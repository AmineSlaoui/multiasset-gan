"""
SF-MarketGAN model (v3).

Improvements over v2:
1. FiLM macro conditioning — replaces concat, reduces OOS generalization gap
2. Correction scale η=0.1 — controls magnitude of learned corrections
3. softplus(σ) — ensures positive volatility
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── TCN Blocks ───────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=self.pad)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = F.gelu(self.norm2(self.conv2(h)))
        h = self.drop(h)
        return x + h


class TCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,
                 num_blocks=6, kernel_size=2, dilation_base=2, dropout=0.2):
        super().__init__()
        self.phi_I = nn.Conv1d(in_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, kernel_size, dilation_base ** i, dropout)
            for i in range(num_blocks)
        ])
        self.phi_O = nn.Conv1d(hidden_dim, out_dim, 1)

    def forward(self, x):
        h = self.phi_I(x)
        for b in self.blocks:
            h = b(h)
        return self.phi_O(h)


# ── FiLM Layer ───────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation.

    Maps macro covariates → (γ, β), then modulates hidden state:
      h ← γ ⊙ h + β

    Applied after the final TCN block. Adds <5K params.
    """
    def __init__(self, d_cov, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_cov, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # γ and β
        )
        # Init γ≈1, β≈0 (identity transform initially)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.mlp[-1].bias.data[:hidden_dim] = 1.0  # γ init to 1

    def forward(self, h, cov):
        """
        h:   (B, hidden_dim, T)
        cov: (B, d_cov, T)
        Returns: (B, hidden_dim, T)
        """
        # Pool covariates over time for global conditioning
        cov_pooled = cov.mean(dim=2)  # (B, d_cov)
        params = self.mlp(cov_pooled)  # (B, 2*hidden_dim)
        gamma, beta = params.chunk(2, dim=1)  # Each (B, hidden_dim)
        gamma = gamma.unsqueeze(2)  # (B, hidden_dim, 1)
        beta = beta.unsqueeze(2)
        return gamma * h + beta


# ── Generator ────────────────────────────────────────────────────────

class Generator(nn.Module):
    """SF-MarketGAN Generator with FiLM conditioning.

    TCN processes noise only → FiLM modulates with macro → output heads.
    Correction scale η=0.1 controls learned correction magnitude.
    """
    def __init__(self, n_assets=60, n_factors=5, d_z=10, d_cov=15,
                 hidden_dim=256, num_blocks=6, dropout=0.2, eta=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors
        self.d_z = d_z
        self.d_cov = d_cov
        self.eta = eta
        self.rfs = 1 + 2 * (2 - 1) * (2 ** num_blocks - 1) // (2 - 1)

        # TCN backbone: processes NOISE ONLY (not macro)
        self.backbone = nn.Sequential(
            nn.Conv1d(d_z, hidden_dim, 1),
            *[ResidualBlock(hidden_dim, 2, 2 ** i, dropout) for i in range(num_blocks)]
        )

        # FiLM: macro conditions the hidden state AFTER TCN
        self.film = FiLMLayer(d_cov, hidden_dim)

        # Output heads
        self.alpha_out = nn.Conv1d(hidden_dim, n_assets, 1)
        self.beta_out = nn.Conv1d(hidden_dim, n_assets * n_factors, 1)
        self.sigma_out = nn.Conv1d(hidden_dim, n_assets, 1)

        # ε network: noise + macro → residuals (1×1 convs, RFS=1)
        self.eps_net = TCN(d_z + d_cov, hidden_dim, n_assets,
                           num_blocks=num_blocks, kernel_size=1,
                           dilation_base=1, dropout=dropout)

    def forward(self, z_full, cov_full, alpha_hat, beta_hat, sigma_hat, factors):
        """
        z_full:    (B, d_z, T_full)     T_full = RFS + T_L + 1
        cov_full:  (B, d_cov, T_full)
        alpha_hat: (B, N, T_L)
        beta_hat:  (B, N, K, T_L)
        sigma_hat: (B, N, T_L)
        factors:   (B, K, T_L)
        Returns:   (B, N, T_L)
        """
        B = z_full.shape[0]
        T_L = factors.shape[2]

        # 1. TCN on noise only (macro not mixed in)
        h = self.backbone(z_full[:, :, :-1])  # (B, hidden_dim, T_full-1)

        # 2. FiLM: modulate with macro covariates
        h = self.film(h, cov_full[:, :, :-1])

        # 3. Take last T_L steps
        h_out = h[:, :, -T_L:]

        # 4. Output heads with η-scaled corrections
        alpha_corr = self.eta * torch.tanh(self.alpha_out(h_out))
        beta_corr = self.eta * torch.tanh(self.beta_out(h_out)).view(
            B, self.n_assets, self.n_factors, T_L)
        sigma_corr = self.eta * torch.tanh(self.sigma_out(h_out))

        # 5. Apply corrections (softplus for σ to ensure positive)
        alpha = alpha_hat * (1 + alpha_corr)
        beta = beta_hat * (1 + beta_corr)
        sigma = sigma_hat * F.softplus(1 + sigma_corr)

        # 6. Residuals (noise + macro, contemporaneous)
        x_eps = torch.cat([z_full[:, :, -T_L:], cov_full[:, :, -(T_L+1):-1]], dim=1)
        eps = self.eps_net(x_eps)

        # 7. r = α + β·F + σ⊙ε
        r_sys = (beta * factors.unsqueeze(1)).sum(dim=2)
        return alpha + r_sys + sigma * eps


# ── Discriminator ────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """WGAN-GP critic with FiLM macro conditioning."""
    def __init__(self, n_assets=60, d_cov=15, hidden_dim=256,
                 num_blocks=6, dropout=0.2):
        super().__init__()
        self.tcn = TCN(n_assets, hidden_dim, hidden_dim,
                       num_blocks=num_blocks, dropout=dropout)
        self.film = FiLMLayer(d_cov, hidden_dim)
        self.head = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, returns_seq, cov_seq):
        """(B, N, T_L), (B, d_cov, T_L) → (B, 1)"""
        h = self.tcn(returns_seq)
        h = self.film(h, cov_seq)
        return self.head(h).mean(dim=2)


# ── Gradient Penalty ─────────────────────────────────────────────────

def compute_gradient_penalty(disc, real, fake, cov, lam=10.0):
    B = real.shape[0]
    alpha = torch.rand(B, 1, 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = disc(interp, cov)
    grads = torch.autograd.grad(
        score, interp, grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True,
    )[0]
    return lam * ((grads.view(B, -1).norm(2, dim=1) - 1) ** 2).mean()
