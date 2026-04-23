"""
SF-MarketGAN — Stylized-Fact-constrained Market GAN.

Final model (V7d): Factor-structured GAN with macro-conditioned alpha+sigma
corrections and Gaussian residuals.

Architecture:
  r_t = alpha_hat_t * (1 + eta*tanh(net_alpha))
      + beta_hat_t * f_t                          [no beta correction]
      + sigma_hat_t * exp(eta_sigma*tanh(net_sigma)) * N(0,I)

Key design choice: no learned residual network (eps_net) and no beta correction.
The GAN learns ONLY alpha and sigma corrections conditioned on macro state.
Factor loadings (beta) are taken directly from rolling OLS — adversarial
perturbation of beta distorts cross-class correlation structure OOS.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from model import ResidualBlock, FiLMLayer


# ── SNFiLMLayer for Discriminator ────────────────────────────────────

class SNFiLMLayer(nn.Module):
    """FiLM layer with spectral norm on linear layers."""
    def __init__(self, d_cov, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            spectral_norm(nn.Linear(d_cov, hidden_dim)),
            nn.GELU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim * 2)),
        )
        nn.init.zeros_(self.mlp[-1].bias)
        self.mlp[-1].bias.data[:hidden_dim] = 1.0
        with torch.no_grad():
            self.mlp[-1].weight_orig.mul_(0.01)

    def forward(self, h, cov):
        B, _, T = cov.shape
        cov_t = cov.permute(0, 2, 1)
        params = self.mlp(cov_t)
        gamma, beta = params.chunk(2, dim=2)
        gamma = gamma.permute(0, 2, 1)
        beta = beta.permute(0, 2, 1)
        return gamma * h + beta


# ── Generator ────────────────────────────────────────────────────────

class Generator(nn.Module):
    """SF-MarketGAN Generator.

    Two primary correction channels + optional factor gate:
    1. alpha: intercept correction (mean shift), eta=0.1
    2. sigma: idiosyncratic volatility correction, eta_sigma=0.3
    3. lambda_f (optional): factor volatility gate, eta_f=0.3
       When trained: scales systematic risk for counterfactual generation.
       When not trained (weights=0): identity, equivalent to FB structure.

    Beta is passed through from rolling OLS (not learned).
    Residuals are i.i.d. Gaussian N(0,I).

    Current best checkpoint: alpha+sigma trained, factor_gate at identity.

    Theoretical support: Pitt & Shephard (1999) factor stochastic volatility,
    Forbes & Rigobon (2002) crisis transmission through common channels.
    """
    def __init__(self, n_assets=60, n_factors=11, d_z=10, d_cov=15,
                 hidden_dim=256, num_blocks=6, dropout=0.2,
                 eta=0.1, eta_sigma=0.3, eta_f=0.3,
                 residual_cholesky=None):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors
        self.d_z = d_z
        self.d_cov = d_cov
        self.eta = eta
        self.eta_sigma = eta_sigma
        self.eta_f = eta_f
        self.rfs = 1 + 2 * (2 - 1) * (2 ** num_blocks - 1) // (2 - 1)

        # Shrunk residual covariance: eps = L @ u instead of N(0,I)
        # L is fixed (not learned), estimated from training residuals
        if residual_cholesky is not None:
            self.register_buffer('L_eps', torch.tensor(residual_cholesky, dtype=torch.float32))
        else:
            self.L_eps = None

        # TCN backbone
        self.input_proj = nn.Conv1d(d_z, hidden_dim, 1)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, 2, 2 ** i, dropout) for i in range(num_blocks)
        ])
        self.film = FiLMLayer(d_cov, hidden_dim)

        # Correction heads
        self.alpha_out = nn.Conv1d(hidden_dim, n_assets, 1)
        self.sigma_out = nn.Conv1d(hidden_dim, n_assets, 1)
        # Factor volatility gate: (B, K, T) — scales each factor's return
        self.factor_gate = nn.Conv1d(hidden_dim, n_factors, 1)
        nn.init.zeros_(self.factor_gate.weight)
        nn.init.zeros_(self.factor_gate.bias)

    def forward(self, z_full, cov_full, alpha_hat, beta_hat, sigma_hat, factors):
        B = z_full.shape[0]
        T_L = factors.shape[2]
        cov_input = cov_full[:, :, :-1]

        h = self.input_proj(z_full[:, :, :-1])
        for block in self.blocks:
            h = block(h)
        h = self.film(h, cov_input)
        h_out = h[:, :, -T_L:]

        # Alpha correction (intercept)
        alpha_corr = self.eta * torch.tanh(self.alpha_out(h_out))
        alpha = alpha_hat * (1 + alpha_corr)

        # Beta: no correction
        beta = beta_hat

        # Factor volatility gate: scales systematic returns
        # lambda_f ∈ [exp(-eta_f), exp(eta_f)]
        # At init: lambda_f ≈ 1 (no change), learns macro-dependent scaling
        lambda_f = torch.exp(self.eta_f * torch.tanh(self.factor_gate(h_out)))  # (B, K, T)
        scaled_factors = lambda_f * factors  # (B, K, T)

        # Idiosyncratic volatility correction
        sigma_corr = self.eta_sigma * torch.tanh(self.sigma_out(h_out))
        sigma = sigma_hat * torch.exp(sigma_corr)

        # Residuals: shrunk covariance if available, else N(0,I)
        u = torch.randn(B, self.n_assets, T_L, device=z_full.device)
        if self.L_eps is not None:
            eps = torch.einsum('ij,bjt->bit', self.L_eps, u)
        else:
            eps = u

        # r = alpha + beta * (lambda_f * f) + sigma * eps
        r_sys = (beta * scaled_factors.unsqueeze(1)).sum(dim=2)
        return alpha + r_sys + sigma * eps


# ── Discriminator ────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """SN-GAN Discriminator with full spectral norm + SNFiLM."""

    def __init__(self, n_assets=60, d_cov=15, hidden_dim=256,
                 num_blocks=6, dropout=0.2):
        super().__init__()

        self.input_proj = spectral_norm(nn.Conv1d(n_assets + d_cov, hidden_dim, 1))
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = ResidualBlock(hidden_dim, 2, 2**i, dropout)
            block.conv1.conv = spectral_norm(block.conv1.conv)
            block.conv2.conv = spectral_norm(block.conv2.conv)
            self.blocks.append(block)
        self.film = SNFiLMLayer(d_cov, hidden_dim)
        self.head = spectral_norm(nn.Conv1d(hidden_dim, 1, 1))

    def forward(self, returns_seq, cov_seq):
        x = torch.cat([returns_seq, cov_seq], dim=1)
        h = self.input_proj(x)
        for b in self.blocks:
            h = b(h)
        h = self.film(h, cov_seq)
        return self.head(h).mean(dim=2)


# Aliases for backward compatibility
GeneratorV7d = Generator
DiscriminatorV7d = Discriminator
