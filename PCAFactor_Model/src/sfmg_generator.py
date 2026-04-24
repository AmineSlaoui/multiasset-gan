"""SF-MarketGAN v9 — regime-conditional factor GAN.

Motivation
----------
Round 2B reviewer's central critique of v7d was that with η=0.01, the GAN's
output is bounded to ≤1% deviation from the rolling-OLS + shrunk-residual
baseline, so the GAN contributes almost nothing. v9 addresses this by giving
the GAN one specific job that FactorBootstrap and DCC-GARCH categorically
cannot do: *regime-conditional* residual-covariance correction.

Design (delta vs. v7d)
----------------------
A small **regime encoder** g(cov_t) ∈ [0, 1] is added. It maps each timestep's
macro covariate vector to a scalar "stress indicator." The encoder can be
supervised with an auxiliary BCE loss against `1{VIX_z > 0.3}` so it learns
an interpretable regime, not an arbitrary latent.

The regime score gates three channels:

1. Correction strength: η_α(t) = η_low + z(t) · (η_high - η_low),
   same for η_σ. So the network can correct OLS more aggressively when stress
   is detected, while staying close to OLS in normal regimes.

2. Residual-covariance correction: ε_t = σ̂_t · (L_ρ + z(t) · L_Δ) · u,
   where L_ρ is the fixed v7d shrunk correlation Cholesky and L_Δ ∈ R^{N×r}
   is a learned rank-r correction (default r=4). When z=0 the residual law is
   exactly v7d's; when z=1 the covariance structure is augmented by a learned
   stress component. This is the mechanism FB/DCC cannot replicate.

The regime encoder is just a 2-layer MLP on macro covariates — a few thousand
parameters. The rank-r residual correction is N·r ≈ 240 parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ResidualBlock, FiLMLayer


class RegimeEncoder(nn.Module):
    """Maps cov_t → z(t) ∈ [0, 1]. 2-layer MLP with scalar output."""
    def __init__(self, d_cov, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_cov, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Init bias negative so initial z is near 0 (collapse to v7d at start)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def forward(self, cov):
        """cov: (B, d_cov, T) → z: (B, 1, T) ∈ [0, 1]"""
        cov_t = cov.permute(0, 2, 1)              # (B, T, d_cov)
        logit = self.mlp(cov_t).permute(0, 2, 1)   # (B, 1, T)
        return torch.sigmoid(logit)


class SFMGGenerator(nn.Module):
    """Regime-conditional factor GAN.

    Regime-gated channels:
      - η_α, η_σ scaled by z(t)
      - Residual cov: L(t) = L_ρ + z(t) · L_Δ, with L_Δ ∈ R^{N×r}
    """
    def __init__(self, n_assets=60, n_factors=11, d_z=10, d_cov=15,
                 hidden_dim=256, num_blocks=4, dropout=0.2,
                 eta_low=0.01, eta_high=0.10,
                 eta_sigma_low=0.05, eta_sigma_high=0.20,
                 residual_rank=4,
                 residual_cholesky=None,
                 trainable_L_rho=False):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors
        self.d_z = d_z
        self.d_cov = d_cov
        self.eta_low = eta_low; self.eta_high = eta_high
        self.eta_sigma_low = eta_sigma_low; self.eta_sigma_high = eta_sigma_high
        self.residual_rank = residual_rank
        self.rfs = 1 + 2 * (2 - 1) * (2 ** num_blocks - 1) // (2 - 1)

        # Residual Cholesky. Either a frozen buffer (legacy v9 behaviour) or
        # a learnable parameter (v9.1: let GAN loss adjust the 60x60 residual
        # correlation structure; rank-4 L_delta is too coarse to specifically
        # fix high-residual-correlation pairs like Visa/Mastercard).
        self.trainable_L_rho = trainable_L_rho
        if residual_cholesky is not None:
            L_init = torch.tensor(residual_cholesky, dtype=torch.float32)
            if trainable_L_rho:
                self.L_rho = nn.Parameter(L_init)
            else:
                self.register_buffer("L_rho", L_init)
        else:
            self.L_rho = None

        # Learned low-rank residual correction (N × r). Init to zero so v9
        # collapses to v7d at epoch 0.
        self.L_delta = nn.Parameter(torch.zeros(n_assets, residual_rank))

        # Regime encoder
        self.regime_encoder = RegimeEncoder(d_cov, hidden_dim=32)

        # TCN backbone + FiLM (same as v7d)
        self.input_proj = nn.Conv1d(d_z, hidden_dim, 1)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, 2, 2 ** i, dropout) for i in range(num_blocks)
        ])
        self.film = FiLMLayer(d_cov, hidden_dim)

        # Correction heads (same as v7d)
        self.alpha_out = nn.Conv1d(hidden_dim, n_assets, 1)
        self.sigma_out = nn.Conv1d(hidden_dim, n_assets, 1)

    def forward(self, z_full, cov_full, alpha_hat, beta_hat, sigma_hat, factors,
                return_regime=False):
        B = z_full.shape[0]
        T_L = factors.shape[2]

        cov_input = cov_full[:, :, :-1]
        h = self.input_proj(z_full[:, :, :-1])
        for block in self.blocks:
            h = block(h)
        h = self.film(h, cov_input)
        h_out = h[:, :, -T_L:]

        # ── Regime-aware η gating ─────────────────────────────────────
        cov_out = cov_full[:, :, -T_L:]
        z_reg = self.regime_encoder(cov_out)                     # (B, 1, T_L)
        eta_a = self.eta_low + z_reg * (self.eta_high - self.eta_low)         # (B, 1, T_L)
        eta_s = self.eta_sigma_low + z_reg * (self.eta_sigma_high - self.eta_sigma_low)

        alpha_corr = eta_a * torch.tanh(self.alpha_out(h_out))   # (B, N, T_L)
        sigma_corr = eta_s * torch.tanh(self.sigma_out(h_out))
        alpha = alpha_hat * (1 + alpha_corr)
        sigma = sigma_hat * torch.exp(sigma_corr)

        # β passed through (OLS)
        beta = beta_hat

        # ── Regime-gated residual covariance ──────────────────────────
        # L(t) = L_ρ + z(t) · L_Δ_pad    (L_Δ_pad: N × N, first r cols = L_Δ)
        # Residual draw: ε_t = σ̂_t · L(t) · u_t,  u_t ~ N(0, I)
        # We keep the base L_ρ frozen and add a rank-r stress correction.
        u = torch.randn(B, self.n_assets, T_L, device=z_full.device)
        if self.L_rho is not None:
            eps_base = torch.einsum('ij,bjt->bit', self.L_rho, u)
        else:
            eps_base = u

        # Low-rank stress perturbation: L_Δ @ (u[:r, :]) then scaled by z(t)
        # u_small: (B, r, T_L), L_Δ: (N, r)
        u_small = u[:, :self.residual_rank, :]
        eps_delta = torch.einsum('ij,bjt->bit', self.L_delta, u_small)
        eps = eps_base + z_reg * eps_delta                        # (B, N, T_L)

        factor_term = torch.einsum('bnkt,bkt->bnt', beta, factors)
        r = alpha + factor_term + sigma * eps

        if return_regime:
            return r, z_reg.squeeze(1)  # (B, T_L)
        return r
