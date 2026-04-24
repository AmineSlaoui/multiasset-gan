"""
Stylized Fact Constraint Losses for SF-MarketGAN (v3).

Active losses (5):
  L_corr     — correlation matrix Frobenius distance
  L_tail     — soft-sigmoid joint tail co-dependence
  L_vc       — volatility clustering (ACF of |returns|)
  L_lev      — leverage effect (return-to-future-vol correlation)
  L_spill    — volatility spillover (|returns| correlation matrix)

Disabled:
  L_kurtosis — overfits non-stationary tail statistics (confirmed by ablation)
"""
import torch
import torch.nn.functional as F
from typing import Optional, Dict


def _corrcoef(x: torch.Tensor) -> torch.Tensor:
    """Correlation matrix from (M, N) data → (N, N)."""
    x_c = x - x.mean(dim=0, keepdim=True)
    cov = (x_c.T @ x_c) / max(x.shape[0] - 1, 1)
    std = torch.sqrt(torch.diag(cov)).clamp(min=1e-8)
    return (cov / (std.unsqueeze(0) * std.unsqueeze(1))).clamp(-1, 1)


def loss_correlation(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """L_corr: Frobenius distance of correlation matrices.

    real, fake: (B, T, N) → pool B×T for cross-sectional stats.
    """
    r = real.reshape(-1, real.shape[-1])
    f = fake.reshape(-1, fake.shape[-1])
    N = r.shape[1]
    return torch.norm(_corrcoef(r) - _corrcoef(f), p="fro") ** 2 / (N * N)


def loss_tail_codependence(
    real: torch.Tensor, fake: torch.Tensor, q: float = 0.05, tau: float = 0.01
) -> torch.Tensor:
    """L_tail: Joint tail probability with soft-sigmoid indicator.

    Differentiable via sigmoid((qi - ri) / tau) instead of hard threshold.
    """
    r = real.reshape(-1, real.shape[-1])
    f = fake.reshape(-1, fake.shape[-1])
    N = r.shape[1]

    # Quantiles from real data
    quantiles = torch.quantile(r, q, dim=0)  # (N,)

    # Soft indicators
    real_below = torch.sigmoid((quantiles.unsqueeze(0) - r) / tau)  # (M, N)
    fake_below = torch.sigmoid((quantiles.unsqueeze(0) - f) / tau)

    # Joint tail probability matrices
    # P(i<q, j<q) ≈ mean of product of soft indicators
    real_joint = (real_below.T @ real_below) / r.shape[0]  # (N, N)
    fake_joint = (fake_below.T @ fake_below) / f.shape[0]

    return F.mse_loss(fake_joint, real_joint)


def loss_volatility_clustering(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """L_vc: MSE of lag-1 ACF of |returns| across assets."""
    def _acf1_abs(x):
        # x: (B, T, N)
        a = x.abs()
        a_c = a - a.mean(dim=1, keepdim=True)
        var = (a_c ** 2).mean(dim=1).clamp(min=1e-12)  # (B, N)
        cov = (a_c[:, 1:] * a_c[:, :-1]).mean(dim=1)   # (B, N)
        return (cov / var).mean(dim=0)  # (N,) averaged over batch

    return F.mse_loss(_acf1_abs(fake), _acf1_abs(real))


def loss_leverage(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """L_lev: MSE of Corr(r_t, |r_{t+1}|) across assets."""
    def _lev_corr(x):
        # x: (B, T, N)
        r = x[:, :-1]           # (B, T-1, N)
        a_next = x[:, 1:].abs()  # (B, T-1, N)
        r_c = r - r.mean(dim=1, keepdim=True)
        a_c = a_next - a_next.mean(dim=1, keepdim=True)
        cov = (r_c * a_c).mean(dim=1)  # (B, N)
        std_r = r_c.std(dim=1).clamp(min=1e-8)
        std_a = a_c.std(dim=1).clamp(min=1e-8)
        return (cov / (std_r * std_a)).mean(dim=0)  # (N,)

    return F.mse_loss(_lev_corr(fake), _lev_corr(real))


def loss_spillover(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """L_spill: Frobenius distance of |returns| correlation matrices.

    Captures cross-asset volatility transmission (shock propagation).
    """
    r = real.reshape(-1, real.shape[-1]).abs()
    f = fake.reshape(-1, fake.shape[-1]).abs()
    N = r.shape[1]
    return torch.norm(_corrcoef(r) - _corrcoef(f), p="fro") ** 2 / (N * N)


class StylizedFactLoss:
    """Combined SF constraint loss with configurable weights.

    Default weights follow a balanced allocation across 5 active losses.
    Kurtosis is DISABLED (weight=0) based on ablation evidence.
    """
    def __init__(
        self,
        w_corr: float = 0.5,
        w_tail: float = 0.3,
        w_vc: float = 0.3,
        w_lev: float = 0.2,
        w_spill: float = 0.3,
        tail_tau: float = 0.01,
        class_indices: Optional[Dict[str, list]] = None,
    ):
        self.tail_tau = tail_tau
        self.w_corr = w_corr
        self.w_tail = w_tail
        self.w_vc = w_vc
        self.w_lev = w_lev
        self.w_spill = w_spill
        self.class_indices = class_indices

    def __call__(self, real: torch.Tensor, fake: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        real, fake: (B, T, N)
        Returns dict with individual + total loss.
        """
        losses = {}

        losses["corr"] = self.w_corr * loss_correlation(real, fake)
        losses["tail"] = self.w_tail * loss_tail_codependence(real, fake, tau=self.tail_tau)

        if real.shape[1] >= 10:
            losses["vc"] = self.w_vc * loss_volatility_clustering(real, fake)
            losses["lev"] = self.w_lev * loss_leverage(real, fake)
        else:
            zero = torch.tensor(0.0, device=real.device)
            losses["vc"] = zero
            losses["lev"] = zero

        losses["spill"] = self.w_spill * loss_spillover(real, fake)

        losses["total"] = sum(losses.values())
        return losses
