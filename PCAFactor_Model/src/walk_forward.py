"""Walk-forward generation + residual-Cholesky helpers.

All paper-evaluation scripts that compute covariance for GMV / MV portfolio
decisions must condition on strictly-past information at each rebalance,
not on the realised future macro/factor trajectory. This module contains
the shared helpers that enforce that invariant.

Key functions:
  - ``residual_cholesky_from_history``: fit L_eps from factor residuals on
    data strictly before a test index, so rolling-year / FB+shrunk does
    not inject later-sample residual correlations.
  - ``gen_paths_walk_forward``: run the v9 generator once per rebalance
    with flat-extrapolated macro/factor/α/β/σ over the holding period.
  - ``fb_paths_walk_forward``: factor-bootstrap equivalent that refits
    residuals and regenerates per rebalance using only past information.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from eval_sfmg import gen_paths_sfmg


def residual_cholesky_from_history(returns, alpha_hat, beta_hat, factors,
                                   history_end_idx, window_start=126,
                                   eig_floor=1e-6, jitter=1e-6):
    """Training-residual Cholesky using only rows [window_start, history_end_idx).

    Parameters
    ----------
    returns : (T, N) array
    alpha_hat : (T, N), beta_hat : (T, N, K) — causal rolling OLS estimates
    factors : (T, K) factor panel
    history_end_idx : int — first index excluded from the residual fit
    """
    res = np.zeros_like(returns)
    for t in range(window_start, history_end_idx):
        res[t] = returns[t] - alpha_hat[t] - beta_hat[t] @ factors[t]
    res_hist = res[window_start:history_end_idx]
    C = np.corrcoef(res_hist.T)
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    w = np.clip(w, eig_floor, None)
    C_psd = (V * w) @ V.T
    d = np.sqrt(np.diag(C_psd))
    C_norm = C_psd / np.outer(d, d)
    return np.linalg.cholesky(C_norm + jitter * np.eye(C_norm.shape[0]))


def _flat_extrapolate(arr, start_abs, length):
    """Return a copy of ``arr`` with rows [start_abs, start_abs+length)
    replaced by ``arr[start_abs - 1]`` (flat-hold forecast). ``arr`` is
    1-D along the time axis 0."""
    out = arr.copy()
    if start_abs <= 0:
        # No history yet — use the first row as the "last known" value.
        last = arr[0]
    else:
        last = arr[start_abs - 1]
    end = min(start_abs + length, out.shape[0])
    out[start_abs:end] = last
    return out


def gen_paths_walk_forward(gen, cov_arr, factors_arr, alpha_hat, beta_hat,
                            sigma_hat, rebal_points_abs, rebal_freq,
                            n_paths, device, batch=20):
    """Run the generator once per rebalance with flat-extrapolated conditioning.

    Parameters
    ----------
    rebal_points_abs : list[int] — absolute indices into the global time axis
        (i.e. returns.index positions), not test-window offsets.
    rebal_freq : int — holding-period length in trading days.

    Returns
    -------
    list of ndarray, one ``(n_paths, rebal_freq, N)`` segment per rebalance.
    """
    segments = []
    for rp_abs in rebal_points_abs:
        cov_wf = _flat_extrapolate(cov_arr, rp_abs, rebal_freq)
        f_wf = _flat_extrapolate(factors_arr, rp_abs, rebal_freq)
        # α, β, σ at times ≥ rp_abs use a 126-day rolling OLS that would
        # contain realised future returns inside the holding window.
        # Freeze them at the rp_abs-1 estimate (strictly past).
        alpha_wf = _flat_extrapolate(alpha_hat, rp_abs, rebal_freq)
        sigma_wf = _flat_extrapolate(sigma_hat, rp_abs, rebal_freq)
        beta_wf = beta_hat.copy()
        if rp_abs > 0:
            last_beta = beta_hat[rp_abs - 1]
        else:
            last_beta = beta_hat[0]
        end = min(rp_abs + rebal_freq, beta_wf.shape[0])
        beta_wf[rp_abs:end] = last_beta

        paths = gen_paths_sfmg(
            gen, cov_wf, f_wf, alpha_wf, beta_wf, sigma_wf,
            test_start_idx=rp_abs, n_paths=n_paths, t_l=rebal_freq,
            batch=batch, device=device,
        )
        segments.append(paths)
    return segments


def fb_paths_walk_forward(fb_ctor, returns, factors, rebal_points_abs,
                           rebal_freq, n_paths, residual_cholesky_fn,
                           shrunk=False, lam=0.2):
    """FactorBootstrap equivalent: refits per rebalance on strictly-past data.

    Parameters
    ----------
    fb_ctor : callable -> FactorBootstrap instance (so each rebalance gets a
        fresh, history-only fit).
    residual_cholesky_fn : callable(rp_abs) -> (N, N) L_eps or ``None``. Only
        used when ``shrunk=True``.
    """
    segments = []
    for rp_abs in rebal_points_abs:
        fb = fb_ctor()
        fb.fit(returns[:rp_abs], factors[:rp_abs])
        if shrunk:
            L_eps = residual_cholesky_fn(rp_abs)
            fb.fit_shrunk_residuals(returns[:rp_abs], factors[:rp_abs],
                                    lam=lam, L_eps=L_eps)
        # flat-extrapolate factors over holding period so FB doesn't peek
        # ahead either.
        f_wf = _flat_extrapolate(factors, rp_abs, rebal_freq)
        paths = fb.generate(f_wf, start_idx=rp_abs, n_paths=n_paths)
        segments.append(paths[:, :rebal_freq, :])
    return segments
