"""MarketGAN-style GMV portfolio evaluation on 2025 OOS.

Addresses reviewer U1: MarketGAN's headline economic-value experiment is
mean-variance / GMV portfolio construction using synthetic scenario
covariance. This script runs that experiment for FB, FB+shrunk, v7d
(reference), and v9 (regime-conditional), plus equal-weight and sample-cov
baselines. All methods use **monthly rebalancing** on the 2025 test window.

At each rebalance date t:
  1. Generate n_paths × 21-day forward paths from each model, conditioned
     on the macro state at t.
  2. Compute the sample covariance Σ̂_model from those paths.
  3. Solve the long-only GMV:  w* = argmin w' Σ̂_model w,  s.t. 1'w = 1, w ≥ 0.
  4. Hold w* for 21 days, realize returns from real test data.

Metrics: annualized Sharpe, volatility, turnover, max drawdown.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import FactorBootstrap
from data_loader import ASSET_CLASSES, load_macro, load_returns, train_val_test_split
from factors_pca import compute_hierarchical_pca_factors, compute_rolling_coefficients
from macro_processor import MacroProcessor
from sfmg_baseline import Generator
from sfmg_generator import SFMGGenerator
from eval_sfmg import gen_paths_sfmg
from walk_forward import (
    residual_cholesky_from_history,
    gen_paths_walk_forward,
    fb_paths_walk_forward,
)


TRADING_DAYS = 252


def gmv_long_only(sigma: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Long-only GMV via projected inverse. Simple closed-form + clipping fallback.

    We solve argmin w'Σw s.t. 1'w=1, w≥0 approximately by:
      - analytical inverse-variance GMV (can be negative)
      - project onto simplex via Euclidean projection
    This is fast, differentiable, and good enough for path-count comparisons.
    """
    N = sigma.shape[0]
    Sigma = sigma + reg * np.eye(N)
    inv1 = np.linalg.solve(Sigma, np.ones(N))
    w = inv1 / inv1.sum()
    # Project onto simplex (sort-based)
    w_sorted = np.sort(w)[::-1]
    cumsum = np.cumsum(w_sorted) - 1
    rho = np.argmax(w_sorted - cumsum / (np.arange(N) + 1) <= 0) if (w_sorted - cumsum / (np.arange(N) + 1) <= 0).any() else N
    theta = cumsum[max(rho - 1, 0)] / max(rho, 1)
    w_proj = np.maximum(w - theta, 0)
    s = w_proj.sum()
    return w_proj / s if s > 0 else np.ones(N) / N


def backtest(weights_per_rebal: np.ndarray, real_test: np.ndarray, rebal_freq: int = 21):
    """weights_per_rebal: (n_rebal, N) — rebalance at days 0, freq, 2*freq, ...
    real_test: (T, N).
    Returns dict with annualized metrics.
    """
    T, N = real_test.shape
    daily_port = np.zeros(T)
    weight_path = np.zeros((T, N))
    turnover = 0.0
    for k, w in enumerate(weights_per_rebal):
        start = k * rebal_freq
        end = min(start + rebal_freq, T)
        weight_path[start:end] = w
        daily_port[start:end] = real_test[start:end] @ w
        if k > 0:
            turnover += np.abs(w - weights_per_rebal[k - 1]).sum()

    ann_ret = daily_port.mean() * TRADING_DAYS
    ann_vol = daily_port.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    cum = np.exp(np.cumsum(daily_port))
    dd = cum / np.maximum.accumulate(cum) - 1
    mdd = dd.min()
    return {"ann_return": float(ann_ret), "ann_vol": float(ann_vol),
            "sharpe": float(sharpe), "max_drawdown": float(mdd),
            "turnover": float(turnover / max(len(weights_per_rebal) - 1, 1))}


def synth_cov_from_paths(paths: np.ndarray, window_start: int, window_len: int) -> np.ndarray:
    """paths: (n_paths, T, N). Slice rows [window_start:window_start+window_len]
    across all paths, stack, return sample cov (N,N)."""
    seg = paths[:, window_start:window_start + window_len, :]
    flat = seg.reshape(-1, seg.shape[-1])
    return np.cov(flat.T)


def _gen_paths_from_gen(gen, cov_arr, f_arr, alpha_hat, beta_hat, sigma_hat,
                        test_start_idx, n_paths, t_l, device):
    """Single full-window generation (n_paths × t_l × N)."""
    N = alpha_hat.shape[1]
    rfs = gen.rfs
    T_full = rfs + t_l + 1
    t_start = max(0, test_start_idx - rfs - 1)
    t_end = min(t_start + T_full, len(cov_arr))
    cov_seq = torch.tensor(cov_arr[t_start:t_end], dtype=torch.float32).T.unsqueeze(0).to(device)
    if t_end - t_start < T_full:
        cov_seq = F.pad(cov_seq, (0, T_full - (t_end - t_start)), mode="replicate")
    ts, te = test_start_idx, min(test_start_idx + t_l, len(f_arr))
    actual_tl = te - ts
    f = torch.tensor(f_arr[ts:te], dtype=torch.float32).T.unsqueeze(0).to(device)
    a = torch.tensor(alpha_hat[ts:te].T, dtype=torch.float32).unsqueeze(0).to(device)
    s = torch.tensor(sigma_hat[ts:te].T, dtype=torch.float32).unsqueeze(0).to(device)
    b = torch.tensor(beta_hat[ts:te], dtype=torch.float32).permute(1, 2, 0).unsqueeze(0).to(device)
    if actual_tl < t_l:
        pad = t_l - actual_tl
        f = F.pad(f, (0, pad), mode="replicate")
        a = F.pad(a, (0, pad), mode="replicate")
        s = F.pad(s, (0, pad), mode="replicate")
        b = F.pad(b, (0, pad, 0, 0), mode="replicate")
    out = np.zeros((n_paths, actual_tl, N))
    bs = 20
    for i in range(0, n_paths, bs):
        k = min(bs, n_paths - i)
        z = torch.randn(k, gen.d_z, T_full, device=device)
        with torch.no_grad():
            r = gen(z, cov_seq.expand(k, -1, -1), a.expand(k, -1, -1),
                    b.expand(k, -1, -1, -1), s.expand(k, -1, -1), f.expand(k, -1, -1))
        for j in range(k):
            out[i + j] = r[j, :, :actual_tl].cpu().numpy().T
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sfmg_ckpt", default="results/v9_s42/best_model.pt")
    p.add_argument("--baseline_ckpt", default="results/v7d/best_model.pt")
    p.add_argument("--n_paths", type=int, default=200)
    p.add_argument("--rebal_freq", type=int, default=21)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/portfolio_oos.json")
    args = p.parse_args()

    device = torch.device("cpu")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    print("=" * 72)
    print("  GMV portfolio OOS — monthly rebalancing on 2025")
    print("=" * 72)

    returns = load_returns(); macro = load_macro()
    train_ret, val_ret, test_ret = train_val_test_split(returns)
    test_start_idx = returns.index.get_loc(test_ret.index[0])
    train_end = str(train_ret.index[-1].date())

    proc = MacroProcessor(n_pca_components=8, n_regimes=3)
    covariates = proc.fit_transform(macro, returns.index, train_end=train_end)
    cov_arr = covariates.reindex(returns.index).fillna(0).values
    d_cov = covariates.shape[1]

    factors_df, _ = compute_hierarchical_pca_factors(
        returns, ASSET_CLASSES, n_global=5, n_class=2, train_end=train_end)
    f_arr = factors_df.values
    n_factors = factors_df.shape[1]
    alpha_hat, beta_hat, sigma_hat = compute_rolling_coefficients(
        returns, factors_df, window=126)

    real_test = test_ret.values
    T, N = real_test.shape
    print(f"  2025 OOS: T={T} days, N={N} assets, rebal every {args.rebal_freq} days")

    # Rebalance dates relative to test start
    rebal_points = list(range(0, T, args.rebal_freq))
    rebal_abs = [test_start_idx + rp for rp in rebal_points]
    n_rebal = len(rebal_points)
    print(f"  Rebalance points: {n_rebal} (walk-forward)")

    # ── Walk-forward path generation per method ────────────────────────
    def _L_eps_at(rp_abs):
        return residual_cholesky_from_history(
            returns.values, alpha_hat, beta_hat, f_arr, rp_abs)

    segments_bank = {}

    np.random.seed(args.seed)
    segments_bank["FB"] = fb_paths_walk_forward(
        fb_ctor=lambda: FactorBootstrap(window=126),
        returns=returns.values, factors=f_arr,
        rebal_points_abs=rebal_abs, rebal_freq=args.rebal_freq,
        n_paths=args.n_paths, residual_cholesky_fn=_L_eps_at, shrunk=False,
    )

    np.random.seed(args.seed)
    segments_bank["FB+shrunk"] = fb_paths_walk_forward(
        fb_ctor=lambda: FactorBootstrap(window=126),
        returns=returns.values, factors=f_arr,
        rebal_points_abs=rebal_abs, rebal_freq=args.rebal_freq,
        n_paths=args.n_paths, residual_cholesky_fn=_L_eps_at,
        shrunk=True, lam=0.2,
    )

    if os.path.exists(args.baseline_ckpt):
        cfg7 = json.load(open(os.path.join(os.path.dirname(args.baseline_ckpt), "config.json")))
        # v7d has no regime gate, so the walk-forward helper (tailored to v9)
        # would need extension. Use the v7d checkpoint's own L_rho (was
        # saved in state_dict) and keep the generator call as before, but
        # flat-extrapolate conditioning per rebalance.
        g7 = Generator(n_assets=N, n_factors=n_factors, d_z=10, d_cov=d_cov,
                       hidden_dim=cfg7.get("hidden_dim", 256),
                       eta=cfg7.get("eta", 0.1), eta_sigma=cfg7.get("eta_sigma", 0.1),
                       residual_cholesky=None).to(device)
        c7 = torch.load(args.baseline_ckpt, map_location=device, weights_only=False)
        g7.load_state_dict(c7["G"], strict=False); g7.eval()
        segments_bank["v7d"] = gen_paths_walk_forward(
            g7, cov_arr, f_arr, alpha_hat, beta_hat, sigma_hat,
            rebal_points_abs=rebal_abs, rebal_freq=args.rebal_freq,
            n_paths=args.n_paths, device=device,
        )

    if os.path.exists(args.sfmg_ckpt):
        cfg9 = json.load(open(os.path.join(os.path.dirname(args.sfmg_ckpt), "config.json")))
        g9 = SFMGGenerator(
            n_assets=N, n_factors=n_factors, d_z=10, d_cov=d_cov,
            hidden_dim=cfg9["hidden_dim"], num_blocks=cfg9["num_blocks"], dropout=0.2,
            eta_low=cfg9["eta_low"], eta_high=cfg9["eta_high"],
            eta_sigma_low=cfg9["eta_sigma_low"], eta_sigma_high=cfg9["eta_sigma_high"],
            residual_rank=cfg9["residual_rank"],
            residual_cholesky=None,      # keep checkpoint's own L_rho (bug #4)
        ).to(device)
        c9 = torch.load(args.sfmg_ckpt, map_location=device, weights_only=False)
        g9.load_state_dict(c9["G"], strict=False); g9.eval()
        f_in = f_arr
        if cfg9.get("lag_factors", False):
            f_in = np.vstack([np.zeros((1, n_factors)), f_arr[:-1]])
        segments_bank["v9"] = gen_paths_walk_forward(
            g9, cov_arr, f_in, alpha_hat, beta_hat, sigma_hat,
            rebal_points_abs=rebal_abs, rebal_freq=args.rebal_freq,
            n_paths=args.n_paths, device=device,
        )

    # ── Backtest each method ───────────────────────────────────────────
    results = {}

    # Baseline: equal weight
    ew = np.ones((n_rebal, N)) / N
    results["EqualWeight"] = backtest(ew, real_test, args.rebal_freq)

    # Baseline: rolling sample covariance (252-day lookback)
    print("\n  Sample-cov 252d-lookback ...")
    sample_w = np.zeros((n_rebal, N))
    for k, rp in enumerate(rebal_points):
        hist_end = test_start_idx + rp
        hist_start = max(0, hist_end - 252)
        cov_hist = np.cov(returns.values[hist_start:hist_end].T)
        sample_w[k] = gmv_long_only(cov_hist)
    results["SampleCov"] = backtest(sample_w, real_test, args.rebal_freq)

    # Model-based GMVs (walk-forward segments).
    for name, segments in segments_bank.items():
        print(f"  {name} ...")
        w_per_rebal = np.zeros((n_rebal, N))
        for k, seg in enumerate(segments):
            cov_model = np.cov(seg.reshape(-1, N).T)
            w_per_rebal[k] = gmv_long_only(cov_model)
        results[name] = backtest(w_per_rebal, real_test, args.rebal_freq)

    # ── Print + save ───────────────────────────────────────────────────
    print(f"\n{'='*72}\n  GMV portfolio 2025 OOS\n{'='*72}")
    print(f"  {'Method':<14s} {'Sharpe':>8s} {'Ann Vol':>9s} {'Ann Ret':>9s} {'MaxDD':>8s} {'Turnover':>10s}")
    order = ["EqualWeight", "SampleCov", "FB", "FB+shrunk", "v7d", "v9"]
    for n in order:
        if n not in results: continue
        r = results[n]
        print(f"  {n:<14s} {r['sharpe']:>8.3f} {r['ann_vol']:>9.4f} "
              f"{r['ann_return']:>9.4f} {r['max_drawdown']:>8.3f} {r['turnover']:>10.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": {"n_paths": args.n_paths,
                               "rebal_freq": args.rebal_freq,
                               "seed": args.seed,
                               "test_start": str(test_ret.index[0].date()),
                               "test_end": str(test_ret.index[-1].date())},
                   "results": results}, f, indent=2)
    print(f"\n  Saved → {args.out}")


if __name__ == "__main__":
    main()
