"""Walk-forward GMV portfolio backtest over multiple rolling cutoffs.

Each test year uses a checkpoint trained through Y-2 / validated on Y-1:
  - y=2023: v9_roll2022 (train_end=2021-12-31)
  - y=2024: v9_roll2023 (train_end=2022-12-31)
  - y=2025: v9_s42      (train_end=2023-12-31)

Walk-forward invariant: at rebalance rp, the covariance estimate is drawn
from paths conditioned only on information known at rp (macro/factor
through rp-1, rolling-OLS coefficients frozen at their rp-1 values, and
flat-extrapolated conditioning over the holding period). The pre-fix
version conditioned on realised future macro/factor over the whole test
window — see CODEX_REVIEW.md issue #2.

For each year, we:
  1. At each monthly rebalance rp, generate (n_paths × rebal_freq × N) paths
     with the generator warm-started on strictly-past cov/factor and
     flat-extrapolated over [rp, rp+rebal_freq).
  2. Sample covariance from those paths, compute long-only GMV weights
     (simplex-projected), hold for rebal_freq days, realise returns.
  3. Aggregate annualised Sharpe per year; pool daily returns across years
     for a unified Sharpe SE (Lo 2002).
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import FactorBootstrap
from data_loader import ASSET_CLASSES, load_macro, load_returns, train_val_test_split
from factors_pca import compute_hierarchical_pca_factors, compute_rolling_coefficients
from macro_processor import MacroProcessor
from eval_sfmg import gen_paths_sfmg, load_sfmg
from eval_portfolio import gmv_long_only, backtest, synth_cov_from_paths
from walk_forward import (
    residual_cholesky_from_history as _residual_cholesky_from_history,
    gen_paths_walk_forward,
    fb_paths_walk_forward,
)

TRADING_DAYS = 252


def evaluate_year(ckpt_path, test_year, n_paths=200, rebal_freq=21, seed=42):
    """Backtest one year using a v9 checkpoint trained up to test_year-2."""
    device = torch.device("cpu")
    np.random.seed(seed); torch.manual_seed(seed)

    returns = load_returns(); macro = load_macro()
    # Figure out the test window
    from pandas import Timestamp
    te_start = Timestamp(f"{test_year-1}-12-31")
    te_end = Timestamp(f"{test_year}-12-31")
    test_slice = returns.loc[returns.index > te_start]
    test_slice = test_slice.loc[test_slice.index <= te_end]
    test_start_idx = returns.index.get_loc(test_slice.index[0])
    T = len(test_slice); N = returns.shape[1]
    real_test = test_slice.values
    train_end = f"{test_year-2}-12-31"
    print(f"  year={test_year} test={test_slice.index[0].date()}→{test_slice.index[-1].date()} ({T}d)")

    # Pipeline fit on the same train_end (matches what the checkpoint saw)
    proc = MacroProcessor(n_pca_components=8, n_regimes=3)
    covariates = proc.fit_transform(macro, returns.index, train_end=train_end)
    cov_arr = covariates.reindex(returns.index).fillna(0).values
    d_cov = covariates.shape[1]

    factors_df, _ = compute_hierarchical_pca_factors(
        returns, ASSET_CLASSES, n_global=5, n_class=2, train_end=train_end)
    f_arr = factors_df.values; n_factors = factors_df.shape[1]
    alpha_hat, beta_hat, sigma_hat = compute_rolling_coefficients(
        returns, factors_df, window=126)
    # FB+shrunk baseline: rebuild L_eps per cutoff from strictly pre-test
    # residuals so we don't inject later-sample residual correlation into
    # earlier-year rolling backtests (bug #4 in CODEX_REVIEW.md).
    L_eps_fb = _residual_cholesky_from_history(
        returns.values, alpha_hat, beta_hat, f_arr, test_start_idx)

    rebal_points = list(range(0, T - rebal_freq + 1, rebal_freq))
    n_rebal = len(rebal_points)
    # Absolute (into full returns index) rebalance points for walk-forward.
    rebal_abs = [test_start_idx + rp for rp in rebal_points]

    # Walk-forward FB+shrunk: refit per rebalance on strictly-past data,
    # then flat-extrapolate the factor path over the holding period.
    fb_segments = fb_paths_walk_forward(
        fb_ctor=lambda: FactorBootstrap(window=126),
        returns=returns.values, factors=f_arr,
        rebal_points_abs=rebal_abs, rebal_freq=rebal_freq, n_paths=n_paths,
        residual_cholesky_fn=lambda rp: _residual_cholesky_from_history(
            returns.values, alpha_hat, beta_hat, f_arr, rp),
        shrunk=True, lam=0.2,
    )

    # Walk-forward SF-MG-R. L_eps=None keeps the checkpoint's own L_rho
    # (bug #4). Generator is run once per rebalance with flat-extrapolated
    # cov/factor/α/β/σ over [rp, rp+rebal_freq).
    gen, cfg = load_sfmg(ckpt_path, N, n_factors, d_cov, None, device)
    f_in = np.vstack([np.zeros((1, n_factors)), f_arr[:-1]]) if cfg.get("lag_factors") else f_arr
    v9_segments = gen_paths_walk_forward(
        gen, cov_arr, f_in, alpha_hat, beta_hat, sigma_hat,
        rebal_points_abs=rebal_abs, rebal_freq=rebal_freq,
        n_paths=n_paths, device=device,
    )

    # Backtest per method using the per-rebalance segments.
    ew_daily = real_test.mean(axis=1)
    target_vol = ew_daily.std(ddof=1)

    def run(segments):
        w = np.zeros((n_rebal, N))
        for k, seg in enumerate(segments):
            # seg shape (n_paths, rebal_freq, N) — pool within-holding-period
            # returns across paths to estimate covariance at rebalance k.
            flat = seg.reshape(-1, N)
            cov = np.cov(flat.T)
            w[k] = gmv_long_only(cov)
        return backtest(w, real_test, rebal_freq)

    fb_res = run(fb_segments)
    v9_res = run(v9_segments)

    # Per-rebalance realised daily returns for the pooled Sharpe SE.
    daily_fb, daily_v9 = np.zeros(T), np.zeros(T)
    for segments, out_arr in [(fb_segments, daily_fb), (v9_segments, daily_v9)]:
        for k, rp in enumerate(rebal_points):
            flat = segments[k].reshape(-1, N)
            w = gmv_long_only(np.cov(flat.T))
            out_arr[rp:rp+rebal_freq] = real_test[rp:rp+rebal_freq] @ w

    return {"test_year": test_year, "n_rebal": n_rebal,
            "target_vol": float(target_vol),
            "FB+shrunk": fb_res, "SF-MG-R": v9_res,
            "daily_fb": daily_fb.tolist(), "daily_v9": daily_v9.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+",
                   default=["results/v9_roll2022/best_model.pt",
                             "results/v9_roll2023/best_model.pt",
                             "results/v9_s42/best_model.pt"])
    p.add_argument("--test_years", nargs="+", type=int, default=[2023, 2024, 2025])
    p.add_argument("--n_paths", type=int, default=200)
    p.add_argument("--rebal_freq", type=int, default=21)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/portfolio_rolling.json")
    args = p.parse_args()

    assert len(args.ckpts) == len(args.test_years)

    yearly = []
    for ckpt, year in zip(args.ckpts, args.test_years):
        yearly.append(evaluate_year(ckpt, year, n_paths=args.n_paths,
                                     rebal_freq=args.rebal_freq, seed=args.seed))

    # Pooled daily returns for unified SE
    fb_daily_all = np.concatenate([np.array(y["daily_fb"]) for y in yearly])
    v9_daily_all = np.concatenate([np.array(y["daily_v9"]) for y in yearly])

    def annualized(daily):
        return (daily.mean() * TRADING_DAYS,
                daily.std(ddof=1) * np.sqrt(TRADING_DAYS))

    fb_ann_r, fb_ann_v = annualized(fb_daily_all)
    v9_ann_r, v9_ann_v = annualized(v9_daily_all)
    fb_sr = fb_ann_r / fb_ann_v
    v9_sr = v9_ann_r / v9_ann_v
    T_pool = len(fb_daily_all)

    # Lo (2002) standard error (using daily observations)
    def lo_se(sr, T_days):
        return np.sqrt((1 + sr**2 / 2) / T_days) * np.sqrt(TRADING_DAYS)
    fb_se = lo_se(fb_sr, T_pool)
    v9_se = lo_se(v9_sr, T_pool)
    # Diff SE (assumes independent — conservative given correlated daily returns)
    paired_diff_daily = v9_daily_all - fb_daily_all
    diff_sr = paired_diff_daily.mean() / paired_diff_daily.std(ddof=1) * np.sqrt(TRADING_DAYS)
    diff_se = 1.0 / np.sqrt(T_pool) * np.sqrt(TRADING_DAYS)
    t_stat = diff_sr / diff_se

    print(f"\n{'='*72}\n  Rolling GMV portfolio — pooled across {len(yearly)} years ({T_pool} days)\n{'='*72}")
    print(f"  FB+shrunk:  Sharpe = {fb_sr:+.3f}  SE = {fb_se:.3f}   Ann.Ret {fb_ann_r:+.3f}  Ann.Vol {fb_ann_v:.3f}")
    print(f"  SF-MG-R:    Sharpe = {v9_sr:+.3f}  SE = {v9_se:.3f}   Ann.Ret {v9_ann_r:+.3f}  Ann.Vol {v9_ann_v:.3f}")
    print(f"  Paired diff (v9 - fb): daily mean*√252/σ = {diff_sr:+.3f}, t-stat ≈ {t_stat:+.2f}")

    print("\n  Per-year Sharpe:")
    for y in yearly:
        print(f"    {y['test_year']}: FB+shrunk={y['FB+shrunk']['sharpe']:+.3f}  "
              f"SF-MG-R={y['SF-MG-R']['sharpe']:+.3f}  "
              f"Δ={y['SF-MG-R']['sharpe']-y['FB+shrunk']['sharpe']:+.3f}")

    out = {
        "config": vars(args),
        "yearly": [{k: v for k, v in y.items() if k not in ("daily_fb", "daily_v9")}
                    for y in yearly],
        "pooled": {
            "n_days": int(T_pool),
            "FB+shrunk": {"sharpe": float(fb_sr), "lo_se": float(fb_se),
                           "ann_ret": float(fb_ann_r), "ann_vol": float(fb_ann_v)},
            "SF-MG-R": {"sharpe": float(v9_sr), "lo_se": float(v9_se),
                         "ann_ret": float(v9_ann_r), "ann_vol": float(v9_ann_v)},
            "paired_diff_sharpe": float(diff_sr),
            "paired_diff_se": float(diff_se),
            "paired_t_stat": float(t_stat),
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {args.out}")


if __name__ == "__main__":
    main()
