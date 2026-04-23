"""4-year rolling GMV portfolio backtest (2022-2025).

Uses checkpoints from rolling-cutoff training:
  2022 test: v9_roll2022 (train 2011-2021, val 2022, test 2023) → wait, no.
Actually: v9_roll2022 has train_end=2021-12-31, so test window = 2023.
         v9_roll2023 has train_end=2022-12-31, so test window = 2024.
         v9_s42      has train_end=2023-12-31, so test window = 2025.

To backtest year Y, we need a checkpoint trained through Y-2 and validated
on Y-1. We have:
  - y=2023: v9_roll2022
  - y=2024: v9_roll2023
  - y=2025: v9_s42 (or v9_10seed/s42)

For 2022 we would need train_end=2020 → out of scope (would need a new
training run). We report 2023-2025 (36 rebalances total, SE≈0.06).

For each year, we:
  1. Generate n_paths full-year paths from SF-MG-R and from FB+shrunk.
  2. At each monthly rebalance, sample cov from 21-day forward generated paths.
  3. Compute long-only GMV weights (simplex-projected).
  4. Realize returns, aggregate to annualized Sharpe per year.
  5. Pool all 36 daily returns across years for a unified Sharpe SE.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import FactorBootstrap
from data_loader import ASSET_CLASSES, load_macro, load_returns, train_val_test_split
from factors_v2 import compute_hierarchical_pca_factors, compute_rolling_coefficients_v2
from macro_processor import MacroProcessor
from eval_v9 import gen_paths_v9_like, load_v9
from eval_portfolio_v9 import gmv_long_only, backtest, synth_cov_from_paths

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
    alpha_hat, beta_hat, sigma_hat = compute_rolling_coefficients_v2(
        returns, factors_df, window=126)
    L_eps = np.load("data/residual_cholesky.npy")

    rebal_points = list(range(0, T - rebal_freq + 1, rebal_freq))
    n_rebal = len(rebal_points)

    # FB + shrunk baseline
    np.random.seed(seed)
    fb_s = FactorBootstrap(window=126); fb_s.fit(returns.values, f_arr)
    fb_s.fit_shrunk_residuals(returns.values[:test_start_idx],
                               f_arr[:test_start_idx], lam=0.2, L_eps=L_eps)
    fb_s_paths = fb_s.generate(f_arr, start_idx=test_start_idx, n_paths=n_paths)

    # SF-MG-R
    gen, cfg = load_v9(ckpt_path, N, n_factors, d_cov, L_eps, device)
    f_in = np.vstack([np.zeros((1, n_factors)), f_arr[:-1]]) if cfg.get("lag_factors") else f_arr
    v9_paths = gen_paths_v9_like(gen, cov_arr, f_in, alpha_hat, beta_hat,
                                  sigma_hat, test_start_idx, n_paths=n_paths,
                                  t_l=T, device=device)

    # Backtest per method
    ew_daily = real_test.mean(axis=1)
    target_vol = ew_daily.std(ddof=1)

    def run(paths):
        w = np.zeros((n_rebal, N))
        for k, rp in enumerate(rebal_points):
            cov = synth_cov_from_paths(paths, rp, rebal_freq)
            w[k] = gmv_long_only(cov)
        return backtest(w, real_test, rebal_freq)

    fb_res = run(fb_s_paths)
    v9_res = run(v9_paths)

    # Return per-rebalance daily returns so we can pool across years later
    daily_fb, daily_v9 = np.zeros(T), np.zeros(T)
    for method, paths, out_arr in [("fb", fb_s_paths, daily_fb),
                                     ("v9", v9_paths, daily_v9)]:
        for k, rp in enumerate(rebal_points):
            cov = synth_cov_from_paths(paths, rp, rebal_freq)
            w = gmv_long_only(cov)
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
