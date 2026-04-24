"""Asset-level paired bootstrap CI on 2025 OOS metrics — vectorised.

Instead of recomputing the 60 per-asset ACF statistics inside the inner
bootstrap loop (1000 replicates × 4 methods × 60 assets × ACF-100), we
precompute per-asset ``distance-to-real'' for each method once, then
bootstrap by resampling asset indices with replacement. Per-asset ACF/VC/
LEV are averaged over all generated paths so this tracks the main
``full_evaluation`` estimator (CODEX_REVIEW.md issue #7); the pre-fix
code used only ``paths[0]`` and therefore understated path-level MC
variance.

CorrFrob is still recomputed per replicate (60x60 corrcoef on resampled
columns) because the full correlation matrix changes with each asset draw.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import FactorBootstrap
from data_loader import ASSET_CLASSES, load_macro, load_returns, train_val_test_split
from evaluate import _acf, _cross_acf
from factors_pca import compute_hierarchical_pca_factors, compute_rolling_coefficients
from macro_processor import MacroProcessor
from sfmg_baseline import Generator
from sfmg_generator import SFMGGenerator
from eval_sfmg import gen_paths_sfmg, load_sfmg


def per_asset_acf(series_2d, max_lag=100):
    """series_2d: (T, N) → (N, max_lag). 1D ACF per asset."""
    T, N = series_2d.shape
    out = np.zeros((N, max_lag))
    for j in range(N):
        out[j] = _acf(series_2d[:, j], max_lag)
    return out


def per_asset_leverage(series_2d, max_lag=100):
    """Cross-corr lev profile between r_t and r^2_{t+k}. (N, max_lag)."""
    T, N = series_2d.shape
    out = np.zeros((N, max_lag))
    for j in range(N):
        out[j] = _cross_acf(series_2d[:, j], series_2d[:, j] ** 2, max_lag)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sfmg_ckpt", default="results/v9_s42/best_model.pt")
    p.add_argument("--baseline_ckpt", default="results/v7d/best_model.pt")
    p.add_argument("--n_paths", type=int, default=100)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--max_lag", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/bootstrap_ci.json")
    args = p.parse_args()

    device = torch.device("cpu")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

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
    f_arr = factors_df.values; n_factors = factors_df.shape[1]
    alpha_hat, beta_hat, sigma_hat = compute_rolling_coefficients(
        returns, factors_df, window=126)

    real_test = test_ret.values
    N = returns.shape[1]
    L_eps = np.load("data/residual_cholesky.npy")

    # ── Generate paths ─────────────────────────────────────────────────
    print("  FB ...")
    np.random.seed(args.seed)
    fb = FactorBootstrap(window=126); fb.fit(returns.values, f_arr)
    fb_paths = fb.generate(f_arr, start_idx=test_start_idx, n_paths=args.n_paths)

    print("  FB+shrunk ...")
    np.random.seed(args.seed)
    fb_s = FactorBootstrap(window=126); fb_s.fit(returns.values, f_arr)
    fb_s.fit_shrunk_residuals(returns.values[:test_start_idx],
                               f_arr[:test_start_idx], lam=0.2, L_eps=L_eps)
    fb_s_paths = fb_s.generate(f_arr, start_idx=test_start_idx, n_paths=args.n_paths)

    print("  fixed-η (v7d) ...")
    cfg7 = json.load(open(os.path.join(os.path.dirname(args.baseline_ckpt), "config.json")))
    g7 = Generator(n_assets=N, n_factors=n_factors, d_z=10, d_cov=d_cov,
                   hidden_dim=cfg7.get("hidden_dim", 256),
                   eta=cfg7.get("eta", 0.1), eta_sigma=cfg7.get("eta_sigma", 0.1),
                   residual_cholesky=L_eps).to(device)
    c7 = torch.load(args.baseline_ckpt, map_location=device, weights_only=False)
    g7.load_state_dict(c7["G"], strict=False); g7.eval()
    fixed_paths = gen_paths_sfmg(g7, cov_arr, f_arr, alpha_hat, beta_hat, sigma_hat,
                                     test_start_idx, n_paths=args.n_paths,
                                     t_l=real_test.shape[0], device=device)

    print("  SF-MarketGAN-R ...")
    g9, cfg9 = load_sfmg(args.sfmg_ckpt, N, n_factors, d_cov, L_eps, device)
    f_in = np.vstack([np.zeros((1, n_factors)), f_arr[:-1]]) if cfg9.get("lag_factors") else f_arr
    v9_paths = gen_paths_sfmg(g9, cov_arr, f_in, alpha_hat, beta_hat, sigma_hat,
                                  test_start_idx, n_paths=args.n_paths,
                                  t_l=real_test.shape[0], device=device)

    method_paths = {"FB": fb_paths, "FB+shrunk": fb_s_paths,
                    "fixed-η": fixed_paths, "SF-MG-R": v9_paths}

    # ── Precompute per-asset ACF / VC / LEV distances ─────────────────
    print(f"  Precomputing per-asset ACF/VC/LEV distances (max_lag={args.max_lag})...")
    real_acf = per_asset_acf(real_test, args.max_lag)
    real_vc  = per_asset_acf(real_test ** 2, args.max_lag)
    real_lev = per_asset_leverage(real_test, args.max_lag)

    per_asset_dist = {m: {} for m in method_paths}
    for name, paths in method_paths.items():
        # Average per-asset ACF / VC / LEV across all generated paths — same
        # estimator as ``full_evaluation`` uses for the paper main table.
        P, T, _ = paths.shape
        acf_accum = np.zeros((N, args.max_lag))
        vc_accum  = np.zeros((N, args.max_lag))
        lev_accum = np.zeros((N, args.max_lag))
        for p_idx in range(P):
            pp = paths[p_idx]
            acf_accum += per_asset_acf(pp, args.max_lag)
            vc_accum  += per_asset_acf(pp ** 2, args.max_lag)
            lev_accum += per_asset_leverage(pp, args.max_lag)
        acf_fake = acf_accum / P
        vc_fake  = vc_accum  / P
        lev_fake = lev_accum / P
        per_asset_dist[name]["ACF"] = np.linalg.norm(real_acf - acf_fake, axis=1)
        per_asset_dist[name]["VC"]  = np.linalg.norm(real_vc  - vc_fake,  axis=1)
        per_asset_dist[name]["LEV"] = np.linalg.norm(real_lev - lev_fake, axis=1)
        print(f"    {name}: per-asset ACF mean={per_asset_dist[name]['ACF'].mean():.3f}  "
              f"VC mean={per_asset_dist[name]['VC'].mean():.3f}  ({P} paths)")

    # ── Bootstrap ──────────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    B = args.n_bootstrap
    print(f"\n  Bootstrapping {B} replicates...")
    stats = {m: {k: np.zeros(B) for k in ["CorrFrob", "ACF", "VC", "LEV"]}
             for m in method_paths}

    # CorrFrob uses fake flattened (P*T, N)
    fake_flat = {m: paths.reshape(-1, N) for m, paths in method_paths.items()}

    for b in range(B):
        idx = rng.choice(N, N, replace=True)
        real_b = real_test[:, idx]
        corr_real = np.nan_to_num(np.corrcoef(real_b.T), nan=0.0)
        for m in method_paths:
            fake_b = fake_flat[m][:, idx]
            corr_fake = np.nan_to_num(np.corrcoef(fake_b.T), nan=0.0)
            stats[m]["CorrFrob"][b] = np.linalg.norm(corr_real - corr_fake, "fro")
            stats[m]["ACF"][b] = per_asset_dist[m]["ACF"][idx].mean()
            stats[m]["VC"][b]  = per_asset_dist[m]["VC"][idx].mean()
            stats[m]["LEV"][b] = per_asset_dist[m]["LEV"][idx].mean()
        if (b + 1) % 200 == 0:
            print(f"    {b+1}/{B}")

    # ── Aggregate + paired diffs ───────────────────────────────────────
    out = {"config": vars(args), "per_method": {}, "paired_diffs": {}}
    for name in method_paths:
        out["per_method"][name] = {}
        for k, arr in stats[name].items():
            out["per_method"][name][k] = {
                "mean": float(arr.mean()),
                "ci_lo": float(np.quantile(arr, 0.025)),
                "ci_hi": float(np.quantile(arr, 0.975)),
                "std": float(arr.std(ddof=1)),
            }

    print(f"\n  Paired 95% CI (SF-MG-R − baseline, negative = SF-MG-R better):")
    print(f"  {'Metric':<10}" + "".join(f"{b:>30s}" for b in ["FB", "FB+shrunk", "fixed-η"]))
    for k in ["CorrFrob", "ACF", "VC", "LEV"]:
        v9_arr = stats["SF-MG-R"][k]
        row = f"  {k:<10}"
        for base in ["FB", "FB+shrunk", "fixed-η"]:
            diff = v9_arr - stats[base][k]
            lo, hi = np.quantile(diff, [0.025, 0.975])
            p_neg = (diff < 0).mean()
            out["paired_diffs"][f"{k}_vs_{base}"] = {
                "mean": float(diff.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
                "p_better": float(p_neg),
            }
            row += f"   [{lo:+.3f},{hi:+.3f}] p={p_neg:.2f}  "
        print(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {args.out}")


if __name__ == "__main__":
    main()
