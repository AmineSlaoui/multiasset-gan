"""Re-run the VIX regime DiD (Section 4.7) for SF-MarketGAN-R.

The earlier paper version reported `79% of VIX effect captured` for a
different model; we regenerate the 2025 OOS paths from a v9 checkpoint and
recompute the DiD statistics so the paper text matches the method described.
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
from sfmg_generator import SFMGGenerator
from eval_sfmg import load_sfmg


def find_regime_windows(vix_z, thr_hi=0.3, thr_lo=-0.3, window=20):
    """Identify 20-day windows with |VIX_z| > threshold for most days."""
    T = len(vix_z)
    hi, lo = [], []
    for t in range(0, T - window + 1, window):
        seg = vix_z[t:t + window]
        mean_z = seg.mean()
        if mean_z > thr_hi:
            hi.append((t, t + window))
        elif mean_z < thr_lo:
            lo.append((t, t + window))
    return hi, lo


def avg_vol(arr, hi, lo):
    """For arr (T, N) or (P, T, N), return mean std per asset in each regime."""
    if arr.ndim == 3:
        # average across paths
        def vol_segments(segs):
            vs = []
            for p in range(arr.shape[0]):
                for (a, b) in segs:
                    vs.append(arr[p, a:b].std(axis=0))
            return np.mean(vs, axis=0)
        return vol_segments(hi), vol_segments(lo)
    else:
        def vol_segments(segs):
            vs = [arr[a:b].std(axis=0) for (a, b) in segs]
            return np.mean(vs, axis=0)
        return vol_segments(hi), vol_segments(lo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/v9_s42/best_model.pt")
    p.add_argument("--n_paths", type=int, default=100)
    p.add_argument("--out", default="results/counterfactual_v9.json")
    args = p.parse_args()
    device = torch.device("cpu")
    np.random.seed(42); torch.manual_seed(42)

    returns = load_returns(); macro = load_macro()
    train_ret, val_ret, test_ret = train_val_test_split(returns)
    test_start_idx = returns.index.get_loc(test_ret.index[0])
    train_end = str(train_ret.index[-1].date())

    proc = MacroProcessor(n_pca_components=8, n_regimes=3)
    covariates = proc.fit_transform(macro, returns.index, train_end=train_end)
    cov_arr = covariates.reindex(returns.index).fillna(0).values

    factors_df, _ = compute_hierarchical_pca_factors(
        returns, ASSET_CLASSES, n_global=5, n_class=2, train_end=train_end)
    f_arr = factors_df.values
    n_factors = factors_df.shape[1]
    alpha_hat, beta_hat, sigma_hat = compute_rolling_coefficients(
        returns, factors_df, window=126)

    real_test = test_ret.values
    n_assets = returns.shape[1]
    T = len(test_ret)
    d_cov = covariates.shape[1]

    # VIX z-score on the test window (vix_level_z is column index 11)
    vix_idx = list(covariates.columns).index("vix_level_z")
    vix_z_test = cov_arr[test_start_idx:test_start_idx + T, vix_idx]

    hi, lo = find_regime_windows(vix_z_test, thr_hi=0.3, thr_lo=-0.3, window=20)
    print(f"  High-VIX windows (|z>0.3|): {len(hi)}, Low-VIX windows (z<-0.3): {len(lo)}")

    L_eps = np.load("data/residual_cholesky.npy")

    # FB baseline
    np.random.seed(42)
    fb = FactorBootstrap(window=126); fb.fit(returns.values, f_arr)
    fb_paths = fb.generate(f_arr, start_idx=test_start_idx, n_paths=args.n_paths)

    # SF-MarketGAN-R
    gen, cfg = load_sfmg(args.ckpt, n_assets, n_factors, d_cov, L_eps, device)
    f_in = f_arr
    if cfg.get("lag_factors", False):
        f_in = np.vstack([np.zeros((1, n_factors)), f_arr[:-1]])
    # Use the same helper as eval_v9 for path generation
    from eval_sfmg import gen_paths_sfmg
    v9_paths = gen_paths_sfmg(gen, cov_arr, f_in, alpha_hat, beta_hat, sigma_hat,
                                  test_start_idx, n_paths=args.n_paths, t_l=T, device=device)

    # Compute volatilities
    real_hi, real_lo = avg_vol(real_test, hi, lo)
    fb_hi, fb_lo = avg_vol(fb_paths, hi, lo)
    v9_hi, v9_lo = avg_vol(v9_paths, hi, lo)

    # Average across assets
    def regime_effect(vol_hi, vol_lo):
        return (vol_hi.mean() - vol_lo.mean()) / vol_lo.mean()

    real_effect = regime_effect(real_hi, real_lo)
    fb_effect = regime_effect(fb_hi, fb_lo)
    v9_effect = regime_effect(v9_hi, v9_lo)

    fb_capture = fb_effect / real_effect
    v9_capture = v9_effect / real_effect

    print(f"\n  Real 2025 high/low-VIX vol effect:    {real_effect*100:+.1f}%")
    print(f"  Factor bootstrap captures:           {fb_effect*100:+.1f}%  ({fb_capture*100:.0f}% of real)")
    print(f"  SF-MarketGAN-R captures:             {v9_effect*100:+.1f}%  ({v9_capture*100:.0f}% of real)")

    # Also compute cross-window correlation tracking:
    # per-window average vol, correlation across windows
    def window_vol_series(arr, windows):
        if arr.ndim == 3:
            return np.array([[arr[:, a:b].std(axis=1).mean() for (a, b) in windows]]).flatten()
        else:
            return np.array([arr[a:b].std(axis=0).mean() for (a, b) in windows])
    all_wins = hi + lo
    all_wins_sorted = sorted(all_wins, key=lambda x: x[0])
    if len(all_wins_sorted) >= 3:
        real_series = window_vol_series(real_test, all_wins_sorted)
        fb_series = window_vol_series(fb_paths, all_wins_sorted)
        v9_series = window_vol_series(v9_paths, all_wins_sorted)
        corr_fb = np.corrcoef(real_series, fb_series)[0, 1]
        corr_v9 = np.corrcoef(real_series, v9_series)[0, 1]
    else:
        corr_fb = corr_v9 = float("nan")
    print(f"\n  Cross-window vol-tracking correlation with real:")
    print(f"    FB:            {corr_fb:.3f}")
    print(f"    SF-MarketGAN-R: {corr_v9:.3f}")

    out = {
        "config": {"n_paths": args.n_paths, "vix_thr_hi": 0.3, "vix_thr_lo": -0.3,
                    "window": 20, "test_start": str(test_ret.index[0].date())},
        "n_hi_windows": len(hi), "n_lo_windows": len(lo),
        "real_effect": float(real_effect),
        "fb_effect": float(fb_effect), "v9_effect": float(v9_effect),
        "fb_capture_fraction": float(fb_capture), "v9_capture_fraction": float(v9_capture),
        "cross_window_tracking_corr": {"FB": float(corr_fb), "v9": float(corr_v9)},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {args.out}")


if __name__ == "__main__":
    main()
