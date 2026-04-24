"""Rolling correlation vs timestep: Real vs Generated for selected pairs.

Style matches the reference figure (window=63, concatenate N generated paths
across the test horizon, tile real test returns N times, plot both curves).
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import ASSET_CLASSES, load_macro, load_returns, train_val_test_split
from factors_pca import compute_hierarchical_pca_factors, compute_rolling_coefficients
from macro_processor import MacroProcessor
from eval_sfmg import gen_paths_sfmg, load_sfmg

PAIRS = [("v", "ma", "Visa (v) vs Mastercard (ma)"),
         ("gold", "copper", "Gold vs Copper")]
WINDOW = 63


def rolling_corr_1d(x, y, w):
    """O(N) rolling Pearson correlation."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    sx = np.concatenate([[0.0], np.cumsum(x)])
    sy = np.concatenate([[0.0], np.cumsum(y)])
    sxx = np.concatenate([[0.0], np.cumsum(x * x)])
    syy = np.concatenate([[0.0], np.cumsum(y * y)])
    sxy = np.concatenate([[0.0], np.cumsum(x * y)])
    n = len(x)
    out = np.full(n, np.nan)
    for t in range(w - 1, n):
        s, e = t - w + 1, t + 1
        mx = (sx[e] - sx[s]) / w
        my = (sy[e] - sy[s]) / w
        vx = (sxx[e] - sxx[s]) / w - mx * mx
        vy = (syy[e] - syy[s]) / w - my * my
        cxy = (sxy[e] - sxy[s]) / w - mx * my
        denom = np.sqrt(max(vx, 0) * max(vy, 0))
        out[t] = cxy / denom if denom > 0 else np.nan
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/v9_s42/best_model.pt")
    p.add_argument("--n_paths", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--t_l", type=int, default=251)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cpu")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    returns = load_returns(); macro = load_macro()
    train_ret, val_ret, test_ret = train_val_test_split(returns)
    train_end = str(train_ret.index[-1].date())
    asset_list = returns.columns.tolist()

    if args.split == "test":
        target_ret = test_ret
        target_start_idx = returns.index.get_loc(test_ret.index[0])
    elif args.split == "val":
        target_ret = val_ret
        target_start_idx = returns.index.get_loc(val_ret.index[0])
    else:  # train — use the LAST t_l days of train so rolling coefs are warm
        train_last_idx = returns.index.get_loc(train_ret.index[-1])
        target_start_idx = train_last_idx - args.t_l + 1
        target_ret = returns.iloc[target_start_idx: target_start_idx + args.t_l]
    print(f"[split={args.split}] start={target_ret.index[0].date()} "
          f"end={target_ret.index[-1].date()}  n={len(target_ret)}")
    if args.out is None:
        args.out = f"figures/pair_corr_gen_{args.split}.png"

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

    n_assets = returns.shape[1]
    L_eps = np.load("data/residual_cholesky.npy") if os.path.exists("data/residual_cholesky.npy") else None

    print(f"[load] {args.ckpt}")
    gen, cfg = load_sfmg(args.ckpt, n_assets, n_factors, d_cov, L_eps, device)
    f_in = f_arr
    if cfg.get("lag_factors", False):
        f_in = np.vstack([np.zeros((1, n_factors)), f_arr[:-1]])

    print(f"[sample] n_paths={args.n_paths}  (CPU)")
    paths = gen_paths_sfmg(
        gen, cov_arr, f_in, alpha_hat, beta_hat, sigma_hat,
        target_start_idx, n_paths=args.n_paths, t_l=args.t_l, batch=10, device=device)
    # paths shape: (n_paths, T_l, N)
    print(f"[sample] shape={paths.shape}")

    real_target = target_ret.values      # (T_l, N)
    T_l = min(real_target.shape[0], paths.shape[1])
    real_target = real_target[:T_l]
    paths = paths[:, :T_l]

    # Concatenate across paths
    gen_concat = paths.reshape(-1, n_assets)              # (N_paths * T_l, N)
    real_tiled = np.tile(real_target, (args.n_paths, 1))  # (N_paths * T_l, N)
    print(f"[concat] total steps = {gen_concat.shape[0]}")

    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(14, 3.2 * len(PAIRS)))
    if len(PAIRS) == 1:
        axes = [axes]
    for ax, (a, b, title) in zip(axes, PAIRS):
        ia, ib = asset_list.index(a), asset_list.index(b)
        r_roll = rolling_corr_1d(real_tiled[:, ia], real_tiled[:, ib], WINDOW)
        g_roll = rolling_corr_1d(gen_concat[:, ia], gen_concat[:, ib], WINDOW)
        ax.plot(r_roll, color="blue", lw=0.8, label="Real")
        ax.plot(g_roll, color="red", ls="--", lw=0.8, label="Generated")
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Timestep (test set)")
        ax.set_ylabel("Correlation")
        ax.set_title(f"Rolling Correlation: {title} (window={WINDOW}d) [{args.split.upper()} SET]")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left")
        rm = np.nanmean((r_roll - g_roll) ** 2) ** 0.5
        print(f"  {title}: real_mean={np.nanmean(r_roll):.3f}  "
              f"gen_mean={np.nanmean(g_roll):.3f}  rmse={rm:.3f}")

    fig.suptitle(f"v9_s42 ({args.n_paths} paths, split={args.split}) — "
                 f"Rolling Correlation vs Timestep", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
