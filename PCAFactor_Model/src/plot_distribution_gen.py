"""Distribution comparison: real 60-asset returns vs v9_final synthetic.

Generates N paths from results/v9_final over the OOS test window, pools
them to match the real panel volume, and plots the same 3-panel figure
as plot_distribution_60.py but with real vs generated overlaid.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import stats

from data_loader import ASSET_CLASSES, load_macro, load_returns, train_val_test_split
from factors_pca import compute_rolling_coefficients
from factors_sector import compute_sector_factors
from macro_processor import MacroProcessor
from eval_sfmg import gen_paths_sfmg, load_sfmg


CLASS_COLORS = {
    "bonds":       "#2E86AB",
    "commodities": "#E09F3E",
    "stocks":      "#C1272D",
}
REAL_COLOR = "#222222"
GEN_COLOR  = "#D62728"


def generate_v9_paths(n_paths: int = 100, seed: int = 42):
    """Run the standard v9 generation pipeline and return (real_test, gen_paths, asset_list)."""
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    np.random.seed(seed); torch.manual_seed(seed)

    returns = load_returns(); macro = load_macro()
    train_ret, val_ret, test_ret = train_val_test_split(returns)
    test_start_idx = returns.index.get_loc(test_ret.index[0])
    train_end = str(train_ret.index[-1].date())

    proc = MacroProcessor(n_pca_components=8, n_regimes=3)
    covariates = proc.fit_transform(macro, returns.index, train_end=train_end)
    cov_arr = covariates.reindex(returns.index).fillna(0).values
    d_cov = covariates.shape[1]

    factors_df, _ = compute_sector_factors(returns, ASSET_CLASSES, train_end=train_end)
    f_arr = factors_df.values
    n_factors = factors_df.shape[1]
    alpha_hat, beta_hat, sigma_hat = compute_rolling_coefficients(
        returns, factors_df, window=126)

    # Recompute L_eps the same way eval_v9 does for v3
    tr_end = returns.index.get_loc(train_ret.index[-1]) + 1
    res = np.zeros_like(returns.values)
    for t in range(tr_end):
        res[t] = returns.values[t] - alpha_hat[t] - beta_hat[t] @ f_arr[t]
    res_tr = res[126:tr_end]
    Ccorr = np.corrcoef(res_tr.T)
    w, V = np.linalg.eigh(0.5 * (Ccorr + Ccorr.T))
    w = np.clip(w, 1e-6, None)
    Cpsd = (V * w) @ V.T
    dd = np.sqrt(np.diag(Cpsd))
    Cnorm = Cpsd / np.outer(dd, dd)
    L_eps = np.linalg.cholesky(Cnorm + 1e-6 * np.eye(Cnorm.shape[0]))

    ckpt = "results/v9_final/best_model.pt"
    gen, cfg = load_sfmg(ckpt, returns.shape[1], n_factors, d_cov, L_eps, device)
    f_in = f_arr
    if cfg.get("lag_factors", False):
        f_in = np.vstack([np.zeros((1, n_factors)), f_arr[:-1]])

    print(f"  generating {n_paths} paths on {device}...")
    paths = gen_paths_sfmg(gen, cov_arr, f_in, alpha_hat, beta_hat, sigma_hat,
                               test_start_idx, n_paths=n_paths, t_l=252, device=device)
    print(f"  gen shape = {paths.shape}  (n_paths, T, N)")

    return test_ret.values, paths, returns.columns.tolist()


def main(n_paths: int = 100, out_path: str = "figures/distribution_gen_vs_real.png") -> None:
    real_test, gen_paths, asset_list = generate_v9_paths(n_paths=n_paths)
    # pool real over full history for a fair distribution comparison (volume)
    real_full = load_returns()
    print(f"  real full: {real_full.shape},  real test: {real_test.shape},  "
          f"gen pooled: {gen_paths.reshape(-1, gen_paths.shape[-1]).shape}")

    real_pool = real_full.values                             # (T, N)
    gen_pool  = gen_paths.reshape(-1, gen_paths.shape[-1])   # (n_paths*T_oos, N)

    # ── sort assets within class by real std ──────────────────────────
    order, class_of = [], {}
    for cls in ["bonds", "commodities", "stocks"]:
        members = [a for a in ASSET_CLASSES[cls] if a in asset_list]
        members.sort(key=lambda a: real_pool[:, asset_list.index(a)].std())
        order.extend(members)
        for a in members:
            class_of[a] = cls
    idx_of = {a: asset_list.index(a) for a in asset_list}

    real_data = [real_pool[:, idx_of[a]] for a in order]
    gen_data  = [gen_pool[:,  idx_of[a]] for a in order]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0],
                          hspace=0.40, wspace=0.22)

    # ═══ Top: paired violins (real left-half, gen right-half) ═════════
    ax = fig.add_subplot(gs[0, :])
    positions = np.arange(len(order))
    # real on left-half
    pv_real = ax.violinplot(real_data, positions=positions - 0.22, widths=0.42,
                            showmeans=False, showmedians=False, showextrema=False)
    for body, a in zip(pv_real["bodies"], order):
        body.set_facecolor(CLASS_COLORS[class_of[a]])
        body.set_alpha(0.85)
        body.set_edgecolor("black")
        body.set_linewidth(0.4)
        # clip to left half
        m = np.mean(body.get_paths()[0].vertices[:, 0])
        body.get_paths()[0].vertices[:, 0] = np.clip(
            body.get_paths()[0].vertices[:, 0], -np.inf, m)

    # generated on right-half (same class color but lighter fill)
    pv_gen = ax.violinplot(gen_data, positions=positions + 0.22, widths=0.42,
                           showmeans=False, showmedians=False, showextrema=False)
    for body, a in zip(pv_gen["bodies"], order):
        body.set_facecolor("white")
        body.set_edgecolor(CLASS_COLORS[class_of[a]])
        body.set_alpha(0.95)
        body.set_linewidth(1.2)
        m = np.mean(body.get_paths()[0].vertices[:, 0])
        body.get_paths()[0].vertices[:, 0] = np.clip(
            body.get_paths()[0].vertices[:, 0], m, np.inf)

    # class separators
    n_bonds = len(ASSET_CLASSES["bonds"])
    n_com   = len(ASSET_CLASSES["commodities"])
    for xline in [n_bonds - 0.5, n_bonds + n_com - 0.5]:
        ax.axvline(xline, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels([a.upper() for a in order], rotation=75, fontsize=7)
    ax.set_ylabel("Daily log-return", fontsize=11)
    ax.set_title("Real (filled) vs v9_final generated (outline) — 60-asset daily log-returns",
                 fontsize=13, fontweight="bold", pad=22)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    ymax = max(abs(np.quantile(real_pool.flatten(), 0.001)),
               abs(np.quantile(real_pool.flatten(), 0.999))) * 1.10
    ax.set_ylim(-ymax, ymax)

    centers = [n_bonds / 2 - 0.5,
               n_bonds + n_com / 2 - 0.5,
               n_bonds + n_com + len(ASSET_CLASSES["stocks"]) / 2 - 0.5]
    trans = ax.get_xaxis_transform()
    for cx, label in zip(centers, ["Bonds (20)", "Commodities (20)", "Stocks (20)"]):
        ax.text(cx, 1.01, label, transform=trans,
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=CLASS_COLORS[label.split()[0].lower()])

    # legend hint
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="lightgrey", edgecolor="black", label="Real (filled)"),
        Patch(facecolor="white", edgecolor="black", label="Generated (outline)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.95)

    # ═══ Bottom-left: pooled histogram per class, real solid vs gen dashed ═══
    ax_kde = fig.add_subplot(gs[1, 0])
    bins = np.linspace(-0.08, 0.08, 121)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for cls in ["bonds", "commodities", "stocks"]:
        members = [a for a in ASSET_CLASSES[cls] if a in asset_list]
        idxs = [idx_of[a] for a in members]
        real_c = real_pool[:, idxs].flatten()
        gen_c  = gen_pool[:,  idxs].flatten()
        h_r, _ = np.histogram(real_c, bins=bins, density=True)
        h_g, _ = np.histogram(gen_c,  bins=bins, density=True)
        c = CLASS_COLORS[cls]
        ax_kde.plot(centers, h_r, color=c, linewidth=2.0, linestyle="-",
                    label=f"{cls.capitalize()} real "
                          f"(σ={real_c.std():.4f}, k={stats.kurtosis(real_c):.1f})")
        ax_kde.plot(centers, h_g, color=c, linewidth=1.6, linestyle="--",
                    label=f"{cls.capitalize()} gen  "
                          f"(σ={gen_c.std():.4f}, k={stats.kurtosis(gen_c):.1f})")
    ax_kde.set_yscale("log")
    ax_kde.set_xlabel("Daily log-return")
    ax_kde.set_ylabel("Density (log)")
    ax_kde.set_title("Pooled return density — real (solid) vs generated (dashed)",
                     fontsize=11, fontweight="bold")
    ax_kde.legend(fontsize=7, loc="lower center", framealpha=0.9, ncol=1)
    ax_kde.grid(alpha=0.3, which="both")
    ax_kde.set_xlim(-0.08, 0.08)
    ax_kde.set_ylim(1e-1, 5e2)

    # ═══ Bottom-right: Q-Q real vs generated (pooled) ═════════════════
    ax_qq = fig.add_subplot(gs[1, 1])
    # sample equal-size quantile grid on both
    n_grid = 1000
    probs = (np.arange(1, n_grid + 1) - 0.5) / n_grid
    for cls in ["bonds", "commodities", "stocks"]:
        members = [a for a in ASSET_CLASSES[cls] if a in asset_list]
        idxs = [idx_of[a] for a in members]
        real_c = real_pool[:, idxs].flatten()
        gen_c  = gen_pool[:,  idxs].flatten()
        # standardise by real moments so scale differences show up
        mu, sd = real_c.mean(), real_c.std()
        real_q = np.quantile((real_c - mu) / sd, probs)
        gen_q  = np.quantile((gen_c  - mu) / sd, probs)
        ax_qq.scatter(real_q, gen_q, s=6, alpha=0.7, color=CLASS_COLORS[cls],
                      label=cls.capitalize())
    lim = 8
    ax_qq.plot([-lim, lim], [-lim, lim], color="black",
               linestyle="--", linewidth=1, alpha=0.7, label="identity")
    ax_qq.set_xlim(-lim, lim); ax_qq.set_ylim(-lim, lim)
    ax_qq.set_xlabel("Real quantile (standardised)")
    ax_qq.set_ylabel("Generated quantile (standardised)")
    ax_qq.set_title("Q-Q: generated vs real (pooled by class)",
                    fontsize=11, fontweight="bold")
    ax_qq.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax_qq.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"  saved → {out_path}")

    # summary
    print("\n  pooled summary (std | kurtosis):")
    print(f"    real: σ={real_pool.std():.4f}, "
          f"kurt={stats.kurtosis(real_pool.flatten()):.2f}")
    print(f"    gen : σ={gen_pool.std():.4f}, "
          f"kurt={stats.kurtosis(gen_pool.flatten()):.2f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_paths", type=int, default=100)
    p.add_argument("--out", default="figures/distribution_gen_vs_real.png")
    args = p.parse_args()
    main(n_paths=args.n_paths, out_path=args.out)
