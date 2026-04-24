"""Distribution figure for the 60-asset panel.

Top panel: violin plot of daily log returns for each asset, grouped
by class (bonds / commodities / stocks), sorted within class by std.
Bottom-left: pooled KDE per class on a common axis.
Bottom-right: Q-Q of the pooled panel vs a Gaussian with matched moments
            (demonstrates the fat tails that motivate the GAN).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

from data_loader import load_returns, ASSET_CLASSES


CLASS_COLORS = {
    "bonds":       "#2E86AB",
    "commodities": "#E09F3E",
    "stocks":      "#C1272D",
}


def _asset_class(name: str) -> str:
    for cls, members in ASSET_CLASSES.items():
        if name in members:
            return cls
    return "unknown"


def main(out_path: str = "figures/distribution_60assets.png") -> None:
    r = load_returns()
    print(f"returns: {r.shape}, {r.index[0].date()} → {r.index[-1].date()}")

    # ── sort assets within class by std (ascending) for a clean readable x-axis ──
    order, class_of = [], {}
    for cls in ["bonds", "commodities", "stocks"]:
        members = [a for a in ASSET_CLASSES[cls] if a in r.columns]
        members.sort(key=lambda a: r[a].std())
        order.extend(members)
        for a in members:
            class_of[a] = cls

    data = [r[a].values for a in order]
    colors = [CLASS_COLORS[class_of[a]] for a in order]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0],
                          hspace=0.35, wspace=0.22)

    # ═══ Top: per-asset violin ════════════════════════════════════════
    ax = fig.add_subplot(gs[0, :])
    parts = ax.violinplot(data, positions=np.arange(len(order)),
                          showmeans=False, showmedians=False,
                          showextrema=False, widths=0.85)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_edgecolor("black")
        body.set_alpha(0.75)
        body.set_linewidth(0.4)

    # 5th/95th percentile whiskers
    q05 = np.array([np.quantile(d, 0.05) for d in data])
    q95 = np.array([np.quantile(d, 0.95) for d in data])
    med = np.array([np.median(d) for d in data])
    x = np.arange(len(order))
    ax.vlines(x, q05, q95, color="black", linewidth=0.8, alpha=0.9)
    ax.scatter(x, med, s=8, color="white", edgecolor="black",
               linewidth=0.5, zorder=3)

    # class separator lines
    n_bonds = len(ASSET_CLASSES["bonds"])
    n_com   = len(ASSET_CLASSES["commodities"])
    for xline in [n_bonds - 0.5, n_bonds + n_com - 0.5]:
        ax.axvline(xline, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in order], rotation=75, fontsize=7)
    ax.set_ylabel("Daily log-return", fontsize=11)
    ax.set_title("Distribution of daily log-returns across the 60-asset panel  "
                 f"(2011-01 → 2025-12, T = {len(r)} days)",
                 fontsize=13, fontweight="bold", pad=22)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    # clip y so extreme outliers don't compress the violins
    ymax = max(abs(np.quantile(r.values.flatten(), 0.001)),
               abs(np.quantile(r.values.flatten(), 0.999))) * 1.05
    ax.set_ylim(-ymax, ymax)

    # class labels across the top (axes-fraction y so position is stable)
    centers = [n_bonds / 2 - 0.5,
               n_bonds + n_com / 2 - 0.5,
               n_bonds + n_com + len(ASSET_CLASSES["stocks"]) / 2 - 0.5]
    trans = ax.get_xaxis_transform()  # x in data, y in axes-fraction
    for cx, label in zip(centers, ["Bonds (20)", "Commodities (20)", "Stocks (20)"]):
        ax.text(cx, 1.01, label, transform=trans,
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=CLASS_COLORS[label.split()[0].lower()])

    # ═══ Bottom-left: pooled KDE per class ════════════════════════════
    ax_kde = fig.add_subplot(gs[1, 0])
    grid = np.linspace(-0.08, 0.08, 600)
    for cls in ["bonds", "commodities", "stocks"]:
        members = [a for a in ASSET_CLASSES[cls] if a in r.columns]
        pooled = r[members].values.flatten()
        kde = stats.gaussian_kde(pooled, bw_method=0.15)
        ax_kde.plot(grid, kde(grid), color=CLASS_COLORS[cls], linewidth=2.2,
                    label=f"{cls.capitalize()}  (σ={pooled.std():.4f}, "
                          f"kurt={stats.kurtosis(pooled):.1f})")
        ax_kde.fill_between(grid, kde(grid), alpha=0.15,
                            color=CLASS_COLORS[cls])
    ax_kde.set_yscale("log")
    ax_kde.set_xlabel("Daily log-return")
    ax_kde.set_ylabel("Density (log)")
    ax_kde.set_title("Pooled return density by asset class", fontsize=11,
                     fontweight="bold")
    ax_kde.legend(fontsize=9, loc="lower center", framealpha=0.9)
    ax_kde.grid(alpha=0.3)
    ax_kde.set_xlim(-0.08, 0.08)

    # ═══ Bottom-right: Q-Q pooled vs Gaussian ═════════════════════════
    ax_qq = fig.add_subplot(gs[1, 1])
    for cls in ["bonds", "commodities", "stocks"]:
        members = [a for a in ASSET_CLASSES[cls] if a in r.columns]
        pooled = r[members].values.flatten()
        # standardise
        z = (pooled - pooled.mean()) / pooled.std()
        # theoretical quantiles
        n = len(z)
        probs = (np.arange(1, n + 1) - 0.5) / n
        theo = stats.norm.ppf(probs)
        emp = np.sort(z)
        # subsample to keep the plot crisp
        idx = np.linspace(0, n - 1, 2000).astype(int)
        ax_qq.scatter(theo[idx], emp[idx], s=4, alpha=0.55,
                      color=CLASS_COLORS[cls], label=cls.capitalize())
    lim = max(abs(ax_qq.get_xlim()[0]), abs(ax_qq.get_xlim()[1]), 6)
    ax_qq.plot([-lim, lim], [-lim, lim], color="black",
               linestyle="--", linewidth=1, alpha=0.7, label="Gaussian")
    ax_qq.set_xlim(-lim, lim)
    ax_qq.set_ylim(-lim, lim)
    ax_qq.set_xlabel("Gaussian quantile")
    ax_qq.set_ylabel("Empirical quantile (standardised)")
    ax_qq.set_title("Q-Q: pooled standardised returns vs Gaussian",
                    fontsize=11, fontweight="bold")
    ax_qq.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax_qq.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved → {out_path}")

    # quick numerical summary
    pooled_all = r.values.flatten()
    print("\npanel summary:")
    print(f"  mean = {pooled_all.mean():+.5f}")
    print(f"  std  = {pooled_all.std():.5f}")
    print(f"  skew = {stats.skew(pooled_all):+.3f}")
    print(f"  kurt = {stats.kurtosis(pooled_all):.2f}  (Gaussian = 0)")


if __name__ == "__main__":
    main()
