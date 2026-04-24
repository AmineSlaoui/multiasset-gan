"""Pairwise correlation analysis for selected asset pairs."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DATA = "/Users/fengyang/Desktop/gan/data/all_assets_log_returns.csv"
OUT = "/Users/fengyang/Desktop/gan/figures/pair_corr.png"

PAIRS = [("v", "ma", "Visa vs Mastercard"),
         ("gold", "copper", "Gold vs Copper")]
ROLL = 126  # ~6 months of trading days


def tail_dep(x, y, q=0.05):
    lo_x, lo_y = np.quantile(x, q), np.quantile(y, q)
    hi_x, hi_y = np.quantile(x, 1 - q), np.quantile(y, 1 - q)
    lower = np.mean((x <= lo_x) & (y <= lo_y)) / q
    upper = np.mean((x >= hi_x) & (y >= hi_y)) / q
    return lower, upper


def main():
    df = pd.read_csv(DATA, parse_dates=["Date"]).set_index("Date")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ["steelblue", "darkred"]

    for (a, b, title), color in zip(PAIRS, colors):
        x = df[a].astype(float)
        y = df[b].astype(float)
        mask = (x != 0) & (y != 0)
        x, y = x[mask], y[mask]
        roll = x.rolling(ROLL).corr(y)
        full = stats.pearsonr(x, y)[0]
        ax.plot(roll.index, roll.values, color=color, lw=1.2,
                label=f"{title}  (full={full:.3f})")
        print(f"{title}: n={len(x)}  pearson={full:.4f}  "
              f"rolling[min,med,max]=[{roll.min():.3f},{roll.median():.3f},{roll.max():.3f}]")

    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_ylim(-0.5, 1.0)
    ax.set_ylabel(f"{ROLL}-day rolling Pearson corr")
    ax.set_xlabel("Date")
    ax.set_title("Pairwise rolling correlation vs time")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
