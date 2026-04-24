"""Stylized facts bar chart: compare models on ACF / vol-clustering /
leverage / full-correlation Frobenius metrics.

Lower = closer to real data on all four metrics.
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt

# ── Source OOS eval JSONs ──────────────────────────────────────────────
MODELS = [
    ("v9_s42 orig",          "results/v9_s42/oos_results.json",       "v9_s42"),
    ("v9 + L_ρ swap",        "results/v9_oos_fixedL.json",            "v9_s42"),
    ("v9 trainL_v2",         "results/v9_trainL_v2_oos.json",         "v9_s42_trainL_v2"),
    ("v9 + v3_GS (final)",   "results/v9_v3gs_oos.json",              "v9_s42_v3_gs"),
]
# Use baseline metrics from the most recent/clean run for apples-to-apples
# comparison reference (v3 run has FB/FB+shrunk on v3 factors — closest
# baseline for the final model).
BASELINE_JSON = "results/v9_v3gs_oos.json"

METRICS = [
    ("full_corr_frob", "Full 60×60 corr Frob. distance"),
    ("acf_score",      "Autocorrelation score (SF)"),
    ("vc_score",       "Vol-clustering score (SF)"),
    ("lev_score",      "Leverage effect score (SF)"),
]


def load_metric(path: str, key: str, metric: str) -> float:
    d = json.load(open(path))
    return d["metrics"][key].get(metric, float("nan"))


def main():
    rows = [(label, path, key) for label, path, key in MODELS]
    baseline = json.load(open(BASELINE_JSON))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    x = np.arange(len(rows))
    colors = ["#888888", "#3b75b0", "#d28535", "#b32224"]   # grey → final: red

    for ax, (metric, title) in zip(axes, METRICS):
        vals = [load_metric(p, k, metric) for _, p, k in rows]
        fb_val = baseline["metrics"]["FB"].get(metric, float("nan"))
        fbs_val = baseline["metrics"]["FB+shrunk"].get(metric, float("nan"))
        bars = ax.bar(x, vals, color=colors, edgecolor="black", lw=0.6)
        ax.axhline(fb_val, color="gray", linestyle="--", lw=1.0, alpha=0.7,
                   label=f"FB baseline = {fb_val:.3f}")
        ax.axhline(fbs_val, color="#444", linestyle=":", lw=1.2, alpha=0.85,
                   label=f"FB+shrunk = {fbs_val:.3f}")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], rotation=22, ha="right", fontsize=8)
        ax.tick_params(labelsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        # highlight the best bar (min value — lower=better on all metrics)
        j = int(np.nanargmin(vals))
        bars[j].set_edgecolor("#222")
        bars[j].set_linewidth(1.8)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
        ax.margins(y=0.15)

    fig.suptitle("Stylized-facts comparison (lower = closer to real; test set)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    out = "figures/stylized_facts_bars.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
