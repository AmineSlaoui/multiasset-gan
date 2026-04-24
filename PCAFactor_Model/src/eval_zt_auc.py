"""z(t) time-series + ROC/AUC vs VIX label. Addresses reviewer A6.

Output: results/zt_auc.json + figures/fig_zt_timeseries.png
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import ASSET_CLASSES, load_returns, load_macro, train_val_test_split
from factors_pca import compute_hierarchical_pca_factors
from macro_processor import MacroProcessor
from eval_sfmg import load_sfmg


def roc_auc(scores, labels):
    """Compute ROC AUC via Mann-Whitney U statistic."""
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Probability that a random positive scores higher than a random negative
    # = rank-based computation
    all_scores = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(all_scores))
    # Handle ties by averaging ranks
    unique_vals, inv, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    rank_sum = np.zeros(len(unique_vals))
    # average rank per unique value
    order_idx = np.argsort(all_scores)
    sorted_labels = all_labels[order_idx]
    # Simple unreliable tie-breaking — use scipy if available
    try:
        from scipy.stats import rankdata
        r = rankdata(all_scores)
        auc = (r[all_labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    except ImportError:
        r = ranks + 1
        auc = (r[all_labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+",
                   default=["results/v9_s42/best_model.pt"])
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--out", default="results/zt_auc.json")
    p.add_argument("--fig", default="figures/fig_zt_timeseries.png")
    args = p.parse_args()

    device = torch.device("cpu")

    returns = load_returns(); macro = load_macro()
    train_ret, val_ret, test_ret = train_val_test_split(returns)
    test_start_idx = returns.index.get_loc(test_ret.index[0])
    train_end = str(train_ret.index[-1].date())

    proc = MacroProcessor(n_pca_components=8, n_regimes=3)
    covariates = proc.fit_transform(macro, returns.index, train_end=train_end)
    cov_arr = covariates.reindex(returns.index).fillna(0).values
    d_cov = covariates.shape[1]
    vix_idx = list(covariates.columns).index("vix_level_z")

    factors_df, _ = compute_hierarchical_pca_factors(
        returns, ASSET_CLASSES, n_global=5, n_class=2, train_end=train_end)
    n_factors = factors_df.shape[1]
    n_assets = returns.shape[1]
    L_eps = np.load("data/residual_cholesky.npy")

    T_oos = len(test_ret)
    dates_oos = test_ret.index
    vix_z_test = cov_arr[test_start_idx:test_start_idx + T_oos, vix_idx]
    labels = (vix_z_test > args.threshold).astype(int)

    out = {"threshold": args.threshold,
           "labels_active_fraction": float(labels.mean()),
           "per_ckpt": {}}

    z_series = {}
    for ckpt in args.ckpts:
        tag = os.path.basename(os.path.dirname(ckpt))
        gen, _ = load_sfmg(ckpt, n_assets, n_factors, d_cov, L_eps, device)
        cov_t = torch.tensor(cov_arr[test_start_idx:test_start_idx + T_oos],
                              dtype=torch.float32).T.unsqueeze(0)
        with torch.no_grad():
            z = gen.regime_encoder(cov_t).squeeze().numpy()
        auc = roc_auc(z, labels)
        # also temporal auto-correlation lag 1
        z_center = z - z.mean()
        ac1 = float((z_center[:-1] * z_center[1:]).sum() / (z_center ** 2).sum())
        out["per_ckpt"][tag] = {
            "auc_vs_vix_threshold": float(auc),
            "mean": float(z.mean()),
            "std": float(z.std()),
            "active_frac": float((z > 0.5).mean()),
            "lag1_autocorrelation": ac1,
        }
        z_series[tag] = z
        print(f"  {tag}: AUC={auc:.3f}  mean={z.mean():.3f}  active={((z>0.5).mean()):.2f}  "
              f"lag1_ac={ac1:.3f}")

    # Figure
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True, gridspec_kw={'height_ratios':[2,1]})
        for tag, z in z_series.items():
            ax[0].plot(dates_oos, z, label=tag, alpha=0.75)
        ax[0].axhline(0.5, color='k', linestyle='--', alpha=0.3, label='0.5')
        ax[0].fill_between(dates_oos, 0, 1, where=labels==1, alpha=0.15, color='red',
                            label=r'$\mathbf{1}[\text{VIX}_t^z > 0.3]$')
        ax[0].set_ylabel(r"regime $z(t)$"); ax[0].set_ylim(-0.02, 1.02)
        ax[0].legend(fontsize=8, loc='upper right')
        ax[0].set_title("Regime encoder activation on 2025 OOS")

        ax[1].plot(dates_oos, vix_z_test, color='gray')
        ax[1].axhline(0.3, color='red', linestyle='--', alpha=0.5)
        ax[1].set_ylabel(r"VIX z"); ax[1].set_xlabel("Date")
        fig.autofmt_xdate()
        plt.tight_layout()
        os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
        plt.savefig(args.fig, dpi=150, bbox_inches="tight")
        print(f"  figure → {args.fig}")
    except ImportError:
        print("  matplotlib not available, skipping figure")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  JSON   → {args.out}")


if __name__ == "__main__":
    main()
