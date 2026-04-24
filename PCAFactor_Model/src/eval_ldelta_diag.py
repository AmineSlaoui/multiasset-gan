"""L_Delta diagnostic: SVD spectrum, effective rank, regime-conditional
magnitude contribution. Addresses reviewer A5.

Output: results/ldelta_diag.json + figures/fig_ldelta_spectrum.png
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import ASSET_CLASSES, load_returns, load_macro, train_val_test_split
from factors_pca import compute_hierarchical_pca_factors
from macro_processor import MacroProcessor
from sfmg_generator import SFMGGenerator
from eval_sfmg import load_sfmg


def effective_rank(singular_values):
    """Entropy-based effective rank (Roy & Vetterli 2007)."""
    sv2 = singular_values ** 2
    p = sv2 / (sv2.sum() + 1e-12)
    H = -(p * np.log(p + 1e-12)).sum()
    return float(np.exp(H))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+",
                   default=["results/v9_s42/best_model.pt",
                             "results/v9_s1/best_model.pt",
                             "results/v9_s2/best_model.pt"],
                   help="v9 checkpoints to diagnose")
    p.add_argument("--out", default="results/ldelta_diag.json")
    p.add_argument("--fig", default="figures/fig_ldelta_spectrum.png")
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
    d_cov = covariates.shape[1]
    vix_idx = list(covariates.columns).index("vix_level_z")

    factors_df, _ = compute_hierarchical_pca_factors(
        returns, ASSET_CLASSES, n_global=5, n_class=2, train_end=train_end)
    n_factors = factors_df.shape[1]
    n_assets = returns.shape[1]
    L_eps = np.load("data/residual_cholesky.npy")

    per_ckpt = {}
    all_specs = []
    for ckpt in args.ckpts:
        if not os.path.exists(ckpt):
            print(f"  skip {ckpt}"); continue
        tag = os.path.basename(os.path.dirname(ckpt))
        print(f"  {tag}: loading ...")
        gen, cfg = load_sfmg(ckpt, n_assets, n_factors, d_cov, L_eps, device)
        L_delta = gen.L_delta.detach().cpu().numpy()  # (N, r)
        r = L_delta.shape[1]
        U, S, Vt = np.linalg.svd(L_delta, full_matrices=False)

        # L_Δ · L_Δ^T has diag norm per asset; global Frobenius
        cov_delta = L_delta @ L_delta.T
        frob = float(np.linalg.norm(cov_delta))

        # Regime activation on 2025 test window
        cov_t = torch.tensor(cov_arr[test_start_idx:test_start_idx + len(test_ret)],
                              dtype=torch.float32).T.unsqueeze(0)
        with torch.no_grad():
            z = gen.regime_encoder(cov_t).squeeze().numpy()   # (T_oos,)
        z_mean = float(z.mean())
        z_active = float((z > 0.5).mean())

        # Effective contribution: std of path innovation σ̂ · L_Δ u' scaled by z(t)
        # Using a rough estimate: per-day residual-correction std per asset =
        #   sqrt((L_Δ L_Δ^T).diag) × z(t). Report average diag and per-regime.
        diag_delta = np.sqrt(np.diag(cov_delta))
        stress_mask = z > 0.5
        contrib_stress = float(diag_delta.mean() * z[stress_mask].mean()) if stress_mask.any() else 0.0
        contrib_normal = float(diag_delta.mean() * z[~stress_mask].mean()) if (~stress_mask).any() else 0.0

        per_ckpt[tag] = {
            "rank_r": int(r),
            "singular_values": [float(x) for x in S],
            "frobenius": frob,
            "effective_rank": effective_rank(S),
            "per_asset_diag_mean": float(diag_delta.mean()),
            "per_asset_diag_max": float(diag_delta.max()),
            "z_stats": {"mean_oos": z_mean, "active_frac_oos": z_active},
            "stress_vs_normal_contribution": {"stress": contrib_stress,
                                               "normal": contrib_normal},
        }
        all_specs.append((tag, S))
        print(f"    Frobenius={frob:.4f}  eff rank={effective_rank(S):.2f}  SV={S}")

    # Figure: SVD spectra across seeds
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 3.2))
        for tag, S in all_specs:
            plt.plot(np.arange(1, len(S)+1), S / S[0] if S[0] > 0 else S, 'o-',
                     label=tag)
        plt.xlabel("Singular value index")
        plt.ylabel(r"$\sigma_i / \sigma_1$")
        plt.title(r"$L_\Delta$ singular-value spectrum (normalised)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
        plt.savefig(args.fig, dpi=150, bbox_inches="tight")
        print(f"  figure → {args.fig}")
    except ImportError:
        print("  matplotlib not available, skipping figure")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(per_ckpt, f, indent=2)
    print(f"  JSON   → {args.out}")


if __name__ == "__main__":
    main()
