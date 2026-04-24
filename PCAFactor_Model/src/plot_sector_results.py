"""Visualise the sector (GS-orth) results against prior specifications.

Figures produced:
  1. figures/pair_corr_sector_vs_prior.png
     6-panel rolling-63d pair correlation on test set for 6 same-sector pairs,
     real (blue) vs generated (red) — three model variants in columns:
       left  = SF-MG (PCA factors, legacy over-shrunk L_rho)
       mid   = SF-MG + L_rho swap (PCA factors, training-sample L_rho)
       right = SF-MG-R (sector factors + GS orth, training-sample L_rho)
  2. figures/corr_matrix_sector_vs_real.png
     60x60 correlation matrix heatmaps (Real / SF-MG-R / Difference).
"""
import os, sys, json, numpy as np, matplotlib.pyplot as plt, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import ASSET_CLASSES, load_returns, load_macro, train_val_test_split
from factors_pca import compute_hierarchical_pca_factors, compute_rolling_coefficients
from factors_sector import compute_sector_factors
from macro_processor import MacroProcessor
from eval_sfmg import load_sfmg, gen_paths_sfmg
from plot_pair_corr_gen import rolling_corr_1d

PAIRS = [
    ("v", "ma", "V vs MA (payments)"),
    ("gold", "silver", "Gold vs Silver (precious)"),
    ("tlt", "ief", "TLT vs IEF (Treasuries)"),
    ("crude_oil", "bno", "Crude Oil vs BNO (oil)"),
    ("aapl", "msft", "AAPL vs MSFT (mega-tech)"),
    ("gold", "copper", "Gold vs Copper (cross-subgroup)"),
]
WINDOW = 63


def prepare(factors):
    returns = load_returns(); macro = load_macro()
    tr, val, test = train_val_test_split(returns)
    train_end = str(tr.index[-1].date())
    test_start = returns.index.get_loc(test.index[0])
    t_l = len(test)
    proc = MacroProcessor(n_pca_components=8, n_regimes=3)
    cov = proc.fit_transform(macro, returns.index, train_end=train_end).reindex(returns.index).fillna(0).values
    if factors == "sector":
        f, _ = compute_sector_factors(returns, ASSET_CLASSES, train_end=train_end)
    else:
        f, _ = compute_hierarchical_pca_factors(returns, ASSET_CLASSES, n_global=5, n_class=2, train_end=train_end)
    a, b, s = compute_rolling_coefficients(returns, f, window=126)
    # build L_eps from training residuals matching this factor set
    val_start = len(tr)
    F = f.values
    res = np.zeros_like(returns.values)
    for t in range(returns.shape[0]): res[t] = returns.values[t] - a[t] - b[t] @ F[t]
    rtr = res[126:val_start]
    Cc = np.corrcoef(rtr.T); w, V = np.linalg.eigh(0.5*(Cc+Cc.T)); w = np.clip(w, 1e-6, None)
    Cp = (V*w)@V.T; d = np.sqrt(np.diag(Cp)); Cn = Cp/np.outer(d,d)
    L_eps = np.linalg.cholesky(Cn + 1e-6*np.eye(len(Cn)))
    return returns, cov, f, a, b, s, L_eps, test_start, t_l, returns.columns.tolist()


def gen_model(ckpt, factors, L_eps_override=None):
    returns, cov, f, a, b, s, L_eps_default, start, t_l, al = prepare(factors)
    L_eps = L_eps_override if L_eps_override is not None else L_eps_default
    F = f.values; n_factors = F.shape[1]
    gen, cfg = load_sfmg(ckpt, returns.shape[1], n_factors, cov.shape[1], L_eps, "cpu")
    F_in = np.vstack([np.zeros((1, n_factors)), F[:-1]]) if cfg.get("lag_factors", False) else F
    torch.manual_seed(42); np.random.seed(42)
    paths = gen_paths_sfmg(gen, cov, F_in, a, b, s, start, n_paths=50, t_l=t_l, batch=10, device="cpu")
    return paths, returns.values[start:start+t_l], al


def main():
    # ── Model A: SF-MG (PCA) with LEGACY over-shrunk L_rho ─────────────
    print("[A] SF-MG (PCA + legacy L_rho)")
    L_legacy = np.load("data/residual_cholesky_v9legacy.npy")
    p_A, real, al = gen_model("results/v9_s42/best_model.pt", "pca", L_eps_override=L_legacy)
    # ── Model B: SF-MG (PCA) + training-sample L_rho swap ──────────────
    print("[B] SF-MG (PCA) + L_rho swap")
    p_B, _, _ = gen_model("results/v9_s42/best_model.pt", "pca")
    # ── Model C: SF-MG-R (sector factors + GS + training-sample L_rho) ──
    print("[C] SF-MG-R (sector + GS)")
    p_C, _, _ = gen_model("results/v9_s42_v3_gs/best_model.pt", "sector")

    t_l = min(real.shape[0], p_A.shape[1], p_B.shape[1], p_C.shape[1])
    real = real[:t_l]
    p_A = p_A[:, :t_l]; p_B = p_B[:, :t_l]; p_C = p_C[:, :t_l]

    # ── Figure 1: pair-corr grid 6 pairs × 3 models ───────────────────
    fig, axes = plt.subplots(len(PAIRS), 3, figsize=(15, 2.1*len(PAIRS)), sharey=True)
    col_titles = [
        "A · SF-MG (PCA, legacy L_ρ)",
        "B · SF-MG (PCA) + L_ρ swap (no retrain)",
        "C · SF-MG-R (sector factors + GS orth + retrain)",
    ]
    for col, (paths, title) in enumerate(zip([p_A, p_B, p_C], col_titles)):
        gen_c = paths.reshape(-1, 60)
        real_t = np.tile(real, (paths.shape[0], 1))
        for row, (a, b, ptitle) in enumerate(PAIRS):
            ax = axes[row, col]
            ia, ib = al.index(a), al.index(b)
            r_roll = rolling_corr_1d(real_t[:, ia], real_t[:, ib], WINDOW)
            g_roll = rolling_corr_1d(gen_c[:, ia], gen_c[:, ib], WINDOW)
            ax.plot(r_roll, color="C0", lw=0.5, alpha=0.8, label="Real")
            ax.plot(g_roll, color="C3", lw=0.5, alpha=0.8, label="Generated")
            ax.axhline(0, color="gray", lw=0.3, alpha=0.5)
            ax.set_ylim(-1, 1)
            rmse = float(np.sqrt(np.nanmean((r_roll - g_roll) ** 2)))
            ax.text(0.02, 0.95, f"rmse={rmse:.3f}", transform=ax.transAxes,
                    fontsize=8, va="top",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8, ec="none"))
            if col == 0:
                ax.set_ylabel(ptitle, fontsize=9)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            if row == len(PAIRS) - 1:
                ax.set_xlabel("timestep (test ×50 paths concat)")
            ax.tick_params(labelsize=7)
            if row == 0 and col == 0:
                ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.suptitle("Rolling-63d pair correlation: three model variants on test set",
                 fontsize=11, y=0.995)
    fig.tight_layout()
    out1 = "figures/pair_corr_sector_vs_prior.png"
    fig.savefig(out1, dpi=140)
    print(f"[save] {out1}")

    # ── Figure 2: correlation matrix Real vs SF-MG-R ───────────────────────
    # Sort assets: bonds, commodities, stocks (as data_loader ASSET_CLASSES)
    order = []
    for cls in ["bonds", "commodities", "stocks"]:
        order.extend([al.index(a) for a in ASSET_CLASSES[cls] if a in al])
    order_names = [al[i] for i in order]
    real_ord = real[:, order]
    C_real = np.corrcoef(real_ord.T)
    # sector-factor generated
    gen_sector = p_C.reshape(-1, 60)[:, order]
    C_gen = np.corrcoef(gen_sector.T)
    diff = C_real - C_gen
    frob = float(np.linalg.norm(diff, "fro"))

    fig2, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, mat, title, cmap, vmin, vmax in zip(
        axes,
        [C_real, C_gen, diff],
        ["Real (test set)", f"Generated · SF-MG-R  (frob={frob:.2f})", "Real − Generated"],
        ["RdBu_r", "RdBu_r", "RdBu_r"],
        [-0.5, -0.5, -0.3],
        [1.0, 1.0, 0.3],
    ):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(title, fontsize=11)
        ax.axhline(20 - 0.5, color="black", lw=0.7)
        ax.axhline(40 - 0.5, color="black", lw=0.7)
        ax.axvline(20 - 0.5, color="black", lw=0.7)
        ax.axvline(40 - 0.5, color="black", lw=0.7)
        ax.set_xticks([10, 30, 50]); ax.set_yticks([10, 30, 50])
        ax.set_xticklabels(["Bonds", "Commodities", "Stocks"], fontsize=9)
        ax.set_yticklabels(["Bonds", "Commodities", "Stocks"], fontsize=9, rotation=90, va="center")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig2.suptitle("60×60 correlation matrix: Real vs SF-MG-R (test set)", fontsize=12, y=1.02)
    fig2.tight_layout()
    out2 = "figures/corr_matrix_sector_vs_real.png"
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"[save] {out2}")


if __name__ == "__main__":
    main()
