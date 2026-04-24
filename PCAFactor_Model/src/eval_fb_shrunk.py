"""FB vs FB+shrunk vs SF-MarketGAN attribution on 2025 OOS.

Addresses Round 2B Reviewer's N1/U2: decompose the ~3.6% CorrFrob improvement
over plain factor bootstrap into (a) shrinkage contribution and (b) GAN
contribution. Quotes the reviewer:
  "则 GAN 真正的独立贡献只有 0.04 (1.1%). 作者必须直接测量这个数字"

Output: results/fb_shrunk_attribution.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import FactorBootstrap
from data_loader import ASSET_CLASSES, load_macro, load_returns, train_val_test_split
from evaluate import full_evaluation
from eval_oos import gen_paths
from factors_pca import compute_hierarchical_pca_factors, compute_rolling_coefficients
from macro_processor import MacroProcessor
from sfmg_baseline import Generator


KEY_METRICS = ["full_corr_frob", "acf_score", "vc_score",
               "lev_score", "swd", "mahalanobis"]


def main():
    device = torch.device("cpu")
    np.random.seed(42); torch.manual_seed(42)

    print("=" * 72)
    print("  FB vs FB+shrunk vs SF-MarketGAN attribution (Reviewer N1/U2)")
    print("=" * 72)

    returns = load_returns()
    macro = load_macro()
    train_ret, val_ret, test_ret = train_val_test_split(returns)
    test_start_idx = returns.index.get_loc(test_ret.index[0])
    train_end = str(train_ret.index[-1].date())

    proc = MacroProcessor(n_pca_components=8, n_regimes=3)
    covariates = proc.fit_transform(macro, returns.index, train_end=train_end)
    cov_arr = covariates.reindex(returns.index).fillna(0).values

    factors_df, _ = compute_hierarchical_pca_factors(
        returns, ASSET_CLASSES, n_global=5, n_class=2, train_end=train_end)
    f_arr = factors_df.values
    alpha_hat, beta_hat, sigma_hat = compute_rolling_coefficients(
        returns, factors_df, window=126)

    real_test = test_ret.values
    N, n_paths, t_l = 60, 100, 252
    asset_list = returns.columns.tolist()
    class_indices = {c: [asset_list.index(a) for a in v]
                     for c, v in ASSET_CLASSES.items()}

    # ── 1. Plain FB ────────────────────────────────────────────────────
    print("\n  [1] Factor Bootstrap (iid Gaussian residuals)")
    np.random.seed(42)
    fb = FactorBootstrap(window=126)
    fb.fit(returns.values, f_arr)
    fb_paths = fb.generate(f_arr, start_idx=test_start_idx, n_paths=n_paths)

    # ── 2. FB + shrunk residuals ───────────────────────────────────────
    # Use the same L_ρ the v7d generator was trained with so the apples-to-
    # apples comparison isolates the GAN's additional contribution on top of
    # the shrunk-correlation prior.
    L_eps_shared = np.load("data/residual_cholesky.npy")
    print("\n  [2] FB + shrunk residuals (using v7d's correlation prior)")
    np.random.seed(42)
    fb_shrunk = FactorBootstrap(window=126)
    fb_shrunk.fit(returns.values, f_arr)
    fb_shrunk.fit_shrunk_residuals(returns.values[:test_start_idx],
                                    f_arr[:test_start_idx], lam=0.2,
                                    L_eps=L_eps_shared)
    fb_shrunk_paths = fb_shrunk.generate(f_arr, start_idx=test_start_idx,
                                          n_paths=n_paths)

    # ── 3. SF-MarketGAN (existing v7d checkpoint) ──────────────────────
    print("\n  [3] SF-MarketGAN V7d (existing main model)")
    np.random.seed(42); torch.manual_seed(42)
    config = json.load(open("results/v7d/config.json"))
    L_eps = np.load("data/residual_cholesky.npy")
    gen = Generator(n_assets=N, n_factors=factors_df.shape[1], d_z=10,
                    d_cov=covariates.shape[1],
                    hidden_dim=config.get("hidden_dim", 256),
                    eta=config.get("eta", 0.1),
                    eta_sigma=config.get("eta_sigma", 0.1),
                    residual_cholesky=L_eps)
    ckpt = torch.load("results/v7d/best_model.pt", map_location=device,
                      weights_only=False)
    gen.load_state_dict(ckpt["G"], strict=False); gen.eval()
    gan_paths = gen_paths(gen, returns.values, f_arr, cov_arr,
                          alpha_hat, beta_hat, sigma_hat,
                          test_start_idx, n_paths=n_paths, t_l=t_l, device=device)

    # ── Evaluate ───────────────────────────────────────────────────────
    print(f"\n{'='*72}\n  Evaluation\n{'='*72}")
    models = [("FB", fb_paths), ("FB+shrunk", fb_shrunk_paths),
              ("SF-MarketGAN", gan_paths)]
    results = {}
    for name, paths in models:
        print(f"  {name} ...")
        results[name] = full_evaluation(real_test, paths,
                                         class_indices=class_indices,
                                         n_paths=n_paths)

    # ── Attribution table ──────────────────────────────────────────────
    print(f"\n{'='*72}\n  Attribution\n{'='*72}")
    header = f"  {'Metric':<14s}" + "".join(f"{n:>14s}" for n in ["FB","FB+shrunk","SF-MarketGAN"])
    print(header)
    print(f"  {'-'*(14 + 14*3)}")
    attribution = {}
    for m in KEY_METRICS:
        a = results["FB"][m]; b = results["FB+shrunk"][m]; c = results["SF-MarketGAN"][m]
        d_total = a - c            # total improvement FB → GAN (positive = improvement)
        d_shrink = a - b           # shrinkage contribution
        d_gan = b - c              # GAN-additional contribution beyond FB+shrunk
        attribution[m] = {"FB": a, "FB+shrunk": b, "SF-MarketGAN": c,
                          "total_improvement": d_total,
                          "shrinkage_component": d_shrink,
                          "gan_additional_component": d_gan,
                          "shrinkage_pct_of_total": float(d_shrink / d_total * 100) if abs(d_total) > 1e-12 else 0.0,
                          "gan_pct_of_total": float(d_gan / d_total * 100) if abs(d_total) > 1e-12 else 0.0}
        print(f"  {m:<14s} {a:>13.4f}  {b:>13.4f}  {c:>13.4f}")

    print(f"\n{'='*72}\n  Attribution decomposition (lower metric = better)\n{'='*72}")
    print(f"  {'Metric':<14s}  {'Δ total':>10s}  {'Δ shrink':>10s}  {'Δ GAN-extra':>12s}  {'shrink%':>8s}  {'GAN%':>8s}")
    for m in KEY_METRICS:
        a = attribution[m]
        print(f"  {m:<14s}  {a['total_improvement']:>+10.4f}  {a['shrinkage_component']:>+10.4f}  "
              f"{a['gan_additional_component']:>+12.4f}  {a['shrinkage_pct_of_total']:>7.1f}%  "
              f"{a['gan_pct_of_total']:>7.1f}%")

    save = {
        "config": {"n_paths": n_paths, "seed": 42, "lam_shrink": 0.2,
                   "test_start": str(test_ret.index[0].date()),
                   "test_end": str(test_ret.index[-1].date())},
        "metrics": {n: {k: float(v) for k, v in results[n].items()
                        if isinstance(v, (float, int, np.floating, np.integer))}
                    for n in results},
        "attribution": {m: {k: float(v) for k, v in d.items()}
                        for m, d in attribution.items()},
    }
    os.makedirs("results", exist_ok=True)
    out = "results/fb_shrunk_attribution.json"
    with open(out, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
