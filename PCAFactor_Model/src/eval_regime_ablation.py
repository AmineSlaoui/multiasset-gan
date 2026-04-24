"""Hard-gate vs soft-gate regime encoder ablation (inference-time).

Reviewer's concern: the regime encoder can trivially recover the
supervision label (VIX z-score > 0.3) because VIX is itself a macro input.
If a deterministic hard-gate achieves the same OOS performance, the learned
encoder adds nothing.

We load a trained SF-MarketGAN-R checkpoint and at evaluation swap the
encoder's forward pass for the deterministic indicator function
  z_hard(t) = 1[VIX_z(t) > 0.3].
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import FactorBootstrap
from data_loader import ASSET_CLASSES, load_macro, load_returns, train_val_test_split
from evaluate import full_evaluation
from factors_pca import compute_hierarchical_pca_factors, compute_rolling_coefficients
from macro_processor import MacroProcessor
from eval_sfmg import gen_paths_sfmg, load_sfmg


class HardGateEncoder(torch.nn.Module):
    """Returns 1[VIX_z > threshold] per timestep."""
    def __init__(self, vix_idx: int, threshold: float = 0.3):
        super().__init__()
        self.vix_idx = vix_idx
        self.threshold = threshold

    def forward(self, cov):
        # cov: (B, d_cov, T) → z: (B, 1, T)
        vix = cov[:, self.vix_idx:self.vix_idx + 1, :]
        return (vix > self.threshold).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/v9_s42/best_model.pt")
    p.add_argument("--n_paths", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--out", default="results/regime_ablation.json")
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
    f_arr = factors_df.values
    n_factors = factors_df.shape[1]
    alpha_hat, beta_hat, sigma_hat = compute_rolling_coefficients(
        returns, factors_df, window=126)

    real_test = test_ret.values
    n_assets = returns.shape[1]
    asset_list = returns.columns.tolist()
    class_indices = {c: [asset_list.index(a) for a in v] for c, v in ASSET_CLASSES.items()}
    L_eps = np.load("data/residual_cholesky.npy")

    # Load the SF-MarketGAN-R checkpoint twice: one with learned encoder (soft),
    # one with hard-gate encoder
    gen_soft, cfg = load_sfmg(args.ckpt, n_assets, n_factors, d_cov, L_eps, device)
    gen_hard, _   = load_sfmg(args.ckpt, n_assets, n_factors, d_cov, L_eps, device)
    # Monkey-patch the regime encoder
    gen_hard.regime_encoder = HardGateEncoder(vix_idx, threshold=args.threshold)

    f_in = f_arr
    if cfg.get("lag_factors", False):
        f_in = np.vstack([np.zeros((1, n_factors)), f_arr[:-1]])

    # FB + FB+shrunk (for reference)
    np.random.seed(42)
    fb = FactorBootstrap(window=126); fb.fit(returns.values, f_arr)
    fb_paths = fb.generate(f_arr, start_idx=test_start_idx, n_paths=args.n_paths)

    np.random.seed(42)
    fb_s = FactorBootstrap(window=126); fb_s.fit(returns.values, f_arr)
    fb_s.fit_shrunk_residuals(returns.values[:test_start_idx],
                               f_arr[:test_start_idx], lam=0.2, L_eps=L_eps)
    fb_s_paths = fb_s.generate(f_arr, start_idx=test_start_idx, n_paths=args.n_paths)

    # Soft + Hard
    paths_soft = gen_paths_sfmg(gen_soft, cov_arr, f_in, alpha_hat, beta_hat,
                                    sigma_hat, test_start_idx,
                                    n_paths=args.n_paths, t_l=252, device=device)
    paths_hard = gen_paths_sfmg(gen_hard, cov_arr, f_in, alpha_hat, beta_hat,
                                    sigma_hat, test_start_idx,
                                    n_paths=args.n_paths, t_l=252, device=device)

    # Diagnostic: how often do the two z(t) disagree?
    test_cov = torch.tensor(cov_arr[test_start_idx:test_start_idx + 252],
                             dtype=torch.float32).T.unsqueeze(0)
    with torch.no_grad():
        z_soft = gen_soft.regime_encoder(test_cov).squeeze().numpy()
        z_hard = gen_hard.regime_encoder(test_cov).squeeze().numpy()
    print(f"\n  Regime activation on 2025 test:")
    print(f"    soft-gate mean={z_soft.mean():.3f}, std={z_soft.std():.3f}, "
          f"active (>0.5) frac={(z_soft > 0.5).mean():.2f}")
    print(f"    hard-gate active frac={z_hard.mean():.2f}")
    print(f"    agreement on (z>0.5 vs z_hard==1) = {((z_soft > 0.5) == (z_hard == 1)).mean():.3f}")

    KEY = ["full_corr_frob","acf_score","vc_score","lev_score","swd","mahalanobis"]
    models = [("FB", fb_paths), ("FB+shrunk", fb_s_paths),
              ("hard-gate", paths_hard), ("soft-gate (SF-MG-R)", paths_soft)]
    print("\n  Evaluating...")
    results = {}
    for n, p in models:
        results[n] = full_evaluation(real_test, p,
                                      class_indices=class_indices,
                                      n_paths=args.n_paths)

    print(f"\n{'='*72}\n  Regime-gate ablation on 2025 OOS\n{'='*72}")
    hdr = f"  {'Metric':<14s}" + "".join(f"{n:>14s}" for n in [m[0] for m in models])
    print(hdr); print(f"  {'-'*(14 + 14*len(models))}")
    for m in KEY:
        vals = [results[n][m] for n, _ in models]
        best = np.nanargmin(vals)
        row = f"  {m:<14s}"
        for i, v in enumerate(vals):
            mk = "*" if i == best else " "
            row += f"{v:>13.4f}{mk}"
        print(row)

    out = {
        "config": {"n_paths": args.n_paths, "threshold": args.threshold,
                   "ckpt": args.ckpt},
        "z_activation": {"soft_mean": float(z_soft.mean()),
                         "soft_std": float(z_soft.std()),
                         "soft_active_frac": float((z_soft > 0.5).mean()),
                         "hard_active_frac": float(z_hard.mean()),
                         "agreement": float(((z_soft > 0.5) == (z_hard == 1)).mean())},
        "metrics": {n: {k: float(v) for k, v in results[n].items()
                        if isinstance(v, (float, int, np.floating, np.integer))}
                    for n in results},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {args.out}")


if __name__ == "__main__":
    main()
