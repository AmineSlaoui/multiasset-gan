"""Recompute residual Cholesky L_rho and evaluate V-MA pair correlation with
each variant injected into the frozen v9 generator (no retraining).

Saves:
  data/residual_cholesky_sample.npy   — plain sample correlation Cholesky
  data/residual_cholesky_lw.npy       — Ledoit–Wolf shrinkage Cholesky
  data/residual_cholesky_oas.npy      — Oracle Approximating Shrinkage Cholesky

Baseline (data/residual_cholesky.npy) is left untouched. Backup already in
data/residual_cholesky_backup.npy.
"""
import os, sys, json, numpy as np, pandas as pd, torch
from sklearn.covariance import LedoitWolf, OAS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import ASSET_CLASSES, load_returns, load_macro, train_val_test_split
from factors_pca import compute_hierarchical_pca_factors, compute_rolling_coefficients
from macro_processor import MacroProcessor
from eval_sfmg import load_sfmg, gen_paths_sfmg
from plot_pair_corr_gen import rolling_corr_1d

PAIRS = [("v", "ma", "V-MA"),
         ("gold", "silver", "Gold-Silver"),
         ("tlt", "ief", "TLT-IEF"),
         ("crude_oil", "bno", "Crude-BNO"),
         ("aapl", "msft", "AAPL-MSFT"),
         ("gold", "copper", "Gold-Copper")]
WINDOW = 63


def cov_to_chol(C, jitter=1e-6):
    """Symmetrize, PSD-clip, Cholesky. C is (N,N) correlation matrix."""
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-6, None)
    C_psd = (V * w) @ V.T
    d = np.sqrt(np.diag(C_psd))
    C_norm = C_psd / np.outer(d, d)
    return np.linalg.cholesky(C_norm + jitter * np.eye(C_norm.shape[0]))


def build_L_variants(residuals_train):
    """residuals_train: (T, N). Return dict of variant -> L (N,N)."""
    T, N = residuals_train.shape
    sample_cov = np.cov(residuals_train.T)
    d = np.sqrt(np.diag(sample_cov))
    sample_corr = sample_cov / np.outer(d, d)

    # Ledoit-Wolf shrinkage on the residuals directly
    lw = LedoitWolf().fit(residuals_train)
    lw_cov = lw.covariance_
    dlw = np.sqrt(np.diag(lw_cov))
    lw_corr = lw_cov / np.outer(dlw, dlw)

    oas = OAS().fit(residuals_train)
    oas_cov = oas.covariance_
    doas = np.sqrt(np.diag(oas_cov))
    oas_corr = oas_cov / np.outer(doas, doas)

    return {
        "sample": cov_to_chol(sample_corr),
        "lw": cov_to_chol(lw_corr),
        "oas": cov_to_chol(oas_corr),
    }


def eval_with_L(ckpt, L_eps, returns, cov_arr, f_arr, alpha, beta, sigma,
                tgt_start, tgt_len, n_paths=50, seed=42, device="cpu"):
    np.random.seed(seed); torch.manual_seed(seed)
    n_assets = returns.shape[1]
    n_factors = f_arr.shape[1]
    d_cov = cov_arr.shape[1]

    gen, cfg = load_sfmg(ckpt, n_assets, n_factors, d_cov, L_eps, device)
    # Inject L AFTER load_state_dict (strict=False already) — overwrite buffer
    if L_eps is not None:
        gen.L_rho = torch.tensor(L_eps, dtype=torch.float32, device=device)

    f_in = np.vstack([np.zeros((1, n_factors)), f_arr[:-1]]) if cfg.get("lag_factors", False) else f_arr
    paths = gen_paths_sfmg(gen, cov_arr, f_in, alpha, beta, sigma,
                               tgt_start, n_paths=n_paths, t_l=tgt_len,
                               batch=10, device=device)
    real = returns.values[tgt_start:tgt_start + tgt_len]
    tl = min(real.shape[0], paths.shape[1])
    paths = paths[:, :tl]; real = real[:tl]
    gen_concat = paths.reshape(-1, n_assets)
    real_tiled = np.tile(real, (n_paths, 1))

    asset_list = returns.columns.tolist()
    results = {}
    for (a, b, name) in PAIRS:
        ia, ib = asset_list.index(a), asset_list.index(b)
        r_roll = rolling_corr_1d(real_tiled[:, ia], real_tiled[:, ib], WINDOW)
        g_roll = rolling_corr_1d(gen_concat[:, ia], gen_concat[:, ib], WINDOW)
        rmse = float(np.sqrt(np.nanmean((r_roll - g_roll) ** 2)))
        results[name] = {
            "real_mean": float(np.nanmean(r_roll)),
            "gen_mean": float(np.nanmean(g_roll)),
            "rmse": rmse,
        }
    return results


def main():
    device = torch.device("cpu")
    returns = load_returns(); macro = load_macro()
    train_ret, val_ret, test_ret = train_val_test_split(returns)
    train_end = str(train_ret.index[-1].date())
    asset_list = returns.columns.tolist()

    proc = MacroProcessor(n_pca_components=8, n_regimes=3)
    covariates = proc.fit_transform(macro, returns.index, train_end=train_end)
    cov_arr = covariates.reindex(returns.index).fillna(0).values

    factors_df, _ = compute_hierarchical_pca_factors(
        returns, ASSET_CLASSES, n_global=5, n_class=2, train_end=train_end)
    f_arr = factors_df.values
    alpha, beta, sigma = compute_rolling_coefficients(returns, factors_df, window=126)

    # ── Step 1: compute training residuals and L variants ───────────────
    T, N = returns.shape
    res = np.zeros_like(returns.values)
    for t in range(T):
        res[t] = returns.values[t] - alpha[t] - beta[t] @ f_arr[t]
    te_start = len(train_ret)
    tr_mask = np.arange(126, te_start)
    res_tr = res[tr_mask]
    print(f"[residuals] training window {tr_mask[0]}..{tr_mask[-1]} "
          f"n={len(res_tr)} assets={N}")

    L_variants = build_L_variants(res_tr)
    for name, L in L_variants.items():
        out = f"data/residual_cholesky_{name}.npy"
        np.save(out, L)
        # Sanity check: reconstructed V-MA residual corr
        C = L @ L.T
        d = np.sqrt(np.diag(C))
        Cn = C / np.outer(d, d)
        iv, ima = asset_list.index("v"), asset_list.index("ma")
        igold, isilv = asset_list.index("gold"), asset_list.index("silver")
        itlt, iief = asset_list.index("tlt"), asset_list.index("ief")
        print(f"[{name}]  saved {out}  V-MA={Cn[iv,ima]:+.3f}  "
              f"G-S={Cn[igold,isilv]:+.3f}  TLT-IEF={Cn[itlt,iief]:+.3f}")

    # ── Step 2: evaluate v9 with each L on test-set pair correlations ──
    ckpt = "results/v9_s42/best_model.pt"
    tgt_start = returns.index.get_loc(test_ret.index[0])
    tgt_len = len(test_ret)

    L_current = np.load("data/residual_cholesky.npy")
    all_variants = {"current_baseline": L_current, **L_variants}

    print("\n" + "=" * 80)
    print(f"V9_s42 test-set pair correlations with different L_rho")
    print("=" * 80)
    all_results = {}
    for vname, L in all_variants.items():
        print(f"\n[eval] L_rho = {vname}")
        r = eval_with_L(ckpt, L, returns, cov_arr, f_arr, alpha, beta, sigma,
                        tgt_start, tgt_len, n_paths=50, device=device)
        all_results[vname] = r

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Pair':<14s}" + "".join(f"{n:>14s}" for n in all_variants))
    print("-" * 80)
    print(f"{'real mean':<14s}" + "".join(
        f"{all_results[n][list(all_results[n])[0]]['real_mean']:>14.3f}"
        for n in all_variants))
    for (_, _, name) in PAIRS:
        print(f"{name:<14s}", end="")
        for vname in all_variants:
            rmse = all_results[vname][name]["rmse"]
            print(f"{rmse:>14.3f}", end="")
        print()

    os.makedirs("results", exist_ok=True)
    with open("results/residual_cholesky_swap.json", "w") as fp:
        json.dump(all_results, fp, indent=2)
    print("\n[save] results/residual_cholesky_swap.json")


if __name__ == "__main__":
    main()
