"""Same-sector pair-correlation comparison across traditional baselines
and the v9 GAN (old + fixed L_rho).

Traditional methods compared:
  - Historical block bootstrap (no parametric structure, resamples days)
  - Factor bootstrap with iid Gaussian residuals (FB)
  - Factor bootstrap with shrunk residual correlation (FB+shrunk)
  - Student-t copula on standardised returns
  - DCC(1,1)-GARCH(1,1) (moment-based, SimpleDCC)
  - v9_s42 GAN (current fixed L_rho)

Outputs the test-set rolling-63d correlation rmse for 6 known same-sector
pairs (V-MA payments, Gold-Silver precious metals, TLT-IEF Treasuries,
Crude-BNO oil, AAPL-MSFT mega-tech, Gold-Copper cross-subgroup).
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import ASSET_CLASSES, load_returns, load_macro, train_val_test_split
from factors_pca import compute_hierarchical_pca_factors, compute_rolling_coefficients
from macro_processor import MacroProcessor
from baselines import FactorBootstrap, SimpleDCC, StudentTCopula, BlockBootstrap
from eval_sfmg import load_sfmg, gen_paths_sfmg
from plot_pair_corr_gen import rolling_corr_1d

PAIRS = [("v", "ma", "V-MA"),
         ("gold", "silver", "Gold-Silver"),
         ("tlt", "ief", "TLT-IEF"),
         ("crude_oil", "bno", "Crude-BNO"),
         ("aapl", "msft", "AAPL-MSFT"),
         ("gold", "copper", "Gold-Copper")]
WINDOW = 63


def eval_pair_rmse(paths: np.ndarray, real: np.ndarray, asset_list: list,
                   n_paths: int) -> dict:
    """paths: (n_paths, T, N)   real: (T, N)"""
    tl = min(real.shape[0], paths.shape[1])
    paths = paths[:, :tl]
    real = real[:tl]
    gen_concat = paths.reshape(-1, paths.shape[-1])
    real_tiled = np.tile(real, (n_paths, 1))
    out = {}
    for (a, b, name) in PAIRS:
        ia, ib = asset_list.index(a), asset_list.index(b)
        r_roll = rolling_corr_1d(real_tiled[:, ia], real_tiled[:, ib], WINDOW)
        g_roll = rolling_corr_1d(gen_concat[:, ia], gen_concat[:, ib], WINDOW)
        rmse = float(np.sqrt(np.nanmean((r_roll - g_roll) ** 2)))
        out[name] = {
            "real_mean": float(np.nanmean(r_roll)),
            "gen_mean": float(np.nanmean(g_roll)),
            "rmse": rmse,
        }
    return out


def main():
    device = torch.device("cpu")
    np.random.seed(42); torch.manual_seed(42)

    returns = load_returns(); macro = load_macro()
    train_ret, val_ret, test_ret = train_val_test_split(returns)
    train_end = str(train_ret.index[-1].date())
    asset_list = returns.columns.tolist()
    test_start_idx = returns.index.get_loc(test_ret.index[0])
    T_oos = len(test_ret)

    proc = MacroProcessor(n_pca_components=8, n_regimes=3)
    covariates = proc.fit_transform(macro, returns.index, train_end=train_end)
    cov_arr = covariates.reindex(returns.index).fillna(0).values

    factors_df, _ = compute_hierarchical_pca_factors(
        returns, ASSET_CLASSES, n_global=5, n_class=2, train_end=train_end)
    f_arr = factors_df.values
    alpha, beta, sigma = compute_rolling_coefficients(returns, factors_df, window=126)

    L_eps_new = np.load("data/residual_cholesky.npy")          # sample-cov
    L_eps_old = np.load("data/residual_cholesky_v9legacy.npy")

    real_test = test_ret.values
    n_paths = 50
    results = {}

    # ── 1. Historical block bootstrap ──────────────────────────────────
    print("\n[Block bootstrap] fitting on train returns")
    bb = BlockBootstrap(block_len=20)
    bb.fit(returns.values[:test_start_idx])
    paths = bb.generate(T_oos, n_paths=n_paths)
    results["BlockBoot"] = eval_pair_rmse(paths, real_test, asset_list, n_paths)

    # ── 2. Factor bootstrap, iid ───────────────────────────────────────
    print("\n[FB iid] fitting")
    fb = FactorBootstrap(window=126)
    fb.fit(returns.values, f_arr)
    paths = fb.generate(f_arr, start_idx=test_start_idx, n_paths=n_paths)
    results["FB_iid"] = eval_pair_rmse(paths, real_test, asset_list, n_paths)

    # ── 3. Factor bootstrap + shrunk residual correlation (old L_rho) ──
    print("\n[FB+shrunk old L_rho]")
    fb2 = FactorBootstrap(window=126)
    fb2.fit(returns.values, f_arr)
    fb2.fit_shrunk_residuals(returns.values[:test_start_idx],
                              f_arr[:test_start_idx], lam=0.2, L_eps=L_eps_old)
    paths = fb2.generate(f_arr, start_idx=test_start_idx, n_paths=n_paths)
    results["FB_shrunk_oldL"] = eval_pair_rmse(paths, real_test, asset_list, n_paths)

    # ── 4. Factor bootstrap + shrunk residual correlation (NEW L_rho) ──
    print("\n[FB+shrunk new L_rho (sample cov)]")
    fb3 = FactorBootstrap(window=126)
    fb3.fit(returns.values, f_arr)
    fb3.fit_shrunk_residuals(returns.values[:test_start_idx],
                              f_arr[:test_start_idx], lam=0.2, L_eps=L_eps_new)
    paths = fb3.generate(f_arr, start_idx=test_start_idx, n_paths=n_paths)
    results["FB_shrunk_newL"] = eval_pair_rmse(paths, real_test, asset_list, n_paths)

    # ── 5. Student-t copula ────────────────────────────────────────────
    print("\n[Student-t copula]")
    sc = StudentTCopula()
    sc.fit(returns.values[:test_start_idx])
    paths = sc.generate(T_oos, n_paths=n_paths)
    results["StudentCopula"] = eval_pair_rmse(paths, real_test, asset_list, n_paths)

    # ── 6. Simple DCC-GARCH ────────────────────────────────────────────
    print("\n[SimpleDCC]")
    dcc = SimpleDCC()
    dcc.fit(returns.values[:test_start_idx])
    paths = dcc.generate(T_oos, n_paths=n_paths)
    results["SimpleDCC"] = eval_pair_rmse(paths, real_test, asset_list, n_paths)

    # ── 7. v9_s42 with new L_rho ───────────────────────────────────────
    print("\n[v9_s42 new L_rho]")
    ckpt = "results/v9_s42/best_model.pt"
    n_assets = len(asset_list); n_factors = f_arr.shape[1]; d_cov = cov_arr.shape[1]
    gen, cfg = load_sfmg(ckpt, n_assets, n_factors, d_cov, L_eps_new, device)
    f_in = np.vstack([np.zeros((1, n_factors)), f_arr[:-1]]) if cfg.get("lag_factors") else f_arr
    paths = gen_paths_sfmg(gen, cov_arr, f_in, alpha, beta, sigma,
                               test_start_idx, n_paths=n_paths, t_l=T_oos,
                               batch=10, device=device)
    results["v9_newL"] = eval_pair_rmse(paths, real_test, asset_list, n_paths)

    # ── 8. v9_s42 with old L_rho (for reference) ───────────────────────
    print("\n[v9_s42 old L_rho]")
    gen, cfg = load_sfmg(ckpt, n_assets, n_factors, d_cov, L_eps_old, device)
    gen.L_rho = torch.tensor(L_eps_old, dtype=torch.float32, device=device)
    f_in = np.vstack([np.zeros((1, n_factors)), f_arr[:-1]]) if cfg.get("lag_factors") else f_arr
    paths = gen_paths_sfmg(gen, cov_arr, f_in, alpha, beta, sigma,
                               test_start_idx, n_paths=n_paths, t_l=T_oos,
                               batch=10, device=device)
    results["v9_oldL"] = eval_pair_rmse(paths, real_test, asset_list, n_paths)

    # ── Table ──────────────────────────────────────────────────────────
    methods = ["BlockBoot", "FB_iid", "FB_shrunk_oldL", "FB_shrunk_newL",
               "StudentCopula", "SimpleDCC", "v9_oldL", "v9_newL"]

    print("\n" + "=" * 110)
    print("ROLLING-63d PAIR CORRELATION RMSE (lower is better)")
    print("=" * 110)
    print(f"{'Pair':<14s} {'real':>7s}  " + "  ".join(f"{m:>14s}" for m in methods))
    print("-" * 110)
    for (_, _, name) in PAIRS:
        real_mean = results["v9_newL"][name]["real_mean"]
        row = f"{name:<14s} {real_mean:>+7.3f}  "
        vals = [results[m][name]["rmse"] for m in methods]
        best = min(vals)
        for v in vals:
            marker = "*" if v == best else " "
            row += f"{v:>13.3f}{marker} "
        print(row)
    print()

    print(f"{'Pair':<14s} {'real':>7s}  " + "  ".join(f"{m:>14s}" for m in methods))
    print("  (gen mean)")
    for (_, _, name) in PAIRS:
        real_mean = results["v9_newL"][name]["real_mean"]
        row = f"{name:<14s} {real_mean:>+7.3f}  "
        for m in methods:
            row += f"{results[m][name]['gen_mean']:>+13.3f}  "
        print(row)

    os.makedirs("results", exist_ok=True)
    with open("results/pair_corr_baselines.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("\n[save] results/pair_corr_baselines.json")


if __name__ == "__main__":
    main()
