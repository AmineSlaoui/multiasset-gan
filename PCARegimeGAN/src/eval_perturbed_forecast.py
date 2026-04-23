"""Perturbed factor-forecast mean-variance simulation (MarketGAN Sec 5.2 analog).

At each monthly rebalance, we:
  1. Take the realized next-21-day factor path as the ``oracle'' signal.
  2. Inject noise to scale the forecast to target $R^2$ levels:
        f_hat = sqrt(R^2) * f_true + sqrt(1 - R^2) * (sigma_f * eps)
  3. Expected return per asset:  mu_hat = beta_hat @ mean(f_hat over 21 days)
  4. Covariance estimate Sigma_hat from the model's synthetic paths at time t.
  5. Mean-variance weight (unconstrained, rescaled to unit norm):
        w ~ Sigma_hat^{-1} mu_hat
     then scale so portfolio volatility equals the equal-weight benchmark vol
     (i.e., we compare *risk-adjusted return per unit vol*).
  6. Realize the portfolio return on the next 21 days.

Output per R^2 level: annualised Sharpe across rebalances for each model.

This tests whether a model's *covariance* quality translates into better MV
decisions as the signal-to-noise ratio in the forecast degrades.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import FactorBootstrap
from data_loader import ASSET_CLASSES, load_macro, load_returns, train_val_test_split
from factors_v2 import compute_hierarchical_pca_factors, compute_rolling_coefficients_v2
from macro_processor import MacroProcessor
from sf_marketgan import Generator
from sf_marketgan_v9 import GeneratorV9
from eval_v9 import gen_paths_v9_like, load_v9


TRADING_DAYS = 252


def mv_portfolio(mu, Sigma, reg=1e-4, target_vol=None):
    """Unconstrained MV weights; if target_vol is given, scale to that daily vol."""
    N = Sigma.shape[0]
    Sigma_reg = Sigma + reg * np.eye(N)
    w = np.linalg.solve(Sigma_reg, mu)
    w = w / np.sum(np.abs(w) + 1e-12)   # normalize by L1 for numerical stability
    if target_vol is not None:
        vol = np.sqrt(w @ Sigma @ w)
        if vol > 1e-12:
            w = w * (target_vol / vol)
    return w


def annualized_sharpe(daily_port_returns):
    m = daily_port_returns.mean() * TRADING_DAYS
    s = daily_port_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return float(m / s) if s > 1e-12 else 0.0


def run_one_r2(r2, real_factors_oos, sigma_f, beta_hats, real_test, model_paths_per_rebal,
                rebal_points, rebal_freq, target_vol, rng):
    """Run MV backtest for a given R^2 across models.

    model_paths_per_rebal[model_name][k] = (n_paths, rebal_freq, N) generated paths
        aligned to rebalance k.
    """
    N = real_test.shape[1]
    portfolios = {m: np.zeros(real_test.shape[0]) for m in model_paths_per_rebal}

    for k, rp in enumerate(rebal_points):
        # Realized next-rebal_freq factors
        seg_f = real_factors_oos[rp:rp + rebal_freq]       # (H, K)
        f_true = seg_f.mean(axis=0)                         # (K,)

        # Perturbed forecast at this R^2
        noise = rng.randn(len(f_true)) * sigma_f
        f_hat = np.sqrt(r2) * f_true + np.sqrt(1 - r2) * noise

        # Expected return per asset: mu_hat = beta_hat_t @ f_hat
        mu_hat = beta_hats[rp] @ f_hat                      # (N,)

        # Realized return over holding period
        seg_r = real_test[rp:rp + rebal_freq]               # (H, N)

        for name, per_rebal in model_paths_per_rebal.items():
            # Sigma_hat from this model's synthetic paths at rebal k
            paths = per_rebal[k]                            # (P, H, N)
            if paths.ndim == 3:
                flat = paths.reshape(-1, N)
            else:
                flat = paths
            Sigma_hat = np.cov(flat.T)
            w = mv_portfolio(mu_hat, Sigma_hat, target_vol=target_vol)
            # Daily portfolio returns on holding period
            portfolios[name][rp:rp + rebal_freq] = seg_r @ w

    return {m: annualized_sharpe(portfolios[m][:rebal_points[-1] + rebal_freq])
            for m in portfolios}


def get_synthetic_paths_per_rebal(paths_full, rebal_points, rebal_freq):
    """paths_full: (P, T, N). Return list of (P, rebal_freq, N) per rebal."""
    out = []
    for rp in rebal_points:
        seg = paths_full[:, rp:rp + rebal_freq, :]
        out.append(seg)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v9_ckpt", default="results/v9_s42/best_model.pt")
    p.add_argument("--v7d_ckpt", default="results/v7d/best_model.pt")
    p.add_argument("--n_paths", type=int, default=200)
    p.add_argument("--rebal_freq", type=int, default=21)
    p.add_argument("--r2_grid", nargs="+", type=float, default=[1.0, 0.5, 0.1, 0.01, 0.001])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/perturbed_forecast.json")
    args = p.parse_args()

    device = torch.device("cpu")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    returns = load_returns(); macro = load_macro()
    train_ret, val_ret, test_ret = train_val_test_split(returns)
    test_start_idx = returns.index.get_loc(test_ret.index[0])
    train_end = str(train_ret.index[-1].date())

    proc = MacroProcessor(n_pca_components=8, n_regimes=3)
    covariates = proc.fit_transform(macro, returns.index, train_end=train_end)
    cov_arr = covariates.reindex(returns.index).fillna(0).values
    d_cov = covariates.shape[1]

    factors_df, _ = compute_hierarchical_pca_factors(
        returns, ASSET_CLASSES, n_global=5, n_class=2, train_end=train_end)
    f_arr = factors_df.values
    n_factors = factors_df.shape[1]
    alpha_hat, beta_hat, sigma_hat = compute_rolling_coefficients_v2(
        returns, factors_df, window=126)

    real_test = test_ret.values
    T = real_test.shape[0]; N = real_test.shape[1]
    rebal_points = list(range(0, T - args.rebal_freq + 1, args.rebal_freq))
    n_rebal = len(rebal_points)
    print(f"  T={T} days, rebal_freq={args.rebal_freq}, n_rebal={n_rebal}")

    # Rebalance-aligned beta hats (absolute indices into returns array)
    beta_at_rebal = {rp: beta_hat[test_start_idx + rp] for rp in rebal_points}

    # Factor training volatility for the noise draw
    sigma_f = f_arr[:test_start_idx].std(axis=0)    # (K,)

    # Realized factors on test window
    real_factors_oos = f_arr[test_start_idx:test_start_idx + T]

    L_eps = np.load("data/residual_cholesky.npy")

    # ── Generate full-window paths for each model ─────────────────────
    print("  FB ...")
    np.random.seed(args.seed)
    fb = FactorBootstrap(window=126); fb.fit(returns.values, f_arr)
    fb_paths = fb.generate(f_arr, start_idx=test_start_idx, n_paths=args.n_paths)

    print("  FB+shrunk ...")
    np.random.seed(args.seed)
    fb_s = FactorBootstrap(window=126); fb_s.fit(returns.values, f_arr)
    fb_s.fit_shrunk_residuals(returns.values[:test_start_idx],
                               f_arr[:test_start_idx], lam=0.2, L_eps=L_eps)
    fb_s_paths = fb_s.generate(f_arr, start_idx=test_start_idx, n_paths=args.n_paths)

    print("  fixed-η (v7d) ...")
    cfg7 = json.load(open(os.path.join(os.path.dirname(args.v7d_ckpt), "config.json")))
    g7 = Generator(n_assets=N, n_factors=n_factors, d_z=10, d_cov=d_cov,
                   hidden_dim=cfg7.get("hidden_dim", 256),
                   eta=cfg7.get("eta", 0.1), eta_sigma=cfg7.get("eta_sigma", 0.1),
                   residual_cholesky=L_eps).to(device)
    c7 = torch.load(args.v7d_ckpt, map_location=device, weights_only=False)
    g7.load_state_dict(c7["G"], strict=False); g7.eval()
    fixed_paths = gen_paths_v9_like(g7, cov_arr, f_arr,
                                     alpha_hat, beta_hat, sigma_hat,
                                     test_start_idx, n_paths=args.n_paths,
                                     t_l=T, device=device)

    print("  SF-MarketGAN-R ...")
    g9, cfg9 = load_v9(args.v9_ckpt, N, n_factors, d_cov, L_eps, device)
    f_in = np.vstack([np.zeros((1, n_factors)), f_arr[:-1]]) if cfg9.get("lag_factors") else f_arr
    v9_paths = gen_paths_v9_like(g9, cov_arr, f_in,
                                  alpha_hat, beta_hat, sigma_hat,
                                  test_start_idx, n_paths=args.n_paths,
                                  t_l=T, device=device)

    paths_by_rebal = {
        "FB":         get_synthetic_paths_per_rebal(fb_paths, rebal_points, args.rebal_freq),
        "FB+shrunk":  get_synthetic_paths_per_rebal(fb_s_paths, rebal_points, args.rebal_freq),
        "Fixed-η":    get_synthetic_paths_per_rebal(fixed_paths, rebal_points, args.rebal_freq),
        "SF-MG-R":    get_synthetic_paths_per_rebal(v9_paths, rebal_points, args.rebal_freq),
    }

    # Target daily vol = equal-weight daily vol on test window (so all
    # portfolios share risk budget)
    ew_daily = real_test.mean(axis=1)
    target_vol = ew_daily.std(ddof=1)
    print(f"  target daily vol (equal-weight): {target_vol:.5f}")

    # Convert beta_hat dict → absolute-index array
    beta_hats_full = beta_hat[test_start_idx:test_start_idx + T]

    results = {}
    n_noise = 20  # noise replicates per R^2 to reduce MC variance
    print(f"  averaging over {n_noise} noise replicates per R^2")
    for r2 in args.r2_grid:
        method_sharpes = {m: [] for m in paths_by_rebal}
        for nrep in range(n_noise):
            rng = np.random.RandomState(args.seed + 1000 * nrep)
            s = run_one_r2(r2, real_factors_oos, sigma_f,
                            beta_hats_full, real_test, paths_by_rebal,
                            rebal_points, args.rebal_freq, target_vol, rng)
            for m in s:
                method_sharpes[m].append(s[m])
        agg = {m: {"mean": float(np.mean(method_sharpes[m])),
                    "std": float(np.std(method_sharpes[m], ddof=1))}
               for m in method_sharpes}
        results[str(r2)] = agg
        print(f"  R^2={r2:<6}: " + "  ".join(
              f"{m}={agg[m]['mean']:.3f}±{agg[m]['std']:.3f}"
              for m in agg))

    out = {"config": {"n_paths": args.n_paths, "rebal_freq": args.rebal_freq,
                       "r2_grid": args.r2_grid, "seed": args.seed,
                       "target_vol": float(target_vol), "n_rebal": n_rebal},
            "sharpe_by_r2": results}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {args.out}")


if __name__ == "__main__":
    main()
