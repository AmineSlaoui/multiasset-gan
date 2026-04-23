"""OOS evaluation for v9 checkpoints (+ FB + FB+shrunk + v7d reference).

Reuses the same generation logic as eval_oos.py / eval_dynamic_sf.py but knows
how to instantiate a v9 generator.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import FactorBootstrap
from data_loader import ASSET_CLASSES, load_macro, load_returns, train_val_test_split
from evaluate import full_evaluation
from factors_v2 import compute_hierarchical_pca_factors, compute_rolling_coefficients_v2
from macro_processor import MacroProcessor
from sf_marketgan import Generator
from sf_marketgan_v9 import GeneratorV9


KEY_METRICS = ["full_corr_frob", "acf_score", "vc_score",
               "lev_score", "swd", "mahalanobis"]


def gen_paths_v9_like(gen, cov_arr, factors_arr, alpha_hat, beta_hat, sigma_hat,
                       test_start_idx, n_paths=100, t_l=252, batch=20, device="cpu"):
    """Generation loop compatible with v9 or v7d (same interface)."""
    N = alpha_hat.shape[1]
    rfs = gen.rfs
    T_full = rfs + t_l + 1

    t_start = max(0, test_start_idx - rfs - 1)
    t_end = min(t_start + T_full, len(cov_arr))
    cov_seq = torch.tensor(cov_arr[t_start:t_end], dtype=torch.float32).T.unsqueeze(0).to(device)
    if t_end - t_start < T_full:
        cov_seq = F.pad(cov_seq, (0, T_full - (t_end - t_start)), mode="replicate")

    ts, te = test_start_idx, min(test_start_idx + t_l, len(factors_arr))
    actual_tl = te - ts
    f = torch.tensor(factors_arr[ts:te], dtype=torch.float32).T.unsqueeze(0).to(device)
    a = torch.tensor(alpha_hat[ts:te].T, dtype=torch.float32).unsqueeze(0).to(device)
    s = torch.tensor(sigma_hat[ts:te].T, dtype=torch.float32).unsqueeze(0).to(device)
    b = torch.tensor(beta_hat[ts:te], dtype=torch.float32).permute(1, 2, 0).unsqueeze(0).to(device)
    if actual_tl < t_l:
        pad = t_l - actual_tl
        f = F.pad(f, (0, pad), mode="replicate")
        a = F.pad(a, (0, pad), mode="replicate")
        s = F.pad(s, (0, pad), mode="replicate")
        b = F.pad(b, (0, pad, 0, 0), mode="replicate")

    out = np.zeros((n_paths, actual_tl, N))
    for i in range(0, n_paths, batch):
        n = min(batch, n_paths - i)
        z = torch.randn(n, gen.d_z, T_full, device=device)
        with torch.no_grad():
            r = gen(z, cov_seq.expand(n, -1, -1), a.expand(n, -1, -1),
                    b.expand(n, -1, -1, -1), s.expand(n, -1, -1), f.expand(n, -1, -1))
        for j in range(n):
            out[i + j] = r[j, :, :actual_tl].cpu().numpy().T
    return out


def load_v9(ckpt_path, n_assets, n_factors, d_cov, L_eps, device):
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    cfg = json.load(open(cfg_path))
    gen = GeneratorV9(
        n_assets=n_assets, n_factors=n_factors, d_z=10, d_cov=d_cov,
        hidden_dim=cfg["hidden_dim"], num_blocks=cfg["num_blocks"], dropout=0.2,
        eta_low=cfg["eta_low"], eta_high=cfg["eta_high"],
        eta_sigma_low=cfg["eta_sigma_low"], eta_sigma_high=cfg["eta_sigma_high"],
        residual_rank=cfg["residual_rank"],
        residual_cholesky=L_eps,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt["G"], strict=False)
    gen.eval()
    return gen, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v9_ckpts", nargs="+", required=True,
                   help="list of results/v9_s*/best_model.pt paths")
    p.add_argument("--include_v7d", action="store_true")
    p.add_argument("--include_fb_shrunk", action="store_true", default=True)
    p.add_argument("--n_paths", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/v9_oos.json")
    p.add_argument("--test_year", type=int, default=None,
                   help="Override: evaluate on the 1-year OOS window starting "
                        "the first trading day after {year}-12-31. Uses the same "
                        "length as the default 2025 test (~251 days).")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    print(f"  device={device}, n_paths={args.n_paths}, seed={args.seed}")

    returns = load_returns(); macro = load_macro()
    if args.test_year is None:
        train_ret, val_ret, test_ret = train_val_test_split(returns)
        test_start_idx = returns.index.get_loc(test_ret.index[0])
        train_end = str(train_ret.index[-1].date())
    else:
        # Custom test window: first trading day after {test_year-1}-12-31,
        # run for 1 year. MacroProcessor / factor PCA must be fit using the
        # same train_end used for the checkpoint's training; we set it to
        # test_year-2 (the year before val) so no test-window data leaks.
        from pandas import Timestamp
        val_year = args.test_year - 1
        train_end_boundary = Timestamp(f"{val_year - 1}-12-31")
        val_end_boundary = Timestamp(f"{val_year}-12-31")
        test_end_boundary = Timestamp(f"{args.test_year}-12-31")
        test_ret = returns.loc[returns.index > val_end_boundary]
        test_ret = test_ret.loc[test_ret.index <= test_end_boundary]
        test_start_idx = returns.index.get_loc(test_ret.index[0])
        train_end = str(train_end_boundary.date())
        train_ret = returns.loc[:train_end]
        print(f"  [test_year override] train_end={train_end}, "
              f"test={test_ret.index[0].date()} -> {test_ret.index[-1].date()} "
              f"({len(test_ret)} days)")

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
    n_assets = returns.shape[1]
    asset_list = returns.columns.tolist()
    class_indices = {c: [asset_list.index(a) for a in v] for c, v in ASSET_CLASSES.items()}

    L_eps = np.load("data/residual_cholesky.npy") if os.path.exists("data/residual_cholesky.npy") else None

    models = []

    # ── Baselines ──────────────────────────────────────────────────────
    print("\n  [FB] Factor bootstrap (iid Gaussian)")
    np.random.seed(args.seed)
    fb = FactorBootstrap(window=126); fb.fit(returns.values, f_arr)
    fb_paths = fb.generate(f_arr, start_idx=test_start_idx, n_paths=args.n_paths)
    models.append(("FB", fb_paths, None))

    if args.include_fb_shrunk:
        print("\n  [FB+shrunk] FB with v7d shrunk correlation prior")
        np.random.seed(args.seed)
        fb_s = FactorBootstrap(window=126); fb_s.fit(returns.values, f_arr)
        fb_s.fit_shrunk_residuals(returns.values[:test_start_idx],
                                   f_arr[:test_start_idx], lam=0.2, L_eps=L_eps)
        fb_s_paths = fb_s.generate(f_arr, start_idx=test_start_idx, n_paths=args.n_paths)
        models.append(("FB+shrunk", fb_s_paths, None))

    # ── v7d reference ──────────────────────────────────────────────────
    if args.include_v7d:
        v7d_ckpt = "results/v7d/best_model.pt"
        if os.path.exists(v7d_ckpt):
            print("\n  [v7d] reference")
            cfg = json.load(open("results/v7d/config.json"))
            gen = Generator(n_assets=n_assets, n_factors=n_factors, d_z=10, d_cov=d_cov,
                            hidden_dim=cfg.get("hidden_dim", 256),
                            eta=cfg.get("eta", 0.1),
                            eta_sigma=cfg.get("eta_sigma", 0.1),
                            residual_cholesky=L_eps).to(device)
            ckpt = torch.load(v7d_ckpt, map_location=device, weights_only=False)
            gen.load_state_dict(ckpt["G"], strict=False); gen.eval()
            paths = gen_paths_v9_like(gen, cov_arr, f_arr, alpha_hat, beta_hat, sigma_hat,
                                       test_start_idx, n_paths=args.n_paths, t_l=252, device=device)
            models.append(("v7d", paths, float(ckpt.get("frob", float("nan")))))

    # ── v9 checkpoints ─────────────────────────────────────────────────
    for ckpt in args.v9_ckpts:
        if not os.path.exists(ckpt):
            print(f"  [skip missing] {ckpt}")
            continue
        tag = os.path.basename(os.path.dirname(ckpt))   # e.g. v9_s42
        print(f"\n  [{tag}] loading v9 checkpoint")
        gen, cfg = load_v9(ckpt, n_assets, n_factors, d_cov, L_eps, device)
        # v9 lag-1: shift factors by 1
        f_in = f_arr
        if cfg.get("lag_factors", False):
            f_in = np.vstack([np.zeros((1, n_factors)), f_arr[:-1]])
        paths = gen_paths_v9_like(gen, cov_arr, f_in, alpha_hat, beta_hat, sigma_hat,
                                   test_start_idx, n_paths=args.n_paths, t_l=252, device=device)
        ckpt_dict = torch.load(ckpt, map_location=device, weights_only=False)
        models.append((tag, paths, float(ckpt_dict.get("frob", float("nan")))))

    # ── Evaluate ───────────────────────────────────────────────────────
    print(f"\n  Evaluating...")
    results = {}
    for name, paths, _ in models:
        print(f"    {name}")
        results[name] = full_evaluation(real_test, paths,
                                         class_indices=class_indices,
                                         n_paths=args.n_paths)

    names = [m[0] for m in models]
    print(f"\n{'='*72}\n  OOS RESULTS\n{'='*72}")
    header = f"  {'Metric':<12s}" + "".join(f"{n:>12s}" for n in names)
    print(header); print(f"  {'-'*(12 + 12*len(names))}")
    wins = {n: 0 for n in names}
    for m in KEY_METRICS:
        vals = {n: results[n].get(m, float("nan")) for n in names}
        finite = {n: v for n, v in vals.items() if np.isfinite(v)}
        best = min(finite, key=lambda k: finite[k]) if finite else None
        if best is not None: wins[best] += 1
        row = f"  {m:<12s}"
        for n in names:
            mk = "*" if n == best else " "
            row += f"{vals[n]:>11.4f}{mk}"
        print(row)

    vf_row = f"  {'val_frob':<12s}"
    for _, _, vf in models:
        vf_row += f"{'   -':>11s} " if vf is None or not np.isfinite(vf) else f"{vf:>11.4f} "
    print(vf_row)
    print(f"\n  Wins: " + ", ".join(f"{n}={w}" for n, w in sorted(wins.items(), key=lambda x: -x[1])))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save = {
        "config": {"n_paths": args.n_paths, "seed": args.seed,
                   "v9_ckpts": args.v9_ckpts},
        "wins": wins,
        "metrics": {n: {k: float(v) for k, v in results[n].items()
                        if isinstance(v, (float, int, np.floating, np.integer))}
                    for n in names},
        "val_frob": {n: None if vf is None else float(vf) for n, _, vf in models},
    }
    with open(args.out, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Saved → {args.out}")


if __name__ == "__main__":
    main()
