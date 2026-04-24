"""OOS evaluation for dynamic SF lambda experiment checkpoints.

Compares 6 SF schedules (none, fixed_1, ema_01, ema_03, cosine, fixed_20)
against Factor-Bootstrap + the V7d champion on the held-out test window.
Each row reports OOS metrics plus the val_frob recorded at checkpoint time
so the schedule ranking can be cross-checked against training dynamics.

Usage
-----
    python src/eval_dynamic_sf.py                         # default
    python src/eval_dynamic_sf.py --ckpt_dir results/dynamic_sf
    python src/eval_dynamic_sf.py --schedules ema_03 cosine
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
from factors_pca import compute_hierarchical_pca_factors, compute_rolling_coefficients
from macro_processor import MacroProcessor
from train_dynamic_sf import DynamicSFGenerator


DEFAULT_SCHEDULES = ["none", "fixed_1", "ema_01", "ema_03", "cosine", "fixed_20"]
KEY_METRICS = [
    "full_corr_frob", "acf_score", "vc_score",
    "lev_score", "swd", "mahalanobis",
]
METRIC_LABELS = ["CorrFrob", "ACF", "VC", "LEV", "SWD", "Mahal"]


def gen_paths(gen, cov_arr, factors_arr, alpha_hat, beta_hat, sigma_hat,
              test_start_idx, n_paths=100, t_l=252, batch=20, device="cpu"):
    """Generate `n_paths` OOS trajectories anchored at `test_start_idx`.

    Mirrors eval_oos.gen_paths: a single conditioning window is tiled across
    paths so the only stochasticity is the latent z and residual noise. This
    keeps comparisons across schedules strictly controlled.
    """
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


def load_generator(ckpt_path, n_assets, n_factors, d_cov, L_eps, device):
    # Read arch from sibling config.json if present (new runs); fall back to
    # the original 128/2b defaults for legacy checkpoints.
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        hidden_dim = cfg.get("hidden_dim", 128)
        num_blocks = cfg.get("num_blocks", 2)
    else:
        hidden_dim, num_blocks = 128, 2
    gen = DynamicSFGenerator(
        n_assets=n_assets, n_factors=n_factors, d_z=10, d_cov=d_cov,
        hidden_dim=hidden_dim, num_blocks=num_blocks, dropout=0.2,
        eta=0.1, eta_sigma=0.1, residual_cholesky=L_eps,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt["G"], strict=False)
    gen.eval()
    return gen, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="results/dynamic_sf")
    p.add_argument("--schedules", nargs="+", default=DEFAULT_SCHEDULES)
    p.add_argument("--n_paths", type=int, default=100)
    p.add_argument("--t_l", type=int, default=252)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    p.add_argument("--out", default=None, help="output JSON path")
    p.add_argument("--include_v7d", action="store_true",
                   help="also evaluate results/v7d as reference")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("  Dynamic SF Lambda — OOS Evaluation")
    print(f"  ckpt_dir={args.ckpt_dir}  n_paths={args.n_paths}  device={device}")
    print("=" * 72)

    # ── Data pipeline (identical to train_dynamic_sf.py) ───────────────
    returns = load_returns()
    macro = load_macro()
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

    alpha_hat, beta_hat, sigma_hat = compute_rolling_coefficients(
        returns, factors_df, window=126)

    real_test = test_ret.values
    n_assets = returns.shape[1]
    asset_list = returns.columns.tolist()
    class_indices = {c: [asset_list.index(a) for a in v] for c, v in ASSET_CLASSES.items()}

    L_eps_path = "data/residual_cholesky.npy"
    L_eps = np.load(L_eps_path) if os.path.exists(L_eps_path) else None

    # ── Baseline: Factor Bootstrap ─────────────────────────────────────
    print("\n  [FB] Factor Bootstrap (rolling OLS + Gaussian residual)")
    fb = FactorBootstrap(window=126)
    fb.fit(returns.values, f_arr)
    fb_paths = fb.generate(f_arr, start_idx=test_start_idx, n_paths=args.n_paths)

    models = [("FB", fb_paths, None)]

    # ── Dynamic SF checkpoints ─────────────────────────────────────────
    missing = []
    for sname in args.schedules:
        ckpt_path = os.path.join(args.ckpt_dir, sname, "best_model.pt")
        if not os.path.exists(ckpt_path):
            missing.append(sname)
            print(f"  [{sname}] checkpoint missing → skip")
            continue
        print(f"  [{sname}] loading checkpoint")
        gen, ckpt = load_generator(ckpt_path, n_assets, n_factors, d_cov, L_eps, device)
        val_frob = float(ckpt.get("frob", float("nan")))
        paths = gen_paths(gen, cov_arr, f_arr, alpha_hat, beta_hat, sigma_hat,
                          test_start_idx, n_paths=args.n_paths, t_l=args.t_l,
                          device=device)
        models.append((sname, paths, val_frob))

    # ── Optional V7d reference ─────────────────────────────────────────
    if args.include_v7d:
        v7d_ckpt = "results/v7d/best_model.pt"
        if os.path.exists(v7d_ckpt):
            print("  [v7d] loading reference checkpoint")
            from sfmg_baseline import Generator as V7dGenerator
            cfg = json.load(open("results/v7d/config.json"))
            gen = V7dGenerator(
                n_assets=n_assets, n_factors=n_factors, d_z=10, d_cov=d_cov,
                hidden_dim=cfg.get("hidden_dim", 256),
                eta=cfg.get("eta", 0.1), eta_sigma=cfg.get("eta_sigma", 0.1),
                residual_cholesky=L_eps,
            ).to(device)
            ckpt = torch.load(v7d_ckpt, map_location=device, weights_only=False)
            gen.load_state_dict(ckpt["G"], strict=False); gen.eval()
            paths = gen_paths(gen, cov_arr, f_arr, alpha_hat, beta_hat, sigma_hat,
                              test_start_idx, n_paths=args.n_paths, t_l=args.t_l,
                              device=device)
            models.append(("v7d", paths, float(ckpt.get("frob", float("nan")))))
        else:
            print("  [v7d] --include_v7d set but results/v7d/best_model.pt missing")

    if len(models) == 1:
        print("\n  No checkpoints evaluated beyond the baseline — nothing to rank.")
        if missing:
            print(f"  Missing schedules: {', '.join(missing)}")
        return

    # ── Evaluate ───────────────────────────────────────────────────────
    print(f"\n{'='*72}\n  EVALUATION\n{'='*72}")
    results = {}
    for name, paths, _ in models:
        print(f"  {name} ...")
        results[name] = full_evaluation(real_test, paths,
                                        class_indices=class_indices,
                                        n_paths=args.n_paths)
    val_frobs = {name: vf for name, _, vf in models}

    # ── Print table ────────────────────────────────────────────────────
    names = [m[0] for m in models]
    print(f"\n{'='*72}\n  OOS RESULTS — Dynamic SF Lambda\n{'='*72}")
    header = f"  {'Metric':<12s}" + "".join(f"{n:>11s}" for n in names)
    print(header)
    print(f"  {'-'*(12 + 11*len(names))}")

    wins = {n: 0 for n in names}
    for m, label in zip(KEY_METRICS, METRIC_LABELS):
        vals = {n: results[n].get(m, float("nan")) for n in names}
        finite = {n: v for n, v in vals.items() if np.isfinite(v)}
        best = min(finite, key=lambda k: finite[k]) if finite else None
        if best is not None:
            wins[best] += 1
        row = f"  {label:<12s}"
        for n in names:
            marker = "*" if n == best else " "
            row += f"{vals[n]:>10.4f}{marker}"
        print(row)

    # val_frob row (training-time reference)
    vf_row = f"  {'val_frob':<12s}"
    for _, _, vf in models:
        cell = "   -" if vf is None or not np.isfinite(vf) else f"{vf:>10.4f}"
        vf_row += f"{cell:>10s} "
    print(vf_row)

    ranked = sorted(wins.items(), key=lambda kv: -kv[1])
    print("\n  Wins ranking: " + ", ".join(f"{n}={w}" for n, w in ranked))
    if missing:
        print(f"  Missing schedules skipped: {', '.join(missing)}")

    # ── Block correlation block (same convention as eval_oos.py) ───────
    block_keys = sorted(k for k in results[names[0]]
                        if "frob" in k and "norm" not in k and k != "full_corr_frob")
    if block_keys:
        print(f"\n  Block correlation (lower = better):")
        for m in block_keys:
            vals = {n: results[n].get(m, float("nan")) for n in names}
            finite = {n: v for n, v in vals.items() if np.isfinite(v)}
            best = min(finite, key=lambda k: finite[k]) if finite else None
            row = f"  {m:<35s}"
            for n in names:
                marker = "*" if n == best else " "
                row += f"{vals[n]:>10.4f}{marker}"
            print(row)

    # ── Save ───────────────────────────────────────────────────────────
    out_path = args.out or os.path.join(args.ckpt_dir, "oos_results.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save = {
        "config": {
            "ckpt_dir": args.ckpt_dir,
            "schedules": args.schedules,
            "n_paths": args.n_paths,
            "t_l": args.t_l,
            "seed": args.seed,
            "test_start": str(test_ret.index[0].date()),
            "test_end": str(test_ret.index[-1].date()),
        },
        "wins": wins,
        "missing": missing,
        "val_frob": {n: val_frobs[n] for n in names},
        "metrics": {
            n: {k: float(v) for k, v in results[n].items()
                if isinstance(v, (float, int, np.floating, np.integer))}
            for n in names
        },
    }
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
