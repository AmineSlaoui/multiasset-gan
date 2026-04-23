"""DCC-MLE vs moment-based DCC vs other baselines on the OOS window.

Addresses reviewer R3.4: the in-Python SimpleDCC uses moment-based shortcuts
that underestimate the baseline. This script runs the same OOS comparison
that eval_oos.py runs but swaps the production rmgarch DCC in, and keeps
SimpleDCC as a side-by-side reference so we can show exactly how much the
estimator choice moves the numbers.

Output: results/dcc_mle_oos.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import FactorBootstrap, RDCC, SimpleDCC, StudentTCopula, BlockBootstrap
from data_loader import ASSET_CLASSES, load_returns, train_val_test_split
from evaluate import full_evaluation


KEY_METRICS = [
    "full_corr_frob", "acf_score", "vc_score",
    "lev_score", "swd", "mahalanobis",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_paths", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/dcc_mle_oos.json")
    p.add_argument("--returns_csv", default="data/all_assets_log_returns.csv")
    p.add_argument("--cache_dir", default="data/dcc",
                   help="directory for caching rmgarch binary output")
    p.add_argument("--skip_other_baselines", action="store_true",
                   help="only run SimpleDCC and RDCC (faster smoke test)")
    args = p.parse_args()

    np.random.seed(args.seed)

    print("=" * 72)
    print("  DCC-MLE vs SimpleDCC — OOS baseline comparison")
    print(f"  n_paths={args.n_paths}  seed={args.seed}")
    print("=" * 72)

    returns = load_returns()
    train_ret, val_ret, test_ret = train_val_test_split(returns)
    test_start_idx = returns.index.get_loc(test_ret.index[0])
    real_test = test_ret.values
    T_oos = len(test_ret)
    asset_list = returns.columns.tolist()
    class_indices = {c: [asset_list.index(a) for a in v] for c, v in ASSET_CLASSES.items()}

    os.makedirs(args.cache_dir, exist_ok=True)
    models = []

    # Determine trim start (first row where every asset has a non-zero return)
    # so SimpleDCC can be fitted on the *same* data window as rmgarch and
    # isolate the MLE-vs-moment effect from the data-window effect.
    r_fit_full = returns.values[:test_start_idx]
    trim_start = int(max((np.argmax(r_fit_full[:, j] != 0)
                          for j in range(r_fit_full.shape[1]))))
    print(f"\n  Trim start (first all-nonzero row): {trim_start} "
          f"→ fit window = {trim_start}:{test_start_idx} "
          f"(T={test_start_idx - trim_start})")

    # ── (1) SimpleDCC on full data (current paper baseline) ────────────
    print("\n  [1] SimpleDCC-full (moment-based, full fit window)")
    dcc_full = SimpleDCC()
    dcc_full.fit(r_fit_full)
    models.append(("SimpleDCC-full", dcc_full.generate(T_oos, n_paths=args.n_paths)))

    # ── (2) SimpleDCC on trimmed data (apples-to-apples with MLE) ──────
    print("\n  [2] SimpleDCC-trim (moment-based, MLE-aligned window)")
    dcc_trim = SimpleDCC()
    dcc_trim.fit(r_fit_full[trim_start:])
    models.append(("SimpleDCC-trim", dcc_trim.generate(T_oos, n_paths=args.n_paths)))

    # ── (3) Production DCC via rmgarch ─────────────────────────────────
    print("\n  [3] RDCC (rmgarch full MLE, trimmed window)")
    rdcc = RDCC(returns_csv=args.returns_csv, train_end_idx=test_start_idx)
    cache_bin = os.path.join(args.cache_dir, f"mle_paths_n{args.n_paths}_s{args.seed}.bin")
    paths_mle = rdcc.generate(T_oos, n_paths=args.n_paths, seed=args.seed,
                              cache_bin=cache_bin)
    models.append(("DCC-MLE", paths_mle))

    if not args.skip_other_baselines:
        print("\n  [3] Factor Bootstrap")
        from factors_v2 import compute_hierarchical_pca_factors
        factors_df, _ = compute_hierarchical_pca_factors(
            returns, ASSET_CLASSES, n_global=5, n_class=2,
            train_end=str(train_ret.index[-1].date()))
        fb = FactorBootstrap(window=126)
        fb.fit(returns.values, factors_df.values)
        fb_paths = fb.generate(factors_df.values, start_idx=test_start_idx,
                               n_paths=args.n_paths)
        models.append(("FB", fb_paths))

        print("\n  [4] Student-t Copula")
        tcop = StudentTCopula()
        tcop.fit(returns.values[:test_start_idx])
        models.append(("t-Copula", tcop.generate(T_oos, n_paths=args.n_paths)))

        print("\n  [5] Block Bootstrap")
        bb = BlockBootstrap(block_len=20)
        bb.fit(returns.values[:test_start_idx])
        models.append(("BlockBS", bb.generate(T_oos, n_paths=args.n_paths)))

    # ── Evaluate ───────────────────────────────────────────────────────
    print(f"\n{'='*72}\n  EVALUATION\n{'='*72}")
    results = {}
    for name, paths in models:
        print(f"  {name} ...")
        results[name] = full_evaluation(real_test, paths,
                                        class_indices=class_indices,
                                        n_paths=args.n_paths)

    names = [m[0] for m in models]
    print(f"\n{'='*72}\n  DCC-MLE vs SimpleDCC — OOS RESULTS\n{'='*72}")
    header = f"  {'Metric':<12s}" + "".join(f"{n:>12s}" for n in names)
    print(header)
    print(f"  {'-'*(12 + 12*len(names))}")

    wins = {n: 0 for n in names}
    for m in KEY_METRICS:
        vals = {n: results[n].get(m, float("nan")) for n in names}
        finite = {n: v for n, v in vals.items() if np.isfinite(v)}
        best = min(finite, key=lambda k: finite[k]) if finite else None
        if best is not None:
            wins[best] += 1
        row = f"  {m:<12s}"
        for n in names:
            marker = "*" if n == best else " "
            row += f"{vals[n]:>11.4f}{marker}"
        print(row)

    ranked = sorted(wins.items(), key=lambda kv: -kv[1])
    print("\n  Wins ranking: " + ", ".join(f"{n}={w}" for n, w in ranked))

    # Isolate two effects:
    #   Δ_window = SimpleDCC-trim − SimpleDCC-full   (effect of trimming)
    #   Δ_mle    = DCC-MLE       − SimpleDCC-trim   (effect of MLE vs moment)
    # If Δ_mle is small, the reviewer's concern that moment-based DCC is
    # under-estimating the baseline is not supported by the data — any gap
    # between SimpleDCC-full and DCC-MLE is mostly a data-window artefact.
    if "SimpleDCC-full" in results and "SimpleDCC-trim" in results and "DCC-MLE" in results:
        d_window = {m: results["SimpleDCC-trim"].get(m, float("nan"))
                        - results["SimpleDCC-full"].get(m, float("nan"))
                    for m in KEY_METRICS}
        d_mle = {m: results["DCC-MLE"].get(m, float("nan"))
                     - results["SimpleDCC-trim"].get(m, float("nan"))
                 for m in KEY_METRICS}
        print("\n  Effect decomposition (negative = better):")
        print(f"    {'Metric':<16s} {'Δ_window':>10s} {'Δ_mle':>10s}")
        for m in KEY_METRICS:
            print(f"    {m:<16s} {d_window[m]:>+10.4f} {d_mle[m]:>+10.4f}")

    # ── Save ───────────────────────────────────────────────────────────
    save = {
        "config": {"n_paths": args.n_paths, "seed": args.seed,
                   "test_start": str(test_ret.index[0].date()),
                   "test_end": str(test_ret.index[-1].date()),
                   "trim_start": trim_start},
        "wins": wins,
        "metrics": {n: {k: float(v) for k, v in results[n].items()
                        if isinstance(v, (float, int, np.floating, np.integer))}
                    for n in names},
    }
    if "SimpleDCC-full" in results and "SimpleDCC-trim" in results and "DCC-MLE" in results:
        save["delta_window"] = {m: float(d_window[m]) for m in KEY_METRICS}
        save["delta_mle"] = {m: float(d_mle[m]) for m in KEY_METRICS}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Saved → {args.out}")


if __name__ == "__main__":
    main()
