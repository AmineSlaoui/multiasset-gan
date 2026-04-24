"""DCC-MLE vs SimpleDCC on the 51-asset full-history subset.

Addresses Codex's concern: the original eval used a globally trimmed fit
window that confounded "MLE vs moment" with "different training history".
Here we compare on a subset of 51 assets that all have full 2011-01
history, so both estimators see the same input data.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import SimpleDCC
from data_loader import ASSET_CLASSES
from evaluate import full_evaluation


KEY_METRICS = [
    "full_corr_frob", "acf_score", "vc_score",
    "lev_score", "swd", "mahalanobis",
]


def main():
    n_paths = 100
    seed = 42
    returns_csv = "data/all_assets_log_returns_fullhist.csv"
    mle_bin = "data/dcc/mle_paths_51assets_s42.bin"
    out = "results/dcc_mle_fullhist_oos.json"

    np.random.seed(seed)

    df = pd.read_csv(returns_csv, parse_dates=["Date"]).set_index("Date")
    # Match the main pipeline's train/val/test convention
    train_end = "2023-12-31"
    val_end = "2024-12-31"
    train = df.loc[:train_end]
    val = df.loc[train_end:val_end].iloc[1:]
    test = df.loc[val_end:].iloc[1:]
    test_start_idx = df.index.get_loc(test.index[0])
    real_test = test.values
    T_oos = len(test)
    n_assets = df.shape[1]

    asset_list = df.columns.tolist()
    class_indices = {c: [asset_list.index(a) for a in v if a in asset_list]
                     for c, v in ASSET_CLASSES.items()}

    print("=" * 72)
    print(f"  DCC-MLE vs SimpleDCC — full-history subset")
    print(f"  assets={n_assets}  T_fit={test_start_idx}  T_oos={T_oos}")
    print("=" * 72)

    # SimpleDCC on full history
    print("\n  [1] SimpleDCC (moment, full 51-asset history)")
    dcc = SimpleDCC()
    dcc.fit(df.values[:test_start_idx])
    paths_simple = dcc.generate(T_oos, n_paths=n_paths)

    # DCC-MLE from cached binary
    print("\n  [2] DCC-MLE (rmgarch, full 51-asset history)")
    paths_mle = np.fromfile(mle_bin, dtype=np.float64).reshape(
        (n_paths, T_oos, n_assets))

    models = [("SimpleDCC", paths_simple), ("DCC-MLE", paths_mle)]

    print("\n  Evaluating ...")
    results = {}
    for name, paths in models:
        results[name] = full_evaluation(real_test, paths,
                                        class_indices=class_indices,
                                        n_paths=n_paths)

    print(f"\n{'='*72}\n  Apples-to-apples DCC comparison (51 assets)\n{'='*72}")
    header = f"  {'Metric':<16s}" + "".join(f"{n:>12s}" for n in ["SimpleDCC", "DCC-MLE", "Δ (MLE-Moment)"])
    print(header)
    print(f"  {'-'*(16 + 12*3)}")
    for m in KEY_METRICS:
        a = results["SimpleDCC"][m]
        b = results["DCC-MLE"][m]
        d = b - a
        mark = "*" if d < 0 else " "
        print(f"  {m:<16s} {a:>11.4f}  {b:>11.4f}  {d:>+11.4f}{mark}")

    # Per-path std sanity check
    real_std = real_test.std(axis=0)
    s_std = paths_simple.reshape(-1, n_assets).std(axis=0)
    m_std = paths_mle.reshape(-1, n_assets).std(axis=0)
    print("\n  path-std / real-std (lower bias = 1.0):")
    print(f"    SimpleDCC: mean={np.mean(s_std/real_std):.3f}  median={np.median(s_std/real_std):.3f}  max={np.max(s_std/real_std):.3f}")
    print(f"    DCC-MLE:   mean={np.mean(m_std/real_std):.3f}  median={np.median(m_std/real_std):.3f}  max={np.max(m_std/real_std):.3f}")

    save = {
        "config": {"n_paths": n_paths, "seed": seed, "n_assets": n_assets,
                   "T_fit": test_start_idx, "T_oos": T_oos,
                   "subset": "full_history_51_assets"},
        "metrics": {n: {k: float(v) for k, v in results[n].items()
                        if isinstance(v, (float, int, np.floating, np.integer))}
                    for n in results},
        "delta_mle_minus_moment": {m: float(results["DCC-MLE"][m] - results["SimpleDCC"][m])
                                    for m in KEY_METRICS},
        "std_bias": {
            "SimpleDCC": {"mean": float(np.mean(s_std/real_std)),
                           "median": float(np.median(s_std/real_std)),
                           "max": float(np.max(s_std/real_std))},
            "DCC-MLE": {"mean": float(np.mean(m_std/real_std)),
                         "median": float(np.median(m_std/real_std)),
                         "max": float(np.max(m_std/real_std))},
        },
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
