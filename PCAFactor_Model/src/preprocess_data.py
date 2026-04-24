"""Precompute data pipeline outputs once, save to disk.

Downstream training scripts can then skip the 3-5 min CPU-bound pipeline
(MacroProcessor PCA + GMM, hierarchical PCA factors, rolling OLS at
window=126 over T=3773 × N=60 × K=11) and just load the numpy arrays.

Outputs (all in data/preprocessed/):
    returns.npy          (T, N)
    f_arr.npy            (T, K)       hierarchical PCA factors
    f_arr_lag1.npy       (T, K)       f_{t-1} with zero-row prepended
    alpha_hat.npy        (T, N)
    beta_hat.npy         (T, N, K)
    sigma_hat.npy        (T, N)
    cov_arr.npy          (T, d_cov)   macro covariates (15-dim)
    meta.json            {train_end, n_factors, d_cov, vix_col_index,
                          class_indices, asset_list, dates[train_end_idx,
                          val_end_idx, test_start_idx, T]}
    L_eps.npy            (N, N)       copy of residual_cholesky

Usage:
    python3 src/preprocess_data.py                                  # default train_end=2023-12-31
    python3 src/preprocess_data.py --train_end 2022-12-31 \
        --out data/preprocessed_roll2022                            # rolling-cutoff variant
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import ASSET_CLASSES, load_macro, load_returns, train_val_test_split
from factors_pca import compute_hierarchical_pca_factors, compute_rolling_coefficients
from macro_processor import MacroProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_end", default="2023-12-31",
                   help="Last date of training period (PCA/OLS fit endpoint)")
    p.add_argument("--out", default="data/preprocessed",
                   help="Output directory for preprocessed arrays")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[preprocess] train_end={args.train_end}, out={args.out}")

    returns = load_returns()
    macro = load_macro()

    # Respect exact train_end (MacroProcessor expects YYYY-MM-DD string)
    # For the default 2023-12-31 this yields the existing protocol.
    proc = MacroProcessor(n_pca_components=8, n_regimes=3)
    covariates = proc.fit_transform(macro, returns.index, train_end=args.train_end)
    cov_arr = covariates.reindex(returns.index).fillna(0).values

    factors_df, _ = compute_hierarchical_pca_factors(
        returns, ASSET_CLASSES, n_global=5, n_class=2, train_end=args.train_end)
    n_factors = factors_df.shape[1]
    alpha_hat, beta_hat, sigma_hat = compute_rolling_coefficients(
        returns, factors_df, window=126)
    f_arr = factors_df.values
    f_arr_lag1 = np.vstack([np.zeros((1, n_factors)), f_arr[:-1]])

    asset_list = list(returns.columns)
    class_indices = {c: [asset_list.index(a) for a in v]
                     for c, v in ASSET_CLASSES.items()}

    # Determine split boundaries relative to the data (not hard-coded to 2025)
    train_ret, val_ret, test_ret = train_val_test_split(returns)
    # For custom train_end overrides we still derive val/test by convention
    if args.train_end != "2023-12-31":
        # Re-derive using the Timestamp-based custom protocol used in train_v9
        from pandas import Timestamp
        te = Timestamp(args.train_end)
        idx = returns.index
        # train ends at te (inclusive of entries with index <= te)
        train_mask = idx <= te
        tr_end_idx = int(train_mask.sum())
        val_end_idx = min(tr_end_idx + 252, len(idx))
        test_start_idx = val_end_idx
        test_end_idx = min(val_end_idx + 251, len(idx))
    else:
        tr_end_idx = returns.index.get_loc(train_ret.index[-1]) + 1
        val_end_idx = returns.index.get_loc(val_ret.index[-1]) + 1
        test_start_idx = returns.index.get_loc(test_ret.index[0])
        test_end_idx = min(test_start_idx + 251, len(returns.index))

    cov_cols = list(covariates.columns)
    vix_idx = cov_cols.index("vix_level_z")

    np.save(os.path.join(args.out, "returns.npy"), returns.values.astype(np.float64))
    np.save(os.path.join(args.out, "f_arr.npy"), f_arr.astype(np.float64))
    np.save(os.path.join(args.out, "f_arr_lag1.npy"), f_arr_lag1.astype(np.float64))
    np.save(os.path.join(args.out, "alpha_hat.npy"), alpha_hat.astype(np.float64))
    np.save(os.path.join(args.out, "beta_hat.npy"), beta_hat.astype(np.float64))
    np.save(os.path.join(args.out, "sigma_hat.npy"), sigma_hat.astype(np.float64))
    np.save(os.path.join(args.out, "cov_arr.npy"), cov_arr.astype(np.float64))

    # Copy residual cholesky if present
    rc = "data/residual_cholesky.npy"
    if os.path.exists(rc):
        np.save(os.path.join(args.out, "L_eps.npy"), np.load(rc))

    meta = {
        "train_end": args.train_end,
        "n_factors": int(n_factors),
        "d_cov": int(cov_arr.shape[1]),
        "n_assets": int(returns.shape[1]),
        "T": int(returns.shape[0]),
        "vix_col_index": int(vix_idx),
        "cov_cols": cov_cols,
        "class_indices": class_indices,
        "asset_list": asset_list,
        "dates": {
            "first": str(returns.index[0].date()),
            "last": str(returns.index[-1].date()),
            "train_end_idx": int(tr_end_idx),
            "val_end_idx": int(val_end_idx),
            "test_start_idx": int(test_start_idx),
            "test_end_idx": int(test_end_idx),
        },
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    sz = sum(os.path.getsize(os.path.join(args.out, f))
             for f in os.listdir(args.out) if os.path.isfile(os.path.join(args.out, f)))
    print(f"[preprocess] wrote {len(os.listdir(args.out))} files "
          f"to {args.out}/ ({sz / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
