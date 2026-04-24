"""v3 factor model: replace class-level PCA with defensible industry factors.

Total: 16 factors (vs v2's 11 PCA-only factors).
  - 5 global PCA factors (unchanged from v2)
  - 2 bonds class PCA factors (unchanged — bonds already work well, duration
    + credit are well captured by class-PCA, GICS-style sectoring does not
    apply to fixed income; see Litterman & Scheinkman 1991 JFI)
  - 5 stock sector factors (custom 5-bucket, FF5-inspired; split Finance
    out of FF5's "Other" bucket so that Visa/Mastercard/JPM/BRK.B share a
    β loading — the specific fix for the V-MA pair-correlation failure)
  - 4 commodity GSCI sub-sector factors (S&P GSCI methodology)

Sector factors are EQUAL-WEIGHT means of the within-bucket asset returns,
then RESIDUALISED against the global PCA factors fitted on TRAINING data
only. This matches the academic "residualised sector portfolio" approach
(Fama–French 1997 JFE "Industry costs of equity"; sum-to-zero constraint
in Barra USE4 is an industry alternative but harder to implement cleanly).

Broad commodity ETFs (DBC, GSG, DJP, PDBC, FTGC) are INTENTIONALLY excluded
from the GSCI sector definitions — they diversify across all sub-sectors
and will load on multiple sector factors automatically via rolling OLS.

References:
  - Fama & French (1997) "Industry costs of equity" JFE 43:153–193
  - Fama & French (2015) "A five-factor asset pricing model" JFE 116:1–22
  - S&P GSCI Methodology (2023 ed.) — https://www.spglobal.com/spdji/
  - Litterman & Scheinkman (1991) "Common factors affecting bond returns"
    Journal of Fixed Income 1(1): 54–61
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Dict, Tuple


# ── Stock sector taxonomy (custom 5-bucket, FF5 + Finance split-out) ──
# Rationale: standard FF5 puts Finance under "Other" which would leave
# Visa, Mastercard, JPM, BRK.B in a heterogeneous bucket with XOM. For
# our V-MA failure-mode fix, the cleanest thing is to give Finance its
# own bucket even though it shrinks "Other" to a single asset (XOM).
STOCK_SECTORS = {
    "stk_hitec":    ["aapl", "msft", "nvda", "intc", "googl", "meta", "nflx"],
    "stk_health":   ["jnj", "unh"],
    "stk_finance":  ["jpm", "brk_b", "v", "ma"],
    "stk_consumer": ["amzn", "tsla", "hd", "pg", "cost", "dis"],
    "stk_energy":   ["xom"],   # single-asset residual bucket
}

# ── Commodity GSCI sub-sector taxonomy (S&P GSCI) ─────────────────────
# Pure single-commodity names only. Diversified ETFs (DBC, GSG, DJP,
# PDBC, FTGC) are not assigned to any bucket — they pick up multi-factor
# exposure via rolling OLS.
COMMODITY_SECTORS = {
    "cmd_energy":     ["crude_oil", "natural_gas", "uga", "bno"],
    "cmd_precious":   ["gold", "silver", "platinum", "pall"],
    "cmd_industrial": ["copper"],    # single-asset
    "cmd_agri":       ["coffee", "corn", "wheat", "soybeans", "cane", "dba"],
}


def _sector_mean_returns(returns: pd.DataFrame, groups: Dict[str, list]) -> pd.DataFrame:
    out = {}
    for name, members in groups.items():
        present = [m for m in members if m in returns.columns]
        if len(present) == 0:
            continue
        out[name] = returns[present].mean(axis=1)
    return pd.DataFrame(out, index=returns.index)


def _sequential_gram_schmidt(
    factor_matrix_full: np.ndarray,
    train_mask: np.ndarray,
    col_names: list,
) -> Tuple[np.ndarray, list]:
    """Sequential Gram-Schmidt: the k-th output column is the residual of
    the k-th raw factor after projection onto columns 0..k-1, with
    projection coefficients estimated ON TRAINING DATA ONLY.

    Factors are processed in priority order (caller decides the order).
    This yields mutually orthogonal factors on training data (approximately
    so out-of-sample) while preserving the interpretation of earlier
    factors — higher-priority factors are unchanged, lower-priority ones
    are orthogonalized against everything above them.

    Final step: standardize each resulting column to unit variance on
    training data. This matches the scale of other factors and keeps the
    rolling-OLS β coefficients numerically well-conditioned.

    Returns:
        (T, K) orthogonalised + standardized factor matrix.
    """
    T, K = factor_matrix_full.shape
    out = np.zeros_like(factor_matrix_full)
    # First factor: keep as-is, only standardize later
    out[:, 0] = factor_matrix_full[:, 0]

    for k in range(1, K):
        # Regress raw factor k on already-orthogonalised factors 0..k-1
        # WITH intercept, using training data only. Intercept handling is
        # needed because factors have small but nonzero drift — without it
        # zero inner-product does not equal zero correlation.
        X_tr = np.column_stack([np.ones(train_mask.sum()), out[train_mask, :k]])
        X_full = np.column_stack([np.ones(T), out[:, :k]])
        y_tr = factor_matrix_full[train_mask, k]
        coef, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        out[:, k] = factor_matrix_full[:, k] - X_full @ coef

    # Standardize each column to training-period unit variance
    std_tr = out[train_mask].std(axis=0)
    out = out / np.clip(std_tr, 1e-10, None)
    return out, col_names


def _residualise(raw: pd.DataFrame, global_factors: np.ndarray,
                 train_mask: np.ndarray) -> pd.DataFrame:
    """Legacy helper kept for reference; v3 now uses _sequential_gram_schmidt."""
    X_full = np.column_stack([np.ones(len(raw)), global_factors])
    X_train = X_full[train_mask]
    out = {}
    for col in raw.columns:
        y_tr = raw.loc[train_mask, col].values
        coef, *_ = np.linalg.lstsq(X_train, y_tr, rcond=None)
        out[col] = raw[col].values - X_full @ coef
    return pd.DataFrame(out, index=raw.index)


def compute_sector_factors(
    returns: pd.DataFrame,
    asset_classes: Dict[str, list],
    train_end: str = "2023-12-31",
    n_global: int = 5,
    n_bonds: int = 2,
) -> Tuple[pd.DataFrame, Dict]:
    """Build the v3 16-factor panel: global PCA + bonds class PCA + sector."""
    T, N = returns.shape
    train_mask = np.asarray(returns.index <= train_end)
    train_data = returns.loc[train_mask].values
    all_data = returns.values

    print(f"  [v3 factors] N={N}, train_days={train_mask.sum()}/{T}")

    # Step 1: Global PCA (unchanged from v2)
    global_pca = PCA(n_components=n_global)
    global_pca.fit(train_data)
    global_factors = global_pca.transform(all_data)                # (T, 5)
    print(f"  [v3] global PCA: {n_global} factors, "
          f"exp_var={global_pca.explained_variance_ratio_.sum():.3f}")

    # Step 2: Bonds class-PCA (kept — bonds do not use industry taxonomy)
    asset_list = returns.columns.tolist()
    bonds_idx = [asset_list.index(a) for a in asset_classes["bonds"]
                 if a in asset_list]
    global_reconstructed = global_pca.inverse_transform(global_factors)
    residuals = all_data - global_reconstructed
    bonds_residuals = residuals[:, bonds_idx]
    bonds_pca = PCA(n_components=n_bonds)
    bonds_pca.fit(bonds_residuals[train_mask])
    bonds_factors = bonds_pca.transform(bonds_residuals)            # (T, 2)
    print(f"  [v3] bonds class-PCA: {n_bonds} factors, "
          f"exp_var={bonds_pca.explained_variance_ratio_.sum():.3f}")

    # Step 3: Raw stock sector factors (equal-weight means)
    stock_raw = _sector_mean_returns(returns, STOCK_SECTORS)
    print(f"  [v3] stock sectors raw ({len(stock_raw.columns)}): "
          f"{list(stock_raw.columns)}")

    # Step 4: Raw commodity GSCI sector factors
    cmdy_raw = _sector_mean_returns(returns, COMMODITY_SECTORS)
    print(f"  [v3] commodity sectors raw ({len(cmdy_raw.columns)}): "
          f"{list(cmdy_raw.columns)}")

    # Step 5: Stack all factors in priority order, then sequential Gram-
    # Schmidt orthogonalisation fitted on training data only. Order:
    #   global PCA (most general, preserved as-is except std normalisation)
    #   bonds class-PCA (orthogonal to global by PCA construction but
    #     re-orthogonalised here for safety)
    #   stock sectors (orthogonal to global and bonds)
    #   commodity sectors (orthogonal to all above)
    # After orthogonalisation every factor is also standardized to unit
    # variance on training data so β coefficients in the rolling OLS
    # remain at a consistent scale across factor types.
    raw_stack = np.concatenate([
        global_factors,
        bonds_factors,
        stock_raw.values,
        cmdy_raw.values,
    ], axis=1)
    col_names = (
        [f"global_{i+1}" for i in range(n_global)]
        + [f"bonds_{i+1}" for i in range(n_bonds)]
        + list(stock_raw.columns)
        + list(cmdy_raw.columns)
    )
    arr, col_names = _sequential_gram_schmidt(raw_stack, train_mask, col_names)
    factors_df = pd.DataFrame(arr, index=returns.index, columns=col_names)

    # Sanity: training-period off-diagonal max |correlation|
    train_corr = factors_df.loc[train_mask].corr().values
    off_diag = train_corr - np.eye(len(col_names))
    print(f"  [v3 orth] post-GS training corr: max|off-diag|="
          f"{np.abs(off_diag).max():.4f}, mean|off-diag|="
          f"{np.abs(off_diag[np.triu_indices(len(col_names), k=1)]).mean():.4f}")
    info = {
        "n_total": len(col_names),
        "n_global": n_global,
        "n_bonds": n_bonds,
        "n_stock_sectors": len(stock_raw.columns),
        "n_commodity_sectors": len(cmdy_raw.columns),
        "global_exp_var": float(global_pca.explained_variance_ratio_.sum()),
        "bonds_exp_var": float(bonds_pca.explained_variance_ratio_.sum()),
    }
    print(f"  [v3] total K = {len(col_names)}")
    return factors_df, info


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_returns, train_val_test_split, ASSET_CLASSES

    returns = load_returns()
    train_ret, _, _ = train_val_test_split(returns)
    train_end = str(train_ret.index[-1].date())
    f, info = compute_sector_factors(returns, ASSET_CLASSES, train_end=train_end)

    # Quick sanity: do the new factors correlate with V-MA and Gold-Silver?
    r = returns
    print("\n[sanity] V correlation with each stock sector factor:")
    for c in [c for c in f.columns if c.startswith("stk_")]:
        print(f"  {c:20s}  corr(V)={np.corrcoef(r['v'], f[c])[0,1]:+.3f}  "
              f"corr(MA)={np.corrcoef(r['ma'], f[c])[0,1]:+.3f}")
    print("\n[sanity] Gold/Silver correlation with each commodity sector factor:")
    for c in [c for c in f.columns if c.startswith("cmd_")]:
        print(f"  {c:20s}  corr(gold)={np.corrcoef(r['gold'], f[c])[0,1]:+.3f}  "
              f"corr(silver)={np.corrcoef(r['silver'], f[c])[0,1]:+.3f}")
