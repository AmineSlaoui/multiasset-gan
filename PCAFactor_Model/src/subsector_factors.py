"""Sub-sector factor extension for the MarketGAN hierarchical factor model.

Adds a third level below (global, class) PCA factors: each named sub-group of
assets contributes one equal-weight-mean factor. This directly captures
within-class industry co-movement (e.g. Visa / Mastercard payments, Gold /
Silver precious metals, TLT / IEF Treasury duration) that class-level PCA
under-explains (stocks class-PCA only accounts for ~30% of within-class
variance in this universe).

Usage:
    subs = compute_subsector_factors(returns)  # DataFrame (T, n_subsectors)
    # concat alongside the 11 hierarchical PCA factors for a 11 + n_sub
    # factor count that will be fed to rolling OLS + the generator.

This is a DATA-side patch. Retraining the v9 generator is required for the
added β channels to be learned. No changes to sf_marketgan_v9.py — only the
input factor tensor width grows.
"""
import numpy as np
import pandas as pd
from typing import Dict


SUBSECTORS: Dict[str, list] = {
    # ── Stocks sub-sectors ──────────────────────────────────────────
    "tech_mega":       ["aapl", "msft", "googl", "meta", "nvda"],
    "semis":           ["nvda", "intc"],
    "payments":        ["v", "ma"],
    "consumer_disc":   ["amzn", "tsla", "hd", "cost", "nflx", "dis"],
    "consumer_staples":["pg", "cost"],
    "health_care":     ["jnj", "unh"],
    "financials":      ["jpm", "brk_b"],
    "energy_eq":       ["xom"],
    # ── Commodities sub-sectors ────────────────────────────────────
    "precious_metals": ["gold", "silver", "platinum", "pall"],
    "energy_futs":     ["crude_oil", "natural_gas", "uga", "bno"],
    "base_metals":     ["copper"],
    "grains":          ["corn", "soybeans", "wheat"],
    "softs":           ["coffee", "cane"],
    "broad_cmdty":     ["dbc", "gsg", "djp", "ftgc", "pdbc", "dba"],
    # ── Bonds sub-sectors ──────────────────────────────────────────
    "short_duration":  ["shy", "mint", "flot", "vcsh", "bwx"],
    "inter_treasury":  ["ief", "govt", "agg", "bnd", "schz", "bndx"],
    "long_treasury":   ["tlt"],
    "credit_ig":       ["lqd", "vcit", "igib"],
    "credit_hy_em":    ["hyg", "emb"],
    "inflation_linked":["tip", "vtip"],
    "muni":            ["mub"],
}


def compute_subsector_factors(
    returns: pd.DataFrame,
    subsectors: Dict[str, list] = None,
    min_assets: int = 2,
) -> pd.DataFrame:
    """Equal-weight mean return per named sub-group.

    Single-asset groups are dropped (a 1-asset "factor" is just the asset
    itself — adds no information, breaks regression identifiability).
    """
    if subsectors is None:
        subsectors = SUBSECTORS

    cols = returns.columns.tolist()
    out = {}
    for name, members in subsectors.items():
        present = [m for m in members if m in cols]
        if len(present) < min_assets:
            continue
        out[f"sub_{name}"] = returns[present].mean(axis=1)

    return pd.DataFrame(out, index=returns.index)


def orthogonalise_against(subsector_factors: pd.DataFrame,
                          base_factors: pd.DataFrame,
                          train_end: str) -> pd.DataFrame:
    """Project sub-sector factors onto the orthogonal complement of the base
    (global + class PCA) factor space, fitted on training period only.

    Without this step the sub-sector means are highly collinear with the
    global market factor (which also contains an average-return component).
    Orthogonalising keeps the Hessian of the rolling OLS well-conditioned.
    """
    train_mask = subsector_factors.index <= train_end
    X = base_factors.loc[train_mask].values              # (T_tr, K_base)
    # Add intercept
    X = np.column_stack([np.ones(len(X)), X])

    out = {}
    for col in subsector_factors.columns:
        y_tr = subsector_factors.loc[train_mask, col].values
        coef = np.linalg.lstsq(X, y_tr, rcond=None)[0]
        X_full = np.column_stack([np.ones(len(base_factors)), base_factors.values])
        resid = subsector_factors[col].values - X_full @ coef
        out[col] = resid

    return pd.DataFrame(out, index=subsector_factors.index)


if __name__ == "__main__":
    # Quick sanity check
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_returns, train_val_test_split, ASSET_CLASSES
    from factors_pca import compute_hierarchical_pca_factors

    returns = load_returns()
    train_ret, _, _ = train_val_test_split(returns)
    train_end = str(train_ret.index[-1].date())

    sub = compute_subsector_factors(returns)
    print(f"[subsector] {sub.shape[1]} groups: {list(sub.columns)}")

    base, _ = compute_hierarchical_pca_factors(
        returns, ASSET_CLASSES, n_global=5, n_class=2, train_end=train_end)
    sub_orth = orthogonalise_against(sub, base, train_end)
    print(f"[orth] sub_payments residual std after orth: "
          f"{sub_orth['sub_payments'].std():.4e} "
          f"(before: {sub['sub_payments'].std():.4e})")

    # Check: does sub_payments capture V-MA co-movement?
    r = returns[["v", "ma"]].values
    print(f"[check] V-MA raw corr: {np.corrcoef(r.T)[0,1]:.3f}")
    print(f"[check] V corr w/ sub_payments orth: "
          f"{np.corrcoef(returns['v'].values, sub_orth['sub_payments'].values)[0,1]:.3f}")
    print(f"[check] MA corr w/ sub_payments orth: "
          f"{np.corrcoef(returns['ma'].values, sub_orth['sub_payments'].values)[0,1]:.3f}")
