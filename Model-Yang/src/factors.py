"""
Factor model for multi-asset SF-MarketGAN.

Five global factors used in rolling OLS and generation:
  r_{t+1} = α_t + β_t · F_{t+1} + σ_t ⊙ ε_{t+1}

  F = [mkt, term_slope, credit, dollar, vol]

  - mkt: equal-weighted all-asset return
  - term_slope: TLT - SHY daily return spread (duration proxy)
  - credit: HYG - AGG daily return spread (credit risk proxy)
  - dollar: -BWX daily return (inverse USD proxy)
  - vol: 21-day rolling realized volatility of market factor

Class-specific factors (equity SMB/HML, bond duration/credit, commodity
energy/metals/agri) are computed for analysis but NOT used in the
generator's beta estimation — only the 5 global factors above are.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple


def compute_global_factors(
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    asset_classes: Dict[str, list],
) -> pd.DataFrame:
    """Compute global factors shared across all asset classes.

    Returns DataFrame with columns: mkt, term_slope, credit, dollar, vol
    """
    factors = pd.DataFrame(index=returns.index)

    # 1. Market factor: equal-weighted return of all 60 assets
    factors["mkt"] = returns.mean(axis=1)

    # 2. Term structure slope from macro data (if available)
    if "usd_10y_swap" in macro.columns and "usd_2y_swap" in macro.columns:
        slope = macro["usd_10y_swap"] - macro["usd_2y_swap"]
        slope = slope.reindex(returns.index, method="ffill")
        factors["term_slope"] = slope
    else:
        # Proxy: TLT - SHY spread
        if "tlt" in returns.columns and "shy" in returns.columns:
            factors["term_slope"] = returns["tlt"] - returns["shy"]
        else:
            factors["term_slope"] = 0.0

    # 3. Credit spread factor from macro (CDX IG)
    if "cdx_ig" in macro.columns:
        cdx = macro["cdx_ig"].reindex(returns.index, method="ffill")
        factors["credit"] = cdx
    else:
        # Proxy: HYG - AGG spread
        if "hyg" in returns.columns and "agg" in returns.columns:
            factors["credit"] = returns["hyg"] - returns["agg"]
        else:
            factors["credit"] = 0.0

    # 4. Dollar factor: proxy via BWX (international bonds, inversely related to USD)
    if "bwx" in returns.columns:
        factors["dollar"] = -returns["bwx"]  # Negative: strong USD = negative BWX
    else:
        factors["dollar"] = 0.0

    # 5. Volatility factor: rolling realized volatility of market factor
    factors["vol"] = factors["mkt"].rolling(21, min_periods=5).std()
    factors["vol"] = factors["vol"].fillna(factors["vol"].median())

    return factors


def compute_equity_factors(
    returns: pd.DataFrame,
    equity_assets: list,
) -> pd.DataFrame:
    """Compute equity-specific factors (SMB, HML style), vectorized."""
    eq_ret = returns[equity_assets]
    factors = pd.DataFrame(index=returns.index)
    n = len(equity_assets)
    top_k = max(n // 3, 1)

    # SMB: high vol - low vol
    roll_vol = eq_ret.rolling(63, min_periods=21).std()
    ranks = roll_vol.rank(axis=1, na_option="bottom")
    high_mask = ranks >= (n - top_k + 1)
    low_mask = ranks <= top_k
    factors["eq_smb"] = (eq_ret * high_mask).sum(axis=1) / high_mask.sum(axis=1).clip(1) - \
                         (eq_ret * low_mask).sum(axis=1) / low_mask.sum(axis=1).clip(1)

    # HML: low momentum - high momentum
    roll_ret = eq_ret.rolling(252, min_periods=63).mean()
    ranks2 = roll_ret.rank(axis=1, na_option="bottom")
    low_mask2 = ranks2 <= top_k
    high_mask2 = ranks2 >= (n - top_k + 1)
    factors["eq_hml"] = (eq_ret * low_mask2).sum(axis=1) / low_mask2.sum(axis=1).clip(1) - \
                          (eq_ret * high_mask2).sum(axis=1) / high_mask2.sum(axis=1).clip(1)

    factors = factors.fillna(0)
    return factors


def compute_bond_factors(
    returns: pd.DataFrame,
    bond_assets: list,
) -> pd.DataFrame:
    """Compute bond-specific factors.

    - Duration factor: long duration - short duration
    - Credit quality factor: credit bonds - govt bonds
    """
    bd_ret = returns[bond_assets]
    factors = pd.DataFrame(index=returns.index)

    # Duration factor: TLT (20+yr) vs SHY (1-3yr)
    long_dur = [a for a in ["tlt", "ief", "govt", "bwx"] if a in bond_assets]
    short_dur = [a for a in ["shy", "flot", "mint", "vcsh"] if a in bond_assets]
    if long_dur and short_dur:
        factors["bd_duration"] = bd_ret[long_dur].mean(axis=1) - bd_ret[short_dur].mean(axis=1)
    else:
        factors["bd_duration"] = 0.0

    # Credit quality factor: credit bonds vs govt bonds
    credit = [a for a in ["hyg", "emb", "lqd", "igib", "vcit"] if a in bond_assets]
    govt = [a for a in ["agg", "bnd", "govt", "schz"] if a in bond_assets]
    if credit and govt:
        factors["bd_credit"] = bd_ret[credit].mean(axis=1) - bd_ret[govt].mean(axis=1)
    else:
        factors["bd_credit"] = 0.0

    return factors


def compute_commodity_factors(
    returns: pd.DataFrame,
    commodity_assets: list,
) -> pd.DataFrame:
    """Compute commodity-specific factors.

    - Energy factor
    - Metals factor
    - Agriculture factor
    """
    cm_ret = returns[commodity_assets]
    factors = pd.DataFrame(index=returns.index)

    energy = [a for a in ["crude_oil", "natural_gas", "uga", "bno", "gsg"] if a in commodity_assets]
    metals = [a for a in ["gold", "silver", "copper", "platinum", "pall"] if a in commodity_assets]
    agri = [a for a in ["corn", "wheat", "soybeans", "coffee", "dba", "cane"] if a in commodity_assets]

    if energy:
        factors["cm_energy"] = cm_ret[energy].mean(axis=1)
    else:
        factors["cm_energy"] = 0.0

    if metals:
        factors["cm_metals"] = cm_ret[metals].mean(axis=1)
    else:
        factors["cm_metals"] = 0.0

    if agri:
        factors["cm_agri"] = cm_ret[agri].mean(axis=1)
    else:
        factors["cm_agri"] = 0.0

    return factors


def build_hierarchical_factors(
    returns: pd.DataFrame,
    macro: pd.DataFrame,
    asset_classes: Dict[str, list],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Build the full hierarchical factor structure.

    Returns:
        global_factors: DataFrame with global factors (N_global columns)
        class_factors: Dict mapping class name -> DataFrame of class-specific factors
        all_factors: Combined DataFrame for convenience
    """
    global_factors = compute_global_factors(returns, macro, asset_classes)

    class_factors = {}
    if "stocks" in asset_classes:
        class_factors["stocks"] = compute_equity_factors(returns, asset_classes["stocks"])
    if "bonds" in asset_classes:
        class_factors["bonds"] = compute_bond_factors(returns, asset_classes["bonds"])
    if "commodities" in asset_classes:
        class_factors["commodities"] = compute_commodity_factors(returns, asset_classes["commodities"])

    # Combined for convenience
    all_parts = [global_factors]
    for name, df in class_factors.items():
        all_parts.append(df)
    all_factors = pd.concat(all_parts, axis=1)

    return global_factors, class_factors, all_factors


def compute_rolling_coefficients(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    asset_classes: Dict[str, list],
    class_factors: Dict[str, pd.DataFrame],
    global_factors: pd.DataFrame,
    window: int = 126,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling OLS factor model coefficients for each asset.

    Vectorized: at each time step t, solve OLS for all N assets simultaneously.

    Returns:
        alpha_hat: (T, N) array of intercepts
        beta_hat: (T, N, K) array of factor loadings (K = n_global only)
        sigma_hat: (T, N) array of residual volatilities
    """
    N = returns.shape[1]
    T = returns.shape[0]
    n_global = global_factors.shape[1]
    K = n_global  # Simplified: only use global factors for β

    alpha_hat = np.zeros((T, N))
    beta_hat = np.zeros((T, N, K))
    sigma_hat = np.ones((T, N)) * 0.01

    Y = returns.values          # (T, N)
    F = global_factors.values   # (T, K_global)

    print(f"  Computing rolling OLS: T={T}, N={N}, K={K}, window={window}...")

    for t in range(window, T):
        if t % 500 == 0:
            print(f"    t={t}/{T}")
        s = t - window
        # Y_win: (window, N), X_win: (window, 1+K)
        Y_win = Y[s:t]
        X_win = np.column_stack([np.ones(window), F[s:t]])  # (window, 1+K)

        # Solve OLS for all N assets at once: coef = (X'X)^{-1} X'Y
        XtX = X_win.T @ X_win + 1e-6 * np.eye(1 + K)
        XtY = X_win.T @ Y_win  # (1+K, N)
        try:
            coef = np.linalg.solve(XtX, XtY)  # (1+K, N)
        except np.linalg.LinAlgError:
            continue

        alpha_hat[t] = coef[0]           # (N,)
        beta_hat[t] = coef[1:].T         # (N, K)

        residuals = Y_win - X_win @ coef  # (window, N)
        sigma_hat[t] = np.maximum(residuals.std(axis=0), 1e-6)

    print(f"  Rolling OLS complete.")
    return alpha_hat, beta_hat, sigma_hat
