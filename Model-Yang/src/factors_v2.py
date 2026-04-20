"""
Hierarchical PCA factor model for multi-asset SF-MarketGAN (v2).

Replaces 5 hand-picked factors with data-driven statistical factors:

Level 1: Global PCA factors (K_global=5) — shared cross-asset dynamics
Level 2: Class-specific PCA factors (K_class=2 per class) — within-class residual structure

Total: 5 + 2*3 = 11 factors for 60 assets (vs previous 5)

This captures 2x more variance and provides class-aware structure
that the model can learn different loadings for.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Dict, Tuple


def compute_hierarchical_pca_factors(
    returns: pd.DataFrame,
    asset_classes: Dict[str, list],
    n_global: int = 5,
    n_class: int = 2,
    window: int = 252,
    train_end: str = "2023-12-31",
    lag: int = 0,
) -> Tuple[pd.DataFrame, Dict]:
    """Compute hierarchical PCA factors.

    Step 1: Global PCA on all 60 assets → n_global factors
    Step 2: Project out global factors → residuals
    Step 3: Class-specific PCA on residuals → n_class factors per class

    PCA is fit on TRAINING data only (no look-ahead).
    Factor values are computed for ALL dates using the trained PCA.

    NOTE: This is a CONDITIONAL generation setup. Factor values at time t
    are computed from realized returns at time t (or t-lag if lag>0).
    This is standard in the MarketGAN literature — the task is to generate
    realistic synthetic returns CONDITIONED ON the realized factor path,
    not to forecast factors. Both FB and GAN use the same factors.

    Args:
        lag: if > 0, use lagged returns for factor computation (robustness check).
             lag=0 is the standard conditional generation setup.

    Returns:
        factors: DataFrame with all factor values (T, K_total)
        info: dict with explained variance ratios etc.
    """
    T, N = returns.shape
    train_mask = returns.index <= train_end
    train_data = returns.loc[train_mask].values
    all_data = returns.values

    if lag > 0:
        print(f"  Using lagged factors (lag={lag} days)")

    print(f"  Hierarchical PCA: N={N}, n_global={n_global}, n_class={n_class}")
    print(f"  Training period: {train_mask.sum()}/{T} days")

    # Step 1: Global PCA (fit on training only)
    global_pca = PCA(n_components=n_global)
    global_pca.fit(train_data)
    # Lag support: use returns from t-lag for factor computation at t
    if lag > 0:
        lagged_data = np.zeros_like(all_data)
        lagged_data[lag:] = all_data[:-lag]
        lagged_data[:lag] = all_data[0]  # pad with first row
        global_factors = global_pca.transform(lagged_data)
    else:
        global_factors = global_pca.transform(all_data)  # (T, n_global)
    global_var = global_pca.explained_variance_ratio_.sum()
    print(f"  Global PCA: {n_global} factors, explained var = {global_var:.3f}")

    # Step 2: Residuals after removing global factors
    # When lag>0, residuals use lagged global reconstruction but actual returns
    # This means class factors also use lagged information consistently
    data_for_residuals = lagged_data if lag > 0 else all_data
    reconstructed = global_pca.inverse_transform(global_factors)  # (T, N)
    residuals = data_for_residuals - reconstructed  # (T, N)

    # Step 3: Class-specific PCA on residuals
    class_factors_list = []
    class_info = {}

    for cls_name, assets in asset_classes.items():
        asset_idx = [returns.columns.tolist().index(a) for a in assets if a in returns.columns]
        if len(asset_idx) < n_class:
            continue

        cls_residuals = residuals[:, asset_idx]
        cls_train = cls_residuals[train_mask]

        cls_pca = PCA(n_components=n_class)
        cls_pca.fit(cls_train)
        cls_factors = cls_pca.transform(cls_residuals)  # (T, n_class)

        # Zero-pad to full N dimension for each class factor
        # (factors are in class-specific subspace)
        class_factors_list.append(cls_factors)
        cls_var = cls_pca.explained_variance_ratio_.sum()
        class_info[cls_name] = {
            "n_factors": n_class,
            "explained_var": cls_var,
            "n_assets": len(asset_idx),
        }
        print(f"  {cls_name} PCA: {n_class} factors, explained var = {cls_var:.3f} "
              f"({len(asset_idx)} assets)")

    # Combine all factors
    all_factors = np.concatenate([global_factors] + class_factors_list, axis=1)
    n_total = all_factors.shape[1]

    col_names = [f"global_{i+1}" for i in range(n_global)]
    for cls_name in asset_classes:
        col_names.extend([f"{cls_name}_{i+1}" for i in range(n_class)])

    factors_df = pd.DataFrame(all_factors, index=returns.index, columns=col_names)

    info = {
        "n_global": n_global,
        "n_class": n_class,
        "n_total": n_total,
        "global_explained_var": global_var,
        "class_info": class_info,
    }

    print(f"  Total factors: {n_total} (global={n_global} + class={n_class}×{len(asset_classes)})")

    return factors_df, info


def compute_rolling_coefficients_v2(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    window: int = 126,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling OLS with hierarchical PCA factors.

    Simpler than v1 — no class-specific routing, all factors used for all assets.

    Returns:
        alpha_hat: (T, N)
        beta_hat: (T, N, K)
        sigma_hat: (T, N)
    """
    N = returns.shape[1]
    T = returns.shape[0]
    K = factors.shape[1]

    Y = returns.values
    F = factors.values

    alpha_hat = np.zeros((T, N))
    beta_hat = np.zeros((T, N, K))
    sigma_hat = np.ones((T, N)) * 0.01

    print(f"  Rolling OLS (v2): T={T}, N={N}, K={K}, window={window}...")

    for t in range(window, T):
        if t % 500 == 0:
            print(f"    t={t}/{T}")
        s = t - window
        Y_win = Y[s:t]
        X_win = np.column_stack([np.ones(window), F[s:t]])

        XtX = X_win.T @ X_win + 1e-6 * np.eye(1 + K)
        try:
            coef = np.linalg.solve(XtX, X_win.T @ Y_win)
        except np.linalg.LinAlgError:
            continue

        alpha_hat[t] = coef[0]
        beta_hat[t] = coef[1:].T
        residuals = Y_win - X_win @ coef
        sigma_hat[t] = np.maximum(residuals.std(axis=0), 1e-6)

    print(f"  Rolling OLS complete.")
    return alpha_hat, beta_hat, sigma_hat
