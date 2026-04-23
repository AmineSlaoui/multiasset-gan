"""
Data loading, preprocessing, and windowed dataset creation for Multi-Asset MarketGAN.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
from pathlib import Path


# ── Asset class definitions ──────────────────────────────────────────
ASSET_CLASSES = {
    "bonds": [
        "agg", "bnd", "emb", "hyg", "ief", "lqd", "mub", "shy", "tip", "tlt",
        "vcit", "vcsh", "govt", "bndx", "igib", "flot", "schz", "bwx", "mint", "vtip",
    ],
    "commodities": [
        "coffee", "copper", "corn", "crude_oil", "gold", "natural_gas",
        "platinum", "silver", "soybeans", "wheat",
        "dba", "dbc", "uga", "bno", "pall", "cane", "gsg", "djp", "ftgc", "pdbc",
    ],
    "stocks": [
        "aapl", "amzn", "jnj", "jpm", "msft", "nflx", "nvda", "tsla", "v", "xom",
        "googl", "meta", "brk_b", "unh", "hd", "pg", "ma", "dis", "cost", "intc",
    ],
}

N_ASSETS = sum(len(v) for v in ASSET_CLASSES.values())  # 60


def load_returns(data_dir: str = "data") -> pd.DataFrame:
    """Load the pre-built log returns CSV."""
    path = Path(data_dir) / "all_assets_log_returns.csv"
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def load_macro(data_dir: str = "data") -> pd.DataFrame:
    """Load macro conditioning features.

    Prefers daily_macro_features.csv (19 features, daily).
    Falls back to macro_conditioning_features.csv (8 features, weekly).
    """
    daily_path = Path(data_dir) / "macro" / "daily_macro_features.csv"
    weekly_path = Path(data_dir) / "macro" / "macro_conditioning_features.csv"

    if daily_path.exists():
        df = pd.read_csv(daily_path, index_col="Date", parse_dates=True)
        print(f"  Loaded daily macro: {df.shape} ({list(df.columns[:5])}...)")
        return df
    else:
        df = pd.read_csv(weekly_path, index_col="Date", parse_dates=True)
        print(f"  Loaded weekly macro (fallback): {df.shape}")
        return df


def train_val_test_split(
    returns: pd.DataFrame,
    train_end: str = "2023-12-31",
    val_end: str = "2024-12-31",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test.

    Train: 2011-01 to 2023-12 (~13 years, 3270 days)
    Val:   2024-01 to 2024-12 (~1 year, 252 days)
    Test:  2025-01 to 2025-12 (~1 year, 251 days, OOS)
    """
    train = returns.loc[:train_end]
    val = returns.loc[train_end:val_end].iloc[1:]
    test = returns.loc[val_end:].iloc[1:]
    return train, val, test


# Rolling window for OLS coefficient estimation (trading days)
# 126d (half-year) balances R² (0.610) and beta stability (0.127)
# vs 252d (1yr, MarketGAN default): slower regime response
# vs 63d (3mo): R² similar but beta 2x more volatile
ROLLING_WINDOW = 126


class MultiAssetDataset(Dataset):
    """Windowed dataset for MarketGAN training.

    Each sample is a window of (returns, factors, covariates, coefficients).
    """

    def __init__(
        self,
        returns: np.ndarray,           # (T, N)
        global_factors: np.ndarray,    # (T, K_global)
        class_factors_list: list,      # List of (T, K_class_i) arrays per class
        covariates: np.ndarray,        # (T, D_cov) macro covariates
        alpha_hat: np.ndarray,         # (T, N)
        beta_hat: np.ndarray,          # (T, N, K)
        sigma_hat: np.ndarray,         # (T, N)
        window_size: int = 127,        # RFS of TCN (same as MarketGAN)
        asset_class_indices: Optional[Dict[str, list]] = None,
    ):
        self.returns = returns.astype(np.float32)
        self.global_factors = global_factors.astype(np.float32)
        self.class_factors_list = [cf.astype(np.float32) for cf in class_factors_list]
        self.covariates = covariates.astype(np.float32)
        self.alpha_hat = alpha_hat.astype(np.float32)
        self.beta_hat = beta_hat.astype(np.float32)
        self.sigma_hat = sigma_hat.astype(np.float32)
        self.window_size = window_size
        self.asset_class_indices = asset_class_indices

        self.T = returns.shape[0]
        self.valid_start = window_size  # Need at least window_size history

    def __len__(self):
        return max(0, self.T - self.valid_start - 1)

    def __getitem__(self, idx):
        t = self.valid_start + idx
        # Window: [t - window_size, t] for history, t+1 for target
        hist_slice = slice(t - self.window_size, t + 1)
        target_t = t + 1 if t + 1 < self.T else t

        return {
            "returns_window": self.returns[hist_slice],          # (W+1, N)
            "returns_target": self.returns[target_t],            # (N,)
            "global_factors_window": self.global_factors[hist_slice],
            "global_factors_target": self.global_factors[target_t] if target_t < self.T else self.global_factors[t],
            "covariates_window": self.covariates[hist_slice],
            "alpha_hat_t": self.alpha_hat[t],                    # (N,)
            "beta_hat_t": self.beta_hat[t],                      # (N, K)
            "sigma_hat_t": self.sigma_hat[t],                    # (N,)
        }


def build_dataset(
    returns: pd.DataFrame,
    global_factors: pd.DataFrame,
    class_factors: Dict[str, pd.DataFrame],
    macro: pd.DataFrame,
    alpha_hat: np.ndarray,
    beta_hat: np.ndarray,
    sigma_hat: np.ndarray,
    window_size: int = 127,
) -> MultiAssetDataset:
    """Build a MultiAssetDataset from DataFrames."""
    # Align macro to returns index
    macro_aligned = macro.reindex(returns.index, method="ffill").fillna(0)

    # Build class factors list (aligned to returns index)
    class_factors_list = []
    for cls_name in ["bonds", "commodities", "stocks"]:
        if cls_name in class_factors:
            cf = class_factors[cls_name].reindex(returns.index).fillna(0)
            class_factors_list.append(cf.values)

    # Asset class indices (which columns belong to which class)
    asset_list = returns.columns.tolist()
    asset_class_indices = {}
    for cls_name, assets in ASSET_CLASSES.items():
        indices = [asset_list.index(a) for a in assets if a in asset_list]
        asset_class_indices[cls_name] = indices

    return MultiAssetDataset(
        returns=returns.values,
        global_factors=global_factors.reindex(returns.index).fillna(0).values,
        class_factors_list=class_factors_list,
        covariates=macro_aligned.values,
        alpha_hat=alpha_hat,
        beta_hat=beta_hat,
        sigma_hat=sigma_hat,
        window_size=window_size,
        asset_class_indices=asset_class_indices,
    )


def get_dataloaders(
    returns: pd.DataFrame,
    global_factors: pd.DataFrame,
    class_factors: Dict[str, pd.DataFrame],
    macro: pd.DataFrame,
    alpha_hat: np.ndarray,
    beta_hat: np.ndarray,
    sigma_hat: np.ndarray,
    window_size: int = 127,
    batch_size: int = 64,
    train_end: str = "2020-12-31",
    val_end: str = "2022-12-31",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders."""
    train_ret, val_ret, test_ret = train_val_test_split(returns, train_end, val_end)

    # Split coefficients accordingly
    train_idx = returns.index.get_indexer(train_ret.index)
    val_idx = returns.index.get_indexer(val_ret.index)
    test_idx = returns.index.get_indexer(test_ret.index)

    datasets = {}
    for name, ret, idx in [
        ("train", train_ret, train_idx),
        ("val", val_ret, val_idx),
        ("test", test_ret, test_idx),
    ]:
        ds = build_dataset(
            returns=ret,
            global_factors=global_factors,
            class_factors=class_factors,
            macro=macro,
            alpha_hat=alpha_hat[idx],
            beta_hat=beta_hat[idx],
            sigma_hat=sigma_hat[idx],
            window_size=window_size,
        )
        datasets[name] = ds

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
