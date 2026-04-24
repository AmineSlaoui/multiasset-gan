"""
Macro variable processing pipeline for SF-MarketGAN.

Pipeline:
1. Rolling z-score normalization (look-back only, no future info)
2. PCA compression (fit on training period only)
3. Regime indicator via walk-forward GMM
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from typing import Optional


class MacroProcessor:
    """Transform macro data into model-ready covariates.

    IMPORTANT: fit_transform() must receive train_end to prevent look-ahead.
    PCA and StandardScaler are fit ONLY on data up to train_end, then
    applied (transform) to the full index including val/test.
    """

    def __init__(
        self,
        n_pca_components: int = 8,
        n_regimes: int = 3,
        zscore_window: int = 252,
        regime_window: int = 504,
    ):
        self.n_pca_components = n_pca_components
        self.n_regimes = n_regimes
        self.zscore_window = zscore_window
        self.regime_window = regime_window

        self.pca = None
        self.scaler = None
        self.feature_names = None

    def fit_transform(
        self,
        macro_raw: pd.DataFrame,
        daily_index: pd.DatetimeIndex,
        train_end: Optional[str] = "2023-12-31",
    ) -> pd.DataFrame:
        """Fit PCA/scaler on training period only, transform full index.

        Args:
            macro_raw: Daily macro features (Date index)
            daily_index: Full daily date index (train+val+test)
            train_end: Last date of training period. PCA/scaler fit
                       ONLY on data up to this date. Set None to fit
                       on all data (NOT recommended for OOS evaluation).
        """
        # Align to daily index
        macro_daily = macro_raw.reindex(daily_index, method="ffill").bfill().fillna(0)

        # Rolling z-score (look-back only — no future info by construction)
        z_scored = self._rolling_zscore(macro_daily, self.zscore_window)
        z_scored = z_scored.fillna(0)

        # Fit PCA/scaler on TRAINING PERIOD ONLY
        if train_end is not None:
            train_mask = z_scored.index <= train_end
            z_train = z_scored.loc[train_mask].values
            print(f"  PCA/scaler fit on training period only (≤{train_end}, "
                  f"{train_mask.sum()}/{len(z_scored)} days)")
        else:
            z_train = z_scored.values
            print(f"  WARNING: PCA/scaler fit on ALL data (no train_end specified)")

        self.scaler = StandardScaler()
        self.scaler.fit(z_train)  # fit on training only
        scaled_all = self.scaler.transform(z_scored.values)  # transform all

        n_comp = min(self.n_pca_components, scaled_all.shape[1])
        self.pca = PCA(n_components=n_comp)
        self.pca.fit(scaled_all[:len(z_train)])  # fit on training only
        pca_all = self.pca.transform(scaled_all)  # transform all

        pca_df = pd.DataFrame(
            pca_all, index=daily_index,
            columns=[f"pc_{i+1}" for i in range(n_comp)],
        )
        print(f"  PCA: {macro_daily.shape[1]} features → {n_comp} PCs "
              f"(explained var: {self.pca.explained_variance_ratio_.cumsum()[-1]:.3f})")

        # Walk-forward regime detection (uses only past data at each step)
        regime_df = self._detect_regimes(pca_df[["pc_1", "pc_2"]])

        # Interpretable raw features (rolling z-score is look-back only)
        raw_keep = [c for c in ["vix_level", "realized_vol_20d", "term_level", "credit_level"]
                    if c in macro_daily.columns]
        raw_df = self._rolling_zscore(macro_daily[raw_keep], self.zscore_window).fillna(0) \
                 if raw_keep else pd.DataFrame(index=daily_index)

        result = pd.concat([pca_df, regime_df, raw_df], axis=1)
        self.feature_names = result.columns.tolist()
        print(f"  Final covariate dim: {result.shape[1]} — {self.feature_names}")

        return result

    def transform(
        self,
        macro_raw: pd.DataFrame,
        daily_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Transform new data using previously fitted PCA/scaler."""
        macro_daily = macro_raw.reindex(daily_index, method="ffill").bfill().fillna(0)
        z_scored = self._rolling_zscore(macro_daily, self.zscore_window).fillna(0)
        scaled = self.scaler.transform(z_scored.values)
        pca_features = self.pca.transform(scaled)

        n_comp = pca_features.shape[1]
        pca_df = pd.DataFrame(
            pca_features, index=daily_index,
            columns=[f"pc_{i+1}" for i in range(n_comp)],
        )
        regime_df = self._detect_regimes(pca_df[["pc_1", "pc_2"]])

        raw_keep = [c for c in ["vix_level", "realized_vol_20d", "term_level", "credit_level"]
                    if c in macro_daily.columns]
        raw_df = self._rolling_zscore(macro_daily[raw_keep], self.zscore_window).fillna(0) \
                 if raw_keep else pd.DataFrame(index=daily_index)

        return pd.concat([pca_df, regime_df, raw_df], axis=1)

    @staticmethod
    def _rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Rolling z-score normalization (look-back only, no future info)."""
        roll_mean = df.rolling(window, min_periods=max(window // 4, 1)).mean()
        roll_std = df.rolling(window, min_periods=max(window // 4, 1)).std()
        roll_std = roll_std.replace(0, 1e-8)
        z = (df - roll_mean) / roll_std
        return z.clip(-5, 5).add_suffix("_z")

    def _detect_regimes(self, pc_df: pd.DataFrame) -> pd.DataFrame:
        """Walk-forward regime detection via rolling GMM.

        At each refit point t, GMM is fit on [t-window, t) only —
        no future data is used. Predictions cover [t, t+step).
        """
        T = len(pc_df)
        regime_labels = np.zeros(T, dtype=int)
        window = self.regime_window
        step = 21

        for t in range(window, T, step):
            data = pc_df.iloc[max(0, t - window):t].values  # past only
            try:
                gmm = GaussianMixture(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    n_init=2,
                    random_state=42,
                )
                gmm.fit(data)
                end = min(t + step, T)
                regime_labels[t:end] = gmm.predict(pc_df.iloc[t:end].values)
            except Exception:
                pass

        regime_df = pd.DataFrame(index=pc_df.index)
        for k in range(self.n_regimes):
            regime_df[f"regime_{k}"] = (regime_labels == k).astype(np.float32)

        return regime_df
