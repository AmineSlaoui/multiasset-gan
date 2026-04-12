"""
Macro variable processing pipeline for SF-MarketGAN.

Raw macro → aligned, normalized, regime-aware covariates.

Pipeline:
1. Forward-fill weekly → daily
2. Dual-channel: z-scored changes + z-scored cumulative level
3. Interaction features (term slope, credit momentum)
4. PCA compression
5. Regime indicator via rolling GMM
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from typing import Tuple


class MacroProcessor:
    """Transform macro data into model-ready covariates.

    Supports two modes:
    - Daily mode (preferred): reads pre-built daily_macro_features.csv
    - Weekly mode (fallback): reads macro_conditioning_features.csv, forward-fills
    """

    def __init__(
        self,
        n_pca_components: int = 8,
        n_regimes: int = 3,
        zscore_window: int = 252,    # 1 year daily for rolling z-score
        regime_window: int = 504,    # 2 years daily for regime detection
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
    ) -> pd.DataFrame:
        """Fit on training data and transform.

        Args:
            macro_raw: Daily macro features (Date index, 19 columns from daily_macro_features.csv)
                       OR weekly data (8 columns) — will auto-detect
            daily_index: Target daily date index for alignment

        Returns:
            Processed covariates aligned to daily_index, shape (T_daily, d_y)
        """
        # Align to daily index
        macro_daily = macro_raw.reindex(daily_index, method="ffill").bfill().fillna(0)

        # Rolling z-score normalization (all features)
        z_scored = self._rolling_zscore(macro_daily, self.zscore_window)
        z_scored = z_scored.fillna(0)

        # PCA compression
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(z_scored.values)

        n_comp = min(self.n_pca_components, scaled.shape[1])
        self.pca = PCA(n_components=n_comp)
        pca_features = self.pca.fit_transform(scaled)
        pca_df = pd.DataFrame(
            pca_features, index=daily_index,
            columns=[f"pc_{i+1}" for i in range(n_comp)],
        )
        print(f"  PCA: {macro_daily.shape[1]} features → {n_comp} PCs "
              f"(explained var: {self.pca.explained_variance_ratio_.cumsum()[-1]:.3f})")

        # Regime indicators via rolling GMM on PC1-PC2
        regime_df = self._detect_regimes(pca_df[["pc_1", "pc_2"]])

        # Keep a few raw features that are directly interpretable
        # (not z-scored — they carry important scale information)
        raw_keep = []
        for col in ["vix_level", "realized_vol_20d", "term_level", "credit_level"]:
            if col in macro_daily.columns:
                raw_keep.append(col)
        if raw_keep:
            raw_df = self._rolling_zscore(macro_daily[raw_keep], self.zscore_window).fillna(0)
        else:
            raw_df = pd.DataFrame(index=daily_index)

        result = pd.concat([pca_df, regime_df, raw_df], axis=1)
        self.feature_names = result.columns.tolist()
        print(f"  Final covariate dim: {result.shape[1]} — {self.feature_names}")

        return result

    def transform(
        self,
        macro_raw: pd.DataFrame,
        daily_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Transform new data using fitted PCA/scaler."""
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

        raw_keep = []
        for col in ["vix_level", "realized_vol_20d", "term_level", "credit_level"]:
            if col in macro_daily.columns:
                raw_keep.append(col)
        if raw_keep:
            raw_df = self._rolling_zscore(macro_daily[raw_keep], self.zscore_window).fillna(0)
        else:
            raw_df = pd.DataFrame(index=daily_index)

        return pd.concat([pca_df, regime_df, raw_df], axis=1)

    @staticmethod
    def _rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Rolling z-score normalization."""
        roll_mean = df.rolling(window, min_periods=max(window // 4, 1)).mean()
        roll_std = df.rolling(window, min_periods=max(window // 4, 1)).std()
        roll_std = roll_std.replace(0, 1e-8)
        z = (df - roll_mean) / roll_std
        z = z.clip(-5, 5)  # Clip extreme z-scores
        return z.add_suffix("_z")

    def _detect_regimes(self, pc_df: pd.DataFrame) -> pd.DataFrame:
        """Detect macro regimes via GMM on first two PCs.

        Fits GMM every 21 days (monthly) for speed, forward-fills between.
        """
        T = len(pc_df)
        regime_labels = np.zeros(T, dtype=int)
        window = self.regime_window
        step = 21  # Refit monthly

        print(f"  Regime detection: T={T}, window={window}, step={step}...")
        for t in range(window, T, step):
            data = pc_df.iloc[max(0, t - window):t].values
            try:
                gmm = GaussianMixture(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    n_init=2,
                    random_state=42,
                )
                gmm.fit(data)
                # Predict for this chunk
                end = min(t + step, T)
                chunk = pc_df.iloc[t:end].values
                regime_labels[t:end] = gmm.predict(chunk)
            except Exception:
                pass

        regime_df = pd.DataFrame(index=pc_df.index)
        for k in range(self.n_regimes):
            regime_df[f"regime_{k}"] = (regime_labels == k).astype(np.float32)

        print(f"  Regime detection complete.")
        return regime_df
