"""
External baselines for SF-MarketGAN comparison.

All baselines generate synthetic returns with shape (n_paths, T, N)
and are evaluated with the same metrics as SF-MarketGAN.

Baselines:
1. Factor Bootstrap — MarketGAN's standard baseline
2. DCC-GARCH — Traditional multivariate volatility model
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats


# ── 1. Factor Bootstrap ──────────────────────────────────────────────

class FactorBootstrap:
    """MarketGAN-style factor-model bootstrap.

    Generates synthetic returns by combining rolling OLS estimates
    with independently drawn Gaussian residuals:
      r̂_{t+1} = α̂_t + β̂_t · F_{t+1} + σ̂_t ⊙ ε_{t+1},  ε ~ N(0, I)

    This is the transparent baseline from the MarketGAN paper.
    """

    def __init__(self, window: int = 126):
        self.window = window

    def fit(self, returns: np.ndarray, factors: np.ndarray):
        """Compute rolling OLS estimates.

        NOTE: Rolling OLS at time t uses ONLY data from [t-window, t).
        This is strictly causal — no future information leaks.
        Fitting on the full time series is correct because each estimate
        only looks backward. This is standard in rolling-window models.

        returns: (T, N) log returns
        factors: (T, K) factor values
        """
        T, N = returns.shape
        K = factors.shape[1]
        self.N = N
        self.K = K

        self.alpha = np.zeros((T, N))
        self.beta = np.zeros((T, N, K))
        self.sigma = np.ones((T, N)) * 0.01

        for t in range(self.window, T):
            s = t - self.window
            Y = returns[s:t]
            X = np.column_stack([np.ones(self.window), factors[s:t]])
            XtX = X.T @ X + 1e-6 * np.eye(1 + K)
            try:
                coef = np.linalg.solve(XtX, X.T @ Y)
            except np.linalg.LinAlgError:
                continue
            self.alpha[t] = coef[0]
            self.beta[t] = coef[1:].T
            residuals = Y - X @ coef
            self.sigma[t] = np.maximum(residuals.std(axis=0), 1e-6)

        print(f"  FactorBootstrap fitted: T={T}, N={N}, K={K}, window={self.window}")

    def fit_shrunk_residuals(self, returns: np.ndarray, factors: np.ndarray,
                              lam: float = 0.2, L_eps: np.ndarray = None):
        """Fit a shrunk residual *correlation* (Ledoit-Wolf toward identity)
        from rolling OLS residuals. The generator then combines this static
        correlation with the time-varying per-asset σ̂_t from `fit`:
          ε_t = diag(σ̂_t) · L_ρ · z,    z ~ N(0, I)
        so Cov(ε_t) = diag(σ̂_t) · C_shrunk · diag(σ̂_t).

        Rationale: v7d's `residual_cholesky.npy` is a shrunk *correlation* with
        unit diagonal, applied together with σ̂_t · ε in the generator. For an
        apples-to-apples FB+shrunk baseline, we replicate that structure.

        If `L_eps` is provided, we use the existing Cholesky of the shrunk
        correlation (the one v7d was trained against); otherwise we fit a new
        one from the training residuals using Ledoit-Wolf shrinkage toward I.
        """
        T, N = returns.shape
        K = factors.shape[1]

        if L_eps is not None:
            self.L_rho = np.asarray(L_eps, dtype=np.float64)
        else:
            resid_stack = []
            for t in range(self.window, T):
                s = t - self.window
                Y = returns[s:t]
                X = np.column_stack([np.ones(self.window), factors[s:t]])
                try:
                    coef = np.linalg.solve(X.T @ X + 1e-6 * np.eye(1 + K), X.T @ Y)
                except np.linalg.LinAlgError:
                    continue
                resid_stack.append(Y - X @ coef)
            resid_all = np.vstack(resid_stack)
            sd = resid_all.std(axis=0)
            sd = np.maximum(sd, 1e-8)
            rho_sample = np.corrcoef(resid_all.T)
            rho_sample = np.nan_to_num(rho_sample, nan=0.0)
            rho_shrunk = (1 - lam) * rho_sample + lam * np.eye(N)
            rho_shrunk = rho_shrunk + 1e-8 * np.eye(N)
            self.L_rho = np.linalg.cholesky(rho_shrunk)
        self.lam_shrink = lam
        off_norm = np.linalg.norm(self.L_rho @ self.L_rho.T - np.eye(N))
        print(f"  FB+shrunk correlation fitted: lam={lam}, ||C − I||_F={off_norm:.3f}")

    def generate(self, factors: np.ndarray, start_idx: int,
                 n_paths: int = 100) -> np.ndarray:
        """Generate synthetic returns for the OOS period.

        factors: (T_full, K) — full factor time series
        start_idx: index of first OOS day
        n_paths: number of synthetic paths

        Returns: (n_paths, T_oos, N)
        """
        T_oos = len(factors) - start_idx
        paths = np.zeros((n_paths, T_oos, self.N))

        use_shrunk = hasattr(self, "L_rho")

        for p in range(n_paths):
            for t_rel in range(T_oos):
                t = start_idx + t_rel
                if t >= len(self.alpha):
                    continue
                if use_shrunk:
                    # ε_t = σ̂_t · L_ρ · z,  z ~ N(0, I)
                    # so Cov(ε_t) = diag(σ̂_t) · C_shrunk · diag(σ̂_t).
                    z = np.random.randn(self.N)
                    eps = self.sigma[t] * (self.L_rho @ z)
                    r = self.alpha[t] + self.beta[t] @ factors[t] + eps
                else:
                    eps = np.random.randn(self.N)
                    r = (self.alpha[t]
                         + self.beta[t] @ factors[t]
                         + self.sigma[t] * eps)
                paths[p, t_rel] = r

        return paths


# ── 2. DCC-GARCH ─────────────────────────────────────────────────────

class SimpleDCC:
    """Simplified DCC(1,1)-GARCH(1,1) for multivariate return generation.

    Two-stage estimation:
    1. Univariate GARCH(1,1) for each asset's conditional variance
    2. DCC(1,1) for dynamic conditional correlation

    Generates returns via: r_t = σ_t ⊙ R_t^{1/2} · z_t, z~N(0,I)
    where σ_t are GARCH volatilities and R_t is DCC correlation matrix.

    Simplified implementation (no MLE, uses moment-based estimation)
    for fair comparison — production DCC would use rmgarch in R.
    """

    def __init__(self):
        self.fitted = False

    def fit(self, returns: np.ndarray):
        """Fit GARCH(1,1) per asset + DCC(1,1).

        returns: (T, N)
        """
        T, N = returns.shape
        self.N = N
        self.T = T

        # Stage 1: Univariate GARCH(1,1) for each asset
        # h_t = omega + alpha * r²_{t-1} + beta * h_{t-1}
        self.garch_params = []  # (omega, alpha, beta) per asset
        self.cond_vol = np.zeros((T, N))

        for i in range(N):
            r = returns[:, i]
            var_r = r.var()

            # Moment-based GARCH estimation (Engle & Bollerslev)
            alpha_g = 0.05
            beta_g = 0.90
            omega = var_r * (1 - alpha_g - beta_g)
            omega = max(omega, 1e-10)

            self.garch_params.append((omega, alpha_g, beta_g))

            # Compute conditional variance path
            h = np.zeros(T)
            h[0] = var_r
            for t in range(1, T):
                h[t] = omega + alpha_g * r[t-1]**2 + beta_g * h[t-1]
                h[t] = max(h[t], 1e-10)
            self.cond_vol[:, i] = np.sqrt(h)

        # Standardized residuals
        self.std_resid = returns / np.maximum(self.cond_vol, 1e-8)

        # Stage 2: DCC(1,1)
        # Q_t = (1-a-b)*Q̄ + a*(e_{t-1} e_{t-1}') + b*Q_{t-1}
        self.Q_bar = np.corrcoef(self.std_resid.T)  # Unconditional correlation
        self.dcc_a = 0.01
        self.dcc_b = 0.95

        # Compute DCC path (store last Q for generation)
        Q = self.Q_bar.copy()
        for t in range(1, T):
            e = self.std_resid[t-1:t].T  # (N, 1)
            Q = ((1 - self.dcc_a - self.dcc_b) * self.Q_bar
                 + self.dcc_a * (e @ e.T)
                 + self.dcc_b * Q)
        self.last_Q = Q
        self.last_vol = self.cond_vol[-1]
        self.last_r = returns[-1]

        self.fitted = True
        print(f"  SimpleDCC fitted: T={T}, N={N}")

    def generate(self, T_oos: int, n_paths: int = 100) -> np.ndarray:
        """Generate synthetic OOS returns.

        Returns: (n_paths, T_oos, N)
        """
        paths = np.zeros((n_paths, T_oos, self.N))

        for p in range(n_paths):
            vol = self.last_vol.copy()
            Q = self.last_Q.copy()
            prev_r = self.last_r.copy()

            for t in range(T_oos):
                # Update GARCH volatilities
                for i in range(self.N):
                    omega, alpha_g, beta_g = self.garch_params[i]
                    h = omega + alpha_g * prev_r[i]**2 + beta_g * vol[i]**2
                    vol[i] = np.sqrt(max(h, 1e-10))

                # Update DCC correlation
                e = prev_r / np.maximum(vol, 1e-8)
                e_outer = np.outer(e, e)
                Q = ((1 - self.dcc_a - self.dcc_b) * self.Q_bar
                     + self.dcc_a * e_outer
                     + self.dcc_b * Q)

                # Q → R (normalize to correlation)
                d = np.sqrt(np.maximum(np.diag(Q), 1e-10))
                R = Q / np.outer(d, d)
                R = np.clip(R, -1, 1)
                np.fill_diagonal(R, 1.0)

                # Generate returns: r = vol * L * z
                try:
                    L = np.linalg.cholesky(R + 1e-6 * np.eye(self.N))
                except np.linalg.LinAlgError:
                    L = np.eye(self.N)

                z = np.random.randn(self.N)
                r = vol * (L @ z)
                paths[p, t] = r
                prev_r = r

        return paths


# ── 2b. DCC-GARCH via rmgarch (full MLE) ─────────────────────────────

class RDCC:
    """Production DCC(1,1)-GARCH(1,1) via R rmgarch (full MLE).

    Shells out to src/dcc_mle.R. Addresses reviewer R3.4: the in-Python
    SimpleDCC uses moment-based α=0.05/β=0.90 shortcuts that underestimate
    the baseline; rmgarch fits each univariate GARCH by ML and then the
    DCC step by ML, matching the production DCC-GARCH benchmark the
    reviewer asked for.

    The R script trims leading-zero history (late-listed assets such as
    pdbc/meta/bndx) before fitting because those zero runs push the
    univariate GARCH estimator to non-stationary (α≈1) regions. The
    fitted DCC object, however, is used to simulate n_paths × T_oos
    returns conditioned on the full training window.
    """

    R_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "dcc_mle.R")

    def __init__(self, returns_csv: str, train_end_idx: int,
                 rscript_bin: str = "Rscript"):
        """
        returns_csv: path to the asset-returns CSV (Date + N asset columns)
        train_end_idx: 0-indexed exclusive end of fit window (matches
            ``returns.values[:train_end_idx]`` used by SimpleDCC, i.e. the
            boundary between fit data and the OOS window).
        """
        self.returns_csv = returns_csv
        self.train_end_idx = int(train_end_idx)
        self.rscript_bin = rscript_bin

    def generate(self, T_oos: int, n_paths: int = 100,
                 seed: int = 42, cache_bin: str | None = None) -> np.ndarray:
        import subprocess, tempfile
        n_assets = self._infer_n_assets()
        close_file = False
        if cache_bin is None:
            f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
            f.close(); cache_bin = f.name; close_file = True

        # R is 1-indexed and the script's train_end arg is an inclusive row
        # count, so pass train_end_idx directly (Python's [:train_end_idx]
        # equals R's [seq_len(train_end_idx), ]).
        cmd = [self.rscript_bin, self.R_SCRIPT, self.returns_csv,
               str(self.train_end_idx), str(T_oos), str(n_paths),
               cache_bin, str(seed)]
        print(f"  [RDCC] running: {' '.join(cmd)}")
        r = subprocess.run(cmd, check=True, capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.strip():
                print(f"    {line}")

        arr = np.fromfile(cache_bin, dtype=np.float64).reshape(
            (n_paths, T_oos, n_assets))
        if close_file:
            os.unlink(cache_bin)
        return arr

    def _infer_n_assets(self) -> int:
        with open(self.returns_csv) as f:
            header = f.readline().rstrip("\n").split(",")
        return len(header) - 1  # drop Date column


# ── Run All Baselines ─────────────────────────────────────────────────

def run_baselines(returns_train: np.ndarray, returns_test: np.ndarray,
                  factors_full: np.ndarray, test_start_idx: int,
                  class_indices: Dict[str, list],
                  n_paths: int = 100) -> Dict[str, np.ndarray]:
    """Run all baselines and return generated paths.

    returns_train: (T_train, N)
    returns_test: (T_test, N)
    factors_full: (T_full, K)
    test_start_idx: index of first test day in full array

    Returns dict: {name: (n_paths, T_test, N)}
    """
    results = {}

    # 1. Factor Bootstrap
    print("\n[Baseline 1] Factor Bootstrap")
    fb = FactorBootstrap(window=126)
    fb.fit(np.concatenate([returns_train, returns_test], axis=0), factors_full)
    fb_paths = fb.generate(factors_full, test_start_idx, n_paths)
    results["factor_bootstrap"] = fb_paths
    print(f"  Generated: {fb_paths.shape}")

    # 2. DCC-GARCH
    print("\n[Baseline 2] DCC-GARCH")
    dcc = SimpleDCC()
    dcc.fit(returns_train)
    dcc_paths = dcc.generate(len(returns_test), n_paths)
    results["dcc_garch"] = dcc_paths
    print(f"  Generated: {dcc_paths.shape}")

    return results


# ── 3. Student-t Copula ──────────────────────────────────────────────

class StudentTCopula:
    """Student-t copula generator for multi-asset returns.

    Standard financial baseline: fits marginal distributions and
    dependence structure separately.

    1. Fit marginal t-distributions per asset
    2. Fit correlation matrix and degrees of freedom on copula
    3. Generate: sample from t-copula, apply marginal quantile functions
    """

    def __init__(self):
        self.fitted = False

    def fit(self, returns: np.ndarray):
        """Fit on training returns (T, N)."""
        T, N = returns.shape
        self.N = N

        # Fit marginal t-distributions
        self.marginal_params = []
        self.uniform_data = np.zeros_like(returns)
        for i in range(N):
            df, loc, scale = stats.t.fit(returns[:, i])
            self.marginal_params.append((df, loc, scale))
            self.uniform_data[:, i] = stats.t.cdf(returns[:, i], df, loc, scale)

        # Fit t-copula: estimate correlation from pseudo-observations
        # Transform to standard normal for correlation estimation
        normal_data = stats.norm.ppf(np.clip(self.uniform_data, 1e-6, 1-1e-6))
        self.corr = np.corrcoef(normal_data.T)
        # Ensure positive definite
        eigvals, eigvecs = np.linalg.eigh(self.corr)
        eigvals = np.maximum(eigvals, 1e-6)
        self.corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        np.fill_diagonal(self.corr, 1.0)

        # Estimate copula degrees of freedom via maximum pseudo-likelihood
        # on the original data (before Gaussian transform)
        kurtoses = [stats.kurtosis(returns[:, i]) for i in range(N)]
        median_kurt = np.median(kurtoses)
        # Excess kurtosis of t-distribution = 6/(df-4) for df>4
        if median_kurt > 0.1:
            self.copula_df = max(5, min(50, 6.0 / median_kurt + 4))
        else:
            self.copula_df = 50  # near-Gaussian

        self.chol = np.linalg.cholesky(self.corr)
        self.fitted = True
        print(f"  StudentTCopula fitted: N={N}, copula_df={self.copula_df:.1f}")

    def generate(self, T_oos: int, n_paths: int = 100) -> np.ndarray:
        """Generate synthetic returns from t-copula."""
        paths = np.zeros((n_paths, T_oos, self.N))

        for p in range(n_paths):
            # Sample from multivariate t via mixing
            z = np.random.randn(T_oos, self.N) @ self.chol.T
            chi2 = np.random.chisquare(self.copula_df, size=(T_oos, 1))
            t_samples = z / np.sqrt(chi2 / self.copula_df)

            # Transform to uniform via t-CDF
            u = stats.t.cdf(t_samples, self.copula_df)

            # Apply marginal quantile functions
            for i in range(self.N):
                df, loc, scale = self.marginal_params[i]
                paths[p, :, i] = stats.t.ppf(np.clip(u[:, i], 1e-6, 1-1e-6), df, loc, scale)

        return paths


# ── 4. Block Bootstrap ───────────────────────────────────────────────

class BlockBootstrap:
    """Stationary block bootstrap on returns.

    Resamples contiguous blocks of returns, preserving temporal
    dependence within blocks. Block length drawn from geometric
    distribution with expected length `block_len`.
    """

    def __init__(self, block_len: int = 20):
        self.block_len = block_len

    def fit(self, returns: np.ndarray):
        """Store training returns for resampling."""
        self.data = returns.copy()
        self.T, self.N = returns.shape
        print(f"  BlockBootstrap fitted: T={self.T}, N={self.N}, block_len={self.block_len}")

    def generate(self, T_oos: int, n_paths: int = 100) -> np.ndarray:
        """Generate paths via stationary block bootstrap."""
        paths = np.zeros((n_paths, T_oos, self.N))
        p_switch = 1.0 / self.block_len  # geometric block length

        for p in range(n_paths):
            t = 0
            idx = np.random.randint(0, self.T)
            while t < T_oos:
                paths[p, t] = self.data[idx % self.T]
                t += 1
                # With probability p_switch, jump to random start
                if np.random.rand() < p_switch:
                    idx = np.random.randint(0, self.T)
                else:
                    idx += 1

        return paths
