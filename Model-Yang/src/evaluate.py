"""
Evaluation metrics for SF-MarketGAN.

Implements all metrics from MarketGAN paper + cross-class metrics:
- Marginal: RMSE, MAE, moments (mean, var, skew, kurtosis)
- Inter-temporal: ACF, VC, Lev scores
- Cross-sectional: XCorr, XCorrE (within-class and cross-class)
- Distributional: FID, SWD, Mahalanobis Distance, DTW
"""
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")


# ── Marginal Metrics ─────────────────────────────────────────────────

def marginal_metrics(real: np.ndarray, fake: np.ndarray) -> Dict[str, float]:
    """RMSE and MAE averaged across assets.

    Compares per-asset distributional statistics (mean, std) rather than
    point-wise alignment, since generated paths are stochastic.

    real: (T_real, N)
    fake: (T_fake, N) — T_fake can differ from T_real (multiple paths concatenated)
    """
    # Compare per-asset mean and std
    real_mean = real.mean(axis=0)
    fake_mean = fake.mean(axis=0)
    real_std = real.std(axis=0)
    fake_std = fake.std(axis=0)

    rmse_mean = np.sqrt(((real_mean - fake_mean) ** 2).mean())
    rmse_std = np.sqrt(((real_std - fake_std) ** 2).mean())
    mae_mean = np.abs(real_mean - fake_mean).mean()

    return {"rmse_mean": rmse_mean, "rmse_std": rmse_std, "mae_mean": mae_mean}


def moment_comparison(real: np.ndarray, fake: np.ndarray) -> Dict[str, np.ndarray]:
    """Compare first four moments per asset.

    Returns dict with real and fake moments arrays of shape (N, 4).
    """
    def _moments(x):
        return np.array([x.mean(0), x.var(0), stats.skew(x, axis=0), stats.kurtosis(x, axis=0)])

    if fake.ndim == 3:
        fake = fake.reshape(-1, fake.shape[-1])
    if real.ndim == 3:
        real = real.reshape(-1, real.shape[-1])

    m_real = _moments(real)  # (4, N)
    m_fake = _moments(fake)
    return {
        "real_moments": m_real,
        "fake_moments": m_fake,
        "moment_rmse": np.sqrt(((m_real - m_fake) ** 2).mean(axis=1)),  # (4,)
    }


# ── Inter-temporal Metrics ───────────────────────────────────────────

def _acf(x: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Autocorrelation function for 1D array."""
    n = len(x)
    x = x - x.mean()
    var = (x ** 2).mean()
    if var < 1e-12:
        return np.zeros(max_lag)
    acf = np.correlate(x, x, mode="full")[n-1:]
    acf = acf[:max_lag + 1] / (var * n)
    return acf[1:]  # Exclude lag 0


def intertemporal_scores(
    real: np.ndarray,
    fake: np.ndarray,
    max_lag: int = 100,
    n_assets_sample: int = 15,
) -> Dict[str, float]:
    """Compute ACF, VC, Lev scores as L2 distance between real and fake.

    real, fake: (T, N)
    """
    N = real.shape[1]
    indices = np.linspace(0, N - 1, min(n_assets_sample, N)).astype(int)

    acf_diff, vc_diff, lev_diff = [], [], []
    for i in indices:
        r_real, r_fake = real[:, i], fake[:, i]

        # ACF of returns
        acf_r = _acf(r_real, max_lag)
        acf_f = _acf(r_fake, max_lag)
        acf_diff.append(np.linalg.norm(acf_r - acf_f))

        # VC: ACF of squared returns
        vc_r = _acf(r_real ** 2, max_lag)
        vc_f = _acf(r_fake ** 2, max_lag)
        vc_diff.append(np.linalg.norm(vc_r - vc_f))

        # Lev: cross-correlation of r²_{t+k} with r_t
        lev_r = _cross_acf(r_real, r_real ** 2, max_lag)
        lev_f = _cross_acf(r_fake, r_fake ** 2, max_lag)
        lev_diff.append(np.linalg.norm(lev_r - lev_f))

    return {
        "acf_score": np.mean(acf_diff),
        "vc_score": np.mean(vc_diff),
        "lev_score": np.mean(lev_diff),
    }


def _cross_acf(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    """Cross-autocorrelation: Corr(y_{t+k}, x_t) for k=1..max_lag."""
    n = len(x)
    x_c = x - x.mean()
    y_c = y - y.mean()
    std_x = x_c.std() + 1e-12
    std_y = y_c.std() + 1e-12

    cc = []
    for k in range(1, max_lag + 1):
        if n - k < 2:
            cc.append(0.0)
            continue
        cov = np.mean(y_c[k:] * x_c[:-k])
        cc.append(cov / (std_x * std_y))
    return np.array(cc)


# ── Cross-sectional Metrics ──────────────────────────────────────────

def xcorr_score(real: np.ndarray, fake: np.ndarray) -> float:
    """XCorr: Frobenius norm of correlation matrix difference."""
    corr_real = np.corrcoef(real.T)
    corr_fake = np.corrcoef(fake.T)
    return np.linalg.norm(corr_real - corr_fake, "fro")


def xcorre_score(
    real: np.ndarray,
    fake: np.ndarray,
    quantile: float = 0.05,
) -> float:
    """XCorrE: Extreme cross-correlation (tail dependence).

    P(r_i < -q_i | r_j < -q_j) comparison.
    """
    N = real.shape[1]
    q_real = np.quantile(real, quantile, axis=0)
    q_fake = np.quantile(fake, quantile, axis=0)

    def _extreme_corr(data, thresholds):
        extreme = data < thresholds
        n_assets = data.shape[1]
        mat = np.zeros((n_assets, n_assets))
        for i in range(n_assets):
            for j in range(n_assets):
                if extreme[:, j].sum() > 0:
                    mat[i, j] = (extreme[:, i] & extreme[:, j]).sum() / extreme[:, j].sum()
        return mat

    ecorr_real = _extreme_corr(real, q_real)
    ecorr_fake = _extreme_corr(fake, q_fake)
    return np.linalg.norm(ecorr_real - ecorr_fake, "fro")


def cross_class_xcorr(
    real: np.ndarray,
    fake: np.ndarray,
    class_indices: Dict[str, list],
) -> Dict[str, float]:
    """Cross-class correlation scores (NEW metric).

    Measures between-class correlation fidelity separately.
    """
    corr_real = np.corrcoef(real.T)
    corr_fake = np.corrcoef(fake.T)

    scores = {}
    class_names = list(class_indices.keys())
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            idx_i = class_indices[class_names[i]]
            idx_j = class_indices[class_names[j]]
            block_real = corr_real[np.ix_(idx_i, idx_j)]
            block_fake = corr_fake[np.ix_(idx_i, idx_j)]
            key = f"xcorr_{class_names[i]}_{class_names[j]}"
            scores[key] = np.linalg.norm(block_real - block_fake, "fro")
    return scores


# ── Distributional Metrics ───────────────────────────────────────────

def fid_score(real: np.ndarray, fake: np.ndarray) -> float:
    """Fréchet Inception Distance (using raw features, no inception net)."""
    mu_r, sigma_r = real.mean(0), np.cov(real.T)
    mu_f, sigma_f = fake.mean(0), np.cov(fake.T)

    diff = mu_r - mu_f
    # Stable sqrt of product of covariance matrices
    try:
        covmean = _sqrtm_stable(sigma_r @ sigma_f)
    except Exception:
        covmean = np.zeros_like(sigma_r)

    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return max(float(fid), 0.0)


def _sqrtm_stable(M: np.ndarray) -> np.ndarray:
    """Stable matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def swd_score(real: np.ndarray, fake: np.ndarray, n_projections: int = 1000) -> float:
    """Sliced Wasserstein Distance."""
    N = real.shape[1]
    projections = np.random.randn(n_projections, N)
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    dists = []
    for proj in projections:
        r_proj = real @ proj
        f_proj = fake @ proj
        r_sorted = np.sort(r_proj)
        f_sorted = np.sort(f_proj)
        # Align lengths
        n = min(len(r_sorted), len(f_sorted))
        r_idx = np.linspace(0, len(r_sorted) - 1, n).astype(int)
        f_idx = np.linspace(0, len(f_sorted) - 1, n).astype(int)
        dists.append(np.mean(np.abs(r_sorted[r_idx] - f_sorted[f_idx])))

    return float(np.mean(dists))


def mahalanobis_distance(real: np.ndarray, fake: np.ndarray) -> float:
    """Average Mahalanobis distance of fake samples w.r.t. real distribution."""
    mu = real.mean(0)
    cov = np.cov(real.T) + 1e-6 * np.eye(real.shape[1])
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    diff = fake - mu
    md = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    return float(md.mean())


# ── Full Evaluation ──────────────────────────────────────────────────

def full_evaluation(
    real: np.ndarray,
    fake: np.ndarray,
    class_indices: Optional[Dict[str, list]] = None,
    n_paths: int = 1,
) -> Dict[str, float]:
    """Run all evaluation metrics.

    real: (T, N) realized returns
    fake: (T, N) or (n_paths, T, N) synthetic returns
    """
    if fake.ndim == 3:
        fake_flat = fake.reshape(-1, fake.shape[-1])
    else:
        fake_flat = fake

    # Subsample fake to manageable size for pairwise metrics
    if len(fake_flat) > 10000:
        idx = np.random.choice(len(fake_flat), 10000, replace=False)
        fake_sub = fake_flat[idx]
    else:
        fake_sub = fake_flat

    results = {}

    # Marginal (distribution-level, not point-wise)
    m = marginal_metrics(real, fake_flat)
    results.update(m)

    # Inter-temporal (use a single path-length chunk)
    T_real = real.shape[0]
    fake_single = fake_flat[:T_real]  # Take first T_real steps
    it = intertemporal_scores(real, fake_single)
    results.update(it)

    # Cross-sectional
    results["xcorr"] = xcorr_score(real, fake_sub)
    results["xcorre"] = xcorre_score(real, fake_sub)

    # Cross-class
    if class_indices:
        cc = cross_class_xcorr(real, fake_sub, class_indices)
        results.update(cc)

    # Distributional
    results["frechet_gaussian"] = fid_score(real, fake_sub)  # Gaussian approx, not inception-based
    results["swd"] = swd_score(real, fake_sub)
    results["mahalanobis"] = mahalanobis_distance(real, fake_sub)

    # Block-decomposed correlation analysis (for single vs multi comparison)
    if class_indices:
        block = block_correlation_analysis(real, fake_sub, class_indices)
        results.update(block)

    return results


# ── Block Correlation Analysis ───────────────────────────────────────

def block_correlation_analysis(
    real: np.ndarray,
    fake: np.ndarray,
    class_indices: Dict[str, list],
) -> Dict[str, float]:
    """Decompose correlation fidelity into within-class and cross-class blocks.

    Enables fair comparison between single-class models (20 assets)
    and multi-asset models (60 assets):
    - Within-class frob: comparable across single/multi
    - Cross-class frob: only meaningful for multi-asset

    Returns normalized frob (divided by block size) for comparability.
    """
    corr_real = np.corrcoef(real.T)
    corr_fake = np.corrcoef(fake.T)

    results = {}
    class_names = list(class_indices.keys())

    # Within-class blocks (diagonal)
    for cls in class_names:
        idx = class_indices[cls]
        n = len(idx)
        if n == 0:
            continue
        block_real = corr_real[np.ix_(idx, idx)]
        block_fake = corr_fake[np.ix_(idx, idx)]
        frob = np.linalg.norm(block_real - block_fake, "fro")
        results[f"within_{cls}_frob"] = frob
        results[f"within_{cls}_frob_norm"] = frob / n  # Normalized by block size

    # Cross-class blocks (off-diagonal)
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            ci, cj = class_names[i], class_names[j]
            idx_i = class_indices[ci]
            idx_j = class_indices[cj]
            ni, nj = len(idx_i), len(idx_j)
            if ni == 0 or nj == 0:
                continue
            block_real = corr_real[np.ix_(idx_i, idx_j)]
            block_fake = corr_fake[np.ix_(idx_i, idx_j)]
            frob = np.linalg.norm(block_real - block_fake, "fro")
            results[f"cross_{ci}_{cj}_frob"] = frob
            results[f"cross_{ci}_{cj}_frob_norm"] = frob / np.sqrt(ni * nj)

    # Mean correlation error (per element)
    total_frob = np.linalg.norm(corr_real - corr_fake, "fro")
    N = real.shape[1]
    results["full_corr_frob"] = total_frob
    results["full_corr_frob_norm"] = total_frob / N

    return results
