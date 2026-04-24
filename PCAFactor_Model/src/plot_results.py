"""
Generate publication-quality comparison plots:
1. Correlation matrix heatmaps (Real vs Generated)
2. Left tail exceedance matrix (Real vs Generated)
3. Standard deviation: Real vs Generated scatter
"""
import sys, os, json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_returns, load_macro, ASSET_CLASSES, train_val_test_split
from factors import build_hierarchical_factors, compute_rolling_coefficients
from macro_processor import MacroProcessor
from model import Generator
from pathlib import Path


def load_and_generate(save_dir, returns_full, gf_arr, cov_arr, alpha, beta, sigma,
                      test_start_idx, asset_class="all", n_paths=100, device="cpu"):
    """Load model and generate OOS paths."""
    if asset_class == "all":
        selected = ASSET_CLASSES
    else:
        selected = {asset_class: ASSET_CLASSES[asset_class]}
    assets = [a for cls in selected.values() for a in cls]
    n_assets = len(assets)

    config = json.load(open(Path(save_dir) / "config.json"))
    gen = Generator(n_assets=n_assets, n_factors=config.get("n_factors", 5),
                    d_z=config.get("d_z", 10), d_cov=config.get("d_cov", 15),
                    hidden_dim=config.get("hidden_dim", 256))
    ckpt = torch.load(Path(save_dir) / "best_model.pt", map_location=device, weights_only=False)
    gen.load_state_dict(ckpt["G"])
    gen.to(device).eval()

    full_cols = returns_full.columns.tolist()
    idx = [full_cols.index(a) for a in assets]
    rfs = gen.rfs
    T_L = min(252, len(returns_full) - test_start_idx)
    gen_end = test_start_idx + T_L
    hist_start = max(0, test_start_idx - rfs)

    cov_s = torch.tensor(cov_arr[hist_start:gen_end+1], dtype=torch.float32)
    a_s = torch.tensor(alpha[test_start_idx:gen_end, idx], dtype=torch.float32)
    b_s = torch.tensor(beta[test_start_idx:gen_end][:, idx], dtype=torch.float32)
    s_s = torch.tensor(sigma[test_start_idx:gen_end, idx], dtype=torch.float32)
    f_s = torch.tensor(gf_arr[test_start_idx:gen_end], dtype=torch.float32)

    all_paths = []
    with torch.no_grad():
        for _ in range(n_paths):
            T_full = rfs + T_L + 1
            z = torch.randn(1, gen.d_z, T_full, device=device)
            cov_t = cov_s.unsqueeze(0).permute(0, 2, 1).to(device)
            if cov_t.shape[2] < T_full:
                cov_t = F.pad(cov_t, (0, T_full - cov_t.shape[2]), mode="replicate")
            r = gen(z, cov_t[:,:,:T_full],
                    a_s.unsqueeze(0).permute(0,2,1).to(device),
                    b_s.unsqueeze(0).permute(0,2,3,1).to(device),
                    s_s.unsqueeze(0).permute(0,2,1).to(device),
                    f_s.unsqueeze(0).permute(0,2,1).to(device))
            all_paths.append(r.squeeze(0).permute(1,0).cpu().numpy())

    return np.array(all_paths), assets  # (n_paths, T_L, N)


def plot_correlation_matrices(real, fake, assets, class_indices, save_path):
    """Side-by-side correlation matrix heatmaps."""
    corr_real = np.corrcoef(real.T)
    corr_fake = np.corrcoef(fake.T)
    N = len(assets)

    # Asset class boundaries for grid lines
    boundaries = []
    offset = 0
    class_labels = []
    for cls in ["bonds", "commodities", "stocks"]:
        if cls in class_indices:
            n = len(class_indices[cls])
            boundaries.append(offset + n)
            class_labels.append((offset + n/2, cls.capitalize()))
            offset += n

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, corr, title in zip(axes[:2], [corr_real, corr_fake], ["Real", "Generated (SF-MarketGAN)"]):
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-0.5, vmax=1.0, aspect='equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        for b in boundaries[:-1]:
            ax.axhline(b - 0.5, color='black', linewidth=1.5)
            ax.axvline(b - 0.5, color='black', linewidth=1.5)
        for pos, label in class_labels:
            ax.text(-3, pos, label, fontsize=9, ha='right', va='center', rotation=90)
        ax.set_xticks([])
        ax.set_yticks([])

    # Difference
    diff = corr_real - corr_fake
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='equal')
    axes[2].set_title("Difference (Real − Generated)", fontsize=14, fontweight='bold')
    for b in boundaries[:-1]:
        axes[2].axhline(b - 0.5, color='black', linewidth=1.5)
        axes[2].axvline(b - 0.5, color='black', linewidth=1.5)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.colorbar(im, ax=axes[1], shrink=0.8, label='Correlation')
    plt.colorbar(im3, ax=axes[2], shrink=0.8, label='Difference')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tail_exceedance(real, fake, assets, class_indices, save_path, q=0.05):
    """Left tail exceedance probability matrices: P(r_i < q_i | r_j < q_j)."""
    N = len(assets)
    quantiles = np.quantile(real, q, axis=0)

    def _tail_matrix(data, thresholds):
        extreme = data < thresholds
        mat = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if extreme[:, j].sum() > 0:
                    mat[i, j] = (extreme[:, i] & extreme[:, j]).sum() / extreme[:, j].sum()
        return mat

    tail_real = _tail_matrix(real, quantiles)
    tail_fake = _tail_matrix(fake, quantiles)

    boundaries = []
    offset = 0
    class_labels = []
    for cls in ["bonds", "commodities", "stocks"]:
        if cls in class_indices:
            n = len(class_indices[cls])
            boundaries.append(offset + n)
            class_labels.append((offset + n/2, cls.capitalize()))
            offset += n

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, mat, title in zip(axes[:2], [tail_real, tail_fake],
                               ["Real", "Generated (SF-MarketGAN)"]):
        im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=0.5, aspect='equal')
        ax.set_title(f"{title}\n(5% Left Tail Exceedance)", fontsize=13, fontweight='bold')
        for b in boundaries[:-1]:
            ax.axhline(b - 0.5, color='black', linewidth=1.5)
            ax.axvline(b - 0.5, color='black', linewidth=1.5)
        for pos, label in class_labels:
            ax.text(-3, pos, label, fontsize=9, ha='right', va='center', rotation=90)
        ax.set_xticks([])
        ax.set_yticks([])

    diff = tail_real - tail_fake
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-0.2, vmax=0.2, aspect='equal')
    axes[2].set_title("Difference (Real − Generated)", fontsize=13, fontweight='bold')
    for b in boundaries[:-1]:
        axes[2].axhline(b - 0.5, color='black', linewidth=1.5)
        axes[2].axvline(b - 0.5, color='black', linewidth=1.5)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.colorbar(im, ax=axes[1], shrink=0.8, label='P(r_i < q | r_j < q)')
    plt.colorbar(im3, ax=axes[2], shrink=0.8, label='Difference')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_std_comparison(real, fake, assets, class_indices, save_path):
    """Scatter plot of per-asset standard deviation: real vs generated."""
    std_real = real.std(axis=0)
    std_fake = fake.std(axis=0)

    colors = {'bonds': '#1f77b4', 'commodities': '#ff7f0e', 'stocks': '#2ca02c'}

    fig, ax = plt.subplots(figsize=(8, 8))

    for cls in ["bonds", "commodities", "stocks"]:
        if cls not in class_indices:
            continue
        idx = class_indices[cls]
        ax.scatter(std_real[idx], std_fake[idx],
                   c=colors[cls], label=cls.capitalize(),
                   s=60, alpha=0.7, edgecolors='white', linewidth=0.5)

    # 45-degree line
    lims = [0, max(std_real.max(), std_fake.max()) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Perfect match')

    ax.set_xlabel('Real Std Dev (daily log returns)', fontsize=12)
    ax.set_ylabel('Generated Std Dev', fontsize=12)
    ax.set_title('Per-Asset Volatility: Real vs Generated', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Annotate RMSE
    rmse = np.sqrt(((std_real - std_fake) ** 2).mean())
    corr = np.corrcoef(std_real, std_fake)[0, 1]
    ax.text(0.05, 0.92, f'RMSE = {rmse:.5f}\nCorr = {corr:.4f}',
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42)
    torch.manual_seed(42)

    print("Generating plots...")

    # Load data
    returns = load_returns()
    macro = load_macro()
    proc = MacroProcessor(n_pca_components=8, n_regimes=3)
    covariates = proc.fit_transform(macro, returns.index)
    gf, cf, af = build_hierarchical_factors(returns, macro, ASSET_CLASSES)
    n_factors = gf.shape[1]
    alpha, beta, sigma = compute_rolling_coefficients(
        returns, af, ASSET_CLASSES, cf, gf, window=126)
    beta = beta[:, :, :n_factors]
    cov_arr = covariates.reindex(returns.index).fillna(0).values
    gf_arr = gf.reindex(returns.index).fillna(0).values

    _, _, test_ret = train_val_test_split(returns)
    test_start_idx = returns.index.get_loc(test_ret.index[0])
    real_test = test_ret.values

    # Class indices
    asset_list = returns.columns.tolist()
    class_indices = {c: [asset_list.index(a) for a in v] for c, v in ASSET_CLASSES.items()}

    # Generate from best model (multi_film_sf)
    save_dir = "results/multi_film_sf"
    if not Path(f"{save_dir}/best_model.pt").exists():
        print(f"Model not found at {save_dir}, skipping")
        return

    print("Loading model and generating OOS paths...")
    paths, assets = load_and_generate(
        save_dir, returns, gf_arr, cov_arr, alpha, beta, sigma,
        test_start_idx, asset_class="all", n_paths=100, device=device)
    fake = paths.reshape(-1, len(assets))
    print(f"  Real: {real_test.shape}, Fake: {fake.shape}")

    os.makedirs("figures", exist_ok=True)

    # Plot 1: Correlation matrices
    print("\n1. Correlation matrices...")
    plot_correlation_matrices(real_test, fake, assets, class_indices, "figures/correlation_matrix.png")

    # Plot 2: Tail exceedance
    print("2. Left tail exceedance...")
    plot_tail_exceedance(real_test, fake, assets, class_indices, "figures/tail_exceedance.png")

    # Plot 3: Std comparison
    print("3. Standard deviation comparison...")
    plot_std_comparison(real_test, fake, assets, class_indices, "figures/std_comparison.png")

    print("\nAll plots saved to figures/")


if __name__ == "__main__":
    main()
