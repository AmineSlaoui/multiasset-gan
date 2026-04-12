"""
OOS evaluation — supports both single-class and multi-asset models.

Generates contiguous 2025 OOS sequences, evaluates with block-decomposed metrics.
"""
import sys, os, json, argparse
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_returns, load_macro, ASSET_CLASSES, train_val_test_split
from factors import build_hierarchical_factors, compute_rolling_coefficients
from macro_processor import MacroProcessor
from model import Generator
from evaluate import full_evaluation, moment_comparison, block_correlation_analysis
from pathlib import Path


def load_model(save_dir, n_assets, n_factors, d_z, d_cov, hidden_dim, device):
    gen = Generator(n_assets=n_assets, n_factors=n_factors, d_z=d_z,
                    d_cov=d_cov, hidden_dim=hidden_dim)
    ckpt = torch.load(Path(save_dir) / "best_model.pt", map_location=device, weights_only=False)
    gen.load_state_dict(ckpt["G"])
    gen.to(device).eval()
    return gen, ckpt.get("epoch", "?"), ckpt.get("frob", "?")


@torch.no_grad()
def generate_oos_paths(gen, returns_full, factors_full, cov_full,
                       alpha_full, beta_full, sigma_full,
                       test_start_idx, n_paths=100, device="cpu"):
    """Generate contiguous OOS return sequences."""
    rfs = gen.rfs
    T_test = len(returns_full) - test_start_idx
    T_L = min(T_test, 252)
    gen_end = min(len(returns_full), test_start_idx + T_L)
    T_L_actual = gen_end - test_start_idx
    hist_start = max(0, test_start_idx - rfs)

    cov_slice = torch.tensor(cov_full[hist_start:gen_end + 1], dtype=torch.float32)
    alpha_slice = torch.tensor(alpha_full[test_start_idx:gen_end], dtype=torch.float32)
    beta_slice = torch.tensor(beta_full[test_start_idx:gen_end], dtype=torch.float32)
    sigma_slice = torch.tensor(sigma_full[test_start_idx:gen_end], dtype=torch.float32)
    factor_slice = torch.tensor(factors_full[test_start_idx:gen_end], dtype=torch.float32)

    all_paths = []
    batch_size = min(n_paths, 50)
    for start in range(0, n_paths, batch_size):
        B = min(batch_size, n_paths - start)
        T_full = rfs + T_L_actual + 1
        z = torch.randn(B, gen.d_z, T_full, device=device)
        cov_t = cov_slice.unsqueeze(0).expand(B, -1, -1).permute(0, 2, 1).to(device)
        if cov_t.shape[2] < T_full:
            cov_t = F.pad(cov_t, (0, T_full - cov_t.shape[2]), mode="replicate")
        cov_t = cov_t[:, :, :T_full]
        a = alpha_slice.unsqueeze(0).expand(B, -1, -1).permute(0, 2, 1).to(device)
        b = beta_slice.unsqueeze(0).expand(B, -1, -1, -1).permute(0, 2, 3, 1).to(device)
        s = sigma_slice.unsqueeze(0).expand(B, -1, -1).permute(0, 2, 1).to(device)
        f = factor_slice.unsqueeze(0).expand(B, -1, -1).permute(0, 2, 1).to(device)
        r_fake = gen(z, cov_t, a, b, s, f)
        all_paths.append(r_fake.permute(0, 2, 1).cpu().numpy())
    return np.concatenate(all_paths, axis=0)


def eval_model(name, save_dir, asset_class, returns_full, gf, cov_arr,
               alpha, beta, sigma, test_start_idx, test_ret, device):
    """Evaluate one model."""
    # Determine assets
    if asset_class == "all":
        selected_classes = ASSET_CLASSES
    else:
        selected_classes = {asset_class: ASSET_CLASSES[asset_class]}
    selected_assets = []
    for assets in selected_classes.values():
        selected_assets.extend(assets)
    n_assets = len(selected_assets)

    # Load config
    config_path = Path(save_dir) / "config.json"
    if config_path.exists():
        config = json.load(open(config_path))
        hidden_dim = config.get("hidden_dim", 256)
        d_z = config.get("d_z", 10)
        d_cov = config.get("d_cov", 15)
        n_factors = config.get("n_factors", 5)
    else:
        hidden_dim, d_z, d_cov, n_factors = 256, 10, 15, 5

    print(f"\n{'='*60}")
    print(f"  {name} ({asset_class}, {n_assets} assets)")
    print(f"{'='*60}")

    # Load model
    gen, best_epoch, best_frob = load_model(
        save_dir, n_assets, n_factors, d_z, d_cov, hidden_dim, device)
    print(f"  Loaded epoch {best_epoch}, val_frob={best_frob}")

    # Get asset indices in full returns
    full_cols = returns_full.columns.tolist()
    asset_indices = [full_cols.index(a) for a in selected_assets]

    # Subset arrays for this model's assets
    alpha_sub = alpha[:, asset_indices]
    beta_sub = beta[:, asset_indices]
    sigma_sub = sigma[:, asset_indices]

    # Generate OOS paths
    print(f"  Generating 100 OOS paths...")
    paths = generate_oos_paths(
        gen, returns_full.values[:, asset_indices], gf,
        cov_arr, alpha_sub, beta_sub, sigma_sub,
        test_start_idx=test_start_idx, n_paths=100, device=device)
    print(f"  Generated: {paths.shape}")

    # Real test returns for this asset subset
    real_test = test_ret[selected_assets].values
    fake_flat = paths.reshape(-1, n_assets)

    # Class indices relative to this model's assets
    asset_list = selected_assets
    class_indices = {}
    for cls_name, assets in selected_classes.items():
        class_indices[cls_name] = [asset_list.index(a) for a in assets if a in asset_list]

    # Full evaluation
    metrics = full_evaluation(real_test, fake_flat, class_indices)
    moments = moment_comparison(real_test, fake_flat)
    metrics["moment_rmse_kurt"] = float(moments["moment_rmse"][3])
    metrics["moment_rmse_skew"] = float(moments["moment_rmse"][2])

    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"OOS Evaluation | device={device}")

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
    print(f"  Test: {test_ret.index[0].date()} ~ {test_ret.index[-1].date()} ({len(test_ret)} days)")

    # Models to evaluate
    rd = args.results_dir
    models = {}

    # Auto-detect available models
    for name, ac in [
        ("single_stocks", "stocks"),
        ("single_bonds", "bonds"),
        ("single_commodities", "commodities"),
        ("multi_film", "all"),
        ("multi_film_sf", "all"),
    ]:
        model_dir = f"{rd}/{name}"
        if Path(f"{model_dir}/best_model.pt").exists():
            models[name] = (model_dir, ac)

    if not models:
        print("No models found!")
        return

    print(f"\nFound {len(models)} models: {list(models.keys())}")

    # Evaluate each
    all_results = {}
    for name, (save_dir, ac) in models.items():
        try:
            metrics = eval_model(
                name, save_dir, ac, returns, gf_arr, cov_arr,
                alpha, beta, sigma, test_start_idx, test_ret, device)
            all_results[name] = metrics
            print(f"  Key metrics:")
            for k in ["full_corr_frob_norm", "within_stocks_frob_norm",
                       "within_bonds_frob_norm", "within_commodities_frob_norm",
                       "xcorr", "xcorre", "vc_score", "lev_score", "swd"]:
                if k in metrics:
                    print(f"    {k:35s}: {metrics[k]:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results[name] = {"error": str(e)}

    # Save
    out = Path(f"{rd}/oos_eval.json")
    json.dump(all_results, open(out, "w"), indent=2)

    # Summary table
    print(f"\n{'='*70}")
    print("OOS COMPARISON: Single-Class vs Multi-Asset")
    print(f"{'='*70}")

    # Within-class normalized frob comparison
    print(f"\nWithin-class correlation fidelity (frob_norm ↓, lower=better):")
    print(f"  {'Model':25s} {'stocks':>10s} {'bonds':>10s} {'commod':>10s}")
    print(f"  {'-'*57}")
    for name in all_results:
        r = all_results[name]
        if "error" in r:
            continue
        s = r.get("within_stocks_frob_norm", float("nan"))
        b = r.get("within_bonds_frob_norm", float("nan"))
        c = r.get("within_commodities_frob_norm", float("nan"))
        print(f"  {name:25s} {s:>10.4f} {b:>10.4f} {c:>10.4f}")

    # Cross-class (multi-asset only)
    print(f"\nCross-class correlation (multi-asset only):")
    for name in all_results:
        r = all_results[name]
        if "error" in r:
            continue
        for k in ["cross_bonds_commodities_frob_norm", "cross_bonds_stocks_frob_norm",
                   "cross_commodities_stocks_frob_norm"]:
            if k in r:
                print(f"  {name:25s} {k:40s}: {r[k]:.4f}")

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
