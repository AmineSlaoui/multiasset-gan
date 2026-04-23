"""Evaluate MarketGAN-60 checkpoints on 2025 OOS, side-by-side with the
SF-MarketGAN-R 10-seed result."""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import ASSET_CLASSES, load_macro, load_returns, train_val_test_split
from evaluate import full_evaluation
from factors_v2 import compute_hierarchical_pca_factors, compute_rolling_coefficients_v2
from macro_processor import MacroProcessor
from train_marketgan_60asset import MarketGANGenerator


def gen_paths(gen, cov_arr, f_arr, alpha_hat, beta_hat, sigma_hat,
              test_start_idx, n_paths, t_l, device):
    """Generate paths from a MarketGAN-60 checkpoint."""
    N = alpha_hat.shape[1]; rfs = gen.rfs
    T_full = rfs + t_l + 1
    t_start = max(0, test_start_idx - rfs - 1)
    t_end = min(t_start + T_full, len(cov_arr))
    cov_seq = torch.tensor(cov_arr[t_start:t_end], dtype=torch.float32).T.unsqueeze(0).to(device)
    if t_end - t_start < T_full:
        cov_seq = F.pad(cov_seq, (0, T_full - (t_end - t_start)), mode="replicate")
    ts, te = test_start_idx, min(test_start_idx + t_l, len(f_arr))
    actual_tl = te - ts
    f = torch.tensor(f_arr[ts:te], dtype=torch.float32).T.unsqueeze(0).to(device)
    a = torch.tensor(alpha_hat[ts:te].T, dtype=torch.float32).unsqueeze(0).to(device)
    s = torch.tensor(sigma_hat[ts:te].T, dtype=torch.float32).unsqueeze(0).to(device)
    b = torch.tensor(beta_hat[ts:te], dtype=torch.float32).permute(1, 2, 0).unsqueeze(0).to(device)
    if actual_tl < t_l:
        pad = t_l - actual_tl
        f = F.pad(f, (0, pad), mode="replicate")
        a = F.pad(a, (0, pad), mode="replicate")
        s = F.pad(s, (0, pad), mode="replicate")
        b = F.pad(b, (0, pad, 0, 0), mode="replicate")
    out = np.zeros((n_paths, actual_tl, N))
    bs = 20
    for i in range(0, n_paths, bs):
        n = min(bs, n_paths - i)
        z = torch.randn(n, 10, T_full, device=device)
        with torch.no_grad():
            r = gen(z, cov_seq.expand(n, -1, -1), a.expand(n, -1, -1),
                    b.expand(n, -1, -1, -1), s.expand(n, -1, -1), f.expand(n, -1, -1))
        for j in range(n):
            out[i + j] = r[j, :, :actual_tl].cpu().numpy().T
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--n_paths", type=int, default=100)
    p.add_argument("--out", default="results/marketgan60_oos.json")
    args = p.parse_args()

    device = torch.device("cpu"); np.random.seed(42); torch.manual_seed(42)

    returns = load_returns(); macro = load_macro()
    train_ret, val_ret, test_ret = train_val_test_split(returns)
    test_start_idx = returns.index.get_loc(test_ret.index[0])
    train_end = str(train_ret.index[-1].date())
    proc = MacroProcessor(n_pca_components=8, n_regimes=3)
    covariates = proc.fit_transform(macro, returns.index, train_end=train_end)
    cov_arr = covariates.reindex(returns.index).fillna(0).values
    factors_df, _ = compute_hierarchical_pca_factors(
        returns, ASSET_CLASSES, n_global=5, n_class=2, train_end=train_end)
    f_arr = factors_df.values; n_factors = factors_df.shape[1]
    f_arr_lag = np.vstack([np.zeros((1, n_factors)), f_arr[:-1]])
    alpha_hat, beta_hat, sigma_hat = compute_rolling_coefficients_v2(
        returns, factors_df, window=126)
    real_test = test_ret.values; N = returns.shape[1]
    asset_list = returns.columns.tolist()
    class_indices = {c: [asset_list.index(a) for a in v] for c, v in ASSET_CLASSES.items()}
    d_cov = covariates.shape[1]

    results = {}
    for ckpt in args.ckpts:
        if not os.path.exists(ckpt): continue
        tag = os.path.basename(os.path.dirname(ckpt))
        cfg = json.load(open(os.path.join(os.path.dirname(ckpt), "config.json")))
        print(f"  {tag} (lr={cfg['lr']}, n_critic={cfg['n_critic']}, val_frob={cfg['best_frob']:.3f}) ...")
        gen = MarketGANGenerator(N, n_factors, d_z=10, d_cov=d_cov,
                                  hidden=cfg.get("hidden_g", 80),
                                  num_blocks=cfg.get("num_blocks", 6)).to(device)
        ckpt_d = torch.load(ckpt, map_location=device, weights_only=False)
        gen.load_state_dict(ckpt_d["G"], strict=False); gen.eval()
        f_in = f_arr_lag if cfg.get("lag_factors", True) else f_arr
        paths = gen_paths(gen, cov_arr, f_in, alpha_hat, beta_hat, sigma_hat,
                           test_start_idx, args.n_paths, real_test.shape[0], device)
        results[tag] = {"val_frob": cfg["best_frob"],
                        "lr": cfg["lr"], "n_critic": cfg["n_critic"]}
        results[tag].update({k: float(v) for k, v in
                              full_evaluation(real_test, paths,
                                              class_indices=class_indices,
                                              n_paths=args.n_paths).items()
                              if isinstance(v, (float, int, np.floating, np.integer))})

    print(f"\n{'='*72}\n  MarketGAN 60-asset HP sweep — 2025 OOS\n{'='*72}")
    print(f"  {'config':<16} {'lr':>8} {'nc':>4} {'val_frob':>10} {'OOS CorrFrob':>14} {'ACF':>8}")
    for n in sorted(results.keys()):
        r = results[n]
        print(f"  {n:<16} {r['lr']:>8.0e} {r['n_critic']:>4} {r['val_frob']:>10.3f} "
              f"{r.get('full_corr_frob', float('nan')):>14.4f} {r.get('acf_score', float('nan')):>8.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {args.out}")


if __name__ == "__main__":
    main()
