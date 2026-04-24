"""Rolling pair-correlation plot for the legacy 20-stock MarketGAN baseline
(archive/old_results/marketgan_fixed). Only v-ma is supported since the old
universe is stocks-only (no commodities)."""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/Users/fengyang/Desktop/gan/archive/old_src_final")

from data_loader import ASSET_CLASSES, load_returns, train_val_test_split
from sklearn.preprocessing import StandardScaler
from train_marketgan import MGGenerator, compute_factors, rolling_ols
from plot_pair_corr_gen import rolling_corr_1d

WINDOW = 63
PAIRS = [("v", "ma", "Visa (v) vs Mastercard (ma)")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="archive/old_results/marketgan_fixed/best_model.pt")
    p.add_argument("--n_paths", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--t_l", type=int, default=251)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cpu")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    returns_full = load_returns()
    stocks = ASSET_CLASSES["stocks"]
    returns = returns_full[stocks]
    sr = returns.values
    tr, val, te = train_val_test_split(returns)
    T, N = sr.shape
    te_idx = len(tr)
    val_idx = te_idx + len(val)

    if args.split == "test":
        target_start = te_idx
    elif args.split == "val":
        target_start = te_idx
    else:
        target_start = te_idx - args.t_l
    target_end = target_start + args.t_l
    target_real = sr[target_start:target_end]
    print(f"[split={args.split}] start={returns.index[target_start].date()} "
          f"end={returns.index[target_end-1].date()}  n={target_end - target_start}")
    if args.out is None:
        args.out = f"figures/pair_corr_marketgan20_{args.split}.png"

    print("[factors] computing 5 hand-crafted factors on 20 stocks")
    fac = compute_factors(sr, 252)
    K = 5
    ah, bh, sh = rolling_ols(sr, fac, 252)
    sc = StandardScaler(); sc.fit(fac[:te_idx]); cov = sc.transform(fac)

    nb = 6
    gen = MGGenerator(N=N, K=K, d_z=10, d_cov=5, hid=80, nb=nb).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt["G"], strict=True); gen.eval()
    rfs = gen.rfs
    T_full = rfs + args.t_l + 1

    t_start = max(0, target_start - rfs - 1)
    t_end = min(t_start + T_full, T)
    if t_end - t_start < T_full:
        t_start = max(0, t_end - T_full)
    cov_seq = torch.tensor(cov[t_start:t_end], dtype=torch.float32).T.unsqueeze(0).to(device)
    fac_t = torch.tensor(fac[target_start:target_end], dtype=torch.float32).T.unsqueeze(0).to(device)
    a_t = torch.tensor(ah[target_start:target_end].T, dtype=torch.float32).unsqueeze(0).to(device)
    s_t = torch.tensor(sh[target_start:target_end].T, dtype=torch.float32).unsqueeze(0).to(device)
    b_t = torch.tensor(bh[target_start:target_end], dtype=torch.float32).permute(1, 2, 0).unsqueeze(0).to(device)

    print(f"[sample] n_paths={args.n_paths}")
    paths = np.zeros((args.n_paths, args.t_l, N))
    bs = 10
    for i in range(0, args.n_paths, bs):
        n = min(bs, args.n_paths - i)
        z = torch.randn(n, 10, T_full, device=device)
        with torch.no_grad():
            r = gen(z, cov_seq.expand(n, -1, -1), a_t.expand(n, -1, -1),
                    b_t.expand(n, -1, -1, -1), s_t.expand(n, -1, -1),
                    fac_t.expand(n, -1, -1))
        for j in range(n):
            paths[i + j] = r[j, :, :args.t_l].cpu().numpy().T

    gen_concat = paths.reshape(-1, N)
    real_tiled = np.tile(target_real, (args.n_paths, 1))
    print(f"[concat] total steps = {gen_concat.shape[0]}")

    fig, ax = plt.subplots(figsize=(14, 3.2))
    for (a, b, title) in PAIRS:
        ia, ib = stocks.index(a), stocks.index(b)
        r_roll = rolling_corr_1d(real_tiled[:, ia], real_tiled[:, ib], WINDOW)
        g_roll = rolling_corr_1d(gen_concat[:, ia], gen_concat[:, ib], WINDOW)
        ax.plot(r_roll, color="blue", lw=0.8, label="Real")
        ax.plot(g_roll, color="red", ls="--", lw=0.8, label="Generated")
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Timestep (test set)")
        ax.set_ylabel("Correlation")
        ax.set_title(f"MarketGAN-20 baseline — Rolling Correlation: {title} "
                     f"(window={WINDOW}d) [{args.split.upper()} SET]")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left")
        rm = np.nanmean((r_roll - g_roll) ** 2) ** 0.5
        print(f"  {title}: real_mean={np.nanmean(r_roll):.3f}  "
              f"gen_mean={np.nanmean(g_roll):.3f}  rmse={rm:.3f}")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
