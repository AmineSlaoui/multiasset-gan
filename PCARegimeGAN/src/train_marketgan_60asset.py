"""MarketGAN reproduction on the 60-asset multi-class universe.

Unlike our SF-MarketGAN-R, this model learns the full return-generating
process adversarially:
  - intercept $\alpha$ correction (macro-conditioned)
  - factor-loading $\beta$ correction (multiplicative, macro-conditioned)
  - idiosyncratic volatility $\sigma$ correction (multiplicative)
  - residual $\varepsilon$ via a learned network (the $\varepsilon$-net)

Training: WGAN-GP (gradient penalty, $n_{\text{critic}}$ iterations), as per
\citet{huh2026marketgans}. Factor inputs are our hierarchical PCA factors
(same as SF-MarketGAN-R), with strictly lagged $f_{t-1}$ conditioning.

Usage:
    python src/train_marketgan_60asset.py \
        --preprocessed_dir data/preprocessed \
        --save_dir results/marketgan60/lr1e4_nc5 \
        --lr 1e-4 --n_critic 5 --seed 42
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer import SequenceDataset, ProperValDataset


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=self.pad)
    def forward(self, x):
        o = self.conv(x)
        return o[:, :, :-self.pad] if self.pad > 0 else o


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=2, dilation=1, dropout=0.2):
        super().__init__()
        self.c1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.c2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        h = F.relu(self.c1(x)); h = self.drop(h)
        h = F.relu(self.c2(h)); h = self.drop(h)
        return x + h


class MarketGANGenerator(nn.Module):
    """Full-GAN generator: learns alpha, beta, sigma corrections + residual net."""
    def __init__(self, n_assets, n_factors, d_z=10, d_cov=15,
                 hidden=80, num_blocks=6, dropout=0.2):
        super().__init__()
        self.N = n_assets; self.K = n_factors; self.d_z = d_z
        self.rfs = 1 + 2 * (2 - 1) * (2 ** num_blocks - 1) // (2 - 1)

        self.backbone = nn.Sequential(
            nn.Conv1d(d_z + d_cov, hidden, 1),
            *[ResBlock(hidden, 2, 2**i, dropout) for i in range(num_blocks)])
        self.alpha_out = nn.Conv1d(hidden, n_assets, 1)
        self.beta_out  = nn.Conv1d(hidden, n_assets * n_factors, 1)
        self.sigma_out = nn.Conv1d(hidden, n_assets, 1)
        # eps-net: another TCN on noise+macro → residuals
        self.eps_net = nn.Sequential(
            nn.Conv1d(d_z + d_cov, hidden, 1),
            *[ResBlock(hidden, 2, 2**i, dropout) for i in range(num_blocks)],
            nn.Conv1d(hidden, n_assets, 1))

    def forward(self, z, cov, alpha_hat, beta_hat, sigma_hat, factors):
        B = z.shape[0]; T = factors.shape[2]
        x = torch.cat([z[:, :, :-1], cov[:, :, :-1]], 1)
        h = self.backbone(x)[:, :, -T:]

        alpha_c = torch.tanh(self.alpha_out(h))
        beta_c  = torch.tanh(self.beta_out(h)).view(B, self.N, self.K, T)
        sigma_c = torch.tanh(self.sigma_out(h))

        alpha = alpha_hat * (1 + alpha_c)
        beta  = beta_hat  * (1 + beta_c)
        sigma = sigma_hat * torch.abs(1 + sigma_c).clamp(min=0.01)

        eps_in = torch.cat([z[:, :, -T:], cov[:, :, -(T+1):-1]], 1)
        eps = self.eps_net(eps_in)

        factor_term = torch.einsum('bnkt,bkt->bnt', beta, factors)
        return alpha + factor_term + sigma * eps


class MarketGANDiscriminator(nn.Module):
    def __init__(self, n_assets, d_cov=15, hidden=160, num_blocks=6, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_assets + d_cov, hidden, 1),
            *[ResBlock(hidden, 2, 2**i, dropout) for i in range(num_blocks)],
            nn.Conv1d(hidden, 1, 1))
    def forward(self, r, cov):
        return self.net(torch.cat([r, cov], 1)).mean(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preprocessed_dir", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--min_save_epoch", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n_critic", type=int, default=5)
    p.add_argument("--vol_penalty", type=float, default=50.0)
    p.add_argument("--grad_penalty", type=float, default=10.0)
    p.add_argument("--hidden_g", type=int, default=80)
    p.add_argument("--hidden_d", type=int, default=160)
    p.add_argument("--num_blocks", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--steps_per_epoch", type=int, default=20)
    p.add_argument("--lag_factors", action="store_true", default=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # Load preprocessed
    pre = args.preprocessed_dir
    meta = json.load(open(os.path.join(pre, "meta.json")))
    returns_vals = np.load(os.path.join(pre, "returns.npy"))
    f_arr = np.load(os.path.join(pre, "f_arr_lag1.npy")) if args.lag_factors else np.load(os.path.join(pre, "f_arr.npy"))
    alpha_hat = np.load(os.path.join(pre, "alpha_hat.npy"))
    beta_hat  = np.load(os.path.join(pre, "beta_hat.npy"))
    sigma_hat = np.load(os.path.join(pre, "sigma_hat.npy"))
    cov_arr   = np.load(os.path.join(pre, "cov_arr.npy"))
    n_factors = meta["n_factors"]; d_cov = meta["d_cov"]
    n_assets  = meta["n_assets"]
    tr_end_idx = meta["dates"]["train_end_idx"]
    val_end_idx = meta["dates"]["val_end_idx"]

    t_l = 252
    rfs = 1 + 2 * (2 - 1) * (2 ** args.num_blocks - 1) // (2 - 1)  # for RFS

    train_ds = SequenceDataset(
        returns_vals[:tr_end_idx], f_arr[:tr_end_idx],
        cov_arr[:tr_end_idx], alpha_hat[:tr_end_idx], beta_hat[:tr_end_idx],
        sigma_hat[:tr_end_idx], rfs=rfs, t_l=t_l)
    val_ds = ProperValDataset(
        returns_vals[:val_end_idx], f_arr[:val_end_idx],
        cov_arr[:val_end_idx], alpha_hat[:val_end_idx], beta_hat[:val_end_idx],
        sigma_hat[:val_end_idx], rfs=rfs, t_l=t_l,
        val_start_idx=tr_end_idx)

    G = MarketGANGenerator(n_assets, n_factors, d_z=10, d_cov=d_cov,
                            hidden=args.hidden_g, num_blocks=args.num_blocks).to(device)
    D = MarketGANDiscriminator(n_assets, d_cov=d_cov,
                                hidden=args.hidden_d, num_blocks=args.num_blocks).to(device)
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[MarketGAN-60] {n_assets} assets, K={n_factors}, d_cov={d_cov}, "
          f"lr={args.lr}, n_critic={args.n_critic}, vol_pen={args.vol_penalty}, device={device}")
    print(f"  G params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"  D params: {sum(p.numel() for p in D.parameters()):,}")

    best_frob = float("inf"); no_improve = 0
    history = {"d_loss": [], "g_loss": [], "val_frob": [], "gp": []}
    T_full = rfs + t_l + 1

    for epoch in range(args.epochs):
        G.train(); D.train()
        d_losses, g_losses, gps = [], [], []

        for step in range(args.steps_per_epoch):
            batch = train_ds.sample_batch(args.batch_size, device)
            r_real = batch["returns"]; B = r_real.shape[0]
            cov = batch["covariates"]
            if cov.shape[2] < T_full:
                cov = F.pad(cov, (0, T_full - cov.shape[2]), mode="replicate")
            cov_d = batch["covariates"][:, :, -t_l:]

            # WGAN-GP critic steps
            for _ in range(args.n_critic):
                z = torch.randn(B, 10, T_full, device=device)
                with torch.no_grad():
                    r_fake = G(z, cov[:, :, :T_full], batch["alpha_hat"],
                               batch["beta_hat"], batch["sigma_hat"], batch["factors"])
                d_real = D(r_real, cov_d).mean()
                d_fake = D(r_fake, cov_d).mean()
                # Gradient penalty
                alpha_int = torch.rand(B, 1, 1, device=device)
                r_int = (alpha_int * r_real + (1 - alpha_int) * r_fake).requires_grad_(True)
                d_int = D(r_int, cov_d)
                grads = torch.autograd.grad(d_int, r_int, torch.ones_like(d_int),
                                             create_graph=True, retain_graph=True)[0]
                gp = args.grad_penalty * ((grads.view(B, -1).norm(2, 1) - 1) ** 2).mean()
                d_loss = d_fake - d_real + gp
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()
                gps.append(gp.item())
            d_losses.append(d_loss.item())

            # Generator step
            z = torch.randn(B, 10, T_full, device=device)
            r_fake_g = G(z, cov[:, :, :T_full], batch["alpha_hat"],
                         batch["beta_hat"], batch["sigma_hat"], batch["factors"])
            g_adv = -D(r_fake_g, cov_d).mean()
            # Volatility penalty (MarketGAN's stability term)
            rv = r_real.std(2); fv = r_fake_g.std(2)
            vol_pen = args.vol_penalty * ((fv / (rv + 1e-8) - 1) ** 2).mean()
            g_loss = g_adv + vol_pen
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()
            g_losses.append(g_loss.item())

        # Validate
        G.eval()
        with torch.no_grad():
            rng = torch.random.get_rng_state()
            torch.manual_seed(999)
            all_r, all_f = [], []
            for _ in range(5):
                vb = val_ds.sample_batch(64, device)
                Bv = vb["returns"].shape[0]
                zv = torch.randn(Bv, 10, T_full, device=device)
                cv = vb["covariates"]
                if cv.shape[2] < T_full:
                    cv = F.pad(cv, (0, T_full - cv.shape[2]), mode="replicate")
                rf = G(zv, cv[:, :, :T_full], vb["alpha_hat"], vb["beta_hat"],
                       vb["sigma_hat"], vb["factors"])
                all_r.append(vb["returns"].permute(0, 2, 1).cpu().numpy().reshape(-1, n_assets))
                all_f.append(rf.permute(0, 2, 1).cpu().numpy().reshape(-1, n_assets))
            torch.random.set_rng_state(rng)
            real = np.concatenate(all_r); fake = np.concatenate(all_f)
            cr = np.nan_to_num(np.corrcoef(real.T), nan=0.0)
            cf = np.nan_to_num(np.corrcoef(fake.T), nan=0.0)
            frob = float(np.linalg.norm(cr - cf, "fro"))

        dmean = float(np.mean(d_losses)); gmean = float(np.mean(g_losses))
        gpmean = float(np.mean(gps))
        history["d_loss"].append(dmean); history["g_loss"].append(gmean)
        history["val_frob"].append(frob); history["gp"].append(gpmean)
        print(f"Ep {epoch+1:3d}/{args.epochs} D={dmean:+.3e} G={gmean:+.3e} "
              f"GP={gpmean:.2f} frob={frob:.4f} (best={best_frob:.4f})")

        if epoch < args.min_save_epoch:
            continue
        if frob < best_frob:
            best_frob = frob; no_improve = 0
            torch.save({"G": G.state_dict(), "D": D.state_dict(),
                        "epoch": epoch, "frob": frob, "args": vars(args)},
                       os.path.join(args.save_dir, "best_model.pt"))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stop at epoch {epoch+1}"); break

    json.dump(history, open(os.path.join(args.save_dir, "history.json"), "w"), indent=2)
    config = {"version": "MarketGAN-60",
              "seed": args.seed, "best_frob": best_frob,
              "lr": args.lr, "n_critic": args.n_critic,
              "vol_penalty": args.vol_penalty, "grad_penalty": args.grad_penalty,
              "hidden_g": args.hidden_g, "hidden_d": args.hidden_d,
              "num_blocks": args.num_blocks, "lag_factors": args.lag_factors,
              "n_factors": n_factors, "d_cov": d_cov}
    json.dump(config, open(os.path.join(args.save_dir, "config.json"), "w"), indent=2)
    print(f"Done. Best val frob: {best_frob:.4f}")


if __name__ == "__main__":
    main()
