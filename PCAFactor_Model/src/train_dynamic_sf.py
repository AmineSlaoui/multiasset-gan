"""Dynamic SF loss lambda experiment.

Tests 5 SF scheduling strategies:
  1. fixed_20   — baseline fixed λ=20 (current default)
  2. fixed_1    — gentle fixed λ=1
  3. ema_03     — EMA balanced τ=0.3 (existing code)
  4. ema_01     — EMA balanced τ=0.1 (lighter SF)
  5. cosine     — cosine warmup from 0→λ_max=5, start after epoch 30
  6. none       — pure adversarial (λ=0, control)

All use Light arch (128/2b) which showed best CorrFrob in ablation.
"""
import argparse, sys, os, json, time, copy
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_returns, load_macro, ASSET_CLASSES, train_val_test_split
from factors_pca import compute_hierarchical_pca_factors, compute_rolling_coefficients
from macro_processor import MacroProcessor
from model import ResidualBlock, FiLMLayer
from sfmg_baseline import SNFiLMLayer, Discriminator
from stylized_losses import StylizedFactLoss
from trainer import SequenceDataset, ProperValDataset


class DynamicSFGenerator(nn.Module):
    """Light generator (128/2b) for dynamic SF experiments."""
    def __init__(self, n_assets=60, n_factors=11, d_z=10, d_cov=15,
                 hidden_dim=128, num_blocks=2, dropout=0.2,
                 eta=0.1, eta_sigma=0.1, residual_cholesky=None):
        super().__init__()
        self.n_assets = n_assets; self.n_factors = n_factors
        self.d_z = d_z; self.d_cov = d_cov
        self.eta = eta; self.eta_sigma = eta_sigma
        self.rfs = 1 + 2 * (2 - 1) * (2 ** num_blocks - 1) // (2 - 1)

        if residual_cholesky is not None:
            self.register_buffer('L_eps', torch.tensor(residual_cholesky, dtype=torch.float32))
        else:
            self.L_eps = None

        self.input_proj = nn.Conv1d(d_z, hidden_dim, 1)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, 2, 2**i, dropout) for i in range(num_blocks)
        ])
        self.film = FiLMLayer(d_cov, hidden_dim)
        self.alpha_out = nn.Conv1d(hidden_dim, n_assets, 1)
        self.sigma_out = nn.Conv1d(hidden_dim, n_assets, 1)

    def forward(self, z_full, cov_full, alpha_hat, beta_hat, sigma_hat, factors):
        B = z_full.shape[0]; T_L = factors.shape[2]
        h = self.input_proj(z_full[:, :, :-1])
        for block in self.blocks: h = block(h)
        h = self.film(h, cov_full[:, :, :-1])
        h_out = h[:, :, -T_L:]

        alpha_corr = self.eta * torch.tanh(self.alpha_out(h_out))
        alpha = alpha_hat * (1 + alpha_corr)
        beta = beta_hat

        sigma_corr = self.eta_sigma * torch.tanh(self.sigma_out(h_out))
        sigma = sigma_hat * torch.exp(sigma_corr)

        u = torch.randn(B, self.n_assets, T_L, device=z_full.device)
        eps = torch.einsum('ij,bjt->bit', self.L_eps, u) if self.L_eps is not None else u

        factor_term = torch.einsum('bnkt,bkt->bnt', beta, factors)
        r = alpha + factor_term + sigma * eps
        return r


class EMALossBalancer:
    """EMA-based adaptive SF scaling."""
    def __init__(self, tau=0.3, ema_decay=0.99, s_min=0.01, s_max=10.0):
        self.tau = tau
        self.ema_decay = ema_decay
        self.s_min = s_min
        self.s_max = s_max
        self.ema_adv = None
        self.ema_sf = None

    def scale(self, l_adv: float, l_sf: float) -> float:
        l_adv_abs = abs(l_adv) + 1e-8
        l_sf_abs = abs(l_sf) + 1e-8
        if self.ema_adv is None:
            self.ema_adv = l_adv_abs
            self.ema_sf = l_sf_abs
        else:
            self.ema_adv = self.ema_decay * self.ema_adv + (1 - self.ema_decay) * l_adv_abs
            self.ema_sf = self.ema_decay * self.ema_sf + (1 - self.ema_decay) * l_sf_abs
        s = (self.tau / (1 - self.tau)) * (self.ema_adv / (self.ema_sf + 1e-8))
        return max(self.s_min, min(self.s_max, s))


def cosine_lambda(epoch, warmup=30, total=300, lam_max=5.0):
    """Cosine warmup schedule: 0 during warmup, then cos ramp to lam_max."""
    if epoch < warmup:
        return 0.0
    progress = (epoch - warmup) / max(total - warmup, 1)
    # Cosine from 0 to lam_max (half cosine)
    return lam_max * 0.5 * (1 - np.cos(np.pi * min(progress, 1.0)))


SCHEDULES = {
    "fixed_20": {"type": "fixed", "lam": 20.0},
    "fixed_1": {"type": "fixed", "lam": 1.0},
    "ema_03": {"type": "ema", "tau": 0.3},
    "ema_01": {"type": "ema", "tau": 0.1},
    "cosine": {"type": "cosine", "lam_max": 5.0, "warmup": 30},
    "none": {"type": "fixed", "lam": 0.0},
}


def train_with_schedule(schedule_name, schedule_cfg, G, D, sf_loss,
                        train_ds, val_ds, class_indices,
                        device, save_dir, epochs=300, batch_size=128,
                        steps_per_epoch=30, patience=50, t_l=252,
                        lr=1e-4, grad_clip=1.0, min_save_epoch=50,
                        run_suffix=""):
    """Train one model with a given SF schedule.

    min_save_epoch: best-checkpoint tracking and patience only engage once the
    epoch index reaches this value. Needed because pure-adversarial warmup
    (epoch < 30) + ramp (30-50) often produces the lowest val_frob purely due
    to the SF term not yet being active — without this gate, all non-SF
    schedules collapse onto the same warmup checkpoint and early-stop before
    their schedules have a chance to differentiate.
    """
    dir_name = f"{schedule_name}{run_suffix}" if run_suffix else schedule_name
    save_path = os.path.join(save_dir, dir_name)
    os.makedirs(save_path, exist_ok=True)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

    balancer = None
    if schedule_cfg["type"] == "ema":
        balancer = EMALossBalancer(tau=schedule_cfg["tau"])

    history = {"d_loss": [], "g_loss": [], "sf": [], "sf_scale": [], "val_frob": []}
    best_frob = float("inf")
    no_improve = 0

    rfs = G.rfs

    print(f"\n{'='*70}")
    print(f"Schedule: {schedule_name} | Config: {schedule_cfg}")
    print(f"  G: {sum(p.numel() for p in G.parameters()):,} params")
    print(f"  D: {sum(p.numel() for p in D.parameters()):,} params")
    print(f"{'='*70}")

    for epoch in range(epochs):
        G.train(); D.train()
        d_losses, g_losses, sf_vals, sf_scales = [], [], [], []

        # Compute current lambda
        if schedule_cfg["type"] == "fixed":
            current_lam = schedule_cfg["lam"]
            use_sf = current_lam > 0 and epoch >= 30  # warmup
        elif schedule_cfg["type"] == "ema":
            use_sf = epoch >= 30  # warmup
            current_lam = None  # computed dynamically
        elif schedule_cfg["type"] == "cosine":
            current_lam = cosine_lambda(epoch, schedule_cfg.get("warmup", 30),
                                         epochs, schedule_cfg["lam_max"])
            use_sf = current_lam > 0

        for step in range(steps_per_epoch):
            batch = train_ds.sample_batch(batch_size, device)
            r_real = batch["returns"]
            T_full = rfs + t_l + 1
            z = torch.randn(batch["factors"].shape[0], G.d_z, T_full, device=device)
            cov = batch["covariates"]
            if cov.shape[2] < T_full:
                cov = F.pad(cov, (0, T_full - cov.shape[2]), mode="replicate")
            cov_d = batch["covariates"][:, :, -t_l:]

            # ── Discriminator ──
            with torch.no_grad():
                r_fake = G(z, cov[:, :, :T_full], batch["alpha_hat"],
                           batch["beta_hat"], batch["sigma_hat"], batch["factors"])
            d_real = D(r_real, cov_d)
            d_fake = D(r_fake, cov_d)
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()
            d_losses.append(d_loss.item())

            # ── Generator ──
            z2 = torch.randn(batch["factors"].shape[0], G.d_z, T_full, device=device)
            r_fake_g = G(z2, cov[:, :, :T_full], batch["alpha_hat"],
                         batch["beta_hat"], batch["sigma_hat"], batch["factors"])
            g_adv = -D(r_fake_g, cov_d).mean()

            if use_sf and sf_loss is not None:
                r_real_t = r_real.permute(0, 2, 1)
                r_fake_t = r_fake_g.permute(0, 2, 1)
                sf = sf_loss(r_real_t, r_fake_t)
                g_sf = sf["total"]

                if schedule_cfg["type"] == "fixed":
                    ramp = min(1.0, (epoch - 30) / 20.0)
                    s = current_lam * ramp
                elif schedule_cfg["type"] == "ema":
                    ramp = min(1.0, (epoch - 30) / 20.0)
                    s = balancer.scale(g_adv.item(), g_sf.item()) * ramp
                elif schedule_cfg["type"] == "cosine":
                    s = current_lam  # already has warmup built in

                g_loss = g_adv + s * g_sf
                sf_vals.append(g_sf.item())
                sf_scales.append(s)
            else:
                g_loss = g_adv
                sf_vals.append(0.0)
                sf_scales.append(0.0)

            opt_G.zero_grad()
            g_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
            opt_G.step()
            g_losses.append(g_loss.item())

        # ── Validate ──
        G.eval()
        with torch.no_grad():
            val_rng = np.random.RandomState(999)
            orig_np = np.random.get_state()
            orig_torch = torch.random.get_rng_state()
            np.random.set_state(val_rng.get_state())
            torch.manual_seed(999)
            all_real, all_fake = [], []
            for _ in range(5):
                vb = val_ds.sample_batch(64, device)
                T_full_v = rfs + t_l + 1
                zv = torch.randn(vb["factors"].shape[0], G.d_z, T_full_v, device=device)
                cv = vb["covariates"]
                if cv.shape[2] < T_full_v:
                    cv = F.pad(cv, (0, T_full_v - cv.shape[2]), mode="replicate")
                rf = G(zv, cv[:, :, :T_full_v], vb["alpha_hat"],
                       vb["beta_hat"], vb["sigma_hat"], vb["factors"])
                all_real.append(vb["returns"].permute(0, 2, 1).cpu().numpy().reshape(-1, G.n_assets))
                all_fake.append(rf.permute(0, 2, 1).cpu().numpy().reshape(-1, G.n_assets))
            np.random.set_state(orig_np)
            torch.random.set_rng_state(orig_torch)
            real = np.concatenate(all_real)
            fake = np.concatenate(all_fake)
            cr = np.nan_to_num(np.corrcoef(real.T), nan=0.0)
            cf = np.nan_to_num(np.corrcoef(fake.T), nan=0.0)
            frob = float(np.linalg.norm(cr - cf, "fro"))

        d_m = np.mean(d_losses); g_m = np.mean(g_losses)
        sf_m = np.mean(sf_vals); ss_m = np.mean(sf_scales)

        history["d_loss"].append(d_m)
        history["g_loss"].append(g_m)
        history["sf"].append(sf_m)
        history["sf_scale"].append(ss_m)
        history["val_frob"].append(frob)

        sf_str = f" SF={sf_m:.4f}(λ={ss_m:.2f})" if sf_m > 0 else ""
        print(f"[{schedule_name}] Epoch {epoch+1:3d}/{epochs} | D={d_m:.4f} "
              f"G={g_m:.4f}{sf_str} | frob={frob:.4f} "
              f"(best={best_frob:.4f}) | {time.time():.0f}")

        if epoch < min_save_epoch:
            # Pre-gate: do not save or accumulate patience. The SF schedule
            # is not in full effect yet, so checkpoints here would bias the
            # comparison toward pure-adversarial models.
            continue

        if frob < best_frob:
            best_frob = frob
            no_improve = 0
            torch.save({"G": G.state_dict(), "D": D.state_dict(),
                        "epoch": epoch, "frob": frob, "schedule": schedule_name},
                       os.path.join(save_path, "best_model.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{schedule_name}] Early stopping at epoch {epoch+1}")
                break

    # Save history
    with open(os.path.join(save_path, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"[{schedule_name}] Best val frob: {best_frob:.4f}")
    return best_frob, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", default="results/dynamic_sf")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--schedules", nargs="+",
                   default=["none", "fixed_1", "ema_01", "ema_03", "cosine", "fixed_20"],
                   help="Which schedules to run")
    p.add_argument("--min_save_epoch", type=int, default=50,
                   help="Only start tracking best/patience after this epoch")
    p.add_argument("--run_suffix", default="",
                   help="Suffix on per-schedule output dir (e.g. _s1 for multi-seed)")
    p.add_argument("--hidden_dim", type=int, default=128, help="G hidden dim")
    p.add_argument("--num_blocks", type=int, default=2, help="G num residual blocks")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data (follow train_ablation.py pattern exactly) ──
    returns = load_returns()
    macro_df = load_macro()

    r_train, r_val, r_test = train_val_test_split(returns)
    proc = MacroProcessor(n_pca_components=8, n_regimes=3)
    covariates = proc.fit_transform(macro_df, returns.index,
                                     train_end=str(r_train.index[-1].date()))

    # compute_hierarchical_pca_factors wants DataFrame + ASSET_CLASSES dict
    factors_df, _ = compute_hierarchical_pca_factors(
        returns, ASSET_CLASSES, n_global=5, n_class=2,
        train_end=str(r_train.index[-1].date()))
    n_factors = factors_df.shape[1]

    alpha_hat, beta_hat, sigma_hat = compute_rolling_coefficients(
        returns, factors_df, window=126)
    cov_arr = covariates.reindex(returns.index).fillna(0).values
    d_cov = cov_arr.shape[1]
    f_arr = factors_df.values

    # Integer class indices for StylizedFactLoss
    asset_list = list(returns.columns)
    class_indices = {c: [asset_list.index(a) for a in v]
                     for c, v in ASSET_CLASSES.items()}

    # Shrunk residual covariance
    chol_path = "data/residual_cholesky.npy"
    L_eps = np.load(chol_path) if os.path.exists(chol_path) else None

    train_end_idx = returns.index.get_loc(r_train.index[-1]) + 1
    val_end_idx = returns.index.get_loc(r_val.index[-1]) + 1
    n_assets = returns.shape[1]
    t_l = 252
    rfs = 127

    train_ds = SequenceDataset(
        returns.values[:train_end_idx], f_arr[:train_end_idx],
        cov_arr[:train_end_idx],
        alpha_hat[:train_end_idx], beta_hat[:train_end_idx],
        sigma_hat[:train_end_idx], rfs=rfs, t_l=t_l)
    val_ds = ProperValDataset(
        returns.values[:val_end_idx], f_arr[:val_end_idx],
        cov_arr[:val_end_idx],
        alpha_hat[:val_end_idx], beta_hat[:val_end_idx],
        sigma_hat[:val_end_idx], rfs=rfs, t_l=t_l,
        val_start_idx=train_end_idx)

    sf_loss = StylizedFactLoss(class_indices=class_indices)

    results = {}

    for sname in args.schedules:
        if sname not in SCHEDULES:
            print(f"Unknown schedule: {sname}, skipping")
            continue

        torch.manual_seed(args.seed); np.random.seed(args.seed)

        G = DynamicSFGenerator(
            n_assets=n_assets, n_factors=n_factors, d_z=10, d_cov=d_cov,
            hidden_dim=args.hidden_dim, num_blocks=args.num_blocks, dropout=0.2,
            eta=0.1, eta_sigma=0.1, residual_cholesky=L_eps
        ).to(device)

        D = Discriminator(
            n_assets=n_assets, d_cov=d_cov,
            hidden_dim=128, num_blocks=4
        ).to(device)

        # Persist arch config so eval can reconstruct the generator
        sdir = os.path.join(args.save_dir,
                            f"{sname}{args.run_suffix}" if args.run_suffix else sname)
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, "config.json"), "w") as cf:
            json.dump({"hidden_dim": args.hidden_dim,
                       "num_blocks": args.num_blocks,
                       "seed": args.seed,
                       "min_save_epoch": args.min_save_epoch,
                       "schedule": sname}, cf, indent=2)

        best_frob, hist = train_with_schedule(
            sname, SCHEDULES[sname], G, D, sf_loss,
            train_ds, val_ds, class_indices,
            device, args.save_dir, epochs=args.epochs,
            batch_size=128, steps_per_epoch=30,
            patience=args.patience, t_l=t_l,
            min_save_epoch=args.min_save_epoch,
            run_suffix=args.run_suffix)

        results[sname] = {"best_val_frob": best_frob}

    # Summary
    print(f"\n{'='*70}")
    print("Dynamic SF Lambda Experiment Summary")
    print(f"{'='*70}")
    print(f"{'Schedule':<15} {'Best Val Frob':>12}")
    print("-" * 30)
    for sname in args.schedules:
        if sname in results:
            print(f"{sname:<15} {results[sname]['best_val_frob']:>12.4f}")

    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
