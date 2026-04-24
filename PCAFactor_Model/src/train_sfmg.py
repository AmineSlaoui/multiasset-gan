"""Train SF-MarketGAN-R (the regime-conditional generator).

Model: regime encoder + regime-gated corrections on (α, σ) + optional
rank-r residual stress perturbation (see ``sfmg_generator.py``). An
auxiliary BCE loss aligns the regime encoder with a VIX-threshold-based
stress label so the learned regime is interpretable.

Usage
-----
    python src/train_sfmg.py --save_dir results/sfmg_seed42 --seed 42 \
        --hidden_dim 256 --num_blocks 4 --epochs 200 --patience 40 \
        --factors sector --lag_factors --regime_aux_weight 1.0

Lag-1 factors (f_{t-1} conditioning) are enabled with --lag_factors.
"""
import argparse, sys, os, json, time
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_returns, load_macro, ASSET_CLASSES, train_val_test_split
from factors_pca import compute_hierarchical_pca_factors, compute_rolling_coefficients
from macro_processor import MacroProcessor
from sfmg_baseline import Discriminator
from sfmg_generator import SFMGGenerator
from stylized_losses import StylizedFactLoss
from trainer import SequenceDataset, ProperValDataset


def make_stress_label(cov_arr, vix_col_idx, z_threshold=0.3):
    """Return binary (T,) array: 1 where VIX z > threshold."""
    return (cov_arr[:, vix_col_idx] > z_threshold).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", default="results/sfmg")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--steps_per_epoch", type=int, default=30)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--residual_rank", type=int, default=4)
    p.add_argument("--trainable_L_rho", action="store_true",
                   help="Make the 60x60 residual Cholesky an nn.Parameter "
                        "so GAN loss can adjust high-residual-correlation pairs "
                        "(e.g. Visa/Mastercard). Initialised from "
                        "data/residual_cholesky.npy. Reported in the paper as "
                        "failing — the discriminator saturates and L_ρ drifts.")
    p.add_argument("--L_rho_lr", type=float, default=1e-6,
                   help="Separate learning rate for L_rho once unfrozen.")
    p.add_argument("--L_rho_freeze_epochs", type=int, default=50,
                   help="Freeze L_rho at its init for this many epochs to "
                        "let the rest of G stabilize before letting L_rho drift.")
    p.add_argument("--warm_start", default=None,
                   help="Path to an SFMG best_model.pt to initialise G/D from. "
                        "Recommended when using --trainable_L_rho to avoid the "
                        "D-dominates-from-scratch failure mode.")
    p.add_argument("--eta_low", type=float, default=0.01)
    p.add_argument("--eta_high", type=float, default=0.10)
    p.add_argument("--eta_sigma_low", type=float, default=0.05)
    p.add_argument("--eta_sigma_high", type=float, default=0.20)
    p.add_argument("--sf_lambda", type=float, default=20.0)
    p.add_argument("--sf_warmup", type=int, default=30)
    p.add_argument("--regime_aux_weight", type=float, default=1.0,
                   help="Weight on BCE(z, VIX stress) auxiliary loss")
    p.add_argument("--vix_threshold", type=float, default=0.3)
    p.add_argument("--min_save_epoch", type=int, default=50)
    p.add_argument("--lag_factors", action="store_true",
                   help="Use lag-1 factors f_{t-1} (reviewer U3)")
    p.add_argument("--train_end_override", default=None,
                   help="YYYY-MM-DD to override split for rolling-cutoff runs")
    p.add_argument("--preprocessed_dir", default=None,
                   help="If set, load returns/factors/OLS/covariates from this "
                        "directory (produced by src/preprocess_data.py) instead of "
                        "recomputing. Skips the 3-5min data pipeline per run. "
                        "Only compatible with --factors pca.")
    p.add_argument("--factors", choices=["pca", "sector"], default="pca",
                   help="pca: 11 hierarchical PCA factors (5 global + 2×3 class). "
                        "sector: 16 factors (5 global + 2 bonds-PCA + 5 stock "
                        "sector + 4 commodity GSCI sector, GS-orthogonalised). "
                        "sector disables --preprocessed_dir.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"[sfmg] device={device}, seed={args.seed}, arch={args.hidden_dim}/{args.num_blocks}b, "
          f"eta={args.eta_low}->{args.eta_high}, residual_rank={args.residual_rank}")

    # ── Data ─────────────────────────────────────────────────────────
    if args.preprocessed_dir is not None:
        if args.factors != "pca":
            raise ValueError(
                "--preprocessed_dir is only compatible with --factors pca; "
                "preprocessed arrays are computed under the PCA pipeline and "
                "silently mismatching them with --factors sector produces a "
                "config.json that misrepresents what the checkpoint actually "
                "trained on (CODEX_REVIEW.md issue #9). Either drop "
                "--preprocessed_dir or rebuild it for sector factors.")
        pre = args.preprocessed_dir
        meta = json.load(open(os.path.join(pre, "meta.json")))
        train_end = meta["train_end"]
        if args.train_end_override is not None and args.train_end_override != train_end:
            raise ValueError(
                f"Preprocessed dir {pre} was built for train_end={train_end}, "
                f"but --train_end_override={args.train_end_override}. "
                "Rebuild the preprocessed dir for the override first.")
        returns_vals = np.load(os.path.join(pre, "returns.npy"))
        f_base = np.load(os.path.join(pre, "f_arr.npy"))
        f_lag = np.load(os.path.join(pre, "f_arr_lag1.npy"))
        alpha_hat = np.load(os.path.join(pre, "alpha_hat.npy"))
        beta_hat = np.load(os.path.join(pre, "beta_hat.npy"))
        sigma_hat = np.load(os.path.join(pre, "sigma_hat.npy"))
        cov_arr = np.load(os.path.join(pre, "cov_arr.npy"))
        L_eps_path = os.path.join(pre, "L_eps.npy")
        L_eps = np.load(L_eps_path) if os.path.exists(L_eps_path) else None
        n_factors = meta["n_factors"]; d_cov = meta["d_cov"]
        n_assets = meta["n_assets"]
        vix_idx = meta["vix_col_index"]
        class_indices = meta["class_indices"]
        tr_end_idx = meta["dates"]["train_end_idx"]
        val_end_idx = meta["dates"]["val_end_idx"]
        f_arr = f_lag if args.lag_factors else f_base
        print(f"[sfmg] loaded preprocessed data from {pre}")
    else:
        returns = load_returns()
        macro = load_macro()

        if args.train_end_override is not None:
            all_idx = returns.index
            tr_end_idx = all_idx.get_loc(all_idx[all_idx <= args.train_end_override][-1]) + 1
            val_end_idx = min(tr_end_idx + 252, len(all_idx))
            test_end_idx = min(val_end_idx + 251, len(all_idx))
            train_ret = returns.iloc[:tr_end_idx]
            val_ret = returns.iloc[tr_end_idx:val_end_idx]
            test_ret = returns.iloc[val_end_idx:test_end_idx]
            train_end = str(train_ret.index[-1].date())
        else:
            train_ret, val_ret, test_ret = train_val_test_split(returns)
            train_end = str(train_ret.index[-1].date())

        proc = MacroProcessor(n_pca_components=8, n_regimes=3)
        covariates = proc.fit_transform(macro, returns.index, train_end=train_end)

        if args.factors == "sector":
            from factors_sector import compute_sector_factors
            factors_df, _ = compute_sector_factors(
                returns, ASSET_CLASSES, train_end=train_end)
        else:
            factors_df, _ = compute_hierarchical_pca_factors(
                returns, ASSET_CLASSES, n_global=5, n_class=2, train_end=train_end)
        n_factors = factors_df.shape[1]
        alpha_hat, beta_hat, sigma_hat = compute_rolling_coefficients(
            returns, factors_df, window=126)

        cov_arr = covariates.reindex(returns.index).fillna(0).values
        d_cov = cov_arr.shape[1]
        f_arr = factors_df.values
        if args.lag_factors:
            f_arr = np.vstack([np.zeros((1, n_factors)), f_arr[:-1]])

        vix_col_name = "vix_level_z"
        cov_cols = list(covariates.columns)
        vix_idx = cov_cols.index(vix_col_name)

        asset_list = list(returns.columns)
        class_indices = {c: [asset_list.index(a) for a in v]
                         for c, v in ASSET_CLASSES.items()}

        if args.factors == "sector":
            # v3 factors → residual structure changed, legacy L_eps doesn't apply.
            # Compute from training-period residuals with sample correlation Cholesky.
            print(f"  [L_eps] computing from v3 training residuals")
            tr_end_idx_for_L = returns.index.get_loc(train_ret.index[-1]) + 1
            res = np.zeros_like(returns.values)
            for t in range(tr_end_idx_for_L):
                res[t] = returns.values[t] - alpha_hat[t] - beta_hat[t] @ f_arr[t]
            res_tr = res[126:tr_end_idx_for_L]   # skip initial OLS warmup
            Ccorr = np.corrcoef(res_tr.T)
            w, V = np.linalg.eigh(0.5 * (Ccorr + Ccorr.T))
            w = np.clip(w, 1e-6, None)
            Cpsd = (V * w) @ V.T
            dd = np.sqrt(np.diag(Cpsd))
            Cnorm = Cpsd / np.outer(dd, dd)
            L_eps = np.linalg.cholesky(Cnorm + 1e-6 * np.eye(Cnorm.shape[0]))
            print(f"  [L_eps] shape={L_eps.shape}  off-diag |mean|="
                  f"{np.abs(Cnorm[np.triu_indices(len(Cnorm), k=1)]).mean():.3f}")
        else:
            L_eps = np.load("data/residual_cholesky.npy") if os.path.exists("data/residual_cholesky.npy") else None

        tr_end_idx = returns.index.get_loc(train_ret.index[-1]) + 1
        val_end_idx = returns.index.get_loc(val_ret.index[-1]) + 1
        n_assets = returns.shape[1]
        returns_vals = returns.values

    t_l = 252
    rfs = 1 + 2 * (2 - 1) * (2 ** args.num_blocks - 1) // (2 - 1)

    train_ds = SequenceDataset(
        returns_vals[:tr_end_idx], f_arr[:tr_end_idx],
        cov_arr[:tr_end_idx],
        alpha_hat[:tr_end_idx], beta_hat[:tr_end_idx],
        sigma_hat[:tr_end_idx], rfs=rfs, t_l=t_l)
    val_ds = ProperValDataset(
        returns_vals[:val_end_idx], f_arr[:val_end_idx],
        cov_arr[:val_end_idx],
        alpha_hat[:val_end_idx], beta_hat[:val_end_idx],
        sigma_hat[:val_end_idx], rfs=rfs, t_l=t_l,
        val_start_idx=tr_end_idx)

    # ── Model ────────────────────────────────────────────────────────
    gen = SFMGGenerator(
        n_assets=n_assets, n_factors=n_factors, d_z=10, d_cov=d_cov,
        hidden_dim=args.hidden_dim, num_blocks=args.num_blocks, dropout=0.2,
        eta_low=args.eta_low, eta_high=args.eta_high,
        eta_sigma_low=args.eta_sigma_low, eta_sigma_high=args.eta_sigma_high,
        residual_rank=args.residual_rank,
        residual_cholesky=L_eps,
        trainable_L_rho=args.trainable_L_rho,
    ).to(device)

    disc = Discriminator(
        n_assets=n_assets, d_cov=d_cov, hidden_dim=args.hidden_dim,
        num_blocks=max(4, args.num_blocks),
    ).to(device)

    # ── Warm start ───────────────────────────────────────────────────
    if args.warm_start is not None:
        print(f"  [warm_start] loading G/D from {args.warm_start}")
        ws = torch.load(args.warm_start, map_location=device, weights_only=False)
        # strict=False so a new L_rho init (from L_eps) is preserved
        gen_incompat = gen.load_state_dict(ws["G"], strict=False)
        if args.trainable_L_rho and "L_rho" in dict(gen.named_parameters()):
            # Warm-start ckpt's L_rho (if buffer) may have overwritten fresh init.
            # Restore sample-cov init from L_eps so trainable_L_rho starts
            # from the corrected residual structure, not the legacy shrunk one.
            if L_eps is not None:
                with torch.no_grad():
                    gen.L_rho.copy_(torch.tensor(L_eps, dtype=torch.float32, device=device))
                    print(f"  [warm_start] restored L_rho to L_eps (sample-cov init)")
        print(f"  [warm_start] missing={len(gen_incompat.missing_keys)} "
              f"unexpected={len(gen_incompat.unexpected_keys)}")
        if "D" in ws:
            disc.load_state_dict(ws["D"], strict=False)

    # ── Optimizer: separate param group for L_rho so it can be (a) frozen
    # during an initial warmup, and (b) updated with a much smaller lr to
    # avoid D-driven drift away from the sample-cov prior.
    if args.trainable_L_rho:
        L_rho_params = [p for n, p in gen.named_parameters() if n == "L_rho"]
        other_params = [p for n, p in gen.named_parameters() if n != "L_rho"]
        opt_G = optim.Adam([
            {"params": other_params, "lr": args.lr, "name": "G_other"},
            {"params": L_rho_params, "lr": 0.0,      "name": "L_rho"},
        ], betas=(0.0, 0.9))
        print(f"  L_rho: trainable, freeze_epochs={args.L_rho_freeze_epochs}, "
              f"unfrozen lr={args.L_rho_lr}")
    else:
        opt_G = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_D = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.0, 0.9))

    sf_loss = StylizedFactLoss(class_indices=class_indices)

    print(f"  G params: {sum(p.numel() for p in gen.parameters()):,}")
    print(f"  D params: {sum(p.numel() for p in disc.parameters()):,}")

    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)
    history = {"d_loss": [], "g_loss": [], "sf": [], "regime_aux": [],
               "z_mean": [], "z_std": [], "val_frob": []}
    best_frob = float("inf"); no_improve = 0

    # ── Training loop ───────────────────────────────────────────────
    for epoch in range(args.epochs):
        gen.train(); disc.train()
        d_losses, g_losses, sf_vals, aux_vals, z_means, z_stds = [], [], [], [], [], []

        # L_rho schedule: freeze at 0 lr for warmup, then switch to L_rho_lr
        if args.trainable_L_rho:
            for pg in opt_G.param_groups:
                if pg.get("name") == "L_rho":
                    pg["lr"] = 0.0 if epoch < args.L_rho_freeze_epochs else args.L_rho_lr

        use_sf = epoch >= args.sf_warmup
        for step in range(args.steps_per_epoch):
            batch = train_ds.sample_batch(args.batch_size, device)
            r_real = batch["returns"]
            B = r_real.shape[0]
            T_full = rfs + t_l + 1
            z = torch.randn(B, gen.d_z, T_full, device=device)
            cov = batch["covariates"]
            if cov.shape[2] < T_full:
                cov = F.pad(cov, (0, T_full - cov.shape[2]), mode="replicate")
            cov_d = batch["covariates"][:, :, -t_l:]

            # D step
            with torch.no_grad():
                r_fake = gen(z, cov[:, :, :T_full], batch["alpha_hat"],
                             batch["beta_hat"], batch["sigma_hat"], batch["factors"])
            d_real = disc(r_real, cov_d)
            d_fake = disc(r_fake, cov_d)
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()
            d_losses.append(d_loss.item())

            # G step
            z2 = torch.randn(B, gen.d_z, T_full, device=device)
            r_fake_g, z_reg = gen(z2, cov[:, :, :T_full], batch["alpha_hat"],
                                   batch["beta_hat"], batch["sigma_hat"],
                                   batch["factors"], return_regime=True)
            g_adv = -disc(r_fake_g, cov_d).mean()

            # SF loss
            if use_sf:
                sf_out = sf_loss(r_real.permute(0, 2, 1), r_fake_g.permute(0, 2, 1))
                g_sf = sf_out["total"]
                ramp = min(1.0, (epoch - args.sf_warmup) / 20.0)
                sf_vals.append(g_sf.item())
            else:
                g_sf = torch.zeros((), device=device)
                ramp = 0.0
                sf_vals.append(0.0)

            # Regime auxiliary supervision: the batch-level stress label comes
            # from the same time window as the target return. We use the
            # covariates' VIX z-score > threshold as the label per timestep.
            stress_target = (cov_d[:, vix_idx, :] > args.vix_threshold).float()
            # z_reg: (B, T_L); stress_target: (B, T_L)
            # Clamp z_reg for numerical stability before BCE
            z_clamped = torch.clamp(z_reg, 1e-6, 1.0 - 1e-6)
            aux = F.binary_cross_entropy(z_clamped, stress_target)
            aux_vals.append(aux.item())
            z_means.append(z_reg.mean().item())
            z_stds.append(z_reg.std().item())

            g_loss = g_adv + args.sf_lambda * ramp * g_sf + args.regime_aux_weight * aux
            opt_G.zero_grad(); g_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(gen.parameters(), args.grad_clip)
            opt_G.step()
            g_losses.append(g_loss.item())

        # ── Validate ────────────────────────────────────────────────
        # Windows that straddle the train/val boundary include training-period
        # days in the target span; we keep only the val-side days via the
        # ``val_mask`` emitted by ProperValDataset. This avoids optimistic
        # val_frob on early windows (CODEX_REVIEW.md issue #1).
        gen.eval()
        with torch.no_grad():
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(999)
            all_real, all_fake = [], []
            for _ in range(5):
                vb = val_ds.sample_batch(64, device)
                Bv = vb["returns"].shape[0]
                T_full_v = rfs + t_l + 1
                zv = torch.randn(Bv, gen.d_z, T_full_v, device=device)
                cv = vb["covariates"]
                if cv.shape[2] < T_full_v:
                    cv = F.pad(cv, (0, T_full_v - cv.shape[2]), mode="replicate")
                rf = gen(zv, cv[:, :, :T_full_v], vb["alpha_hat"],
                         vb["beta_hat"], vb["sigma_hat"], vb["factors"])
                mask_np = vb["val_mask"].cpu().numpy()             # (B, t_l)
                real_bt = vb["returns"].permute(0, 2, 1).cpu().numpy()  # (B, t_l, N)
                fake_bt = rf.permute(0, 2, 1).cpu().numpy()
                all_real.append(real_bt[mask_np])
                all_fake.append(fake_bt[mask_np])
            torch.random.set_rng_state(rng_state)
            real = np.concatenate(all_real); fake = np.concatenate(all_fake)
            cr = np.nan_to_num(np.corrcoef(real.T), nan=0.0)
            cf = np.nan_to_num(np.corrcoef(fake.T), nan=0.0)
            frob = float(np.linalg.norm(cr - cf, "fro"))

        history["d_loss"].append(float(np.mean(d_losses)))
        history["g_loss"].append(float(np.mean(g_losses)))
        history["sf"].append(float(np.mean(sf_vals)))
        history["regime_aux"].append(float(np.mean(aux_vals)))
        history["z_mean"].append(float(np.mean(z_means)))
        history["z_std"].append(float(np.mean(z_stds)))
        history["val_frob"].append(frob)

        print(f"Ep {epoch+1:3d}/{args.epochs} "
              f"| D={history['d_loss'][-1]:.3f} G={history['g_loss'][-1]:.3f} "
              f"SF={history['sf'][-1]:.4f} aux={history['regime_aux'][-1]:.3f} "
              f"z̄={history['z_mean'][-1]:.3f}±{history['z_std'][-1]:.3f} "
              f"| frob={frob:.4f} (best={best_frob:.4f})")

        if epoch < args.min_save_epoch:
            continue
        if frob < best_frob:
            best_frob = frob; no_improve = 0
            torch.save({"G": gen.state_dict(), "D": disc.state_dict(),
                        "epoch": epoch, "frob": frob,
                        "args": vars(args)},
                       os.path.join(save_path, "best_model.pt"))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stop at epoch {epoch+1}"); break

    # ── Save history + config ───────────────────────────────────────
    with open(os.path.join(save_path, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    config = {
        "arch": "sfmg", "seed": args.seed, "best_frob": best_frob,
        "hidden_dim": args.hidden_dim, "num_blocks": args.num_blocks,
        "residual_rank": args.residual_rank,
        "eta_low": args.eta_low, "eta_high": args.eta_high,
        "eta_sigma_low": args.eta_sigma_low, "eta_sigma_high": args.eta_sigma_high,
        "sf_lambda": args.sf_lambda, "sf_warmup": args.sf_warmup,
        "regime_aux_weight": args.regime_aux_weight, "vix_threshold": args.vix_threshold,
        "lag_factors": args.lag_factors,
        "factors": args.factors,
        "trainable_L_rho": args.trainable_L_rho,
        "L_rho_lr": args.L_rho_lr,
        "L_rho_freeze_epochs": args.L_rho_freeze_epochs,
        "warm_start": args.warm_start,
        "train_end": train_end, "n_factors": n_factors, "d_cov": d_cov,
        "vix_col_index": vix_idx,
    }
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nDone. Best val frob: {best_frob:.4f}")


if __name__ == "__main__":
    main()
