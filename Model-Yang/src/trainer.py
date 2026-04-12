"""
SF-MarketGAN Trainer (v3).

Key improvement: EMA-based loss-magnitude balancing for SF losses.
Different from MacroFactor-GAN's dual-backward-pass gradient scaling.

Our approach: Track EMA of |L_adv| / |L_sf| ratio, scale SF loss
dynamically so it contributes ~τ fraction of total loss magnitude.
Single backward pass (more efficient than dual-pass).

Note: This scales loss magnitudes, not gradient norms directly.
The effect is similar (scaling loss scales its gradient proportionally)
but the mechanism is simpler and avoids a second backward pass.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict

from model import Generator, Discriminator, compute_gradient_penalty
from stylized_losses import StylizedFactLoss


class SequenceDataset:
    """Windowed sequence dataset."""
    def __init__(self, returns, factors, covariates,
                 alpha_hat, beta_hat, sigma_hat, rfs=127, t_l=252):
        self.returns = torch.tensor(returns, dtype=torch.float32)
        self.factors = torch.tensor(factors, dtype=torch.float32)
        self.covariates = torch.tensor(covariates, dtype=torch.float32)
        self.alpha_hat = torch.tensor(alpha_hat, dtype=torch.float32)
        self.beta_hat = torch.tensor(beta_hat, dtype=torch.float32)
        self.sigma_hat = torch.tensor(sigma_hat, dtype=torch.float32)
        self.T, self.N = returns.shape
        self.rfs = rfs
        self.t_l = t_l
        self.t_full = rfs + t_l + 1
        self.valid_starts = list(range(0, self.T - self.t_full))

    def sample_batch(self, batch_size, device):
        indices = np.random.choice(self.valid_starts, size=batch_size, replace=True)
        cov_list, ret_list = [], []
        alpha_list, beta_list, sigma_list, factor_list = [], [], [], []
        for idx in indices:
            e = idx + self.t_full
            cov_list.append(self.covariates[idx:e])
            ts = idx + self.rfs
            te = ts + self.t_l
            ret_list.append(self.returns[ts:te])
            factor_list.append(self.factors[ts:te])
            alpha_list.append(self.alpha_hat[ts:te])
            beta_list.append(self.beta_hat[ts:te])
            sigma_list.append(self.sigma_hat[ts:te])
        return {
            "covariates": torch.stack(cov_list).permute(0, 2, 1).to(device),
            "returns": torch.stack(ret_list).permute(0, 2, 1).to(device),
            "factors": torch.stack(factor_list).permute(0, 2, 1).to(device),
            "alpha_hat": torch.stack(alpha_list).permute(0, 2, 1).to(device),
            "sigma_hat": torch.stack(sigma_list).permute(0, 2, 1).to(device),
            "beta_hat": torch.stack(beta_list).to(device).permute(0, 2, 3, 1),
        }


class EMALossBalancer:
    """EMA-based loss-magnitude balancing for SF losses.

    Tracks the running ratio of |L_adv| / |L_sf| via exponential moving
    average, and dynamically scales L_sf so it contributes ~τ fraction
    of total loss magnitude. Single backward pass.

    Different from MacroFactor-GAN's dual-backward-pass (which scales
    gradient norms directly via two separate backward passes):
    - We scale loss magnitudes → single backward pass, more efficient
    - We use EMA smoothing → more stable than instantaneous ratio
    - We clamp the scale factor to [s_min, s_max] → prevents instability
    """
    def __init__(self, tau=0.3, ema_decay=0.99, s_min=0.01, s_max=10.0):
        self.tau = tau          # Target SF contribution fraction
        self.ema_decay = ema_decay
        self.s_min = s_min
        self.s_max = s_max
        self.ema_adv = None     # EMA of |L_adv|
        self.ema_sf = None      # EMA of |L_sf|

    def scale(self, l_adv: float, l_sf: float) -> float:
        """Compute dynamic scale factor for SF loss.

        Returns s such that: G_loss = L_adv + s * L_sf
        with s chosen so |s * L_sf| / (|L_adv| + |s * L_sf|) ≈ τ
        """
        l_adv_abs = abs(l_adv) + 1e-8
        l_sf_abs = abs(l_sf) + 1e-8

        # Update EMAs
        if self.ema_adv is None:
            self.ema_adv = l_adv_abs
            self.ema_sf = l_sf_abs
        else:
            self.ema_adv = self.ema_decay * self.ema_adv + (1 - self.ema_decay) * l_adv_abs
            self.ema_sf = self.ema_decay * self.ema_sf + (1 - self.ema_decay) * l_sf_abs

        # Target: s * ema_sf / (ema_adv + s * ema_sf) = τ
        # → s = τ * ema_adv / ((1-τ) * ema_sf)
        s = (self.tau / (1 - self.tau)) * (self.ema_adv / (self.ema_sf + 1e-8))
        s = max(self.s_min, min(self.s_max, s))
        return s


class Trainer:
    """Unified trainer with EMA loss-magnitude balancing."""

    def __init__(self, generator, discriminator, device,
                 sf_loss=None, class_indices=None,
                 mode="macro_only",
                 lr=5e-4, n_critic=5, lambda_gp=10.0,
                 sf_tau=0.3, sf_warmup=30, t_l=252):
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.sf_loss = sf_loss
        self.class_indices = class_indices or {}
        self.mode = mode
        self.device = device
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.sf_warmup = sf_warmup
        self.t_l = t_l

        # EMA loss balancer (our differentiator vs MacroFactor-GAN)
        self.loss_balancer = EMALossBalancer(tau=sf_tau) if sf_loss else None

        self.opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(0.0, 0.9))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(0.0, 0.9))
        self.history = {"d_loss": [], "g_loss": [], "sf": [], "sf_scale": [], "val_frob": []}

    def _generate(self, batch):
        B = batch["factors"].shape[0]
        T_full = self.G.rfs + self.t_l + 1
        z = torch.randn(B, self.G.d_z, T_full, device=self.device)
        cov = batch["covariates"]
        if cov.shape[2] < T_full:
            cov = F.pad(cov, (0, T_full - cov.shape[2]), mode="replicate")
        return self.G(z, cov[:, :, :T_full],
                      batch["alpha_hat"], batch["beta_hat"],
                      batch["sigma_hat"], batch["factors"])

    def train_epoch(self, dataset, batch_size, steps, epoch):
        self.G.train(); self.D.train()
        d_losses, g_losses, sf_vals, sf_scales = [], [], [], []
        use_sf = (self.mode == "sf_balanced" and self.sf_loss is not None
                  and epoch >= self.sf_warmup)

        for step in range(steps):
            batch = dataset.sample_batch(batch_size, self.device)
            r_real = batch["returns"]
            cov_d = batch["covariates"][:, :, -self.t_l:]

            # ── Discriminator ──
            for _ in range(self.n_critic):
                with torch.no_grad():
                    r_fake = self._generate(batch)
                d_real = self.D(r_real, cov_d).mean()
                d_fake = self.D(r_fake, cov_d).mean()
                gp = compute_gradient_penalty(self.D, r_real, r_fake, cov_d, self.lambda_gp)
                d_loss = d_fake - d_real + gp
                self.opt_D.zero_grad()
                d_loss.backward()
                self.opt_D.step()
            d_losses.append(d_loss.item())

            # ── Generator ──
            r_fake_g = self._generate(batch)
            g_adv = -self.D(r_fake_g, cov_d).mean()

            if use_sf:
                r_real_t = r_real.permute(0, 2, 1)  # (B, T_L, N)
                r_fake_t = r_fake_g.permute(0, 2, 1)
                sf = self.sf_loss(r_real_t, r_fake_t)
                g_sf = sf["total"]

                # EMA-based adaptive scaling (single backward pass)
                ramp = min(1.0, (epoch - self.sf_warmup) / 20.0)
                s = self.loss_balancer.scale(g_adv.item(), g_sf.item())
                g_loss = g_adv + s * ramp * g_sf

                sf_vals.append(g_sf.item())
                sf_scales.append(s)
            else:
                g_loss = g_adv
                sf_vals.append(0.0)
                sf_scales.append(0.0)

            self.opt_G.zero_grad()
            g_loss.backward()
            self.opt_G.step()
            g_losses.append(g_loss.item())

        return {
            "d_loss": np.mean(d_losses), "g_loss": np.mean(g_losses),
            "sf": np.mean(sf_vals), "sf_scale": np.mean(sf_scales),
        }

    @torch.no_grad()
    def validate(self, dataset, batch_size=64, n_samples=5):
        self.G.eval()
        # Fix both numpy and torch RNG for deterministic validation
        val_rng_state = np.random.RandomState(999).get_state()
        orig_np = np.random.get_state()
        orig_torch = torch.random.get_rng_state()
        np.random.set_state(val_rng_state)
        torch.manual_seed(999)
        all_real, all_fake = [], []
        for _ in range(n_samples):
            batch = dataset.sample_batch(batch_size, self.device)
            r_fake = self._generate(batch)
            all_real.append(batch["returns"].permute(0, 2, 1).reshape(-1, self.G.n_assets).cpu())
            all_fake.append(r_fake.permute(0, 2, 1).reshape(-1, self.G.n_assets).cpu())
        np.random.set_state(orig_np)
        torch.random.set_rng_state(orig_torch)
        real = torch.cat(all_real).numpy()
        fake = torch.cat(all_fake).numpy()
        return float(np.linalg.norm(np.corrcoef(real.T) - np.corrcoef(fake.T), "fro"))

    def train(self, train_ds, val_ds, n_epochs=300, batch_size=128,
              steps_per_epoch=30, save_dir="results", patience=50,
              finetune_epochs=10, finetune_lr_factor=0.1):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        best_frob = float("inf")
        no_improve = 0

        print(f"Training [{self.mode}] | T_L={self.t_l} | lr={self.opt_G.param_groups[0]['lr']}")
        print(f"  G: {sum(p.numel() for p in self.G.parameters()):,} params")
        print(f"  D: {sum(p.numel() for p in self.D.parameters()):,} params")
        if self.mode == "sf_balanced":
            print(f"  SF: EMA loss balancing (τ={self.loss_balancer.tau}), warmup={self.sf_warmup}")
        print("-" * 70)

        for epoch in range(n_epochs):
            t0 = time.time()
            m = self.train_epoch(train_ds, batch_size, steps_per_epoch, epoch)
            frob = self.validate(val_ds)
            dt = time.time() - t0

            self.history["d_loss"].append(m["d_loss"])
            self.history["g_loss"].append(m["g_loss"])
            self.history["sf"].append(m["sf"])
            self.history["sf_scale"].append(m["sf_scale"])
            self.history["val_frob"].append(frob)

            sf_str = f" SF={m['sf']:.4f}(s={m['sf_scale']:.2f})" if m["sf"] > 0 else ""
            print(f"Epoch {epoch+1:3d}/{n_epochs} | D={m['d_loss']:.4f} "
                  f"G={m['g_loss']:.4f}{sf_str} | frob={frob:.4f} "
                  f"(best={best_frob:.4f}) | {dt:.1f}s")

            if frob < best_frob:
                best_frob = frob
                no_improve = 0
                torch.save({"G": self.G.state_dict(), "D": self.D.state_dict(),
                            "epoch": epoch, "frob": frob},
                           save_path / "best_model.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Fine-tune
        print(f"\nFine-tuning {finetune_epochs} epochs...")
        for pg in self.opt_G.param_groups + self.opt_D.param_groups:
            pg["lr"] *= finetune_lr_factor
        for ep in range(finetune_epochs):
            m = self.train_epoch(val_ds, batch_size, steps_per_epoch, n_epochs + ep)
            print(f"  FT {ep+1}/{finetune_epochs} | D={m['d_loss']:.4f} G={m['g_loss']:.4f}")

        torch.save({"G": self.G.state_dict(), "D": self.D.state_dict(),
                    "history": self.history}, save_path / "final_model.pt")
        with open(save_path / "history.json", "w") as f:
            json.dump({k: [float(v) for v in vs] for k, vs in self.history.items()}, f)

        print(f"\nDone. Best val_frob={best_frob:.4f}")
        return self.history
