from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

MODULE_FILE = globals().get("__file__")
ROOT_DIR = Path(MODULE_FILE).resolve().parent.parent if MODULE_FILE else Path.cwd()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from FamaFrenchModel.tcn_marketgan_model import (
    REPO_ROOT,
    MarketCritic,
    MarketGANConfig,
    MarketGANSequenceDataset,
    MarketGANTrainer,
    MarketGenerator,
    PreparedMarketData,
    prepare_marketgan_data,
)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


@dataclass
class TrainingRunConfig:
    num_epochs: int = 25
    train_fraction: float = 0.75   # 75% train, 12.5% val, 12.5% test
    val_fraction: float = 0.125    # val used for checkpoint selection + fine-tuning
    # test_fraction = 1 - train_fraction - val_fraction = 0.125 (held out, final eval only)
    num_workers: int = 4
    pin_memory: bool = True
    show_progress: bool = True
    log_every_epoch: int = 1
    checkpoint_every: int = 10
    max_train_batches_per_epoch: Optional[int] = None
    max_validation_batches: Optional[int] = None
    checkpoint_root: Path = REPO_ROOT / "artifacts" / "marketgan_tcn"
    run_name: str = "baseline_proxy_factor"
    resume_checkpoint: Optional[Path] = None
    save_history_csv: bool = True
    plot_history: bool = True
    fine_tune_epochs: int = 50
    fine_tune_lr: float = 1e-4   # 1/10th of main learning rate per paper

    @property
    def run_dir(self) -> Path:
        return self.checkpoint_root / self.run_name

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def generated_dir(self) -> Path:
        return self.run_dir / "generated"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_sequence_dataset(
    dataset: Dataset, train_fraction: float, val_fraction: float
) -> Tuple[Subset, Subset, Subset]:
    """Split dataset chronologically into train / val / test.

    train: first train_fraction of windows  → gradient updates
    val:   next  val_fraction of windows    → checkpoint selection + fine-tuning
    test:  remaining windows                → held-out, final evaluation only
    """
    test_fraction = 1.0 - train_fraction - val_fraction
    if test_fraction <= 0.0:
        raise ValueError("train_fraction + val_fraction must be < 1.0 to leave room for test.")
    total = len(dataset)
    if total < 3:
        raise ValueError("Need at least 3 windows for a train/val/test split.")

    train_end = max(1, int(total * train_fraction))
    val_end   = max(train_end + 1, int(total * (train_fraction + val_fraction)))
    val_end   = min(val_end, total - 1)

    train_indices = list(range(train_end))
    val_indices   = list(range(train_end, val_end))
    test_indices  = list(range(val_end, total))
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def build_data_loaders(
    dataset: Dataset,
    config: MarketGANConfig,
    run_config: TrainingRunConfig,
) -> Tuple[Subset, Subset, Subset, DataLoader, DataLoader, DataLoader]:
    train_dataset, val_dataset, test_dataset = split_sequence_dataset(
        dataset,
        train_fraction=run_config.train_fraction,
        val_fraction=run_config.val_fraction,
    )
    pin_memory = bool(run_config.pin_memory and torch.cuda.is_available())

    def _loader(ds, shuffle, drop_last):
        return DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=run_config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=run_config.num_workers > 0,
        )

    train_loader = _loader(train_dataset, shuffle=True,  drop_last=len(train_dataset) >= config.batch_size)
    val_loader   = _loader(val_dataset,   shuffle=False, drop_last=False)
    test_loader  = _loader(test_dataset,  shuffle=False, drop_last=False)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_full_training_stack(
    config: MarketGANConfig,
    run_config: TrainingRunConfig,
    factor_frame: Optional[pd.DataFrame] = None,
    factor_columns: Optional[Sequence[str]] = None,
    covariate_columns: Optional[Sequence[str]] = None,
):
    prepared_data = prepare_marketgan_data(
        config=config,
        factor_frame=factor_frame,
        factor_columns=factor_columns,
        covariate_columns=covariate_columns,
    )
    dataset = MarketGANSequenceDataset(prepared_data, sequence_length=config.total_sequence_length)
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_data_loaders(
        dataset=dataset,
        config=config,
        run_config=run_config,
    )
    generator = MarketGenerator(prepared_data.dimensions, config).to(config.device)
    critic = MarketCritic(prepared_data.dimensions, config).to(config.device)
    trainer = MarketGANTrainer(generator=generator, critic=critic, config=config)
    return prepared_data, dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, generator, critic, trainer


def _trim_real_returns(batch: Dict[str, torch.Tensor], config: MarketGANConfig) -> torch.Tensor:
    if config.warmup_period <= 0:
        return batch["real_returns"]
    return batch["real_returns"][:, config.warmup_period :, :]


@torch.no_grad()
def evaluate_market_gan(
    trainer: MarketGANTrainer,
    dataloader: DataLoader,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    trainer.generator.eval()
    trainer.critic.eval()
    rows = []

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        batch = trainer.move_batch(batch)
        outputs = trainer.generator(
            covariates=batch["covariates"],
            factor_returns=batch["factors"],
            alpha_hat=batch["alpha_hat"],
            beta_hat=batch["beta_hat"],
            sigma_hat=batch["sigma_hat"],
            trim_output=True,
        )

        fake_full = outputs["generated_returns"]
        fake_trimmed = outputs["generated_returns_trimmed"]
        real_trimmed = _trim_real_returns(batch, trainer.config)

        critic_real = trainer.critic(batch["real_returns"], batch["covariates"], trim_warmup=True)
        critic_fake = trainer.critic(fake_full, batch["covariates"], trim_warmup=True)

        fake_mean = fake_trimmed.mean(dim=(0, 1))
        real_mean = real_trimmed.mean(dim=(0, 1))
        fake_vol = fake_trimmed.var(dim=(0, 1), unbiased=False).sqrt()
        real_vol = real_trimmed.var(dim=(0, 1), unbiased=False).sqrt()

        rows.append(
            {
                "generator_loss": float((-critic_fake.mean()).detach().cpu()),
                "critic_real_score": float(critic_real.mean().detach().cpu()),
                "critic_fake_score": float(critic_fake.mean().detach().cpu()),
                "wasserstein_estimate": float((critic_real.mean() - critic_fake.mean()).detach().cpu()),
                "mean_gap": float((fake_mean - real_mean).abs().mean().detach().cpu()),
                "vol_gap": float((fake_vol - real_vol).abs().mean().detach().cpu()),
            }
        )

    trainer.generator.train()
    trainer.critic.train()

    if not rows:
        return {}
    return pd.DataFrame(rows).mean(numeric_only=True).to_dict()


def _serializable_config_dict(config_object: object) -> Dict[str, object]:
    output = {}
    for key, value in config_object.__dict__.items():
        if isinstance(value, Path):
            output[key] = str(value)
        elif isinstance(value, tuple):
            output[key] = list(value)
        else:
            output[key] = value
    return output


def save_checkpoint(
    trainer: MarketGANTrainer,
    path: Path,
    epoch: int,
    history,
    model_config: MarketGANConfig,
    run_config: TrainingRunConfig,
    prepared_data: PreparedMarketData,
    extra_state: Optional[Dict[str, object]] = None,
) -> None:
    ensure_directory(path.parent)
    payload = {
        "epoch": epoch,
        "history": list(history),
        "model_config": _serializable_config_dict(model_config),
        "run_config": _serializable_config_dict(run_config),
        "generator_state_dict": trainer.generator.state_dict(),
        "critic_state_dict": trainer.critic.state_dict(),
        "generator_optimizer_state_dict": trainer.generator_optimizer.state_dict(),
        "critic_optimizer_state_dict": trainer.critic_optimizer.state_dict(),
        "asset_columns": prepared_data.asset_columns,
        "factor_columns": prepared_data.factor_columns,
        "covariate_columns": prepared_data.covariate_columns,
    }
    if extra_state is not None:
        payload["extra_state"] = extra_state
    torch.save(payload, path)


def load_checkpoint(trainer: MarketGANTrainer, checkpoint_path: Path) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=trainer.config.device)
    trainer.generator.load_state_dict(checkpoint["generator_state_dict"])
    trainer.critic.load_state_dict(checkpoint["critic_state_dict"])
    trainer.generator_optimizer.load_state_dict(checkpoint["generator_optimizer_state_dict"])
    trainer.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    return checkpoint


def plot_training_history(history_frame: pd.DataFrame) -> None:
    if history_frame.empty:
        print("No history to plot.")
        return

    figure, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(history_frame["epoch"], history_frame["train_generator_loss"], label="train")
    axes[0].plot(history_frame["epoch"], history_frame["val_generator_loss"], label="val")
    axes[0].set_title("Generator Loss")
    axes[0].legend()

    axes[1].plot(history_frame["epoch"], history_frame["train_critic_loss"], label="train")
    if "val_wasserstein_estimate" in history_frame.columns:
        axes[1].plot(history_frame["epoch"], history_frame["val_wasserstein_estimate"], label="val wasserstein")
    axes[1].set_title("Critic / Wasserstein")
    axes[1].legend()

    axes[2].plot(history_frame["epoch"], history_frame["val_mean_gap"], label="mean gap")
    axes[2].plot(history_frame["epoch"], history_frame["val_vol_gap"], label="vol gap")
    axes[2].set_title("Validation Gaps")
    axes[2].legend()

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def train_market_gan(
    trainer: MarketGANTrainer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    prepared_data: PreparedMarketData,
    model_config: MarketGANConfig,
    run_config: TrainingRunConfig,
) -> pd.DataFrame:
    ensure_directory(run_config.run_dir)
    ensure_directory(run_config.checkpoints_dir)

    start_epoch = 0
    history = []
    best_validation_loss = math.inf

    if run_config.resume_checkpoint is not None:
        checkpoint = load_checkpoint(trainer, run_config.resume_checkpoint)
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint.get("history", []))
        best_validation_loss = min([row.get("val_generator_loss", math.inf) for row in history] or [math.inf])
        print(f"Resumed from checkpoint at epoch {start_epoch}.")

    for epoch in range(start_epoch, run_config.num_epochs):
        trainer.generator.train()
        trainer.critic.train()
        epoch_rows = []

        progress_bar = train_loader
        if run_config.show_progress:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{run_config.num_epochs}")

        for batch_index, batch in enumerate(progress_bar):
            if run_config.max_train_batches_per_epoch is not None and batch_index >= run_config.max_train_batches_per_epoch:
                break

            critic_metrics = {}
            for _ in range(model_config.critic_steps):
                critic_metrics = trainer.critic_step(batch)
            generator_metrics = trainer.generator_step(batch)
            batch_metrics = {**critic_metrics, **generator_metrics}
            epoch_rows.append(batch_metrics)

            if run_config.show_progress:
                progress_bar.set_postfix(
                    critic_loss=f"{batch_metrics['critic_loss']:.4f}",
                    generator_loss=f"{batch_metrics['generator_loss']:.4f}",
                )

        train_metrics = pd.DataFrame(epoch_rows).mean(numeric_only=True).to_dict()
        validation_metrics = evaluate_market_gan(
            trainer=trainer,
            dataloader=validation_loader,
            max_batches=run_config.max_validation_batches,
        )

        epoch_metrics = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(epoch_metrics)

        save_checkpoint(
            trainer=trainer,
            path=run_config.checkpoints_dir / "latest.pt",
            epoch=epoch,
            history=history,
            model_config=model_config,
            run_config=run_config,
            prepared_data=prepared_data,
        )

        current_validation_loss = epoch_metrics.get("val_vol_gap", math.inf)
        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            save_checkpoint(
                trainer=trainer,
                path=run_config.checkpoints_dir / "best.pt",
                epoch=epoch,
                history=history,
                model_config=model_config,
                run_config=run_config,
                prepared_data=prepared_data,
                extra_state={"best_validation_loss": best_validation_loss},
            )

        if run_config.checkpoint_every > 0 and (epoch + 1) % run_config.checkpoint_every == 0:
            save_checkpoint(
                trainer=trainer,
                path=run_config.checkpoints_dir / f"epoch_{epoch + 1:04d}.pt",
                epoch=epoch,
                history=history,
                model_config=model_config,
                run_config=run_config,
                prepared_data=prepared_data,
            )

        history_frame = pd.DataFrame(history)
        if run_config.save_history_csv:
            history_frame.to_csv(run_config.run_dir / "history.csv", index=False)

        if (epoch + 1) % run_config.log_every_epoch == 0:
            print(f"\n── Epoch {epoch_metrics['epoch'] + 1}/{run_config.num_epochs} ──────────────────────────")
            print(f"  Train │ critic: {epoch_metrics['train_critic_loss']:+.4f}  "
                  f"gen: {epoch_metrics['train_generator_loss']:8.4f}  "
                  f"vol_penalty: {epoch_metrics.get('train_vol_penalty', 0):.2e}  "
                  f"gp: {epoch_metrics['train_gradient_penalty']:.4f}")
            print(f"  Val   │ wasserstein: {epoch_metrics['val_wasserstein_estimate']:.4f}  "
                  f"mean_gap: {epoch_metrics['val_mean_gap']:.6f}  "
                  f"vol_gap: {epoch_metrics['val_vol_gap']:.6f}")

    history_frame = pd.DataFrame(history)
    if run_config.plot_history:
        plot_training_history(history_frame)
    return history_frame


def fine_tune_market_gan(
    trainer: MarketGANTrainer,
    fine_tune_loader: DataLoader,
    prepared_data: PreparedMarketData,
    model_config: MarketGANConfig,
    run_config: TrainingRunConfig,
) -> pd.DataFrame:
    """Fine-tune on validation (most recent) data at 1/10th learning rate.

    Per the MarketGAN paper: after main training, fine-tune on the validation
    set so the model captures recent market dynamics that were excluded from
    the main training loop.
    """
    ensure_directory(run_config.checkpoints_dir)

    # Lower learning rate for both optimizers
    for param_group in trainer.generator_optimizer.param_groups:
        param_group["lr"] = run_config.fine_tune_lr
    for param_group in trainer.critic_optimizer.param_groups:
        param_group["lr"] = run_config.fine_tune_lr

    print(f"\n── Fine-tuning for {run_config.fine_tune_epochs} epochs at lr={run_config.fine_tune_lr} ──")

    history = []
    for epoch in range(run_config.fine_tune_epochs):
        trainer.generator.train()
        trainer.critic.train()
        epoch_rows = []

        progress_bar = fine_tune_loader
        if run_config.show_progress:
            progress_bar = tqdm(fine_tune_loader, desc=f"Fine-tune {epoch + 1}/{run_config.fine_tune_epochs}")

        for batch in progress_bar:
            for _ in range(model_config.critic_steps):
                critic_metrics = trainer.critic_step(batch)
            generator_metrics = trainer.generator_step(batch)
            epoch_rows.append({**critic_metrics, **generator_metrics})

        epoch_metrics = {"epoch": epoch, **pd.DataFrame(epoch_rows).mean(numeric_only=True).to_dict()}
        history.append(epoch_metrics)

        if (epoch + 1) % run_config.log_every_epoch == 0:
            print(f"  Fine-tune {epoch + 1}/{run_config.fine_tune_epochs} │ "
                  f"critic: {epoch_metrics['critic_loss']:+.4f}  "
                  f"gen: {epoch_metrics['generator_loss']:.4f}")

    save_checkpoint(
        trainer=trainer,
        path=run_config.checkpoints_dir / "fine_tuned.pt",
        epoch=run_config.fine_tune_epochs - 1,
        history=history,
        model_config=model_config,
        run_config=run_config,
        prepared_data=prepared_data,
    )
    print(f"Fine-tuned checkpoint saved → {run_config.checkpoints_dir / 'fine_tuned.pt'}")

    history_frame = pd.DataFrame(history)
    if run_config.save_history_csv:
        history_frame.to_csv(run_config.run_dir / "fine_tune_history.csv", index=False)
    return history_frame


def get_window_dates(
    window_index: int,
    prepared_data: PreparedMarketData,
    config: MarketGANConfig,
    trim_output: bool = True,
) -> pd.DatetimeIndex:
    start = window_index
    if trim_output:
        start += config.warmup_period
    length = config.target_horizon if trim_output else config.total_sequence_length
    stop = start + length
    return prepared_data.target_dates[start:stop]


def resolve_window_index(dataset: Dataset, dataset_index: int) -> int:
    if isinstance(dataset, Subset):
        return int(dataset.indices[dataset_index])
    return int(dataset_index)


def sample_dataset_batch(dataset: Dataset, index: int) -> Dict[str, torch.Tensor]:
    sample = dataset[index]
    return {key: value.unsqueeze(0) for key, value in sample.items()}


def outputs_to_dataframes(
    outputs: Dict[str, torch.Tensor],
    prepared_data: PreparedMarketData,
    dates: pd.DatetimeIndex,
    sample_index: int = 0,
    trim_output: bool = True,
) -> Dict[str, pd.DataFrame]:
    returns_key = "generated_returns_trimmed" if trim_output else "generated_returns"
    alpha_key = "alpha_trimmed" if trim_output else "alpha"
    sigma_key = "sigma_trimmed" if trim_output else "sigma"
    beta_key = "beta_trimmed" if trim_output else "beta"

    generated_returns = outputs[returns_key][sample_index].numpy()
    alpha_values = outputs[alpha_key][sample_index].numpy()
    sigma_values = outputs[sigma_key][sample_index].numpy()
    beta_values = outputs[beta_key][sample_index].numpy()

    beta_columns = pd.MultiIndex.from_product(
        [prepared_data.asset_columns, prepared_data.factor_columns],
        names=["asset", "factor"],
    )

    return {
        "generated_returns": pd.DataFrame(generated_returns, index=dates, columns=prepared_data.asset_columns),
        "alpha": pd.DataFrame(alpha_values, index=dates, columns=prepared_data.asset_columns),
        "sigma": pd.DataFrame(sigma_values, index=dates, columns=prepared_data.asset_columns),
        "beta": pd.DataFrame(beta_values.reshape(len(dates), -1), index=dates, columns=beta_columns),
    }


def generate_synthetic_sample(
    trainer: MarketGANTrainer,
    dataset: Dataset,
    prepared_data: PreparedMarketData,
    dataset_index: int = 0,
    trim_output: bool = True,
    export_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    batch = sample_dataset_batch(dataset, dataset_index)
    window_index = resolve_window_index(dataset, dataset_index)
    outputs = trainer.generate(batch, trim_output=trim_output)
    dates = get_window_dates(
        window_index=window_index,
        prepared_data=prepared_data,
        config=trainer.config,
        trim_output=trim_output,
    )
    frames = outputs_to_dataframes(
        outputs=outputs,
        prepared_data=prepared_data,
        dates=dates,
        sample_index=0,
        trim_output=trim_output,
    )

    if export_dir is not None:
        ensure_directory(export_dir)
        frames["generated_returns"].to_csv(export_dir / f"sample_{dataset_index:04d}_returns.csv")
        frames["alpha"].to_csv(export_dir / f"sample_{dataset_index:04d}_alpha.csv")
        frames["sigma"].to_csv(export_dir / f"sample_{dataset_index:04d}_sigma.csv")
        frames["beta"].to_csv(export_dir / f"sample_{dataset_index:04d}_beta.csv")

    return frames


if __name__ == "__main__":
    RUN_TRAINING = True
    RUN_GENERATION_EXAMPLE = False
    

    model_config = MarketGANConfig(
        batch_size=128,
        target_horizon=252 * 4,
        learning_rate_generator=1e-4,
        learning_rate_critic=1e-4,
        critic_steps=5,
    )
    run_config = TrainingRunConfig(
        num_epochs=300,
        train_fraction=0.75,    # 75% train
        val_fraction=0.125,     # 12.5% val  → checkpoint selection + fine-tuning
        # test = remaining 12.5% → held-out, final evaluation only
        checkpoint_every=10,
        run_name="marketgan_famafrench_50assets_run1",
    )

    print("Device:", model_config.device)

    # ── Fama-French 3 factors (Mkt-RF, SMB, HML) loaded from factor_path ──────
    # factor_frame=None → model loads from config.factor_path = data/factors/fama_french_daily.csv
    prepared_data, full_dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, generator, critic, trainer = build_full_training_stack(
        config=model_config,
        run_config=run_config,
        factor_frame=None,
    )

    print("Assets     :", prepared_data.dimensions.num_assets)
    print("Factors    :", prepared_data.dimensions.num_factors)
    print("Covariates :", prepared_data.dimensions.num_covariates)
    print("Full windows :", len(full_dataset))
    print("Train windows:", len(train_dataset))
    print("Val windows  :", len(val_dataset))
    print("Test windows :", len(test_dataset), " ← held-out, not used until final eval")
    print("Generator parameters:", f"{count_parameters(generator):,}")
    print("Critic parameters   :", f"{count_parameters(critic):,}")
    print("Run directory:", run_config.run_dir)

    if RUN_TRAINING:
        history_frame = train_market_gan(
            trainer=trainer,
            train_loader=train_loader,
            validation_loader=val_loader,
            prepared_data=prepared_data,
            model_config=model_config,
            run_config=run_config,
        )
        # Load best checkpoint then fine-tune on val (most recent training data)
        best_checkpoint = run_config.checkpoints_dir / "best.pt"
        if best_checkpoint.exists():
            load_checkpoint(trainer, best_checkpoint)
            print(f"Loaded best checkpoint for fine-tuning: {best_checkpoint}")
        fine_tune_market_gan(
            trainer=trainer,
            fine_tune_loader=val_loader,
            prepared_data=prepared_data,
            model_config=model_config,
            run_config=run_config,
        )
    else:
        print("Set RUN_TRAINING = True pour lancer le training complet.")

    if RUN_GENERATION_EXAMPLE:
        best_checkpoint = run_config.checkpoints_dir / "best.pt"
        if best_checkpoint.exists():
            load_checkpoint(trainer, best_checkpoint)
        sample_frames = generate_synthetic_sample(
            trainer=trainer,
            dataset=val_dataset,
            prepared_data=prepared_data,
            dataset_index=0,
            trim_output=True,
            export_dir=run_config.generated_dir,
        )
        print(sample_frames["generated_returns"].head())
    else:
        print("Set RUN_GENERATION_EXAMPLE = True pour charger le meilleur checkpoint et exporter un sample synthetique.")
