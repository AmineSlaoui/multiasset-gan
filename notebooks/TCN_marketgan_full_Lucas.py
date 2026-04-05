# %% [markdown]
# # MarketGAN TCN
#
# Implementation complete du TCN de MarketGAN directement dans ce notebook, sans lancer l'entrainement pour l'instant.
#
# Le notebook contient:
# - la config du modele suivant le papier `MarketGANs.pdf`;
# - le pipeline de preparation des donnees;
# - l'estimation rolling de `alpha_hat`, `beta_hat`, `sigma_hat`;
# - le backbone TCN causal, le generateur, le critique et l'utilitaire d'entrainement WGAN-GP;
# - un smoke test de shapes seulement, sans training.

# %% [markdown]
# ## Hyperparametres repris du papier
#
# - TCN partage des coefficients: `kernel_size=2`, `dilation base=2`, `6` blocs, `80` canaux, `latent_dim=10`.
# - Champ receptif du generateur: `127`.
# - Generateur des residus: TCN de type pointwise avec `kernel_size=1`.
# - Critique: TCN avec `160` canaux.
# - Procedure d'entrainement prevue: `WGAN-GP`.
#
# Par defaut, si aucun fichier de facteurs n'est fourni, le notebook construit un facteur proxy en prenant la moyenne cross-sectionnelle des rendements. Pour coller strictement au papier, remplace ce proxy par tes facteurs observes quand on branchera le training.

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import weight_norm


def resolve_repo_root(start: Optional[Path] = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / 'MarketGANs.pdf').exists() or (candidate / '.git').exists():
            return candidate
    return current


REPO_ROOT = resolve_repo_root()
DATA_ROOT = REPO_ROOT / 'data'


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


@dataclass
class ZScoreScaler:
    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, frame: pd.DataFrame, eps: float = 1e-8) -> "ZScoreScaler":
        std = frame.std().replace(0.0, eps).fillna(eps)
        return cls(mean=frame.mean(), std=std)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return (frame - self.mean) / self.std

    def inverse_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame * self.std + self.mean


@dataclass
class ModelDimensions:
    num_assets: int
    num_factors: int
    num_covariates: int


@dataclass
class MarketGANConfig:
    returns_path: Path = DATA_ROOT / "all_assets_log_returns.csv"
    macro_path: Path = DATA_ROOT / "macro" / "macro_conditioning_features.csv"
    factor_path: Optional[Path] = None
    rolling_window: int = 252
    generator_blocks: int = 6
    generator_channels: int = 80
    generator_kernel_size: int = 2
    generator_dilation_base: int = 2
    generator_dropout: float = 0.10
    residual_blocks: int = 1
    residual_channels: int = 80
    residual_kernel_size: int = 1
    residual_dilation_base: int = 1
    residual_dropout: float = 0.00
    critic_blocks: int = 6
    critic_channels: int = 160
    critic_kernel_size: int = 2
    critic_dilation_base: int = 2
    critic_dropout: float = 0.10
    latent_dim: int = 10
    target_horizon: int = 252 * 4
    batch_size: int = 16
    critic_steps: int = 5
    learning_rate_generator: float = 1e-4
    learning_rate_critic: float = 1e-4
    adam_betas: Tuple[float, float] = (0.0, 0.9)
    gradient_penalty_lambda: float = 10.0
    sigma_floor: float = 1e-6
    standardize_covariates: bool = True

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.generator_kernel_size - 1) * sum(
            self.generator_dilation_base ** block for block in range(self.generator_blocks)
        )

    @property
    def warmup_period(self) -> int:
        return self.receptive_field - 1

    @property
    def total_sequence_length(self) -> int:
        return self.target_horizon + self.warmup_period


@dataclass
class PreparedMarketData:
    reference_dates: pd.DatetimeIndex
    target_dates: pd.DatetimeIndex
    asset_columns: list[str]
    factor_columns: list[str]
    covariate_columns: list[str]
    covariates: np.ndarray
    factors: np.ndarray
    real_returns: np.ndarray
    alpha_hat: np.ndarray
    beta_hat: np.ndarray
    sigma_hat: np.ndarray
    scalers: Dict[str, ZScoreScaler]

    @property
    def dimensions(self) -> ModelDimensions:
        return ModelDimensions(
            num_assets=len(self.asset_columns),
            num_factors=len(self.factor_columns),
            num_covariates=len(self.covariate_columns),
        )

    def __len__(self) -> int:
        return int(self.real_returns.shape[0])


def load_time_series_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "Date" not in frame.columns:
        raise ValueError(f"Expected a Date column in {path}.")
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").set_index("Date")
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame.apply(pd.to_numeric, errors="coerce")


def align_covariates_to_returns(covariates: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    covariates = covariates.sort_index()
    return covariates.reindex(target_index).ffill().bfill()


def build_market_factor_proxy(returns_frame: pd.DataFrame, column_name: str = "market_proxy") -> pd.DataFrame:
    return returns_frame.mean(axis=1).rename(column_name).to_frame()


def prepare_factor_frame(
    returns_frame: pd.DataFrame,
    factor_frame: Optional[pd.DataFrame] = None,
    factor_path: Optional[Path] = None,
    factor_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if factor_frame is not None:
        factors = factor_frame.copy()
        if "Date" in factors.columns:
            factors["Date"] = pd.to_datetime(factors["Date"])
            factors = factors.sort_values("Date").set_index("Date")
    elif factor_path is not None:
        factors = load_time_series_csv(Path(factor_path))
    else:
        factors = build_market_factor_proxy(returns_frame)

    factors = factors.apply(pd.to_numeric, errors="coerce")
    factors = factors[~factors.index.duplicated(keep="last")]
    if factor_columns is not None:
        factors = factors.loc[:, list(factor_columns)]
    return factors.sort_index()


def estimate_rolling_factor_coefficients(
    returns_frame: pd.DataFrame,
    factors_frame: pd.DataFrame,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(returns_frame) != len(factors_frame):
        raise ValueError("Returns and factors must be aligned before rolling regression.")
    if len(returns_frame) <= window:
        raise ValueError("Not enough observations for the rolling regression window.")

    returns_values = returns_frame.to_numpy(dtype=np.float64)
    factors_values = factors_frame.to_numpy(dtype=np.float64)
    num_assets = returns_values.shape[1]
    num_factors = factors_values.shape[1]
    num_steps = len(returns_frame) - window

    alpha_hat = np.zeros((num_steps, num_assets), dtype=np.float32)
    beta_hat = np.zeros((num_steps, num_assets, num_factors), dtype=np.float32)
    sigma_hat = np.zeros((num_steps, num_assets), dtype=np.float32)

    for step, end_index in enumerate(range(window - 1, len(returns_frame) - 1)):
        start_index = end_index - window + 1
        factor_window = factors_values[start_index : end_index + 1]
        return_window = returns_values[start_index : end_index + 1]
        design_matrix = np.concatenate(
            [np.ones((window, 1), dtype=np.float64), factor_window],
            axis=1,
        )
        coefficients, _, _, _ = np.linalg.lstsq(design_matrix, return_window, rcond=None)
        fitted = design_matrix @ coefficients
        residuals = return_window - fitted
        degrees_of_freedom = max(window - design_matrix.shape[1], 1)

        alpha_hat[step] = coefficients[0]
        beta_hat[step] = coefficients[1:].T
        sigma_hat[step] = np.sqrt((residuals ** 2).sum(axis=0) / degrees_of_freedom).astype(np.float32)

    return alpha_hat, beta_hat, sigma_hat


def prepare_marketgan_data(
    config: MarketGANConfig,
    factor_frame: Optional[pd.DataFrame] = None,
    factor_columns: Optional[Sequence[str]] = None,
    covariate_columns: Optional[Sequence[str]] = None,
) -> PreparedMarketData:
    returns_frame = load_time_series_csv(config.returns_path).dropna(how="any")
    covariates_frame = load_time_series_csv(config.macro_path)

    if covariate_columns is not None:
        covariates_frame = covariates_frame.loc[:, list(covariate_columns)]

    covariates_frame = align_covariates_to_returns(covariates_frame, returns_frame.index)
    factors_frame = prepare_factor_frame(
        returns_frame=returns_frame,
        factor_frame=factor_frame,
        factor_path=config.factor_path,
        factor_columns=factor_columns,
    )
    factors_frame = factors_frame.reindex(returns_frame.index).ffill().bfill()

    shared_index = returns_frame.index.intersection(covariates_frame.index).intersection(factors_frame.index)
    returns_frame = returns_frame.loc[shared_index].dropna(how="any")
    covariates_frame = covariates_frame.loc[shared_index].dropna(how="any")
    factors_frame = factors_frame.loc[shared_index].dropna(how="any")

    alpha_hat, beta_hat, sigma_hat = estimate_rolling_factor_coefficients(
        returns_frame=returns_frame,
        factors_frame=factors_frame,
        window=config.rolling_window,
    )

    reference_dates = returns_frame.index[config.rolling_window - 1 : -1]
    target_dates = returns_frame.index[config.rolling_window :]
    aligned_covariates = covariates_frame.iloc[config.rolling_window - 1 : -1].copy()
    aligned_factors = factors_frame.iloc[config.rolling_window :].copy()
    aligned_returns = returns_frame.iloc[config.rolling_window :].copy()

    scalers: Dict[str, ZScoreScaler] = {}
    if config.standardize_covariates:
        scalers["covariates"] = ZScoreScaler.fit(aligned_covariates)
        aligned_covariates = scalers["covariates"].transform(aligned_covariates)

    return PreparedMarketData(
        reference_dates=reference_dates,
        target_dates=target_dates,
        asset_columns=list(aligned_returns.columns),
        factor_columns=list(aligned_factors.columns),
        covariate_columns=list(aligned_covariates.columns),
        covariates=aligned_covariates.to_numpy(dtype=np.float32),
        factors=aligned_factors.to_numpy(dtype=np.float32),
        real_returns=aligned_returns.to_numpy(dtype=np.float32),
        alpha_hat=alpha_hat.astype(np.float32),
        beta_hat=beta_hat.astype(np.float32),
        sigma_hat=sigma_hat.astype(np.float32),
        scalers=scalers,
    )


class MarketGANSequenceDataset(Dataset):
    def __init__(self, prepared_data: PreparedMarketData, sequence_length: int):
        self.prepared_data = prepared_data
        self.sequence_length = sequence_length
        if len(prepared_data) < sequence_length:
            raise ValueError(f"Need at least {sequence_length} aligned observations, got {len(prepared_data)}.")
        self.window_starts = np.arange(0, len(prepared_data) - sequence_length + 1)

    def __len__(self) -> int:
        return int(len(self.window_starts))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        start = int(self.window_starts[index])
        end = start + self.sequence_length
        selection = slice(start, end)
        return {
            "covariates": torch.tensor(self.prepared_data.covariates[selection], dtype=torch.float32),
            "factors": torch.tensor(self.prepared_data.factors[selection], dtype=torch.float32),
            "real_returns": torch.tensor(self.prepared_data.real_returns[selection], dtype=torch.float32),
            "alpha_hat": torch.tensor(self.prepared_data.alpha_hat[selection], dtype=torch.float32),
            "beta_hat": torch.tensor(self.prepared_data.beta_hat[selection], dtype=torch.float32),
            "sigma_hat": torch.tensor(self.prepared_data.sigma_hat[selection], dtype=torch.float32),
        }


def build_dataloader(prepared_data: PreparedMarketData, config: MarketGANConfig, shuffle: bool = True) -> DataLoader:
    dataset = MarketGANSequenceDataset(prepared_data, sequence_length=config.total_sequence_length)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, drop_last=True)

# %%
def initialize_linear(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        nn.init.normal_(conv.weight, mean=0.0, std=0.01)
        nn.init.zeros_(conv.bias)
        self.conv = weight_norm(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv(x)
        if self.padding > 0:
            output = output[:, :, :-self.padding]
        return output


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.residual_projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            nn.init.normal_(self.residual_projection.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.residual_projection.bias)
        else:
            self.residual_projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_projection(x)
        output = self.dropout(self.activation(self.conv1(x)))
        output = self.dropout(self.activation(self.conv2(output)))
        return self.activation(output + residual)


class TCNBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_blocks: int,
        kernel_size: int,
        dilation_base: int,
        dropout: float,
    ):
        super().__init__()
        blocks = []
        current_channels = in_channels
        for block_index in range(num_blocks):
            dilation = dilation_base ** block_index
            blocks.append(
                TemporalResidualBlock(
                    in_channels=current_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            current_channels = hidden_channels
        self.network = nn.Sequential(*blocks)
        self.out_channels = hidden_channels
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilation_base ** i for i in range(num_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.network(x)
        return x.transpose(1, 2)


class SharedCoefficientGenerator(nn.Module):
    def __init__(self, dimensions: ModelDimensions, config: MarketGANConfig):
        super().__init__()
        self.dimensions = dimensions
        self.config = config
        input_channels = dimensions.num_covariates + config.latent_dim
        self.backbone = TCNBackbone(
            in_channels=input_channels,
            hidden_channels=config.generator_channels,
            num_blocks=config.generator_blocks,
            kernel_size=config.generator_kernel_size,
            dilation_base=config.generator_dilation_base,
            dropout=config.generator_dropout,
        )
        self.alpha_head = nn.Linear(config.generator_channels, dimensions.num_assets)
        self.beta_head = nn.Linear(config.generator_channels, dimensions.num_assets * dimensions.num_factors)
        self.sigma_head = nn.Linear(config.generator_channels, dimensions.num_assets)
        initialize_linear(self.alpha_head)
        initialize_linear(self.beta_head)
        initialize_linear(self.sigma_head)

    def forward(
        self,
        covariates: torch.Tensor,
        alpha_hat: torch.Tensor,
        beta_hat: torch.Tensor,
        sigma_hat: torch.Tensor,
        latent_noise: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.backbone(torch.cat([covariates, latent_noise], dim=-1))
        alpha_delta = self.alpha_head(hidden)
        beta_delta = self.beta_head(hidden).view(
            hidden.size(0),
            hidden.size(1),
            self.dimensions.num_assets,
            self.dimensions.num_factors,
        )
        sigma_delta = self.sigma_head(hidden)

        alpha = alpha_hat * (1.0 + alpha_delta)
        beta = beta_hat * (1.0 + beta_delta)
        sigma = torch.clamp(sigma_hat * (1.0 + sigma_delta), min=self.config.sigma_floor)

        return {
            "hidden": hidden,
            "alpha_delta": alpha_delta,
            "beta_delta": beta_delta,
            "sigma_delta": sigma_delta,
            "alpha": alpha,
            "beta": beta,
            "sigma": sigma,
        }


class ResidualGenerator(nn.Module):
    def __init__(self, dimensions: ModelDimensions, config: MarketGANConfig):
        super().__init__()
        input_channels = dimensions.num_covariates + config.latent_dim
        self.backbone = TCNBackbone(
            in_channels=input_channels,
            hidden_channels=config.residual_channels,
            num_blocks=config.residual_blocks,
            kernel_size=config.residual_kernel_size,
            dilation_base=config.residual_dilation_base,
            dropout=config.residual_dropout,
        )
        self.output_head = nn.Linear(config.residual_channels, dimensions.num_assets)
        initialize_linear(self.output_head)

    def forward(self, covariates: torch.Tensor, latent_noise: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(torch.cat([covariates, latent_noise], dim=-1))
        return self.output_head(hidden)


class MarketGenerator(nn.Module):
    def __init__(self, dimensions: ModelDimensions, config: MarketGANConfig):
        super().__init__()
        self.dimensions = dimensions
        self.config = config
        self.coefficient_generator = SharedCoefficientGenerator(dimensions, config)
        self.residual_generator = ResidualGenerator(dimensions, config)

    def sample_latent(
        self,
        batch_size: int,
        sequence_length: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or next(self.parameters()).device
        z_coefficients = torch.randn(batch_size, sequence_length, self.config.latent_dim, device=device)
        z_residuals = torch.randn(batch_size, sequence_length, self.config.latent_dim, device=device)
        return z_coefficients, z_residuals

    def trim_sequence(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.config.warmup_period <= 0:
            return tensor
        return tensor[:, self.config.warmup_period :, ...]

    def compose_returns(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        sigma: torch.Tensor,
        factors: torch.Tensor,
        residuals: torch.Tensor,
    ) -> torch.Tensor:
        systematic = alpha + torch.einsum("blnk,blk->bln", beta, factors)
        return systematic + sigma * residuals

    def forward(
        self,
        covariates: torch.Tensor,
        factor_returns: torch.Tensor,
        alpha_hat: torch.Tensor,
        beta_hat: torch.Tensor,
        sigma_hat: torch.Tensor,
        z_coefficients: Optional[torch.Tensor] = None,
        z_residuals: Optional[torch.Tensor] = None,
        trim_output: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size, sequence_length, _ = covariates.shape
        if z_coefficients is None or z_residuals is None:
            sampled_coefficients, sampled_residuals = self.sample_latent(
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=covariates.device,
            )
            if z_coefficients is None:
                z_coefficients = sampled_coefficients
            if z_residuals is None:
                z_residuals = sampled_residuals

        coefficient_outputs = self.coefficient_generator(
            covariates=covariates,
            alpha_hat=alpha_hat,
            beta_hat=beta_hat,
            sigma_hat=sigma_hat,
            latent_noise=z_coefficients,
        )
        residuals = self.residual_generator(covariates=covariates, latent_noise=z_residuals)
        generated_returns = self.compose_returns(
            alpha=coefficient_outputs["alpha"],
            beta=coefficient_outputs["beta"],
            sigma=coefficient_outputs["sigma"],
            factors=factor_returns,
            residuals=residuals,
        )

        outputs = {
            **coefficient_outputs,
            "residuals": residuals,
            "generated_returns": generated_returns,
        }
        if trim_output:
            outputs["alpha_trimmed"] = self.trim_sequence(outputs["alpha"])
            outputs["beta_trimmed"] = self.trim_sequence(outputs["beta"])
            outputs["sigma_trimmed"] = self.trim_sequence(outputs["sigma"])
            outputs["residuals_trimmed"] = self.trim_sequence(outputs["residuals"])
            outputs["generated_returns_trimmed"] = self.trim_sequence(outputs["generated_returns"])
        return outputs


class MarketCritic(nn.Module):
    def __init__(self, dimensions: ModelDimensions, config: MarketGANConfig):
        super().__init__()
        self.config = config
        self.backbone = TCNBackbone(
            in_channels=dimensions.num_assets + dimensions.num_covariates,
            hidden_channels=config.critic_channels,
            num_blocks=config.critic_blocks,
            kernel_size=config.critic_kernel_size,
            dilation_base=config.critic_dilation_base,
            dropout=config.critic_dropout,
        )
        self.output_head = nn.Linear(config.critic_channels, 1)
        initialize_linear(self.output_head)

    def forward(self, returns: torch.Tensor, covariates: torch.Tensor, trim_warmup: bool = True) -> torch.Tensor:
        hidden = self.backbone(torch.cat([returns, covariates], dim=-1))
        if trim_warmup and self.config.warmup_period > 0:
            hidden = hidden[:, self.config.warmup_period :, :]
        scores = self.output_head(hidden).squeeze(-1)
        return scores.mean(dim=1)

# %%
def gradient_penalty(
    critic: MarketCritic,
    real_returns: torch.Tensor,
    fake_returns: torch.Tensor,
    covariates: torch.Tensor,
) -> torch.Tensor:
    batch_size = real_returns.size(0)
    epsilon = torch.rand(batch_size, 1, 1, device=real_returns.device)
    interpolated_returns = epsilon * real_returns + (1.0 - epsilon) * fake_returns
    interpolated_returns.requires_grad_(True)
    interpolated_scores = critic(interpolated_returns, covariates, trim_warmup=True)
    gradients = torch.autograd.grad(
        outputs=interpolated_scores,
        inputs=interpolated_returns,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


class MarketGANTrainer:
    def __init__(self, generator: MarketGenerator, critic: MarketCritic, config: MarketGANConfig):
        self.generator = generator
        self.critic = critic
        self.config = config
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate_generator,
            betas=config.adam_betas,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate_critic,
            betas=config.adam_betas,
        )

    def move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.config.device) for key, value in batch.items()}

    def critic_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        batch = self.move_batch(batch)
        self.critic_optimizer.zero_grad(set_to_none=True)

        generated = self.generator(
            covariates=batch["covariates"],
            factor_returns=batch["factors"],
            alpha_hat=batch["alpha_hat"],
            beta_hat=batch["beta_hat"],
            sigma_hat=batch["sigma_hat"],
            trim_output=False,
        )
        fake_returns = generated["generated_returns"].detach()

        real_scores = self.critic(batch["real_returns"], batch["covariates"], trim_warmup=True)
        fake_scores = self.critic(fake_returns, batch["covariates"], trim_warmup=True)
        gp = gradient_penalty(
            critic=self.critic,
            real_returns=batch["real_returns"],
            fake_returns=fake_returns,
            covariates=batch["covariates"],
        )
        loss = fake_scores.mean() - real_scores.mean() + self.config.gradient_penalty_lambda * gp
        loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": float(loss.detach().cpu()),
            "critic_real_score": float(real_scores.mean().detach().cpu()),
            "critic_fake_score": float(fake_scores.mean().detach().cpu()),
            "gradient_penalty": float(gp.detach().cpu()),
        }

    def generator_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        batch = self.move_batch(batch)
        self.generator_optimizer.zero_grad(set_to_none=True)

        generated = self.generator(
            covariates=batch["covariates"],
            factor_returns=batch["factors"],
            alpha_hat=batch["alpha_hat"],
            beta_hat=batch["beta_hat"],
            sigma_hat=batch["sigma_hat"],
            trim_output=False,
        )
        fake_scores = self.critic(generated["generated_returns"], batch["covariates"], trim_warmup=True)
        loss = -fake_scores.mean()
        loss.backward()
        self.generator_optimizer.step()

        return {
            "generator_loss": float(loss.detach().cpu()),
            "generator_score": float(fake_scores.mean().detach().cpu()),
        }

    def fit(self, dataloader: DataLoader, num_epochs: int, log_every: int = 25) -> pd.DataFrame:
        history = []
        global_step = 0
        for epoch in range(num_epochs):
            for batch in dataloader:
                critic_metrics = {}
                for _ in range(self.config.critic_steps):
                    critic_metrics = self.critic_step(batch)
                generator_metrics = self.generator_step(batch)
                step_metrics = {
                    "epoch": epoch,
                    "step": global_step,
                    **critic_metrics,
                    **generator_metrics,
                }
                history.append(step_metrics)
                if global_step % log_every == 0:
                    print(step_metrics)
                global_step += 1
        return pd.DataFrame(history)

    @torch.no_grad()
    def generate(self, batch: Dict[str, torch.Tensor], trim_output: bool = True) -> Dict[str, torch.Tensor]:
        batch = self.move_batch(batch)
        outputs = self.generator(
            covariates=batch["covariates"],
            factor_returns=batch["factors"],
            alpha_hat=batch["alpha_hat"],
            beta_hat=batch["beta_hat"],
            sigma_hat=batch["sigma_hat"],
            trim_output=trim_output,
        )
        return {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in outputs.items()
        }


def build_training_stack(
    config: MarketGANConfig,
    factor_frame: Optional[pd.DataFrame] = None,
    factor_columns: Optional[Sequence[str]] = None,
    covariate_columns: Optional[Sequence[str]] = None,
) -> Tuple[PreparedMarketData, MarketGANSequenceDataset, DataLoader, MarketGenerator, MarketCritic, MarketGANTrainer]:
    prepared_data = prepare_marketgan_data(
        config=config,
        factor_frame=factor_frame,
        factor_columns=factor_columns,
        covariate_columns=covariate_columns,
    )
    dataset = MarketGANSequenceDataset(prepared_data, sequence_length=config.total_sequence_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    generator = MarketGenerator(prepared_data.dimensions, config).to(config.device)
    critic = MarketCritic(prepared_data.dimensions, config).to(config.device)
    trainer = MarketGANTrainer(generator=generator, critic=critic, config=config)
    return prepared_data, dataset, dataloader, generator, critic, trainer


def run_smoke_test() -> None:
    config = MarketGANConfig()

    print(f"Repo root: {REPO_ROOT}")
    print(f"Generator receptive field: {config.receptive_field}")
    print(f"Warmup trimmed from each sequence: {config.warmup_period}")
    print(f"Total sequence length expected by the trainer: {config.total_sequence_length}")

    dummy_dimensions = ModelDimensions(num_assets=30, num_factors=1, num_covariates=8)
    generator = MarketGenerator(dummy_dimensions, config).to(config.device)
    critic = MarketCritic(dummy_dimensions, config).to(config.device)

    batch_size = 2
    sequence_length = config.total_sequence_length
    dummy_covariates = torch.randn(batch_size, sequence_length, dummy_dimensions.num_covariates, device=config.device)
    dummy_factors = torch.randn(batch_size, sequence_length, dummy_dimensions.num_factors, device=config.device)
    dummy_alpha_hat = torch.randn(batch_size, sequence_length, dummy_dimensions.num_assets, device=config.device) * 0.001
    dummy_beta_hat = torch.randn(
        batch_size,
        sequence_length,
        dummy_dimensions.num_assets,
        dummy_dimensions.num_factors,
        device=config.device,
    ) * 0.05
    dummy_sigma_hat = torch.rand(batch_size, sequence_length, dummy_dimensions.num_assets, device=config.device) * 0.02 + 0.001

    with torch.no_grad():
        generated = generator(
            covariates=dummy_covariates,
            factor_returns=dummy_factors,
            alpha_hat=dummy_alpha_hat,
            beta_hat=dummy_beta_hat,
            sigma_hat=dummy_sigma_hat,
            trim_output=True,
        )
        critic_scores = critic(generated["generated_returns"], dummy_covariates)

    print("Generated returns shape:", tuple(generated["generated_returns"].shape))
    print("Trimmed returns shape:", tuple(generated["generated_returns_trimmed"].shape))
    print("Critic scores shape:", tuple(critic_scores.shape))


if __name__ == "__main__":
    run_smoke_test()

