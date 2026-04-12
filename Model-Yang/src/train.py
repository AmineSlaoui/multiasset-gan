"""
SF-MarketGAN: Unified training entry point.

Usage:
  # Single asset class
  python train.py --asset_class stocks --save_dir results/single_stocks

  # Multi-asset, FiLM only
  python train.py --asset_class all --save_dir results/multi_film

  # Multi-asset, FiLM + SF balanced
  python train.py --asset_class all --use_sf --save_dir results/multi_film_sf
"""
import argparse, sys, os, json
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_returns, load_macro, ASSET_CLASSES, train_val_test_split
from factors import build_hierarchical_factors, compute_rolling_coefficients
from macro_processor import MacroProcessor
from model import Generator, Discriminator
from stylized_losses import StylizedFactLoss
from trainer import SequenceDataset, Trainer
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="SF-MarketGAN training")
    # Asset selection
    p.add_argument("--asset_class", default="all",
                   choices=["stocks", "bonds", "commodities", "all"])
    # Model
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--d_z", type=int, default=10)
    p.add_argument("--eta", type=float, default=0.1, help="Correction scale")
    # SF constraints
    p.add_argument("--use_sf", action="store_true", help="Enable SF balanced loss")
    p.add_argument("--sf_tau", type=float, default=0.3, help="SF gradient target ratio")
    p.add_argument("--sf_warmup", type=int, default=30)
    # Training
    p.add_argument("--t_l", type=int, default=252)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--steps_per_epoch", type=int, default=30)
    p.add_argument("--window", type=int, default=126)
    p.add_argument("--seed", type=int, default=42)
    # Output
    p.add_argument("--save_dir", default="results/default")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Asset selection ──
    if args.asset_class == "all":
        selected_classes = ASSET_CLASSES
    else:
        selected_classes = {args.asset_class: ASSET_CLASSES[args.asset_class]}

    selected_assets = []
    for assets in selected_classes.values():
        selected_assets.extend(assets)
    n_assets = len(selected_assets)

    mode = "sf_balanced" if args.use_sf else "macro_film"
    print(f"[{mode}] {args.asset_class} ({n_assets} assets) | device={device} | seed={args.seed}")

    # ── Data ──
    returns_full = load_returns()
    macro = load_macro()
    returns = returns_full[selected_assets]

    # Macro processing
    proc = MacroProcessor(n_pca_components=8, n_regimes=3)
    covariates = proc.fit_transform(macro, returns.index)
    d_cov = covariates.shape[1]

    # Factors (computed from full dataset for consistency)
    gf, cf, af = build_hierarchical_factors(returns_full, macro, ASSET_CLASSES)
    n_factors = gf.shape[1]

    # Rolling OLS on selected assets
    alpha, beta, sigma = compute_rolling_coefficients(
        returns, af, selected_classes, cf, gf, window=args.window)
    beta = beta[:, :, :n_factors]

    # ── Datasets ──
    train_ret, val_ret, _ = train_val_test_split(returns)
    cov_arr = covariates.reindex(returns.index).fillna(0).values
    gf_arr = gf.reindex(returns.index).fillna(0).values
    train_end = returns.index.get_loc(train_ret.index[-1]) + 1
    val_end = returns.index.get_loc(val_ret.index[-1]) + 1
    rfs = 127

    train_ds = SequenceDataset(
        returns.values[:train_end], gf_arr[:train_end], cov_arr[:train_end],
        alpha[:train_end], beta[:train_end], sigma[:train_end],
        rfs=rfs, t_l=args.t_l)
    val_ds = SequenceDataset(
        returns.values[:val_end], gf_arr[:val_end], cov_arr[:val_end],
        alpha[:val_end], beta[:val_end], sigma[:val_end],
        rfs=rfs, t_l=args.t_l)

    print(f"  n_assets={n_assets}, K={n_factors}, d_cov={d_cov}")
    print(f"  Train: {len(train_ds.valid_starts)} windows | Val: {len(val_ds.valid_starts)}")

    # ── Model ──
    gen = Generator(n_assets=n_assets, n_factors=n_factors, d_z=args.d_z,
                    d_cov=d_cov, hidden_dim=args.hidden_dim, eta=args.eta)
    disc = Discriminator(n_assets=n_assets, d_cov=d_cov, hidden_dim=args.hidden_dim)

    # SF loss
    asset_list = returns.columns.tolist()
    class_indices = {c: [asset_list.index(a) for a in v if a in asset_list]
                     for c, v in selected_classes.items()}
    sf_loss = StylizedFactLoss(class_indices=class_indices) if args.use_sf else None

    trainer = Trainer(
        generator=gen, discriminator=disc, device=device,
        sf_loss=sf_loss, class_indices=class_indices,
        mode=mode, lr=args.lr, n_critic=5, lambda_gp=10.0,
        sf_tau=args.sf_tau, sf_warmup=args.sf_warmup, t_l=args.t_l)

    trainer.train(
        train_ds, val_ds, n_epochs=args.epochs, batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch, save_dir=args.save_dir,
        patience=args.patience)

    # Save config
    config = vars(args)
    config.update({"n_assets": n_assets, "n_factors": n_factors, "d_cov": d_cov,
                   "selected_assets": selected_assets, "mode": mode})
    json.dump(config, open(Path(args.save_dir) / "config.json", "w"), indent=2)


if __name__ == "__main__":
    main()
