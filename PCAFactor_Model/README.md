# SF-MarketGAN-R

A factor-structured, regime-conditional GAN for multi-asset synthetic financial data.

On 60 assets (20 bonds + 20 commodities + 20 stocks), SF-MarketGAN-R improves out-of-sample correlation Frobenius norm by 23.3% over factor bootstrap and 17.4% over FB+shrunk (strictly lagged factors, sector-augmented + Gram–Schmidt representation). On six same-sector pair-correlation benchmarks it matches the best classical method while being the only method below both factor-bootstrap baselines on every structural stylized-fact metric (ACF, volatility clustering, leverage effect).

Paper: [`paper/main.pdf`](paper/main.pdf).
Code-review audit: [`CODEX_REVIEW.md`](CODEX_REVIEW.md).

## Repository layout

```
paper/         LaTeX source + compiled PDF
src/           Training and evaluation code (renamed from legacy v9/v7d/v3 namespace)
data/          Returns, macro covariates, preprocessed inputs
results/       Result JSONs referenced by the paper (folders keep the legacy
               v9_*/ names for continuity with saved checkpoints; the trained
               architecture is SF-MarketGAN-R throughout)
figures/       Figures used in paper
papers/        Related-work PDFs (literature)
archive/       Superseded code and old results (legacy baseline: sf_marketgan.py
               with GeneratorV7d alias lives at archive/legacy_baseline/)
```

### Code naming map (legacy → current)

| Legacy file | Current name |
|---|---|
| `train_v9.py` | `train_sfmg.py` |
| `eval_v9.py` | `eval_sfmg.py` |
| `eval_portfolio_v9.py` | `eval_portfolio.py` |
| `eval_counterfactual_v9.py` | `eval_counterfactual.py` |
| `plot_v3_results.py` | `plot_sector_results.py` |
| `sf_marketgan_v9.py` | `sfmg_generator.py` (class `SFMGGenerator`) |
| `sf_marketgan.py` | `sfmg_baseline.py` |
| `factors_v2.py` | `factors_pca.py` (11-factor hierarchical PCA) |
| `factors_v3.py` | `factors_sector.py` (16-factor sector + GS) |
| CLI `--factors_version {v2,v3}` | `--factors {pca,sector}` |
| CLI `--v9_ckpt(s)`, `--v7d_ckpt` | `--sfmg_ckpt(s)`, `--baseline_ckpt` |

Existing `config.json` files that still carry the legacy `"factors_version": "v2"/"v3"` key are loaded transparently by `eval_sfmg.load_sfmg` — they map to `"pca"/"sector"`.

## Reproduce the headline numbers

### 1. Main 60-asset OOS result (single seed, 2025 test year)

```bash
# 1a. Preprocess once (PCA factors + rolling OLS + shrunk residual).
python src/preprocess_data.py --train_end 2023-12-31 \
    --out data/preprocessed

# 1b. Train a single sector-spec seed (~20 min on a 5090).
python src/train_sfmg.py --save_dir results/sfmg_sector_seed42 --seed 42 \
    --factors sector --lag_factors \
    --hidden_dim 256 --num_blocks 4 --epochs 200 --patience 40 \
    --sf_lambda 20.0 --regime_aux_weight 1.0 --residual_rank 4

# 1c. Evaluate OOS on 2025 (~3 min on CPU).
python src/eval_sfmg.py --device cuda --n_paths 100 \
    --sfmg_ckpts results/sfmg_sector_seed42/best_model.pt \
    --out results/oos_headline.json
```

### 2. Same-sector pair-correlation benchmark (Table 2 / Figure 2)

```bash
python src/plot_sector_results.py  # writes figures/pair_corr_sector_vs_prior.png
                                    #        figures/corr_matrix_sector_vs_real.png
```

### 3. Rolling GMV portfolio backtest (§4.6, walk-forward)

```bash
python src/eval_portfolio_rolling.py \
    --ckpts results/v9_roll2022/best_model.pt \
            results/v9_roll2023/best_model.pt \
            results/v9_s42/best_model.pt \
    --test_years 2023 2024 2025 \
    --out results/portfolio_rolling.json
```

Each rebalance generates paths conditioned only on information known at that date — macro/factor up to `rp-1`, rolling-OLS coefficients frozen at `rp-1`, flat-extrapolated conditioning over the holding period. The pre-fix version conditioned on realised future values (see `CODEX_REVIEW.md` item 2).

### 4. Perturbed-forecast mean-variance simulation (Appendix H)

```bash
python src/eval_perturbed_forecast.py \
    --sfmg_ckpt results/v9_s42/best_model.pt \
    --baseline_ckpt results/v7d/best_model.pt \
    --out results/perturbed_forecast.json
```

Uses the same walk-forward generator as the rolling portfolio backtest.

### 5. MarketGAN-60 reproduction (Appendix B)

```bash
python src/train_marketgan_60asset.py --lr 1e-4 --n_critic 5 \
    --save_dir results/marketgan60_lr1e-4_nc5 --seed 42 --lag_factors
python src/eval_marketgan60.py --ckpts results/marketgan60_*/best_model.pt \
    --out results/marketgan60_oos.json
```

## Paper build

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Requirements

- Python 3.12, PyTorch ≥ 2.6 with CUDA
- For the DCC-MLE comparison (Appendix H): R 4.x with the `rmgarch` package
