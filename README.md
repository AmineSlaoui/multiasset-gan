# Multi-Asset GAN — Synthetic Return Generation

Generates realistic synthetic daily log-returns for multi-asset portfolios using WGAN-GP with a Temporal Convolutional Network (TCN) backbone. Two models are implemented, with different factor structure and asset universe.

---

## Models

### 1. FamaFrench Model (`FamaFrenchModel/`)

Model conditioned on **Fama-French 3 factors** and **Welch-Goyal macro variables** (T-bill rate, term spread, default spread, realized variance).

- **Asset universe:** 50 large-cap US equities
- **Factor model:** Rolling OLS (252-day window) on FF3 factors → TCN learns stochastic corrections to OLS estimates
- **Training:** 300 epochs, WGAN-GP with vol penalty, train/val/test split (75/12.5/12.5)
- **Entry point:** `python -m FamaFrenchModel.tcn_marketgan_train`
- **Evaluation:** `notebooks/evaluate_gan.ipynb`
- **Results:** `artifacts/marketgan_tcn/marketgan_famafrench_50assets_run1/`

For more information go to `FamaFrenchModel/FamaFrench.md`

### 2. PCARegime Model (`PCAFactor_Model/`)

Model conditioned on **hierarchical PCA factors** and **8 macro variables**, with a regime encoder that gates correction strength based on market stress.

- **Asset universe:** 60 assets across 3 classes — 20 stocks, 20 bonds, 20 commodities
- **Factor model:** Hierarchical PCA (5 global + 2 per class = 11 factors) → Rolling OLS → TCN corrections
- **Macro conditioning:** 8 Bloomberg variables (CDX spreads, swap rates, LIBOR, CRB index) compressed to 8 PCA components via FiLM modulation
- **Regime encoder:** Maps macro covariates to a stress score z(t) ∈ [0,1] that scales correction magnitude and residual covariance
- **Training:** WGAN-GP, train/val/test split
- **Entry point:** `python PCAFactor_Model/src/train_sfmg.py`
- **Evaluation:** `notebooks/distribution_analysis_pcafactor_model.ipynb`
- **Results:** `PCAFactor_Model/results/sfmg_sector_seed42/`

---

## Setup

### 1. Install Anaconda or Miniconda
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### 2. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate gan-project
```

### 3. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name gan-project --display-name "gan-project"
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

---

## Data Pipeline

```
notebooks/extract_stocks.ipynb          → data/stocks/raw_<ticker>.csv         (50 stocks)
notebooks/extract_bonds.ipynb           → data/bonds/raw_<ticker>.csv           (20 bond ETFs)
notebooks/extract_commodities.ipynb     → data/commodities/raw_<name>.csv       (20 commodities)
notebooks/process_data.ipynb            → data/stocks_log_returns.csv           (FamaFrench model)
PCAFactor_Model/src/preprocess_data.py  → PCAFactor_Model/data/                 (PCARegime model)
notebooks/extract_macro_covariates.ipynb → data/macro/welch_goyal_features.csv
```

All data covers **January 2011 – December 2025** sourced from Yahoo Finance and FRED.

---

## Results

| Model | Checkpoints | Generated Returns | Plots |
|---|---|---|---|
| FamaFrench | `artifacts/marketgan_tcn/marketgan_famafrench_50assets_run1/checkpoints/` | `artifacts/marketgan_tcn/marketgan_famafrench_50assets_run1/generated_returns_*.csv` | `artifacts/marketgan_tcn/marketgan_famafrench_50assets_run1/eval_*.png` |
| PCARegime | `PCAFactor_Model/results/sfmg_sector_seed42/best_model.pt` | `PCAFactor_Model/data/generated_returns_test.csv` | `PCAFactor_Model/figures/` |

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/extract_stocks.ipynb` | Download 50 stock price series |
| `notebooks/extract_bonds.ipynb` | Download 20 bond ETF price series |
| `notebooks/extract_commodities.ipynb` | Download 20 commodity futures series |
| `notebooks/extract_macro_covariates.ipynb` | Build Welch-Goyal macro features |
| `notebooks/process_data.ipynb` | Build log returns CSV for FamaFrench model |
| `notebooks/evaluate_gan.ipynb` | Full evaluation of FamaFrench model (all splits) |
| `notebooks/distribution_analysis_famafrench_model.ipynb` | Return distribution comparison — FamaFrench |
| `notebooks/distribution_analysis_pcafactor_model.ipynb` | Joint distribution analysis — PCARegime |