# GAN Project 

## Requirements

Please download Anaconda or Miniconda so we can have a consolidated package for everyone
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
```

### 2. Activate the environment

```bash
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

## Notebooks

### `notebooks/extract_data.ipynb`
Downloads raw OHLCV data from Yahoo Finance (2017–2026) and saves:

```
data/raw_btc.csv       # Bitcoin (BTC-USD)
data/raw_sp500.csv     # S&P 500 (^GSPC)
data/raw_gold.csv      # Gold Futures (GC=F)
```

### `notebooks/preprocess.ipynb`
Aligns the three assets to a common trading calendar and computes log returns. Run this after `extract_data.ipynb`.

Steps:
1. Resamples BTC to business days (last close of each business day)
2. Inner-joins all three on the S&P 500 (NYSE) trading calendar
3. Computes daily log returns: `ln(P_t / P_{t-1})`

Outputs:
```
data/aligned_prices.csv   # closing prices on shared trading days
data/log_returns.csv      # daily log returns for all three assets
```
