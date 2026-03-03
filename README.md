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
jupyter notebook notebooks/extract_data.ipynb
```

## What it does

Runs `notebooks/extract_data.ipynb`, which downloads raw OHLCV data from Yahoo Finance (2017–2026) and saves three CSV files:

```
data/raw_btc.csv       # Bitcoin (BTC-USD)
data/raw_sp500.csv     # S&P 500 (^GSPC)
data/raw_gold.csv      # Gold Futures (GC=F)
```
