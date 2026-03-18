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


### `notebooks/extract_stocks.ipynb`
Downloads daily OHLCV data for 10 stocks from **WRDS** (CRSP Daily Stock File) covering January 1, 2011 to January 1, 2026.

**Requires a WRDS account.** You will be prompted for your WRDS username and password when running the notebook. Request access at [wrds-www.wharton.upenn.edu](https://wrds-www.wharton.upenn.edu).

Stocks included:
```
AAPL   # Apple
MSFT   # Microsoft
NVDA   # Nvidia
AMZN   # Amazon
JPM    # JPMorgan Chase
JNJ    # Johnson & Johnson
XOM    # ExxonMobil
TSLA   # Tesla
NFLX   # Netflix
V      # Visa
```

Output saved to `data/stocks/raw_<ticker>.csv`.

### `notebooks/extract_commodities.ipynb`
Downloads daily OHLCV data for 10 commodity futures from **Yahoo Finance** via `yfinance` covering January 1, 2011 to January 1, 2026. Uses `curl_cffi` to impersonate a browser and bypass rate limits.

Commodities included:
```
GC=F   # Gold
SI=F   # Silver
CL=F   # Crude Oil (WTI)
NG=F   # Natural Gas
HG=F   # Copper
ZW=F   # Wheat
ZC=F   # Corn
ZS=F   # Soybeans
PL=F   # Platinum
KC=F   # Coffee
```

Output saved to `data/commodities/raw_<name>.csv`.

