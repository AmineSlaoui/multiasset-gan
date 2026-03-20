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
Downloads daily OHLCV data for 10 stocks from **Yahoo Finance** via `yfinance` covering January 1, 2011 to January 1, 2026. Uses `curl_cffi` to impersonate a browser and bypass rate limits.

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

### `notebooks/extract_bonds.ipynb`
Downloads daily OHLCV data for 10 bond ETFs from **Yahoo Finance** via `yfinance` covering January 1, 2011 to January 1, 2026.

ETFs included:
```
SHY   # iShares 1-3 Year Treasury Bond ETF
IEF   # iShares 7-10 Year Treasury Bond ETF
TLT   # iShares 20+ Year Treasury Bond ETF
LQD   # iShares Investment Grade Corporate Bond ETF
HYG   # iShares High Yield Corporate Bond ETF
MUB   # iShares National Municipal Bond ETF
TIP   # iShares TIPS Bond ETF
EMB   # iShares Emerging Market Bond ETF
AGG   # iShares Core US Aggregate Bond ETF
BND   # Vanguard Total Bond Market ETF
```

Output saved to `data/bonds/raw_<ticker>.csv`.

