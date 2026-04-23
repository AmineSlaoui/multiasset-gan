# SF-MarketGAN — Model Overview

## What It Is

A WGAN-GP with a TCN backbone (**SF-MarketGAN**, Stylized Fact MarketGAN), designed for **60 assets across 3 asset classes** (equities, bonds, commodities). Unlike the FamaFrench model which uses hand-picked Fama-French 3 factors, this model learns its conditioning factors directly from the data using **hierarchical PCA**.

---

## Factor Model — Hierarchical PCA

Instead of Fama-French 3 factors, this model uses **data-driven statistical factors**:

| Level | Description | # Factors |
|-------|-------------|-----------|
| **Global PCA** | Shared cross-asset dynamics across all 60 assets | 5 |
| **Class PCA** | Within-class residual structure (per asset class) | 2 × 3 = 6 |
| **Total** | | **11 factors** |

**Step-by-step:**
1. Run PCA on all 60 assets → 5 global factors (fitted on training data only, no look-ahead)
2. Subtract out global factor contribution → residuals
3. Run PCA on residuals within each asset class → 2 class-specific factors per class

This captures 2× more variance than 5 hand-picked factors and gives the model class-aware structure.

---

## Macro Conditioning — FiLM

Macro variables are injected using **FiLM (Feature-wise Linear Modulation)** instead of concatenation:

```
h ← γ(cov_t) ⊙ h + β(cov_t)
```

A small MLP maps the macro covariate vector at each timestep to a scale (γ) and shift (β) that modulates the TCN hidden state. This is causal — no future macro data leaks into earlier timesteps.

---

## Regime Conditioning (v9)

The latest version adds a **regime encoder** that maps macro covariates to a stress indicator z(t) ∈ [0, 1], supervised against a VIX-threshold-based stress label:

- **Correction strength** is gated by z(t) — the model corrects OLS estimates more aggressively during stress regimes
- **Residual covariance** is augmented by a learned rank-r stress component when z(t) is high:

```
ε_t = σ̂_t · (L_ρ + z(t) · L_Δ) · u
```

where L_ρ is the baseline (shrunk) Cholesky and L_Δ is a small rank-4 learned stress correction. This is the key capability that factor bootstrap (FB) and DCC-GARCH baselines cannot replicate.

---

## Macro Covariates

8 market microstructure variables sourced from Bloomberg (vs Welch-Goyal in FamaFrench model):

| Variable | What it is |
|---|---|
| `cdx_ig` | CDX Investment Grade credit spread index |
| `cdx_hy` | CDX High Yield credit spread index |
| `initial_jobless_claims` | Weekly unemployment claims |
| `fed_funds` | Fed funds rate |
| `usd_10y_swap` | 10Y USD interest rate swap |
| `usd_2y_swap` | 2Y USD interest rate swap |
| `usd_3m_libor` | 3M LIBOR rate |
| `crb_commodity_index` | CRB Commodity index |

These are designed for a multi-asset class universe (equities + bonds + commodities). Before being fed into the TCN, they are **PCA-compressed** (fit on training period only) down to 8 principal components — so the actual covariate dimension the model sees is 8 PCs, not the raw variables.

---

## Key Differences vs FamaFrench Model

| | **FamaFrench Model** | **This Model** |
|---|---|---|
| Assets | 20–50 equities | 60 (equities + bonds + commodities) |
| Factors | Fama-French 3 (hand-picked) | Hierarchical PCA (data-driven, 11 factors) |
| Macro conditioning | Concatenation | FiLM modulation |
| Regime awareness | None | Regime encoder + stress-gated corrections |
| Factor fitting | Rolling OLS | Rolling PCA + OLS on PCA factors |

---

## Suggested Rename

Since `FamaFrenchModel` already occupies the "factor + TCN" name space, this model should be called:

**`HierarchicalPCAGAN`** or **`PCARe​gimeGAN`**

The name reflects the two defining contributions: hierarchical PCA factors + regime-conditional generation.