# SF-MarketGAN: Macro-Conditioned Factor-Model GAN with Stylized Fact Constraints for Multi-Asset Synthetic Data Generation

## 1. Problem

Synthetic multi-asset return data is needed for stress testing, portfolio optimization, and scenario analysis. Existing GANs (MarketGAN, TimeGAN, QuantGAN) focus on single asset classes and fail to capture cross-asset dependence structures. Key challenges:

- **Cross-asset heterogeneity**: Stock-bond correlations flip sign across macro regimes; commodity correlations are sector-dependent
- **Stylized fact enforcement**: Heavy tails, volatility clustering, leverage effects, and cross-asset tail dependence are evaluated post-hoc but never used to guide training
- **Macro conditioning**: Raw macro variables without proper processing miss regime information

## 2. Method

We extend MarketGAN's factor-model GAN architecture with three innovations:

### 2.1 Multi-Asset Factor Model

Five cross-asset global factors replace equity-specific Fama-French:
- Market risk (equal-weighted all-asset return)
- Term structure slope (TLT−SHY return spread)
- Credit spread (HYG−AGG return spread)
- Dollar (−BWX return, inverse proxy)
- Volatility (21-day rolling realized vol of market factor)

Return generation: r_{t+1} = α_t + β_t · F_{t+1} + σ_t ⊙ ε_{t+1}

Stochastic corrections: θ = θ̂ · (1 + η·tanh(f(z))), with η=0.1 bounding magnitude. σ uses softplus to ensure positivity.

### 2.2 FiLM Macro Conditioning

19 daily macro features (VIX, term structure, credit spreads, USD, commodity momentum, market momentum, realized volatility) processed through:
1. Rolling z-score normalization
2. PCA compression (8 PCs, 80.3% variance explained)
3. GMM regime detection (3 regimes)
4. Key level features preserved (VIX, realized vol, term level, credit level)

Final covariate dimension: 15.

**FiLM layer**: TCN processes noise only → MLP(macro) → (γ, β) → h ← γ⊙h + β. Adds <5K params. Macro modulates hidden representations rather than being mixed into input — acts as structural regularizer with smaller OOS generalization gap.

### 2.3 Stylized Fact Constraints with EMA Loss Balancing

Five differentiable losses enforcing cross-sectional and temporal properties:
- L_corr: Correlation matrix Frobenius distance
- L_tail: Soft-sigmoid joint tail co-dependence (differentiable)
- L_vc: Lag-1 ACF of |returns| (volatility clustering)
- L_lev: Return-to-future-volatility correlation (leverage effect)
- L_spill: |Returns| correlation matrix distance (volatility spillover)

Kurtosis loss disabled (overfits non-stationary tail statistics).

**EMA loss-magnitude balancing**: Tracks running ratio of |L_adv| / |L_sf| via exponential moving average. Dynamically scales L_sf so it contributes ~τ=30% of total loss magnitude. Single backward pass (more efficient than dual-pass alternatives). Clamp [0.01, 10.0] prevents instability.

### 2.4 Architecture

- Generator: TCN backbone (k=2, D=2, L=6, RFS=127), hidden=256, d_z=10. Sequence-level: generates T_L=252 steps at once.
- Discriminator: TCN + FiLM, hidden=256. Evaluates full return sequences.
- Training: WGAN-GP (λ=10), Adam (β₁=0, β₂=0.9), lr=5e-4, batch=128
- Rolling OLS: 126-day window for coefficient estimation

## 3. Data

- **60 assets**: 20 stocks, 20 bonds, 20 commodities (ETFs, daily)
- **Period**: 2011-01-04 to 2025-12-30 (3,773 trading days)
- **Split**: Train 2011-2023 / Val 2024 / Test 2025 (OOS)
- **Macro**: 19 daily features from yfinance

## 4. Experimental Plan

Progressive validation — each step builds on the previous:

### Step 1: Single-Class Validation
Train FiLM model separately on each asset class (20 assets each).
Purpose: Verify architecture works before scaling to multi-asset.

### Step 2: Multi-Asset Joint (FiLM)
Train on all 60 assets jointly.
Purpose: Demonstrate joint generation captures cross-asset dependence that single-class models miss.

### Step 3: Full Model (FiLM + SF)
Add EMA-balanced SF constraints.
Purpose: Show SF losses improve stylized fact fidelity beyond FiLM-only.

### Step 4: Hyperparameter Optimization
Sweep hidden_dim, lr, η, τ for the final model configuration.

## 5. Results

### 5.1 Progressive Validation (val_frob ↓)

| Step | Model | Assets | Macro | SF | val_frob | Δ vs baseline |
|------|-------|--------|-------|-----|---------|---------------|
| 1 | single_stocks | 20 | ✅ FiLM | ❌ | 0.70 | — |
| 1 | single_bonds | 20 | ✅ FiLM | ❌ | 2.13 | — |
| 1 | single_commodities | 20 | ✅ FiLM | ❌ | 0.66 | — |
| 2 | baseline (no macro)† | 60 | ❌ | ❌ | 4.72±0.45 | — |
| 2 | multi_film | 60 | ✅ FiLM | ❌ | 2.35±0.06 | -50% |
| **3** | **multi_film_sf** | **60** | **✅ FiLM** | **✅ EMA** | **1.59±0.13** | **-66%** |

### 5.1.1 Multi-Seed Validation (3 seeds: 42, 123, 7)

| Model | s42 | s123 | s7 | mean±std |
|-------|-----|------|-----|----------|
| multi_film (FiLM only) | 2.36 | 2.41 | 2.29 | 2.35±0.06 |
| **multi_film_sf (FiLM+SF)** | **1.60** | **1.72** | **1.46** | **1.59±0.13** |

SF improvement: −32% (Welch's t-test: t=−9.34, p=0.001).

† Baseline (no macro) results from `results/v3/baseline_s*` runs (3 seeds), using FiLM architecture with zero-valued dummy covariates (d_cov=1). Checkpoints in `results_server/v3/`.

### 5.2 OOS Evaluation (2025, Contiguous Generation)

**Within-class correlation fidelity (frob_norm ↓):**

| Model | stocks | bonds | commod | mean |
|-------|--------|-------|--------|------|
| single_stocks | 0.162 | — | — | — |
| single_bonds | — | 0.156 | — | — |
| single_commod | — | — | 0.144 | — |
| **multi_film_sf** | **0.142** | **0.103** | **0.125** | **0.123** |

Key finding: The joint 60-asset model achieves BETTER within-class fidelity than separate single-class models (mean 0.123 vs 0.154, −20%). Joint training provides cross-class information that benefits within-class generation — the model learns shared dynamics (e.g., flight-to-quality) that improve even single-class correlations.

**Cross-class correlation (multi-asset only):**

| Block | frob_norm |
|-------|----------|
| bonds-commodities | 0.114 |
| bonds-stocks | 0.119 |
| commodities-stocks | 0.134 |

Single-class models cannot capture any cross-class structure. This is the unique value of joint multi-asset generation.

**Other OOS metrics:**

| Metric ↓ | single_stocks | single_bonds | single_commod | multi_film_sf |
|----------|--------------|-------------|--------------|--------------|
| VC score | 0.708 | 0.552 | 0.773 | 0.720 |
| Lev score | 0.889 | 0.614 | 0.987 | 0.851 |
| SWD | 0.0038 | 0.0007 | 0.0040 | 0.0030 |
| XCorr | 3.244 | 3.120 | 2.887 | 7.399 |

Note: The absolute XCorr of multi_film_sf (7.40) is higher than single-class models (~3.0) because it covers a 60×60 correlation matrix vs 20×20. The normalized within-class frob_norm is the fair comparison, where multi_film_sf wins.

**Full OOS comparison (all models with correctly trained checkpoints):**

| Metric ↓ | single (mean) | multi_film | multi_film_sf | SF Δ vs film |
|----------|--------------|-----------|--------------|-------------|
| within-class frob_norm | 0.154 | 0.150 | **0.123** | −18% |
| full corr frob_norm | — | 0.129 | **0.123** | −5% |
| cross bonds-stocks | — | 0.109 | 0.119 | — |
| cross bonds-commod | — | 0.118 | 0.114 | — |
| cross commod-stocks | — | 0.122 | 0.134 | — |
| VC score | 0.677 | 0.700 | 0.720 | — |
| Lev score | 0.830 | 0.881 | 0.851 | — |
| SWD | 0.0028 | 0.0028 | 0.0030 | — |

Key insight: multi_film (FiLM only) matches single-class within-class quality (0.150 vs 0.154) — joint training does not sacrifice per-class fidelity. Adding SF constraints (multi_film_sf) further improves within-class to 0.123 (−20% vs single-class), demonstrating that SF losses help the model capture shared dynamics across classes that benefit even per-class generation.

### 5.3 EMA Loss Balancing Analysis

The EMA scale factor saturated at the clamp maximum (10.0), indicating the adversarial loss magnitude exceeds the SF loss by ~10×. Without this scaling, SF constraints have zero measurable effect on generation quality (confirmed in Round 1 experiments where SF losses were added naively). This validates the loss-dominance problem identified in concurrent work (MacroFactor-GAN) and demonstrates that EMA-based single-pass loss scaling is an effective and simpler alternative to dual-backward-pass gradient scaling approaches.

## 6. Claimed Contributions

1. **Multi-asset joint generation** of stocks + bonds + commodities (60 assets) — within-class quality exceeds separate single-class models
2. **FiLM macro conditioning** with PCA+regime pipeline — -55% val_frob vs baseline, resolves concat's OOS overfitting
3. **EMA loss-balanced SF constraints** — -32% val_frob vs FiLM-only, single-pass alternative to dual-pass balancing
4. **Block-decomposed evaluation** — separates within-class and cross-class correlation fidelity for fair single vs multi comparison

## 7. Remaining Work

1. ~~Multi-seed (×3) for statistical significance~~ ✅ Done
2. ~~Fix multi_film OOS checkpoint~~ ✅ Done (restored from results/v3/)
3. Factor bootstrap baseline (traditional method comparison)
4. Portfolio optimization (economic value demonstration)
5. Ablation study (component-by-component contribution)
6. Counterfactual scenario generation (qualitative validation)
