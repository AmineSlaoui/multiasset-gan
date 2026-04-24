# Codex Code Review — SF-MarketGAN Submission Audit

_Automated review at nightmare difficulty. Issues are listed by severity; each bullet cites file:line and the expected failure mode. Fix these before paper submission — several are claim-invalidating._

## Critical (blocks submission if paper claims depend on the affected path)

1. **Validation contamination in model selection.**
   `src/train_v9.py:202`, `src/train_v9.py:347`, `src/trainer.py:75`.
   `ProperValDataset` keeps any window with ≥ half of the 252-day target in the validation span, but `sample_batch()` still returns the full target window. The `val_frob` used for checkpointing therefore includes pre-`val_start_idx` returns for the earliest windows, so early stopping is not based on pure validation data.

2. **GMV backtest is not truly walk-forward.**
   `src/eval_portfolio_rolling.py:84`, `:96`; `src/eval_v9.py:37`, `:43`.
   Each test year's full synthetic path is generated once using the realised OOS factor path `factors_arr[test_start_idx:test_start_idx+t_l]` and the realised OOS macro path `cov_arr[t_start:t_end]`. Month-k covariance slices are then carved from those already-future-conditioned paths, so at rebalance `rp` the covariance estimate has seen realised future conditioning inputs through year-end.

3. **Perturbed-forecast covariance is not walk-forward either.**
   `src/eval_perturbed_forecast.py:181`, `:189`; `src/eval_v9.py:37`, `:43`.
   Same structural issue: noise is added only to the mean forecast, while the covariance side is drawn from paths conditioned on the realised future factor/macro trajectory over the full test window.

4. **Rolling-year evaluation can inject later-sample `L_rho`.**
   `src/eval_portfolio_rolling.py:71`, `:84`; `src/eval_v9.py:82`.
   `load_v9()` overwrites the frozen checkpoint `L_rho` with whatever matrix is passed in, and the rolling driver passes the single global `data/residual_cholesky.npy`. For `v9_roll2022`/`v9_roll2023`, this leaks a later-sample residual-correlation prior into earlier-year backtests unless `residual_cholesky.npy` was rebuilt per cutoff.

5. **Hard import failure.**
   `src/eval_portfolio_rolling.py:37` imports `eval_portfolio_v9`, but there is no `eval_portfolio_v9.py` in `src/`. The rolling GMV script cannot execute as committed.

## Medium (tightens claims, may require one re-run or one paragraph of caveat text)

6. **"10-seed ensemble" is not averaged.**
   `src/eval_v9.py:227`, `:248`. Each checkpoint is scored as its own model entry — no seed-level path averaging, no metric aggregation. If the paper claims an ensemble result, this file does not produce it.

7. **Bootstrap CIs don't match the main evaluator.**
   `src/eval_bootstrap_ci.py:124`, `:129`; `src/evaluate.py:288`. `full_evaluation()` averages ACF/VC/LEV across up to 20 paths; `eval_bootstrap_ci.py` uses `paths[0]` only. The docstring's "identical results" claim is false and the CI suppresses path-level Monte Carlo variation.

8. **Orthogonalisation is training-only but not rolling-causal.**
   `src/factors_v3.py:103`, `:195`; `src/subsector_factors.py:86`. Projection coefficients are estimated once on the full training span and applied to all dates. Avoids OOS leakage, but paper text shouldn't call the GS step "rolling" or "time-local".

9. **`--preprocessed_dir` silently wins over `--factors_version v3`.**
   `src/train_v9.py:80`, `:97`, `:403`. Help text says preprocessed mode is v2-only, but no runtime enforcement. A run can land with `factors_version=v3` in `config.json` while actually using v2 arrays.

10. **Pair-correlation RMSE diagnostic crosses path boundaries.**
    `src/fix_residual_cholesky.py:83`, `:86`, `:93`. Concatenates independent generated paths plus a tiled real test path, then runs 63-day rolling correlation across the concatenation. Cross-boundary windows are artefacts; this is not the same estimand as averaging within-path RMSEs.

## Minor

11. `src/eval_perturbed_forecast.py:145`: `beta_at_rebal` computed, unused — stale rebalance indexing.
12. Unused imports (`GeneratorV9`, `pandas`) in `src/eval_bootstrap_ci.py:25`, `src/eval_perturbed_forecast.py:31`, `src/fix_residual_cholesky.py:12`.

## Not a bug

`L_rho` is not accidentally updated in the default frozen-training path. In `src/sf_marketgan_v9.py:89` it's a buffer, so `opt_G.step()` at `src/train_v9.py:257` cannot touch it. The only `L_rho` mutations are the explicit overwrite paths called out under issue 4.
