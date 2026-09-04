# META_LABELER_FIX — FASTWIN-01 implementation spec

Goal: make the meta-labeler **healthy, accurate, helpful** as a true **AFML precision filter** on the
fast-winner label `eff_pnl >= 3% & duration <= 7h`, with present (not deferred) results.

All numbers below were **executed on real data and read back from output** — none assumed. The one
empirical study is committed at `scripts/meta_labeler_feature_study.py` (Step 1, done).

---

## Verified facts (executed this session)

- Pool source: `data/backtest_candidates/training_data_*.csv` -> 5,591 rows -> **633 distinct candidate_ids**.
- The **authoritative geometric pool** = rows passing the full `build()` Step-5 gate. VERIFIED filter
  equivalence on `_load_backtest_rows()` output (5,591 rows):
  - `is_authoritative==True` alone = **3503** (WRONG — 3247 are `mode=NaN`, only 256 geometric)
  - `version_gated==False` = **256**  (non-geometric rows are marked `version_gated=True` by the gate)
  - `~version_gated & is_authoritative & source_class!=reconstruction` (build()-style) = **256, n_pos=111**
  => `build_meta_labeler_pool()` MUST use the build()-style filter, NOT `is_authoritative` alone.
- Fast-winner label on the 256-row geometric pool: **n_pos = 111, base_rate = 0.434** (duration is constant
  6h here so `<=7h` is non-binding; keep it for forward generality once live durations vary).
- **Feature set = 20 ex-ante, leakage+circularity clean.** Calibrated L2 logistic, symbol-grouped purged CV,
  bootstrap 95% CI (reproduced twice identically):
  - geometric (n=256/111): **OOF AUC 0.704 [0.641, 0.761]**, ECE 0.056, decile lift 1.75
  - broad (n=633/143): **OOF AUC 0.752 [0.702, 0.797]** (top-k 0.767; GBM 0.783)
  - current production 7-feat GBM baseline: 0.53 (geom) / 0.64 (broad) — the noise floor it sits at
  - `leak_check` (no feature univariate AUC>0.90): EMPTY on both pools
- k=floor(n_pos/10)=11 truncation by MI scored LOWER (0.659) than the full-20 L2 logistic (0.704). Operator
  chose: **keep all 20, control complexity by L2 regularization**; the OOF-CV CI excluding 0.5 is the direct
  non-overfit proof (and the promotion gate enforces it).

## CRITICAL — `primary_pipeline_score` is CIRCULAR; it MUST stay excluded
The scanner ranking score (`scan.py` -> `score`/`scan_score` -> mapped to `primary_pipeline_score`) is produced
by `mi_weighted_scorer_v20260311.py` from signals `(similarity_score, profile_proba, ev_score, meta_prob)` —
it can CONTAIN `meta_prob`. Feeding it to the meta-labeler is feeding the model its own output back.
`tests/test_afml_compliance.py:521` enforces `"primary_pipeline_score" not in cfg.features`; `CHANGELOG.md:942`
records its removal as "a circular feature." KEEP that guard green. `ev_score` is independent (derived from
`PnLRanker.compute_score().rank_score`, builder line 834 — NOT from meta_prob) and IS retained.

## THE FORMULA (final, compliant)
```
LABEL:    eff_pnl = net_pnl_pct if finite else pnl_pct
          y = 1[ eff_pnl >= 3.0%  AND  duration_hours <= 7.0h ]
FEATURES (20): ev_score, adx_1h, adx_15m, adx_5m, atr_pct_15m, ou_halflife, rsi_15m, ema_slope_1h,
  ema_crosses_5m, vwap_crosses_5m, hurst_exponent, range_size_pct, bb_width, grid_spacing_pct,
  profit_per_grid_pct, num_grids, quote_volume_24h, open_interest, micro_round_trip_cost_pct, funding_rate
ESTIMATOR: calibrated L2 logistic; meta_prob = Platt(sigma(b0 + sum bj*standardize(median_impute(xj)))).
STAGE-B GATE (fail-closed): reject data_missing:meta on NaN; deploy iff meta_prob >= tau.
KELLY: b = E[eff|y=1]/|E[eff|y=0]| estimated from pool; f* = (p*b - q)/b. (Already wired in _compute_kelly.)
PROMOTION GATE: pass iff (a) purged OOF-AUC 95% CI excludes 0.50 AND (b) n_pos>=70 AND (c) ECE<=0.10.
  Current geometric pool PASSES all three.
```

## Ground-truthed code anchors (verified via git show HEAD)
- `unified_training_builder.py`: `build()` ENDS line 557 (`return combined`); `_load_snapshots` at 561;
  `_load_backtest_rows` at 1419; `_apply_ingestion_gate` at 1233. NO `_rank/_finalize/_compute_sample_weights`
  in HEAD (those were duplicates from a reverted spike). Insert `build_meta_labeler_pool()` between 557 and 561.
  `_load_backtest_rows()` already returns `is_authoritative`, `version_gated`, `source_class`, `mode`,
  `sample_weight_override`, `start_time_utc`, and all 20 FW features (verified: 0 missing).
- `meta_labeler.py`: `ACTIVE_*` consts ~212; profile tuples + `META_FEATURE_PROFILES` ~127/218; `_make_model()`
  at 907; `MetaLabelerConfig` at 435; `use_sequential_bootstrap` flag at 483; `LogisticRegression` imported
  locally at 1326.
- `retrain_meta_labeler.py`: label hardcoded `pnl.gt(0.0)` in `_compute_fast_target_population`;
  `MetaLabelerConfig(...)` built in `main()`; `--feature-profile` default = `ACTIVE_BOOTSTRAP_META_FEATURE_PROFILE`.

## Steps (each: edit -> git diff -> pyright -> targeted pytest -> revert+retry if red)
2. Builder: add `build_meta_labeler_pool()` = `_load_backtest_rows` -> build()-style filter
   (`~version_gated & is_authoritative & source_class!=reconstruction`) -> ensure `ALL_META_FEATURES` cols ->
   sort by start_time_utc. Returns 256 rows / n_pos 111. Do NOT touch `build()`.
3. Model contract: add 20-feat `SNAPSHOT_META_FEATURES_V20260530_FASTWIN`; point ACTIVE at it; register profile;
   `ACTIVE_META_TARGET_PNL_THRESHOLD_PCT=3.0`, contract `..._pnl_ge_3`, corrected docstring; add median defaults
   for ev_score/adx_5m/ema_crosses_5m/vwap_crosses_5m/range_size_pct; add `estimator_type` + logistic branch.
4. Retrain wiring: `ACTIVE_META_FEATURE_PROFILE` default; route to `build_meta_labeler_pool()`; label `>=3.0`;
   `estimator_type="logistic"`; keep bootstrap imputation guard.
5. Feature Pipeline Update Rule: verify 20 feats present in all 3 files (`verify-feature-pipeline`); add any gap.
6. Promotion gate: bootstrap OOF-AUC CI + ECE + n_pos -> `promotion_status` in metadata.json; fail-closed.
7. Test cascade: update contract/afml/inference/bypass tests to FASTWIN; keep primary_pipeline_score guard
   green; add promotion-gate test. `pytest tests/` green (read real failures, don't trust predictions).
8. Re-enable consumption gated on `promotion_status`: `deployment_meta_prob = meta_prob if authoritative else
   None`; SOFT Stage-B meta gate (`meta_gate_enabled`, `min_meta_prob` in TwoStageConfig); update the ~4 tests
   asserting `diagnostic_only`.
9. Lineage backfill (`scripts/backfill_training_features.py --default-artifact-version <active_hmm>` fresh path)
   + `verify-hmm-lineage` + `retrain_meta_labeler.py`; confirm metadata `promotion_status`.

## The three failures (per user) — handling
1. Fabricated numbers -> only write a statistic after reading its run output. Study script prints+saves+re-reads.
2. Label leakage -> strict ex-ante allowlist + outcome regex + CIRCULAR_EXCLUDE; assert no AUC>0.90.
3. Broken tree -> one coherent change, git diff+pyright+pytest each step, revert+retry on red.

## Implementation status — COMPLETE (2026-05-30)
All steps 2-9 landed, pyright clean on package files, `pytest tests/` green except two
PRE-EXISTING `btk_funding` failures (confirmed unrelated via stash; not meta-labeler).
- Step 2-4 (builder pool / model contract / retrain wiring): landed as specified.
- Step 5: Feature Pipeline Update Rule verified for all 20 features across the 3 files
  (`micro_round_trip_cost_pct` flows via `_SCANNER_TO_FEATURE` + the line-1131 fill list,
  not the literal `TRAINING_OUTPUT_COLUMNS`; still present in every mechanism).
- Step 6: promotion gate -> `MetaLabeler._evaluate_promotion_gate` (pass iff CI-low>0.50,
  n_pos>=70, ECE<=0.10; fail-closed), `_bootstrap_auc_ci`, fields on `MetaLabelerMetrics`,
  persisted to `eval_metrics`, restored onto `MetaLabeler.promotion_status`/`is_promoted` on load.
- Step 7: contract test bumped `v20260408 -> v20260530`; inference-alias + bypass-gate
  fixtures updated to FASTWIN; +new promotion-gate / soft-meta-gate tests; afml circular guard
  stays green.
- Step 8: `deployment_meta_prob` + `meta_prob_authority` gated on `promotion_status=="pass"`
  (enrich_grid_params + run_full_pipeline post-scoring); SOFT Stage-B meta gate
  (`TwoStageConfig.meta_gate_enabled`/`min_meta_prob`, default OFF, `data_missing:meta`).
- Step 9: retrained on the geometric pool (`build_meta_labeler_pool`, 256 rows), PINNED to
  active HMM `rolling_180d_20260524_140738`; **promotion_status=pass**.

Two refinements required by the real retrain (both verified):
- FIX: the L2-logistic estimator was unscaled (train() built for scale-invariant GBM) ->
  `_make_preprocessor` folds `StandardScaler` after the mean imputer for the logistic path,
  fit per-fold (leakage-safe), persisted with the model.
- GATE FIDELITY (operator-approved): the model's training CPCV is degenerate on this
  256-row/100-symbol pool (3 splits, 135/256 scored -> noisy 0.558), so the gate is judged
  by `_evaluate_promotion_oof` (study-faithful: symbol-grouped, time-purged, all-rows-scored,
  CALIBRATED OOF) -> AUC 0.640 [0.568, 0.712], ECE 0.058, n_pos 110 -> PASS. The unreliable
  native-CPCV `auc_cv`/OOS-calibration on small symbol-diverse pools is tracked as ERR-053 (WATCH).
- The verified geometric numbers (line 24) come from the study CV; the deployed gate's
  numbers (0.640/0.058) are the production estimate under the same methodology with the
  added time-purge + calibration. Promotion criteria still hold.
