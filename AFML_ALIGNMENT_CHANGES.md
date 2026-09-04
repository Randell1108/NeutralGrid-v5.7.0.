# AFML Alignment Changes (NeutralGrid Scanner)

This revision aligns the repository with three AFML-critical properties:

1. Each observation is an **event** with a start time (t0) and an end time (t1).
2. Cross-validation applies **purging/embargo** using event spans to avoid leakage.
3. Inference uses the **same preprocessing** used during training.

## Implemented changes

### 1) Outcome labels now include `t1` (event end) and barrier metadata
**File:** `src/neutralgrid/training/data_generator.py`

- `BarrierLabelGenerator.compute_label_from_final_pnl` now accepts `symbol`, `start_time_utc`, and optional `end_time_utc`, and it produces:
  - `t1` (observed end time when available; otherwise `start_time_utc + horizon_hours`).
  - `barrier_touched` inferred from terminal PnL when the full intrahorizon price path is unavailable.
- `ExistingDataMapper.map_dataframe` and `TrainingDataBuilder.build_from_snapshots` were updated to populate `t1`, `barrier_touched`, and related columns so downstream CPCV and sample-weighting can operate on event spans.

### 2) CPCV now prefers time-based purging when `t1` is present
**File:** `src/neutralgrid/models/meta_labeler.py`

- When a `t1` column exists (and has values), the meta-labeler uses `CPCVConfig(purge_hours=..., embargo_hours=...)` rather than percentage-based purging.
- This makes purging/embargo consistent with event durations, which is the intended AFML usage.

### 3) Training-serving consistency for imputation (no “fill with 0” at inference)
**File:** `src/neutralgrid/models/meta_labeler.py`

- The final training imputer is stored in the model state and is reused during inference.
- `predict_proba` and `_prepare_features` were updated so inference uses the saved feature ordering and saved imputer.
- The imputer is now persisted in `save()` and restored in `load()`.

### 4) Removed scoring assumptions during the initial scan step
**File:** `scanner/scan.py`

- The scan step no longer fabricates values such as `survival_prob = 0.5` or estimates grid economics (profit per grid, number of grids).
- EV scoring is computed only when all required inputs are present. Otherwise the system falls back to legacy scoring (similarity + profile probability) and records the reason in `scoring_flags`.
- The scan step now stores `similarity_score` to enable deterministic post-enrichment scoring.

### 5) Post-enrichment AFML scoring is applied after grid parameters are known
**File:** `run_full_pipeline.py`

- After `enrich_with_grid_params`, the pipeline computes:
  - EV via `PnLRanker` (requires enriched grid params and regime metrics).
  - Meta-label probability via the trained `MetaLabeler` (if available).
  - `score_afml` when all components are available; otherwise the original scan `score` is retained.

### 6) Microstructure gating no longer uses an arbitrary default range
**File:** `scanner/enrich_grid_params.py`

- Microstructure volatility proxy is derived from computed grid bounds.
- If the range size cannot be computed, microstructure analysis is skipped and recorded as `micro_reason = range_size_unavailable` (no hidden defaults).
