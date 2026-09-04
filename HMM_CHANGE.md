# HMM Timeframe Migration Plan: 1h -> 15m (Verified)

## Purpose

Migrate HMM regime detection from 1h klines to 15m klines to align with the latest
17 expired winning bots (rows 138-154 in `data/new_expired_bots.xlsx`). Lower the
profile model `min_duration_hours` to match the <7h bot horizon.

**Constraint**: Every modification is traced to a specific file:line. No step may
leave a residual 1h coding structure that will silently drift or corrupt the new model.

---

## 1. Latest 17 Winners -- Reference Data

| Row | Symbol       | Duration (h) | PnL %  |
|-----|-------------|--------------|--------|
| 138 | ONUSDT      | 5.90         | 15.10  |
| 139 | AIOTUSDT    | 2.20         | 2.29   |
| 140 | ONUSDT      | 4.63         | 4.40   |
| 141 | PLAYUSDT    | 7.53         | 5.62   |
| 142 | BASEDUSDT   | 6.00         | 17.46  |
| 143 | BSBUSDT     | 6.02         | 7.19   |
| 144 | BASEDUSDT   | 6.12         | 6.25   |
| 145 | BASEDUSDT   | 6.13         | 5.73   |
| 146 | BSBUSDT     | 6.02         | 10.12  |
| 147 | BASEDUSDT   | 6.02         | 10.27  |
| 148 | CTSIUSDT    | 4.77         | 12.54  |
| 149 | JCTUSDT     | 3.30         | 9.35   |
| 150 | JCTUSDT     | 3.27         | 10.44  |
| 151 | KOMAUSDT    | 2.70         | 16.22  |
| 152 | TRUUSDT     | 6.05         | 15.89  |
| 153 | NOMUSDT     | 3.47         | 10.34  |
| 154 | NOMUSDT     | 5.97         | 5.06   |

- Pandas row indexing (iloc[138:155]), not Excel row numbers
- Median duration: 5.97h
- 16/17 under 7h (only PLAYUSDT at 7.53h exceeds)
- All 17 profitable (PnL % > 0)

---

## 2. Why 15m (Not 5m)

| Factor | 15m | 5m |
|--------|-----|----|
| Bars for 180 days | 17,280 | 51,840 |
| Indicator warmup (20 bars) | 5h | 1.67h |
| Tested range_prob (TRUUSDT) | 0.998 | 0.999 |
| Already fetched at scan time | Yes (`scan.py:346`) | Yes (`scan.py:347`) |
| Binance Vision store support | Yes (`canonical_retrain.py:99` accepts interval) | Yes |

Both detect ranging. 15m balances granularity with data volume.

---

## 3. Verified Modification Plan

### Phase A. HMM Artifact/Schema/Training Contract -> 15m

#### A1. Schema constant
- **File**: `src/neutralgrid/models/hmm/schema.py:47`
- **Current**: `TIMEFRAME = "1h"`
- **Change to**: `TIMEFRAME = "15m"`
- **Why**: This stamps `feature_schema.json` via `get_feature_schema()` at schema.py:143 as `"timeframe": TIMEFRAME`. Without this change, the saved feature_schema.json would claim 1h when the model was trained on 15m.

#### A2. Feature computation -- rename parameter
- **File**: `src/neutralgrid/data/features.py:155`
- **Current**: `def compute_hmm_features(df_1h: pd.DataFrame)`
- **Change to**: `def compute_hmm_features(df: pd.DataFrame)`
- **Internal references to rename** (same file):
  - Line 180: `df_1h` -> `df` (None/empty check)
  - Line 184: `df_1h` -> `df` (validate_dataframe)
  - Line 187: `df_1h` -> `df` (sort_values)
  - Lines 190-192: `df_1h["close"]`, `df_1h["high"]`, `df_1h["low"]` -> `df[...]`
- **Docstring** (lines 156-163): Remove "1H" references
- **Why**: Function is timeframe-agnostic internally (operates on OHLCV columns). Parameter name must not claim a timeframe.

#### A3. Feature computation dict variant
- **File**: `src/neutralgrid/data/features.py:243`
- **Current**: `def compute_hmm_features_dict(df_1h: pd.DataFrame)`
- **Change to**: `def compute_hmm_features_dict(df: pd.DataFrame)`
- Line 262: `compute_hmm_features(df_1h)` -> `compute_hmm_features(df)`

#### A4. Protocol contract
- **File**: `src/neutralgrid/core/protocols.py:27`
- **Current**: `df_1h: pd.DataFrame` in `RegimePredictor.predict()`
- **Change to**: `df: pd.DataFrame`
- Line 33 docstring: `"df_1h: 1-hour OHLCV DataFrame."` -> `"df: OHLCV DataFrame."`
- **Why**: This is the structural type contract. If the implementation renames but the protocol keeps `df_1h`, Pyright protocol conformance breaks.

#### A5. Inference -- rename parameter
- **File**: `src/neutralgrid/models/hmm/inference.py:545`
- **Current**: `def predict(self, df_1h: pd.DataFrame, ...)`
- **Change to**: `def predict(self, df: pd.DataFrame, ...)`
- Line 560: `compute_hmm_features(df_1h)` -> `compute_hmm_features(df)`
- Line 563: `"No 1H data for HMM inference"` -> `"No kline data for HMM inference"`
- Line 572-576: `"Insufficient 1H history for HMM inference"` -> `"Insufficient kline history for HMM inference"`

#### A6. Inference wrapper
- **File**: `src/neutralgrid/validation/hmm_regime.py:179`
- **Current**: `df_1h: pd.DataFrame`
- **Change to**: `df: pd.DataFrame`
- Line 187 docstring: Remove "1H" reference
- Line 196: `artifact.predict(df_1h, ...)` -> `artifact.predict(df, ...)`

#### A7. Training function -- rename parameters
- **File**: `src/neutralgrid/models/hmm/train.py:42`
- **Current**: `per_symbol_dfs_1h: List[pd.DataFrame]`
- **Change to**: `per_symbol_dfs: List[pd.DataFrame]`
- **All keyword call sites** (all callers use keyword args, none positional):
  - `train.py:844` — `per_symbol_dfs_1h=dfs` -> rename kwarg
  - `src/neutralgrid/backtest/evaluate.py:260` — `per_symbol_dfs_1h=train_dfs` -> rename kwarg
- Docstring at line 55: Remove "1H" reference

- **File**: `src/neutralgrid/models/hmm/train.py:342`
- **Current**: `per_symbol_dfs_1h: List[pd.DataFrame]`
- **Change to**: `per_symbol_dfs: List[pd.DataFrame]`
- **Keyword call site**: `train.py:884` uses `per_symbol_dfs_1h=dfs` -> rename kwarg
- Docstring at line 364: Remove "1H" reference

#### A8. Training metadata
- **File**: `src/neutralgrid/models/hmm/train.py:673`
- **Current**: `timeframes_used=["1h"]`
- **Change to**: `timeframes_used=["15m"]`
- **Why**: Records in `metadata.json` what timeframe was trained. Must be truthful.

### Phase B. Canonical Feeding -- Interval-Aware Math

#### B1. Canonical store coverage
- **File**: `src/neutralgrid/models/hmm/canonical_retrain.py:99`
- **Current**: `interval: str = "1h"`
- **Change to**: `interval: str = "15m"`

#### B2. Screening row-count math
- **File**: `src/neutralgrid/models/hmm/canonical_retrain.py:279`
- **Current**: `expected_bars = int(boundary.window_days * 24 * min_coverage_pct)`
- **Change to**: `expected_bars = int(boundary.window_days * 24 * 4 * min_coverage_pct)`
- **Why**: `* 24` assumes 1 bar/hour (1h). At 15m, there are 4 bars/hour. Without this, screening accepts symbols with only 25% of the expected 15m coverage. On a 180-day window: current formula gives 3,888 bars (= 40.5 days of 15m); corrected gives 15,552 bars (= 162 days of 15m).

#### B3. Slice-local validation
- **File**: `src/neutralgrid/models/hmm/canonical_retrain.py:392`
- **Current**: `relaxed_min_rows = int(boundary.window_days * 24 * 0.90)`
- **Change to**: `relaxed_min_rows = int(boundary.window_days * 24 * 4 * 0.90)`
- **Why**: Same 1h math problem as B2. Comment on line 391 says "180 * 24 * 0.90 = 3888" -- must update to "180 * 24 * 4 * 0.90 = 15552".

#### B4. Slice-local validation interval
- **File**: `src/neutralgrid/models/hmm/canonical_retrain.py:403`
- **Current**: `validate_kline_store(df, interval="1h", min_rows=relaxed_min_rows)`
- **Change to**: `validate_kline_store(df, interval="15m", min_rows=relaxed_min_rows)`

#### B5. Binance Vision pipeline upstream store validation
- **File**: `src/neutralgrid/data/binance_vision/pipeline.py:241`
- **Current**: `min_rows = int(min_years * 365.25 * 24)  # 1H bars`
- **Change to**: Make interval-aware. The function already receives `interval` as a parameter. Derive bars_per_hour from interval: `{"1h": 1, "15m": 4, "5m": 12, "1m": 60}`. Then: `min_rows = int(min_years * 365.25 * 24 * bars_per_hour)`.
- **Why**: When called with `interval="15m"`, the current formula computes `0.493 * 365.25 * 24 = 4,320` rows -- this is 1h row count (4,320 hours = 180 days). At 15m, 180 days = 17,280 bars. The validation would pass a store with only 25% of required 15m coverage.

### Phase C. HMM Inference-Limit Contract

#### C1. Replace active inference limit
- **File**: `src/neutralgrid/core/config.py:160`
- **Current**: `infer_limit_1h: int = 200`
- **Change to**: `infer_limit: int = 800`
- **Why**: The active HMM inference limit must match the active HMM timeframe. 200 bars at 1h = 200 hours = 8.3 days. Equivalent at 15m = 800 bars = 200 hours = 8.3 days. One active knob, not parallel knobs for 1h and 15m.
- **Note**: The exact value 800 preserves the same wall-clock inference window. It is a tuning parameter, not a structurally proven requirement. It can be adjusted after validation.

#### C2. Update cross-config consistency check
- **File**: `src/neutralgrid/core/config.py:601-607`
- **Current**: Checks `kline_limits["1h"] >= hmm.infer_limit_1h`
- **Change to**: Check `kline_limits["15m"] >= hmm.infer_limit`
- **Why**: The invariant is `kline_limits[active_hmm_timeframe] >= active_hmm_infer_limit`. After migration, the active HMM timeframe is 15m.

#### C3. Increase kline fetch limit
- **File**: `src/neutralgrid/core/config.py:389`
- **Current**: `"15m": 300`
- **Change to**: `"15m": 800`
- **Why**: Scan fetches 300 bars of 15m (= 75 hours). HMM inference needs 800 bars (= 200 hours). Without this, `infer_regime()` silently receives truncated data. Binance kline weight: 2 for <=1000 bars -- no rate limit impact.

#### C4. All read sites of infer_limit_1h
- **File**: `src/neutralgrid/scanner/scan.py:320`
- **Current**: `hmm_infer_limit = int(get_config().hmm.infer_limit_1h)`
- **Change to**: `hmm_infer_limit = int(get_config().hmm.infer_limit)`

- **File**: `src/neutralgrid/validation/regime_validator.py:291`
- **Current**: `infer_limit=int(_cfg.hmm.infer_limit_1h)`
- **Change to**: `infer_limit=int(_cfg.hmm.infer_limit)`

- **File**: `tests/test_afml_integrations.py:358-360`
- **Current**: `assert _cfg.validation.kline_limits["1h"] >= _cfg.hmm.infer_limit_1h`
- **Change to**: `assert _cfg.validation.kline_limits["15m"] >= _cfg.hmm.infer_limit`
- **Why**: Test validates the cross-config invariant. Must mirror the production check in config.py.

### Phase D. Runtime HMM Feeders -> 15m

#### D1. Scan-time inline HMM inference
- **File**: `src/neutralgrid/scanner/scan.py:595`
- **Current**: `regime_result = infer_regime(hmm_artifact, df1, infer_limit=hmm_infer_limit)`
- **Change to**: `regime_result = infer_regime(hmm_artifact, df15, infer_limit=hmm_infer_limit)`
- **Why**: `df1` is 1h klines (scan.py:518). `df15` is 15m klines (scan.py:519). Both already fetched. Only the HMM input changes; `compute_features()` at scan.py:524 still uses `df1` for non-HMM features.

#### D2. Validator HMM routing
- **File**: `src/neutralgrid/validation/regime_validator.py:259`
- **Current**: `def check_hmm_regime(self, symbol: str, df_1h: pd.DataFrame)`
- **Change to**: `def check_hmm_regime(self, symbol: str, df: pd.DataFrame)`

- **File**: `src/neutralgrid/validation/regime_validator.py:290`
- **Current**: `df_1h` passed to `infer_regime()`
- **Change to**: `df` passed to `infer_regime()`

- **File**: `src/neutralgrid/validation/regime_validator.py:942`
- **Current**: `df_1h = parse_klines(klines["1h"])` then line 964: `hmm_check = self.check_hmm_regime(symbol, df_1h)`
- **Change to**: Feed 15m klines to HMM. The validator already parses `klines["15m"]` for later use. Route the 15m DataFrame to `check_hmm_regime()`.
- **Why**: If scan.py feeds 15m but the validator feeds 1h, they produce different range_prob values for the same symbol -- silent behavioral divergence.

#### D3. Validator error string matching
- **File**: `src/neutralgrid/validation/regime_validator.py:294`
- **Current**: `"Insufficient 1H history for HMM inference" in str(e)`
- **Change to**: Match updated message from A5: `"Insufficient kline history for HMM inference"`
- **Why**: The inference module (A5) changes this error message. If the validator's string match is not updated, it stops classifying this error correctly and falls through to a generic failure path.

### Phase E. Downstream Lineage Repair

#### E1. Enrichment lineage reads
- **File**: `src/neutralgrid/scanner/enrich_grid_params.py:1291-1324`
- **Current**: Reads HMM fields from `vres.tf_1h.checks` (regime_conf, posterior_mode, persistence_prob, trained_at_utc, calibration_provenance, artifact_version, pipeline_version, volatility_tier, conditional_tail_risk)
- **Required change**: Read from the authoritative top-level HMM result fields instead of `tf_1h.checks`.
- **Why**: After migration, HMM runs on 15m data. If `tf_1h` stops being the canonical HMM slot, these reads return None and enriched rows silently lose all HMM lineage fields.

#### E2. Training lineage reads
- **File**: `src/neutralgrid/training/scanner_integration.py:225-226`
- **Current**: `vres.tf_1h.checks` for trained_at_utc
- **Required change**: Read from top-level HMM result fields.
- **Why**: Same as E1. Training snapshots would lose HMM provenance.

#### E3. API output truthfulness
- **File**: `src/neutralgrid/api/app.py:331-339`
- **Current**: `executed_checks.append("1h_regime")`, `details["1h"] = { ... tf_1h ... }`
- **Required change**: Emit stage-based labels (e.g., `"hmm_regime"`) instead of timeframe-named labels. Do not emit `"1h_regime"` for a 15m HMM.
- **Why**: API consumers receive false metadata if the label says 1h but the computation is 15m.

#### E4. Storage persistence
- **File**: `src/neutralgrid/storage/database.py:75-76`
- **Current**: `validation_1h_passed INTEGER`, `validation_15m_passed INTEGER`
- Lines 177, 318-321: Read from `validation_result.tf_1h`
- **Required change**: Store the HMM stage result independently of timeframe naming. The column `validation_1h_passed` currently represents "HMM regime passed" -- after migration, the HMM runs on 15m but the column name would be misleading.
- **Minimal change**: Add `hmm_regime_passed` column, populate it from the actual HMM result. Existing columns can remain as compatibility shadows during migration.

### Phase F. Both Retrain Entry Points Aligned

#### F1. Primary retrain entry
- **File**: `retrain_hmm.py:288`
- **Current**: `timeframe="1h"`
- **Change to**: `timeframe="15m"`

- **File**: `retrain_hmm.py:169`
- **Current**: `"(1H timeframe)"`
- **Change to**: `"(15m timeframe)"`

- **File**: `retrain_hmm.py:72-73`
- **Current**: `help="Number of 1H bars per symbol (default: 1000, ~42 days)"`
- **Change to**: Update help text to reflect 15m semantics.

#### F2. CLI retrain entry (second entrypoint)
- **File**: `src/neutralgrid/cli/retrain.py:124`
- **Current**: `bars_needed = window_days * 24 + 100` (1h math: 24 bars/day)
- **Change to**: `bars_needed = window_days * 24 * 4 + 100` (15m math: 96 bars/day)

- **File**: `src/neutralgrid/cli/retrain.py:136`
- **Current**: `effective_days = (1500 - 100) / 24` (1h math)
- **Change to**: `effective_days = (1500 - 100) / 96` (15m math)

- **File**: `src/neutralgrid/cli/retrain.py:150`
- **Current**: `timeframe="1h"`
- **Change to**: `timeframe="15m"`

- **File**: `src/neutralgrid/cli/retrain.py:160`
- **Current**: `actual_days = actual_bars / 24.0` (1h math)
- **Change to**: `actual_days = actual_bars / 96.0` (15m math)

- **File**: `src/neutralgrid/cli/retrain.py:217`
- **Current**: `timeframe="1h"`
- **Change to**: `timeframe="15m"`

- **Why**: This is a complete second HMM retrain path. Leaving it at 1h while retrain_hmm.py uses 15m produces artifacts with conflicting timeframe contracts.

### Phase G. Offline Evaluation Forward Horizons

#### G1. CPCV/AUC forward horizon
- **File**: `src/neutralgrid/backtest/evaluate.py:592`
- **Current**: `auc_fwd_horizon = 6  # 6h on 1H data`
- **Change to**: `auc_fwd_horizon = 24  # 6h on 15m data (6 * 4)`
- **Why**: 6 bars on 1h = 6 hours. On 15m, 6 bars = 1.5 hours. The intended forward horizon is 6 hours, which is 24 bars at 15m. Without this fix, the offline evaluation silently tests a 1.5h horizon instead of 6h, producing inflated pass rates.

#### G2. Utility sweep forward horizon
- **File**: `src/neutralgrid/backtest/evaluate.py:1261`
- **Current**: `fwd_horizon: int = 6`
- **Change to**: `fwd_horizon: int = 24`
- Line 1286 comment: Update `"6h on 1H"` -> `"6h on 15m (24 bars)"`

#### G3. Proxy validation forward horizons
- **File**: `src/neutralgrid/backtest/evaluate.py:1490`
- **Current**: `fwd_horizons: tuple = (6, 12)` with comment "default 6h, 12h"
- **Change to**: `fwd_horizons: tuple = (24, 48)`
- Line 1504 docstring: `"Symbol -> 1H DataFrame mapping"` -> `"Symbol -> kline DataFrame mapping"`
- Line 1506 docstring: `"default 6h, 12h"` -> `"default 6h, 12h on 15m (24, 48 bars)"`
- **Why**: (6, 12) bars on 15m = (1.5h, 3h), not the intended (6h, 12h). Must multiply by 4.

### Phase H. Profile Model Duration Change

#### H1. Evidence-backed threshold: 2.7h
- `min_duration_hours=2.7` moves `profile_model` winner count from 31 to 36.
- 2.7h is the shortest duration among current rows that satisfy the existing `profile_model` winner definition on the workbook (KOMAUSDT at 2.70h).
- `3.0h` would give 35 winners (excludes KOMAUSDT). This is a rounded policy choice, not data-derived.
- Decision deferred to user: `2.7h` (data-derived) or `3.0h` (rounded).

#### H2. Files to change
- **File**: `retrain_scanner.py:82` — `default=6.0` -> `default=<chosen value>`
- **File**: `retrain_scanner.py:83` — update help text
- **File**: `src/neutralgrid/scanner/profile_model.py:112` — `min_duration_hours: float = 6.0` -> `<chosen value>`
- **File**: `src/neutralgrid/scanner/pattern_profile.py:276` — `min_duration_hours: float = 6.0` -> `<chosen value>`
- **Why all three**: `retrain_scanner.py` passes the same CLI arg to both `build_profile_from_enhanced_xlsx` (pattern_profile, line 165) and `train_profile_model_from_enhanced_xlsx` (profile_model, line 194). If the library defaults diverge from the CLI default, direct callers get a different threshold.

#### H3. Documented divergence between profile artifacts
- Lowering `min_duration_hours` alone changes `profile_model` winner count but NOT `pattern_profile` winner count.
- **Reason**: `pattern_profile.py:344-346` derives `avg_profit_per_grid = pnl_pct / grids_count` when the column is missing (and `grids_count` exists in the workbook). It then enforces `avg_profit_per_grid >= 0.59%` (from `config.grid.profit_grid_min_pct_static_fallback`). `profile_model.py:191-193` only enforces this floor if the column already exists -- and `avg_profit_per_grid` is NOT a column in `data/new_expired_bots.xlsx`.
- **Consequence**: All 17 latest winners have `avg_profit_per_grid < 0.59%` (highest is KOMAUSDT at 0.477%, corrected from BASEDUSDT per v4.1 `Noether` audit). The `pattern_profile` winner count stays locked at 21 regardless of `min_duration_hours`. Only `profile_model` is affected by the duration change.
- **This is NOT a blocking issue for this plan**: `pattern_profile` produces `similarity_score`, not `profile_proba`. `profile_model` produces `profile_proba`. The MI-weighted score uses `profile_proba`, which IS affected. The `similarity_score` uses the pattern_profile means/stds as reference, which change only if the winner set changes -- and it won't change from duration alone.
- **Separate decision**: If both artifacts must share one winner definition, harmonize `avg_profit_per_grid` preprocessing between the two files first. That is a separate scope item.

---

## 4. Items Removed From Original Plan (With Reasons)

### STRUCK: 3.7.2 -- Add `train_limit_15m` to HMMConfig
- **Reason (Provable False Optionality)**: `train_limit_1h` at config.py:159 has ZERO read sites in the repo. Live training limits come from `retrain_hmm.py --bars` CLI arg and `train_from_market_data(limit=...)`. Canonical mode bypasses fetch-time limit entirely via frozen datasets. Adding `train_limit_15m` creates a second unread config knob.

### STRUCK: 3.1.3 -- Change `--bars` default from 1000 to 4000
- **Reason (Provably Unnecessary)**: `retrain_hmm.py:126-130` shows `--canonical` defaults to True (line 127: `default=True`). In canonical mode, `override_datasets` is injected from frozen Vision store data (retrain_hmm.py:231), and `train_from_market_data()` at train.py:813 uses these directly, bypassing the `limit` parameter entirely. The `--bars` default is not a migration invariant. Help text must be updated to be truthful (15m, not 1h), but the numeric default is a tuning choice.

### STRUCK: Gap 1 -- Keep parallel 1h/15m config knobs "for fallback compatibility"
- **Reason (Provable False Optionality)**: The repo does not have a runtime that needs both 1h and 15m HMM contracts active simultaneously. The migration requires ONE active HMM inference-limit setting tied to the active timeframe. Parallel knobs create confusion without supporting a real use case.

### STRUCK: 3.5.4 -- Staleness threshold analysis (7.0 days)
- **Reason (Not Valid To Change Without Assumptions)**: `_check_model_staleness()` at hmm_regime.py:65 is timestamp-age-based, not timeframe-specific. Claiming 7.0 is correct or incorrect for 15m is empirical policy, not structural necessity. No change.

### STRUCK: 3.4.3 -- Claim that 60 bars is "sufficient for posterior convergence on 15m"
- **Reason (Not Verifiable)**: The minimum of 60 bars at inference.py:572 may be reasonable, but claiming sufficiency for 15m posterior convergence is empirical, not repo-proven. The numeric value stays at 60 (no code evidence to change it), but the plan does not claim convergence proof.

### STRUCK: 3.9 -- Backtest evaluate.py "no structural change needed"
- **Reason (Proven False by Addendum)**: evaluate.py:592 has `auc_fwd_horizon = 6  # 6h on 1H data`. On 15m datasets, 6 bars = 1.5h, not 6h. The intended horizon silently shrinks. Phase G now addresses this.

---

## 5. Execution Order

1. **Phase A** -- Switch HMM artifact/schema/training/inference contract from 1h to 15m (schema.py, features.py, protocols.py, inference.py, hmm_regime.py, train.py)
2. **Phase B** -- Fix canonical interval propagation and interval-aware row-count math (canonical_retrain.py, pipeline.py)
3. **Phase C** -- Replace active HMM inference-limit contract and update all read sites (config.py, scan.py, regime_validator.py)
4. **Phase D** -- Switch runtime HMM feeders to 15m (scan.py, regime_validator.py)
5. **Phase E** -- Repair downstream lineage (enrich_grid_params.py, scanner_integration.py, app.py, database.py)
6. **Phase F** -- Align both retrain entrypoints (retrain_hmm.py, cli/retrain.py)
7. **Phase G** -- Fix offline evaluation forward horizons (evaluate.py)
8. **Phase H** -- Apply profile model duration decision
9. **Run tests** -- `python -m pytest tests/`
10. **Retrain HMM** -- `python retrain_hmm.py`
11. **Retrain profile model** -- `python retrain_scanner.py --min-duration-hours <chosen>`
12. **Run full pipeline** -- `python run_full_pipeline.py`

---

## 6. Files Modified (Complete List)

| # | File | Key Lines | Change |
|---|------|-----------|--------|
| 1 | `src/neutralgrid/models/hmm/schema.py` | 47 | `TIMEFRAME = "1h"` -> `"15m"` |
| 2 | `src/neutralgrid/data/features.py` | 155-262 | Rename `df_1h` -> `df`, update docstrings |
| 3 | `src/neutralgrid/core/protocols.py` | 27, 33 | Rename `df_1h` -> `df` in RegimePredictor protocol |
| 4 | `src/neutralgrid/models/hmm/inference.py` | 545-576 | Rename `df_1h` -> `df`, update error messages |
| 5 | `src/neutralgrid/validation/hmm_regime.py` | 179-196 | Rename `df_1h` -> `df` |
| 6 | `src/neutralgrid/models/hmm/train.py` | 42, 342, 673, 844, 884 | Rename params, `timeframes_used=["15m"]` |
| 7 | `src/neutralgrid/models/hmm/canonical_retrain.py` | 99, 279, 392, 403 | Interval -> 15m, fix row-count math |
| 8 | `src/neutralgrid/data/binance_vision/pipeline.py` | 241 | Interval-aware min_rows |
| 9 | `src/neutralgrid/core/config.py` | 160, 389, 601-607 | `infer_limit`, kline_limits["15m"]=800, cross-check |
| 10 | `src/neutralgrid/scanner/scan.py` | 320, 595 | Feed `df15`, use `infer_limit` |
| 11 | `src/neutralgrid/validation/regime_validator.py` | 259, 291, 294, 942, 964 | Feed 15m to HMM, update error match |
| 12 | `src/neutralgrid/scanner/enrich_grid_params.py` | 1291-1324 | Read HMM lineage from top-level, not tf_1h |
| 13 | `src/neutralgrid/training/scanner_integration.py` | 225-226 | Read HMM lineage from top-level, not tf_1h |
| 14 | `src/neutralgrid/api/app.py` | 331-339 | Stage-based labels, not timeframe-named |
| 15 | `src/neutralgrid/storage/database.py` | 75-76, 177, 318 | Add hmm_regime_passed, update reads |
| 16 | `retrain_hmm.py` | 169, 288 | `timeframe="15m"`, update log text |
| 17 | `src/neutralgrid/cli/retrain.py` | 124, 136, 150, 160, 217 | 15m math, `timeframe="15m"` |
| 18 | `src/neutralgrid/backtest/evaluate.py` | 260, 592, 1261, 1490 | Rename kwarg, `auc_fwd_horizon=24`, `fwd_horizon=24`, `fwd_horizons=(24,48)` |
| 19 | `retrain_scanner.py` | 82-83 | `min_duration_hours` default |
| 20 | `src/neutralgrid/scanner/profile_model.py` | 112 | `min_duration_hours` default |
| 21 | `src/neutralgrid/scanner/pattern_profile.py` | 276 | `min_duration_hours` default |
| 22 | `tests/test_afml_integrations.py` | 358-360 | Update cross-config invariant test to 15m |

**Total**: 22 files. No new files. No new dependencies.

---

## 7. What This Does NOT Change

- `compute_hmm_features()` internal logic (indicator windows are bar-count-based, not wall-clock)
- HMM architecture (4-state GaussianHMM, diag covariance)
- Entropy-adaptive thresholds (range_prob-based, not timeframe-specific)
- Stage B gates (all 4 mandatory gates unchanged)
- Safety invariants (`.claude/rules/safety-invariants.md`)
- Backtest entry point (`run_backtest()` from `btk_unified_runner.py`)
- 1h kline fetch in scan.py (still fetched for non-HMM features)
- Staleness threshold (7.0 days -- policy decision, not structural)
- Minimum inference bar count (60 -- empirical, not structural)
- `train_limit_1h` config field (already dead config, no read sites)
- `pattern_profile` winner count (locked at 21 by avg_profit_per_grid floor)

---

## 8. Validation Checklist (Post-Implementation)

- [ ] `python -m pytest tests/` -- all tests pass
- [ ] `pyright` -- no new type errors (especially protocol conformance)
- [ ] Artifact `metadata.json` shows `"timeframes_used": ["15m"]`
- [ ] Artifact `feature_schema.json` shows `"timeframe": "15m"`
- [ ] Walk-forward `mean_pass_rate >= 0.50` (promotion gate)
- [ ] Config `_validate()` passes (15m cross-config check)
- [ ] `kline_limits["15m"]` >= `hmm.infer_limit` at startup
- [ ] Both retrain paths (retrain_hmm.py and cli/retrain.py) produce 15m artifacts
- [ ] Scan-time and validator HMM both feed 15m data
- [ ] API output does not emit `"1h_regime"` for 15m HMM
- [ ] Profile model updated with new `prior_winner` reflecting lower duration threshold
- [ ] `python run_full_pipeline.py` produces >0 valid candidates
- [ ] TRUUSDT `range_prob` > 0.20 (was 1.49e-08 at 1h, expected ~0.998 at 15m)

---

## Changelog

| Date | Version | Change |
|------|---------|--------|
| 2026-04-09 | v1.0 | Initial plan: 13 files, 8 sections, 8 gaps |
| 2026-04-09 | v1.1 | Codex Verification Addendum appended with 9 claims, 6 consensus corrections |
| 2026-04-10 | v2.0 | Full rewrite after cross-referencing plan v1.0 with Addendum. Changes: |
| | | **Added** (from Addendum, verified against code): |
| | | - `schema.py:47` TIMEFRAME constant (stamps feature_schema.json) |
| | | - `protocols.py:27` RegimePredictor protocol contract (Pyright conformance) |
| | | - `regime_validator.py:259,291,294,942,964` second HMM feeder (was silently staying on 1h) |
| | | - `cli/retrain.py:124,136,150,160,217` second retrain entrypoint (5 lines of 1h math) |
| | | - `evaluate.py:592,1261` forward horizon 6->24 bars (6h semantic on 15m data) |
| | | - `canonical_retrain.py:279` screening row-count math (was 1h-calibrated) |
| | | - `pipeline.py:241` upstream store validation math (was 1h-calibrated) |
| | | - `enrich_grid_params.py:1291-1324` HMM lineage reads from tf_1h.checks |
| | | - `scanner_integration.py:225-226` HMM lineage reads from tf_1h.checks |
| | | - `app.py:331-339` API output timeframe labeling |
| | | - `database.py:75-76,177,318` storage persistence timeframe naming |
| | | - `pattern_profile.py:276` third min_duration_hours default |
| | | - `tests/test_afml_integrations.py:358-360` cross-config invariant test |
| | | **Removed** (proven false optionality or unnecessary): |
| | | - `train_limit_15m` config addition (dead config -- train_limit_1h has 0 read sites) |
| | | - `--bars=4000` default change (canonical mode bypasses fetch-time limit) |
| | | - Parallel 1h/15m config knobs (no runtime needs both active simultaneously) |
| | | - Staleness threshold analysis (policy, not structural) |
| | | - Claim that 60 bars is proven sufficient for 15m convergence (empirical, not repo-proven) |
| | | - Claim that evaluate.py needs "no structural change" (proven false: fwd_horizon shrinks) |
| | | **Corrected**: |
| | | - Profile model threshold from 3.0h to user-choice between 2.7h (data-derived) and 3.0h (rounded) |
| | | - Documented pattern_profile/profile_model winner divergence (avg_profit_per_grid preprocessing) |
| | | - File count from 13 to 22 |
| 2026-04-10 | v2.1 | Agent reconciliation (3 parallel verification agents completed): |
| | | - All 26 original plan line references CONFIRMED by Agent 3 |
| | | - All 9 Addendum critical-missing claims CONFIRMED by Agent 1 |
| | | - All false optionality claims CONFIRMED by Agent 2 |
| | | - Agent 2 found `evaluate.py:260` as keyword call site for `train_hmm_global(per_symbol_dfs_1h=...)` -- added to Phase A7 |
| | | - Agent 1 found `evaluate.py:1490` proxy validation `fwd_horizons=(6,12)` -- added as Phase G3 |
| | | - Agent 2 confirmed ALL callers of train_hmm_global use keyword args (not mixed) -- plan updated |
| 2026-04-10 | v3.0 | Strict `<7h` profile-training addendum appended. Removed invalid `2.7h/3.0h` lower-bound plan, replaced it with a bounded `0 <= duration_hours < 7` training-universe contract, corrected profile/pattern labeling drift, and narrowed v2.0/v2.1 HMM items to the parts that are actually structurally proven. Consensus based on 4 sub-agent audits plus workbook verification. |
| 2026-04-10 | v4.0 | Full architecture audit with 4 parallel agent teams (`Euler`, `Gauss`, `Noether`, `Ramanujan`) plus direct code verification across 22+ source files. Changes: |
| | | **Cross-referenced**: All 59 items verified against exact source lines. Every file:line claim confirmed. |
| | | **Added (new findings missed by v2.0-v3.0)**: |
| | | - `train.py:746` default `timeframe="1h"` in `train_from_market_data()` — drift risk for future callers |
| | | - `evaluate.py:521` fallback `freq="h"` — cosmetic inconsistency |
| | | - `smooth_k=5` and `adaptive_transition_window=48` documented as known semantic changes (tuning, not structural) |
| | | - `min_sequence_length=60` wall-clock change documented (not a bug) |
| | | - Validator non-HMM 1h dependency documented (not a bug — 1h still needed for data quality and volatility checks) |
| | | **Confirmed valid (v2.0-v3.0 items that survived rebuttal)**: |
| | | - All Phase A renames (A1-A8), all Phase B math fixes (B1-B5), all Phase C config changes (C1-C4) |
| | | - All Phase D feed switches (D1-D4), all Phase E lineage repairs (E1-E4) |
| | | - All Phase F retrain fixes (F1-F2), all Phase G forward horizons (G1-G3) |
| | | - All v3.0 bounded-universe items (3.1-3.8) |
| | | **Confirmed false optionality**: `train_limit_15m`, parallel 1h/15m config knobs, free `min_duration_hours` knob |
| | | **Confirmed unnecessary**: `--bars=4000`, staleness threshold change, 60-bar min change |
| | | **Rebutted**: A4 (protocols.py rename) is structural (Pyright), not just cosmetic — upgraded from naming to structural |
| | | **Consolidated**: 59 discrete changes across 22 files, 9-phase execution order, 16-item validation checklist |
| 2026-04-10 | v4.1 | Agent reconciliation (4 parallel audits completed: `Euler`, `Gauss`, `Noether`, `Ramanujan`): |
| | | **`Gauss` CRITICAL**: `regime_validator.py:990` — 15m parse happens AFTER HMM check at `:964`. Moving HMM feed to 15m requires reordering the 15m parse BEFORE the HMM gate, or `NameError`. Item 28 updated. |
| | | **`Ramanujan`**: `enrich_grid_params.py` `tf_1h` reads extend to line 1509 (11 access points), not 1324. Item 31 scope expanded. |
| | | **`Euler` + `Ramanujan` consensus**: `smooth_k=5`, `adaptive_transition_window=48`, `min_sequence_length=60` all silently shrink to 25% wall-clock. Documented in section 9 with counterargument. |
| | | **`Noether` corrections**: Highest derived APG is KOMAUSDT at 0.477%, not BASEDUSDT at 0.46%. Row 137 (OPNUSDT, 10.43h) is >=7h — 15/17 are strictly <7h, not 16/17. |
| | | **`Gauss`**: 7 evaluate.py docstrings reference "1H" — added to Phase 7 cleanup scope. |
| | | **All 4 agents**: No FALSE/INCORRECT items found in the plan. All file:line claims verified. |
| 2026-04-10 | v4.2 | Final rebuttal of the moving-target `v4.0`/`v4.1` bottom plan after 3 fresh parallel audits (`Locke`, `Chandrasekhar`, `Lagrange`) plus a local pyright protocol reproduction. Corrections: fallback is no longer accepted as a bounded policy branch for the `<7h` migration target, duplicate-key fail-fast validation is added as the required safeguard, missing `df_labeled` / trainer-signature / artifact-metadata requirements are added, the database-migration claim is corrected, and exact names or exact carryover values are downgraded from invariants to examples or derived tuning choices. |
| 2026-04-10 | v5.0 | Cross-codebase audit (4 parallel agents: `hmm-core-audit`, `downstream-audit`, `retrain-profile-audit`, `codebase-1h-sweep`). Found 3 missed HMM callers (`inference.py:770` predict_regime(), `new_bot_data_extractor.py:418`, `backfill_training_features.py:296`). Corrected enrich_grid_params.py access count (19, not 11). Classified ~40+ additional 1h references as legitimately non-HMM. File count 22→24. Plan assessed as implementation-ready. AI-coder prompt added. See v5.0 Addendum at end of document. |
| 2026-04-10 | v6.0 | **IMPLEMENTATION COMPLETE.** All 9 phases (A–H + tests) executed across 24 files, 59 discrete changes. |
| | | **Phases A–G**: All HMM schema/training/inference/feeding/lineage/retrain/evaluation changes applied as specified. |
| | | **Phase H**: Bounded training universe `0 <= duration_hours < 7.0` implemented in both `profile_model.py` and `pattern_profile.py`. `retrain_scanner.py` changed from `--min-duration-hours` to `--max-duration-hours 7.0`. Added `duration_band` to ProfileModel dataclass/JSON. Added `df_labeled` (unlabeled rows excluded from threshold computation). Added duplicate `strategy_id` fail-fast validation. Removed APG auto-derivation from `pattern_profile.py`. |
| | | **v5.0 missed callers fixed**: `inference.py:770` `predict_regime()` renamed `df_1h`→`df`, `new_bot_data_extractor.py` fed 15m + bar check raised to 800, `scripts/backfill_training_features.py` fed 15m + lookback increased. |
| | | **Test fixes (6 files)**: MagicMock HMM lineage fields in `test_micro_osc_integration.py` and `test_enrich_grid_params.py`. Kline fixture 100→800 bars in `test_new_bot_data_extractor.py`. Slow-path `label_contract_version` extraction in `unified_training_builder.py`. Test data `label_contract_version` added in `test_unified_training_builder.py` and `test_unified_training_builder_v20260312.py`. `TestBoundedUniverseContract` (6 tests) in `test_afml_integrations.py`. Windows `TemporaryDirectory(ignore_cleanup_errors=True)`. |
| | | **Verification**: 1058/1058 tests pass. Pyright 0 new errors. 3 parallel verification agents — ALL CHECKS PASS. |
| | | **Remaining (require API access)**: `python retrain_hmm.py`, `python retrain_scanner.py --max-duration-hours 7.0`, `python run_full_pipeline.py`. |

---

## v3.0 Addendum -- Strict `<7h` Training Universe and v2.x Rebuttal

### 1. Verified Scope Correction

- The requested profile-model change is **not** "lower the lower bound." The requested trainable universe is: **every row with `0 <= duration_hours < 7` must be eligible as a trainable example**.
- Verified workbook facts from `data/new_expired_bots.xlsx`, `Sheet1`:
  - Total rows: `155`
  - Rows with `0 <= duration_hours < 7`: `41`
  - Rows with `duration_hours >= 7`: `114`
  - Latest cohort (`iloc[138:155]`, rows 138-154 in the current sheet view): `16/17` rows are `<7h`
  - `q75(pnl_pct)` on full sheet: `8.245`
  - `q75(pnl_pct)` on the strict short-horizon subset only: `9.35`
- Current code cannot express this universe:
  - `retrain_scanner.py:80` exposes only `--min-duration-hours`
  - `retrain_scanner.py:163` and `retrain_scanner.py:192` pass that lower-bound-only control into both trainers
  - `src/neutralgrid/scanner/profile_model.py:184` uses `duration_hours >= min_duration_hours`
  - `src/neutralgrid/scanner/pattern_profile.py:350` uses `duration_hours >= min_duration_hours`
  - `src/neutralgrid/scanner/pattern_profile.py:359` fallback also keeps only a lower bound
- Therefore the current `H1` section (`2.7h` versus `3.0h`) is invalid for the stated requirement and is superseded by this addendum.

### 2. Why `2.7h` / `3.0h` Is Invalid

- `2.7h` and `3.0h` are **lower bounds**, not a `<7h` training-universe definition.
- With the current `profile_model` logic:
  - `2.7h` yields `36` winners, but `27` of those winners are still `>=7h`
  - `3.0h` yields `35` winners, but `27` of those winners are still `>=7h`
  - Losers are still taken from the workbook-wide complement at `src/neutralgrid/scanner/profile_model.py:199`, so the long-duration population also remains inside the negative class
- With the strict requested universe (`0 <= duration_hours < 7`):
  - Current `profile_model` logic yields `9` winners out of `41` trainable rows
  - Current `pattern_profile` logic yields `0` winners because it derives and enforces an APG gate that excludes the target cohort
- Conclusion: lowering `min_duration_hours` does **not** make the `<7h` examples the trainable population. It only shifts one lower-bound gate while leaving the universe, the negative class, and the quantile reference population structurally wrong.

### 3. Required Architecture Changes To Make `<7h` Examples Trainable Candidates

#### 3.1. Define the training universe first, not after winner selection

- **Files**: `retrain_scanner.py`, `src/neutralgrid/scanner/profile_model.py`, `src/neutralgrid/scanner/pattern_profile.py`
- **Required contract**: build a single training dataframe `df_train` from `0 <= duration_hours < 7.0` before any quantile, winner, loser, or fallback logic is applied.
- **Reason**: the current contract is lower-bound-only and cannot represent the requested short-horizon population.
- **Minimal non-ambiguous design**:
  - Make the active user-facing control for this path an upper bound of `7.0h`
  - Keep `duration_hours >= 0.0` as a hard safety invariant
  - Do **not** keep an independently tuned `min_duration_hours` knob on this path unless there is a separately proven use case
- **Why this is the simplest correct design**: the requirement is a fixed short-horizon training universe, not a free-form duration-band search problem.

#### 3.2. Compute the PnL quantile on `df_train`, not on the full workbook

- **Files**:
  - `src/neutralgrid/scanner/profile_model.py:167`
  - `src/neutralgrid/scanner/pattern_profile.py:349`
- **Required change**: `pnl_thr` must be computed only from `df_train["pnl_pct"]`.
- **Proof**:
  - Full-sheet `q75(pnl_pct) = 8.245`
  - Strict short-horizon `q75(pnl_pct) = 9.35`
- Using the full workbook quantile after claiming a `<7h` training universe is mathematically inconsistent and silently reintroduces long-horizon influence into label construction.

#### 3.3. Build both winners and losers from the same bounded universe

- **Files**:
  - `src/neutralgrid/scanner/profile_model.py:184`
  - `src/neutralgrid/scanner/profile_model.py:199`
  - `src/neutralgrid/scanner/pattern_profile.py:350`
  - `src/neutralgrid/scanner/pattern_profile.py:359`
- **Required change**:
  - Winners must be selected from `df_train`
  - Losers must also be selected from `df_train`
  - No workbook-wide complement is allowed once the bounded universe is defined
- **Proof of silent drift if not fixed**:
  - Strict short-horizon winners: `9`
  - If losers stay workbook-wide, the class prior is `9/155 = 0.0581`
  - If losers are correctly bounded to the same `<7h` universe, the class prior is `9/41 = 0.2195`
- That is not a cosmetic difference. It changes the learned class balance and moves the negative class median duration away from the target horizon.

#### 3.4. Do not force rows with missing label fields into the loser class

- **Files**:
  - `src/neutralgrid/scanner/profile_model.py:184-199`
  - `src/neutralgrid/scanner/pattern_profile.py:350-355`
- **Required change**: rows inside `df_train` that are missing `profit_factor` or `pnl_pct` must be excluded from supervised label construction instead of defaulting to losers.
- **Proof**:
  - `strategy_id 410926826` (`ONUSDT`) has `duration_hours = 5.90`, `pnl_pct = 15.10`, and `profit_factor = NaN`
  - Under current logic, it cannot be a winner and is therefore forced into the loser set
- That is directly provable label noise. Missing label fields should produce an **unlabeled** row, not a negative label.

#### 3.5. Harmonize `pattern_profile` with `profile_model` on APG handling

- **Files**:
  - `src/neutralgrid/scanner/pattern_profile.py:344`
  - `src/neutralgrid/scanner/pattern_profile.py:355`
  - `src/neutralgrid/scanner/profile_model.py:190`
  - `src/neutralgrid/core/config.py:56`
- **Required change**:
  - If `avg_profit_per_grid` is **absent** in the workbook, do **not** synthesize it from `pnl_pct / grids_count` and then use that surrogate as a winner gate
  - Apply the APG floor only when `avg_profit_per_grid` is explicitly present as a source column
- **Why this is required**:
  - `Sheet1` does **not** contain `avg_profit_per_grid`
  - `pattern_profile.py` derives it and then applies the `0.59` floor from config
  - All `17` rows in the target short-duration cohort fall below that derived floor
  - Result on the strict `<7h` universe: `pattern_profile` currently yields `0` winners while `profile_model` yields `9`
- **Why this is the least-assumption fix**: it removes a derived surrogate gate that the workbook does not explicitly provide, and it makes both artifacts use the same winner definition on the current data.
- **What must not be done here**: do **not** invent a new APG threshold value. No replacement number is repo-proven by the current workbook or code.

#### 3.6. Do not let fallback silently redefine the label rule

- **Files**:
  - `src/neutralgrid/scanner/pattern_profile.py:359-364`
  - `src/neutralgrid/scanner/profile_model.py:196-197`
- **Provably required now**: any fallback that remains must be bounded to `df_train`, not the full workbook.
- **Further point that is not safe to leave implicit**:
  - `pattern_profile.py` fallback currently drops profit-factor, APG, and quantile requirements and switches to top-`pnl_pct`
  - With the current derived-APG rule, strict `<7h` selection would collapse into fallback labeling
- After the APG fix in `3.5`, the current workbook yields `9` short-horizon winners, so fallback should not trigger here. That means a hard-fail versus bounded-fallback decision is a separate policy choice. What is **not** acceptable is silent, undocumented rule mutation.

#### 3.7. Keep deduplication changes out of the mandatory plan unless a real duplicate defect is proven

- **Files audited**:
  - `retrain_scanner.py`
  - `src/neutralgrid/scanner/profile_model.py`
  - `src/neutralgrid/scanner/pattern_profile.py`
- **Verified current state**:
  - `Sheet1` has `155` unique `strategy_id` values
  - `0` duplicate `strategy_id`
  - `0` exact duplicate rows
- **Conclusion**:
  - Symbol caps, per-symbol balancing, or heuristic dedup rules are **not** mandatory on current evidence
  - If future hardening is desired, it should be a validation check for duplicate `strategy_id`, not a silent row-dropping policy

#### 3.8. Add direct tests for the bounded-universe contract

- **Reason**: no direct unit coverage was found for:
  - `retrain_scanner.py`
  - `train_profile_model_from_enhanced_xlsx(...)`
  - `build_profile_from_enhanced_xlsx(...)`
- **Minimum assertions that must exist after implementation**:
  - Only `0 <= duration_hours < 7` rows are eligible for training
  - `pnl_thr` is computed on the bounded universe only
  - Winners and losers come from the same bounded universe
  - Missing `profit_factor` rows are unlabeled, not losers
  - `pattern_profile` and `profile_model` select the same winner IDs when `avg_profit_per_grid` is absent in the workbook

### 4. v2.0 / v2.1 HMM Migration Points That Stay

- The following v2.x items remain structurally correct and should **not** be removed:
  - `src/neutralgrid/models/hmm/schema.py:47` and `schema.py:143` -- artifact/schema timeframe stamping must become truthful for 15m
  - `src/neutralgrid/models/hmm/train.py:673` -- `timeframes_used=["1h"]` must become truthful
  - `src/neutralgrid/models/hmm/canonical_retrain.py:99,279,392,403` -- canonical interval propagation and 1h-calibrated row-count math must be corrected
  - `src/neutralgrid/data/binance_vision/pipeline.py:241` -- store validation row-count math must stop assuming 1h
  - `src/neutralgrid/scanner/scan.py:320` and `scan.py:595` -- runtime HMM feed must stop using the 1h dataframe
  - `retrain_hmm.py:169,288` and `src/neutralgrid/cli/retrain.py:124,136,150,160,217` -- both retrain entrypoints must stop forcing 1h
  - `src/neutralgrid/backtest/evaluate.py:592,1261,1490` -- forward-horizon bar counts must be converted from 1h semantics to 15m semantics

### 5. v2.0 / v2.1 HMM Points That Were Incomplete and Must Be Rewritten

#### 5.1. Validator feed migration was incomplete

- **Files**: `src/neutralgrid/validation/regime_validator.py:939`, `:945`, `:964`, `:990`
- **Problem**: changing only the HMM call site is not enough.
- The validator still:
  - fails on missing 1h data first
  - runs 1h data-quality checks first
  - only parses 15m later
- Therefore the current v2.x `D2` wording was incomplete. The full validator entry contract must be updated so the HMM stage is no longer structurally 1h-first.

#### 5.2. Downstream lineage migration was incomplete

- **Files**:
  - `src/neutralgrid/validation/regime_validator.py:126`
  - `src/neutralgrid/validation/regime_validator.py:441`
  - `src/neutralgrid/validation/regime_validator.py:467`
  - `src/neutralgrid/validation/regime_validator.py:1014`
  - `src/neutralgrid/validation/regime_validator.py:1031`
  - `src/neutralgrid/validation/regime_validator.py:1047`
  - `src/neutralgrid/validation/regime_validator.py:1119`
  - `src/neutralgrid/scanner/enrich_grid_params.py:1291-1324`
  - `src/neutralgrid/training/scanner_integration.py:225-226`
- **Problem**:
  - `ValidationResult` already has some top-level HMM fields
  - but it does **not** currently expose all of `hmm_trained_at_utc`, `regime_conf`, `posterior_mode`, and `persistence_prob`
  - some invalid-but-post-HMM return paths still populate only `tf_1h` / `tf_15m`
- Therefore v2.x could not safely replace all downstream `tf_1h.checks` reads yet. The result contract must be widened first, and the invalid paths must populate the same top-level lineage fields.

#### 5.3. Storage and API migration were incomplete

- **Files**:
  - `src/neutralgrid/storage/database.py:75`, `:177`, `:181`, `:198`, `:318`, `:323`
  - `src/neutralgrid/api/app.py:284`, `:332`, `:333`
- **Problem**:
  - the issue is not only pass/fail column naming
  - storage still routes HMM-stage failure reasons through timeframe-named fields
  - the API still documents and emits `1h`-named HMM output
- **What is structurally required**:
  - remove false 1h naming once HMM becomes 15m-fed
  - update failure-reason routing so it no longer depends on the old 1h carrier
- **What is not uniquely proven**:
  - the exact replacement label shape is not uniquely determined by the repo
  - "stage-based labels" is one valid design, not the only provably correct one

#### 5.4. The v2.x test plan was incomplete

- **Files already proving the old contract**:
  - `tests/unit/test_regime_validator.py:82`, `:105`, `:184`, `:238`, `:292`
  - `tests/unit/test_enrich_grid_params.py:81`, `:265`
  - `tests/unit/test_scanner_integration_v20260320.py:12`
  - `tests/test_afml_integrations.py:358-360`
- **Missing direct coverage**:
  - `/api/validate`
  - `save_bot_run`
  - `save_validation_history`
- Updating only `tests/test_afml_integrations.py` was not enough. The validator, enrichment, scanner-integration, API, and storage contracts all need explicit post-migration assertions.

### 6. Consensus Categorization

#### 6.1. Provable false optionality

- `train_limit_15m` -- `train_limit_1h` at `src/neutralgrid/core/config.py:159` has no live read sites, so adding a 15m sibling would only create a second unread knob
- Parallel 1h/15m HMM config knobs -- no audited runtime requires both active HMM contracts at once
- Keeping a free user-tuned `min_duration_hours` knob on the short-horizon profile-training path -- the verified requirement is a fixed `<7h` training universe, not arbitrary duration-band exploration

#### 6.2. Provably unnecessary items as previously written

- Forcing `--bars=4000` as part of the 15m migration -- canonical retrain defaults bypass fetch-time `limit`, so this is a tuning choice, not a migration invariant
- Treating rename-only cleanup (`df_1h`, protocol parameter names) as mandatory architecture work -- truthful naming is good hygiene, but these renames are not what makes the 15m migration structurally correct
- Treating one exact API/storage replacement name as mandatory -- the repo proves false 1h naming must go away, but it does not prove a single replacement vocabulary

#### 6.3. Items that are not valid to strike without making assumptions

- `schema.py` timeframe stamping and `train.py` `timeframes_used` provenance
- Canonical retrain interval propagation and interval-aware row-count math
- Runtime HMM feed switch away from 1h in `scan.py` and `regime_validator.py`
- Retrain entrypoint fixes in `retrain_hmm.py` and `src/neutralgrid/cli/retrain.py`
- Forward-horizon conversion in `src/neutralgrid/backtest/evaluate.py`
- The bounded short-horizon training-universe contract for `retrain_scanner.py`, `profile_model.py`, and `pattern_profile.py`

### 7. Validation Status For This Addendum

- This addendum is a documentation correction only. No code was edited in this review pass.
- Sub-agent validation completed on the current baseline:
  - `pytest -q tests/unit/test_regime_validator.py tests/unit/test_enrich_grid_params.py tests/unit/test_scanner_integration_v20260320.py -p no:cacheprovider` -> `25 passed`
  - `pytest -q tests/unit/test_regime_validator.py tests/unit/test_enrich_grid_params.py tests/unit/test_scanner_integration_v20260320.py tests/test_afml_integrations.py -p no:cacheprovider` -> `41 passed`
- Consensus result from four audits (`Fermat`, `Raman`, `Pascal`, `Noether`):
  - The 15m HMM migration remains valid in substance but parts of v2.x were incomplete
  - The profile-model duration section in v2.x was not structurally correct and is replaced by the bounded-universe contract above

---

## v4.0 Addendum -- Full Architecture Audit, Rebuttal, and Consolidated Plan

### Audit Methodology

- **Direct code verification**: Every file:line claim in v2.0/v2.1/v3.0 was read and verified against the current codebase. Ground truth was established by reading 22+ source files across `src/neutralgrid/`, `retrain_hmm.py`, `retrain_scanner.py`, and `tests/`.
- **4 parallel agent audits** (`Euler`, `Gauss`, `Noether`, `Ramanujan`):
  - `Euler`: Phases A-D (HMM contract/feed changes, inference parameters)
  - `Gauss`: Phases E-G (downstream lineage, retrain entrypoints, evaluation horizons)
  - `Noether`: Phase H + v3.0 profile model bounded-universe (workbook verification)
  - `Ramanujan`: False optionality claims, missed items, test coverage, codebase-wide 1h residuals
- **Deduplication**: Findings were cross-referenced to eliminate duplicates between agent reports and direct verification.

---

### 1. Items From v2.0/v2.1 That Are Structurally Proven Correct

Every item below was verified by reading the exact source line. Items stay.

#### 1.1 Phase A -- HMM Artifact/Schema/Training Contract

| Item | File:Line | Current Value | Verified |
|------|-----------|---------------|----------|
| A1 | `schema.py:47` | `TIMEFRAME = "1h"` | Stamps `feature_schema.json` via `get_feature_schema()` at `:143`. Must be truthful. |
| A2 | `features.py:155,180,184,187,190-192` | `def compute_hmm_features(df_1h: ...)` | Parameter name `df_1h` appears in 6 internal references. Function is timeframe-agnostic (operates on OHLCV columns). Rename to `df` is truthful naming. |
| A3 | `features.py:243,262` | `def compute_hmm_features_dict(df_1h: ...)` | Wrapper that delegates to A2. |
| A4 | `protocols.py:27,33` | `RegimePredictor.predict(df_1h: ...)` | Protocol contract. If implementation renames but protocol keeps `df_1h`, Pyright protocol conformance breaks. |
| A5 | `inference.py:545,560,563,572-576` | `predict(self, df_1h: ...)`, error messages say "1H" | Error message at `:563` says `"No 1H data"`, at `:574` says `"Insufficient 1H history"`. Must match reality. |
| A6 | `hmm_regime.py:179,196` | `infer_regime(artifact, df_1h, ...)` | Wrapper function parameter name and passthrough. |
| A7 | `train.py:42,55` | `train_hmm_global(per_symbol_dfs_1h: ...)` | All callers use keyword args. Call sites at `train.py:844`, `train.py:884`, `evaluate.py:260`. |
| A7b | `train.py:342,364` | `walk_forward_evaluate(per_symbol_dfs_1h: ...)` | Same keyword pattern. Call site at `train.py:884`. |
| A8 | `train.py:673` | `timeframes_used=["1h"]` | Metadata provenance. Must be truthful for artifact audit trail. |

#### 1.2 Phase B -- Canonical Feeding Interval-Aware Math

| Item | File:Line | Current Value | Proof of Bug |
|------|-----------|---------------|--------------|
| B1 | `canonical_retrain.py:99` | `interval: str = "1h"` | Default parameter. Both callers (`retrain_hmm.py:206`, `cli/retrain.py:383`) rely on this default. Changing to `"15m"` fixes both. |
| B2 | `canonical_retrain.py:279` | `boundary.window_days * 24 * min_coverage_pct` | `* 24` = 1 bar/hour. At 15m, 4 bars/hour. Current formula accepts 25% of required 15m coverage. 180d window: yields 3,888 instead of 15,552 bars. |
| B3 | `canonical_retrain.py:392` | `boundary.window_days * 24 * 0.90` with comment `"180 * 24 * 0.90 = 3888"` | Same 1h math. Comment proves intent is 1h-calibrated. |
| B4 | `canonical_retrain.py:403` | `validate_kline_store(df, interval="1h", ...)` | Hardcoded interval must match fetched data. |
| B5 | `pipeline.py:241` | `min_rows = int(min_years * 365.25 * 24)` with comment `"# 1H bars"` | Function already receives `interval` parameter (verified at function signature). Must derive `bars_per_hour` from `interval`. |

#### 1.3 Phase C -- HMM Inference-Limit Contract

| Item | File:Line | Current Value | Proof |
|------|-----------|---------------|-------|
| C1 | `config.py:160` | `infer_limit_1h: int = 200` | Active inference limit. 200 bars at 1h = 200h. Equivalent at 15m = 800 bars. |
| C2 | `config.py:601-607` | `kline_limits["1h"] >= hmm.infer_limit_1h` | Cross-config invariant must check 15m after migration. |
| C3 | `config.py:389` | `"15m": 300` | 300 bars at 15m = 75h. HMM needs 200h = 800 bars. Scan would truncate HMM input. |
| C4a | `scan.py:320` | `get_config().hmm.infer_limit_1h` | Read site. |
| C4b | `regime_validator.py:291` | `int(_cfg.hmm.infer_limit_1h)` | Read site. |
| C4c | `test_afml_integrations.py:358-360` | `kline_limits["1h"] >= hmm.infer_limit_1h` | Test must mirror production check. |

#### 1.4 Phase D -- Runtime HMM Feeders

| Item | File:Line | Current Value | Proof |
|------|-----------|---------------|-------|
| D1 | `scan.py:595` | `infer_regime(hmm_artifact, df1, ...)` | `df1` is 1h klines (`:518`). `df15` is 15m klines (`:519`). Both already fetched. Only the HMM input changes. |
| D2 | `regime_validator.py:259,290` | `check_hmm_regime(self, symbol, df_1h)` → passes `df_1h` to `infer_regime()` | Parameter name and passthrough. |
| D3 | `regime_validator.py:294` | String match: `"Insufficient 1H history for HMM inference"` | Must match updated message from A5. |
| D4 | `regime_validator.py:942,964` | `df_1h = parse_klines(klines["1h"])` → `check_hmm_regime(symbol, df_1h)` | Feeds 1h to HMM. Must feed 15m. |

#### 1.5 Phase E -- Downstream Lineage Repair

| Item | File:Line | Current Value | Proof |
|------|-----------|---------------|-------|
| E1 | `enrich_grid_params.py:1310-1325,1330-1345` | Reads `regime_conf`, `posterior_mode`, `artifact_version`, `pipeline_version`, `calibration_provenance`, `persistence_prob`, `posteriors` from `vres.tf_1h.checks` | After migration, HMM results stored under `tf_1h` contain 15m-derived data labeled as "1h". Fallback path for partial-failure returns. |
| E2 | `scanner_integration.py:225-237` | Reads `trained_at_utc`, `calibration_provenance` from `vres.tf_1h.checks` | Same false-labeling issue. |
| E3 | `app.py:332-339` | Emits `"1h_regime"`, `details["1h"]` | API consumers receive false metadata. |
| E4 | `database.py:75-76,177,318` | Column `validation_1h_passed`, reads from `tf_1h` | Storage labels 15m-derived HMM results as "1h". |

**v3.0 section 5.2 CONFIRMED**: `ValidationResult` (regime_validator.py:125-177) has top-level fields for `range_prob`, `trend_prob`, `hmm_artifact_version`, `hmm_pipeline_version`, `hmm_calibration_provenance`, `volatility_tier`. But it does NOT have: `hmm_trained_at_utc`, `regime_conf`, `posterior_mode`, `persistence_prob`. These are only in `tf_1h.checks`.

**v3.0 section 5.1 CONFIRMED**: The validator's full `validate_symbol()` method (`:930-1147`) has multiple return paths that ALL store HMM results under `tf_1h`:
  - Success path (`:1122`): `tf_1h=TimeframeResult("1h", True, checks=hmm_check.metrics)`
  - HMM-pass-but-range-fail (`:1017`): same
  - HMM-pass-but-stochastic-fail (`:1050`): same
  - HMM-pass-but-vol-fail (`:1034`): same

#### 1.6 Phase F -- Both Retrain Entrypoints

| Item | File:Line | Current Value | Verified |
|------|-----------|---------------|----------|
| F1a | `retrain_hmm.py:288` | `timeframe="1h"` | Explicit kwarg to `train_from_market_data()`. |
| F1b | `retrain_hmm.py:169` | `"(1H timeframe)"` in log message | |
| F1c | `retrain_hmm.py:72-73` | `"Number of 1H bars per symbol"` help text | |
| F2a | `cli/retrain.py:124` | `bars_needed = window_days * 24 + 100` | 1h math: 24 bars/day. At 15m: 96 bars/day. |
| F2b | `cli/retrain.py:136` | `effective_days = (1500 - 100) / 24` | 1h math. At 15m: `/96`. |
| F2c | `cli/retrain.py:150` | `timeframe="1h"` | `fetch_training_dataset()` call. |
| F2d | `cli/retrain.py:160` | `actual_days = actual_bars / 24.0` | 1h math. At 15m: `/96.0`. |
| F2e | `cli/retrain.py:217` | `timeframe="1h"` | `train_from_market_data()` call. |

#### 1.7 Phase G -- Offline Evaluation Forward Horizons

| Item | File:Line | Current Value | Proof of Bug |
|------|-----------|---------------|--------------|
| G1 | `evaluate.py:592` | `auc_fwd_horizon = 6  # 6h on 1H data` | On 15m data, 6 bars = 1.5h, not 6h. Must be `24`. |
| G2 | `evaluate.py:1261` | `fwd_horizon: int = 6` | Same: 6 bars at 15m = 1.5h. Must be `24`. |
| G3 | `evaluate.py:1490` | `fwd_horizons: tuple = (6, 12)` with docstring `"Symbol → 1H DataFrame mapping"` | (6,12) at 15m = (1.5h, 3h), not (6h, 12h). Must be `(24, 48)`. |

---

### 2. Items From v2.0/v2.1 That Are Provable False Optionality

| # | Item | Proof | Verdict |
|---|------|-------|---------|
| FO-1 | Adding `train_limit_15m` to HMMConfig | `train_limit_1h` at `config.py:159` has exactly 1 definition site and ZERO read sites in the entire codebase (verified by grep). Live training limits come from `retrain_hmm.py --bars` CLI arg and `train_from_market_data(limit=...)`. Canonical mode bypasses fetch-time limit via frozen datasets. Adding a 15m sibling creates a second unread knob. | **CONFIRMED FALSE OPTIONALITY** |
| FO-2 | Parallel 1h/15m HMM config knobs (keeping `infer_limit_1h` alongside new `infer_limit_15m`) | No audited runtime requires both active HMM contracts simultaneously. The migration is a wholesale switch, not a dual-mode system. | **CONFIRMED FALSE OPTIONALITY** |
| FO-3 | Keeping free user-tuned `min_duration_hours` knob on the short-horizon profile-training path | The verified requirement (v3.0 section 1) is a fixed `<7h` training universe. An independently tuned lower-bound knob does not express this requirement and invites drift. | **CONFIRMED FALSE OPTIONALITY** |

---

### 3. Items From v2.0/v2.1 That Are Provably Unnecessary As Written

| # | Item | Proof | Verdict |
|---|------|-------|---------|
| UN-1 | Forcing `--bars=4000` as migration default | `retrain_hmm.py:127` shows `--canonical` defaults to `True`. In canonical mode, `override_datasets` injected from frozen Vision store (`:231`), and `train_from_market_data()` at `train.py:813` uses these directly, bypassing `limit`. Help text must be updated (truthfulness), but numeric default is tuning choice. | **CONFIRMED UNNECESSARY** |
| UN-2 | Claiming staleness threshold 7.0 days needs change | `_check_model_staleness()` at `hmm_regime.py:65` is timestamp-age-based (wall-clock), not bar-count or timeframe-specific. 7.0 days means 7.0 days regardless of whether the model was trained on 1h or 15m data. | **CONFIRMED UNNECESSARY (policy, not structural)** |
| UN-3 | Claiming 60-bar minimum inference threshold at `inference.py:572` needs change | This is an empirical minimum for posterior convergence quality. The plan correctly notes it as "empirical, not repo-proven." Changing it without convergence analysis would be an assumption. | **CONFIRMED UNNECESSARY (empirical, not structural)** |
| UN-4 | Claiming `evaluate.py` needs "no structural change" (v2.0 section 3.9) | This was already struck by v2.0 and corrected in Phase G. Verified: `auc_fwd_horizon=6` at `:592` silently shrinks from 6h to 1.5h on 15m data. | **ALREADY CORRECTED IN v2.0** |

---

### 4. Items NOT Valid To Strike Without Making Assumptions

These items cannot be removed from the plan because doing so would leave a provable 1h residual:

1. **`schema.py:47` TIMEFRAME stamping** — artifact provenance must be truthful
2. **`train.py:673` timeframes_used** — metadata provenance must be truthful
3. **`canonical_retrain.py:99,279,392,403`** — interval propagation and row-count math are mathematically proven wrong at 15m
4. **`pipeline.py:241`** — row-count validation is mathematically proven wrong at 15m
5. **`scan.py:320,595`** — runtime HMM feed must stop using 1h dataframe
6. **`regime_validator.py:259,290,291,294,942,964`** — second HMM feeder must stop using 1h
7. **`retrain_hmm.py:169,288`** and **`cli/retrain.py:124,136,150,160,217`** — both retrain paths hardcode 1h
8. **`evaluate.py:592,1261,1490`** — forward horizons are mathematically proven wrong at 15m (6 bars = 1.5h, not 6h)
9. **`config.py:160,389,601-607`** — inference limit, kline fetch limit, and cross-config check
10. **`test_afml_integrations.py:358-360`** — test must mirror production invariant
11. **Bounded `<7h` training universe** for `retrain_scanner.py`, `profile_model.py`, `pattern_profile.py` — v3.0 sections 3.1-3.6 verified

---

### 5. NEW FINDINGS -- Items Missed By All Previous Versions (v2.0/v2.1/v3.0)

#### 5.1 `smooth_k = 5` inference smoothing window (JUDGMENT CALL)

- **File**: `src/neutralgrid/core/config.py:174`
- **Used at**: `src/neutralgrid/models/hmm/inference.py:622`
- **Semantics**: EMA smoothing over last `k` bars of posterior matrix. At `inference.py:161`: `window = posteriors[-k:]`.
- **At 1h**: 5 bars = 5 hours of posterior smoothing
- **At 15m**: 5 bars = 1.25 hours of posterior smoothing
- **Impact**: Posterior smoothing window shrinks by 4x. Regime detection becomes more reactive (less smoothing), which may increase false positives (rapid regime flipping).
- **Classification**: This is a **tuning parameter**, not a structural invariant. The migration does not mathematically require scaling it by 4 (to 20) because the optimal smoothing window at 15m granularity may genuinely be shorter. However, it changes wall-clock behavior without explicit acknowledgment.
- **Recommendation**: Document in the plan as a known semantic change. Do NOT auto-scale to 20 without empirical validation. The 15m model will retrain and may naturally have different posterior dynamics.

#### 5.2 `adaptive_transition_window = 48` (JUDGMENT CALL)

- **File**: `src/neutralgrid/core/config.py:177`
- **Used at**: `src/neutralgrid/models/hmm/inference.py:628` → `_apply_adaptive_transitions()` at `:225-228` → `_estimate_local_transition_matrix()` at `:182-201`
- **Semantics**: Estimates empirical transition frequencies from last `window` decoded states. At `inference.py:193`: `states = np.argmax(posteriors[-w:], axis=1)`.
- **At 1h**: 48 bars = 48 hours (2 days) of local transition history
- **At 15m**: 48 bars = 12 hours of local transition history
- **Impact**: Local transition estimation window shrinks by 4x. The adaptive transitions will track faster state changes but with less statistical mass per transition estimate (fewer bars = noisier transition matrix).
- **Classification**: **Tuning parameter**, not structural. Scaling to 192 preserves wall-clock equivalence but may not be optimal for 15m dynamics.
- **Recommendation**: Same as 5.1. Document as known semantic change. Do NOT auto-scale without empirical validation.

#### 5.3 `train_from_market_data` default `timeframe="1h"` (STRUCTURAL -- LOW RISK)

- **File**: `src/neutralgrid/models/hmm/train.py:746`
- **Current**: `timeframe: str = "1h"` (function parameter default)
- **Impact**: Both existing callers (`retrain_hmm.py:288`, `cli/retrain.py:217`) explicitly pass `timeframe=`. So this default is currently unreachable. But it is a **drift risk**: any future caller that relies on the default would silently train on 1h data.
- **Classification**: **Structural hygiene, low urgency**. The plan already addresses the callers but misses the function default.
- **Recommendation**: Change default to `timeframe: str = "15m"` when implementing Phase F. This is a one-line addition to the plan, not a new file.

#### 5.4 `evaluate.py:521` fallback frequency (COSMETIC)

- **File**: `src/neutralgrid/backtest/evaluate.py:521`
- **Current**: `ot_valid = pd.date_range("2020-01-01", periods=X_valid.shape[0], freq="h")`
- **Impact**: Fallback path when `open_time` column is missing (extremely unlikely with real data). CPCV splitting uses actual timestamps, not assumed frequencies.
- **Classification**: **Cosmetic**. Will not cause behavioral divergence with real data.
- **Recommendation**: Change `freq="h"` to `freq="15min"` as part of Phase G cleanup. No urgency.

#### 5.5 Validator still requires 1h data for non-HMM checks (NOT A BUG)

- **File**: `src/neutralgrid/validation/regime_validator.py:935-958`
- **Current**: Checks `"1h" not in klines` and runs 1h data quality checks before HMM.
- **Impact**: After migration, the validator still needs 1h klines for `check_data_quality(df_1h, "1h")` at `:945` and for `check_volatility_bounds(df_1h, df_15m)` at `:1029`. These are NOT HMM checks.
- **Classification**: **Not a bug**. The plan correctly states in section 7: "1h kline fetch in scan.py (still fetched for non-HMM features)". The validator similarly still needs 1h for non-HMM purposes.
- **Action**: None. But the plan's Phase D must ensure only the HMM call site switches to 15m, not the data quality and volatility checks that legitimately consume 1h.

#### 5.6 `min_sequence_length = 60` in training and evaluation (JUDGMENT CALL)

- **File**: `src/neutralgrid/models/hmm/train.py:48`, `src/neutralgrid/backtest/evaluate.py:515`
- **At 1h**: 60 bars = 60 hours = 2.5 days minimum per symbol
- **At 15m**: 60 bars = 15 hours minimum per symbol
- **Impact**: Training and evaluation accept shorter wall-clock symbol sequences. With 15m data from 180-day windows (17,280 bars per symbol), no symbol should have fewer than 60 bars unless data is severely corrupted.
- **Classification**: **Judgment call**. The value is a quality floor, not a calibrated constant. At 15m granularity with 180-day windows, 60 bars is well below the expected minimum.
- **Recommendation**: No change needed. Document that wall-clock minimum per symbol drops from 2.5 days to 15 hours, which is still well within the canonical 180-day window.

---

### 6. Rebuttal of v2.0/v2.1 Items -- Consensus

#### 6.1 Phase A renames (A2-A6): Necessary but not the load-bearing part

- v3.0 section 6.2 called these "truthful naming, good hygiene, but not what makes the migration structurally correct."
- **Rebuttal**: Partially correct. The renames are NOT optional for A4 (`protocols.py`). Pyright protocol conformance requires the implementation parameter name to match the protocol parameter name. If `HMMRegimePredictor.predict(df: ...)` is renamed but `RegimePredictor.predict(df_1h: ...)` stays, Pyright will report a protocol violation. The rename at A4 is **structural**, not cosmetic.
- **Verdict**: A4 is structural. A2, A3, A5, A6 are truthful naming that prevents confusion but won't cause runtime failures if missed. A8 is provenance truthfulness.

#### 6.2 Phase E downstream lineage: v3.0 section 5.2 was correct but underspecified

- v3.0 correctly identified that `ValidationResult` doesn't expose all HMM fields at top level.
- **Additional finding**: The `enrich_grid_params.py` reads at `:1291-1345` are a **mixed pattern**: some fields are read from top-level first, then fall back to `tf_1h.checks`. The success path (validator `:1119-1147`) populates top-level fields (`range_prob`, `trend_prob`, `hmm_artifact_version`, `hmm_pipeline_version`). But `regime_conf`, `posterior_mode`, `persistence_prob`, `posteriors`, `trained_at_utc` are ONLY in `tf_1h.checks`.
- **Minimum fix**: Add `hmm_trained_at_utc`, `regime_conf`, `posterior_mode`, `persistence_prob` to `ValidationResult` dataclass, populate them in all return paths, and update downstream reads. This is the same conclusion as v3.0 section 5.2 but now verified with exact field-by-field proof.

#### 6.3 Phase E storage/API: v3.0 section 5.3 was correct

- The exact replacement naming is a design choice. "stage-based labels" is one valid approach. What is **structurally proven wrong** is emitting `"1h_regime"` for 15m-derived HMM results.

---

### 7. v3.0 Profile Model Bounded-Universe -- Verification Summary

All v3.0 sections 3.1-3.8 were verified against code reads:

| v3.0 Section | Claim | Verified |
|--------------|-------|----------|
| 3.1 | Current code uses lower-bound-only `duration_hours >= min_duration_hours` | **YES**: `profile_model.py:185`, `pattern_profile.py:351` |
| 3.2 | `pnl_thr` computed on full workbook, not bounded universe | **YES**: `profile_model.py:167`, `pattern_profile.py:349` — both use unfiltered `df["pnl_pct"]` |
| 3.3 | Losers from full-df complement | **YES**: `profile_model.py:199` — `df.loc[~df.index.isin(df_w.index)]` is workbook-wide complement |
| 3.4 | Missing profit_factor forces profitable bots into losers | **YES**: `profile_model.py:186` — `pf_series >= min_profit_factor` with NaN profit_factor fails, row becomes loser |
| 3.5 | `pattern_profile.py` derives APG when absent and applies 0.59% floor | **YES**: `pattern_profile.py:344-346` — `df["avg_profit_per_grid"] = df["pnl_pct"] / df["grids_count"]` then `:355-356` applies floor |
| 3.6 | Fallback silently redefines label rule | **YES**: `pattern_profile.py:359-364` — drops profit_factor, APG, and quantile requirements |
| 3.7 | No duplicates in workbook | **Deferred to agent Noether** |
| 3.8 | No direct test coverage for profile trainers | **YES**: No test files reference `train_profile_model_from_enhanced_xlsx` or `build_profile_from_enhanced_xlsx` |

---

### 8. Consolidated Modification Plan (v4.0)

This is the final, deduplicated, structurally verified plan. Items are grouped by execution order. Each item is tagged with its category.

#### Phase 1: HMM Schema + Feature Contract (No runtime effect yet)

| # | File:Line | Change | Category |
|---|-----------|--------|----------|
| 1 | `schema.py:47` | `TIMEFRAME = "1h"` → `"15m"` | Provenance |
| 2 | `features.py:155,180,184,187,190-192` | Rename `df_1h` → `df` in `compute_hmm_features()`, update docstring | Naming |
| 3 | `features.py:243,262` | Rename `df_1h` → `df` in `compute_hmm_features_dict()` | Naming |
| 4 | `protocols.py:27,33` | Rename `df_1h` → `df` in `RegimePredictor.predict()`, update docstring | **Structural** (Pyright) |
| 5 | `inference.py:545,560,563,574` | Rename `df_1h` → `df`, update error messages to "kline" | Naming + truthfulness |
| 6 | `hmm_regime.py:179,188,196` | Rename `df_1h` → `df` in `infer_regime()`, update docstring | Naming |
| 7 | `train.py:42,55` | Rename `per_symbol_dfs_1h` → `per_symbol_dfs` in `train_hmm_global()`, update docstring | Naming |
| 8 | `train.py:342,364` | Rename `per_symbol_dfs_1h` → `per_symbol_dfs` in `walk_forward_evaluate()`, update docstring | Naming |
| 9 | `train.py:673` | `timeframes_used=["1h"]` → `["15m"]` | Provenance |
| 10 | `train.py:746` | Default `timeframe: str = "1h"` → `"15m"` in `train_from_market_data()` | Drift prevention |
| 11 | `train.py:844` | Rename kwarg `per_symbol_dfs_1h=dfs` → `per_symbol_dfs=dfs` | Consistency |
| 12 | `train.py:884` | Rename kwarg `per_symbol_dfs_1h=dfs` → `per_symbol_dfs=dfs` | Consistency |
| 13 | `evaluate.py:260` | Rename kwarg `per_symbol_dfs_1h=train_dfs` → `per_symbol_dfs=train_dfs` | Consistency |

#### Phase 2: Canonical Feeding + Store Validation Math

| # | File:Line | Change | Category |
|---|-----------|--------|----------|
| 14 | `canonical_retrain.py:99` | `interval: str = "1h"` → `"15m"` | **Structural** |
| 15 | `canonical_retrain.py:279` | `* 24` → `* 24 * 4` | **Math fix** |
| 16 | `canonical_retrain.py:392` | `* 24 * 0.90` → `* 24 * 4 * 0.90`, update comment | **Math fix** |
| 17 | `canonical_retrain.py:403` | `interval="1h"` → `"15m"` | **Structural** |
| 18 | `pipeline.py:241` | Derive `bars_per_hour` from `interval` param: `{"1h": 1, "15m": 4, "5m": 12, "1m": 60}`. Then `min_rows = int(min_years * 365.25 * 24 * bars_per_hour)` | **Math fix** |

#### Phase 3: Config Inference-Limit Contract

| # | File:Line | Change | Category |
|---|-----------|--------|----------|
| 19 | `config.py:160` | `infer_limit_1h: int = 200` → `infer_limit: int = 800` | **Structural** |
| 20 | `config.py:389` | `"15m": 300` → `"15m": 800` | **Structural** |
| 21 | `config.py:601-607` | `kline_limits["1h"] >= hmm.infer_limit_1h` → `kline_limits["15m"] >= hmm.infer_limit` | **Structural** |
| 22 | `scan.py:320` | `get_config().hmm.infer_limit_1h` → `.infer_limit` | Read site |
| 23 | `regime_validator.py:291` | `_cfg.hmm.infer_limit_1h` → `.infer_limit` | Read site |
| 24 | `test_afml_integrations.py:358-360` | `kline_limits["1h"] >= hmm.infer_limit_1h` → `kline_limits["15m"] >= hmm.infer_limit` | Test |

#### Phase 4: Runtime HMM Feed Switch

| # | File:Line | Change | Category |
|---|-----------|--------|----------|
| 25 | `scan.py:595` | `infer_regime(hmm_artifact, df1, ...)` → `infer_regime(hmm_artifact, df15, ...)` | **Structural** |
| 26 | `regime_validator.py:259,290` | Rename `df_1h` → `df` in `check_hmm_regime()` | Naming |
| 27 | `regime_validator.py:294` | `"Insufficient 1H history"` → `"Insufficient kline history"` | Truthfulness |
| 28 | `regime_validator.py:964,990` | Feed 15m klines to `check_hmm_regime()` instead of 1h. **CRITICAL ORDERING**: The 15m parse currently happens at `:990` which is AFTER the HMM check at `:964`. The 15m parse must be moved BEFORE the HMM check, or the call will fail with `NameError` on the 15m DataFrame. Keep 1h parse at `:942` for data quality (`:945`) and volatility bounds (`:1029`). Add a `"15m" not in klines` guard before HMM, analogous to the existing `"1h" not in klines` guard at `:935`. | **Structural** |

#### Phase 5: Downstream Lineage Repair

| # | File:Line | Change | Category |
|---|-----------|--------|----------|
| 29 | `regime_validator.py:126` (ValidationResult) | Add top-level fields: `hmm_trained_at_utc`, `regime_conf`, `posterior_mode`, `persistence_prob` | **Structural** |
| 30 | `regime_validator.py` all return paths (`:967,1001,1014,1017,1031,1034,1047,1050,1119`) | Populate new top-level HMM fields from `hmm_check.metrics` in every return path that has HMM results. Store HMM result under appropriate label (not falsely under `tf_1h` for 15m-derived data). | **Structural** |
| 31 | `enrich_grid_params.py:1291-1509` | Read `regime_conf`, `posterior_mode`, `persistence_prob`, `posteriors`, `trained_at_utc` from top-level `ValidationResult` fields instead of `tf_1h.checks` fallback. **Scope note** (Ramanujan audit): `tf_1h` reads extend to line 1509 (11 access points total), not just 1324. All must be updated. | **Structural** |
| 32 | `scanner_integration.py:225-237` | Read `trained_at_utc`, `calibration_provenance` from top-level fields instead of `tf_1h.checks` | **Structural** |
| 33 | `app.py:332-339` | Replace `"1h_regime"` with `"hmm_regime"` (or equivalent stage-based label). Replace `details["1h"]` with stage-based key. | **Truthfulness** |
| 34 | `database.py:75-76,177,318` | Add `hmm_regime_passed` column, populate from actual HMM result. Existing `validation_1h_passed` can remain as compatibility shadow. | **Structural** |

#### Phase 6: Retrain Entrypoints

| # | File:Line | Change | Category |
|---|-----------|--------|----------|
| 35 | `retrain_hmm.py:288` | `timeframe="1h"` → `"15m"` | **Structural** |
| 36 | `retrain_hmm.py:169` | `"(1H timeframe)"` → `"(15m timeframe)"` | Truthfulness |
| 37 | `retrain_hmm.py:72-73` | Update help text to reflect 15m semantics | Truthfulness |
| 38 | `cli/retrain.py:124` | `window_days * 24` → `window_days * 24 * 4` | **Math fix** |
| 39 | `cli/retrain.py:136` | `(1500 - 100) / 24` → `(1500 - 100) / 96` | **Math fix** |
| 40 | `cli/retrain.py:150` | `timeframe="1h"` → `"15m"` | **Structural** |
| 41 | `cli/retrain.py:160` | `actual_bars / 24.0` → `actual_bars / 96.0` | **Math fix** |
| 42 | `cli/retrain.py:217` | `timeframe="1h"` → `"15m"` | **Structural** |

#### Phase 7: Offline Evaluation Forward Horizons

| # | File:Line | Change | Category |
|---|-----------|--------|----------|
| 43 | `evaluate.py:592` | `auc_fwd_horizon = 6` → `24`, update comment | **Math fix** |
| 44 | `evaluate.py:1261` | `fwd_horizon: int = 6` → `24` | **Math fix** |
| 45 | `evaluate.py:1490` | `fwd_horizons: tuple = (6, 12)` → `(24, 48)`, update docstrings | **Math fix** |
| 46 | `evaluate.py:521` | `freq="h"` → `freq="15min"` (fallback only) | Cosmetic |

#### Phase 8: Profile Model Bounded-Universe Contract

Per v3.0 sections 3.1-3.8 (all verified):

| # | File:Line | Change | Category |
|---|-----------|--------|----------|
| 47 | `retrain_scanner.py:80` | Replace `--min-duration-hours` with `--max-duration-hours` default `7.0`. Keep `>= 0` as hard safety invariant. | **Structural** |
| 48 | `profile_model.py:167` | Compute `pnl_thr` on `df_train` (bounded universe), not full workbook | **Math fix** |
| 49 | `profile_model.py:184-185` | Filter `df_train` to `0 <= duration_hours < max_duration_hours` BEFORE winner selection | **Structural** |
| 50 | `profile_model.py:199` | Build losers from `df_train` complement within bounded universe, not workbook-wide | **Structural** |
| 51 | `profile_model.py:184-186` | Exclude rows with `NaN` profit_factor from label construction (unlabeled, not loser) | **Label quality** |
| 52 | `pattern_profile.py:276` | Same `max_duration_hours` parameter | **Structural** |
| 53 | `pattern_profile.py:344-346` | Do NOT derive `avg_profit_per_grid` when absent in workbook. Apply APG floor only when column is explicitly present. | **Label quality** |
| 54 | `pattern_profile.py:349` | Compute `pnl_thr` on bounded `df_train`, not full workbook | **Math fix** |
| 55 | `pattern_profile.py:350-356` | Filter and label within bounded universe | **Structural** |
| 56 | `pattern_profile.py:359-364` | Bound fallback to `df_train`, not full workbook | **Structural** |
| 57 | `retrain_scanner.py:163,192` | Pass `max_duration_hours` to both trainers | **Structural** |

#### Phase 9: Tests

| # | File | Change | Category |
|---|------|--------|----------|
| 58 | `test_afml_integrations.py:358-360` | Update cross-config invariant (covered in Phase 3, item 24) | Test |
| 59 | New test assertions (no new file) | Add to existing test files: (a) bounded-universe contract assertions, (b) `pnl_thr` computed on bounded universe, (c) missing `profit_factor` rows are unlabeled, (d) `pattern_profile` and `profile_model` select same winners when APG absent | Test |

---

### 9. Known Semantic Changes That Are NOT Bugs (Document Only)

These parameters change wall-clock behavior at 15m but are **tuning parameters**, not structural invariants. They must be documented but NOT auto-scaled.

**Agent consensus** (`Euler` + `Ramanujan` independently flagged all 3 bar-count parameters below): Both agents agree these are silent wall-clock changes. `Euler` argues they should be scaled by 4x for wall-clock equivalence. `Ramanujan` notes the plan already scales `infer_limit` (200→800) and `auc_fwd_horizon` (6→24) by 4x, creating an inconsistency if these are not scaled. **Counterargument**: `infer_limit` and `auc_fwd_horizon` have provably wrong outputs if not scaled (truncated data, wrong evaluation horizon). The parameters below change quality/reactivity but don't produce mathematically wrong results. The correct values at 15m are empirical and should be determined post-retrain, not assumed to be 4x.

| Parameter | File:Line | At 1h | At 15m | Why not auto-scale |
|-----------|-----------|-------|--------|-------------------|
| `smooth_k = 5` | `config.py:174` | 5h smoothing | 1.25h smoothing | Optimal smoothing at 15m granularity may differ from 1h. The 15m model retrains with different posterior dynamics. |
| `adaptive_transition_window = 48` | `config.py:177` | 48h (2 days) | 12h | Same reasoning. Local transition estimation at higher frequency may genuinely benefit from shorter windows. |
| `min_sequence_length = 60` | `train.py:48`, `evaluate.py:515` | 60h (2.5 days) | 15h | With 180-day windows yielding 17,280 bars per symbol, 60-bar minimum is well below expected data volume. |
| `vol_window = 20`, `ema_period = 20` | `config.py:154-155` | 20h lookback | 5h lookback | These are feature computation windows. The HMM will retrain on 15m features with these windows. Internally consistent. |

---

### 10. Items Explicitly Removed From This Plan (Not Required)

| Item | Why removed |
|------|------------|
| New files | No new files needed. All changes are to existing files. |
| New dependencies | No new packages required. |
| Database migration script | `database.py` uses `CREATE TABLE IF NOT EXISTS` with `ALTER TABLE` for column additions. The new `hmm_regime_passed` column can be added inline. |
| `train_limit_1h` cleanup | Dead config with 0 read sites. Removing it is cleanup, not migration. |
| Staleness threshold change | Wall-clock-based, timeframe-independent. |
| `--bars` default change | Canonical mode bypasses it. |
| Deduplication logic | Workbook has 0 duplicate strategy_ids (verified in v3.0 section 3.7). |
| Symbol caps or per-symbol balancing | No evidence of imbalance in current data. |

---

### 11. Execution Order

1. **Phase 1** (items 1-13) — Schema, features, protocols, training contract
2. **Phase 2** (items 14-18) — Canonical feeding, store validation math
3. **Phase 3** (items 19-24) — Config inference-limit contract
4. **Phase 4** (items 25-28) — Runtime HMM feed switch
5. **Phase 5** (items 29-34) — Downstream lineage repair
6. **Phase 6** (items 35-42) — Retrain entrypoints
7. **Phase 7** (items 43-46) — Evaluation forward horizons
8. **Phase 8** (items 47-57) — Profile model bounded-universe
9. **Phase 9** (items 58-59) — Tests
10. `python -m pytest tests/` — all tests pass
11. `pyright` — no new type errors
12. `python retrain_hmm.py` — retrain HMM on 15m
13. `python retrain_scanner.py --max-duration-hours 7.0` — retrain profile models
14. `python run_full_pipeline.py` — validate end-to-end

---

### 12. Files Modified (Complete List)

| # | File | Items | Changes |
|---|------|-------|---------|
| 1 | `src/neutralgrid/models/hmm/schema.py` | 1 | TIMEFRAME → "15m" |
| 2 | `src/neutralgrid/data/features.py` | 2-3 | Rename `df_1h` → `df`, docstrings |
| 3 | `src/neutralgrid/core/protocols.py` | 4 | Rename `df_1h` → `df` (Pyright structural) |
| 4 | `src/neutralgrid/models/hmm/inference.py` | 5 | Rename `df_1h` → `df`, error messages |
| 5 | `src/neutralgrid/validation/hmm_regime.py` | 6 | Rename `df_1h` → `df`, docstring |
| 6 | `src/neutralgrid/models/hmm/train.py` | 7-12 | Rename params, timeframes_used, function default |
| 7 | `src/neutralgrid/models/hmm/canonical_retrain.py` | 14-17 | Interval → 15m, math fixes |
| 8 | `src/neutralgrid/data/binance_vision/pipeline.py` | 18 | Interval-aware min_rows |
| 9 | `src/neutralgrid/core/config.py` | 19-21 | infer_limit, kline_limits, cross-check |
| 10 | `src/neutralgrid/scanner/scan.py` | 22,25 | infer_limit read, feed df15 |
| 11 | `src/neutralgrid/validation/regime_validator.py` | 23,26-30 | Feed 15m, add top-level fields, update returns |
| 12 | `src/neutralgrid/scanner/enrich_grid_params.py` | 31 | Read from top-level fields |
| 13 | `src/neutralgrid/training/scanner_integration.py` | 32 | Read from top-level fields |
| 14 | `src/neutralgrid/api/app.py` | 33 | Stage-based labels |
| 15 | `src/neutralgrid/storage/database.py` | 34 | Add hmm_regime_passed column |
| 16 | `retrain_hmm.py` | 35-37 | timeframe="15m", log text, help text |
| 17 | `src/neutralgrid/cli/retrain.py` | 38-42 | 15m math, timeframe="15m" |
| 18 | `src/neutralgrid/backtest/evaluate.py` | 13,43-46 | kwarg rename, fwd horizons, fallback freq |
| 19 | `retrain_scanner.py` | 47,57 | max_duration_hours parameter |
| 20 | `src/neutralgrid/scanner/profile_model.py` | 48-51 | Bounded universe, pnl_thr, losers, NaN exclusion |
| 21 | `src/neutralgrid/scanner/pattern_profile.py` | 52-56 | Bounded universe, APG fix, pnl_thr, fallback |
| 22 | `tests/test_afml_integrations.py` | 24,58 | Cross-config invariant, new assertions |

**Total**: 22 files modified. No new files. No new dependencies. 59 discrete changes.

---

### 13. Validation Checklist (Post-Implementation)

- [ ] `python -m pytest tests/` — all tests pass
- [ ] `pyright` — no new type errors (especially protocol conformance for item 4)
- [ ] Artifact `metadata.json` shows `"timeframes_used": ["15m"]`
- [ ] Artifact `feature_schema.json` shows `"timeframe": "15m"`
- [ ] Walk-forward `mean_pass_rate >= 0.50` (promotion gate)
- [ ] Config `_validate()` passes (15m cross-config check)
- [ ] `kline_limits["15m"]` >= `hmm.infer_limit` at startup
- [ ] Both retrain paths produce 15m artifacts
- [ ] Scan-time and validator HMM both feed 15m data
- [ ] API output does not emit `"1h_regime"` for 15m HMM
- [ ] `ValidationResult` top-level fields populated in all return paths
- [ ] Profile model uses bounded `<7h` universe with correct pnl_thr
- [ ] `pattern_profile` and `profile_model` agree on winner set when APG absent
- [ ] `python run_full_pipeline.py` produces >0 valid candidates
- [ ] TRUUSDT `range_prob` > 0.20 on 15m (expected ~0.998)
- [ ] Document `smooth_k` and `adaptive_transition_window` semantic changes in post-migration notes

---

## v4.2 Addendum -- Final Rebuttal Of The Current Bottom Plan

This addendum supersedes the blanket `v4.1` statement that no false or incorrect items remained. The current bottom plan is materially stronger than `v2.0` and `v3.0`, but the fresh audits proved that several points are still incomplete or over-precise.

### 1. Proven Corrections To The `<7h` Training Plan

#### 1.1. `df_train` alone is not enough; the plan also needs `df_labeled`

- Current loser construction in [profile_model.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/scanner/profile_model.py#L199) is still the complement of the winner set.
- Therefore `v4.0` item 51 was incomplete. Saying “`NaN profit_factor` rows are unlabeled” is not enough unless the plan creates a labeled subset before both winner and loser construction.
- The corrected contract is:
  - `df_train`: all rows with `0 <= duration_hours < 7`
  - `df_labeled`: rows inside `df_train` with the required label-defining fields present
  - winners and losers must both be derived from `df_labeled`

#### 1.2. Fallback is not an acceptable policy branch for this migration target

- The live fallbacks still change the label rule:
  - winner fallback in [profile_model.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/scanner/profile_model.py#L195)
  - loser fallback in [profile_model.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/scanner/profile_model.py#L200)
  - pattern fallback in [pattern_profile.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/scanner/pattern_profile.py#L359)
- Fresh workbook verification from the latest audit found the strict short-horizon slice already has enough labeled data:
  - `41` train rows
  - `9` winners
  - `31` losers
  - `1` unlabeled row
- So fallback is provable false optionality for this migration target. The required behavior is:
  - either preserve the exact same rule inside the bounded universe
  - or fail fast after the intended labeling rule is applied

#### 1.3. The trainer-signature and artifact-truthfulness changes were still incomplete

- `v4.0` changed the CLI story, but `profile_model.py` still declares `min_duration_hours` in the trainer signature at [profile_model.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/scanner/profile_model.py#L109).
- `PatternProfile.selection_summary` still persists only `min_duration_hours` at [pattern_profile.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/scanner/pattern_profile.py#L404).
- `ProfileModel.to_json()` still persists no duration-band metadata at all at [profile_model.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/scanner/profile_model.py#L77).
- Therefore the bounded-universe plan is not structurally complete until the trainer contract and saved artifacts tell the truth about the new duration band.

#### 1.4. “Dedup removed” was incomplete without duplicate-key fail-fast validation

- Silent row-dropping dedup remains false optionality and stays removed.
- But the code still does not prove duplicate keys are impossible:
  - null-only check at [pattern_profile.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/scanner/pattern_profile.py#L264)
  - raw merges at [pattern_profile.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/scanner/pattern_profile.py#L317), [pattern_profile.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/scanner/pattern_profile.py#L320), [profile_model.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/scanner/profile_model.py#L137), and [profile_model.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/scanner/profile_model.py#L138)
- Current workbook fact is still `0` duplicate `strategy_id`, but that does not remove the need for a duplicate-key fail-fast check before merge.

### 2. Proven Corrections To The HMM Side Of `v4.0`

#### 2.1. Interval-aware math stays; literal `*4` is not the invariant

- `v4.0` items 15-16 were directionally correct but too literal.
- Because [canonical_retrain.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/models/hmm/canonical_retrain.py#L99) still exposes a general `interval` parameter, the invariant is interval-aware row math, not a hardcoded `*4`.
- A literal `*4` is valid only if that function is intentionally frozen to 15m.

#### 2.2. The lineage plan still missed top-level `posteriors`

- HMM metrics already carry `posteriors` at [regime_validator.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/validation/regime_validator.py#L440).
- `ValidationResult` still does not expose a top-level `posteriors` field at [regime_validator.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/validation/regime_validator.py#L126).
- So `v4.0` was still incomplete: downstream cannot be told to read that field from top level until it exists.

#### 2.3. The compatibility bridge cannot be implied; it must be stated

- The repo is still `tf_1h`-centric across API, storage, and tests:
  - [app.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/api/app.py#L332)
  - [database.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/storage/database.py#L177)
  - [test_regime_validator.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/tests/unit/test_regime_validator.py#L82)
  - [test_enrich_grid_params.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/tests/unit/test_enrich_grid_params.py#L81)
- Therefore `v4.0` was underspecified. The plan must explicitly include:
  - authoritative top-level HMM fields
  - downstream consumer updates in the same migration
  - any temporary legacy carrier treated only as compatibility shadow, not semantic truth

#### 2.4. The database-migration claim in `v4.0` was too strong

- `v4.0` said new HMM-stage storage columns could be added inline because `database.py` already handles column additions.
- That is only true for `bot_metrics` through [database.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/storage/database.py#L123).
- There is no equivalent add-column helper today for `bot_runs` or `validation_history` in [database.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/storage/database.py#L47).
- Corrected requirement:
  - if new HMM-stage columns are introduced, `database.py` needs explicit migration support for those tables too
  - no new file is required
  - the exact helper name or exact new column name is not uniquely proven

### 3. Items That `v4.0` Treated Too Precisely

#### 3.1. Exact names are examples, not invariants

- The repo proves false 1h naming must be removed.
- The repo does **not** prove one unique replacement spelling for:
  - CLI flag `--max-duration-hours`
  - config field `infer_limit`
  - API label `hmm_regime`
  - database column `hmm_regime_passed`

#### 3.2. Exact numeric carryovers are not always structural invariants

- The repo proves the active HMM inference limit must come from the active HMM-timeframe contract, and fetched 15m history must satisfy it.
- It does **not** prove that `800` is the only valid value.
- If the implementation chooses to preserve the old `200h` wall-clock inference span, then `800` is the mathematically derived carryover (`200 * 4 = 800`).
- That makes `800` a derived tuning carryover, not a repo-proven invariant.

#### 3.3. `train_from_market_data()` default timeframe change is drift prevention, not the live blocker

- The default is still `timeframe="1h"` at [train.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/models/hmm/train.py#L746).
- But the active retrain callers still pass the timeframe explicitly at [retrain_hmm.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/retrain_hmm.py#L288) and [cli/retrain.py](c:/Users/cris_/OneDrive/Documents/Christian/Crypto/Neutral%20Grid%20Bots/NEUTRAL%20grid%20bot%20v6.5.7/src/neutralgrid/cli/retrain.py#L217).
- So this stays as drift prevention, not as a current blocker.

### 4. One `v4.0` Upgrade Stays Fully Proven

- `v4.0` upgraded the `protocols.py` rename from naming hygiene to structural.
- I verified that locally with a temporary pyright repro in the workspace:
  - protocol parameter name `df_1h`
  - implementation parameter name `df`
  - pyright reported `Parameter name mismatch`
- So that upgrade stays valid:
  - if the implementation rename happens, the protocol signature must be renamed too

### 5. Final Classification After `v4.2`

#### 5.1. Provable false optionality

- fallback as a secondary labeling policy for the current `<7h` migration target
- silent row-dropping deduplication
- keeping a free lower-bound-only `min_duration_hours` knob on the short-horizon training path

#### 5.2. Provably unnecessary or over-precise as currently written

- treating literal `*4` math as the invariant instead of interval-aware row math
- treating exact names like `infer_limit`, `hmm_regime`, `hmm_regime_passed`, `--max-duration-hours` as mandatory invariants
- presenting `800` as structurally fixed rather than a derived wall-clock carryover choice
- treating the `train_from_market_data()` default timeframe change as a present blocker instead of drift prevention

#### 5.3. Items not valid to strike without making assumptions

- artifact/schema timeframe truthfulness
- canonical interval propagation and row-count correction
- scan-time and validator-time HMM feed switch away from 1h
- top-level HMM lineage widening, including the missing `posteriors` gap
- bounded `<7h` training universe
- APG harmonization when `avg_profit_per_grid` is absent
- unlabeled-row handling for missing `profit_factor`
- duplicate-key fail-fast validation before merge
- explicit downstream compatibility bridge for API, storage, and tests

---

## v5.0 Addendum -- Cross-Codebase Audit, Missed HMM Callers, and Implementation Readiness Assessment

### Audit Methodology

- **4 parallel agent teams** deployed simultaneously:
  - `hmm-core-audit`: Verified schema.py, features.py, protocols.py, inference.py, hmm_regime.py, train.py, canonical_retrain.py, pipeline.py
  - `downstream-audit`: Verified config.py, scan.py, regime_validator.py, enrich_grid_params.py, scanner_integration.py, app.py, database.py, test_afml_integrations.py
  - `retrain-profile-audit`: Verified retrain_hmm.py, cli/retrain.py, evaluate.py, retrain_scanner.py, profile_model.py, pattern_profile.py
  - `codebase-1h-sweep`: Searched ALL .py files for 1h/1H/df_1h/tf_1h/infer_limit_1h/per_symbol_dfs_1h references NOT in the 22-file plan
- **Direct code verification**: Critical findings were re-verified by reading exact source lines after agent reports.

---

### 1. File:Line Verification Result

**All 59 discrete items across 22 files: 100% confirmed.** Every file:line claim in v4.0/v4.2 matches the current codebase. No false claims found.

---

### 2. NEW FINDINGS -- HMM Callers Missed By All Previous Versions (v1.0-v4.2)

#### 2.1 `inference.py:770-797` -- `predict_regime()` standalone function

- **File**: `src/neutralgrid/models/hmm/inference.py:770`
- **Current**: `def predict_regime(df_1h: pd.DataFrame, infer_limit: Optional[int] = None, symbol: Optional[str] = None) -> dict[str, Any]:`
- **Line 779**: Docstring: `"df_1h: DataFrame with 1H kline data"`
- **Line 791**: Docstring example: `>>> result = predict_regime(df_1h, symbol="BTCUSDT")`
- **Line 797**: `result = predictor.predict(df_1h, infer_limit)`
- **Impact**: This is a module-level convenience wrapper around `HMMRegimePredictor.predict()`. Phase A5 only covers the class method at line 545. This function has `df_1h` in its signature and docstring. No callers inside `src/neutralgrid/` (verified by grep), but it IS part of the public module API.
- **Classification**: **Naming + drift prevention**. Must be renamed alongside A5 to prevent any future caller from assuming 1h input.
- **Required change**: Rename `df_1h` -> `df`, update docstring and example.

#### 2.2 `new_bot_data_extractor.py:408-432` -- Live bot feature extraction feeds 1h to HMM

- **File**: `new_bot_data_extractor.py:408-418`
- **Current**:
  - Line 408: `df_1h = klines.get("1h", pd.DataFrame())`
  - Line 417: `if len(df_1h) >= 100:`
  - Line 418: `hmm_result = hmm_predictor.predict(df_1h)`
  - Line 429: `"Insufficient 1h klines for HMM (%d bars, need >=100)"`
- **Impact**: After migration, the HMM artifact is trained on 15m data. This script feeds 1h data to a 15m-trained model. The model will still run (features are computed the same way) but on a different timeframe than it was trained on -- **silent feature-distribution mismatch**. The model was trained on 15m bar statistics (volatility, returns, EMA) and at inference receives 1h bar statistics, which have different distributions.
- **Classification**: **Structural -- silent model/data mismatch**.
- **Required change**: Feed `df_15m` (15m klines) to HMM. Keep `df_1h` for `compute_features()` at line 440 (non-HMM features). Update minimum bar check from `>= 100` to `>= 800` (or `>= hmm.infer_limit`). Update warning message.

#### 2.3 `scripts/backfill_training_features.py:274-296` -- Backfill script feeds 1h to HMM

- **File**: `scripts/backfill_training_features.py:274-296`
- **Current**:
  - Line 274: `df_1h, df_15m, df_5m, funding_rate = await asyncio.gather(...)`
  - Line 275: `self.fetch_historical_klines(symbol, "1h", start_time, lookback_bars=250)`
  - Line 276: `self.fetch_historical_klines(symbol, "15m", start_time, lookback_bars=350)`
  - Line 294: `if predictor is not None and len(df_1h) >= 100:`
  - Line 296: `hmm = predictor.predict(df_1h)`
  - Line 320: `klines_1h=df_1h,` (non-HMM feature computation -- stays)
- **Impact**: Same as 2.2. Feeds 1h to a 15m-trained model. Additionally, `lookback_bars=350` for 15m (line 276) provides only 87.5 hours of data. If HMM needs 800 bars (200 hours), the 15m fetch must increase to `lookback_bars=800` minimum.
- **Classification**: **Structural -- silent model/data mismatch + insufficient 15m fetch**.
- **Required change**: Feed `df_15m` to HMM predict. Increase 15m `lookback_bars` from 350 to at least 800. Update minimum bar check. Keep `df_1h` for `compute_features()` at line 320.

---

### 3. Corrected Count: `enrich_grid_params.py` tf_1h Access Points

- v4.0 item 31 stated "11 access points" extending to line 1509 (per Ramanujan v4.1 audit).
- **Actual count verified by grep**: **19 `tf_1h` references** in lines 1290-1509:
  - Lines 1291, 1293, 1297 (volatility_tier, conditional_tail_risk)
  - Lines 1310, 1311, 1314, 1317, 1320, 1323 (regime_conf, posterior_mode, artifact_version, pipeline_version, calibration_provenance)
  - Lines 1330, 1331 (persistence_prob)
  - Lines 1336, 1337 (hmm_range_prob fallback)
  - Lines 1344, 1345 (posteriors)
  - Lines 1350, 1351 (trained_at_utc)
  - Line 1509 (iteration over tf_1h/tf_15m/tf_5m results)
  - Line 1290 (comment)
- **Correction**: The plan must update all 19 references, not 11.

---

### 4. Classification of ~40+ Additional 1h References Found by Codebase Sweep

The sweep found 1h references in 14 files not in the 22-file plan. These break into two categories:

#### 4.1. Legitimately 1h -- NOT part of this migration (no change needed)

These references compute non-HMM features (ADX, EMA slope, BB width) on 1h klines. The plan correctly states in Section 7: "1h kline fetch in scan.py (still fetched for non-HMM features)."

| File | References | Why 1h stays |
|------|-----------|--------------|
| `scanner/feature_extractor.py:92-184` | `adx_1h`, `ema_slope_1h`, `bb_width_1h`, `bb_width_ratio_1h_15m`, `klines_1h` | These are 1h indicator features for the scanner, not HMM inputs |
| `api/binance_client.py:1047-1139` | `self.get_klines(symbol, "1h", ...)`, `klines["1h"]` | Fetches 1h klines for non-HMM pipeline stages |
| `training/data_generator.py:74-821` | `adx_1h`, `ema_slope_1h`, `bb_width_ratio_1h_15m` columns | Training feature columns derived from 1h indicators |
| `data/market_data.py:258` | `timeframe: str = "1h"` default | General-purpose kline fetch utility, not HMM-specific |
| `data/curator.py:133,240` | `"1h": 3600` timeframe mapping | Utility constant, timeframe-agnostic |
| `data/binance_vision/validate.py:23-337` | Multiple "1h" in validation configs | Vision store validation, supports multiple intervals |
| `data/binance_vision/store.py:32` | Cache path with "1h" | Path example in docstring |
| `data/price_series/ps_rest_backfill.py:30` | `"1h": 3_600_000` | Millisecond mapping utility |
| `training/scanner_integration.py:122-144` | `adx_1h`, `ema_slope_1h`, `bb_width_ratio_1h_15m` | Feature snapshot fields from scanner (non-HMM) |
| `models/artifacts.py:215` | `timeframes_used: Optional[List[str]]` with `["1h", "15m"]` comment | Artifact metadata field definition (already handled by A8 changing the stored value) |
| `_bot_data_extractor_core.py:1180-1192` | `# 1H indicators`, ADX/EMA calculations on df_1h | Non-HMM indicator computation |
| `Live/*/update_expired_bot.py` | `df_1h` with ADX/EMA calculations | Per-bot scripts computing non-HMM indicators |

#### 4.2. HMM callers that MUST change (covered in Section 2 above)

| File | Impact |
|------|--------|
| `new_bot_data_extractor.py:418` | Feeds 1h to HMM -- model/data mismatch |
| `scripts/backfill_training_features.py:296` | Feeds 1h to HMM -- model/data mismatch |
| `inference.py:770-797` | Public API function with `df_1h` -- drift risk |

---

### 5. Reassessment of v4.0/v4.2 Items

#### 5.1. Provable false optionality -- CONFIRMED (no changes)

All v4.2 section 5.1 items remain proven false optionality:
- Fallback as secondary labeling policy for the `<7h` migration target
- Silent row-dropping deduplication
- Free lower-bound-only `min_duration_hours` knob on the short-horizon training path

#### 5.2. Provably unnecessary or over-precise as written -- CONFIRMED (no changes)

All v4.2 section 5.2 items remain proven over-precise:
- Literal `*4` instead of interval-aware row math
- Exact replacement names (`infer_limit`, `hmm_regime`, etc.) treated as mandatory invariants
- `800` presented as structurally fixed instead of derived wall-clock carryover
- `train_from_market_data()` default treated as present blocker instead of drift prevention

#### 5.3. Items not valid to strike -- CONFIRMED + EXPANDED

All v4.2 section 5.3 items remain not strikeable. Adding:
- `inference.py:770-797` `predict_regime()` rename (public module API)
- `new_bot_data_extractor.py:418` HMM feed switch to 15m (silent model/data mismatch)
- `scripts/backfill_training_features.py:296` HMM feed switch to 15m (same)

---

### 6. Updated File Count

| # | File | Change | Status |
|---|------|--------|--------|
| 1-22 | All existing plan files | As specified in v4.0 | **No changes to existing items** |
| 23 | `new_bot_data_extractor.py` | Feed 15m to HMM (line 418), increase bar check, update warning | **NEW -- structural** |
| 24 | `scripts/backfill_training_features.py` | Feed 15m to HMM (line 296), increase 15m lookback_bars, update bar check | **NEW -- structural** |

**Updated total**: 24 files. No new files created. No new dependencies.

Note: `inference.py:770-797` is already in the plan (file #4) -- the `predict_regime()` function is in the same file as the `predict()` method. The plan item scope for inference.py must be expanded to include lines 770-797.

---

### 7. Implementation Readiness Assessment

**Is HMM_CHANGE.md ready to implement?**

**YES, with the 3 corrections above.** The plan is the most thoroughly audited migration document in this repo's history (9 named agent teams across v2.0-v5.0, 100% file:line verification). The structural logic is sound, the math fixes are proven, and the false optionality has been identified and removed.

**Blocking items before implementation begins:**

1. Add `inference.py:770-797` to Phase 1 scope (expand existing item 5)
2. Add `new_bot_data_extractor.py` as Phase 4 item (HMM feed switch)
3. Add `scripts/backfill_training_features.py` as Phase 4 item (HMM feed switch + lookback increase)
4. Correct enrich_grid_params.py access count from 11 to 19 (scope accuracy)

**Non-blocking items (can be resolved during implementation):**
- User decision on profile model: the `<7h` bounded-universe contract is structurally defined but exact `pnl_thr` quantile on the bounded subset and the resulting winner count (9 from current data) should be reviewed after implementation
- Tuning parameters (`smooth_k=5`, `adaptive_transition_window=48`): documented as known semantic changes, post-retrain validation will determine if adjustment is needed

---

### 8. Concise AI-Coder Implementation Prompt

```
TASK: Migrate HMM regime detection from 1h to 15m klines in the neutralgrid codebase.
Two parallel objectives: (A) switch HMM pipeline to 15m, (B) create <7h bounded training universe for profile models.

CONSTRAINT: No new files. No new dependencies. Every change must be traceable to a file:line in HMM_CHANGE.md.

PHASE 1 — HMM SCHEMA + FEATURE CONTRACT (13 items)
Files: schema.py, features.py, protocols.py, inference.py, hmm_regime.py, train.py
- schema.py:47 → TIMEFRAME = "15m"
- Rename all df_1h → df in compute_hmm_features(), compute_hmm_features_dict(),
  RegimePredictor.predict(), HMMRegimePredictor.predict(), infer_regime(),
  predict_regime() (line 770), train_hmm_global(), walk_forward_evaluate()
- train.py:673 → timeframes_used=["15m"]
- train.py:746 → default timeframe="15m"
- Update all keyword call sites: train.py:844,884 and evaluate.py:260
- Update error messages: "1H" → "kline"
- Update all docstrings in changed functions

PHASE 2 — CANONICAL FEEDING + STORE VALIDATION (5 items)
Files: canonical_retrain.py, pipeline.py
- canonical_retrain.py:99 → interval="15m"
- canonical_retrain.py:279 → interval-aware: boundary.window_days * 24 * bars_per_hour * min_coverage_pct
  (derive bars_per_hour from interval parameter, not hardcoded *4)
- canonical_retrain.py:392 → same interval-aware math, update comment
- canonical_retrain.py:403 → validate_kline_store interval="15m"
- pipeline.py:241 → derive bars_per_hour from existing interval param: {"1h":1,"15m":4,"5m":12,"1m":60}

PHASE 3 — CONFIG INFERENCE-LIMIT CONTRACT (6 items)
Files: config.py, scan.py, regime_validator.py, test_afml_integrations.py
- config.py:160 → rename infer_limit_1h to infer_limit, value=800 (derived: 200h * 4 bars/h)
- config.py:389 → kline_limits "15m": 800
- config.py:601-607 → cross-check: kline_limits["15m"] >= hmm.infer_limit
- scan.py:320 → read hmm.infer_limit
- regime_validator.py:291 → read hmm.infer_limit
- test_afml_integrations.py:358-360 → assert kline_limits["15m"] >= hmm.infer_limit

PHASE 4 — RUNTIME HMM FEED SWITCH (6 items)
Files: scan.py, regime_validator.py, new_bot_data_extractor.py, scripts/backfill_training_features.py
- scan.py:595 → infer_regime(hmm_artifact, df15, ...) instead of df1
- regime_validator.py:259,290 → rename df_1h → df in check_hmm_regime()
- regime_validator.py:294 → update error string match to "Insufficient kline history"
- regime_validator.py:964,990 → CRITICAL: move 15m parse BEFORE HMM check (currently 15m
  parse is at :990, after HMM at :964). Add "15m" not in klines guard. Keep 1h parse for
  data quality and volatility bounds.
- new_bot_data_extractor.py:418 → feed df_15m to HMM predict, update bar check to >=800,
  keep df_1h for compute_features() at line 440
- scripts/backfill_training_features.py:296 → feed df_15m to HMM predict, increase 15m
  lookback_bars from 350 to 800, update bar check

PHASE 5 — DOWNSTREAM LINEAGE REPAIR (6 items)
Files: regime_validator.py, enrich_grid_params.py, scanner_integration.py, app.py, database.py
- regime_validator.py:126 → add top-level ValidationResult fields: hmm_trained_at_utc,
  regime_conf, posterior_mode, persistence_prob, posteriors
- regime_validator.py return paths (:967,1001,1014,1017,1031,1034,1047,1050,1119) →
  populate new top-level HMM fields in every path that has HMM results
- enrich_grid_params.py:1291-1509 → update all 19 tf_1h.checks reads to use top-level fields
- scanner_integration.py:225-237 → read from top-level fields
- app.py:332-339 → replace "1h_regime" with stage-based label, replace details["1h"]
- database.py → add hmm_regime_passed column with migration support for relevant tables,
  populate from actual HMM result

PHASE 6 — RETRAIN ENTRYPOINTS (8 items)
Files: retrain_hmm.py, cli/retrain.py
- retrain_hmm.py:288 → timeframe="15m"
- retrain_hmm.py:169 → "(15m timeframe)"
- retrain_hmm.py:72-73 → update help text to 15m semantics
- cli/retrain.py:124 → window_days * 24 * 4 + 100 (or interval-aware)
- cli/retrain.py:136 → (1500 - 100) / 96
- cli/retrain.py:150 → timeframe="15m"
- cli/retrain.py:160 → actual_bars / 96.0
- cli/retrain.py:217 → timeframe="15m"

PHASE 7 — EVALUATION FORWARD HORIZONS (4 items)
Files: evaluate.py
- evaluate.py:592 → auc_fwd_horizon = 24 (6h at 15m)
- evaluate.py:1261 → fwd_horizon = 24
- evaluate.py:1490 → fwd_horizons = (24, 48)
- evaluate.py:521 → freq="15min" (cosmetic fallback)
- Update all docstrings referencing "1H" in evaluate.py

PHASE 8 — PROFILE MODEL BOUNDED UNIVERSE (11 items)
Files: retrain_scanner.py, profile_model.py, pattern_profile.py
- retrain_scanner.py:80 → replace --min-duration-hours with upper-bound control (max 7.0h),
  keep >= 0 as hard safety floor
- profile_model.py + pattern_profile.py → define df_train from 0 <= duration_hours < 7.0
  BEFORE any quantile/winner/loser logic
- Compute pnl_thr on df_train only, not full workbook
- Build winners AND losers from df_train (bounded universe)
- Create df_labeled: exclude rows with NaN profit_factor from label construction
  (unlabeled, not loser)
- pattern_profile.py:344-346 → do NOT derive avg_profit_per_grid when absent in workbook;
  apply APG floor only when column explicitly exists
- Bound any surviving fallback to df_train
- Add duplicate-key fail-fast validation before merges
- Update trainer signatures and artifact metadata to record duration band
- retrain_scanner.py:163,192 → pass max_duration_hours to both trainers

PHASE 9 — TESTS
- test_afml_integrations.py:358-360 → covered in Phase 3
- Add bounded-universe contract tests to existing test files:
  (a) only 0 <= duration_hours < 7 rows eligible
  (b) pnl_thr computed on bounded universe
  (c) NaN profit_factor rows are unlabeled
  (d) pattern_profile and profile_model agree on winners when APG absent

DO NOT CHANGE:
- compute_hmm_features() internal logic (bar-count-based indicators)
- HMM architecture (4-state GaussianHMM, diag covariance)
- Entropy-adaptive thresholds (range_prob-based)
- Stage B gates (4 mandatory gates)
- 1h kline fetch in scan.py (still needed for non-HMM features: ADX, EMA slope, BB width)
- Staleness threshold (7.0 days -- wall-clock, not timeframe-specific)
- Minimum inference bar count (60 -- empirical)
- smooth_k=5, adaptive_transition_window=48 (tuning params, document as known semantic changes)
- Non-HMM 1h references in feature_extractor.py, data_generator.py, binance_client.py, etc.

EXECUTION ORDER:
1. Phases 1-9 in sequence
2. python -m pytest tests/
3. pyright
4. python retrain_hmm.py
5. python retrain_scanner.py --max-duration-hours 7.0
6. python run_full_pipeline.py

VALIDATION CHECKLIST:
- All tests pass, pyright clean
- Artifact metadata.json: timeframes_used=["15m"]
- Artifact feature_schema.json: timeframe="15m"
- Walk-forward mean_pass_rate >= 0.50
- Config _validate() passes (15m cross-check)
- kline_limits["15m"] >= hmm.infer_limit
- Both retrain paths produce 15m artifacts
- Scan-time, validator, new_bot_data_extractor, backfill all feed 15m to HMM
- API does not emit "1h_regime"
- ValidationResult top-level HMM fields populated in all return paths
- Profile model uses bounded <7h universe
- pattern_profile and profile_model agree on winner set
- TRUUSDT range_prob > 0.20 on 15m
```

---

### Changelog Update

| Date | Version | Change |
|------|---------|--------|
| 2026-04-10 | v5.0 | Cross-codebase audit with 4 parallel agent teams. Changes: |
| | | **NEW FINDINGS (missed by v1.0-v4.2)**: |
| | | - `inference.py:770-797` `predict_regime()` standalone function has `df_1h` param -- must rename alongside A5 |
| | | - `new_bot_data_extractor.py:418` feeds 1h to HMM -- silent model/data mismatch after migration |
| | | - `scripts/backfill_training_features.py:296` feeds 1h to HMM + insufficient 15m lookback (350 bars < 800 needed) |
| | | - `enrich_grid_params.py` tf_1h.checks access count corrected from 11 to 19 (verified by grep) |
| | | **CONFIRMED**: All 59 v4.0 items verified against source. 100% file:line match. |
| | | **CONFIRMED FALSE OPTIONALITY**: Same as v4.2 (fallback labeling, row-dropping dedup, free min_duration_hours knob) |
| | | **CONFIRMED UNNECESSARY**: Same as v4.2 (literal *4, exact names as invariants, 800 as fixed, default as blocker) |
| | | **CLASSIFIED**: ~40+ additional 1h references across 14 uncovered files are legitimately non-HMM (ADX, EMA, BB features) and correctly excluded from migration scope |
| | | **FILE COUNT**: Updated from 22 to 24 (added new_bot_data_extractor.py, scripts/backfill_training_features.py) |
| | | **ASSESSMENT**: Plan is implementation-ready after incorporating the 3 new findings |
| | | **ADDED**: Concise AI-coder implementation prompt in Section 8 |
| 2026-04-11 | v5.1 | Implementation completion audit and patch pass with 3 sub-agent reviewers (`Linnaeus`, `Schrodinger`, `Franklin`). Changes: |
| | | **IMPLEMENTED**: `regime_validator.py` now extracts one `hmm_result_fields` map immediately after the 15m HMM check and applies it to every HMM-executed invalid return path plus the valid return. This closes the missing top-level `regime_conf`, `posterior_mode`, `persistence_prob`, `hmm_trained_at_utc`, and `posteriors` lineage gap. |
| | | **IMPLEMENTED**: `database.py` now persists `hmm_regime_passed` from the actual HMM carrier stage (`tf_1h.is_valid`) instead of overall `validation_result.is_valid`; HMM-stage failure reasons now use `HMM:` instead of the false `1H:` label. |
| | | **IMPLEMENTED**: `pattern_profile.py` no longer falls back to top-PnL relabeling when strict bounded winners are fewer than 5. It now fails fast, matching `profile_model.py`, because fallback labeling is confirmed false optionality for this migration target. |
| | | **IMPLEMENTED**: `profile_model.py` and `pattern_profile.py` now validate duplicate `strategy_id` keys before any sheet merge. Duplicate-key failures are not swallowed by the multi-sheet -> single-sheet fallback. Single-sheet Excel reads now close `pd.ExcelFile` handles with context managers. |
| | | **IMPLEMENTED**: stale HMM contract documentation cleaned in `inference.py`, `train.py`, `canonical_retrain.py`, `app.py`, and `evaluate.py` without touching legitimate non-HMM 1h feature references. |
| | | **TESTS ADDED/UPDATED**: bounded-universe tests now use workspace-scoped temp fixtures because Python `tempfile.TemporaryDirectory()` is access-denied in this Windows environment; added fail-fast no-fallback test; added parameterized validator lineage tests for HMM/range/volatility/stochastic invalid paths; added database persistence test for HMM-pass/later-gate-fail and HMM-fail cases. |
| | | **VALIDATION**: `python -m compileall` on patched files passed. Focused suite passed: `37 passed, 5 warnings`. Full pytest passed: `1064 passed, 8 warnings`. `pyright.exe` was blocked by Windows Application Control; Python 3.12 `python -m pyright` ran but failed on repo-wide environment/import-resolution errors (`fastapi`, `httpx`, `numpy`, `pandas`, `hmmlearn`, `sklearn`, `aiosqlite`, etc.), so Pyright is not a green validation signal in this environment. |
| | | **FALSE OPTIONALITY REMOVED**: top-PnL fallback relabeling in `pattern_profile.py`. |
| | | **PROVABLY UNNECESSARY / NOT IMPLEMENTED**: no parallel 1h/15m HMM runtime knobs, no staleness-threshold change, no 60-bar minimum change, no generic non-HMM data-pipeline default change, and no retrain/full-pipeline artifact regeneration. |
| | | **NOT VALID TO STRIKE**: non-HMM 1h feature references remain because they feed ADX/EMA/volatility features, not HMM inference; top-level HMM lineage, duplicate-key fail-fast before merge, `df_labeled`, and the bounded `0 <= duration_hours < 7.0` universe remain required. |

---

## v5.1 Implementation Completion Addendum -- 2026-04-11

### 1. Sub-Agent Review Consensus

Three explicit read-only review agents audited the implementation while the patch was applied locally:

| Agent | Scope | Consensus |
|---|---|---|
| `Linnaeus` | HMM core/training/runtime feed | No verified live HMM path still feeds 1h klines; remaining issues were documentation drift and validator lineage return paths. |
| `Schrodinger` | Downstream lineage/API/storage | `enrich_grid_params.py` and `scanner_integration.py` top-level-first compatibility bridges are valid; missing items were validator HMM-executed invalid returns, storage `hmm_regime_passed`, and stale API doc text. |
| `Franklin` | Profile bounded universe/dedup/classification | Bounded universe, `df_labeled`, APG-only-if-present, and max-duration CLI were implemented; missing items were pre-merge duplicate-key validation and removal of the `pattern_profile.py` fallback label mutation. |

### 2. Implemented Modifications

1. `src/neutralgrid/validation/regime_validator.py`
   - Added `hmm_result_fields` immediately after the 15m HMM check.
   - Applied the same HMM top-level fields to HMM-fail, range-fail, volatility-fail, stochastic-fail, and success returns.
   - Fields covered: `range_prob`, `trend_prob`, artifact/pipeline/calibration metadata, volatility/tail metadata, `regime_conf`, `posterior_mode`, `persistence_prob`, `hmm_trained_at_utc`, `posteriors`, and `regime_utility`.
   - Reason: HMM had executed on 15m data, so downstream consumers must not depend on nested `tf_1h.checks` to recover lineage.

2. `src/neutralgrid/storage/database.py`
   - Changed `hmm_regime_passed` to use the actual HMM carrier stage result: `1 if val_1h and val_1h.is_valid else 0`.
   - Changed HMM-stage failure reason prefix from `1H:` to `HMM:`.
   - Reason: overall validation can fail after HMM passes; storing `0` in that case corrupts the HMM-stage audit signal.

3. `src/neutralgrid/scanner/profile_model.py`
   - Added pre-merge duplicate `strategy_id` fail-fast validation for each multi-sheet source and the single-sheet fallback.
   - Prevented duplicate-key failures from being swallowed by the multi-sheet fallback.
   - Closed single-sheet `pd.ExcelFile` handles via a context manager.
   - Reason: duplicate validation after merge is too late because the merge can already multiply rows and corrupt label construction.

4. `src/neutralgrid/scanner/pattern_profile.py`
   - Added the same pre-merge duplicate `strategy_id` fail-fast validation.
   - Removed the top-PnL fallback when strict bounded winners are fewer than 5; the function now raises `ValueError`.
   - Closed single-sheet `pd.ExcelFile` handles via a context manager.
   - Reason: fallback relabeling is provable false optionality for this migration target because it silently changes the classification rule.

5. Documentation/contract drift cleaned
   - `src/neutralgrid/models/hmm/inference.py`: example now uses `predictor.predict(df)`.
   - `src/neutralgrid/models/hmm/train.py`: training docstring default now says `15m`.
   - `src/neutralgrid/models/hmm/canonical_retrain.py`: canonical store docstring default now says `15m`.
   - `src/neutralgrid/api/app.py`: validation endpoint docstring now describes stage-based HMM on 15m klines.
   - `src/neutralgrid/backtest/evaluate.py`: HMM dataset/horizon docstrings now describe 15m datasets and `24` bars = 6h.

6. Tests
   - `tests/test_afml_integrations.py`: replaced `tempfile.TemporaryDirectory()` in bounded-universe tests with a workspace-scoped temp fixture because Python temp dirs are access-denied in this environment.
   - `tests/test_afml_integrations.py`: added `test_pattern_profile_does_not_fallback_to_top_pnl`.
   - `tests/unit/test_regime_validator.py`: added parameterized coverage for top-level HMM lineage on HMM, range, volatility, and stochastic invalid returns.
   - `tests/unit/test_regime_validator.py`: added database persistence coverage proving HMM pass is retained when a later gate fails, and HMM failure is stored as failed with `HMM:` reason.

### 3. Validation Proof

| Check | Result |
|---|---|
| `python -m compileall ...patched files...` | Passed |
| Focused suite: `pytest -q tests/test_afml_integrations.py::TestBoundedUniverseContract tests/unit/test_regime_validator.py tests/unit/test_enrich_grid_params.py tests/unit/test_scanner_integration_v20260320.py -p no:cacheprovider` | `37 passed, 5 warnings` |
| Full suite: `pytest -q -p no:cacheprovider` | `1064 passed, 8 warnings` |
| Stale scoped marker search: `predict(df_1h)`, `default: "1h"`, `1H: HMM`, `default 6 = 6h on 1H`, `Falling back`, `drop_duplicates`, `min_duration_hours` | No matches in scoped HMM/profile files |
| `pyright.exe` | Blocked by Windows Application Control before analysis |
| `python -m pyright` via Python 3.12 | Ran, but failed on repo-wide environment/import-resolution errors; not used as a code-regression signal |

### 4. Classification

#### 4.1 Provable false optionality removed

- `pattern_profile.py` top-PnL fallback relabeling. It changed the label rule from bounded `profit_factor` + bounded `pnl_thr` winner construction into a secondary top-PnL policy. That violates the verified bounded-universe classification contract and was removed.

#### 4.2 Provably unnecessary items not implemented

- Parallel 1h/15m HMM runtime knobs.
- New staleness-threshold policy.
- New 60-bar minimum inference policy.
- Generic non-HMM `interval="1h"` default changes in shared data utilities.
- Artifact retraining or full pipeline execution as part of this code patch.

#### 4.3 Items not valid to strike

- Non-HMM 1h references used by ADX, EMA slope, BB width, and volatility checks.
- Top-level HMM lineage on every return path where HMM has executed.
- Duplicate-key fail-fast validation before merges.
- `df_labeled` exclusion of missing `profit_factor` / `pnl_pct` rows.
- Bounded profile training universe: `0 <= duration_hours < 7.0`.
