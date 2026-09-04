# Safety Invariants

## Leakage Prevention
- 'hlabel' is NOT a model feature ('hlabel==3 ⟺ y==1' = information leakage)
- '_KNOWN_LABEL_COLUMNS' enforces this in '_prepare_features()' and 'train()'
- Both guard sites must remain — removing either re-introduces leakage

## Fail-Closed Behavior
- Stage B ('two_stage_selector.py') returns 'StageBResult.approved'
- Gates fail-closed: missing inputs yield 'data_missing' rejection codes (approved=False)
- All 4 mandatory gates must pass; gate 5 (conformal) is optional
- Gate 4 tests regime suitability. Standard archetype: range_prob >= threshold
  (entropy-adaptive). When micro_osc.enabled AND the row carries
  micro_osc_bypass provenance AND micro_osc_score >= min_score (ERR-093):
  Gate 4 tests survival_prob >= min_survival_prob (MC containment). A raw
  micro_osc_score alone must NOT flip the mode — bypass provenance is set at
  enrichment eligibility from scan-phase score AND scan-phase survival.
  Gate 4 remains mandatory in both modes.
- Utility validator: when 'artifacts/utility/current.json' is absent or invalid,
  'UtilityConfig.from_artifact()' raises 'UtilityCalibratorUnavailable'
  (UTILFIX-01). Decision-time callers propagate; offline callers (scan-time
  feature snapshot, training builder, backfill, regime validator) catch and
  emit utility_score=None / NaN with a logger.warning. Never silently
  substitute pinned v0 defaults at the runtime path. The fallback constants
  in 'validation/utility.py' remain in use for:
  (a) UtilityConfig dataclass field defaults (direct UtilityConfig(...)
      instantiation continues to work without arguments), and
  (b) the calibrator's internal G6 baseline at
      'utility_calibrator._build_artifact()' which compares candidate metrics
      against a v0 reference. Neither uses the silent runtime fallback.
- Future Stage B utility gate (if added) MUST use rejection code
  'data_missing:utility', parallel to 'data_missing:tos' /
  'data_missing:range_prob'.

## Backtest Entry Point
- All production code must call 'run_backtest()' from 'btk_unified_runner.py'
- Only unit tests may import 'RealisticGridBacktester' directly
- Training configs must use 'build_training_config()', not raw 'GridConfig'
  - Enforces: continuous funding, taker fees, 2-bar delay

## Realism Profile Authority
- 'legacy' is the canonical default for 'backtest_candidates.py'.
- 'candidate_time_geometric_v1' and 'candidate_time_public_market_v1' are
  shadow-only challengers. They require an explicit isolated output directory
  and must not write 'training_data_*.csv' or 'backtest_results_*.csv' into
  'data/backtest_candidates/', 'data/fastwin_dataset/', or any descendant of
  either canonical training tree.
- Authoritative training admission must classify any explicitly stamped
  non-legacy profile as non-authoritative. An explicit blank or unknown profile
  fails closed; a wholly absent profile column remains a documented compatibility
  path for pre-profile historical files only.
- A challenger result is diagnostic evidence, not authority for model retrain,
  artifact promotion, scanner ranking, or deployment until a materially larger
  bot-disjoint temporal-OOS cohort and event-complete historical execution
  evidence demonstrate consistent superiority.
- Candidate-ID temporal holdout statistics alone do not satisfy that promotion
  gate; downstream profile experiments must retain an explicit shadow-policy
  blocker until the bot-disjoint and event-complete evidence contract exists.

## Grid Mode Authority
- The backtest engine validates BOTH 'arithmetic' and 'geometric' modes
  ('btk_label_contract.validate_engine_result'). Arithmetic remains a valid
  backtest mode and must not be rejected at the engine/contract layer.
- The AUTHORITATIVE TRAINING pool is geometric-only (alignment-v2-geometric-realism).
  The ingestion gate ('unified_training_builder._apply_ingestion_gate') marks any
  row with 'mode != "geometric"' as 'version_gated=True' /
  'source_class="non_authoritative"'. This asymmetry is intentional: arithmetic
  rows are valid for backtest but NOT authoritative for training. Confirmed by
  operator 2026-05-22. Do not "fix" the gate to accept arithmetic.

## Config Integrity
- 'HierarchicalLabelConfig.hurdle_pct' must equal 'BarrierConfig.meta_hurdle_pct'
- Cross-validated at startup via '_validate()' — mismatches raise immediately

## Artifact Naming
- HMM artifacts must follow 'rolling_180d_YYYYMMDD_HHMMSS' format
- Walk-forward evaluation is mandatory before promotion (not optional)
- 'mean_pass_rate >= 0.50' required for 'promote_hmm_version()'
- If data is truncated (Binance 1500-bar limit), artifact will not be promoted

## HMM Lineage Authority
- Every row's 'hmm_artifact_version' must accurately reflect which HMM produced its features.
- Backfill ('scripts/backfill_training_features.py'): '--default-artifact-version',
  when set, is AUTHORITATIVE (UTILFIX-01). Rows whose merge-preserved
  hmm_artifact_version differs from the explicit default have their preserved
  HMM_DERIVED_COLUMNS and HMM_LINEAGE_COLUMNS invalidated, forcing re-inference
  against the explicit version. There is no silent retention of mismatched
  lineage.
- Calibrator inputs must come from a uniform-HMM-lineage workbook;
  cross-HMM-lineage rows in the same calibration pool violate FIXPIPELINE-01.

## Feature Pipeline Update Rule
- When adding a meta-labeler feature, update ALL three files:
  1. '_SCANNER_TO_FEATURE' + 'TRAINING_OUTPUT_COLUMNS' in 'candidate_pipeline.py'
  2. 'FeatureSnapshot' + 'to_dict()' in 'data_generator.py'
  3. 'EXTRA_META_FEATURES' in 'unified_training_builder.py' + a 'scan_<name>'
     alias in '_INFERENCE_FEATURE_ALIASES' in 'meta_labeler.py' (the live
     scan_* normalization applied via 'normalize_inference_feature_frame').
     ERR-083: the former '_SCAN_TO_FEATURE' mapping was dead code and has been
     removed; do not reintroduce it.

## Rate Limits
- Binance rate limits are per-IP, not per-key
- Multiple API keys from same machine share the 1200 weight/min budget

## Version Constants
- 'LABEL_CONTRACT_VERSION', 'FORMULA_VERSION', 'ENGINE_VERSION' all sourced from 'src/neutralgrid/core/constants.py'
- Never duplicate these values — always import from the single source
