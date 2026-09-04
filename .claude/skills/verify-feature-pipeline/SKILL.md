---
name: verify-feature-pipeline
description: Enforce the Feature Pipeline Update Rule (safety-invariants.md). Use whenever a meta-labeler feature is added, renamed, or removed, or whenever a contract test fails on missing/extra features. Diffs the feature column lists across the three files that must stay in lockstep - candidate_pipeline.py, data_generator.py, unified_training_builder.py - and reports per-file missing or extra entries. Read-only.
---

# verify-feature-pipeline

## Purpose
Confirm the three files that together define the meta-labeler feature schema are in sync. The Feature Pipeline Update Rule (`.claude/rules/safety-invariants.md`) requires all three to be updated together; a partial update silently leaks NaN/missing features into training rows and is only caught later by the dated contract test.

## Files to inspect
1. `src/neutralgrid/backtest/candidate_pipeline.py` - symbols `_SCANNER_TO_FEATURE`, `TRAINING_OUTPUT_COLUMNS`.
2. `src/neutralgrid/training/data_generator.py` - class `FeatureSnapshot` and its `to_dict()` method.
3. `src/neutralgrid/training/unified_training_builder.py` - symbol `EXTRA_META_FEATURES` - plus `src/neutralgrid/models/meta_labeler.py` - symbol `_INFERENCE_FEATURE_ALIASES` (the LIVE scan_* alias normalization; ERR-083 removed the dead `_SCAN_TO_FEATURE`).

## Procedure
1. Read each of the three files.
2. Extract the feature column lists from each:
   - From `_SCANNER_TO_FEATURE` and `TRAINING_OUTPUT_COLUMNS` (the union of feature columns, excluding identifier and outcome columns).
   - From `FeatureSnapshot` field names and `to_dict()` keys.
   - From `EXTRA_META_FEATURES` and the value side of `_INFERENCE_FEATURE_ALIASES`.
3. Compute the symmetric difference between each pair.
4. Report:
   - Features present in file A but missing in file B (per pair).
   - Whether `to_dict()` keys exactly match `FeatureSnapshot` fields (no rename drift).
5. If anything is out of sync, name the exact line(s) to add/remove and stop. Do not edit.
6. Also run the latest dated contract test:
   ```powershell
   python -m pytest tests/unit/ -k contract -v
   ```
   The contract test is the authoritative tripwire; this skill is the pre-check.

## Output format
A table with one row per feature and columns: `feature | candidate_pipeline | data_generator | unified_training_builder`. Mark each cell present/absent. Highlight any row that is not all-present.

## Refuse / fail-closed
If any of the three files cannot be parsed, stop and report which one. Do not infer.

## Verification
See `.claude/rules/skill-verification.md`. Negative test: remove a column from one file and re-run; this skill must flag the inconsistency.

<!-- Verified: 2026-07-10 (trio audited in session 981a7d2c: all 20 active features consistent; ERR-083 fixed micro_round_trip_cost_pct declaration and repointed site 3 to the live mapping) -->
