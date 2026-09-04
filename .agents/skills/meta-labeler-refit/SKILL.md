---
name: meta-labeler-refit
description: Retrain the meta-labeler pinned to the active HMM. Use after an hmm-rotate, after a feature schema change (Feature Pipeline Update Rule), or when the latest dated contract test fails on stale lineage. Validates uniform HMM lineage in the training backbone before fitting (FIXPIPELINE-01), runs python retrain_meta_labeler.py, verifies pinning is written to model metadata, and runs the version-stamped contract test.
---

# meta-labeler-refit

## Purpose
Per FIXPIPELINE-01, meta-labeler training rows must come from a uniform HMM lineage matching the active HMM. The retrain script auto-pins to active HMM, but the input pool and the feature schema are operator-controlled and must be checked first. This skill is the wrapper that enforces those checks.

## Pre-conditions
- Active HMM is the intended target (run `hmm-rotate` first if a rotation is in progress).
- Meta-labeler feature schema is in sync across the three pipeline files (run `verify-feature-pipeline`).
- The training pool is uniform-lineage against the active HMM (run `verify-hmm-lineage`).

## Procedure
1. Resolve active HMM:
   ```powershell
   python -c "import json; print(json.load(open('artifact_manifest.json'))['hmm']['active_version'])"
   ```
2. Run `verify-feature-pipeline` skill - must PASS.
3. Run `verify-hmm-lineage` skill against the training pool - must PASS.
   NOTE (FASTWIN-01, snapshot_v20260530_fastwin): the active fast-winner profile
   uses NO HMM-derived features, so its authoritative pool
   (`build_meta_labeler_pool()` -> geometric backtest CSVs) carries no
   `hmm_artifact_version` column and lineage uniformity is N/A. `verify-hmm-lineage`
   and `backfill-features` apply only to HMM-feature profiles. The retrain still
   pins the model metadata to the active HMM for compatibility.
4. For an active FASTWIN retrain, first create a new unbounded full-pool backtest run with `python backtest_candidates.py --fastwin-full-pool --output <fresh_run_dir>`. Do not use the prior dated replay directory as the source of new outcomes.
   ```powershell
   python scripts/backfill_training_features.py --input <fresh_run_dir>/training_data_*.csv --output <fresh_backfilled_csv> --default-artifact-version <active_hmm> --hmm-only --feature-cutoff-source candidate_id_scan_time --replay-scope hmm_lineage_only --require-fresh-output
   python scripts/finalize_fresh_authoritative_meta_pool.py --source <fresh_backfilled_csv> --run-manifest <fresh_run_dir>/backtest_run_manifest.json --output-dir <finalized_fresh_pool> --start-date <pool_start> --end-date <pool_end> --active-hmm-artifact-version <active_hmm>
   python retrain_meta_labeler.py --input data/new_expired_bots.xlsx --backtest-results-dir <finalized_fresh_pool>
   ```
   Historical exact replay is diagnostic only and requires explicit `--allow-historical-replay`. Use `--analyze-only` first if you want a dry diagnostic without overwriting the artifact.
5. Verify pinning:
   - Open the new meta-labeler model metadata.
   - Confirm `hmm_artifact_version` matches active HMM.
   - Confirm `LABEL_CONTRACT_VERSION` and `FORMULA_VERSION` come from `src/neutralgrid/core/constants.py` (no inlined duplicates).
6. Run the version-stamped contract test:
   ```powershell
   python -m pytest tests/unit/ -k "meta_labeler_retrain_contract" -v
   ```
   Find the latest dated contract file (currently `test_meta_labeler_retrain_contract_v20260530.py`) and confirm it passes. If a feature was added, the contract test must be updated separately - do not silently relax it.
7. Log a trial record (the script does this via `trial_tracker`; confirm the entry is written).

## Refuse / fail-closed
- Refuse to refit on a mixed-lineage pool. Run `backfill-features` first.
- Refuse to refit if `verify-feature-pipeline` is not PASS - partial schemas produce silently degraded models.
- Refuse to mark complete until the contract test passes.

## Critical files
- `retrain_meta_labeler.py`
- `src/neutralgrid/models/meta_labeler.py`
- `src/neutralgrid/training/unified_training_builder.py`
- `src/neutralgrid/training/data_generator.py`
- `tests/unit/test_meta_labeler_retrain_contract_v<latest>.py`

## Verification
See `.claude/rules/skill-verification.md`.

<!-- Verified: 2026-07-10 against rolling_180d_20260710_025615 (exercised live: retrain produced artifact 20260710_031243, lineage pinned, contract v20260530 current) -->
