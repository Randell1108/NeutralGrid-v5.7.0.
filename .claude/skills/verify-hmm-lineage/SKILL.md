---
name: verify-hmm-lineage
description: Audit a training/calibration workbook (Excel or Parquet) for uniform HMM lineage before any calibrator fit. Use before running utility recalibration or meta-labeler retrain on a pool. Verifies all rows share one hmm_artifact_version and have finite range_prob/trend_prob/persistence_prob. Required by FIXPIPELINE-01 to prevent cross-HMM-lineage pool corruption. Read-only.
---

# verify-hmm-lineage

## Purpose
Calibration and meta-labeler training pools must be uniform-lineage: every row must have been inferenced by the same HMM artifact. FIXPIPELINE-01 documented that mixed-lineage pools silently corrupt calibrators. This skill is the pre-fit gate.

## Inputs
- A workbook path (xlsx or parquet) - usually `data/new_expired_bots.xlsx` or a backfill output.

## Procedure
1. Resolve the active HMM (ERR-085: the previously documented
   `resolve_active_hmm` never existed):
   ```powershell
   python -c "from neutralgrid.models.artifacts import get_active_hmm_version; print(get_active_hmm_version())"
   ```
   (Or read `artifact_manifest.json` at the repo root directly:
   key `hmm.active_version`.)
2. Load the workbook (pandas, single sheet for canonical xlsx).
3. Check the lineage columns are present:
   - `hmm_artifact_version`
   - `hmm_trained_at_utc`
   - `hmm_feature_semantics_version`
4. Group by `hmm_artifact_version` and report row counts per version.
5. Check finiteness of `range_prob`, `trend_prob`, `persistence_prob` per row; report counts of NaN / non-finite values.
6. Verdict:
   - PASS - single `hmm_artifact_version` matching the active HMM, all regime probs finite.
   - SPLIT - multiple `hmm_artifact_version` values; list each with its row count and recommend `python scripts/backfill_training_features.py --input <path> --output <fresh_path> --default-artifact-version <active_hmm>` (always write to a FRESH output path - never merge a backfill into the input workbook in place).
   - STALE - single version but it differs from active HMM; same recommendation.
   - INCOMPLETE - any non-finite regime prob; surface row indices and recommend re-inference.

## Refuse / fail-closed
- Refuse to PASS if even one row has a non-finite `range_prob` / `trend_prob` / `persistence_prob`.
- Refuse to PASS if `hmm_artifact_version` column is missing (this is a structural failure, not a data-only one).

## Verification
See `.claude/rules/skill-verification.md`. Negative test: feed a synthetic workbook with two distinct `hmm_artifact_version` values; this skill must return SPLIT, not PASS.

<!-- Verified: 2026-07-10 against rolling_180d_20260710_025615 (step-1 command executed OK; ERR-085 API fix) -->
