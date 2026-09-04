---
name: hmm-rotate
description: Orchestrate an HMM rotation end-to-end. Runs python retrain_hmm.py through the canonical pipeline (build universe -> ensure Vision coverage -> freeze boundary -> screen -> freeze slice -> train -> CPCV evaluate -> promotion gate). Verifies artifact naming, walk-forward mean_pass_rate >= 0.50, identity temperature scaler, and atomic artifact_manifest.json update. After promotion, flags downstream artifacts (meta-labeler, utility) as stale and queues their refits.
---

# hmm-rotate

## Purpose
HMM rotation is the most consequential operation in the pipeline: every downstream artifact (meta-labeler, calibrators, utility) is pinned to the active HMM's lineage. A botched rotation cascades. This skill enforces the canonical pipeline, the promotion gates, and the post-rotation invalidation queue.

## Pre-conditions
- Probe universe + Binance Vision coverage available (the canonical pipeline will fetch + index if absent, but a recent run is faster).
- No in-flight retrains for downstream artifacts (avoid racing).
- Operator has authority to update `artifact_manifest.json` (this is a write to the single source of truth for active HMM).

## Procedure
1. Capture current active HMM (for rollback reference):
   ```powershell
   python -c "import json; print(json.load(open('artifact_manifest.json'))['hmm']['active_version'])"
   ```
2. Run the canonical retrain:
   ```powershell
   python retrain_hmm.py
   ```
   This runs build universe -> freeze boundary -> screen -> freeze slice -> train -> CPCV -> promotion-gate evaluation. Do not bypass any stage.
3. Inspect the candidate artifact:
   - Name matches `rolling_180d_\d{8}_\d{6}`.
   - Metadata records walk-forward eval with `mean_pass_rate >= 0.50`.
   - Metadata is NOT flagged as Binance-1500-bar-truncated (truncated -> no promotion).
   - Temperature scaler is identity (self-labeled HMM).
4. Promotion decision is made by `retrain_hmm.py` per the gates above. Do not override. If gates failed, the candidate stays unpromoted and `artifact_manifest.json` is unchanged - investigate the failure (CPCV pass rate, data truncation, scaler config) rather than forcing promotion.
5. On successful promotion:
   - Confirm `artifact_manifest.json` -> `active_version` updated atomically to the new artifact.
   - Flag the following downstream artifacts as stale and queue refits in this order:
     1. `meta-labeler-refit` skill (meta-labeler must be re-pinned to the new HMM).
     2. Backfill expired-bot pool: `backfill-features` skill (without `--skip-if-fresh` - every row is stale w.r.t. the new HMM).
     3. Utility calibrator (separate workflow; not a skill in this batch).
6. Log a CHANGELOG.md entry (use the `changelog-entry` skill).

## Refuse / fail-closed
- Refuse to manually edit `artifact_manifest.json` if `retrain_hmm.py` did not produce a promoted artifact. Promotion is gated; bypassing is exactly the failure mode that motivated FIXHMM-01.
- Refuse to declare rotation complete until at least the meta-labeler refit has been queued or run. A promoted HMM with stale downstreams is worse than no rotation.

## Critical files
- `retrain_hmm.py`
- `src/neutralgrid/models/hmm/canonical_retrain.py`
- `src/neutralgrid/models/hmm/retrain_orchestration.py`
- `src/neutralgrid/models/hmm/train.py`
- `artifact_manifest.json`
- `artifacts/hmm/<rolling_180d_*>/`

## Verification
See `.claude/rules/skill-verification.md`.

<!-- Verified: 2026-07-10 against rolling_180d_20260710_025615 (exercised live: full canonical rotation + downstream refresh chain ran in session 981a7d2c exactly per this runbook) -->
