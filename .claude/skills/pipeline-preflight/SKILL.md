---
name: pipeline-preflight
description: Read-only health check before running run_full_pipeline.py (scan -> enrich -> deploy). Confirms the active HMM artifact is valid, the utility calibrator is present (or that decisions will run with utility_score=NaN), pattern profile + profile model are resolvable, and the meta-labeler feature schema is in sync across the three pipeline files. Use before any production scan or before a paper-trading deploy.
---

# pipeline-preflight

## Purpose
Catch the classes of misconfiguration that have historically caused pipeline runs to be discarded (UTILFIX-01, FIXPIPELINE-01, GRIDFIX-001) before they consume a scan window.

## Checks
Run all in order; surface any that fail. Do not edit, just report.

### 1. HMM artifact health
- Read `artifact_manifest.json` -> `hmm.active_version` (nested under the `hmm` key).
- Confirm `artifacts/hmm/<active_version>/` exists.
- Confirm `<active_version>` matches regex `rolling_180d_\d{8}_\d{6}`.
- Confirm metadata records walk-forward eval with `mean_pass_rate >= 0.50`.
- Confirm metadata is not flagged as Binance-1500-bar-truncated.

### 2. Utility calibrator presence
- Confirm `artifacts/utility/current.json` exists and is parseable.
- If absent: this is not a hard fail (offline callers degrade to `utility_score=NaN` per UTILFIX-01), but flag clearly so the operator knows decisions will run without a utility weighting.

### 3. Pattern profile + profile model
- Confirm a pattern profile resolves (preferred: a `current.json` under the pattern-profile artifact dir; fallback: bootstrap profile shipped with the package).
- Confirm a profile model resolves the same way.
- Confirm pattern features match the active `DEFAULT_FEATURES` constant.

### 4. Meta-labeler lineage
- Read meta-labeler model metadata.
- Confirm `hmm_artifact_version` recorded in the meta-labeler matches `active_version` from step 1.
- If mismatch: STALE-METALABELER. Recommend `meta-labeler-refit` skill.

### 5. Feature pipeline sync
- Delegate to `verify-feature-pipeline` (or run the same diff inline). PASS only if all three files are in sync.

### 6. Dependency check
```powershell
python scripts/check_deps.py
```

## Output
- One-line verdict: PASS / WARN / FAIL.
- Per-check status with the exact remedy for each FAIL/WARN.
- If any FAIL, recommend the operator does not launch `run_full_pipeline.py` until cleared.

## Refuse / fail-closed
- If `artifact_manifest.json` does not exist or has no `hmm.active_version`, FAIL immediately. The pipeline cannot run without an active HMM.
- If the meta-labeler is pinned to an HMM version that is no longer present on disk, FAIL - do not silently fall back.

## Verification
See `.claude/rules/skill-verification.md`.

<!-- Verified: 2026-07-10 against rolling_180d_20260710_025615 (manifest key path corrected to hmm.active_version per ERR-085; checks match current artifact layout) -->
