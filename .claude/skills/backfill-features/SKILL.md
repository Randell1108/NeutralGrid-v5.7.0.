---
name: backfill-features
description: Run scripts/backfill_training_features.py with the correct HMM lineage authority semantics (UTILFIX-01). Use when re-inferring HMM features on an expired-bot workbook after an HMM rotation, after appending new rows, or to refresh stale lineage before calibration. Always passes --default-artifact-version <active_hmm> as AUTHORITATIVE; decides --skip-if-fresh based on whether the active HMM has rotated since last run. Writes to a fresh output path to avoid the merge-contamination pitfall.
---

# backfill-features

## Purpose
The backfill script is the single point where row-level HMM lineage is brought into agreement with the active HMM. UTILFIX-01 made `--default-artifact-version` AUTHORITATIVE: rows whose merge-preserved lineage differs from the explicit default are invalidated and re-inferenced. This skill captures the right invocation and the decisions around `--skip-if-fresh` and output-path hygiene.

## Inputs
- `--input <path>` (default: `data/new_expired_bots.xlsx`; CSV is supported).
- The current active HMM, resolved from `artifact_manifest.json`.

## Procedure
1. Resolve the active HMM:
   ```powershell
   python -c "import json; print(json.load(open('artifact_manifest.json'))['hmm']['active_version'])"
   ```
2. Choose an output path that does NOT already exist (writing to an existing output reuses prior lineage via the merge-preserve path). Suggested pattern: `data/new_expired_bots_backfilled_<YYYYMMDD>.xlsx`.
3. Choose the point-in-time feature cutoff explicitly:
   - Use `--feature-cutoff-source start_time_utc` for expired-bot rows whose
     strategy start is the feature observation time.
   - Use `--feature-cutoff-source candidate_id_scan_time` for scanner-origin
     backtest rows. Their `start_time_utc` is the later backtest/outcome start,
     while the canonical candidate ID proves the scan time of the feature
     snapshot. Pair it with `--replay-scope hmm_lineage_only` so independent
     scanner-snapshot features are preserved and only HMM plus HMM-transitive
     EV fields are recomputed. Invalid/noncausal candidate timestamps fail closed.
4. Decide on `--skip-if-fresh`:
   - **Use it** for routine ingestions (new rows appended, active HMM unchanged). Stale-lineage rows are still re-inferenced because the merge invalidates them; skipping is bounded to truly fresh rows.
   - **Omit it** immediately after an HMM rotation. (Even with the flag set, every row is stale relative to the new active HMM, so `--skip-if-fresh` is effectively a no-op for that run - but omitting it is the explicit, auditable choice.)
5. Invoke:
   ```powershell
   python scripts/backfill_training_features.py `
     --input <input> `
     --output <fresh_output> `
     --default-artifact-version <active_hmm> `
     --feature-cutoff-source <start_time_utc|candidate_id_scan_time> `
     --replay-scope <full_feature_refresh|hmm_lineage_only> `
     --require-fresh-output `
     [--skip-if-fresh]
   ```
6. After the run, validate with the `verify-hmm-lineage` skill against the output. Verdict must be PASS.
7. If any row's `utility_score` is NaN, that is expected when `artifacts/utility/current.json` is absent (UTILFIX-01 fallback path). Do not silently substitute defaults.

## Refuse / fail-closed
- Refuse to run if `--default-artifact-version` would be empty or unresolved.
- Refuse to overwrite an existing output. Pick a fresh path.
- Refuse to infer scanner-origin rows at their later outcome start; use the
  canonical candidate scan timestamp.
- After the run, refuse to mark complete until `verify-hmm-lineage` returns PASS on the output.

## Critical files
- `scripts/backfill_training_features.py`
- `src/neutralgrid/scanner/feature_extractor.py`
- `src/neutralgrid/training/data_generator.py`
- `src/neutralgrid/validation/utility.py`

## Verification
See `.claude/rules/skill-verification.md`.

<!-- Verified: 2026-07-10 against rolling_180d_20260710_025615 (exercised live: 278-row uniform-lineage backfill to a fresh dated output; stranded memory citation removed per ERR-085) -->
