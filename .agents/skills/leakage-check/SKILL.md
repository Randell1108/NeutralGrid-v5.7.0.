---
name: leakage-check
description: Verify the meta-labeler label-leakage guards remain intact. Use before and after any edit to src/neutralgrid/models/meta_labeler.py or to feature lists feeding the meta-labeler. Confirms hlabel is excluded from all feature paths and that the two _KNOWN_LABEL_COLUMNS guard sites in _prepare_features() and train() are present and identical. Read-only.
---

# leakage-check

## Purpose
`hlabel == 3 <=> y == 1` - including `hlabel` in features would be perfect leakage. `safety-invariants.md` requires two guard sites in `meta_labeler.py` to enforce this. This skill confirms both are intact and that no feature list elsewhere reintroduces the leak.

## Procedure
1. Read `src/neutralgrid/models/meta_labeler.py`. Confirm:
   - `_KNOWN_LABEL_COLUMNS` is defined and contains `"hlabel"` (and any other label columns the codebase has added).
   - `_prepare_features()` filters by `_KNOWN_LABEL_COLUMNS`.
   - `train()` filters by `_KNOWN_LABEL_COLUMNS`.
   - Both guard sites still exist; neither has been deleted, commented, or bypassed.
2. Grep the rest of the codebase for any feature list that mentions `hlabel`:
   - `src/neutralgrid/training/unified_training_builder.py` (`EXTRA_META_FEATURES`)
   - `src/neutralgrid/models/meta_labeler.py` (`_INFERENCE_FEATURE_ALIASES` - the live scan_* alias map; ERR-083 removed the dead `_SCAN_TO_FEATURE`)
   - `src/neutralgrid/training/data_generator.py` (`FeatureSnapshot`)
   - `src/neutralgrid/backtest/candidate_pipeline.py` (`_SCANNER_TO_FEATURE`, `TRAINING_OUTPUT_COLUMNS`)
3. Verify version-stamped contract tests still assert leakage prevention:
   ```powershell
   python -m pytest tests/unit/ -k "contract and (leak or hlabel or label)" -v
   ```
4. Report PASS only when:
   - Both guard sites present with identical column sets.
   - No feature list references `hlabel`.
   - Contract test(s) pass.

## Refuse / fail-closed
- If only one of the two guard sites exists, report LEAKAGE-RISK and stop. Do not propose a unilateral fix; the guard sites must be re-added by the engineer who removed them with a CHANGELOG entry explaining why.
- If a feature list mentions `hlabel`, report LEAKAGE-PRESENT with the file:line.

## Verification
See `.claude/rules/skill-verification.md`. Negative test: temporarily insert `"hlabel"` into a feature list; this skill must report LEAKAGE-PRESENT.

<!-- Verified: 2026-07-10 against rolling_180d_20260710_025615 (both guard sites confirmed intact; ERR-079 extended the guard with outcome columns + prefix families and added negative tests in tests/unit/test_afml_compliance_fixes.py) -->
