# Skill Verification Rubric

This file is the canonical reference for confirming that any skill in `.claude/skills/` is doing what it was intended to do, and for diagnosing drift if a skill misbehaves. Each `SKILL.md` references this file under its **Verification** heading rather than restating the criteria.

## When to apply this rubric

- Immediately after authoring or modifying a skill.
- When a skill produces output that contradicts `safety-invariants.md`, `CLAUDE.md`, or current repo state.
- Periodically (suggested: after every HMM rotation, contract-test version bump, or major refactor of `src/neutralgrid/`) to catch drift.
- When a skill names a file, flag, function, or artifact that is reported missing.

## Rubric

### 1. Dry-run check
Invoke the skill with no destructive arguments and inspect what it would do.
- Does the checklist it produces match the invariants enforced by `.claude/rules/safety-invariants.md` and the operational rules in `CLAUDE.md`?
- Does it surface the right pre-conditions before any action?
- Does it explicitly state when it would refuse to act (fail-closed)?

If the checklist is silent on a known invariant the skill is supposed to enforce, the skill is incomplete.

### 2. Reuse check (runbook skills)
Runbook skills must call existing scripts, not re-implement logic.
- Does the skill invoke the canonical entry points (`retrain_hmm.py`, `retrain_meta_labeler.py`, `retrain_scanner.py`, `scripts/backfill_training_features.py`, `run_full_pipeline.py`) rather than spelling out their internals?
- If a script has a flag (e.g. `--default-artifact-version`, `--skip-if-fresh`, `--analyze-only`), does the skill respect the documented authority/precedence rather than introducing its own?

A runbook skill that re-implements logic is a drift hazard the moment the underlying script changes.

### 3. Negative test (invariant-guard skills)
Guard skills must detect a deliberately introduced violation.
- For `verify-feature-pipeline`: temporarily remove a column from one of the three files; the skill must flag the inconsistency.
- For `verify-hmm-lineage`: feed a workbook with two distinct `hmm_artifact_version` values; the skill must refuse / surface the split.
- For `leakage-check`: temporarily insert `hlabel` into a feature list; the skill must flag the leakage.

A guard that does not fail in the negative case is providing false assurance.

### 4. Regression check (any code-touching skill execution)
After running a skill that edits code or artifacts:
- `python -m pytest tests/` must remain green.
- `pyright` must remain clean (basic mode, `pythonVersion=3.11`).
- The skill must not have orphaned imports, dead references, or partial state.

### 5. Drift indicators
A skill is drifting and needs review when any of these are observed:
- It names a file path that no longer exists (e.g. a script was renamed or moved).
- It references a flag or function that has been removed.
- Its checklist includes invariants that have since been relaxed or replaced (e.g. an old `mean_pass_rate` threshold).
- It points to a contract test name with an outdated date stamp.
- It contradicts a memory entry under `C:\Users\cris_\.claude\projects\...\memory\` that captures a more recent decision.

When drift is found: open an `ERR-###` entry referencing the skill, fix or retire the skill, and update any cross-referenced memory.

## Invariants every skill must respect

These are repeated here so a skill author has them in one place. The authoritative source remains `safety-invariants.md`.

- **Leakage**: `hlabel` is never a feature; the two `_KNOWN_LABEL_COLUMNS` guard sites must stay intact.
- **Fail-closed**: missing inputs yield `data_missing:*` rejection codes, not silent NaN substitution.
- **HMM lineage authority**: `--default-artifact-version` is AUTHORITATIVE in backfill (UTILFIX-01); calibration pools must be uniform-lineage (FIXPIPELINE-01).
- **Feature Pipeline Update Rule**: a meta-labeler feature change must touch all three files (`candidate_pipeline.py`, `data_generator.py`, `unified_training_builder.py`).
- **Artifact naming**: HMM artifacts must match `rolling_180d_YYYYMMDD_HHMMSS`; promotion requires walk-forward eval with `mean_pass_rate >= 0.50`.
- **Utility fallback**: absent utility artifact yields `UtilityCalibratorUnavailable` at decision time and `utility_score=NaN` + warning offline; never a silent v0 default.
- **Version constants**: `LABEL_CONTRACT_VERSION`, `FORMULA_VERSION`, `ENGINE_VERSION` come from `src/neutralgrid/core/constants.py` only.

## Recording verification outcomes

When a skill is verified, note the date and HMM artifact version (if relevant) in a one-line comment at the bottom of the skill's `SKILL.md`:

```
<!-- Verified: 2026-05-06 against rolling_180d_20260423_120000 -->
```

When a skill fails verification, do not silently fix and re-verify. Open an `ERR-###` entry, link it from the skill's `SKILL.md`, and only close once the rubric passes end-to-end.
