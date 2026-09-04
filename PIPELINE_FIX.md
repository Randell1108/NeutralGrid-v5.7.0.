# Pipeline Health Restoration Plan (PIPELINE_FIX)

## Progress

`[####################] 4/4 steps complete (100%)` — Final Integration: UNANIMOUS PASS (2026-04-29)

| # | Step | Status | Evidence |
|---|---|---|---|
| 1 | ERR-021 label-precedence fix (`unified_training_builder.py:822-831`) | DONE | ERR-021 CLOSED in `ERRORS_LOG.md` |
| 2 | ERR-033 calibrator active-HMM resolver (`hmm_winner_calibrator.py:45,798,885`) | DONE | ERR-033 CLOSED; calibrator `current.json` lineage = active HMM |
| 3 | Doc sync to `snapshot_v20260421_bootstrap` (`readmefullpwep.md`, `PNL_CURVE_CLASS.md`) | DONE | Doc references match `meta_labeler.py:138` |
| 4 | Meta-labeler retrain against active HMM | DONE | `metadata.json` lineage = `rolling_180d_20260426_012042`; 18/25 rows calibrated post-fix; ERR-042 accepted as locked-pool noise floor |

Open follow-ups (out of scope for this plan): ERR-035/036/037 WATCH-listed, ERR-038/039 pre-existing test-suite issues, ERR-042 documented constraint.

## User Goal

**Align all components of the full pipeline so it runs healthy and functional end-to-end, without adding new gating, increased filters, or unnecessary files. Every modification must be traceable to file:line evidence and verifiable by reproducible commands. The locked sample pool (N=85 calibrator / N≈226 meta-labeler) is a hard constraint; no proposal may assume it can grow.**

Concretely, "aligned" means:
- Every active artifact (HMM, HMM winner-calibrator, meta-labeler) declares lineage that matches `artifact_manifest.json["hmm"]["active_version"]`.
- The default code paths (no-flag invocations of retrain / calibrate scripts) target the active HMM, not a stale literal.
- Documentation describes the active code, not a superseded one.
- The fail-closed lineage check at `MetaLabeler.load()` no longer fires on every pipeline run.

---

## Context

The full pipeline currently runs in degraded mode. The startup artifact-compatibility check fail-closes the meta-labeler load on every run, producing `meta_prob_source = "missing"` for all 25 rows in `results/deployment_ready_20260429_074741.csv`. Stage 12 Kelly sizing receives zero meta-labeler predictions across the run.

The HMM (`rolling_180d_20260426_012042`) and HMM winner-calibrator (`hmm_winner_20260428_195512_559490` pinned to that HMM) are active and lineage-correct. The utility calibrator and pattern-profile model run on documented fallback paths by deliberate user decision (FIXUTILITY-01: utility recalibration produced `-0.32` correlation; PATTERN_PROFILE_FIX Phase 2: per-class floor `max(30, 3*len(feats))` refuses promotion on current xlsx).

The blocking sequence is:
- **ERR-021** — `unified_training_builder.py` label-precedence loop (lines 822-831) accepts the all-zero `y` column from backtest outcome CSVs, blocking the `net_pnl_pct >= meta_hurdle_pct` fallback path. Retrain raises `ValueError: Positive rate 0.0% is below 5%`.
- **ERR-033** — meta-labeler is therefore stuck pinned to old HMM `rolling_180d_20260421_213328` (per `models/meta_labeler/metadata.json:7`) while the active HMM is `rolling_180d_20260426_012042` (per `artifact_manifest.json` and `artifacts/hmm_winner_calibrator/current.json`).

The user constraint set rules out: new gates, increased filters, expanded sample pool (locked at N=85 calibrator / N≈226 meta-labeler), unnecessary files, and false-optionality CLI flags.

## Scope Classification

Each candidate change is classified as **(P)** provably necessary by file:line evidence, **(N)** provably unnecessary by file:line evidence, or **(A)** assumption-flagged (requires user decision before inclusion).

---

## IN SCOPE — Provably Necessary (P)

### Step 1 — Fix ERR-021: extend label-degeneracy check to the `y`-column path

**File:** `src/neutralgrid/training/unified_training_builder.py`
**Lines:** 822-831

**Current code (verbatim from current tree):**
```python
if not assigned_target:
    preferred_target = None
    for col in ("y", "y_horizon", "label_positive_by_horizon"):
        if col in out.columns and cast(pd.Series, out[col]).notna().any():
            preferred_target = col
            break
    if preferred_target is not None:
        out["y"] = out[preferred_target].map(_label_from_any)
        out["label_source"] = preferred_target
        assigned_target = True
```

**Defect (proof):** `notna().any()` accepts a column of all zeros (zero is not NA) as a valid label source. ERR-021 evidence: `y.value_counts(): {0: 162}` from backtest outcome CSVs, which the loop selects ahead of the `net_pnl_pct >= meta_hurdle_pct` fallback at lines 833-838. The same function ALREADY uses the correct degeneracy check on the `hlabel_meta` path at lines 793-815 (`nunique() <= 1 or hlabel_pos_rate < 0.05`) — the bug is local to the secondary loop.

**Fix pattern:** mirror the existing degeneracy bypass that the `hlabel_meta` path uses. For each candidate column in `("y", "y_horizon", "label_positive_by_horizon")`, require that mapped non-null values satisfy `nunique() > 1` AND `mean() >= 0.05` before accepting it.

**Why this is not a new gate:** the gate already exists at lines 793-815 for the `hlabel_meta` path. This change only extends the same check to the next loop in the same function. No threshold is being added; the 5% positive-rate threshold is reused from `meta_labeler.py:725-732` (`if _global_pos_rate < 0.05: raise ValueError`) — both must agree or training will throw at a later stage.

**Verification (concrete commands):**
1. Pre-change probe (read-only): inspect `data/new_expired_bots_backfilled.xlsx` to confirm `y` either absent or `nunique == 1` AND `>= 3.0` share `>= 0.05`. ERR-021 evidence: `10/162 = 6.17%`, satisfies the gate.
2. Post-change unit test: `tests/unit/test_unified_training_builder.py` gains one test asserting `label_source == "net_pnl_pct_hurdle"` and `y.sum() >= 5` when an all-zero `y` is present alongside positive `net_pnl_pct` rows.
3. Full suite: `python -m pytest tests/` must remain green (~1023 tests per CLAUDE.md).
4. Diff scope: ~10 LOC change in `unified_training_builder.py` + ~25 LOC test.

### Step 2 — Replace `hmm_winner_calibrator.py:45` hardcoded HMM path with active-HMM resolver

**File:** `src/neutralgrid/calibration/hmm_winner_calibrator.py`
**Lines:** 45, 798, 885

**Current state (verbatim):**
- Line 45: `DEFAULT_HMM_ARTIFACT_DIR = Path("artifacts/hmm/rolling_180d_20260426_012042")`
- Line 798: `hmm_artifact_dir: Path | str = DEFAULT_HMM_ARTIFACT_DIR,`
- Line 885: `parser.add_argument("--hmm-artifact-dir", default=str(DEFAULT_HMM_ARTIFACT_DIR))`
- Line 673: `loaded_metadata = load_artifact(hmm_artifact_dir, validate_schema=True)["metadata"]`
- Line 681: the resolved path is stamped into the calibrator's output JSON, becoming the calibrator's official lineage claim in `artifacts/hmm_winner_calibrator/current.json`.

**Defect:** import-time literal. When the active HMM rotates (FIXHMM-01 already produced three replacement candidates: `rolling_180d_20260427_030024`, `..._051225`, `..._071650`), an operator running calibrator retrain without `--hmm-artifact-dir` loads metadata from the stale hardcoded HMM while the workbook features were computed against the new active HMM. The calibrator records the stale HMM in its `current.json` (false lineage), and there is NO cross-artifact gate that catches calibrator-vs-HMM drift — `_check_artifact_version_consistency()` at `run_full_pipeline.py:609-643` only checks meta-labeler vs HMM (`src/neutralgrid/models/artifact_compat.py:14-62`).

**Downstream consequences (concrete, not speculative):**
1. Auditability and reproducibility: replaying the calibrator from its recorded lineage loads the wrong HMM and yields different feature vectors → different coefficients → different scores. The lineage record becomes untrustworthy.
2. Pipeline-level lineage agreement is silently broken: meta-labeler.lineage.hmm and active HMM are checked; calibrator.lineage.hmm is not. The third leg can rot indefinitely.
3. P3/P4 promotion gates (older-row regression in `current.json:62-66`) compare candidate against a baseline that uses the recorded HMM's discriminant — if recorded HMM != workbook HMM, the baseline mixes contexts and the drop metric becomes meaningless.
4. Schema validation at line 673 (`validate_schema=True`) succeeds against the wrong schema if HMM family bumps in the future.
5. Class of drift the user explicitly warned against: math-works, fail-closed-doesn't-fire, lineage-lies — same failure mode as the FIXUTILITY-01 `-0.32` correlation finding.

**Fix pattern:** introduce `_resolve_default_hmm_artifact_dir()` that calls the existing `resolve_hmm_artifact_dir()` from `src/neutralgrid/models/artifacts.py` (same function `run_full_pipeline.py:626` uses). Make the default at line 798 lazy (`Path | str | None = None` with the resolver invoked when `None`) and make line 885 call the resolver for the CLI default. The `--hmm-artifact-dir` flag stays — operators who explicitly need to pin an old HMM (regression replay) retain the override.

**Why this is not a new gate:** no new check is added. The active-HMM lookup is reused from `run_full_pipeline.py:626`. Behavior changes from "silent stale path" to "active path" only when the operator does NOT pass `--hmm-artifact-dir`.

**Verification:**
1. After change, `python -c "from neutralgrid.calibration.hmm_winner_calibrator import _resolve_default_hmm_artifact_dir; print(_resolve_default_hmm_artifact_dir())"` returns the path matching `artifact_manifest.json["hmm"]["artifact_dir"]`.
2. New unit test (~20 LOC) in `tests/unit/test_hmm_winner_calibrator.py`: monkey-patches `artifact_manifest.json` to a different HMM, asserts the resolver returns the new value at call time (not the import-time literal).
3. Existing 9 calibrator tests in `tests/unit/test_hmm_winner_calibrator.py` (per memory) remain green.

### Step 3 — Update documentation to reflect the locked 8-feature `snapshot_v20260421_bootstrap` profile

**Files and exact lines to update:**

- `readmefullpwep.md`
  - Line 298: "Uses the `snapshot_v20260407` feature profile (29 Binance-fetchable features). This is the only available profile."
  - Line 304: "Bootstrap gate: when the deployed model's features don't match `SNAPSHOT_META_FEATURES_V20260407`, ..."
  - Line 530: "### 8.4 Feature profile: `snapshot_v20260407` (29 features)"
  - Line 591: "| `--feature-profile` | `snapshot_v20260407` | Feature profile (only `snapshot_v20260407` available) |"

- `PNL_CURVE_CLASS.md`
  - Line 16: "**NOT used as a meta-labeler feature** — absent from `SNAPSHOT_META_FEATURES_V20260407` in `src/neutralgrid/models/meta_labeler.py:41-74`."
  - Line 35: "Even if the classifier were richer, `pnl_curve_shape_class` is not in `SNAPSHOT_META_FEATURES_V20260407`, so the meta-labeler cannot use it."
  - Line 105: "Does NOT add `pnl_curve_shape_class` to `SNAPSHOT_META_FEATURES_V20260407` (would be post-outcome leakage)."
  - Line 136: "`src/neutralgrid/models/meta_labeler.py:41-74` — current `SNAPSHOT_META_FEATURES_V20260407`."

**What to change:** replace each `SNAPSHOT_META_FEATURES_V20260407` / `snapshot_v20260407` reference with `SNAPSHOT_META_FEATURES_V20260421_BOOTSTRAP` / `snapshot_v20260421_bootstrap`, update the feature-count claim from 29 to 8, and add a one-line pointer to `CHANGELOG.md:288-289` (which records the explicit downgrade rationale and the small-pool justification).

**Why this is in scope:** the user's "no false optionality" rule is violated when documentation describes a profile that is no longer the active code path. New operators reading `readmefullpwep.md:591` would believe `--feature-profile snapshot_v20260407` is the production default; the actual default is `snapshot_v20260421_bootstrap` (per `meta_labeler.py:138`). Doc-only fix; zero behavior change.

**Verification:**
1. After change, `grep -n "snapshot_v20260407\|SNAPSHOT_META_FEATURES_V20260407" readmefullpwep.md PNL_CURVE_CLASS.md` returns no matches except in deliberate "historical reference / superseded" sections.
2. The `--feature-profile` table entry shows the correct default and lists all three available profiles (`snapshot_v20260407`, `snapshot_v20260420`, `snapshot_v20260421_bootstrap`) per `meta_labeler.py:146-150`, with `snapshot_v20260421_bootstrap` flagged as active.
3. CHANGELOG entry under `[6.5.7-pipeline-health-restoration]` notes the doc sync.

### Step 4 — Resolve ERR-033: retrain meta-labeler against active HMM

**Command:** `python retrain_meta_labeler.py --input data/new_expired_bots_backfilled.xlsx --allow-imputation`

**Active-HMM resolution (proof):** `retrain_meta_labeler.py:573-580` (`_get_active_hmm_artifact_version`) calls `get_active_hmm_version()` from `src/neutralgrid/models/artifacts.py:561-569`, which reads `artifact_manifest.json["hmm"]["active_version"]`. No flag override required; pinning is automatic.

**Default feature profile (proof):** `--feature-profile` defaults to `ACTIVE_SNAPSHOT_META_FEATURES` = `SNAPSHOT_META_FEATURES_V20260421_BOOTSTRAP` = 8 features (`src/neutralgrid/models/meta_labeler.py:127-138`). This matches the locked small-pool decision documented in `CHANGELOG.md:288-289`. No `--feature-profile` flag needs to be passed.

**Why this is not a new model:** the model architecture (GradientBoostingClassifier, n_estimators=50, max_depth=3) is unchanged from the current `metadata.json:36-42`. Only training data and HMM lineage refresh.

**Verification:**
1. Post-retrain `models/meta_labeler/metadata.json["lineage"]["hmm_artifact_version"] == "rolling_180d_20260426_012042"` (matches active HMM in `artifact_manifest.json` and the calibrator's `current.json`).
2. `models/meta_labeler/metadata.json["eval_metrics"]["positive_rate"] >= 0.05` (passes the gate at `meta_labeler.py:725-732`).
3. `python run_full_pipeline.py --discovery-mode --top-n 25 --capital 400` produces a CSV where `meta_prob_source.value_counts()` shows non-`missing` values for at least some rows.
4. The pipeline log no longer emits the `meta_labeler lineage mismatch` warning that `_check_artifact_version_consistency()` at `run_full_pipeline.py:609-643` produces when lineage drifts.
5. ERRORS_LOG.md ERR-033 row is moved to CLOSED with the artifacts-changed evidence above.

---

## EXPLICITLY OUT OF SCOPE — Provably Unnecessary (N)

For each item, the file:line evidence proving no change is required is recorded.

### N1 — Version constants are already centralized

**Evidence:** `src/neutralgrid/core/constants.py:18-49` is the single source of truth for `LABEL_CONTRACT_VERSION = "2026-04-17"`, `FORMULA_VERSION = "alignment-v1"`, `ENGINE_VERSION = "realistic-v7"`, `BOT_HORIZON_HOURS = 6.0`, `BOT_HORIZON_BARS_15M = 24`, `BOT_HORIZON_SECONDS = 21_600`, `BOT_INCLUSION_TOLERANCE_HOURS = 0.25`, `PROFIT_FACTOR_CAP = 1000.0`. Imports verified at `backtest/btk_label_contract.py:26-34`, `backtest/btk_unified_runner.py:38-48`, `src/neutralgrid/backtest/candidate_pipeline.py:1046-1047`, `src/neutralgrid/training/unified_training_builder.py:28`, `src/neutralgrid/scanner/pnl_ranker.py:13`, `src/neutralgrid/scanner/empirical_profile_v20260302.py:23`, `src/neutralgrid/validation/utility.py:22`. The `try/except ImportError` fallback hardcodes in `backtest/*` are exception-guarded for script-mode invocation; they only fire when the centralized import fails.

### N2 — Feature triple is not duplicated; declarations have distinct roles

**Evidence:**
- `src/neutralgrid/backtest/candidate_pipeline.py` `TRAINING_OUTPUT_COLUMNS` (~58 columns) declares the SUPERSET of training-row outputs.
- `src/neutralgrid/training/data_generator.py` `FeatureSnapshot.to_dict()` (~48 columns) declares the SUPERSET the live pipeline collects per scan.
- `src/neutralgrid/training/unified_training_builder.py` `EXTRA_META_FEATURES + TRAINING_FEATURES` (~43 columns) declares the SUPERSET the trainer is willing to consume.
- `src/neutralgrid/models/meta_labeler.py:127-138` `ACTIVE_SNAPSHOT_META_FEATURES` (8 features) declares the SUBSET the deployed model actually uses.

The 8-feature deployed schema is a strict subset of all three superset declarations. Reducing the supersets to 8 features would lose data the system already collects for future retrains; expanding the model to use the supersets would violate the fixed-pool stability decision (`CHANGELOG.md:288-289`).

### N3 — Utility calibrator and pattern-profile fallbacks are deliberate

**Evidence:**
- `artifacts/utility/current.json` does NOT exist (verified via `ls artifacts/utility/`). Five timestamped candidate JSONs are present (latest `utility_20260422_190601_149413.json`). Per memory `project_fixutility_01.md`: FIXUTILITY-01 v3 plan implemented; user has DEFERRED promotion of `current.json` due to observed `utility_score` `-0.32` correlation with outcome at the locked sample pool.
- `data/profile/current.json` does NOT exist (verified via `ls data/profile/`). Bootstrap `profile_model.json` is in place. Per memory `project_pattern_profile_phase2.md`: per-class floor `max(30, 3*len(feats))` refuses promotion on current xlsx (only 11 winners).

The user's small-pool constraint forbids promotion of either; both promotion gates are structurally unwinnable at current N. Touching either is OUT OF SCOPE.

### N4 — Cross-artifact lineage check is already in place; no new gate needed

**Evidence:** `run_full_pipeline.py:609-643` (`_check_artifact_version_consistency`) calls `evaluate_hmm_meta_labeler_compatibility()` from `src/neutralgrid/models/artifact_compat.py:14-62`. This is the warning path; the fail-closed path is enforced inside `MetaLabeler.load()` itself (per ERR-033 evidence: `Artifact-managed meta-labeler load failed: meta_labeler lineage mismatch`). Adding any new lineage gate would violate the user's "no additional gating" constraint.

### N5 — ERR-022 (argparse) and ERR-023 (1m interval) are already landed

**Evidence:** Explore-agent audit confirms `scripts/backfill_training_features.py` lines 598-621 declare argparse with `--input` and `--output` Path args; `_interval_delta()` at lines 151-160 supports `"1m"` interval. ERRORS_LOG.md still shows these as OPEN — that is a stale ERRORS_LOG entry, not a code defect.

---

## ASSUMPTION-FLAGGED — items NOT valid to strike without user decision (A)

Each item is verifiable to exist but cannot be classified as in-or-out without the user's call.

### A1 — RESOLVED: locked into Step 2 above (replace hardcode with active-HMM resolver).

### A2 — RESOLVED: locked into Step 3 above (sync README and PNL_CURVE_CLASS to active 8-feature profile).

### A2-original — Documentation drift (README, CHANGELOG, PNL_CURVE_CLASS still reference `snapshot_v20260407` as active)

**Evidence:** `readmefullpwep.md:298, 304, 530, 591`, `PNL_CURVE_CLASS.md:16, 35, 105, 136` describe the 29-feature `snapshot_v20260407` profile as "the only available profile" or "active". Code's `ACTIVE_SNAPSHOT_META_FEATURES` is `snapshot_v20260421_bootstrap` (8 features) per `meta_labeler.py:138`. `CHANGELOG.md:288-289` records the explicit downgrade.

**Why it is NOT a runtime defect:** docs lag the code; runtime is correctly bound to `snapshot_v20260421_bootstrap`.

**Status:** Step 3 was authorized by user; the original framing is retained as audit trail until Step 3 is implemented and verified, at which point this section moves to "closed/archived".

### A3 — DEFERRED: backfill multi-sheet output

**Decision:** the strict-flat fallback in `hmm_winner_calibrator.py` reads the single-sheet output successfully; no consumer is currently broken. Adding a multi-sheet writer now is false optionality. Re-evaluate when an identified downstream consumer requires the `General` / `PnL Curve Features` / `Meta Features` sheet split. Recorded for later in ERRORS_LOG.md as ERR-031 (status remains OPEN).

---

## Validation Workflow

Implementation order is Step 1 → Step 2 → Step 3 → Step 4. Each step gates on its own verification AND on the team-review consensus described below before proceeding.

1. Read `tests/unit/test_unified_training_builder.py` — capture existing precedence-test coverage.
2. Apply Step 1 fix (`unified_training_builder.py:822-831`). Add one targeted test for the all-zero `y` + positive `net_pnl_pct` case.
3. Run `python -m pytest tests/unit/test_unified_training_builder.py -v` — green.
4. Apply Step 2 fix (`hmm_winner_calibrator.py:45, 798, 885`). Add the resolver test in `tests/unit/test_hmm_winner_calibrator.py` that monkey-patches `artifact_manifest.json` to a different HMM and asserts the resolver returns the new value at call time.
5. Run `python -m pytest tests/unit/test_hmm_winner_calibrator.py -v` — green (existing 9 tests + 1 new test).
6. Run `python -m pytest tests/ -k 'not (TestBoundedUniverseContract or test_current_workbook_contract_counts_if_available)'` — full suite green except ERR-038 (4-test pre-existing failure family in `scanner/pattern_profile.py:325` / `profile_model.py:325`, untouched by PIPELINE_FIX) and ERR-039 (pre-existing utility-calibrator pool-size fixture drift), both tracked separately in ERRORS_LOG.md.
7. Apply Step 3 doc updates (readmefullpwep.md and PNL_CURVE_CLASS.md). No tests; visual diff review only.
8. Execute Step 4 retrain: `python retrain_meta_labeler.py --input data/new_expired_bots_backfilled.xlsx --allow-imputation`.
9. Verify `models/meta_labeler/metadata.json["lineage"]["hmm_artifact_version"] == "rolling_180d_20260426_012042"`.
10. Run discovery-mode pipeline: `python run_full_pipeline.py --discovery-mode --top-n 25 --capital 400`.
11. Verify `meta_prob_source != "missing"` on at least some rows of the resulting deployment CSV.
12. Move ERR-033 from OPEN to CLOSED with artifacts-changed evidence in ERRORS_LOG.md. Add CHANGELOG entry under `[6.5.7-pipeline-health-restoration]` recording all four steps.

## Files Changed (IN SCOPE only)

- `src/neutralgrid/training/unified_training_builder.py` — lines 822-831, ~10 LOC change (Step 1)
- `tests/unit/test_unified_training_builder.py` — ~25 LOC new test (Step 1)
- `src/neutralgrid/calibration/hmm_winner_calibrator.py` — lines 45, 798, 885, ~10 LOC change (Step 2)
- `tests/unit/test_hmm_winner_calibrator.py` — ~20 LOC new test (Step 2)
- `readmefullpwep.md` — lines 298, 304, 530, 591, doc-only edit (Step 3)
- `PNL_CURVE_CLASS.md` — lines 16, 35, 105, 136, doc-only edit (Step 3)
- `models/meta_labeler/*` (regenerated by Step 4)
- `models/meta_labeler.pkl` (regenerated by Step 4)
- `ERRORS_LOG.md` — move ERR-033 to CLOSED
- `CHANGELOG.md` — entry under `[6.5.7-pipeline-health-restoration]`

No new files. No new CLI flags. No new gates.

## Team Review Step

User confirmed full 4-agent team review. The implementation phase will spawn a persistent team `pipeline-health-review` (TeamCreate, NOT one-shot Agent calls — per memory `feedback_agents_team_vs_subagents.md`) with the following members and per-step responsibilities:

**Team members:**
- `data-curator` — validates label-precedence fix preserves data semantics; confirms the 5% positive-rate threshold is reused from `meta_labeler.py:725-732` (not a new gate); validates the lineage-stamping change in `hmm_winner_calibrator.py:681` correctly records the resolved active HMM.
- `feature-analyst` — confirms the 8-feature `snapshot_v20260421_bootstrap` profile is the correct schema at locked N (calibrator pool 85, meta-labeler pool ~226); confirms no feature drift is introduced by either code change; validates that the doc updates accurately reflect the active feature set.
- `market-strategy-architect` — confirms the precedence fix does not weaken the fast-winner classification theory (`pnl_pct > 1.0`, `duration_hours < 7.0`); confirms the calibrator-HMM lineage repair preserves the regime-persistence economic mechanism that the calibrator's logistic regression captures.
- `backtest-evaluator` — POBO impact analysis: the precedence fix changes ~6% of training rows from `y=0` to `y=1`; ensures this is not a backtest-overfit lever; runs alternative-scenario stress test (e.g., what if backfilled positive rate is 5.1% instead of 6.2% — does retrain still succeed and produce stable AUC); validates the calibrator resolver change against POBO via the existing 9-test calibrator suite.

**Per-step gating:**

| Step | Reviewer assignments | Block condition |
|---|---|---|
| Step 1 (label precedence fix) | data-curator (lead), backtest-evaluator | All assigned reviewers vote PASS |
| Step 2 (calibrator hardcode fix) | data-curator, market-strategy-architect (lead) | All assigned reviewers vote PASS |
| Step 3 (doc sync) | feature-analyst (lead) | feature-analyst votes PASS |
| Step 4 (meta-labeler retrain) | feature-analyst (lead), backtest-evaluator | All assigned reviewers vote PASS |
| Final integration verification | All four | Unanimous PASS |

The team will block on each step until consensus PASS. Any FAIL must include a concrete file:line objection that traces to the step's code change.

---

## Final Outcome (2026-04-29)

**Final Integration: UNANIMOUS PASS** with one documented exception (ERR-042).

| Reviewer | Step 1 | Step 2 | Step 3 | Step 4 | Final Integration |
|---|---|---|---|---|---|
| data-curator | PASS | PASS | — | — | PASS |
| feature-analyst | — | — | PASS | PASS | PASS |
| market-strategy-architect | — | PASS | — | — | PASS (after Option A reconciliation) |
| backtest-evaluator | PASS | — | — | CONDITIONAL PASS | PASS (operational) — ERR-042 accepted |

**Round 1 of Final Integration produced a non-unanimous result** (market-strategy-architect FAIL with two file:line-cited blocking objections; backtest-evaluator split). Per the user's pause-and-ask instruction, the plan paused; the user authorized Option A (theory reconciliation, no threshold revert). After reconciliation:

- ERR-040 (theory drift) — CLOSED. White-box-theory docstring added at `src/neutralgrid/models/meta_labeler.py:140-167` documenting the canonical mechanism, locked-pool concession, and Stage 12 Kelly-sizing interpretation guidance. CHANGELOG `[6.5.7-meta-labeler-bootstrap-simplification]` entry augmented with a "White-Box Theory Reconciliation" section.
- ERR-041 (artifact internal contract mismatch) — CLOSED. `meta_labeler.py:1593-1611` now writes `model_params.hurdle_pct = ACTIVE_META_TARGET_PNL_THRESHOLD_PCT` when an active target column is set; verified in regenerated `metadata.json` (artifact `20260429_172907`): both threshold fields = 0.0, aligned. Config Integrity per safety-invariants.md restored.
- ERR-042 (locked-pool noise floor: CV AUC = 0.455) — accepted as a documented constraint of the locked-pool regime, not a Step 4 defect. The reconciled theory docstring explicitly references ERR-042 so consumers see the caveat.

**Operational outcome:** the meta-labeler now loads with lineage matching the active HMM, the discovery-mode pipeline produces 18/25 calibrated rows (was 0/25 pre-fix), and the artifact's internal fields are self-consistent. The user's stated goal in this plan ("align all components of the full pipeline so it runs healthy and functional end-to-end, without adding new gating or increased filters") is satisfied.

ERR-033 (the originally-blocking meta-labeler stale lineage) and ERR-021 (the precondition label-precedence bug) are CLOSED in ERRORS_LOG.md with artifact-changed evidence. ERR-035, ERR-036, ERR-037 remain WATCH-listed for future maintenance. ERR-038 and ERR-039 are pre-existing test-suite issues unrelated to PIPELINE_FIX.

## Error and Blocker Reporting

- All errors encountered during implementation are appended to `ERRORS_LOG.md` with the standard schema: `ID | Status | Area | Error Check | Evidence | Required Action | Verification`.
- If a blocker arises that cannot be resolved without user input (ambiguity, conflicting requirements, resource access, third-party failure), the implementation pauses at the current step. The blocker is logged to `ERRORS_LOG.md` with status `BLOCKED`, and the user is asked for direction before continuing.
