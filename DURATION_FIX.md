# DURATION_FIX - Unified Current Plan

Status: structural fix complete. P0 slow-path parser and P1 tmp/cache ACL blocker are both closed. Advisory items (s*, tau) remain open as post-fix monitoring, not structural work.

Date: 2026-04-17

Canonical bot horizon:

```text
H* = 6 hours = 24 bars at 15 minutes = 360 bars at 1 minute = 21,600 seconds
```

Progress:

```text
Structural and verification progress: [####################] 100% (20/20 tracked items complete)
```

The 100% count is limited to structural drift and verification items tracked in this document. Advisory calibration work (s* threshold re-monitoring and tau measurement) is tracked separately and is not part of the structural contract.

Completed:

1. H* is centralized in `src/neutralgrid/core/constants.py`.
2. Grid lifecycle seconds, barrier horizons, CPCV/t1 synthesis, stochastic survival horizon, utility, PnL ranking, microstructure, candidate conversion, and BTK label defaults are routed from H*.
3. The label contract version is bumped to `2026-04-17`.
4. `horizon_censored` is required by the BTK output contract.
5. The realistic backtest emits `horizon_censored`.
6. Candidate training rows propagate native boolean `horizon_censored`.
7. Training ingestion gates legacy or older label contracts through `version_gated=True`.
8. Old or missing label-contract rows are excluded from the final training frame.
9. Survival horizon is H* = 24 bars at 15 minutes.
10. OU half-life max remains 48 bars because it is an estimator validity bound, not the bot inventory horizon.
11. CRITERION implementation notes now carry a supersession header pointing back to this document.
12. D14.5 runtime TTL/executor abstraction is closed and must not be reintroduced. The user is the sole deployment teardown operator.
13. P0 - slow-path `horizon_censored` parsing is hardened through `_as_bool_strict()` in `src/neutralgrid/training/unified_training_builder.py`. Native bools, numpy bools, int 0/1, and case-insensitive `true|false|1|0|yes|no` strings are accepted; anything else raises `ValueError`.
14. P0 follow-up (§11 rail) - a current-contract row whose `horizon_censored` field is non-missing but unparseable is now gated with `version_gated=True` and `source_class="non_authoritative"` rather than silently admitted with an omitted field. The slow-path loop no longer clobbers this tag when it later stamps `source_class="reconstruction"`.
15. Two focused tests cover P0 - `test_slow_path_horizon_censored_string_false_stays_false` proves the string `"False"` does not coerce to `True`, and `test_slow_path_horizon_censored_invalid_string_gates_row` proves a malformed value causes the row to be dropped from final training while a well-formed companion row survives.
16. P1 - the local Windows pytest temp/cache ACL blocker under `C:\Users\cris_\AppData\Local\Temp\pytest-of-cris_` is cleared, and the tmp-path-heavy training-builder tests have been revalidated in the current environment.

Advisory, not structural blockers:

1. Re-monitor the survival threshold `s* = 0.60` after H* alignment and grid-bound survival recalculation.
2. Measure the duration inclusion tolerance `tau = 0.25h` empirically from teardown latency when reliable deployment logs exist.

## 1. Scope And Authority

This file is the current source of truth for bot-duration structural alignment. Earlier repeated audit sections have been consolidated into this single plan. Historical details should be recovered from git history if needed; the operative plan is the one below.

This document only accepts a claim as implemented when it is traceable to code, tests, or a directly observed command result. Anything not verifiable from local files is classified as advisory, assumption, or blocked.

The plan is intentionally narrow:

1. Fix the structural horizon drift around H*.
2. Preserve AFML separation between label horizon, sample eligibility, validation horizon, estimator validity, and deployment operations.
3. Avoid new registries, new files, live-execution abstractions, and optional alternate paths unless a current code dependency proves they are required.

## 2. Root Structural Problem

The root problem was not a single wrong number. The root problem was semantic drift: numerically similar quantities were allowed to behave as if they were interchangeable.

The codebase had several horizon-like concepts:

1. Bot inventory lifecycle: H*, the intended maximum lifecycle of one grid bot.
2. Triple-barrier label horizon: the time barrier used to define the supervised label.
3. Survival containment horizon: the Monte Carlo horizon used to estimate whether price remains inside the proposed grid.
4. CPCV purge/t1 horizon: the leakage-prevention horizon for training validation.
5. OU estimator validity window: a statistical upper bound for admissible mean-reversion half-life.
6. PnL validation/reporting horizon: an operator reporting window.
7. Deployment teardown authority: an operational decision made outside this codebase by the user.

The structural drift happened because the same kind of value, "duration", appeared in multiple modules without an explicit semantic owner. Once `24`, `56`, `6`, and `48` existed as local constants or defaults, downstream modules could not prove whether a number represented the bot lifecycle, a survival simulation horizon, a calibration observation, or an estimator cutoff.

The correct structural repair is therefore classification plus routing, not broad replacement:

1. Values that mathematically define the label or bot lifecycle must route from H*.
2. Values that define estimator admissibility must stay local or config-routed under their own semantic name.
3. Values that define operator validation/reporting must not be silently coupled to H* unless the code uses them for labeling or training leakage.
4. Runtime teardown must not be modeled as code-owned TTL when the actual operator is the user.

## 3. Theory Break

The theory break was caused by mixing the lifecycle horizon and the survival horizon.

For a bot with intended lifecycle H*, the supervised event label and the containment estimate should refer to the same economic exposure window unless there is a documented reason not to. If labels are created at H* but survival is estimated at another horizon, the model is trained and filtered under two different event definitions.

Mathematically:

```text
H* = 6h
bar_15m = 0.25h
H*_bars_15m = H* / bar_15m = 6 / 0.25 = 24

bar_1m = 1/60h
H*_bars_1m = H* / bar_1m = 6 * 60 = 360

H*_seconds = 6 * 3600 = 21,600
```

For survival probability:

```text
S(t) = P(no grid breach before or at time t)
```

For any valid containment process, survival is non-increasing in t:

```text
t2 > t1  =>  S(t2) <= S(t1)
```

Therefore, using 56 bars instead of 24 bars is not a harmless parameter change. It changes the event being estimated:

```text
S(56 bars) <= S(24 bars)
```

The correct structural value is `S(24)` because the current bot label horizon is H* = 24 bars at 15 minutes. A 56-bar value may be useful for separate duration analysis, but it cannot be the canonical H* survival horizon unless H* itself changes.

The survival threshold `s* = 0.60` is separate from horizon routing. Routing H* proves the event window. It does not mathematically prove the threshold is optimal. That threshold must be monitored empirically after the event definition is stable.

## 4. Current Implementation Proof

The following items are currently verified from local code.

| Item | Status | Verification |
| --- | --- | --- |
| Canonical label contract version | Implemented | `src/neutralgrid/core/constants.py` defines `LABEL_CONTRACT_VERSION: str = "2026-04-17"`. |
| Canonical H* hours | Implemented | `src/neutralgrid/core/constants.py` defines `BOT_HORIZON_HOURS: float = 6.0`. |
| Canonical H* 15m bars | Implemented | `src/neutralgrid/core/constants.py` defines `BOT_HORIZON_BARS_15M: int = 24`. |
| Canonical H* seconds | Implemented | `src/neutralgrid/core/constants.py` defines `BOT_HORIZON_SECONDS: int = 21_600`. |
| Inclusion tolerance | Implemented as provisional policy | `src/neutralgrid/core/constants.py` defines `BOT_INCLUSION_TOLERANCE_HOURS: float = 0.25`. |
| Grid max lifecycle | Implemented | `GridConfig.max_holding_seconds = BOT_HORIZON_SECONDS`. |
| Barrier time horizon | Implemented | `BarrierConfig.time_hours = BOT_HORIZON_HOURS`. |
| Stochastic survival horizon | Implemented | `StochasticConfig.survival_horizon_bars = BOT_HORIZON_BARS_15M`. |
| CPCV t1/purge horizon | Implemented | `CPCVConfig.horizon_hours = BOT_HORIZON_HOURS`. |
| PnL validation window | Correctly not coupled as structural proof | `ValidationConfig.pnl_horizon_hours = 6` is marked as an operator validation window, not H*. |
| BTK default holding bars | Implemented | `TRAINING_ENGINE_DEFAULTS["max_holding_bars"] = int(BOT_HORIZON_HOURS * 60)`. |
| Realistic backtest holding bars | Implemented | `backtest/backtest_realistic.py` uses `int(BOT_HORIZON_HOURS * 60)`. |
| BTK contract requires censor metadata | Implemented | `REQUIRED_LABEL_FIELDS` contains `horizon_censored`. |
| BTK contract validates censor type | Implemented | `validate_btk_output_contract()` requires `horizon_censored` to be bool-like. |
| Realistic backtest emits censor metadata | Implemented | Result includes `'horizon_censored': bool(self._stale_closes > 0)`. |
| Candidate fast path preserves censor metadata | Implemented | `convert_to_training_row()` emits `"horizon_censored": bool(backtest_result.get("horizon_censored", False))`. |
| Censor metadata excluded from model features | Implemented | Tests assert `horizon_censored` is not in `ALL_META_FEATURES`. |
| Training ingestion version gate | Implemented | `unified_training_builder.py` marks missing or old `label_contract_version` as `version_gated=True`. |
| Version-gated rows excluded | Implemented | Final combined builder filters `version_gated` rows before training output. |
| Slow-path censor preservation | Implemented | `_as_bool_strict()` in `unified_training_builder.py` parses native bools, numpy bools, int 0/1, and case-insensitive `true\|false\|1\|0\|yes\|no` strings. Invalid non-missing values gate the row with `version_gated=True` + `source_class="non_authoritative"` so §11's "known and current" invariant is enforced. |
| Survival validator horizon | Implemented | `src/neutralgrid/validation/stochastic.py` defines `survival_horizon = BOT_HORIZON_BARS_15M`. |
| OU max half-life | Correctly retained | `ou_halflife_max = 48` remains an estimator validity bound, not H*. |
| PnL ranker horizon | Implemented | `src/neutralgrid/scanner/pnl_ranker.py` defaults to `BOT_HORIZON_HOURS`. |
| Utility horizon fallback | Implemented | `src/neutralgrid/validation/utility.py` defaults to `BOT_HORIZON_HOURS`. |
| Microstructure horizon | Implemented | `src/neutralgrid/validation/microstructure.py` defaults to `BOT_HORIZON_HOURS`. |
| Grid spacing duration inclusion | Implemented | `src/neutralgrid/grid/spacing_profile.py` uses grid max holding seconds plus `BOT_INCLUSION_TOLERANCE_HOURS`. |
| D14.5 runtime TTL/executor | Closed | No `max_runtime_hours` matches outside this document. |
| SURVIVAL_PROB_ADJUST.md | Not available in root | `Test-Path SURVIVAL_PROB_ADJUST.md` returned `False`; it cannot be used as a source of proof. |

## 5. Drifts Resolved

| Drift | Resolution | Proof Type |
| --- | --- | --- |
| H* scattered as local `6`, `24`, or `21600` | Central constants define the canonical contract. | Code. |
| Grid lifecycle separate from label horizon | `GridConfig.max_holding_seconds` routes from `BOT_HORIZON_SECONDS`. | Code. |
| Barrier horizon separate from H* | `BarrierConfig.time_hours` routes from `BOT_HORIZON_HOURS`. | Code. |
| Candidate backtest default separate from H* | `run_single_backtest()` defaults to `int(BOT_HORIZON_HOURS * 60)`. | Code. |
| Training label contract did not require censor metadata | `horizon_censored` is required and type-validated. | Code and tests. |
| Engine censoring could disappear before training | Native candidate fast path now preserves `horizon_censored`. | Code and tests. |
| Legacy rows could mix with new-label rows | Missing or older contract versions are gated and excluded. | Code. |
| Survival horizon drifted to historical 56-bar guidance | Survival horizon routes from 24-bar H*. | Code. |
| OU max half-life was at risk of being mistaken for H* | OU max remains 48 and is classified as estimator validity. | Code and classification. |
| PnL, utility, and microstructure horizons could drift | Defaults route from H* where they model bot exposure. | Code. |
| Grid spacing exclusion was too brittle around H* | Duration cap includes H* plus tau. | Code. |
| CRITERION guidance still contained old 56-bar plan | Supersession header points to this file for bot lifetime horizon. | Documentation. |
| Deployment TTL abstraction was proposed despite no code owner | Closed as false optionality. | Code search and architecture. |

## 6. What Is Already Aligned

The following are aligned and should not be reopened without new evidence:

1. H* is `6h`, not 14h.
2. H* at 15-minute resolution is 24 bars.
3. H* at 1-minute backtest resolution is 360 bars.
4. H* in seconds is 21,600.
5. Survival horizon must be 24 bars at 15 minutes for the canonical H* event.
6. OU half-life max of 48 bars is not a horizon drift by itself.
7. `ValidationConfig.pnl_horizon_hours = 6` is acceptable as an explicitly local operator validation window.
8. `horizon_censored` is metadata, not a model feature.
9. D14.5 deployment executor and runtime TTL are out of scope for this codebase.

## 7. Blockers In Priority Order

### P0 - Harden slow-path `horizon_censored` parsing

Status: closed.

The unsafe coercion `bool(raw.get("horizon_censored"))` has been replaced with `_as_bool_strict()` in `src/neutralgrid/training/unified_training_builder.py`. The parser:

1. Accepts native `bool` and `numpy.bool_` values.
2. Accepts integer `0` and `1`; raises on other numeric values.
3. Accepts case-insensitive `true`, `false`, `1`, `0`, `yes`, `no`.
4. Treats `None`, `NaN`, empty strings, and the literal `"nan"` as missing.
5. Raises `ValueError` on any other non-missing value rather than silently coercing.

The call site in the slow-path row loop gates invalid non-missing values by setting `row["version_gated"] = True` and `row["source_class"] = "non_authoritative"`. The later unconditional `source_class = "reconstruction"` assignment has been guarded with `if "source_class" not in row` so the gating tag is not clobbered. The existing `version_gated` filter in the combined pipeline then excludes the row from final training. This enforces §11's "known and current" invariant on a per-field basis.

Coverage is provided by `tests/unit/test_unified_training_builder.py::test_slow_path_horizon_censored_string_false_stays_false` and `...::test_slow_path_horizon_censored_invalid_string_gates_row`.

### P1 - Resolve pytest temp/cache ACL blocker

Status: closed.

The stale pytest temp tree under `C:\Users\cris_\AppData\Local\Temp\pytest-of-cris_` was removed. The tmp-path-heavy training-builder tests and the focused H* contract suite were re-run in the current environment:

```text
tests/unit/test_unified_training_builder.py
Result: 16 passed

tests/unit/test_btk_output_contract.py
tests/unit/test_bot_horizon_contract.py
tests/unit/test_enrich_grid_params_survival_recalc.py
Result: 28 passed

tests/
Result: 1095 passed, 7 warnings
```

The prior blocker was a Windows ACL state issue, not a code defect. With the stale tree cleared, no training-builder assertion failed.

### P2 - Re-monitor survival threshold `s* = 0.60`

Status: advisory, not a structural blocker.

The code can prove horizon alignment. It cannot prove threshold optimality without fresh post-fix validation data.

Valid continuation:

1. Keep `s* = 0.60` until post-fix distributions are available.
2. Recompute containment and label-quality diagnostics under H* = 24 bars.
3. Change `s*` only if the diagnostics prove a better threshold.

Invalid continuation:

1. Do not change `s*` merely because the horizon changed.
2. Do not claim `s* = 0.60` is optimal without post-fix evidence.

### P3 - Measure tau for duration inclusion

Status: advisory, not a structural blocker.

`BOT_INCLUSION_TOLERANCE_HOURS = 0.25` is a simple, explicit, provisional policy. It is valid because it is classified as tolerance, not H*.

Valid continuation:

1. Keep tau local and named.
2. Measure actual teardown/observation latency when reliable deployment logs exist.
3. Tune tau only from observed latency, not from assumed runtime behavior.

Invalid continuation:

1. Do not use tau to redefine H*.
2. Do not convert tau into an executor TTL.

## 8. Provable False Optionality

These integrations are false optionality because current code and architecture prove they are not required to solve the H* structural drift.

| Item | Classification | Reason |
| --- | --- | --- |
| Deployment executor | False optionality | `deployment_payload_v20260304.py` is payload sizing only and does not place orders. The user is the operational teardown actor. |
| Runtime TTL field such as `max_runtime_hours` | False optionality | No `max_runtime_hours` matches exist outside this document; adding one would create an unused abstraction. |
| Reintroducing D14.5 as code work | False optionality | There is no code path that can enforce it without building a new runtime owner. |
| Making `horizon_censored` a model feature | False optionality | It is label provenance metadata. Using it as a feature would leak outcome-generation information into training. |
| Treating engine-only censor metadata as sufficient | False optionality, already corrected | If not propagated to training rows, the metadata cannot protect training data. |
| Broad search-and-replace of every `6`, `24`, `48`, or `56` | False optionality | These numbers have different semantics. Blind replacement would create new theory breaks. |
| Changing OU max half-life from 48 to 24 | False optionality | OU max is an estimator validity criterion, not the bot lifecycle horizon. |
| New schema registry for the current fix | False optionality | The existing label contract version and training gate already provide the needed contract boundary. |
| Using absent `SURVIVAL_PROB_ADJUST.md` as proof | False optionality | The file is not present in the workspace root, so it cannot validate current code. |

## 9. Provably Unnecessary Items As Written

These are unnecessary for the current plan and should not be implemented unless a future dependency proves otherwise.

1. A new deployment scheduler or teardown service.
2. A new runtime owner for payload files.
3. A new document-only abstraction layer over existing constants.
4. A separate constants file for survival horizon when `BOT_HORIZON_BARS_15M` already represents the canonical H* survival event.
5. A full rewrite of `CRITERION_IMPLEMENT.md`; its supersession header is sufficient for the H* contradiction.
6. A full model retrain solely to prove constants routing. Retraining may be needed operationally, but routing correctness is proven by code and tests.
7. Extra optional code paths for 56-bar survival. A separate 56-bar study can exist only if explicitly labeled as non-H* analysis.
8. Moving `horizon_censored` into `ALL_META_FEATURES`.
9. Adding files for the bool parser. A local helper and focused test are enough.

## 10. Items Not Valid To Strike Without Assumptions

These items must remain in the plan unless new verifiable evidence removes them.

| Item | Why it cannot be struck |
| --- | --- |
| P0 slow-path bool parser | Closed. `_as_bool_strict()` replaces the unsafe coercion and malformed non-missing values now gate the row via `version_gated=True`; struck by direct code and test evidence. |
| Training contract gate | Without it, legacy and current labels can mix after H* semantics change. |
| `horizon_censored` metadata | Without it, samples closed by horizon cannot be distinguished from economically resolved samples. |
| Survival threshold monitoring | H* routing does not prove `s* = 0.60` remains optimal. |
| Tau monitoring | The current 0.25h value is a policy, not an empirically proven latency estimate. |
| Pytest ACL blocker | Closed. Stale pytest temp tree removed; training-builder tests (`16 passed`), focused H* suite (`28 passed`), and full suite (`1095 passed`) revalidated in the current environment. |
| CRITERION supersession | Historical 56-bar guidance remains in that file; the supersession header prevents it from being treated as current authority. |

## 11. Deduplication And Healthy Training Rules

The healthy-training rule is: classify before deduplicating, then deduplicate by contract.

Required classification:

1. Current authoritative row: has `label_contract_version >= 2026-04-17` and required contract fields.
2. Legacy row: missing `label_contract_version`.
3. Older-contract row: has a version older than `2026-04-17`.
4. Non-authoritative row: reconstructed, missing required label provenance, or otherwise not eligible for final training.
5. Horizon-censored row: authoritative row where the engine closed at horizon rather than through economic resolution.

Required deduplication:

1. Prefer current authoritative rows over legacy rows.
2. If duplicate symbol/timestamp rows exist across label versions, keep the newest valid contract version.
3. Exclude `version_gated=True` rows from final model training.
4. Preserve `horizon_censored` as metadata after deduplication.
5. Do not use `horizon_censored` as a feature.

Maintenance invariant:

```text
No row may enter model training unless its label semantics are known and current.
```

## 12. Maintenance Process To Prevent Drift

Every future horizon-related change must pass this checklist:

1. Name the semantic class before editing code: lifecycle, label, survival, CPCV, estimator validity, validation/reporting, or deployment operation.
2. If the value represents bot lifecycle exposure, route it from `BOT_HORIZON_HOURS`, `BOT_HORIZON_BARS_15M`, or `BOT_HORIZON_SECONDS`.
3. If the value represents estimator validity, keep it under its estimator config and document why it is not H*.
4. If the value represents operator reporting, keep it local and document that it is not label horizon.
5. If the value changes label semantics, bump `LABEL_CONTRACT_VERSION`.
6. If the label contract changes, update the BTK contract test and training ingestion gate.
7. If a new field is label provenance metadata, keep it out of model features.
8. If a doc contains older contradictory horizon guidance, add a supersession note rather than rewriting unrelated historical context.
9. Run focused tests for the touched contract boundary before broader tests.

Minimal verification commands:

```powershell
$env:PYTHONPATH='.;src'; pytest -q tests/unit/test_btk_output_contract.py tests/unit/test_bot_horizon_contract.py tests/unit/test_enrich_grid_params_survival_recalc.py
```

After the ACL blocker is fixed:

```powershell
$env:PYTHONPATH='.;src'; pytest -q tests/unit/test_unified_training_builder.py
$env:PYTHONPATH='.;src'; pytest -q tests
```

## 13. Final Concrete Continuation Plan

Step 1 - Fix P0 bool parsing.

Edit only `src/neutralgrid/training/unified_training_builder.py` and `tests/unit/test_unified_training_builder.py`.

Acceptance proof:

1. Native `True` remains true.
2. Native `False` remains false.
3. String `"True"` becomes true.
4. String `"False"` becomes false.
5. Missing value remains absent/defaulted according to current builder behavior.
6. Invalid non-missing string does not silently become true.

Step 2 - Revalidate the training-builder tests.

Clear the local pytest temp/cache ACL issue and run:

```powershell
$env:PYTHONPATH='.;src'; pytest -q tests/unit/test_unified_training_builder.py
```

Acceptance proof:

1. The new string-false test passes.
2. Existing version-gate tests pass.
3. Existing slow-path preservation tests pass.

Step 3 - Revalidate the focused H* contract suite.

Run:

```powershell
$env:PYTHONPATH='.;src'; pytest -q tests/unit/test_btk_output_contract.py tests/unit/test_bot_horizon_contract.py tests/unit/test_enrich_grid_params_survival_recalc.py
```

Acceptance proof:

1. BTK contract still requires `horizon_censored`.
2. H* constants still route into configs and BTK defaults.
3. Post-enrichment survival recalculation still passes existing tests.

Step 4 - Re-run full test suite when local ACL permits.

Run:

```powershell
$env:PYTHONPATH='.;src'; pytest -q tests
```

Acceptance proof:

1. No H* contract regression.
2. No training-builder regression.
3. No unrelated module regression caused by the parser change.

Step 5 - Operational retraining after code verification.

Only after contract tests pass, regenerate training data and retrain using the current label contract. This is not required to prove code routing, but it is required to make deployed model artifacts reflect the corrected H* label semantics.

Step 6 - Post-fix threshold monitoring.

After enough fresh H*-contract rows exist, evaluate survival probability calibration and decide whether `s* = 0.60` remains appropriate.

Step 7 - Tau monitoring.

When reliable deployment teardown logs exist, estimate observed latency and tune `BOT_INCLUSION_TOLERANCE_HOURS` only if the data proves a better value.

## 14. Verification Record

Latest verified commands and observations:

1. `rg` verified H* constants in `src/neutralgrid/core/constants.py`.
2. `rg` verified label contract, `horizon_censored`, backtest emission, candidate propagation, and training version gating.
3. `rg` verified survival, PnL, utility, microstructure, and grid-spacing H* routing.
4. `rg -n "max_runtime_hours" --glob "!DURATION_FIX.md" .` returned no matches.
5. `Test-Path SURVIVAL_PROB_ADJUST.md` returned `False`.
6. `_as_bool_strict()` and its gating call site verified in `src/neutralgrid/training/unified_training_builder.py`.
7. Stale pytest temp tree removed: `cmd /c rmdir /s /q C:\Users\cris_\AppData\Local\Temp\pytest-of-cris_`.
8. Training-builder tests revalidated: `16 passed` in the current environment.
9. Focused H* contract suite revalidated: `28 passed`.
10. Full test suite revalidated: `1095 passed, 7 warnings` (up from the prior `1093 passed` due to the two new P0 tests).
11. Independent review performed by the `deployment-engineering` agent (APPROVE-WITH-NOTES, parser and call site verified, no other unsafe `bool(raw.get("horizon_censored"))` coercion sites found) and the `backtest-evaluator` agent (LOW overfitting risk, H*/CPCV/OU routing unchanged; §11 classification gap flagged and subsequently closed by the Option B gating switch recorded above).

## 15. Final Status

The plan is structurally complete after deduplication. Both tracked blockers (P0 slow-path bool parser and P1 pytest temp/cache ACL) are closed, and the full test suite has been revalidated at `1095 passed, 7 warnings`.

The current code is aligned with AFML principles because it separates:

1. event-label horizon,
2. containment horizon,
3. leakage/purge horizon,
4. estimator validity,
5. operator validation,
6. deployment operation.

Structural changes that closed the plan:

1. `_as_bool_strict()` in `src/neutralgrid/training/unified_training_builder.py` eliminates the silent `bool("False") is True` coercion bug on the slow-path.
2. Option B gating (`version_gated=True` + `source_class="non_authoritative"`) enforces §11's "known and current" label-semantics invariant on a per-field basis for malformed `horizon_censored` values, so a corrupted audit field cannot admit a row with ambiguous censor provenance.
3. The slow-path `source_class = "reconstruction"` assignment is guarded so it no longer clobbers a more specific gating tag set upstream in the same loop.
4. Two focused tests lock in the parser and the gating behavior.
5. The pytest temp/cache ACL blocker is cleared and the full suite is re-verified locally.

Remaining work is advisory only:

1. Re-monitor `s* = 0.60` once enough fresh post-fix rows exist.
2. Measure `tau = 0.25h` empirically from real teardown latency when reliable deployment logs are available.

No deployment executor, runtime TTL, extra constants registry, 56-bar optional survival path, or additional document layer is required. The rails that prevent corruption of the bot-horizon contract are now in place end-to-end: central constants route H*, the BTK contract requires censor metadata, the slow-path parser refuses silent coercion, and the ingestion gate drops any row whose label semantics cannot be proven current.

## 16. Read-Only Audit Findings Appended 2026-04-17

This addendum records the latest read-only audit findings. No code was changed during the audit. The audit verified source integration with `rg`, reproduced one slow-path edge case in a throwaway temp directory, reran focused tests outside the sandbox, and reran the full test suite outside the sandbox.

Current test evidence from the audit:

1. Training-builder suite: `16 passed`.
2. Focused H* and censor-contract suite: `44 passed`.
3. Full test suite: `1095 passed, 35 warnings`.
4. The pass count in the existing verification record is correct, but the warning count is not current: this audit observed `35 warnings`, not `7 warnings`.

### 16.1 False Optionality

The following remain false optionality because current code and architecture prove they are not required for the H* structural fix:

1. Deployment executor or runtime TTL integration. `rg -n "max_runtime_hours" --glob "!DURATION_FIX.md" .` returned no matches, and the live payload path is not an order-placement or teardown executor. Adding a TTL field without a consumer would create an unenforced safety claim.
2. Treating D14.5 as code work. The user is the deployment teardown operator in the current architecture. No verified runtime component exists that could enforce teardown.
3. Making `horizon_censored` a model feature. It is label-process provenance. Using it as a feature would leak label-generation mechanics into the model.
4. Reintroducing a 56-bar survival horizon as an H* option. For canonical H* labels, survival must use 24 bars at 15m. A 56-bar study can exist only as a separately named non-H* analysis.
5. Replacing OU max half-life `48` with H* `24`. OU max is an estimator validity bound, not the bot inventory lifecycle.
6. Broad replacement of horizon-like literals. The audit verified separate semantic classes: label horizon, survival horizon, CPCV/t1 horizon, estimator validity, operator validation, and deployment operation.
7. Treating independent agent approval notes as implementation proof. They may be historical review context, but this audit can only treat local source, command output, and tests as proof.

### 16.2 Provably Unnecessary As Written

The following are unnecessary as written for the current fix:

1. A new constants registry. `src/neutralgrid/core/constants.py` already owns `BOT_HORIZON_HOURS`, `BOT_HORIZON_BARS_15M`, `BOT_HORIZON_SECONDS`, and `LABEL_CONTRACT_VERSION`.
2. A separate file for boolean parsing. `_as_bool_strict()` is local to `src/neutralgrid/training/unified_training_builder.py`, where the slow-path coercion risk exists.
3. A payload-only runtime TTL. Serialization without a consuming executor would not enforce teardown.
4. Rewriting all historical 56-bar documentation. `CRITERION_IMPLEMENT.md` already has a supersession header that marks 56-bar guidance as historical for bot-lifetime horizon.
5. Moving `horizon_censored` into `ALL_META_FEATURES`. Tests already assert it is excluded from model features.
6. Re-deriving OU max half-life solely because H* is 24 bars. OU admissibility must be tuned as an estimator policy, not inherited from bot lifecycle.
7. A full retrain as proof of source routing. Retraining may be operationally required before deployment, but code routing is proven by source and tests.
8. A 56-bar optional survival path in the active H* contract. It would add a second event definition without solving the current structural drift.

### 16.3 Not Valid To Strike Without Assumptions

The following items cannot be struck unless new evidence removes them:

1. Current-contract slow-path rows missing `horizon_censored`. The audit reproduced that a CSV row with `label_contract_version = 2026-04-17` but no `horizon_censored` survives when `include_reconstruction=True`: `rows 1`, `columns_has_horizon_censored False`, `candidate_survived True`, `source_class reconstruction`, `version_gated_col False`. This does not break the default builder path because `include_reconstruction=False` by default, but it is still an unclosed provenance edge case if reconstruction rows are intentionally included.
2. Survival threshold monitoring for `s* = 0.60`. H* routing proves the event window; it does not prove that the existing threshold remains optimal after the survival distribution changes.
3. Tau measurement for `BOT_INCLUSION_TOLERANCE_HOURS = 0.25`. The value is an explicit policy, not an empirically measured teardown-latency estimate.
4. Warning-count correction in the verification record. The current audit observed `1095 passed, 35 warnings`; treating `7 warnings` as current would be an assumption.
5. Sub-agent approval notes as proof. Without locally inspectable outputs, they cannot replace direct source/test verification.
6. Missing `horizon_censored` gating. Malformed non-missing values are now gated, but missing current-contract values are a separate case and are not proven closed by the existing tests.
7. The CRITERION supersession guard. Historical 56-bar text remains in `CRITERION_IMPLEMENT.md`, so the supersession header remains necessary to prevent old guidance from being treated as canonical.
