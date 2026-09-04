# SCORING_FIX

Progress: [######] 6/6 phases complete

| Phase | Status | Pyright | Tests | Proof | Notes |
| --- | --- | --- | --- | --- | --- |
| Phase 0 | Complete | Pass | Blocked by temp ACL cleanup after 23 passes | Pass | Proof scaffolding landed in `SCORING_FIX.md`, `retrain_meta_labeler.py`, and `utility_calibrator.py` |
| Phase 1 | Complete | Pass | Pass (`30` passed) | Pass | Latest runtime CSV dropped bypass `grid_generation + regime_validation_failed` from baseline `40` to `0`; bypass rows now surface Stage B / approved outcomes |
| Phase 2 | Complete | Pass | Pass on non-temp meta slices; `tmp_path` bundle blocked by temp ACL | Pass | Active meta retrain now uses `fast_winner_target`; proof shows `8` features, `75` positives, `151` negatives, active HMM lineage match, and post-save loadability |
| Phase 3 | Complete | Pass | Pass (`28` passed) | Runtime CSV stale; current code recompute passes | Saved runtime CSV still shows old EV-first ordering, but reapplying current `_apply_afml_post_scoring()` to that CSV produces meta-first order immediately |
| Phase 4 | Complete | Pass | Pass on non-temp parity slices; one existing temp-fixture test blocked | Pass | Empirical profile now fits `duration_hours <= 7.0`; saved runtime CSV predates this scoring change, but current code recompute uses the fast-horizon EV basis |
| Phase 5 | Complete | Pass | Pass (`14` passed) | Pass | Utility calibration now uses the fast-horizon pool (`69` rows, `45` positives, `24` negatives); promotion stayed `false` only because `G7_finite_nonnegative_not_boundary_pinned=false` |
| Phase 6 | Complete | Pass | Pass (`74` passed combined regression sweep) | Manual full-pipeline CSV pending | Final review done; environment blockers recorded; review agents shut down after this pass |

## Goal

Make `enrich + post scoring` select candidates closer to the current fast-winner cohort:

- profitable
- `duration_hours < 7`

Production code has been modified. This document now serves as the implementation tracker and proof log.

## Audited Scope

Runtime and ranking:

- `src/neutralgrid/scanner/enrich_grid_params.py`
- `src/neutralgrid/scanner/two_stage_selector.py`
- `src/neutralgrid/grid/calculator.py`
- `src/neutralgrid/validation/microstructure_hard_gate.py`
- `src/neutralgrid/scanner/scan.py`
- `run_full_pipeline.py`
- `src/neutralgrid/scanner/pnl_ranker.py`
- `src/neutralgrid/scanner/empirical_profile_v20260302.py`
- `src/neutralgrid/models/meta_labeler.py`
- `src/neutralgrid/models/artifact_compat.py`

Training and artifact alignment:

- `retrain_meta_labeler.py`
- `src/neutralgrid/training/unified_training_builder.py`
- `src/neutralgrid/training/data_generator.py`
- `src/neutralgrid/training/scanner_integration.py`
- `src/neutralgrid/validation/utility.py`
- `src/neutralgrid/calibration/utility_calibrator.py`
- `src/neutralgrid/training/live_outcome_ingestor.py`

Reference and evidence:

- `reports/winners_vs_architecture.py`
- `data/new_expired_bots.xlsx`
- `results/deployment_ready_20260422_121821.csv`
- `models/meta_labeler/metadata.json`
- `artifact_manifest.json`
- `artifacts/utility/utility_20260422_125814_312065.json`

Tests reviewed or executed:

- `tests/unit/test_artifact_compatibility.py`
- `tests/unit/test_meta_labeler_inference_aliases.py`
- `tests/unit/test_enrich_grid_params.py`
- `tests/unit/test_enrich_grid_params_survival_recalc.py`
- `tests/test_micro_osc_integration.py`
- `tests/unit/test_utility_calibrator.py`
- `tests/unit/test_unified_training_builder.py`
- `tests/unit/test_scanner_integration_v20260320.py`

## Verified Baseline

Current fast-winner reference:

- `reports/winners_vs_architecture.py:19-20` defines fast winners as profitable rows with `duration_hours < 7`.
- Current workbook check on `data/new_expired_bots.xlsx`:
  - total rows: `192`
  - fast winners: `52`
  - fast-winner median `duration_hours`: `4.6`
  - fast-winner median `pnl_pct`: `5.535`
  - fast-winner median `grids_count`: `40`
  - fast-winner median `grid_spacing_pct`: `0.7535`
  - fast-winner median `range_size_pct`: `9.282`
  - fast-winner median `adx_1h`: `36.68`

Current live pipeline output:

- `results/deployment_ready_20260422_121821.csv` has `250` rows.
- Only `1` row is `grid_is_valid=True`.
- Only `1` row is `stage_b_approved=True`.
- `49` rows have `micro_osc_bypass=True`.
- Of those `49` bypass rows:
  - `45` fail at `failure_stage=grid_generation`
  - `40` of those `45` fail with `grid_reason=regime_validation_failed`
- `meta_prob` is non-null on `0` rows.
- `ev_score` is non-null on `5` rows.
- `deployment_score` is non-null on `1` row.

Current backtest evidence already supports a fast `<7h` retrain target:

- deduped backtest rows in `data/backtest_candidates`: `621`
- deduped rows with `duration_hours <= 7`: `463`
- of those `463` rows:
  - positive `net_pnl_pct > 0`: `251`
  - non-positive `net_pnl_pct <= 0`: `212`

## Why The Discrepancy Exists

### 1. Enrich is still blocking the intended fast path before post scoring can help.

Verified facts:

- `enrich_grid_params.py:347-352` creates `micro_osc_bypass` as an eligibility bypass.
- `enrich_grid_params.py:1769-1774` avoids early regime return for bypass rows.
- But `enrich_grid_params.py:991-996` only force-validates an invalid `ValidationResult` in `discovery_mode`.
- `calculator.py:513-518` still rejects any invalid validation object immediately.

Result:

- The current bypass is not a full live-path bypass.
- Most bypass candidates still die in `grid_generation` because geometry sees `validation_result.is_valid=False`.
- This is the biggest live enrich mismatch with the current fast-winner behavior.

### 2. The first strict post-geometry authority already exists, but too few rows reach it.

Verified facts:

- `enrich_grid_params.py:1240-1242` recalculates `survival_prob` from generated geometry.
- `two_stage_selector.py:157-166` uses `survival_prob` for micro-oscillator Gate 4.
- In the latest live output only `1` row fails at Stage B, and that failure is survival-based.

Result:

- TOS tuning is not the first justified fix.
- Hard-gate tuning is not the first justified fix.
- The immediate problem is that bypass rows are not reaching Stage B in the first place.

### 3. Post scoring is not using the fast-winner classifier at all.

Verified facts:

- `run_full_pipeline.py:104` loads meta "for audit, not used in ranking".
- `run_full_pipeline.py:196-205` computes `meta_prob` only for audit/logging.
- `run_full_pipeline.py:243-249` builds `deployment_score` from EV ranking over valid rows.

Result:

- Even with a healthy meta model, current deployment ranking is still EV-only.
- This is a direct structural mismatch with a fast-winner objective.

### 4. The current meta artifact is offline in practice.

Verified facts:

- `models/meta_labeler/metadata.json:7` ties the saved meta artifact to HMM `rolling_180d_20260412_001444`.
- `artifact_manifest.json:3` shows the active HMM is `rolling_180d_20260421_213328`.
- `artifact_compat.py:65-76` raises on lineage mismatch.
- `run_full_pipeline.py:657-668` swallows load failure and falls back to `None`.

Result:

- The current meta model cannot load against the active HMM lineage.
- `meta_prob` stays null.
- The discrepancy is hidden because the load failure degrades to `meta=None`.

### 5. The active meta contract is intentionally minimal, but the current retrain path is not explicitly fast-7h.

Verified facts:

- `meta_labeler.py:123-134` sets the active bootstrap profile to exactly 8 features:
  - `range_prob`
  - `survival_prob`
  - `utility_score`
  - `ou_halflife`
  - `profit_per_grid_pct`
  - `num_grids`
  - `grid_spacing_pct`
  - `adx_1h`
- `retrain_meta_labeler.py:784-796` trains using precomputed `y`.
- `unified_training_builder.py:755-818` populates `y` from hierarchical or hurdle-based precedence, not an explicit `duration_hours < 7 and profit > 0` contract.

Result:

- The active feature set is already bootstrap-sized, which is good.
- The target contract is still not explicitly the fast-winner contract.
- That must be fixed before the classifier can be trusted as the primary post-score.

### 6. EV calibration is trained on the wrong cohort and a different EV basis.

Verified facts:

- `empirical_profile_v20260302.py:149-154` keeps good deterministic dedup.
- `empirical_profile_v20260302.py` fits on all deduped rows; it does not restrict to `duration_hours <= 7`.
- Deduped profile population currently includes `158` rows with `duration_hours > 7`.
- `empirical_profile_v20260302.py:100-109` defines analytical EV.
- `pnl_ranker.py:265-271` applies funding on a leverage-adjusted margin basis during live ranking.

Result:

- The current EV alignment profile is not fast-winner specific.
- The fit-time EV basis and live EV basis are not the same.
- Current post-scoring is therefore ranking with a calibration layer learned on the wrong target and a mismatched formula.

### 7. Utility calibration is misaligned to `<7h`, but it is not phase-1 critical.

Verified facts:

- `utility_calibrator.py:6-8` defines labels using `duration_hours > 7`.
- `utility_calibrator.py:479-482` writes that same `>7h` contract into the candidate artifact.
- `utility.py:82-115` falls back to pinned v0 when `artifacts/utility/current.json` is absent.
- `artifacts/utility/current.json` is absent in this workspace.
- `artifacts/utility/utility_20260422_125814_312065.json:61-62` is non-promotable.
- `utility.py:418-423` shows governed provisional utility already has one consistent meaning across runtime/training/backfill.

Result:

- The utility artifact path is misaligned to the fast-winner target.
- But the runtime is not currently using any promoted utility artifact anyway.
- Utility recalibration should be phase 2, not the first blocker on enrich/post scoring.

## Immediate Integration Plan

This is the shortest architecture-coherent path. It uses the existing pipeline shape and avoids new subsystems.

### Step 1. Complete the existing `micro_osc_bypass` through live grid generation.

Files:

- `src/neutralgrid/scanner/enrich_grid_params.py`
- `tests/test_micro_osc_integration.py`
- `tests/unit/test_enrich_grid_params.py`

Change:

- For `rd.micro_osc_bypass=True`, reuse the same non-gating `grid_vres` pattern currently used only in `discovery_mode` at `enrich_grid_params.py:991-996`.
- Do this only when geometry inputs already exist or can be recovered by `_fill_discovery_geometry_from_market_data()`.
- Do not add a second bypass subsystem.

Why this is required:

- The current intended fast path is already present conceptually.
- It is simply not completed for the live path.
- This is the cleanest way to stop losing bypass rows at `grid_generation -> regime_validation_failed`.

Verification proof:

- Add one unit test proving a bypass row with invalid regime validation can still reach grid generation on the live path.
- Re-run pipeline and compare against the current baseline:
  - current baseline: `40` rows in `micro_osc_bypass + grid_generation + regime_validation_failed`
  - expected proof: this count must drop

### Step 2. Keep Stage B micro-osc survival gating as the first strict authority after geometry.

Files:

- `src/neutralgrid/scanner/two_stage_selector.py`
- `tests/unit/test_enrich_grid_params_survival_recalc.py`
- `tests/test_micro_osc_integration.py`

Change:

- Do not retune TOS first.
- Do not retune hard gate first.
- Preserve the current architecture where bypass rows reach geometry first, then face Stage B survival gating.

Why this is required:

- The current code already expresses the right authority order for this path.
- The problem is insufficient flow into that authority, not the authority itself.

Verification proof:

- After Step 1, newly surviving bypass rows should fail or pass with explicit Stage B reasons instead of dying as `regime_validation_failed`.

### Step 3. Retrain the active bootstrap meta model on an explicit fast-winner target, then restore lineage compatibility.

Files:

- `retrain_meta_labeler.py`
- `models/meta_labeler.pkl`
- `models/meta_labeler/metadata.json`
- `tests/unit/test_artifact_compatibility.py`
- `tests/unit/test_meta_labeler_inference_aliases.py`

Change:

- Keep the active 8-feature bootstrap profile from `meta_labeler.py:123-134`.
- After `UnifiedTrainingBuilder.build()` in `retrain_meta_labeler.py`, derive an explicit fast target from the joined training table:
  - positive: `duration_hours <= 7.0 and net_pnl_pct > 0`
  - negative: `duration_hours <= 7.0 and net_pnl_pct <= 0`
- Train the bootstrap profile on that explicit target instead of relying on the current precomputed `y` precedence.
- Refresh the artifact so its lineage matches the active HMM in `artifact_manifest.json`.
- Keep exact-feature, non-imputed bootstrap behavior.

Why this is required:

- The current active feature contract is already minimal enough for bootstrap.
- The current target contract is not explicitly the fast-winner contract.
- The current artifact cannot load anyway because its lineage is stale.

Verification proof:

- `MetaLabeler.load(Path("models/meta_labeler.pkl"))` must succeed.
- `models/meta_labeler/metadata.json` lineage must match `artifact_manifest.json`.
- Metadata must state the explicit fast target used for retrain.
- Post-enrichment rows with full 8-feature coverage must start receiving non-null `meta_prob`.

### Step 4. Make post scoring use `meta_prob` as the primary deployment ranking and `ev_score` as the secondary economics tie-breaker.

Files:

- `run_full_pipeline.py`
- `tests/test_afml_bugfixes.py`

Change:

- Keep meta strictly post-enrichment; do not try to move it earlier than geometry.
- Stop treating `meta_prob` as audit-only.
- For `grid_is_valid=True` rows:
  - primary sort key: `meta_prob` descending
  - secondary sort key: `ev_score` descending
- Preserve the original scan-time `score` column.
- Do not invent a weighted blend constant.
- Do not create a new `score_afml` path.

Why this is required:

- The classifier models fast-horizon win likelihood.
- EV models economic quality conditional on a viable candidate.
- The reviewed code does not prove a correct blend weight, so a weighted blend would be assumption, not evidence.

Verification proof:

- Add one deterministic unit test on a small synthetic dataframe proving sort precedence.
- In runtime output, `deployment_score` must no longer be equivalent to pure EV percentile order.

### Step 5. Refit empirical EV alignment on the same fast-horizon universe and unify the EV formula.

Files:

- `src/neutralgrid/scanner/empirical_profile_v20260302.py`
- `src/neutralgrid/scanner/pnl_ranker.py`
- relevant AFML scoring tests

Change:

- Keep existing dedup.
- Restrict profile-fit rows to the same fast target universe used by the deployment objective: `duration_hours <= 7.0`.
- Make fit-time analytical EV use the same funding/leverage basis as live `PnLRanker.compute_score()`.
- Keep current context system (`symbol`, `regime`, `global`) for phase 1.

Why this is required:

- Current EV alignment mixes fast and slow cohorts.
- Current fit-time EV and live-time EV are not the same quantity.

Verification proof:

- Profile metadata must record the fast-horizon eligible universe.
- Add one parity test proving fit-time analytical EV and live EV use the same basis on the same row schema.

### Step 6. Keep utility and feature expansion in phase 2, after data flow is stable.

Files:

- phase 2 only: `src/neutralgrid/calibration/utility_calibrator.py`
- phase 2 only: any follow-up meta feature-contract change

Change now:

- Do not block phase 1 on utility recalibration.
- Do not expand the active 8-feature bootstrap contract yet.
- Keep governed provisional utility as-is for phase 1.

Why this is required:

- Runtime is still on fallback utility because `artifacts/utility/current.json` does not exist.
- Retargeting the utility calibrator alone will not fix the current enrich/post bottleneck.
- The active 8-feature contract is already the repo's deliberate bootstrap profile.
- Feature additions such as `range_size_pct` may be reasonable later, but the reviewed code does not prove they are the first necessary change.

Phase-2 reactivation gate:

- start phase 2 only after all of these are true:
  - the meta artifact loads against the active HMM lineage
  - post-enrichment rows receive non-null `meta_prob`
  - bypass rows are no longer predominantly dying as `grid_generation + regime_validation_failed`

## Provable False Optionality

- Treating `micro_osc_bypass` as a full live bypass today
  - false as written; live geometry still rejects invalid validation objects
- Treating meta availability as meaningful to ranking today
  - false as written; ranking is EV-only
- Treating `survival_prob` as a geometry input today
  - false as written; `calculator.py` accepts it but only references it in the function signature and docstring
- Treating hard-gate profitability as the live profitability lever
  - false as written; final profitability rejection happens in `calculator.py:602-608`
- Treating `RankingConfig.pnl_hurdle_pct` as an active ranking control
  - false as written; it is defined in `pnl_ranker.py:32-33` but not used in live ranking logic

## Provably Unnecessary Items As Written

- Retuning TOS before fixing the incomplete bypass path
- Retuning `edge_tier.medium_buffer_pct` or `big_buffer_pct` before fixing the incomplete bypass path
- Rewriting `LiveOutcomeIngestor` for the immediate enrich/post fix
- Pulling `scripts/build_calibration_dataset.py` or `scripts/calibrate_from_execution.py` into phase 1
- Creating a new scoring pipeline, new pointer system, or new artifact family for this fix
- Creating a weighted `meta + EV` blend constant without evidence

## Items Not Valid To Strike Without Assumptions

- dedup in the backtest/training builders
- two-class retention for training/calibration
- HMM/meta lineage compatibility guard
- exact non-imputed active bootstrap feature contract
- hard-gate checks 1-4 for round-trip cost, spread/profit, funding, and liquidity
- the distinction between `dynamic_profit_floor_pct` and `micro_min_profit_required_pct`
- governed provisional utility fallback semantics while no promoted utility artifact exists

Also not valid to change without new evidence:

- adding or removing features from the active 8-feature bootstrap profile
- introducing a new context hierarchy for EV alignment
- lowering hard-coded thresholds globally before the current flow bottleneck is fixed

## Sub-Agent Validation

`Herschel` validated the enrich path:

- confirmed the bypass is incomplete on the live path
- confirmed TOS and hard gate are not the first-order blocker
- confirmed current profitability misses are geometry-time, not hard-gate-time

`Parfit` validated post-scoring:

- confirmed ranking is EV-only today
- confirmed the active meta artifact fails lineage compatibility
- confirmed EV alignment is trained on the wrong cohort and a mismatched EV basis

`Planck` validated the training and artifact slice:

- correctly confirmed the utility calibrator is still on a `>7h` contract
- correctly confirmed no promoted utility artifact is live
- correctly confirmed the unified builder does not train directly from `new_expired_bots.xlsx`

One point was narrowed after local verification:

- `scanner_integration.py` does default `min_features_required` to the full active profile when validation is enabled
- but `run_full_pipeline.py:857` creates the collector through `create_collector(...)`, and `scanner_integration.py:438-449` defaults that runtime path to `validate_snapshots=False`
- so snapshot-validation relaxation is not a primary live fix in the current call path

## Test Evidence

Executed successfully:

- `pytest -q tests/unit/test_artifact_compatibility.py tests/unit/test_meta_labeler_inference_aliases.py tests/unit/test_enrich_grid_params.py tests/unit/test_enrich_grid_params_survival_recalc.py tests/test_micro_osc_integration.py -k "not slow"`
- result: `34 passed`

Historically session-dependent, not a global repo blocker:

- `tests/unit/test_utility_calibrator.py`
- `tests/unit/test_unified_training_builder.py`
- `tests/unit/test_scanner_integration_v20260320.py`

Observed outcome in the agent session that produced the original note:

- `17` tests passed
- `16` tests in `test_unified_training_builder.py` errored on Windows temp-directory permission access to `pytest-of-cris_`

Later user-run verification reported the same bundle passing cleanly (`37` tests passed). Treat this as a session-local temp-directory issue, not a standing repo defect.

Operational rule:

- do not run ACL cleanup commands preemptively
- only consider temp-directory cleanup if the exact `PermissionError` reproduces in the session that is currently running pytest

## Bottom Line

Priority order is:

1. finish the live `micro_osc_bypass` path into grid generation
2. keep Stage B as the first strict authority after geometry
3. retrain the existing 8-feature bootstrap meta model on an explicit fast `<7h` target and restore HMM lineage compatibility
4. rank valid candidates by `meta_prob` first and `ev_score` second
5. refit EV alignment on the same `<7h` universe using the same EV basis as live ranking
6. leave utility recalibration and feature expansion for phase 2, after data flow is stable

That is the smallest verifiable plan that directly addresses the current structural discrepancy without adding false optionality.
