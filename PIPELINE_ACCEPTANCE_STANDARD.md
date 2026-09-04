# Pipeline Acceptance Standard

`Current status: Green`

## Purpose

This document defines the exact conditions under which
`run_full_pipeline.py` can be considered safe to use again for
deployment-candidate generation in this repository.

This is a plan-only release standard. It does not authorize implementation.
Implementation should not begin until the user explicitly approves it.

The target is not "guaranteed profit". That is not achievable in live
markets. The target is:

- valid deployment candidates
- positive-expectancy candidates under the repo's own economic model
- AFML-aligned training, validation, promotion, and deployment contracts
- no silent semantic drift between scan, enrich, backtest, and retraining

## Scope

This standard covers the real path that determines whether
`run_full_pipeline.py` can be trusted:

- `run_full_pipeline.py`
- `src/neutralgrid/scanner/scan.py`
- `src/neutralgrid/scanner/enrich_grid_params.py`
- `src/neutralgrid/scanner/pattern_profile.py`
- `src/neutralgrid/scanner/profile_model.py`
- `src/neutralgrid/scanner/pnl_ranker.py`
- `src/neutralgrid/scanner/empirical_profile_v20260302.py`
- `src/neutralgrid/scanner/mi_weighted_scorer_v20260311.py`
- `src/neutralgrid/scanner/two_stage_selector.py`
- `src/neutralgrid/validation/regime_validator.py`
- `src/neutralgrid/models/hmm/train.py`
- `src/neutralgrid/models/hmm/inference.py`
- `src/neutralgrid/models/hmm/retrain_orchestration.py`
- `src/neutralgrid/backtest/candidate_pipeline.py`
- `src/neutralgrid/backtest/cpcv.py`
- `src/neutralgrid/training/unified_training_builder.py`
- `src/neutralgrid/training/scanner_integration.py`
- `src/neutralgrid/training/data_generator.py`
- `src/neutralgrid/models/meta_labeler.py`
- `backtest/btk_unified_runner.py`
- `backtest_candidates.py`
- `retrain_hmm.py`
- `retrain_meta_labeler.py`

Out of scope as first-order release blockers:

- `backtest_candidates_current.py`
- version-constant fallback cleanup
- directional dead-state cleanup
- preventive config hardening not currently causing live-path drift
- retrain entrypoint default cleanup such as `n_iter` parity

These remain worthwhile hardening tasks, but they are not the main reasons the
current full pipeline is unsafe.

## What Safe Means Here

For this repo, safe to use again means all of the following are true:

- the active HMM, meta-labeler, and live scoring artifacts are built under the
  same semantics the runtime actually consumes
- scan-time ranking, enrichment-time validation, backtest export, and
  retraining rows use one stable meaning for regime, utility, range, score,
  horizon, and row authority
- ranking cannot override absolute economic invalidity, and the economic model
  used for that floor is unit-consistent
- non-authoritative, reconstructed, or version-gated rows cannot silently
  influence training or live artifact generation
- the authoritative rebuild path works end to end
- enrichment-time hard-gate failures fail closed explicitly rather than through
  accidental truthiness

Safe does not mean:

- guaranteed profitability
- immunity to regime shifts
- immunity to slippage, funding shocks, or exchange risk

## Why This Revision Changed The Prior Draft

The current version adds only code-backed missing contracts and trims points
that were too broad or too implementation-specific.

### Added because they are real live-path contracts

- `pattern_profile.py` and `profile_model.py` are now explicit scope items.
  Reason: scan-time similarity and profile probabilities are real runtime
  signals. They must stay aligned with the repaired range/score contract.
- Optional live artifacts now require fail-closed load behavior.
  Reason: runtime currently loads `mi_weights.json`, `conformal_quantile.json`,
  and threshold artifacts opportunistically. Rebuild policy alone is not
  enough; incompatible artifacts must be ignored or rejected at load time.
- Empirical EV-profile rerun governance is now explicit.
  Reason: `empirical_profile_v20260302.py` concatenates historical
  `backtest_results_*.csv` files without cross-run dedup or authoritative-row
  filtering.
- Green now requires authoritative `training_data_*.csv` as the rebuild-path
  source of truth.
  Reason: reconstruction fallback from raw `backtest_results_*.csv` is not an
  authoritative release path.
- `retrain_meta_labeler.py --include-backtest-data` is now treated as a bypass
  path that must enforce the same authority/version rules as the unified
  builder.
- Builder-side `t1` synthesis is now part of the horizon-integrity gate.
  Reason: horizon drift is not only an export problem; it can be silently
  perpetuated at ingestion.
- MI parity now means signal semantics and signal availability.
  Reason: scan-time MI scoring often lacks live `ev_score`, while MI artifact
  training currently assumes fuller backtest-time signal availability.
- Enrichment hard-gate exception semantics are now explicit.
  Reason: the current exception path stores `hard_gate_passed=None`, which only
  fails closed through incidental boolean coercion instead of an explicit
  failed-gate contract.
- EV funding-cost basis is now part of the economic-floor contract.
  Reason: if funding drag is not measured on the same capital basis as revenue
  and stop-loss terms, the absolute economic floor can remain mathematically
  optimistic even after rank-floor separation is repaired.

### Trimmed or narrowed because they were too broad

- The prior long audit transcript was removed from the standard.
  Reason: the acceptance standard should contain gates and reasons, not a full
  forensic log.
- Version gating is now described as inconsistent across entrypoints, not
  missing everywhere.
  Reason: `retrain_meta_labeler.py` already excludes `version_gated` rows by
  default; the blocker is uneven enforcement across the whole rebuild path.
- The identity gate no longer bans all `merge_asof` fallback logic.
  Reason: the blocking requirement is exact identity for newly produced rows.
  Legacy fallback logic can remain for historical data.
- The economic-floor gate is now defined as "ranking cannot override the
  authoritative absolute floor".
  Reason: the issue is percentile ranking overriding economic validity, not a
  need for a separate parallel profitability rule.

## Blocking Gates

### Gate 1: Authoritative Rebuild Path Must Execute

Why necessary:

- The repo cannot be recovered if the canonical backtest rebuild path does not
  compile and run.

Green requires:

- `backtest_candidates.py` compiles cleanly
- the script can regenerate authoritative `training_data_*.csv`
- conformal and MI artifact generation blocks are syntactically reachable

Current failure:

- `python -m py_compile backtest_candidates.py` fails at line 397

### Gate 2: HMM Promotion, Runtime, and Threshold Artifacts Must Share One Contract

Why necessary:

- The active HMM cannot be trusted if promotion, runtime gating, and optional
  threshold artifacts measure different regime semantics.

Green requires:

- walk-forward evaluation matches runtime regime semantics
- aggregated state-group probabilities, dominance logic, pass mode, and any
  post-processing used in runtime are the same ones measured for promotion
- threshold artifacts are rebuilt under the corrected contract, ignored, or
  rejected when incompatible
- rolling-window promotion policy matches actual realized training coverage

### Gate 3: HMM Posterior-Derived Training Metadata Must Respect Sequence Boundaries

Why necessary:

- Runtime tail correction and uncertainty logic consume metadata produced by
  training. That metadata must not be contaminated by stacked cross-symbol
  posterior decoding.

Green requires:

- posterior-derived state assignments used for uncertainty profiling preserve
  per-symbol boundaries
- regime-uncertainty, tail-risk, and volatility-tier metadata are boundary-safe

### Gate 4: Feature, Similarity, and Score Contracts Must Be Unique

Why necessary:

- The live path is unsafe when one field name or one artifact means different
  things across scan, enrichment, training, and backtest rebuild.

Green requires:

- `range_size_pct` has exactly one meaning end to end
- provisional scan volatility and validated enriched range are stored under
  distinct names if both are kept
- `utility_score` has one authoritative contract, or provisional and final
  utility are explicitly separated
- scan-stage score, deployment-stage score, and audit-only score lineage are
  explicit and do not share one overloaded field
- `PatternProfile` and `profile_model` consume the repaired scan feature
  contract, not the old overloaded one

### Gate 5: New-Row Identity and Lineage Must Be Exact

Why necessary:

- Newly produced scan, enrich, and backtest rows should not depend on time
  heuristics for lineage when exact identity can exist.

Green requires:

- every new scan-stage snapshot carries an immutable scan decision identifier
- that identifier survives through enrichment, export, and training ingestion
- exact joins are the default for newly produced rows
- HMM lineage fields are populated on both live and backtest rows
- `hmm_feature_source` is explicit rather than silently `None`

Not required:

- removing all legacy `merge_asof` fallback logic for historical rows

### Gate 6: Backtest Authority, Horizon, and Ingestion Governance Must Be Enforced

Why necessary:

- AFML training rows must represent the same event contract the live system is
  trying to predict.

Green requires:

- authoritative `training_data_*.csv` is the release-path backtest input
- Green does not rely on reconstruction from raw `backtest_results_*.csv`
- non-authoritative rows are excluded from training by default
- `source_class=reconstruction` rows are excluded from training by default
- `version_gated=True` rows are excluded by default across all ingestion paths
- fallback label synthesis is treated as non-authoritative
- row-level provenance includes `backtest_timestamp`, `backtest_run_id`,
  `engine_version`, `label_contract_version`, `formula_version`,
  `is_authoritative`, and HMM lineage
- truncated backtests do not retain a canonical 12h `t1`
- builder-side `t1` synthesis cannot silently legitimize broken horizons
- `retrain_meta_labeler.py --include-backtest-data` enforces the same
  authority/version rules as the unified builder

### Gate 7: Meta-Labeler Validation Must Be Temporal and Live-Compatible

Why necessary:

- The live pipeline uses the meta-labeler at scan time. Both the validation
  method and the deployed feature profile must match that reality.

Green requires:

- `MetaLabeler.train()` sorts internally by `timestamp_col` before any split
- CPCV preserves time-based purge and embargo even when symbol grouping is used
- calibration holdout is explicitly chronological after sorting
- the active meta-labeler artifact used at scan time is scan-safe, or scan-time
  meta-prob consumption is deferred until enrichment
- if the active artifact uses context features, synthetic context-feature
  backfills become blocking train/serve-skew issues rather than deferred cleanup

### Gate 8: Live Artifacts Must Match Runtime Semantics, Availability, and Provenance

Why necessary:

- `run_full_pipeline.py` and enrichment-time gating consume external artifacts
  that can silently skew ranking even if core code is repaired.

Green requires:

- conformal artifacts are fit from authoritative `y` on the authoritative path
- MI weights are trained from the same live signals they weight at runtime
- MI artifacts match runtime signal availability, not only signal naming
- empirical EV-profile and fill-rate artifacts use authoritative rows and
  deterministic rerun dedup
- pattern-profile artifacts are rebuilt when scan feature semantics change
- artifacts are rebuilt, disabled, or rejected at load time when incompatible
- HMM/meta-labeler artifact compatibility warnings are zero for the active pair

### Gate 9: Ranking Cannot Override The Absolute Economic Floor

Why necessary:

- Relative ranking alone is not enough. A negative-economics batch cannot be
  allowed to manufacture apparently strong candidates through percentile rank.
- The floor itself must be computed on consistent units. If funding drag is on
  a different capital basis from revenue and stop-loss terms, the repo can pass
  candidates whose economics are only apparently positive.

Green requires:

- exported deployment candidates satisfy one authoritative absolute economic
  floor under the repo's own economics
- fill revenue, funding drag, and adverse-loss terms in the EV floor use one
  consistent capital basis
- leverage-sensitive costs are not implicitly understated relative to
  margin-based revenue and loss terms
- percentile re-ranking cannot override that floor
- backtest candidate selection does not treat overloaded or purely relative
  scores as proof of profitability

### Gate 10: Enrichment Hard-Gate Exceptions Must Fail Closed Explicitly

Why necessary:

- A safety gate cannot rely on downstream truthiness accidents. The current
  behavior stores a tri-state sentinel on exception and only behaves
  fail-closed because `bool(None)` is false.
- The repo's own safety invariant expects exception-driven gate failure to
  return `False`, not an implicit falsy placeholder.

Green requires:

- hard-gate exception paths emit `hard_gate_passed=False`, not `None`
- exception reason metadata remains available separately from the boolean gate
- no downstream path can reinterpret exception-state candidates as
  "unknown-but-possibly-pass"

## Sequential Integration Plan

The steps below are ordered by dependency and live-path impact.
This is the implementation order to follow after user approval.

### Step 1: Restore the authoritative rebuild path

Files:

- `backtest_candidates.py`

Change:

- fix the syntax error
- make the conformal generation block logically reachable on the authoritative
  path

Why first:

- nothing else can be release-validated while the rebuild script is broken

### Step 2: Repair score, similarity, and identity contracts together

Files:

- `run_full_pipeline.py`
- `src/neutralgrid/scanner/scan.py`
- `src/neutralgrid/scanner/pattern_profile.py`
- `src/neutralgrid/scanner/profile_model.py`
- `src/neutralgrid/training/scanner_integration.py`
- `src/neutralgrid/training/data_generator.py`
- `src/neutralgrid/backtest/candidate_pipeline.py`
- `backtest_candidates.py`

Change:

- introduce a stable scan-stage decision identifier
- stop overloading one `score` field across scan, deployment rank, and rebuild
- make similarity/profile artifacts consume the repaired scan feature contract

Why second:

- score drift and identity drift are both already affecting the live path and
  rebuild path

### Step 3: Align HMM promotion, runtime, and uncertainty metadata

Files:

- `src/neutralgrid/models/hmm/train.py`
- `src/neutralgrid/models/hmm/inference.py`
- `src/neutralgrid/models/hmm/retrain_orchestration.py`
- `src/neutralgrid/validation/regime_validator.py`
- `retrain_hmm.py`

Change:

- make walk-forward evaluation match runtime regime semantics
- make posterior-derived training metadata boundary-safe
- rebuild or reject incompatible threshold artifacts
- keep rolling-window promotion aligned with real training coverage

Why here:

- the active HMM contract must be repaired before any rebuilt artifact is
  trustworthy

### Step 4: Unify feature semantics for range and utility

Files:

- `src/neutralgrid/scanner/feature_extractor.py`
- `src/neutralgrid/scanner/scan.py`
- `src/neutralgrid/scanner/enrich_grid_params.py`
- `src/neutralgrid/backtest/candidate_pipeline.py`
- `src/neutralgrid/training/unified_training_builder.py`
- `src/neutralgrid/training/scanner_integration.py`

Change:

- separate scan-time volatility proxy from validated enriched range
- separate or unify provisional versus authoritative utility semantics

Why here:

- candidate economics, similarity scoring, and model features cannot be trusted
  while the same field names mean different things across stages

### Step 5: Bind exported ranking to the absolute economic floor

Files:

- `run_full_pipeline.py`
- `src/neutralgrid/scanner/enrich_grid_params.py`
- `src/neutralgrid/scanner/pnl_ranker.py`
- `backtest_candidates.py`

Change:

- enforce one authoritative absolute economic floor
- make EV funding drag use the same capital basis as revenue and adverse-loss
  terms
- prevent leverage-sensitive costs from being understated inside the floor
- make enrichment hard-gate exception paths return explicit `False` rather than
  relying on falsy sentinel values
- ensure exported rank cannot override that floor

Why here:

- `backtest_candidates.py` already consumes exported `score`, so the repaired
  score contract must be economically grounded before artifact rebuild

### Step 6: Enforce authoritative backtest, horizon, and ingestion rules

Files:

- `src/neutralgrid/backtest/candidate_pipeline.py`
- `src/neutralgrid/training/unified_training_builder.py`
- `retrain_meta_labeler.py`

Change:

- hard-gate non-authoritative, reconstructed, and version-gated rows
- gate fallback label synthesis as non-authoritative
- preserve full provenance
- stamp horizon and authority from the actual realized event window

Why here:

- the meta-labeler and live artifacts cannot be rebuilt cleanly until
  authoritative rows are the default

### Step 7: Restore AFML-temporal meta-label validation and live feature compatibility

Files:

- `src/neutralgrid/models/meta_labeler.py`
- `src/neutralgrid/backtest/cpcv.py`
- `retrain_meta_labeler.py`

Change:

- sort internally before calibration and CPCV
- preserve time purge with symbol grouping
- ensure the deployed feature profile matches the scan-time call site, or defer
  scan-time meta-prob usage
- if context features are active, remove or replace synthetic training backfills

Why here:

- a rebuilt model is still unsafe if its validation is temporally wrong or if
  live inference relies on non-live synthetic features

### Step 8: Rebuild or fail-close all live artifacts on matching contracts

Files:

- `backtest_candidates.py`
- `src/neutralgrid/scanner/empirical_profile_v20260302.py`
- `src/neutralgrid/scanner/mi_weighted_scorer_v20260311.py`
- `src/neutralgrid/scanner/enrich_grid_params.py`
- `src/neutralgrid/validation/regime_validator.py`
- `run_full_pipeline.py`

Change:

- rebuild conformal, MI, empirical EV-profile, pattern-profile, and threshold
  artifacts under corrected authoritative contracts
- add compatibility validation so incompatible artifacts fail closed at load time
- dedup empirical-profile historical reruns deterministically

Why here:

- optional and semi-optional artifacts directly influence live ranking and
  gating, so stale or incompatible ones cannot be tolerated

### Step 9: Rebuild core artifacts after the contract fixes

Operational steps:

1. retrain and promote the HMM
2. regenerate authoritative `training_data_*.csv`
3. retrain the meta-labeler from corrected authoritative rows
4. rebuild or remove conformal, MI, empirical EV-profile, pattern-profile,
   and threshold artifacts

Why here:

- code fixes do not make the live system safe until stale artifacts are replaced

### Step 10: Verification, smoke run, and manual audit

Verification must confirm:

- the authoritative rebuild path compiles and runs
- corrected HMM and meta-labeler artifacts are active
- live artifacts are rebuilt, disabled, or rejected at load time when incompatible
- zero active HMM/meta-labeler compatibility issues remain
- no duplicate new-row identities
- no valid row is missing lineage
- no overloaded score or range fields survive final output
- no deployment-valid row violates the absolute economic floor
- no enrichment hard-gate exception path emits an ambiguous tri-state pass value
- manual review of the top 20 candidates finds no semantic contradictions

## Verification Gates

Green status requires all of the following:

- authoritative rebuild path passes
- corrected HMM artifacts are active
- corrected meta-labeler artifact is active
- live artifacts are compatible, rebuilt, disabled, or fail-closed
- targeted regression tests pass
- smoke run passes
- manual top-20 audit passes

## Deferred Hardening After Green

These remain valid, but they should not displace the first integration pass.

- unify retrain entrypoint defaults such as `n_iter`
- harden directional dead-state mapping
- add stricter config cross-validation in `config.py`
- finish cleaning `primary_pipeline_score` into pure audit metadata everywhere
- improve output filename uniqueness for data-management hygiene

## Final Acceptance Rule

`run_full_pipeline.py` is release-accepted only if all blocking gates and all
verification gates pass after artifact rebuild.

If any blocking gate fails, the repo remains Red.

If code is fixed but artifacts are not rebuilt, or live artifacts are disabled
pending rebuild, the repo is Amber.

Only a full pass is Green.

## One-Line Summary

This repo is safe to use again only when the live path, rebuild path, and all
active artifacts agree on the same definitions of regime, score, similarity,
range, utility, horizon, row authority, and positive expectancy.
