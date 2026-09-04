# METALABELER_LIVETELEMETRY

> **Status report — regenerated 2026-06-18** from a 10-agent, read-only verification
> (workflow run `wf_abd5b8c8-514`) against the live repo, with an independent adversarial
> recheck of every non-confirmed claim and reproduction of the doc's own commands.
> Every **DONE** item below is backed by `file:line` evidence or observed command output.
> Temporary file — delete only after the operator confirms the live-decision goal is achieved.
>
> **2026-06-20 implementation update.** Phase 1 of the live-decision wiring is **CODE
> COMPLETE and shipped DISABLED (shadow-mode)**: D2 (full-fidelity features by reuse),
> D3 (live authority + fidelity), D4 (bounded **caution-only** tilt, `meta_tilt_enabled=False`
> default), and D7 (live audit). `meta_proba` is now computed full-fidelity, logged, and
> **wired into `recommender.decide()`** as a shadow signal that changes **no verdict** until
> calibrated (D5) and proven OOS (D6) — both Phase 2. A pre-existing live-path defect was
> fixed in passing (funding_rate/open_interest were raw Binance payloads, not scalars —
> the overlay had never run against real Binance). Verified live `--once`: full-fidelity
> `meta_proba`, `meta_authoritative=true`, complete audit trail. See the 2026-06-20
> changelog entry + Assumptions section at the bottom.

## Goal

Correctly implement the meta-labeler into **LIVE telemetry scanner decision-making** so it
can improve deployment accuracy **without leaking labels or creating circular scanner feedback**.

## Progress

```
Foundation & batch admission (8 items)  ████████████████████ 100%   (8/8 DONE)
Documentation hygiene        (1 item)   ████████████████████ 100%   (1/1 DONE)
LIVE decision-making — THE GOAL (7)     ██████████████░░░░░░  71%   (5/7 DONE; D5/D6 Phase 2)
────────────────────────────────────────────────────────────────────────────
OVERALL                      (16 items) █████████████████░░░  88%  (14/16 DONE)
```

> ✅ **The core goal is wired (shadow-mode), not yet enabled.** As of 2026-06-20 `meta_proba`
> is computed full-fidelity (D2), gated by live authority + fidelity (D3), persisted to the
> live audit trail (D7), and **consumed by `recommender.decide()`** as a bounded **caution-only**
> tilt (D4). The tilt ships **DISABLED** (`meta_tilt_enabled=False`, `meta_tilt_low_threshold=0.0`)
> so it changes **no live verdict** — it is computed and logged (`meta_would_tilt`) for shadow
> analysis only. The goal is **ACHIEVED (enabled)** only after **D5** (calibrate the threshold
> from shadow data) and **D6** (prove OOS accuracy improvement), then flipping the flag.

## Current state (verified on-disk truth)

| Fact | Value | Evidence |
|---|---|---|
| Active HMM | `rolling_180d_20260617_151954` (promoted `2026-06-18T05:31:13Z`) | `artifact_manifest.json:3,5` |
| Meta-labeler lineage pin | `rolling_180d_20260617_151954` (**matches active**) | `models/meta_labeler/metadata.json:7` |
| Meta-labeler artifact | `20260617_153639`, trained `2026-06-17T15:36:39Z`, `total_samples=5737` | `metadata.json:2,3,15` |
| `MetaLabeler.load(...)` | **succeeds → `True pass`** (no longer raises a lineage `ValueError`) | runtime, reproduced |
| Live monitor load | `MonitorContext.create()` → `meta_labeler` present, `meta_unavailable=False` | runtime, reproduced |
| Deployed estimator | `LogisticRegression` (beta-OOS calibrated), `oof_auc=0.7807`, `n_pos=2611`, `pass` | `joblib.load` + `metadata.json:73,77,78` |

## Implementation checklist

Legend: ✅ DONE (verified) · ☐ TODO

### A. Lineage & artifact — foundation
- ✅ **A1 — Resolve HMM/meta lineage so the artifact loads.** Active == linked == `rolling_180d_20260617_151954`; `MetaLabeler.load` returns `True pass`. Promotion done via the canonical `promote_hmm_version` (no manifest hand-edit, no `artifact_compat.py` bypass). *(`artifact_manifest.json:3`; `metadata.json:7`; `artifact_compat.py:40-45,76` gate enforced at `meta_labeler.py:2138-2141`)*
- ✅ **A2 — Loaded model is trained + promoted.** `is_trained=True`, `promotion_status='pass'`, `is_promoted=True`. *(loaded pkl; `metadata.json:78`)*
- ✅ **A3 — HMM eligible for promotion.** 0617 HMM: `mean_pass_rate=1.0` (≥0.50), `total_samples=858000` (not truncated), identity temperature scaler (`temp=1.0`, `fitted=false`). *(`artifacts/hmm/rolling_180d_20260617_151954/metadata.json:62,97`; `temperature_scaler.json`)*

### B. Batch deployment-admission path
- ✅ **B1 — Scan-time `scan_meta_prob` kept telemetry-only.** Written at `scan.py:715`, passthrough at `enrich_grid_params.py:477`, carried to training at `live_outcome_ingestor.py:76`; never consumed for ranking / sizing / admission.
- ✅ **B2 — Enrich-time fail-closed authority.** `deployment_meta_prob` is set iff `promotion_status=='pass'` (string-equality, fail-closed for None/legacy/mock). *(`enrich_grid_params.py:1892-1897`; `run_full_pipeline.py:144,297-301`)* Tests `test_promoted_meta_labeler_is_authoritative_and_drives_kelly` + `test_runtime_capital_and_live_sizers_are_authoritative` pass.
- ✅ **B3 — Batch deployment audit columns stamped.** `deployment_ready_*.csv` carries `meta_feature_profile`, `meta_labeler_hmm_artifact_version`, `stage_b_meta_gate_enabled` (additive +21-line block right after the `active_hmm_artifact_version` stamp). Deploy-linkage append-only header deliberately **not** schema-changed (joins by `candidate_id`). *(`run_full_pipeline.py:1062-1083`; `candidate_deploy_linker.py:53-80` unchanged)*
- ✅ **B4 — Leakage / circular-feedback guards intact.** 20 active features, **zero** overlap with `_KNOWN_LABEL_COLUMNS`, `hlabel` not a feature; both guard sites present and identical; FASTWIN profile HMM-free (no `*_prob`). *(`meta_labeler.py:705-708`, `1056-1059`, `137-167`)* Leak/contract subset `53 passed / 1149 deselected`; full suite `1418 passed`; pyright clean.
- ✅ **B5 — Stage B meta gate intentionally OFF + diagnostic.** `meta_gate_enabled=False`, `min_meta_prob=0.50` (static, **uncalibrated**); gate is fail-closed with rejection code `data_missing:meta`. Satisfies Done-Criterion 5 in its "off + diagnostic" form. *(`config.py:306-307`; `two_stage_selector.py:225-243`)* See **D5** for the enable path.

### C. Documentation hygiene
- ✅ **C1 — Stale top "Current blocker" section removed.** The prior file asserted, in present tense, that the active HMM was `…0601_134113` and that `MetaLabeler.load` raises a lineage mismatch — both false after the 2026-06-18 promotion, with no strike or forward-pointer. Resolved by this rewrite; current state is stated above from verified disk truth.

### D. LIVE telemetry decision-making — **the actual goal** (D1–D4, D7 DONE; D5/D6 Phase 2)
- ✅ **D1 — Live fail-OPEN feature guard fixed (2026-06-18).** The required-feature count now comes from `get_missing_feature_names({})` (the loaded model exposes no `get_feature_names` accessor), so the >50%-missing **ratio guard actually runs**, and the `AttributeError` branch now **fails closed** (`meta_overlay_inactive`) instead of silently predicting on imputed features. The `_build_meta_feature_dict` docstring inventory was corrected to the true imputed-5 (`ev_score`, `hurst_exponent`, `micro_round_trip_cost_pct`, `ou_halflife`, `profit_per_grid_pct`; dropped `survival_prob`, which is not an active feature). Regression tests added in `tests/unit/test_decision_monitor.py`: `test_meta_overlay_inactive_when_majority_features_missing`, `…_active_with_minority_features_missing`, `…_fails_closed_without_introspection_method`. Validated: pyright 0 errors; `pytest tests/` → **1421 passed**. *(`src/neutralgrid/live/decision/monitor.py` meta block + `_build_meta_feature_dict`)*
- ✅ **D2 — Live feature parity by REUSE (2026-06-20).** All 20 features now computed live via the canonical paths (no train-serve skew, no new scripts): `hurst_exponent`/`ou_halflife` (+`survival_prob` for EV) via `scanner.scan._compute_stochastic_features` (≥300-bar guard inside); `micro_round_trip_cost_pct` threaded from the micro `costs` (estimated once, `monitor._estimate_micro_costs`); **geometric** `grid_spacing_pct`/`profit_per_grid_pct` (PERCENT) via `grid.formulas` + config fees (`monitor._geometric_grid_features`); `ev_score` via `PnLRanker.compute_score().rank_score` (`monitor._compute_ev_score`). Missing features are left **absent (not imputed)** so the strict-0-missing fidelity gate fails closed. **Also fixed a pre-existing defect:** funding_rate/open_interest were passed to `compute_features` as raw Binance payloads (history list / OI dict), not scalars — `_extract_funding_rate`/`_extract_open_interest` now mirror `scan.py`. *(`monitor.py` `_build_meta_feature_dict`, `_geometric_grid_features`, `_compute_ev_score`, `_estimate_micro_costs`, `_extract_funding_rate`, `_extract_open_interest`)* Verified live `--once`: full-fidelity `meta_proba`, 0 missing features.
- ✅ **D3 — Live authority + fidelity semantics (2026-06-20).** `MonitorContext.meta_promoted` set from `MetaLabeler.is_promoted` (== `promotion_status=='pass'`, fail-closed), mirroring `enrich_grid_params.py:1892-1894`. `BotEvaluation` gains `meta_authoritative` (promoted **and** `meta_proba is not None` **and** full fidelity) + `meta_full_fidelity` (strict 0-missing). Only an authoritative + full-fidelity `meta_proba` may influence a verdict.
- ✅ **D4 — `meta_proba` wired into `recommender.decide()` as a bounded caution-only tilt (2026-06-20, shipped DISABLED).** Inside `if not end_reasons:` only: when authoritative + full-fidelity + `meta_proba < meta_tilt_low_threshold` **and an ADJUST already exists**, it escalates one tick sooner and adds a `meta_low_confidence` reason. It **never** creates/cancels an END, **never** originates or suppresses an ADJUST, and is **never** fed back into scan-time selection. New `RecommenderConfig` knobs `meta_tilt_enabled=False` + `meta_tilt_low_threshold=0.0` make it inert by default (computed + logged as `meta_would_tilt` for shadow analysis). Focused tests cover authority/fidelity gating, never-flips-END, never-originates-ADJUST, escalate-sooner, and default-inert.
- ☐ **D5 — Calibrate `meta_tilt_low_threshold` (Phase 2)** from the shadow-logged `meta_proba` vs ingested outcomes before enabling. No operating point is persisted in the artifact (`metadata.json:55-80`) and the F1 threshold targets the fast-winner label, not the live ADJUST decision — so the threshold is chosen empirically from shadow data, then set in the live `--config-file`. (The separate scan-time Stage-B `min_meta_prob` gate is out of scope; leave OFF.)
- ☐ **D6 — Prove OOS accuracy improvement (Phase 2)** as offline analysis (no new permanent script/gate): join the D7 JSONL (`meta_proba`, `meta_would_tilt`, verdict) against `live_outcome_ingestor` outcomes; compute false-END/missed-END, ADJUST precision/recall, and `meta_proba` calibration drift. The shadow `meta_would_tilt` log makes this a pure before/after with zero verdict risk.
- ✅ **D7 — Live audit evidence persisted (2026-06-20).** `renderer._to_jsonl_record` now emits `meta_authoritative`, `meta_full_fidelity`, `meta_missing_features`, `meta_source`, `active_hmm_version`, `linked_hmm_version`, `meta_feature_profile` (+ top-level `meta_would_tilt`/`meta_influenced_verdict`). Additive/backward-compatible. Verified live `--once`: complete audit record, JSON-serializable. (`TickSummary` remains a lossy 20-tick ring buffer; the JSONL is the audit-of-record.)

## Done Criteria mapping (original contract)

| # | Criterion | Status | Note |
|---|---|---|---|
| 1 | `MetaLabeler.load(...)` succeeds in the `run_full_pipeline.py` env | ✅ DONE | `True pass` (A1) |
| 2 | Loaded model trained + promoted | ✅ DONE | A2 |
| 3 | HMM/meta lineage compatibility returns no issues | ✅ DONE | A1/A3 |
| 4 | Enrich rows get `meta_prob_authority=="authoritative"` only for a promoted compatible model | ✅ DONE | Authority decision is at enrich (B2, tested). **Caveat:** the `meta_prob` *value* is populated in post-scoring backfill, not at enrich, because `ev_score` (feature #1) is absent at enrich and the row self-gates MISSING (`enrich_grid_params.py:1738-1767,1880`; `run_full_pipeline.py:282,309-318`). |
| 5 | Stage B gating off + diagnostic OR enabled + calibrated + fail-closed | ✅ DONE | Off + diagnostic (B5) |
| 6 | Logs preserve enough metadata to audit every meta-influenced decision | ✅ DONE | Batch (B3) + live JSONL audit (D7, 2026-06-20) |
| 7 | Leakage + circular-feedback tests pass | ✅ DONE | B4 |

## Safety invariants (still binding for D1–D7)

- Do not add label columns to model features; do not use `hlabel`, `hlabel_meta`, `y`, `pnl_pct`, `net_pnl_pct`, `duration_hours`, realized-PnL or post-decision outcome fields as live inputs.
- Do not feed `score` / `scan_score` / any scanner composite that already blends `meta_prob` into the active feature set; do not let `meta_prob` change the candidate set that generates its own future labels (circular feedback).
- Do not bypass missing-feature checks with silent imputation in live admission logic (this is exactly the D1 defect).
- Do not treat a lineage-mismatched or unloadable model as diagnostic success; do not bypass `artifact_compat.py`.
- Decision-time authority is fail-closed: `meta_prob` may become authoritative only when `promotion_status=='pass'`.

## Verification evidence (reproduced 2026-06-18, run `wf_abd5b8c8-514`)

```
MetaLabeler.load(models/meta_labeler.pkl) → is_trained, promotion_status   → True pass
MonitorContext.create() → meta_labeler present, meta_unavailable            → True False
pytest tests/unit/ -k "contract and (leak or hlabel or label)"             → 53 passed, 1149 deselected
pytest test_meta_labeler_retrain_contract_v20260530.py test_stage_b_meta_gate_fastwin.py \
       test_enrich_grid_params.py::test_promoted_meta_labeler_is_authoritative_and_drives_kelly → 38 passed
python scripts/check_deps.py                                               → 6/6 OK, exit 0
pytest tests/                                                              → 1421 passed (1418 + 3 D1 regression tests)
pyright run_full_pipeline.py                                               → 0 errors
```

## Provenance / changelog

- **2026-06-20 (D2/D3/D4/D7 — Phase 1, shipped DISABLED)** — Wired the meta-labeler into live verdicts in shadow-mode. `monitor.py`: computed all 20 features via canonical reuse (`_compute_stochastic_features`, `grid.formulas`, `PnLRanker`, threaded micro `costs`), strict-0-missing fidelity gate, `meta_promoted`/`meta_authoritative`/`meta_full_fidelity`, audit fields + HMM versions + feature profile; **fixed pre-existing funding_rate/open_interest scalar-extraction defect** (`_extract_funding_rate`/`_extract_open_interest`); fixed the phantom `LIVE_BOT_DECISION.md v2` docstring reference. `recommender.py`: `BotEvaluation` authority/fidelity/audit fields, `Recommendation.meta_would_tilt`/`meta_influenced_verdict`, `RecommenderConfig.meta_tilt_enabled=False`/`meta_tilt_low_threshold=0.0`, bounded caution-only tilt in `decide()`. `renderer.py`: D7 JSONL audit fields. Tests: +19 in `test_decision_monitor.py`/`test_decision_recommender.py`. Verified: `pytest tests/` → **1440 passed**; `pyright` 0 errors; `check_deps` 6/6; `leakage-check` PASS (53 contract tests); dated contract test 29 passed; live `--once` shadow run → full-fidelity authoritative `meta_proba` + complete audit trail, verdicts unchanged (tilt disabled). Uncommitted in the working tree.
- **2026-06-18** — Promoted `rolling_180d_20260617_151954` to active via canonical `promote_hmm_version`; added the batch audit-stamp block to `run_full_pipeline.py`. Both changes are **uncommitted** in the working tree (`git status`: `M artifact_manifest.json`, `M run_full_pipeline.py`).
- **2026-06-18 (D1)** — Fixed the live-monitor fail-OPEN feature guard + corrected the `_build_meta_feature_dict` docstring in `src/neutralgrid/live/decision/monitor.py`; added 3 regression tests to `tests/unit/test_decision_monitor.py`. Validated: pyright clean, `pytest tests/` → 1421 passed. Uncommitted in the working tree (`M src/neutralgrid/live/decision/monitor.py`, `M tests/unit/test_decision_monitor.py`).
- Cross-reference: the LIVE-monitor-focused plan at `C:\Users\cris_\.claude\plans\curious-chasing-liskov.md` (verified to exist) is complementary; its premise that the overlay is "loaded/called every tick" is currently false on disk (the D1 guard + D4 gap), and its "HMM lineage authority: unaffected" framing is misleading — `MetaLabeler.load` hard-gates on HMM `artifact_version` regardless of whether features use the HMM.

## Minor notes (verified, non-blocking)

- `metadata.json:47` `model_type="GradientBoostingClassifier"` is a **cosmetic writer-bug** — a hardcoded string at `meta_labeler.py:2000` that never inspects the estimator. The pkl's actual base estimator is `LogisticRegression`; the pkl is authoritative.
- `auc_cv` (`metadata.json:56` ≈ 0.58, training-CPCV mean-of-folds) and `oof_auc` (0.781, study-faithful purged OOF) are two distinct fields, not a contradiction. Promotion gates on `oof_auc` (CI low `0.7686` > 0.50).
- `models/meta_labeler_verification.json` is **stale** (generated `2026-06-01`, 256/254 rows, lineage matched against the OLD 0601 HMM). Do **not** cite it as proof the current model loads.
- `pyproject.toml:7` and `src/neutralgrid/__init__.py:3` still read `6.5.7` inside the v6.5.8 tree. Immaterial to the load gate (which keys on HMM `artifact_version`), but it contradicts the memory note `project_basedir_split_v657_v658`, which is therefore inaccurate.

## Assumptions (operator-validated 2026-06-20)

These were flagged as assumptions during planning and checked on disk / confirmed by the
operator with line citations before any code was written. Recorded here per the operator's
request so each modification point is traceable.

1. **Grid features "match training" → REFINED (was too broad).** Enrich writes
   `grid_spacing_pct`/`profit_per_grid_pct` from `gd.g` (`enrich_grid_params.py:111,1951`),
   but that only matches training rows snapshotted from enriched output. Expired-bot rows
   copy the **stored telemetry** `grid_spacing_pct` (`data_generator.py:843`) and **recompute**
   `profit_per_grid_pct` from workbook geometry/mode (`:868`). ⇒ the two grid features have
   **heterogeneous training provenance**; no single live value matches the whole pool. The
   live builder computes the **geometric** value (matching the expired-bot recompute
   convention via the canonical `grid.formulas`, default legacy-line-count semantics, config
   fees), **records it + its basis in the D7 audit**, and the residual basis vs the
   enriched-scan subset is **quantified by shadow-mode / D6 calibration drift before
   enabling** — never assumed away. The other 18 features (13 `SymbolFeatures` + 3 stochastic
   + `ev_score` + `micro_round_trip_cost_pct`) are unambiguous and fully consistent.
2. **`LiveBotSpec` leverage/fees → leverage YES, fees NO.** `leverage` is required
   (`loader.py:134,141`) and is passed to `PnLRanker` (cast to `int`). Fees are **not** on the
   spec; they resolve from `config.grid` inside `grid.formulas`/`PnLRanker` exactly as
   training does (`PnLRanker` receives only `profit_per_grid_pct` + `leverage`, never
   recomputes fees). The prior `monitor.py:496` "EV call deferred until survival_prob is
   plumbed" note is now obsolete — survival_prob is plumbed and the deferral removed.
3. **Feature-profile name-match → VALID.** `run_full_pipeline.py:1071-1076` reads
   `meta_labeler._feature_names` against `META_FEATURE_PROFILES`; there is no public
   `get_feature_names()` accessor (public introspection is `get_missing_feature_names()`).
   D7 reuses this exact recipe (private-attr read, mirroring the batch path) — verified live
   to resolve `meta_feature_profile == "snapshot_v20260530_fastwin"`.
4. **`f1_threshold` from the pickle → irrelevant to Phase 1.** Training sets
   `MetaLabelerMetrics.f1_threshold` (`meta_labeler.py:1784,1870`); legacy load restores
   `_metrics` but **not** `_f1_threshold` as its own attribute (`:2193`), and the metadata
   JSON persists **no** operating point (`metadata.json:55-80`). Phase 1 needs only
   probabilities + audit columns, so this blocks nothing; D5 (Phase 2) selects
   `meta_tilt_low_threshold` empirically from shadow data, not from the artifact.

### New assumption surfaced during implementation (2026-06-20)

5. **funding_rate / open_interest were non-scalar in the live path (FIXED).** `get_all_market_data`
   returns `funding_rate` as the raw Binance history **list** and `open_interest` as the raw
   **dict** (`binance_client.py:1073-1074`); the model features are scalars. The live monitor
   had passed these raw payloads to `compute_features`, so the meta probe failed with an
   ambiguous-truth-value error (the overlay had never run against real Binance). Fixed by
   `_extract_funding_rate` (mirrors `scan.py:408` `premium_index.lastFundingRate`, fallback to
   latest history entry) and `_extract_open_interest` (mirrors `scan.py:417`
   `float(oi["openInterest"])`). Regression-tested.

## Reversal

Re-promote the prior HMM via `promote_hmm_version("rolling_180d_20260601_134113", ...)`; revert the `run_full_pipeline.py` audit-stamp block. The 2026-06-20 live-decision edits are confined to `src/neutralgrid/live/decision/{monitor,recommender,renderer}.py` + the two decision test files; they are additive and the D4 tilt ships DISABLED (`meta_tilt_enabled=False`), so reverting them (or simply leaving the flag off) restores prior live behavior with no verdict change. No `artifact_compat.py` or feature-pipeline files were touched.

## Delete-this-file criteria

Phase 1 (D1–D4, D7) is DONE and verified. Delete only when **D5 + D6 are DONE and the tilt is ENABLED**: `meta_tilt_low_threshold` calibrated from shadow data (D5), an OOS-validated accuracy improvement demonstrated (D6), `meta_tilt_enabled` flipped on, and live audit evidence confirms meta-influenced verdicts — with `leakage-check`, `verify-feature-pipeline`, full `pytest`, `pyright`, and `check_deps` all green.

## Verification refresh - 2026-06-21 UTC / 2026-06-20 America-Lima

Read/rechecked this file against the current working tree and active artifacts.
Conclusion: **do not enable the meta tilt yet**. Phase 1 remains correctly wired
and validated in disabled shadow mode, but D5/D6 are still not complete because
the available shadow sample is not large or clean enough to calibrate an
operating threshold or prove OOS improvement.

Current on-disk truth:

- `artifact_manifest.json` active HMM == `rolling_180d_20260617_151954`.
- `models/meta_labeler/metadata.json` lineage == `rolling_180d_20260617_151954`.
- `MetaLabeler.load(models/meta_labeler.pkl)` succeeds with
  `is_trained=True`, `promotion_status="pass"`, `is_promoted=True`, and 20
  loaded features.
- `MonitorContext.create()` loads the meta-labeler with
  `meta_unavailable=False`, `meta_promoted=True`,
  `meta_feature_profile="snapshot_v20260530_fastwin"`, and active HMM
  `rolling_180d_20260617_151954`.

Live shadow evidence currently available:

- `logs/live_decisions_20260618.jsonl`: 7 rows, 0 non-null `meta_proba`, 0
  authoritative rows, 0 `meta_would_tilt`; all rows have `meta_predict_failed`
  from the now-documented scalar/container defect.
- `logs/live_decisions_20260621.jsonl`: 2 rows for ENAUSDT, 1 non-null
  full-fidelity authoritative `meta_proba` (`0.44690049255488784`), 0
  `meta_would_tilt`, 0 `meta_influenced_verdict`.
- The first 2026-06-21 row still records `meta_predict_failed` and
  `micro_estimate_failed:TypeError`; the next row succeeds full-fidelity. This
  does not change any verdict because the tilt is disabled and both rows were
  already END, but it is a watch item before D5 calibration.
- Current rows have `candidate_id=null` / `deploy_snapshot=null`, so they are
  not enough by themselves to join a robust D7 shadow series to finalized
  `live_outcome_ingestor` outcomes.

Validation commands run in this refresh:

- `python -m pytest tests/unit/test_decision_monitor.py tests/unit/test_decision_recommender.py -q`
  -> 57 passed.
- `python -m pytest tests/unit/ -k "contract and (leak or hlabel or label)" -q`
  -> 53 passed / 1171 deselected.
- `python scripts/check_deps.py` -> pyarrow/numpy/pandas/scikit-learn/hmmlearn/joblib OK.
- `python -m pyright src\neutralgrid\live\decision\monitor.py src\neutralgrid\live\decision\recommender.py src\neutralgrid\live\decision\renderer.py`
  -> 0 errors.
- `python -m pytest tests/ -q` -> 1440 passed.
- `python -m pyright` -> 0 errors.

Correct next gate: keep `meta_tilt_enabled=False`, continue collecting D7 JSONL
shadow records until there are enough full-fidelity authoritative rows with
joinable outcomes, then perform D5 threshold calibration and D6 OOS proof before
flipping the flag.

## Implementation progress - 2026-06-21 UTC / 2026-06-21 America-Lima

Implemented and verified one concrete D5/D6 blocker fix: live shadow rows can now
be made joinable when a live YAML lacks `candidate_id` but has a single causal
deployment-ready geometry match.

What changed:

- `src/neutralgrid/live/deployment_link_backfill.py` now has an explicit
  `--allow-geometry-match` mode. Default behavior is unchanged: missing
  `candidate_id` still fails. The opt-in resolver writes only when symbol,
  grid lower/upper, `num_grids`, leverage, and scan timestamp <= `deploy_ts`
  identify exactly one deployment-ready row.
- Focused tests cover default refusal, unique causal geometry success,
  future-candidate rejection, and ambiguous-match rejection.
- ENAUSDT was safely backfilled in `data/linkage/deploy_linkage_log.csv`:
  strategy `412730355` -> candidate
  `ENAUSDT_20260618_115611_25245053` from
  `results/deployment_ready_20260618_115611.csv`.
- AVAXUSDT was not backfilled. The nearest geometry row is
  `AVAXUSDT_20260620_134434_9ab18c75`, but the recorded active-bot
  `deploy_ts` is earlier than that candidate scan timestamp, so the new
  causal resolver correctly rejects it.

Live proof after the ENA linkage backfill:

- `MonitorContext.create().lookup_deploy_row()` resolves active ENAUSDT by
  `strategy_id` to `ENAUSDT_20260618_115611_25245053`; active AVAXUSDT still
  resolves to no row.
- `python live_decision_scanner.py --once --bots "active bots\ENAUSDT.yaml" --no-discord --config-file config\live_discord_emit_all.yaml`
  appended a new `logs/live_decisions_20260621.jsonl` row:
  `meta_proba=0.4658274630904182`, `meta_authoritative=true`,
  `meta_full_fidelity=true`, `deploy_snapshot.candidate_id=ENAUSDT_20260618_115611_25245053`,
  `deploy_snapshot.meta_prob=0.4414962624473774`,
  `delta_meta_prob=0.02433120064304084`, `meta_would_tilt=false`,
  `meta_influenced_verdict=false`, diagnostics only
  `utility_calibrator_unavailable`.

Follow-up source-stamp proof:

- `active bots/ENAUSDT.yaml` and
  `Live/2026-06-18/ENAUSDT/ENAUSDT_live_bot_data_scanner.yaml` now both carry
  `candidate_id: ENAUSDT_20260618_115611_25245053`. AVAX remains
  `candidate_id: null` because its mapping is not causally proven.
- Loader validation after the stamp: active ENA and canonical Live ENA both
  parse with zero warnings, direct `candidate_id`, and a resolvable linker row;
  active AVAX parses with zero warnings but still has no candidate/linker row.
- `python -m neutralgrid.live.deployment_link_backfill "active bots\ENAUSDT.yaml" --dry-run`
  returns `wrote_row=False`, `reason=link already exists`, proving the stamped
  candidate ID is consistent with the append-only linkage log.
- A second `python live_decision_scanner.py --once --bots "active bots\ENAUSDT.yaml" --no-discord --config-file config\live_discord_emit_all.yaml`
  appended a new `logs/live_decisions_20260621.jsonl` row with top-level
  `candidate_id=ENAUSDT_20260618_115611_25245053`,
  `meta_proba=0.4653767410072977`, `meta_authoritative=true`,
  `meta_full_fidelity=true`,
  `deploy_snapshot.candidate_id=ENAUSDT_20260618_115611_25245053`,
  `delta_meta_prob=0.023880478559920315`, `meta_would_tilt=false`,
  and `meta_influenced_verdict=false`.

Validation for this progress:

- `python -m pytest tests/unit/test_deployment_link_backfill.py -q` -> 8 passed.
- `python -m pyright src\neutralgrid\live\deployment_link_backfill.py` -> 0 errors.
- ENA geometry dry-runs (canonical Live YAML and active YAML) both resolved the
  same candidate ID before the real backfill was written.
- AVAX geometry dry-run failed closed with `candidate_id not found by geometry`.

Current D5/D6 status after this progress:

- ENA future shadow rows are now joinable through `strategy_id` -> linkage ->
  candidate snapshot, and the latest ENA live audit row already proves
  `deploy_snapshot`/`delta_meta_prob` population.
- Existing pre-backfill JSONL rows are not retroactively rewritten.
- AVAX remains unjoinable until the operator provides the true `candidate_id`
  or a separate causal source proves the deploy timestamp / candidate mapping.
- D5 is still not complete: no calibrated `meta_tilt_low_threshold` has been
  selected from enough shadow rows with finalized outcomes.
- D6 is still not complete: no OOS before/after proof exists yet because the
  currently joinable ENA evidence is one live shadow row and no finalized
  outcome set.
- Keep `meta_tilt_enabled=False`.

## D6 ingestion correctness update - 2026-06-21 UTC / 2026-06-21 America-Lima

The outcome side of the D6 proof path was tightened after inspecting
`LiveOutcomeIngestor`, which is the component that will join finalized expired
bot outcomes to deployment-ready candidate snapshots.

What changed:

- `src/neutralgrid/training/live_outcome_ingestor.py` now uses the latest
  append-only linkage row when a `strategy_id` appears more than once, matching
  the live monitor's `MonitorContext` last-match-wins semantics.
- When a deployment-ready CSV has `candidate_id`, scan features are extracted by
  exact `candidate_id`, not by first same-symbol row. If the linked candidate is
  absent from that scan file, the scan-feature extraction returns empty instead
  of borrowing features from a different same-symbol candidate.
- `tests/unit/test_live_outcome_ingestor.py` now covers both failure modes:
  repeated `strategy_id` rows and same-symbol duplicate candidates in one scan.

Validation:

- `python -m pytest tests/unit/test_live_outcome_ingestor.py -q` -> 10 passed.
- `python -m pyright src\neutralgrid\training\live_outcome_ingestor.py` -> 0 errors.
- Current workbook smoke check:
  `LiveOutcomeIngestor(data/new_expired_bots.xlsx, linkage_dir=data/linkage, scanner_results_dir=results)`
  produced 202 rows with match methods `unmatched=104`, `forensic=98`, no active
  ENA/AVAX strategy hits, and no finalized outcome for
  `ENAUSDT_20260618_115611_25245053`.

Status impact:

- This removes a D6 proof-risk in the outcome joiner; it does not complete D6.
- D6 still needs finalized outcome rows for the joinable shadow decisions before
  false-END, missed-END, ADJUST precision/recall, and calibration drift can be
  computed.
- Keep `meta_tilt_enabled=False`.

## D5/D6 offline analyzer added - 2026-06-21 UTC / 2026-06-21 America-Lima

Added the reproducible, non-gating offline analysis path needed to finish D5/D6
once finalized outcomes exist.

What changed:

- `src/neutralgrid/live/decision/meta_shadow_analysis.py` loads D7 JSONL rows,
  recovers `candidate_id` from top-level JSONL or `deploy_snapshot.candidate_id`,
  joins to finalized outcomes from `LiveOutcomeIngestor`, filters to
  authoritative full-fidelity `meta_proba`, and computes:
  false-END / missed-END proxy counts, ADJUST precision/recall for bad outcomes,
  `meta_proba` Brier/ECE calibration drift, fit-split threshold sweep for
  `meta_tilt_low_threshold`, and OOS precision lift for the low-confidence
  ADJUST subset.
- The analyzer is explicitly non-gating: it writes/report metrics but never
  edits config and never flips `meta_tilt_enabled`.
- `tests/unit/test_meta_shadow_analysis.py` covers `deploy_snapshot.candidate_id`
  fallback joins, insufficient-data refusal, and a synthetic successful
  threshold + OOS-lift path.

Validation:

- `python -m pytest tests/unit/test_meta_shadow_analysis.py -q` -> 3 passed.
- `python -m pyright src\neutralgrid\live\decision\meta_shadow_analysis.py` -> 0 errors.
- Current-data run:
  `python -m neutralgrid.live.decision.meta_shadow_analysis --decisions "logs/live_decisions_*.jsonl" --expired-bots data\new_expired_bots.xlsx --linkage-dir data\linkage --scanner-results-dir results --output outputs\meta_tilt_shadow_analysis_20260621.json`
  produced `decision_rows=703`, `outcome_rows=202`, `joined_rows=0`,
  `eligible_rows=0`, `status=insufficient_joined_rows`,
  `recommended_meta_tilt_low_threshold=null`, `d5_calibrated=false`,
  `d6_oos_proven=false`, `ready_to_enable=false`.

Status impact:

- D5/D6 now have a reproducible analysis path and a fail-closed current-data
  artifact at `outputs/meta_tilt_shadow_analysis_20260621.json`.
- D5 is still not complete because no threshold was calibrated from sufficient
  joined shadow/outcome rows.
- D6 is still not complete because the current live JSONL rows do not yet join
  to finalized outcome rows.
- Keep `meta_tilt_enabled=False`.

## D5/D6 analyzer hardening - 2026-06-21 UTC / 2026-06-21 America-Lima

Hardened the offline analyzer so the current-data artifact explains *why* D5/D6
are still blocked instead of reporting only a zero candidate join.

What changed:

- `src/neutralgrid/live/decision/meta_shadow_analysis.py` now keeps exact
  `candidate_id` as the preferred join key, including the existing
  `deploy_snapshot.candidate_id` fallback, and adds a legacy fallback only for
  decision rows whose candidate ID is blank: exact `strategy_id` + `symbol`
  against `LiveOutcomeIngestor` outcomes.
- Rows that already claim a `candidate_id` do **not** use the strategy fallback,
  so candidate mismatches remain visible instead of being hidden by a weaker
  recovery path.
- The analyzer output now includes `join_coverage`, `metric_availability`,
  `gate_audit`, and `recommended_live_config`, making the fail-closed action
  explicit (`keep_meta_tilt_enabled_false`) until D5 and D6 are both proven.
- `_bool_series()` was cleaned up to avoid pandas downcast FutureWarnings during
  analyzer runs.

Validation:

- `python -m pytest tests/unit/test_meta_shadow_analysis.py -q` -> 4 passed.
- `python -m pyright src\neutralgrid\live\decision\meta_shadow_analysis.py` -> 0 errors.
- Current-data run:
  `python -m neutralgrid.live.decision.meta_shadow_analysis --decisions "logs/live_decisions_*.jsonl" --expired-bots data\new_expired_bots.xlsx --linkage-dir data\linkage --scanner-results-dir results --output outputs\meta_tilt_shadow_analysis_20260621.json`
  produced `decision_rows=703`, `outcome_rows=202`, `joined_rows=358`
  (`strategy_symbol=358`), `decision_rows_with_meta_proba=6`,
  `decision_rows_authoritative_full_fidelity=4`, but `eligible_rows=0`,
  `metric_availability.*=false`, `recommended_meta_tilt_low_threshold=null`,
  `d5_calibrated=false`, `d6_oos_proven=false`, `ready_to_enable=false`,
  `gate_audit.config_action=keep_meta_tilt_enabled_false`.

Status impact:

- The old "0 joined rows" read was too coarse. The analyzer can now prove that
  358 legacy decision ticks do join finalized outcomes by `strategy_id`+`symbol`,
  but none are eligible for D5/D6 because they are not authoritative full-fidelity
  D7 meta rows with usable `meta_proba`.
- The remaining blocker is therefore narrower and clearer: collect finalized
  outcomes for authoritative full-fidelity D7 rows (or ingest new expired bots
  that correspond to those rows), then rerun D5 threshold calibration and D6 OOS
  proof.
- Keep `meta_tilt_enabled=False`.

## D6 group-safe OOS split hardening - 2026-06-21 UTC / 2026-06-21 America-Lima

Hardened the D6 analyzer again so an eventual "OOS lift observed" result cannot
come from repeated ticks of the same live bot leaking across the fit/OOS split.

What changed:

- `src/neutralgrid/live/decision/meta_shadow_analysis.py` now performs the
  temporal fit/OOS split by unique `join_key`, not by individual decision row.
  This keeps all ticks for one candidate / recovered `strategy_id|symbol` live
  bot on exactly one side of the split.
- The analyzer records `split_audit` with `split_strategy=temporal_join_key`,
  fit/OOS join-key counts, and overlap count.
- D5/D6 now require minimum unique join-key support, not only row counts:
  `min_fit_adjust_join_keys=5`, `min_oos_join_keys=5`, and
  `min_oos_tilt_join_keys=3`.
- Threshold metrics now report unique join-key support for the selected
  low-confidence subset, so a future D6 pass cannot be driven by many repeated
  ticks from one bot.

Validation:

- `python -m pytest tests/unit/test_meta_shadow_analysis.py -q` -> 5 passed.
- `python -m pyright src\neutralgrid\live\decision\meta_shadow_analysis.py` -> 0 errors.
- Focused D5/D6/Phase-D cluster:
  `python -m pytest tests/unit/test_meta_shadow_analysis.py tests/unit/test_live_outcome_ingestor.py tests/unit/test_deployment_link_backfill.py tests/unit/test_decision_phase_d.py -q`
  -> 40 passed.
- Focused pyright cluster:
  `python -m pyright src\neutralgrid\live\decision\meta_shadow_analysis.py src\neutralgrid\training\live_outcome_ingestor.py src\neutralgrid\live\deployment_link_backfill.py src\neutralgrid\live\decision\monitor.py`
  -> 0 errors.
- Current-data analyzer rerun still fails closed:
  `decision_rows=703`, `outcome_rows=202`, `joined_rows=358`,
  `eligible_rows=0`, `decision_rows_authoritative_full_fidelity=4`,
  `ready_to_enable=false`, `recommended_live_config.meta_tilt_enabled=false`.

Status impact:

- This does not complete D5/D6 because no authoritative full-fidelity D7 rows
  currently join finalized outcomes.
- It improves the future proof standard: once eligible rows exist, the threshold
  calibration and OOS lift check must be supported by multiple independent live
  bot join keys, not just repeated ticks.
- Keep `meta_tilt_enabled=False`.

## D5/D6 eligibility-funnel diagnostic - 2026-06-21 UTC / 2026-06-21 America-Lima

Added an explicit eligibility funnel to the offline analyzer so the blocker is
diagnosed at the row-filter level, not inferred from `eligible_rows=0`.

What changed:

- `src/neutralgrid/live/decision/meta_shadow_analysis.py` now reports, for joined
  shadow/outcome rows, how many have a join key, `meta_proba`, `pnl_pct`,
  timestamp, `meta_authoritative=true`, `meta_full_fidelity=true`, and all
  eligibility requirements at once.
- The funnel also reports fail counts for missing join key, missing meta
  probability, missing outcome PnL, missing timestamp, non-authoritative meta,
  and non-full-fidelity meta.
- `tests/unit/test_meta_shadow_analysis.py` now covers a joined-but-ineligible
  case so the analyzer cannot silently collapse D5/D6 blockers into one
  undiagnosed `insufficient_joined_rows` result.

Validation:

- `python -m pytest tests/unit/test_meta_shadow_analysis.py -q` -> 6 passed.
- `python -m pyright src\neutralgrid\live\decision\meta_shadow_analysis.py` -> 0 errors.
- Current-data analyzer rerun:
  `joined_rows=358`, `with_join_key=358`, `with_pnl_pct=358`,
  `with_ts_utc=358`, `with_meta_proba=2`, `meta_authoritative_true=0`,
  `meta_full_fidelity_true=0`, `with_authoritative_full_fidelity_meta=0`,
  `all_eligible=0`, `missing_meta_proba=356`, `not_authoritative=358`,
  `not_full_fidelity=358`, `ready_to_enable=false`.

Status impact:

- The current blocker is now precise: historical rows do join outcomes, but they
  were produced before the D7 authoritative/full-fidelity meta audit fields were
  available. They cannot calibrate or prove the live tilt.
- D5/D6 need finalized outcomes for new authoritative full-fidelity D7 rows, not
  merely more legacy outcome joins.
- Keep `meta_tilt_enabled=False`.

## D5/D6 pending-outcome target list - 2026-06-21 UTC / 2026-06-21 America-Lima

Extended the analyzer to list authoritative full-fidelity D7 rows that are
ready on the decision side but still missing finalized outcomes.

What changed:

- `src/neutralgrid/live/decision/meta_shadow_analysis.py` now emits
  `pending_authoritative_outcomes`: total authoritative/full-fidelity D7 rows,
  how many already joined outcomes, pending row count, pending unique join-key
  count, and grouped pending keys with symbol, strategy ID, candidate ID, row
  count, and first timestamp.
- `tests/unit/test_meta_shadow_analysis.py` now covers the pending-outcome
  summary, including candidate-ID and strategy-symbol join keys.

Validation:

- `python -m pytest tests/unit/test_meta_shadow_analysis.py -q` -> 7 passed.
- `python -m pyright src\neutralgrid\live\decision\meta_shadow_analysis.py` -> 0 errors.
- Current-data analyzer rerun:
  `pending_authoritative_outcomes.authoritative_full_fidelity_rows=4`,
  `matched_rows=0`, `pending_rows=4`, `pending_unique_join_keys=3`.
  Pending join keys:
  `412730355|ENAUSDT` (1 row, candidate blank),
  `412770639|AVAXUSDT` (1 row, candidate blank), and
  `ENAUSDT_20260618_115611_25245053` (2 rows, candidate ID present).

Status impact:

- D5/D6 are now blocked by a concrete outcome-ingestion target, not by unknown
  analyzer behavior: finalized outcomes for the pending authoritative D7 join
  keys must appear in `LiveOutcomeIngestor` output before threshold calibration
  and OOS proof can proceed.
- Keep `meta_tilt_enabled=False`.

## D7 JSONL candidate-ID fallback - 2026-06-21 UTC / 2026-06-21 America-Lima

Improved future D7 joinability by making the JSONL audit row use the strongest
candidate identifier already known at render time.

What changed:

- `src/neutralgrid/live/decision/renderer.py` now emits top-level
  `candidate_id` from `result.spec.candidate_id` when present, otherwise from
  `evaluation.deploy_snapshot.candidate_id` when the monitor resolved a deploy
  snapshot by `strategy_id`.
- Explicit YAML `candidate_id` still takes precedence over the deploy snapshot.
- This does not rewrite existing JSONL rows; it only makes future resolved rows
  easier to join by exact candidate ID instead of needing analyzer-side
  `deploy_snapshot` or `strategy_id|symbol` recovery.

Validation:

- `python -m pytest tests/unit/test_meta_shadow_analysis.py tests/unit/test_live_outcome_ingestor.py tests/unit/test_deployment_link_backfill.py tests/unit/test_decision_phase_d.py tests/unit/test_decision_renderer.py -q`
  -> 55 passed.
- `python -m pyright src\neutralgrid\live\decision\renderer.py src\neutralgrid\live\decision\meta_shadow_analysis.py src\neutralgrid\training\live_outcome_ingestor.py src\neutralgrid\live\deployment_link_backfill.py src\neutralgrid\live\decision\monitor.py`
  -> 0 errors.
- Current-data analyzer rerun remains fail-closed (`ready_to_enable=false`) with
  the same four pending authoritative rows; existing rows are intentionally not
  mutated.

Status impact:

- Future ENA-style rows where the YAML lacks `candidate_id` but linkage resolves
  a deploy snapshot will carry a top-level candidate ID in D7 JSONL.
- D5/D6 still require finalized outcomes for authoritative full-fidelity D7 rows.
- Keep `meta_tilt_enabled=False`.

## D5 config enablement regression - 2026-06-21 UTC / 2026-06-21 America-Lima

Verified the final enablement path can consume the future D5 recommendation
through the existing live scanner `--config-file` mechanism.

What changed:

- `tests/unit/test_decision_phase_d.py` now covers loading
  `meta_tilt_enabled: true` and `meta_tilt_low_threshold: 0.45` from YAML via
  `RecommenderConfig.from_file()`.
- No live config was changed and no tilt was enabled.

Validation:

- `python -m pytest tests/unit/test_decision_phase_d.py tests/unit/test_decision_recommender.py tests/unit/test_meta_shadow_analysis.py -q`
  -> 61 passed.
- `python -m pyright src\neutralgrid\live\decision\recommender.py src\neutralgrid\live\decision\meta_shadow_analysis.py`
  -> 0 errors.

Status impact:

- Once D5/D6 eventually produce a proven threshold, the operator-facing config
  field path is regression-tested.
- Current-data analyzer status remains fail-closed: `eligible_rows=0`,
  `d5_calibrated=false`, `d6_oos_proven=false`, `ready_to_enable=false`.
- Keep `meta_tilt_enabled=False`.

<!-- Verified: 2026-06-18 against rolling_180d_20260617_151954 (run wf_abd5b8c8-514) -->
<!-- Verified: 2026-06-20 D1-D4/D7 shipped (shadow, disabled) against rolling_180d_20260617_151954; pytest 1440 passed, pyright clean, live --once full-fidelity authoritative meta_proba -->
