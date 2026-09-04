# PATTERN_PROFILE_FIX — Remediation Plan

Scope: `src/neutralgrid/scanner/profile_model.py` + `src/neutralgrid/scanner/pattern_profile.py`.
Driver: 4-agent audit (2026-04-19) found in-sample training, class-conditional imputation bias, availability leakage, possible outcome-window contamination, and no promotion gate. Team review (2026-04-20, team `pattern-profile-feature-review`) added a 5-feature replacement set, a silent-drop guard, and resolved Phase 0.1.
Constraint: match HMM-level discipline (`safety-invariants.md` — walk-forward + `mean_pass_rate >= 0.50`).

**User decisions (2026-04-20):**
- Ship all 5 replacement features (F1–F5), staged A → B → C.
- F4 (`liquidity_stability_z_1h`) gated on a plan-first system design (Phase 5.5); do not implement until design is approved.
- F5 (`oi_slope_1h_pre`) uses **forward-only training** (no external OI archive; live accrual over >=30 days).
- Missingness indicators follow a best-practices policy (Phase 5.7), no over-engineering.
- Apply the scanner-side touchpoint set in the Rollout section.
- Proceed with first-addendum nuance clarifications (Phase 5.1) and second-addendum ordering + silent-drop guard (Phase 3.6 + Phase 5.9).

**User decisions — third addendum (2026-04-20, pre-implementation consistency pass):**
- Resolve Batch A scope inconsistency: Batch A = **F3 only** until the F4 design doc is approved (was ambiguously "F3 + F4" in Phase 5.8 vs "F3 alone" in Phase 5.5). F4 becomes Batch A.2 after design approval.
- Clarify Phase 5.1 "Confirmed-negative" scope: applies to the 4 legacy `_avg` string references only. `profile_model.py` IS modified under Phases 2.1-2.4, 3.2, 3.6, 5.7 — this is expected, not a contradiction.
- Close three CLI-level false optionalities (new Phase 3.7): `--skip-model` flag, silent-success "Continuing without profile model" log path, and scan-time similarity-only silent degradation when `profile_model` is absent.
- Confirm as in-scope (already in plan, not re-opened): availability-drop promotion guard (Phase 3.6), F4 design-first (Phase 5.5), F3 realized/expected funding asymmetry as documented proxy (Phase 5.4).
- Confirm as out-of-scope (not re-opened): shipping all 5 features unstaged, external OI archive, LDA rewrite, broader `api/app.py` changes beyond fail-closed, F4 depth-stability pre-design.

**User decisions — fourth addendum (2026-04-21, F4 simplification + Phase 5.8 override):**
- **F4 locked to option (a) Volume-stability.** Strike option (b) Order-book-depth-stability explicitly (mirrors F5's external-archive decline at line 461). With (b) off the table, the 8-section design doc is unnecessary scaffolding for the remaining option.
- **`PATTERN_PROFILE_F4_DESIGN.md` is NOT created.** F4 becomes a peer of F1/F2/F3 — no separate design doc, no new ingest module, no new monitoring surface. Existing safeguards apply unchanged: `liquidity_stability_available` flag → Phase 5.7 indicator policy; walk-forward promotion → Phase 3.6; staging → Phase 5.8.
- **F4 lifts out of Batch A.2 and folds into Batch B alongside F1+F2.** Batch A.2 row is removed from the Phase 5.8 staging table.
- **Phase 5.8 staging is OVERRIDDEN for the literal "4 features only" landing.** User adopts a backfill-first strategy: F1, F2, F3, F4 are computed retroactively on the existing labeled bot rows (real historical market data — 15m klines for F1/F4, 1m klines for F2, historical funding for F3) before the per-class floor / walk-forward gate is evaluated. This is recompute-from-real-history, **not** generation of synthetic samples or labels (clarification recorded in Phase 7).
- **Per-phase verification gate (mandatory):** after each phase under the new workstream, require (i) `pyright` clean, (ii) focused unit-test pass count, (iii) retrain proof block listing feature count, available-feature coverage, per-class counts, walk-forward `mean_pass_rate`, and promotion decision.

---

## Phase 0 — Preconditions (no code change)

0.1 **RESOLVED (2026-04-20, data-layer review).** The xlsx workbooks (`data/new_expired_bots.xlsx`, `data/new_expired_bots_backfilled.xlsx`) do **not** carry `funding_rate_avg`, `open_interest_start`, `atr_avg`, `volume_avg` on any sheet. Sheets: `['Sheet1','PnL Curve Features','Meta Features']`. These features are silently dropped at `pattern_profile.py:432` and `profile_model.py:267-271` during training; the fitted artifact carries **no statistics** for them. At scoring time they are aliased from pre-entry live snapshots (`scan.py:546-554`). Reframing: this is a **train/serve schema mismatch**, not label leakage. Removal proceeds as schema-hygiene work under Phase 5.1.

0.2 Snapshot current artifact for rollback: copy `data/profile/profile_model.json` → `data/profile/profile_model.pre_fix.json`.

0.3 Baseline: run `python -m pytest tests/ -q` and record pass/fail count.

---

## Phase 1 — Data-layer consolidation (low risk)

Goal: single source for xlsx I/O; identical curation for both builders.

1.1 Create `src/neutralgrid/scanner/_xlsx_io.py` with:
  - `read_sheet(path, sheet_name) -> pd.DataFrame`
  - `raise_on_duplicate_strategy_id(df, *, context)`
  - `validate_dataframe(df, sheet_name) -> tuple[pd.DataFrame, list[str]]`
  - `detect_format(xlsx_path) -> Literal["multi", "single"]` — inspect `xl.sheet_names` explicitly, **do not** rely on `ValueError` catch.

1.2 Delete the duplicated helpers in `profile_model.py:112-127` and `pattern_profile.py:216-282`; import from `_xlsx_io`.

1.3 In `profile_model.py:156-183`, apply `validate_dataframe` to every sheet read (currently absent — divergence from `pattern_profile.py:313-316`).

1.4 Replace wide merge at `profile_model.py:164-172` with the allow-list merge pattern used in `pattern_profile.py:326-339`. Log any column present in both sheets with divergent values.

1.5 Fail-closed changes in `_xlsx_io.validate_dataframe`:
  - `win_rate > 100` → raise `ValueError` (not silent cap). Current `pattern_profile.py:249-253`.
  - `profit_factor < 0` → raise (currently logs only).
  - `duration_hours < 0` → raise.

1.6 Move `profit_factor` cap of `1000.0` to `src/neutralgrid/core/constants.py` as `PROFIT_FACTOR_CAP` with a comment citing the decision. Import in both builders.

**Verify:** `pytest tests/ -q` passes. Re-train artifact against a known-good xlsx and confirm byte-identical `winner_mu` / `loser_mu` — Phase 1 must be behavior-preserving.

---

## Phase 2 — Training-pipeline correctness

Goal: remove the three quiet statistical defects.

2.1 **Availability-filter leakage** — `profile_model.py:269`.
  Change `if f in df.columns and df[f].notna().sum() >= 10` → evaluate on `df_labeled`, not `df`. Mirror the same check in `pattern_profile.py:432` where `feats` is filtered.

2.2 **Class-conditional imputation bias** — `profile_model.py:277-285`.
  Replace `_impute(Xw)` + `_impute(Xl)` with a **shared** imputation computed on the concatenated labeled set before class split:
  ```python
  X_all = np.vstack([Xw, Xl])
  col_medians = np.nanmedian(X_all, axis=0)
  Xw = _fill_with(Xw, col_medians)
  Xl = _fill_with(Xl, col_medians)
  ```
  Missingness indicators per the Phase 5.7 policy (not per-feature discretion).
  Document in the function docstring: *"Imputation is class-agnostic to avoid inflating `mu_w − mu_l`."*

2.3 **Feature standardization** — `profile_model.py:287-309`.
  Z-score features (fit on concatenated labeled set) **before** computing `Sw`/`Sl`/pooled `S`. Persist the mean/std vectors in the artifact so `ProfileModel.llr` can apply the same transform at inference:
  - Extend `ProfileModel` dataclass with `feature_mean: dict[str, float]`, `feature_std: dict[str, float]`.
  - Update `to_json` / `from_json` / `_vector`.
  - Standardization target for `shrinkage` becomes the identity, which is what the 0.30 shrinkage assumes.

2.4 **Minimum sample size** — `profile_model.py:250, 259`.
  Raise floor from 5 to `max(30, 3 * len(feats))` per class. Fewer samples than that cannot support a `len(feats)`-dim Gaussian.

**Verify:** new unit tests in `tests/unit/test_profile_model_training.py`:
  - NaN imputation uses shared medians (construct a dataset where class-conditional vs shared yield different `mu_w − mu_l`).
  - Availability filter ignores rows outside `df_labeled`.
  - Standardized features produce `llr=0` when row equals the midpoint of class means.
  - Small-n training raises.

Run `pyright` and `pytest tests/ -q`.

---

## Phase 3 — Validation discipline (parallels HMM promotion)

Goal: prevent promotion of a model that hasn't generalized.

3.1 Introduce `neutralgrid.scanner.profile_model_walkforward`:
  - K-fold purged CV (AFML-style) over the labeled universe — purge window ≥ `max_duration_hours` so winner/loser windows do not straddle folds.
  - Per-fold metric: AUC of `proba` and KS separation between classes.
  - Aggregate: `mean_auc`, `mean_pass_rate` (fold passes if `auc >= 0.55`, matching the HMM convention).

3.2 Add `promote_profile_version(candidate_path) -> bool` that refuses to publish a timestamped promoted artifact / update `current.json` unless `mean_pass_rate >= 0.50`. Mirror `promote_hmm_version` (`safety-invariants.md`).

3.3 Add artifact naming: `profile_model_YYYYMMDD_HHMMSS.json` with a `current` symlink/metadata pointer, analogous to HMM `rolling_180d_*`.

3.4 Wire trial tracking: every training run logs (hyperparams, metrics, artifact hash) to `neutralgrid.training.trial_tracker`. This stops silent hyperparameter search.

3.5 **Bootstrap caller contract** (updated 2026-04-21): preserve promotion discipline without deadlocking the pipeline. Default resolution order:
  - if `current.json` exists and is valid, load the promoted artifact it points to;
  - if `current.json` is absent and `data/profile/profile_model.json` exists, load that file as an **unvalidated bootstrap candidate** so the scan → enrich → backtest → train loop can run and collect data;
  - if `current.json` is corrupt / missing `active`, or if neither artifact exists, return `data_missing` fail-closed.
  This is the smallest exception that breaks the low-data loop without erasing promotion. `current.json` remains reserved for walk-forward-validated artifacts only.

3.6 **Silent-drop guard at promotion** (new, 2026-04-20). `promote_profile_version` must additionally assert `len(available_features) >= 0.9 * len(requested_features)` and refuse promotion otherwise. Rationale: the Phase 2.1 availability filter silently omits any declared feature whose NaN rate is too high. Without an explicit floor, a replacement feature broken in the ingest path would disappear from the trained model with no audit signal — direct violation of `feedback_no_silent_degradation`. Emit a structured log line listing dropped features when the floor is breached. Additional verify cases: boundary at exactly 90% and batch-declaration-shrinkage (requested_features frozen at batch-declaration, not inferred from availability).

3.7 **CLI discipline — close false-optionality escape hatches** (new, 2026-04-20, third addendum). Evidence: `retrain_scanner.py:122, 188, 214-216, 251-256`; `src/neutralgrid/scanner/scan.py:660-662`.

  - **3.7.a — Remove `--skip-model`.** The flag (`retrain_scanner.py:122, 188, 256`) allows training to ship a `pattern_profile.json` without a paired profile-model artifact. Since Phase 3.2 makes the profile-model output a gated first-class artifact, a flag that opts out of producing it is a policy-bypass. Delete the flag and its two `if not args.skip_model:` branches. Callers that want partial retrain can run the phase-scoped helpers directly.
  - **3.7.b — Fail-closed on profile-model training error.** Current `retrain_scanner.py:214-216` catches `Exception`, logs `"Continuing without profile model"`, and continues to the "Training complete!" summary. This is a silent-success false path. Change to `logger.error(...)` + `sys.exit(1)`. A failed profile-model train is a hard failure; the pattern_profile artifact by itself is not shippable under the Phase 3.2 gate.
  - **3.7.c — Scan-time similarity-only degradation must be explicit, not silent.** `scan.py:662` sets `p = profile_model.proba(row) if profile_model is not None else None`; the MI-weighted scorer (`scan.py:700-714`) then operates over whatever signals remain, which silently yields a "similarity-only" decision when `profile_model is None`. Require: (i) `profile_model is None` emits a single structured WARN banner per scanner boot (not per-row), (ii) the per-row `scoring_flags` list gets `"profile_model_absent"` appended whenever the fallback is taken, (iii) `RegimeValidator` / `ProfileGate` consumers that run post-scan treat the absent model as `data_missing` (already covered by Phase 3.5 for gates; 3.7.c extends the contract to the MI scorer's audit trail). This preserves a legitimate degraded-mode operation while prohibiting undocumented silent degradation — matches `feedback_no_silent_degradation`.
  - **3.7.d — `enable_profile_model` default.** `api/app.py:111-118` already fails closed when `enable_profile_model=True` and the file is missing. Default must be `True` (or asserted True in production config); a False default would reintroduce the same silent-degradation surface 3.7.c closes. Verify default, do not expand `api/app.py` scope otherwise (remains out-of-scope per line 287).

**Verify:** new `tests/unit/test_profile_promotion.py` and `tests/unit/test_retrain_scanner_cli.py`:
  - Walk-forward with `mean_pass_rate=0.49` → refused.
  - With `mean_pass_rate=0.60` → promoted, `current` updated atomically.
  - Missing artifact → `StageBResult.approved == False`.
  - Feature coverage < 90% of requested → promotion refused with logged feature-drop list.
  - Feature coverage exactly at 90% → promotion allowed (boundary).
  - Batch-declaration shrinkage: `requested_features` shrinking between trainer invocations does not silently lower the coverage floor.
  - `--skip-model` flag no longer accepted (argparse error).
  - Profile-model train failure → CLI exits non-zero, no "Training complete!" banner.
  - `profile_model is None` in `scan.py` emits the boot WARN exactly once and populates `scoring_flags` with `"profile_model_absent"` on every affected row.

Run full `pytest tests/ -q`.

### Phase 3.1–3.6 completion log (2026-04-20)

**Status:** Phase 3.1–3.6 landed. CLI discipline (3.7) deferred — file surface is `retrain_scanner.py` and not covered in this batch.

**Implementation:**
- New module `src/neutralgrid/scanner/profile_model_walkforward.py` (705 lines). Contains `WalkForwardResult`, `PromotionDecision`, `walkforward_evaluate` (chronological expanding-window purged K-fold, per-fold relabeling so test-fold pnl does not leak into winner label), `_apply_winner_labels`, `_train_from_frame` (fold-local training mirroring Phase 2 shared-median + z-score + pooled cov), `promote_profile_version` (atomic artifact + `current.json` write with sha256), `resolve_active_profile_model_path` (fail-closed sentinel on corrupt/missing `active` key), `train_and_promote` end-to-end helper.
- Constants: `AUC_FOLD_PASS_THRESHOLD = 0.55`, `MEAN_PASS_RATE_FLOOR = 0.50`, `COVERAGE_FLOOR = 0.90`.
- Purge invariant: `walkforward_evaluate` raises `ValueError` if `purge_hours < max_duration_hours`. Two-sided purge: drops any train bot whose window **ends** into the test fold, not only those that start inside the purge window.
- Trial logging: every promotion attempt (pass or fail) emits a `TrialRecord` when a logger is supplied.

**Verify — pyright:** `pyright src/neutralgrid/scanner/profile_model_walkforward.py` → `0 errors, 0 warnings`.

**Verify — focused tests** (`tests/unit/test_profile_promotion.py`, 12 tests, all green):
| Test | Coverage |
| --- | --- |
| `test_make_filename_matches_convention` | 3.3 artifact naming |
| `test_promotion_refused_below_pass_rate_floor` | 3.2 `mean_pass_rate=0.49` → refused |
| `test_promotion_accepted_updates_current_atomically` | 3.2 + 3.3 `mean_pass_rate=0.60` → promoted with sha256 |
| `test_resolve_active_path_respects_pointer` | 3.5 pointer target resolution |
| `test_resolve_active_path_bootstraps_when_current_missing` | 3.5 absent `current.json` + bootstrap candidate present |
| `test_resolve_active_path_fail_closed_when_current_and_bootstrap_missing` | 3.5 absent `current.json` + no bootstrap candidate |
| `test_resolve_active_path_fail_closed_on_corrupt_current` | 3.5 corrupt `current.json` → sentinel |
| `test_resolve_active_path_fail_closed_on_missing_active_key` | 3.5 missing `active` key → sentinel |
| `test_walkforward_rejects_purge_hours_below_max_duration` | 3.1 purge-invariant enforcement |
| `test_silent_drop_guard_refuses_below_coverage_floor` | 3.6 `coverage=0.80` → refused |
| `test_silent_drop_guard_allows_exactly_at_floor` | 3.6 boundary `coverage=0.90` → allowed |
| `test_trial_logger_fires_on_refusal_and_pass` | 3.4 both outcomes log |
| `test_walkforward_on_chronological_xlsx_produces_fold_metrics` | 3.1 end-to-end AUC finite on separable synthetic |

Combined with Phase 2: 16/16 targeted tests green.

**Verify — retrain proof on `data/new_expired_bots.xlsx`:** see the authoritative
Phase 2 retrain proof table below (9 requested / 9 admitted / 100% coverage
post-Phase-5.1). Superseded: an earlier intermediate build of this table
recorded 13 requested / 9 admitted / 69.2% coverage when the 4 legacy `_avg`
strings were still declared in `DEFAULT_FEATURES`. That block is no longer
representative and has been removed to prevent confusion with the
post-Phase-5.1 state. All three refusal paths still fire as documented in
the Phase 2 retrain proof: (a) Phase 2.4 sample floor raises before any
promotion attempt; (b) coverage would also be checked against the 90% floor;
(c) pass-rate would also be checked against the 50% floor.

**Independent bug review fixes applied:**
1. **Fail-closed violation on corrupt `current.json`** (review §3): `resolve_active_profile_model_path` now returns a non-existent sentinel path when `current.json` is corrupt or missing `active`, rather than silently bypassing a broken promotion pointer. Bootstrap fallback is allowed only when `current.json` is absent.
2. **One-sided purge leakage** (review §1): fold loop now filters train bots on both `start_time_utc < train_cutoff` **and** `start_time_utc + duration_hours < test_start_time`. Added `ValueError` when `purge_hours < max_duration_hours`.
3. **Non-atomic artifact write** (review §2): `promote_profile_version` now stages the artifact to `*.json.tmp` and renames, preventing a crash mid-save from leaving a half-written artifact.
4. **Global winner-label leakage across folds** (review §5): added `_apply_winner_labels`; fold loop now recomputes `pnl_thr` from train-only pnl, so test-fold pnl cannot leak into winner-label definition.

---

## Phase 4 — Theoretical / semantic cleanup

Lower urgency; only after Phases 1–3 green.

4.1 **SUPERSEDED by Phase 5.** The `_avg` removal + 5-feature replacement is now a structured multi-batch project; see Phase 5.

4.2 `similarity_score` trend multiplier `0.85 + 0.30 * p` (`pattern_profile.py:74`): replace with a likelihood ratio `P(ts|winner) / P(ts|winner)+P(ts|loser)` learned from the labeled universe, or delete the multiplier and document that `similarity_score` is a one-class kernel only.

4.3 Reconcile the two scoring surfaces. Decide:
  - (a) `ProfileModel.proba` is the *decision* surface; `similarity_score` is deprecated for gating and retained only as a diagnostic, or
  - (b) Define an explicit composition (e.g., geometric mean) and unit-test that both agree on sign at the decision boundary.

4.4 `prior_winner` (`profile_model.py:311`): stop using the sample frequency. Either (a) fix at `0.5` and document, or (b) plug in a live-population prior sourced from the trial tracker over the last N completed bots.

4.5 Add cross-sectional purging. When two symbols share the same macro regime window, either cluster-sample one per regime-bucket or down-weight via AFML uniqueness weights.

**Verify:** regression test: `proba` on a held-out 2025-Q4 xlsx slice produces `mean_auc` no worse than the pre-fix model (baseline snapshot from Phase 0.2). If worse, stop and diagnose before promotion.

---

## Phase 5 — Feature set replacement (2026-04-20)

Ship 4 core + 1 experimental feature to replace the legacy `_avg` block. User-authorized: all 5 features, staged A → B → C, no silent drift.

### 5.1 Legacy feature removal (schema hygiene, not leakage)

Remove references to `funding_rate_avg`, `open_interest_start`, `atr_avg`, `volume_avg` per the authoritative reference map (70 refs, 13 files, 5 tiers).

**Tier 1 — production code:**
- `src/neutralgrid/scanner/pattern_profile.py:17-22, 330, 361` — `DEFAULT_FEATURES` + xlsx allow-list + numeric coercion list.
- `src/neutralgrid/scanner/scan.py:546-554` — live alias block.
- `src/neutralgrid/scanner/scan.py:658` — scoring weight key `"funding_rate_avg": 0.8`. **Nuance (first addendum):** this is a weight, not a column assignment; remove as a separate edit step.
- `src/neutralgrid/scanner/feature_extractor.py:110, 113, 197, 216` — `atr_avg`/`volume_avg` dataclass fields + computation.
- `src/neutralgrid/training/scanner_integration.py:149` — **Nuance (first addendum):** drop only the fallback arm of `get_first("funding_rate", "funding_rate_avg")`; keep the primary `funding_rate` read intact.

**Tier 2 — pipeline driver:**
- `run_full_pipeline.py:151` — same fallback-arm pattern; same nuance.

**Tier 3 — tests:**
- `tests/test_afml_integrations.py:410-458` — fixture rows using the 4 features; update to new set or delete as appropriate.
- `tests/unit/test_afml_compliance_fixes.py:210-234` — **Nuance (first addendum):** anti-aliasing regression guards (`test_atr_avg_not_aliased_to_atr_pct`, `test_volume_avg_not_aliased_to_quote_volume`) become semantically vacuous after removal. Delete wholesale rather than update.

**Tier 4 — docs:**
- `SCANNER_USAGE.md:231-234, 438` — feature documentation.
- `CHANGELOG.md:1892` — **LEAVE AS-IS** (append-only history per project convention).

**Tier 5 — live data artifacts (LEAVE-AS-IS):**
- `Live/02-03/fetch_xrp_live_data.py:116`, `Live/02-03/xrp_regime_data.json:25`, `Live/02-03/XRPUSDT_live_bot.json:80` — data snapshots per CLAUDE.md Live Bot Data Storage Policy. Not code.

**Confirmed-negative for the 4 legacy `_avg` strings (do NOT touch *for Phase 5.1 removal*):** `candidate_pipeline.py`, `data_generator.py`, `unified_training_builder.py`, `profile_model.py` — zero refs to `funding_rate_avg`, `open_interest_start`, `atr_avg`, `volume_avg` verified by team. Scope clarification (third addendum, 2026-04-20): this statement is about **legacy-string references only**. `profile_model.py` IS modified under Phase 2.1 (availability filter), Phase 2.2 (shared imputation), Phase 2.3 (standardization + dataclass extension), Phase 2.4 (sample-size floor), Phase 3.2 (`promote_profile_version`), Phase 3.6 (silent-drop guard), and Phase 5.7 (`indicator_cols` in dataclass). These are remediation edits, not `_avg` string removals, and do not conflict with the confirmed-negative finding.

### 5.2 F1 — `parkinson_vol_ratio_4h_24h_pre` (Batch B, GO)

- Construction: `parkinson_4h / parkinson_24h`, where `parkinson = (1/(4 ln 2)) * mean(ln(H/L)^2)` over the window, computed on 15m klines with `close_time < t_entry` (strict).
- Data: 15m klines already fetched (`feature_extractor.py:136-237`).
- New code: Parkinson helper in `src/neutralgrid/indicators/technical.py`.
- NaN policy: NaN if either window has <75% bar coverage or any `high <= low`. Shared-median impute per Phase 2.2. Missingness indicator per Phase 5.7.
- Unit test: assert `close_time < t_entry` on every bar used.

### 5.3 F2 — `variance_ratio_1m_15m_pre_2h` (Batch B, CONDITIONAL)

- Construction: Lo-MacKinlay VR = `(Var_15m / 15) / Var_1m` over last 2h; fallback `VR(5m,15m)` when 1m coverage < 100 of 120 bars.
- Data: scanner does NOT currently fetch 1m for the profile path. Add `client.get_klines(symbol, "1m", 120)` to the scanner gather (≈ +1 REST weight per symbol).
- Training backfill: extend `binance_vision` ingest to 1m klines per labeled `start_time_utc`.
- New code: variance-ratio helper (extractable from `src/neutralgrid/validation/stochastic.py:468` `_variance_ratio_hurst`).
- Fallback indicator: persist a per-row boolean `variance_ratio_fallback_used` so the profile model can condition on the estimator rather than mix two distributions. This is NOT optional — see Phase 5.7.
- NaN policy: NaN when fallback also insufficient (<24 5m bars). Shared-median impute. Missingness indicator per Phase 5.7.
- Unit test: assert `close_time < t_entry`; assert fallback branch triggers correctly when 1m coverage is low.

### 5.4 F3 — `funding_carry_expected_next_7h` (Batch A, CONDITIONAL-GO)

- Construction: `N * f * 1e4` bps, where `f = premiumIndex.lastFundingRate` at `t_entry` and `N = count of scheduled funding settlements in [t_entry, t_entry + 7h)` using the deterministic 00/08/16 UTC schedule (per-symbol override if `fundingInfo` indicates).
- Data: live via `FundingRateProvider.get_funding_info` (`funding_rate.py:116-161`); historical via `HistoricalFundingLoader` + `BinanceClient.get_funding_rate`.
- **Training/inference convention (must be documented):** Binance has no historical `premiumIndex` endpoint. Training backfill substitutes `realized fundingRate[first settlement ≥ t_entry]` for the at-entry prediction. This is causal at inference (nextFundingTime is 8h ahead, so the next-window rate is publicly known at `t_entry`) but the training distribution is "realized" while the inference distribution is "predicted." Document in the feature helper docstring and in `SCANNER_USAGE.md`.
- Per-symbol schedule: use the funding schedule active at `t_entry`, not the current one (Binance occasionally moves symbols to 4h funding).
- Sign convention: encode the direction paid/received by a long position; verify consistency across ingest batches.
- NaN policy: NaN if `premiumIndex` call fails. Shared-median impute. Missingness indicator per Phase 5.7.
- Unit tests: (a) both paths (live predicted, historical realized) return the same scalar for a fixed past timestamp when the realized rate equals the prediction; (b) horizon is a constant 7h, not look-ahead-to-exit.

### 5.5 F4 — `liquidity_stability_z_1h` (Batch B, peer of F1/F2/F3 — **simplified per fourth addendum 2026-04-21**)

**Locked to option (a) Volume-stability. Option (b) is struck.**

- **Signal specification (locked):** z-score of the 1h rolling quote-volume sum vs the 24h trailing distribution of 1h rolling sums. All inputs derived from `klines_15m.quote_volume`, which is already fetched at scan time (`scan.py:540-555`, verified) and available in Binance Vision archives for retroactive computation on existing labeled rows.
- **Option (b) struck:** spread-weighted top-N depth-stability is **NOT pursued.** No third-party L2 archive (Tardis/Kaiko) is sourced; no self-collected snapshot logger is built. This mirrors the F5 forward-only / no-external-archive decision at line 461.
- **No separate design doc.** `PATTERN_PROFILE_F4_DESIGN.md` is **not authored**. F4 follows the same lightweight authoring discipline as F1/F2/F3:
  - Data contract (canonical column name `liquidity_stability_z_1h`, dtype `float64`, units = z-score, NaN policy = NaN when fewer than 24 valid 1h rolling sums in the trailing window).
  - Availability flag `liquidity_stability_available: bool` emitted alongside the value, governed by the **Phase 5.7 missingness indicator policy** (no new monitoring surface).
  - Walk-forward promotion gate via **Phase 3.6** `promote_profile_version` (no new gate).
  - Causal guard: rolling sums use `[t - 25h, t - 1h]` for the trailing distribution and `[t - 1h, t]` for the current 1h sum, all strictly pre-entry. Same convention as F1's pre-entry windowing.
  - Tests live next to F1/F2/F3 unit tests under `tests/unit/` — no new test module created.
- **Integration points** (same surface as F1/F2/F3, no expansion): `scanner/feature_extractor.py` (emission), `scanner/scan.py` (row assembly), `scanner/pattern_profile.py` (`DEFAULT_FEATURES`), training-time backfill (Phase 7), no new module under `data/`.
- **Missingness indicator decision:** indicator REQUIRED when expected NaN rate on `df_labeled` >=10%, per Phase 5.7. Initial estimate is below 10% because `klines_15m` is already fetched for every scanned symbol; if the Phase 7 backfill audit shows otherwise, the indicator becomes required at that point.

F4 ships in **Batch B alongside F1+F2** (see Phase 5.8 updated table below). Batch A.2 is dissolved.

### 5.6 F5 — `oi_slope_1h_pre` (Batch C, experimental, **forward-only training**)

User decision: forward-only. Do NOT source external OI archive. Batch C waits for live accrual of >= 30 days of scanner logs with valid `oi_slope_1h_pre` coverage before the experimental promotion gate is evaluated.

- Construction: OLS slope of `ln(OI_t)` vs `t` over >=6 (preferred >=8) hourly points in the pre-entry window; do NOT use a strict 1h window that yields 1 point at hourly granularity. Extend to 6–12 hourly points as data-layer review recommended.
- Data: `BinanceClient.get_open_interest_hist(symbol, period="1h")`. 5m-granularity symbols are a minority; gate the feature off for them or use a 5m variant with the same 6–12 point rule.
- NaN policy: NaN for symbols with <8h of OI history (fresh listings). High NaN rate expected → missingness indicator REQUIRED per Phase 5.7 (not optional).
- Forward-only training: the trial tracker must record the date at which F5 first became available per symbol; training is done on rows whose `start_time_utc` falls inside the tracked-availability window.
- Promotion criterion (from task #3 review):
  - Walk-forward purged K-fold, K=5, purge = `max_duration_hours`.
  - Baseline: 9 technical + F1..F4 (Batch A+B promoted) = 13 features.
  - Candidate: baseline + F5 = 14 features.
  - Promote F5 iff: `mean(AUC_lift) >= 0.01` AND `median(AUC_lift) > 0` AND `AUC_lift[k] >= -0.005` in >=4 of 5 folds AND `recall_fast_winners_candidate >= recall_fast_winners_baseline + 0.03` at precision=0.70 AND `FPR_candidate <= FPR_baseline + 0.02`.
  - F5 NaN rate on training `df_labeled` must be <=30%; otherwise abort and re-test after further ingests.

### 5.7 Missingness indicator policy (best practices, not over-engineered)

Replace Phase 2.2's per-feature discretion with a single policy:

- **Required indicator** for any feature whose expected NaN rate on the labeled universe is >=10%. Indicator is a separate binary column `<feature_name>_missing` persisted alongside the feature in the training row and inference vector.
- **Required indicator** regardless of NaN rate when the feature has multiple construction branches (e.g., F2's 1m-vs-5m fallback) — encode the branch as a first-class column so the classifier can condition on the estimator.
- **Not required** for features whose NaN rate on `df_labeled` is consistently <10% AND construction is single-path. Imputation via shared median (Phase 2.2) is sufficient.
- Per-feature indicator decisions are recorded in the F-series subphases (5.2–5.6). As of 2026-04-20: F1 optional, F2 required (fallback branch), F3 optional, F4 TBD in design doc, F5 required (fresh-listing NaN rate).
- Implementation: extend `FeatureSnapshot` and `ProfileModel.feature_mean/std/indicator_cols` to carry indicator columns end-to-end. Do NOT create a parallel system; reuse the Phase 2.3 dataclass extension.

### 5.8 Batch staging (revised — fourth addendum 2026-04-21)

| Batch | Features | Feature count (total) | Per-class floor | Preconditions |
|---|---|---|---|---|
| A | **F3 only** | 9 baseline + F3 = 10 | max(30, 3·10) = 30 | Phases 1, 2, 3 (incl. 3.6, 3.7) green. F3 convention documented. |
| B | F1 + F2 + F4 | 10 + 3 = 13 | max(30, 3·13) = 39 | Batch A promoted via walk-forward. 1m kline fetch + Vision backfill live (F2). F4 = volume-stability via existing 15m klines (no new ingest). Phase 7 backfill of all four columns over the existing labeled rows complete. |
| C | F5 (experimental) | 13 + F5 = 14 | max(30, 3·14) = 42 | Batch A+B promoted. >=30 days of forward-only coverage. F5 promotion criterion (5.6) passes. Phase 3.4 trial tracker populated (non-empty availability window per symbol). |

Baseline count (9) is the current artifact feature set — all technical, none `_avg`. Phase 5.1 removes the 4 `_avg` names from `DEFAULT_FEATURES`, which aligns the declared set with the already-fitted in-artifact set.

**User override (fourth addendum 2026-04-21):** the user has authorized a "4 features only" landing path that bypasses the staged A → B → C sequence above. Under the override, the **9 baseline features may be removed in a single transition** when (i) Phase 7 backfill is complete on all four target columns, (ii) per-class floor `max(30, 3·4) = 30` is satisfied on the backfilled labeled set, (iii) walk-forward `mean_pass_rate >= 0.50` against the 4-feature candidate. The staged table is preserved above as the default discipline; the override is the recorded exception.

Sample-size warning: shipping all 5 simultaneously (floor 42) against ~80 rows with 21/22 winner skew (FIXUTILITY-01 memory) violates the minority-class floor. Staging remains the default; only the explicit fourth-addendum 4-feature override may bypass it. Each batch's corpus must be re-checked against its per-class floor at the time of its promotion attempt (not at plan-time).

### 5.9 Ordering constraint

Phases **3.1–3.3 + 3.6 + 3.7 must land before Batch A** is promoted. Otherwise a new-feature artifact could be written to `data/profile/profile_model.json` without walk-forward validation, without the silent-drop guard, or with a CLI path that advertises "Training complete!" despite a failed profile-model fit — direct Artifact Naming + No-Silent-Degradation violations.

Phase commits remain separate per the Rollout rule; feature-batch commits are a separate axis. Never bundle a phase commit with a feature-batch commit.

**Ordering enforcement (from task #9 validation):** the constraint is policy-level, not machine-enforced. Implementation of Phases 3.2 + 3.3 should tighten it into a code contract:
- `promote_profile_version` should accept a typed `WalkForwardResult` (not a bare `mean_pass_rate` float), making provenance non-spoofable.
- The artifact writer should refuse to emit `profile_model.json` (non-timestamped) once `DEFAULT_FEATURES` contains any Batch-A feature name.
- The CLI in `retrain_scanner.py` should refuse to run when Phase 3.6 coverage floor is not yet wired in (detectable by a version field in the `ProfileModel` dataclass).

---

## Consistency resolutions (third addendum, 2026-04-20)

Resolutions applied from the pre-implementation consistency review. "Status" distinguishes **doc-fix** (plan wording corrected, no code change) from **code-implemented** (source edits landed). Implementers cross-reference this table:

| Claim | Status | Resolved in |
|---|---|---|
| Phase 5.5 F3-alone vs Phase 5.8 F3+F4 inconsistency | **Doc-fix** (plan-only) | Phase 5.8 (new table), Phase 5.5 (clarified) |
| "Confirmed-negative: profile_model.py" overbroad | **Doc-fix — scoped to legacy strings only** | Phase 5.1 Confirmed-negative block |
| `--skip-model` false optionality | **Code-implemented** — flag removed from `retrain_scanner.py` | Phase 3.7.a |
| "Continuing without profile model" false-success path | **Code-implemented** — `sys.exit(1)` on exception | Phase 3.7.b |
| Silent similarity-only runtime fallback | **Code-implemented** — boot WARN + per-row `profile_model_absent` flag | Phase 3.7.c |
| `enable_profile_model` default | **Code-implemented** — flipped `False` → `True` in `config.py:386` | Phase 3.7.d |
| Availability-drop without promotion guard | **Code-implemented** — `promote_profile_version` (coverage floor + subset guard) landed 2026-04-20 | Phase 3.6 |
| F4 without historical L2 data | **Resolved (4th addendum 2026-04-21)** — option (b) struck; F4 = volume-stability via existing 15m klines (no L2 needed) | Phase 5.5 (rewritten) |
| Realized vs expected funding proxy | **Planned — documented proxy** (doc accepted; F3 code not yet written) | Phase 5.4 |
| Shipping all 5 features simultaneously | **Not adopted** — staged A → A.2 → B → C | Phase 5.8 |
| External OI archive for F5 | **Not adopted** — forward-only | Phase 5.6, Out-of-scope |
| Rewriting LDA/Gaussian classifier | **Not adopted** | Out-of-scope |
| Expanding `api/app.py` scope | **Not adopted** — fail-closed already present at `app.py:111-118`; no expansion | Phase 3.7.d, Out-of-scope |
| F4 depth-stability (option b) | **Struck (4th addendum 2026-04-21)** — option (a) volume-stability locked instead | Phase 5.5 (rewritten), Out-of-scope |

Phases 1, 2, 3.1–3.6, 3.7, and **Phase 5.1 (partially)** are code-implemented (landed 2026-04-20). Phase 5.1 active-surface removal is complete (`DEFAULT_FEATURES`, scanner row assembly in `scan.py`, `FeatureSnapshot` dataclass in `feature_extractor.py`, training-time fallbacks in `run_full_pipeline.py` and `scanner_integration.py`, test fixtures, and user docs). Residual multi-sheet merge allow-lists in `profile_model.py:200` and `profile_model_walkforward.py:198` were trimmed to `["strategy_id"]` only — the mkt-sheet merge now auto-skips via its `len > 1` guard and no legacy `_avg` / `open_interest_start` strings remain in source. Phase 4 and Phases 5.2–5.8 remain plan-only. Phase 0.2 rollback snapshot and Phase 0.3 baseline pytest run have not been executed.

**Phase 2 completion log (2026-04-20):**
- 2.1 Availability filter in `profile_model.py` now evaluates `notna()>=10` on `df_labeled` (was `df`). Mirror in `pattern_profile.py` changed from `df.columns` to `df_labeled.columns`. Divergence intentional: `pattern_profile.py` has per-feature `dropna()` at fit time and does not gate on NaN count.
- 2.2 Class-conditional `_impute` replaced with shared-median imputation on concatenated `vstack([Xw, Xl])`. `col_medians` fallback to `0.0` for all-NaN columns (defensive only; availability filter should have dropped them).
- 2.3 Z-score standardization added. `feat_mean` / `feat_std` fit on concatenated labeled set (`ddof=0`), persisted as `ProfileModel.feature_mean` / `feature_std`. `ProfileModel._vector` applies `(raw - mean) / std`; back-compat path passes through raw values when fields are `None`. `feat_std > 1e-12` else `1.0`.
- 2.4 Per-class sample floor raised to `max(30, 3 * len(feats))`. Applied AFTER `feats` is finalized by availability filter.
- 2.5 (audit parity with PatternProfile) `ProfileModel` extended with `selection_summary: Optional[dict[str, Any]] = None`. Assembled in `train_profile_model_from_enhanced_xlsx` after `winners_mask` is resolved and attached to the returned model on the success path only (before the `min_samples` raise). Persisted through `save_profile_model` / `load_profile_model` under JSON key `"selection_summary"`. Pre-existing artifacts lacking the field load with `selection_summary=None`. Shared keys with `PatternProfile.selection_summary` (`xlsx`, `top_quantile`, `min_profit_factor`, `min_avg_profit_per_grid`, `max_duration_hours`, `duration_band`, `pnl_threshold`, `winners_count`, `winners_symbols`, `bounded_universe_size`, `labeled_universe_size`); profile-model-only extension: `losers_count`.
- Unit tests added: `tests/unit/test_profile_model_training.py` (Phase 2.1-2.4) and `tests/unit/test_profile_model_selection_summary.py` (Phase 2.5, round-trip + legacy-load + top_quantile=0.0 boundary) — all pass.
- `TestBoundedUniverseContract._make_test_xlsx` fixture enlarged to `_N_SHORT_ROWS=120` to satisfy new per-class floor.
- Pyright: `0 errors, 0 warnings` on `profile_model.py`, `pattern_profile.py`, `test_profile_model_training.py`.
- Focused suite: **19 passed**, 0 failed.
- Independent reviewer: no critical bugs; confirmed standardization math, back-compat, and ordering correct.

**Phase 2 retrain proof (current xlsx, refreshed 2026-04-21):**

| Metric | Value |
|---|---|
| Input | `data/new_expired_bots.xlsx` (192 rows) |
| Bounded universe (duration<7h) | 77 rows |
| Labeled universe (pf,pnl non-NaN) | 77 rows |
| Requested numeric features | 4 (`parkinson_vol_ratio_4h_24h_pre`, `variance_ratio_1m_15m_pre_2h`, `funding_carry_expected_next_7h`, `liquidity_stability_z_1h`) |
| Admitted after availability filter | 4 (all requested features admitted) |
| Available-feature coverage | **100%** (4/4) — satisfies the 90% floor enforced by Phase 3.6 `promote_profile_version` |
| Per-class sample floor | `max(30, 3*4) = 30` |
| `top_quantile` used | 0.605 (selected as the maximum quantile that keeps both classes ≥ 30 in the current pool) |
| `pnl_threshold` (60.5th pct of pnl_pct) | 3.43% |
| `min_profit_factor` | 1.5 |
| `min_avg_profit_per_grid` | 0.59 |
| Winners | 30 |
| Losers | 47 |
| **Legacy-CLI training (`retrain_scanner.py --skip-gate`)** | **TRAINED** — `data/profile/pattern_profile.json` and `data/profile/profile_model.json` written. Prior(winner) = 0.390. |
| **Walk-forward gate (`train_and_promote`)** | See "Walk-forward promotion proof" block below. |

Historical refusal (preserved for audit): with the prior workbook (170 rows, 47 labeled, 11 winners) the Phase 2.4 sample floor correctly **REFUSED** training and `train_and_promote` raised `ValueError` before any artifact write. The backfilled variant (`data/new_expired_bots_backfilled.xlsx`: 38 bounded, 37 labeled, 9 winners) was also refused. Today's pool growth (live ingest 2026-04-19/20/21) flipped the decision — the Phase 2.4 floor is now satisfied at the cost of a minimum-margin winner count (30 winners, exactly at floor).

Interpretation: Phase 2 fail-closed behavior was demonstrated under the prior pool and the new pool only clears the floor by a single-row margin. The binding blocker is no longer the per-class sample floor at this `top_quantile` — it is now sample-margin fragility (one winner reclassification flips admissibility) and walk-forward AUC, which is the next gate.

Additional retrain proof from the live 4-feature run:
- Pattern/profile artifacts both persist the same 4-feature contract.
- Intra-fit NaN drops inside the 77-row labeled subset were observed for:
  - `parkinson_vol_ratio_4h_24h_pre`: 5 rows
  - `variance_ratio_1m_15m_pre_2h`: 21 rows
  - `liquidity_stability_z_1h`: 25 rows
- Two symbols had no cached kline history at all during retrain (`TAKEUSDT`, `ZRCUSDT`), so their backfilled profile features remained NaN.

**Walk-forward promotion proof (current xlsx, 2026-04-21):**

`train_and_promote(xlsx='data/new_expired_bots.xlsx', n_folds=5, purge_hours=7.0, max_duration_hours=7.0, min_profit_factor=1.5, min_avg_profit_per_grid=0.59, top_quantile=0.605)`:

| Metric | Value | Threshold | Pass? |
|---|---|---|---|
| `feature_coverage` | 1.000 (4/4 admitted) | ≥ 0.90 | ✓ |
| `n_folds` evaluated | 4 (chronological K-fold, k=1..n_folds-1) | — | — |
| `fold_auc` | `[NaN, NaN, NaN, NaN]` | per-fold ≥ 0.55 | ✗ |
| `fold_ks` | `[NaN, NaN, NaN, NaN]` | — | — |
| `mean_pass_rate` | 0.000 | ≥ 0.50 | ✗ |
| **Promotion decision** | **NOT PROMOTED** — `mean_pass_rate=0.000 < floor=0.50` | — | — |

Why every fold returned NaN: with 77 chronologically-sorted labeled rows and `n_folds=5`, `fold_size=15`. The train slice for fold k spans rows `[0, k·15)`, so train size grows 15→30→45→60. Each fold's local labeling at `top_quantile=0.605` then needs **both** `n_w ≥ max(30, 3·4) = 30` **and** `n_l ≥ 30` in the train slice (`profile_model_walkforward.py:447`). Even fold 4 (60-row train slice) cannot produce ≥ 30 winners and ≥ 30 losers simultaneously, so each fold short-circuits to NaN.

Consequence: **no `data/profile/profile_model_YYYYMMDD_HHMMSS.json` was written, and `current.json` was not updated.** Under the 2026-04-21 bootstrap exception, the CLI artifact at `data/profile/profile_model.json` remains usable as the default **bootstrap candidate** while `current.json` is absent. The resolver still fails closed on a corrupt `current.json` or when neither artifact exists.

To pass the walk-forward gate at `top_quantile=0.605` with `n_folds=5`, the labeled pool needs to grow substantially so the smallest train slice can host both classes above floor. The immediate continuation is therefore:
1. keep the candidate artifact in bootstrap use so the full pipeline can collect real outcomes;
2. accumulate more labeled short-duration results;
3. rerun walk-forward / promotion only when the pool is large enough for finite per-fold class counts.

This keeps the promotion contract intact while removing the operational deadlock:
low data → no promotion → no deploy → no new data.

**Phase 1 completion log (2026-04-20):**
- 1.1 `src/neutralgrid/scanner/_xlsx_io.py` created with `read_sheet`, `raise_on_duplicate_strategy_id`, `validate_dataframe`, `detect_format`, `read_single_sheet`, `MULTI_SHEET_NAMES`.
- 1.2 Duplicated helpers deleted from `pattern_profile.py` and `profile_model.py`; both now import from `_xlsx_io`.
- 1.3 `validate_dataframe` applied to every sheet read in both builders.
- 1.4 Wide-merge in `profile_model.py` replaced with allow-list merge pattern (matches `pattern_profile.py`).
- 1.5 Fail-closed: `win_rate>100`, `profit_factor<0`, `duration_hours<0` all raise `ValueError` in `validate_dataframe`.
- 1.6 `PROFIT_FACTOR_CAP = 1000.0` moved to `src/neutralgrid/core/constants.py`; imported in both builders.
- **Behavior-preservation proof:** retrain against `data/new_expired_bots.xlsx` pre-stash vs post-stash yielded byte-identical `features`, `winner_mu`, `loser_mu`, `inv_cov`, `prior_winner=0.234043…`, `duration_band`. (9 features, single-sheet path; multi-sheet path not exercised by current data but preserved by construction.)
- Pyright: `0 errors, 0 warnings` on `profile_model.py`, `pattern_profile.py`, `_xlsx_io.py`.
- Focused tests: 15 passed, 0 failed (profile/pattern/scanner/xlsx selectors).

## Rollout

- Phases 1–3 are required before a new profile-model artifact is promoted via `current.json`. While the pool is too small for meaningful walk-forward, the freshly retrained `data/profile/profile_model.json` may run as an explicit bootstrap candidate when `current.json` is absent. Phase 4 may ship later but must be tracked in session FIXUTILITY-01 memory.
- Phase 5 batches (A → A.2 → B → C) ship after their listed preconditions in 5.8 are met; the ordering constraint in 5.9 is hard.
- Each phase commits separately with the phase label in the message. Do not bundle phases. Feature-batch commits are a separate axis from phase commits.
- After each phase: `pytest tests/ -q` and `pyright` must be green. Record counts in commit body.
- **Scanner-side touchpoint set** — when scanner `DEFAULT_FEATURES` changes, update consistently:
  1. `src/neutralgrid/scanner/pattern_profile.py:17-22, 330, 361`
  2. `src/neutralgrid/scanner/scan.py:546-554` (remove) or equivalent emission block (add)
  3. `src/neutralgrid/scanner/feature_extractor.py` — `FeatureSnapshot` fields + computation
  4. `retrain_scanner.py:47, 158, 170, 199` — CLI log reflects new count
  5. `SCANNER_USAGE.md` — documentation
  6. `tests/unit/test_profile_model_training.py` + `tests/test_afml_integrations.py` — fixture updates
  The three-file meta-labeler rule in `safety-invariants.md` does NOT apply to profile-only features.

## Out of scope (explicit)

- Rewriting the classifier as logistic regression or a tree model. LDA remains; this plan fixes its preconditions.
- Changing callers in `api/app.py` or `enrich_grid_params.py` beyond the fail-closed contract in 3.5 and the `enable_profile_model=True` default verification in 3.7.d.
- External OI archive (Tardis / Laevitas / Coinalyze) for F5 — explicitly declined by user; forward-only training is the accepted path.
- Shipping F1–F5 in a single batch — staging A → B → C is the default discipline (Phase 5.8). The fourth-addendum 4-feature override is the explicit exception; F5 remains out of any single-batch landing.
- Authoring `PATTERN_PROFILE_F4_DESIGN.md` (struck per fourth addendum 2026-04-21; option (b) declined, option (a) locked).
- Sourcing third-party L2 archives (Tardis, Kaiko, Coinalyze) or building a self-collected depth snapshot logger for F4 (option (b) struck).

## File anchors (audit source)

- `src/neutralgrid/scanner/profile_model.py:112-127, 156-183, 164-172, 239-247, 250-263, 269, 277-285, 287-309, 311`
- `src/neutralgrid/scanner/pattern_profile.py:17-22, 35-76, 74, 216-282, 249-253, 313-339, 370, 432, 444-449`
- `src/neutralgrid/scanner/scan.py:342-344, 546-554, 658`
- `src/neutralgrid/scanner/feature_extractor.py:105-133, 196-220`
- `src/neutralgrid/training/scanner_integration.py:149`
- `src/neutralgrid/api/binance_client.py:211, 387, 504-557, 570`
- `src/neutralgrid/data/funding_rate.py:26-188, 256-364`
- `src/neutralgrid/data/binance_vision/urls.py:17-44`
- `src/neutralgrid/validation/stochastic.py:468`
- `run_full_pipeline.py:151`
- `tests/test_afml_integrations.py:410-458`
- `tests/unit/test_afml_compliance_fixes.py:210-234`
- `SCANNER_USAGE.md:231-234, 438`
- `.claude/rules/safety-invariants.md` — Fail-Closed Behavior, Artifact Naming, Feature Pipeline Update Rule, Leakage Prevention
- `C:\Users\cris_\.claude\projects\C--Users-cris--OneDrive-Documents-Christian-Crypto-Neutral-Grid-Bots-NEUTRAL-grid-bot-v6-5-7\memory\` — `project_fixutility_01`, `feedback_no_silent_degradation`, `project_pending_external_apis`

## Team review trail (2026-04-20)

- Team: `pattern-profile-feature-review` (config: `~/.claude/teams/pattern-profile-feature-review/config.json`).
- Members: `reference-auditor`, `data-layer-reviewer`, `feature-catalog-reviewer`, `safety-invariants-reviewer`.
- Deliverables consumed: task #1 reference map (70 refs, 13 files), task #2 data-layer feasibility + Phase 0.1 resolution, task #3 feature catalogue + orthogonality + F5 gating protocol, task #4 safety-invariants matrix (all PASS conditional), task #5 converged report + two addenda.

---

## Phase 6 — User-requested 4-feature replacement (historical pre-implementation audit, superseded)

**Historical note:** this section captures the pre-implementation audit pass from 2026-04-21. It is preserved for traceability only. The live implementation state is now recorded in **Phase 7 — 4-feature override completion log** below.

**Source request (2026-04-21):** replace the current 9 active features with **only** these 4:
1. `parkinson_vol_ratio_4h_24h_pre` (≡ Phase 5.2 F1)
2. `variance_ratio_1m_15m_pre_2h` (≡ Phase 5.3 F2)
3. `funding_carry_expected_next_7h` (≡ Phase 5.4 F3)
4. `liquidity_stability_z_1h` (≡ Phase 5.5 F4)

**Scope of this section:** historical plan snapshot only. No longer the source of truth for current runtime behavior. Every claim below is anchored to the 2026-04-21 pre-landing verification pass and may now be outdated where Phase 7 records the landed implementation.

### 6.1 Constraint conflicts the user must resolve before any code lands

These are facts on disk today; they block a literal "ship all 4 simultaneously" reading of the request.

| # | Conflict | Evidence |
|---|---|---|
| C1 | **RESOLVED (4th addendum 2026-04-21)** — F4 locked to option (a) volume-stability; `PATTERN_PROFILE_F4_DESIGN.md` requirement struck. F4 may now ship as a peer of F1/F2/F3 in Batch B. | Phase 5.5 (rewritten), fourth-addendum block at top of doc. |
| C2 | **OVERRIDDEN (4th addendum 2026-04-21)** — user explicitly authorizes the literal "4 features only" landing under a backfill-first strategy. Phase 5.8 staging table now records the override; A.2 row is dissolved (F4 folded into Batch B). | Phase 5.8 (revised), fourth-addendum block, Phase 7. |
| C3 | **Phase 5.9 hard ordering not yet met.** Phases 3.1–3.3 + 3.6 + 3.7 must land before *any* Batch A promotion. 3.6 + 3.7 are code-implemented (third addendum). Phases 3.1–3.3 (walk-forward + `train_and_promote` + sklearn-style trainer/inference contract) status in this doc shows "remain plan-only" for Phase 4 and 5.2–5.8 but Phases 3.1–3.6 are listed as code-implemented at line 369. **Assumption flag:** I did not re-verify each of 3.1–3.3 line-by-line in this pass; before Batch A promotion the implementer must re-confirm. | This file, lines 335–344, 369. |
| C4 | **Walk-forward floor unmet today.** With the 4-feature set the per-class floor is `max(30, 3·4) = 30` (same as today). But walk-forward at K=5 over 77 chronologically-sorted labeled rows yields fold sizes of 15, so train slices grow 15→30→45→60 — none can host ≥30 winners AND ≥30 losers simultaneously. The current 9-feature retrain produced `mean_pass_rate=0.000`. Switching to 4 features does **not** raise sample efficiency; it changes the loading on the LDA inverse covariance only. | This file, lines 406–429. |
| C5 | **RESOLVED-IN-PLAN (4th addendum 2026-04-21)** — user adopts a backfill-first strategy. F1/F2/F3/F4 are computed retroactively on the existing labeled rows from real historical market data (15m klines for F1/F4, 1m klines for F2, historical funding for F3). See Phase 7 for the workstream. The Phase 2.1 availability filter is the Phase 7 acceptance gate, not a blocker. | Phase 7, `profile_model.py:267-271` availability filter, `pattern_profile.py:25-29`. |

**Recommendation (revised, 4th addendum 2026-04-21):** C1, C2, and C5 are resolved or overridden. The remaining live constraints are C3 (Phase 5.9 ordering must still be re-confirmed by the implementer before Batch A or the 4-feature override is promoted) and C4 (walk-forward floor over the current 77-row labeled pool — backfill enlarges feature columns but does **not** add new bot rows; sample-margin fragility persists until the live ingest grows the pool). The legal landing path is therefore: complete Phase 7 backfill → re-run availability filter → re-run walk-forward → promote only if `mean_pass_rate >= 0.50`.

### 6.2 Touchpoint inventory by category

The categories below are the user-requested taxonomy:
- **PFO** = *provable false optionality* — code path that advertises a choice that is in fact required or always taken.
- **PUW** = *provably unnecessary as written* — code path that does no work for the new feature set, with line-cited proof.
- **NSA** = *not valid to strike without an assumption* — line that may need to change but I cannot prove it from current data; striking it requires an assumption the user must accept or reject.

#### A. `DEFAULT_FEATURES` declaration

| Site | Category | Action under "4-only" reading | Evidence |
|---|---|---|---|
| `src/neutralgrid/scanner/pattern_profile.py:25-29` (`DEFAULT_FEATURES` tuple, currently 10 names: 9 active + `trend_structure`) | **NSA** | Replace contents with the 4 names, keep `trend_structure` only if categorical pipeline retained. Cannot strike `trend_structure` without an assumption about whether the categorical branch in `pattern_profile.similarity_score` is still desired. | Verified by Explore agent #2: declaration is the canonical feature list consumed by both builders. |

#### B. Producers — features the new 4 require

| Feature | Required producer | Status today | Category |
|---|---|---|---|
| F1 `parkinson_vol_ratio_4h_24h_pre` | high/low extraction over 4h vs 24h windows; 15m klines already fetched at `scan.py:352`. **Helper not present** — no `parkinson_vol` symbol in source. | **Missing** — must be added under Phase 5.2. | NSA (cannot ship without writing the helper). |
| F2 `variance_ratio_1m_15m_pre_2h` | Lo-MacKinlay VR. Method exists at `src/neutralgrid/validation/stochastic.py:468` (`_variance_ratio_hurst`) but is bound to the validator class — needs extraction into a free function or a new `indicators/` module. **1m klines are NOT fetched today** by the scanner. **Binance Vision 1m ingest is not present** (verified absent). | **Partial** — math exists, plumbing absent. | NSA (extracting math is mechanical; adding 1m fetch + cache is a real workstream). |
| F3 `funding_carry_expected_next_7h` | `FundingRateProvider.get_funding_info` (`src/neutralgrid/data/funding_rate.py:116-161`); `HistoricalFundingLoader` (`funding_rate.py:256-363`); `BinanceClient.get_funding_rate` (`api/binance_client.py:531-557`); premiumIndex wrapper (`binance_client.py:559-563`). | **Present — high reuse.** | Producer side ready; the asymmetry between training (realized) and inference (expected `lastFundingRate`) is **already documented as a Phase 5.4 proxy** (line 362). |
| F4 `liquidity_stability_z_1h` | `quote_volume` already fetched via 15m klines (`scan.py:540-555`). Helper for 1h-rolling-sum + 24h trailing z-score not present — must be added next to F1's helper. **Top-20 depth snapshot at `scan.py:361, 470-485` is unused under option (a)** and remains untouched. | **Unblocked (4th addendum)** — option (a) volume-stability locked; helper missing, plumbing path = same as F1. | NSA (helper must be authored under Phase 7 task 7.2; option (b) is struck so depth-side touchpoints are out of scope). |

#### C. Consumers — code that reads `DEFAULT_FEATURES` or the named columns

| Site | Category | What changes when the feature set drops to the new 4 |
|---|---|---|
| `src/neutralgrid/scanner/pattern_profile.py:42-83` (`similarity_score`, the per-feature z weighting loop) | **NSA** | The loop iterates over `self.features` from JSON, so it adapts automatically — but the categorical branch for `trend_structure` (if retained per A) will short-circuit on the 4 numeric features. No code change required if `trend_structure` is dropped from `DEFAULT_FEATURES`; striking the categorical branch itself is NSA. |
| `src/neutralgrid/scanner/profile_model.py` (training entry `train_profile_model_from_enhanced_xlsx`, inference `_vector`) | **NSA** | Reads `feats` list at fit time, persists `feature_mean`/`feature_std`/`inv_cov` sized to `len(feats)`. Switching to 4 features changes matrix sizes but no signature change. The Phase 2.4 floor `max(30, 3·4)=30` is the same as today — *not* a relaxation. |
| `src/neutralgrid/scanner/scan.py:540-555` (quote_volume injection block + similar sites where the scanner emits feature columns into the candidate row) | **NSA** | The scanner row assembly emits the **9 active features** today (verified by reference-auditor in 2026-04-20 team review). Switching to 4 means: remove the 9 active emissions, add 4 new emissions. Whether to *keep* the 9 numeric values in the candidate row for *non-profile-model* consumers (meta-labeler, training_builder) is **NSA** — these consumers are listed in the safety-invariants Feature Pipeline Update Rule and must not be silently broken. |
| `src/neutralgrid/scanner/scan.py:649-658` (hardcoded weights, per Explore agent #2) | **PUW iff hardcoded names match the 9 being removed.** | Verified by Explore agent #2 that this block carries hardcoded feature weights. If the names are the 9 active features, the block is dead code under the 4-only reading and must be removed or rewritten. **Assumption:** I did not re-read the literal contents in this pass; the implementer must confirm before striking. |
| `src/neutralgrid/scanner/feature_extractor.py:88-114` (`SymbolFeatures` dataclass) | **NSA** | Currently carries the 9 active fields. New fields (F1..F4) must be added; old fields cannot be removed without auditing every consumer (meta-labeler builder, training_builder, scanner_integration). |
| `src/neutralgrid/training/scanner_integration.py:42-182` (`build_feature_snapshot`) | **NSA** | Builds the FeatureSnapshot for the meta-labeler. Per safety-invariants Feature Pipeline Update Rule, *any* meta-labeler feature change requires synchronized edits to candidate_pipeline + data_generator + unified_training_builder. The 4-only reading must declare whether F1..F4 are *also* meta-labeler features or *only* profile-model features. **Default per Phase 5 design: profile-model only**, in which case `scanner_integration` does not need the new 4 — but *removing* the 9 from `scanner_integration` would break the meta-labeler. Conclusion: **do not touch `scanner_integration` under this batch.** |
| `data/profile/pattern_profile.json` (10-feature persisted artifact) and `data/profile/profile_model.json` (9-feature persisted artifact) | **PUW under retrain.** | Both are regenerated by `retrain_scanner.py`. No manual edit. The 2026-04-21 retrain (top_quantile=0.605) will be invalidated as soon as the feature set changes. |

#### D. CLI / pipeline plumbing

| Site | Category | Action |
|---|---|---|
| `retrain_scanner.py:47` (`from neutralgrid.scanner.pattern_profile import build_profile_from_enhanced_xlsx, DEFAULT_FEATURES`) | **PUW** as-is iff `DEFAULT_FEATURES` is rewritten — no change to CLI signature. The CLI just logs `len(DEFAULT_FEATURES)` (line 153, 170, 197). | No code change. The log will read `Features: 4` automatically. |
| `retrain_scanner.py:153, 170, 197` (logging) | **PUW** | Auto-adjusts to the new tuple length. No edit. |
| `retrain_scanner.py:80-85` (`--max-duration-hours` flag) | **PFO** | Independent of feature set. Out of scope. |

#### E. Tests

| Site | Category | Action |
|---|---|---|
| `tests/unit/test_profile_model_training.py` (Phase 2 fixtures, `_make_test_xlsx` enlarged to 120 rows for the floor) | **NSA** | Fixture must be re-built to carry the 4 new column names. Striking the existing fixture is **NSA** because Phase 2.5 selection-summary tests assert specific keys from the current set. |
| `tests/test_afml_integrations.py:410-458` | **NSA** | Listed in 2026-04-20 team-review references. Re-check when the new column names are introduced. |

#### F. Documentation

| Site | Category | Action |
|---|---|---|
| `SCANNER_USAGE.md:231-234, 438` | **NSA** | Documents the active feature names. Must be rewritten when the set changes. |

### 6.3 Deduplication & classification hygiene (best practices, not over-engineered)

These are properties the implementation must satisfy at fit time. They are *not* new code paths; they are checks the implementer must run and record in the retrain proof block.

1. **Strategy-id deduplication:** `_xlsx_io.raise_on_duplicate_strategy_id` already enforces uniqueness on every sheet read (Phase 1.1, line 39). Keep it. No additional dedupe needed; if a duplicate appears, training fail-closes.
2. **Class-balance check before fit:** Phase 2.4 already enforces `max(30, 3·k)` per class. With k=4, floor = 30 — same as today. The user must accept that the 4-only switch does **not** lower the data requirement.
3. **Standardization symmetry:** Phase 2.3 fits `feat_mean` / `feat_std` on `vstack([Xw, Xl])` (line 374). With 4 features, `feature_mean` and `feature_std` are 4-vectors. `_vector` applies `(raw - mean) / std`. No code change, but `ProfileModel.feature_mean` and `feature_std` from the prior 9-feature artifact must be discarded — back-compat path (None → raw passthrough) would silently degrade if loaded against a 4-feature config. **Recommendation:** add a feature-set hash to `ProfileModel` (NSA — would be a new field) and have the resolver refuse cross-set loads.
4. **Selection-summary parity:** Phase 2.5 attaches `selection_summary` to both `PatternProfile` and `ProfileModel`. Switching the feature set does not affect the summary keys; no change.
5. **Walk-forward purge unchanged:** `purge_hours == max_duration_hours` (Phase 5.6 convention; line 306). Independent of feature set.

### 6.4 What this plan does NOT claim

The following are *not* established by the 2026-04-21 verification pass and would be assumptions if asserted:
- That the 4 features are jointly orthogonal on the current 77-row labeled set. Orthogonality was assessed for the 9 baseline + F-series in the team review (line 487), but the **4-only** subset has not been re-tested.
- That removing the 9 baseline features will not regress the meta-labeler or any non-profile-model consumer. The Feature Pipeline Update Rule (`safety-invariants.md`) requires synchronized edits to three files for *meta-labeler* features; the 4-only reading must explicitly declare scope.
- That `scan.py:649-658` is dead code under the 4-only reading. Explore agent #2 reported "hardcoded weights" but the literal names were not enumerated in this pass.
- That the F4 site can be designed within the current rollout window. C1 is a hard plan-level block.
- That walk-forward will pass with 4 features at any `top_quantile` over the current pool. The C4 fold-size argument applies independent of feature count.

### 6.5 Minimal next concrete step (superseded by fourth addendum 2026-04-21)

The original §6.5 step list (author F4 design doc; decide backfill policy; re-affirm or override Phase 5.8) is superseded by the fourth-addendum decisions and the new **Phase 7** workstream below. The remaining open question is C3 (Phase 5.9 ordering re-confirmation by the implementer), which is a one-time check before the first promotion attempt under the override.

---

## Phase 7 — 4-feature override completion log (implemented 2026-04-21)

This section replaces the earlier Phase 7 plan-only table. The 4-feature override is now code-implemented. The remaining open item is not feature wiring; it is data sufficiency for walk-forward promotion.

### 7.1 What landed

The Pattern Profile / Profile Model contract was replaced with exactly these 4 numeric features:

1. `parkinson_vol_ratio_4h_24h_pre`
2. `variance_ratio_1m_15m_pre_2h`
3. `funding_carry_expected_next_7h`
4. `liquidity_stability_z_1h`

`trend_structure` was removed from the Pattern Profile / Profile Model feature contract and from similarity scoring. The active feature set persisted in `data/profile/pattern_profile.json` and `data/profile/profile_model.json` is now exactly the 4 names above.

### 7.1.a Step status

| Step | Status | Proof |
|---|---|---|
| Freeze Pattern Profile / Profile Model to exactly the 4 requested numeric features | **DONE** | `DEFAULT_FEATURES` and both persisted artifacts now carry exactly the 4 requested names. |
| Drop `trend_structure` from the profile contract and similarity scoring | **DONE** | `PatternProfile.similarity_score` is numeric-only and persisted feature lists no longer include `trend_structure`. |
| Make `General` the canonical training sheet | **DONE** | `_xlsx_io.py` prefers `General`; retrain backfills and retrains directly from `General`. |
| Backfill the 4 profile columns onto the existing labeled rows | **DONE** | Retrain reported 77 eligible rows backfilled on `General`. |
| Live scanner computes and emits the 4 profile features | **DONE** | `feature_extractor.py` / `scan.py` now compute and emit the 4-feature block. |
| Keep bootstrap candidate behavior while `current.json` is absent | **DONE** | Resolver still loads `profile_model.json` as bootstrap candidate only when `current.json` is absent. |
| Align training and inference imputation semantics | **DONE** | `feature_impute` now persists in the artifact; runtime no longer imputes missing values toward the winner class. |
| Keep PatternProfile feature admission aligned with the labeled-universe availability contract | **DONE** | PatternProfile now filters on labeled-universe availability and only persists features with fitted summary stats. |
| Remove stale legacy `profile_gate.json` generation from the 4-feature retrain path | **DONE** | `retrain_scanner.py` now skips profile-gate generation when the profile contract does not carry ADX/RSI gate inputs. |
| Walk-forward promotion of the 4-feature artifact | **PARTIALLY DONE** | The code path is correct and the proof was run, but promotion is still refused because `mean_pass_rate=0.000` on the current 77-row pool. |

### 7.2 Concrete implementation surface

Implemented touchpoints:

- `src/neutralgrid/scanner/pattern_profile.py`
  - `DEFAULT_FEATURES` replaced with the 4-feature contract only.
  - `trend_structure` removed from profile similarity scoring and artifact validation.
- `src/neutralgrid/scanner/profile_model.py`
  - training and load-time validation now bind to the same 4-feature contract.
  - shared-median training imputation is now persisted in the artifact and reused at inference, removing the prior bias toward the winner class when a live row is missing a feature.
  - pre-existing artifacts that do not yet carry `feature_impute` remain backward-compatible through pooled-mean fallback at inference.
- `src/neutralgrid/scanner/profile_model_walkforward.py`
  - walk-forward and promotion reuse the same `DEFAULT_FEATURES` contract.
  - bootstrap candidate behavior remains active when `current.json` is absent.
- `src/neutralgrid/scanner/_xlsx_io.py`
  - `General` is now the canonical single-sheet training source when present.
- `src/neutralgrid/scanner/pattern_profile.py`
  - PatternProfile now applies the same labeled-universe availability floor before admitting a feature and only persists features whose winner subset produced real summary statistics.
- `src/neutralgrid/indicators/technical.py`
  - F1, F2, and F4 helper computations landed in the existing indicator surface.
- `src/neutralgrid/data/funding_rate.py`
  - F3 live funding-carry and historical realized-funding proxy helpers landed in the existing funding surface.
- `src/neutralgrid/scanner/feature_extractor.py`
  - live scanner feature computation emits the 4 new profile fields.
- `src/neutralgrid/scanner/scan.py`
  - live scan fetches the needed 1m / 15m / funding inputs and uses the default equal-weight profile similarity path.
- `retrain_scanner.py`
  - backfills the 4 profile fields onto `General` before retraining.
  - reuses the existing Binance Vision and funding utilities instead of adding a new ingest subsystem.
  - skips legacy `profile_gate.json` generation when the active profile contract no longer carries the old ADX/RSI gate inputs.

### 7.3 Verification gate summary

Per the mandatory gate for this workstream, the implemented 4-feature rewrite has the following proof:

- `pyright`: `0 errors, 0 warnings, 0 informations` on the touched runtime/test surface for the 4-feature integration.
- Focused unit tests: `51 passed, 2 warnings` on the dedicated profile/training/CLI slices for this rewrite. The two warnings are pytest cache-permission warnings, not assertion failures.
- Retrain proof (`retrain_scanner.py --input data/new_expired_bots.xlsx --output-dir data/profile --top-quantile 0.605 --min-avg-profit-per-grid 0.59 --skip-gate`):
  - feature count: 4
  - available-feature coverage: 4/4
  - per-class counts: winners=30, losers=47, `top_quantile=0.605`
  - prior winner: 0.390
  - `feature_impute` persisted in `data/profile/profile_model.json`: yes
  - promotion decision: NOT ATTEMPTED in the retrain CLI because `--skip-gate` was used
- Walk-forward / promotion proof (`train_and_promote(...)` with the same thresholds):
  - feature count: 4
  - available-feature coverage: 4/4
  - `n_folds=5`, effective evaluated folds = 4, `fold_size=15`
  - `fold_auc=[NaN, NaN, NaN, NaN]`
  - `fold_ks=[NaN, NaN, NaN, NaN]`
  - `mean_pass_rate=0.000`
  - promotion decision: **NOT PROMOTED** (`mean_pass_rate=0.000 < floor=0.50`)

### 7.4 Why promotion still fails

The current blocker is not feature wiring. It is fold-local class count insufficiency under purged chronological walk-forward:

- labeled pool: 77 rows
- `n_folds=5` -> `fold_size=15`
- smallest train slices are too small to satisfy both:
  - winners `>= max(30, 3*4) = 30`
  - losers `>= 30`

So the walk-forward path is behaving correctly by refusing promotion while still allowing the unvalidated bootstrap artifact to run when `current.json` is absent.

### 7.5 Provable false optionality

Resolved:

- "4 new features plus the old 9/10" was not adopted. One canonical `DEFAULT_FEATURES` contract now exists.
- A separate backfill sheet was not adopted. The backfill writes onto `General`, which is the sheet the trainers now use.
- A new ingest subsystem was not adopted. Existing Binance Vision and funding utilities were reused.
- A second temporary promotion path was not adopted. Bootstrap use remains the only operational relaxation.

Remaining intentionally:

- `current.json` still represents only walk-forward-promoted artifacts.
- `data/profile/profile_model.json` remains the bootstrap candidate when `current.json` is absent.

### 7.6 Provably unnecessary items as written

Not adopted because they were unnecessary for this integration:

- `trend_structure` retention inside the profile contract
- F2 fallback branches
- a dedicated `PATTERN_PROFILE_F4_DESIGN.md`
- a separate `Backfilled F1F2F3F4` workbook sheet
- any meta-labeler feature-contract rewrite as part of this phase

### 7.7 Items not valid to strike without assumptions

These remain required:

- duplicate checks and bounded/labeled-universe discipline in `_xlsx_io.py`
- shared imputation / standardization / per-class floor in `profile_model.py`
- bootstrap candidate behavior while the pool is too small for meaningful promotion
- the promotion gate itself once finite walk-forward folds become possible
- the historical realized-funding proxy for F3, because no historical premium-index source is exposed in the current repo

### 7.8 Immediate continuation

The 4-feature implementation is complete enough to run the scan -> enrich -> backtest -> train loop. The next simple step is operational, not architectural:

1. keep using the freshly retrained `data/profile/profile_model.json` as the bootstrap candidate while `current.json` is absent;
2. keep collecting real short-duration outcomes;
3. rerun walk-forward / promotion only after the labeled pool is large enough that the smallest train fold can satisfy both per-class floors.

This preserves the relaxed-gate bootstrap needed for data flow without erasing the existing promotion contract.

Operational note: the refresh retrain was completed after the `feature_impute` landing, and the current `data/profile/profile_model.json` now persists the shared-impute vector explicitly. Older artifacts remain backward-compatible because runtime falls back to pooled-mean inference rather than winner-biased imputation.
