# AUDIT_01 — NEUTRAL Grid Bot v6.5.7 Full Pipeline Audit

**Date:** 2026-05-05
**Scope:** read-only audit of the worktree at
`C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7\.claude\worktrees\add-reviewer`
**Branch:** `main`
**Method:** static review (Read/Grep/Glob) + pyright. No code was changed.
All paths are absolute. Verbatim line excerpts and pyright output are quoted directly.

---

## 1. Executive summary

- **Pyright (basic mode, src/) — 104 errors, 0 warnings, 0 informations**, all confined to two files: `src\neutralgrid\calibration\hmm_winner_calibrator.py` (48) and `src\neutralgrid\grid\spacing_profile.py` (56). Most are pandas typing fan-out (one source-line generates ~10 reports because pyright enumerates Series union variants); the underlying defects cluster around 6–8 distinct call sites.
- **Leakage guards intact (high confidence)**: `_KNOWN_LABEL_COLUMNS` is enforced at both required sites — `meta_labeler._prepare_features` (line 600) and `meta_labeler.train` (line 775). `hlabel` is in the guard set (line 282).
- **`build_training_config()` / `run_backtest()` invariant intact (high confidence)**: production callers route through `backtest.btk_unified_runner.run_backtest`. The only direct uses of `RealisticGridBacktester` outside tests are docstring/comment references (no actual imports).
- **`hierarchical_label.hurdle_pct == barrier.meta_hurdle_pct` invariant ENFORCED at startup**: `core/config.py:616-622` raises on drift.
- **Version constants centralized correctly**: only `core/constants.py` defines `LABEL_CONTRACT_VERSION` / `FORMULA_VERSION` / `ENGINE_VERSION`; `backtest/btk_label_contract.py:28-33` re-exports through `try/except ImportError` fallback that duplicates the literals — this is a documented mirroring pattern, but the duplication is real and creates a drift surface (AUDIT-003).
- **`UtilityCalibratorUnavailable` propagation is correct at the runtime path** but I found one fail-closed gap: `regime_validator.check_stochastic_regime` hard-codes `survival_min = 0.0` (line 814) and there is no `StochasticConfig.survival_min` field in `core/config.py:213-222`. The MC containment gate is therefore degenerate — `result.survival_prob >= 0.0` is true by construction (AUDIT-004).
- **Stage B Gate 4 (`two_stage_selector.approve`)**: branching matches the safety-invariants spec (standard archetype uses entropy-adaptive `range_prob`; micro-oscillator archetype uses `survival_prob >= min_survival_prob`). Both branches preserve `data_missing:range_prob` rejection codes; no `data_missing:utility` code is emitted because no utility gate is wired into Stage B yet (matches CLAUDE.md "Future Stage B utility gate" wording — AUDIT-008).
- **Backfill `--default-artifact-version` authoritative behaviour confirmed (high confidence)**: `scripts/backfill_training_features.py:558-619` invalidates preserved `HMM_DERIVED_COLUMNS ∪ HMM_LINEAGE_COLUMNS` when stale; `--skip-if-fresh` short-circuit at lines 654-671 only triggers when preserved version equals explicit default AND derived probs are finite — the docstring claim is faithful.
- **Feature Pipeline Update Rule — three-file consistency holds for the active feature set**, but `regime_conf`, `micro_round_trip_cost_pct`, and `long_short_ratio` are present in some lists and absent from `TRAINING_OUTPUT_COLUMNS`. They are added at runtime via `OPTIONAL_LIVE_PLUS_FIELDS_V20260312` and a hard-coded extension list in `build_feature_dict_from_scanner_row` (candidate_pipeline.py:798-800). This is intentional plumbing but creates a non-obvious contract surface (AUDIT-007).
- **Training-feature coverage of the meta-labeler's active profile is OK at runtime** (`ACTIVE_SNAPSHOT_META_FEATURES = SNAPSHOT_META_FEATURES_V20260421_BOOTSTRAP`, 8 fields, all in `FeatureSnapshot.to_dict` and `_SCANNER_TO_FEATURE`), but the `TRAINING_FEATURES` symbol is shadowed: `unified_training_builder.py:48` reassigns `TRAINING_FEATURES = list(DEFAULT_SCHEMA.features)` — different from `candidate_pipeline.TRAINING_FEATURES = TRAINING_OUTPUT_COLUMNS`. Two unrelated value sets share one identifier in two modules (AUDIT-009).

**Categories:** confirmed bug = AUDIT-004, AUDIT-005; safety-invariant violation = none confirmed; data-gap = AUDIT-003, AUDIT-007, AUDIT-008; code-quality = AUDIT-001, AUDIT-002, AUDIT-006, AUDIT-009; open-question = AUDIT-OQ-1, AUDIT-OQ-2.

---

## 2. Pyright summary

Command: `pyright` from worktree (`pyrightconfig.json` + `pyproject.toml [tool.pyright]` both set `typeCheckingMode = "basic"`, `include = ["src"]`).

**Totals (verbatim, last line of `.pyright_audit.log`):**
```
104 errors, 0 warnings, 0 informations
```

**Distribution:**
- `src\neutralgrid\calibration\hmm_winner_calibrator.py` — 48 error lines (clustered at lines 348, 356, 463, 464, 465).
- `src\neutralgrid\grid\spacing_profile.py` — 56 error lines (clustered at lines 153–156, 318, 320, 321, 322, 366, 457, 555, 557, 586, 589).

**Top issue classes (verbatim from `.pyright_audit.log`):**

1. `hmm_winner_calibrator.py:348:71 — error: Cannot access attribute "fillna" for class "float"` (and ~9 other Series-union variants). Site: `frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(float(medians[column]))` (`hmm_winner_calibrator.py:348`).
2. `hmm_winner_calibrator.py:356:31 — error: Cannot access attribute "median" for class ...`. Site: `median = float(values.median())` (`hmm_winner_calibrator.py:356`).
3. `hmm_winner_calibrator.py:463:67 — error: Cannot access attribute "to_numpy" for class "float"` plus `463:82 — error: Argument of type "type[float]" cannot be assigned to parameter "dtype" of type "dtype[Any] | str | None" in function "to_numpy"`. Site: `range_prob = pd.to_numeric(df["range_prob"], errors="coerce").to_numpy(dtype=float)` (`hmm_winner_calibrator.py:463`). Same pattern at `464:67`, `464:82` for `trend_prob`.
4. `hmm_winner_calibrator.py:465:12 — error: Operator "-" not supported for types "ndarray ... | timedelta64 | datetime64" ...` and `465:12 — error: Type ... is not assignable to return type "ndarray[_AnyShape, dtype[Any]]"`. Site: `return range_prob - trend_prob` — pyright cannot prove the operands are float arrays because the union from `to_numpy` is too wide.
5. `spacing_profile.py:153:19, 154:19, 156:19 — error: Argument of type "Series | ndarray | Any | Unknown" cannot be assigned to parameter "x" of type "ConvertibleToFloat" in function "__new__"`. Site: `float(row["price_range_low"])`, `float(row["price_range_high"])`, `float(row["grid_spacing_pct"]) ...`.
6. `spacing_profile.py:155:17 — error: Argument of type "Series | ndarray | Any | Unknown" cannot be assigned to parameter "x" of type "ConvertibleToInt"` plus `155:40 — error: Invalid conditional operand of type "Series | NDArray[bool_]"`. Site: `int(row["grids_count"]) if pd.notna(row["grids_count"]) else 0`.
7. `spacing_profile.py:318:53 — error: Cannot access attribute "dropna" for class "float"`, `320:25 — error: Cannot access attribute "quantile" for class "DatetimeIndex"`, `321:29 — median`, `322:25 — quantile`. Site: the `_iqr(series: pd.Series)` body at 317-323. Pyright resolves `series` as `pd.Series` but `pd.to_numeric(series, errors="coerce")` returns the wide pandas union; `.dropna()` then fans out across the union.
8. `spacing_profile.py:366:26 — error: Argument of type "Series | int | Unknown" cannot be assigned to parameter "x" of type "ConvertibleToInt"`. Site: `winner_symbols = int(winners["symbol"].nunique()) if "symbol" in winners.columns else 0`.
9. `spacing_profile.py:457:33 — error: Argument of type "Expression | Unknown | Any" cannot be assigned to parameter "x" of type "ConvertibleToFloat"`. Site: `range_edges=tuple(float(v) for v in raw_edges)` — pyright cannot narrow the iterated element type because `pd.qcut(..., retbins=True)` returns a heterogeneous tuple.
10. `spacing_profile.py:555:28 .. 589:23 — float(row["..."])` cluster inside `for _, row in geometry.iterrows()` body.

All 104 errors are **type-narrowing / pandas-stub** problems, not logic bugs. None of them violate a safety invariant. None are leakage or fail-open issues. They cluster around three legitimate root causes: (a) `pd.to_numeric` returning a union type that includes scalar variants (issues 1, 2, 7), (b) `pd.Series.to_numpy(dtype=float)` not matching pyright's signature (issue 3), and (c) `row["col"]` from `iterrows()` returning a wide union (issues 5, 6, 8, 10). This is the same `df[col]` Series-narrowing pattern called out in CLAUDE.md "pandas Pyright Patterns".

---

## 3. Findings (numbered AUDIT-NNN)

### AUDIT-001 — pyright: pandas Series-narrowing fan-out in spacing_profile.py
- **Category:** code-quality
- **Location:** `src\neutralgrid\grid\spacing_profile.py:153-156`, `318`, `320-322`, `366`, `457`, `555`, `557`, `586`, `589`
- **Verbatim excerpt** (`spacing_profile.py:151-157`):
  ```python
  inferred = [
      _infer_mode(
          float(row["price_range_low"]),
          float(row["price_range_high"]),
          int(row["grids_count"]) if pd.notna(row["grids_count"]) else 0,
          float(row["grid_spacing_pct"]) if pd.notna(row["grid_spacing_pct"]) else float("nan"),
      )
      for _, row in df.iterrows()
  ]
  ```
- **Observation:** pyright infers `row["price_range_low"]` as `Series | ndarray | Any | Unknown`. `float(...)` rejects this union because `Series.__float__` is not defined. The generator produces 56 reports for 8 distinct call sites, dwarfing the actual defect count.
- **Why it matters:** the file owns the spacing target — the production winner pool computation. CLAUDE.md "pandas Pyright Patterns" prescribes `cast(pd.Series, df[col]).iloc[...]` or per-cell `pd.to_numeric(..., errors="coerce").iloc[0]`. The current code reads from `iterrows()` rows directly with `row["col"]` and then `float(...)`, bypassing the convention.
- **Validation proof:** pyright excerpt `spacing_profile.py:153:19 — error: Argument of type "Series | ndarray[_AnyShape, dtype[Any]] | Any | Unknown" cannot be assigned to parameter "x" of type "ConvertibleToFloat" in function "__new__"`.
- **Confidence:** high (verbatim pyright output).

### AUDIT-002 — pyright: hmm_winner_calibrator.py pandas typing
- **Category:** code-quality
- **Location:** `src\neutralgrid\calibration\hmm_winner_calibrator.py:348`, `356`, `463-465`
- **Verbatim excerpt** (`hmm_winner_calibrator.py:346-349`):
  ```python
  frame = df.loc[:, list(feature_columns)].copy()
  for column in feature_columns:
      frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(float(medians[column]))
  return frame.to_numpy(dtype=float)
  ```
  And `462-465`:
  ```python
  def _baseline_scores(df: pd.DataFrame) -> np.ndarray:
      range_prob = pd.to_numeric(df["range_prob"], errors="coerce").to_numpy(dtype=float)
      trend_prob = pd.to_numeric(df["trend_prob"], errors="coerce").to_numpy(dtype=float)
      return range_prob - trend_prob
  ```
- **Observation:** `pd.to_numeric(...).to_numpy(dtype=float)` and `pd.to_numeric(...).fillna(...)` both produce a wide pandas union under pyright basic mode. Same mechanical issue as AUDIT-001.
- **Why it matters:** this is the active HMM-winner calibrator, part of the FIXPIPELINE-01 close-out (per Memory). Pyright noise here makes future regressions harder to detect; it does not represent a runtime bug.
- **Validation proof:** pyright excerpt `hmm_winner_calibrator.py:463:82 — error: Argument of type "type[float]" cannot be assigned to parameter "dtype" of type "dtype[Any] | str | None" in function "to_numpy"`.
- **Confidence:** high.

### AUDIT-003 — Version-constant duplication in `backtest/btk_label_contract.py`
- **Category:** data-gap (drift surface) / code-quality
- **Location:** `backtest\btk_label_contract.py:28-33`
- **Verbatim excerpt** (from `Grep` LABEL_CONTRACT_VERSION):
  ```python
  28:        FORMULA_VERSION as FORMULA_VERSION,
  29:        LABEL_CONTRACT_VERSION as LABEL_CONTRACT_VERSION,
  ...
  32:    LABEL_CONTRACT_VERSION: str = "2026-04-17"
  33:    FORMULA_VERSION: str = "alignment-v1"
  ```
- **Observation:** `btk_label_contract.py` re-exports the version constants under a `try / except ImportError` fallback that re-defines them with the same literal values. Today they match `core/constants.py:18,23` (`"2026-04-17"`, `"alignment-v1"`).
- **Why it matters:** safety-invariants.md §"Version Constants" requires a single source of truth and forbids duplicate literals. The `try/except ImportError` fallback keeps `backtest/` runnable when the package is not installed, but if a future bump in `core/constants.py` is not mirrored, the fallback path silently uses the stale value. There is no test that asserts `core.constants.LABEL_CONTRACT_VERSION == backtest.btk_label_contract.LABEL_CONTRACT_VERSION` under fallback conditions.
- **Validation proof:** grep results above; `core/constants.py:18,23` define the canonical values.
- **Confidence:** high for the structural duplication; medium for "is it ever exercised at runtime" — the import path is `from neutralgrid.core.constants import ...` which works when the package is installed, so the fallback should be dead in normal runs. Worth confirming.

### AUDIT-004 — Stochastic gate `survival_min` is degenerate (always 0.0)
- **Category:** bug
- **Location:** `src\neutralgrid\validation\regime_validator.py:807-825` and `src\neutralgrid\core\config.py:213-222`
- **Verbatim excerpt** (`regime_validator.py:807-825`):
  ```python
  # Get config parameters
  survival_horizon = int(_cfg.stochastic.survival_horizon_bars)
  mc_paths = int(_cfg.stochastic.survival_mc_paths)
  hurst_max = float(_cfg.stochastic.hurst_max_trending)
  ou_halflife_min = int(_cfg.stochastic.ou_halflife_min_bars)
  ou_halflife_max = int(_cfg.stochastic.ou_halflife_max_bars)
  survival_min = 0.0
  ...
      stoch_config = StochasticConfig(
          ...
          survival_min=survival_min,
          ...
  ```
  And `core/config.py:213-222` (the `Config` / `_cfg.stochastic` source):
  ```python
  @dataclass
  class StochasticConfig:
      """Stochastic regime check parameters."""
      enable: bool = True
      survival_horizon_bars: int = BOT_HORIZON_BARS_15M
      survival_mc_paths: int = 10000
      hurst_max_trending: float = 0.65
      ou_halflife_min_bars: int = 4
      ou_halflife_max_bars: int = 48
  ```
- **Observation:** `core/config.py StochasticConfig` (the runtime config dataclass) **does not expose a `survival_min` field**. Only the per-call `validation/stochastic.py StochasticConfig` (line 54) defines it (`survival_min: float = 0.0`). Inside `RegimeValidator.check_stochastic_regime` the value is hard-coded to `0.0` (line 814) before being passed to the per-call `StochasticConfig`. Result: `survival_ok = survival >= 0.0` is true for every finite survival probability, so the survival arm of the stochastic gate is a no-op.
- **Why it matters:** safety-invariants.md says "Gate 4 tests survival_prob >= min_survival_prob (MC containment). Gate 4 remains mandatory in both modes." That refers to **Stage B** Gate 4 in `two_stage_selector.py`, which uses a different threshold (`get_config().micro_osc.min_survival_prob`). The `RegimeValidator` stochastic gate is separate. But its current form means failures CAN only come from `hurst_ok` and `halflife_ok`; the named "survival" sub-check is structurally inert. Same pattern is repeated at `scanner\enrich_grid_params.py:295` and `scanner\scan.py:224`.
- **Validation proof:** grep `survival_min` results above; no `survival_min` attribute appears anywhere in `core/config.py`.
- **Confidence:** high that the value is hard-coded; medium that this is unintentional vs. intentional (could be by design as "containment is checked by the Stage B micro-oscillator branch instead"). Recommend confirming with the original author.

### AUDIT-005 — `mode` field carried through training pipeline as a string column
- **Category:** bug-risk / code-quality
- **Location:** `src\neutralgrid\training\unified_training_builder.py:79-84`, `src\neutralgrid\backtest\candidate_pipeline.py:161-162`, `src\neutralgrid\models\meta_labeler.py:138`
- **Verbatim excerpt** (`unified_training_builder.py:79-84`):
  ```python
  # GRID_SYNCH §1.5 — `mode` is bot-side metadata carried for audit / training-time
  # branching only. It must NOT enter the model X-matrix (categorical string would
  # crash classifiers / mis-encode under label-encoding). See plan §6 invariant C3
  # and meta_labeler.py:138 (ACTIVE_SNAPSHOT_META_FEATURES does NOT contain it).
  "mode",
  ```
- **Observation:** `mode` is intentionally appended to `EXTRA_META_FEATURES` (line 83) and `TRAINING_OUTPUT_COLUMNS` (line 162) for audit. The active feature profile `SNAPSHOT_META_FEATURES_V20260421_BOOTSTRAP` (line 127–136 of meta_labeler.py) does NOT contain it. So the runtime guard relies on `ACTIVE_SNAPSHOT_META_FEATURES`. There is no `mode` entry in `_KNOWN_LABEL_COLUMNS` and no explicit "string column" filter in `_prepare_features`.
- **Why it matters:** if a future refactor accidentally widens `MetaLabelerConfig.features` from `ACTIVE_SNAPSHOT_META_FEATURES` to "all dataset columns", the string column would crash `pd.to_numeric(..., errors="coerce")` softly (becomes NaN) but then sklearn's class-weight check would still fit — silent feature corruption rather than a crash. The current design puts the entire load of this invariant on the `MetaLabelerConfig.features` default factory at meta_labeler.py:453 (`field(default_factory=lambda: list(ACTIVE_SNAPSHOT_META_FEATURES))`).
- **Validation proof:** cross-file grep; `meta_labeler.py:138` defines the active profile; `unified_training_builder.py:79-84` and `candidate_pipeline.py:161-162` document the intent in comments only.
- **Confidence:** medium. The current invariant holds; the risk is that no test asserts "no string columns can leak into `X_raw`". A `mode` value of e.g. `"geometric"` reaching `pd.to_numeric` returns NaN with `errors="coerce"`, which the existing imputer would silently fill — feature mass with no signal.

### AUDIT-006 — `_PostProbabilityCalibrator._apply` does not normalize raw probability shape
- **Category:** code-quality
- **Location:** `src\neutralgrid\models\meta_labeler.py:368-388`
- **Verbatim excerpt:**
  ```python
  def _apply(self, proba: np.ndarray) -> np.ndarray:
      p = np.asarray(proba, dtype=float).reshape(-1)
      if self.calibrator is None or self.method is None:
          return np.clip(p, 0.0, 1.0)
      if self.method == "isotonic_oos":
          p = np.asarray(self.calibrator.predict(p), dtype=float)
      elif self.method == "sigmoid_oos":
          p = np.asarray(
              self.calibrator.predict_proba(p.reshape(-1, 1)), dtype=float
          )[:, 1]
      elif self.method == "beta_oos":
          # BetaCalibrator uses predict_proba(X) → (N, 2) interface
          p = np.asarray(
              self.calibrator.predict_proba(p.reshape(-1, 1)), dtype=float
          )[:, 1]
      return np.clip(p, 0.0, 1.0)
  ```
- **Observation:** if `self.method` is set to a string outside `{"isotonic_oos","sigmoid_oos","beta_oos"}` (e.g., a serialized artifact from a removed calibrator family), the function silently returns the unclipped raw probabilities without the requested calibration step. There is no `else: raise` branch. Combined with `_evaluate_oos_calibration_gate` (line 679) — that gate validates ECE/Brier, but it cannot detect "method name was unrecognized so calibration was a no-op."
- **Why it matters:** safety-invariants.md §Fail-Closed Behavior. For the calibration layer, "fail closed" should mean an unknown method aborts inference rather than silently degrading. This is a soft fail-open path.
- **Validation proof:** source excerpt above.
- **Confidence:** high. The branch is missing a default-case raise.

### AUDIT-007 — `regime_conf`, `micro_round_trip_cost_pct`, `long_short_ratio` flow inconsistency
- **Category:** data-gap (contract surface)
- **Location:** `src\neutralgrid\backtest\candidate_pipeline.py:103-163, 168-177, 798-803`, `src\neutralgrid\training\data_generator.py:169, 173, 181`, `src\neutralgrid\training\unified_training_builder.py:49-84`
- **Observations:**
  - `regime_conf` is in `OPTIONAL_LIVE_PLUS_FIELDS_V20260312` (candidate_pipeline.py:176) and in `FeatureSnapshot` (data_generator.py:169) and in `_SCANNER_TO_FEATURE` (candidate_pipeline.py:62), but NOT in `TRAINING_OUTPUT_COLUMNS`.
  - `micro_round_trip_cost_pct` is in `_SCANNER_TO_FEATURE` (line 65), `FeatureSnapshot` (data_generator.py:173) and `EXTRA_META_FEATURES` (unified_training_builder.py:51), but NOT in `TRAINING_OUTPUT_COLUMNS`.
  - `long_short_ratio` is in `_SCANNER_TO_FEATURE` (line 75), `FeatureSnapshot` (line 181) and `EXTRA_META_FEATURES` (line 57), but NOT in `TRAINING_OUTPUT_COLUMNS`.
  - `build_feature_dict_from_scanner_row` (candidate_pipeline.py:798-803) compensates by appending `micro_round_trip_cost_pct` and `long_short_ratio` to the NaN-filler loop, but `regime_conf` is only filled via `OPTIONAL_LIVE_PLUS_FIELDS_V20260312`.
- **Verbatim excerpt** (candidate_pipeline.py:797-803):
  ```python
  for feat in TRAINING_OUTPUT_COLUMNS + OPTIONAL_LIVE_PLUS_FIELDS_V20260312 + [
      "micro_round_trip_cost_pct", "long_short_ratio",
  ]:
      if feat not in features:
          features[feat] = float("nan")
  ```
- **Why it matters:** the safety-invariants Feature Pipeline Update Rule names three sites (`_SCANNER_TO_FEATURE` + `TRAINING_OUTPUT_COLUMNS` in candidate_pipeline.py; `FeatureSnapshot` + `to_dict()` in data_generator.py; `EXTRA_META_FEATURES` + `_SCAN_TO_FEATURE` in unified_training_builder.py). Three feature names live in 5 of those 6 places but not in `TRAINING_OUTPUT_COLUMNS`. The actual runtime contract (which columns end up in the training CSV) is governed by `TRAINING_OUTPUT_COLUMNS + OPTIONAL_LIVE_PLUS_FIELDS_V20260312 + ["micro_round_trip_cost_pct", "long_short_ratio"]`. This hidden third extension is what creates the asymmetry.
- **Validation proof:** grep results above; line numbers verified.
- **Confidence:** high. This is documented as intentional plumbing in `meta_labeler.py:138` (active profile excludes them) but is fragile: a developer following the safety-invariants update rule literally would only edit 3 of the 4 effective lists.

### AUDIT-008 — No `data_missing:utility` rejection code in Stage B
- **Category:** data-gap (per CLAUDE.md, expected future work)
- **Location:** `src\neutralgrid\scanner\two_stage_selector.py:120-220`
- **Verbatim excerpt** (CLAUDE.md `safety-invariants.md` §Fail-Closed Behavior, last bullet):
  ```
  Future Stage B utility gate (if added) MUST use rejection code
  'data_missing:utility', parallel to 'data_missing:tos' /
  'data_missing:range_prob'.
  ```
  And the actual selector — there is no utility gate (only Gates 1 hard, 2 tos, 3 sizer, 4 regime, 5 conformal). `validation/utility.py` raises `UtilityCalibratorUnavailable`, but Stage B never queries the calibrator.
- **Observation:** this matches the current spec ("Future" — not yet added) and is therefore not a violation. I list it so it is not lost from the audit horizon. The existing path is: `RegimeValidator.check_hmm_regime` catches `UtilityCalibratorUnavailable` and emits `utility_score=None, utility_passed=False` (regime_validator.py:393-411). When `_cfg.hmm.pass_mode in {"utility", "hybrid"}`, `base_passed = utility_passed = False` → reject with `hmm_regime_fail`. So fail-closed is satisfied for the regime path. But the symmetry the spec asks for (an explicit `data_missing:utility` Stage B code) is not implemented.
- **Validation proof:** source excerpts above.
- **Confidence:** high (explicit absence; no test triggers the code).

### AUDIT-009 — `TRAINING_FEATURES` symbol shadowed across modules
- **Category:** code-quality
- **Location:** `src\neutralgrid\backtest\candidate_pipeline.py:166`, `src\neutralgrid\training\unified_training_builder.py:48`
- **Verbatim excerpts:**
  - `candidate_pipeline.py:165-166`:
    ```python
    # Backward-compatible alias (deprecated — use TRAINING_OUTPUT_COLUMNS)
    TRAINING_FEATURES = TRAINING_OUTPUT_COLUMNS
    ```
  - `unified_training_builder.py:48`:
    ```python
    TRAINING_FEATURES = list(DEFAULT_SCHEMA.features)
    ```
- **Observation:** same name, two semantically distinct lists, in two modules. `tests/unit/test_enhancements_v653.py:745-746` imports it from `neutralgrid.backtest.candidate_pipeline`; `tests/unit/test_unified_training_builder.py:19` imports it from `neutralgrid.training.unified_training_builder`. Each test exercises a different list.
- **Why it matters:** future readers can confuse the two. No active bug, but a subtle drift surface — if one of the two definitions is updated, the other will not flag.
- **Validation proof:** grep results.
- **Confidence:** high.

### AUDIT-OQ-1 (open question) — Provisional utility callers and `from_artifact()` propagation
- **Category:** open-question
- **Location:** `src\neutralgrid\validation\utility.py:450-480`
- **Excerpt:**
  ```python
  def compute_governed_provisional_utility(
      *,
      range_prob: float,
      trend_prob: float,
      ...
  ) -> UtilityComponents:
      ...
      scorer = UtilityScorer(UtilityConfig.from_artifact(artifact_path=artifact_path, **overrides))
      return scorer.compute_utility(...)
  ```
- **Observation:** `UtilityConfig.from_artifact()` raises `UtilityCalibratorUnavailable` when missing. `compute_governed_provisional_utility` does NOT catch — it propagates. Callers in `regime_validator.py:393-411` and `unified_training_builder.py` (offline) and `scripts/backfill_training_features.py:399-416` all catch. **Question for the user:** does any other call path invoke `compute_governed_provisional_utility` and *not* catch the exception (i.e., a decision-time path that should propagate rather than catch)? I did not enumerate every call site.
- **Confidence:** medium that all known offline callers catch; I cannot verify "all decision-time callers propagate" without exhaustive inspection.

### AUDIT-OQ-2 (open question) — Whether `_FEATURE_MEDIAN_DEFAULTS` for `range_prob = 0.5` could mask missing features in inference
- **Category:** open-question
- **Location:** `src\neutralgrid\models\meta_labeler.py:225-260` (default-imputation map for legacy models without fitted imputer)
- **Excerpt:** `"range_prob": 0.5,` and the legacy fallback at line 622-633:
  ```python
  # Fallback for legacy models without fitted imputer: use per-feature
  # median defaults instead of zero (which distorts bounded features).
  logger.warning("Using median-default imputation (no fitted imputer)")
  fill_values = np.array(
      [_FEATURE_MEDIAN_DEFAULTS.get(f, 0.0) for f in feats],
      dtype=float,
  )
  ```
- **Observation:** when the active artifact ships without an imputer (legacy), missing `range_prob` is replaced with `0.5` — neutral but not signaling "missing." A downstream Stage B that compares `range_prob >= min_range_prob` will then transparently fail for `min > 0.5` and pass for `min <= 0.5`, with no `data_missing:range_prob` code emitted. Whether the active artifact uses this fallback or not is an artifact-state question I cannot answer from source.
- **Confidence:** medium. The fallback exists; whether it's reached at runtime depends on the deployed artifact.

---

## 4. Cross-file contract checks

### 4.1 Feature Pipeline Update Rule (3 files)

The required sites and their authoritative list members for the active profile (`SNAPSHOT_META_FEATURES_V20260421_BOOTSTRAP`):

| Active feature       | candidate_pipeline `_SCANNER_TO_FEATURE` | candidate_pipeline `TRAINING_OUTPUT_COLUMNS` | data_generator `FeatureSnapshot` + `to_dict` | unified_training_builder `EXTRA_META_FEATURES` | unified_training_builder `_SCAN_TO_FEATURE` |
|----------------------|------------------------------------------|----------------------------------------------|----------------------------------------------|------------------------------------------------|----------------------------------------------|
| `range_prob`         | yes (line 36-38)                         | yes (104)                                    | yes (135, 222)                               | (in `TRAINING_FEATURES`)                       | yes (89-90)                                  |
| `survival_prob`      | yes (43)                                 | yes (107)                                    | yes (146, 231)                               | (in `TRAINING_FEATURES`)                       | yes (95)                                     |
| `utility_score`      | yes (40-42)                              | yes (106)                                    | yes (137, 224)                               | (in `TRAINING_FEATURES`)                       | yes (93-94)                                  |
| `ou_halflife`        | yes (45)                                 | yes (109)                                    | yes (148, 233)                               | (in `TRAINING_FEATURES`)                       | yes (97)                                     |
| `profit_per_grid_pct`| yes (46)                                 | yes (110)                                    | yes (151, 234)                               | (in `TRAINING_FEATURES`)                       | yes (98)                                     |
| `num_grids`          | yes (47)                                 | yes (111)                                    | yes (152, 235)                               | (in `TRAINING_FEATURES`)                       | yes (99)                                     |
| `grid_spacing_pct`   | yes (48)                                 | yes (112)                                    | yes (153, 236)                               | (in `TRAINING_FEATURES`)                       | yes (100)                                    |
| `adx_1h`             | yes (52)                                 | yes (114)                                    | yes (158, 239)                               | (in `TRAINING_FEATURES`)                       | yes (102)                                    |

**Verdict:** all 8 active features are uniformly present across the rule's 6 lists (with `TRAINING_FEATURES` from `DEFAULT_SCHEMA.features` covering the union via the `unified_training_builder.py:48` aliasing).

The cross-list asymmetries in AUDIT-007 (`regime_conf`, `micro_round_trip_cost_pct`, `long_short_ratio`) involve *additional* features that are intentionally outside the active profile but carried for audit/future reactivation.

### 4.2 Version constants

- `core/constants.py:18,23,28`: `LABEL_CONTRACT_VERSION = "2026-04-17"`, `FORMULA_VERSION = "alignment-v1"`, `ENGINE_VERSION = "realistic-v7"`.
- `backtest/btk_label_contract.py:32-33`: defines `LABEL_CONTRACT_VERSION = "2026-04-17"` and `FORMULA_VERSION = "alignment-v1"` as fallback under `try/except ImportError`. Currently in sync; structural duplication = AUDIT-003.
- `tests/` and `CHANGELOG.md` reference these as documentation only (no executable duplicates).

### 4.3 Leakage guards

`_KNOWN_LABEL_COLUMNS` is enforced at:
- `meta_labeler.py:600` (in `_prepare_features`):
  ```python
  leaked = _KNOWN_LABEL_COLUMNS.intersection(feats)
  if leaked:
      logger.warning("Dropping label columns from features: %s", sorted(leaked))
      feats = [f for f in feats if f not in _KNOWN_LABEL_COLUMNS]
  ```
- `meta_labeler.py:775` (in `train`):
  ```python
  leaked = _KNOWN_LABEL_COLUMNS.intersection(available)
  if leaked:
      logger.warning("Dropping label columns from training features: %s", sorted(leaked))
      available = [f for f in available if f not in _KNOWN_LABEL_COLUMNS]
  ```
- `_KNOWN_LABEL_COLUMNS` definition (line 281-286) includes `hlabel`, `hlabel_meta`, `hlabel_L1..L3`, `hlabel_reason`, `y`, `y_horizon`, `label`, `pnl_pct`, `net_pnl_pct`, `sl_hit`, `realized_net_pnl_pct`, `realized_pnl_per_margin_hour`, `rotation_score`, `unrealized_fraction`. Matches CLAUDE.md.

Both guard sites are **present and operative**. Both call `_KNOWN_LABEL_COLUMNS.intersection(...)` and remove matches; the warning is logged but training/inference is not aborted — the guard quietly drops rather than fails-closed. Acceptable per the existing safety-invariants wording ("Both guard sites must remain — removing either re-introduces leakage"). The guard does not need to abort because the test suite (`tests/unit/test_afml_compliance_fixes.py:169-176, 559-563`) verifies the set membership directly.

### 4.4 Backtest entry-point invariant

Direct importers of `RealisticGridBacktester` outside tests: 0 production code files. Excerpts:
- `backtest/backtest_realistic.py:1044-1059` (CLI module): re-routes through `run_backtest()`.
- `backtest/btk_unified_runner.py:36`: imports `RealisticGridBacktester` to call it once at `run_backtest()` body line 166. This is the SOLE allowed call site.
- `backtest_candidates.py`, `backtest_candidates_current.py`, `src/neutralgrid/backtest/candidate_pipeline.py`, `src/neutralgrid/grid/equity_circuit_breaker.py`: docstring/comment references only (verified by `Grep -C 3`).
- 16 test files import directly — explicitly allowed by safety-invariants.md.

### 4.5 Hurdle-pct invariant (`HierarchicalLabelConfig.hurdle_pct == BarrierConfig.meta_hurdle_pct`)

`core/config.py:616-622` (verbatim):
```python
if self.hierarchical_label.hurdle_pct != self.barrier.meta_hurdle_pct:
    raise ValueError(
        f"Config drift: HierarchicalLabelConfig.hurdle_pct "
        f"({self.hierarchical_label.hurdle_pct}) != "
        f"BarrierConfig.meta_hurdle_pct ({self.barrier.meta_hurdle_pct}). "
        f"These must be synchronized to prevent label/deployment divergence."
    )
```
Defaults: both 3.0 (`config.py:96`, `config.py:338`). Enforced at `Config._validate()` startup.

---

## 5. Open questions

1. **AUDIT-OQ-1**: are there decision-time callers of `compute_governed_provisional_utility` that should not catch `UtilityCalibratorUnavailable`? (Static enumeration suggests no, but the audit was bounded.)
2. **AUDIT-OQ-2**: does the deployed meta-labeler artifact ship a fitted imputer, or does it fall back to `_FEATURE_MEDIAN_DEFAULTS`? The fallback masks `range_prob` missingness with 0.5 — relevant for Stage B downstream.
3. **AUDIT-004 follow-up**: was `survival_min = 0.0` hard-coded intentionally because the Stage B micro-osc Gate 4 owns containment, or is the `RegimeValidator` stochastic survival check meant to be active? `core/config.py StochasticConfig` has no `survival_min` field, so making this configurable would require a new attribute. Confirm with the original author before any change.
4. The `_PostProbabilityCalibrator._apply` (AUDIT-006) silent passthrough behaviour — was that intentional (defensive) or an oversight? A test that loads a model with `method="unknown"` and checks for an exception would settle this.
5. Whether the `mode` string column (AUDIT-005) is ever exposed to a future `MetaLabelerConfig.features` list outside `ACTIVE_SNAPSHOT_META_FEATURES`. Would benefit from an assertion in `_prepare_features` that `feats ∩ {string-typed columns} == ∅`.

---

## 6. Suggested next steps (read-only — not code changes)

1. Run `python -m pytest tests/unit/test_afml_compliance_fixes.py -v` to confirm `_KNOWN_LABEL_COLUMNS` test still passes after the audit window. (Static check passed; runtime should match.)
2. Inspect `artifacts/utility/current.json` presence and validate that `RegimeValidator.check_hmm_regime` does not silently log "utility calibrator unavailable" in recent logs — would confirm AUDIT-OQ-1's offline propagation contract is functioning.
3. Add a one-shot pyright inline-suppression review only after the user confirms they want the noise-reduction work; do not start it autonomously.
4. Add a regression test that asserts `core.constants.LABEL_CONTRACT_VERSION == backtest.btk_label_contract.LABEL_CONTRACT_VERSION` to close AUDIT-003's drift surface (proposed by user, not by me).
5. Consider whether `RegimeValidator.check_stochastic_regime` should consume a `core/config.StochasticConfig.survival_min` field rather than hard-coding 0.0 — but this requires user authorization (config schema change is a behaviour change).
6. AUDIT-008: if/when a Stage B utility gate is added, ensure rejection code = `data_missing:utility` per the spec.

---

## 7. Files of greatest relevance (absolute paths)

- `C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7\.claude\worktrees\add-reviewer\.pyright_audit.log`
- `C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7\.claude\worktrees\add-reviewer\src\neutralgrid\calibration\hmm_winner_calibrator.py`
- `C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7\.claude\worktrees\add-reviewer\src\neutralgrid\grid\spacing_profile.py`
- `C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7\.claude\worktrees\add-reviewer\src\neutralgrid\models\meta_labeler.py`
- `C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7\.claude\worktrees\add-reviewer\src\neutralgrid\scanner\two_stage_selector.py`
- `C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7\.claude\worktrees\add-reviewer\src\neutralgrid\validation\regime_validator.py`
- `C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7\.claude\worktrees\add-reviewer\src\neutralgrid\validation\utility.py`
- `C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7\.claude\worktrees\add-reviewer\src\neutralgrid\backtest\candidate_pipeline.py`
- `C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7\.claude\worktrees\add-reviewer\src\neutralgrid\training\unified_training_builder.py`
- `C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7\.claude\worktrees\add-reviewer\src\neutralgrid\training\data_generator.py`
- `C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7\.claude\worktrees\add-reviewer\src\neutralgrid\core\constants.py`
- `C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7\.claude\worktrees\add-reviewer\src\neutralgrid\core\config.py`
- `C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7\.claude\worktrees\add-reviewer\backtest\btk_unified_runner.py`
- `C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7\.claude\worktrees\add-reviewer\backtest\btk_label_contract.py`
- `C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7\.claude\worktrees\add-reviewer\scripts\backfill_training_features.py`

---

# AUDIT_02 — 2026-05-18 follow-up audit

**Date:** 2026-05-18 (13 days after AUDIT_01).
**Scope:** Re-verify AUDIT_01 findings against current HEAD; audit the ~10.8 kLOC of new code added since 2026-05-05 (live decision tool, reconciliation script, training builder changes, version bumps).
**Method:** Four-agent panel (data-curator, feature-analyst, deployment-engineering, backtest-evaluator) on team `audit-02-review`, each tasked with a non-overlapping mandate requiring file:line evidence and severity tags. All read-only; no source edits.
**Baseline state at audit start:** pyright 104 errors / 0 warnings (identical totals to AUDIT_01); active HMM `rolling_180d_20260516_220545`; meta-labeler artifact `20260517_181209` with lineage matching active HMM; HMM winner calibrator `hmm_winner_20260517_181910_112991` (promoted this session).

## A02-1. Executive summary

- **No CRITICAL findings.** Leakage guards intact, no fail-closed gate silently relaxed, no HMM promotion bypass.
- **5 HIGH findings**, of which 2 are AUDIT_01 items that **worsened** and 3 are **net-new** in the post-2026-05-05 code surface.
- **Multi-agent convergence on ERR-048**: three of four agents independently flagged the same dynamic — the live recommender's `meta_proba` consumption is reading a model whose **importance-weighted 56% of MDI is being imputed at every tick**, while the fail-closed switch is count-weighted (3/7 = 43% missing, below the 50% threshold). The "secondary signal" doctrine that originally justified Option A on ERR-048 no longer holds at this MDI distribution.
- **pyright totals unchanged**. No new files contribute errors despite 10.8 kLOC added — credit to the new code's typing discipline.

## A02-2. Severity-ranked findings

| # | Severity | Finding | State vs AUDIT_01 |
|---|---|---|---|
| A02-F01 | HIGH | `backtest\btk_label_contract.py:32-33` `except ImportError` fallback hardcodes `LABEL_CONTRACT_VERSION="2026-04-17"` / `FORMULA_VERSION="alignment-v1"` while `src\neutralgrid\core\constants.py:18,23,28` is now `"2026-05-09"` / `"alignment-v2-geometric-realism"` / `"realistic-v8"`. AUDIT-003 widened from "in sync" to **active drift**. | WIDENED |
| A02-F02 | HIGH | `tests\unit\test_plan_v6_steps.py:150-152` asserts the OLD literals `LABEL_CONTRACT_VERSION == "2026-04-17"` and `FORMULA_VERSION == "alignment-v1"` and is **currently failing** pytest collection (`AssertionError: assert '2026-05-09' == '2026-04-17'`). Same forbidden-duplicate pattern flagged by `safety-invariants.md §Version Constants`. Shares root with A02-F01. | NEW |
| A02-F03 | HIGH | `scripts\validate_backtest_live_reconciliation.py:28,606,641` (new, 941 LOC) imports `RealisticGridBacktester` directly outside `tests/` — violates `safety-invariants.md §Backtest Entry Point`. `_run_model_trade_metrics` (line 641) runs trade-cadence analysis under a config path that did NOT go through `build_training_config()`'s safety knobs (continuous funding, taker fees, 2-bar delay). | NEW |
| A02-F04 | HIGH | **AUDIT-004 still present.** `src\neutralgrid\validation\regime_validator.py:814` still hard-codes `survival_min = 0.0`; `src\neutralgrid\core\config.py:213-222` `StochasticConfig` still exposes no `survival_min` field. MC containment gate in the regime validator remains degenerate. Stage B Gate 4 micro_osc branch uses a different field (`MicroOscConfig.min_survival_prob`), so the user-tunable survival floor lives there — the regime_validator stochastic path is untunable. | STILL PRESENT (13 days) |
| A02-F05 | HIGH | **HMM lineage not stamped on live decision outputs.** `BotEvaluation` (`src\neutralgrid\live\decision\recommender.py:96-148`) carries regime probs and meta_proba but no `hmm_artifact_version`. `monitor.evaluate_bot` (`monitor.py:263-267, 338-360`) consumes `context.hmm` without recording the artifact id on emitted JSONL. Live decisions cannot be tied to an HMM lineage during reconciliation — UTILFIX-01 lineage-authority intent does not propagate into the decision-time audit trail. | NEW |
| A02-F06 | HIGH | **ERR-048 severity reassessment** — convergent verdict from three agents. Today's active artifact MDI mass on imputed features = 0.5628 (ou_halflife 0.2533 + profit_per_grid_pct 0.2138 + survival_prob 0.0957); computed-live mass = 0.4372. The `meta_overlay_inactive` gate (`monitor.py:457-460`) checks **count** (3/7 ~ 43% missing, below 50% threshold), not **importance-weighted** missing share (56%). Gate is mis-specified for the current model. ERR-044 (uncalibrated recommender thresholds) compounds. | SEVERITY UPGRADE |
| A02-F07 | MEDIUM | **AUDIT-005 still present.** `mode` (string) carried through `EXTRA_META_FEATURES` (`unified_training_builder.py:87`), `TRAINING_OUTPUT_COLUMNS` (`candidate_pipeline.py:158`), `FeatureSnapshot.to_dict()` (`data_generator.py:152,234`). Filtered out only via `ACTIVE_SNAPSHOT_META_FEATURES`; no explicit drop in `_prepare_features`. | STILL PRESENT |
| A02-F08 | MEDIUM | **AUDIT-007 still present.** `regime_conf` / `micro_round_trip_cost_pct` / `long_short_ratio` plumbed only via hard-coded extension at `candidate_pipeline.py:1076-1078`, not in `TRAINING_OUTPUT_COLUMNS`. | STILL PRESENT |
| A02-F09 | MEDIUM | **AUDIT-OQ-2 now operationally relevant.** `_FEATURE_MEDIAN_DEFAULTS` (`meta_labeler.py:268-279`) still maps `range_prob -> 0.5`, `trend_prob -> 0.25`, `survival_prob -> 0.5`. Coupled to A02-F06: the imputed features dominate MDI, so the median defaults are now load-bearing in live inference. | STILL PRESENT (now load-bearing) |
| A02-F10 | MEDIUM | **`_coerce_deploy_ts` silently treats naive timestamps as UTC** at `src\neutralgrid\live\decision\loader.py:442-445`. New occurrence of the ERR-043 defect class (manual-ingest +5h offset) in the live YAML loader — separate code path, same silent-coercion pattern. | NEW location |
| A02-F11 | LOW | **AUDIT-006 still present.** `_PostProbabilityCalibrator._apply` (`meta_labeler.py:411-426`) silently returns raw probability for unrecognized `method`. Benign today (active artifact `is_calibrated: false`) but unchanged. | STILL PRESENT |
| A02-F12 | LOW | **AUDIT-009 still present.** `TRAINING_FEATURES` reassigned at `candidate_pipeline.py:162` (= `TRAINING_OUTPUT_COLUMNS`) AND `unified_training_builder.py:52` (= `list(DEFAULT_SCHEMA.features)`) — two unrelated value sets, one identifier. Line numbers shifted from AUDIT-009's 166/48; logic unchanged. | STILL PRESENT |
| A02-F13 | LOW | New ingestion gates `_apply_ingestion_gates` (`unified_training_builder.py:1216-1339`) are mark-and-continue, not fail-closed. Bootstrap-relaxation branch (lines 458-472) now filters by contract reason, but a future contract column added at the wrong location could silently corrupt the relaxed pool. | NEW |
| A02-F14 | INFO | **AUDIT-008 unchanged.** Stage B still has no `data_missing:utility` rejection code — matches `safety-invariants.md` "future" framing. | STILL PRESENT (designed) |

## A02-3. Resolved since AUDIT_01

- **AUDIT-OQ-1** — utility fallback propagation. All four callers of `compute_governed_provisional_utility` catch `UtilityCalibratorUnavailable`: `regime_validator.py:386-411`, `scanner/scan.py:251-270`, `unified_training_builder.py:766-784`, and `utility.py:450` (definition). Closed by enumeration.

## A02-4. Invariants verified intact (high confidence)

- **Leakage guards** — `_KNOWN_LABEL_COLUMNS` definition at `meta_labeler.py:324-329` (includes `hlabel`); enforcement at `_prepare_features` line 643 and `train` line 818 (line numbers shifted from AUDIT_01's 600/775 due to new code, logic unchanged).
- **Hierarchical hurdle cross-validation** — `core/config.py:615-622` still raises on drift.
- **Stage B 4-gate fail-closed structure** — `two_stage_selector.py:124-195`. All 4 gates emit `data_missing:*` codes; archetype branching at 156-167 (micro_osc) / 168-192 (standard) intact.
- **HMM rotation chain** — regex `^rolling_180d_\d{8}_\d{6}$` at `artifacts.py:34`; `MIN_PASS_RATE = 0.50` enforced at `retrain_orchestration.py:21,122-125`; not silently relaxed.
- **Backtest entry-point invariant** (excluding A02-F03) — production callers in `candidate_pipeline.py`, `equity_circuit_breaker.py`, `backtest_candidates.py`, `backtest_candidates_current.py` are docstring/comment mentions only; the only production-grade violation is the new `validate_backtest_live_reconciliation.py`.
- **Live decision Binance error handling** — fail-closed at `monitor.py:197-210` (4xx -> `symbol_unavailable`, 5xx/timeout/OSError -> `transient_fetch_error`).
- **State-store atomic writes** — `state_store.py:153-186` (`tempfile.mkstemp` + `fsync` + `os.replace`).
- **No new credential-exposure paths** in live decision code.
- **Loader datetime handling** — tz-aware at `loader.py:228,434-447,673` (excluding the `_coerce_deploy_ts` case in A02-F10).
- **Imputation warning path** — `meta_labeler.py:1579-1586` `n_missing > 2` warning still fires (today's n=3 per ERR-048 evidence).
- **No new feature added since AUDIT_01** has broken the Feature Pipeline Update Rule — only an alias removal (`scan_utility_score` / `scan_regime_utility` from `_SCAN_TO_FEATURE`), resolved downstream via `_INFERENCE_FEATURE_ALIASES`.

## A02-5. ERR entries filed for HIGH findings

Per user direction 2026-05-18:

- **ERR-049** (this session) — A02-F01 + A02-F02 combined root (btk_label_contract.py fallback drift + test_plan_v6_steps.py asserting stale literals).
- **ERR-050** (this session) — A02-F05 (HMM lineage not stamped on live decision outputs).
- **ERR-051** (this session) — A02-F04 (AUDIT-004 survival_min hard-coded 0.0, 13 days open since AUDIT_01).

A02-F03 and A02-F06 were not assigned new ERR IDs at user direction (A02-F03 has clear remediation path without ERR tracking; A02-F06 is folded into existing ERR-048 as a severity reassessment).

## A02-6. Recommended remediation order

1. **A02-F02** first — test suite is currently red. Replace literals with imports from `core.constants`.
2. **A02-F01** — collapse `btk_label_contract.py:32-33` fallback to single-source import or hard-fail on missing package.
3. **A02-F03** — route `validate_backtest_live_reconciliation.py` through `run_backtest()` or open documented exemption in `safety-invariants.md`.
4. **A02-F05** — add `hmm_artifact_version` (and `meta_artifact_version`) to `BotEvaluation` + emit on JSONL.
5. **A02-F06** — ERR-048 needs operator decision. Importance-weighted gate (instead of count) is a low-cost interim fix for the `meta_overlay_inactive` switch; the larger Option B/C/D choice remains open.
6. **A02-F04** — open 13 days; add `survival_min` field to `StochasticConfig`, thread through `regime_validator._validate_stochastic_regime`.
7. A02-F07 through A02-F13 — backlog, not blockers.

## A02-7. Process notes

- TaskCreate-then-TeamCreate ordering orphans pre-team tasks into the session task list; agents joining the team see the team's (empty) task list and cannot find the pre-team task IDs. For future audits: **TeamCreate first, then TaskCreate**.
- All four agents shut down cleanly via `shutdown_request`; team `audit-02-review` directories and worktrees deleted post-shutdown.
- No source files were edited during the audit.

---

# AUDIT_03 — 2026-05-19 Updated remediation priority order

**Date:** 2026-05-19 (validated against current code; second-AI reviewer cross-check incorporated).
**Method:** direct file reads + live `pytest` execution for failure confirmation. No source edits.
**Resolution recorded:** P0 fixes the test FIXTURES (treats `_apply_ingestion_gate()` behavior as correct). Decision per operator 2026-05-19.

## Progress

`[####------] 39% (7/18 fix items complete)`

Update this bar as items are marked DONE below. Counter = items marked DONE in the priority list / 18 total fix items (P0 through P4, excluding P4 #18 which is "no-action by design").

## P0 — Make baseline green (one PR, "tests reflect current contract")

**Closure proof:** `python -m pytest tests/ -k contract` returns 0 failures; `python -m pytest tests/unit/test_plan_v6_steps.py -v` returns all green.

1. **A02-F02** — `tests/unit/test_plan_v6_steps.py:144-152` (`test_constants_module_exists`). Replace literal `"2026-04-17"` / `"alignment-v1"` / `"realistic-v7"` assertions. The test already imports the constants at lines 145-148; delete the literal assertions or rebind to imported names. Better: delete and rely on `test_btk_label_contract_uses_same_versions` / `test_btk_unified_runner_uses_same_engine_version` at lines 154-169 to enforce consistency.  [STATUS: PENDING]  [Re-validated 2026-05-22: line range is 144-152 (was cited as 150-152); confirmed FAILING — `assert '2026-05-09' == '2026-04-17'`.]

2. **A02-F01** — `backtest/btk_label_contract.py:31-34`. Collapse the `except ImportError` fallback. Either re-import via path-bootstrap, or hard-fail on missing package. No more local literals.  [STATUS: PENDING]  [Re-validated 2026-05-22: fallback block is lines 31-34 (was cited as 32-33). It hardcodes THREE literals — `LABEL_CONTRACT_VERSION="2026-04-17"`, `FORMULA_VERSION="alignment-v1"`, and `BOT_HORIZON_HOURS=6.0`. NOTE: `ENGINE_VERSION` is NOT in this fallback. The first two drift from constants (`"2026-05-09"` / `"alignment-v2-geometric-realism"`); `BOT_HORIZON_HOURS=6.0` still matches and is not drifting.]

3. **`test_matching_version_not_gated`** — `tests/unit/test_plan_v6_steps.py:175-193`. Extend fixture with the 7 fields the gate now checks: `engine_version=ENGINE_VERSION`, `formula_version=FORMULA_VERSION`, `mode="geometric"`, `fill_mode="wick"`, `global_cooldown_bars=0.0`, `cb_enabled=False`, `is_authoritative=True`. Test name says "matching version not gated" — to assert that, row must satisfy every gate the function applies.  [STATUS: PENDING]

4. **`test_missing_version_marked_legacy`** — `tests/unit/test_plan_v6_steps.py:215-234`. Same fixture extension: supply the 7 additional fields with passing values. Test isolates the missing-`label_contract_version` path; the row must pass every other gate so the only failure mode under test is the missing version.  [STATUS: PENDING — BLOCKED on E7]  [Re-validated 2026-05-22: the prior prescription was WRONG. `_apply_ingestion_gate` sets `source_class="legacy"` at line 1274 BUT THEN re-enters `_mark_gated(no_version, "missing_label_contract_version")` at line 1291, which OVERWRITES `source_class` to `"non_authoritative"` (line 1242). So the legacy marker is NOT preserved; the test fails `assert 'non_authoritative' == 'legacy'`. This is a code/contract contradiction (docstring at 1225-1226 says missing-version rows are `legacy`, code yields `non_authoritative`). Fixing the FIXTURE alone will not make this test green — the legacy-vs-non_authoritative vocabulary question (E7) must be settled first: either the code clobber is a bug (fix code, keep test) or the contract intentionally changed (update docstring + test to expect `non_authoritative`).]

5. **`test_current_workbook_contract_counts_if_available`** — `tests/unit/test_utility_calibrator.py:401-413`. Workbook is a moving target by design. **E1 SETTLED 2026-05-22: use ALL available samples** — replace the hardcoded magic counts (`len(pool)==85`, winners==57, losers==28, fit==64, holdout==21) with structural/dynamic assertions derived from the pool itself (winners+losers==len(pool); fit+holdout==len(pool); pool non-empty; label binary). Utility calibrator not currently in use; correctness of code/test is the goal.  [STATUS: PENDING — E1 settled, ready to implement]

6. **4 bounded-universe failures in `tests/test_afml_integrations.py`** — reviewer reported; not independently verified in this audit. Re-run `pytest tests/test_afml_integrations.py -v` to enumerate. Treat each as P0 fixture or count-shift fix. If any are real safety-invariant breaches rather than fixture drift, escalate to P1.  [STATUS: PENDING — enumerate first]

## P1 — Safety-invariant violations (one PR each, in this order)

7. **A02-F03** — `scripts/validate_backtest_live_reconciliation.py:28,864,899`. Direct `RealisticGridBacktester` instantiation outside `tests/` breaches `safety-invariants.md §Backtest Entry Point`. Route the two call sites through `backtest.btk_unified_runner.run_backtest()` so `build_training_config()` safety knobs apply, OR add an explicit documented exemption in `safety-invariants.md` with rationale. **Operator confirms path before code change** — routing is the safer default; exemption is faster but enlarges the bypass surface.  [STATUS: PENDING — awaiting operator election; see Open Elections below]

8. **A02-F05** — ERR-050. Add `hmm_artifact_version`, `meta_artifact_version`, `calibrator_artifact_version` to `BotEvaluation` (`src/neutralgrid/live/decision/recommender.py:96-144`). Stamp into the JSONL emission. Additive change; no model logic touched. High audit value.  [STATUS: PENDING]  [Re-validated 2026-05-22: the JSONL writer is `src/neutralgrid/live/decision/renderer.py::_to_jsonl_record` (lines 169-211), NOT `monitor.py:263-360` (that range is the `BotEvaluation` construction site). The fix must stamp the version fields in `renderer.py:_to_jsonl_record`. Today the record stamps only `contract_version` (174) and `hmm_artifact_missing` (207); no `*_artifact_version` token exists anywhere in monitor.py/recommender.py/renderer.py.]

9. **A02-F04** — ERR-051. **E4 SETTLED 2026-05-22: option (a) — inert by design; DELETE the dead `survival_ok` check** rather than adding a config field. Stage B micro_osc Gate 4 owns containment. Behavior-preserving (check was always True). Remove the inert `survival_min=0.0` plumbing + dead `survival_ok` reason/metrics branch in `regime_validator.py:814,892-893,876,871` and the two parallel hardcodes at `scanner/enrich_grid_params.py:295` and `scanner/scan.py:224` to the extent they only fed the dead check.  [STATUS: PENDING — E4 settled, ready to implement]

## P2 — Live-model risk (operator decisions)

10. **A02-F06 + A02-F09 (coupled)** — ERR-048. MDI-weighted 56% imputed-feature mass makes `_FEATURE_MEDIAN_DEFAULTS` load-bearing. Two-track decision:
    - **Interim:** switch `meta_overlay_inactive` (`monitor.py:457-460`) from count-weighted to MDI-weighted threshold. Disables overlay for the current artifact; lossless when overlay is sound.
    - **Strategic:** ERR-048 Option B/C/D (retrain on full features vs. accept reduced model vs. change which features are missing-imputable). Operator decision required.
    [STATUS: PENDING — awaiting operator election; see Open Elections below]

11. **A02-F10** — user's baseline-policy approach (`time_source` discriminator + config-pinned operator timezone + recorded `time_provenance`). Three preconditions before implementation:
    - Source-discriminator field added to row schema (`time_source ∈ {manual_ingest_user_local, live_yaml_user_local, extractor_utc, binance_api_utc}`).
    - `OPERATOR_TIMEZONE` (or `OPERATOR_UTC_OFFSET_HOURS`) pinned in `.env` / `core/config.py`; no scattered `-5` literals.
    - Every converted row carries `time_provenance` column; double-conversion guarded by idempotent check.

    Supersedes UTC_FIX.md Steps 2-3 policy-selection framing (no policy choice at load time once discriminator + config exist). Also closes the same ERR-043 defect class on the extractor side if extractor stamps `time_source: "user_local"` at write time.  [STATUS: PENDING — awaiting operator election on OPERATOR_TIMEZONE identifier; see Open Elections below]

## P3 — Backlog (code quality / latent risk)

12. **A02-F07** (AUDIT-005) — add explicit drop in `_prepare_features` of any string-typed columns; harden the `mode` invariant beyond the implicit `ACTIVE_SNAPSHOT_META_FEATURES` filter.  [STATUS: PENDING]

13. **A02-F08** (AUDIT-007) — move `micro_round_trip_cost_pct` / `long_short_ratio` into `TRAINING_OUTPUT_COLUMNS`, OR update `safety-invariants.md §Feature Pipeline Update Rule` to name the hidden hard-coded extension at `candidate_pipeline.py:1076-1078` as a fourth required site.  [STATUS: PENDING]  [Re-validated 2026-05-22: ONLY `micro_round_trip_cost_pct` and `long_short_ratio` are in the hard-coded extension at candidate_pipeline.py:1076-1078. `regime_conf` is NOT plumbed there — it appears only in `_FEATURE_MEDIAN_DEFAULTS` (meta_labeler.py:292). The prior text overstated the scope by including `regime_conf`.]

14. **A02-F11** (AUDIT-006) — `_PostProbabilityCalibrator._apply` (`meta_labeler.py:411-426`): add default-case `raise` for unknown `method`. Benign today (active artifact `is_calibrated: false`); harden before next calibration.  [STATUS: PENDING]

15. **A02-F12** (AUDIT-009) — rename `TRAINING_FEATURES` in one of the two modules. Suggest `unified_training_builder.TRAINING_FEATURES` -> `DEFAULT_SCHEMA_FEATURES` since the value is `list(DEFAULT_SCHEMA.features)`. Touch `tests/unit/test_unified_training_builder.py:19` to follow.  [STATUS: PENDING]

16. **A02-F13** — `_apply_ingestion_gates` mark-and-continue posture. Decide whether ingestion gates should be fail-closed (parallel to Stage B). Currently mark-only; future contract column added at the wrong location could silently corrupt a relaxed pool.  [STATUS: PENDING]

17. **AUDIT-001 / AUDIT-002** — pyright pandas-narrowing noise (`spacing_profile.py` 56 errors, `hmm_winner_calibrator.py` 48 errors). Apply `cast(pd.Series, ...)` / `.iloc[0]` per CLAUDE.md `pandas Pyright Patterns`. Pure noise reduction; **do not bundle** with any other fix in this list — pyright noise cleanup is its own PR.  [STATUS: PENDING]

## P4 — Designed / no-action

18. **A02-F14** (AUDIT-008) — Future Stage B utility gate. Matches CLAUDE.md "Future" wording. When/if added, rejection code MUST be `data_missing:utility`. NO ACTION REQUIRED.  [STATUS: BY DESIGN — excluded from progress counter]

## Open elections (operator must settle before P0c / P1 / P2 PRs land)

The following are NOT engineering choices; they require operator authority because they change either contracts, defaults, or scope.

| # | Election | Blocks | Options |
|---|---|---|---|
| E1 | **Workbook count test** (P0 #5) — how to handle the moving-target workbook size in `test_current_workbook_contract_counts_if_available`. | P0 close | (a) range assertion `80 <= len(pool) <= 120`; (b) parametrized snapshot recording the canonical count + date; (c) `pytest.skip` with TODO referencing how the canonical count should be sourced. |
| | **SETTLED 2026-05-22 — use ALL available samples in the pool.** Replace the hardcoded magic counts with structural/dynamic assertions derived from the pool itself (e.g. winners + losers == len(pool); len(fit_df) + len(holdout_df) == len(pool); pool non-empty; label column binary). Operator note: the utility calibrator is NOT currently in use and will remain so for the foreseeable future, so the count is operationally unimportant — but the code/test must be correct. | | |
| E2 | **`test_afml_integrations.py` triage** (P0 #6) — once the 4 failures are enumerated, decide if any are real invariant breaches (escalate to P1) or all are fixture drift (resolve in P0). | P0 close | (a) all P0 fixture fixes; (b) mixed P0/P1 escalation per finding. |
| E3 | **A02-F03 path** (P1 #7) — route `validate_backtest_live_reconciliation.py` through `run_backtest()` (safer) or carve a documented exemption in `safety-invariants.md` (faster but enlarges bypass surface). | P1 #7 | (a) route through `run_backtest()`; (b) documented exemption. |
| E4 | **A02-F04 author intent** (P1 #9) — was `survival_min=0.0` intentionally inert because Stage B micro_osc Gate 4 owns containment? | P1 #9 | (a) inert by design -> fix is to DELETE the dead `survival_ok` check; (b) oversight -> fix is to ADD `survival_min` to `StochasticConfig` with default 0.0 (behavior-preserving) and thread it through. |
| | **SETTLED 2026-05-22 — option (a): DELETE the dead `survival_ok` check.** Behavior-preserving (the check was always True since `survival_prob >= 0.0` always holds). Remove the inert `survival_min=0.0` plumbing in `regime_validator.py` (and the dead `survival_ok` reason branch / metrics key) and the two parallel hardcodes at `scanner/enrich_grid_params.py:295` and `scanner/scan.py:224` to the extent they only fed the dead check. Containment is owned by Stage B micro_osc Gate 4. | | |
| E7 | **`source_class` legacy vs non_authoritative** (P0 #3/#4) — code sets `source_class="legacy"` (1274) then `_mark_gated` clobbers it to `"non_authoritative"` (1242 via 1291). Docstring (1225-1226) and test expect `legacy`. | P0 close | (a) clobber is a code bug -> fix code to preserve `legacy`; (b) contract intentionally collapsed `legacy` into `non_authoritative` -> update docstring + test. |
| | **SETTLED 2026-05-22 — option (a): treat the clobber as a code bug and preserve `source_class="legacy"` for missing `label_contract_version` rows.** Fix `_apply_ingestion_gate` so the missing-version legacy class is not overwritten by `_mark_gated`'s `non_authoritative` assignment; keep `test_missing_version_marked_legacy` asserting `legacy`. | | |
| E5 | **ERR-048 / A02-F06 track** (P2 #10) — interim only, or commit to a strategic track now? | P2 #10 | (a) interim only: MDI-weighted `meta_overlay_inactive` switch, defer B/C/D; (b) Option B: retrain on full features; (c) Option C: accept reduced model; (d) Option D: change which features are missing-imputable. |
| E6 | **A02-F10 operator timezone identifier** (P2 #11) — what value gets pinned for `OPERATOR_TIMEZONE` / `OPERATOR_UTC_OFFSET_HOURS`? Must be a non-DST UTC-5 zone per ERR-043 evidence. | P2 #11 | (a) `OPERATOR_TIMEZONE="America/Lima"`; (b) `"America/Bogota"`; (c) `"America/Guayaquil"`; (d) `"America/Panama"`; (e) numeric offset `OPERATOR_UTC_OFFSET_HOURS=-5` (timezone-agnostic, DST-unaware — only safe if user confirms they will NEVER operate from a DST zone). |
| | **SETTLED 2026-05-22 — option (a): `OPERATOR_TIMEZONE="America/Lima"`.** Operator confirmed Lima is the operator time zone. (Lima is non-DST UTC-5, consistent with the ERR-043 constraint.) | | |

## DONE markers protocol

When a fix item lands:
1. Replace its `[STATUS: PENDING]` (or `[STATUS: PENDING — awaiting operator election; see Open Elections below]`) with `[STATUS: DONE — <YYYY-MM-DD> — <commit-hash-short> — <one-line proof: e.g., "pytest green; pyright clean">]`.
2. Increment the progress bar at the top of AUDIT_03. Bar uses 10 cells; each cell = 10% (1.8 items). Round to nearest cell.
3. For elections E1-E6: when an election is settled, append a single line under the option row noting the date and resolution. Do not delete unresolved options — the audit trail is the value.
4. If a fix is reverted or superseded, change `DONE` to `REVERTED — <date> — <reason>` and decrement the counter. Do not delete the row.

## Closing rule

AUDIT_03 closes (entire section marked CLOSED at the top) only when:
- All 17 actionable items (P0 through P3) are DONE, and
- All 6 open elections are settled, and
- `pytest tests/ -k contract` returns 0 failures, and
- `pyright` is at or below the AUDIT_01 baseline (104 errors, 0 warnings) — net new errors from any AUDIT_03 fix must be cleaned up before close.

Item P4 #18 (A02-F14) does not block close — it is BY DESIGN per CLAUDE.md.

---

## Implementation Log — 2026-05-22 (session AUDIT-04-FIX)

All changes verified read-evidence-driven; no assumptions. Gating: pyright (basic, src) + focused pytest after each phase. Full pyright held at the 104-error baseline (zero new errors). All edits proven independent of pre-existing failures via a stash-revert re-run.

**DONE (7/18):**
- **P0-1 (A02-F02)** — `tests/unit/test_plan_v6_steps.py::test_constants_module_exists` now asserts str/non-empty instead of frozen literals. Proof: test green.
- **P0-2 (A02-F01)** — `backtest/btk_label_contract.py` `except ImportError` fallback now bootstraps `src/` onto sys.path and re-imports from `core.constants` (no local version literals; hard-fails if package truly absent). Verified primary import resolves LCV=2026-05-09. Proof: `test_btk_label_contract_uses_same_versions` green.
- **P0-3** — `test_matching_version_not_gated` fixture extended with the 7 engine-settings gate fields. Proof: green.
- **P0-4 + E7** — `_apply_ingestion_gate._mark_gated` no longer downgrades `source_class="legacy"` to `non_authoritative` (origin class preserved; gating still by `is_authoritative`/`version_gated`, unchanged). `test_missing_version_marked_legacy` fixture extended. Proof: green.
- **P0-5 (E1)** — `test_current_workbook_contract_counts_if_available` now uses structural assertions over ALL samples (winners+losers==len(pool); fit+holdout==len(pool); both non-empty) instead of magic counts. Proof: `test_utility_calibrator.py` 20 passed.
- **P0-6 (E2)** — root cause was FEATURE-SET drift, not fixture size: `pattern_profile.DEFAULT_FEATURES` (also used by `profile_model`) changed to 4 microstructure/pre-event features; the synthetic fixture still emitted obsolete columns, so the availability filter dropped all features. Fixed `_make_test_xlsx` to emit the 4 current features + corrected stale "13 features" comment. The availability filter / per-class floor is correct production behavior (no escalation). Proof: `test_afml_integrations.py` 23 passed.
- **P1-9 (E4 option a)** — DELETED the inert `survival_ok` check (always True; survival_min was always 0.0). Removed `survival_min` config field + `survival_ok` result field + `all_passed` term + reason branch + metrics keys in `stochastic.py`/`regime_validator.py`, and the parallel hardcodes in `scanner/enrich_grid_params.py` and `scanner/scan.py`; updated `test_hurst_quality.py`. `survival_prob` is still computed/reported. Behavior-preserving. Proof: 78 stochastic/regime/enrich/afml-bugfix tests passed.

**New items surfaced this session:**
- **Grid Mode Authority** (operator-confirmed): arithmetic is valid for backtest but NOT authoritative for training. Added `safety-invariants.md` section + code comment at the gate + new `test_arithmetic_mode_gated` lock test.
- **10 pre-existing full-suite failures beyond the AUDIT_03 catalogue** (proven NOT caused by this session's edits via stash-revert) — triaged by team `audit-04-fix` (verifier + triage-btk + triage-builder), ALL VERDICT A (fixture/test drift, zero real regressions), now RESOLVED:
  - `test_unified_training_builder_v20260312` (2) — backtest rows lacked the 7 engine-settings contract fields; fixtures extended with authoritative `geometric`/`wick` values (NOT arithmetic). FIXED.
  - `test_btk_order_lifecycle` (2) + `test_btk_exchange_rounding` (1) — grid default flipped arithmetic->geometric (commit 04a15b4); hand-computed integer-level scenarios pinned to `mode="arithmetic"`. FIXED.
  - `test_btk_global_cooldown` (1) — `TRAINING_ENGINE_DEFAULTS["global_cooldown_bars"]` 120->0 (commit 04a15b4, deliberate). **Operator-confirmed intentional 2026-05-22**; test updated to expect 0. FIXED.
  - `test_bot_data_extractor_v2` (4) — manual-paste timestamps now interpreted America/Lima -> UTC (+5h, commit d1e5591), consistent with E6=Lima. **Operator approved fixing to Lima +5h 2026-05-22**; expected UTC values updated (forward-compatible with the future A02-F10 config-pinned OPERATOR_TIMEZONE). FIXED.

**Full suite:** `pytest tests/` → **1377 passed, 0 failed** (was 10 failed). `pytest tests/ -k contract` → 74 passed, 0 failed (was 5 failed). pyright → 104 errors, 0 warnings (unchanged baseline).

**Still PENDING (not in this session's scope):** P1-7 (E3, untouched per operator), P1-8 (A02-F05 lineage stamping), P2-10 (E5 meta_overlay, left intact per operator), P2-11 (A02-F10 full timezone impl; E6 value = America/Lima recorded), P3-12/13/14/15/16 backlog, P3-17 pyright noise.
