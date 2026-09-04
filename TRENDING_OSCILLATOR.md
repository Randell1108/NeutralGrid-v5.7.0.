# TRENDING_OSCILLATOR

Status: Revised draft v7 — awaiting user approval
Date: 2026-03-28
Scope: Add a "trending micro-oscillator" archetype to the pipeline so it can detect and approve symbols like ONUSDT.

## Approval Gate

Do not implement until explicit user approval.

## Evidence

ONUSDT (strategy_id 410926826) produced +60.40 USDT (+15.10%) in 5h 54m with 137 matched trades. The pipeline scored it 27.83 (threshold 45), HMM gave range_prob=0.0008, and it was never enriched.

Post-mortem: hurst=0.4934, survival_prob=0.9992, vwap_crosses_5m=36.

## Architectural Diagnosis

The pipeline blocks trending micro-oscillators at five sequential points:

| Gate | Current Logic | ONUSDT Result | Problem |
|------|---------------|---------------|---------|
| Scanner score | `score >= 45.0` | 27.83 | No trending-oscillator archetype in similarity profile |
| Enrichment eligibility | `range_prob >= 0.50` | 0.0008 | HMM is 1H-only; sees trend, not micro-oscillation |
| Pre-enrichment early-reject | `range_prob >= 0.30` (line 317) | 0.0008 | Hardcoded in enrichment loop |
| Regime rejection | `rd.vres.is_valid` (line 1351) | N/A under defaults | `soft_gating=True` makes HMM always pass; not a blocker under current config |
| Stage B Gate 4 | `range_prob >= 0.35-0.45` (line 167) | Never reached | Symbol never enriched |

Plus two infrastructure blockers and one sort-truncation blocker:

| Blocker | Current State | Effect |
|---------|--------------|--------|
| `kline_limits["15m"] = 200` (config.py line 385) | Stochastic requires >= 300 bars (scan.py line 159) | `survival_prob` and `hurst_exponent` never computed |
| `--compute-stochastic` is `action="store_true"` (run_full_pipeline.py line 323) | Defaults to `False`, overrides `STOCHASTIC_VALIDATION_AVAILABLE` fallback | Stochastic computation off even with enough bars |
| Sort-then-truncate (enrich_grid_params.py lines 159-161) | Sorts by `range_prob DESC, score DESC` then `head(max_symbols)` | Low range_prob + low score micro-osc candidates sorted to bottom and dropped |

## Plan: 6 Steps (10 files)

### Step 0: Fix stochastic computation prerequisites

Two changes that unblock `survival_prob` and `hurst_exponent` at scan time.

**0a — File: `src/neutralgrid/core/config.py` line 385**

Change `kline_limits["15m"]` from `200` to `300`.

Why: `scan.py` line 159 requires `len(df_15m) >= 300` for `_compute_stochastic_features()`. With 200 bars, the guard always returns early and `survival_prob` / `hurst_exponent` are never computed. The enrichment bypass requires `survival_prob >= 0.60` in the scan DataFrame — without this fix, the bypass always fails.

Validated: API weight is per-endpoint (5 points per `/fapi/v1/klines` call regardless of bar count), so 200→300 has zero rate-limit impact.

**0b — File: `run_full_pipeline.py` line 323**

Add `default=None` to `--compute-stochastic` argument.

```python
# Current (line 323-326):
parser.add_argument(
    "--compute-stochastic",
    action="store_true",
    help="Compute stochastic features (survival prob, Hurst, OU halflife)",
)

# Fixed:
parser.add_argument(
    "--compute-stochastic",
    action="store_true",
    default=None,
    help="Compute stochastic features (survival prob, Hurst, OU halflife)",
)
```

Why: `action="store_true"` without explicit default sets value to `False` when flag absent. `scan.py` line 318 receives `compute_stochastic=False` and assigns it directly — the `None` fallback to `STOCHASTIC_VALIDATION_AVAILABLE` at line 319-320 never triggers. Setting `default=None` lets the fallback activate. Reference implementation: `scan_top100.py` line 156-161 already uses this pattern.

### Step 1: Compute `micro_osc_score` in scan.py after stochastic features

**File: `src/neutralgrid/scanner/scan.py`**

`feature_extractor.py` runs at line 365-370, BEFORE stochastic features are computed at lines 412-420. `hurst_exponent` is not available inside `feature_extractor.py`. Therefore `micro_osc_score` must be computed in `scan.py` after `_compute_stochastic_features()` returns.

Compute in `scan.py` after line 422 (after stochastic features are assigned to `row`):

```python
_vwap_c = row.get("vwap_crosses_5m")
_ema_c = row.get("ema_crosses_5m")
_hurst = row.get("hurst_exponent")
if _vwap_c is not None and _ema_c is not None:
    _k_vwap, _k_ema = 25.0, 20.0
    _vwap_norm = _vwap_c / (_vwap_c + _k_vwap)
    _ema_norm = _ema_c / (_ema_c + _k_ema)
    if _hurst is not None:
        _hurst_norm = max(0.0, 1.0 - 2.0 * abs(_hurst - 0.5))
        row["micro_osc_score"] = round(0.45 * _vwap_norm + 0.35 * _hurst_norm + 0.20 * _ema_norm, 4)
    else:
        row["micro_osc_score"] = round(0.60 * _vwap_norm + 0.40 * _ema_norm, 4)
```

Always computed regardless of feature flag (training data collection).

Saturation constants are hardcoded inline. They are not exposed as config fields because this is a scoring formula under validation — tuning knobs are premature before threshold calibration is complete.

### Step 2: Add enrichment bypass with full propagation

File: `src/neutralgrid/scanner/enrich_grid_params.py`

Five sub-steps in file order.

**2a — Eligibility gate (line 154):** Add 3rd OR condition.

```python
from neutralgrid.core.config import get_config
_mosc = get_config().micro_osc

if _mosc.enabled and "micro_osc_score" in df.columns and "survival_prob" in df.columns:
    micro_osc_mask = (
        (df["micro_osc_score"].fillna(0) >= _mosc.min_score)
        & (df["survival_prob"].fillna(0) >= _mosc.min_survival_prob)
    )
else:
    micro_osc_mask = pd.Series(False, index=df.index)

eligible = df.loc[score_mask | range_mask | micro_osc_mask].copy()
eligible["micro_osc_bypass"] = micro_osc_mask.reindex(eligible.index).fillna(False)
```

Why: ONUSDT fails both score_mask (27.83 < 45) and range_mask (0.0008 < 0.50). Without a 3rd condition, it is never eligible. The mask requires BOTH `micro_osc_score >= 0.45` AND `survival_prob >= 0.60` — this is a dual-gate, not a single threshold.

**2b — Sort-and-truncate (lines 157-165):** Guarantee bypass candidates survive `head(max_symbols)`.

```python
if "micro_osc_bypass" in eligible.columns:
    eligible = eligible.sort_values(
        by=["micro_osc_bypass", range_prob_col, cfg.score_column],
        ascending=[False, False, False],
    ).head(int(cfg.max_symbols))
else:
    eligible = eligible.sort_values(
        by=[range_prob_col, cfg.score_column], ascending=[False, False]
    ).head(int(cfg.max_symbols))
```

Why: `micro_osc_bypass` is boolean. `False=0, True=1`. Sorting descending puts `True` rows first. Within bypass rows, existing range_prob/score sort applies. Within non-bypass rows, existing sort is unchanged.

**2c — Propagate `micro_osc_bypass` back into `df` (BEFORE line 286):**

Must execute before the threshold tag so that 2d can condition on it.

```python
df["micro_osc_bypass"] = False
if "micro_osc_bypass" in eligible.columns:
    df.loc[eligible.index, "micro_osc_bypass"] = eligible["micro_osc_bypass"]
```

Why: `_enforce_threshold_gate()` runs on `df` at line 1493. The bypass flag must exist on `df` before `below_threshold_tag` is set at line 286.

**2d — Threshold tag (line 286-287):** Exclude bypass rows.

```python
bypass_col = df["micro_osc_bypass"] if "micro_osc_bypass" in df.columns else pd.Series(False, index=df.index)
below_threshold_mask = (
    (df[cfg.score_column] < float(cfg.score_threshold))
    & ~bypass_col.astype(bool)
)
df.loc[below_threshold_mask, "below_threshold_tag"] = True
```

Why: Without this, bypass rows with score < 45 are tagged `below_threshold_tag=True`, and `_enforce_threshold_gate()` at line 1493 sets `grid_is_valid=False` and nullifies all grid parameters.

**2e — Pre-enrichment early-reject (line 317):** Condition range_prob < 0.30 check.

```python
_is_bypass = bool(row.get("micro_osc_bypass")) if "micro_osc_bypass" in eligible.columns else False
if rp is not None and pd.notna(rp) and float(rp) < 0.30 and not _is_bypass:
    pre_reject.add(sym)
```

Why: ONUSDT has range_prob=0.0008 < 0.30. Without this condition, it is pre-rejected at line 318 before any enrichment API calls.

### Step 3: Redefine Gate 4 for Micro-Oscillator Archetype

File: `src/neutralgrid/scanner/two_stage_selector.py`

Gate 4 remains MANDATORY. Its check changes depending on archetype.

Add `micro_osc_score: float = 0.0` and `survival_prob: float = 0.0` to `approve()` signature (line 70).

```python
_mosc = get_config().micro_osc

if (_mosc.enabled
    and micro_osc_score >= _mosc.min_score):
    regime_pass = survival_prob >= _mosc.min_survival_prob
    details["gate4_mode"] = "micro_oscillator_survival"
    details["gate4_survival_prob"] = round(survival_prob, 4)
    details["gate4_micro_osc_score"] = round(micro_osc_score, 4)
    if not regime_pass:
        codes.append(f"micro_osc_survival_below_min({survival_prob:.4f})")
else:
    effective_min_range_prob = cfg.min_range_prob
    if posteriors is not None:
        effective_min_range_prob, _tier = select_range_prob_threshold(posteriors)
    regime_pass = range_prob >= effective_min_range_prob
    details["gate4_mode"] = "hmm_range_prob"

gates["regime_confidence"] = regime_pass
```

Why: For standard symbols, range_prob is the correct regime test (HMM classifies range vs trend). For trending micro-oscillators, range_prob is structurally low (the HMM correctly sees a trend). The substitute test — `survival_prob >= 0.60` — validates that price is statistically contained within the grid range (Monte Carlo mean-reversion containment), which is the actual property a grid bot needs.

Safety invariant preserved: `all(gates.values())` at line 202 still requires all 4 mandatory gates. Gate 4 is not removed — only its test condition is swapped. Both modes test a probability threshold and return a boolean into `gates["regime_confidence"]`.

Call site (enrich_grid_params.py lines 607-619): pass `micro_osc_score` and `survival_prob` from enrichment data:
```python
sb_result = stage_b_selector.approve(
    ...,
    micro_osc_score=float(rd.micro_osc_score) if rd.micro_osc_score is not None else 0.0,
    survival_prob=float(rd.survival_prob) if rd.survival_prob is not None else 0.0,
)
```

Where `rd.micro_osc_score` must be added to `_RegimeData` dataclass (read from scan DataFrame column propagated through enrichment).

### Step 4: Register Feature in Pipeline

Per `safety-invariants.md` Feature Pipeline Update Rule — three mandatory files, plus scanner integration and imputation default.

| File | Location | Addition |
|------|----------|----------|
| `candidate_pipeline.py` | `_SCANNER_TO_FEATURE` (line 34) | `"micro_osc_score": "micro_osc_score"`, `"scan_micro_osc_score": "micro_osc_score"` |
| `candidate_pipeline.py` | `TRAINING_OUTPUT_COLUMNS` (line 80) | `"micro_osc_score"` |
| `data_generator.py` | `FeatureSnapshot` (line 114) | `micro_osc_score: Optional[float] = None` |
| `data_generator.py` | `to_dict()` (line 184) | `"micro_osc_score": self.micro_osc_score` |
| `unified_training_builder.py` | `EXTRA_META_FEATURES` (line 41) | `"micro_osc_score"` |
| `unified_training_builder.py` | `_SCAN_TO_FEATURE` (line 56) | `"scan_micro_osc_score": "micro_osc_score"` |
| `scanner_integration.py` | `build_feature_snapshot()` (line 42) | `micro_osc_score=_safe_float(scanner_row.get("micro_osc_score")),` |
| `meta_labeler.py` | `_FEATURE_MEDIAN_DEFAULTS` (line 91) | `"micro_osc_score": 0.0` |

All identifiers verified to exist at the stated locations.

### Step 5: Config and Feature Flag

File: `src/neutralgrid/core/config.py`

**5a — New dataclass (before `Config` class at line 483):**

```python
@dataclass
class MicroOscConfig:
    enabled: bool = False
    min_score: float = 0.45
    min_survival_prob: float = 0.60
```

Three fields. Each consumed by at least one live code path:
- `enabled`: read by Step 2a (eligibility gate) and Step 3 (Gate 4 archetype)
- `min_score`: read by Step 2a (micro_osc_mask threshold) and Step 3 (Gate 4 archetype trigger)
- `min_survival_prob`: read by Step 2a (micro_osc_mask dual-gate) and Step 3 (Gate 4 survival test)

**5b — Add to top-level `Config` class:**

```python
micro_osc: MicroOscConfig = field(default_factory=MicroOscConfig)
```

This makes `get_config().micro_osc` available everywhere.

**5c — Env-var override (in config loading section):**

```python
if val := os.getenv("MICRO_OSC_ENABLED"):
    config.micro_osc.enabled = val.lower() in ("1", "true", "yes")
```

**Config access pattern summary:**

| Consumer | Reads |
|----------|-------|
| `enrich_grid_params.py` (Step 2a, 2b, 2d, 2e) | `get_config().micro_osc.enabled`, `.min_score`, `.min_survival_prob` |
| `two_stage_selector.py` (Step 3) | `get_config().micro_osc.enabled`, `.min_score`, `.min_survival_prob` |
| `scan.py` (Step 1) | No config read — always computes `micro_osc_score` for training data |

**Update `safety-invariants.md`:**

Add after "All 4 mandatory gates must pass":
```
Gate 4 tests regime suitability. Standard archetype: range_prob >= threshold.
When micro_osc.enabled and micro_osc_score >= min_score:
Gate 4 tests survival_prob >= min_survival_prob (MC containment).
Gate 4 remains mandatory in both modes.
```

## Files Modified

| File | Change | Risk |
|------|--------|------|
| `config.py` | `kline_limits["15m"]`: 200 → 300, `MicroOscConfig` dataclass (3 fields), `Config.micro_osc` field | Low |
| `run_full_pipeline.py` | `--compute-stochastic` default: `store_true` → `default=None` | Low |
| `scan.py` | `micro_osc_score` computation after stochastic features | Low — additive |
| `enrich_grid_params.py` | Eligibility bypass + sort-tier + bypass propagation + threshold tag exclusion + pre-reject conditioning | Medium — gate logic |
| `two_stage_selector.py` | Gate 4 archetype check + 2 new params | Medium — gate logic |
| `candidate_pipeline.py` | Feature mappings | Low — additive |
| `data_generator.py` | Field + to_dict | Low — additive |
| `unified_training_builder.py` | Feature mappings | Low — additive |
| `scanner_integration.py` | `micro_osc_score` mapping in `build_feature_snapshot()` | Low — additive |
| `meta_labeler.py` | Imputation default | Low — additive |
| `safety-invariants.md` | Gate 4 archetype documentation | Low — docs |

11 files total.

## What This Does NOT Change

- HMM model or training
- Meta-labeler model
- Backtest engine
- Scanner score formula
- Conformal risk control
- Label contract
- TOS (Tradable Oscillation Scorer) weights or sub-signals
- Existing pipeline behavior when flag is disabled

## What Was Removed From v6 and Why

### Struck: Provable False Optionality

| Item | Proof |
|------|-------|
| `MicroOscConfig.k_vwap` (was line 336) | Defined in config; Step 1 hardcodes `_k_vwap=25.0` and config summary says "No config read" (was line 379). Config field has no consumer. |
| `MicroOscConfig.k_ema` (was line 337) | Same proof as k_vwap. Hardcoded `_k_ema=20.0` with no config wiring. |
| `MicroOscConfig.w_vwap` (was line 338) | Hardcoded as `0.45` in Step 1 score formula. Config field has no consumer. |
| `MicroOscConfig.w_hurst` (was line 339) | Hardcoded as `0.35` in Step 1 score formula. Config field has no consumer. |
| `MicroOscConfig.w_ema` (was line 340) | Hardcoded as `0.20` in Step 1 score formula. Config field has no consumer. |
| `oscillation_scorer.micro_osc_enabled` (was line 360) | Force-synced from `micro_osc.enabled` at line 369. Derived duplicate with no independent business meaning. TOS can read `get_config().micro_osc.enabled` directly if needed. |

### Struck: Provably Unnecessary As Written

| Item | Proof |
|------|-------|
| `SymbolFeatures.micro_osc_score` field (was line 79) | `compute_features()` returns `SymbolFeatures` which is immediately converted to `row = feats.as_dict()` (scan.py:375). `micro_osc_score` is computed on `row` (dict) after line 422. The dataclass is never used downstream after conversion. Field would be populated as `None` by `as_dict()` then immediately overwritten. |

### Struck: Separate `enrich_bypass_threshold` and `stage_b_bypass_threshold`

Both were 0.45 with no demonstrated need to diverge. Consolidated to single `min_score` field. If validation proves they need different values, split at that time.

### Deferred to v2: Not Proven Necessary for ONUSDT Miss Path

| Item | Proof | Future Path |
|------|-------|-------------|
| Step 3 / `micro_gcf_5m` (was lines 236-270) | ONUSDT was never enriched (line 13). TOS runs inside enrichment (enrich_grid_params.py:543-584). TOS enhancement cannot fix an upstream miss. | Add when micro-osc symbols reach enrichment and TOS quality matters. |
| `weights_8` (was line 359) | Exists solely to support 8th TOS signal. No consumer without Step 3. | Defers with Step 3 as a unit. |
| Step 2f / HMM suppression (was lines 171-231) | Under default `soft_gating=True` (config.py:192), HMM check always passes: `base_passed=True` (regime_validator.py:389) → `hmm_check.passed=True` → `vres.is_valid=True`. `_check_regime_rejection` at line 1351 returns `None` when `vres.is_valid=True`. Step 2f's narrowing logic is unreachable under defaults. Hysteresis (default 3 bars) does not change this: `base_passed` is unconditionally `True` under soft gating, so hysteresis count always trends positive. | Add if `soft_gating` is ever set to `False`, or as defensive hardening. |

## Validation Criteria (Before Enabling Flag)

1. Collect `micro_osc_score` across 250+ symbols. Verify separation between winners and losers.
2. Using `new_expired_bots.xlsx`, find threshold that maximizes F1 for profitable micro-oscillator bots.
3. Among micro_osc bypass candidates, compute false positive rate via backtest.
4. For trending symbols with survival_prob >= 0.60, verify MC range-containment accuracy.
5. Full pipeline integration test with flag enabled. Zero regressions + ONUSDT-like symbols appear.
6. Verify `all(gates.values())` produces correct results with archetype-dependent Gate 4.

## Approval Reminder

Wait for explicit user approval before making any runtime code changes.
