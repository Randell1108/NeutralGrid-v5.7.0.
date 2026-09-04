# GRID_SYNCH.md - Synchronizing Model Grid Calculations with Binance (Geometric Mode)

**Authored:** 2026-05-03
**Author:** code audit + 2x web cross-check + empirical verification on 19 geometric expired bots
**Status:** IMPLEMENTATION COMPLETE — see progress bar below. Closed 2026-05-04 under session GRIDFIX-001.

## Implementation Progress

**Overall:** [6/6 steps fully complete]  `[██████████] 100%`

| Step | Progress | Status | Updated |
|---|---|---|---|
| 1. Add `mode` to data layer | `[██████████] 100%` | DONE (GRIDFIX-001 2026-05-04) — extractor row dict emits `mode` (`_bot_data_extractor_core.py:1613`); `FeatureSnapshot` field + `to_dict` added; `ExistingDataMapper.COLUMN_MAP` carries it; both feature-pipeline contracts (`candidate_pipeline.py`, `unified_training_builder.py`) updated atomically; xlsx populated 209/209 rows (166 arithmetic, 43 geometric, 0 NaN) | 2026-05-04 |
| 2. Create `grid/formulas.py` (deduplicate F1/F2/F3) | `[██████████] 100%` | DONE — new file `src/neutralgrid/grid/formulas.py` (80 LOC); 6 tests in `tests/unit/test_grid_formulas.py` pass | 2026-05-03 |
| 3. Switch divisor `n` → `(n - 1)` | `[██████████] 100%` | DONE — F1/F2/F3/F6 landed Apr 30 (commit 2bbd55d); F5 (`backtest/backtest_realistic.py:161,276`) and `spacing_profile.py:120-121` consolidated 2026-05-04 under GRIDFIX-001; full `(n-1)` divisor convention with N grid LINES across all sites | 2026-05-04 |
| 4. Live default → geometric | `[██████████] 100%` | DONE — commit 2bbd55d landed for F1/F3 + min/max field strike + grid-count inversion (Apr 30); F2 corrected this branch | 2026-05-03 |
| 5. Stored values authoritative for past rows | `[██████████] 100%` | DONE (GRIDFIX-001 2026-05-04) — F1 wrapper `compute_profit_per_grid` requires explicit `mode: str` (no default), threads from `row["mode"]` in `map_dataframe`; silent `0.0` defaults replaced with `float("nan")`; row-exclusion gate at end of `map_dataframe` drops rows with both `grid_spacing_pct` and `profit_per_grid_pct` NaN (zero rows dropped today, defensive); `c` derivation pinned to F2/F3 pattern `(maker_fee + close_fee_rate)/2` for cross-site equivalence; meta-labeler distribution diff at `reports/grid_synch_step5_distribution_diff_20260504_165655.md` shows arithmetic-row corrections (mean +1.04 bps, max +38.47 bps) and geometric-row delta = 0.0 (unchanged as expected) | 2026-05-04 |
| 6. Update locked tests | `[██████████] 100%` | DONE — `tests/unit/test_btk_exchange_rounding.py` (rounding suite), `test_btk_seed_state.py` (5 fixture sites), `test_btk_order_lifecycle.py` (`_config`), `test_btk_global_cooldown.py` (`_base_config`), `test_btk_gap_fixes.py` (`_base_config`), `test_btk_funding_modes.py` (`_base_config`), `test_new_bot_data_extractor.py` (FakeExistingDataMapper signature + base_kwargs) all updated with GRID_SYNCH §3.1 references in docstrings; 209/209 affected unit tests pass | 2026-05-04 |

**Re-test result (post-impl geometric comparison):**
- Spacing delta: PRE 11.59 bps mean → POST 0.003 bps mean (**100% reduction**, all 19 rows < 1 bps)
- PPG delta: residual ~3 bps from unspecified Binance fee constant (Step 7, out of scope)
- Full data: `reports/ppg_geometric_comparison_postimpl_20260503_120000.md` + `reports/ppg_geometric_comparison_data_20260503_postimpl.csv`
- Branch: `grid-synch-impl` (separate environment from main)

## Closure Note (2026-05-04 — session GRIDFIX-001)

Four-agent team review (data-curator, feature-analyst, deployment-engineering,
backtest-evaluator) of the original GRID_SYNCH.md found that the prior 90%
progress bar overstated completion. Specifically:
- F5 (`backtest/backtest_realistic.py:161,276`) Step 3 fix had not actually
  landed — spacing still divided by `n` instead of `n-1`, and the level
  list used `range(n+1)` for `n` intervals.
- `grid/spacing_profile.py:120-121` contained an inline duplicate of the
  geometric/arithmetic identity, bypassing `grid/formulas.py`.
- The xlsx had no stored `profit_per_grid_pct` column, so the v2 plan's
  "passthrough stored value" branch was dead code.
- The `data/manual_input/*.txt` pool overlapped the locked candidate-id
  pool by only 5 rows (out of 82); a TXT-only mode tagging would have
  left 77 of 82 locked-pool rows with `mode=NaN`, dramatically reducing
  PPG coverage.

The v3 plan (and this closure) addresses all of the above:
1. F5 divisor + level count corrected; n-grid-LINES Binance convention
   adopted across the codebase.
2. `spacing_profile.py:_infer_mode` delegates to `grid.formulas`.
3. xlsx mode column populated end-to-end (3 stages: TXT-tagged + pre-launch
   arithmetic rule + user-supplied final 30 entries) with zero data loss.
4. F1 wrapper threads per-row `mode`; silent geometric flip eliminated;
   silent `0.0` defaults replaced with NaN; fail-closed exclusion gate
   added.

**Step 7 (`c` measurement) remains explicitly DEFERRED** per §8 — narrowed
in v3 plan to maker-only fit because the available pool has zero
taker-dominant rows. The current `taker_fee = 0.0005` (5 bps) at
`core/config.py:80` is preserved as Binance's published taker fee,
explicitly labeled as assumption-of-record.

Progress legend: `█` = done sub-task, `░` = pending.

---

---

## 0. Scope and ground rules

This document captures structural differences between Binance's live grid bot calculations and the model's calculations, and presents a step-by-step plan to close the gap. Per the user directive: when the model uses arithmetic and Binance uses geometric, the model must move to geometric.

Ground rules applied throughout:

- **No assumption claims.** Every cited line and formula has a file path, line number, and a verifiable artifact. Where a quantity could not be verified at the time of writing, it is labeled `UNVERIFIED` and excluded from the recommended plan.
- **Provable false optionality** is explicitly flagged. A "false option" is a configuration knob, fallback, or branch that does not change observable behaviour or whose only role is to silently mask the problem under audit.
- **Deduplication > addition.** Where the same formula is implemented in three places, the recommendation is to consolidate, not to add a fourth.
- **Feature Pipeline Update Rule** (`.claude/rules/safety-invariants.md`) is binding. Any new column in the training table must update all three contracts atomically: `_SCANNER_TO_FEATURE` + `TRAINING_OUTPUT_COLUMNS` (`backtest/candidate_pipeline.py`), `FeatureSnapshot` + `to_dict` (`training/data_generator.py`), `EXTRA_META_FEATURES` + `_SCAN_TO_FEATURE` (`training/unified_training_builder.py`).

---

## 1. Authoritative Binance formulas (web cross-check)

### 1.1 Grid spacing - geometric mode (official)

> "The price range of each cell of the geometric grid is proportional. The geometric grid divides the price range from the Lower Price to the Upper Price into the number of grids by equal price ratio."
> "Ratio = (Upper Price / Lower Price) ^ (1 / n) * 100%; where n = the number of grids."
> -- Binance Support FAQ, [What Is Futures Grid Trading?](https://www.binance.com/en/support/faq/what-is-futures-grid-trading-f4c453bab89648beb722aa26634120c3)

### 1.2 Profit per grid - geometric (official)

> "For Geometric Grids, Profit/Grid is a fixed value, since the price ratio of each Grid price is the same."
> "Grid Profit is the realized profit of filled grid orders that are matched by one buy-order and one sell-order, and the trading fees incurred during the strategy are already deducted in the Grid Profit."
> -- Binance Support FAQ, [Binance Spot Grid Trading Parameters](https://www.binance.com/en/support/faq/detail/688ff6ff08734848915de76a07b953dd) (parameters apply to futures grids identically per the same FAQ)

### 1.3 Profit per grid - arithmetic (official, for comparison)

> Maximum Profit/Grid = (1 - c) * d / Grid_Lower - 2c
> Minimum Profit/Grid = Upper * (1 - c) / (Upper - d) - 1 - c
> where d = (Upper - Lower) / n and c = trading fee per side.
> -- same source

### 1.4 Documentation ambiguity (provable)

Binance's text says "n = the number of grids" but the empirical extraction confirms the exponent uses `(grids_count - 1)` (intervals, not lines). Verification: 19 geometric bots in `data/new_expired_bots.xlsx` reproduce their stored `grid_spacing_pct` to within 4.88e-5 percentage points using `((high/low) ^ (1/(n-1)) - 1) * 100` — see `reports/ppg_geometric_comparison_data_20260503_113000.csv`, column `binance_spacing_geo_recompute_pct` vs `binance_spacing_stored_pct`. Thus the empirical Binance formula uses **`(n - 1)`** in the exponent, regardless of the FAQ wording.

### 1.5 Profit per grid - empirical net offset (UNVERIFIED to a specific fee constant)

The 19-row geometric subset shows the Binance UI's `Profit Per Grid` value sits ~5 to ~7 bps below the gross geometric spacing. Examples (from the CSV):

| symbol | binance_spacing_stored_pct | binance_ppg_reported_pct | gross - net (bps) |
|---|---:|---:|---:|
| BASEDUSDT | 0.6211 | 0.56 | 6.11 |
| CTSIUSDT | 0.7013 | 0.64 | 6.13 |
| SIRENUSDT | 0.4721 | 0.42 | 5.21 |
| CHIPUSDT | 0.9126 | 0.85 | 6.26 |

The offset is consistent with one round-trip fee deduction (~6 bps per round trip), but **no Binance source explicitly defines the c constant used in the futures-grid UI**. It is between maker (2 bps × 2 = 4 bps) and taker (5 bps × 2 = 10 bps), and is also impacted by 2-decimal rounding in the displayed value. This document does NOT claim a specific fee constant; the model fee constant is treated as a separate calibration issue (Step 7).

---

## 2. Current state of the model - call-site map

All 24 source files that touch grid spacing or profit-per-grid were audited. The map below is exhaustive.

### 2.1 Formula sites (creators of the value)

| # | File | Lines | Function | Mode | Formula | Fee constant |
|---|---|---|---|---|---|---|
| F1 | `src/neutralgrid/training/data_generator.py` | 730-753 | `ExistingDataMapper.compute_profit_per_grid` | **arithmetic only** | `d = (upper-lower)/n`; max=(1-c)*d/lower - 2c; min=upper*(1-c)/(upper-d) - 1 - c; avg=(min+max)/2 | hard-coded `taker_fee=0.0005` |
| F2 | `src/neutralgrid/scanner/empirical_profile_v20260302.py` | 78-100 | `_compute_profit_per_grid_pct` | **arithmetic only** | identical to F1 | `c = (maker_fee + close_fee_rate)/2` from config |
| F3 | `src/neutralgrid/grid/calculator.py` | 246-311 | `GridCalculator.calculate_profit_per_grid` | **arithmetic only** | identical to F1 (returns tuple of min/max/avg) | `c = (maker_fee + close_fee_rate)/2` from config |
| F4 | `_bot_data_extractor_core.py` | 88-104 | `ExtractedBotData.compute_derived_fields` | **mode-aware** (arithmetic + geometric branches) | geometric: `((high/low)^(1/(n-1)) - 1) * 100`; arithmetic: `(high-low)/low/(n-1) * 100` | none (gross) |
| F5 | `backtest/backtest_realistic.py` | 161-163 | inline in `BacktestEngine` `__init__` | **arithmetic only** | `grid_spacing = (upper-lower)/n`; `grid_spacing_pct = grid_spacing/avg_price * 100` | none |
| F6 | `src/neutralgrid/grid/calculator.py` | 400-407, 434-439, 580-590 | `compute_regime_adjusted_grids` and `generate_params` | **arithmetic only** | implied spacing = `(upper-lower)/lower/n * 100` | none (it's a spacing-only routine) |

**Observations (provable):**

- F1, F2, F3 are **three independent copies of the same arithmetic formula**. They diverge in fee handling only (F1 is hard-coded; F2/F3 read config). This is a deduplication target (Step 5).
- F4 is the **only mode-aware** site. Mode is parsed from manual TXT extracts and used to branch the spacing computation, then **the `mode` field is dropped before xlsx serialization** (verified: column does not exist in `data/new_expired_bots.xlsx`).
- F5 and F6 are arithmetic-only **grid-construction** sites. F5 builds the actual level list at backtest time. F6 derives spacing from final geometry inside the live params generator.
- The "model" in this codebase has **no geometric branch anywhere outside F4** (the extractor).

### 2.2 Consumer sites (read the value as a feature or gate)

| File | Lines | Use |
|---|---|---|
| `src/neutralgrid/models/meta_labeler.py` | 103-105, 132-134, 219, 229-231 | Listed in `ACTIVE_SNAPSHOT_META_FEATURES` and `_FEATURE_MEDIAN_DEFAULTS`. Median default for `profit_per_grid_pct = 0.3`, for `grid_spacing_pct = 0.75`. |
| `src/neutralgrid/grid/spacing_profile.py` | 67, 86-100, 333-348 | Winner IQR profiling; per-bucket medians and IQR bands. |
| `src/neutralgrid/scanner/pnl_ranker.py` | 137-322 | Plugs `profit_per_grid_pct` into analytical EV. |
| `src/neutralgrid/scanner/tradable_oscillation.py` | TOS-score input |
| `src/neutralgrid/validation/microstructure.py` and `microstructure_hard_gate.py` | Profit floor hard gate. |
| `src/neutralgrid/scanner/enrich_grid_params.py` | feeds `GridParams.profit_per_grid_pct` |
| `src/neutralgrid/training/unified_training_builder.py` | 42-76 | Lists in `EXTRA_META_FEATURES` and `_SCAN_TO_FEATURE`. |
| `src/neutralgrid/backtest/candidate_pipeline.py` | 35, 101-162, 775-794 | `_SCANNER_TO_FEATURE` and `TRAINING_OUTPUT_COLUMNS`. |
| `src/neutralgrid/training/data_generator.py` | 119, 294, 843-854 | `FeatureSnapshot` and `map_dataframe`. |

### 2.3 Tests that lock current arithmetic-only behaviour

These five files assert specific arithmetic-mode values and **must be updated atomically** with any formula change:

| File | What it locks |
|---|---|
| `tests/unit/test_bot_data_extractor_v2.py` | mode-aware `compute_derived_fields()` arithmetic + geometric expected values (already mode-aware; only fee/divisor changes affect this) |
| `tests/unit/test_backfill_training_features_v20260312.py` | hard-coded `grid_spacing_pct=0.40` |
| `tests/unit/test_bde_coherence_and_parsing.py` | grid_spacing_pct sanity bounds |
| `tests/test_afml_integrations.py` | hard-coded `grid_spacing_pct=0.5` in feature snapshots |
| `tests/test_micro_osc_integration.py` | hard-coded `grid_spacing_pct=0.5` fixture |

---

## 3. Structural differences (model -> Binance) - verified

### 3.1 Mode

- **Binance live grids:** the user picks arithmetic or geometric per bot; geometric is common in volatile / wide-range setups.
- **Model:** silently treats every bot as arithmetic (F1, F2, F3, F5, F6). Mode is parsed by F4 but discarded before any consumer can read it.

**Empirical proof:** `data/manual_input/` contains 38 TXT extracts; 19 are tagged `Geometric` and all 19 match short-duration rows (`duration_hours < 7.0`) in `data/new_expired_bots.xlsx`. The xlsx has no `mode` column - confirmed by `pd.read_excel(...).columns`.

### 3.2 Grid spacing formula

| Side | Formula | Result on row 4 (LYNUSDT, n=150, lo=0.04057, hi=0.11948) |
|---|---|---|
| Binance (geometric) | `((high/low)^(1/(n-1)) - 1) * 100` | 0.7275% |
| Model (arithmetic, `d/lower`) | `(high-low)/low/n * 100` | 1.2966% |
| Delta | model - binance | **+56.91 bps** (model OVER-estimates) |

For the 19-row geometric subset: mean delta `+9.45` bps, median `+7.06` bps, max `+56.91` bps.

### 3.3 Profit per grid

| Side | Formula | Result on row 4 |
|---|---|---|
| Binance (UI value, parsed from TXT) | `Profit Per Grid` displayed | 0.99% |
| Model (arithmetic `compute_profit_per_grid`) | F1 / F3 | 1.0790% |
| Delta | model - binance | **+8.90 bps** (model OVER-estimates) |

For 19-row subset: 18 of 19 rows show model UNDER-estimates (mean `-3.71` bps, median `-4.75` bps). The single positive outlier is the wide-range row above.

### 3.4 Divisor mismatch (`n` vs `n - 1`)

- Binance geometric uses `(n - 1)` in the exponent (proven empirically in section 1.4).
- Model arithmetic uses `n` in `(high - low) / n` (F1, F2, F3, F5, F6).

This is independent of mode and is a known issue from the prior arithmetic-mode test (`reports/ppg_comparison_report_20260503_112035.md` section 5.2).

### 3.5 Fee constant

- F1 hard-codes `taker_fee = 0.0005` (5 bps).
- F2, F3 use `c = (maker_fee + close_fee_rate) / 2` from config (typically (2 + 5)/2 = 3.5 bps depending on `close_fee_mode`).
- F4 is gross.
- The Binance UI value is empirically gross-spacing minus ~6 bps per round trip (section 1.5). The exact c is **UNVERIFIED**; the model and Binance therefore differ by an unquantified rounding/fee constant (~1-2 bps in either direction). Treat as separate calibration; **do not change it as part of this synchronization** without an explicit measurement plan.

---

## 4. Step-by-step plan (concise, traceable, no false optionality)

Each step is **mandatory** unless explicitly tagged `[REMOVABLE - provably unnecessary]`. There are NO conditional / opt-in steps. Every step has a verification command that emits a binary pass/fail signal.

### Step 1 - Add `mode` to the data layer

**Goal:** Make mode a first-class column from extraction through training.

**Changes:**
1. `_bot_data_extractor_core.py`: add `"mode": bot.mode,` immediately after the `"grids_count"` entry on line 1612 of the dict at lines 1602-1647. **Verified 2026-05-03 by Read:** the current dict does not contain a `"mode"` key, while `bot.mode` is populated by `compute_derived_fields` and listed in the `valid_fields` set at line 587. This is the only site where the parsed mode is silently dropped.
2. `data/new_expired_bots.xlsx` schema: re-extract from `data/manual_input/*.txt` to populate the new `mode` column. For rows without a TXT extract, set `mode = NaN` (no inference).
3. `src/neutralgrid/backtest/candidate_pipeline.py:35-100` - add `"mode"` to `_SCANNER_TO_FEATURE` and `TRAINING_OUTPUT_COLUMNS`.
4. `src/neutralgrid/training/data_generator.py:119` `FeatureSnapshot` - add `mode: Optional[str]` and update `to_dict()`.
5. `src/neutralgrid/training/unified_training_builder.py:42-76` - add `"mode"` to `EXTRA_META_FEATURES` and the `_SCAN_TO_FEATURE` map.

**Verification:**
- `python -c "import pandas as pd; df = pd.read_excel('data/new_expired_bots.xlsx'); assert 'mode' in df.columns; print(df['mode'].value_counts(dropna=False))"` -> expect 19 geometric, ~30+ arithmetic, rest NaN.
- `python -m pytest tests/unit/test_bot_data_extractor_v2.py -v` -> all green (already covers both modes).
- Feature Pipeline Update Rule check: grep `grep -n '"mode"' src/neutralgrid/backtest/candidate_pipeline.py src/neutralgrid/training/data_generator.py src/neutralgrid/training/unified_training_builder.py` -> three hits across three files.

**Why this is required (not optional):** without a `mode` column, no downstream branching can be correct. The current implicit "arithmetic" assumption corrupts the 19/94 geometric rows in training.

### Step 2 - Create one shared formula module; deprecate F1, F2, F3

**Goal:** Eliminate the triplicate arithmetic formula. Add the geometric formula in the same module.

**Changes:**
1. New module: `src/neutralgrid/grid/formulas.py` with two pure functions:
   ```python
   def grid_spacing_pct(low: float, high: float, n: int, mode: str) -> float
   def profit_per_grid_pct(low: float, high: float, n: int, mode: str, c: float) -> float
   ```
   - `mode in {"arithmetic", "geometric"}`. No default. No silent fallback. Raises `ValueError` for unknown modes.
   - `c` is required, no default. Caller must pass the explicit fee constant.
   - Geometric branch:
     - `r = (high/low) ** (1.0 / (n - 1))`
     - `gross_spacing_pct = (r - 1) * 100`
     - `profit_pct = (r - 1 - 2*c) * 100`
   - Arithmetic branch: copies F3's formula verbatim (it is the most complete of the three).
2. Replace F1, F2, F3 call sites with calls to `grid.formulas`. F1's hard-coded `taker_fee=0.0005` is replaced by an explicit `c` argument from the caller.

**Verification:**
- `pyright` clean.
- New unit tests in `tests/unit/test_grid_formulas.py`:
  - Reproduces Binance's documented arithmetic example (Upper=450, Lower=400, n=5, c=0.001 -> max=2.30%, min=2.07%) within 1e-3 percentage points.
  - Reproduces stored `grid_spacing_pct` for all 19 geometric rows in the CSV within 1e-3 percentage points.
  - Raises `ValueError` for `mode="cubic"`.
- `python -m pytest tests/ -v` -> existing tests still green (F1/F2/F3 call sites now route through the shared module; behaviour unchanged for arithmetic).

**Why this is required:** three independent copies of the same formula is a maintenance hazard. Adding a fourth (geometric) copy in each location would compound the problem.

### Step 3 - Switch divisor to `(n - 1)` everywhere

**Goal:** Eliminate the `n` vs `n - 1` mismatch (section 3.4).

**Changes:**
1. In `grid.formulas.grid_spacing_pct` arithmetic branch: use `(high - low) / low / (n - 1) * 100`.
2. In `grid.formulas.profit_per_grid_pct` arithmetic branch: use `d = (high - low) / (n - 1)`.
3. F5 (`backtest/backtest_realistic.py:161`): change `(upper-lower)/num_grids` to `(upper-lower)/(num_grids-1)`.
4. F6 (`grid/calculator.py:400-407, 434-439, 580-590`): change all four occurrences of `(grid_upper - grid_lower) / grid_lower / num_grids` to `... / (num_grids - 1)`.

**Verification:**
- `python -m pytest tests/ -v` -> some assertions will need numeric updates; each updated test must include a one-line reference to GRID_SYNCH.md Step 3 in its docstring.
- Re-run the geometric comparison test: max `|stored - geometric_recomputed|` should remain < 1e-3 percentage points (this step does not change the geometric branch, only arithmetic, so the geometric closure check is untouched).

**Why this is required:** Binance's empirical formula is `(n - 1)`. The model's `n` divisor is a provable error of factor `(n-1)/n`, which on `n=5` is -20% on the spacing.

### Step 4 - Live-deployment default switches to geometric

**Goal:** Per the user directive, the model now uses geometric by default for new grid construction.

**Changes:**
1. `src/neutralgrid/grid/calculator.py:140-203` `calculate_grid_spacing` - irrelevant (returns a target spacing; unaffected).
2. `src/neutralgrid/grid/calculator.py:246-311` `calculate_profit_per_grid` - replace body with a call to `grid.formulas.profit_per_grid_pct(low, high, n, mode="geometric", c=c)`.
3. `src/neutralgrid/grid/calculator.py:483-645` `generate_params`:
   - Set `grid_spacing_pct` from `grid.formulas.grid_spacing_pct(grid_lower, grid_upper, adjusted_num_grids, "geometric")`.
   - Set `profit_per_grid_pct` from the geometric branch.
   - **Collapse** the min/max profit fields (`profit_per_grid_min_pct`, `profit_per_grid_max_pct` in `GridParams` lines 34-35) into the single `profit_per_grid_pct`. **[REMOVABLE - provably unnecessary in geometric, see classification 5.1.]** justification: geometric grids have constant per-grid percent profit by definition (Binance docs section 1.2); min and max are mathematically equal. Keeping both fields is false optionality.
4. Update `GridParams.to_dict` (`calculator.py:75-101`) to drop `profit_per_grid_min_pct` and `profit_per_grid_max_pct`.
5. Update consumer sites that currently read `profit_per_grid_min_pct` / `profit_per_grid_max_pct`. **Verified 2026-05-03 by grep:** consumers exist in `src/neutralgrid/scanner/enrich_grid_params.py`, `tests/unit/test_enrich_grid_params.py`, and `tests/test_micro_osc_integration.py`. Each consumer must be migrated to use the single `profit_per_grid_pct` value, OR keep reading min/max but receive `profit_per_grid_min_pct == profit_per_grid_max_pct == profit_per_grid_pct` (transitional bridge). Recommendation: migrate atomically; do not introduce a transitional bridge (false optionality). The migration is mechanical (replace `min(...)` / `max(...)` clamps with the single value) and bounded to those three files.

**Verification:**
- `grep -rn "profit_per_grid_min_pct\|profit_per_grid_max_pct" src/ tests/` -> only the dataclass definition and any updated tests.
- `python -m pytest tests/unit/test_enrich_grid_params.py -v` -> green after numeric assertion updates.
- New live-bot smoke: invoke `GridCalculator.generate_params(...)` on a known regime; assert `grid_spacing_pct` matches `((high/low)^(1/(n-1)) - 1) * 100` to 1e-6.

**Why this is required:** The user directive is explicit. Skipping it (e.g., adding a `mode` config flag with arithmetic as a soft default) is *false optionality* because the user has already chosen.

### Step 5 - Training-time recomputation uses stored `grid_spacing_pct` as ground truth, not the model formula

**Goal:** Past Binance bots had a fixed mode at construction time. The training pipeline should not recompute spacing for them; it should pass through the stored value.

**Changes:**
1. `src/neutralgrid/training/data_generator.py:843-854` `map_dataframe`: when `df["grid_spacing_pct"]` is present and not NaN, copy it directly to `result["grid_spacing_pct"]`. Only call `compute_profit_per_grid` (now `grid.formulas.profit_per_grid_pct`) when `df["grid_spacing_pct"]` is NaN AND `df["mode"]` is set. If both are missing, the row is **excluded from training** (not silently defaulted).
2. Same for `profit_per_grid_pct` if such a column exists; if it does not, derive from `(low, high, n, mode)` only when all four are present.

**Verification:**
- `python -m pytest tests/unit/test_unified_training_builder.py -v` -> green.
- `python -c "from neutralgrid.training.data_generator import ExistingDataMapper; ..."` smoke that loads a row from the xlsx and asserts the mapped feature equals the source column when both exist.

**Why this is required:** Recomputing arithmetic spacing for geometric bots (the current behaviour) silently corrupts 19/94 training rows. Per memory note `feedback_no_silent_degradation.md`, this is the kind of silent-corruption the project rules forbid.

### Step 6 - Tests: update locked numeric expectations

**Goal:** Five tests lock arithmetic-only values. Update them to:
- Assert mode-aware behaviour (parameterized by mode).
- Assert the new `(n - 1)` divisor.
- Drop `profit_per_grid_min/max` assertions where geometric is the active mode.

**Changes per file** (concrete numeric updates in each `expected = ...` line; cross-checked against `grid.formulas`):
- `tests/unit/test_bot_data_extractor_v2.py` — already mode-aware; verify `(n-1)` divisor used in both branches.
- `tests/unit/test_backfill_training_features_v20260312.py` — replace `0.40` with the value produced by the geometric branch on that fixture's geometry.
- `tests/unit/test_bde_coherence_and_parsing.py` — bounds unchanged; validate that mode is now a column.
- `tests/test_afml_integrations.py` — replace `0.5` and re-derive from fixture geometry.
- `tests/test_micro_osc_integration.py` — replace `0.5` and re-derive.

**Verification:**
- `python -m pytest tests/ -v` -> all green.
- Confirm test count is unchanged (no tests removed, only updated).

### Step 7 - Calibrate fee constant `c` (separate, optional, NOT part of this synchronization)

The empirical offset in section 1.5 (~6 bps gross-vs-displayed) is **UNVERIFIED** to a specific Binance constant. This step is intentionally **out of scope** of GRID_SYNCH:

- Without measurement, choosing `c = 0.0005` (current F1 default), `c = 0.0002` (maker), or `c = 0.001` (Binance docs example) would be an assumption.
- A separate calibration job — not described here — would compare 50+ extracted bots' `Profit Per Grid` UI values to the gross spacing and fit `c`.

**This step is listed only to make explicit that it is not in the plan.** The model's `c` value is left as-is during synchronization.

---

## 5. Categorization (provable false-optionality / unnecessary / not strikable)

### 5.1 Provably FALSE OPTIONALITY (must be removed in implementation)

| Item | Site | Proof |
|---|---|---|
| `profit_per_grid_min_pct` and `profit_per_grid_max_pct` fields | `grid/calculator.py:34-35`, `to_dict()` 86-87 | After Step 4, geometric grids have a single profit value by Binance's definition (section 1.2). Min/max are mathematically equal. Keeping both is a knob with no observable distinction. |
| Hard-coded `taker_fee=0.0005` default in F1 | `training/data_generator.py:735-737` | Two of the three sister formulas (F2, F3) read `c` from config. The default in F1 is a silent override of the config — the only path where the function is called without an explicit `c` is the training mapper (Step 5 removes that path). |
| The implicit "arithmetic" assumption when `mode` is missing | F1, F2, F3, F5, F6 | After Step 1 `mode` is always present (or row excluded). Defaulting to arithmetic on a `None` mode is a silent fallback. |
| `compute_profit_per_grid` exposed as a method on `ExistingDataMapper` (data class) | F1 | The function is pure; coupling it to a stateful mapper is unnecessary. Moves to `grid.formulas` in Step 2. |

### 5.2 Provably UNNECESSARY items as written

| Item | Site | Proof |
|---|---|---|
| Three independent copies of the arithmetic formula | F1, F2, F3 | Diff shows identical structure; only fee handling differs. Step 2 deduplicates. |
| `grid_spacing_pct` recomputation in `backtest_realistic.py:161-163` and in `calculator.py:400-590` (multiple lines) | F5, F6 | After Step 5 the stored value is authoritative for past data; for live grid construction the value is computed once in `grid.formulas`. The duplicate recomputations exist only to recover from missing inputs and are dead code under the new contract. |
| Median default `profit_per_grid_pct = 0.3` and `grid_spacing_pct = 0.75` in `meta_labeler.py:229-231` | meta-labeler | After Step 5 these features are never imputed at training time (rows without the value are excluded). The medians remain only for inference paths where a runtime artifact may lack the field; **NOT removable** without further investigation of inference paths (see 5.3). |

### 5.3 Items NOT VALID to strike without making assumptions

| Item | Why I refuse to recommend deletion |
|---|---|
| `_FEATURE_MEDIAN_DEFAULTS` for `profit_per_grid_pct` and `grid_spacing_pct` (`meta_labeler.py:229-231`) | Removing them would change inference behaviour on legacy artifacts that may not carry the field. I have not verified every inference path. Recommendation: keep, document the new training contract above the dict. |
| `compute_regime_adjusted_grids` and the regime-widening multiplier in `calculator.py:353-445` | The widening logic is orthogonal to mode; geometric grids can be regime-widened too. Removing or rewriting requires regime-test review beyond the scope of synchronization. |
| Existing `_SCANNER_TO_FEATURE` arithmetic-only entries | They map scan columns to feature columns; they do not depend on mode. No change. |
| Dropping the arithmetic branch entirely from `grid.formulas` | The user said "if model uses arithmetic, change to geometric." That is a default switch, not a directive to remove arithmetic. Past bots that were arithmetic still need correct feature reproduction in training. |
| Changing the fee constant `c` | Section 1.5 is unverified; changing this without measurement is an assumption. Step 7 is explicitly out of scope. |

---

## 6. Self-review of each plan step

Before recommending implementation, every step is reviewed against five criteria:

| # | Criterion | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Step 6 |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | Traceable to a specific file:line | Yes (every change) | Yes | Yes | Yes | Yes | Yes |
| 2 | Verification produces a binary pass/fail | Yes | Yes | Yes | Yes | Yes | Yes |
| 3 | No silent fallback / no false optionality | Yes (`mode` is required, no default) | Yes (no default `c`, no default mode) | Yes (single divisor, no flag) | Yes (geometric is the new default; no opt-out flag) | Yes (rows without inputs are excluded, not defaulted) | Yes (no skipped tests) |
| 4 | Respects Feature Pipeline Update Rule | Yes (Step 1 explicitly updates all three contracts) | N/A (no new feature column) | N/A | Yes (no new feature column; only formula change) | N/A | Yes |
| 5 | Respects Leakage Prevention | Yes (mode is regime-orthogonal; not a label proxy) | Yes | Yes | Yes | Yes | Yes |

**Cross-step contradictions checked:**

- Step 1 adds `mode`; Step 2 requires `mode` as a function argument. Consistent.
- Step 3 changes divisor; Step 6 updates the locked numeric tests. Consistent.
- Step 4 makes geometric the default; Step 5 makes stored values authoritative for past rows. Consistent — the model formula changes only the future-grid-design path.
- Step 7 is explicitly out of scope — no other step assumes a particular `c`. Consistent.

**Backwards-compatibility ratchet (per `safety-invariants.md`):**

- HMM artifact format unchanged (mode is bot-side metadata, not a regime feature).
- `LABEL_CONTRACT_VERSION` unchanged (label generator is mode-blind).
- Conformal / utility calibrators unaffected.

**Memory checks:**

- `feedback_no_silent_degradation.md`: Step 5 refuses silent recomputation. Pass.
- `feedback_pre_impl_consistency_pass.md`: completed in this section.
- `feedback_sample_pool_fixed.md`: no new training rows proposed; the locked pool is preserved. Pass.
- `feedback_no_unnecessary_files.md`: only one new file is added (`src/neutralgrid/grid/formulas.py`) and its purpose (deduplication) is explicit. Pass.

---

## 7. Validation proof per implemented modification

Every modification listed in Steps 1-6 carries a verification artifact. Below is the verification map; running the listed command after implementation will produce evidence that can be reviewed line by line.

| Step | What to verify | Concrete command | Expected signal |
|---|---|---|---|
| 1 | `mode` column present in xlsx | `python -c "import pandas as pd; df = pd.read_excel('data/new_expired_bots.xlsx'); print(df['mode'].value_counts(dropna=False))"` | Counts include `geometric` and `arithmetic` |
| 1 | Feature Pipeline Update Rule contracts updated | `grep -n '"mode"' src/neutralgrid/backtest/candidate_pipeline.py src/neutralgrid/training/data_generator.py src/neutralgrid/training/unified_training_builder.py` | three hits |
| 2 | Shared formulas reproduce Binance docs example | `python -m pytest tests/unit/test_grid_formulas.py::test_binance_arithmetic_doc_example -v` | passes within 1e-3 % points |
| 2 | F1/F2/F3 sites no longer contain inline formulas | `grep -nE "max_profit\s*=\s*\(1\s*-\s*c\)" src/neutralgrid/training src/neutralgrid/scanner src/neutralgrid/grid/calculator.py` | only `src/neutralgrid/grid/formulas.py` matches |
| 3 | `(n - 1)` divisor consistent | `grep -nE "(num_grids|grid_count|adjusted_num_grids)\s*\)?\s*$|/\s*(num_grids|grid_count|adjusted_num_grids)\b" src/neutralgrid/grid/calculator.py backtest/backtest_realistic.py` and audit each remaining hit. | no `... / num_grids` in spacing arithmetic |
| 4 | `GridParams` exposes only the single geometric profit value | `grep -n "profit_per_grid_min_pct\|profit_per_grid_max_pct" src/ tests/` | empty (after migration of `enrich_grid_params.py`, `test_enrich_grid_params.py`, `test_micro_osc_integration.py`) |
| 4 | Live `generate_params` produces geometric spacing | unit test in `tests/unit/test_grid_calculator.py` asserts geometric formula on fresh fixture | passes |
| 5 | Training mapper passes through stored value | `python -m pytest tests/unit/test_unified_training_builder.py -v` | green |
| 5 | Rows without `mode` and without `grid_spacing_pct` are excluded, not defaulted | added test asserts `len(out) < len(in)` for a fixture with one row missing both | green |
| 6 | All tests green | `python -m pytest tests/` | exit code 0 |

---

## 8. What this plan does NOT do (out of scope, by design)

- Does **not** change the fee constant `c` (Step 7 deferred).
- Does **not** retrain HMM, meta-labeler, or any calibrator. The synchronization affects feature *values* for past geometric rows but does not retrain. A subsequent retrain may follow once the data is correct, gated by `mean_pass_rate >= 0.50` per `safety-invariants.md`.
- Does **not** add or remove training rows. The locked pool (`feedback_sample_pool_fixed.md`) is preserved.
- Does **not** add new feature columns to `meta_labeler.py:ACTIVE_SNAPSHOT_META_FEATURES`. The mode field is bot-side metadata, not a regime feature; if it later becomes a learned input, that is a separate decision with its own audit.
- Does **not** modify `backtest/backtest_realistic.py` beyond the divisor fix (Step 3). The backtester still constructs arithmetic level lists; geometric backtest support is a follow-up.

---

## 9. Sources

Web (cross-checked, two independent searches):
- [Binance Support FAQ - What Is Futures Grid Trading?](https://www.binance.com/en/support/faq/what-is-futures-grid-trading-f4c453bab89648beb722aa26634120c3)
- [Binance Support FAQ - Binance Spot Grid Trading Parameters](https://www.binance.com/en/support/faq/detail/688ff6ff08734848915de76a07b953dd)
- [Binance Blog - Step-by-step Guide to Grid Trading on Binance Futures](https://www.binance.com/en/blog/futures/stepbystep-guide-to-grid-trading-on-binance-futures-1221278002770616377)
- [Binance Blog - Demystifying the Futures Grid Bot](https://www.binance.com/en/blog/markets/demystifying-the-futures-grid-bot-5592639286007523221)

Empirical artifacts (in this repo):
- `data/new_expired_bots.xlsx` (209 bots, 19 confirmed geometric in the duration-<7h subset)
- `data/manual_input/*.txt` (38 TXT extracts; 19 tagged `Geometric`)
- `reports/ppg_geometric_comparison_data_20260503_113000.csv` (per-row geometric verification)
- `reports/ppg_geometric_comparison_report_20260503_113000.md` (analysis)
- `reports/ppg_comparison_report_20260503_112035.md` (prior arithmetic-mode comparison)

Source-code anchor points (already cited inline above):
- `_bot_data_extractor_core.py:88-104, 297-305, 1612`
- `src/neutralgrid/training/data_generator.py:730-753, 119, 843-854`
- `src/neutralgrid/scanner/empirical_profile_v20260302.py:78-100`
- `src/neutralgrid/grid/calculator.py:246-311, 353-445, 580-590`
- `backtest/backtest_realistic.py:161-163`
- `src/neutralgrid/models/meta_labeler.py:103-138, 219, 229-231`
- `src/neutralgrid/training/unified_training_builder.py:42-76`
- `src/neutralgrid/backtest/candidate_pipeline.py:35-162, 775-794`
