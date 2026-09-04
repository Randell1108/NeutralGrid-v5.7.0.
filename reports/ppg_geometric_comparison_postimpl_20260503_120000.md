# Post-Implementation Geometric Comparison Report

**Run date:** 2026-05-03 (post-implementation)
**Branch:** `grid-synch-impl` (separate environment from main)
**Subset:** `duration_hours < 7.0` AND mode == "geometric" (mode confirmed via TXT extracts)
**Population:** 19 rows / 19 unique symbols (matches the pre-impl population exactly)

---

## 0. What changed

Commit history shows the geometric switch was already largely landed in `2bbd55d` (April 30, 2026, merged to main April 30 via PR #1). On top of that, this branch added:

1. **`src/neutralgrid/grid/formulas.py` (new, 80 LOC)** — single-source-of-truth for `grid_spacing_pct(low, high, n, mode)` and `profit_per_grid_pct(low, high, n, mode, c)`. Both arithmetic and geometric branches. Mode and `c` required (no defaults), unknown mode raises `ValueError`.
2. **`tests/unit/test_grid_formulas.py` (new, 6 tests)** — verifies arithmetic internal consistency, geometric reproduces 19-row CSV, unknown mode raises, invalid inputs raise.
3. **F1 (`training/data_generator.py:734`)** — `compute_profit_per_grid` now delegates to `grid.formulas.profit_per_grid_pct(..., GEOMETRIC, taker_fee)`. Was previously inline geometric.
4. **F2 (`scanner/empirical_profile_v20260302.py:78`)** — `_compute_profit_per_grid_pct` now delegates to `grid.formulas.profit_per_grid_pct(..., GEOMETRIC, c)`. **Was still arithmetic until this branch.**
5. **F3 (`grid/calculator.py:282`)** — `calculate_profit_per_grid` now delegates to `grid.formulas.profit_per_grid_pct(..., GEOMETRIC, c)`. Was previously inline geometric.

The three formula sites are now deduplicated through one shared module.

---

## 1. Verification (passed)

| Check | Result |
|---|---|
| Filter correctness: `max(duration_hours)` in subset | `6.2200h` (< 7.0) **OK** |
| `grid.formulas.grid_spacing_pct(..., GEOMETRIC)` reproduces stored values | max abs diff `4.88e-5` percentage points **OK** |
| Targeted test files (test_grid_formulas, test_enrich_grid_params, test_unified_training_builder, test_bot_data_extractor_v2) | 202/202 green **OK** |
| Full suite | 1177 pass + 5 PRE-EXISTING fails (calibration pool size, unrelated to grid math; reproduced via `git stash` test) **OK** |

---

## 2. Headline results (post-implementation)

### 2.1 Distributions (n=19)

| Metric | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|
| Binance spacing (stored geometric) (%) | +0.7251 | +0.7175 | 0.1433 | +0.4721 | +1.1699 |
| MODEL spacing (geometric, post-impl) (%) | +0.7251 | +0.7175 | 0.1432 | +0.4721 | +1.1699 |
| **Spacing delta (bps)** | **+0.0007** | **+0.0013** | 0.0032 | -0.0049 | +0.0047 |
| Binance PPG (TXT, UI value) (%) | +0.6489 | +0.6400 | 0.1373 | +0.4200 | +1.0800 |
| MODEL PPG (geometric, c=0.0002) (%) | +0.6851 | +0.6775 | 0.1432 | +0.4321 | +1.1299 |
| MODEL PPG (geometric, c=0.0005) (%) | +0.6251 | +0.6175 | 0.1432 | +0.3721 | +1.0699 |
| PPG delta (c=maker) (bps) | +3.6154 | +2.6903 | 2.6593 | +0.7547 | +11.2818 |
| PPG delta (c=taker) (bps) | -2.3846 | -3.3097 | 2.6593 | -5.2453 | +5.2818 |

### 2.2 Sign of deltas

| | Spacing | PPG (c=maker) | PPG (c=taker) |
|---|---:|---:|---:|
| Model > Binance | 13 | 19 | 3 |
| Equal | 0 | 0 | 0 |
| Model < Binance | 6 | 0 | 16 |
| `|delta| <= 1 bps` (spacing only) | **19/19** | — | — |
| `|delta| <= 2 bps` | — | 5/19 | 5/19 |

### 2.3 Before vs After (this is the headline)

|  | PRE-impl mean (abs bps) | POST-impl mean (abs bps) | Reduction |
|---|---:|---:|---:|
| Spacing delta | 11.588 | **0.003** | **-99.97%** |
| PPG delta | 4.643 | 3.615 (c=maker) | -22.1% |

The spacing-formula gap is **fully closed** (residual is floating-point noise). The PPG residual is dominated by the Binance UI fee constant, which remains unspecified in any source and is intentionally out of scope (Step 7 of GRID_SYNCH.md).

---

## 3. Variables and indicators captured per row

The full per-row dump (deleted with the temp output file) included, for each of the 19 rows:

**Identity & geometry**
`strategy_id, symbol, txt_file, duration_hours, grids_count, price_range_low, price_range_high, high_over_low, leverage, invested_margin_usdt`

**Comparison outputs**
`binance_spacing_stored_pct, model_spacing_geo_pct, model_spacing_arith_pct_REF, spacing_delta_bps, binance_ppg_reported_pct, model_ppg_geo_c_maker_pct, model_ppg_geo_c_taker_pct, ppg_delta_c_maker_bps, ppg_delta_c_taker_bps`

**Realized PnL**
`pnl_pct, total_profit_usdt, realized_pnl_usdt, unrealized_pnl_usdt, funding_fee_usdt, commission_usdt, profit_factor`

**Execution**
`total_trades, maker_count, taker_count, executed_qty, notional_completed`

**Indicators**
`adx_1h, adx_15m, adx_5m, rsi_15m, ema_slope_1h, ema_crosses_5m, vwap_crosses_5m, range_size_pct, bb_width, trend_structure`

**Excursion**
`mae, mfe, mae_pct_initial, mfe_pct_initial`

These indicators are not consumed by either compared formula — both are purely geometric. They are reported per the user's "all variables and indicators" request so the reader can correlate post-impl geometric agreement with realized outcomes.

The full per-row data is preserved in `reports/ppg_geometric_comparison_data_20260503_postimpl.csv` (19 rows × 38 columns).

### 3.1 Population summary

- `duration_hours`: min 0.72, median 4.47, max 6.22
- `grids_count`: min 8, median 35, max 150 (range 8–150 grids)
- `trend_structure`: uptrend 7, range 6, downtrend 5
- `status`: cancelled 13, expired 6 (counts unchanged from pre-impl since same population)

---

## 4. Differences between model and Binance — POST-IMPLEMENTATION

### 4.1 Grid spacing — fully synchronized

`((high / low)^(1 / (n - 1)) - 1) * 100` is now used by BOTH model and Binance/extractor. Residual is floating-point precision: max delta `0.005 bps` (i.e. 5e-7 percentage points) across all 19 rows. The 100% reduction in mean spacing delta is the principal outcome of the GRID_SYNCH.md Step 2-4 implementation.

### 4.2 Profit per grid — residual ~3 bps from fee-constant ambiguity

The model now uses geometric profit-per-grid: `((r - 1) - 2c) * 100`. With `c = 0.0002` (maker, default config), the model is uniformly **above** Binance's UI value by ~3-4 bps. With `c = 0.0005` (taker), it is uniformly **below** by ~2-3 bps. The truth lies in between — Binance's UI uses an effective `c ≈ 0.0003` (3 bps per side, 6 bps round-trip), which matches the gross-vs-displayed offset measured in the pre-impl test.

This is documented in `GRID_SYNCH.md` section 1.5 as **UNVERIFIED to a specific Binance constant** and Step 7 (out of scope of this synchronization). Fixing `c` requires a calibration job over more bots, which is a separate piece of work.

### 4.3 LYNUSDT outlier (row 4)

Pre-impl `spacing_delta = +56.91 bps` (the row that drove the worst arithmetic-vs-geometric mismatch).
Post-impl `spacing_delta ≈ +0.003 bps`. The wide-range bot (`high/low = 2.95`) that was the worst outlier is now perfectly aligned.

### 4.4 What is NOT closed (and why)

| Discrepancy | Status | Reason |
|---|---|---|
| Grid spacing geometric | **CLOSED** | Both sides use the same formula |
| Profit-per-grid formula | **CLOSED** | Model now uses geometric `(r-1) - 2c` |
| `n` vs `n-1` divisor | **CLOSED** in geometric branch (Step 3 in GRID_SYNCH was already done in commit 2bbd55d for the calculator; F2 corrected on this branch) | Both use `n - 1` |
| Fee constant `c` | **OPEN** | Binance UI `c` is unspecified (~0.0003 empirically); not safe to set without measurement (Step 7) |
| `mode` column propagation | OPEN (not blocking re-test) | `mode` is bot-side metadata; not a regime feature; deferred per GRID_SYNCH section 8 |
| Backtest engine geometric levels | OPEN | `backtest/backtest_realistic.py` still constructs arithmetic levels; separate follow-up |

---

## 5. Files modified on this branch

```
modified:   src/neutralgrid/training/data_generator.py     (F1: import + delegate to grid.formulas)
modified:   src/neutralgrid/scanner/empirical_profile_v20260302.py (F2: arithmetic -> geometric via shared module)
modified:   src/neutralgrid/grid/calculator.py              (F3: inline geometric -> delegate to shared module)
new file:   src/neutralgrid/grid/formulas.py                (80 LOC; deduplication anchor)
new file:   tests/unit/test_grid_formulas.py                (6 tests, all green)
```

---

## 6. Cleanup performed

- Deleted: `_ppg_geo_postimpl_TEMP.py` (test script, in `.claude/worktrees/add-gridcount/`)
- Deleted: `_ppg_geo_postimpl_TEMP_OUTPUT.txt` (raw stdout dump)
- Retained as deliverables:
  - `reports/ppg_geometric_comparison_postimpl_20260503_120000.md` (this report)
  - `reports/ppg_geometric_comparison_data_20260503_postimpl.csv` (full per-row data)
- Branch `grid-synch-impl` retained for review (not merged; user can review and merge or discard)

---

## 7. Sources

- `GRID_SYNCH.md` (repo root) — design document
- `reports/ppg_geometric_comparison_report_20260503_113000.md` — pre-impl report (for direct comparison)
- `reports/ppg_geometric_comparison_data_20260503_113000.csv` — pre-impl per-row data
- Commit `2bbd55d` — `feat(grid): switch to geometric grid identity` (Apr 30, 2026, merged main via PR #1)
- This branch's commits: visible via `git log grid-synch-impl ^main`
