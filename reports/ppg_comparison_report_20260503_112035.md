# Hypothetical Test Report - Model `profit_per_grid_pct` vs Binance `grid_spacing_pct`

**Run date:** 2026-05-03
**Subset filter:** `duration_hours < 7.0` from `data/new_expired_bots.xlsx`
**Population:** 94 rows / 70 unique symbols (out of 209 in the file)
**Methodology:** read-only inline computation; both formulas applied to the SAME `(grid_lower, grid_upper, grids_count)` per row.

---

## 1. Inputs and formulas (as audited in source)

### 1.1 Model side - `compute_profit_per_grid()`
File: `src/neutralgrid/training/data_generator.py:730-753`
```
d = (upper - lower) / num_grids
c = taker_fee = 0.0005
max_profit = (1 - c) * d / lower      - 2c
min_profit = upper * (1 - c) / (upper - d) - 1 - c
ppg_pct    = max(0, (min + max) / 2) * 100
```
Used by: `unified_training_builder` (live ingestion path, `EXTRA_META_FEATURES`), scanner promotion features.

### 1.2 Binance side - `grid_spacing_pct` column
File: `_bot_data_extractor_core.py:99-104` (arithmetic mode, the documented path)
```
grid_spacing_pct = (high - low) / low * 100 / (n - 1)     # n grid LINES, n-1 GAPS
```

---

## 2. Verification (passed)

| Check | Result |
|---|---|
| Filter correctness: `max(duration_hours)` in subset | `6.6000h` (< 7.0) **OK** |
| Decomposition closure: `delta_bps == divisor_bps + avg_bps + fee_bps` | `max\|err\| = 7.10e-15 bps` **OK** |
| Spot-check NIGHTUSDT formula (manual recomputation) | matches |

**Caveat surfaced during verification:** `BINANCE RECOMP MAX|diff|` was `0.578` percentage points. The stored `grid_spacing_pct` does **not** uniformly follow the documented `(n-1)` arithmetic formula across this dataset. See section 5.1.

---

## 3. Headline results

### 3.1 Distributions (n=94)

| Metric | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|
| Binance `grid_spacing_pct` (%) | +0.9895 | +0.7535 | 0.5168 | +0.4478 | +3.1984 |
| Model `profit_per_grid_pct` (%) | +0.7756 | +0.6415 | 0.3559 | +0.3370 | +2.4074 |
| **Delta = model - binance (bps)** | **-21.3937** | **-14.7533** | 18.4897 | -117.2368 | +4.1532 |

### 3.2 Sign of delta

| Direction | rows |
|---|---:|
| Model **lower** than Binance (`delta < 0`) | **93** |
| Equal (`delta == 0`) | 0 |
| Model **higher** than Binance (`delta > 0`) | 1 |
| `\|delta\| >= 30 bps` (large divergence) | 20 |
| `\|delta\| <  5 bps` (close match) | 1 |

The model is **systematically below** Binance's reported spacing across the entire short-duration subset.

### 3.3 Decomposition of the gap (mean contribution to delta_bps)

| Component | Mean (bps) | Sign | Source |
|---|---:|---|---|
| Divisor effect (`n` vs `n-1`) | +1.41 | mixed (std 17.99, range -53 to +57) | Binance/extractor uses `n-1`; model uses `n` |
| Min/max averaging effect | -12.76 | always negative on this subset | model averages bottom/top-of-range; Binance reports bottom-of-range |
| Fee deduction effect | -10.04 | always negative (fixed ~10 bps) | model subtracts taker fee `c=0.0005`; Binance is gross |
| **Net** | **-21.39** | -- | -- |

The fee component is essentially constant at -10 bps (median -10.037, std 0.018 bps); it is the **dominant deterministic** contribution.

The min/max-averaging component drives nearly all the variability and pulls the model down by another -12.76 bps on average, with worst cases at -102 bps for low-grid-count, wide-range bots (`grids_count = 4..7`).

The divisor component is small in mean but bimodal (44 rows negative, 46 positive, 4 zero), confirming the input-data heterogeneity in section 5.1.

---

## 4. Variables and indicators captured per row

The full per-row dump (printed during the run, not retained as a file) includes the following columns for each of the 94 rows:

**Identity & geometry**
`symbol, strategy_id, status, duration_hours, grids_count, price_range_low, price_range_high, leverage, invested_margin_usdt`

**Comparison outputs**
`binance` (= `grid_spacing_pct`, %), `model_full` (= `compute_profit_per_grid` output, %), `delta_bps`, `divisor_bps`, `avg_bps`, `fee_bps`

**Realized PnL**
`pnl_pct, total_profit_usdt, realized_pnl_usdt, unrealized_pnl_usdt, funding_fee_usdt, commission_usdt, profit_factor`

**Execution**
`total_trades, maker_count, taker_count, executed_qty, notional_completed`

**Indicators (regime/microstructure context)**
`adx_1h, adx_15m, adx_5m, rsi_15m, ema_slope_1h, ema_crosses_5m, vwap_crosses_5m, range_size_pct, bb_width, trend_structure`

**Excursion**
`mae, mfe, mae_pct_initial, mfe_pct_initial`

These indicators are **not** consumed by either compared formula - both formulas operate purely on grid geometry. They are listed here per the user's request for a "detailed analysis of all variables and indicators" so the reader can correlate the geometric divergence with realized outcomes if desired.

### 4.1 Population summary

- `duration_hours`: min 0.000, median 4.275, max 6.600
- `grids_count`: min 4, median 38.0, max 150 (peak at n=50: 11 rows; long tail of micro-grid 4-8 levels: 21 rows)
- `trend_structure`: uptrend 39, range 29, downtrend 23
- `status`: cancelled 64, Canceled 15, expired 15

---

## 5. Differences between the model and the Binance system

The two systems disagree along **four distinct axes**. Items 1-3 are mathematically inherent in the formulas and reproduce on every row; item 4 is dataset-specific and was discovered during this run.

### 5.1 Inconsistent divisor in the stored `grid_spacing_pct` column (NEW FINDING)

The extractor docstring claims `(n - 1)` (arithmetic mode), but only **44 of 94** rows in this subset are consistent with that convention. The breakdown:

| Implied divisor of stored value | rows |
|---|---:|
| Strict `(n - 1)` (matches extractor spec) | 44 |
| Strict `n` (matches model) | 4 |
| Other / inconsistent (denominator implied > n) | 46 |

The 46 "other" rows have `divisor_bps > 0`, meaning the stored value is **smaller** than even the `(d/low)/n` arithmetic spacing. Possible upstream causes (not investigated in this test):
- Manual entry where the user typed a Binance-displayed range different from the recomputed `(high - low) / low`
- Geometric-mode bots whose stored value follows `(high/low)^(1/(n-1)) - 1`
- Post-extraction rounding or a separate code path overwriting `grid_spacing_pct`

This heterogeneity should be flagged before any downstream feature uses `grid_spacing_pct` as a clean canonical input.

### 5.2 Divisor (`n` vs `n - 1`)

- **Binance/extractor**: `n - 1` (gaps between `n` grid lines).
- **Model**: `n` (treats `grids_count` as interval count).

Effect: for a given range, model `d` is smaller than Binance `d` by factor `(n-1)/n`. For `n=50` this is -2%, for `n=5` this is -20%. On rows where Binance does follow `(n-1)`, this contributes up to -53 bps to delta. On rows where Binance also uses `n`, the divisor effect is zero.

### 5.3 Min/max averaging

- **Binance**: reports a single bottom-of-range value `d / low * 100`.
- **Model**: averages `max_profit` (bottom-of-range, `(1-c)*d/lower - 2c`) and `min_profit` (top-of-range, `upper*(1-c)/(upper-d) - 1 - c`).

For arithmetic grids, `d/(upper-d) < d/lower` (a grid step is a smaller percentage of the upper price than of the lower price), so averaging pulls the model below Binance's bottom-of-range value. Always negative on this subset (mean -12.76 bps, worst -102.74 bps for low-`n` wide-range bots).

### 5.4 Fee deduction

- **Binance**: gross (no fees).
- **Model**: subtracts `2*c` from `max_profit` and `c` from `min_profit`, so on average `-1.5*c = -7.5 bps` plus second-order interaction with `d` (full effect ~-10.04 bps).

Always negative; very low variance (std 0.018 bps); essentially a fixed ~10 bps offset.

### 5.5 `max(0, ...)` clamp (model only)

Model floors negative average profit at zero. Not triggered on this subset (all spacings are positive enough that fees do not flip the sign).

---

## 6. What this comparison does NOT test

- Both formulas are purely **a-priori, geometric** estimators. Neither uses fill counts, funding, holding time, leverage, MAE/MFE, microstructure, volatility, or regime indicators.
- The stored `total_profit_usdt`, `realized_pnl_usdt`, `pnl_pct`, `profit_factor`, etc., are **realized outcomes**. They cannot be predicted by, nor used to validate, either of the two compared spacing formulas.
- A meaningful "model vs Binance realized" comparison would require a different test (Option B/C/D from the plan-mode disambiguation).

---

## 7. Recommendations (informational, no action taken)

1. **Audit the extractor**: `grid_spacing_pct` column is heterogeneously produced across this dataset. Either (a) rerun the extractor with a single canonical formula, or (b) add a `grid_spacing_pct_source` column to disambiguate.
2. **Decide divisor convention**: the model uses `n`, the extractor docstring claims `n - 1`. Pick one and align both. The `n - 1` convention matches Binance's UI behavior on arithmetic grids; the model would then need to be rebuilt with `d = (upper - lower) / (n - 1)`.
3. **Document fee constant**: `taker_fee = 0.0005` is hard-coded inside `compute_profit_per_grid`. If the deployment uses maker-mostly execution, the model under-estimates by ~5 bps relative to actual fees; if it uses higher-tier fees, vice versa.
4. **No code edits performed**. This was a hypothetical test only.

---

## 8. Cleanup performed

- Deleted: `_ppg_test_TEMP.py` (script)
- Deleted: `_ppg_test_TEMP_OUTPUT.txt` (raw stdout dump)
- Deliverable retained: this report (`reports/ppg_comparison_report_20260503_112035.md`)
- No artifacts, configs, or model files were touched.
