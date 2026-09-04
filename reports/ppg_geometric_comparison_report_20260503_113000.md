# Hypothetical Test Report - GEOMETRIC Mode: Model (arithmetic) vs Binance (geometric)

**Run date:** 2026-05-03
**Subset filter:** `duration_hours < 7.0` AND mode == "geometric" (mode confirmed from manual TXT extracts)
**Population:** 19 rows / 19 unique symbols (out of 94 short-duration rows; 209 total in the file)
**Methodology:** read-only inline computation; both formulas applied to the SAME `(grid_lower, grid_upper, grids_count)` per row. Mode recovered by parsing the `Mode` field from each row's matching TXT in `data/manual_input/`.

---

## 1. Inputs and formulas (as audited in source)

### 1.1 Model side (existing, arithmetic - unchanged)

The model has **NO geometric branch**. Per Q2 answer (c), we apply the existing arithmetic formula to geometric bots, which quantifies the silent-arithmetic-assumption error.

| Output | Formula | Source |
|---|---|---|
| `model_spacing_arith` | `d / lower * 100` where `d = (high - lower) / num_grids` | implicit in `compute_profit_per_grid` |
| `model_ppg_arith` | `max(0, (min_profit + max_profit) / 2) * 100` with fee `c=0.0005`, `d = (high-low)/n` | `src/neutralgrid/training/data_generator.py:730-753` |

### 1.2 Binance side (geometric)

| Output | Formula / Source | Reference |
|---|---|---|
| `binance_spacing_stored` | column `grid_spacing_pct` from `data/new_expired_bots.xlsx` | extracted via `_bot_data_extractor_core.py:99-100` geometric branch: `((high/low)^(1/(n-1)) - 1) * 100` |
| `binance_ppg_reported` | UI-displayed value parsed from each row's TXT under `Profit Per Grid` | `_bot_data_extractor_core.py:298-305` regex |

For geometric grids the stored spacing is constant across all levels: `r - 1` where `r = (high/low)^(1/(n-1))`.

---

## 2. Verification (passed)

| Check | Result |
|---|---|
| Filter correctness: `max(duration_hours)` in subset | `6.2200h` (< 7.0) **OK** |
| Stored `grid_spacing_pct` reproduces with the geometric formula | `max\|diff\| = 4.88e-5` percentage points **OK** (numerical precision; confirms these 19 rows ARE geometric) |
| TXT-extract mode tag = "geometric" for all 19 rows | 19/19 **OK** |
| All 19 rows have a parsed `Profit Per Grid` value | 19/19 **OK** |
| `binance_ppg_high` is None for all 19 rows | 19/19 **OK** (geometric grids show a single value, not a range) |

---

## 3. Headline results

### 3.1 Distributions (n=19)

| Metric | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|
| Binance spacing (stored, geometric) (%) | +0.7251 | +0.7175 | 0.1433 | +0.4721 | +1.1699 |
| Model spacing (arithmetic, `d/lower`) (%) | +0.8196 | +0.7488 | 0.2142 | +0.5851 | +1.3177 |
| **Spacing delta = model - binance (bps)** | **+9.4468** | **+7.0582** | 14.2744 | -8.5641 | **+56.9142** |
| Binance Profit Per Grid (TXT, %) | +0.6489 | +0.6400 | 0.1373 | +0.4200 | +1.0800 |
| Model `profit_per_grid` (arithmetic, %) | +0.6119 | +0.5997 | 0.1460 | +0.3839 | +1.0442 |
| **PPG delta = model - binance (bps)** | **-3.7056** | **-4.7549** | 3.1961 | -5.6395 | +8.9032 |

### 3.2 Sign of deltas

| | Spacing delta | PPG delta |
|---|---:|---:|
| Model **higher** than Binance (`>0`) | 14 | 1 |
| Equal | 0 | 0 |
| Model **lower** than Binance (`<0`) | 5 | 18 |
| `\|delta\| >= 30 bps` | 1 | 0 |

The **two metrics move in opposite directions**:
- **Spacing**: model OVER-estimates (14/19), worst case +56.9 bps on CHIPUSDT (n=56, range 0.0753-0.1240).
- **Profit per grid**: model UNDER-estimates almost universally (18/19), but tightly clustered (-5.64 to +8.90 bps).

---

## 4. Per-row results (compact)

Full per-row dump (with all indicators) was generated and reviewed during the run; full file is no longer retained per cleanup policy. Compact extract below shows the comparison columns and key context.

| # | symbol | sid | n | low | high | bin_spc% | mod_spc% | spc_d_bps | bin_ppg% | mod_ppg% | ppg_d_bps | dur_h | trend |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | BASEDUSDT | 410995172 | 38 | 0.10020 | 0.12600 | 0.6211 | 0.6776 | +5.64 | 0.56 | 0.5094 | -5.06 | 6.13 | downtrend |
| 2 | CTSIUSDT | 411062151 | 50 | 0.03820 | 0.05380 | 0.7013 | 0.8168 | +11.55 | 0.64 | 0.5997 | -4.03 | 4.77 | range |
| 3 | SIRENUSDT | 410995039 | 100 | 0.85000 | 1.35500 | 0.4721 | 0.5941 | +12.20 | 0.42 | 0.3839 | -3.61 | 2.28 | * |
| 4 | LYNUSDT | 410756101 | 150 | 0.04057 | 0.11948 | 0.7275 | 1.2966 | +56.91 | 0.99 | 1.0790 | +8.90 | 2.23 | * |
| 5 | CHIPUSDT | 411517173 | 56 | 0.07525 | 0.12402 | 0.9126 | 1.1571 | +24.48 | 0.85 | 0.8318 | -1.82 | 1.17 | * |
| 6 | RIVERUSDT | 411518920 | 15 | 6.10300 | 6.71900 | 0.6892 | 0.6729 | -1.63 | 0.62 | 0.5644 | -5.64 | 2.97 | * |
| 7 | PIEVERSEUSDT | 411522164 | 8 | 0.96530 | 1.02300 | 0.8328 | 0.7472 | -8.56 | 0.69 | 0.6382 | -5.18 | 0.97 | * |
| 8 | METUSDT | 411514902 | 15 | 0.18040 | 0.20030 | 0.7502 | 0.7354 | -1.48 | 0.69 | 0.6307 | -4.93 | 6.22 | * |
| 9 | GRASSUSDT | 411521724 | 10 | 0.39120 | 0.41720 | 0.7175 | 0.6646 | -5.29 | 0.63 | 0.5755 | -5.45 | 4.33 | * |
| 10 | OPGUSDT | 411525629 | 30 | 0.35940 | 0.42870 | 0.6099 | 0.6428 | +3.29 | 0.55 | 0.5346 | -4.80 | 0.72 | * |
| 11 | CLOUSDT | 411521755 | 12 | 0.12918 | 0.13825 | 0.6188 | 0.5851 | -3.37 | 0.52 | 0.4979 | -5.29 | 5.05 | * |
| 12 | RAVEUSDT | 411521836 | 80 | 1.01024 | 1.61000 | 0.5917 | 0.7421 | +15.04 | 0.55 | 0.5147 | -3.53 | 5.02 | * |
| 13 | TACUSDT | 411541927 | 50 | 0.00701 | 0.01000 | 0.7270 | 0.8525 | +12.52 | 0.65 | 0.6280 | -4.36 | 5.35 | * |
| 14 | MAGMAUSDT | 411541856 | 35 | 0.17614 | 0.22230 | 0.6869 | 0.7488 | +6.19 | 0.60 | 0.5525 | -4.75 | 5.53 | * |
| 15 | STABLEUSDT | 411564395 | 48 | 0.02965 | 0.04118 | 0.7014 | 0.8101 | +10.88 | 0.65 | 0.5949 | -4.19 | 5.88 | * |
| 16 | BSBUSDT | 411565162 | 60 | 0.41352 | 0.63500 | 0.7296 | 0.8926 | +16.30 | 0.62 | 0.5868 | -3.17 | 3.58 | * |
| 17 | BASUSDT | 411565493 | 32 | 0.01368 | 0.01740 | 0.7778 | 0.8487 | +7.06 | 0.66 | 0.6096 | -5.04 | 5.52 | * |
| 18 | KATUSDT | 411570289 | 28 | 0.01778 | 0.02434 | 1.1699 | 1.3177 | +14.78 | 1.08 | 1.0442 | -3.58 | 4.47 | * |
| 19 | (CN-symbol) | 411589243 | 24 | 0.33661 | 0.39877 | 0.7395 | 0.7695 | +2.99 | 0.69 | 0.6411 | -4.88 | 3.03 | * |

(Trend/status not reproduced for every row to keep the table narrow; full dump included `adx_1h, adx_15m, adx_5m, rsi_15m, ema_slope_1h, ema_crosses_5m, vwap_crosses_5m, range_size_pct, bb_width, trend_structure, mae, mfe, mae_pct_initial, mfe_pct_initial, pnl_pct, total_profit_usdt, realized_pnl_usdt, unrealized_pnl_usdt, funding_fee_usdt, commission_usdt, profit_factor, total_trades, maker_count, taker_count, executed_qty, notional_completed, leverage, invested_margin_usdt, status` per row.)

### 4.1 Population summary

- `duration_hours`: min 0.72, median 4.47, max 6.22
- `grids_count`: min 8, median 35, max 150 (rows 4-150 grids; widely spread)
- `trend_structure`: uptrend 7, range 6, downtrend 5
- `status`: cancelled 13, expired 6

---

## 5. Differences between the model and the Binance system

These 19 bots were created by Binance with the **geometric** option, which means every grid level is spaced by the SAME percentage `r - 1`. The model has no geometric branch and computes everything as arithmetic. The mismatch produces three structural divergences.

### 5.1 Grid spacing - model uses bottom-of-range, geometric is uniform

- **Binance (geometric):** `d_pct_const = ((high/low)^(1/(n-1)) - 1) * 100` — same percent at every level.
- **Model (arithmetic):** `d_pct_at_lower = (high - low) / low / n * 100` — only correct at the bottom of the range; absolute step `d` is constant, so percent step shrinks as price rises.

For arithmetic-versus-geometric grids of the same `(low, high, n)`:
```
arithmetic_spacing_at_lower / geometric_spacing  =  ((high-low)/low/n)  /  ((high/low)^(1/(n-1)) - 1)
```
This ratio grows with `high/low`. The two row extremes confirm this:

| Row | symbol | high/low | n | spc_delta_bps |
|---|---|---:|---:|---:|
| 4 | LYNUSDT | 2.945 | 150 | **+56.9** (largest divergence) |
| 7 | PIEVERSEUSDT | 1.060 | 8 | -8.6 (model below; small range, n-1 vs n divisor dominates) |

So **wide-range bots** drive the model to over-estimate spacing dramatically; **narrow-range bots** (high/low close to 1) flip the sign because the n vs n-1 divisor effect dominates.

### 5.2 Profit per grid - tight clustering, almost always slightly under-reported by the model

PPG delta std = 3.20 bps, range only [-5.64, +8.90]. The model's arithmetic profit-per-grid is **systematically about 4-5 bps below** Binance's UI value across this subset. Reasons:

1. Binance's UI rounds to two decimal places (e.g., `0.56%`, `0.64%`). The model returns 6+ decimal places and applies fees explicitly. Some of the -3 to -5 bps gap is rounding noise in Binance's display.
2. For geometric bots, the *exact* gross profit per grid is `r - 1`. Binance's UI displays the gross-or-net value (Binance's documentation for grid bots labels it as the post-fee per-grid profit estimate; precise definition not visible in the codebase).
3. The model's `(min_profit + max_profit) / 2` averaging on geometric input introduces a small bias because `(high-low)/n` is not the same as `low * (r-1)`.

The single positive outlier (LYNUSDT, +8.90 bps) is the same wide-range row that drives the spacing extreme — for `high/low = 2.95`, the arithmetic average over [bottom, top] of the range happens to overshoot the geometric uniform value.

### 5.3 Divisor mismatch (`n` vs `n-1`)

This is the same effect documented in the prior arithmetic-mode test. In geometric mode it is even more visible because:
- Binance geometric uses `(n-1)` in the **exponent** of `(high/low)^(1/(n-1))`.
- Model arithmetic uses `n` in the **denominator** of `(high-low)/n`.

For narrow ranges these competing conventions can flip the sign of the spacing delta (rows 6, 7, 8, 9, 11, where `spc_delta_bps < 0`).

### 5.4 What this test does NOT find

- The model's `compute_profit_per_grid` is **not catastrophically wrong** for geometric bots with small `range_size_pct`; PPG delta stays under 6 bps on 18/19 rows.
- The realized PnL and realized profit-per-grid (e.g., row 1 BASEDUSDT realized 11.51 USDT / 38 grids ≈ 0.30 USDT/grid; row 2 CTSIUSDT realized 107.92 USDT / 50 grids ≈ 2.16 USDT/grid) cannot be predicted by either formula since both are purely geometric.

---

## 6. Recommendations (informational, no action taken)

1. **Decide if the model needs a geometric branch.** On this subset, using the arithmetic formula on geometric bots costs ~10 bps in spacing and ~4 bps in profit-per-grid. If the strategy population shifts toward wide-range geometric grids (e.g., LYNUSDT-class with `high/low > 2`), the spacing error explodes (>50 bps) and the assumption becomes meaningfully wrong. A geometric variant is cheap to add (`profit_per_grid_pct = ((high/low)^(1/(n-1)) - 1 - 2c) * 100`).
2. **Resolve the `n` vs `n-1` divisor mismatch.** Affects both arithmetic and geometric comparisons. Pick one convention end-to-end.
3. **Re-extract `grid_spacing_pct` consistently.** Prior test found 46/94 arithmetic rows do NOT match either pure `n-1` or pure `n` arithmetic formulas; the geometric subset on the other hand reproduces cleanly here (max diff 4.88e-5 percentage points), suggesting the inconsistency is concentrated in the arithmetic path of the extractor.
4. **No code edits performed.** This was a hypothetical test only.

---

## 7. Cleanup performed

- Deleted: `_ppg_geo_TEMP.py` (script)
- Deleted: `_ppg_geo_TEMP_OUTPUT.txt` (raw stdout dump)
- Deliverable retained: this report (`reports/ppg_geometric_comparison_report_20260503_113000.md`)
- No artifacts, configs, or model files were touched.
