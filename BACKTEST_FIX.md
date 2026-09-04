Progress: [##########] 100% (10/10 canonical integration steps DONE; candidate-time profile implemented as diagnostic-only; manual-export seeded validation remains upper-bound only)

# BACKTEST_FIX.md Geometric Backtest Trustability Plan

Goal: **"Make the geometric backtest model similarly realistic to live conditions which would make the model trustable for candidate selection."**

The scope is now geometric-only. Arithmetic workbook rows are historical rows and are not part of the acceptance cohort for this goal.

The full trustability goal is not completely achieved yet. The completed work
now proves that optional manual-export seeding materially improves geometric
backtest realism without changing unseeded full-pipeline behavior. Remaining
trustability is evidence-gated because many rows still have partial, not
complete, live evidence.

## Hard Boundaries

- Do not modify files that could provenly affect current full-pipeline results until the geometric-only diagnostic proves the exact mechanism.
- Do not touch scanner admission, scanner ranking, deploy-ready output logic, HMM, meta-labeler, utility scoring, live deployment code, or model artifacts.
- Do not retrain anything.
- Do not write model files.
- Do not create a new label contract, new engine version, new formula version, or new backtesting identity.
- Do not add `mode`, source coverage, seed source, or fee source as model features. They are validation metadata only.
- Keep `arithmetic` and `geometric` as the only grid modes.
- Keep explicit historical arithmetic rows as arithmetic.
- Use geometric rows only for the current trustability objective.
- Do not treat missing live data, missing seed state, or unknown intrabar path as verified truth.

## Side Note - Loader Scope And Data Growth

- The manual-export loader may scan files containing many symbols, but each
  backtest run must receive only the exact `SeedState` matching one bot by
  `symbol`, `strategy_id`, and time window.
- More complete symbol data helps validation because it increases verified
  coverage: actual ladder, quantities, trade timestamps, maker/taker flags,
  fees, realized PnL, funding, and final state.
- More data hurts only if partial or conflicting evidence is treated as
  complete. Therefore every row must be classified as `complete`, `partial`,
  `missing`, or `conflicting`; partial evidence remains diagnostic only.
- The loader must stay an adapter. It must not change engine behavior
  silently, create model features, or infer missing live state.

## Canonical Integration Steps

### Step 1 - Narrow Goal To Geometric Mode - DONE

Path:

- `BACKTEST_FIX.md`

Validation proof:

```text
workbook rows:       221
arithmetic rows:     166
geometric rows:       55
mode_values_tested: ["geometric"]
```

Reason:

- The user clarified that the current objective is geometric backtesting. A mixed 221-row acceptance set is structurally wrong because it blends historical arithmetic rows with geometric rows.

Rebuke:

- Full-workbook validation is still useful as a historical diagnostic, but it is false optionality for the current goal if it treats arithmetic rows as part of geometric model acceptance.

### Step 2 - Audit Existing Geometric Architecture - DONE

Paths audited:

- `backtest/btk_unified_runner.py`
- `backtest/backtest_realistic.py`
- `backtest/btk_label_contract.py`
- `scripts/validate_backtest_live_reconciliation.py`
- `data/new_expired_bots.xlsx`
- `data/manual_exports`

Validated architecture facts:

- `backtest/btk_unified_runner.py` is the production entry point for backtest execution and wraps `RealisticGridBacktester`.
- `backtest/backtest_realistic.py` already supports `GridConfig.mode`.
- Geometric grid levels are computed as a multiplicative ratio between lower and upper levels.
- Tick-size rounding fails closed if duplicate levels are created.
- Current training-style defaults use geometric mode, wick fills, maker close fees, no global cooldown, and no circuit breaker.
- `scripts/validate_backtest_live_reconciliation.py` is diagnostic-only and is not imported by scanner, training, deployment, or artifact code.

Potential gap found:

- The current wick-fill branch uses 1-minute OHLC high/low and bar direction to infer fills. A 1-minute candle does not prove the intrabar order of high and low. Therefore this path can over-count or under-count fills depending on the real sequence of touches.

Critical architecture constraint:

- A direct edit to `backtest/backtest_realistic.py` can affect candidate labels and therefore full-pipeline behavior. It is not allowed until the diagnostic proves the change on geometric rows.

### Step 3 - Refresh External Evidence - DONE

Web evidence used:

- Binance explains grid trading as preset orders inside a configured range and distinguishes arithmetic equal-price spacing from geometric equal-ratio spacing:
  <https://www.binance.com/en/academy/articles/step-by-step-guide-to-grid-trading-on-binance-futures>.
- Binance USD-M exchange information exposes trading rules and symbol filters including `PRICE_FILTER.tickSize`, `LOT_SIZE.stepSize`, and `MIN_NOTIONAL`, and warns that precision fields are not tick or step size:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Exchange-Information>.
- Binance USD-M klines expose OHLCV fields by open time:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data>.
- Binance USD-M mark-price klines exist separately from regular klines and are appropriate for mark-to-market, liquidation, and funding-related checks:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Mark-Price-Kline-Candlestick-Data>.
- Binance USD-M funding history exposes `fundingRate`, `fundingTime`, and the associated mark price:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History>.
- Binance USD-M account trades expose `commission`, `maker`, `price`, `qty`, `realizedPnl`, `side`, `positionSide`, and `time`, which are the correct fields for validating live fills and fee behavior:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api/Account-Trade-List>.
- Binance USD-M order-book depth is a current symbol order-book query through `/fapi/v1/depth`; it has no `startTime` or `endTime` parameter, so it cannot reconstruct historical queue state for expired bots unless depth snapshots or websocket diffs were archived separately:
  <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Order-Book>.
- HftBacktest documents the central assumptions behind market-data replay: replayed orders do not alter the historical market, market impact is excluded, and queue-position modelling needs market depth and trade data. This supports queue modelling as a diagnostic or future replay layer only when historical depth/trade feeds exist:
  <https://hft.readthedocs.io/en/latest/order_fill.html>.
- Jain, Firoozye, Kochems, and Treleaven review limit-order-book simulation models and classify LOB simulation as a distinct market-microstructure problem. This supports not treating 1-minute OHLC candles as equivalent to order-book replay:
  <https://arxiv.org/abs/2402.17359>.
- Bailey, Borwein, Lopez de Prado, and Zhu warn that repeated backtest tuning can create false positives:
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253>.

Integration consequence:

- The plan must prioritize structural physics and paired live evidence, not threshold tuning, retraining, or optimizing against the same backtest errors.
- The backtester can be optimized only where the input evidence supports the optimization. Account trades can validate actual fills, fees, realized PnL, and maker/taker state. Regular klines can validate trade-price touch ranges, but not intrabar high/low order. Mark-price klines can validate mark-to-market, liquidation, and funding-related valuation, but not trade triggers. Funding history can validate funding charges. Exchange information can validate tick, step, and minimum-notional rounding. Order-book depth can validate current book state only, unless historical book snapshots or diffs were already stored.

Research-backed consensus:

- Accepted for this plan: keep the existing `RealisticGridBacktester` and the diagnostic reconciliation harness as the integration surface. This is structurally aligned because the current pipeline already routes backtests through the unified runner and because a new engine would create a second behavior path before the current discrepancy is causally explained.
- Accepted for this plan: classify every geometric validation row by evidence completeness before judging model accuracy. A row with account trades, transaction history, order history, mark/funding data, and exchange filters can be evaluated more strictly than a workbook-only row. This does not change the pipeline because the class is validation metadata, not a model feature.
- Accepted for this plan: treat OHLC path variants as diagnostic envelopes only. A bar can prove that a level was inside the minute's high/low range, but it cannot prove whether the buy-side or sell-side touch happened first.
- Accepted for this plan: use mark-price klines only for valuation, drawdown, liquidation, and funding timing checks. Using mark price as the trade-trigger source would be structurally wrong because Binance exposes regular trade klines and mark-price klines as separate data sources.
- Accepted for this plan: use official exchange filters for rounding checks when validating realism. The exchange-info source explicitly provides `PRICE_FILTER.tickSize`, `LOT_SIZE.stepSize`, and `MIN_NOTIONAL`; precision fields are not valid substitutes.
- Accepted for this plan: keep deduplication and classification in the validation/reporting layer. The current goal is trustable candidate selection, so validation must report both PnL error and classification behavior such as winner recall and non-winner specificity.
- Rejected as production change for now: importing a full order-book replay engine. It is valid research direction, but false optionality for this patch because the current evidence does not prove archived historical depth/trade feeds exist for every geometric candidate-selection row, and adding a second engine would make silent drift more likely.
- Rejected as production change for now: queue-position modelling from 1-minute candles. Queue position requires depth and trade-sequence evidence. Without historical order-book depth or account-order placement state, any queue model would be an assumption.
- Rejected as production change for now: tuning fill/cooldown/liquidation parameters until the latest cohort matches live PnL. Bailey et al. identify repeated backtest selection as an overfitting risk; in this repo that would create a model that fits the validation cohort without proving candidate-selection trustability.
- Rejected as production change for now: scanner, ranking, threshold, or retraining edits. They would consume the current backtest labels but would not repair label realism.

### Step 4 - Implement Geometric-Only Diagnostic Harness - DONE

Path modified:

- `scripts/validate_backtest_live_reconciliation.py`

Implemented diagnostic-only capabilities:

- `--mode-filter geometric`
- explicit diagnostic override reporting
- fill-mode override for isolated tests
- order-delay override for isolated tests
- global-cooldown override for isolated tests
- deterministic OHLC path envelopes:
  - `--bar-path ohlc`
  - `--bar-path olhc`
- classification metrics:
  - PnL sign match
  - winner recall for `pnl_pct > 0`
  - non-winner specificity for `pnl_pct <= 0`
  - fast-winner recall for `pnl_pct > 1.0` and `duration_hours < 7.0`
- capital error reporting
- MAE percent initial pairing where workbook evidence exists
- workbook live fill count using `maker_count + taker_count` when available
- manual bot text trade-list parsing from `data/manual_input`
- manual-export and manual-input fill timing summaries:
  - first trade time
  - last trade time
  - intertrade gaps
  - same-minute clustering
  - sub-120-second gap share

Validation proof:

```text
python -m py_compile scripts\validate_backtest_live_reconciliation.py
python -m pyright scripts\validate_backtest_live_reconciliation.py

pyright result: 0 errors, 0 warnings, 0 informations
```

Rebuke:

- This is not a new backtest engine. It is a no-save diagnostic harness around the existing runner.
- This is not a new pipeline. It is not imported by production code.

### Step 5 - Run Geometric-Only Baseline - DONE

Command shape:

```text
python scripts\validate_backtest_live_reconciliation.py --scope all --mode-filter geometric --duration-source workbook
```

Validated result:

```text
rows tested:                         55
model rows completed:                55
missing kline rows:                   0
strict manual-export rows:           22

live mean PnL:                       1.739818%
model mean PnL:                     -3.062997%
mean absolute PnL error:            13.037998%
median absolute PnL error:           6.818518%
PnL sign match rate:                 0.654545

live positive rows:                  39
live non-positive rows:              16
winner recall, pnl > 0:              0.717949
non-winner specificity, pnl <= 0:    0.562500

live mean fill count:               181.200000
model mean trade count:             148.672727
mean absolute fill-count error:     122.709091

capital abs error mean:               0.000000
strict manual maker ratio:            0.999150
strict model mean abs PnL error:      7.330611%
```

Interpretation:

- The geometric model is not trustable yet.
- Capital shrink is not the remaining explanation because capital error is zero.
- Maker fee behavior is directionally supported by strict manual rows.
- The remaining gap is geometric execution realism: fill sequencing, intrabar path, start state, and risk path behavior.

### Step 6 - Run Isolated Geometric Variant Sweep - DONE

Scratch validation:

```text
SCRATCH_EXISTS_AFTER=False
```

Tested variants:

```text
current
close_only
wick_delay_0
wick_delay_1
wick_delay_3
wick_cooldown_2
path_ohlc
path_olhc
path_ohlc_cooldown_2
path_olhc_cooldown_2
```

Best aggregate diagnostic variant after corrected live-fill-count comparison:

```text
variant:                         wick_cooldown_2
rows:                            55
model rows:                      55
mean absolute PnL error:         10.371792%
median absolute PnL error:        5.638167%
PnL sign match:                   0.672727
winner recall, pnl > 0:           0.717949
non-winner specificity:           0.625000
fast-winner recall:               0.717949
mean abs trade-count error:       72.127273
capital abs error mean:            0.000000
MAE pct initial pair rows:        33
MAE pct initial mean abs error:   13.938882%
```

Current baseline for comparison:

```text
mean absolute PnL error:          13.037998%
median absolute PnL error:         6.818518%
PnL sign match:                    0.654545
non-winner specificity:            0.562500
mean abs trade-count error:        97.272727
MAE pct initial mean abs error:    16.972081%
```

Interpretation:

- `global_cooldown_bars=2` is the best isolated aggregate variant tested so far.
- It improves PnL error, median error, sign match, non-winner specificity, trade-count error, and MAE error without shrinking capital.
- It is not accepted as the production fix yet because a global cooldown can block all levels after any fill, while real grid bots can have multiple resting orders already open. Aggregate improvement alone is not proof of structural correctness.

Rebuke:

- Promoting `global_cooldown_bars=2` only because it is the best aggregate variant would be backtest tuning. It must be validated against strict manual fill timing and top-error rows first.

### Step 7 - Prove The Actual Causal Mechanism - DONE

Paths to use:

- `scripts/validate_backtest_live_reconciliation.py`
- `data/manual_exports`
- `data/new_expired_bots.xlsx`

Required next diagnostic:

- Restrict paired causal validation to strict manual-export geometric rows first.
- Separate rows with live strategy-linked orders/trades from rows without strict evidence.
- Compare per-row:
  - live trade timestamps
  - model fill timestamps
  - live maker/taker flags
  - model fee assumptions
  - live trade count
  - model trade count
  - live realized PnL
  - model realized and mark-to-market PnL
- Focus top-error rows first:
  - `CTSIUSDT`
  - `BSBUSDT`
  - `BULLAUSDT`
  - `PRLUSDT`
  - `NOMUSDT`
  - `TRUUSDT`
  - `DAMUSDT`

Acceptance condition:

- A candidate mechanism must improve mean and median absolute PnL error.
- It must not degrade winner recall or non-winner specificity.
- It must improve trade-count error or explain why live and model trade counts are not comparable.
- It must not reduce capital silently.
- It must be supported by strict manual rows, not only aggregate error.

Completed validation:

```text
strict manual-export geometric rows:      22
manual-export-or-input geometric rows:    23
manual-any mean trade rows:               67.043478
manual-any mean live-fill coverage:        0.311987
manual-any sub-120s gap share:             0.660605
current model sub-120s gap share:          0.592529
cooldown2 model sub-120s gap share:        0.372713
OLHC model sub-120s gap share:             0.786108
```

Corrected geometric baseline:

```text
current mean abs PnL error:          13.037998%
current median abs PnL error:         6.818518%
current sign match:                   0.654545
current non-winner specificity:       0.562500
current mean abs fill-count error:  122.709091
manual-any mean abs PnL error:       12.426296%
capital abs error:                    0.000000
```

Best aggregate diagnostic variant:

```text
variant:                            global_cooldown_bars=2
mean abs PnL error:                 10.371792%
median abs PnL error:                5.638167%
sign match:                          0.672727
non-winner specificity:              0.625000
mean abs fill-count error:         108.690909
manual-any mean abs PnL error:      11.241565%
capital abs error:                   0.000000
```

Critical causal finding:

- The `global_cooldown_bars=2` variant improves aggregate PnL and fill-count
  errors, but its fill timing is less like live strict evidence. Live manual rows
  have a `0.660605` sub-120-second gap share, current model has `0.592529`, and
  cooldown2 has only `0.372713`.
- Therefore cooldown2 is an improved diagnostic variant, but not a proven live
  mechanism. Promoting it would be parameter tuning, not verified realism.

Top-error risk-path finding:

- The largest current errors are correlated with model drawdown and risk path,
  not only fill count.
- `CTSIUSDT` is the worst error: live `+12.54%`, current model `-111.99%`,
  cooldown2 `-103.99%`.
- `CTSIUSDT` has manual bot text in `data/manual_input`, but no strict CSV
  account-trade export coverage. It can improve timing evidence, but it does
  not provide the same authenticated order/trade/transaction reconciliation as
  the CSV exports.
- A diagnostic `maintenance_margin_rate=0` test did not improve the geometric
  cohort, so disabling liquidation is not a valid fix.

Step 7 conclusion:

- The causal mechanism is not proven.
- The best tested variant is not structurally safe to promote.
- Proceeding to production Step 8 would require an assumption unless the user
  approves a lower proof standard or provides additional live evidence.

Additional reduced-cohort validation after removing the first top-error symbols:

Excluded symbols:

```text
CTSIUSDT
BSBUSDT
BULLAUSDT
PRLUSDT
NOMUSDT
TRUUSDT
DAMUSDT
```

Filtered cohort:

```text
original geometric rows:       55
excluded rows:                 11
tested rows:                   44
strict manual rows:            20
manual export/input rows:      20
```

Filtered current-model result:

```text
mean abs PnL error:             6.051523%
median abs PnL error:           4.422208%
PnL sign match:                 0.704545
winner recall, pnl > 0:         0.794118
non-winner specificity:         0.400000
mean abs fill-count error:     90.295455
capital abs error:              0.000000
```

Filtered variant comparison:

```text
variant                  mean abs PnL   median abs PnL   sign match   specificity   fill error
current                  6.051523%      4.422208%        0.704545     0.400000      90.295455
cooldown2                6.254584%      5.489321%        0.704545     0.400000      82.090909
close_only               6.762919%      4.715650%        0.772727     0.600000      89.840909
path_olhc                8.183684%      6.776253%        0.772727     0.400000     150.272727
path_olhc_cooldown2      9.723715%      6.112804%        0.750000     0.600000     104.022727
```

Remaining-top-error subset after first exclusions:

```text
included symbols:
AGTUSDT, BASUSDT, PIEVERSEUSDT, RAVEUSDT, JCTUSDT, AXSUSDT, DUSDT, ENJUSDT, CLOUSDT

rows tested:                   14
strict manual rows:             8
manual export/input rows:       8
```

Remaining-top-error subset result:

```text
variant                  mean abs PnL   median abs PnL   sign match
cooldown2                10.781025%     11.967933%       0.571429
current                  10.837719%     11.522065%       0.428571
close_only               10.982071%      6.903632%       0.642857
path_olhc                11.223429%      8.669383%       0.714286
path_olhc_cooldown2      15.345863%      9.689300%       0.642857
```

Reduced-cohort conclusion:

- Removing the first top-error symbols improves the current model substantially,
  but the remaining cohort still fails candidate-selection trustability because
  non-winner specificity is only `0.400000`.
- The remaining top-error subset does not pick a clean winner. `cooldown2`
  slightly improves mean PnL error but has weak sign match; `close_only` and
  `path_olhc` improve sign match but worsen mean PnL error.
- Therefore the reduced-cohort test does not remove the Step 8 blocker.

### Step 8 - Implement Optional Manual-Export Seed Realism - DONE

Paths modified:

- `backtest/btk_seed_state.py`
- `backtest/btk_replay_seed_loader.py`
- `backtest/backtest_realistic.py`
- `tests/unit/test_btk_seed_state.py`

Implemented:

- Added direct Binance `Order history*.csv` seed loading as an adapter into the
  existing `SeedState` interface.
- Preserved existing replay `levels.csv` seed loading.
- Filtered seed evidence by exact `symbol`, optional exact `strategy_id`, and
  active order window at bot start.
- Added seed metadata:
  - `qty_per_order`
  - `qty_source`
  - `evidence_class`
  - `source`
- Seeded engine runs now restrict initial active levels to verified open ladder
  levels. Unseeded runs preserve legacy level availability semantics.
- Verified live order quantity overrides model sizing only when exactly one
  positive per-order quantity is present. Conflicting quantities fail closed to
  model sizing and are labelled `conflicting`.

Validation proof:

```text
python -m pyright backtest/btk_replay_seed_loader.py
result: 0 errors, 0 warnings, 0 informations

python -m pytest -q tests/unit/test_btk_seed_state.py
result: 22 passed

python -m pyright backtest/backtest_realistic.py backtest/btk_seed_state.py backtest/btk_replay_seed_loader.py
result: 0 errors, 0 warnings, 0 informations

python -m pytest -q tests/unit/test_btk_seed_state.py tests/unit/test_btk_label_runner.py
result: 66 passed
```

Rebuke:

- This is not a new engine.
- This is not a scanner, ranking, training, or artifact change.
- This does not promote `global_cooldown_bars=2`, close-only fills, OHLC/OLHC
  path order, or liquidation disablement.
- The loader can scan many symbols, but each backtest run receives only the
  exact matching seed state for one bot.

### Step 9 - Validate Candidate-Selection Trustability - DONE FOR SEEDED EVIDENCE PATH

Path modified:

- `scripts/validate_backtest_live_reconciliation.py`

Implemented:

- Added `--seed-from-manual-exports`.
- Added order-history UTC start anchoring when strict strategy evidence exists.
- Added evidence classification:
  - `complete`
  - `partial`
  - `missing`
  - `conflicting`
- Added seed diagnostics:
  - `seed_state_source`
  - `seed_evidence_class`
  - `seed_active_level_count`
  - `seed_qty_per_order`
  - `seed_qty_source`
  - `model_position_size_source`
- Added seeded subset summary metrics and top-error reporting.

Validation proof:

```text
python -m pyright scripts/validate_backtest_live_reconciliation.py backtest/backtest_realistic.py backtest/btk_replay_seed_loader.py
result: 0 errors, 0 warnings, 0 informations

python -m pytest -q tests/unit/test_btk_seed_state.py tests/unit/test_btk_label_runner.py tests/unit/test_candidate_pipeline_bypass.py
result: 74 passed
```

Geometric all-row baseline, unseeded:

```text
rows:                              57
model rows:                        55
missing kline rows:                 2
mean abs PnL error:                13.037998%
median abs PnL error:               6.818518%
PnL sign match:                     0.654545
winner recall, pnl > 0:             0.717949
non-winner specificity:             0.562500
fast-winner recall:                 0.717949
mean abs trade-count error:       118.763636
capital abs error mean:             0.000000
MAE pct initial mean abs error:    16.972081%
```

Geometric all-row seeded/manual-export validation:

```text
rows:                              57
model rows:                        55
seeded model rows:                 55
missing kline rows:                 2
evidence class counts:
  complete:                        19
  partial:                         36
mean abs PnL error:                 4.418476%
median abs PnL error:               2.011418%
PnL sign match:                     0.872727
winner recall, pnl > 0:             0.846154
non-winner specificity:             0.937500
fast-winner recall:                 0.769231
mean abs trade-count error:        89.381818
capital abs error mean:             0.000000
MAE pct initial mean abs error:     5.263589%
position size source:               order_history for 55/55 seeded model rows
```

DOGEUSDT `strategy_id=411991896` focused validation with scratch-only kline cache:

```text
live PnL:                           2.720000%
seeded model PnL:                   3.217126%
absolute PnL error:                 0.497126%
live fill count:                    5
model trade count:                 10
manual export trade rows:           2
evidence class:                     partial
position size source:               order_history
missing kline rows:                 0
```

Readiness conclusion:

- The seeded/manual-export path materially improves geometric realism.
- It improves average error, median error, sign match, winner recall,
  non-winner specificity, fast-winner recall, trade-count error, and MAE error.
- It does not improve by hidden capital shrink; capital error remains zero.
- It is not enough to mark the whole geometric model fully trustable because
  only 19 of 55 modelable seeded rows are currently classified as complete.

### Step 10 - Mark Goal Achieved Only After Production-Safe Validation - PARTIAL

The optional seeded evidence path is achieved:

- implementation path stays inside the existing backtest architecture;
- normal unseeded backtests are unchanged unless `seed_state` is explicitly supplied;
- no scanner, training, deployment, workbook, or model artifacts are silently changed;
- pyright passes;
- focused unit tests pass;
- isolated scratch validation was run.

The full goal remains partially open:

- More complete live evidence is required before every geometric validation row
  can be treated as trustable.
- Rows classified as `partial`, `missing`, or `conflicting` remain diagnostic
  only.
- Scanner or ranking changes remain out of scope.
- Retraining remains out of scope.

### Step 11 - Candidate-Time Realism Profile - DONE, NOT PROMOTED

Paths modified:

- `backtest/btk_seed_state.py`
- `backtest/btk_unified_runner.py`
- `src/neutralgrid/backtest/candidate_pipeline.py`
- `backtest_candidates.py`
- `scripts/validate_backtest_live_reconciliation.py`
- `tests/unit/test_btk_seed_state.py`
- `tests/unit/test_btk_label_runner.py`
- `tests/unit/test_candidate_pipeline_bypass.py`

Implemented:

- Added explicit realism profiles:
  - `legacy`
  - `candidate_time_geometric_v1`
- Kept default behavior as `legacy`.
- Added a candidate-time geometry seed helper that uses only:
  - configured grid levels,
  - first replay close,
  - symbol,
  - timestamp.
- The candidate-time seed labels source as `candidate_time_geometry`.
- The candidate-time seed keeps quantity model-sized and does not copy
  manual-export quantities.
- `backtest_candidates.py` now accepts:

```text
--realism-profile legacy|candidate_time_geometric_v1
```

Validation proof:

```text
python -m pyright backtest/btk_seed_state.py backtest/btk_unified_runner.py src/neutralgrid/backtest/candidate_pipeline.py backtest_candidates.py scripts/validate_backtest_live_reconciliation.py
result: 0 errors, 0 warnings, 0 informations

python -m pytest -q tests/unit/test_btk_seed_state.py tests/unit/test_btk_label_runner.py tests/unit/test_candidate_pipeline_bypass.py
result: 82 passed
```

Isolated holdout validation:

```text
scope: all geometric rows
split: chronological
scratch cleanup: SCRATCH_EXISTS_AFTER_CLEANUP=False
```

Holdout results:

```text
legacy:
  rows:                              23
  model rows:                        21
  missing kline rows:                 2
  mean abs PnL error:                13.642216%
  median abs PnL error:               8.201958%
  PnL sign match:                     0.571429
  winner recall, pnl > 0:             0.615385
  non-winner specificity:             0.500000
  mean abs trade-count error:        88.857143
  capital abs error mean:             0.000000

candidate_time_geometric_v1:
  rows:                              23
  model rows:                        21
  missing kline rows:                 2
  mean abs PnL error:                13.642216%
  median abs PnL error:               8.201958%
  PnL sign match:                     0.571429
  winner recall, pnl > 0:             0.615385
  non-winner specificity:             0.500000
  mean abs trade-count error:        88.857143
  capital abs error mean:             0.000000

manual_order_history seeded validation:
  rows:                              23
  model rows:                        21
  missing kline rows:                 2
  mean abs PnL error:                 2.368012%
  median abs PnL error:               2.011418%
  PnL sign match:                     1.000000
  winner recall, pnl > 0:             1.000000
  non-winner specificity:             1.000000
  mean abs trade-count error:        76.285714
  capital abs error mean:             0.000000
```

Promotion decision:

- `candidate_time_geometric_v1` is not promoted because holdout mean absolute
  PnL error and median absolute PnL error did not improve versus `legacy`.
- Sign match, winner recall, non-winner specificity, and capital error did not
  degrade, but the promotion rule required PnL-error improvement.
- Manual-order-history seeded validation improved materially, but it remains
  an upper-bound validation path because it uses live order data unavailable to
  future candidate selection.

Rebuke:

- Promoting `candidate_time_geometric_v1` as the default would be false
  optionality as written because it did not improve holdout error.
- Promoting manual-order-history seeding for future candidates would be
  invalid because it would use future live data.

## Provable False Optionality

- Full 221-row validation as the acceptance set: false for this goal. It includes 166 arithmetic rows.
- New backtest engine: false. The existing `RealisticGridBacktester` and unified runner are the correct integration points.
- New training pipeline: false. The issue is geometric simulation realism, not a training ingestion architecture gap.
- New label contracts or versions: false. The user explicitly ruled them out.
- New model feature for `mode`: false. `mode` is a simulation contract field, not a predictive feature.
- Retraining as the fix: false. Retraining cannot make labels trustworthy if geometric backtest physics still diverge from live behavior.
- Promoting `global_cooldown_bars=2` solely because it improves aggregate error: false. It improves aggregate metrics but does not match live sub-120-second fill timing well enough.
- Treating manual export trade-row count as a direct replacement for workbook `total_trades`: false as written. Prior validation showed the units are not stable across rows.
- Treating OHLC or OLHC path as live truth: false. They are diagnostic envelopes only because 1-minute OHLC does not reveal high/low order.
- Treating Binance `/fapi/v1/depth` as a historical order-book source: false. The official request has `symbol` and `limit`, not `startTime` or `endTime`; it cannot reconstruct expired-bot queues unless this repo already archived historical snapshots or websocket diffs.
- Treating a queue-position model as valid without depth/trade-sequence input: false. Queue position is a market-microstructure state. It cannot be derived from a workbook PnL row or a 1-minute OHLC candle without adding assumptions.
- Treating an external replay engine as a shortcut to trustability: false. A replay engine can help only if the required historical depth, trade, latency, and account-order evidence exists. Without that evidence it adds complexity without adding proof.

## Provably Unnecessary Items As Written

- A broad downloader for every Binance endpoint is unnecessary. Only missing evidence tied to geometric PnL, fills, fees, funding, filters, mark price, and liquidation should be fetched.
- A new workbook mode backfill script is unnecessary. The workbook already has a `mode` column.
- A new scanner gate is unnecessary. Scanner gates do not fix incorrect geometric PnL labels.
- A new threshold optimizer is unnecessary. Optimizing thresholds over inaccurate labels creates a new problem.
- A blanket mark-price replacement is unnecessary. Mark price is needed for valuation, liquidation, and funding checks, not for trade-trigger OHLC.
- A production dependency on HftBacktest or another replay engine is unnecessary for the next step. The next verifiable step is evidence classification and strict diagnostic reconciliation inside the existing harness.
- A new training feature for evidence completeness is unnecessary and risky. Evidence completeness is an audit/classification field for validation quality, not a predictive property of the market.
- A blanket production default change before strict manual-row causal validation is unnecessary and unsafe.

## Items Not Valid To Strike Without Assumptions

- Live seed state from orders, positions, or replay exports.
- Account-trade timestamps.
- Maker/taker classification.
- Commission and realized PnL from live account trades.
- Tick size, step size, and minimum notional.
- Mark-price valuation and liquidation checks.
- Funding history inside the bot window.
- Intrabar high/low path ambiguity.
- Geometric non-winner rows.
- Historical order-book snapshots or websocket depth diffs, if already archived.
- Account-order creation, cancellation, and replacement timestamps.
- Latency and replacement-delay evidence, if present in live exports.

These items cannot be struck because each can change PnL, drawdown, trade count, deployed capital, or candidate-selection accuracy.

## Current Consensus

The geometric-only target is correctly scoped. The manual-export seed path
improves realism without corrupting the pipeline because it is opt-in and uses
the existing `SeedState` interface. The candidate-time profile is implemented
and tested, but it is diagnostic-only because holdout PnL error did not improve
over `legacy`. The additional web and research review does not justify a
production engine replacement, scanner change, retrain, tuned parameter
promotion, or default profile switch. The next safe optimization remains
evidence completeness: account trades for fills/fees/realized PnL, regular
klines for price-touch ranges, mark-price klines for valuation/liquidation/
funding checks, funding history for funding charges, exchange filters for
rounding, and archived depth/trade feeds only if they actually exist. Full
candidate-selection trustability remains evidence-gated because many rows are
still partial.
