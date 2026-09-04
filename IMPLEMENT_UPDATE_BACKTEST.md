Progress: [##########] 100% (8/8 implementation checkpoints complete; candidate-time public-market profile remains diagnostic-only)

# Backtest Realism Integration Plan

Goal: separate seeded backtest realism from full-pipeline candidate-selection
realism, then implement only verifiable changes that use available data without
future leakage, silent drift, or model corruption.

Status: implemented and validated on 2026-05-17. The implementation is
complete, but the promotion gate did not pass for the candidate-time
public-market profile, so default full-pipeline behavior remains unchanged.

Repository root:

```text
C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.7
```

## Non-Negotiable Boundary

There are two different backtest problems. They must remain separate.

1. Seeded backtest for live telemetry reconciliation:
   - Purpose: make historical or active bot reconciliation more realistic after
     a bot exists.
   - Valid inputs: live telemetry, order history, trade history, transaction
     history, setup screenshots, order-book screenshots, workbook outcomes.
   - Invalid claim: this directly improves future candidate selection.

2. Candidate-time backtest for full-pipeline candidate selection:
   - Purpose: improve the backtest labels and validation used by future
     candidate selection.
   - Valid inputs: data available at candidate time or during the public
     forward backtest window, such as scanner row features, candidate grid
     geometry, public klines, mark-price klines, funding history, exchange
     filters, and prospectively archived public market data.
   - Invalid inputs: future order history, future trade history, future
     transaction history, realized live PnL, screenshots from a bot that has
     not yet existed, or any post-selection live ladder state.

## Current Architecture Proof

These are verified local paths and code responsibilities.

- `run_full_pipeline.py`
  - `run_full_pipeline.py:269-323` ranks deployment candidates using
    `grid_is_valid`, `meta_prob`, `ev_score`, and `score`.
  - It does not pass manual-export `SeedState` into candidate ranking.

- `src/neutralgrid/backtest/candidate_pipeline.py`
  - `src/neutralgrid/backtest/candidate_pipeline.py:556-678` extracts
    exchange filters, attaches mark-price closes, and converts funding rows
    into an explicit funding-series status.
  - `src/neutralgrid/backtest/candidate_pipeline.py:792-1072` runs one
    candidate backtest through `backtest.btk_unified_runner.run_backtest()`.
  - `replay_seed_dir` is optional and explicit.
  - `realism_profile` defaults to `legacy`, preserving default behavior.
  - `src/neutralgrid/backtest/candidate_pipeline.py:1090` serializes
    backtest results into training rows with provenance fields.

- `backtest/btk_unified_runner.py`
  - `backtest/btk_unified_runner.py:138-270` is the unified runner that creates
    and annotates engine results.
  - External `seed_state` and non-legacy realism profiles cannot be combined.
  - `candidate_time_public_market_v1` remains explicit and opt-in.

- `backtest/btk_replay_seed_loader.py`
  - `backtest/btk_replay_seed_loader.py:182-387` loads manual order-history
    seed state by exact symbol, optional strategy id, and active-at-start
    window.
  - This is a seeded-reconciliation adapter, not a candidate-time input.

- `backtest/backtest_realistic.py`
  - `backtest/backtest_realistic.py:49-115` defines `GridConfig`, including
    geometric mode, tick/step filters, funding series, and valuation source.
  - `backtest/backtest_realistic.py:436-1001` executes the grid replay,
    keeps last-price fills separate from optional mark-price valuation, and
    reports fill/valuation provenance.

- `backtest/btk_label_contract.py`
  - `backtest/btk_label_contract.py:71-146` defines current training engine
    defaults.
  - No new label contract or model artifact promotion is part of this plan.

- `src/neutralgrid/training/unified_training_builder.py`
  - `src/neutralgrid/training/unified_training_builder.py:1218-1373`
    applies ingestion gates and public-market provenance checks.
  - `src/neutralgrid/training/unified_training_builder.py:1645` preserves
    public-market provenance columns without turning them into model features.

## Available Data Inventory

Use these data sources only according to the track where they are valid.

### Historical Live Outcome Data

- `data/new_expired_bots.xlsx`
  - Verified sheets: `General`, `PnL Curve Features`, `Meta Features`.
  - Relevant `General` columns include `strategy_id`, `symbol`,
    `start_time_utc`, `end_time_utc`, `duration_hours`,
    `invested_margin_usdt`, `leverage`, `grids_count`,
    `price_range_low`, `price_range_high`, `pnl_pct`, `total_trades`,
    `mae`, `mfe`, and `mode`.
  - Valid for validation and label-quality measurement.
  - Not valid as an input feature for future candidates except where the same
    field is already known at candidate time.

### Manual Export Data

- `data/manual_exports/Order History*.csv`
  - Available columns include order time, order id, symbol, side, price,
    amount, executed amount, status, and update time.
  - `Order history 4.csv` through `Order history 8.csv` include `Strategy Id`.
  - `Order History.csv` and `Order History 1.csv` through
    `Order History 3.csv` do not include `Strategy Id`, so exact bot seeding
    cannot be proven from those files alone.

- `data/manual_exports/Trade History*.csv`
  - Available columns include time, symbol, side, price, quantity, fee,
    realized profit, maker flag, trade id, and order id.
  - Valid for historical reconciliation and maker/taker validation.

- `data/manual_exports/Transaction History*.csv`
  - Available columns include UTC date, type, amount, asset, symbol, and
    transaction id.
  - Valid for historical realized PnL, commission, and funding reconciliation.

### Public Binance Data Already Reachable In Code

- `src/neutralgrid/api/binance_client.py:387-451`
  - `get_klines()` for last-price klines.

- `src/neutralgrid/api/binance_client.py:453-493`
  - `get_mark_price_klines()` for mark-price klines.

- `src/neutralgrid/api/binance_client.py:531-557`
  - `get_funding_rate()` for funding history.

- `src/neutralgrid/api/binance_client.py:654-663`
  - `get_exchange_info()` for symbol filters and exchange rules.

- `src/neutralgrid/api/binance_client.py:570-584`
  - `get_order_book()` for current public order book snapshots.
  - Current order book snapshots are not historical replay data by themselves.

### Authenticated Live Data Already Reachable In Code

- `src/neutralgrid/api/binance_client.py:706-739`
  - `get_user_trades()` for account trade fills.

- `src/neutralgrid/api/binance_client.py:947-986`
  - `get_grid_bot_session_data()` for user trades, realized PnL, commissions,
    funding fees, and mark klines.

Authenticated live data is valid for live telemetry reconciliation after a bot
exists. It is not a candidate-time input for a future bot.

## External Evidence Guardrails

- Binance USD-M `exchangeInfo` documents symbol trading rules and warns that
  `pricePrecision` and `quantityPrecision` must not be used as tick size or
  step size substitutes:
  https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Exchange-Information

- Binance USD-M public endpoints exist for last-price klines, mark-price
  klines, and funding history:
  https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data
  https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Mark-Price-Kline-Candlestick-Data
  https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History

- Binance USD-M aggregate trades are public, but documented historical access
  is limited; therefore they are valid for prospective capture or recent
  windows, not for silently reconstructing old candidates where data is no
  longer available:
  https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Compressed-Aggregate-Trades-List

- HftBacktest documents that market replay assumes orders do not alter the
  replayed market, and fill realism depends on explicit exchange and queue
  assumptions. This supports explicit fill-model labeling rather than silent
  behavior changes:
  https://hftbacktest.readthedocs.io/en/py-v2.1.0/order_fill.html

- Bailey, Borwein, Lopez de Prado, and Zhu warn that repeated backtest tuning
  can create false positives. This plan uses chronological validation and
  refuses promotion based only on a tuned cohort:
  https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf

## Track A: Seeded Backtest For Live Telemetry Reconciliation

Purpose: improve historical or active bot reconciliation by using evidence that
exists only after a bot exists.

This track may improve live telemetry scanner analysis, but it must not be used
to claim that future deployment candidates are more accurate.

### A1. Preserve Seed Loader As Adapter - DONE

Paths:

- `backtest/btk_replay_seed_loader.py`
- `backtest/btk_seed_state.py`
- `backtest/backtest_realistic.py`
- `tests/unit/test_btk_seed_state.py`

Plan:

- Keep manual-export loading filtered by exact `symbol`, `strategy_id`, and
  UTC time window.
- Keep support for replay `levels.csv`.
- Keep seed state classified as `complete`, `partial`, `missing`, or
  `conflicting`.
- Do not infer missing ladder state.
- Do not create model features from manual-export seed fields.

Verification:

```text
python -m pyright backtest/btk_replay_seed_loader.py backtest/btk_seed_state.py backtest/backtest_realistic.py
python -m pytest -q tests/unit/test_btk_seed_state.py
```

Acceptance:

- Loader accepts many-symbol files but returns only the exact bot state.
- Rows without exact strategy evidence are not upgraded to complete evidence.
- Seeded mode remains opt-in.
- Unseeded mode is unchanged.

### A2. Reconcile Order, Trade, And Transaction Evidence - DONE

Paths:

- `scripts/validate_backtest_live_reconciliation.py`
- `data/manual_exports/`
- `data/new_expired_bots.xlsx`

Plan:

- Use order history to reconstruct verified active ladder at bot start.
- Use trade history to validate fill count, maker/taker mix, fees, quantity,
  and realized profit.
- Use transaction history to validate commission, funding fee, and realized
  PnL totals.
- Treat screenshot inputs as evidence only after the user supplies them and
  after the parser stores their extracted values with provenance.

Verification:

```text
python -m pyright scripts/validate_backtest_live_reconciliation.py backtest/btk_replay_seed_loader.py
python -m pytest -q tests/unit/test_btk_seed_state.py tests/unit/test_btk_label_runner.py
```

Acceptance:

- Report includes evidence class, seed source, position-size source, PnL error,
  trade-count error, max DD where available, capital used, and top-error rows.
- Scratch validation output is written only to an isolated scratch directory.
- Scratch output is deleted and cleanup is verified.

### A3. Live Telemetry Boundary - DONE

Paths:

- `src/neutralgrid/live/decision/`
- `src/neutralgrid/api/binance_client.py`
- `backtest/btk_seed_state.py`

Plan:

- Live telemetry may call seeded reconciliation only when a bot has a real
  `symbol`, `strategy_id`, and verified live/manual evidence.
- Do not change scanner admission, ranking, deploy-ready output, HMM,
  meta-labeler, utility, or model artifacts in this track.

Verification:

```text
python -m pyright src/neutralgrid/live/decision/ backtest/btk_seed_state.py
python -m pytest -q tests/unit/test_decision_loader.py tests/unit/test_decision_recommender.py
```

Acceptance:

- Any live telemetry use is advisory/reconciliation only.
- No full-pipeline candidate output changes because of seeded evidence.

## Track B: Candidate-Time Backtest For Full-Pipeline Candidate Selection

Purpose: improve future candidate-selection trustability using only information
available to the model at candidate time or during the public forward backtest
window.

This is the only track that can eventually improve full-pipeline candidate
accuracy without future leakage.

### B1. Establish Candidate-Time Data Contract - DONE

Paths:

- `src/neutralgrid/backtest/candidate_pipeline.py`
- `backtest/btk_unified_runner.py`
- `backtest/btk_label_contract.py`
- `backtest_candidates.py`
- `tests/unit/test_candidate_pipeline_bypass.py`
- `tests/unit/test_btk_label_runner.py`

Plan:

- Define the candidate-time input whitelist:
  - candidate row fields already produced by scanner/enrichment;
  - grid geometry: lower, upper, grids, mode, leverage, capital fraction;
  - public last-price klines for the forward backtest window;
  - public mark-price klines for valuation or liquidation diagnostics;
  - public funding-rate history for the forward backtest window;
  - exchange filters from `exchangeInfo`;
  - prospectively archived public market data captured at or after scan time.
- Define the forbidden future-data list:
  - manual order history for that future candidate;
  - manual trade history for that future candidate;
  - manual transaction history for that future candidate;
  - realized live PnL;
  - live ladder state after deployment;
  - screenshots of a bot that has not yet existed.

Verification:

```text
python -m pyright src/neutralgrid/backtest/candidate_pipeline.py backtest/btk_unified_runner.py backtest_candidates.py
python -m pytest -q tests/unit/test_candidate_pipeline_bypass.py tests/unit/test_btk_label_runner.py
```

Acceptance:

- Candidate-time tests fail if manual-export data can enter the profile.
- Default `legacy` behavior remains unchanged until explicit promotion is
  separately approved.

### B2. Apply Exchange Filter Realism In Candidate-Time Backtests - DONE

Paths:

- `src/neutralgrid/api/binance_client.py`
- `src/neutralgrid/backtest/candidate_pipeline.py`
- `backtest/backtest_realistic.py`
- `backtest/btk_unified_runner.py`
- `tests/unit/test_btk_label_runner.py`
- `tests/unit/test_candidate_pipeline_bypass.py`

Plan:

- Fetch `exchangeInfo` for each candidate symbol through the existing client.
- Extract `PRICE_FILTER.tickSize`, `LOT_SIZE.stepSize`, and notional minimum
  from symbol filters.
- Pass verified `tick_size` and `step_size` into `GridConfig`.
- If tick rounding collapses grid levels, fail closed; do not repair by
  silently changing grid bounds or grid count.
- If rounded quantity violates lot or notional constraints, classify the row as
  invalid for the candidate-time profile and report the reason.

Verification:

```text
python -m pyright src/neutralgrid/backtest/candidate_pipeline.py backtest/backtest_realistic.py backtest/btk_unified_runner.py
python -m pytest -q tests/unit/test_btk_label_runner.py tests/unit/test_candidate_pipeline_bypass.py
```

Acceptance:

- `pricePrecision` and `quantityPrecision` are not used as tick/step
  substitutes.
- The profile records filter provenance.
- Invalid exchange-filter rows fail closed with an explicit reason.
- No existing default full-pipeline output changes unless the explicit
  candidate-time profile is selected.

### B3. Use Public Funding Series Without Silent Fallback - DONE

Paths:

- `src/neutralgrid/api/binance_client.py`
- `src/neutralgrid/backtest/candidate_pipeline.py`
- `backtest/backtest_realistic.py`
- `backtest/btk_label_contract.py`
- `tests/unit/test_btk_label_runner.py`

Plan:

- Use `get_funding_rate()` for the symbol and forward backtest window.
- Convert returned rates into the engine's `funding_rate_series` shape.
- If no funding event exists inside a short 6h window, record
  `funding_series_status=no_event_in_window`.
- If the API is unavailable, record `funding_series_status=missing` and do not
  silently pretend the static rate is verified series data.
- Keep static funding as the legacy/default path unless explicit candidate-time
  public-market realism is selected.

Verification:

```text
python -m pyright src/neutralgrid/backtest/candidate_pipeline.py backtest/backtest_realistic.py backtest/btk_label_contract.py
python -m pytest -q tests/unit/test_btk_label_runner.py tests/unit/test_candidate_pipeline_bypass.py
```

Acceptance:

- Result rows distinguish static funding, verified funding series, no event,
  and missing funding.
- No row silently changes funding semantics without recorded provenance.

### B4. Separate Last-Price Fills From Mark-Price Valuation - DONE

Paths:

- `src/neutralgrid/backtest/candidate_pipeline.py`
- `backtest/backtest_realistic.py`
- `tests/unit/test_btk_label_runner.py`

Plan:

- Continue to use last-price klines for grid level touch/fill logic unless a
  separately validated fill source is added.
- Use mark-price klines for valuation, drawdown, liquidation, or diagnostic
  comparison only when the data is fetched for the same window.
- Do not mix last and mark prices without result columns that state which
  series drove fills and which series drove valuation.

Verification:

```text
python -m pyright src/neutralgrid/backtest/candidate_pipeline.py backtest/backtest_realistic.py
python -m pytest -q tests/unit/test_btk_label_runner.py
```

Acceptance:

- Result rows include explicit `fill_price_source` and `valuation_price_source`
  or equivalent fields.
- Missing mark-price data does not silently alter fill logic.

### B5. Prospective Public Market Capture For Future Validation - DONE

Paths:

- `src/neutralgrid/api/binance_client.py`
- `src/neutralgrid/backtest/candidate_pipeline.py`
- `data/`

Plan:

- Do not claim historical depth replay unless data was actually captured.
- For future candidates, optionally archive public order book snapshots and
  recent aggregate trades at scan/backtest time with timestamp and symbol
  provenance.
- Use archived public data first as validation evidence, not as a promoted
  fill model.
- Promote no queue or fill assumption unless chronological validation proves it
  improves candidate-time metrics.

Verification:

```text
python -m pyright src/neutralgrid/api/binance_client.py src/neutralgrid/backtest/candidate_pipeline.py
python -m pytest -q tests/unit/test_candidate_pipeline_bypass.py
```

Acceptance:

- Archived public market data never contains manual order/trade/transaction
  exports.
- Missing archived data is classified as missing, not inferred.

### B6. Chronological Holdout Validation - DONE

Paths:

- `scripts/validate_backtest_live_reconciliation.py`
- `data/new_expired_bots.xlsx`
- `.codex_scratch/`

Plan:

- Validate on geometric rows only unless the user explicitly asks for
  arithmetic rows.
- Compare:
  - legacy candidate-time baseline;
  - candidate-time public-market profile;
  - seeded/manual-export profile as historical upper bound only.
- Split chronologically so tuning cannot learn from future rows.
- Report mean absolute PnL error, median absolute PnL error, sign match,
  winner recall, non-winner specificity, fast-winner recall, trade-count error,
  capital error, evidence class counts, and top-error rows.
- Delete scratch output and verify cleanup.

Verification:

```text
python scripts/validate_backtest_live_reconciliation.py --scope all --mode-filter geometric --duration-source workbook --validation-split chronological
python scripts/validate_backtest_live_reconciliation.py --scope all --mode-filter geometric --duration-source workbook --validation-split chronological --seed-from-manual-exports
```

Acceptance:

- Candidate-time profile must improve holdout mean absolute PnL error.
- Candidate-time profile must improve or preserve holdout median absolute PnL
  error.
- Candidate-time profile must not degrade winner recall or non-winner
  specificity.
- Seeded/manual-export improvement alone is not sufficient for promotion.

### B7. Training And Classification Hygiene - DONE

Paths:

- `src/neutralgrid/training/unified_training_builder.py`
- `src/neutralgrid/backtest/candidate_pipeline.py`
- `tests/unit/test_unified_training_builder.py`
- `tests/unit/test_candidate_pipeline_bypass.py`

Plan:

- Preserve deduplication by `candidate_id`.
- Keep raw backtest metrics separate from adjusted or candidate-time-profile
  metrics.
- Add provenance fields for any candidate-time profile output.
- Reject or quarantine rows with missing required provenance.
- Do not add `mode`, seed evidence, manual-export flags, or live-result fields
  as model features unless a separate leakage review proves they are valid
  candidate-time features.

Verification:

```text
python -m pyright src/neutralgrid/training/unified_training_builder.py src/neutralgrid/backtest/candidate_pipeline.py
python -m pytest -q tests/unit/test_unified_training_builder.py tests/unit/test_candidate_pipeline_bypass.py
```

Acceptance:

- No manual-export leakage into training features.
- No duplicate candidate labels survive deduplication.
- Adjusted metrics cannot overwrite raw metrics silently.

### B8. Promotion Decision Gate - DONE

Paths:

- `IMPLEMENT_UPDATE_BACKTEST.md`
- `BACKTEST_FIX.md`
- `CHANGELOG.md`
- `backtest_candidates.py`
- `src/neutralgrid/backtest/candidate_pipeline.py`
- `run_full_pipeline.py`

Plan:

- Keep new candidate-time realism opt-in until validation passes.
- Do not change `run_full_pipeline.py` ranking, scanner admission, HMM,
  meta-labeler, utility, deploy-ready CSV generation, workbook cells, or model
  artifacts during the first implementation.
- If validation passes, ask the user before making the profile default or
  wiring it into full-pipeline candidate output.
- If validation fails, keep the profile diagnostic-only and record the failure.

Verification:

```text
python -m pyright backtest/backtest_realistic.py backtest/btk_unified_runner.py src/neutralgrid/backtest/candidate_pipeline.py backtest_candidates.py scripts/validate_backtest_live_reconciliation.py
python -m pytest -q tests/unit/test_btk_seed_state.py tests/unit/test_btk_label_runner.py tests/unit/test_candidate_pipeline_bypass.py tests/unit/test_unified_training_builder.py
git diff -- IMPLEMENT_UPDATE_BACKTEST.md BACKTEST_FIX.md CHANGELOG.md backtest/backtest_realistic.py backtest/btk_unified_runner.py src/neutralgrid/backtest/candidate_pipeline.py backtest_candidates.py scripts/validate_backtest_live_reconciliation.py tests/unit/test_btk_seed_state.py tests/unit/test_btk_label_runner.py tests/unit/test_candidate_pipeline_bypass.py tests/unit/test_unified_training_builder.py
```

Acceptance:

- The user can review exactly which paths changed.
- The default full-pipeline output remains unchanged unless separately
  authorized.
- Any promoted behavior has chronological holdout proof.

## Implementation Validation Results

These results came from isolated scratch runs. The scratch directories were
deleted after metrics were collected:

```text
.codex_scratch exists=False
.codex_scratch\backtest_update_validation exists=False
.codex_scratch\doge_focus_validation exists=False
```

### Latest-20 Geometric Workbook Cohort

Scope:

```text
scripts/validate_backtest_live_reconciliation.py --scope latest20 --mode-filter geometric --duration-source workbook --validation-split chronological
```

Result:

| profile | rows | model rows | missing klines | mean abs PnL error | median abs PnL error | sign match | winner recall | fast-winner recall | mean abs trade-count error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `legacy` | 19 | 17 | 2 | 6.368633907600 | 3.800947130855 | 0.764705882353 | 0.764705882353 | 0.764705882353 | 93.000000000000 |
| `candidate_time_public_market_v1` | 19 | 17 | 2 | 9.059605185196 | 6.819644101974 | 0.764705882353 | 0.764705882353 | 0.705882352941 | 91.764705882353 |
| seeded/manual-export upper bound | 19 | 17 | 2 | 1.985181973495 | 1.732114138521 | 1.000000000000 | 1.000000000000 | 0.882352941176 | 74.235294117647 |

Validation decision:

- `candidate_time_public_market_v1` failed the promotion gate because mean abs
  PnL error, median abs PnL error, and fast-winner recall degraded versus
  `legacy`.
- Seeded/manual-export validation improved substantially, but it remains an
  historical reconciliation upper bound because manual exports are unavailable
  for future candidates.
- Default full-pipeline output remains unchanged.

### Chronological Holdout Subset

| profile | holdout model rows | holdout mean abs PnL error | holdout median abs PnL error | holdout sign match |
|---|---:|---:|---:|---:|
| `legacy` | 6 | 8.123829630140 | 6.520473565427 | 0.666666666667 |
| `candidate_time_public_market_v1` | 6 | 9.522266958055 | 7.757316340001 | 0.500000000000 |
| seeded/manual-export upper bound | 6 | 1.534625170670 | 1.763249572998 | 1.000000000000 |

Validation decision:

- The candidate-time public-market profile also failed the holdout gate.
- This prevents silent promotion into the full pipeline.
- Review pass found and fixed one diagnostic reporting bug: validation
  summaries now mark public-market evidence as requested whenever
  `candidate_time_public_market_v1` is selected.

### Focused DOGEUSDT `strategy_id=411991896`

Scope:

```text
scripts/validate_backtest_live_reconciliation.py --include-symbol DOGEUSDT --mode-filter geometric --duration-source workbook --validation-split chronological
```

Result:

| profile | validation start UTC | live PnL | model PnL | abs PnL error | model max DD | live trades | model trades | evidence class |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `legacy` | 2026-05-13 17:12:25+00:00 | 2.720000 | 1.153360 | 1.566640 | 4.063080 | 5 | 8 | partial |
| `candidate_time_public_market_v1` | 2026-05-13 17:12:25+00:00 | 2.720000 | 2.417570 | 0.302430 | 1.586589 | 5 | 7 | partial |
| seeded/manual-export | 2026-05-13 22:12:25+00:00 | 2.720000 | 3.217126 | 0.497126 | 2.397686 | 5 | 10 | partial |

DOGE-specific validation decision:

- `candidate_time_public_market_v1` improved the DOGE focused row versus
  `legacy`.
- This single-symbol improvement is not enough to promote the profile because
  the latest-20 geometric cohort and chronological holdout were worse.
- The five-hour difference between workbook start and seeded/manual-export
  start is recorded as a data-alignment fact. Binance CSV columns labeled UTC
  were not shifted by UTC-5.

### Verification Commands

```text
python -m pyright backtest/btk_replay_seed_loader.py backtest/btk_seed_state.py backtest/backtest_realistic.py
python -m pytest -q tests/unit/test_btk_seed_state.py
python -m pyright src/neutralgrid/backtest/candidate_pipeline.py backtest/backtest_realistic.py backtest/btk_unified_runner.py backtest/btk_label_contract.py backtest_candidates.py
python -m pytest -q tests/unit/test_btk_label_runner.py tests/unit/test_candidate_pipeline_bypass.py
python -m pyright scripts/validate_backtest_live_reconciliation.py src/neutralgrid/training/unified_training_builder.py src/neutralgrid/backtest/candidate_pipeline.py backtest/backtest_realistic.py backtest/btk_unified_runner.py
python -m pytest -q tests/unit/test_unified_training_builder.py tests/unit/test_candidate_pipeline_bypass.py tests/unit/test_btk_label_runner.py tests/unit/test_btk_seed_state.py
```

Final observed results after the reporting fix:

```text
pyright targeted checks: 0 errors
focused seed tests: 29 passed
focused label/candidate tests: 61 passed
focused full backtest-realism unit set: 113 passed
summary flag smoke validation: public_market_evidence_requested=true
.codex_scratch exists=False
```

## Potential Malfunctions To Guard

- Manual export files `Order History 1.csv`, `Order History 2.csv`, and
  `Order History 3.csv` do not include `Strategy Id`.
  - Consequence: exact bot seeding is not verifiable from those files alone.
  - Guard: classify those rows as incomplete unless another exact
    `strategy_id` source proves the bot identity.
  - Best-practice fix: use an exact-identity contract for seeded evidence:
    `symbol + strategy_id + UTC window`. If `Strategy Id` is absent, the row
    may only become `partial` unless a separate verified source proves the same
    strategy id for that exact bot.
  - Best-practice fix: allow order-id linkage through trade history only as
    diagnostic support. It cannot upgrade evidence to `complete` unless the
    linked order ids map uniquely to one workbook bot in the same UTC window.
  - User-help path: the user can provide newer Binance order-history exports
    that include `Strategy Id`, or bot setup screenshots containing strategy
    id, symbol, mode, price range, grid count, and start time. Screenshot times
    must carry explicit local-time provenance before UTC normalization.

- Manual export CSV files are already labeled `Time(UTC)`.
  - Consequence: applying UTC-5 to those CSV timestamps would shift verified
    exchange times by five hours and corrupt the reconstruction.
  - Guard: use CSV `Time(UTC)`, `Update Time`, and `Date(UTC)` as UTC. Apply
    UTC-5 conversion only to user screenshots or manually supplied local-time
    inputs that are explicitly not already UTC.
  - Best-practice fix: add timestamp provenance for every parsed time:
    `raw_time_text`, `raw_time_source`, `raw_timezone`, and
    `normalized_time_utc`.
  - Best-practice fix: fail validation if any Binance CSV timestamp is shifted
    by the local UTC-5 offset. A known CSV row should round-trip as the same UTC
    instant after parsing.
  - Best-practice fix: screenshot/manual local-time inputs must be converted
    exactly once from UTC-5 to UTC, then stored as UTC with provenance. Never
    apply local-time conversion to columns already labeled UTC.

- `tick_size` and `step_size` already exist in the engine, but current training
  defaults keep both at `0.0`.
  - Consequence: exchange-filter realism is not yet being fed into ordinary
    candidate backtests.
  - Guard: candidate-time exchange-filter realism must explicitly fetch
    `exchangeInfo`, extract filter values, pass them into `GridConfig`, and
    stamp provenance before any result can be treated as exchange-filtered.
  - Best-practice fix: extract `PRICE_FILTER.tickSize`, `LOT_SIZE.stepSize`,
    and `MIN_NOTIONAL.notional` from `exchangeInfo` symbol filters. Do not use
    `pricePrecision` as tick size or `quantityPrecision` as step size.
  - Best-practice fix: perform deterministic exchange rounding before the run:
    rounded grid levels must remain strictly increasing, rounded quantity must
    be positive, and each order notional must satisfy the symbol's minimum
    notional rule.
  - Best-practice fix: fail closed with an explicit rejection reason when
    rounding collapses grid levels or violates quantity/notional filters. Do
    not silently change grid bounds, grid count, capital, leverage, or sizing to
    force the candidate to pass.
  - Best-practice fix: result rows from this path must stamp
    `exchange_filter_source`, `tick_size`, `step_size`, `min_notional`, and an
    exchange-filter validation status before they can be compared with the
    legacy baseline.

- Public order book depth is a current snapshot, not historical replay by
  itself.
  - Consequence: using a current depth snapshot to explain an old candidate
    would be a time-leak and a false reconstruction.
  - Guard: use depth snapshots only when timestamped and prospectively archived
    for that candidate window; otherwise classify depth evidence as missing.

- Seeded improvements are not proof of full-pipeline candidate accuracy unless
  replicated with candidate-time-only inputs.
  - Consequence: a better seeded/manual-export result improves historical or
    live telemetry reconciliation, but it does not prove future candidate
    labels are better.
  - Guard: full-pipeline accuracy claims require chronological holdout
    validation from Track B, without manual order/trade/transaction exports,
    future live ladder state, screenshots from future bots, or realized live
    PnL as inputs.

## False Optionality And Rebuttal Consensus

### Provable False Optionality

- Using manual order history for future candidates:
  - False because future candidates do not have future order history.
  - Valid only after deployment or for historical reconciliation.

- Claiming seeded backtest improvements improve full-pipeline output:
  - False unless `run_full_pipeline.py` or the candidate-label generation path
    consumes a candidate-time-safe profile and passes validation.

- Using public current order book snapshots as historical replay:
  - False because a current snapshot is not a historical event sequence.
  - Valid only when timestamped and prospectively archived.

- Retuning parameters until the latest cohort matches live PnL:
  - False because it creates backtest-overfitting risk without proving
    generalization.

### Provably Unnecessary As Written

- New backtest engine:
  - Unnecessary because `backtest/btk_unified_runner.py` already centralizes
    execution through `RealisticGridBacktester`.

- New label contract or new model artifact promotion:
  - Unnecessary for this plan. The first goal is candidate-time backtest
    realism validation, not retraining or artifact release.

- Adding manual-export columns as model features:
  - Unnecessary and unsafe because those columns are not available for future
    candidates.

- Scanner ranking changes before label validation:
  - Unnecessary and risky because the current defect under review is backtest
    realism, not scanner admission.

### Not Valid To Strike Without Assumptions

- Exchange filter rounding:
  - Cannot be struck because engine fields already exist and Binance exposes
    tick/step rules that affect grid levels, quantity, and notional validity.

- Funding-series replay:
  - Cannot be struck because funding affects neutral grid PnL and public
    funding history is available.

- Mark-price valuation:
  - Cannot be struck because futures liquidation and margin valuation can
    differ from last-price fill candles.

- Prospective public market capture:
  - Cannot be struck because it is the only non-leaking route to improved fill
    realism beyond OHLC bars for future candidates.

## Final Implementation Rule

Implementation proceeded checkpoint by checkpoint after user confirmation.

After each checkpoint, the required rule was:

1. Run the listed pyright command.
2. Run the listed focused tests.
3. Run isolated validation when the checkpoint changes behavior.
4. Delete scratch files and prove cleanup.
5. Review changed paths for unintended full-pipeline drift.
6. Fix bugs only in touched or directly affected files.
7. Report whether the checkpoint passed, failed, or remains diagnostic-only.

Final decision:

- Seeded/manual-export reconciliation improved and remains opt-in.
- `candidate_time_public_market_v1` is implemented and testable, but remains
  diagnostic-only because chronological holdout validation failed versus
  `legacy`.
- No result in this implementation is allowed to claim improved full-pipeline
  candidate accuracy because the candidate-time track did not pass the
  chronological holdout gate.

## Support System: Evidence-Matched UTC Correction For Canonical Backtest Accuracy

Status: audit support path defined. This section does not promote a model
artifact, overwrite the canonical workbook, or change scanner/deploy behavior.

Goal:

- Improve the main backtest/training path only when historical workbook rows
  have provably wrong UTC windows and corrected windows produce better
  validation results.
- Keep manual order/trade/transaction exports as provenance and validation
  evidence. They are not model features and must not be available to future
  candidate inference.

Verified evidence already available:

- `scripts/validate_backtest_live_reconciliation.py` exposes
  `--timestamp-policy evidence_matched` and emits row-level timestamp
  provenance fields.
- `tests/unit/test_backtest_timestamp_policy.py` proves the timestamp safety
  boundary: Binance CSV columns labeled UTC are not shifted, workbook
  `stored_utc` remains available, local UTC-5 conversion is explicit, missing
  manual evidence rejects under `evidence_matched`, and `dual_diagnostic` is
  non-promotable.
- Current geometric validation evidence showed `evidence_matched` improved
  holdout mean absolute PnL error from `14.140301546151198` under `stored_utc`
  to `2.7434760256981576`, with one `conflicting_manual_evidence` row excluded
  from proof.
- This evidence validates timestamp correction as a reconciliation improvement.
  It is not yet proof of improved canonical model accuracy until corrected
  historical labels/features are rebuilt and evaluated.

Canonical step-by-step path:

1. Generate an evidence-matched correction manifest.
   - Run the existing validator against historical workbook rows with
     `--timestamp-policy evidence_matched`.
   - Required manifest fields:
     `strategy_id`, `symbol`, workbook stored start/end, local-adjusted
     start/end, manual order-history start/end, selected UTC start/end,
     timestamp deltas in seconds, evidence class, source export files, and
     rejection reason.
   - Classification must be one of:
     `correctable`, `stored_utc_valid`, `conflicting_manual_evidence`, or
     `missing_manual_evidence`.
   - Validation proof required:
     rows classified `correctable` must have exact `symbol + strategy_id`
     manual evidence and exact local-adjusted UTC match. Rows without exact
     evidence are not correctable.

2. Correct only rows with exact evidence.
   - For `correctable` rows, use the selected evidence-matched UTC window.
   - For `stored_utc_valid` rows, leave the stored UTC window unchanged.
   - For `conflicting_manual_evidence` and `missing_manual_evidence`, do not
     change labels/features; keep them diagnostic until additional evidence is
     supplied.
   - Do not globally apply UTC-5 to the workbook. A global shift would be an
     assumption for rows without exact evidence.

3. Regenerate historical labels/features from validated UTC windows.
   - Use the existing feature/backtest rebuild paths.
   - Fetch market data using the selected UTC window only.
   - Preserve manual order/trade/transaction data as provenance columns in the
     manifest or validation report only; do not feed them into model feature
     columns.
   - Write corrected outputs copy-on-write. Do not overwrite
     `data/new_expired_bots.xlsx` in this support step.

4. Rebuild training/evaluation data from corrected historical rows.
   - Build a corrected training/evaluation dataset from the copy-on-write
     output.
   - Deduplicate by the repository's existing candidate identity rules after
     sorting corrected rows by newest validated timestamp/contract lineage.
   - Keep classification fields available for audit, but exclude provenance
     fields from model features.

5. Compare current canonical data versus corrected-data validation.
   - Required comparison:
     row count, eligible corrected rows, excluded rows by evidence class,
     label distribution, mean/median absolute PnL error where live PnL exists,
     sign match, winner recall, non-winner specificity, and top-error rows.
   - Improvement can be claimed only on the corrected evidence-matched subset
     and only if the comparison is run in an isolated scratch directory whose
     outputs are deleted after capture.
   - If corrected data improves validation, the result supports a future
     user-approved rebuild/retrain. It does not automatically promote any
     artifact.

Guardrails:

- Do not shift CSV timestamps already labeled `Time(UTC)`, `Update Time`, or
  `Date(UTC)`.
- Do not infer a missing `strategy_id` from symbol-only matches.
- Do not correct conflicting rows.
- Do not add manual order/trade/transaction values as model features.
- Do not change scanner admission, scanner ranking, HMM artifacts,
  meta-labeler artifacts, utility artifacts, deploy-ready CSV generation, or
  live telemetry scanner behavior.
- Do not claim full-pipeline candidate accuracy improved until corrected
  historical labels/features are rebuilt and the corrected-data validation beats
  the current canonical baseline.

Validation commands:

```text
python -m pyright scripts/validate_backtest_live_reconciliation.py
python -m pytest -q tests/unit/test_backtest_timestamp_policy.py tests/unit/test_btk_seed_state.py tests/unit/test_btk_label_runner.py tests/unit/test_candidate_pipeline_bypass.py
```

Isolated validation requirement:

- Run evidence-matched reconciliation into a scratch directory.
- Capture stored-UTC versus evidence-matched summary metrics.
- Delete the scratch directory.
- Report cleanup proof.

Rebuttal consensus:

- Provable false optionality: a new backtest engine is not required because the
  timestamp support path already routes through the existing reconciliation
  validator and `RealisticGridBacktester` execution surface.
- Provable false optionality: manual exports cannot be candidate-time features
  because future candidates do not have future manual order/trade/transaction
  files.
- Provably unnecessary as written: overwriting the canonical workbook is not
  required to prove the support path. Copy-on-write corrected outputs are enough
  for validation and prevent data loss.
- Not valid to strike without assumptions: corrected historical labels/features
  remain a valid route to full-pipeline improvement because the canonical model
  can learn only from cleaner historical labels after they are rebuilt and
  validated.

## Support Implementation: ERR-043-SUPPORT Canonical UTC Repair, Flat Backfill Only

Goal: fix the manual-ingestion UTC root cause, repair historical workbook UTC
windows in a copy-on-write raw workbook, then rebuild derived backfilled/training
data in a separate flat workbook. Backfill/HMM/training fields must not be
written into `data/new_expired_bots.xlsx` or into any raw workbook sheet.

Canonical integration steps:

1. DONE - Fix the manual UTC ingestion root cause.
   - Implemented in `_bot_data_extractor_core.py` with
     `parse_manual_ui_datetime_to_utc()`.
   - Manual Binance UI/OCR timestamps and pasted manual trade/matched-profit
     timestamps are interpreted as `America/Lima` local time and converted to
     UTC.
   - Binance CSV fields labeled `Time(UTC)`, `Update Time`, and `Date(UTC)`
     remain parsed as UTC and are not shifted.

2. DONE - Add copy-on-write raw workbook repair.
   - Implemented in `scripts/repair_expired_bot_utc_windows.py`.
   - Input stays `data/new_expired_bots.xlsx`; output is a copy such as
     `data/new_expired_bots_utc_corrected.xlsx` or a scratch-path equivalent.
   - The manifest classifies rows as `correctable`, `stored_utc_valid`,
     `conflicting_manual_evidence`, or `missing_manual_evidence`.
   - Rows are corrected only with exact `symbol + strategy_id` order-history
     evidence. Files without `Strategy Id` cannot trigger correction.
   - The corrected raw workbook updates only the extractor-contract timestamp
     fields currently implemented by the repair adapter:
     `start_time_utc` and `end_time_utc`.

3. DONE - Preserve raw workbook/backfill separation.
   - `scripts/backfill_training_features.py` remains the derived backfill path.
   - The corrected raw workbook is used as input, and the output is a separate
     flat workbook such as `data/new_expired_bots_utc_corrected_backfilled.xlsx`.
   - The flat derived workbook may contain outcome, extractor, HMM, stochastic,
     utility, funding, and lineage columns because it is not the raw canonical
     workbook.
   - No backfill/HMM/training columns are written into the corrected raw
     workbook or into `data/new_expired_bots.xlsx`.

4. DONE - Add focused tests for healthy integration.
   - `tests/unit/test_new_bot_data_extractor.py` proves local manual UI time is
     converted to UTC for bot windows, OCR windows, trade fills, and matched
     profit events.
   - `tests/unit/test_repair_expired_bot_utc_windows.py` proves exact-evidence
     correction, missing/conflicting rejection, `Strategy Id` requirement, and
     no added backfill columns in the raw workbook.
   - `tests/unit/test_backfill_training_features_v20260312.py` proves the raw
     corrected workbook remains unchanged while derived fields are written only
     to the separate flat backfilled workbook.

5. DONE - Run isolated validation and delete scratch outputs.
   - Full scratch validation repaired `227` rows, produced a manifest, ran the
     separate flat backfill output, inspected both workbooks, and deleted the
     scratch folder.
   - Manifest counts: `correctable=109`, `stored_utc_valid=19`,
     `conflicting_manual_evidence=51`, `missing_manual_evidence=48`.
   - Raw corrected workbook check: `RAW_HAS_RANGE_PROB=False`.
   - Flat derived workbook check: `FLAT_HAS_RANGE_PROB=True`,
     `FLAT_RANGE_PROB_NONNA=216`, `FLAT_HMM_VERSION_NONNA=227`.
   - Cleanup proof: `SCRATCH_EXISTS_AFTER_CLEANUP=False`.

Guardrails preserved:

- Do not globally shift all workbook rows. Only exact evidence-matched rows are
  corrected.
- Do not shift CSV timestamps already labeled UTC.
- Do not infer missing `strategy_id` from symbol-only matches.
- Do not use manual order/trade/transaction exports as model features.
- Do not overwrite `data/new_expired_bots.xlsx` during validation.
- Do not promote model artifacts from this support step. Any future promotion
  still requires corrected-data validation and separate user approval.

Superseded temporary limitation:

- Before promotion, the repair adapter corrected a raw workbook copy and rebuilt
  a separate flat workbook without overwriting canonical data. That temporary
  limitation is now closed by the canonical replacement below.

## Gate Closure: Corrected UTC Data Is Canonical

Status: CLOSED.

The one-time baseline-vs-corrected gate passed and the corrected data was
promoted into the canonical raw and flat backfilled paths:

- `data/new_expired_bots.xlsx` is now the corrected canonical raw workbook.
- `data/new_expired_bots_backfilled.xlsx` is now regenerated from that corrected
  canonical raw workbook.
- The temporary repair gate script and gate-only test were removed after
  production-path validation passed.
- No model artifact was promoted by this data replacement step.

Closure proof:

- Pre-replacement gate: `gate_pass=true`.
- Baseline vs corrected holdout mean absolute PnL error:
  `14.140301546151198` -> `2.479104243919372`.
- Baseline vs corrected holdout median absolute PnL error:
  `8.201957766028666` -> `1.45761421948092`.
- Canonical post-replacement reconciliation:
  `rows=61`, `model_rows=55`, `missing_kline_rows=6`,
  `holdout_model_rows=19`,
  `holdout_model_mean_abs_pnl_error=2.479104243919372`,
  `holdout_non_winner_specificity_pnl_lte_0=1.0`.
- Canonical flat backfill coverage:
  `FLAT_RANGE_PROB_NONNA=216`, `FLAT_HMM_VERSION_NONNA=227`.
- Scratch cleanup proof: `SCRATCH_EXISTS_AFTER_CLEANUP=False`.

Long-lived invariant:

- The only remaining code behavior from this support path is the root-cause
  fix: manual UI/OCR timestamps are converted from `America/Lima` local time to
  UTC before they are written to the workbook.

## Support System: Corrected Base To Training Inputs And Runtime Artifacts

Goal: make the model training inputs and promoted runtime artifacts learn from
the corrected canonical base. Full-pipeline accuracy can improve only after the
derived backtest outcome side is rebuilt and validated from
`data/new_expired_bots.xlsx`, then the runtime artifacts that consume those
training inputs are staged and promoted through the existing gates.

Verified architecture anchors:

- `retrain_meta_labeler.py` keeps `--input` as a compatibility/reference
  workbook only; active retraining uses `UnifiedTrainingBuilder`.
- `UnifiedTrainingBuilder` reads `data/training_snapshots/*.parquet` as the
  authoritative feature source and `data/backtest_candidates/training_data_*.csv`
  as the authoritative backtest outcome source, joined by `candidate_id`.
- `backtest_candidates.py` writes `training_data_*.csv` and
  `backtest_results_*.csv` into its output directory.
- `hmm_winner_calibrator.py` and `scripts/recalibrate_utility.py` read
  `data/new_expired_bots_backfilled.xlsx` by default.
- Therefore, replacing the canonical workbook and regenerating the flat backfill
  fixed the corrected base, but it does not by itself force the active
  meta-labeler training backbone to learn from corrected backtest outcomes.

Canonical integration steps:

1. DONE - Document the corrected-base propagation plan.
   - `CHANGELOG.md` records that the corrected canonical data is in place and
     that the next accuracy step is rebuilding the derived backtest/training
     outcome inputs plus staged runtime artifacts.
   - This section is the single plan location. No new planning file is needed.

2. DONE - Rebuild derived backtest outcome inputs in scratch first.
   - Run `backtest_candidates.py` against `data/new_expired_bots.xlsx` and
     write to a scratch output directory, not directly to
     `data/backtest_candidates`.
   - Preserve current pipeline behavior unless a realism-profile change is
     separately approved. The current CLI default is `--realism-profile legacy`.
   - Compare scratch `training_data_*.csv` and `backtest_results_*.csv` against
     the current canonical `data/backtest_candidates` files before any
     replacement.
   - Validation result: selected `250` candidates, completed `249`, skipped `1`
     (`WLFIUSDT` empty DataFrame), and wrote scratch-only
     `training_data_20260520.csv` plus `backtest_results_20260520.csv`.
   - Corrected-derived raw training summary: `249` rows, `249` unique
     `candidate_id` values, `0` duplicate `candidate_id` rows,
     positive-label rate `0.4457831325301205`, mean `pnl_pct`
     `1.7469390405398564`, and mean `duration_hours`
     `5.948728246318608`.

3. DONE - Apply healthy training deduplication and classification.
   - Deduplicate by `candidate_id` when present.
   - If `candidate_id` is missing, use exact
     `strategy_id + symbol + start_time_utc`.
   - Never deduplicate, join, or promote by symbol alone.
   - Classify every dropped or rejected row as one of:
     `accepted_authoritative`, `excluded_duplicate_superseded`,
     `excluded_missing_snapshot`, `excluded_unmodelable`, or
     `excluded_stale_or_invalid_contract`.
   - Promotion is blocked if any dropped row is unclassified.
   - Validation result for the scratch run:
     `accepted_authoritative=54`, `excluded_missing_snapshot=195`,
     `excluded_unmodelable=1`, `excluded_duplicate_superseded=0`, and
     `excluded_stale_or_invalid_contract=0`.

4. DONE - Validate unified training inputs.
   - Run `retrain_meta_labeler.py` in dry-run/export mode against the scratch
     backtest outcome directory.
   - Compare current versus corrected-derived training inputs by row count,
     duplicate identity count, modelable rows, feature completeness, label
     distribution, chronological split metrics, sign match, winner recall,
     non-winner specificity, and top-error rows where live outcomes exist.
   - Use explicit metrics:
     `abs_error = abs(model_pnl_pct - live_pnl_pct)`,
     `sign_match = sign(model_pnl_pct) == sign(live_pnl_pct)`,
     `winner_recall = true_positive_winners / actual_winners`,
     `non_winner_specificity = true_negative_non_winners / actual_non_winners`,
     and `label_delta = corrected_label_rate - current_label_rate`.
   - Validation result: current baseline and corrected-derived scratch exports
     both produced `54` modelable rows, `54` unique `candidate_id` values, and
     identical candidate IDs.
   - Baseline positive-label rate:
     `0.24074074074074073`.
   - Corrected-derived positive-label rate:
     `0.24074074074074073`.
   - `label_delta=0.0`.
   - Overlapping-row `net_pnl_pct` delta:
     `mean_delta=-0.24426459181431362`,
     `median_abs_delta=0.0`, `max_abs_delta=9.421167586265716`.
   - Interpretation: corrected live-matched durations changed some PnL values,
     but this scratch gate did not change the active fast-target label set.

5. DONE - Stage runtime artifact checks only after input validation.
   - Stage or dry-run the meta-labeler from corrected-derived training inputs.
   - Stage or dry-run the HMM winner calibrator from
     `data/new_expired_bots_backfilled.xlsx`.
   - Stage or dry-run the utility calibrator from
     `data/new_expired_bots_backfilled.xlsx`.
   - Do not retrain the HMM regime model in this step. The corrected expired-bot
     workbook is historical bot outcome evidence, not the HMM market-regime
     training source.
   - Do not promote any artifact until staged validation passes and the user
     separately approves promotion.
   - Validation result: meta-labeler dry-run/export passed feature-contract
     validation with no artifact write.
   - HMM winner calibrator result: `promotable=false`,
     `pool_rows=103`, `fit_rows=68`, `holdout_rows=35`,
     `holdout_auc_delta=-0.09333333333333332`.
   - Utility calibrator result: `promotable=false` because
     `G7_finite_nonnegative_not_boundary_pinned=false`.
   - No runtime artifact was promoted.

6. DONE - Promotion blocked by gates; canonical runtime path unchanged.
   - If the gate passes and the user approves, replace only the canonical
     runtime artifacts used by the current pipeline.
   - Delete staging outputs after promotion so alternate artifact paths do not
     silently drift.
   - Run a no-deploy full-pipeline proof and compare candidate counts, failure
     stages, meta-label coverage, HMM winner coverage, utility coverage, grid
     validity, and deploy-ready output count.
   - Validation result: no-deploy proof was run with feature snapshot logging
     disabled in memory and output redirected to scratch, preventing mutation of
     `data/training_snapshots`.
   - Full-pipeline proof output: `250` rows, `3` grid-valid rows, valid symbols
     `HIGHUSDT`, `SENTUSDT`, and `PROVEUSDT`.
   - Failure-stage counts: `pre_reject=101`, `regime=49`,
     `score_threshold=32`, `hard_gate=25`, `stage_b=21`,
     `grid_generation=19`, `approved=3`.
   - Coverage: `meta_prob` non-null on `47` rows, `hmm_winner_score` non-null
     on `250` rows, `range_prob` non-null on `250` rows, `survival_prob`
     non-null on `250` rows, `deployment_score` non-null on `3` rows, and
     `utility_score` non-null on `0` rows because no utility `current.json` is
     available.
   - Promotion status: blocked. The corrected base is valid, but this gate did
     not prove that replacing runtime artifacts would improve the full pipeline.

Rebuttal consensus:

- Provable false optionality: keeping parallel corrected and legacy workbook
  paths after canonical replacement is false. The canonical data paths are
  already `data/new_expired_bots.xlsx` and
  `data/new_expired_bots_backfilled.xlsx`.
- Provable false optionality: treating the backfilled workbook alone as enough
  to retrain the meta-labeler is false. The active builder reads
  `data/backtest_candidates/training_data_*.csv` outcomes plus
  `data/training_snapshots/*.parquet` features.
- Provable false optionality: retraining the HMM regime model from corrected
  expired-bot rows is false for this goal. The HMM is a market-regime model; the
  corrected workbook is historical bot outcome evidence.
- Provable false optionality: adding manual export fields as model features is
  false. Manual exports are provenance and validation evidence, and future
  candidates do not have future manual order/trade/transaction exports.
- Provably unnecessary as written: a new training data store is unnecessary
  because the repo already has `data/backtest_candidates` and
  `data/training_snapshots`.
- Provably unnecessary as written: a new model architecture is unnecessary
  because the goal is corrected input propagation, not replacing the learner.
- Provably unnecessary as written: scanner admission or ranking changes are
  unnecessary in this step because scanner behavior should not change until
  corrected training inputs and staged artifacts pass validation.
- Provably unnecessary as written: symbol-only matching is unnecessary and
  unsafe because stronger identities already exist.
- Not valid to strike without assumptions: rebuilding
  `data/backtest_candidates/training_data_*.csv` is required because active
  meta-labeler training reads that outcome path.
- Not valid to strike without assumptions: dry-run/export validation before
  artifact promotion is required because corrected inputs can still shift label
  balance, feature coverage, or chronological validation behavior.
- Not valid to strike without assumptions: staging artifact refits are required
  to prove runtime artifacts can learn from the corrected base without
  corrupting current production paths.
- Not valid to strike without assumptions: a no-deploy full-pipeline proof is
  required to verify runtime integration without placing trades.

Acceptance criteria:

- Scratch backtest outcome rebuild completes or reports a concrete blocker.
- Scratch outputs are deleted unless the user separately approves promotion.
- Unified training dry-run/export compares current versus corrected-derived
  training inputs.
- No manual export data becomes a model feature.
- No runtime artifact is promoted without a separate approval.
- After any future approved promotion, only one canonical runtime artifact path
  remains active.
