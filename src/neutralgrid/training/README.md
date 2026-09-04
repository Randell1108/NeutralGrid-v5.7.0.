# Meta-Labeler Training Data Guide

This document describes the training data requirements for the meta-labeler,
feature collection procedures, and known data gaps.

## Training Table Schema

Each row in the training table represents one candidate at decision time:

### Identifiers (Required)
| Column | Type | Description |
|--------|------|-------------|
| `symbol` | str | Trading pair (e.g., "BTCUSDT") |
| `start_time_utc` | datetime | Decision timestamp (UTC) |

### Features (14 Expected)

#### A. HMM/Regime Features (from 1H OHLCV)
| Feature | Source | Computable from Historical? |
|---------|--------|---------------------------|
| `range_prob` | HMM posterior | **NO** - requires 1H OHLCV at decision time |
| `trend_prob` | HMM posterior | **NO** - requires 1H OHLCV at decision time |
| `utility_score` | Utility scorer | **NO** - requires HMM outputs |

#### B. Stochastic/Survival Features (from 15M OHLCV)
| Feature | Source | Computable from Historical? |
|---------|--------|---------------------------|
| `survival_prob` | Monte Carlo simulation | **NO** - requires 15M OHLCV + range bounds |
| `hurst_exponent` | R/S analysis | **NO** - requires 15M log-prices |
| `ou_halflife` | OU parameter fit | **NO** - requires 15M log-prices |

#### C. Grid Parameters (from grid calculator)
| Feature | Source | Computable from Historical? |
|---------|--------|---------------------------|
| `profit_per_grid_pct` | Grid calculator | **YES** - from price_range_low/high + grids_count |
| `num_grids` | Grid calculator | **YES** - from grids_count column |
| `range_size_pct` | Range calculation | **YES** - from BB width or range bounds |

#### D. Technical Indicators (from OHLCV)
| Feature | Source | Computable from Historical? |
|---------|--------|---------------------------|
| `adx_1h` | ADX(14) on 1H | **YES** - if 1H OHLCV available |
| `adx_15m` | ADX(14) on 15M | **YES** - if 15M OHLCV available |
| `rsi_15m` | RSI(14) on 15M | **YES** - if 15M OHLCV available |

#### E. Market Features
| Feature | Source | Computable from Historical? |
|---------|--------|---------------------------|
| `funding_rate` | Binance API | **NO** - not captured in historical bot data |
| `ev_score` | PnL ranker | **NO** - requires survival_prob + other components |

### Labels (Required for Training)
| Column | Type | Description |
|--------|------|-------------|
| `pnl_pct` | float | Realized PnL percentage (e.g., 5.0 = 5%) |
| `sl_hit` | bool | Whether stop-loss was triggered |
| `y` | int | Binary label: 1 if success, 0 otherwise |

## Current Data Availability (new_expired_bots.xlsx)

### Available Features (6/14)
- `num_grids` (from grids_count)
- `profit_per_grid_pct` (computed from grid params)
- `range_size_pct`
- `adx_1h`
- `adx_15m`
- `rsi_15m`

### Missing Features (8/14)
- `range_prob` - Requires historical 1H OHLCV at decision time
- `trend_prob` - Requires historical 1H OHLCV at decision time
- `utility_score` - Requires HMM outputs
- `survival_prob` - Requires historical 15M OHLCV + range bounds
- `hurst_exponent` - Requires historical 15M log-prices
- `ou_halflife` - Requires historical 15M log-prices
- `funding_rate` - Not captured in historical data
- `ev_score` - Requires survival_prob + other components

## Training with Limited Data

When training with limited features, the meta-labeler will:
1. Use available features only
2. Fill missing features with 0.0 (configurable)
3. Report feature importance only for used features

**Expected Performance Impact:**
- Training on 6/14 features will produce a weaker model
- The model will rely heavily on technical indicators
- HMM regime features (range_prob, trend_prob) are typically the most predictive
- Stochastic features (survival_prob, hurst) add significant discriminative power

## Collecting Future Training Data

### Option 1: Feature Snapshot Logging (Recommended)

Integrate the FeatureCollector into the scanner to log features at scan time:

```python
from neutralgrid.training import FeatureCollector, build_feature_snapshot

# Create collector at scanner startup
collector = FeatureCollector(log_dir="data/training_snapshots")

# In fetch_one(), after computing features:
snapshot = build_feature_snapshot(
    symbol=symbol,
    scanner_row=row,
    validation_result=validation_result,
)
collector.log(snapshot)

# At end of session:
collector.close()
```

Snapshots are stored as daily Parquet files in `data/training_snapshots/`.

### Option 2: Manual Logging

For each grid bot launched:
1. Log all 14 features at launch time
2. Track the bot's performance over 48h
3. Record final PnL, whether SL was hit, and exit reason
4. Combine into training table

## Label Generation (AFML-Compliant)

Labels are generated using triple-barrier method:

### Configuration
```python
LabelConfig(
    hurdle_pct=5.0,      # Take-profit: +5%
    sl_pct=-10.0,        # Stop-loss: -10%
    horizon_hours=48.0,  # Maximum holding time
)
```

### Label Definition
```
y = 1 if:
    pnl_pct >= hurdle_pct (5%) AND
    stop-loss NOT hit before hurdle

y = 0 if:
    pnl_pct < hurdle_pct OR
    stop-loss hit before hurdle OR
    vertical barrier (48h) reached without hurdle
```

## Cross-Validation Requirements

### Purging and Embargo
Because labels are forward-looking (48h horizon), proper CV requires:
- **Purging**: Remove samples near test boundaries
- **Embargo**: Gap after test set to prevent overlap

### Configuration
```python
CVConfig(
    n_folds=5,
    purge_pct=0.02,    # Purge 2% of samples
    embargo_pct=0.01,  # 1% embargo gap
)
```

### Timestamp Requirements
- Every row must have `start_time_utc`
- Timestamps must be accurate (not synthetic)
- CV will sort by timestamp and apply purging

## Avoiding Common Pitfalls

### 1. Training-Serving Skew
**Problem:** Features computed differently in training vs inference.

**Solution:** Use the same feature functions:
- Scanner uses `compute_features()` and `infer_regime()`
- Training data should capture outputs of these exact functions
- Never compute features from different code paths

### 2. Indicator Drift
**Problem:** ADX/RSI computed from different candle sources.

**Solution:** Ensure consistent parameters:
```python
ADX_PERIOD = 14  # Same everywhere
RSI_PERIOD = 14
BB_PERIOD = 20
```

### 3. Funding Rate Missingness
**Problem:** Systematic nulls become implicit signals when filled with 0.0.

**Solution:** Track missingness separately or use funding_rate_available flag.

### 4. Survival/Range Bound Inconsistency
**Problem:** survival_prob depends on range bounds; if range changes, survival changes.

**Solution:** Log range_high and range_low with each snapshot.

### 5. Non-Stationary Regime
**Problem:** Older examples become less relevant.

**Solution:** Consider:
- Time-decay weighting (AFML discusses this)
- Rolling training windows
- Regime-specific models

## Running Training

### Basic Training
```bash
python retrain_meta_labeler.py \
    --input data/new_expired_bots.xlsx \
    --output models/meta_labeler.pkl
```

### With Analysis
```bash
python retrain_meta_labeler.py \
    --input data/new_expired_bots.xlsx \
    --analyze-only
```

### Export Training Data for Inspection
```bash
python retrain_meta_labeler.py \
    --input data/new_expired_bots.xlsx \
    --export-training-data data/training_data_preview.csv \
    --dry-run
```

## Future Improvements

1. **Backfill Historical Data**: Fetch 1H/15M OHLCV at decision times
   - Risk: May introduce subtle leakage if not careful
   - Recommendation: Only backfill if timestamps are reliable

2. **Live Feature Logging**: Enable FeatureCollector in production scanner
   - Benefit: Full feature set for future training
   - Cost: Additional storage and processing

3. **Outcome Tracking**: Automated outcome collection from Binance API
   - Track PnL, SL hits, and exit reasons
   - Join with logged features by (symbol, timestamp)

4. **Time-Weighted Training**: Weight recent examples higher
   - Addresses regime drift
   - AFML recommends exponential decay
