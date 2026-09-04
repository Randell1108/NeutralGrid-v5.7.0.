# Scanner Usage Guide

## Overview

The scanner system consists of two main scripts for finding and validating trading opportunities:

1. **`retrain_scanner.py`** - Train models from historical bot data
2. **`scan_top100.py`** - Scan live markets for trading opportunities

---

## Training Models (`retrain_scanner.py`)

### Purpose

Train or retrain the pattern profile and probabilistic models from your historical bot performance data.

### Input

Excel file with three sheets:
- **Entry Validation Metrics** - Entry features (ADX, RSI, slopes, etc.)
- **Performance Risk-Adjusted** - PnL, profit factor, duration
- **Market and Volatility** - Funding rates, open interest, ATR, volume

### Output

Three JSON files in `data/profile/`:
1. **`pattern_profile.json`** - Statistical profile of winning bots
   - Means, standard deviations
   - 10th/90th percentiles
   - Trend distribution

2. **`profile_model.json`** - Gaussian discriminant classifier training artifact
   - Winner vs. loser means
   - Pooled inverse covariance
   - Prior probability
   - Serves as the bootstrap candidate when `current.json` is absent

3. **`profile_gate.json`** - Hard threshold gates
   - ADX bounds (1h, 15m, 5m)
   - RSI range (15m)

### Basic Usage

```bash
# Train with default settings
python retrain_scanner.py

# Train with custom input file
python retrain_scanner.py --input data/my_bots.xlsx

# Save to versioned directory
python retrain_scanner.py --output-dir models/v1.0.0/profile/
```

### Advanced Options

```bash
# More selective winner criteria
python retrain_scanner.py \
  --min-duration-hours 12.0 \
  --min-profit-factor 2.0 \
  --top-quantile 0.80

# Only train profile and model (skip gate)
python retrain_scanner.py --skip-gate

# Control covariance shrinkage (0.0-1.0)
python retrain_scanner.py --shrinkage 0.20

# Widen gate thresholds by 20%
python retrain_scanner.py --gate-widen 0.20
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | `data/new_expired_bots.xlsx` | Training data file |
| `--output-dir` | `data/profile/` | Output directory |
| `--min-duration-hours` | `6.0` | Minimum bot runtime |
| `--min-profit-factor` | `1.5` | Minimum profit factor |
| `--min-avg-profit-per-grid` | `0.70` | Minimum avg profit per grid |
| `--top-quantile` | `0.75` | PnL threshold (top 25%) |
| `--shrinkage` | `0.30` | Covariance regularization |
| `--gate-widen` | `0.15` | Gate threshold widening |

### When to Retrain

- **Weekly/Monthly:** If actively collecting new bot data
- **After major market changes:** Regime shifts, volatility changes
- **When performance degrades:** Lower win rate, worse predictions
- **New features added:** Additional technical indicators

### Training Output Example

```
======================================================================
NEUTRAL Grid Scanner - Retrain Profile Models
======================================================================
Input file: data/new_expired_bots.xlsx
Output directory: data/profile/
----------------------------------------------------------------------
Training parameters:
  Min duration hours: 6.0
  Min profit factor: 1.5
  Min avg profit per grid: 0.7
  Top quantile: 0.75
  Features: 16 features
----------------------------------------------------------------------
Training pattern profile...
✓ Pattern profile trained
  Winners count: 45
  PnL threshold: 8.23%
  Features: 16
  Trend distribution: {'ranging': 0.42, 'uptrend': 0.33, 'downtrend': 0.25}
✓ Saved to: data/profile/pattern_profile.json
----------------------------------------------------------------------
Training profile model...
  Shrinkage: 0.3
✓ Profile model trained
  Features: 16
  Prior winner: 0.375
✓ Saved bootstrap candidate to: data/profile/profile_model.json
----------------------------------------------------------------------
Generating profile gate...
  Widening factor: 0.15
✓ Profile gate generated
  ADX 1h max: 45.23
  ADX 15m max: 38.67
  ADX 5m max: 42.89
  RSI 15m range: [35.20, 68.45]
✓ Saved to: data/profile/profile_gate.json
======================================================================
```

---

## Scanning Markets (`scan_top100.py`)

### Purpose

Scan live markets for trading opportunities using trained models.

### How It Works

1. **Fetch top symbols** by 24h quote volume
2. **Download market data** for each symbol (1h, 15m, 5m klines)
3. **Compute features** (ADX, RSI, slopes, crosses, etc.)
4. **Score against profile** using similarity + model probability
5. **Output ranked candidates** to CSV

### Basic Usage

```bash
# Scan top 100 symbols (default)
python scan_top100.py

# Scan top 50 symbols
python scan_top100.py --top-n 50

# Save to custom location
python scan_top100.py --output my_scan.csv

# Filter by minimum score
python scan_top100.py --min-score 60.0
```

### Rate Limiting

```bash
# More conservative (slower, safer)
python scan_top100.py \
  --max-concurrency 4 \
  --min-delay 0.1

# More aggressive (faster, higher API usage)
python scan_top100.py \
  --max-concurrency 12 \
  --min-delay 0.02
```

### Model Selection

By default, the profile model is resolved via `resolve_active_profile_model_path()`,
which prefers `data/profile/current.json` (the promotion pointer written by
`promote_profile_version` — see `profile_model_walkforward.py`). If
`current.json` is absent, the resolver falls back to
`data/profile/profile_model.json` as an **unvalidated bootstrap candidate** so
the pipeline can continue collecting outcome data. Standard scans should pass no
model-related flags and let the resolver choose promoted-or-bootstrap default
loading.

Manual overrides (diagnostic / non-production use only):

```bash
# Force-load a specific model JSON, bypassing the promotion pointer.
# Useful when comparing a candidate artifact against the promoted one.
python scan_top100.py \
  --profile-path models/v1.0.0/profile/pattern_profile.json \
  --model-path models/v1.0.0/profile/profile_model.json

# Run in degraded similarity-only mode (no probabilistic model).
# Emits a single WARN at boot; every affected row's scoring_flags
# gets "profile_model_absent" appended. Intended for diagnostics, not
# routine operation.
python scan_top100.py --no-model
```

### Scan Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--top-n` | `100` | Number of symbols to scan |
| `--output` | Auto-generated | Output CSV path |
| `--max-concurrency` | `8` | Concurrent API requests |
| `--min-delay` | `0.05` | Delay between requests (s) |
| `--profile-path` | `data/profile/pattern_profile.json` | Pattern profile path |
| `--model-path` | `None` → resolved from `data/profile/current.json`, else `data/profile/profile_model.json` bootstrap fallback | Explicit profile-model path (diagnostic override) |
| `--no-model` | `False` | Skip profile model (degraded similarity-only, diagnostic) |
| `--min-score` | `None` | Minimum score filter |

### Output Format

CSV file with columns:

**Identification:**
- `symbol` - Trading pair (e.g., BTCUSDT)

**Scores:**
- `score` - Combined score (0-100, higher = better fit)
- `winner_proba` - Model probability (0-1, if model enabled)

**Notes:**
- `notes` - Qualitative assessments (e.g., "ADX_1h in winner band")

**Features:** (16+ columns)
- `adx_1h`, `adx_15m`, `adx_5m` - ADX indicators
- `rsi_15m` - RSI indicator
- `ema_slope_1h` - EMA slope
- `ema_crosses_5m`, `vwap_crosses_5m` - Cross counts
- `range_size_pct` - Price range percentage
- `bb_width` - Bollinger Band width
- `quote_volume_24h` - 24h quote volume

### Scan Output Example

```
======================================================================
NEUTRAL Grid Scanner - Top Symbols by Quote Volume
======================================================================
Loading pattern profile from: data/profile/pattern_profile.json
✓ Pattern profile loaded (16 features)
Loading default profile model via current.json/bootstrap → data/profile/profile_model_YYYYMMDD_HHMMSS.json
✓ Profile model loaded (16 features)
Initializing Binance client...
✓ Connected to Binance (auth: True)
----------------------------------------------------------------------
Scanning top 100 symbols by quote volume...
Concurrency: 8, Min delay: 0.05s
----------------------------------------------------------------------
✓ Results saved to: results/neutralgrid_candidates_20260112_143022.csv
Total candidates: 87
Score range: 42.31 - 89.67
Top 5 symbols by score:
  1. SOLUSDT      score= 89.67 (ADX_1h in winner band; Funding near flat)
  2. LINKUSDT     score= 85.23 (ADX_15m in winner band; BB width in winner band)
  3. AVAXUSDT     score= 82.45 (High VWAP cross count (mean-reverting))
  4. MATICUSDT    score= 79.12 (Range size in winner band)
  5. ATOMUSDT     score= 76.89 (ADX_5m in winner band)
======================================================================
Scan complete
======================================================================
```

### Understanding Scores

**Score Components:**

1. **Similarity Score (65% weight):**
   - Weighted Euclidean distance to winner profile
   - Considers all features with custom weights
   - Trend structure bonus/penalty

2. **Model Probability (35% weight, if enabled):**
   - Gaussian discriminant classifier
   - P(winner | features)
   - Based on pooled covariance

**Score Interpretation:**

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| 80-100 | Excellent fit | Strong candidate |
| 60-80 | Good fit | Review details |
| 40-60 | Moderate fit | Check notes |
| 0-40 | Poor fit | Likely skip |

**Notes Interpretation:**

- `ADX_Xh in winner band` - ADX within historical winner range
- `BB width in winner band` - Bollinger Band width similar to winners
- `Range size in winner band` - Price range similar to winners
- `High EMA cross count (choppy)` - Many EMA crosses (ranging market)
- `High VWAP cross count (mean-reverting)` - Many VWAP crosses
- `Funding near flat` - Funding rate close to zero (balanced)

---

## Typical Workflow

### 1. Initial Setup

```bash
# Train models from historical data
python retrain_scanner.py

# Verify output files exist
ls -lh data/profile/
```

### 2. Daily/Regular Scanning

```bash
# Run scan
python scan_top100.py --min-score 65.0

# Review results
cat results/neutralgrid_candidates_*.csv | head -20
```

### 3. Periodic Retraining

```bash
# Monthly or when adding new data
python retrain_scanner.py \
  --input data/new_expired_bots.xlsx \
  --output-dir data/profile/

# Compare old vs. new models
diff data/profile/pattern_profile.json \
     models/archive/pattern_profile_old.json
```

### 4. Model Versioning (Recommended)

```bash
# Train new version
python retrain_scanner.py \
  --output-dir models/v1.1.0/profile/

# Test new version
python scan_top100.py \
  --profile-path models/v1.1.0/profile/pattern_profile.json \
  --model-path models/v1.1.0/profile/profile_model.json \
  --top-n 20

# If good, promote to production via current.json
# (do not promote by copying files directly into data/profile/)
```

---

## Integration with Main Application

The scanner can work alongside the main FastAPI application:

### Scenario 1: Pre-screening

```bash
# 1. Scan for candidates
python scan_top100.py --min-score 70.0 --output candidates.csv

# 2. Extract top symbols
top_symbols=$(cat candidates.csv | tail -n +2 | cut -d, -f1 | head -10)

# 3. Validate through API
for symbol in $top_symbols; do
  curl -X POST http://localhost:8000/api/validate \
    -H "Content-Type: application/json" \
    -d "{\"symbol\": \"$symbol\"}"
done
```

### Scenario 2: Automated Pipeline

```bash
# cron job: 0 */6 * * * (every 6 hours)
python scan_top100.py --min-score 75.0
python process_candidates.py  # Your custom logic
```

---

## Troubleshooting

### Models Not Found

```bash
# Error: Pattern profile not found
python retrain_scanner.py  # Train first; default loading then uses
                           # current.json if present, else profile_model.json
                           # as the bootstrap candidate
```

### API Rate Limiting

```bash
# Error: 429 Too Many Requests
python scan_top100.py \
  --max-concurrency 4 \
  --min-delay 0.2  # Slower
```

### Empty Results

```bash
# Try lowering threshold
python scan_top100.py --min-score 50.0

# Or scan more symbols
python scan_top100.py --top-n 200
```

### Feature Mismatch

```bash
# Retrain with same features
python retrain_scanner.py
python scan_top100.py  # Should work now
```

---

## Advanced: Custom Features

If you modify `scanner/feature_extractor.py`:

1. **Update `DEFAULT_FEATURES`** in `scanner/pattern_profile.py`
2. **Retrain models** with new features
3. **Verify** features appear in output CSV

```python
# In scanner/pattern_profile.py
DEFAULT_FEATURES = [
    "adx_1h", "adx_15m", "adx_5m", "rsi_15m", "ema_slope_1h",
    "ema_crosses_5m", "vwap_crosses_5m", "range_size_pct", "bb_width",
    "trend_structure",
    # Add your new features here
    "my_custom_indicator",
]
```

---

## FAQ

**Q: How often should I scan?**
A: Depends on market conditions. During volatile periods: hourly. Stable markets: 4-6 hours.

**Q: Can I run scans in parallel?**
A: Yes, but respect Binance API limits. Use different output files.

**Q: How do I backtest the scanner?**
A: Use `backtest/backtest_profile_models.py` with historical data.

**Q: Should I always use the profile model?**
A: For normal scans, yes. Default loading uses the promoted model when
available, otherwise the bootstrap candidate. `--no-model` is diagnostic /
degraded mode, not the standard path.

**Q: Can I scan specific symbols instead of top 100?**
A: Not directly. Modify `scan_top100.py` or use the API validation endpoints.

---

*Last updated: 2026-01-12*
*Related: [QUICK_START.md](QUICK_START.md), [REFACTOR_PLAN.md](REFACTOR_PLAN.md)*
