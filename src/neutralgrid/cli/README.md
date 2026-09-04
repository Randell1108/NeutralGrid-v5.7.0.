# NeutralGrid CLI Tools

Command-line tools for model training, evaluation, and maintenance.

## Available Commands

### `retrain` - Rolling Window Retraining

Retrain HMM models with rolling time windows to handle non-stationary markets.

**Usage**:
```bash
python -m neutralgrid.cli.retrain [OPTIONS]
```

**Options**:
- `--window WINDOW` - Training window size (e.g., `180d`, `90d`, `12w`, `6m`). Default: `180d`
- `--symbols SYMBOLS [SYMBOLS ...]` - Specific symbols to train on (e.g., `BTCUSDT ETHUSDT`). If not specified, uses top-N by volume
- `--top-n TOP_N` - Number of top symbols by volume to use if `--symbols` not specified. Default: `30`
- `--output OUTPUT` - Output directory for trained model. Default: `artifacts/hmm/rolling_<window>_<timestamp>`
- `--n-components N_COMPONENTS` - Number of HMM states. Default: from config (`3`)
- `--evaluate` - Run walk-forward evaluation after training
- `--save-dataset` - Save training dataset to disk for reproducibility
- `--dataset-name DATASET_NAME` - Name for saved dataset. Default: `rolling_<window>_<date>`
- `--force-refresh` - Force refresh data from API even if cached

**Examples**:

```bash
# Basic: Retrain with 180-day rolling window
python -m neutralgrid.cli.retrain --window 180d

# With evaluation and dataset saving (recommended)
python -m neutralgrid.cli.retrain --window 180d --evaluate --save-dataset

# Custom symbols
python -m neutralgrid.cli.retrain --window 90d --symbols BTCUSDT ETHUSDT BNBUSDT

# Top 50 symbols by volume
python -m neutralgrid.cli.retrain --window 180d --top-n 50 --evaluate

# Custom output parent directory (versioned child is auto-created)
python -m neutralgrid.cli.retrain --window 180d --output artifacts/hmm

# Force refresh data from API (ignore cache)
python -m neutralgrid.cli.retrain --window 180d --force-refresh

# Custom HMM components
python -m neutralgrid.cli.retrain --window 180d --n-components 4 --evaluate
```

**Output Structure**:

After running retraining, artifacts are saved to the specified output directory:

```
artifacts/hmm/rolling_180d_YYYYMMDD_HHMMSS/
├── model.joblib                   # Trained HMM model
├── scaler.joblib                  # RobustScaler for features
├── state_means_unscaled.npy      # State means in original scale
├── metadata.json                  # Complete provenance
├── feature_schema.json            # Feature names and order
└── eval.json                      # Walk-forward evaluation results (if --evaluate)
```

If `--save-dataset` is used, training data is also saved:

```
data/training_sets/rolling_180d_20260112/
├── metadata.json                  # Dataset metadata
├── BTCUSDT.parquet               # Cached market data
├── ETHUSDT.parquet
└── ...
```

**Window Size Guidelines**:
- **180 days (6 months)**: Good balance, adapts to recent regimes (recommended starting point)
- **90 days (3 months)**: More adaptive, may be noisy
- **365 days (1 year)**: Stable, but may lag regime changes

**Evaluation Metrics**:

When `--evaluate` is used, the following metrics are calculated via walk-forward evaluation:

- `mean_pass_rate`: Percentage of predictions that passed validation (target: 60-70%)
- `std_pass_rate`: Standard deviation of pass rate across splits (lower is better)
- `mean_range_prob`: Average probability assigned to range state (target: > 0.55)
- `mean_trend_prob`: Average probability assigned to trend state (target: < 0.35)

**Automated Retraining**:

Set up scheduled retraining using cron (Linux/Mac) or Task Scheduler (Windows):

```bash
# Weekly retraining (Sundays at 2 AM UTC)
# Cron: 0 2 * * 0
python -m neutralgrid.cli.retrain \
    --window 180d \
    --top-n 50 \
    --evaluate \
    --save-dataset \
    --output artifacts/hmm/weekly/$(date +%Y%m%d)

# Monthly retraining (1st of month at 3 AM UTC)
# Cron: 0 3 1 * *
python -m neutralgrid.cli.retrain \
    --window 365d \
    --top-n 100 \
    --evaluate \
    --save-dataset \
    --output artifacts/hmm/monthly/$(date +%Y%m%d)
```

**Integration with API**:

After retraining, restart your API to load the new model:

```bash
# API loads the active rolling artifact from artifact_manifest.json
uvicorn neutralgrid.api.app:app --reload
```

---

## Future CLI Tools (Planned)

### `evaluate` - Standalone Model Evaluation

Evaluate existing models without retraining.

```bash
python -m neutralgrid.cli.evaluate --artifact artifacts/hmm/rolling_180d_YYYYMMDD_HHMMSS --symbols BTCUSDT ETHUSDT
```

### `backtest` - Grid Performance Backtesting

Backtest grid strategies on historical data.

```bash
python -m neutralgrid.cli.backtest --symbols BTCUSDT --start 2025-01-01 --end 2025-12-31
```

### `monitor` - Production Model Monitoring

Monitor live model performance and drift detection.

```bash
python -m neutralgrid.cli.monitor --artifact artifacts/hmm/rolling_180d_YYYYMMDD_HHMMSS --alert-email user@example.com
```

---

## Development

### Adding New CLI Commands

1. Create a new file in `src/neutralgrid/cli/` (e.g., `evaluate.py`)
2. Follow the pattern from `retrain.py`:
   - Use `argparse` for CLI argument parsing
   - Use structured logging with `get_logger()` and `set_run_id()`
   - Use `asyncio.run(main())` for async entry point
   - Add docstrings and examples
3. Update this README with usage instructions
4. Add tests to `tests/cli/` (if applicable)

### Best Practices

- **Structured logging**: Always use `get_logger()` and set context variables (`run_id`, `symbol`, `artifact_version`)
- **Error handling**: Catch exceptions and log with `exc_info=True` for debugging
- **Argument validation**: Validate all user inputs before processing
- **Output metadata**: Save complete metadata for all artifacts (provenance tracking)
- **Help text**: Provide clear help text and usage examples in `--help`
- **Exit codes**: Use `sys.exit(1)` for errors, `sys.exit(0)` for success

---

## Troubleshooting

### Common Issues

**ImportError: No module named 'src.neutralgrid'**

Ensure you're running from the project root directory:
```bash
cd "c:\Users\cris_\OneDrive\Documents\Christian\Crypto\Antigravity - NEUTRAL grid v2"
python -m neutralgrid.cli.retrain --help
```

**API rate limits**

Use `--force-refresh` sparingly to avoid hitting Binance API rate limits. The CLI uses disk caching by default to minimize API calls.

**Out of memory**

Reduce `--top-n` or `--window` if you encounter memory issues:
```bash
python -m neutralgrid.cli.retrain --window 90d --top-n 20
```

**Slow evaluation**

Evaluation can be time-consuming for large datasets. Disable it for faster retraining:
```bash
python -m neutralgrid.cli.retrain --window 180d  # No --evaluate flag
```

---

## See Also

- [Phase 5 Summary](../../../PHASE_5_SUMMARY.md) - Detailed documentation of rolling retraining
- [src/neutralgrid/models/hmm/train.py](../models/hmm/train.py) - Training implementation
- [src/neutralgrid/backtest/evaluate.py](../backtest/evaluate.py) - Evaluation implementation
- [src/neutralgrid/core/config.py](../core/config.py) - Configuration management
- [src/neutralgrid/core/logging.py](../core/logging.py) - Structured logging
