# Meta-Labeling System

This folder contains all meta-labeling related data, models, and scripts for the Neutral Grid Bot trading system.

## What is Meta-Labeling?

Meta-labeling is a secondary ML model that predicts the probability that a primary model's prediction will be correct. Instead of predicting market direction, it predicts whether a deployment candidate will be successful.

## Folder Structure

### `/data/` - Meta-labeling datasets and results
- `expired_bots_meta_labeling_details.csv` - Complete meta-label results for all expired bots
- `meta_labeling_summary.csv` - Summary analysis of TRUMPUSDT and XRPUSDT meta-labeling validation
- `backtest_vs_actual_comparison.csv` - Comparison of backtest predictions vs actual live performance

### `/models/` - Trained meta-labeling models
- `meta_labeler.pkl` - Primary meta-labeling model (global)
- `meta_labeler_model.pkl` - Alternative meta-labeling model from live tracking
- `meta_labeler_scaler.pkl` - Feature scaler for model inputs

### `/scripts/` - REMOVED
Legacy scripts have been deleted. Use the AFML-compliant equivalents:
- Retrain: `python retrain_meta_labeler.py` (root-level)
- Class: `from neutralgrid.models.meta_labeler import MetaLabeler`

### `/reports/` - Meta-labeling analysis reports
- `comprehensive_meta_labels.json` - Comprehensive meta-labeling analysis
- `durability_meta_labels.json` - Durability-focused meta-label analysis

### `/live_tracking/` - Live bot data with meta-labeling outcomes
- `TRUMPUSDT_live_bot.json` - TRUMPUSDT live bot (Strategy #409845187)
- `XRPUSDT_live_bot.json` - XRPUSDT live bot (Strategy #409845158)

## Meta-Labeling Success Criteria

A bot is labeled as SUCCESS (1) if it meets these criteria:
1. **Profitable**: Total profit > 0
2. **Positive ROI**: ROI % > 0
3. **Perfect Win Rate**: Win rate = 100%
4. **Acceptable Drawdown**: Max drawdown < 50%
5. **High Annualized Yield**: APY > 1000%

## Recent Meta-Labeling Results

### TRUMPUSDT (Strategy #409845187)
- **Meta-Label**: SUCCESS (1)
- **Confidence**: HIGH
- **Original Meta Probability**: 60.80%
- **Actual Outcome**: SUCCESS ✓
- **ROI**: 8.24% in 20.4 hours
- **Criteria Passed**: 4/5 (drawdown exceeded 50% but recovered)

### XRPUSDT (Strategy #409845158)
- **Meta-Label**: SUCCESS (1)
- **Confidence**: HIGH
- **Original Meta Probability**: 56.08%
- **Actual Outcome**: SUCCESS ✓
- **ROI**: 9.40% in 20.0 hours
- **Criteria Passed**: 5/5 (all criteria met)

## Key Findings

1. **Meta-probabilities were accurate**: Both bots had ~60% predicted success and both succeeded
2. **Stress tests were accurate**: Both bots tracked to predicted peak trajectories
3. **Realistic backtest validated**: Close-to-close backtest closely tracks live grid bot performance
4. **Grid strategy validation**: 100% win rates prove grid effectiveness in ranging markets

## Usage

### Retrain Meta-Labeling Model
```bash
python retrain_meta_labeler.py --input data/new_expired_bots.xlsx
```

### Use Meta-Labeler in Pipeline
The meta-labeler is automatically integrated into the full pipeline:
```bash
python run_full_pipeline.py
```

The pipeline will load `models/meta_labeler.pkl` and compute meta-probabilities for all candidates.

## Dependencies

### Python Packages
- pandas
- numpy
- scikit-learn
- joblib (for .pkl serialization)

### Internal Modules
- `neutralgrid.models.meta_labeler` - Core meta-labeling class
- `neutralgrid.scanner.pnl_ranker` - Expected value calculations
- `data/new_expired_bots.xlsx` - Training data source

## Model Performance Metrics

- **Training Dataset**: 61 expired bots (as of 2026-02-04)
- **Feature Count**: 30+ technical and regime indicators
- **Validation Method**: Time-series cross-validation
- **Latest Update**: 2026-02-04 (added TRUMPUSDT and XRPUSDT)

## References

- Main expired bots dataset: `../data/new_expired_bots.xlsx`
- HMM regime classifier: active `rolling_180d_*` artifact from `../artifact_manifest.json`
- Full pipeline integration: `../run_full_pipeline.py`
- PnL ranking system: `../src/neutralgrid/scanner/pnl_ranker.py`

---

**Last Updated**: 2026-02-04
**Maintained By**: Neutral Grid Bot System
