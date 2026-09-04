# Quick Start Guide - AFML Enhanced Features

**Date**: 2026-02-04
**Purpose**: Get started with newly implemented AFML features

---

## What's New

Three major AFML enhancements are now available:

1. **Backfilled Training Features** - 7/8 missing features recovered
2. **Trial Tracking** - Automatic logging for deflated Sharpe calculation
3. **Holdout Validation** - Station 4 compliance tracking

---

## 1. Using Backfilled Training Data

### Training with Full Features

**Before**:
```bash
python retrain_meta_labeler.py --input data/new_expired_bots.xlsx --output models/meta_labeler.pkl
# Only 6/14 features available (43% coverage)
# Model AUC ~0.60-0.65
```

**After**:
```bash
python retrain_meta_labeler.py --input data/new_expired_bots_backfilled.xlsx --output models/meta_labeler.pkl
# 13/14 features available (93% coverage)
# Expected AUC improvement: ~0.65-0.75
```

### Verify Feature Coverage

```python
import pandas as pd

df = pd.read_excel('data/new_expired_bots_backfilled.xlsx')

features = [
    'range_prob', 'trend_prob', 'utility_score',
    'survival_prob', 'hurst_exponent', 'ou_halflife',
    'funding_rate', 'ev_score',
    'adx_1h', 'adx_15m', 'rsi_15m',
    'range_size_pct', 'num_grids', 'profit_per_grid_pct'
]

for feat in features:
    if feat in df.columns:
        coverage = (~df[feat].isna()).sum()
        print(f"{feat:25s}: {coverage}/{len(df)} ({coverage/len(df)*100:.1f}%)")
```

**Expected Output**:
```
range_prob               : 61/61 (100.0%)
trend_prob               : 61/61 (100.0%)
utility_score            : 61/61 (100.0%)
survival_prob            : 61/61 (100.0%)
hurst_exponent           : 61/61 (100.0%)
ou_halflife              : 55/61 (90.2%)
funding_rate             : 0/61 (0.0%)   # API limitation
ev_score                 : 61/61 (100.0%)
```

---

## 2. Trial Tracking

### Automatic Logging

**Every training run now logs a trial automatically**:

```bash
python retrain_meta_labeler.py --input data/new_expired_bots_backfilled.xlsx --output models/meta_labeler.pkl

# Output will include:
# Logging trial to trial tracker...
#   Total trials logged: 1
#   (After subsequent runs: Mean CV AUC, Best CV AUC, etc.)
```

### View Trial History

```python
from neutralgrid.training.trial_tracker import TrialTracker

tracker = TrialTracker()

# Get summary
stats = tracker.summary_stats('meta_labeler')
print(f"Total trials: {stats['count']}")
print(f"Mean CV AUC: {stats.get('cv_score_mean', 0):.4f}")
print(f"Best CV AUC: {stats.get('cv_score_max', 0):.4f}")

# Get recent trials
recent = tracker.get_trials('meta_labeler', limit=5)
for trial in recent:
    print(f"{trial.trial_id}: AUC={trial.cv_score:.4f}, Features={len(trial.feature_set)}")
```

### Compute Deflated Sharpe

**After accumulating trials**:

```python
from neutralgrid.training.trial_tracker import TrialTracker

tracker = TrialTracker()

# Compute deflated Sharpe for observed performance
result = tracker.compute_deflated_sharpe(
    observed_sharpe=1.5,           # Your backtest Sharpe
    model_type='meta_labeler',
    n_observations=252             # Number of daily returns
)

print(f"Raw Sharpe:        {result['raw_sharpe']:.4f}")
print(f"Deflated Sharpe:   {result['deflated_sharpe']:.4f}")
print(f"Haircut:           {result['haircut_pct']:.1f}%")
print(f"N Trials:          {result['n_trials']}")
print(f"P(False Discovery): {result['prob_false_discovery']:.4f}")

if result['prob_false_discovery'] > 0.5:
    print("⚠️ High overfitting risk!")
```

**Interpretation**:
- **Haircut < 30%**: Reasonable given trial count
- **Haircut 30-50%**: Moderate multiple testing penalty
- **Haircut > 50%**: High overfitting risk
- **P(False Discovery) > 0.5**: Likely overfitted

### Manual Trial Logging

```python
from neutralgrid.training.trial_tracker import TrialTracker, TrialRecord
from datetime import datetime

tracker = TrialTracker()

# Log a manual trial (e.g., from notebook experiments)
trial = TrialRecord(
    trial_id='experiment_custom_features_001',
    timestamp=datetime.now(),
    model_type='meta_labeler',
    hyperparameters={
        'n_estimators': 150,
        'max_depth': 7,
        'learning_rate': 0.05,
    },
    cv_score=0.73,
    feature_set=['range_prob', 'trend_prob', 'utility_score'],
    notes='Testing custom feature subset'
)

tracker.log_trial(trial)
```

---

## 3. Holdout Validation

### Mark Holdout Creation

**When first creating holdout split**:

```python
from neutralgrid.training.holdout_validator import HoldoutValidator
from neutralgrid.backtest.cpcv import CPCV, CPCVConfig

# Set up CPCV with holdout
config = CPCVConfig(
    n_groups=6,
    n_test_groups=2,
    purge_hours=48.0,
    embargo_hours=6.0,
    holdout_pct=0.2  # 20% holdout
)

cpcv = CPCV(config)
cv_df, holdout_df, cv_splits = cpcv.split_with_holdout(
    df=training_data,
    timestamp_col='start_time_utc',
    t1_col='t1'
)

# Log holdout creation
validator = HoldoutValidator()
validator.mark_holdout_created(
    cv_samples=len(cv_df),
    holdout_samples=len(holdout_df),
    cv_end_date=cv_df['start_time_utc'].max().isoformat(),
    holdout_start_date=holdout_df['start_time_utc'].min().isoformat(),
    notes='Initial holdout for meta-labeler training'
)

print(f"✅ Holdout created: {len(holdout_df)} samples reserved")
```

### Mark Holdout Access

**Before final evaluation ONLY**:

```python
validator.mark_holdout_accessed(
    purpose='final_evaluation',
    caller='evaluate_final_model.py',
    notes='Final out-of-sample evaluation before deployment'
)
```

**⚠️ Never access holdout for**:
- Hyperparameter tuning
- Feature selection
- Model selection
- Performance debugging

### Validate Integrity

**Before deployment**:

```python
result = validator.validate_integrity()

print(validator.summary())

if result['contaminated']:
    raise ValueError(
        f"Holdout contaminated! {result['premature_access_count']} premature access(es)"
    )

print("✅ Holdout integrity verified")
```

**Example Output (Valid)**:
```
✅ VALID: Holdout integrity preserved
  Created: 2026-02-04T16:00:00
  Access count: 1
```

**Example Output (Contaminated)**:
```
❌ CONTAMINATED: 2 premature access(es)
  - 2026-02-03T14:30:00: model_selection by tune_hyperparams.py
  - 2026-02-04T10:00:00: feature_analysis by analyze_features.py
```

---

## 4. Complete Training Workflow

### Full End-to-End Example

```bash
#!/bin/bash

# Step 1: Verify backfilled data
python -c "
import pandas as pd
df = pd.read_excel('data/new_expired_bots_backfilled.xlsx')
print(f'Samples: {len(df)}')
print(f'Features: {len([c for c in df.columns if c in ['range_prob', 'trend_prob', 'utility_score', 'survival_prob', 'hurst_exponent', 'ou_halflife', 'ev_score']])}')
"

# Step 2: Retrain meta-labeler (auto-logs trial)
python retrain_meta_labeler.py \
  --input data/new_expired_bots_backfilled.xlsx \
  --output models/meta_labeler.pkl \
  --hurdle-pct 5.0 \
  --sl-pct -10.0 \
  --horizon-hours 48.0

# Step 3: View trial history
python -c "
from neutralgrid.training.trial_tracker import TrialTracker
tracker = TrialTracker()
stats = tracker.summary_stats('meta_labeler')
print(f'Total trials: {stats['count']}')
if stats['count'] > 0:
    print(f'Mean AUC: {stats.get('cv_score_mean', 0):.4f}')
"

# Step 4: Run deployment pipeline
python run_full_pipeline.py --top-n 50 --min-score 40

# Step 5: Check deployment results
ls -lh results/deployment_ready*.csv | tail -1
```

---

## 5. Best Practices

### Trial Tracking

✅ **DO**:
- Run training with different hyperparameters
- Log trials from notebooks/experiments
- Track feature subsets
- Compute deflated Sharpe after 10+ trials

❌ **DON'T**:
- Delete trial log
- Fake trial scores
- Skip logging for "quick tests"

### Holdout Validation

✅ **DO**:
- Mark holdout creation immediately
- Only access for final evaluation
- Validate before deployment
- Document all access in notes

❌ **DON'T**:
- Peek at holdout during development
- Use holdout for hyperparameter tuning
- Access multiple times for different purposes
- Skip validation checks

### Feature Backfilling

✅ **DO**:
- Use backfilled data for retraining
- Log features prospectively for new bots
- Document data sources
- Verify coverage before training

❌ **DON'T**:
- Backfill with future data (look-ahead bias)
- Use different feature computation for backfill vs live
- Ignore NaN values without investigation

---

## 6. Troubleshooting

### Issue: "Trial log not found"

**Solution**:
```bash
# Trial log is created on first use
# Just run training once and it will be created automatically
python retrain_meta_labeler.py --input data/new_expired_bots_backfilled.xlsx --output models/meta_labeler.pkl
```

### Issue: "Holdout access log not found"

**Solution**:
```python
# Create manually if needed
from neutralgrid.training.holdout_validator import HoldoutValidator
validator = HoldoutValidator()
# Log will be created automatically
```

### Issue: "Low feature coverage in backfilled data"

**Check**:
```python
import pandas as pd
df = pd.read_excel('data/new_expired_bots_backfilled.xlsx')
print(df[['range_prob', 'trend_prob', 'survival_prob']].describe())

# If all NaN, re-run backfill:
# python scripts/backfill_training_features.py
```

### Issue: "Deflated Sharpe shows huge haircut"

**This is normal if**:
- You have many trials (>100)
- Sharpe is low (<1.0)
- High variability in trial scores

**Action**: Report both raw and deflated Sharpe in papers/reports

---

## 7. Quick Reference

### Files Locations

```
data/
├── new_expired_bots.xlsx              # Original data
├── new_expired_bots_backfilled.xlsx   # Backfilled (USE THIS)
├── trial_log.json                     # Trial history
└── holdout_access_log.json            # Holdout access log

scripts/
└── backfill_training_features.py      # Backfill script

src/neutralgrid/training/
├── trial_tracker.py                   # Trial tracking
└── holdout_validator.py               # Holdout validation

models/
└── meta_labeler.pkl                   # Trained model
```

### Command Cheat Sheet

```bash
# Retrain with backfilled data
python retrain_meta_labeler.py --input data/new_expired_bots_backfilled.xlsx --output models/meta_labeler.pkl

# View trial history
python -c "from neutralgrid.training.trial_tracker import TrialTracker; print(TrialTracker().summary_stats('meta_labeler'))"

# Validate holdout
python -c "from neutralgrid.training.holdout_validator import HoldoutValidator; print(HoldoutValidator().summary())"

# Run full pipeline
python run_full_pipeline.py --top-n 50

# Check backfill coverage
python -c "import pandas as pd; df=pd.read_excel('data/new_expired_bots_backfilled.xlsx'); print(f'Range prob: {(~df.range_prob.isna()).sum()}/{len(df)}')"
```

---

## 8. Expected Improvements

### Before Backfill

- Features: 6/14 (43%)
- Meta-labeler AUC: ~0.60-0.65
- Model relies heavily on grid parameters

### After Backfill

- Features: 13/14 (93%)
- Meta-labeler AUC: ~0.65-0.75 (expected +0.05-0.10)
- Model can leverage regime and stochastic features

### With Trial Tracking

- Can compute deflated Sharpe after 10+ trials
- Quantify overfitting probability
- More rigorous performance claims

### With Holdout Validation

- Station 4 compliant
- Provable out-of-sample guarantee
- Increased credibility for research

---

## Need Help?

See full documentation:
- `AFML_PIPELINE_COMPLETE_ANALYSIS.md` - Complete pipeline analysis
- `AFML_IMPLEMENTATION_STATUS.md` - Implementation details
- `AFML_QUICK_REFERENCE.md` - Code snippets

---

**Document Version**: 1.0
**Last Updated**: 2026-02-04
**Quick Start Time**: ~5 minutes
