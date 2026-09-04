# AUC & Meta-Labeler Calibration — Data Composition & Code Integrity Plan

## Problem Statement

Adding more training data does not improve the meta-labeler unless the data
**maintains a healthy class balance**.  Backtest rows from lower-score candidates
(score 45-85) almost universally fail the hierarchical labeler (L1 AND L2 AND L3),
producing extreme class imbalance that destroys AUC:

| Run       | Samples | Positive Rate | CV AUC | Failure Mode                       |
|-----------|---------|---------------|--------|------------------------------------|
| 263 rows  | 263     | 26.6%         | 0.645  | Baseline — best achieved           |
| 175 rows  | 175     | 18.3%         | 0.501  | Borderline — signal barely present |
| 701 rows  | 701     | 7.4%          | 0.372  | Collapsed — worse than random      |

Root cause: 438 new negatives added, near-zero new positives.  The GBM class
weight formula (`n_total / (2 * n_class)`) amplified 52 positives by 6.7x,
destabilizing the loss surface across CV folds.

A code audit revealed **6 structural issues** in the calibration pipeline that
compound the data composition problem.  This plan addresses both: the operational
workflow (data quality) and the code defects (calibration integrity).

---

## Audit Summary

Four independent audits examined the meta-labeler training path, label pipeline,
governance, and calibration math.  Three review agents cross-validated the
critical findings.  A final pass verified every line number, variable name, and
code path against the actual source.

### Confirmed Issues (Code Changes Required)

| #  | Issue                                 | Severity | File                              | Exact Lines     |
|----|---------------------------------------|----------|-----------------------------------|-----------------|
| C1 | Double calibration chain              | CRITICAL | `src/neutralgrid/models/meta_labeler.py` | 980, 1153-1157 |
| C2 | Calibration holdout zero-positive     | MEDIUM   | `src/neutralgrid/models/meta_labeler.py` | 935-960        |
| C3 | Positive rate guard missing           | MEDIUM   | `src/neutralgrid/models/meta_labeler.py` | after 531      |
| C4 | Hurdle sync — MetaLabeler unvalidated | LOW      | `retrain_meta_labeler.py`         | after config load |
| C5 | F1 threshold range too narrow         | LOW      | `src/neutralgrid/models/meta_labeler.py` | 1206           |
| C6 | Model backup before overwrite         | LOW      | `retrain_meta_labeler.py`         | before 857      |

### Structural Errors Found in Previous Plan (Corrected Below)

| Error | What Was Wrong | Why It Would Fail | Correction |
|-------|---------------|-------------------|------------|
| Step 1 referenced `self._base_model` | Attribute does not exist in MetaLabeler. Grep returned zero matches. | `AttributeError` at runtime. | Use local variable `base_model` (in scope at line 1153 within `train()`). |
| Step 3 placed guard at line 793 | Line 793 is inside the per-fold CV loop. Guard would execute 6 times, checking fold-level y_train instead of full dataset y. | Guard checks partial data; per-fold positive rate differs from global rate. | Place guard after line 531 where full `y` is created, before CV loop starts at line 748. |
| Step 4 used `hasattr(self, 'meta_labeler')` in Config | Config class (`config.py:483-516`) has no `meta_labeler` field. MetaLabelerConfig is defined in `meta_labeler.py`, separate class hierarchy. | `hasattr` always returns False — fix is a no-op. | Move validation to `retrain_meta_labeler.py` where both Config and MetaLabelerConfig are accessible. |

### Rebuked Findings (No Code Change Needed)

| Finding | Reason Not a Bug |
|---------|-----------------|
| Live rows missing `label_contract_version` | Live rows take legacy path which bypasses version gate. `version_gated=NaN` → `fillna(False)` → included. Verified: not excluded in any current workflow. |
| PnL column precedence in `create_labels()` | Intentional alignment-v1 design: `net_pnl_pct` preferred over gross `pnl_pct`. The `_label_pnl_col_used` audit field records which was used. |
| Feature list divergence (meta_labeler vs unified_builder) | Handled by `resolve_feature_profile()` which intersects config features with available columns. Missing features are logged. |
| Duration filter inconsistency (scanner 6h vs labeler 1h) | Intentional: scanner profile trains on mature bots (6h+), meta-labeler accepts shorter durations for broader coverage. |
| N_eff guard scope (time-decay only) | With 15-30% positive rate, class weights stay in 1.7-3.3x — not enough to reduce N_eff below 30. |
| ECE instability under extreme imbalance | Addressed by composition targets. With 15-30% positive rate, ECE bins have sufficient samples. |
| Brier paradox (low Brier + low AUC) | Implicitly captured by the AUC >= 0.55 deployment gate. Not actionable as a separate guard. |
| Sample weight compounding without per-fold clip | `SampleWeightConfig.max_weight=10.0` cap exists in `sample_weights.py:362`. At target composition, per-fold class weights stay below 3.3x. |
| Weighted AUC reporting mismatch | Test weights at `meta_labeler.py:838-841` use uniqueness + time-decay only, NOT class weights. Reported AUC IS on natural distribution. |
| Legacy path dedup missing | Unified path handles dedup by candidate_id with source ranking. Legacy is the fallback for simple workflows. |
| `_is_calibrated` flag logic at line 1158 | Correctly evaluates: True when at least one calibrator succeeded, False when both failed. `_PostProbabilityCalibrator._apply()` returns raw clipped probabilities when `calibrator=None`. |
| Save format (artifact vs pickle) inconsistency | `save()` and `load()` handle both paths internally. Users call `labeler.save()`/`MetaLabeler.load()` — never raw `joblib.load()`. |

---

## Implementation Plan

### Step 1 — Fix Double Calibration Chain (C1)

**File:** `src/neutralgrid/models/meta_labeler.py`
**Line:** 1153

**What's wrong:** When temporal holdout calibration succeeds (line 980),
`self._model` becomes a `CalibratedClassifierCV`.  Then line 1153 wraps
`self._model` in `_PostProbabilityCalibrator` without checking its type.
At inference (line 191), `_PostProbabilityCalibrator.predict_proba()` calls
`self.base_model.predict_proba(X)` — which returns already-calibrated
probabilities — then applies the OOS calibrator (which was trained on raw
CV predictions from uncalibrated per-fold models at line 833).

Result: `raw GBM → sigmoid-calibrated → OOS-calibrated` (double transformation).
The OOS calibrator learned `raw → frequency` but receives `calibrated → ?`,
producing distorted deployment probabilities.

**Why this must be fixed:** This is the highest-impact code defect.  Double
calibration compresses the probability scale incorrectly.  It can invert
candidate rankings and compress all meta_prob values toward a narrow band,
making the model unable to discriminate good from bad candidates.

**Fix:** At line 1153, use the local variable `base_model` (the raw GBM,
fitted at line 927 and still in scope within `train()`) when OOS calibration
succeeded.  Preserve temporal calibration when OOS did not apply:

```python
# Line 1153 — replace current wrapping:
if oos_calibrator is not None:
    # OOS calibrator was trained on raw CV predictions (line 833),
    # so it must receive raw model output — not temporally-calibrated output.
    base_for_oos = base_model
else:
    # No OOS calibrator — preserve temporal calibration if it succeeded.
    base_for_oos = self._model
self._model = _PostProbabilityCalibrator(
    base_model=base_for_oos,
    calibrator=oos_calibrator,
    method=oos_method,
)
```

**Why `base_model` and not `self._base_model`:** `self._base_model` does not
exist (grep confirms zero matches).  `base_model` is the local variable
created by `_make_model()` at line 626 and fitted at line 927.  It is in scope
at line 1153 because both are within the `train()` method body.

**Why conditional on `oos_calibrator is not None`:** If OOS calibration did not
run (too few samples, single class), `oos_calibrator` is `None` and
`_PostProbabilityCalibrator._apply()` returns raw clipped probabilities
(line 175-176).  In that case, preserving the temporal calibration in
`self._model` is correct — it's the only calibration available.

**Risk:** Low.  Does not alter the training loop, CV metrics, or OOS
calibrator's training data.  Only changes which model feeds the OOS wrapper.

---

### Step 2 — Add Calibration Holdout Class Balance Guard (C2)

**File:** `src/neutralgrid/models/meta_labeler.py`
**Lines:** 935-936 (insert after existing size guard)

**What's wrong:** The temporal holdout guard at line 936 checks only sample
count (`cal_split >= 50` and `holdout >= 20`), not class distribution.
`CalibratedClassifierCV.fit()` with all-zero `y` produces a degenerate
sigmoid where `predict_proba()` returns NaN.  The downstream clip at line 188
(`np.clip(p, 0.0, 1.0)`) does not catch NaN — `np.clip(NaN, 0, 1)` returns
NaN — which propagates to deployment decisions.

**Why this must be fixed:** With 68 samples at 7% positive rate (5 positives)
and 20% holdout (14 samples), P(zero positives in holdout) ≈ 5%.  This is a
realistic failure mode at current dataset size.

The OOS calibration phase already has this guard at line 1007:
`len(set(oos_y_true)) >= 2`.  The temporal holdout phase lacks the equivalent.

**Fix:** Insert class distribution check inside the existing size-guard block,
after line 936 and before line 937:

```python
# After line 936: if cal_split >= 50 and (len(X) - cal_split) >= 20:
    cal_y = np.asarray(y)[cal_split:]
    if len(np.unique(cal_y)) < 2:
        logger.warning(
            "Calibration holdout has only class %d (%d samples); "
            "skipping temporal calibration",
            int(cal_y[0]), len(cal_y),
        )
        self._model = base_model
        self._is_calibrated = False
    else:
        # ... existing calibration code (lines 937-985) goes here
```

**Risk:** None.  Mirrors the existing OOS guard.  When triggered, falls back
to the raw GBM — same as when `calibrate_probabilities=False`.

---

### Step 3 — Add Positive Rate Guard in train() (C3)

**File:** `src/neutralgrid/models/meta_labeler.py`
**Line:** After line 531 (where full `y` is created), before CV loop at line 748

**What's wrong:** Class weights are applied unconditionally when
`use_class_weights=True` (line 793).  At 7% positive rate, the positive class
weight is 7.1x — far beyond what the GBM (`max_depth=3`, `min_samples_leaf=15`)
can handle stably.  No warning is emitted.

**Why line 531, not line 793:** Line 793 is inside the per-fold CV loop — it
would execute 6 times checking fold-level `y_train`, not the global positive
rate.  The guard must check the full `y` (created at line 523 or 531) once
before training begins.

**Fix:** Insert after line 531 (after `y` is created from either pre-computed
labels or `create_labels()`):

```python
# After line 531, before feature preparation:
_global_pos_rate = float(np.mean(np.asarray(y) == 1))
if _global_pos_rate < 0.05:
    raise ValueError(
        f"Positive rate {_global_pos_rate:.1%} is below 5%. "
        f"Training would produce an unstable model (class weight "
        f">{1.0 / (2.0 * _global_pos_rate):.0f}x). "
        f"Fix data composition before retraining."
    )
if _global_pos_rate < 0.15:
    logger.warning(
        "Positive rate %.1f%% is below recommended 15%%. "
        "Class weights will amplify positives by %.1fx. "
        "Consider raising --min-score or downsampling negatives.",
        _global_pos_rate * 100,
        1.0 / (2.0 * _global_pos_rate),
    )
```

**Thresholds rationale:**
- 5% hard floor: positive weight = 10x (hits `SampleWeightConfig.max_weight`
  cap).  GBM with `min_samples_leaf=15` cannot learn stable splits.
- 15% warning: positive weight = 3.3x.  Workable but marginal.

**Risk:** Low.  The 5% floor prevents obviously broken models.  The 15%
warning informs without blocking.

---

### Step 4 — Add Hurdle Synchronization for MetaLabeler (C4)

**File:** `retrain_meta_labeler.py`
**Line:** After config loading, before `MetaLabelerConfig` construction

**What's wrong:** `Config._validate()` at `config.py:591-598` checks
`HierarchicalLabelConfig.hurdle_pct` against `BarrierConfig.meta_hurdle_pct`,
but `MetaLabelerConfig.hurdle_pct` is never validated against either.

**Why not in `config.py`:** The `Config` class (`config.py:483-516`) has no
`meta_labeler` field.  `MetaLabelerConfig` is defined in `meta_labeler.py:197`
as a separate class.  Adding it to `Config` would require a schema change —
over-engineering for this fix.

**Why `retrain_meta_labeler.py`:** This is where both `Config` (via
`get_config()`) and `MetaLabelerConfig` (via `--hurdle-pct` CLI arg) are
accessible.  The CLI arg `--hurdle-pct` (default: 3.0) overrides the
dataclass default.  If someone passes `--hurdle-pct 3.5` while Config has
`meta_hurdle_pct=3.0`, labels diverge silently.

**Fix:** After config loading in `retrain_meta_labeler.py`, add:

```python
# After get_config() call, before MetaLabelerConfig construction:
cfg = get_config()
if abs(args.hurdle_pct - cfg.barrier.meta_hurdle_pct) > 1e-6:
    raise ValueError(
        f"--hurdle-pct ({args.hurdle_pct}) != "
        f"BarrierConfig.meta_hurdle_pct ({cfg.barrier.meta_hurdle_pct}). "
        f"These must match to prevent label/deployment divergence. "
        f"Update .env or config if the hurdle has changed."
    )
```

**Risk:** None.  Pure validation before training starts.  Uses existing
`get_config()` which is already imported in the file.

---

### Step 5 — Widen F1 Threshold Tuning Range (C5)

**File:** `src/neutralgrid/models/meta_labeler.py`
**Line:** 1206

**What's wrong:** The F1 threshold is tuned over `np.arange(0.30, 0.75, 0.05)`.
With positive rate below 30%, the optimal F1 threshold is often below 0.30
because the model's predicted probabilities concentrate in a lower range.  The
current range misses the optimum and defaults to 0.30 (the lower bound).

**Fix:** Replace line 1206 with adaptive range:

```python
# Line 1206 — replace fixed range:
_oos_pos_rate = float(np.mean(oos_yt))
_f1_lower = max(0.05, _oos_pos_rate * 0.5)
_f1_upper = min(0.80, max(_oos_pos_rate * 4.0, _f1_lower + 0.10))
for thr in np.arange(_f1_lower, _f1_upper, 0.02):
```

**Why these bounds:** For a well-calibrated classifier, the optimal decision
threshold is approximately equal to the positive rate (Bayes-optimal
threshold).  Searching from `0.5 × pos_rate` to `4 × pos_rate` covers the
plausible range.  The finer step (0.02 vs 0.05) improves resolution.  The
`max(..., _f1_lower + 0.10)` ensures the range is at least 0.10 wide.

**Variables `oos_yt` and `oos_yp`:** Already defined at lines 1204-1205.
No new variables needed.

**Risk:** Low.  Only affects which threshold is stored in
`MetaLabelerMetrics.f1_threshold`.  Existing thresholds in 0.30-0.75 remain
reachable when positive rate is high enough.

---

### Step 6 — Add Model Backup Before Overwrite (C6)

**File:** `retrain_meta_labeler.py`
**Line:** Before line 857 (`labeler.save(output_path)`)

**What's wrong:** Line 857 calls `labeler.save(output_path)` which writes
directly to `--output` (default: `models/meta_labeler.pkl`).  No backup.  A
poor retrain permanently overwrites the previous model with no rollback.

**Why this must be fixed:** The AUC 0.645 model was nearly lost when Run 3
(AUC 0.372) overwrote it.  A timestamped backup costs nothing.

**Fix:** Insert before line 857:

```python
import shutil
from datetime import datetime

output_path = Path(args.output)
if output_path.exists():
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = output_path.with_name(
        f"{output_path.stem}_backup_{timestamp}{output_path.suffix}"
    )
    shutil.copy2(output_path, backup_path)
    logger.info("Backed up previous model to %s", backup_path)
```

**Risk:** None.  Pure defensive copy.  Does not touch training or inference.

---

## Operational Workflow

The following sections are operational guidance.  They do not require code
changes — they document best practices for data composition.

### Composition Targets

These targets must be met **before** calling `retrain_meta_labeler.py`:

| Metric               | Minimum | Target   | Maximum | Why                                           |
|----------------------|---------|----------|---------|-----------------------------------------------|
| Positive rate        | 15%     | 20-30%   | 50%     | Class weight stays in 1.0-3.3x range          |
| Effective samples    | 60      | 100-200  | —       | CV folds have enough signal per fold           |
| Live row fraction    | 10%     | 30%+     | —       | Live rows (1.0x weight) anchor the model       |
| Feature coverage     | 70%     | 90%+     | —       | Avoid imputation-dominated features            |

### Class Weight Stability Zone

The balanced formula in `meta_labeler.py:797`:
```
w_class[c] = n_total / (2 * n_class[c])
```

| Positive Rate | Positive Weight | Negative Weight | Stability |
|---------------|-----------------|-----------------|-----------|
| 7%            | 7.1x            | 1.08x           | Unstable — blocked by Step 3 (< 5% raises, < 15% warns) |
| 15%           | 3.3x            | 1.18x           | Marginal — Step 3 warns |
| 20%           | 2.5x            | 1.25x           | Good |
| 25%           | 2.0x            | 1.33x           | Good |
| 30%           | 1.7x            | 1.43x           | Good |

### The Accumulation Loop

```
  Run pipeline  ──>  Backtest top candidates  ──>  Deploy best  ──>  Harvest outcomes
       |                     |                          |                    |
       v                     v                          v                    v
  deployment_ready_*.csv  training_data_*.csv    live bots expire    new_expired_bots.xlsx
                          (weight=0.5)                               (weight=1.0)
                               |                                          |
                               └──────────> retrain_meta_labeler <────────┘
                                                     |
                                                     v
                                            models/meta_labeler.pkl
                                                     |
                                                     v
                                            run_full_pipeline.py  (better candidates)
```

### Pre-Retrain Checklist

1. **Positive rate >= 15%** across combined live + backtest data
2. **Feature coverage >= 70%** — run `--backfill` if below
3. **Label contract version matches** — check for `version_gated` warnings
4. **Duration alignment** — all rows generated at current `STANDARD_HORIZON_HOURS`
5. **No stale backtest rows** — archive `training_data_*.csv` from before
   any horizon/hurdle config change

### Score-Gated Backtest Inclusion

When running `backtest_candidates.py`, control the score floor:

```bash
python backtest_candidates.py --min-score 75 --max-candidates 300
```

Check the output positive rate before including in training:

```bash
python -c "
import pandas as pd
from pathlib import Path

latest = sorted(Path('data/backtest_candidates').glob('training_data_*.csv'))[-1]
df = pd.read_csv(latest)
y_col = 'y' if 'y' in df.columns else 'hlabel_meta'
pos = (df[y_col] == 1).sum()
print(f'{latest.name}: {len(df)} rows, {pos} pos ({100*pos/max(len(df),1):.1f}%)')
if pos / max(len(df), 1) < 0.15:
    print('WARNING: Positive rate too low. Raise --min-score.')
"
```

---

## Anti-Patterns

### 1. Bulk-Adding Low-Score Backtests

**Wrong:** `--min-score 45 --max-candidates 800` → 7% positive rate → AUC collapse.
**Right:** `--min-score 75 --max-candidates 300` → ~20% positive rate.

### 2. Mixing Horizon Periods

**Wrong:** Training on rows from 6h + 12h horizons.  Labels encode different
time windows — the model cannot learn a consistent decision boundary.
**Right:** Archive old training CSVs when changing `STANDARD_HORIZON_HOURS`.

### 3. Forcing Calibration on Bad Data

**Wrong:** AUC = 0.37, but calibrator still enabled.  The calibrator fits a
monotonic transform on inverted predictions.
**Right:** Step 3 guard now blocks training at < 5% positive rate and warns
at < 15%.  Fix composition first.

### 4. Retraining on Every New Data Point

**Wrong:** Retrain after each expired bot.  The model oscillates.
**Right:** Batch retrains after 10+ new live outcomes.

---

## Decision Flowchart

```
Have new data to add?
    |
    v
Check positive rate of combined set
    |
    +--> >= 15% and <= 50%
    |       |
    |       v
    |    Check feature coverage >= 70%
    |       |
    |       +--> Yes --> Check duration alignment
    |       |               |
    |       |               +--> Aligned --> Retrain
    |       |               |               (Step 3 guard validates at train time)
    |       |               |
    |       |               +--> Misaligned --> Archive old CSVs,
    |       |                                   re-backtest, retrain
    |       |
    |       +--> No --> Run --backfill, check again
    |
    +--> < 15%
    |       |
    |       v
    |    Step 3 guard will WARN at train time.
    |    Options:
    |    A) Raise --min-score floor
    |    B) Downsample negatives
    |    C) Wait for more live outcomes
    |
    +--> < 5%
    |       |
    |       v
    |    Step 3 guard will BLOCK training with ValueError.
    |    Must fix data before proceeding.
    |
    +--> > 50%
            |
            v
         Check for label inflation:
         - hurdle_pct too low?
         - min_fills too lenient?
```

---

## Summary

### Code Changes (6 steps, ordered by criticality)

| Step | What                              | File                              | Exact Location                    | Lines of Code | Risk |
|------|-----------------------------------|-----------------------------------|-----------------------------------|---------------|------|
| 1    | Fix double calibration            | `meta_labeler.py`                 | Replace lines 1153-1157           | ~8            | Low  |
| 2    | Holdout class balance guard       | `meta_labeler.py`                 | Insert after line 936             | ~8            | None |
| 3    | Positive rate guard               | `meta_labeler.py`                 | Insert after line 531             | ~12           | Low  |
| 4    | Hurdle sync for MetaLabeler       | `retrain_meta_labeler.py`         | After config load                 | ~6            | None |
| 5    | Adaptive F1 threshold range       | `meta_labeler.py`                 | Replace line 1206                 | ~4            | Low  |
| 6    | Model backup before overwrite     | `retrain_meta_labeler.py`         | Insert before line 857            | ~7            | None |

Total: ~45 lines of code across 2 files.  No new files.  No schema changes.
No dependency additions.

### Golden Rules (Operational)

1. **Positive rate 15-30%** — checked before every retrain, enforced by Step 3
2. **Score floor >= 75** for backtest inclusion
3. **Live rows > backtest rows** over time
4. **Same horizon everywhere**
5. **Batch retrains** — 10+ new live outcomes
6. **AUC >= 0.55 or don't deploy**
7. **Check calibration health post-retrain**
