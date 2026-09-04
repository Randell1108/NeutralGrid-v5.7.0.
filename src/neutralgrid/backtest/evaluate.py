"""
Walk-forward evaluation for HMM regime detection.

AFML-aligned principle: "Don't optimize by backtest; validate out-of-sample."

This module implements rolling time-series splits to evaluate model performance
without look-ahead bias. Models are trained on historical windows and tested
on subsequent unseen periods.

Design principles:
- Time-series splits (no random shuffling)
- Train on past, test on future (no look-ahead)
- Multiple splits for robustness
- Clear evaluation metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, cast
import numpy as np
import pandas as pd
import warnings

logger = logging.getLogger(__name__)

from neutralgrid.core.config import get_config
from neutralgrid.models.hmm.train import train_hmm_global
from neutralgrid.data.features import compute_hmm_features

# Phase 2: Temperature scaling for CPCV evaluation
try:
    from neutralgrid.calibration.temperature_scaling_v20260311 import TemperatureScaler
    TEMP_SCALING_AVAILABLE = True
except ImportError:
    TEMP_SCALING_AVAILABLE = False


@dataclass
class WalkForwardSplit:
    """A single train/test split in walk-forward evaluation."""

    split_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_symbols: List[str]
    test_symbols: List[str]


@dataclass
class SplitEvaluationResult:
    """Evaluation result for a single split."""

    split_id: int
    train_size: int
    test_size: int

    # Classification metrics
    test_range_prob_mean: float
    test_trend_prob_mean: float
    test_range_prob_std: float
    test_trend_prob_std: float

    # Gating metrics (using config thresholds)
    num_pass_gate: int
    num_fail_gate: int
    pass_rate: float

    # Detailed predictions
    predictions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WalkForwardEvaluation:
    """Complete walk-forward evaluation results."""

    num_splits: int
    splits: List[WalkForwardSplit]
    results: List[SplitEvaluationResult]

    # Aggregated metrics
    mean_pass_rate: float
    std_pass_rate: float
    mean_range_prob: float
    mean_trend_prob: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_splits": self.num_splits,
            "mean_pass_rate": float(self.mean_pass_rate),
            "std_pass_rate": float(self.std_pass_rate),
            "mean_range_prob": float(self.mean_range_prob),
            "mean_trend_prob": float(self.mean_trend_prob),
            "splits": [
                {
                    "split_id": r.split_id,
                    "train_size": r.train_size,
                    "test_size": r.test_size,
                    "pass_rate": float(r.pass_rate),
                    "range_prob_mean": float(r.test_range_prob_mean),
                    "trend_prob_mean": float(r.test_trend_prob_mean),
                }
                for r in self.results
            ],
        }


def create_walk_forward_splits(
    datasets: Dict[str, pd.DataFrame],
    train_days: int = 90,
    test_days: int = 14,
    step_days: int = 14,
    min_train_bars: int = 200,
    embargo_days: float = 2.0,
) -> List[WalkForwardSplit]:
    """
    Create rolling time-series splits for walk-forward evaluation.

    Args:
        datasets: Dict mapping symbol to DataFrame with 15m data
        train_days: Training window size in days
        test_days: Test window size in days
        step_days: Step size between splits in days
        min_train_bars: Minimum bars required in train set
        embargo_days: Gap between train_end and test_start to prevent
            leakage from labels spanning the boundary (default 2.0 days,
            matching ~6h label horizon).

    Returns:
        List of WalkForwardSplit objects

    Example (embargo_days=2):
        Split 0: Train [Day 0 - Day 90], Test [Day 92 - Day 106]
        Split 1: Train [Day 14 - Day 104], Test [Day 106 - Day 120]
        ...
    """
    if not datasets:
        return []

    # Find earliest and latest timestamps across all symbols
    all_starts = [df["open_time"].min() for df in datasets.values() if not df.empty]
    all_ends = [df["close_time"].max() for df in datasets.values() if not df.empty]

    if not all_starts or not all_ends:
        return []

    data_start = cast(datetime, min(all_starts))
    data_end = cast(datetime, max(all_ends))

    splits = []
    split_id = 0

    # Walk forward through time
    current_train_start = data_start

    while True:
        train_end = current_train_start + timedelta(days=train_days)
        test_start = train_end + timedelta(days=embargo_days)
        test_end = test_start + timedelta(days=test_days)

        # Check if we have enough data
        if test_end > data_end:
            break

        # Validate train size
        train_bars = 0
        for df in datasets.values():
            mask = (df["open_time"] >= current_train_start) & (df["close_time"] <= train_end)
            train_bars += mask.sum()

        if train_bars < min_train_bars:
            warnings.warn(f"Split {split_id} has insufficient training data ({train_bars} bars)")
            current_train_start += timedelta(days=step_days)
            continue

        # Create split
        split = WalkForwardSplit(
            split_id=split_id,
            train_start=current_train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_symbols=list(datasets.keys()),
            test_symbols=list(datasets.keys()),
        )

        splits.append(split)
        split_id += 1

        # Move to next split
        current_train_start += timedelta(days=step_days)

    return splits


def evaluate_split(
    split: WalkForwardSplit,
    datasets: Dict[str, pd.DataFrame],
    n_components: int = 4,
    covariance_type: str = "diag",
    n_iter: int = 200,
    tol: float = 1e-3,
    random_state: int = 7,
) -> SplitEvaluationResult:
    """
    Evaluate a single walk-forward split.

    Note: This uses last-bar EMA smoothing (``_ema_smooth_posteriors``) for
    backward compatibility with walk-forward evaluation.  The CPCV per-bar
    evaluation in ``run_cpcv_hmm_evaluation`` uses the full-matrix variant
    (``_ema_smooth_posteriors_full``) as the primary method.

    Args:
        split: WalkForwardSplit to evaluate
        datasets: Dict mapping symbol to DataFrame
        n_components: HMM components
        covariance_type: HMM covariance type
        n_iter: HMM iterations
        tol: HMM tolerance
        random_state: Random seed

    Returns:
        SplitEvaluationResult with evaluation metrics
    """
    # Extract train data
    train_dfs = []
    for symbol in split.train_symbols:
        if symbol not in datasets:
            continue
        df = datasets[symbol]
        mask = (df["open_time"] >= split.train_start) & (df["close_time"] <= split.train_end)
        df_train = df[mask].copy()
        if len(df_train) >= 60:
            train_dfs.append(df_train)

    if len(train_dfs) < 3:
        warnings.warn(f"Split {split.split_id}: Insufficient training sequences")
        return SplitEvaluationResult(
            split_id=split.split_id,
            train_size=0,
            test_size=0,
            test_range_prob_mean=0.0,
            test_trend_prob_mean=0.0,
            test_range_prob_std=0.0,
            test_trend_prob_std=0.0,
            num_pass_gate=0,
            num_fail_gate=0,
            pass_rate=0.0,
        )

    # Train model on train window
    try:
        result = train_hmm_global(
            per_symbol_dfs=train_dfs,
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
        )
    except Exception as e:
        warnings.warn(f"Split {split.split_id}: Training failed: {e}")
        return SplitEvaluationResult(
            split_id=split.split_id,
            train_size=sum(len(df) for df in train_dfs),
            test_size=0,
            test_range_prob_mean=0.0,
            test_trend_prob_mean=0.0,
            test_range_prob_std=0.0,
            test_trend_prob_std=0.0,
            num_pass_gate=0,
            num_fail_gate=0,
            pass_rate=0.0,
        )

    model = result["model"]
    scaler = result["scaler"]
    trend_state = result["trend_state"]
    range_state = result["range_state"]

    # Extract test data and make predictions
    predictions = []
    range_probs = []
    trend_probs = []

    for symbol in split.test_symbols:
        if symbol not in datasets:
            continue

        df = datasets[symbol]
        mask = (df["open_time"] >= split.test_start) & (df["close_time"] <= split.test_end)
        df_test = df[mask].copy()

        if len(df_test) < 60:
            continue

        # Compute features
        X, valid = compute_hmm_features(cast(pd.DataFrame, df_test))
        X_valid = X[valid]

        if X_valid.shape[0] < 60:
            continue

        # Scale and predict
        X_scaled = scaler.transform(X_valid)
        posteriors = model.predict_proba(X_scaled)

        # Apply same EMA smoothing as production inference
        from neutralgrid.models.hmm.inference import _ema_smooth_posteriors
        _cfg = get_config()
        smooth_k = int(_cfg.hmm.smooth_k)
        smoothed = _ema_smooth_posteriors(posteriors, smooth_k)
        range_prob = float(smoothed[range_state])
        trend_prob = float(smoothed[trend_state])

        range_probs.append(range_prob)
        trend_probs.append(trend_prob)

        predictions.append({
            "symbol": symbol,
            "range_prob": range_prob,
            "trend_prob": trend_prob,
        })

    # Compute metrics
    if not range_probs:
        return SplitEvaluationResult(
            split_id=split.split_id,
            train_size=sum(len(df) for df in train_dfs),
            test_size=0,
            test_range_prob_mean=0.0,
            test_trend_prob_mean=0.0,
            test_range_prob_std=0.0,
            test_trend_prob_std=0.0,
            num_pass_gate=0,
            num_fail_gate=0,
            pass_rate=0.0,
        )

    # Get thresholds from config
    _cfg = get_config()
    range_prob_min = float(_cfg.hmm.range_prob_min)
    trend_prob_max = float(_cfg.hmm.trend_prob_max)

    # Count passes/fails
    num_pass = sum(
        1 for rp, tp in zip(range_probs, trend_probs)
        if rp >= range_prob_min and tp <= trend_prob_max
    )
    num_fail = len(range_probs) - num_pass

    return SplitEvaluationResult(
        split_id=split.split_id,
        train_size=sum(len(df) for df in train_dfs),
        test_size=len(predictions),
        test_range_prob_mean=float(np.mean(range_probs)),
        test_trend_prob_mean=float(np.mean(trend_probs)),
        test_range_prob_std=float(np.std(range_probs)),
        test_trend_prob_std=float(np.std(trend_probs)),
        num_pass_gate=num_pass,
        num_fail_gate=num_fail,
        pass_rate=float(num_pass / len(range_probs)) if range_probs else 0.0,
        predictions=predictions,
    )


def run_walk_forward_evaluation(
    datasets: Dict[str, pd.DataFrame],
    train_days: int = 90,
    test_days: int = 14,
    step_days: int = 14,
    n_components: int = 4,
    covariance_type: str = "diag",
    n_iter: int = 200,
    tol: float = 1e-3,
    random_state: int = 7,
) -> WalkForwardEvaluation:
    """
    Run complete walk-forward evaluation.

    Args:
        datasets: Dict mapping symbol to DataFrame with 15m data
        train_days: Training window size
        test_days: Test window size
        step_days: Step size between splits
        n_components: HMM components
        covariance_type: HMM covariance type
        n_iter: HMM iterations
        tol: HMM tolerance
        random_state: Random seed

    Returns:
        WalkForwardEvaluation with complete results

    Example:
        >>> datasets = load_training_dataset(Path("data/training_sets/2025-01-01"))
        >>> evaluation = run_walk_forward_evaluation(
        ...     datasets,
        ...     train_days=90,
        ...     test_days=14,
        ...     step_days=14,
        ... )
        >>> print(f"Mean pass rate: {evaluation.mean_pass_rate:.2%}")
    """
    # Create splits
    splits = create_walk_forward_splits(
        datasets=datasets,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )

    logger.info("Created %d walk-forward splits", len(splits))

    # Evaluate each split
    results = []
    for i, split in enumerate(splits):
        logger.info("Evaluating split %d/%d...", i + 1, len(splits))
        result = evaluate_split(
            split=split,
            datasets=datasets,
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
        )
        results.append(result)

    # Compute aggregated metrics
    pass_rates = [r.pass_rate for r in results if r.test_size > 0]
    range_probs = [r.test_range_prob_mean for r in results if r.test_size > 0]
    trend_probs = [r.test_trend_prob_mean for r in results if r.test_size > 0]

    return WalkForwardEvaluation(
        num_splits=len(splits),
        splits=splits,
        results=results,
        mean_pass_rate=float(np.mean(pass_rates)) if pass_rates else 0.0,
        std_pass_rate=float(np.std(pass_rates)) if pass_rates else 0.0,
        mean_range_prob=float(np.mean(range_probs)) if range_probs else 0.0,
        mean_trend_prob=float(np.mean(trend_probs)) if trend_probs else 0.0,
    )


def _ema_smooth_posteriors_full(posteriors: np.ndarray, k: int) -> np.ndarray:
    """Apply causal EMA smoothing to every row of a posterior matrix.

    Unlike ``_ema_smooth_posteriors`` (which returns only the *last* row),
    this returns the full smoothed matrix — one row per bar.  This is
    needed for per-bar CPCV evaluation.

    Note: Walk-forward evaluation (``evaluate_split``) uses the last-bar
    smoothing variant (``_ema_smooth_posteriors``) for backward
    compatibility.  CPCV per-bar evaluation uses this full-matrix
    variant as the primary method.

    Args:
        posteriors: (n_samples, n_states) raw posteriors from ``predict_proba``.
        k: EMA window size (1 = no smoothing).

    Returns:
        (n_samples, n_states) smoothed posteriors (each row sums to ~1).
    """
    if k <= 1 or posteriors.shape[0] <= 1:
        return posteriors.copy()

    # Allow explicit alpha override from config
    _cfg = get_config()
    alpha_override = _cfg.hmm.smooth_alpha
    if alpha_override is not None:
        alpha = float(alpha_override)
    else:
        alpha = 2.0 / (k + 1)

    smoothed = np.empty_like(posteriors)
    smoothed[0] = posteriors[0]
    for i in range(1, posteriors.shape[0]):
        smoothed[i] = alpha * posteriors[i] + (1.0 - alpha) * smoothed[i - 1]
        row_sum = smoothed[i].sum()
        if row_sum > 0:
            smoothed[i] /= row_sum
    return smoothed


def _build_cpcv_feature_matrix(
    datasets: Dict[str, pd.DataFrame],
) -> tuple:
    """Concatenate per-symbol HMM features into a single matrix.

    Returns:
        (X_all, cpcv_df, close_prices, symbol_ids)
        - X_all: (N, n_features)
        - cpcv_df: DataFrame with ``start_time_utc`` for CPCV splitting
        - close_prices: (N,) float array of close prices (aligned with X_all)
        - symbol_ids: (N,) int array identifying which symbol each row belongs to
        Returns (None, None, None, None) if no usable data.
    """
    all_features = []
    all_open_times = []
    all_close_prices = []
    all_symbol_ids = []

    for sym_idx, (_symbol, df) in enumerate(datasets.items()):
        X, valid = compute_hmm_features(df)
        if len(X) == 0:
            continue
        X_valid = X[valid]
        if X_valid.shape[0] < 60:
            continue
        if "open_time" in df.columns:
            ot = np.asarray(pd.to_datetime(df["open_time"]))
            ot_valid = ot[valid]
        else:
            ot_valid = pd.date_range("2020-01-01", periods=X_valid.shape[0], freq="15min")

        close_valid = np.asarray(df["close"])[valid].astype(float)

        all_features.append(X_valid)
        all_open_times.extend(ot_valid)
        all_close_prices.extend(close_valid.tolist())
        all_symbol_ids.extend([sym_idx] * X_valid.shape[0])

    if not all_features:
        return None, None, None, None

    X_all = np.vstack(all_features)
    cpcv_df = pd.DataFrame({"start_time_utc": all_open_times})
    cpcv_df["start_time_utc"] = pd.to_datetime(cpcv_df["start_time_utc"], utc=True)
    cpcv_df["t1"] = cpcv_df["start_time_utc"] + timedelta(hours=24)
    close_arr = np.array(all_close_prices)
    sym_ids_arr = np.array(all_symbol_ids)
    return X_all, cpcv_df, close_arr, sym_ids_arr


def run_cpcv_hmm_evaluation(
    datasets: Dict[str, pd.DataFrame],
    n_components: int = 4,
    covariance_type: str = "diag",
    n_iter: int = 200,
    tol: float = 1e-3,
    random_state: int = 7,
) -> Dict[str, Any]:
    """
    CPCV-based evaluation for HMM regime detection with purge/embargo.

    **Per-bar evaluation**: instead of testing only the smoothed *last bar*
    per path, this evaluates every test bar, yielding thousands of data
    points for threshold calibration.

    Args:
        datasets: Dict mapping symbol to DataFrame with 15m data
        n_components: HMM components
        covariance_type: HMM covariance type
        n_iter: HMM iterations
        tol: HMM tolerance
        random_state: Random seed

    Returns:
        Dictionary with CPCV evaluation results including per-bar
        distribution_stats and per-path summary statistics.
    """
    from neutralgrid.backtest.cpcv import (
        CPCV,
        CPCVConfig,
        DeflatedSharpeCalculator,
    )

    # Read CPCV config
    _cfg = get_config()
    n_groups = int(_cfg.cpcv.n_groups)
    n_test_groups = int(_cfg.cpcv.n_test_groups)
    purge_pct = float(_cfg.cpcv.purge_pct)
    embargo_pct = float(_cfg.cpcv.embargo_pct)
    smooth_k = int(_cfg.hmm.smooth_k)
    range_prob_min = float(_cfg.hmm.range_prob_min)
    trend_prob_max = float(_cfg.hmm.trend_prob_max)
    dominance_min = float(_cfg.hmm.range_dominance_min)
    dominance_eps = float(_cfg.hmm.dominance_eps)

    X_all, cpcv_df, close_prices, symbol_ids = _build_cpcv_feature_matrix(datasets)
    if X_all is None:
        return {"num_paths": 0, "mean_pass_rate": 0.0, "paths": []}

    # AUC forward-return horizon (bars) for ground-truth label construction
    auc_fwd_horizon = 24  # 6h on 15m data (6 * 4 bars)

    # AFML Fix #2: Wire holdout (Station 4 compliance)
    holdout_pct = float(_cfg.cpcv.holdout_pct)
    purge_hours = float(_cfg.cpcv.purge_hours)
    embargo_hours = float(_cfg.cpcv.embargo_hours)
    cpcv_config = CPCVConfig(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge_pct=purge_pct,
        embargo_pct=embargo_pct,
        purge_hours=purge_hours,
        embargo_hours=embargo_hours,
        holdout_pct=holdout_pct,
    )
    cpcv = CPCV(cpcv_config)

    # Split with holdout: CV on older data, holdout is most recent 20%
    cv_df, holdout_df, cv_splits = cpcv.split_with_holdout(
        cpcv_df, timestamp_col="start_time_utc", t1_col="t1"
    )
    holdout_metrics = cpcv.get_holdout_metrics(cv_df, holdout_df, "start_time_utc")

    # AFML Station 4: Log holdout creation via HoldoutValidator
    if not holdout_df.empty:
        try:
            from neutralgrid.training.holdout_validator import HoldoutValidator
            hv = HoldoutValidator()
            cv_ts = pd.to_datetime(cv_df["start_time_utc"])
            ho_ts = pd.to_datetime(holdout_df["start_time_utc"])
            hv.mark_holdout_created(
                cv_samples=len(cv_df),
                holdout_samples=len(holdout_df),
                cv_end_date=str(cv_ts.max().date()) if not cv_ts.empty else "",
                holdout_start_date=str(ho_ts.min().date()) if not ho_ts.empty else "",
                notes="Created by run_cpcv_hmm_evaluation",
            )
        except Exception as e:
            warnings.warn(f"HoldoutValidator logging failed (non-fatal): {e}")

    # Build indices mapping for holdout-aware splitting
    cv_indices = cv_df.index.to_numpy()
    _X_cv = X_all[cv_indices] if len(cv_indices) > 0 else X_all

    path_results = []
    # Collect all per-bar values across paths for distribution stats
    all_bar_range = []
    all_bar_trend = []
    all_bar_dominance = []
    # AFML Fix #3: Collect OOS returns per path for PBO computation
    path_oos_returns: Dict[int, np.ndarray] = {}
    # AUC per path
    path_aucs: List[float] = []

    for path_idx, (train_idx, test_idx) in enumerate(cv_splits):
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]

        if X_train.shape[0] < 100 or X_test.shape[0] < 10:
            continue

        try:
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import RobustScaler
            from neutralgrid.models.hmm.schema import infer_state_mapping, compute_state_groups

            scaler = RobustScaler(quantile_range=(25, 75))
            X_train_scaled = scaler.fit_transform(X_train)

            model = GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol,
                random_state=random_state,
            )
            model.fit(X_train_scaled)

            state_means_unscaled = scaler.inverse_transform(model.means_)
            _trend_state, _range_state = infer_state_mapping(
                state_means_unscaled, transmat=model.transmat_
            )
            groups = compute_state_groups(state_means_unscaled)

            X_test_scaled = scaler.transform(X_test)
            posteriors = model.predict_proba(X_test_scaled)

            # Per-bar smoothed posteriors
            smoothed_all = _ema_smooth_posteriors_full(posteriors, smooth_k)

            # Aggregated range/trend probs across state groups
            bar_range = smoothed_all[:, groups["range_states"]].sum(axis=1).astype(float)
            bar_trend = smoothed_all[:, groups["trend_states"]].sum(axis=1).astype(float)
            bar_dominance = bar_range / (bar_range + bar_trend + dominance_eps)

            # Per-bar pass rates
            hard_pass_arr = (bar_range >= range_prob_min) & (bar_trend <= trend_prob_max)
            dom_pass_arr = bar_dominance >= dominance_min

            all_bar_range.extend(bar_range.tolist())
            all_bar_trend.extend(bar_trend.tolist())
            all_bar_dominance.extend(bar_dominance.tolist())

            # AFML Fix #1: Compute per-bar grid-survival payoff for PBO
            # Uses the same forward-return pattern as AUC computation
            cap = 0.05
            payoffs = []
            for local_i, global_i in enumerate(test_idx):
                fwd_i = global_i + auc_fwd_horizon
                if (fwd_i < len(close_prices)
                        and symbol_ids[fwd_i] == symbol_ids[global_i]
                        and close_prices[global_i] > 0):
                    fwd_logret = abs(float(np.log(
                        close_prices[fwd_i] / close_prices[global_i]
                    )))
                    payoffs.append(max(0.0, cap - fwd_logret))
            if payoffs:
                payoff_arr = np.array(payoffs)
                # Center by subtracting mean so Sharpe can go negative
                centered = payoff_arr - payoff_arr.mean()
                path_oos_returns[path_idx] = centered
            else:
                path_oos_returns[path_idx] = np.array([0.0])

            # AUC: use forward |log-return| < cap as ground-truth range label
            path_auc = None
            try:
                from sklearn.metrics import roc_auc_score as _roc_auc
                fwd_labels = []
                dominance_scores = []
                cap = 0.05  # same cap as utility sweep
                for local_i, global_i in enumerate(test_idx):
                    fwd_i = global_i + auc_fwd_horizon
                    if (fwd_i < len(close_prices)
                            and symbol_ids[fwd_i] == symbol_ids[global_i]
                            and close_prices[global_i] > 0):
                        fwd_ret = abs(float(np.log(
                            close_prices[fwd_i] / close_prices[global_i]
                        )))
                        fwd_labels.append(1 if fwd_ret < cap else 0)
                        dominance_scores.append(float(bar_dominance[local_i]))
                if len(fwd_labels) >= 20 and len(set(fwd_labels)) == 2:
                    path_auc = float(_roc_auc(fwd_labels, dominance_scores))
                    path_aucs.append(path_auc)
            except Exception as e:
                logger.debug("AUC computation failed for path %d: %s", path_idx, e)

            path_result = {
                "path_idx": path_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "n_test_bars": len(bar_range),
                "range_prob_agg_mean": float(np.mean(bar_range)),
                "range_prob_agg_std": float(np.std(bar_range)),
                "trend_prob_agg_mean": float(np.mean(bar_trend)),
                "trend_prob_agg_std": float(np.std(bar_trend)),
                "range_dominance_mean": float(np.mean(bar_dominance)),
                "range_dominance_std": float(np.std(bar_dominance)),
                "pass_rate_hard": float(np.mean(hard_pass_arr)),
                "pass_rate_dominance": float(np.mean(dom_pass_arr)),
                "auc": path_auc,
            }

            # Phase 2: Temperature scaling diagnostics
            if TEMP_SCALING_AVAILABLE and len(posteriors) > 10:
                try:
                    _ts = TemperatureScaler()
                    _ts_labels = np.argmax(posteriors, axis=1)
                    _ts_result = _ts.fit(posteriors, _ts_labels)
                    path_result["temp_scale_T"] = _ts_result.temperature
                    path_result["temp_scale_nll_before"] = _ts_result.nll_before
                    path_result["temp_scale_nll_after"] = _ts_result.nll_after
                except Exception:
                    pass

            path_results.append(path_result)
        except Exception as e:
            warnings.warn(f"CPCV path {path_idx} failed: {e}")
            continue

    if not path_results:
        return {"num_paths": 0, "mean_pass_rate": 0.0, "paths": []}

    # Distribution stats across all per-bar values
    percentiles = [5, 10, 15, 20, 25, 50, 75, 80, 85, 90, 95]
    bar_range_arr = np.array(all_bar_range)
    bar_trend_arr = np.array(all_bar_trend)
    bar_dom_arr = np.array(all_bar_dominance)

    distribution_stats = {
        "n_total_bars": len(all_bar_range),
        "range_prob": {
            "mean": float(np.mean(bar_range_arr)),
            "std": float(np.std(bar_range_arr)),
            **{f"p{p}": float(np.percentile(bar_range_arr, p)) for p in percentiles},
        },
        "trend_prob": {
            "mean": float(np.mean(bar_trend_arr)),
            "std": float(np.std(bar_trend_arr)),
            **{f"p{p}": float(np.percentile(bar_trend_arr, p)) for p in percentiles},
        },
        "range_dominance": {
            "mean": float(np.mean(bar_dom_arr)),
            "std": float(np.std(bar_dom_arr)),
            **{f"p{p}": float(np.percentile(bar_dom_arr, p)) for p in percentiles},
        },
    }

    mean_pass_hard = float(np.mean([p["pass_rate_hard"] for p in path_results]))
    mean_pass_dom = float(np.mean([p["pass_rate_dominance"] for p in path_results]))

    # AFML Fix #3: Compute PBO (Probability of Backtest Overfitting)
    from neutralgrid.backtest.cpcv import compute_pbo

    pbo_result = compute_pbo(path_oos_returns, len(path_results))
    pbo_summary = {
        "pbo": pbo_result.pbo,
        "n_paths": pbo_result.n_paths,
        "n_overfit": pbo_result.n_overfit,
    }

    # AFML Fix #2: Compute Deflated Sharpe from aggregated centered payoffs
    # (not pseudo-Sharpe from pass_rates — that was structurally broken)
    all_payoffs = np.concatenate([
        path_oos_returns[k] for k in sorted(path_oos_returns.keys())
    ]) if path_oos_returns else np.array([])

    if len(all_payoffs) > 1 and np.std(all_payoffs) > 0:
        raw_sharpe = float(np.mean(all_payoffs) / np.std(all_payoffs))
    else:
        raw_sharpe = 0.0

    # AFML Fix #12: Use TrialTracker for correct n_trials (independent strategy
    # trials), falling back to a conservative default (not 1) since CPCV paths are
    # folds, not independent strategy trials.
    _default_trials = get_config().cpcv.default_trial_count
    try:
        from neutralgrid.training.trial_tracker import get_global_tracker
        _tracker = get_global_tracker()
        # Count HMM trials including threshold sweeps (AFML: all trials count)
        _n_hmm = _tracker.get_trial_count(model_type="hmm")
        _n_sweep = _tracker.get_trial_count(model_type="hmm_threshold_sweep")
        _n_meta = _tracker.get_trial_count(model_type="meta_labeler")
        _n_trials = _n_hmm + _n_sweep + _n_meta
        if _n_trials == 0:
            logger.warning(
                "Trial tracker has 0 logged trials — using default_trial_count=%d. "
                "Deflated Sharpe may be overstated. Consider running with --trial-log.",
                _default_trials,
            )
            _n_trials = _default_trials
    except Exception:
        logger.warning(
            "Trial tracker unavailable — using default_trial_count=%d. "
            "Deflated Sharpe may be overstated. Consider running with --trial-log.",
            _default_trials,
        )
        _n_trials = _default_trials
    _n_trials = max(1, _n_trials)

    # AFML Fix #3: Compute var_sharpe from per-path Sharpe of centered payoff arrays
    path_sharpes = []
    for pidx in sorted(path_oos_returns.keys()):
        parr = path_oos_returns[pidx]
        if len(parr) > 1 and np.std(parr) > 0:
            path_sharpes.append(float(np.mean(parr) / np.std(parr)))
        else:
            path_sharpes.append(0.0)
    var_sharpe = float(np.var(path_sharpes)) if len(path_sharpes) > 1 else 1.0
    var_sharpe = max(var_sharpe, 1e-6)  # avoid zero variance

    dsr_calc = DeflatedSharpeCalculator(n_trials=_n_trials)
    dsr_result = dsr_calc.deflate(
        raw_sharpe=raw_sharpe,
        n_observations=len(all_payoffs),
        var_sharpe=var_sharpe,
    )
    dsr_summary = {
        "raw_sharpe": dsr_result.raw_sharpe,
        "deflated_sharpe": dsr_result.deflated_sharpe,
        "haircut_pct": dsr_result.haircut_pct,
        "prob_false_discovery": dsr_result.prob_false_discovery,
        "n_trials": _n_trials,
        "var_sharpe": round(var_sharpe, 6),
        "metric_type": "proxy_payoff",
    }

    # Mean AUC across paths
    mean_auc = float(np.mean(path_aucs)) if path_aucs else None

    # Phase 2: Aggregate temperature scaling diagnostics
    if TEMP_SCALING_AVAILABLE:
        ts_temps = [r.get("temp_scale_T") for r in path_results if r.get("temp_scale_T") is not None]
        if ts_temps:
            results_ts: Dict[str, Any] = {}
            results_ts["temp_scale_mean_T"] = float(np.mean(ts_temps))
            results_ts["temp_scale_std_T"] = float(np.std(ts_temps))
            ts_nll_deltas = [
                r["temp_scale_nll_before"] - r["temp_scale_nll_after"]
                for r in path_results
                if r.get("temp_scale_nll_before") is not None
            ]
            if ts_nll_deltas:
                results_ts["temp_scale_mean_nll_improvement"] = float(np.mean(ts_nll_deltas))
        else:
            results_ts = {}
    else:
        results_ts = {}

    # AFML Station 4: Evaluate holdout data (not just carve it out)
    if not holdout_df.empty and len(cv_df) >= 100:
        try:
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import RobustScaler
            from neutralgrid.models.hmm.schema import infer_state_mapping, compute_state_groups

            holdout_indices = holdout_df.index.to_numpy()
            X_holdout = X_all[holdout_indices]
            X_cv_full = X_all[cv_indices]

            if X_cv_full.shape[0] >= 100 and X_holdout.shape[0] >= 10:
                ho_scaler = RobustScaler(quantile_range=(25, 75))
                X_cv_scaled = ho_scaler.fit_transform(X_cv_full)

                ho_model = GaussianHMM(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    tol=tol,
                    random_state=random_state,
                )
                ho_model.fit(X_cv_scaled)

                ho_means_unscaled = ho_scaler.inverse_transform(ho_model.means_)
                _ho_trend_state, _ho_range_state = infer_state_mapping(
                    ho_means_unscaled, transmat=ho_model.transmat_
                )
                ho_groups = compute_state_groups(ho_means_unscaled)

                X_holdout_scaled = ho_scaler.transform(X_holdout)
                ho_posteriors = ho_model.predict_proba(X_holdout_scaled)

                ho_smoothed = _ema_smooth_posteriors_full(ho_posteriors, smooth_k)
                ho_bar_range = ho_smoothed[:, ho_groups["range_states"]].sum(axis=1).astype(float)
                ho_bar_trend = ho_smoothed[:, ho_groups["trend_states"]].sum(axis=1).astype(float)
                ho_bar_dom = ho_bar_range / (ho_bar_range + ho_bar_trend + dominance_eps)

                ho_hard_pass = (ho_bar_range >= range_prob_min) & (ho_bar_trend <= trend_prob_max)
                ho_dom_pass = ho_bar_dom >= dominance_min

                holdout_metrics["holdout_evaluated"] = True
                holdout_metrics["holdout_pass_rate_hard"] = float(np.mean(ho_hard_pass))
                holdout_metrics["holdout_pass_rate_dominance"] = float(np.mean(ho_dom_pass))
                holdout_metrics["holdout_range_prob_mean"] = float(np.mean(ho_bar_range))
                holdout_metrics["holdout_trend_prob_mean"] = float(np.mean(ho_bar_trend))
                holdout_metrics["holdout_dominance_mean"] = float(np.mean(ho_bar_dom))

                # Log holdout access via HoldoutValidator
                try:
                    from neutralgrid.training.holdout_validator import HoldoutValidator
                    hv = HoldoutValidator()
                    hv.mark_holdout_accessed(
                        purpose="final_evaluation",
                        caller="run_cpcv_hmm_evaluation",
                        notes=f"pass_rate={float(np.mean(ho_hard_pass)):.4f}",
                    )
                except ImportError:
                    logger.debug("HoldoutValidator not available, skipping holdout logging")
                except Exception as e:
                    logger.debug("Holdout access logging failed: %s", e)

        except Exception as ho_err:
            warnings.warn(f"Holdout evaluation failed (non-fatal): {ho_err}")
            holdout_metrics["holdout_evaluated"] = False
            holdout_metrics["holdout_error"] = str(ho_err)

    return {
        "num_paths": len(path_results),
        "mean_pass_rate": mean_pass_hard,
        "mean_pass_rate_dominance": mean_pass_dom,
        "mean_auc": mean_auc,
        "mean_range_prob_agg": float(np.mean([p["range_prob_agg_mean"] for p in path_results])),
        "mean_trend_prob_agg": float(np.mean([p["trend_prob_agg_mean"] for p in path_results])),
        "distribution_stats": distribution_stats,
        "holdout_metrics": holdout_metrics,
        "pbo": pbo_summary,
        "deflated_sharpe": dsr_summary,
        "cpcv_config": {
            "n_groups": n_groups,
            "n_test_groups": n_test_groups,
            "purge_pct": purge_pct,
            "embargo_pct": embargo_pct,
            "holdout_pct": holdout_pct,
        },
        "n_components": n_components,
        "paths": path_results,
        **results_ts,
    }


def run_cpcv_threshold_sweep(
    datasets: Dict[str, pd.DataFrame],
    n_components: int = 4,
    covariance_type: str = "diag",
    n_iter: int = 200,
    tol: float = 1e-3,
    random_state: int = 7,
    target_pass_rate_low: float = 0.15,
    target_pass_rate_high: float = 0.25,
) -> Dict[str, Any]:
    """
    Sweep dominance and hard thresholds on CPCV per-bar posteriors.

    Collects per-bar test posteriors from all CPCV paths, then evaluates
    a grid of candidate thresholds to find the ones that produce a pass
    rate inside the target band [target_pass_rate_low, target_pass_rate_high].

    Args:
        datasets: Dict mapping symbol to 15m DataFrame.
        n_components: HMM states.
        covariance_type: HMM covariance type.
        n_iter: HMM EM iterations.
        tol: HMM convergence tolerance.
        random_state: Random seed.
        target_pass_rate_low: Lower bound of target pass-rate band.
        target_pass_rate_high: Upper bound of target pass-rate band.

    Returns:
        Dictionary with distribution_stats, sweep_results, and recommended
        thresholds for dominance, hard, and percentile-anchored modes.
    """
    from neutralgrid.backtest.cpcv import CPCV, CPCVConfig

    # Read CPCV config
    _cfg = get_config()
    n_groups = int(_cfg.cpcv.n_groups)
    n_test_groups = int(_cfg.cpcv.n_test_groups)
    purge_pct = float(_cfg.cpcv.purge_pct)
    embargo_pct = float(_cfg.cpcv.embargo_pct)
    smooth_k = int(_cfg.hmm.smooth_k)
    dominance_eps = float(_cfg.hmm.dominance_eps)

    X_all, cpcv_df, _, _ = _build_cpcv_feature_matrix(datasets)
    if X_all is None:
        return {"error": "no usable data", "sweep_results": {}}

    # AFML Fix #2: Wire holdout into threshold sweep
    holdout_pct = float(_cfg.cpcv.holdout_pct)
    purge_hours = float(_cfg.cpcv.purge_hours)
    embargo_hours = float(_cfg.cpcv.embargo_hours)
    cpcv_config = CPCVConfig(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge_pct=purge_pct,
        embargo_pct=embargo_pct,
        purge_hours=purge_hours,
        embargo_hours=embargo_hours,
        holdout_pct=holdout_pct,
    )
    cpcv = CPCV(cpcv_config)

    _cv_df, _holdout_df, cv_splits = cpcv.split_with_holdout(
        cpcv_df, timestamp_col="start_time_utc", t1_col="t1"
    )

    # Collect all per-bar posteriors
    all_bar_range: list[float] = []
    all_bar_trend: list[float] = []
    all_bar_dominance: list[float] = []

    for path_idx, (train_idx, test_idx) in enumerate(cv_splits):
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]

        if X_train.shape[0] < 100 or X_test.shape[0] < 10:
            continue

        try:
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import RobustScaler
            from neutralgrid.models.hmm.schema import infer_state_mapping, compute_state_groups

            scaler = RobustScaler(quantile_range=(25, 75))
            X_train_scaled = scaler.fit_transform(X_train)

            model = GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol,
                random_state=random_state,
            )
            model.fit(X_train_scaled)

            state_means_unscaled = scaler.inverse_transform(model.means_)
            _trend_state, _range_state = infer_state_mapping(
                state_means_unscaled, transmat=model.transmat_
            )
            groups = compute_state_groups(state_means_unscaled)

            X_test_scaled = scaler.transform(X_test)
            posteriors = model.predict_proba(X_test_scaled)
            smoothed_all = _ema_smooth_posteriors_full(posteriors, smooth_k)

            # Aggregated range/trend probs across state groups
            bar_range = smoothed_all[:, groups["range_states"]].sum(axis=1).astype(float)
            bar_trend = smoothed_all[:, groups["trend_states"]].sum(axis=1).astype(float)
            bar_dom = bar_range / (bar_range + bar_trend + dominance_eps)

            all_bar_range.extend(bar_range.tolist())
            all_bar_trend.extend(bar_trend.tolist())
            all_bar_dominance.extend(bar_dom.tolist())
        except Exception as e:
            warnings.warn(f"Sweep CPCV path {path_idx} failed: {e}")
            continue

    if not all_bar_range:
        return {"error": "no bar data collected", "sweep_results": {}}

    bar_range_arr = np.array(all_bar_range)
    bar_trend_arr = np.array(all_bar_trend)
    bar_dom_arr = np.array(all_bar_dominance)
    n_bars = len(bar_range_arr)

    # Distribution stats
    percentiles = [5, 10, 15, 20, 25, 50, 75, 80, 85, 90, 95]
    distribution_stats = {
        "n_total_bars": n_bars,
        "range_prob": {
            "mean": float(np.mean(bar_range_arr)),
            "std": float(np.std(bar_range_arr)),
            **{f"p{p}": float(np.percentile(bar_range_arr, p)) for p in percentiles},
        },
        "trend_prob": {
            "mean": float(np.mean(bar_trend_arr)),
            "std": float(np.std(bar_trend_arr)),
            **{f"p{p}": float(np.percentile(bar_trend_arr, p)) for p in percentiles},
        },
        "range_dominance": {
            "mean": float(np.mean(bar_dom_arr)),
            "std": float(np.std(bar_dom_arr)),
            **{f"p{p}": float(np.percentile(bar_dom_arr, p)) for p in percentiles},
        },
    }

    target_mid = (target_pass_rate_low + target_pass_rate_high) / 2.0

    # ── Dominance sweep ──────────────────────────────────────────────
    dominance_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    dominance_sweep: list[dict] = []
    best_dom_dist = float("inf")
    best_dom_entry: Optional[dict] = None

    for threshold in dominance_grid:
        pass_rate = float(np.mean(bar_dom_arr >= threshold))
        entry = {"range_dominance_min": threshold, "pass_rate": round(pass_rate, 4)}
        dominance_sweep.append(entry)
        dist = abs(pass_rate - target_mid)
        if dist < best_dom_dist:
            best_dom_dist = dist
            best_dom_entry = entry

    # ── Hard threshold sweep ─────────────────────────────────────────
    range_prob_grid = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    trend_prob_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    hard_sweep: list[dict] = []
    best_hard_dist = float("inf")
    best_hard_entry: Optional[dict] = None

    for rp_min in range_prob_grid:
        for tp_max in trend_prob_grid:
            pass_rate = float(np.mean(
                (bar_range_arr >= rp_min) & (bar_trend_arr <= tp_max)
            ))
            entry = {
                "range_prob_min": rp_min,
                "trend_prob_max": tp_max,
                "pass_rate": round(pass_rate, 4),
            }
            hard_sweep.append(entry)
            dist = abs(pass_rate - target_mid)
            if dist < best_hard_dist:
                best_hard_dist = dist
                best_hard_entry = entry

    # ── Percentile-anchored thresholds ───────────────────────────────
    pct_anchored = {}
    for label, pct in [("p80", 80), ("p85", 85), ("p90", 90)]:
        threshold = float(np.percentile(bar_dom_arr, pct))
        pass_rate = float(np.mean(bar_dom_arr >= threshold))
        pct_anchored[label] = {
            "range_dominance_min": round(threshold, 4),
            "pass_rate": round(pass_rate, 4),
            "description": f"top-{100 - pct}% pass",
        }

    recommended = {
        "dominance": best_dom_entry,
        "hard": best_hard_entry,
        "percentile_anchored": pct_anchored,
        "target_band": {
            "low": target_pass_rate_low,
            "high": target_pass_rate_high,
            "midpoint": target_mid,
        },
    }

    # AFML Fix #4: Log sweep trials via TrialTracker for deflated Sharpe
    try:
        from neutralgrid.training.trial_tracker import TrialTracker, TrialRecord
        tracker = TrialTracker()

        # Log dominance sweep trials
        for entry in dominance_sweep:
            trial = TrialRecord(
                trial_id=f"hmm_dom_sweep_{entry['range_dominance_min']:.2f}",
                timestamp=datetime.now(timezone.utc),
                model_type="hmm_threshold_sweep",
                hyperparameters={"range_dominance_min": entry["range_dominance_min"]},
                cv_score=entry["pass_rate"],
                notes=f"Dominance sweep: threshold={entry['range_dominance_min']:.2f}",
            )
            tracker.log_trial(trial)

        # Log best hard threshold trial
        if best_hard_entry:
            trial = TrialRecord(
                trial_id=f"hmm_hard_sweep_{best_hard_entry['range_prob_min']:.2f}_{best_hard_entry['trend_prob_max']:.2f}",
                timestamp=datetime.now(timezone.utc),
                model_type="hmm_threshold_sweep",
                hyperparameters={
                    "range_prob_min": best_hard_entry["range_prob_min"],
                    "trend_prob_max": best_hard_entry["trend_prob_max"],
                },
                cv_score=best_hard_entry["pass_rate"],
                notes="Best hard threshold from sweep",
            )
            tracker.log_trial(trial)
    except Exception as e:
        warnings.warn(f"Trial logging failed (non-fatal): {e}")

    return {
        "distribution_stats": distribution_stats,
        "sweep_results": {
            "dominance": dominance_sweep,
            "hard": hard_sweep,
        },
        "recommended": recommended,
        "n_components": n_components,
        "cpcv_config": {
            "n_groups": n_groups,
            "n_test_groups": n_test_groups,
            "purge_pct": purge_pct,
            "embargo_pct": embargo_pct,
            "holdout_pct": holdout_pct,
        },
    }


def run_cpcv_utility_sweep(
    datasets: Dict[str, pd.DataFrame],
    n_components: int = 4,
    covariance_type: str = "diag",
    n_iter: int = 200,
    tol: float = 1e-3,
    random_state: int = 7,
    dominance_grid: Optional[List[float]] = None,
    cap: float = 0.05,
    lambda_risk: float = 0.5,
    fwd_horizon: int = 24,
) -> Dict[str, Any]:
    """Select dominance threshold by maximizing expected grid-survival utility OOS.

    Instead of targeting an arbitrary pass-rate band, this sweep maximizes a
    utility function that rewards bars where forward price movement is small
    (grid survives) and penalizes variance (AFML-aligned).
    fwd_horizon default is 24 bars = 6h on 15m data.

    For each CPCV path and candidate threshold *t*:
        pass_mask = (dominance >= t)
        payoff(bar) = max(0, cap - |fwd_log_return_at_fwd_horizon|)
        U(t, path) = mean(payoff[pass]) * pass_rate - lambda * std(payoff[pass])
        If pass_rate == 0 → U = -inf

    The threshold with the highest mean U across all paths is selected.

    Also computes a PBO-style rank-persistence check: do winning thresholds
    persist across different OOS folds?

    Args:
        datasets: Symbol → 15m DataFrame mapping.
        n_components: HMM states.
        dominance_grid: Candidate thresholds to evaluate.
        cap: Max payoff cap (grid survives if fwd return < cap).
        lambda_risk: Risk-aversion coefficient for std penalty.
        fwd_horizon: Forward bars to measure return (default 24 = 6h on 15m).

    Returns:
        Dictionary with selected_threshold, per-threshold stats, per-path
        rankings, and artifact-ready output.
    """
    from neutralgrid.backtest.cpcv import CPCV, CPCVConfig

    if dominance_grid is None:
        dominance_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    _cfg = get_config()
    n_groups = int(_cfg.cpcv.n_groups)
    n_test_groups = int(_cfg.cpcv.n_test_groups)
    purge_pct = float(_cfg.cpcv.purge_pct)
    embargo_pct = float(_cfg.cpcv.embargo_pct)
    holdout_pct = float(_cfg.cpcv.holdout_pct)
    smooth_k = int(_cfg.hmm.smooth_k)
    dominance_eps = float(_cfg.hmm.dominance_eps)

    X_all, cpcv_df, close_prices, symbol_ids = _build_cpcv_feature_matrix(datasets)
    if X_all is None:
        return {"error": "no usable data"}

    purge_hours = float(_cfg.cpcv.purge_hours)
    embargo_hours = float(_cfg.cpcv.embargo_hours)
    cpcv_config = CPCVConfig(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge_pct=purge_pct,
        embargo_pct=embargo_pct,
        purge_hours=purge_hours,
        embargo_hours=embargo_hours,
        holdout_pct=holdout_pct,
    )
    cpcv = CPCV(cpcv_config)

    _cv_df, _holdout_df, cv_splits = cpcv.split_with_holdout(
        cpcv_df, timestamp_col="start_time_utc", t1_col="t1"
    )

    _n_thresholds = len(dominance_grid)
    # U(t, path) matrix: rows=paths, cols=thresholds
    path_utilities: List[List[float]] = []
    path_pass_rates: List[List[float]] = []
    path_rankings: List[List[int]] = []  # per-path ranking of thresholds

    for path_idx, (train_idx, test_idx) in enumerate(cv_splits):
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]

        if X_train.shape[0] < 100 or X_test.shape[0] < 10:
            continue

        try:
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import RobustScaler
            from neutralgrid.models.hmm.schema import compute_state_groups

            scaler = RobustScaler(quantile_range=(25, 75))
            X_train_scaled = scaler.fit_transform(X_train)

            model = GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol,
                random_state=random_state,
            )
            model.fit(X_train_scaled)

            state_means_unscaled = scaler.inverse_transform(model.means_)
            groups = compute_state_groups(state_means_unscaled)

            X_test_scaled = scaler.transform(X_test)
            posteriors = model.predict_proba(X_test_scaled)
            smoothed_all = _ema_smooth_posteriors_full(posteriors, smooth_k)

            bar_range = smoothed_all[:, groups["range_states"]].sum(axis=1).astype(float)
            bar_trend = smoothed_all[:, groups["trend_states"]].sum(axis=1).astype(float)
            bar_dominance = bar_range / (bar_range + bar_trend + dominance_eps)

            # Compute forward |log-return| for each test bar
            n_test = len(test_idx)
            fwd_abs_logret = np.full(n_test, np.nan)
            for local_i, global_i in enumerate(test_idx):
                fwd_i = global_i + fwd_horizon
                if (fwd_i < len(close_prices)
                        and symbol_ids[fwd_i] == symbol_ids[global_i]
                        and close_prices[global_i] > 0):
                    fwd_abs_logret[local_i] = abs(float(np.log(
                        close_prices[fwd_i] / close_prices[global_i]
                    )))

            # Evaluate utility for each candidate threshold
            path_u = []
            path_pr = []
            for t in dominance_grid:
                pass_mask = (bar_dominance >= t) & np.isfinite(fwd_abs_logret)
                n_pass = int(pass_mask.sum())
                n_total = int(np.isfinite(fwd_abs_logret).sum())

                if n_pass == 0 or n_total == 0:
                    path_u.append(float("-inf"))
                    path_pr.append(0.0)
                    continue

                payoff = np.maximum(0.0, cap - fwd_abs_logret[pass_mask])
                pass_rate = n_pass / n_total
                mean_payoff = float(np.mean(payoff))
                std_payoff = float(np.std(payoff))
                u = mean_payoff * pass_rate - lambda_risk * std_payoff
                path_u.append(u)
                path_pr.append(pass_rate)

            path_utilities.append(path_u)
            path_pass_rates.append(path_pr)

            # Rank thresholds for this path (higher utility = better rank)
            u_arr = np.array(path_u)
            ranking = list(np.argsort(-u_arr))  # descending
            path_rankings.append(ranking)

        except Exception as e:
            warnings.warn(f"Utility sweep CPCV path {path_idx} failed: {e}")
            continue

    if not path_utilities:
        return {"error": "no paths completed"}

    # Average utility across paths for each threshold
    u_matrix = np.array(path_utilities)  # (n_paths, n_thresholds)
    # Replace -inf with NaN for mean computation, then use nanmean
    u_matrix_clean = np.where(np.isinf(u_matrix), np.nan, u_matrix)
    mean_u = np.nanmean(u_matrix_clean, axis=0)

    # Select best threshold
    best_idx = int(np.nanargmax(mean_u))
    selected_threshold = dominance_grid[best_idx]
    selected_utility = float(mean_u[best_idx])

    # Per-threshold stats
    pr_matrix = np.array(path_pass_rates)
    per_threshold = []
    for i, t in enumerate(dominance_grid):
        per_threshold.append({
            "threshold": t,
            "mean_utility": float(mean_u[i]) if np.isfinite(mean_u[i]) else None,
            "mean_pass_rate": float(np.mean(pr_matrix[:, i])),
            "std_utility": float(np.nanstd(u_matrix_clean[:, i])),
        })

    # PBO-style rank correlation: do winning thresholds persist OOS?
    # Compute Spearman rank correlation between consecutive path rankings
    rank_correlations = []
    if len(path_rankings) >= 2:
        from scipy.stats import spearmanr
        for i in range(len(path_rankings) - 1):
            sr_result = spearmanr(path_rankings[i], path_rankings[i + 1])
            corr = float(cast(float, sr_result[0]))
            if np.isfinite(corr):
                rank_correlations.append(corr)

    mean_rank_corr = float(np.mean(rank_correlations)) if rank_correlations else None

    # Compute mean pass rate at selected threshold
    selected_pass_rate = float(np.mean(pr_matrix[:, best_idx]))

    # Build artifact-ready output
    artifact_output = {
        "dominance_threshold": selected_threshold,
        "utility": round(selected_utility, 6),
        "pass_rate": round(selected_pass_rate, 4),
        "sweep_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "schema_version": str(_cfg.hmm.feature_schema),
    }

    return {
        "selected_threshold": selected_threshold,
        "selected_utility": selected_utility,
        "selected_pass_rate": selected_pass_rate,
        "per_threshold": per_threshold,
        "n_paths": len(path_utilities),
        "rank_persistence": {
            "mean_spearman_corr": mean_rank_corr,
            "n_comparisons": len(rank_correlations),
        },
        "config": {
            "dominance_grid": dominance_grid,
            "cap": cap,
            "lambda_risk": lambda_risk,
            "fwd_horizon": fwd_horizon,
        },
        "artifact": artifact_output,
    }


def run_cpcv_proxy_outcome_report(
    datasets: Dict[str, pd.DataFrame],
    n_components: int = 4,
    covariance_type: str = "diag",
    n_iter: int = 200,
    tol: float = 1e-3,
    random_state: int = 7,
    fwd_horizons: tuple = (24, 48),
) -> Dict[str, Any]:
    """Conditional proxy-outcome report: forward returns for pass vs fail bars.

    For each CPCV test bar that passes/fails the dominance gate (with
    trend_prob veto), compute forward absolute log-return over *fwd_horizons*
    bars (within the same symbol to avoid cross-symbol contamination).

    A well-calibrated gate should show:
    - **Pass bars**: lower forward |log-return| (price stays in range)
    - **Fail bars**: higher forward |log-return| (trending / breakout)
    - ``separation_ratio > 1.0`` (fail_mean / pass_mean)

    Args:
        datasets: Symbol -> kline DataFrame mapping.
        n_components: HMM states.
        fwd_horizons: Tuple of forward bar counts (default 6h, 12h on 15m = 24, 48 bars).

    Returns:
        Dictionary with gate_config, counts, forward_returns per horizon
        (pass vs fail statistics + separation metrics), and dominance_stats.
    """
    from neutralgrid.backtest.cpcv import CPCV, CPCVConfig

    # Read config
    _cfg = get_config()
    n_groups = int(_cfg.cpcv.n_groups)
    n_test_groups = int(_cfg.cpcv.n_test_groups)
    purge_pct = float(_cfg.cpcv.purge_pct)
    embargo_pct = float(_cfg.cpcv.embargo_pct)
    smooth_k = int(_cfg.hmm.smooth_k)
    dominance_min = float(_cfg.hmm.range_dominance_min)
    dominance_eps = float(_cfg.hmm.dominance_eps)
    trend_prob_max = float(_cfg.hmm.trend_prob_max)

    X_all, cpcv_df, close_prices, symbol_ids = _build_cpcv_feature_matrix(datasets)
    if X_all is None:
        return {"error": "no usable data"}

    purge_hours = float(_cfg.cpcv.purge_hours)
    embargo_hours = float(_cfg.cpcv.embargo_hours)
    holdout_pct = float(_cfg.cpcv.holdout_pct)
    cpcv = CPCV(CPCVConfig(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge_pct=purge_pct,
        embargo_pct=embargo_pct,
        purge_hours=purge_hours,
        embargo_hours=embargo_hours,
        holdout_pct=holdout_pct,
    ))

    # Use split_with_holdout for AFML Station 4 compliance
    _cv_df, _holdout_df, cv_splits = cpcv.split_with_holdout(
        cpcv_df, timestamp_col="start_time_utc", t1_col="t1"
    )

    # Accumulators keyed by horizon
    pass_fwd: Dict[int, list] = {h: [] for h in fwd_horizons}
    fail_fwd: Dict[int, list] = {h: [] for h in fwd_horizons}
    pass_dominance: list = []
    fail_dominance: list = []
    n_pass = 0
    n_fail = 0

    for path_idx, (train_idx, test_idx) in enumerate(cv_splits):
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]
        if X_train.shape[0] < 100 or X_test.shape[0] < 10:
            continue

        try:
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import RobustScaler
            from neutralgrid.models.hmm.schema import infer_state_mapping, compute_state_groups

            scaler = RobustScaler(quantile_range=(25, 75))
            X_train_scaled = scaler.fit_transform(X_train)

            model = GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol,
                random_state=random_state,
            )
            model.fit(X_train_scaled)

            state_means_unscaled = scaler.inverse_transform(model.means_)
            _trend_state, _range_state = infer_state_mapping(
                state_means_unscaled, transmat=model.transmat_
            )
            groups = compute_state_groups(state_means_unscaled)

            X_test_scaled = scaler.transform(X_test)
            posteriors = model.predict_proba(X_test_scaled)
            smoothed_all = _ema_smooth_posteriors_full(posteriors, smooth_k)

            for local_i, global_i in enumerate(test_idx):
                # Aggregated range/trend probs across state groups
                rp = float(smoothed_all[local_i, groups["range_states"]].sum())
                tp = float(smoothed_all[local_i, groups["trend_states"]].sum())
                dom = rp / (rp + tp + dominance_eps)

                passed = (dom >= dominance_min) and (tp <= trend_prob_max)

                if passed:
                    n_pass += 1
                    pass_dominance.append(dom)
                else:
                    n_fail += 1
                    fail_dominance.append(dom)

                # Forward absolute log-returns (within same symbol only)
                for h in fwd_horizons:
                    fwd_i = global_i + h
                    if (fwd_i < len(close_prices)
                            and symbol_ids[fwd_i] == symbol_ids[global_i]
                            and close_prices[global_i] > 0):
                        fwd_ret = abs(float(np.log(
                            close_prices[fwd_i] / close_prices[global_i]
                        )))
                        if passed:
                            pass_fwd[h].append(fwd_ret)
                        else:
                            fail_fwd[h].append(fwd_ret)

        except Exception as e:
            warnings.warn(f"Proxy-outcome path {path_idx} failed: {e}")
            continue

    total = n_pass + n_fail
    if total == 0:
        return {"error": "no test bars evaluated"}

    # Build report
    report: Dict[str, Any] = {
        "gate_config": {
            "mode": "dominance",
            "range_dominance_min": dominance_min,
            "trend_prob_max_veto": trend_prob_max,
        },
        "counts": {
            "pass": n_pass,
            "fail": n_fail,
            "total": total,
            "pass_rate": round(n_pass / total, 4),
        },
        "forward_returns": {},
    }

    for h in fwd_horizons:
        p_arr = np.array(pass_fwd[h]) if pass_fwd[h] else np.array([])
        f_arr = np.array(fail_fwd[h]) if fail_fwd[h] else np.array([])

        def _stats(arr: np.ndarray) -> dict[str, Any]:
            if len(arr) == 0:
                return {"count": 0}
            return {
                "count": len(arr),
                "mean": round(float(np.mean(arr)), 6),
                "median": round(float(np.median(arr)), 6),
                "std": round(float(np.std(arr)), 6),
                "p25": round(float(np.percentile(arr, 25)), 6),
                "p75": round(float(np.percentile(arr, 75)), 6),
                "p95": round(float(np.percentile(arr, 95)), 6),
            }

        h_report: Dict[str, Any] = {
            "horizon_bars": h,
            "pass": _stats(p_arr),
            "fail": _stats(f_arr),
        }

        # Separation metrics
        if len(p_arr) > 1 and len(f_arr) > 1:
            p_mean = float(np.mean(p_arr))
            f_mean = float(np.mean(f_arr))
            if p_mean > 0:
                h_report["separation_ratio"] = round(f_mean / p_mean, 4)

            # Welch's t-test
            try:
                from scipy.stats import ttest_ind
                tt_result = ttest_ind(f_arr, p_arr, equal_var=False)
                h_report["welch_t_stat"] = round(float(cast(float, tt_result[0])), 4)
                h_report["welch_p_value"] = float(cast(float, tt_result[1]))
            except ImportError:
                # Compute t-stat manually (skip p-value without scipy)
                na, nb = len(p_arr), len(f_arr)
                va = float(np.var(p_arr, ddof=1))
                vb = float(np.var(f_arr, ddof=1))
                se = np.sqrt(va / na + vb / nb)
                if se > 0:
                    h_report["welch_t_stat"] = round((f_mean - p_mean) / se, 4)

        report["forward_returns"][f"{h}bar"] = h_report

    # Dominance distribution for pass vs fail
    report["dominance_stats"] = {
        "pass_mean": round(float(np.mean(pass_dominance)), 6) if pass_dominance else None,
        "fail_mean": round(float(np.mean(fail_dominance)), 6) if fail_dominance else None,
    }

    return report


def save_evaluation_results(
    evaluation: WalkForwardEvaluation,
    output_path: Path,
) -> None:
    """
    Save evaluation results to JSON.

    Args:
        evaluation: WalkForwardEvaluation results
        output_path: Path to save JSON file
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation.to_dict(), f, indent=2)

    logger.info("Evaluation results saved to: %s", output_path)


def compute_wf_stability_report(
    datasets: Dict[str, pd.DataFrame],
    n_components: int = 4,
    covariance_type: str = "diag",
    n_iter: int = 200,
    tol: float = 1e-3,
    random_state: int = 7,
    n_folds: int = 3,
) -> Dict[str, Any]:
    """Walk-forward stability report: check regime model consistency across retrains.

    Trains the HMM on expanding windows (50/50, 66/33, 75/25) and compares:
    - Mean regime duration per state
    - Transition matrix (element-wise delta)
    - Dominance distribution (mean, std, p20, p80)

    Stability flags:
    - STABLE: transition_max_delta < 0.15 AND dominance_drift < 0.10
    - CAUTION: either exceeds soft limit
    - UNSTABLE: transition_max_delta > 0.25 OR dominance_drift > 0.20

    Args:
        datasets: Symbol → 15m DataFrame mapping.
        n_components: HMM states.
        n_folds: Number of expanding folds (default 3).

    Returns:
        Dictionary with per-fold metrics, cross-fold deltas, and stability flag.
    """
    from neutralgrid.models.hmm.schema import compute_state_groups

    _cfg = get_config()
    smooth_k = int(_cfg.hmm.smooth_k)
    dominance_eps = float(_cfg.hmm.dominance_eps)

    X_all, _cpcv_df, _, _ = _build_cpcv_feature_matrix(datasets)
    if X_all is None:
        return {"error": "no usable data", "stability": "UNKNOWN"}

    n_total = X_all.shape[0]

    # Define expanding window folds: train on first fraction, test on remainder
    fold_fractions = []
    for i in range(1, n_folds + 1):
        frac = (i) / (n_folds + 1)
        fold_fractions.append(frac)
    # e.g. for n_folds=3: [0.25, 0.50, 0.75] — but plan says 50/50, 66/33, 75/25
    fold_fractions = [0.50, 0.667, 0.75][:n_folds]

    fold_results = []

    for fold_idx, train_frac in enumerate(fold_fractions):
        train_end = int(n_total * train_frac)
        X_train = X_all[:train_end]
        X_test = X_all[train_end:]

        if X_train.shape[0] < 100 or X_test.shape[0] < 10:
            fold_results.append({"fold": fold_idx, "error": "insufficient data"})
            continue

        try:
            from hmmlearn.hmm import GaussianHMM
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler(quantile_range=(25, 75))
            X_train_scaled = scaler.fit_transform(X_train)

            model = GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol,
                random_state=random_state,
            )
            model.fit(X_train_scaled)

            state_means_unscaled = scaler.inverse_transform(model.means_)
            groups = compute_state_groups(state_means_unscaled)
            transmat = model.transmat_

            # Mean regime duration = 1 / (1 - self-transition probability)
            durations = []
            for s in range(n_components):
                self_p = float(transmat[s, s])
                dur = 1.0 / (1.0 - self_p) if self_p < 1.0 else float("inf")
                durations.append(dur)

            # Test-set dominance distribution
            X_test_scaled = scaler.transform(X_test)
            posteriors = model.predict_proba(X_test_scaled)
            smoothed = _ema_smooth_posteriors_full(posteriors, smooth_k)

            bar_range = smoothed[:, groups["range_states"]].sum(axis=1).astype(float)
            bar_trend = smoothed[:, groups["trend_states"]].sum(axis=1).astype(float)
            bar_dom = bar_range / (bar_range + bar_trend + dominance_eps)

            # Range state duration
            range_durations = [durations[s] for s in groups["range_states"]]
            mean_range_duration = float(np.mean(range_durations)) if range_durations else 0.0

            fold_results.append({
                "fold": fold_idx,
                "train_frac": round(train_frac, 3),
                "train_size": X_train.shape[0],
                "test_size": X_test.shape[0],
                "transmat": transmat.tolist(),
                "durations": [round(d, 2) for d in durations],
                "mean_range_duration": round(mean_range_duration, 2),
                "dominance": {
                    "mean": round(float(np.mean(bar_dom)), 4),
                    "std": round(float(np.std(bar_dom)), 4),
                    "p20": round(float(np.percentile(bar_dom, 20)), 4),
                    "p80": round(float(np.percentile(bar_dom, 80)), 4),
                },
            })

        except Exception as e:
            fold_results.append({"fold": fold_idx, "error": str(e)})
            continue

    # Compute cross-fold stability metrics
    valid_folds = [f for f in fold_results if "error" not in f]

    if len(valid_folds) < 2:
        return {
            "folds": fold_results,
            "stability": "UNKNOWN",
            "reason": "insufficient valid folds",
        }

    # Transition matrix max delta between consecutive folds
    max_transmat_delta = 0.0
    for i in range(1, len(valid_folds)):
        t_prev = np.array(valid_folds[i - 1]["transmat"])
        t_curr = np.array(valid_folds[i]["transmat"])
        delta = float(np.max(np.abs(t_curr - t_prev)))
        max_transmat_delta = max(max_transmat_delta, delta)

    # Dominance mean drift: first fold vs last fold
    first_dom_mean = valid_folds[0]["dominance"]["mean"]
    last_dom_mean = valid_folds[-1]["dominance"]["mean"]
    dominance_drift = abs(last_dom_mean - first_dom_mean)

    # Duration CV: coefficient of variation of mean range-state duration
    range_durations_all = [f["mean_range_duration"] for f in valid_folds
                           if np.isfinite(f["mean_range_duration"])]
    if range_durations_all and np.mean(range_durations_all) > 0:
        duration_cv = float(np.std(range_durations_all) / np.mean(range_durations_all))
    else:
        duration_cv = float("inf")

    # Determine stability flag
    if max_transmat_delta > 0.25 or dominance_drift > 0.20:
        stability = "UNSTABLE"
    elif max_transmat_delta > 0.15 or dominance_drift > 0.10:
        stability = "CAUTION"
    else:
        stability = "STABLE"

    return {
        "folds": fold_results,
        "cross_fold_metrics": {
            "transition_matrix_max_delta": round(max_transmat_delta, 4),
            "dominance_mean_drift": round(dominance_drift, 4),
            "duration_cv": round(duration_cv, 4),
        },
        "stability": stability,
        "n_valid_folds": len(valid_folds),
    }
