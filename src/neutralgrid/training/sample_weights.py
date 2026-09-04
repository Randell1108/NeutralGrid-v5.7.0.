"""
AFML-Compliant Sample Weighting for Meta-Labeler Training.

Implements sample weighting strategies from AFML Chapter 4:
1. Uniqueness weighting based on label overlap
2. Concurrency-based sample weights
3. Time-decay weighting for recency bias

These weights help correct for:
- Over-counting clustered regimes (many overlapping events)
- Non-stationarity (older examples less relevant)
- Information leakage from concurrent labels

Reference: AFML Chapter 4 (Sample Weights)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, cast
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _to_datetime_utc_mixed(values: Any) -> pd.Series:
    """Parse timestamps with mixed-format tolerance and UTC coercion."""
    index = values.index if isinstance(values, pd.Series) else None
    try:
        parsed = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if isinstance(parsed, pd.Series):
        return parsed
    return pd.Series(parsed, index=index)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class SampleWeightConfig:
    """Configuration for sample weighting strategies."""

    # Uniqueness weighting (AFML core)
    use_uniqueness: bool = True

    # Time-decay weighting (for regime drift)
    use_time_decay: bool = True
    time_decay_halflife_days: float = 30.0

    # Weight bounds
    min_weight: float = 0.1  # Floor to prevent zero weights
    max_weight: float = 10.0  # Cap to prevent outlier dominance

    # Normalization
    normalize: bool = True  # Normalize weights to sum to N


# =============================================================================
# CONCURRENCY COMPUTATION
# =============================================================================

def compute_concurrency_matrix(
    t0: pd.Series,
    t1: pd.Series,
) -> np.ndarray:
    """
    Compute the concurrency matrix for events.

    Concurrency[i, j] = 1 if event i and event j overlap in time.
    Two events overlap if: t0[i] <= t1[j] AND t0[j] <= t1[i]

    Args:
        t0: Series of event start times (datetime)
        t1: Series of event end times (datetime)

    Returns:
        N x N boolean matrix where True indicates overlap
    """
    _n = len(t0)
    t0_arr = np.asarray(_to_datetime_utc_mixed(t0))
    t1_arr = np.asarray(_to_datetime_utc_mixed(t1))

    # Vectorized overlap check
    # Event i overlaps with event j if t0[i] <= t1[j] and t0[j] <= t1[i]
    t0_matrix = t0_arr.reshape(-1, 1)  # N x 1
    t1_matrix = t1_arr.reshape(1, -1)  # 1 x N

    # Condition 1: t0[i] <= t1[j]
    cond1 = t0_matrix <= t1_matrix

    # Condition 2: t0[j] <= t1[i]
    cond2 = t0_arr.reshape(1, -1) <= t1_arr.reshape(-1, 1)

    overlap_matrix = cond1 & cond2

    return overlap_matrix


def compute_concurrent_labels(
    t0: pd.Series,
    t1: pd.Series,
) -> np.ndarray:
    """
    Compute number of concurrent labels for each event.

    An event i is concurrent with event j if their [t0, t1] intervals overlap.
    This counts self-overlap, so minimum concurrency is 1.

    Args:
        t0: Series of event start times (datetime)
        t1: Series of event end times (datetime)

    Returns:
        Array of concurrency counts per event
    """
    overlap_matrix = compute_concurrency_matrix(t0, t1)
    concurrency = overlap_matrix.sum(axis=1)

    return concurrency


def compute_avg_uniqueness(
    t0: pd.Series,
    t1: pd.Series,
) -> np.ndarray:
    """
    Compute average uniqueness for each event (AFML Section 4.3).

    For each event i with span [t0_i, t1_i], compute c_t (number of
    concurrent events active at time t) for every bar t within the span,
    then average 1/c_t over all bars in the span.

    Args:
        t0: Series of event start times
        t1: Series of event end times

    Returns:
        Array of average uniqueness values (0 to 1)
    """
    n = len(t0)
    t0_arr = np.asarray(_to_datetime_utc_mixed(t0))
    t1_arr = np.asarray(_to_datetime_utc_mixed(t1))

    # Build a sorted array of all unique time points (bar boundaries)
    all_times = np.unique(np.concatenate([t0_arr, t1_arr]))
    all_times.sort()

    # For each time point, compute c_t: number of events active at that time
    # Event i is active at time t if t0_arr[i] <= t <= t1_arr[i]
    n_times = len(all_times)
    c_t = np.zeros(n_times, dtype=np.float64)
    for i in range(n):
        mask = (all_times >= t0_arr[i]) & (all_times <= t1_arr[i])
        c_t[mask] += 1.0

    # For each event i, average 1/c_t over bars in [t0_i, t1_i]
    uniqueness = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mask = (all_times >= t0_arr[i]) & (all_times <= t1_arr[i])
        bars_ct = c_t[mask]
        if len(bars_ct) > 0:
            bars_ct = np.maximum(bars_ct, 1.0)
            uniqueness[i] = np.mean(1.0 / bars_ct)
        else:
            uniqueness[i] = 1.0

    return uniqueness


# =============================================================================
# WEIGHT COMPUTATION
# =============================================================================

def compute_uniqueness_weights(
    t0: pd.Series,
    t1: pd.Series,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute uniqueness-based sample weights.

    Weight[i] = uniqueness[i] = mean(1/c_t) for t in [t0_i, t1_i]

    This ensures overlapping events don't dominate training.
    Events that are more "unique" (less overlap) get higher weight.

    Args:
        t0: Series of event start times
        t1: Series of event end times
        normalize: If True, normalize weights to sum to N

    Returns:
        Array of sample weights
    """
    uniqueness = compute_avg_uniqueness(t0, t1)

    if normalize:
        n = len(uniqueness)
        total = uniqueness.sum()
        weights = uniqueness / total * n if total > 0 else uniqueness
    else:
        weights = uniqueness

    return weights


def compute_time_decay_weights(
    timestamps: pd.Series,
    halflife_days: float = 30.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute time-decay weights (more recent = higher weight).

    Uses exponential decay: weight = exp(-ln(2) * age / halflife)

    This addresses regime drift by giving more importance to recent events.

    Args:
        timestamps: Series of event timestamps
        halflife_days: Half-life in days (after this many days, weight is halved)
        normalize: If True, normalize weights to sum to N

    Returns:
        Array of time-decay weights
    """
    timestamps = _to_datetime_utc_mixed(timestamps)
    valid = timestamps.notna()
    if not valid.any():
        return np.ones(len(timestamps), dtype=float)
    if not valid.all():
        logger.warning(
            "Time decay weights: %d rows have invalid timestamps; using oldest fallback",
            int((~valid).sum()),
        )
        oldest = timestamps[valid].min()
        timestamps = timestamps.fillna(oldest)
    latest = timestamps.max()

    # Compute age in days
    age_seconds = (latest - timestamps).dt.total_seconds()
    age_days = age_seconds / 86400.0

    # Exponential decay
    decay_rate = np.log(2) / halflife_days
    weights = np.exp(-decay_rate * age_days)

    if normalize:
        n = len(weights)
        _wsum = weights.sum()
        weights = weights / _wsum * n if _wsum > 1e-12 else np.ones(n)

    return weights


def compute_sample_weights(
    df: pd.DataFrame,
    config: Optional[SampleWeightConfig] = None,
    t0_col: str = "start_time_utc",
    t1_col: str = "t1",
) -> np.ndarray:
    """
    Compute combined sample weights for training.

    Combines multiple weighting strategies:
    1. Uniqueness weights (if use_uniqueness=True and t1 available)
    2. Time-decay weights (if use_time_decay=True)

    Weights are multiplied together, then clipped and normalized.

    Args:
        df: Training DataFrame with timestamp columns
        config: SampleWeightConfig (uses defaults if None)
        t0_col: Column name for event start time
        t1_col: Column name for event end time (vertical barrier)

    Returns:
        Array of sample weights, one per row in df
    """
    config = config or SampleWeightConfig()
    n = len(df)

    # Start with uniform weights
    weights = np.ones(n)

    # Apply uniqueness weighting
    if config.use_uniqueness and t1_col in df.columns and t0_col in df.columns:
        t0 = _to_datetime_utc_mixed(df[t0_col])
        t1 = _to_datetime_utc_mixed(df[t1_col])

        # Check for valid windows (both t0 and t1)
        valid_window = t0.notna() & t1.notna()
        if valid_window.any():
            if valid_window.all():
                uniqueness_weights = compute_uniqueness_weights(t0, t1, normalize=False)
            else:
                # Partial t1: compute uniqueness for valid rows, default 1.0 for missing
                logger.info(
                    f"timestamp columns have {(~valid_window).sum()} invalid rows; "
                    "computing uniqueness for valid subset, using 1.0 for missing"
                )
                uniqueness_weights = np.ones(n)
                valid_mask = np.asarray(valid_window)
                if valid_mask.sum() > 1:
                    t0_valid = cast(pd.Series, t0[valid_mask])
                    t1_valid = cast(pd.Series, t1[valid_mask])
                    valid_uniqueness = compute_uniqueness_weights(
                        t0_valid, t1_valid, normalize=False
                    )
                    uniqueness_weights[valid_mask] = valid_uniqueness
            weights *= uniqueness_weights
            logger.debug(
                f"Applied uniqueness weights: min={uniqueness_weights.min():.3f}, "
                f"max={uniqueness_weights.max():.3f}, mean={uniqueness_weights.mean():.3f}"
            )
        else:
            logger.warning("No valid t0/t1 windows; skipping uniqueness weighting")

    # Apply time-decay weighting with N_eff guard
    if config.use_time_decay:
        timestamps = _to_datetime_utc_mixed(df[t0_col])
        time_weights = compute_time_decay_weights(
            timestamps,
            halflife_days=config.time_decay_halflife_days,
            normalize=False,
        )

        # N_eff guard: disable time-decay if effective sample size too low
        try:
            from neutralgrid.calibration.effective_sample_size_v20260311 import (
                should_disable_time_decay as _should_disable,
            )
            _disable, _n_eff = _should_disable(time_weights, min_effective_n=30)
            if _disable:
                logger.warning(
                    "Time-decay disabled by N_eff guard (N_eff=%.1f < 30); "
                    "using uniform weights instead",
                    _n_eff,
                )
                time_weights = np.ones(n)
        except ImportError:
            pass  # Guard not available — proceed without check

        weights *= time_weights
        logger.debug(
            f"Applied time-decay weights: min={time_weights.min():.3f}, "
            f"max={time_weights.max():.3f}, mean={time_weights.mean():.3f}"
        )

    # Normalize to sum to N first, then clip (AFML: normalize before clipping
    # to avoid distorting the distribution)
    if config.normalize:
        total = weights.sum()
        if total > 0:
            weights = weights / total * n

    # Apply weight bounds after normalization
    weights = np.clip(weights, config.min_weight, config.max_weight)

    # Re-normalize after clipping to maintain sum = N
    if config.normalize:
        total = weights.sum()
        if total > 0:
            weights = weights / total * n

    logger.info(
        f"Final sample weights: min={weights.min():.3f}, max={weights.max():.3f}, "
        f"mean={weights.mean():.3f}, std={weights.std():.3f}"
    )

    return weights


# =============================================================================
# SEQUENTIAL BOOTSTRAPPING (AFML Ch 4, Section 4.5)
# =============================================================================

def sequential_bootstrap_indices(
    t0: pd.Series,
    t1: pd.Series,
    n_samples: Optional[int] = None,
    random_state: int = 42,
) -> np.ndarray:
    """
    Generate a bootstrapped sample respecting event concurrency.

    Standard bootstrapping draws with replacement, ignoring temporal overlap.
    Sequential bootstrapping draws events with probability proportional to
    their average uniqueness at the time of drawing, reducing the chance
    of selecting highly concurrent (redundant) events.

    Algorithm (AFML 4.5.2):
    1. Start with empty sample set phi.
    2. For each draw:
       a. Compute average uniqueness of each event given phi.
       b. Draw one event with probability proportional to uniqueness.
       c. Add drawn event to phi.

    Args:
        t0: Series of event start times
        t1: Series of event end times
        n_samples: Number of samples to draw (default: len(t0))
        random_state: Random seed

    Returns:
        Array of selected indices (may contain repeats)
    """
    rng = np.random.RandomState(random_state)
    n = len(t0)
    if n_samples is None:
        n_samples = n

    t0_arr = np.asarray(_to_datetime_utc_mixed(t0))
    t1_arr = np.asarray(_to_datetime_utc_mixed(t1))

    # Build indicator matrix (events x time_points) -- O(N*T) precomputation
    all_times = np.unique(np.concatenate([t0_arr, t1_arr]))
    T = len(all_times)

    # indicator[i, t] = 1 if event i spans time point t
    indicator = np.zeros((n, T), dtype=np.float64)
    for i in range(n):
        mask = (all_times >= t0_arr[i]) & (all_times <= t1_arr[i])
        indicator[i, mask] = 1.0

    # Running concurrency vector: c_t[t] = number of selected events spanning time t
    c_t = np.zeros(T, dtype=np.float64)

    # Precompute per-event bar counts for averaging
    bar_counts = indicator.sum(axis=1)  # shape (n,)
    no_bars = bar_counts == 0

    selected = []
    for _ in range(n_samples):
        # Vectorized uniqueness: indicator dot (1/(c_t+1)), divided by bar count
        inv_ct = 1.0 / (c_t + 1.0)  # shape (T,)
        uniqueness = indicator.dot(inv_ct)  # shape (n,) = sum of 1/(c_t+1) over active bars
        np.divide(uniqueness, bar_counts, out=uniqueness, where=~no_bars)
        uniqueness[no_bars] = 1.0

        # Probability proportional to uniqueness
        total = uniqueness.sum()
        probs = uniqueness / total if total > 0 else np.ones(n) / n

        # Draw one event
        idx = rng.choice(n, p=probs)
        selected.append(idx)

        # Incrementally update c_t
        c_t += indicator[idx]

    return np.array(selected)


def compute_return_attribution(
    t0: pd.Series,
    t1: pd.Series,
    returns: np.ndarray,
) -> np.ndarray:
    """
    Compute return attribution weights (AFML Ch 4).

    Each event's weight is proportional to its absolute return,
    divided by the number of concurrent events at each point.
    This attributes returns to unique events rather than
    double-counting overlapping periods.

    Args:
        t0: Series of event start times
        t1: Series of event end times
        returns: Array of event returns (e.g., pnl_pct)

    Returns:
        Array of attribution weights (sums to total absolute return)
    """
    n = len(t0)
    abs_rets = np.abs(returns)

    # Compute uniqueness
    uniqueness = compute_avg_uniqueness(t0, t1)

    # Attribution = |return| * uniqueness
    attribution = abs_rets * uniqueness

    # Normalize to sum to N (consistent with other weight schemes)
    total = attribution.sum()
    if total > 0:
        attribution = attribution / total * n

    return attribution


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def analyze_concurrency(
    df: pd.DataFrame,
    t0_col: str = "start_time_utc",
    t1_col: str = "t1",
) -> dict[str, Any]:
    """
    Analyze concurrency patterns in the training data.

    Useful for understanding how much overlap exists and whether
    uniqueness weighting will have a significant effect.

    Args:
        df: Training DataFrame
        t0_col: Column name for event start time
        t1_col: Column name for event end time

    Returns:
        Dictionary with concurrency statistics
    """
    if t1_col not in df.columns:
        return {"error": f"Column '{t1_col}' not found in DataFrame"}

    t0 = _to_datetime_utc_mixed(df[t0_col])
    t1 = _to_datetime_utc_mixed(df[t1_col])

    concurrency = compute_concurrent_labels(t0, t1)
    uniqueness = compute_avg_uniqueness(t0, t1)

    return {
        "n_samples": len(df),
        "concurrency": {
            "min": int(concurrency.min()),
            "max": int(concurrency.max()),
            "mean": float(concurrency.mean()),
            "median": float(np.median(concurrency)),
            "std": float(concurrency.std()),
        },
        "uniqueness": {
            "min": float(uniqueness.min()),
            "max": float(uniqueness.max()),
            "mean": float(uniqueness.mean()),
            "median": float(np.median(uniqueness)),
        },
        "high_concurrency_pct": float((concurrency > 5).mean() * 100),
        "low_uniqueness_pct": float((uniqueness < 0.2).mean() * 100),
    }


def get_weight_diagnostics(
    weights: np.ndarray,
) -> dict[str, Any]:
    """
    Get diagnostic statistics for computed sample weights.

    Args:
        weights: Array of sample weights

    Returns:
        Dictionary with weight statistics
    """
    return {
        "n_samples": len(weights),
        "min": float(weights.min()),
        "max": float(weights.max()),
        "mean": float(weights.mean()),
        "median": float(np.median(weights)),
        "std": float(weights.std()),
        "sum": float(weights.sum()),
        "effective_n": float(weights.sum() ** 2 / (weights ** 2).sum()) if (weights ** 2).sum() > 0 else 0.0,
        "max_min_ratio": float(weights.max() / weights.min()) if weights.min() > 0 else float('inf'),
    }
