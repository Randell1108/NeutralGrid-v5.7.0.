"""
Combinatorial Purged Cross-Validation (CPCV) for AFML-Compliant Model Selection.

AFML explicitly warns that without reporting trials you cannot assess false
discovery probability. This module implements:

1. CPCV (Combinatorial Purged CV): Multiple paths through data to reduce
   selection bias and false discoveries.

2. Deflated Sharpe Ratio: Adjusts Sharpe for multiple testing to account
   for the probability of finding spurious patterns.

3. Trial Tracking: Records number of trials for proper deflation.

Design principles:
- All paths through the data are explored (combinatorial)
- Purging and embargo prevent look-ahead bias
- Deflated metrics account for multiple testing
- Explicit trial counting for reproducibility

Reference: AFML Chapter 11 (Cross-Validation), Chapter 14 (Backtest Statistics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Iterator, Dict, Any, cast
from itertools import combinations
import logging
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

from neutralgrid.core.config import CPCVConfig  # Single source of truth


def _to_datetime_utc_mixed(values: Any) -> pd.Series:
    """
    Parse datetimes robustly across mixed string formats.

    Some training tables combine values like:
    - 2026-02-17 22:49:16+00:00
    - 2026-02-17T22:49:16+00:00
    """
    index = values.index if isinstance(values, pd.Series) else None
    try:
        parsed = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
    except TypeError:
        # pandas versions without format="mixed"
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if isinstance(parsed, pd.Series):
        return parsed
    return pd.Series(parsed, index=index)


@dataclass
class CPCVSplitMetrics:
    """Metrics for a single CPCV split."""

    path_idx: int
    test_groups: Tuple[int, ...]
    train_size: int
    test_size: int
    purged_count: int
    embargo_count: int


@dataclass
class CPCVMetrics:
    """Aggregate metrics for CPCV."""

    n_paths: int  # Total number of paths
    n_groups: int
    n_test_groups: int
    total_samples: int
    n_yielded: int = 0  # Number of splits actually yielded (may be < n_paths)
    splits: List[CPCVSplitMetrics] = field(default_factory=list)

    @property
    def avg_train_size(self) -> float:
        return float(np.mean([s.train_size for s in self.splits])) if self.splits else 0

    @property
    def avg_test_size(self) -> float:
        return float(np.mean([s.test_size for s in self.splits])) if self.splits else 0


@dataclass
class DeflatedSharpeResult:
    """Result of deflated Sharpe ratio calculation."""

    raw_sharpe: float  # Unadjusted Sharpe ratio
    deflated_sharpe: float  # Adjusted for multiple testing
    haircut_pct: float  # Percentage reduction

    n_trials: int  # Number of trials/comparisons
    expected_max_sharpe: float  # Expected max Sharpe under null

    # Probability that observed Sharpe is due to chance
    prob_false_discovery: float

    # Confidence interval
    sharpe_ci_lower: float
    sharpe_ci_upper: float

    # Minimum Track Record Length (AFML Ch 14)
    min_track_record_length: Optional[int] = None


class CPCV:
    """
    Combinatorial Purged Cross-Validation.

    Explores all combinations of test groups to reduce selection bias.
    Each path uses different groups as test set, with purging and embargo.

    Usage:
        cpcv = CPCV(CPCVConfig(n_groups=6, n_test_groups=2))

        for train_idx, test_idx in cpcv.split(df, timestamp_col="time"):
            # Train on train_idx, evaluate on test_idx
            pass

        print(f"Total paths: {cpcv.metrics.n_paths}")
    """

    def __init__(self, config: Optional[CPCVConfig] = None):
        """
        Initialize CPCV.

        Args:
            config: CPCVConfig with n_groups, n_test_groups, purge/embargo
        """
        self.config = config or CPCVConfig()
        self._metrics: Optional[CPCVMetrics] = None

    @property
    def n_paths(self) -> int:
        """Number of combinatorial paths."""
        from math import comb
        return comb(self.config.n_groups, self.config.n_test_groups)

    @property
    def metrics(self) -> Optional[CPCVMetrics]:
        return self._metrics

    def split(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "start_time_utc",
        t1_col: Optional[str] = "t1",
        group_col: Optional[str] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for all combinatorial paths.

        AFML Compliance:
        - If t1_col is available and time-based purging is enabled,
          purges training events whose [t0, t1] overlaps the test period.
        - This prevents information leakage from events spanning train/test boundary.
        - Falls back to percent-based purging if t1 not available.
        - If group_col is provided (e.g. "symbol"), all rows sharing the same
          group value are assigned to the same temporal group.  This prevents
          leakage through persistent asset characteristics (AFML Ch 7 extension
          for multi-asset datasets).

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column (t0, event start)
            t1_col: Name of t1 column (event end, vertical barrier time)
            group_col: Optional column for group-blocked assignment (e.g. "symbol")

        Yields:
            Tuple of (train_indices, test_indices)
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"DataFrame must have '{timestamp_col}' column")

        # Parse and sort by timestamp using mixed-format tolerant parser.
        parsed_ts = _to_datetime_utc_mixed(df[timestamp_col])
        valid_ts = parsed_ts.notna()
        if not valid_ts.all():
            dropped = int((~valid_ts).sum())
            logger.warning(
                "CPCV dropped %d rows with invalid %s timestamps",
                dropped,
                timestamp_col,
            )
        if int(valid_ts.sum()) < self.config.n_groups + 1:
            raise ValueError(
                f"Not enough valid timestamp rows ({int(valid_ts.sum())}) for "
                f"{self.config.n_groups} groups"
            )

        df_sorted = df.loc[valid_ts].copy()
        df_sorted["_cpcv_ts"] = parsed_ts.loc[valid_ts].values
        df_sorted = df_sorted.sort_values("_cpcv_ts")
        sorted_indices = df_sorted.index.to_numpy()
        n = len(sorted_indices)

        if n < self.config.n_groups + 1:
            raise ValueError(f"Not enough samples ({n}) for {self.config.n_groups} groups")

        # Get timestamps for time-based purging
        timestamps = _to_datetime_utc_mixed(df_sorted["_cpcv_ts"])

        # Check if we can use time-based purging
        use_time_based = (
            self.config.use_time_based_purging and
            t1_col is not None and
            t1_col in df_sorted.columns
        )

        if use_time_based:
            t1_times = _to_datetime_utc_mixed(df_sorted[t1_col])
            # Fill missing t1 with t0 + horizon.
            # Use horizon_hours (actual training horizon) for t1 synthesis,
            # NOT purge_hours — purge_hours controls the purge window and
            # may not equal the event duration (e.g. 12h purge vs 6h horizon).
            _synth_hours = (
                self.config.horizon_hours
                if self.config.horizon_hours is not None
                else self.config.purge_hours
            )
            missing_t1 = t1_times.isna()
            if missing_t1.any():
                horizon_delta = pd.Timedelta(hours=_synth_hours)
                t1_times = t1_times.fillna(timestamps + horizon_delta)
                logger.info(
                    "CPCV t1 synthesis: filled %d missing t1 values using "
                    "horizon_hours=%.1f (purge_hours=%.1f)",
                    int(missing_t1.sum()), _synth_hours, self.config.purge_hours,
                )
        else:
            # Fallback: assume t1 = t0 + horizon_hours (or use index-based)
            if self.config.use_time_based_purging:
                _synth_hours_fb = (
                    self.config.horizon_hours
                    if self.config.horizon_hours is not None
                    else self.config.purge_hours
                )
                horizon_delta = pd.Timedelta(hours=_synth_hours_fb)
                t1_times = timestamps + horizon_delta
            else:
                t1_times = None

        # Split into groups
        used_group_col = False
        if group_col is not None and group_col in df_sorted.columns:
            # Symbol-blocked assignment: all rows with the same group value
            # go into the same temporal group based on median timestamp.
            used_group_col = True
            grp_vals = df_sorted[group_col].values
            unique_grps = []
            seen: set = set()
            for v in grp_vals:
                if v not in seen:
                    unique_grps.append(v)
                    seen.add(v)
            # unique_grps is in temporal order (df_sorted is time-sorted)
            block_size = max(1, len(unique_grps) // self.config.n_groups)
            groups: list[np.ndarray] = [np.array([], dtype=sorted_indices.dtype)] * self.config.n_groups
            for i in range(self.config.n_groups):
                start_b = i * block_size
                end_b = start_b + block_size if i < self.config.n_groups - 1 else len(unique_grps)
                block_grp_vals = set(unique_grps[start_b:end_b])
                mask = np.array([v in block_grp_vals for v in grp_vals])
                groups[i] = sorted_indices[mask]
            logger.info(
                "CPCV symbol-blocked groups: %d unique %s values across %d groups "
                "(sizes: %s)",
                len(unique_grps), group_col,
                self.config.n_groups,
                [len(g) for g in groups],
            )
        else:
            group_size = n // self.config.n_groups
            groups = []
            for i in range(self.config.n_groups):
                start = i * group_size
                end = start + group_size if i < self.config.n_groups - 1 else n
                groups.append(sorted_indices[start:end])

        # Initialize metrics
        self._metrics = CPCVMetrics(
            n_paths=self.n_paths,
            n_groups=self.config.n_groups,
            n_test_groups=self.config.n_test_groups,
            total_samples=n,
        )

        # Generate all combinations of test groups
        for path_idx, test_group_indices in enumerate(combinations(range(self.config.n_groups), self.config.n_test_groups)):
            # Test indices: combine selected test groups
            test_idx = np.concatenate([groups[i] for i in test_group_indices])

            # Train indices: all other groups
            train_groups = [i for i in range(self.config.n_groups) if i not in test_group_indices]
            train_idx_raw = np.concatenate([groups[i] for i in train_groups])

            if used_group_col:
                # GROUP-BLOCKED PURGING: When groups are defined by an identity
                # column (e.g. symbol), symbol blocking prevents cross-symbol
                # leakage (AFML Ch 7 extension).  However, within each temporal
                # slice there can still be temporal leakage if training events'
                # t1 overlaps the test group's time range.  Apply within-group
                # temporal purge to remove such overlapping training samples.
                if t1_times is not None and self.config.use_time_based_purging:
                    embargo_delta = pd.Timedelta(
                        hours=self.config.embargo_hours if self.config.embargo_hours is not None else 0.0
                    )
                    # Locate training sample positions in sorted_indices
                    train_mask = np.isin(sorted_indices, train_idx_raw)
                    train_t0_series = timestamps[train_mask]
                    train_t1_series = t1_times[train_mask]

                    exclude_mask = np.zeros(len(train_idx_raw), dtype=bool)
                    for grp_i in test_group_indices:
                        grp_idx = groups[grp_i]
                        grp_ts = timestamps.iloc[
                            np.where(np.isin(sorted_indices, grp_idx))[0]
                        ]
                        grp_start = grp_ts.min()
                        grp_end = grp_ts.max()
                        grp_embargo_end = grp_end + embargo_delta

                        # Exclude training events whose t1 overlaps this
                        # test group's time range (within-group temporal purge)
                        grp_exclude = (
                            np.asarray(train_t1_series >= grp_start) &
                            np.asarray(train_t0_series <= grp_embargo_end)
                        )
                        exclude_mask |= grp_exclude

                    keep_mask = ~exclude_mask
                    train_idx = train_idx_raw[keep_mask]
                    purged_count = len(train_idx_raw) - len(train_idx)
                    if purged_count > 0:
                        logger.debug(
                            "CPCV symbol-grouped temporal purge: removed %d "
                            "training samples with t1 overlap (path %d)",
                            purged_count, path_idx,
                        )
                else:
                    # No t1 available — symbol blocking alone is the guard
                    train_idx = train_idx_raw
                    purged_count = 0

            elif use_time_based or self.config.use_time_based_purging:
                # TIME-BASED PURGING (AFML-compliant)
                # Handle each test group independently to avoid over-purging
                # when test groups are non-contiguous (e.g., groups 0 and 4)
                embargo_delta = pd.Timedelta(hours=self.config.embargo_hours if self.config.embargo_hours is not None else 0.0)

                # Get train event times
                train_mask = np.isin(sorted_indices, train_idx_raw)
                train_t0_series = timestamps[train_mask]
                train_t1_series = t1_times[train_mask] if t1_times is not None else train_t0_series

                # Start with all training samples kept
                exclude_mask = np.zeros(len(train_idx_raw), dtype=bool)

                # Apply purge/embargo per test group independently
                for grp_i in test_group_indices:
                    grp_idx = groups[grp_i]
                    grp_timestamps = timestamps.iloc[np.where(np.isin(sorted_indices, grp_idx))[0]]
                    grp_start = grp_timestamps.min()
                    grp_end = grp_timestamps.max()
                    grp_embargo_end = grp_end + embargo_delta

                    # Exclude training events whose t1 overlaps this group's test period
                    # or whose t0 falls within the embargo window after this group
                    grp_exclude = (
                        np.asarray(train_t1_series >= grp_start) &
                        np.asarray(train_t0_series <= grp_embargo_end)
                    )
                    exclude_mask |= grp_exclude

                keep_mask = ~exclude_mask
                train_idx = train_idx_raw[keep_mask]
                purged_count = len(train_idx_raw) - len(train_idx)

            else:
                # PERCENT-BASED PURGING (legacy) — boundary-based
                purge_size = int(n * self.config.purge_pct)
                embargo_size = int(n * self.config.embargo_pct)

                # Map between positional indices and original DataFrame indices
                # so offset arithmetic works on contiguous positions
                idx_to_pos = {idx: pos for pos, idx in enumerate(sorted_indices)}

                # Boundary-based: for each test group, compute the
                # positional range [grp_start - purge_size, grp_end + embargo_size]
                # and remove any training sample whose position falls in it.
                train_positions = np.array([idx_to_pos[i] for i in train_idx_raw])
                exclude_mask = np.zeros(len(train_idx_raw), dtype=bool)

                for grp_i in test_group_indices:
                    grp_idx = groups[grp_i]
                    grp_positions = np.array([idx_to_pos[i] for i in grp_idx])
                    grp_start = int(grp_positions.min()) - purge_size
                    grp_end = int(grp_positions.max()) + embargo_size
                    exclude_mask |= (train_positions >= grp_start) & (train_positions <= grp_end)

                train_idx = train_idx_raw[~exclude_mask]
                purged_count = int(exclude_mask.sum())

            # Track metrics
            # Convert embargo to sample count based on data frequency
            if use_time_based:
                median_interval_obj = timestamps.diff().dropna().median()
                if isinstance(median_interval_obj, pd.Timedelta) and median_interval_obj > pd.Timedelta(0):
                    embargo_td = cast(
                        pd.Timedelta, pd.Timedelta(hours=self.config.embargo_hours or 0.0)
                    )
                    embargo_sample_count = int(embargo_td / median_interval_obj)
                else:
                    embargo_sample_count = 0
            else:
                embargo_sample_count = max(1, int(n * self.config.embargo_pct))

            split_metrics = CPCVSplitMetrics(
                path_idx=path_idx,
                test_groups=test_group_indices,
                train_size=len(train_idx),
                test_size=len(test_idx),
                purged_count=purged_count,
                embargo_count=embargo_sample_count,
            )
            self._metrics.splits.append(split_metrics)

            # Yield only if we have enough samples
            if len(train_idx) >= 50 and len(test_idx) >= 20:
                self._metrics.n_yielded += 1
                yield train_idx, test_idx

    def split_with_holdout(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "start_time_utc",
        t1_col: Optional[str] = "t1",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Iterator[Tuple[np.ndarray, np.ndarray]]]:
        """
        Split data into holdout and CV portions, then generate CV splits.

        AFML Compliance (Station 4):
        - True holdout period is reserved from the most recent data
        - This data is NEVER seen during CV - only for final out-of-sample evaluation
        - Addresses the "no true holdout" finding from Station 4

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            t1_col: Name of t1 column (event end time)

        Returns:
            Tuple of (cv_df, holdout_df, cv_splits_iterator)
            - cv_df: Data to use for cross-validation
            - holdout_df: True holdout data (most recent, never used in CV)
            - cv_splits_iterator: Iterator over (train_idx, test_idx) for CV
        """
        if not self.config.has_holdout:
            # No holdout configured - use all data for CV
            return df, pd.DataFrame(), self.split(df, timestamp_col, t1_col)

        parsed_ts = _to_datetime_utc_mixed(df[timestamp_col])
        valid_ts = parsed_ts.notna()
        if not valid_ts.all():
            dropped = int((~valid_ts).sum())
            logger.warning(
                "CPCV holdout split dropped %d rows with invalid %s timestamps",
                dropped,
                timestamp_col,
            )
        if int(valid_ts.sum()) <= 0:
            raise ValueError(f"No valid rows in timestamp column '{timestamp_col}'")

        df_sorted = df.loc[valid_ts].copy()
        df_sorted["_cpcv_ts"] = parsed_ts.loc[valid_ts].values
        df_sorted = df_sorted.sort_values("_cpcv_ts")
        n = len(df_sorted)

        # Calculate holdout size (most recent data)
        holdout_size = int(n * self.config.holdout_pct)
        cv_size = n - holdout_size

        if cv_size < self.config.n_groups + 10:
            raise ValueError(
                f"Not enough samples ({cv_size}) for CV after {self.config.holdout_pct:.0%} holdout. "
                f"Need at least {self.config.n_groups + 10}."
            )

        # Apply embargo gap between CV and holdout to prevent serial correlation leakage
        embargo_gap = 0
        if self.config.embargo_hours and self.config.embargo_hours > 0 and timestamp_col in df_sorted.columns:
            embargo_td = pd.Timedelta(hours=self.config.embargo_hours)
            cv_end_time = pd.to_datetime(df_sorted.iloc[cv_size - 1]["_cpcv_ts"], utc=True)
            holdout_mask = _to_datetime_utc_mixed(df_sorted["_cpcv_ts"]) > (cv_end_time + embargo_td)
            embargo_gap = cv_size - holdout_mask.sum() - (n - cv_size)  # approximate
            # CV stays the same; holdout starts after embargo
            holdout_df = df_sorted.loc[holdout_mask].copy()
            cv_df = df_sorted.iloc[:cv_size].copy()
            if embargo_gap > 0:
                logger.info(
                    f"Holdout embargo: dropped {n - cv_size - len(holdout_df)} samples "
                    f"in {self.config.embargo_hours}h gap between CV and holdout"
                )
        else:
            cv_df = df_sorted.iloc[:cv_size].copy()
            holdout_df = df_sorted.iloc[cv_size:].copy()

        # Log holdout time range to confirm chronological split (Gate 7 Fix 4).
        cv_ts = _to_datetime_utc_mixed(cv_df[timestamp_col])
        holdout_ts = (
            _to_datetime_utc_mixed(holdout_df[timestamp_col])
            if not holdout_df.empty else pd.Series(dtype="datetime64[ns, UTC]")
        )
        logger.info(
            "CPCV holdout split — CV: %s to %s (%d samples), "
            "Holdout: %s to %s (%d samples). "
            "Holdout is chronologically after CV: %s",
            cv_ts.min() if not cv_ts.empty else "N/A",
            cv_ts.max() if not cv_ts.empty else "N/A",
            len(cv_df),
            holdout_ts.min() if not holdout_ts.empty else "N/A",
            holdout_ts.max() if not holdout_ts.empty else "N/A",
            len(holdout_df),
            bool(holdout_ts.empty or cv_ts.empty or holdout_ts.min() > cv_ts.max()),
        )

        # Generate CV splits on the CV portion only
        cv_splits = self.split(cv_df, timestamp_col, t1_col)

        return cv_df, holdout_df, cv_splits

    def get_holdout_metrics(
        self,
        cv_df: pd.DataFrame,
        holdout_df: pd.DataFrame,
        timestamp_col: str = "start_time_utc",
    ) -> Dict[str, Any]:
        """
        Get metrics about the holdout split for artifact metadata.

        Args:
            cv_df: Cross-validation data
            holdout_df: True holdout data
            timestamp_col: Name of timestamp column

        Returns:
            Dictionary with holdout metrics for artifact metadata
        """
        if holdout_df.empty:
            return {
                "holdout_configured": False,
                "holdout_pct": 0.0,
            }

        cv_timestamps = _to_datetime_utc_mixed(cv_df[timestamp_col])
        holdout_timestamps = _to_datetime_utc_mixed(holdout_df[timestamp_col])

        return {
            "holdout_configured": True,
            "holdout_pct": self.config.holdout_pct,
            "cv_samples": len(cv_df),
            "holdout_samples": len(holdout_df),
            "cv_start": cv_timestamps.min().isoformat() if not cv_timestamps.empty else None,
            "cv_end": cv_timestamps.max().isoformat() if not cv_timestamps.empty else None,
            "holdout_start": holdout_timestamps.min().isoformat() if not holdout_timestamps.empty else None,
            "holdout_end": holdout_timestamps.max().isoformat() if not holdout_timestamps.empty else None,
        }


@dataclass
class PBOResult:
    """Result of Probability of Backtest Overfitting computation."""

    pbo: float  # Probability of Backtest Overfitting [0, 1]
    logit_distribution: np.ndarray  # Distribution of logit(lambda) across paths
    n_paths: int  # Number of CPCV paths evaluated
    n_overfit: int  # Number of paths where IS-best underperforms OOS


def compute_pbo(
    path_returns: Dict[int, np.ndarray],
    n_paths: int,
) -> PBOResult:
    """
    Compute Probability of Backtest Overfitting (PBO) per AFML Ch 11.

    PBO measures the fraction of CPCV paths where the strategy selected
    as best in-sample turns out to be below median out-of-sample.

    Algorithm:
    1. For each CPCV path, split strategy performance into IS and OOS.
    2. Rank strategies by IS performance, pick the best.
    3. Check if that best-IS strategy's OOS rank is below median.
    4. PBO = fraction of paths where best-IS is below-median OOS.

    For a single-strategy system (our case), we treat each CPCV path as
    an independent trial. IS performance = train Sharpe, OOS = test Sharpe.
    PBO = fraction of paths where OOS Sharpe < 0 (underperforms benchmark).

    Args:
        path_returns: Dict mapping path_idx -> array of OOS returns
        n_paths: Total number of CPCV paths

    Returns:
        PBOResult with PBO probability and diagnostics
    """
    if not path_returns or n_paths == 0:
        return PBOResult(pbo=1.0, logit_distribution=np.array([]), n_paths=0, n_overfit=0)

    oos_sharpes = []
    for path_idx in sorted(path_returns.keys()):
        rets = path_returns[path_idx]
        if len(rets) > 1 and np.std(rets) > 0:
            sr = np.mean(rets) / np.std(rets)
        else:
            sr = 0.0
        oos_sharpes.append(sr)

    oos_sharpes = np.array(oos_sharpes)

    # For single-strategy: overfit when OOS Sharpe < 0
    n_overfit = int(np.sum(oos_sharpes < 0))
    pbo = n_overfit / len(oos_sharpes) if len(oos_sharpes) > 0 else 1.0

    # Compute logit distribution: logit(rank / n_paths)
    # rank = relative rank of the OOS sharpe within all paths
    ranks = np.argsort(np.argsort(oos_sharpes)) + 1  # 1-based ranks
    lambdas = ranks / (len(ranks) + 1)  # normalize to (0, 1)
    # Clip to avoid log(0)
    lambdas = np.clip(lambdas, 1e-6, 1.0 - 1e-6)
    logit_dist = np.log(lambdas / (1.0 - lambdas))

    return PBOResult(
        pbo=pbo,
        logit_distribution=logit_dist,
        n_paths=len(oos_sharpes),
        n_overfit=n_overfit,
    )


def compute_min_track_record_length(
    observed_sharpe: float,
    benchmark_sharpe: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> int:
    """
    Compute Minimum Track Record Length (MinTRL) per AFML Ch 14.

    MinTRL is the minimum number of observations needed before we can
    statistically distinguish a strategy's Sharpe ratio from the benchmark
    at the given confidence level.

    Formula (AFML 14.12):
        MinTRL = 1 + (1 - skew*SR + (kurt-1)/4 * SR^2) * (z_alpha / (SR - SR*))^2

    Args:
        observed_sharpe: Observed Sharpe ratio (SR)
        benchmark_sharpe: Benchmark Sharpe ratio (SR*, default 0)
        skewness: Return skewness
        kurtosis: Return kurtosis (excess kurtosis + 3; normal = 3)
        confidence: Confidence level (default 0.95)

    Returns:
        Minimum number of observations required (at least 1)

    Raises:
        ValueError: If confidence is not in (0, 1)
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    sr_diff = observed_sharpe - benchmark_sharpe
    if abs(sr_diff) < 1e-10:
        # SR equals benchmark: infinite observations needed
        return 2**31 - 1  # max int sentinel

    if abs(observed_sharpe) < 1e-10:
        # SR is zero: formula simplifies (skew term vanishes)
        z_alpha = stats.norm.ppf(confidence)
        mintrl = 1 + (z_alpha / sr_diff) ** 2
        return max(1, int(np.ceil(mintrl)))

    z_alpha = stats.norm.ppf(confidence)

    # Variance adjustment for non-normal returns (AFML Eq 14.4)
    variance_factor = (
        1.0
        + 0.5 * observed_sharpe ** 2
        - skewness * observed_sharpe
        + (kurtosis - 3) / 4.0 * observed_sharpe ** 2
    )

    mintrl = 1 + variance_factor * (z_alpha / sr_diff) ** 2
    return max(1, int(np.ceil(mintrl)))


class DeflatedSharpeCalculator:
    """
    Calculate deflated Sharpe ratio to account for multiple testing.

    AFML warns that high Sharpe ratios from many trials are often spurious.
    The deflated Sharpe adjusts for the expected maximum Sharpe under the null.

    Usage:
        calc = DeflatedSharpeCalculator(n_trials=100)
        result = calc.deflate(raw_sharpe=1.5, n_observations=252)
        print(f"Deflated Sharpe: {result.deflated_sharpe:.2f}")
    """

    def __init__(
        self,
        n_trials: int = 1,
        annual_trading_days: int = 365,
    ):
        """
        Initialize DeflatedSharpeCalculator.

        Args:
            n_trials: Number of strategy trials/comparisons
            annual_trading_days: Trading days per year
        """
        self.n_trials = max(1, n_trials)
        self.annual_trading_days = annual_trading_days

    def expected_max_sharpe(
        self,
        n_trials: Optional[int] = None,
        var_sharpe: float = 1.0,
    ) -> float:
        """
        Calculate expected maximum Sharpe ratio under the null.

        Under the null (no skill), the expected maximum Sharpe from N trials is:
        E[max(SR)] ≈ sqrt(2 * log(N)) * σ(SR)

        Args:
            n_trials: Number of trials (default: self.n_trials)
            var_sharpe: Variance of Sharpe estimates

        Returns:
            Expected maximum Sharpe under null
        """
        if n_trials is None:
            n_trials = self.n_trials

        if n_trials <= 1:
            return 0.0

        # Bailey and López de Prado (2014) approximation
        gamma = 0.5772  # Euler-Mascheroni constant
        expected_max = (
            (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials) +
            gamma * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        ) * np.sqrt(var_sharpe)

        return max(0.0, expected_max)

    def prob_sharpe_ratio(
        self,
        observed_sharpe: float,
        benchmark_sharpe: float,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """
        Calculate probability that observed Sharpe > benchmark.

        Probabilistic Sharpe Ratio (PSR) from AFML.

        Args:
            observed_sharpe: Observed Sharpe ratio
            benchmark_sharpe: Benchmark Sharpe (often 0 or expected max)
            n_observations: Number of return observations
            skewness: Return skewness
            kurtosis: Return kurtosis

        Returns:
            Probability that true Sharpe > benchmark
        """
        if n_observations <= 1:
            return 0.5

        # Standard error of Sharpe ratio
        se_sharpe = np.sqrt(
            (1.0 + 0.5 * observed_sharpe ** 2 -
             skewness * observed_sharpe +
             (kurtosis - 3) / 4.0 * observed_sharpe ** 2) / (n_observations - 1)
        )

        if se_sharpe <= 0:
            return 0.5

        # Z-score
        z = (observed_sharpe - benchmark_sharpe) / se_sharpe

        # Probability
        return float(stats.norm.cdf(z))

    def deflate(
        self,
        raw_sharpe: float,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        n_trials: Optional[int] = None,
        var_sharpe: float = 1.0,
    ) -> DeflatedSharpeResult:
        """
        Calculate deflated Sharpe ratio.

        Args:
            raw_sharpe: Unadjusted Sharpe ratio
            n_observations: Number of return observations
            skewness: Return skewness
            kurtosis: Return kurtosis
            n_trials: Override for number of trials
            var_sharpe: Variance of Sharpe estimates across paths/trials

        Returns:
            DeflatedSharpeResult with deflated Sharpe and statistics
        """
        if n_trials is None:
            n_trials = self.n_trials

        # Expected max Sharpe under null
        exp_max = self.expected_max_sharpe(n_trials, var_sharpe=var_sharpe)

        # Deflated Sharpe: raw - expected max under null
        deflated = raw_sharpe - exp_max
        haircut_pct = (1.0 - deflated / raw_sharpe) * 100 if raw_sharpe != 0 else 0.0

        # Probability of false discovery
        prob_fd = 1.0 - self.prob_sharpe_ratio(
            raw_sharpe, exp_max, n_observations, skewness, kurtosis
        )

        # Confidence interval — full AFML SE formula matching prob_sharpe_ratio()
        se_sharpe = np.sqrt(
            (1.0 + 0.5 * raw_sharpe ** 2 -
             skewness * raw_sharpe +
             (kurtosis - 3) / 4.0 * raw_sharpe ** 2) / max(1, n_observations - 1)
        )
        ci_lower = raw_sharpe - 1.96 * se_sharpe
        ci_upper = raw_sharpe + 1.96 * se_sharpe

        # Minimum Track Record Length (AFML Ch 14)
        try:
            mintrl = compute_min_track_record_length(
                observed_sharpe=raw_sharpe,
                benchmark_sharpe=exp_max,
                skewness=skewness,
                kurtosis=kurtosis,
            )
        except (ValueError, ZeroDivisionError):
            mintrl = None

        return DeflatedSharpeResult(
            raw_sharpe=raw_sharpe,
            deflated_sharpe=deflated,
            haircut_pct=haircut_pct,
            n_trials=n_trials,
            expected_max_sharpe=exp_max,
            prob_false_discovery=prob_fd,
            sharpe_ci_lower=ci_lower,
            sharpe_ci_upper=ci_upper,
            min_track_record_length=mintrl,
        )


def create_cpcv(
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge_pct: float = 0.0,
    embargo_pct: float = 0.0,
    purge_hours: float = 12.0,
    embargo_hours: float = 1.5,
    holdout_pct: float = 0.20,
) -> CPCV:
    """
    Factory function to create CPCV.

    Args:
        n_groups: Number of groups to split data
        n_test_groups: Number of groups for testing per path
        purge_pct: Purge fraction
        embargo_pct: Embargo fraction
        purge_hours: Purge window in hours (time-based purging)
        embargo_hours: Embargo window in hours (time-based purging)
        holdout_pct: Fraction of data reserved as true holdout

    Returns:
        Configured CPCV instance
    """
    config = CPCVConfig(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge_pct=purge_pct,
        embargo_pct=embargo_pct,
        purge_hours=purge_hours,
        embargo_hours=embargo_hours,
        holdout_pct=holdout_pct,
    )
    return CPCV(config)


def create_deflated_sharpe_calculator(
    n_trials: int = 1,
    annual_trading_days: int = 365,
) -> DeflatedSharpeCalculator:
    """
    Factory function to create DeflatedSharpeCalculator.

    Args:
        n_trials: Number of strategy trials
        annual_trading_days: Trading days per year

    Returns:
        Configured DeflatedSharpeCalculator instance
    """
    return DeflatedSharpeCalculator(
        n_trials=n_trials,
        annual_trading_days=annual_trading_days,
    )
