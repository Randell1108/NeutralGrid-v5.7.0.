"""
Data Curator Layer for AFML-Compliant Data Quality.

Implements centralized data quality controls before feature generation
and model inference.

AFML stresses that data pipelines need controls against:
- Missing bars / irregular sampling
- Stale candles (especially if including the "current forming candle")
- Outliers and bad ticks
- Unit consistency (percent vs decimal)
- NaN/inf propagation

Design principles:
- Schema + checks + metrics before feature generation
- Explicit quality contracts with pass/fail
- Configurable thresholds for different use cases
- Audit trail for data quality issues
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, cast
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataQualityConfig:
    """Configuration for data quality checks."""

    # Missing data thresholds
    max_missing_bars_pct: float = 0.05  # Max 5% missing bars
    max_gap_multiplier: float = 3.0  # Max gap = 3x expected interval

    # Stale data detection
    max_stale_seconds: int = 300  # 5 minutes max staleness

    # Outlier detection
    price_change_max_pct: float = 10.0  # Max 10% single-bar price change
    volume_spike_mult: float = 20.0  # Volume > 20x median is outlier

    # NaN/Inf handling
    max_nan_pct: float = 0.02  # Max 2% NaN values

    # Unit consistency
    expected_decimal_columns: List[str] = field(default_factory=lambda: [
        "funding_rate", "profit_factor", "pnl_pct"
    ])
    expected_percent_columns: List[str] = field(default_factory=lambda: [
        "grid_spacing_pct", "profit_per_grid_pct", "range_size_pct"
    ])


@dataclass
class DataQualityResult:
    """Result of data quality validation."""

    passed: bool
    checks: Dict[str, bool]
    metrics: Dict[str, Any]
    issues: List[str]
    warnings: List[str]

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "PASS" if self.passed else "FAIL"
        issues_str = "; ".join(self.issues) if self.issues else "none"
        warnings_str = "; ".join(self.warnings) if self.warnings else "none"
        return f"[{status}] Issues: {issues_str} | Warnings: {warnings_str}"


class DataCurator:
    """
    Centralized data quality controller.

    Validates OHLCV data and feature DataFrames before model inference.

    Usage:
        curator = DataCurator(DataQualityConfig())

        # Validate OHLCV data
        result = curator.validate_ohlcv(df, timeframe="15m")
        if not result.passed:
            logger.warning(f"Data quality issues: {result.summary()}")

        # Clean data
        df_clean, clean_report = curator.clean_ohlcv(df)
    """

    def __init__(self, config: Optional[DataQualityConfig] = None):
        """
        Initialize DataCurator.

        Args:
            config: DataQualityConfig with thresholds
        """
        self.config = config or DataQualityConfig()

    def check_missing_bars(
        self,
        df: pd.DataFrame,
        timeframe: str,
        timestamp_col: str = "open_time",
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Check for missing bars in OHLCV data.

        Args:
            df: DataFrame with timestamp column
            timeframe: Expected timeframe ("1m", "5m", "15m", "1h", etc.)
            timestamp_col: Name of timestamp column

        Returns:
            Tuple of (passed, metrics, issues)
        """
        issues = []
        metrics = {}

        if len(df) < 2:
            return (True, {"bar_count": len(df)}, [])

        # Parse timeframe to expected interval
        interval_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }
        expected_interval = interval_map.get(timeframe, 3600)

        # Ensure timestamp is datetime
        timestamps = pd.to_datetime(df[timestamp_col], errors="coerce")
        # AFML: Log dropped rows instead of silent dropna
        dropped_ts = timestamps.isna().sum()
        if dropped_ts > 0:
            logger.warning(f"check_missing_bars: dropping {dropped_ts} rows with invalid timestamps")
        timestamps = timestamps.dropna()

        if len(timestamps) < 2:
            return (True, {"bar_count": len(df)}, [])

        # Calculate gaps
        time_diffs = timestamps.diff()
        # AFML: Log dropped rows instead of silent dropna (first row is always NaT after diff)
        dropped_diffs = time_diffs.isna().sum()
        if dropped_diffs > 1:  # First row is expected to be NaT
            logger.warning(f"check_missing_bars: dropping {dropped_diffs - 1} unexpected NaT time differences")
        time_diffs = time_diffs.dropna()
        time_diffs_seconds = time_diffs.dt.total_seconds()

        # Find gaps that are too large
        max_allowed_gap = expected_interval * self.config.max_gap_multiplier
        large_gaps = time_diffs_seconds[time_diffs_seconds > max_allowed_gap]

        # Calculate missing bar percentage
        expected_bars = (timestamps.max() - timestamps.min()).total_seconds() / expected_interval
        if expected_bars > 0:
            missing_pct = 1.0 - (len(df) / expected_bars)
        else:
            missing_pct = 0.0

        metrics = {
            "bar_count": len(df),
            "expected_bars": int(expected_bars),
            "missing_pct": round(missing_pct * 100, 2),
            "large_gaps_count": len(large_gaps),
            "max_gap_seconds": float(time_diffs_seconds.max()) if len(time_diffs_seconds) > 0 else 0,
        }

        passed = True

        if missing_pct > self.config.max_missing_bars_pct:
            passed = False
            issues.append(f"missing_bars({missing_pct*100:.1f}%>{self.config.max_missing_bars_pct*100}%)")

        if len(large_gaps) > 0:
            issues.append(f"large_gaps({len(large_gaps)})")

        return (passed, metrics, issues)

    def check_stale_data(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "open_time",
        reference_time: Optional[datetime] = None,
        timeframe: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Check if data is stale (too old).

        The staleness threshold is timeframe-aware: the last *complete* bar can
        be up to one full interval old, so the effective max staleness is
        ``interval_seconds + config.max_stale_seconds`` (the configured value
        acts as a buffer on top of the bar interval).

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            reference_time: Reference time (default: now)
            timeframe: Bar timeframe (e.g. "1h", "15m").  When provided the
                       threshold accounts for the bar interval.

        Returns:
            Tuple of (passed, metrics, issues)
        """
        issues = []

        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        if len(df) == 0:
            return (False, {"staleness_seconds": float("inf")}, ["empty_dataframe"])

        # Get most recent timestamp
        timestamps = pd.to_datetime(df[timestamp_col], errors="coerce")
        if timestamps.isna().all():
            return (False, {"staleness_seconds": float("inf")}, ["invalid_timestamps"])

        most_recent = timestamps.max()
        if pd.isna(most_recent):
            return (False, {"staleness_seconds": float("inf")}, ["invalid_timestamps"])

        # Ensure timezone-aware comparison
        if most_recent.tzinfo is None:
            most_recent = most_recent.replace(tzinfo=timezone.utc)

        staleness = (reference_time - most_recent).total_seconds()

        # Timeframe-aware threshold: 2 * bar_interval + configured buffer.
        # The factor of 2 accounts for open_time semantics: the most recent
        # bar's open_time is 1 interval behind its close, and the current
        # (incomplete) bar can add up to 1 more interval before it appears
        # as a completed row.
        interval_map = {
            "1m": 60, "5m": 300, "15m": 900,
            "1h": 3600, "4h": 14400, "1d": 86400,
        }
        bar_interval = interval_map.get(timeframe, 0) if timeframe else 0
        effective_max = 2 * bar_interval + self.config.max_stale_seconds

        metrics = {
            "most_recent": most_recent.isoformat(),
            "staleness_seconds": round(staleness, 1),
            "effective_max_stale_seconds": effective_max,
        }

        passed = staleness <= effective_max

        if not passed:
            issues.append(f"stale_data({staleness:.0f}s>{effective_max}s)")

        return (passed, metrics, issues)

    def check_outliers(
        self,
        df: pd.DataFrame,
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Check for outliers in OHLCV data.

        Args:
            df: OHLCV DataFrame

        Returns:
            Tuple of (passed, metrics, issues)
        """
        _issues = []
        warnings = []

        if len(df) < 2:
            return (True, {}, [])

        # Initialize outlier containers to avoid unbound variables
        price_outliers = pd.Series(dtype=float)
        volume_outliers = pd.Series(dtype=float)

        # Price change outliers
        if "close" in df.columns:
            closes = cast(pd.Series, pd.to_numeric(df["close"], errors="coerce"))
            pct_changes = cast(pd.Series, closes.pct_change()).abs() * 100
            price_outliers = cast(
                pd.Series, pct_changes[pct_changes > self.config.price_change_max_pct]
            )

            if len(price_outliers) > 0:
                warnings.append(f"price_spikes({len(price_outliers)})")

        # Volume outliers
        if "volume" in df.columns:
            volumes = cast(pd.Series, pd.to_numeric(df["volume"], errors="coerce"))
            vol_median = float(volumes.median()) if len(volumes) > 0 else 0.0
            if vol_median > 0:
                volume_outliers = cast(
                    pd.Series,
                    volumes[volumes > vol_median * self.config.volume_spike_mult],
                )
                if len(volume_outliers) > 0:
                    warnings.append(f"volume_spikes({len(volume_outliers)})")

        metrics = {
            "price_outliers": len(price_outliers) if "close" in df.columns else 0,
            "volume_outliers": len(volume_outliers) if "volume" in df.columns else 0,
        }

        # Outliers are warnings, not failures
        passed = True

        return (passed, metrics, warnings)

    def check_nan_inf(
        self,
        df: pd.DataFrame,
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Check for NaN and Inf values.

        Args:
            df: DataFrame to check

        Returns:
            Tuple of (passed, metrics, issues)
        """
        issues = []

        total_cells = df.size
        if total_cells == 0:
            return (True, {"nan_pct": 0, "inf_count": 0}, [])

        # Count NaN
        nan_count = df.isna().sum().sum()
        nan_pct = nan_count / total_cells

        # Count Inf (for numeric columns)
        numeric_df = df.select_dtypes(include=[np.number])
        inf_count = np.isinf(np.asarray(numeric_df)).sum() if len(numeric_df.columns) > 0 else 0

        metrics = {
            "nan_count": int(nan_count),
            "nan_pct": round(nan_pct * 100, 2),
            "inf_count": int(inf_count),
        }

        passed = True

        if nan_pct > self.config.max_nan_pct:
            passed = False
            issues.append(f"high_nan({nan_pct*100:.1f}%>{self.config.max_nan_pct*100}%)")

        if inf_count > 0:
            passed = False
            issues.append(f"inf_values({inf_count})")

        return (passed, metrics, issues)

    def check_unit_consistency(
        self,
        df: pd.DataFrame,
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Check for unit consistency (percent vs decimal).

        Args:
            df: DataFrame to check

        Returns:
            Tuple of (passed, metrics, issues)
        """
        warnings = []

        # Check decimal columns (should be in range [-1, 1] or small)
        for col in self.config.expected_decimal_columns:
            if col in df.columns:
                values_raw = cast(pd.Series, pd.to_numeric(df[col], errors="coerce"))
                # AFML: Log dropped rows instead of silent dropna
                dropped_count = int(values_raw.isna().sum())
                if dropped_count > 0:
                    logger.warning(f"check_unit_consistency: dropping {dropped_count} NaN values in decimal column '{col}'")
                values = cast(pd.Series, values_raw.dropna())
                if len(values) > 0:
                    max_abs = float(values.abs().max())
                    if max_abs > 1.0:
                        warnings.append(f"{col}_possibly_percent(max={max_abs:.2f})")

        # Check percent columns (should be > 1 typically)
        for col in self.config.expected_percent_columns:
            if col in df.columns:
                values_raw = cast(pd.Series, pd.to_numeric(df[col], errors="coerce"))
                # AFML: Log dropped rows instead of silent dropna
                dropped_count = int(values_raw.isna().sum())
                if dropped_count > 0:
                    logger.warning(f"check_unit_consistency: dropping {dropped_count} NaN values in percent column '{col}'")
                values = cast(pd.Series, values_raw.dropna())
                if len(values) > 0:
                    max_abs = float(values.abs().max())
                    if max_abs < 0.1:
                        warnings.append(f"{col}_possibly_decimal(max={max_abs:.4f})")

        metrics = {"unit_warnings": len(warnings)}

        # Unit issues are warnings, not failures
        passed = True

        return (passed, metrics, warnings)

    def check_ohlcv_semantics(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
        """
        Validate OHLCV semantic relationships and drop invalid bars.

        Checks: high >= max(open, close), low <= min(open, close),
        high >= low, volume >= 0.

        Returns:
            Tuple of (cleaned DataFrame, metrics, issues)
        """
        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            return df, {"skipped": True}, []

        o = cast(pd.Series, pd.to_numeric(df["open"], errors="coerce"))
        h = cast(pd.Series, pd.to_numeric(df["high"], errors="coerce"))
        lo = cast(pd.Series, pd.to_numeric(df["low"], errors="coerce"))
        c = cast(pd.Series, pd.to_numeric(df["close"], errors="coerce"))

        invalid = cast(
            pd.Series,
            (h < o) | (h < c) |
            (lo > o) | (lo > c) |
            (h < lo)
        )
        if "volume" in df.columns:
            v = cast(pd.Series, pd.to_numeric(df["volume"], errors="coerce"))
            invalid = cast(pd.Series, invalid | (v < 0))

        n_invalid = int(invalid.sum())
        issues: List[str] = []
        metrics = {"invalid_ohlcv_bars": n_invalid, "total_bars": len(df)}

        if n_invalid > 0:
            logger.warning(
                "Dropping %d / %d bars with invalid OHLCV relationships",
                n_invalid, len(df),
            )
            issues.append(f"invalid_ohlcv_bars({n_invalid})")
            df = cast(pd.DataFrame, df.loc[~invalid].reset_index(drop=True))

        return df, metrics, issues

    def check_duplicates(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "open_time",
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Check for duplicate timestamps in OHLCV data."""
        issues = []
        if timestamp_col not in df.columns:
            return True, {"duplicates": 0}, []

        n_dupes = df[timestamp_col].duplicated().sum()
        metrics = {"duplicates": int(n_dupes), "total_rows": len(df)}

        if n_dupes > 0:
            issues.append(f"duplicate_timestamps({n_dupes})")
            logger.warning(f"Found {n_dupes} duplicate timestamps in {timestamp_col}")

        return n_dupes == 0, metrics, issues

    def validate_ohlcv(
        self,
        df: pd.DataFrame,
        timeframe: str = "15m",
        timestamp_col: str = "open_time",
        reference_time: Optional[datetime] = None,
    ) -> DataQualityResult:
        """
        Full validation of OHLCV data.

        Args:
            df: OHLCV DataFrame
            timeframe: Expected timeframe
            timestamp_col: Timestamp column name
            reference_time: Reference time for staleness check

        Returns:
            DataQualityResult with all checks
        """
        all_issues = []
        all_warnings = []
        all_metrics = {}
        all_checks = {}

        # Check missing bars
        passed, metrics, issues = self.check_missing_bars(df, timeframe, timestamp_col)
        all_checks["missing_bars"] = passed
        all_metrics["missing_bars"] = metrics
        all_issues.extend(issues)

        # Check stale data (timeframe-aware threshold)
        passed, metrics, issues = self.check_stale_data(
            df, timestamp_col, reference_time, timeframe=timeframe,
        )
        all_checks["stale_data"] = passed
        all_metrics["stale_data"] = metrics
        all_issues.extend(issues)

        # Check outliers
        passed, metrics, warnings = self.check_outliers(df)
        all_checks["outliers"] = passed
        all_metrics["outliers"] = metrics
        all_warnings.extend(warnings)

        # Check NaN/Inf
        passed, metrics, issues = self.check_nan_inf(df)
        all_checks["nan_inf"] = passed
        all_metrics["nan_inf"] = metrics
        all_issues.extend(issues)

        # Check duplicate timestamps
        passed, metrics, issues = self.check_duplicates(df, timestamp_col)
        all_checks["duplicates"] = passed
        all_metrics["duplicates"] = metrics
        all_issues.extend(issues)

        # Check OHLCV semantic relationships
        _, metrics, issues = self.check_ohlcv_semantics(df)
        all_checks["ohlcv_semantics"] = len(issues) == 0
        all_metrics["ohlcv_semantics"] = metrics
        all_issues.extend(issues)

        # Check unit consistency
        passed, metrics, warnings = self.check_unit_consistency(df)
        all_checks["unit_consistency"] = passed
        all_metrics["unit_consistency"] = metrics
        all_warnings.extend(warnings)

        # Overall pass: all checks must pass
        overall_passed = all(all_checks.values())

        return DataQualityResult(
            passed=overall_passed,
            checks=all_checks,
            metrics=all_metrics,
            issues=all_issues,
            warnings=all_warnings,
        )

    def clean_ohlcv(
        self,
        df: pd.DataFrame,
        fill_method: str = "ffill",
        remove_outliers: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean OHLCV data by handling NaN and optionally removing outliers.

        Args:
            df: OHLCV DataFrame
            fill_method: Method for filling NaN ("ffill", "bfill", "interpolate")
            remove_outliers: Whether to remove price spike rows

        Returns:
            Tuple of (cleaned DataFrame, cleaning report dict)
        """
        report: Dict[str, Any] = {
            "input_rows": len(df),
            "steps": [],
        }
        df_clean = cast(pd.DataFrame, df.copy())

        # Handle NaN based on fill method
        numeric_cols = list(df_clean.select_dtypes(include=[np.number]).columns)
        nan_before = int(df_clean[numeric_cols].isna().sum().sum())

        if fill_method == "ffill":
            df_clean[numeric_cols] = df_clean[numeric_cols].ffill()
        elif fill_method == "bfill":
            df_clean[numeric_cols] = df_clean[numeric_cols].bfill()
        elif fill_method == "interpolate":
            df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method="linear")

        nan_after_fill = int(df_clean[numeric_cols].isna().sum().sum())
        filled_by_method = nan_before - nan_after_fill
        if filled_by_method > 0:
            report["steps"].append({
                "action": f"fill_nan_{fill_method}",
                "cells_filled": filled_by_method,
            })

        # Fill any remaining NaN with column median
        median_filled = 0
        for col in numeric_cols:
            col_series = cast(pd.Series, df_clean[col])
            if bool(col_series.isna().any()):
                count = int(col_series.isna().sum())
                median_value = float(col_series.median()) if bool(col_series.notna().any()) else 0.0
                df_clean[col] = col_series.fillna(median_value)
                median_filled += count
        if median_filled > 0:
            report["steps"].append({
                "action": "fill_nan_median",
                "cells_filled": median_filled,
            })

        # Replace inf with NaN then fill
        inf_count = int(np.isinf(np.asarray(df_clean[numeric_cols])).sum()) if len(numeric_cols) > 0 else 0
        df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df_clean[numeric_cols] = df_clean[numeric_cols].ffill().bfill()
        if inf_count > 0:
            report["steps"].append({
                "action": "replace_inf",
                "cells_replaced": inf_count,
            })

        # Remove outliers if requested
        if remove_outliers and "close" in df_clean.columns:
            rows_before = len(df_clean)
            closes = cast(pd.Series, pd.to_numeric(df_clean["close"], errors="coerce"))
            pct_changes = cast(pd.Series, closes.pct_change()).abs() * 100
            mask = cast(pd.Series, pct_changes <= self.config.price_change_max_pct)
            mask = cast(pd.Series, mask | pct_changes.isna())  # Keep first row
            df_clean = cast(pd.DataFrame, df_clean.loc[mask].reset_index(drop=True))
            rows_removed = rows_before - len(df_clean)
            if rows_removed > 0:
                report["steps"].append({
                    "action": "remove_outliers",
                    "rows_removed": rows_removed,
                })

        report["output_rows"] = len(df_clean)
        report["total_nan_before"] = nan_before
        report["total_nan_after"] = int(df_clean[numeric_cols].isna().sum().sum()) if len(numeric_cols) > 0 else 0

        return cast(pd.DataFrame, df_clean), report
