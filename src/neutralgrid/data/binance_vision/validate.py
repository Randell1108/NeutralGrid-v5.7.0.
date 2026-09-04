"""
Data quality gates for Binance Vision kline stores.

Follows the DataQualityResult pattern from curator.py but tuned for
long-range historical data (months/years rather than live feeds).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Thresholds ---------------------------------------------------------------

MAX_GAP_RATIO: float = 0.002          # max fraction of expected bars that are gaps
MAX_SINGLE_GAP_HOURS: int = 6         # largest tolerable contiguous gap
MIN_ROWS_FOR_TRAINING: int = 39_000   # ~4.5 yr of 1H bars
OUTLIER_PCT_THRESHOLD: float = 20.0   # max abs close-to-close pct change for 1H bars

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


# --- Result dataclass (mirrors curator.DataQualityResult) --------------------

@dataclass
class ValidationResult:
    """Outcome of kline-store validation."""

    passed: bool
    checks: Dict[str, bool]
    metrics: Dict[str, Any]
    issues: List[str]
    warnings: List[str]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        issues_str = "; ".join(self.issues) if self.issues else "none"
        warnings_str = "; ".join(self.warnings) if self.warnings else "none"
        return f"[{status}] Issues: {issues_str} | Warnings: {warnings_str}"


# --- Individual checks -------------------------------------------------------

def check_ohlcv_integrity(
    df: pd.DataFrame,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Verify OHLCV price/volume invariants.

    * high >= max(open, close)
    * low  <= min(open, close)
    * volume >= 0
    * prices > 0
    """
    issues: List[str] = []
    metrics: Dict[str, Any] = {}

    if df.empty:
        return True, {"rows": 0}, []

    o = cast(pd.Series, pd.to_numeric(df["open"], errors="coerce"))
    h = cast(pd.Series, pd.to_numeric(df["high"], errors="coerce"))
    l = cast(pd.Series, pd.to_numeric(df["low"], errors="coerce"))  # noqa: E741
    c = cast(pd.Series, pd.to_numeric(df["close"], errors="coerce"))
    v = cast(pd.Series, pd.to_numeric(df["volume"], errors="coerce"))

    o_np = o.to_numpy(dtype=float, na_value=np.nan)
    h_np = h.to_numpy(dtype=float, na_value=np.nan)
    l_np = l.to_numpy(dtype=float, na_value=np.nan)
    c_np = c.to_numpy(dtype=float, na_value=np.nan)
    v_np = v.to_numpy(dtype=float, na_value=np.nan)

    bad_high = int((h_np < np.maximum(o_np, c_np)).sum())
    bad_low = int((l_np > np.minimum(o_np, c_np)).sum())
    neg_vol = int((v_np < 0).sum())
    non_pos = int((
        (o_np <= 0) | (h_np <= 0) | (l_np <= 0) | (c_np <= 0)
        | np.isnan(o_np) | np.isnan(h_np) | np.isnan(l_np) | np.isnan(c_np)
    ).sum())

    metrics = {
        "bad_high": bad_high,
        "bad_low": bad_low,
        "negative_volume": neg_vol,
        "non_positive_price": non_pos,
    }

    passed = True
    if bad_high > 0:
        passed = False
        issues.append(f"high<max(open,close) in {bad_high} rows")
    if bad_low > 0:
        passed = False
        issues.append(f"low>min(open,close) in {bad_low} rows")
    if neg_vol > 0:
        passed = False
        issues.append(f"negative volume in {neg_vol} rows")
    if non_pos > 0:
        passed = False
        issues.append(f"non-positive price in {non_pos} rows")

    return passed, metrics, issues


def check_time_cadence(
    df: pd.DataFrame,
    interval: str = "1h",
    max_gap_ratio: float = MAX_GAP_RATIO,
    max_single_gap_hours: float = MAX_SINGLE_GAP_HOURS,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Verify monotonicity and detect gaps in open_time.

    Args:
        df: Must have ``open_time`` as datetime.
        interval: Kline interval string (e.g. ``"1h"``).
        max_gap_ratio: Fraction of total expected bars allowed as gaps.
        max_single_gap_hours: Largest tolerable single gap in hours.
    """
    issues: List[str] = []
    metrics: Dict[str, Any] = {}

    if len(df) < 2:
        return True, {"rows": len(df)}, []

    ts = cast(pd.Series, pd.to_datetime(df["open_time"], errors="coerce"))
    diffs_ms = ts.diff().dt.total_seconds() * 1000
    diffs_ms = diffs_ms.dropna()

    interval_ms = INTERVAL_MS.get(interval, 3_600_000)

    # Monotonicity
    non_mono = int((diffs_ms <= 0).sum())
    if non_mono > 0:
        issues.append(f"non-monotonic open_time in {non_mono} rows")

    # Gap detection
    gap_threshold_ms = interval_ms * 1.5
    gaps = diffs_ms[diffs_ms > gap_threshold_ms]
    gap_count = len(gaps)
    max_gap_ms = float(cast(float, gaps.max())) if gap_count > 0 else 0.0
    max_gap_hours = max_gap_ms / 3_600_000

    # Gap ratio: how many bars are "missing" relative to a perfect series
    total_span_ms = (ts.max() - ts.min()).total_seconds() * 1000
    expected_bars = total_span_ms / interval_ms if interval_ms > 0 else len(df)
    gap_ratio = 1.0 - (len(df) / expected_bars) if expected_bars > 0 else 0.0

    metrics = {
        "gap_count": int(gap_count),
        "max_gap_hours": round(max_gap_hours, 2),
        "gap_ratio": round(gap_ratio, 6),
        "non_monotonic": int(non_mono),
    }

    passed = (non_mono == 0)

    if gap_ratio > max_gap_ratio:
        passed = False
        issues.append(f"gap_ratio {gap_ratio:.4f} > {max_gap_ratio}")

    if max_gap_hours > max_single_gap_hours:
        passed = False
        issues.append(f"max_gap {max_gap_hours:.1f}h > {max_single_gap_hours}h")

    return bool(passed), metrics, issues


def check_row_count(
    df: pd.DataFrame,
    min_rows: int = MIN_ROWS_FOR_TRAINING,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Ensure the store has enough rows for training."""
    row_count = len(df)
    passed = row_count >= min_rows
    issues: List[str] = []
    if not passed:
        issues.append(f"row_count {row_count} < minimum {min_rows}")
    return passed, {"row_count": row_count, "min_required": min_rows}, issues


MAX_NAN_INF_RATIO: float = 0.05     # max fraction of rows with NaN/Inf in OHLCV


def check_nan_inf(
    df: pd.DataFrame,
    max_ratio: float = MAX_NAN_INF_RATIO,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Check OHLCV columns for NaN and Inf values.

    Args:
        df: DataFrame with open, high, low, close, volume columns.
        max_ratio: Maximum fraction of affected rows before failure.

    Returns:
        (passed, metrics, issues) following the standard check pattern.
    """
    issues: List[str] = []
    metrics: Dict[str, Any] = {}

    if df.empty:
        return True, {"rows": 0}, []

    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    present_cols = [c for c in ohlcv_cols if c in df.columns]

    total_rows = len(df)
    nan_counts: Dict[str, int] = {}
    inf_counts: Dict[str, int] = {}

    for col in present_cols:
        series = cast(pd.Series, pd.to_numeric(df[col], errors="coerce"))
        nan_counts[col] = int(series.isna().sum())
        series_np = series.fillna(0).to_numpy(dtype=float, na_value=0.0)
        inf_counts[col] = int(np.isinf(series_np).sum())

    total_nan = sum(nan_counts.values())
    total_inf = sum(inf_counts.values())
    # Count rows where ANY ohlcv column has NaN or Inf
    affected_mask = pd.Series(False, index=df.index, dtype=bool)
    for col in present_cols:
        series = cast(pd.Series, pd.to_numeric(df[col], errors="coerce"))
        nan_mask = cast(pd.Series, series.isna())
        inf_mask = pd.Series(
            np.isinf(series.fillna(0).to_numpy(dtype=float, na_value=0.0)),
            index=series.index,
            dtype=bool,
        )
        affected_mask = cast(pd.Series, affected_mask | nan_mask | inf_mask)
    affected_rows = int(affected_mask.sum())
    affected_ratio = affected_rows / total_rows if total_rows > 0 else 0.0

    metrics = {
        "nan_per_column": nan_counts,
        "inf_per_column": inf_counts,
        "total_nan": total_nan,
        "total_inf": total_inf,
        "affected_rows": affected_rows,
        "affected_ratio": round(affected_ratio, 6),
    }

    passed = affected_ratio <= max_ratio
    if not passed:
        issues.append(
            f"NaN/Inf in {affected_rows}/{total_rows} rows "
            f"({affected_ratio:.2%} > {max_ratio:.2%} threshold)"
        )

    return passed, metrics, issues


# --- Outlier / spike check ---------------------------------------------------

# Default thresholds per interval — single-bar |close-to-close| pct change
# above which a bar is flagged as an outlier.
_OUTLIER_PCT_THRESHOLDS: Dict[str, float] = {
    "1m": 5.0,
    "5m": 8.0,
    "15m": 12.0,
    "1h": 20.0,
    "4h": 30.0,
    "1d": 40.0,
}


def check_price_outliers(
    df: pd.DataFrame,
    interval: str = "1h",
    threshold_pct: float | None = None,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Flag bars where close-to-close pct change exceeds a threshold.

    Consistent with curator philosophy: outliers are flagged in metrics and
    warnings but do NOT cause the gate to fail (``passed`` is always True).
    Downstream consumers can inspect ``metrics["outlier_count"]`` to decide.
    """
    issues: List[str] = []

    if len(df) < 2 or "close" not in df.columns:
        return True, {"outlier_count": 0, "outlier_pct": 0.0}, issues

    thr = threshold_pct if threshold_pct is not None else _OUTLIER_PCT_THRESHOLDS.get(interval, 20.0)
    closes = cast(pd.Series, pd.to_numeric(df["close"], errors="coerce"))
    pct_change = cast(pd.Series, closes.pct_change().abs() * 100.0)
    outlier_mask = pct_change > thr
    outlier_count = int(outlier_mask.sum())
    outlier_pct = round(outlier_count / len(df) * 100.0, 4)

    metrics: Dict[str, Any] = {
        "outlier_count": outlier_count,
        "outlier_pct": outlier_pct,
        "threshold_pct": thr,
    }

    if outlier_count > 0:
        # Report up to 5 worst spikes for diagnostics
        worst = cast(pd.Series, pct_change[outlier_mask]).nlargest(5)
        worst_idx: list[tuple[int, float]] = [
            (int(idx), round(float(val), 2))  # type: ignore[arg-type]
            for idx, val in worst.items()
        ]
        metrics["worst_spikes"] = worst_idx
        issues.append(
            f"price_outliers({outlier_count} bars > {thr}% change, "
            f"worst={worst_idx[0][1]}%)"
        )
        logger.warning(
            "Binance Vision outlier check: %d bars exceed %.1f%% threshold",
            outlier_count, thr,
        )

    # Flag only — never fail the gate (consistent with curator)
    return True, metrics, issues


# --- Aggregate validator -----------------------------------------------------

def validate_kline_store(
    df: pd.DataFrame,
    interval: str = "1h",
    min_rows: int = MIN_ROWS_FOR_TRAINING,
    max_gap_ratio: float = MAX_GAP_RATIO,
    max_single_gap_hours: float = MAX_SINGLE_GAP_HOURS,
) -> ValidationResult:
    """Run all quality gates on a kline DataFrame.

    Returns a :class:`ValidationResult` following the ``curator.py`` pattern.
    """
    all_checks: Dict[str, bool] = {}
    all_metrics: Dict[str, Any] = {}
    all_issues: List[str] = []
    all_warnings: List[str] = []

    # 1. OHLCV integrity
    ok, m, iss = check_ohlcv_integrity(df)
    all_checks["ohlcv_integrity"] = ok
    all_metrics["ohlcv_integrity"] = m
    all_issues.extend(iss)

    # 2. Time cadence
    ok, m, iss = check_time_cadence(
        df,
        interval=interval,
        max_gap_ratio=max_gap_ratio,
        max_single_gap_hours=max_single_gap_hours,
    )
    all_checks["time_cadence"] = ok
    all_metrics["time_cadence"] = m
    all_issues.extend(iss)

    # 3. Row count
    ok, m, iss = check_row_count(df, min_rows=min_rows)
    all_checks["row_count"] = ok
    all_metrics["row_count"] = m
    all_issues.extend(iss)

    # 4. NaN/Inf check
    ok, m, iss = check_nan_inf(df)
    all_checks["nan_inf"] = ok
    all_metrics["nan_inf"] = m
    all_issues.extend(iss)

    # 5. Price outlier / spike check (flag only, does not fail the gate)
    ok, m, iss = check_price_outliers(df, interval=interval)
    all_checks["price_outliers"] = ok
    all_metrics["price_outliers"] = m
    all_warnings.extend(iss)

    overall = all(all_checks.values())

    return ValidationResult(
        passed=overall,
        checks=all_checks,
        metrics=all_metrics,
        issues=all_issues,
        warnings=all_warnings,
    )
