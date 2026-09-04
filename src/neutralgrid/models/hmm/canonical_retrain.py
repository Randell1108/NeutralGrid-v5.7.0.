"""
Canonical HMM retrain infrastructure.

Provides a shared, auditable pipeline for HMM retrain runs:

1. **Universe selection** -- probe deeper than target, freeze ranked list.
2. **Store coverage** -- ensure Binance Vision kline stores are up-to-date.
3. **Run boundary** -- freeze a common [start, end] window across all symbols.
4. **Screening** -- walk the ranked list, accept symbols that meet coverage
   and quality gates, stop at exactly *target_accepted*.
5. **Training slice** -- load, validate, persist the frozen training data.

All functions are deterministic given the same store state and produce
frozen / immutable outputs so the training run can be fully reproduced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, cast

import pandas as pd

from neutralgrid.data.binance_vision.pipeline import ensure_kline_store
from neutralgrid.data.binance_vision.store import load_parquet_range, parquet_root
from neutralgrid.data.binance_vision.validate import validate_kline_store
from neutralgrid.data.market_data import save_training_dataset
from neutralgrid.scanner.scan import get_top_usdtm_symbols_by_quote_volume

logger = logging.getLogger(__name__)

# Interval → bars per hour mapping (used for interval-aware row-count math)
_BARS_PER_HOUR: Dict[str, int] = {"1h": 1, "15m": 4, "5m": 12, "1m": 60}


# ---------------------------------------------------------------------------
# Step 1 -- Frozen probe universe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrozenProbeUniverse:
    """Immutable snapshot of the probe universe at selection time."""

    selection_time_utc: datetime
    ranked_symbols: List[str]
    ranked_volumes: List[float]
    target_accepted: int


async def build_frozen_probe_universe(
    client: Any,
    *,
    target_accepted: int = 50,
    probe_depth_multiplier: float = 2.0,
) -> FrozenProbeUniverse:
    """Select a probe universe deeper than the target accepted count.

    Calls the canonical scanner selector and freezes the result.

    Args:
        client: Binance API client.
        target_accepted: How many symbols the downstream screen should accept.
        probe_depth_multiplier: How many multiples of *target_accepted* to fetch
            so the screen has headroom to reject poor-coverage symbols.

    Returns:
        A frozen dataclass with ranked symbols, volumes, and metadata.
    """
    probe_n = int(target_accepted * probe_depth_multiplier)
    rows = await get_top_usdtm_symbols_by_quote_volume(client, top_n=probe_n)

    selection_time_utc = datetime.now(timezone.utc)
    ranked_symbols = [sym for sym, _ in rows]
    ranked_volumes = [vol for _, vol in rows]

    logger.info(
        "Probe universe: %d symbols fetched (target_accepted=%d, depth=%.1fx)",
        len(ranked_symbols),
        target_accepted,
        probe_depth_multiplier,
    )
    logger.info("Top-10 probe symbols: %s", ", ".join(ranked_symbols[:10]))

    return FrozenProbeUniverse(
        selection_time_utc=selection_time_utc,
        ranked_symbols=ranked_symbols,
        ranked_volumes=ranked_volumes,
        target_accepted=target_accepted,
    )


# ---------------------------------------------------------------------------
# Step 2 -- Canonical store coverage
# ---------------------------------------------------------------------------

async def ensure_canonical_store_coverage(
    symbols: List[str],
    *,
    interval: str = "15m",
    market: Literal["futures_um", "spot"] = "futures_um",
    min_years: float = 180 / 365.25,
    cache_root: str = "data/cache/klines",
) -> Dict[str, Path]:
    """Ensure every symbol has a Binance Vision kline store with coverage.

    For each symbol, delegates to :func:`ensure_kline_store` to backfill or
    incrementally update the Hive-partitioned Parquet store.

    Args:
        symbols: List of symbol strings to ensure stores for.
        interval: Kline interval (default ``"15m"``).
        market: Binance market type (default ``"futures_um"``).
        min_years: Minimum history depth in years (0.5 = 180 days).
        cache_root: Root directory for kline cache.

    Returns:
        Dict mapping symbol -> :func:`parquet_root` path.
    """
    store_roots: Dict[str, Path] = {}
    initialised = 0
    updated = 0

    for sym in symbols:
        pq_root = parquet_root(cache_root, market, sym, interval)
        existed_before = pq_root.exists()

        try:
            await ensure_kline_store(
                sym,
                interval=interval,
                market=market,
                min_years=min_years,
                cache_root=cache_root,
                mode="auto",
            )
        except Exception:
            logger.warning("Store coverage failed for %s -- skipping", sym, exc_info=True)
            continue

        store_roots[sym] = pq_root

        if existed_before:
            updated += 1
        else:
            initialised += 1

    logger.info(
        "Store coverage: %d symbols ready (%d initialised, %d updated)",
        len(store_roots),
        initialised,
        updated,
    )
    return store_roots


# ---------------------------------------------------------------------------
# Step 3 -- Canonical run boundary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalRunBoundary:
    """Frozen time boundary for a canonical retrain run."""

    selection_time_utc: datetime
    target_end_utc: datetime
    common_start_utc: datetime
    common_end_utc: datetime
    window_days: int
    timestamp_basis: str  # always "open_time"


def freeze_run_boundary(
    store_roots: Dict[str, Path],
    selection_time_utc: datetime,
    window_days: int = 180,
) -> CanonicalRunBoundary:
    """Determine the common [start, end] window across all symbols.

    For each symbol store, loads the full parquet and finds the latest
    ``open_time``.  The *target_end_utc* is the **minimum** of all
    symbols' latest bar so that every symbol shares the same end.

    Args:
        store_roots: Dict mapping symbol -> parquet root path.
        selection_time_utc: When the probe universe was selected.
        window_days: Size of the rolling training window in days.

    Returns:
        A frozen :class:`CanonicalRunBoundary`.

    Raises:
        ValueError: If no usable store data is found.
    """
    if not store_roots:
        raise ValueError("freeze_run_boundary: store_roots is empty -- nothing to bound")

    latest_per_symbol: Dict[str, datetime] = {}

    for sym, root in store_roots.items():
        df = load_parquet_range(root)
        if df.empty:
            logger.warning("freeze_run_boundary: store for %s is empty", sym)
            continue
        max_ts = cast(Any, df["open_time"].max())
        if max_ts is None or bool(pd.isna(max_ts)):
            logger.warning("freeze_run_boundary: %s has NaT max open_time", sym)
            continue
        latest_per_symbol[sym] = pd.to_datetime(max_ts, utc=True).to_pydatetime()

    if not latest_per_symbol:
        raise ValueError(
            "freeze_run_boundary: no valid open_time found in any store"
        )

    target_end_utc = min(latest_per_symbol.values())
    common_start_utc = target_end_utc - timedelta(days=window_days)

    logger.info(
        "Run boundary: %s to %s (%d days, %d symbols)",
        common_start_utc.isoformat(),
        target_end_utc.isoformat(),
        window_days,
        len(latest_per_symbol),
    )

    return CanonicalRunBoundary(
        selection_time_utc=selection_time_utc,
        target_end_utc=target_end_utc,
        common_start_utc=common_start_utc,
        common_end_utc=target_end_utc,
        window_days=window_days,
        timestamp_basis="open_time",
    )


# ---------------------------------------------------------------------------
# Step 4 -- Screen to exactly N accepted symbols
# ---------------------------------------------------------------------------

@dataclass
class ScreeningResult:
    """Outcome of screening the probe universe."""

    accepted: List[str]
    rejected: List[Tuple[str, str]]  # (symbol, reason)
    common_window: CanonicalRunBoundary


def screen_probe_universe(
    probe: FrozenProbeUniverse,
    store_roots: Dict[str, Path],
    boundary: CanonicalRunBoundary,
    *,
    interval: str = "15m",
    min_coverage_pct: float = 0.90,
) -> ScreeningResult:
    """Walk ranked symbols and accept up to *target_accepted* that pass QA.

    Checks per symbol:
    - Enough bars (>= ``window_days * 24 * bars_per_hour * min_coverage_pct``).
    - Monotonic ``open_time`` (no backwards jumps).
    - No duplicate ``open_time``.

    Args:
        probe: Frozen probe universe from :func:`build_frozen_probe_universe`.
        store_roots: Dict mapping symbol -> parquet root path.
        boundary: Frozen run boundary from :func:`freeze_run_boundary`.
        interval: Kline interval (e.g. "1h", "15m"). Determines bars-per-hour.
        min_coverage_pct: Minimum fraction of expected bars required.

    Returns:
        A :class:`ScreeningResult` with accepted and rejected lists.

    Raises:
        ValueError: If fewer than *target_accepted* symbols pass screening.
    """
    accepted: List[str] = []
    rejected: List[Tuple[str, str]] = []
    target = probe.target_accepted
    bars_per_hour = _BARS_PER_HOUR.get(interval, 1)

    expected_bars = int(boundary.window_days * 24 * bars_per_hour * min_coverage_pct)

    start_ms = int(boundary.common_start_utc.timestamp() * 1000)
    end_ms = int(boundary.common_end_utc.timestamp() * 1000)

    for sym in probe.ranked_symbols:
        if len(accepted) >= target:
            break

        if sym not in store_roots:
            reason = "no store available"
            rejected.append((sym, reason))
            logger.info("Rejected %s: %s", sym, reason)
            continue

        root = store_roots[sym]
        df = load_parquet_range(root, start_ms=start_ms, end_ms=end_ms)

        # -- Check 1: enough bars --
        if len(df) < expected_bars:
            reason = f"insufficient bars ({len(df)} < {expected_bars})"
            rejected.append((sym, reason))
            logger.info("Rejected %s: %s", sym, reason)
            continue

        # -- Check 2: monotonic open_time --
        open_times = cast(pd.Series, pd.to_datetime(df["open_time"], errors="coerce", utc=True))
        diffs = open_times.diff().dropna()
        non_mono = int((diffs.dt.total_seconds() <= 0).sum())
        if non_mono > 0:
            reason = f"non-monotonic open_time ({non_mono} violations)"
            rejected.append((sym, reason))
            logger.info("Rejected %s: %s", sym, reason)
            continue

        # -- Check 3: no duplicate open_time --
        n_dupes = int(open_times.duplicated().sum())
        if n_dupes > 0:
            reason = f"duplicate open_time ({n_dupes} duplicates)"
            rejected.append((sym, reason))
            logger.info("Rejected %s: %s", sym, reason)
            continue

        accepted.append(sym)

    if len(accepted) < target:
        raise ValueError(
            f"Screening: only {len(accepted)} symbols accepted "
            f"(target={target}). "
            f"Rejected {len(rejected)} symbols. "
            "Increase probe_depth_multiplier or reduce target_accepted."
        )

    logger.info(
        "Screening: %d accepted, %d rejected (target=%d)",
        len(accepted),
        len(rejected),
        target,
    )

    return ScreeningResult(
        accepted=accepted,
        rejected=rejected,
        common_window=boundary,
    )


# ---------------------------------------------------------------------------
# Step 5 -- Freeze training slice
# ---------------------------------------------------------------------------

@dataclass
class FrozenTrainingSlice:
    """Immutable training dataset ready for HMM training."""

    datasets: Dict[str, pd.DataFrame]
    metadata: Dict[str, Any]
    output_dir: Path


def freeze_training_slice(
    screening: ScreeningResult,
    store_roots: Dict[str, Path],
    boundary: CanonicalRunBoundary,
    output_dir: Path,
    *,
    interval: str = "15m",
) -> FrozenTrainingSlice:
    """Load the exact training window for each accepted symbol and persist.

    For each accepted symbol:
    1. Load the ``[common_start_utc, common_end_utc]`` slice from the store.
    2. Inject ``symbol`` column (not present in the Vision store).
    3. Run :func:`validate_kline_store` with relaxed ``min_rows``.
    4. Persist via :func:`save_training_dataset`.

    Args:
        screening: Screening result with accepted symbols.
        store_roots: Dict mapping symbol -> parquet root.
        boundary: Frozen run boundary.
        output_dir: Directory to save the training dataset.
        interval: Kline interval (e.g. "15m", "1h"). Determines bars-per-hour for row math.

    Returns:
        A :class:`FrozenTrainingSlice` containing loaded DataFrames and
        metadata.

    Raises:
        ValueError: If any accepted symbol fails slice-level validation.
    """
    datasets: Dict[str, pd.DataFrame] = {}

    start_ms = int(boundary.common_start_utc.timestamp() * 1000)
    end_ms = int(boundary.common_end_utc.timestamp() * 1000)

    bars_per_hour = _BARS_PER_HOUR.get(interval, 1)
    # Relaxed min_rows: e.g. 180 * 24 * 4 * 0.90 = 15552 for 15m
    relaxed_min_rows = int(boundary.window_days * 24 * bars_per_hour * 0.90)

    for sym in screening.accepted:
        root = store_roots[sym]
        df = load_parquet_range(root, start_ms=start_ms, end_ms=end_ms)

        # Inject symbol column (Vision store doesn't have it)
        df = df.copy()
        df["symbol"] = sym

        # Slice-local validation with relaxed min_rows
        result = validate_kline_store(df, interval=interval, min_rows=relaxed_min_rows)

        if not result.passed:
            raise ValueError(
                f"freeze_training_slice: validation failed for {sym}: "
                f"{result.summary()}"
            )

        datasets[sym] = df

    # Build metadata
    rejected_summary = [
        {"symbol": sym, "reason": reason}
        for sym, reason in screening.rejected
    ]

    metadata: Dict[str, Any] = {
        "canonical_source": "binance_vision",
        "selection_time_utc": boundary.selection_time_utc.isoformat(),
        "target_end_utc": boundary.target_end_utc.isoformat(),
        "common_start_utc": boundary.common_start_utc.isoformat(),
        "common_end_utc": boundary.common_end_utc.isoformat(),
        "window_days": boundary.window_days,
        "timestamp_basis": boundary.timestamp_basis,
        "accepted_symbols": screening.accepted,
        "num_accepted": len(screening.accepted),
        "num_rejected": len(screening.rejected),
        "rejected_summary": rejected_summary,
    }

    # Persist via save_training_dataset
    output_dir = Path(output_dir)
    save_training_dataset(datasets, output_dir, metadata=metadata)

    logger.info(
        "Frozen training slice: %d symbols, %d total bars, saved to %s",
        len(datasets),
        sum(len(df) for df in datasets.values()),
        output_dir,
    )

    return FrozenTrainingSlice(
        datasets=datasets,
        metadata=metadata,
        output_dir=output_dir,
    )
