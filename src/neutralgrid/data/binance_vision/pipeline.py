"""
Single orchestrator entry point for Binance Vision kline stores.

``ensure_kline_store()`` is the only function callers need.  It handles:

1. Checking the manifest to decide backfill vs. incremental update.
2. Downloading monthly or daily ZIPs (with checksum verification).
3. Ingesting ZIPs into a canonical DataFrame.
4. Running data-quality validation gates.
5. Writing Hive-partitioned Parquet.
6. Saving the manifest for next time.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, cast

import pandas as pd

from .downloader import download_kline_batch
from .ingest import ingest_zips
from .manifest import (
    create_manifest,
    get_last_open_time_ms,
    load_manifest,
    save_manifest,
)
from .store import parquet_root, write_partitioned, load_parquet_range
from .urls import Market, daily_dates, monthly_dates
from .validate import validate_kline_store

logger = logging.getLogger(__name__)

Mode = Literal["auto", "backfill", "update"]


def _previous_complete_month() -> date:
    """First day of the most recently *completed* month."""
    today = datetime.now(timezone.utc).date()
    if today.month == 1:
        return date(today.year - 1, 12, 1)
    return date(today.year, today.month - 1, 1)


def _yesterday() -> date:
    """UTC yesterday — avoids partial daily archives."""
    from datetime import timedelta
    return datetime.now(timezone.utc).date() - timedelta(days=1)


async def ensure_kline_store(
    symbol: str,
    interval: str = "1h",
    market: Market = "futures_um",
    min_years: float = 4.5,
    cache_root: str = "data/cache/klines",
    mode: Mode = "auto",
    force: bool = False,
    max_concurrency: int = 5,
) -> Dict[str, Any]:
    """Download, validate, and store historical klines.

    Args:
        symbol: Trading pair (e.g. ``"BTCUSDT"``).
        interval: Kline interval (default ``"1h"``).
        market: ``"futures_um"`` or ``"spot"``.
        min_years: Minimum history depth in years.
        cache_root: Root directory for all cached data.
        mode: ``"auto"`` (default), ``"backfill"``, or ``"update"``.
        force: Force re-download of all archives.
        max_concurrency: Max parallel downloads.

    Returns:
        The manifest dict written to disk.

    Raises:
        ValueError: If validation fails (training must not use bad data).
    """
    cache_dir = Path(cache_root)
    pq_root = parquet_root(str(cache_dir), market, symbol, interval)
    manifest_path = pq_root.parent / "manifest.json"
    existing = load_manifest(manifest_path)

    # Decide mode
    if mode == "auto":
        mode = "update" if existing is not None else "backfill"

    logger.info(
        "ensure_kline_store: %s %s mode=%s min_years=%.1f",
        symbol, interval, mode, min_years,
    )

    # --- Date boundaries -----------------------------------------------------

    today = datetime.now(timezone.utc).date()

    if mode == "backfill":
        # Monthly range covering min_years
        start_month = today.month
        start_year = today.year - int(min_years)
        # Adjust for fractional years — ceil ensures we fetch enough history
        import math
        frac_months = math.ceil((min_years - int(min_years)) * 12)
        start_month -= frac_months
        while start_month < 1:
            start_month += 12
            start_year -= 1

        end_m = _previous_complete_month()
        dates = monthly_dates(start_year, start_month, end_m.year, end_m.month)
        granularity = "monthly"

        logger.info(
            "Backfill: %d monthly archives (%d-%02d to %d-%02d)",
            len(dates), start_year, start_month, end_m.year, end_m.month,
        )

    else:  # update
        last_ms = get_last_open_time_ms(existing)
        if last_ms is None:
            raise ValueError(
                "Cannot update: manifest has no end_ms. Run backfill first."
            )
        # Start the day after the last known bar
        from datetime import timedelta
        last_dt = datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc).date()
        start = last_dt + timedelta(days=1)
        end = _yesterday()
        dates = daily_dates(start, end)
        granularity = "daily"

        logger.info(
            "Update: %d daily archives (%s to %s)",
            len(dates), start, end,
        )

    if not dates:
        logger.info("No new archives to download")
        return existing or {}

    # --- Download ------------------------------------------------------------

    zip_paths = await download_kline_batch(
        symbol=symbol,
        interval=interval,
        dates=dates,
        market=market,
        granularity=granularity,
        cache_dir=cache_dir,
        max_concurrency=max_concurrency,
        force=force,
    )

    logger.info("Downloaded %d / %d archives", len(zip_paths), len(dates))

    if not zip_paths:
        logger.warning("No archives downloaded — nothing to ingest")
        return existing or {}

    # --- Ingest ---------------------------------------------------------------

    df, ingest_report = ingest_zips(zip_paths)
    logger.info(
        "Ingested %d rows (raw=%d, deduped=%d removed)",
        ingest_report["final_rows"],
        ingest_report["total_raw_rows"],
        ingest_report["duplicates_removed"],
    )

    if df.empty:
        logger.warning("Ingestion produced empty DataFrame")
        return existing or {}

    # --- Daily bridge after backfill ------------------------------------------
    # Monthly archives end at the last complete month.  Bridge the gap to
    # yesterday with daily archives so the store is as fresh as possible.
    if mode == "backfill" and not df.empty:
        from datetime import timedelta as _td
        from datetime import date as _date_type
        last_bar_dt = cast(pd.Series, df["open_time"]).max()
        bridge_dates = []
        if not bool(pd.isna(last_bar_dt)):
            if hasattr(last_bar_dt, "date"):
                _bridge_date = last_bar_dt.date() + _td(days=1)
            else:
                _bridge_date = pd.Timestamp(str(last_bar_dt), tz="UTC").date() + _td(days=1)
            bridge_start = cast(_date_type, _bridge_date)
            bridge_end = _yesterday()
            bridge_dates = daily_dates(bridge_start, bridge_end)
        if bridge_dates:
            logger.info(
                "Daily bridge: %d daily archives (%s to %s)",
                len(bridge_dates), bridge_start, bridge_end,
            )
            bridge_zips = await download_kline_batch(
                symbol=symbol,
                interval=interval,
                dates=bridge_dates,
                market=market,
                granularity="daily",
                cache_dir=cache_dir,
                max_concurrency=max_concurrency,
                force=force,
            )
            if bridge_zips:
                bridge_df, bridge_report = ingest_zips(bridge_zips)
                if not bridge_df.empty:
                    pre_dedup = len(pd.concat([df, bridge_df], ignore_index=True))
                    df = (
                        pd.concat([df, bridge_df], ignore_index=True)
                        .sort_values("open_time")
                        .drop_duplicates(subset="open_time", keep="last")
                        .reset_index(drop=True)
                    )
                    dupes_removed = pre_dedup - len(df)
                    if dupes_removed > 0:
                        logger.info("Daily bridge merge: removed %d duplicate bars", dupes_removed)
                    logger.info(
                        "Daily bridge added %d bars (total: %d)",
                        bridge_report["final_rows"], len(df),
                    )

    # If updating, merge with existing data
    if mode == "update" and pq_root.exists():
        existing_df = load_parquet_range(pq_root)
        if not existing_df.empty:
            df = (
                pd.concat([existing_df, df], ignore_index=True)
                .sort_values("open_time")
                .drop_duplicates(subset="open_time", keep="last")
                .reset_index(drop=True)
            )
            logger.info("Merged with existing store: %d total rows", len(df))

    # --- Validate -------------------------------------------------------------

    # Scale min_rows to requested history (interval-aware)
    _bars_per_hour = {"1h": 1, "15m": 4, "5m": 12, "1m": 60}.get(interval, 1)
    min_rows = int(min_years * 365.25 * 24 * _bars_per_hour)
    result = validate_kline_store(df, interval=interval, min_rows=min_rows)

    logger.info("Validation: %s", result.summary())

    if not result.passed:
        raise ValueError(
            f"Kline store validation failed for {symbol}: {result.summary()}"
        )

    # --- Write Parquet --------------------------------------------------------

    write_partitioned(df, pq_root)

    # --- Manifest -------------------------------------------------------------

    ts_min = df["open_time"].min()
    ts_max = df["open_time"].max()
    if bool(pd.isna(ts_min)) or bool(pd.isna(ts_max)):
        logger.warning("open_time min/max is NaT after processing — cannot build manifest")
        return existing or {}

    start_ms = int(ts_min.timestamp() * 1000)
    end_ms = int(ts_max.timestamp() * 1000)

    manifest = create_manifest(
        symbol=symbol,
        interval=interval,
        market=market,
        parquet_root=str(pq_root),
        start_ms=start_ms,
        end_ms=end_ms,
        bar_count=len(df),
        files_downloaded=len(zip_paths),
        gap_report=result.metrics.get("time_cadence"),
        dedupe_report=ingest_report,
        validation_summary={
            "passed": result.passed,
            "checks": result.checks,
        },
    )

    save_manifest(manifest, manifest_path)
    logger.info(
        "Manifest saved: %d bars, %s → %s",
        len(df),
        df["open_time"].min(),
        df["open_time"].max(),
    )

    return manifest
