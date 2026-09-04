"""
Parquet read/write with Hive-style partitioning for kline stores.

Layout::

    parquet_root / year=YYYY / month=MM / part-000.parquet

Uses PyArrow for writing (Hive-partitioned dataset) and Pandas/PyArrow
for reading with partition pruning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, cast

import pandas as pd

logger = logging.getLogger(__name__)


def parquet_root(
    cache_root: str,
    market: str,
    symbol: str,
    interval: str,
) -> Path:
    """Canonical path for a symbol's Parquet store.

    Example:
        ``data/cache/klines/futures_um/BTCUSDT/1h/parquet``
    """
    return Path(cache_root) / market / symbol.upper() / interval / "parquet"


def write_partitioned(df: pd.DataFrame, root: Path) -> None:
    """Write *df* as a Hive-partitioned Parquet dataset under *root*.

    Adds ``year`` and ``month`` columns derived from ``open_time`` for
    partitioning, then writes via :func:`pyarrow.dataset.write_dataset`.
    """
    import pyarrow as pa
    import pyarrow.dataset as pds

    if df.empty:
        logger.warning("write_partitioned: empty DataFrame, nothing to write")
        return

    df = df.copy()

    # Ensure timestamp columns are UTC-aware before writing.
    # Production data (via normalize_dataframe) is already UTC, but
    # defensively localize naive timestamps so the Parquet schema
    # is always timestamp[ns, tz=UTC] — matching load_parquet_range filters.
    for col in ("open_time", "close_time"):
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce")
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize("UTC")
            df[col] = ts

    # Derive partition columns
    open_time = df["open_time"]
    df["year"] = open_time.dt.year.astype(str)
    df["month"] = open_time.dt.strftime("%m")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=False)
    pds.write_dataset(
        table,
        base_dir=str(root),
        format="parquet",
        partitioning=pds.partitioning(
            pa.schema([
                pa.field("year", pa.string()),
                pa.field("month", pa.string()),
            ]),
            flavor="hive",
        ),
        existing_data_behavior="overwrite_or_ignore",
    )

    logger.info(
        "Wrote %d rows to %s (%d partitions)",
        len(df),
        root,
        df.groupby(["year", "month"]).ngroups,
    )


def load_parquet_range(
    root: Path,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
) -> pd.DataFrame:
    """Load klines from a Hive-partitioned Parquet store.

    Performs partition-level pruning when *start_ms* / *end_ms* are given,
    then applies precise row-level filtering on ``open_time``.

    Returns:
        DataFrame matching the canonical kline schema (without year/month).
    """
    import pyarrow.dataset as pds

    root = Path(root)
    if not root.exists():
        return pd.DataFrame()

    import pyarrow.compute as pc

    dataset = pds.dataset(
        str(root),
        format="parquet",
        partitioning="hive",
    )

    # Build PyArrow filter for predicate pushdown (avoids loading entire dataset)
    pa_filter = None
    if start_ms is not None:
        start_dt = pd.to_datetime(start_ms, unit="ms", utc=True)
        f = pc.field("open_time") >= pc.scalar(start_dt)
        pa_filter = f if pa_filter is None else pa_filter & f
    if end_ms is not None:
        end_dt = pd.to_datetime(end_ms, unit="ms", utc=True)
        f = pc.field("open_time") <= pc.scalar(end_dt)
        pa_filter = f if pa_filter is None else pa_filter & f

    # Exclude partition columns from the read
    all_cols = dataset.schema.names
    columns = [c for c in all_cols if c not in ("year", "month")]

    table = dataset.to_table(filter=pa_filter, columns=columns)
    df = table.to_pandas()

    if df.empty:
        return df

    # Ensure open_time is datetime
    df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce", utc=True)

    df = (
        df.sort_values("open_time")
        .drop_duplicates(subset="open_time", keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    return df


def get_latest_open_time_ms(root: Path) -> Optional[int]:
    """Return the latest ``open_time`` in the store as epoch milliseconds."""
    df = load_parquet_range(root)
    if df.empty:
        return None
    latest = cast(Any, df["open_time"].max())
    if latest is None or bool(pd.isna(latest)):
        return None
    latest_ts = pd.to_datetime(latest, utc=True, errors="coerce")
    if not isinstance(latest_ts, pd.Timestamp) or pd.isna(latest_ts):
        return None
    return int(latest_ts.timestamp() * 1000)
