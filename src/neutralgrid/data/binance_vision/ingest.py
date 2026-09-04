"""
Ingest Binance Vision ZIP archives into canonical DataFrames.

Each ZIP contains a single headerless CSV with the 12-column Binance kline
format.  Column names MUST match ``market_data.py`` exactly:

    open_time, open, high, low, close, volume, close_time,
    quote_volume, trades, taker_buy_base, taker_buy_quote, ignore
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

NUMERIC_COLS = ["open", "high", "low", "close", "volume", "quote_volume"]
TIMESTAMP_COLS = ["open_time", "close_time"]


# --- Helpers -----------------------------------------------------------------

def parse_zip(zip_path: Path) -> pd.DataFrame:
    """Extract the single CSV inside a Binance Vision ZIP and parse it.

    Handles both formats:
    - Pre-2022: headerless 12-column CSV
    - 2022+: CSV with header row (``open_time,open,high,...``)

    Returns:
        DataFrame with raw string/numeric values (no type conversion yet).

    Raises:
        ValueError: If the ZIP does not contain exactly one CSV.
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(
                f"Expected 1 CSV in {zip_path.name}, found {len(csv_names)}: {csv_names}"
            )
        with zf.open(csv_names[0]) as csv_file:
            raw_bytes = csv_file.read()

    # Detect header: if the first field is non-numeric, the CSV has a header
    first_line = raw_bytes.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    first_field = first_line.split(",", 1)[0].strip()
    try:
        int(first_field)
        has_header = False
    except (ValueError, TypeError):
        has_header = True

    if has_header:
        df = pd.read_csv(io.BytesIO(raw_bytes), header=0)
        # Rename to our canonical column names (Binance uses 'count'
        # instead of 'trades' and 'taker_buy_volume' instead of
        # 'taker_buy_base' in newer files)
        df.columns = KLINE_COLUMNS
    else:
        df = pd.read_csv(io.BytesIO(raw_bytes), header=None, names=KLINE_COLUMNS)

    expected = len(KLINE_COLUMNS)
    actual = len(df.columns)
    if actual != expected:
        raise ValueError(
            f"{zip_path.name}: expected {expected} columns, got {actual}"
        )

    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert types to match ``market_data.py:116-175``.

    * open_time, close_time → ``pd.to_datetime(col, unit="ms")``
    * OHLCV + quote_volume  → ``pd.to_numeric``
    * Sort by open_time, drop exact duplicates, enforce monotonicity.
    """
    df = df.copy()

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in TIMESTAMP_COLS:
        df[col] = pd.to_datetime(df[col], unit="ms", utc=True)

    df = df.sort_values("open_time").reset_index(drop=True)
    # Policy: latest data wins — keep="last" for consistency across all data paths
    df = df.drop_duplicates(subset="open_time", keep="last").reset_index(drop=True)

    return df


# --- Public API --------------------------------------------------------------

def ingest_zips(
    zip_paths: List[Path],
    sort: bool = True,
    dedupe: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Parse and concatenate multiple ZIP archives into one canonical DataFrame.

    Args:
        zip_paths: Ordered list of ZIP file paths.
        sort: Sort by open_time after concatenation.
        dedupe: Drop duplicate open_time rows.

    Returns:
        (df, report) where *report* contains ingestion statistics.
    """
    frames: List[pd.DataFrame] = []
    total_raw = 0
    errors: List[str] = []

    for zp in zip_paths:
        try:
            raw = parse_zip(zp)
            total_raw += len(raw)
            frames.append(raw)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", zp.name, e)
            errors.append(f"{zp.name}: {e}")

    if not frames:
        empty = pd.DataFrame(columns=pd.Index(KLINE_COLUMNS))
        return empty, {
            "total_raw_rows": 0,
            "final_rows": 0,
            "duplicates_removed": 0,
            "parse_errors": errors,
        }

    combined = pd.concat(frames, ignore_index=True)
    pre_normalize = len(combined)

    # Normalize types (includes sort + dedupe on open_time)
    combined = normalize_dataframe(combined)

    if sort:
        combined = combined.sort_values("open_time").reset_index(drop=True)
    if dedupe:
        # Policy: latest data wins — keep="last" for consistency across all data paths
        combined = combined.drop_duplicates(subset="open_time", keep="last").reset_index(drop=True)

    report = {
        "total_raw_rows": total_raw,
        "final_rows": len(combined),
        "duplicates_removed": pre_normalize - len(combined),
        "parse_errors": errors,
    }

    return combined, report
