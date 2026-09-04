"""
Folder-based CSV file scanner and classifier.

Recursively finds .csv files and classifies them as 'orders' or 'trades'
by inspecting header columns.  Files that match neither schema are logged
and skipped.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Column sets that MUST be present for classification
_ORDERS_REQUIRED = {"Order No", "Time(UTC)", "Update Time", "Status", "Symbol"}
_TRADES_REQUIRED = {"Order Id", "Time(UTC)", "Symbol"}


@dataclass
class ScanResult:
    """Result of scanning an input directory for CSV exports."""
    order_files: list[Path]
    trade_files: list[Path]
    unknown_files: list[Path]


def _read_header(path: Path) -> set[str]:
    """Read the first line of a CSV and return column names as a set."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return set()
            return {col.strip() for col in header}
    except Exception as exc:
        logger.warning("Could not read header from %s: %s", path, exc)
        return set()


def scan_directory(input_dir: Path) -> ScanResult:
    """Scan *input_dir* recursively for .csv files and classify each.

    Returns a ``ScanResult`` with three lists: order files, trade files,
    and unknown (unclassified) files.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    order_files: list[Path] = []
    trade_files: list[Path] = []
    unknown_files: list[Path] = []

    csv_paths = sorted(input_dir.rglob("*.csv"))
    logger.info("Found %d CSV file(s) in %s", len(csv_paths), input_dir)

    for path in csv_paths:
        columns = _read_header(path)
        if not columns:
            logger.warning("Empty or unreadable CSV: %s", path)
            unknown_files.append(path)
            continue

        if _ORDERS_REQUIRED.issubset(columns):
            order_files.append(path)
            logger.info("  [orders] %s", path.name)
        elif _TRADES_REQUIRED.issubset(columns):
            trade_files.append(path)
            logger.info("  [trades] %s", path.name)
        else:
            unknown_files.append(path)
            logger.info("  [unknown] %s  (cols: %s)", path.name,
                        ", ".join(sorted(columns)[:5]))

    logger.info(
        "Classification complete: %d order file(s), %d trade file(s), "
        "%d unknown",
        len(order_files), len(trade_files), len(unknown_files),
    )
    return ScanResult(
        order_files=order_files,
        trade_files=trade_files,
        unknown_files=unknown_files,
    )
