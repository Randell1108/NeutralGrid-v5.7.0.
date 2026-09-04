"""
Manifest create / load / save for Binance Vision kline stores.

Each kline store (``parquet_root``) gets a sibling ``manifest.json`` that
records download provenance, bar count, gap reports, and the last
``open_time`` so incremental updates know where to resume.

Writes are atomic (write to tmp then rename) to avoid corrupted manifests
on crash.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA_VERSION = "1.0.0"


def create_manifest(
    symbol: str,
    interval: str,
    market: str,
    parquet_root: str,
    start_ms: int,
    end_ms: int,
    bar_count: int,
    files_downloaded: int,
    gap_report: Optional[Dict[str, Any]] = None,
    dedupe_report: Optional[Dict[str, Any]] = None,
    validation_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a manifest dict with full provenance."""
    return {
        "schema_version": SCHEMA_VERSION,
        "symbol": symbol.upper(),
        "interval": interval,
        "market": market,
        "parquet_root": str(parquet_root),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "bar_count": bar_count,
        "files_downloaded": files_downloaded,
        "gap_report": gap_report or {},
        "dedupe_report": dedupe_report or {},
        "validation_summary": validation_summary or {},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_manifest(manifest: Dict[str, Any], path: Path) -> None:
    """Atomically write *manifest* to *path* (tmp + rename)."""
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=".manifest_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        # On Windows, target must not exist for os.replace
        os.replace(tmp, str(path))
    except BaseException:
        # Clean up temp file on failure
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    """Load manifest from *path*, returning ``None`` if absent or corrupt."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_last_open_time_ms(manifest: Optional[Dict[str, Any]]) -> Optional[int]:
    """Extract end_ms from a manifest (the last open_time in the store)."""
    if manifest is None:
        return None
    return manifest.get("end_ms")
