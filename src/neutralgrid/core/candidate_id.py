"""
Stable candidate_id generation with config-hash uniqueness.

A candidate_id uniquely identifies a (symbol, scan_time, grid_config) tuple.
Format: ``{SYMBOL}_{YYYYMMDD_HHMMSS}_{hash8}``

The 8-character hash is derived from the grid configuration fields that
define the trading setup.  Two scans of the same symbol at the same time
but with different grid parameters will produce different candidate_ids.

Usage::

    from neutralgrid.core.candidate_id import make_candidate_id

    cid = make_candidate_id(
        symbol="BTCUSDT",
        scan_ts="20260227_143000",
        grid_lower=40000.0,
        grid_upper=42000.0,
        num_grids=30,
        leverage=10,
    )
    # -> "BTCUSDT_20260227_143000_a1b2c3d4"
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Optional


# Fields included in the config hash (order matters — keep sorted by key name)
_CONFIG_HASH_KEYS = (
    "grid_lower",
    "grid_upper",
    "leverage",
    "num_grids",
)


def _config_hash(fields: Dict[str, Any], length: int = 8) -> str:
    """Compute a deterministic short hash from grid config fields.

    Parameters
    ----------
    fields : dict
        Keys must be a subset of ``_CONFIG_HASH_KEYS``.
    length : int
        Number of hex characters to return (max 32).

    Returns
    -------
    str
        Lowercase hex digest truncated to *length* characters.
    """
    # Canonicalise: sort by key, round floats to 8 decimal places
    parts: list[str] = []
    for key in _CONFIG_HASH_KEYS:
        val = fields.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            parts.append(f"{key}=")
        elif isinstance(val, float):
            parts.append(f"{key}={val:.8f}")
        else:
            parts.append(f"{key}={val}")

    payload = "|".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return digest[:length]


def make_candidate_id(
    symbol: str,
    scan_ts: str,
    *,
    grid_lower: Optional[float] = None,
    grid_upper: Optional[float] = None,
    num_grids: Optional[int] = None,
    leverage: Optional[int] = None,
    config_fields: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a stable, unique ``candidate_id``.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g. ``"BTCUSDT"``).
    scan_ts : str
        Scan timestamp in ``YYYYMMDD_HHMMSS`` format.
    grid_lower, grid_upper, num_grids, leverage
        Individual config fields.  Ignored when *config_fields* is given.
    config_fields : dict, optional
        Full config dict — only ``_CONFIG_HASH_KEYS`` are used.

    Returns
    -------
    str
        ``"{SYMBOL}_{scan_ts}_{hash8}"``
    """
    if config_fields is not None:
        fields = config_fields
    else:
        fields = {
            "grid_lower": grid_lower,
            "grid_upper": grid_upper,
            "num_grids": num_grids,
            "leverage": leverage,
        }

    h = _config_hash(fields)
    return f"{symbol}_{scan_ts}_{h}"


def make_candidate_id_from_row(row: Dict[str, Any], scan_ts: str) -> str:
    """Convenience: build candidate_id from a scanner DataFrame row.

    Parameters
    ----------
    row : dict-like
        Must contain ``"symbol"`` and the grid config columns.
    scan_ts : str
        Scan timestamp in ``YYYYMMDD_HHMMSS`` format.
    """
    symbol = str(row.get("symbol", ""))
    fields = {k: row.get(k) for k in _CONFIG_HASH_KEYS}
    return make_candidate_id(symbol, scan_ts, config_fields=fields)


def extract_parts(candidate_id: str) -> Dict[str, str]:
    """Parse a candidate_id into its components.

    Returns
    -------
    dict
        ``{"symbol": ..., "scan_ts": ..., "config_hash": ...}``
        If the id is in legacy format (no hash), ``config_hash`` is ``""``.
    """
    parts = candidate_id.split("_")
    if len(parts) >= 4:
        # SYMBOL_YYYYMMDD_HHMMSS_hash
        symbol = parts[0]
        scan_ts = f"{parts[1]}_{parts[2]}"
        config_hash = parts[3]
    elif len(parts) == 3:
        # Legacy: SYMBOL_YYYYMMDD_HHMMSS (no hash)
        symbol = parts[0]
        scan_ts = f"{parts[1]}_{parts[2]}"
        config_hash = ""
    else:
        symbol = candidate_id
        scan_ts = ""
        config_hash = ""

    return {
        "symbol": symbol,
        "scan_ts": scan_ts,
        "config_hash": config_hash,
    }
