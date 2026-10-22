"""Original premium-index evidence and causal, explicitly bounded replay.

This module never obtains historical forecasts from settlement history.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4
from zoneinfo import ZoneInfo

from neutralgrid.core.config import get_config
from neutralgrid.data.funding_rate import expected_funding_carry_next_hours

SCHEMA = "binance_premium_index_observation_v1"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Funding timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("Funding event timestamps must be integer milliseconds")
    result = int(value)
    if result <= 0:
        raise ValueError("Funding event timestamps must be positive")
    return result


def premium_funding_info(
    payload: Mapping[str, Any], *, symbol: str, cutoff: datetime,
) -> dict[str, Any]:
    """Validate one original response at a cutoff; do not clamp stale times."""
    cutoff = _utc(cutoff)
    if payload.get("symbol") != symbol or not re.fullmatch(r"[A-Z0-9]+", symbol):
        raise ValueError("Funding response symbol mismatch")
    observed_ms = _integer(payload.get("time"))
    next_ms = _integer(payload.get("nextFundingTime"))
    cutoff_ms = int(cutoff.timestamp() * 1000)
    raw_rate = payload.get("lastFundingRate")
    if isinstance(raw_rate, bool) or not isinstance(raw_rate, (str, int, float)):
        raise ValueError("Funding rate is missing or invalid")
    rate = float(raw_rate)
    if not math.isfinite(rate) or observed_ms > cutoff_ms or next_ms < cutoff_ms:
        raise ValueError("Funding observation is future-dated, stale or non-finite")
    return {
        "symbol": symbol,
        "funding_rate_pct": rate * 100.0,
        "funding_interval_hours": (next_ms - cutoff_ms) / 3_600_000.0,
        "next_funding_time": next_ms,
        "observed_at_ms": observed_ms,
        "fetched_at": cutoff.isoformat(),
        "mark_price": payload.get("markPrice"),
        "index_price": payload.get("indexPrice"),
    }


def _payload_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def premium_rate_from_market_data(
    market_data: Mapping[str, Any], *, symbol: str, cutoff: datetime,
) -> float | None:
    """Enrichment rate in decimal units; settlement history is never fallback."""
    try:
        info = premium_funding_info(market_data["premium_index"], symbol=symbol, cutoff=cutoff)
        return float(info["funding_rate_pct"]) / 100.0
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return None


def persist_funding_observation(
    payload: Mapping[str, Any], *, symbol: str, received_at: datetime,
    live_root: Path | None = None, strategy_id: str | None = None,
) -> Path:
    """Write a new immutable receipt; never overwrite an earlier capture."""
    received_at = _utc(received_at)
    premium_funding_info(payload, symbol=symbol, cutoff=received_at)
    root = live_root if live_root is not None else get_config().base_dir / "Live"
    day = received_at.astimezone(ZoneInfo("America/Lima")).strftime("%Y-%m-%d")
    folder = root / day / symbol / "funding_observations"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"premium_{received_at.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex}.json"
    record = {
        "schema_version": SCHEMA,
        "source": "/fapi/v1/premiumIndex",
        "symbol": symbol,
        "strategy_id": strategy_id,
        "received_at_utc": received_at.isoformat(),
        "payload_sha256": _payload_hash(payload),
        "payload": dict(payload),
    }
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    return path


async def capture_funding_observation(
    client: Any, symbol: str, *, live_root: Path | None = None,
    strategy_id: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Capture receipt time immediately after the request, before other work."""
    payload = await client.get_premium_index(symbol)
    received_at = datetime.now(timezone.utc)
    if not isinstance(payload, dict):
        raise ValueError("premiumIndex did not return a symbol object")
    path = persist_funding_observation(
        payload, symbol=symbol, received_at=received_at,
        live_root=live_root, strategy_id=strategy_id,
    )
    return premium_funding_info(payload, symbol=symbol, cutoff=received_at), path


def replay_funding_observation(
    path: Path, *, symbol: str, cutoff: datetime, max_age_seconds: float,
    strategy_id: str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Replay an explicitly chosen receipt under an explicit freshness policy."""
    cutoff = _utc(cutoff)
    if isinstance(max_age_seconds, bool) or not math.isfinite(max_age_seconds) or max_age_seconds < 0:
        raise ValueError("An explicit finite nonnegative funding maximum age is required")
    receipt_bytes = path.read_bytes()
    record = json.loads(receipt_bytes)
    if record.get("schema_version") != SCHEMA or record.get("source") != "/fapi/v1/premiumIndex":
        raise ValueError("Unsupported funding observation source/schema")
    if record.get("symbol") != symbol:
        raise ValueError("Funding receipt symbol mismatch")
    if record.get("strategy_id") is not None and str(record["strategy_id"]) != str(strategy_id):
        raise ValueError("Funding receipt strategy mismatch")
    payload = record["payload"]
    if _payload_hash(payload) != record.get("payload_sha256"):
        raise ValueError("Funding payload hash mismatch")
    received_at = _utc(datetime.fromisoformat(record["received_at_utc"]))
    # Both exchange time and actual receipt must precede the entry cutoff.
    premium_funding_info(payload, symbol=symbol, cutoff=received_at)
    info = premium_funding_info(payload, symbol=symbol, cutoff=cutoff)
    age = cutoff.timestamp() - info["observed_at_ms"] / 1000.0
    if received_at > cutoff or age > max_age_seconds:
        raise ValueError("Funding observation was unavailable or too old at cutoff")
    value = expected_funding_carry_next_hours(info)
    if value is None:
        raise ValueError("Funding observation cannot define the requested forecast")
    provenance = {
        "source_path": str(path.resolve()), "source_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "observed_at_ms": info["observed_at_ms"], "received_at_utc": received_at.isoformat(),
        "cutoff_utc": cutoff.isoformat(), "max_age_seconds": max_age_seconds,
        "next_funding_time_ms": info["next_funding_time"], "units": "percentage_points",
        "funding_carry_expected_next_7h": value,
    }
    return value, provenance
