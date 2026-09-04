"""Strategy-scoped private execution events for live telemetry evaluation.

The scanner consumes an append-only canonical JSONL stream produced by a
separate authenticated collector or a reviewed manual-export ingestion.  This
module deliberately does not infer bot ownership from symbol and time alone:
every record and manifest must carry the exact strategy ID of the active bot.

Event completeness is declared by the producer and transported as evidence.
REST order history is normally a latest-state snapshot, not an event-complete
order-update stream, so callers must not label it ``event_complete``.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast


PRIVATE_EVENT_SCHEMA_VERSION = "neutralgrid_private_event_v1"
PRIVATE_EVENT_MANIFEST_SCHEMA_VERSION = "neutralgrid_private_event_manifest_v1"
_EVENT_TYPES = {"order_update", "trade_fill", "income", "account_update"}
_COMPLETENESS = {"event_complete", "history_complete", "snapshot_only", "unknown"}


class PrivateEventError(RuntimeError):
    """Private-event evidence is missing, stale, ambiguous, or invalid."""


@dataclass(frozen=True)
class PrivateEventStreamRef:
    event_path: Path
    symbol: str
    strategy_id: str
    run_id: str | None = None
    manifest_path: Path | None = None
    max_age_seconds: float = 600.0
    history_window_seconds: float = 86400.0


@dataclass(frozen=True)
class PrivateEventEvidence:
    """Bounded observational summary of exact-strategy private events."""

    source: str
    observed_at_utc: datetime
    age_seconds: float
    run_id: str
    symbol: str
    strategy_id: str
    capture_mode: str
    event_completeness: str
    source_scopes: tuple[str, ...]
    manifest_total_records: int
    records_in_window: int
    history_coverage_seconds: float
    order_update_count: int
    order_status_counts: dict[str, int]
    trade_fill_count: int
    unique_trade_order_count: int
    maker_fill_count: int
    taker_fill_count: int
    unknown_liquidity_fill_count: int
    buy_fill_notional_usdt: float
    sell_fill_notional_usdt: float
    realized_pnl_usdt: float
    commission_usdt: float
    funding_fee_usdt: float
    other_income_usdt: float
    duplicate_records_dropped: int
    rejected_records: int
    first_event_at_utc: datetime | None
    last_event_at_utc: datetime | None
    last_fill_at_utc: datetime | None
    scope_note: str = (
        "Exact symbol/strategy private-event evidence. Completeness is producer-"
        "declared and preserved; it is observational and does not alter verdicts."
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "observed_at_utc",
            "first_event_at_utc",
            "last_event_at_utc",
            "last_fill_at_utc",
        ):
            value = payload[key]
            if isinstance(value, datetime):
                payload[key] = value.astimezone(timezone.utc).isoformat()
        payload["source_scopes"] = list(self.source_scopes)
        return payload


def private_event_dedup_key(record: Mapping[str, Any]) -> tuple[object, ...]:
    """Return the stable exchange-identity key for a canonical record."""

    event_type = str(record.get("event_type", ""))
    if event_type == "trade_fill":
        return (
            event_type,
            str(record.get("symbol", "")).upper(),
            str(record.get("trade_id", "")),
        )
    if event_type == "income":
        return (
            event_type,
            str(record.get("symbol", "")).upper(),
            str(record.get("transaction_id", "")),
            str(record.get("income_type", "")).upper(),
            str(record.get("trade_id", "")),
        )
    if event_type == "order_update":
        return (
            event_type,
            str(record.get("symbol", "")).upper(),
            str(record.get("order_id", "")),
            str(record.get("event_time_utc", "")),
            str(record.get("status", "")).upper(),
            str(record.get("executed_qty", "")),
        )
    return (
        event_type,
        str(record.get("symbol", "")).upper(),
        str(record.get("event_id", "")),
    )


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise PrivateEventError(f"{field} must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PrivateEventError(f"{field} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise PrivateEventError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _nonnegative_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrivateEventError(f"{key} must be a non-negative integer")
    return value


def _finite_number(
    data: Mapping[str, Any],
    key: str,
    *,
    required: bool = False,
    positive: bool = False,
) -> float | None:
    value = data.get(key)
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrivateEventError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise PrivateEventError(f"{key} must be {qualifier}")
    return number


def _read_tail_records(path: Path, *, max_records: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PrivateEventError(f"private event stream does not exist: {path}")
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        buffer = b""
        while position > 0 and buffer.count(b"\n") <= max_records:
            block = min(65536, position)
            position -= block
            handle.seek(position)
            buffer = handle.read(block) + buffer
    lines = [line for line in buffer.splitlines() if line.strip()][-max_records:]
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            decoded = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PrivateEventError(f"invalid JSONL tail in {path}") from exc
        if not isinstance(decoded, dict):
            raise PrivateEventError(f"private JSONL record is not an object in {path}")
        records.append(cast(dict[str, Any], decoded))
    return records


def _validate_manifest(
    ref: PrivateEventStreamRef,
    *,
    now: datetime,
) -> dict[str, Any]:
    if ref.manifest_path is None:
        raise PrivateEventError("private event manifest_path is required")
    try:
        decoded = json.loads(ref.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivateEventError(
            f"cannot read private event manifest {ref.manifest_path}: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise PrivateEventError("private event manifest is not an object")
    if decoded.get("schema_version") != PRIVATE_EVENT_MANIFEST_SCHEMA_VERSION:
        raise PrivateEventError("unsupported private event manifest schema")
    symbol = str(decoded.get("symbol", "")).upper()
    strategy_id = str(decoded.get("strategy_id", ""))
    if symbol != ref.symbol.upper() or strategy_id != ref.strategy_id:
        raise PrivateEventError(
            "private event manifest identity mismatch: "
            f"expected={ref.symbol.upper()}/{ref.strategy_id}, "
            f"got={symbol}/{strategy_id or 'missing'}"
        )
    run_id = decoded.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise PrivateEventError("private event manifest run_id is missing")
    if ref.run_id is not None and run_id != ref.run_id:
        raise PrivateEventError(
            f"private event run_id mismatch: expected {ref.run_id}, got {run_id}"
        )
    if decoded.get("status") not in {"running", "complete"}:
        raise PrivateEventError(
            f"private event collector status is not usable: {decoded.get('status')!r}"
        )
    updated_at = _parse_utc(decoded.get("updated_at_utc"), field="manifest.updated_at_utc")
    age = (now.astimezone(timezone.utc) - updated_at).total_seconds()
    if age < -5 or age > ref.max_age_seconds:
        raise PrivateEventError(
            f"private event manifest heartbeat is stale or future-dated: age={age:.1f}s"
        )
    completeness = decoded.get("event_completeness")
    if completeness not in _COMPLETENESS:
        raise PrivateEventError(
            f"manifest.event_completeness must be one of {sorted(_COMPLETENESS)}"
        )
    scopes = decoded.get("source_scopes")
    if not isinstance(scopes, list) or not all(
        isinstance(item, str) and item.strip() for item in scopes
    ):
        raise PrivateEventError("manifest.source_scopes must be a list of strings")
    _nonnegative_int(decoded, "total_records")
    _nonnegative_int(decoded, "duplicate_records_dropped")
    _nonnegative_int(decoded, "rejected_records")
    manifest_event_path = decoded.get("event_path")
    if manifest_event_path is not None:
        if not isinstance(manifest_event_path, str) or not manifest_event_path.strip():
            raise PrivateEventError("manifest.event_path must be a non-empty string")
        declared_path = Path(manifest_event_path)
        if not declared_path.is_absolute():
            declared_path = ref.manifest_path.parent / declared_path
        if declared_path.resolve() != ref.event_path.resolve():
            raise PrivateEventError(
                "private event path mismatch between stream reference and manifest"
            )
    validated = dict(decoded)
    validated["updated_at"] = updated_at
    return validated


def _validate_record(
    record: Mapping[str, Any],
    ref: PrivateEventStreamRef,
    *,
    run_id: str,
) -> dict[str, Any]:
    if record.get("schema_version") != PRIVATE_EVENT_SCHEMA_VERSION:
        raise PrivateEventError("unsupported private event schema")
    event_type = record.get("event_type")
    if event_type not in _EVENT_TYPES:
        raise PrivateEventError(f"unsupported private event type: {event_type!r}")
    symbol = str(record.get("symbol", "")).upper()
    strategy_id = str(record.get("strategy_id", ""))
    if symbol != ref.symbol.upper() or strategy_id != ref.strategy_id:
        raise PrivateEventError(
            "private event record identity mismatch: "
            f"expected={ref.symbol.upper()}/{ref.strategy_id}, "
            f"got={symbol}/{strategy_id or 'missing'}"
        )
    if record.get("run_id") != run_id:
        raise PrivateEventError("private event record run_id mismatch")
    event_at = _parse_utc(record.get("event_time_utc"), field="event_time_utc")
    validated = dict(record)
    validated["event_at"] = event_at
    if event_type == "trade_fill":
        if not str(record.get("trade_id", "")) or not str(record.get("order_id", "")):
            raise PrivateEventError("trade_fill requires trade_id and order_id")
        side = str(record.get("side", "")).upper()
        if side not in {"BUY", "SELL"}:
            raise PrivateEventError("trade_fill.side must be BUY or SELL")
        validated["price_value"] = _finite_number(record, "price", required=True, positive=True)
        validated["qty_value"] = _finite_number(record, "qty", required=True, positive=True)
        _finite_number(record, "commission_usdt")
        _finite_number(record, "realized_pnl_usdt")
        maker = record.get("maker")
        if maker is not None and not isinstance(maker, bool):
            raise PrivateEventError("trade_fill.maker must be boolean or null")
    elif event_type == "income":
        if not str(record.get("transaction_id", "")):
            raise PrivateEventError("income requires transaction_id")
        if not str(record.get("income_type", "")).strip():
            raise PrivateEventError("income requires income_type")
        _finite_number(record, "income_usdt", required=True)
    elif event_type == "order_update":
        if not str(record.get("order_id", "")):
            raise PrivateEventError("order_update requires order_id")
        if not str(record.get("status", "")).strip():
            raise PrivateEventError("order_update requires status")
        _finite_number(record, "executed_qty")
    elif not str(record.get("event_id", "")):
        raise PrivateEventError("account_update requires event_id")
    return validated


def load_private_event_evidence(
    ref: PrivateEventStreamRef,
    *,
    now: datetime,
) -> PrivateEventEvidence:
    """Load a fresh exact-strategy event stream into a bounded scanner summary."""

    manifest, window = load_validated_private_event_window(ref, now=now)
    run_id = str(manifest["run_id"])
    now_utc = now.astimezone(timezone.utc)
    ordered = list(window)
    orders = [record for record in ordered if record["event_type"] == "order_update"]
    fills = [record for record in ordered if record["event_type"] == "trade_fill"]
    income = [record for record in ordered if record["event_type"] == "income"]
    status_counts = Counter(str(record.get("status", "")).upper() for record in orders)
    maker_count = sum(record.get("maker") is True for record in fills)
    taker_count = sum(record.get("maker") is False for record in fills)
    unknown_count = len(fills) - maker_count - taker_count
    buy_notional = sum(
        float(record["price_value"]) * float(record["qty_value"])
        for record in fills
        if str(record.get("side", "")).upper() == "BUY"
    )
    sell_notional = sum(
        float(record["price_value"]) * float(record["qty_value"])
        for record in fills
        if str(record.get("side", "")).upper() == "SELL"
    )
    realized = sum(float(record.get("realized_pnl_usdt") or 0.0) for record in fills)
    commissions = sum(float(record.get("commission_usdt") or 0.0) for record in fills)
    funding = sum(
        float(record.get("income_usdt") or 0.0)
        for record in income
        if str(record.get("income_type", "")).upper() == "FUNDING_FEE"
    )
    other_income = sum(
        float(record.get("income_usdt") or 0.0)
        for record in income
        if str(record.get("income_type", "")).upper() != "FUNDING_FEE"
    )
    event_times = [cast(datetime, record["event_at"]) for record in ordered]
    fill_times = [cast(datetime, record["event_at"]) for record in fills]
    coverage = (
        (event_times[-1] - event_times[0]).total_seconds()
        if len(event_times) >= 2
        else 0.0
    )
    return PrivateEventEvidence(
        source="canonical_private_event_stream",
        observed_at_utc=cast(datetime, manifest["updated_at"]),
        age_seconds=(now_utc - cast(datetime, manifest["updated_at"])).total_seconds(),
        run_id=run_id,
        symbol=ref.symbol.upper(),
        strategy_id=ref.strategy_id,
        capture_mode=str(manifest.get("capture_mode", "unknown")),
        event_completeness=str(manifest["event_completeness"]),
        source_scopes=tuple(str(item) for item in manifest["source_scopes"]),
        manifest_total_records=int(manifest["total_records"]),
        records_in_window=len(ordered),
        history_coverage_seconds=coverage,
        order_update_count=len(orders),
        order_status_counts=dict(sorted(status_counts.items())),
        trade_fill_count=len(fills),
        unique_trade_order_count=len({str(record["order_id"]) for record in fills}),
        maker_fill_count=maker_count,
        taker_fill_count=taker_count,
        unknown_liquidity_fill_count=unknown_count,
        buy_fill_notional_usdt=buy_notional,
        sell_fill_notional_usdt=sell_notional,
        realized_pnl_usdt=realized,
        commission_usdt=commissions,
        funding_fee_usdt=funding,
        other_income_usdt=other_income,
        duplicate_records_dropped=int(manifest["duplicate_records_dropped"]),
        rejected_records=int(manifest["rejected_records"]),
        first_event_at_utc=event_times[0] if event_times else None,
        last_event_at_utc=event_times[-1] if event_times else None,
        last_fill_at_utc=fill_times[-1] if fill_times else None,
    )


def load_validated_private_event_window(
    ref: PrivateEventStreamRef,
    *,
    now: datetime,
    max_records: int = 4096,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Return the validated manifest and exact-strategy bounded event window."""

    if now.tzinfo is None:
        raise PrivateEventError("now must be timezone-aware")
    if ref.max_age_seconds <= 0 or ref.history_window_seconds <= 0:
        raise PrivateEventError("private event age/window settings must be positive")
    if max_records < 1:
        raise PrivateEventError("max_records must be positive")
    manifest = _validate_manifest(ref, now=now)
    run_id = str(manifest["run_id"])
    raw_records = _read_tail_records(ref.event_path, max_records=max_records)
    if int(manifest["total_records"]) < len(raw_records):
        raise PrivateEventError(
            "private event manifest total_records is smaller than persisted records"
        )
    validated = [
        _validate_record(record, ref, run_id=run_id) for record in raw_records
    ]
    seen: set[tuple[object, ...]] = set()
    for record in validated:
        key = private_event_dedup_key(record)
        if key in seen:
            raise PrivateEventError(f"duplicate canonical private event in stream: {key}")
        seen.add(key)
    now_utc = now.astimezone(timezone.utc)
    window_start = now_utc.timestamp() - ref.history_window_seconds
    ordered = tuple(
        sorted(
            (
                record
                for record in validated
                if window_start <= cast(datetime, record["event_at"]).timestamp()
                and cast(datetime, record["event_at"]) <= now_utc
            ),
            key=lambda record: cast(datetime, record["event_at"]),
        )
    )
    return manifest, ordered
