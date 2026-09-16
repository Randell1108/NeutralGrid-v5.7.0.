"""Fail-closed normalization and storage for Binance USD-M user-data events.

Account-level frames are preserved as redacted raw evidence for every exact
active target.  Only events carrying an exact strategy ID, or order IDs present
in a reviewed linkage artifact, enter strategy-scoped canonical events.  Symbol
and time proximity are never treated as ownership evidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from neutralgrid.data.diff_depth import AppendOnlyJsonl, atomic_write_json
from neutralgrid.live.decision.private_events import (
    PRIVATE_EVENT_MANIFEST_SCHEMA_VERSION,
    PRIVATE_EVENT_SCHEMA_VERSION,
    private_event_dedup_key,
)


PRIVATE_WIRE_SCHEMA_VERSION = "binance_usdm_private_wire_v1"
PRIVATE_CONTROL_SCHEMA_VERSION = "binance_usdm_private_control_v1"
PRIVATE_SERVICE_MANIFEST_SCHEMA_VERSION = "binance_usdm_private_service_manifest_v1"
LINKAGE_SCHEMA_VERSION = "neutralgrid_strategy_order_linkage_v1"
SYMBOL_RE = re.compile(r"^[A-Z0-9]+USDT$")
IDENTITY_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


class PrivateUserStreamError(RuntimeError):
    """Private-stream configuration, payload, or persistence is invalid."""


class PrivateUserPayloadError(PrivateUserStreamError):
    """A documented private-stream event violated its required shape."""


@dataclass(frozen=True, order=True)
class StrategyTarget:
    symbol: str
    strategy_id: str


@dataclass(frozen=True)
class ExactOrderLinkage:
    target: StrategyTarget
    order_ids: frozenset[str]
    provenance: str
    source_path: Path


@dataclass(frozen=True)
class PrivateNormalization:
    source_event_type: str
    source_symbols: tuple[str, ...]
    classification: str
    canonical_records: tuple[dict[str, Any], ...]
    target: StrategyTarget | None


def validate_target(symbol: str, strategy_id: str) -> StrategyTarget:
    normalized_symbol = symbol.strip().upper()
    normalized_strategy = strategy_id.strip()
    if not SYMBOL_RE.fullmatch(normalized_symbol):
        raise PrivateUserStreamError(f"invalid USD-M symbol: {symbol!r}")
    if not normalized_strategy or not IDENTITY_RE.fullmatch(normalized_strategy):
        raise PrivateUserStreamError("strategy_id contains unsafe characters")
    return StrategyTarget(normalized_symbol, normalized_strategy)


def load_exact_order_linkages(
    paths: Sequence[Path],
    *,
    expected_targets: Iterable[StrategyTarget],
) -> tuple[dict[tuple[str, str], StrategyTarget], tuple[ExactOrderLinkage, ...]]:
    """Load reviewed, collision-free order-to-strategy ownership."""

    expected = set(expected_targets)
    lookup: dict[tuple[str, str], StrategyTarget] = {}
    loaded: list[ExactOrderLinkage] = []
    seen_targets: set[StrategyTarget] = set()
    for source_path in paths:
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PrivateUserStreamError(
                f"cannot read linkage artifact {source_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise PrivateUserStreamError("linkage artifact root must be an object")
        if payload.get("schema_version") != LINKAGE_SCHEMA_VERSION:
            raise PrivateUserStreamError("unsupported linkage schema")
        target = validate_target(
            str(payload.get("symbol", "")),
            str(payload.get("strategy_id", "")),
        )
        if target not in expected:
            raise PrivateUserStreamError(
                f"linkage target is not in the exact active roster: {target}"
            )
        if target in seen_targets:
            raise PrivateUserStreamError(f"duplicate linkage target: {target}")
        provenance = str(payload.get("provenance", "")).strip()
        if not provenance or provenance == "symbol_and_time_only":
            raise PrivateUserStreamError("linkage provenance is missing or ambiguous")
        raw_order_ids = payload.get("order_ids")
        if not isinstance(raw_order_ids, list) or not raw_order_ids:
            raise PrivateUserStreamError("linkage order_ids must be a non-empty list")
        order_ids: set[str] = set()
        for value in raw_order_ids:
            order_id = str(value).strip()
            if not order_id.isdigit() or int(order_id) <= 0:
                raise PrivateUserStreamError("linkage order_ids must be positive integers")
            if order_id in order_ids:
                raise PrivateUserStreamError("linkage order_ids contain duplicates")
            key = (target.symbol, order_id)
            owner = lookup.get(key)
            if owner is not None and owner != target:
                raise PrivateUserStreamError(
                    f"order ID ownership collision for {target.symbol}/{order_id}"
                )
            order_ids.add(order_id)
            lookup[key] = target
        linkage = ExactOrderLinkage(
            target=target,
            order_ids=frozenset(order_ids),
            provenance=provenance,
            source_path=source_path.resolve(),
        )
        loaded.append(linkage)
        seen_targets.add(target)
    return lookup, tuple(sorted(loaded, key=lambda item: item.target))


def redact_secret(value: str, secret: str | None) -> tuple[str, bool]:
    """Remove an in-memory listen key from text before durable persistence."""

    if not secret or secret not in value:
        return value, False
    return value.replace(secret, "<REDACTED_LISTEN_KEY>"), True


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PrivateUserPayloadError(f"{field} must be an object")
    return value


def _require_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if value is None or isinstance(value, bool):
        raise PrivateUserPayloadError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PrivateUserPayloadError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise PrivateUserPayloadError(f"{field} must be non-negative")
    return parsed


def _require_positive_int(payload: Mapping[str, Any], field: str) -> int:
    parsed = _require_int(payload, field)
    if parsed <= 0:
        raise PrivateUserPayloadError(f"{field} must be positive")
    return parsed


def _decimal(
    value: Any,
    *,
    field: str,
    positive: bool = False,
) -> float:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PrivateUserPayloadError(f"{field} must be decimal") from exc
    if not parsed.is_finite():
        raise PrivateUserPayloadError(f"{field} must be finite")
    if positive and parsed <= 0:
        raise PrivateUserPayloadError(f"{field} must be positive")
    return float(parsed)


def _event_iso(event_time_ms: int) -> str:
    try:
        return datetime.fromtimestamp(
            event_time_ms / 1000.0, tz=timezone.utc
        ).isoformat()
    except (OverflowError, OSError, ValueError) as exc:
        raise PrivateUserPayloadError("E is outside the supported timestamp range") from exc


def _normalize_symbol(value: Any, *, field: str) -> str:
    symbol = str(value or "").strip().upper()
    if not SYMBOL_RE.fullmatch(symbol):
        raise PrivateUserPayloadError(f"{field} is not a USD-M symbol")
    return symbol


def extract_private_event_symbols(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract only explicit symbols; absence remains account-wide evidence."""

    nested = payload.get("data", payload)
    event = _require_mapping(nested, field="payload")
    candidates: list[Any] = [event.get("s")]
    for field in ("o", "su", "gu", "ao", "ac"):
        value = event.get(field)
        if isinstance(value, Mapping):
            candidates.append(value.get("s"))
    account = event.get("a")
    if isinstance(account, Mapping):
        candidates.append(account.get("S"))
        positions = account.get("P")
        if isinstance(positions, list):
            for position in positions:
                if isinstance(position, Mapping):
                    candidates.append(position.get("s"))
    margin_positions = event.get("p")
    if isinstance(margin_positions, list):
        for position in margin_positions:
            if isinstance(position, Mapping):
                candidates.append(position.get("s"))
    symbols = {
        str(value).strip().upper()
        for value in candidates
        if value is not None and SYMBOL_RE.fullmatch(str(value).strip().upper())
    }
    return tuple(sorted(symbols))


def normalize_private_user_event(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    exact_targets: Iterable[StrategyTarget],
    order_lookup: Mapping[tuple[str, str], StrategyTarget],
) -> PrivateNormalization:
    """Normalize only ownership-proven events into private-event v1 records."""

    nested = payload.get("data", payload)
    event = _require_mapping(nested, field="payload")
    event_type = str(event.get("e", "")).strip()
    if not event_type:
        raise PrivateUserPayloadError("private event type is missing")
    event_time_ms = _require_int(event, "E")
    event_time_utc = _event_iso(event_time_ms)
    source_symbols = extract_private_event_symbols(event)
    target_set = set(exact_targets)

    if event_type == "ORDER_TRADE_UPDATE":
        order = _require_mapping(event.get("o"), field="o")
        symbol = _normalize_symbol(order.get("s"), field="o.s")
        order_id = str(_require_positive_int(order, "i"))
        target = order_lookup.get((symbol, order_id))
        if target is None:
            return PrivateNormalization(
                source_event_type=event_type,
                source_symbols=(symbol,),
                classification="unlinked_order_preserved_raw_only",
                canonical_records=(),
                target=None,
            )
        if target not in target_set:
            raise PrivateUserPayloadError("linked order target is not active")
        status = str(order.get("X", "")).strip().upper()
        execution_type = str(order.get("x", "")).strip().upper()
        if not status or not execution_type:
            raise PrivateUserPayloadError("order status or execution type is missing")
        order_record: dict[str, Any] = {
            "schema_version": PRIVATE_EVENT_SCHEMA_VERSION,
            "event_type": "order_update",
            "symbol": symbol,
            "strategy_id": target.strategy_id,
            "run_id": run_id,
            "event_time_utc": event_time_utc,
            "transaction_time_ms": _require_int(event, "T"),
            "order_id": order_id,
            "client_order_id": str(order.get("c", "")),
            "status": status,
            "execution_type": execution_type,
            "executed_qty": _decimal(order.get("z", "0"), field="o.z"),
            "source_event": "ORDER_TRADE_UPDATE",
        }
        records = [order_record]
        if execution_type == "TRADE":
            trade_id = str(_require_positive_int(order, "t"))
            last_qty = _decimal(order.get("l"), field="o.l", positive=True)
            last_price = _decimal(order.get("L"), field="o.L", positive=True)
            side = str(order.get("S", "")).strip().upper()
            if side not in {"BUY", "SELL"}:
                raise PrivateUserPayloadError("o.S must be BUY or SELL")
            commission_asset = str(order.get("N", "") or "").strip().upper()
            commission_amount = _decimal(order.get("n") or "0", field="o.n")
            trade_record: dict[str, Any] = {
                "schema_version": PRIVATE_EVENT_SCHEMA_VERSION,
                "event_type": "trade_fill",
                "symbol": symbol,
                "strategy_id": target.strategy_id,
                "run_id": run_id,
                "event_time_utc": event_time_utc,
                "transaction_time_ms": _require_int(event, "T"),
                "trade_time_ms": _require_int(order, "T"),
                "trade_id": trade_id,
                "order_id": order_id,
                "side": side,
                "price": last_price,
                "qty": last_qty,
                "maker": order.get("m") if isinstance(order.get("m"), bool) else None,
                "realized_pnl_usdt": _decimal(order.get("rp") or "0", field="o.rp"),
                "commission_asset": commission_asset or None,
                "commission_amount": commission_amount,
                "source_event": "ORDER_TRADE_UPDATE",
            }
            if commission_asset == "USDT":
                trade_record["commission_usdt"] = -abs(commission_amount)
            records.append(trade_record)
        return PrivateNormalization(
            source_event_type=event_type,
            source_symbols=(symbol,),
            classification="accepted_exact_order_linkage",
            canonical_records=tuple(records),
            target=target,
        )

    if event_type in {"STRATEGY_UPDATE", "GRID_UPDATE"}:
        field = "su" if event_type == "STRATEGY_UPDATE" else "gu"
        strategy = _require_mapping(event.get(field), field=field)
        symbol = _normalize_symbol(strategy.get("s"), field=f"{field}.s")
        strategy_id = str(strategy.get("si", "")).strip()
        target = validate_target(symbol, strategy_id)
        if target not in target_set:
            return PrivateNormalization(
                source_event_type=event_type,
                source_symbols=(symbol,),
                classification="inactive_strategy_preserved_raw_only",
                canonical_records=(),
                target=None,
            )
        transaction_time = _require_int(event, "T")
        status = str(strategy.get("ss", "")).strip().upper()
        opcode = str(strategy.get("c", strategy.get("opCode", ""))).strip()
        record = {
            "schema_version": PRIVATE_EVENT_SCHEMA_VERSION,
            "event_type": "account_update",
            "symbol": symbol,
            "strategy_id": target.strategy_id,
            "run_id": run_id,
            "event_time_utc": event_time_utc,
            "event_id": (
                f"{event_type}:{event_time_ms}:{transaction_time}:"
                f"{target.strategy_id}:{status}:{opcode}"
            ),
            "transaction_time_ms": transaction_time,
            "strategy_status": status or None,
            "strategy_opcode": opcode or None,
            "source_event": event_type,
            "deprecated_source": event_type == "GRID_UPDATE",
        }
        return PrivateNormalization(
            source_event_type=event_type,
            source_symbols=(symbol,),
            classification="accepted_exact_strategy_identity",
            canonical_records=(record,),
            target=target,
        )

    return PrivateNormalization(
        source_event_type=event_type,
        source_symbols=source_symbols,
        classification="account_or_unsupported_event_preserved_raw_only",
        canonical_records=(),
        target=None,
    )


class PrivateStrategyStorage:
    """Per-strategy raw replica and exact-link canonical event storage."""

    def __init__(
        self,
        run_dir: Path,
        *,
        target: StrategyTarget,
        run_id: str,
        fsync_every: int = 1,
    ) -> None:
        self.run_dir = run_dir
        self.target = target
        self.run_id = run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.wire = AppendOnlyJsonl(
            run_dir / "wire_events.jsonl", fsync_every=fsync_every
        )
        self.events = AppendOnlyJsonl(
            run_dir / "events.jsonl", fsync_every=fsync_every
        )
        self.control = AppendOnlyJsonl(
            run_dir / "control.jsonl", fsync_every=fsync_every
        )
        self.manifest_path = run_dir / "manifest.json"
        self._dedup_records: dict[tuple[object, ...], dict[str, Any]] = {}
        self.counters: dict[str, int] = {
            "wire_events": 0,
            "redacted_wire_events": 0,
            "canonical_records": 0,
            "duplicate_records_dropped": 0,
            "conflicting_records_rejected": 0,
            "unlinked_order_events": 0,
            "raw_only_events": 0,
            "parse_errors": 0,
        }

    def append_wire(
        self,
        *,
        connection_id: str,
        wire_sequence: int,
        received_at_utc: str,
        received_monotonic_ns: int,
        original_sha256: str,
        persisted_text: str,
        credential_redacted: bool,
    ) -> None:
        self.wire.append(
            {
                "schema_version": PRIVATE_WIRE_SCHEMA_VERSION,
                "record_type": "account_scope_wire_replica",
                "symbol": self.target.symbol,
                "strategy_id": self.target.strategy_id,
                "connection_id": connection_id,
                "wire_sequence": wire_sequence,
                "received_at_utc": received_at_utc,
                "received_monotonic_ns": received_monotonic_ns,
                "original_sha256": original_sha256,
                "credential_redacted": credential_redacted,
                "persisted_text": persisted_text,
                "scope_note": (
                    "One account-level frame replicated to each exact active target; "
                    "never aggregate replicas across targets."
                ),
            }
        )
        self.counters["wire_events"] += 1
        if credential_redacted:
            self.counters["redacted_wire_events"] += 1

    def append_canonical(self, record: Mapping[str, Any]) -> str:
        candidate = dict(record)
        if candidate.get("symbol") != self.target.symbol or candidate.get(
            "strategy_id"
        ) != self.target.strategy_id:
            raise PrivateUserStreamError("canonical event identity mismatch")
        key = private_event_dedup_key(candidate)
        previous = self._dedup_records.get(key)
        if previous is not None:
            if previous == candidate:
                self.counters["duplicate_records_dropped"] += 1
                return "duplicate"
            self.counters["conflicting_records_rejected"] += 1
            raise PrivateUserStreamError(
                f"conflicting canonical private event for key {key}"
            )
        self.events.append(candidate)
        self._dedup_records[key] = candidate
        self.counters["canonical_records"] += 1
        return "appended"

    def append_control(self, kind: str, details: Mapping[str, Any]) -> None:
        self.control.append(
            {
                "schema_version": PRIVATE_CONTROL_SCHEMA_VERSION,
                "record_type": "control",
                "kind": kind,
                **dict(details),
            }
        )

    def write_manifest(self, payload: Mapping[str, Any]) -> None:
        atomic_write_json(
            self.manifest_path,
            {
                "schema_version": PRIVATE_EVENT_MANIFEST_SCHEMA_VERSION,
                "run_id": self.run_id,
                "symbol": self.target.symbol,
                "strategy_id": self.target.strategy_id,
                "event_path": "events.jsonl",
                "capture_mode": "binance_user_data_stream",
                "event_completeness": "unknown",
                "source_scopes": [
                    "account_raw",
                    "exact_order_linked_orders",
                    "exact_order_linked_trades",
                    "exact_strategy_updates",
                ],
                "total_records": self.counters["canonical_records"],
                "duplicate_records_dropped": self.counters[
                    "duplicate_records_dropped"
                ],
                "rejected_records": (
                    self.counters["conflicting_records_rejected"]
                    + self.counters["unlinked_order_events"]
                    + self.counters["parse_errors"]
                ),
                "counters": dict(self.counters),
                "raw_replication_scope": "all_exact_active_targets",
                "raw_credential_policy": (
                    "Original frame SHA-256 retained; listen key redacted before "
                    "durable persistence."
                ),
                "runtime_effect": "observational_only",
                **dict(payload),
            },
        )

    def close(self) -> None:
        self.wire.close()
        self.events.close()
        self.control.close()
