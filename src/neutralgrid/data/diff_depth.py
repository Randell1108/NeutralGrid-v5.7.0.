"""Prospective Binance USD-M diff-depth capture and deterministic replay.

The module deliberately separates three evidence layers:

* ``wire_events.jsonl`` stores each WebSocket text frame before parsing.
* ``timeline.jsonl`` stores the exact normalized inputs offered to the
  deterministic sequence engine.
* ``engine_actions.jsonl`` stores the engine's deterministic decisions.

An event-complete segment is only a contiguous portion of Binance's published
diff-depth stream.  A sequence break, connection boundary, parse failure, or
unfinished bootstrap closes that segment and must be represented explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast


WIRE_SCHEMA_VERSION = "binance_usdm_diff_depth_wire_v1"
TIMELINE_SCHEMA_VERSION = "binance_usdm_diff_depth_timeline_v1"
ACTION_SCHEMA_VERSION = "binance_usdm_diff_depth_action_v1"
SNAPSHOT_SCHEMA_VERSION = "binance_usdm_depth_snapshot_v1"
MANIFEST_SCHEMA_VERSION = "binance_usdm_diff_depth_manifest_v1"
PUBLIC_AGG_TRADE_SCHEMA_VERSION = "binance_usdm_public_agg_trade_v1"
PUBLIC_MARK_PRICE_SCHEMA_VERSION = "binance_usdm_public_mark_price_v1"


class DiffDepthError(RuntimeError):
    """Base class for fail-closed diff-depth errors."""


class PayloadValidationError(DiffDepthError):
    """A WebSocket event or REST snapshot violated its schema."""


class SequenceBufferOverflow(DiffDepthError):
    """The pre-bootstrap or resynchronization buffer exceeded its limit."""


class BookInvariantError(DiffDepthError):
    """Applying an event would produce an invalid local order book."""


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _parse_decimal(value: Any, *, field: str, allow_zero: bool) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PayloadValidationError(f"{field} is not decimal: {value!r}") from exc
    if not parsed.is_finite():
        raise PayloadValidationError(f"{field} is not finite: {value!r}")
    if parsed < 0 or (not allow_zero and parsed == 0):
        comparator = "non-negative" if allow_zero else "positive"
        raise PayloadValidationError(f"{field} must be {comparator}: {value!r}")
    return parsed


def _parse_signed_decimal(value: Any, *, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PayloadValidationError(f"{field} is not decimal: {value!r}") from exc
    if not parsed.is_finite():
        raise PayloadValidationError(f"{field} is not finite: {value!r}")
    return parsed


def _parse_levels(
    value: Any,
    *,
    field: str,
) -> tuple[tuple[Decimal, Decimal], ...]:
    if not isinstance(value, list):
        raise PayloadValidationError(f"{field} must be a list")
    levels: list[tuple[Decimal, Decimal]] = []
    for index, raw_level in enumerate(value):
        if not isinstance(raw_level, (list, tuple)) or len(raw_level) != 2:
            raise PayloadValidationError(
                f"{field}[{index}] must be a two-item price/quantity pair"
            )
        price = _parse_decimal(
            raw_level[0],
            field=f"{field}[{index}].price",
            allow_zero=False,
        )
        quantity = _parse_decimal(
            raw_level[1],
            field=f"{field}[{index}].quantity",
            allow_zero=True,
        )
        levels.append((price, quantity))
    return tuple(levels)


def _require_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if value is None or isinstance(value, bool):
        raise PayloadValidationError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PayloadValidationError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise PayloadValidationError(f"{field} must be non-negative")
    return parsed


@dataclass(frozen=True)
class DiffDepthEvent:
    """One validated Binance USD-M diff-depth event."""

    symbol: str
    event_time_ms: int
    transaction_time_ms: int
    first_update_id: int
    final_update_id: int
    previous_final_update_id: int
    bids: tuple[tuple[Decimal, Decimal], ...]
    asks: tuple[tuple[Decimal, Decimal], ...]
    connection_id: str
    wire_sequence: int
    received_at_utc: str
    received_monotonic_ns: int

    def to_timeline_record(self) -> dict[str, Any]:
        """Return the canonical deterministic replay input."""

        return {
            "schema_version": TIMELINE_SCHEMA_VERSION,
            "record_type": "event",
            "connection_id": self.connection_id,
            "wire_sequence": self.wire_sequence,
            "received_at_utc": self.received_at_utc,
            "received_monotonic_ns": self.received_monotonic_ns,
            "payload": {
                "e": "depthUpdate",
                "E": self.event_time_ms,
                "T": self.transaction_time_ms,
                "s": self.symbol,
                "U": self.first_update_id,
                "u": self.final_update_id,
                "pu": self.previous_final_update_id,
                "b": [
                    [_decimal_text(price), _decimal_text(quantity)]
                    for price, quantity in self.bids
                ],
                "a": [
                    [_decimal_text(price), _decimal_text(quantity)]
                    for price, quantity in self.asks
                ],
            },
        }


@dataclass(frozen=True)
class PublicAggTrade:
    """One validated Binance USD-M aggregate-trade stream event."""

    symbol: str
    event_time_ms: int
    aggregate_trade_id: int
    price: Decimal
    quantity: Decimal
    first_trade_id: int
    last_trade_id: int
    trade_time_ms: int
    buyer_is_maker: bool
    connection_id: str
    wire_sequence: int
    received_at_utc: str
    received_monotonic_ns: int

    @property
    def aggressive_side(self) -> str:
        return "SELL" if self.buyer_is_maker else "BUY"

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": PUBLIC_AGG_TRADE_SCHEMA_VERSION,
            "record_type": "public_agg_trade",
            "symbol": self.symbol,
            "event_time_ms": self.event_time_ms,
            "aggregate_trade_id": self.aggregate_trade_id,
            "price": _decimal_text(self.price),
            "quantity": _decimal_text(self.quantity),
            "notional_usdt": _decimal_text(self.price * self.quantity),
            "first_trade_id": self.first_trade_id,
            "last_trade_id": self.last_trade_id,
            "trade_time_ms": self.trade_time_ms,
            "buyer_is_maker": self.buyer_is_maker,
            "aggressive_side": self.aggressive_side,
            "connection_id": self.connection_id,
            "wire_sequence": self.wire_sequence,
            "received_at_utc": self.received_at_utc,
            "received_monotonic_ns": self.received_monotonic_ns,
        }


@dataclass(frozen=True)
class PublicMarkPrice:
    """One validated Binance USD-M one-second mark-price update."""

    symbol: str
    event_time_ms: int
    mark_price: Decimal
    index_price: Decimal
    estimated_settle_price: Decimal
    funding_rate: Decimal
    next_funding_time_ms: int
    connection_id: str
    wire_sequence: int
    received_at_utc: str
    received_monotonic_ns: int

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": PUBLIC_MARK_PRICE_SCHEMA_VERSION,
            "record_type": "public_mark_price_update",
            "symbol": self.symbol,
            "event_time_ms": self.event_time_ms,
            "mark_price": _decimal_text(self.mark_price),
            "index_price": _decimal_text(self.index_price),
            "estimated_settle_price": _decimal_text(self.estimated_settle_price),
            "funding_rate": _decimal_text(self.funding_rate),
            "next_funding_time_ms": self.next_funding_time_ms,
            "connection_id": self.connection_id,
            "wire_sequence": self.wire_sequence,
            "received_at_utc": self.received_at_utc,
            "received_monotonic_ns": self.received_monotonic_ns,
        }


@dataclass(frozen=True)
class DepthSnapshot:
    """Validated REST depth snapshot used to bootstrap a segment."""

    symbol: str
    last_update_id: int
    bids: tuple[tuple[Decimal, Decimal], ...]
    asks: tuple[tuple[Decimal, Decimal], ...]
    connection_id: str
    request_started_at_utc: str
    received_at_utc: str
    received_monotonic_ns: int
    wire_sequence_seen: int
    request_latency_ms: float

    def to_timeline_record(self) -> dict[str, Any]:
        """Return the canonical deterministic replay input."""

        return {
            "schema_version": TIMELINE_SCHEMA_VERSION,
            "record_type": "snapshot",
            "connection_id": self.connection_id,
            "request_started_at_utc": self.request_started_at_utc,
            "received_at_utc": self.received_at_utc,
            "received_monotonic_ns": self.received_monotonic_ns,
            "wire_sequence_seen": self.wire_sequence_seen,
            "request_latency_ms": self.request_latency_ms,
            "payload": {
                "lastUpdateId": self.last_update_id,
                "bids": [
                    [_decimal_text(price), _decimal_text(quantity)]
                    for price, quantity in self.bids
                ],
                "asks": [
                    [_decimal_text(price), _decimal_text(quantity)]
                    for price, quantity in self.asks
                ],
            },
        }


@dataclass(frozen=True)
class EngineAction:
    """One deterministic decision made by :class:`DepthSequenceEngine`."""

    kind: str
    details: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": ACTION_SCHEMA_VERSION,
            "kind": self.kind,
            "details": self.details,
        }


@dataclass(frozen=True)
class ReplayVerification:
    """Result of replaying one symbol's stored deterministic timeline."""

    symbol: str
    timeline_records: int
    expected_actions: int
    replayed_actions: int
    raw_events: int
    raw_hash_failures: int
    actions_match: bool

    @property
    def passed(self) -> bool:
        return self.raw_hash_failures == 0 and self.actions_match


def parse_diff_depth_event(
    raw_payload: str | Mapping[str, Any],
    *,
    expected_symbol: str,
    connection_id: str,
    wire_sequence: int,
    received_at_utc: str,
    received_monotonic_ns: int,
) -> DiffDepthEvent:
    """Parse and validate one raw or combined-stream diff-depth event."""

    if isinstance(raw_payload, str):
        try:
            decoded = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise PayloadValidationError("WebSocket frame is not valid JSON") from exc
    else:
        decoded = dict(raw_payload)
    if not isinstance(decoded, dict):
        raise PayloadValidationError("WebSocket payload must be an object")
    nested = decoded.get("data", decoded)
    if not isinstance(nested, dict):
        raise PayloadValidationError("WebSocket data payload must be an object")
    payload = cast(dict[str, Any], nested)
    if payload.get("e") != "depthUpdate":
        raise PayloadValidationError(
            f"unexpected WebSocket event type: {payload.get('e')!r}"
        )
    symbol = str(payload.get("s", "")).upper()
    if symbol != expected_symbol.upper():
        raise PayloadValidationError(
            f"event symbol {symbol!r} does not match {expected_symbol.upper()!r}"
        )
    first_update_id = _require_int(payload, "U")
    final_update_id = _require_int(payload, "u")
    previous_final_update_id = _require_int(payload, "pu")
    if first_update_id > final_update_id:
        raise PayloadValidationError(
            f"U must be <= u, got U={first_update_id}, u={final_update_id}"
        )
    bids = _parse_levels(payload.get("b"), field="b")
    asks = _parse_levels(payload.get("a"), field="a")
    if not bids and not asks:
        raise PayloadValidationError("depth event contains no bid or ask updates")
    return DiffDepthEvent(
        symbol=symbol,
        event_time_ms=_require_int(payload, "E"),
        transaction_time_ms=_require_int(payload, "T"),
        first_update_id=first_update_id,
        final_update_id=final_update_id,
        previous_final_update_id=previous_final_update_id,
        bids=bids,
        asks=asks,
        connection_id=connection_id,
        wire_sequence=wire_sequence,
        received_at_utc=received_at_utc,
        received_monotonic_ns=received_monotonic_ns,
    )


def parse_public_agg_trade(
    raw_payload: str | Mapping[str, Any],
    *,
    expected_symbol: str,
    connection_id: str,
    wire_sequence: int,
    received_at_utc: str,
    received_monotonic_ns: int,
) -> PublicAggTrade:
    """Parse one raw or combined-stream aggregate-trade event."""

    if isinstance(raw_payload, str):
        try:
            decoded = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise PayloadValidationError("WebSocket frame is not valid JSON") from exc
    else:
        decoded = dict(raw_payload)
    if not isinstance(decoded, dict):
        raise PayloadValidationError("WebSocket payload must be an object")
    nested = decoded.get("data", decoded)
    if not isinstance(nested, dict):
        raise PayloadValidationError("WebSocket data payload must be an object")
    payload = cast(dict[str, Any], nested)
    if payload.get("e") != "aggTrade":
        raise PayloadValidationError(
            f"unexpected WebSocket event type: {payload.get('e')!r}"
        )
    symbol = str(payload.get("s", "")).upper()
    if symbol != expected_symbol.upper():
        raise PayloadValidationError(
            f"event symbol {symbol!r} does not match {expected_symbol.upper()!r}"
        )
    buyer_is_maker = payload.get("m")
    if not isinstance(buyer_is_maker, bool):
        raise PayloadValidationError("m must be a boolean")
    first_trade_id = _require_int(payload, "f")
    last_trade_id = _require_int(payload, "l")
    if first_trade_id > last_trade_id:
        raise PayloadValidationError(
            f"f must be <= l, got f={first_trade_id}, l={last_trade_id}"
        )
    return PublicAggTrade(
        symbol=symbol,
        event_time_ms=_require_int(payload, "E"),
        aggregate_trade_id=_require_int(payload, "a"),
        price=_parse_decimal(payload.get("p"), field="p", allow_zero=False),
        quantity=_parse_decimal(payload.get("q"), field="q", allow_zero=False),
        first_trade_id=first_trade_id,
        last_trade_id=last_trade_id,
        trade_time_ms=_require_int(payload, "T"),
        buyer_is_maker=buyer_is_maker,
        connection_id=connection_id,
        wire_sequence=wire_sequence,
        received_at_utc=received_at_utc,
        received_monotonic_ns=received_monotonic_ns,
    )


def parse_public_mark_price(
    raw_payload: str | Mapping[str, Any],
    *,
    expected_symbol: str,
    connection_id: str,
    wire_sequence: int,
    received_at_utc: str,
    received_monotonic_ns: int,
) -> PublicMarkPrice:
    """Parse one raw or combined-stream one-second mark-price update."""

    if isinstance(raw_payload, str):
        try:
            decoded = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise PayloadValidationError("WebSocket frame is not valid JSON") from exc
    else:
        decoded = dict(raw_payload)
    nested = decoded.get("data", decoded)
    if not isinstance(nested, dict):
        raise PayloadValidationError("WebSocket data payload must be an object")
    payload = cast(dict[str, Any], nested)
    if payload.get("e") != "markPriceUpdate":
        raise PayloadValidationError(
            f"unexpected WebSocket event type: {payload.get('e')!r}"
        )
    symbol = str(payload.get("s", "")).upper()
    if symbol != expected_symbol.upper():
        raise PayloadValidationError(
            f"event symbol {symbol!r} does not match {expected_symbol.upper()!r}"
        )
    return PublicMarkPrice(
        symbol=symbol,
        event_time_ms=_require_int(payload, "E"),
        mark_price=_parse_decimal(payload.get("p"), field="p", allow_zero=False),
        index_price=_parse_decimal(payload.get("i"), field="i", allow_zero=False),
        estimated_settle_price=_parse_decimal(
            payload.get("P"), field="P", allow_zero=True
        ),
        funding_rate=_parse_signed_decimal(payload.get("r"), field="r"),
        next_funding_time_ms=_require_int(payload, "T"),
        connection_id=connection_id,
        wire_sequence=wire_sequence,
        received_at_utc=received_at_utc,
        received_monotonic_ns=received_monotonic_ns,
    )


def parse_depth_snapshot(
    payload: Mapping[str, Any],
    *,
    symbol: str,
    connection_id: str,
    request_started_at_utc: str,
    received_at_utc: str,
    received_monotonic_ns: int,
    wire_sequence_seen: int,
    request_latency_ms: float,
) -> DepthSnapshot:
    """Parse and validate a Binance REST depth snapshot."""

    last_update_id = _require_int(payload, "lastUpdateId")
    bids = _parse_levels(payload.get("bids"), field="bids")
    asks = _parse_levels(payload.get("asks"), field="asks")
    if not bids or not asks:
        raise PayloadValidationError("depth snapshot must contain bids and asks")
    snapshot = DepthSnapshot(
        symbol=symbol.upper(),
        last_update_id=last_update_id,
        bids=bids,
        asks=asks,
        connection_id=connection_id,
        request_started_at_utc=request_started_at_utc,
        received_at_utc=received_at_utc,
        received_monotonic_ns=received_monotonic_ns,
        wire_sequence_seen=wire_sequence_seen,
        request_latency_ms=float(request_latency_ms),
    )
    LocalOrderBook.from_snapshot(snapshot)
    return snapshot


class LocalOrderBook:
    """Exact decimal price-level state reconstructed from a REST snapshot."""

    def __init__(
        self,
        bids: dict[Decimal, Decimal],
        asks: dict[Decimal, Decimal],
    ) -> None:
        self.bids = bids
        self.asks = asks
        self._validate()

    @classmethod
    def from_snapshot(cls, snapshot: DepthSnapshot) -> LocalOrderBook:
        bids = {price: quantity for price, quantity in snapshot.bids if quantity > 0}
        asks = {price: quantity for price, quantity in snapshot.asks if quantity > 0}
        return cls(bids=bids, asks=asks)

    @property
    def best_bid(self) -> Decimal | None:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return min(self.asks) if self.asks else None

    def _validate(self) -> None:
        best_bid = self.best_bid
        best_ask = self.best_ask
        if best_bid is None or best_ask is None:
            raise BookInvariantError("local book must contain bids and asks")
        if best_bid >= best_ask:
            raise BookInvariantError(
                f"crossed local book: best_bid={best_bid}, best_ask={best_ask}"
            )

    def apply(self, event: DiffDepthEvent) -> None:
        """Apply absolute quantities transactionally and validate the result."""

        bid_before = {price: self.bids.get(price) for price, _ in event.bids}
        ask_before = {price: self.asks.get(price) for price, _ in event.asks}
        for price, quantity in event.bids:
            if quantity == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = quantity
        for price, quantity in event.asks:
            if quantity == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = quantity
        try:
            self._validate()
        except BookInvariantError:
            self._restore(self.bids, bid_before)
            self._restore(self.asks, ask_before)
            raise

    @staticmethod
    def _restore(
        side: dict[Decimal, Decimal],
        previous: Mapping[Decimal, Decimal | None],
    ) -> None:
        for price, quantity in previous.items():
            if quantity is None:
                side.pop(price, None)
            else:
                side[price] = quantity

    def digest(self) -> str:
        payload = {
            "bids": [
                [_decimal_text(price), _decimal_text(self.bids[price])]
                for price in sorted(self.bids, reverse=True)
            ],
            "asks": [
                [_decimal_text(price), _decimal_text(self.asks[price])]
                for price in sorted(self.asks)
            ],
        }
        canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def checkpoint(self, *, top_n: int) -> dict[str, Any]:
        bid_prices = sorted(self.bids, reverse=True)
        ask_prices = sorted(self.asks)
        return {
            "best_bid": (
                _decimal_text(bid_prices[0]) if bid_prices else None
            ),
            "best_ask": (
                _decimal_text(ask_prices[0]) if ask_prices else None
            ),
            "bid_level_count": len(bid_prices),
            "ask_level_count": len(ask_prices),
            "top_bids": [
                [_decimal_text(price), _decimal_text(self.bids[price])]
                for price in bid_prices[:top_n]
            ],
            "top_asks": [
                [_decimal_text(price), _decimal_text(self.asks[price])]
                for price in ask_prices[:top_n]
            ],
            "book_sha256": self.digest(),
        }


class DepthSequenceEngine:
    """Pure sequence/bootstrap state machine for one connection and symbol."""

    def __init__(
        self,
        symbol: str,
        *,
        segment_prefix: str,
        max_buffer_events: int = 100_000,
    ) -> None:
        self.symbol = symbol.upper()
        self.segment_prefix = segment_prefix
        self.max_buffer_events = max_buffer_events
        self.phase = "buffering"
        self.buffer: list[DiffDepthEvent] = []
        self.snapshot: DepthSnapshot | None = None
        self.book: LocalOrderBook | None = None
        self.last_u: int | None = None
        self.current_segment_id: str | None = None
        self.segment_index = 0
        self.segment_applied_count = 0
        self.segment_first_u: int | None = None
        self.last_event_time_ms: int | None = None
        self.last_transaction_time_ms: int | None = None

    def offer_event(self, event: DiffDepthEvent) -> list[EngineAction]:
        if event.symbol != self.symbol:
            raise PayloadValidationError(
                f"engine {self.symbol} received event for {event.symbol}"
            )
        if self.phase == "live":
            if self.last_u is None or self.book is None:
                raise DiffDepthError("live engine is missing sequence or book state")
            if event.previous_final_update_id != self.last_u:
                actions = [
                    EngineAction(
                        "sequence_gap",
                        {
                            "symbol": self.symbol,
                            "segment_id": self.current_segment_id,
                            "expected_pu": self.last_u,
                            "received_pu": event.previous_final_update_id,
                            "event_first_update_id": event.first_update_id,
                            "event_final_update_id": event.final_update_id,
                            "wire_sequence": event.wire_sequence,
                        },
                    )
                ]
                actions.extend(self._close_segment("sequence_gap"))
                self._reset_to_buffer([event])
                actions.append(self._buffered_action(event, "sequence_gap"))
                return actions
            try:
                return [self._apply_event(event)]
            except BookInvariantError as exc:
                actions = [
                    EngineAction(
                        "book_invariant_failure",
                        {
                            "symbol": self.symbol,
                            "segment_id": self.current_segment_id,
                            "wire_sequence": event.wire_sequence,
                            "event_final_update_id": event.final_update_id,
                            "error": str(exc),
                        },
                    )
                ]
                actions.extend(self._close_segment("book_invariant_failure"))
                self._reset_to_buffer([event])
                actions.append(self._buffered_action(event, "book_invariant_failure"))
                return actions

        self._append_buffer(event)
        actions = [self._buffered_action(event, "awaiting_snapshot")]
        if self.snapshot is not None:
            actions.extend(self._attempt_bootstrap())
        return actions

    def offer_snapshot(self, snapshot: DepthSnapshot) -> list[EngineAction]:
        if snapshot.symbol != self.symbol:
            raise PayloadValidationError(
                f"engine {self.symbol} received snapshot for {snapshot.symbol}"
            )
        if snapshot.connection_id != self.segment_prefix:
            raise PayloadValidationError(
                "snapshot connection does not match the sequence engine"
            )
        if self.phase != "buffering":
            raise DiffDepthError("cannot offer a bootstrap snapshot to a live engine")
        self.snapshot = snapshot
        actions = [
            EngineAction(
                "snapshot_offered",
                {
                    "symbol": self.symbol,
                    "connection_id": snapshot.connection_id,
                    "last_update_id": snapshot.last_update_id,
                    "wire_sequence_seen": snapshot.wire_sequence_seen,
                },
            )
        ]
        actions.extend(self._attempt_bootstrap())
        return actions

    def _append_buffer(self, event: DiffDepthEvent) -> None:
        self.buffer.append(event)
        if len(self.buffer) > self.max_buffer_events:
            raise SequenceBufferOverflow(
                f"{self.symbol} buffered {len(self.buffer)} events; "
                f"limit is {self.max_buffer_events}"
            )

    def _buffered_action(self, event: DiffDepthEvent, reason: str) -> EngineAction:
        return EngineAction(
            "event_buffered",
            {
                "symbol": self.symbol,
                "connection_id": event.connection_id,
                "wire_sequence": event.wire_sequence,
                "event_first_update_id": event.first_update_id,
                "event_final_update_id": event.final_update_id,
                "reason": reason,
            },
        )

    def _attempt_bootstrap(self) -> list[EngineAction]:
        snapshot = self.snapshot
        if snapshot is None or not self.buffer:
            return []
        actions: list[EngineAction] = []
        discarded: list[DiffDepthEvent] = []
        while self.buffer and self.buffer[0].final_update_id < snapshot.last_update_id:
            discarded.append(self.buffer.pop(0))
        if discarded:
            actions.append(
                EngineAction(
                    "events_discarded_before_snapshot",
                    {
                        "symbol": self.symbol,
                        "count": len(discarded),
                        "wire_sequences": [
                            event.wire_sequence for event in discarded
                        ],
                        "last_update_id": snapshot.last_update_id,
                    },
                )
            )
        if not self.buffer:
            return actions
        first = self.buffer[0]
        if first.first_update_id > snapshot.last_update_id:
            actions.append(
                EngineAction(
                    "snapshot_behind_buffer",
                    {
                        "symbol": self.symbol,
                        "last_update_id": snapshot.last_update_id,
                        "first_buffered_first_update_id": first.first_update_id,
                        "first_buffered_final_update_id": first.final_update_id,
                        "wire_sequence": first.wire_sequence,
                    },
                )
            )
            self.snapshot = None
            return actions
        if not (
            first.first_update_id
            <= snapshot.last_update_id
            <= first.final_update_id
        ):
            raise DiffDepthError("bootstrap bridge predicate is inconsistent")

        self.segment_index += 1
        self.current_segment_id = (
            f"{self.segment_prefix}-segment-{self.segment_index:06d}"
        )
        self.book = LocalOrderBook.from_snapshot(snapshot)
        self.phase = "live"
        self.last_u = None
        self.segment_applied_count = 0
        self.segment_first_u = None
        actions.append(
            EngineAction(
                "segment_started",
                {
                    "symbol": self.symbol,
                    "segment_id": self.current_segment_id,
                    "connection_id": self.segment_prefix,
                    "snapshot_last_update_id": snapshot.last_update_id,
                    "snapshot_wire_sequence_seen": snapshot.wire_sequence_seen,
                    "bridging_wire_sequence": first.wire_sequence,
                    "bridging_first_update_id": first.first_update_id,
                    "bridging_final_update_id": first.final_update_id,
                },
            )
        )
        pending = self.buffer
        self.buffer = []
        for index, event in enumerate(pending):
            if self.last_u is not None and event.previous_final_update_id != self.last_u:
                actions.append(
                    EngineAction(
                        "sequence_gap",
                        {
                            "symbol": self.symbol,
                            "segment_id": self.current_segment_id,
                            "expected_pu": self.last_u,
                            "received_pu": event.previous_final_update_id,
                            "event_first_update_id": event.first_update_id,
                            "event_final_update_id": event.final_update_id,
                            "wire_sequence": event.wire_sequence,
                        },
                    )
                )
                actions.extend(self._close_segment("sequence_gap"))
                self._reset_to_buffer(pending[index:])
                self.snapshot = None
                return actions
            try:
                actions.append(self._apply_event(event))
            except BookInvariantError as exc:
                actions.append(
                    EngineAction(
                        "book_invariant_failure",
                        {
                            "symbol": self.symbol,
                            "segment_id": self.current_segment_id,
                            "wire_sequence": event.wire_sequence,
                            "event_final_update_id": event.final_update_id,
                            "error": str(exc),
                        },
                    )
                )
                actions.extend(self._close_segment("book_invariant_failure"))
                self._reset_to_buffer(pending[index:])
                self.snapshot = None
                return actions
        return actions

    def _apply_event(self, event: DiffDepthEvent) -> EngineAction:
        if self.book is None or self.current_segment_id is None:
            raise DiffDepthError("cannot apply event without an active segment")
        self.book.apply(event)
        self.last_u = event.final_update_id
        if self.segment_first_u is None:
            self.segment_first_u = event.first_update_id
        self.segment_applied_count += 1
        self.last_event_time_ms = event.event_time_ms
        self.last_transaction_time_ms = event.transaction_time_ms
        return EngineAction(
            "event_applied",
            {
                "symbol": self.symbol,
                "segment_id": self.current_segment_id,
                "wire_sequence": event.wire_sequence,
                "event_first_update_id": event.first_update_id,
                "event_final_update_id": event.final_update_id,
                "event_pu": event.previous_final_update_id,
            },
        )

    def _close_segment(self, reason: str) -> list[EngineAction]:
        if self.current_segment_id is None:
            return []
        checkpoint = self.book.checkpoint(top_n=20) if self.book is not None else {}
        action = EngineAction(
            "segment_closed",
            {
                "symbol": self.symbol,
                "segment_id": self.current_segment_id,
                "connection_id": self.segment_prefix,
                "reason": reason,
                "first_U": self.segment_first_u,
                "final_u": self.last_u,
                "applied_event_count": self.segment_applied_count,
                **checkpoint,
            },
        )
        self.current_segment_id = None
        self.book = None
        self.last_u = None
        self.segment_applied_count = 0
        self.segment_first_u = None
        self.last_event_time_ms = None
        self.last_transaction_time_ms = None
        return [action]

    def _reset_to_buffer(self, events: Sequence[DiffDepthEvent]) -> None:
        self.phase = "buffering"
        self.snapshot = None
        self.book = None
        self.last_u = None
        self.current_segment_id = None
        self.last_event_time_ms = None
        self.last_transaction_time_ms = None
        self.buffer = list(events)
        if len(self.buffer) > self.max_buffer_events:
            raise SequenceBufferOverflow(
                f"{self.symbol} buffered {len(self.buffer)} events during resync"
            )

    def end_connection(self, reason: str) -> list[EngineAction]:
        """Close any live segment and label an unfinished bootstrap."""

        actions = self._close_segment(reason)
        if self.buffer:
            actions.append(
                EngineAction(
                    "bootstrap_incomplete",
                    {
                        "symbol": self.symbol,
                        "connection_id": self.segment_prefix,
                        "reason": reason,
                        "buffered_event_count": len(self.buffer),
                        "first_wire_sequence": self.buffer[0].wire_sequence,
                        "last_wire_sequence": self.buffer[-1].wire_sequence,
                    },
                )
            )
        self._reset_to_buffer([])
        return actions


class AppendOnlyJsonl:
    """Append-only UTF-8 JSONL writer with configurable durability."""

    def __init__(self, path: Path, *, fsync_every: int = 1) -> None:
        if fsync_every < 0:
            raise ValueError("fsync_every must be non-negative")
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        self._fsync_every = fsync_every
        self._records_since_sync = 0
        self.count = 0

    def append(self, payload: Mapping[str, Any]) -> None:
        line = json.dumps(
            dict(payload),
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=True,
        )
        self._handle.write(line + "\n")
        self._handle.flush()
        self.count += 1
        self._records_since_sync += 1
        if self._fsync_every and self._records_since_sync >= self._fsync_every:
            os.fsync(self._handle.fileno())
            self._records_since_sync = 0

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()

    def __enter__(self) -> AppendOnlyJsonl:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically, retrying transient Windows/OneDrive locks."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(6):
            try:
                temp_path.replace(path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(min(0.05 * (2**attempt), 0.5))
    finally:
        temp_path.unlink(missing_ok=True)


class SymbolCaptureStorage:
    """Durable files and counters for one symbol capture run."""

    def __init__(
        self,
        run_dir: Path,
        *,
        symbol: str,
        run_id: str,
        fsync_every: int = 1,
    ) -> None:
        self.run_dir = run_dir
        self.symbol = symbol.upper()
        self.run_id = run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.wire = AppendOnlyJsonl(
            run_dir / "wire_events.jsonl",
            fsync_every=fsync_every,
        )
        self.snapshots = AppendOnlyJsonl(
            run_dir / "snapshots.jsonl",
            fsync_every=fsync_every,
        )
        self.timeline = AppendOnlyJsonl(
            run_dir / "timeline.jsonl",
            fsync_every=fsync_every,
        )
        self.actions = AppendOnlyJsonl(
            run_dir / "engine_actions.jsonl",
            fsync_every=fsync_every,
        )
        self.control = AppendOnlyJsonl(
            run_dir / "control.jsonl",
            fsync_every=fsync_every,
        )
        self.risk_snapshots = AppendOnlyJsonl(
            run_dir / "l2_risk_snapshots.jsonl",
            fsync_every=fsync_every,
        )
        self.public_trades = AppendOnlyJsonl(
            run_dir / "public_agg_trades.jsonl",
            fsync_every=fsync_every,
        )
        self.public_mark_prices = AppendOnlyJsonl(
            run_dir / "public_mark_prices.jsonl",
            fsync_every=fsync_every,
        )
        self.manifest_path = run_dir / "manifest.json"
        self.counters: dict[str, int] = {
            "wire_events": 0,
            "parsed_events": 0,
            "snapshots": 0,
            "events_applied": 0,
            "events_discarded": 0,
            "sequence_gaps": 0,
            "segments_started": 0,
            "segments_closed": 0,
            "parse_errors": 0,
            "connections": 0,
            "market_connections": 0,
            "market_coverage_gaps": 0,
            "market_parse_errors": 0,
            "resync_requests": 0,
            "risk_snapshots": 0,
            "public_agg_trades": 0,
            "public_agg_trade_id_discontinuities": 0,
            "public_agg_trade_duplicates_dropped": 0,
            "public_agg_trade_out_of_order_dropped": 0,
            "public_mark_price_updates": 0,
        }

    def append_wire(
        self,
        *,
        connection_id: str,
        wire_sequence: int,
        received_at_utc: str,
        received_monotonic_ns: int,
        raw_text: str,
    ) -> None:
        """Persist a frame before any JSON or schema parsing."""

        self.wire.append(
            {
                "schema_version": WIRE_SCHEMA_VERSION,
                "record_type": "wire_event",
                "symbol": self.symbol,
                "connection_id": connection_id,
                "wire_sequence": wire_sequence,
                "received_at_utc": received_at_utc,
                "received_monotonic_ns": received_monotonic_ns,
                "raw_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                "raw_text": raw_text,
            }
        )
        self.counters["wire_events"] += 1

    def append_event_timeline(self, event: DiffDepthEvent) -> None:
        self.timeline.append(event.to_timeline_record())
        self.counters["parsed_events"] += 1

    def append_snapshot_evidence(self, snapshot: DepthSnapshot) -> None:
        record = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            **snapshot.to_timeline_record(),
        }
        self.snapshots.append(record)
        self.counters["snapshots"] += 1

    def append_snapshot_timeline(self, snapshot: DepthSnapshot) -> None:
        """Record when a completed snapshot is offered to the engine."""

        self.timeline.append(snapshot.to_timeline_record())

    def append_connection_start(self, connection_id: str) -> None:
        self.timeline.append(
            {
                "schema_version": TIMELINE_SCHEMA_VERSION,
                "record_type": "connection_start",
                "symbol": self.symbol,
                "connection_id": connection_id,
            }
        )
        self.counters["connections"] += 1

    def append_connection_end(self, connection_id: str, reason: str) -> None:
        self.timeline.append(
            {
                "schema_version": TIMELINE_SCHEMA_VERSION,
                "record_type": "connection_end",
                "symbol": self.symbol,
                "connection_id": connection_id,
                "reason": reason,
            }
        )

    def append_actions(self, actions: Iterable[EngineAction]) -> None:
        for action in actions:
            self.actions.append(action.to_record())
            if action.kind == "event_applied":
                self.counters["events_applied"] += 1
            elif action.kind == "events_discarded_before_snapshot":
                self.counters["events_discarded"] += int(
                    action.details.get("count", 0)
                )
            elif action.kind == "sequence_gap":
                self.counters["sequence_gaps"] += 1
                self.counters["resync_requests"] += 1
            elif action.kind == "snapshot_behind_buffer":
                self.counters["resync_requests"] += 1
            elif action.kind == "segment_started":
                self.counters["segments_started"] += 1
            elif action.kind == "segment_closed":
                self.counters["segments_closed"] += 1

    def append_control(self, kind: str, details: Mapping[str, Any]) -> None:
        self.control.append(
            {
                "schema_version": ACTION_SCHEMA_VERSION,
                "kind": kind,
                "details": dict(details),
            }
        )
        if kind == "parse_error":
            self.counters["parse_errors"] += 1
        if kind == "snapshot_retry":
            self.counters["resync_requests"] += 1
        if kind in {"public_trade_parse_error", "public_mark_price_parse_error"}:
            self.counters["parse_errors"] += 1
        if kind == "public_trade_duplicate_dropped":
            self.counters["public_agg_trade_duplicates_dropped"] += 1
        if kind == "public_trade_out_of_order_dropped":
            self.counters["public_agg_trade_out_of_order_dropped"] += 1

    def append_risk_snapshot(self, record: Mapping[str, Any]) -> None:
        """Persist one bounded derivative of the current contiguous book."""

        self.risk_snapshots.append(dict(record))
        self.counters["risk_snapshots"] += 1

    def append_public_agg_trade(
        self,
        event: PublicAggTrade,
        *,
        id_discontinuity: bool,
        segment_id: str | None,
    ) -> None:
        """Persist a validated public trade with conservative continuity metadata."""

        record = event.to_record()
        record["run_id"] = self.run_id
        record["l2_segment_id"] = segment_id
        record["aggregate_id_discontinuity_observed"] = id_discontinuity
        self.public_trades.append(record)
        self.counters["public_agg_trades"] += 1
        if id_discontinuity:
            self.counters["public_agg_trade_id_discontinuities"] += 1

    def append_public_mark_price(self, event: PublicMarkPrice) -> None:
        """Persist one validated one-second mark-price update."""

        record = event.to_record()
        record["run_id"] = self.run_id
        self.public_mark_prices.append(record)
        self.counters["public_mark_price_updates"] += 1

    def write_manifest(self, payload: Mapping[str, Any]) -> None:
        atomic_write_json(
            self.manifest_path,
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "symbol": self.symbol,
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
                "counters": dict(self.counters),
                **dict(payload),
            },
        )

    def close(self) -> None:
        self.wire.close()
        self.snapshots.close()
        self.timeline.close()
        self.actions.close()
        self.control.close()
        self.risk_snapshots.close()
        self.public_trades.close()
        self.public_mark_prices.close()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DiffDepthError(
                    f"{path}:{line_number} is not valid JSON"
                ) from exc
            if not isinstance(decoded, dict):
                raise DiffDepthError(f"{path}:{line_number} is not an object")
            records.append(cast(dict[str, Any], decoded))
    return records


def replay_symbol_capture(run_dir: Path) -> ReplayVerification:
    """Replay one stored symbol timeline and compare every engine action."""

    timeline = _read_jsonl(run_dir / "timeline.jsonl")
    expected_records = _read_jsonl(run_dir / "engine_actions.jsonl")
    wire_records = _read_jsonl(run_dir / "wire_events.jsonl")
    manifest_payload = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(manifest_payload, dict):
        raise DiffDepthError("symbol manifest is not an object")
    symbol = str(manifest_payload.get("symbol", "")).upper()
    if not symbol:
        raise DiffDepthError("symbol manifest is missing symbol")

    hash_failures = 0
    for wire in wire_records:
        raw_text = wire.get("raw_text")
        expected_hash = wire.get("raw_sha256")
        if not isinstance(raw_text, str) or not isinstance(expected_hash, str):
            hash_failures += 1
            continue
        actual_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        if actual_hash != expected_hash:
            hash_failures += 1

    engine: DepthSequenceEngine | None = None
    replayed: list[dict[str, Any]] = []
    for record in timeline:
        record_type = record.get("record_type")
        if record_type == "connection_start":
            connection_id = str(record.get("connection_id", ""))
            if engine is not None:
                raise DiffDepthError("timeline starts a connection before ending one")
            engine = DepthSequenceEngine(
                symbol,
                segment_prefix=connection_id,
            )
            continue
        if engine is None:
            raise DiffDepthError(
                f"timeline record {record_type!r} has no active connection"
            )
        if record_type == "event":
            payload = record.get("payload")
            if not isinstance(payload, dict):
                raise DiffDepthError("event timeline record has no payload")
            event = parse_diff_depth_event(
                cast(dict[str, Any], payload),
                expected_symbol=symbol,
                connection_id=str(record.get("connection_id", "")),
                wire_sequence=int(record.get("wire_sequence", 0)),
                received_at_utc=str(record.get("received_at_utc", "")),
                received_monotonic_ns=int(
                    record.get("received_monotonic_ns", 0)
                ),
            )
            actions = engine.offer_event(event)
        elif record_type == "snapshot":
            payload = record.get("payload")
            if not isinstance(payload, dict):
                raise DiffDepthError("snapshot timeline record has no payload")
            snapshot = parse_depth_snapshot(
                cast(dict[str, Any], payload),
                symbol=symbol,
                connection_id=str(record.get("connection_id", "")),
                request_started_at_utc=str(
                    record.get("request_started_at_utc", "")
                ),
                received_at_utc=str(record.get("received_at_utc", "")),
                received_monotonic_ns=int(
                    record.get("received_monotonic_ns", 0)
                ),
                wire_sequence_seen=int(record.get("wire_sequence_seen", 0)),
                request_latency_ms=float(record.get("request_latency_ms", 0.0)),
            )
            actions = engine.offer_snapshot(snapshot)
        elif record_type == "connection_end":
            actions = engine.end_connection(str(record.get("reason", "")))
            engine = None
        else:
            raise DiffDepthError(f"unknown timeline record type: {record_type!r}")
        replayed.extend(action.to_record() for action in actions)
    if engine is not None:
        raise DiffDepthError("timeline ended without a connection_end record")
    return ReplayVerification(
        symbol=symbol,
        timeline_records=len(timeline),
        expected_actions=len(expected_records),
        replayed_actions=len(replayed),
        raw_events=len(wire_records),
        raw_hash_failures=hash_failures,
        actions_match=replayed == expected_records,
    )


def verification_to_dict(result: ReplayVerification) -> dict[str, Any]:
    return {
        **asdict(result),
        "passed": result.passed,
    }
