"""Durable Binance USD-M market-stream evidence.

This module is intentionally separate from diff-depth collection.  Binance
routes regular market traffic through ``/market`` and high-frequency order-book
traffic through ``/public``.  Keeping their storage and manifests independent
prevents a market-stream reconnect from being mistaken for an L2 sequence gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from neutralgrid.data.diff_depth import (
    AppendOnlyJsonl,
    PublicAggTrade,
    PublicMarkPrice,
    atomic_write_json,
)


MARKET_WIRE_SCHEMA_VERSION = "binance_usdm_market_wire_v1"
MARKET_KLINE_SCHEMA_VERSION = "binance_usdm_closed_kline_v1"
MARKET_MANIFEST_SCHEMA_VERSION = "binance_usdm_market_stream_manifest_v1"


class MarketStreamError(RuntimeError):
    """A market-stream payload or storage invariant failed."""


class MarketPayloadError(MarketStreamError):
    """A market-stream payload violated its documented shape."""


def _require_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if value is None or isinstance(value, bool):
        raise MarketPayloadError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MarketPayloadError(f"{field} must be an integer") from exc
    if parsed < 0:
        raise MarketPayloadError(f"{field} must be non-negative")
    return parsed


def _decimal(
    value: Any,
    *,
    field: str,
    allow_zero: bool,
) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise MarketPayloadError(f"{field} must be decimal") from exc
    if not parsed.is_finite():
        raise MarketPayloadError(f"{field} must be finite")
    if parsed < 0 or (not allow_zero and parsed == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise MarketPayloadError(f"{field} must be {qualifier}")
    return parsed


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


@dataclass(frozen=True)
class MarketKline:
    """One validated Binance kline update from the regular market route."""

    symbol: str
    event_time_ms: int
    open_time_ms: int
    close_time_ms: int
    interval: str
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    base_volume: Decimal
    quote_volume: Decimal
    trade_count: int
    taker_buy_base_volume: Decimal
    taker_buy_quote_volume: Decimal
    closed: bool
    connection_id: str
    wire_sequence: int
    received_at_utc: str
    received_monotonic_ns: int

    @property
    def dedup_key(self) -> tuple[str, str, int]:
        return (self.symbol, self.interval, self.open_time_ms)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": MARKET_KLINE_SCHEMA_VERSION,
            "record_type": "closed_kline" if self.closed else "open_kline_update",
            "symbol": self.symbol,
            "event_time_ms": self.event_time_ms,
            "open_time_ms": self.open_time_ms,
            "close_time_ms": self.close_time_ms,
            "interval": self.interval,
            "open": _decimal_text(self.open_price),
            "high": _decimal_text(self.high_price),
            "low": _decimal_text(self.low_price),
            "close": _decimal_text(self.close_price),
            "base_volume": _decimal_text(self.base_volume),
            "quote_volume": _decimal_text(self.quote_volume),
            "trade_count": self.trade_count,
            "taker_buy_base_volume": _decimal_text(self.taker_buy_base_volume),
            "taker_buy_quote_volume": _decimal_text(
                self.taker_buy_quote_volume
            ),
            "closed": self.closed,
            "connection_id": self.connection_id,
            "wire_sequence": self.wire_sequence,
            "received_at_utc": self.received_at_utc,
            "received_monotonic_ns": self.received_monotonic_ns,
        }


def parse_market_kline(
    payload: Mapping[str, Any],
    *,
    expected_symbol: str,
    expected_intervals: set[str],
    connection_id: str,
    wire_sequence: int,
    received_at_utc: str,
    received_monotonic_ns: int,
) -> MarketKline:
    """Validate a raw or combined-stream Binance kline message."""

    nested = payload.get("data", payload)
    if not isinstance(nested, Mapping) or nested.get("e") != "kline":
        raise MarketPayloadError("payload is not a kline event")
    symbol = str(nested.get("s", "")).upper()
    if symbol != expected_symbol.upper():
        raise MarketPayloadError(
            f"kline symbol mismatch: expected {expected_symbol.upper()}, got {symbol}"
        )
    raw_kline = nested.get("k")
    if not isinstance(raw_kline, Mapping):
        raise MarketPayloadError("kline payload.k must be an object")
    interval = str(raw_kline.get("i", ""))
    if interval not in expected_intervals:
        raise MarketPayloadError(f"unexpected kline interval: {interval!r}")
    closed = raw_kline.get("x")
    if not isinstance(closed, bool):
        raise MarketPayloadError("kline closed flag must be boolean")

    open_price = _decimal(raw_kline.get("o"), field="k.o", allow_zero=False)
    high_price = _decimal(raw_kline.get("h"), field="k.h", allow_zero=False)
    low_price = _decimal(raw_kline.get("l"), field="k.l", allow_zero=False)
    close_price = _decimal(raw_kline.get("c"), field="k.c", allow_zero=False)
    if high_price < max(open_price, low_price, close_price):
        raise MarketPayloadError("kline high is below an observed price")
    if low_price > min(open_price, high_price, close_price):
        raise MarketPayloadError("kline low is above an observed price")

    open_time_ms = _require_int(raw_kline, "t")
    close_time_ms = _require_int(raw_kline, "T")
    if close_time_ms < open_time_ms:
        raise MarketPayloadError("kline close time precedes open time")

    return MarketKline(
        symbol=symbol,
        event_time_ms=_require_int(nested, "E"),
        open_time_ms=open_time_ms,
        close_time_ms=close_time_ms,
        interval=interval,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
        base_volume=_decimal(raw_kline.get("v"), field="k.v", allow_zero=True),
        quote_volume=_decimal(raw_kline.get("q"), field="k.q", allow_zero=True),
        trade_count=_require_int(raw_kline, "n"),
        taker_buy_base_volume=_decimal(
            raw_kline.get("V"), field="k.V", allow_zero=True
        ),
        taker_buy_quote_volume=_decimal(
            raw_kline.get("Q"), field="k.Q", allow_zero=True
        ),
        closed=closed,
        connection_id=connection_id,
        wire_sequence=wire_sequence,
        received_at_utc=received_at_utc,
        received_monotonic_ns=received_monotonic_ns,
    )


class MarketStreamStorage:
    """Append-only evidence owned by one symbol/market-stream run."""

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
            run_dir / "wire_events.jsonl", fsync_every=fsync_every
        )
        self.trades = AppendOnlyJsonl(
            run_dir / "public_agg_trades.jsonl", fsync_every=fsync_every
        )
        self.mark_prices = AppendOnlyJsonl(
            run_dir / "public_mark_prices.jsonl", fsync_every=fsync_every
        )
        self.closed_klines = AppendOnlyJsonl(
            run_dir / "closed_klines.jsonl", fsync_every=fsync_every
        )
        self.control = AppendOnlyJsonl(
            run_dir / "control.jsonl", fsync_every=fsync_every
        )
        self.manifest_path = run_dir / "manifest.json"
        self.counters: dict[str, int] = {
            "wire_events": 0,
            "connections": 0,
            "coverage_gaps": 0,
            "parse_errors": 0,
            "unexpected_events": 0,
            "public_agg_trades": 0,
            "agg_trade_duplicates_dropped": 0,
            "agg_trade_out_of_order_dropped": 0,
            "agg_trade_id_discontinuities": 0,
            "public_mark_price_updates": 0,
            "mark_price_duplicates_dropped": 0,
            "mark_price_out_of_order_dropped": 0,
            "non_final_kline_updates": 0,
            "closed_klines": 0,
            "closed_kline_duplicates_dropped": 0,
            "closed_kline_conflicts": 0,
        }

    def append_wire(
        self,
        *,
        connection_id: str,
        wire_sequence: int,
        received_at_utc: str,
        received_monotonic_ns: int,
        raw_text: str,
        raw_sha256: str,
    ) -> None:
        self.wire.append(
            {
                "schema_version": MARKET_WIRE_SCHEMA_VERSION,
                "record_type": "wire_event",
                "symbol": self.symbol,
                "connection_id": connection_id,
                "wire_sequence": wire_sequence,
                "received_at_utc": received_at_utc,
                "received_monotonic_ns": received_monotonic_ns,
                "raw_sha256": raw_sha256,
                "raw_text": raw_text,
            }
        )
        self.counters["wire_events"] += 1

    def append_trade(self, event: PublicAggTrade) -> None:
        self.trades.append(event.to_record())
        self.counters["public_agg_trades"] += 1

    def append_mark_price(self, event: PublicMarkPrice) -> None:
        self.mark_prices.append(event.to_record())
        self.counters["public_mark_price_updates"] += 1

    def append_closed_kline(self, event: MarketKline) -> None:
        if not event.closed:
            raise MarketStreamError("non-final kline cannot enter closed-kline storage")
        self.closed_klines.append(event.to_record())
        self.counters["closed_klines"] += 1

    def append_control(self, kind: str, details: Mapping[str, Any]) -> None:
        self.control.append(
            {
                "schema_version": MARKET_MANIFEST_SCHEMA_VERSION,
                "record_type": "control",
                "kind": kind,
                **dict(details),
            }
        )

    def write_manifest(self, payload: Mapping[str, Any]) -> None:
        atomic_write_json(
            self.manifest_path,
            {
                "schema_version": MARKET_MANIFEST_SCHEMA_VERSION,
                "symbol": self.symbol,
                "run_id": self.run_id,
                "counters": dict(self.counters),
                **dict(payload),
            },
        )

    def close(self) -> None:
        self.wire.close()
        self.trades.close()
        self.mark_prices.close()
        self.closed_klines.close()
        self.control.close()
