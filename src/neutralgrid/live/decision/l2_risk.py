"""Sequence-linked L2 risk snapshots for live bot evaluation.

Raw diff-depth frames remain the event-complete evidence source.  The records
defined here are bounded, periodic derivatives of a verified local order book
so a scanner tick can consume current liquidity without replaying an unbounded
capture.  Removal/addition metrics are explicitly book-update proxies: without
the synchronized trade stream they must not be called cancellations or sweeps.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence, cast

from neutralgrid.data.diff_depth import (
    MANIFEST_SCHEMA_VERSION,
    DiffDepthEvent,
    LocalOrderBook,
)


L2_RISK_SCHEMA_VERSION = "neutralgrid_l2_risk_snapshot_v1"


class L2RiskError(RuntimeError):
    """A persisted L2 risk stream is missing, stale, or invalid."""


@dataclass(frozen=True)
class L2StreamRef:
    feature_path: Path
    symbol: str
    run_id: str | None = None
    strategy_id: str | None = None
    manifest_path: Path | None = None
    public_trade_path: Path | None = None
    max_age_seconds: float = 15.0
    history_window_seconds: float = 300.0
    deterioration_min_duration_seconds: float = 60.0
    deterioration_min_observations: int = 3
    deterioration_fraction: float = 0.80


@dataclass(frozen=True)
class PositionNormalizedL2Risk:
    source: str
    captured_at_utc: datetime
    age_seconds: float
    run_id: str
    segment_id: str
    final_update_id: int
    wire_sequence: int
    snapshot_count: int
    history_coverage_seconds: float
    history_max_gap_seconds: float | None
    best_bid: float
    best_ask: float
    mid_price: float
    spread_bps: float
    spread_median_bps: float
    spread_p90_bps: float
    spread_current_to_median: float | None
    book_imbalance: float
    book_imbalance_median: float
    bid_depth_notional_usdt: float
    ask_depth_notional_usdt: float
    position_side: str | None
    position_size_base: float | None
    position_notional_usdt: float | None
    exit_book_side: str | None
    exit_depth_notional_usdt: float | None
    exit_depth_to_position_ratio: float | None
    exit_fill_ratio: float | None
    expected_exit_vwap: float | None
    expected_exit_impact_bps: float | None
    exit_depth_current_to_median: float | None
    exit_side_removed_notional_usdt: float | None
    exit_side_added_notional_usdt: float | None
    exit_side_removal_to_addition_ratio: float | None
    top_n_levels: int
    scope_note: str = (
        "Sequence-linked public diff-depth derivative. Book removals/additions are "
        "proxies, not classified cancellations or sweeps; private fills are separate."
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["captured_at_utc"] = self.captured_at_utc.astimezone(
            timezone.utc
        ).isoformat()
        return payload


class L2IntervalAccumulator:
    """Accumulate absolute-book update deltas between derived snapshots."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.bid_added_notional = Decimal("0")
        self.bid_removed_notional = Decimal("0")
        self.ask_added_notional = Decimal("0")
        self.ask_removed_notional = Decimal("0")
        self.bid_update_count = 0
        self.ask_update_count = 0
        self.bid_delete_count = 0
        self.ask_delete_count = 0

    def observe(self, book: LocalOrderBook, event: DiffDepthEvent) -> None:
        self._observe_side(book.bids, event.bids, side="bid")
        self._observe_side(book.asks, event.asks, side="ask")

    def _observe_side(
        self,
        current: Mapping[Decimal, Decimal],
        updates: Sequence[tuple[Decimal, Decimal]],
        *,
        side: str,
    ) -> None:
        for price, new_quantity in updates:
            old_quantity = current.get(price, Decimal("0"))
            delta_notional = price * (new_quantity - old_quantity)
            if side == "bid":
                self.bid_update_count += 1
                if new_quantity == 0 and old_quantity > 0:
                    self.bid_delete_count += 1
                if delta_notional >= 0:
                    self.bid_added_notional += delta_notional
                else:
                    self.bid_removed_notional -= delta_notional
            else:
                self.ask_update_count += 1
                if new_quantity == 0 and old_quantity > 0:
                    self.ask_delete_count += 1
                if delta_notional >= 0:
                    self.ask_added_notional += delta_notional
                else:
                    self.ask_removed_notional -= delta_notional

    def to_record(self) -> dict[str, Any]:
        return {
            "bid_added_notional_usdt": float(self.bid_added_notional),
            "bid_removed_notional_usdt": float(self.bid_removed_notional),
            "ask_added_notional_usdt": float(self.ask_added_notional),
            "ask_removed_notional_usdt": float(self.ask_removed_notional),
            "bid_update_count": self.bid_update_count,
            "ask_update_count": self.ask_update_count,
            "bid_delete_count": self.bid_delete_count,
            "ask_delete_count": self.ask_delete_count,
        }


def build_l2_risk_record(
    *,
    book: LocalOrderBook,
    accumulator: L2IntervalAccumulator,
    symbol: str,
    run_id: str,
    connection_id: str,
    segment_id: str,
    captured_at_utc: str,
    wire_sequence: int,
    final_update_id: int,
    top_n: int,
    exchange_event_time_ms: int | None = None,
    exchange_transaction_time_ms: int | None = None,
) -> dict[str, Any]:
    """Build one JSON-safe derivative from a contiguous local book."""

    if top_n < 1:
        raise L2RiskError(f"top_n must be >= 1, got {top_n}")
    best_bid = book.best_bid
    best_ask = book.best_ask
    if best_bid is None or best_ask is None or best_bid >= best_ask:
        raise L2RiskError("cannot derive risk from an empty or crossed book")
    mid = (best_bid + best_ask) / Decimal("2")
    spread_bps = (best_ask - best_bid) / mid * Decimal("10000")
    bid_prices = sorted(book.bids, reverse=True)[:top_n]
    ask_prices = sorted(book.asks)[:top_n]
    bid_notional = sum((price * book.bids[price] for price in bid_prices), Decimal("0"))
    ask_notional = sum((price * book.asks[price] for price in ask_prices), Decimal("0"))
    total_notional = bid_notional + ask_notional
    imbalance = (
        (bid_notional - ask_notional) / total_notional
        if total_notional > 0
        else Decimal("0")
    )

    return {
        "schema_version": L2_RISK_SCHEMA_VERSION,
        "record_type": "l2_risk_snapshot",
        "symbol": symbol.upper(),
        "run_id": run_id,
        "connection_id": connection_id,
        "segment_id": segment_id,
        "captured_at_utc": captured_at_utc,
        "wire_sequence": int(wire_sequence),
        "final_update_id": int(final_update_id),
        "exchange_event_time_ms": exchange_event_time_ms,
        "exchange_transaction_time_ms": exchange_transaction_time_ms,
        "book_sha256": book.digest(),
        "top_n_levels": int(top_n),
        "best_bid": float(best_bid),
        "best_ask": float(best_ask),
        "mid_price": float(mid),
        "spread_bps": float(spread_bps),
        "bid_depth_notional_usdt": float(bid_notional),
        "ask_depth_notional_usdt": float(ask_notional),
        "book_imbalance": float(imbalance),
        "top_bids": [
            [format(price, "f"), format(book.bids[price], "f")]
            for price in bid_prices
        ],
        "top_asks": [
            [format(price, "f"), format(book.asks[price], "f")]
            for price in ask_prices
        ],
        "interval_book_update_proxies": accumulator.to_record(),
    }


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise L2RiskError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise L2RiskError(f"{field} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise L2RiskError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _finite_float(record: Mapping[str, Any], key: str, *, positive: bool = False) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise L2RiskError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise L2RiskError(f"{key} must be {qualifier}")
    return number


def _positive_int(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise L2RiskError(f"{key} must be a positive integer")
    return value


def _parse_levels(record: Mapping[str, Any], key: str) -> tuple[tuple[float, float], ...]:
    raw = record.get(key)
    if not isinstance(raw, list):
        raise L2RiskError(f"{key} must be a list")
    levels: list[tuple[float, float]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, list) or len(item) != 2:
            raise L2RiskError(f"{key}[{index}] must be a price/quantity pair")
        try:
            price = float(item[0])
            quantity = float(item[1])
        except (TypeError, ValueError) as exc:
            raise L2RiskError(f"{key}[{index}] is not numeric") from exc
        if not math.isfinite(price) or not math.isfinite(quantity) or price <= 0 or quantity < 0:
            raise L2RiskError(f"{key}[{index}] has invalid price/quantity")
        levels.append((price, quantity))
    return tuple(levels)


def _read_tail_records(path: Path, *, max_records: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise L2RiskError(f"L2 risk stream does not exist: {path}")
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
            raise L2RiskError(f"invalid JSONL tail in {path}") from exc
        if not isinstance(decoded, dict):
            raise L2RiskError(f"L2 JSONL record is not an object in {path}")
        records.append(cast(dict[str, Any], decoded))
    if not records:
        raise L2RiskError(f"L2 risk stream is empty: {path}")
    return records


def _validate_record(record: Mapping[str, Any], ref: L2StreamRef) -> dict[str, Any]:
    if record.get("schema_version") != L2_RISK_SCHEMA_VERSION:
        raise L2RiskError("unsupported L2 risk schema")
    symbol = str(record.get("symbol", "")).upper()
    if symbol != ref.symbol.upper():
        raise L2RiskError(f"L2 symbol mismatch: expected {ref.symbol}, got {symbol}")
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise L2RiskError("L2 run_id is missing")
    if ref.run_id is not None and run_id != ref.run_id:
        raise L2RiskError(f"L2 run_id mismatch: expected {ref.run_id}, got {run_id}")
    segment_id = record.get("segment_id")
    if not isinstance(segment_id, str) or not segment_id:
        raise L2RiskError("L2 segment_id is missing")
    captured_at = _parse_utc(record.get("captured_at_utc"), field="captured_at_utc")
    exchange_event_time_ms = record.get("exchange_event_time_ms")
    if exchange_event_time_ms is not None and (
        isinstance(exchange_event_time_ms, bool)
        or not isinstance(exchange_event_time_ms, int)
        or exchange_event_time_ms < 0
    ):
        raise L2RiskError("exchange_event_time_ms must be a non-negative integer or null")
    observation_at = (
        datetime.fromtimestamp(exchange_event_time_ms / 1000.0, tz=timezone.utc)
        if isinstance(exchange_event_time_ms, int)
        else captured_at
    )
    best_bid = _finite_float(record, "best_bid", positive=True)
    best_ask = _finite_float(record, "best_ask", positive=True)
    mid = _finite_float(record, "mid_price", positive=True)
    if not best_bid < mid < best_ask:
        raise L2RiskError("L2 best bid/mid/ask ordering is invalid")
    validated = dict(record)
    validated["captured_at"] = captured_at
    validated["observation_at"] = observation_at
    validated["top_bids_parsed"] = _parse_levels(record, "top_bids")
    validated["top_asks_parsed"] = _parse_levels(record, "top_asks")
    _positive_int(record, "wire_sequence")
    _positive_int(record, "final_update_id")
    _positive_int(record, "top_n_levels")
    for key in (
        "spread_bps",
        "bid_depth_notional_usdt",
        "ask_depth_notional_usdt",
        "book_imbalance",
    ):
        _finite_float(record, key)
    return validated


def load_validated_l2_manifest(
    ref: L2StreamRef,
    *,
    now: datetime,
) -> dict[str, Any] | None:
    """Validate and return the collector manifest when one is configured."""

    path = ref.manifest_path
    if path is None:
        return None
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise L2RiskError(f"cannot read L2 symbol manifest {path}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise L2RiskError("L2 symbol manifest is not an object")
    if decoded.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise L2RiskError("unsupported L2 symbol manifest schema")
    symbol = str(decoded.get("symbol", "")).upper()
    if symbol != ref.symbol.upper():
        raise L2RiskError(
            f"L2 manifest symbol mismatch: expected {ref.symbol}, got {symbol}"
        )
    run_id = decoded.get("run_id")
    if ref.run_id is not None and run_id != ref.run_id:
        raise L2RiskError(
            f"L2 manifest run_id mismatch: expected {ref.run_id}, got {run_id}"
        )
    if decoded.get("status") != "running" or decoded.get("current_phase") != "live":
        raise L2RiskError(
            "L2 collector is not in a running contiguous live segment: "
            f"status={decoded.get('status')!r}, phase={decoded.get('current_phase')!r}"
        )
    updated_at = _parse_utc(decoded.get("updated_at_utc"), field="manifest.updated_at_utc")
    age = (now.astimezone(timezone.utc) - updated_at).total_seconds()
    if age < -5 or age > ref.max_age_seconds:
        raise L2RiskError(
            f"L2 manifest heartbeat is stale or future-dated: age={age:.1f}s"
        )
    return cast(dict[str, Any], decoded)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _exit_metrics(
    record: Mapping[str, Any],
    *,
    position_size_base: float | None,
    position_size_usdt: float | None,
) -> dict[str, Any]:
    mid = float(record["mid_price"])
    signed_base = position_size_base
    if signed_base is None and position_size_usdt is not None:
        signed_base = position_size_usdt / mid
    if signed_base is None or signed_base == 0:
        return {
            "position_side": None,
            "position_size_base": signed_base,
            "position_notional_usdt": 0.0 if signed_base == 0 else None,
            "exit_book_side": None,
            "exit_depth_notional_usdt": None,
            "exit_depth_to_position_ratio": None,
            "exit_fill_ratio": None,
            "expected_exit_vwap": None,
            "expected_exit_impact_bps": None,
        }

    long_position = signed_base > 0
    levels_key = "top_bids_parsed" if long_position else "top_asks_parsed"
    levels = cast(tuple[tuple[float, float], ...], record[levels_key])
    required_qty = abs(signed_base)
    remaining = required_qty
    filled_qty = 0.0
    filled_notional = 0.0
    available_notional = 0.0
    for price, quantity in levels:
        available_notional += price * quantity
        take = min(quantity, remaining)
        if take > 0:
            filled_qty += take
            filled_notional += price * take
            remaining -= take
        if remaining <= 0:
            continue
    position_notional = (
        abs(position_size_usdt)
        if position_size_usdt is not None
        else required_qty * mid
    )
    fill_ratio = min(filled_qty / required_qty, 1.0)
    vwap = filled_notional / filled_qty if filled_qty > 0 else None
    impact = None
    if vwap is not None:
        impact = (
            (mid - vwap) / mid * 10000.0
            if long_position
            else (vwap - mid) / mid * 10000.0
        )
    return {
        "position_side": "long" if long_position else "short",
        "position_size_base": signed_base,
        "position_notional_usdt": position_notional,
        "exit_book_side": "bids" if long_position else "asks",
        "exit_depth_notional_usdt": available_notional,
        "exit_depth_to_position_ratio": (
            available_notional / position_notional if position_notional > 0 else None
        ),
        "exit_fill_ratio": fill_ratio,
        "expected_exit_vwap": vwap,
        "expected_exit_impact_bps": impact,
    }


def load_position_normalized_l2_risk(
    ref: L2StreamRef,
    *,
    now: datetime,
    position_size_base: float | None,
    position_size_usdt: float | None,
) -> PositionNormalizedL2Risk:
    """Load fresh same-segment records and calculate exit-liquidity evidence."""

    if now.tzinfo is None:
        raise L2RiskError("now must be timezone-aware")
    if ref.max_age_seconds <= 0 or ref.history_window_seconds <= 0:
        raise L2RiskError("L2 stream age/window settings must be positive")
    window = load_validated_l2_window(ref, now=now)
    return position_normalized_l2_risk_from_validated_window(
        window,
        now=now,
        position_size_base=position_size_base,
        position_size_usdt=position_size_usdt,
    )


def position_normalized_l2_risk_from_validated_window(
    window: Sequence[Mapping[str, Any]],
    *,
    now: datetime,
    position_size_base: float | None,
    position_size_usdt: float | None,
) -> PositionNormalizedL2Risk:
    """Calculate exit evidence from one already validated immutable window."""

    if now.tzinfo is None:
        raise L2RiskError("now must be timezone-aware")
    if not window:
        raise L2RiskError("validated L2 history window is empty")
    latest = window[-1]
    latest_at = cast(datetime, latest["captured_at"])
    age_seconds = (now.astimezone(timezone.utc) - latest_at).total_seconds()
    segment_id = str(latest["segment_id"])
    spreads = [float(record["spread_bps"]) for record in window]
    imbalances = [float(record["book_imbalance"]) for record in window]
    timestamps = [cast(datetime, record["captured_at"]).timestamp() for record in window]
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]

    current_exit = _exit_metrics(
        latest,
        position_size_base=position_size_base,
        position_size_usdt=position_size_usdt,
    )
    exit_depth_series: list[float] = []
    for record in window:
        metrics = _exit_metrics(
            record,
            position_size_base=position_size_base,
            position_size_usdt=position_size_usdt,
        )
        depth = metrics["exit_depth_notional_usdt"]
        if isinstance(depth, float):
            exit_depth_series.append(depth)
    current_exit_depth = current_exit["exit_depth_notional_usdt"]
    median_exit_depth = median(exit_depth_series) if exit_depth_series else None
    exit_depth_current_to_median = (
        current_exit_depth / median_exit_depth
        if isinstance(current_exit_depth, float)
        and median_exit_depth is not None
        and median_exit_depth > 0
        else None
    )

    proxy = latest.get("interval_book_update_proxies")
    proxy_map = proxy if isinstance(proxy, dict) else {}
    exit_prefix = "bid" if current_exit["exit_book_side"] == "bids" else "ask"
    removed = None
    added = None
    removal_ratio = None
    if current_exit["exit_book_side"] is not None:
        removed_value = proxy_map.get(f"{exit_prefix}_removed_notional_usdt")
        added_value = proxy_map.get(f"{exit_prefix}_added_notional_usdt")
        if isinstance(removed_value, (int, float)) and math.isfinite(float(removed_value)):
            removed = float(removed_value)
        if isinstance(added_value, (int, float)) and math.isfinite(float(added_value)):
            added = float(added_value)
        if removed is not None and added is not None and added > 0:
            removal_ratio = removed / added

    spread_median = median(spreads)
    return PositionNormalizedL2Risk(
        source="binance_usdm_diff_depth_derived",
        captured_at_utc=latest_at,
        age_seconds=max(age_seconds, 0.0),
        run_id=str(latest["run_id"]),
        segment_id=segment_id,
        final_update_id=int(latest["final_update_id"]),
        wire_sequence=int(latest["wire_sequence"]),
        snapshot_count=len(window),
        history_coverage_seconds=(max(timestamps) - min(timestamps)) if timestamps else 0.0,
        history_max_gap_seconds=max(gaps) if gaps else None,
        best_bid=float(latest["best_bid"]),
        best_ask=float(latest["best_ask"]),
        mid_price=float(latest["mid_price"]),
        spread_bps=float(latest["spread_bps"]),
        spread_median_bps=spread_median,
        spread_p90_bps=_percentile(spreads, 0.90),
        spread_current_to_median=(
            float(latest["spread_bps"]) / spread_median if spread_median > 0 else None
        ),
        book_imbalance=float(latest["book_imbalance"]),
        book_imbalance_median=median(imbalances),
        bid_depth_notional_usdt=float(latest["bid_depth_notional_usdt"]),
        ask_depth_notional_usdt=float(latest["ask_depth_notional_usdt"]),
        position_side=cast(str | None, current_exit["position_side"]),
        position_size_base=cast(float | None, current_exit["position_size_base"]),
        position_notional_usdt=cast(float | None, current_exit["position_notional_usdt"]),
        exit_book_side=cast(str | None, current_exit["exit_book_side"]),
        exit_depth_notional_usdt=cast(float | None, current_exit_depth),
        exit_depth_to_position_ratio=cast(
            float | None, current_exit["exit_depth_to_position_ratio"]
        ),
        exit_fill_ratio=cast(float | None, current_exit["exit_fill_ratio"]),
        expected_exit_vwap=cast(float | None, current_exit["expected_exit_vwap"]),
        expected_exit_impact_bps=cast(
            float | None, current_exit["expected_exit_impact_bps"]
        ),
        exit_depth_current_to_median=exit_depth_current_to_median,
        exit_side_removed_notional_usdt=removed,
        exit_side_added_notional_usdt=added,
        exit_side_removal_to_addition_ratio=removal_ratio,
        top_n_levels=int(latest["top_n_levels"]),
    )


def load_validated_l2_window(
    ref: L2StreamRef,
    *,
    now: datetime,
    max_records: int = 512,
) -> tuple[dict[str, Any], ...]:
    """Return a fresh same-segment validated window for derived evidence."""

    if now.tzinfo is None:
        raise L2RiskError("now must be timezone-aware")
    if ref.max_age_seconds <= 0 or ref.history_window_seconds <= 0:
        raise L2RiskError("L2 stream age/window settings must be positive")
    if max_records < 1:
        raise L2RiskError("max_records must be positive")
    load_validated_l2_manifest(ref, now=now)
    records = [
        _validate_record(record, ref)
        for record in _read_tail_records(ref.feature_path, max_records=max_records)
    ]
    latest = records[-1]
    latest_at = cast(datetime, latest["captured_at"])
    age_seconds = (now.astimezone(timezone.utc) - latest_at).total_seconds()
    if age_seconds < -5:
        raise L2RiskError(f"L2 risk snapshot is {abs(age_seconds):.1f}s in the future")
    if age_seconds > ref.max_age_seconds:
        raise L2RiskError(
            f"L2 risk snapshot is stale: age={age_seconds:.1f}s > {ref.max_age_seconds:.1f}s"
        )
    segment_id = str(latest["segment_id"])
    window_start = latest_at.timestamp() - ref.history_window_seconds
    window = tuple(
        record
        for record in records
        if record["segment_id"] == segment_id
        and cast(datetime, record["captured_at"]).timestamp() >= window_start
        and cast(datetime, record["captured_at"]) <= latest_at
    )
    if not window:
        raise L2RiskError("validated L2 history window is empty")
    captured_times = [cast(datetime, record["captured_at"]) for record in window]
    observation_times = [cast(datetime, record["observation_at"]) for record in window]
    wire_sequences = [int(record["wire_sequence"]) for record in window]
    final_update_ids = [int(record["final_update_id"]) for record in window]
    if any(current < previous for previous, current in zip(captured_times, captured_times[1:])):
        raise L2RiskError("L2 captured timestamps regress within the active segment")
    if any(current < previous for previous, current in zip(observation_times, observation_times[1:])):
        raise L2RiskError("L2 exchange observation timestamps regress within the active segment")
    if any(current < previous for previous, current in zip(wire_sequences, wire_sequences[1:])):
        raise L2RiskError("L2 wire sequence regresses within the active segment")
    if any(current < previous for previous, current in zip(final_update_ids, final_update_ids[1:])):
        raise L2RiskError("L2 final update ID regresses within the active segment")
    latest_observation_at = observation_times[-1]
    observation_age_seconds = (
        now.astimezone(timezone.utc) - latest_observation_at
    ).total_seconds()
    if observation_age_seconds < -5:
        raise L2RiskError(
            "L2 exchange observation timestamp is future-dated: "
            f"age={observation_age_seconds:.1f}s"
        )
    return window
