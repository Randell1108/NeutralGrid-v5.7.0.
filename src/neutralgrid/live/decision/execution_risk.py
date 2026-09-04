"""Event-linked execution and liquidity deterioration evidence.

This module joins three observational streams without changing live verdicts:

* contiguous sequence-derived L2 snapshots;
* public aggregate trades captured on the same WebSocket connection; and
* exact-strategy private fills/order updates.

The classifications are intentionally conservative. Public book removals that
cannot be aligned to aggressive trades are called *unexplained removals*, not
cancellations. A sweep is a trade-aligned removal proxy, not queue-level proof.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Sequence, cast

from neutralgrid.data.diff_depth import PUBLIC_AGG_TRADE_SCHEMA_VERSION
from neutralgrid.live.decision.l2_risk import (
    L2RiskError,
    L2StreamRef,
    load_validated_l2_manifest,
    load_validated_l2_window,
)
from neutralgrid.live.decision.private_events import (
    PrivateEventError,
    PrivateEventStreamRef,
    load_validated_private_event_window,
)


class ExecutionRiskError(RuntimeError):
    """Required L2 execution-risk evidence is invalid or unavailable."""


@dataclass(frozen=True)
class LatestFillEstimate:
    event_time_utc: datetime
    trade_id: str
    order_id: str
    side: str
    fill_price: float
    fill_qty: float
    reference_mid: float
    reference_lag_seconds: float
    estimated_slippage_bps: float
    adverse_selection_5s_bps: float | None
    adverse_selection_5s_actual_horizon_seconds: float | None
    adverse_selection_30s_bps: float | None
    adverse_selection_30s_actual_horizon_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["event_time_utc"] = self.event_time_utc.astimezone(
            timezone.utc
        ).isoformat()
        return payload


@dataclass(frozen=True)
class ExecutionRiskEvidence:
    source: str
    captured_at_utc: datetime
    l2_run_id: str
    l2_segment_id: str
    strategy_id: str | None
    l2_snapshot_count: int
    history_coverage_seconds: float
    position_side: str | None
    exit_book_side: str | None
    position_notional_usdt: float | None
    current_spread_bps: float
    baseline_spread_median_bps: float | None
    current_exit_depth_notional_usdt: float | None
    baseline_exit_depth_median_usdt: float | None
    exit_depth_current_to_baseline: float | None
    exit_side_imbalance: float | None
    recent_observation_count: int
    recent_spread_worse_fraction: float | None
    recent_exit_depth_worse_fraction: float | None
    sustained_spread_deterioration: bool | None
    sustained_exit_depth_deterioration: bool | None
    sustained_joint_deterioration: bool | None
    temporary_joint_deterioration: bool | None
    joint_deterioration_trailing_duration_seconds: float | None
    deterioration_min_duration_seconds: float
    deterioration_min_observations: int
    deterioration_fraction_threshold: float
    liquidity_state: str
    public_trade_identity_status: str
    public_trade_status: str
    public_trade_count: int
    public_trade_notional_usdt: float
    aggressive_exit_side_trade_notional_usdt: float | None
    aggressive_exit_side_trade_to_position_ratio: float | None
    exit_side_removed_notional_usdt: float | None
    exit_side_removed_to_position_ratio: float | None
    exit_side_added_notional_usdt: float | None
    exit_side_added_to_position_ratio: float | None
    exit_side_net_withdrawal_notional_usdt: float | None
    exit_side_net_withdrawal_to_position_ratio: float | None
    trade_aligned_removal_proxy_usdt: float | None
    trade_aligned_removal_to_position_ratio: float | None
    unexplained_removal_proxy_usdt: float | None
    unexplained_removal_to_position_ratio: float | None
    refill_proxy_usdt: float | None
    refill_to_position_ratio: float | None
    sweep_proxy_interval_count: int | None
    private_event_status: str
    private_fill_count: int
    private_order_update_count: int
    private_cancel_update_count: int
    private_cancel_update_fraction: float | None
    fill_l2_linked_count: int
    fill_l2_unlinked_count: int
    mean_estimated_slippage_bps: float | None
    p90_estimated_slippage_bps: float | None
    mean_adverse_selection_5s_bps: float | None
    mean_adverse_selection_30s_bps: float | None
    latest_fill_estimate: LatestFillEstimate | None
    scope_notes: tuple[str, ...] = (
        "No field in this record changes CONTINUE/ADJUST/END.",
        "Trade-aligned removal and refill are interval proxies, not queue-level proof.",
        "Unexplained public removal is not labelled cancellation.",
        "Public trades are attributed through the exact collector target; exchange market trades do not carry strategy IDs.",
        "Fill slippage uses the nearest preceding bounded L2 derivative, not an exact raw-book replay.",
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["captured_at_utc"] = self.captured_at_utc.astimezone(
            timezone.utc
        ).isoformat()
        payload["scope_notes"] = list(self.scope_notes)
        payload["latest_fill_estimate"] = (
            self.latest_fill_estimate.to_dict()
            if self.latest_fill_estimate is not None
            else None
        )
        return payload


def _read_tail_jsonl(path: Path, *, max_records: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ExecutionRiskError(f"public aggregate-trade stream is missing: {path}")
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        buffer = b""
        while position > 0 and buffer.count(b"\n") <= max_records:
            block = min(position, 65536)
            position -= block
            handle.seek(position)
            buffer = handle.read(block) + buffer
    records: list[dict[str, Any]] = []
    for line in [item for item in buffer.splitlines() if item.strip()][-max_records:]:
        try:
            decoded = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExecutionRiskError(f"invalid public-trade JSONL in {path}") from exc
        if not isinstance(decoded, dict):
            raise ExecutionRiskError(f"public-trade JSONL record is not an object in {path}")
        records.append(cast(dict[str, Any], decoded))
    return records


def _finite_float(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ExecutionRiskError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExecutionRiskError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ExecutionRiskError(f"{field} must be {qualifier}")
    return number


def _nonnegative_int(value: Any, *, field: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExecutionRiskError(f"{field} must be an integer")
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ExecutionRiskError(f"{field} must be {qualifier}")
    return value


def _parse_utc_text(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ExecutionRiskError(f"{field} must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExecutionRiskError(f"{field} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ExecutionRiskError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _load_public_trades_validated(
    ref: L2StreamRef,
    *,
    segment_id: str,
    start: datetime,
    end: datetime,
    now: datetime,
) -> tuple[list[dict[str, Any]], str]:
    if ref.public_trade_path is None:
        return [], "unavailable:no_public_trade_path"
    if ref.strategy_id is None or not ref.strategy_id.strip():
        return [], "unavailable:public_trade_strategy_identity_missing"
    try:
        manifest = load_validated_l2_manifest(ref, now=now)
    except L2RiskError as exc:
        return [], f"unavailable:{exc}"
    status = "available"
    if manifest is not None:
        if manifest.get("collect_agg_trades") is not True:
            return [], "unavailable:public_trade_collection_disabled"
        if manifest.get("trade_subscription_acknowledged") is not True:
            return [], "unavailable:public_trade_subscription_unacknowledged"
        counters_raw = manifest.get("counters")
        if not isinstance(counters_raw, dict):
            return [], "unavailable:public_trade_manifest_counters_missing"
        anomaly_keys = (
            "public_agg_trade_id_discontinuities",
            "public_agg_trade_duplicates_dropped",
            "public_agg_trade_out_of_order_dropped",
        )
        anomaly_count = 0
        for key in anomaly_keys:
            value = counters_raw.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return [], f"unavailable:invalid_public_trade_counter:{key}"
            anomaly_count += value
        if anomaly_count:
            status = "available_with_observed_transport_anomalies"
    try:
        raw = _read_tail_jsonl(ref.public_trade_path, max_records=8192)
    except ExecutionRiskError as exc:
        return [], f"unavailable:{exc}"
    accepted: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    last_wire_sequence: int | None = None
    last_trade_time_ms: int | None = None
    for record in raw:
        if record.get("schema_version") != PUBLIC_AGG_TRADE_SCHEMA_VERSION:
            return [], "unavailable:unsupported_public_trade_schema"
        if str(record.get("symbol", "")).upper() != ref.symbol.upper():
            return [], "unavailable:public_trade_symbol_mismatch"
        if ref.run_id is not None and record.get("run_id") != ref.run_id:
            return [], "unavailable:public_trade_run_id_mismatch"
        if record.get("l2_segment_id") != segment_id:
            continue
        aggregate_id = _nonnegative_int(
            record.get("aggregate_trade_id"),
            field="public_trade.aggregate_trade_id",
        )
        if aggregate_id in seen_ids:
            return [], "unavailable:duplicate_public_aggregate_trade_id"
        seen_ids.add(aggregate_id)
        trade_time_ms = _nonnegative_int(
            record.get("trade_time_ms"),
            field="public_trade.trade_time_ms",
        )
        _nonnegative_int(
            record.get("event_time_ms"),
            field="public_trade.event_time_ms",
        )
        first_trade_id = _nonnegative_int(
            record.get("first_trade_id"),
            field="public_trade.first_trade_id",
        )
        last_trade_id = _nonnegative_int(
            record.get("last_trade_id"),
            field="public_trade.last_trade_id",
        )
        if first_trade_id > last_trade_id:
            return [], "unavailable:invalid_public_trade_id_range"
        wire_sequence = _nonnegative_int(
            record.get("wire_sequence"),
            field="public_trade.wire_sequence",
            positive=True,
        )
        _nonnegative_int(
            record.get("received_monotonic_ns"),
            field="public_trade.received_monotonic_ns",
            positive=True,
        )
        _parse_utc_text(
            record.get("received_at_utc"),
            field="public_trade.received_at_utc",
        )
        connection_id = record.get("connection_id")
        if not isinstance(connection_id, str) or not connection_id:
            return [], "unavailable:invalid_public_trade_connection_id"
        if not segment_id.startswith(f"{connection_id}-segment-"):
            return [], "unavailable:public_trade_connection_segment_mismatch"
        if last_wire_sequence is not None and wire_sequence <= last_wire_sequence:
            return [], "unavailable:public_trade_wire_sequence_not_increasing"
        if last_trade_time_ms is not None and trade_time_ms < last_trade_time_ms:
            return [], "unavailable:public_trade_exchange_time_regression"
        last_wire_sequence = wire_sequence
        last_trade_time_ms = trade_time_ms
        trade_at = datetime.fromtimestamp(trade_time_ms / 1000.0, tz=timezone.utc)
        aggressive_side = str(record.get("aggressive_side", "")).upper()
        if aggressive_side not in {"BUY", "SELL"}:
            return [], "unavailable:invalid_public_trade_side"
        buyer_is_maker = record.get("buyer_is_maker")
        if not isinstance(buyer_is_maker, bool):
            return [], "unavailable:invalid_public_trade_maker_flag"
        expected_side = "SELL" if buyer_is_maker else "BUY"
        if aggressive_side != expected_side:
            return [], "unavailable:public_trade_side_maker_mismatch"
        discontinuity = record.get("aggregate_id_discontinuity_observed")
        if not isinstance(discontinuity, bool):
            return [], "unavailable:invalid_public_trade_discontinuity_flag"
        price = _finite_float(
            record.get("price"),
            field="public_trade.price",
            positive=True,
        )
        quantity = _finite_float(
            record.get("quantity"),
            field="public_trade.quantity",
            positive=True,
        )
        notional = _finite_float(
            record.get("notional_usdt"),
            field="public_trade.notional_usdt",
            positive=True,
        )
        if not math.isclose(notional, price * quantity, rel_tol=1e-9, abs_tol=1e-9):
            return [], "unavailable:public_trade_notional_mismatch"
        if not start <= trade_at <= end:
            continue
        validated = dict(record)
        validated["trade_at"] = trade_at
        validated["notional_value"] = notional
        accepted.append(validated)
    accepted.sort(key=lambda record: cast(datetime, record["trade_at"]))
    return accepted, status


def _load_public_trades(
    ref: L2StreamRef,
    *,
    segment_id: str,
    start: datetime,
    end: datetime,
    now: datetime,
) -> tuple[list[dict[str, Any]], str]:
    try:
        return _load_public_trades_validated(
            ref,
            segment_id=segment_id,
            start=start,
            end=end,
            now=now,
        )
    except ExecutionRiskError as exc:
        return [], f"unavailable:{exc}"


def _public_trade_usable(status: str) -> bool:
    return status.startswith("available")


def _position_side(
    *,
    position_size_base: float | None,
    position_size_usdt: float | None,
) -> tuple[str | None, str | None]:
    signed = position_size_base if position_size_base is not None else position_size_usdt
    if signed is None or signed == 0:
        return None, None
    return ("long", "bids") if signed > 0 else ("short", "asks")


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


def _weighted_mean(values: Sequence[tuple[float, float]]) -> float | None:
    denominator = sum(weight for _, weight in values)
    if denominator <= 0:
        return None
    return sum(value * weight for value, weight in values) / denominator


def _trailing_true_duration_seconds(
    flags: Sequence[bool],
    records: Sequence[dict[str, Any]],
) -> float:
    """Wall-clock duration of the current contiguous true suffix."""

    if not flags or not records or len(flags) != len(records) or not flags[-1]:
        return 0.0
    start_index = len(flags) - 1
    while start_index > 0 and flags[start_index - 1]:
        start_index -= 1
    started_at = cast(datetime, records[start_index]["observation_at"])
    ended_at = cast(datetime, records[-1]["observation_at"])
    return max((ended_at - started_at).total_seconds(), 0.0)


def _adverse_selection_bps(side: str, reference_mid: float, future_mid: float) -> float:
    if side == "BUY":
        return (reference_mid - future_mid) / reference_mid * 10000.0
    return (future_mid - reference_mid) / reference_mid * 10000.0


def _link_private_fills(
    private_ref: PrivateEventStreamRef | None,
    *,
    l2_window: Sequence[dict[str, Any]],
    now: datetime,
) -> tuple[
    str,
    int,
    int,
    int,
    int,
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
    LatestFillEstimate | None,
]:
    if private_ref is None:
        return "unavailable:no_private_event_stream", 0, 0, 0, 0, [], [], [], None
    try:
        _manifest, records = load_validated_private_event_window(private_ref, now=now)
    except PrivateEventError as exc:
        return f"unavailable:{exc}", 0, 0, 0, 0, [], [], [], None
    fills = [record for record in records if record.get("event_type") == "trade_fill"]
    order_updates = [
        record for record in records if record.get("event_type") == "order_update"
    ]
    cancels = sum(
        str(record.get("status", "")).upper() in {"CANCELED", "CANCELLED"}
        for record in order_updates
    )
    snapshots = sorted(
        l2_window,
        key=lambda record: cast(datetime, record["observation_at"]),
    )
    slippage: list[tuple[float, float]] = []
    adverse_5s: list[tuple[float, float]] = []
    adverse_30s: list[tuple[float, float]] = []
    latest: LatestFillEstimate | None = None
    linked = 0
    for fill in fills:
        fill_at = cast(datetime, fill["event_at"])
        before = [
            record
            for record in snapshots
            if cast(datetime, record["observation_at"]) <= fill_at
        ]
        after = [
            record
            for record in snapshots
            if cast(datetime, record["observation_at"]) > fill_at
        ]
        if not before or not after:
            continue
        reference = before[-1]
        reference_at = cast(datetime, reference["observation_at"])
        reference_mid = float(reference["mid_price"])
        side = str(fill.get("side", "")).upper()
        fill_price = float(fill["price_value"])
        fill_qty = float(fill["qty_value"])
        weight = fill_price * fill_qty
        slip = (
            (fill_price - reference_mid) / reference_mid * 10000.0
            if side == "BUY"
            else (reference_mid - fill_price) / reference_mid * 10000.0
        )
        slippage.append((slip, weight))
        linked += 1

        horizon_values: dict[int, tuple[float, float] | None] = {}
        for horizon in (5, 30):
            target = fill_at.timestamp() + horizon
            future = next(
                (
                    record
                    for record in after
                    if cast(datetime, record["observation_at"]).timestamp() >= target
                ),
                None,
            )
            if future is None:
                horizon_values[horizon] = None
                continue
            future_at = cast(datetime, future["observation_at"])
            value = _adverse_selection_bps(
                side,
                reference_mid,
                float(future["mid_price"]),
            )
            horizon_values[horizon] = (
                value,
                (future_at - fill_at).total_seconds(),
            )
            (adverse_5s if horizon == 5 else adverse_30s).append((value, weight))
        h5 = horizon_values[5]
        h30 = horizon_values[30]
        estimate = LatestFillEstimate(
            event_time_utc=fill_at,
            trade_id=str(fill.get("trade_id", "")),
            order_id=str(fill.get("order_id", "")),
            side=side,
            fill_price=fill_price,
            fill_qty=fill_qty,
            reference_mid=reference_mid,
            reference_lag_seconds=(fill_at - reference_at).total_seconds(),
            estimated_slippage_bps=slip,
            adverse_selection_5s_bps=h5[0] if h5 is not None else None,
            adverse_selection_5s_actual_horizon_seconds=(
                h5[1] if h5 is not None else None
            ),
            adverse_selection_30s_bps=h30[0] if h30 is not None else None,
            adverse_selection_30s_actual_horizon_seconds=(
                h30[1] if h30 is not None else None
            ),
        )
        if latest is None or estimate.event_time_utc > latest.event_time_utc:
            latest = estimate
    return (
        "available",
        len(fills),
        len(order_updates),
        cancels,
        linked,
        slippage,
        adverse_5s,
        adverse_30s,
        latest,
    )


def derive_execution_risk_evidence(
    l2_ref: L2StreamRef,
    *,
    private_ref: PrivateEventStreamRef | None,
    now: datetime,
    position_size_base: float | None,
    position_size_usdt: float | None,
) -> ExecutionRiskEvidence:
    """Join bounded L2/public/private events into verdict-inert evidence."""

    try:
        window = load_validated_l2_window(l2_ref, now=now)
    except L2RiskError as exc:
        raise ExecutionRiskError(str(exc)) from exc
    return execution_risk_from_validated_l2_window(
        l2_ref,
        window,
        private_ref=private_ref,
        now=now,
        position_size_base=position_size_base,
        position_size_usdt=position_size_usdt,
    )


def execution_risk_from_validated_l2_window(
    l2_ref: L2StreamRef,
    window: Sequence[dict[str, Any]],
    *,
    private_ref: PrivateEventStreamRef | None,
    now: datetime,
    position_size_base: float | None,
    position_size_usdt: float | None,
) -> ExecutionRiskEvidence:
    """Join other event streams to one immutable, already validated L2 window."""

    if not window:
        raise ExecutionRiskError("validated L2 history window is empty")
    if (
        l2_ref.deterioration_min_duration_seconds < 0
        or l2_ref.deterioration_min_observations < 2
        or not 0.5 <= l2_ref.deterioration_fraction <= 1.0
    ):
        raise ExecutionRiskError("invalid L2 deterioration persistence contract")
    latest = window[-1]
    captured_at = cast(datetime, latest["captured_at"])
    first_observation_at = cast(datetime, window[0]["observation_at"])
    latest_observation_at = cast(datetime, latest["observation_at"])
    position_side, exit_book_side = _position_side(
        position_size_base=position_size_base,
        position_size_usdt=position_size_usdt,
    )
    exit_depth_key = (
        "bid_depth_notional_usdt" if exit_book_side == "bids"
        else "ask_depth_notional_usdt" if exit_book_side == "asks"
        else None
    )
    split = len(window) // 2
    baseline = window[:split]
    recent = window[split:]
    baseline_spread = (
        median(float(record["spread_bps"]) for record in baseline)
        if baseline and recent
        else None
    )
    baseline_exit_depth = (
        median(float(record[exit_depth_key]) for record in baseline)
        if exit_depth_key is not None and baseline and recent
        else None
    )
    current_exit_depth = (
        float(latest[exit_depth_key]) if exit_depth_key is not None else None
    )
    spread_flags = (
        [float(record["spread_bps"]) > baseline_spread for record in recent]
        if baseline_spread is not None
        else []
    )
    depth_flags = (
        [float(record[exit_depth_key]) < baseline_exit_depth for record in recent]
        if exit_depth_key is not None and baseline_exit_depth is not None
        else []
    )
    joint_flags = [
        spread and depth for spread, depth in zip(spread_flags, depth_flags)
    ]
    recent_contract_available = bool(
        len(baseline) >= 2 and spread_flags and depth_flags
    )
    spread_fraction = (
        sum(spread_flags) / len(spread_flags) if spread_flags else None
    )
    depth_fraction = sum(depth_flags) / len(depth_flags) if depth_flags else None
    joint_fraction = sum(joint_flags) / len(joint_flags) if joint_flags else None
    spread_duration = _trailing_true_duration_seconds(spread_flags, recent)
    depth_duration = _trailing_true_duration_seconds(depth_flags, recent)
    joint_duration = _trailing_true_duration_seconds(joint_flags, recent)

    def _is_sustained(fraction: float | None, duration: float) -> bool | None:
        if not recent_contract_available or fraction is None:
            return None
        return bool(
            len(recent) >= l2_ref.deterioration_min_observations
            and fraction >= l2_ref.deterioration_fraction
            and duration >= l2_ref.deterioration_min_duration_seconds
        )

    sustained_spread = _is_sustained(spread_fraction, spread_duration)
    sustained_depth = _is_sustained(depth_fraction, depth_duration)
    sustained = _is_sustained(joint_fraction, joint_duration)
    current_joint = (
        spread_flags[-1] and depth_flags[-1]
        if spread_flags and depth_flags
        else None
    )
    temporary = (
        bool(current_joint and sustained is False)
        if current_joint is not None and sustained is not None
        else None
    )
    if exit_book_side is None:
        liquidity_state = "position_unavailable"
    elif sustained is None:
        liquidity_state = "insufficient_history"
    elif sustained:
        liquidity_state = "sustained_joint_deterioration"
    elif temporary:
        liquidity_state = "temporary_joint_deterioration"
    else:
        liquidity_state = "joint_deterioration_not_observed"

    latest_mid = float(latest["mid_price"])
    if position_size_usdt is not None and math.isfinite(position_size_usdt):
        position_notional = abs(position_size_usdt)
    elif position_size_base is not None and math.isfinite(position_size_base):
        position_notional = abs(position_size_base) * latest_mid
    else:
        position_notional = None
    if position_notional == 0:
        position_notional = None

    public_trades, public_status = _load_public_trades(
        l2_ref,
        segment_id=str(latest["segment_id"]),
        start=first_observation_at,
        end=latest_observation_at,
        now=now,
    )
    public_notional = sum(float(record["notional_value"]) for record in public_trades)
    aggressive_side = (
        "SELL" if exit_book_side == "bids"
        else "BUY" if exit_book_side == "asks"
        else None
    )
    aggressive_exit_notional = (
        sum(
            float(record["notional_value"])
            for record in public_trades
            if record.get("aggressive_side") == aggressive_side
        )
        if aggressive_side is not None and _public_trade_usable(public_status)
        else None
    )
    prefix = "bid" if exit_book_side == "bids" else "ask"
    removed_total = 0.0
    added_total = 0.0
    trade_aligned = 0.0
    refill = 0.0
    sweep_intervals = 0
    if exit_book_side is not None and _public_trade_usable(public_status):
        for prior, record in zip(window, window[1:]):
            interval_start = cast(datetime, prior["observation_at"])
            interval_end = cast(datetime, record["observation_at"])
            proxy = record.get("interval_book_update_proxies")
            proxy_map = proxy if isinstance(proxy, dict) else {}
            removed = _finite_float(
                proxy_map.get(f"{prefix}_removed_notional_usdt", 0.0),
                field="interval.removed_notional",
            )
            added = _finite_float(
                proxy_map.get(f"{prefix}_added_notional_usdt", 0.0),
                field="interval.added_notional",
            )
            interval_trade = sum(
                float(trade["notional_value"])
                for trade in public_trades
                if trade.get("aggressive_side") == aggressive_side
                and interval_start < cast(datetime, trade["trade_at"]) <= interval_end
            )
            aligned = min(removed, interval_trade)
            removed_total += removed
            added_total += added
            trade_aligned += aligned
            refill += min(removed, added)
            if removed > 0 and interval_trade > 0:
                sweep_intervals += 1
    else:
        removed_total = math.nan
        added_total = math.nan
        trade_aligned = math.nan
        refill = math.nan

    def _to_position_ratio(value: float | None) -> float | None:
        if (
            value is None
            or not math.isfinite(value)
            or position_notional is None
            or position_notional <= 0
        ):
            return None
        return value / position_notional

    removed_value = removed_total if math.isfinite(removed_total) else None
    added_value = added_total if math.isfinite(added_total) else None
    trade_aligned_value = trade_aligned if math.isfinite(trade_aligned) else None
    unexplained_value = (
        max(removed_total - trade_aligned, 0.0)
        if math.isfinite(removed_total) and math.isfinite(trade_aligned)
        else None
    )
    refill_value = refill if math.isfinite(refill) else None
    net_withdrawal = (
        max(removed_total - added_total, 0.0)
        if math.isfinite(removed_total) and math.isfinite(added_total)
        else None
    )

    (
        private_status,
        private_fill_count,
        private_order_update_count,
        private_cancel_count,
        linked_count,
        slippage,
        adverse_5s,
        adverse_30s,
        latest_fill,
    ) = _link_private_fills(private_ref, l2_window=window, now=now)
    current_imbalance = float(latest["book_imbalance"])
    exit_imbalance = (
        current_imbalance if exit_book_side == "bids"
        else -current_imbalance if exit_book_side == "asks"
        else None
    )
    return ExecutionRiskEvidence(
        source="sequence_linked_l2_public_private_events",
        captured_at_utc=captured_at,
        l2_run_id=str(latest["run_id"]),
        l2_segment_id=str(latest["segment_id"]),
        strategy_id=l2_ref.strategy_id,
        l2_snapshot_count=len(window),
        history_coverage_seconds=(
            latest_observation_at - first_observation_at
        ).total_seconds(),
        position_side=position_side,
        exit_book_side=exit_book_side,
        position_notional_usdt=position_notional,
        current_spread_bps=float(latest["spread_bps"]),
        baseline_spread_median_bps=baseline_spread,
        current_exit_depth_notional_usdt=current_exit_depth,
        baseline_exit_depth_median_usdt=baseline_exit_depth,
        exit_depth_current_to_baseline=(
            current_exit_depth / baseline_exit_depth
            if current_exit_depth is not None
            and baseline_exit_depth is not None
            and baseline_exit_depth > 0
            else None
        ),
        exit_side_imbalance=exit_imbalance,
        recent_observation_count=len(recent),
        recent_spread_worse_fraction=spread_fraction,
        recent_exit_depth_worse_fraction=depth_fraction,
        sustained_spread_deterioration=sustained_spread,
        sustained_exit_depth_deterioration=sustained_depth,
        sustained_joint_deterioration=sustained,
        temporary_joint_deterioration=temporary,
        joint_deterioration_trailing_duration_seconds=(
            joint_duration if recent_contract_available else None
        ),
        deterioration_min_duration_seconds=(
            l2_ref.deterioration_min_duration_seconds
        ),
        deterioration_min_observations=l2_ref.deterioration_min_observations,
        deterioration_fraction_threshold=l2_ref.deterioration_fraction,
        liquidity_state=liquidity_state,
        public_trade_identity_status=(
            "exact_collector_target"
            if l2_ref.public_trade_path is not None
            and l2_ref.strategy_id is not None
            else "unavailable:no_public_trade_stream"
        ),
        public_trade_status=public_status,
        public_trade_count=len(public_trades),
        public_trade_notional_usdt=public_notional,
        aggressive_exit_side_trade_notional_usdt=aggressive_exit_notional,
        aggressive_exit_side_trade_to_position_ratio=_to_position_ratio(
            aggressive_exit_notional
        ),
        exit_side_removed_notional_usdt=removed_value,
        exit_side_removed_to_position_ratio=_to_position_ratio(removed_value),
        exit_side_added_notional_usdt=added_value,
        exit_side_added_to_position_ratio=_to_position_ratio(added_value),
        exit_side_net_withdrawal_notional_usdt=net_withdrawal,
        exit_side_net_withdrawal_to_position_ratio=_to_position_ratio(
            net_withdrawal
        ),
        trade_aligned_removal_proxy_usdt=trade_aligned_value,
        trade_aligned_removal_to_position_ratio=_to_position_ratio(
            trade_aligned_value
        ),
        unexplained_removal_proxy_usdt=unexplained_value,
        unexplained_removal_to_position_ratio=_to_position_ratio(
            unexplained_value
        ),
        refill_proxy_usdt=refill_value,
        refill_to_position_ratio=_to_position_ratio(refill_value),
        sweep_proxy_interval_count=(
            sweep_intervals
            if _public_trade_usable(public_status) and exit_book_side
            else None
        ),
        private_event_status=private_status,
        private_fill_count=private_fill_count,
        private_order_update_count=private_order_update_count,
        private_cancel_update_count=private_cancel_count,
        private_cancel_update_fraction=(
            private_cancel_count / private_order_update_count
            if private_order_update_count > 0
            else None
        ),
        fill_l2_linked_count=linked_count,
        fill_l2_unlinked_count=private_fill_count - linked_count,
        mean_estimated_slippage_bps=_weighted_mean(slippage),
        p90_estimated_slippage_bps=(
            _percentile([value for value, _weight in slippage], 0.90)
            if slippage
            else None
        ),
        mean_adverse_selection_5s_bps=_weighted_mean(adverse_5s),
        mean_adverse_selection_30s_bps=_weighted_mean(adverse_30s),
        latest_fill_estimate=latest_fill,
    )
