from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from neutralgrid.data.diff_depth import (
    MANIFEST_SCHEMA_VERSION,
    PUBLIC_AGG_TRADE_SCHEMA_VERSION,
    LocalOrderBook,
)
from neutralgrid.live.decision.execution_risk import derive_execution_risk_evidence
from neutralgrid.live.decision.l2_risk import (
    L2IntervalAccumulator,
    L2StreamRef,
    build_l2_risk_record,
)
from neutralgrid.live.decision.private_events import (
    PRIVATE_EVENT_MANIFEST_SCHEMA_VERSION,
    PRIVATE_EVENT_SCHEMA_VERSION,
    PrivateEventStreamRef,
)


UTC = timezone.utc
T0 = datetime(2026, 8, 2, 2, 0, tzinfo=UTC)


def _l2_record(
    at: datetime,
    *,
    wire: int,
    bid: str,
    ask: str,
    bid_qty: str,
    removed: float,
    added: float,
) -> dict[str, object]:
    record = build_l2_risk_record(
        book=LocalOrderBook(
            bids={Decimal(bid): Decimal(bid_qty)},
            asks={Decimal(ask): Decimal("10")},
        ),
        accumulator=L2IntervalAccumulator(),
        symbol="TESTUSDT",
        run_id="l2-run",
        connection_id="connection",
        segment_id="connection-segment-000001",
        captured_at_utc=at.isoformat(),
        wire_sequence=wire,
        final_update_id=100 + wire,
        top_n=10,
    )
    record["interval_book_update_proxies"] = {
        "bid_removed_notional_usdt": removed,
        "bid_added_notional_usdt": added,
        "ask_removed_notional_usdt": 0.0,
        "ask_added_notional_usdt": 0.0,
    }
    return record


def _write_l2(tmp_path: Path, records: list[dict[str, object]]) -> L2StreamRef:
    risk_path = tmp_path / "l2_risk_snapshots.jsonl"
    risk_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    public_path = tmp_path / "public_agg_trades.jsonl"
    public_path.write_text(
        json.dumps(
            {
                "schema_version": PUBLIC_AGG_TRADE_SCHEMA_VERSION,
                "record_type": "public_agg_trade",
                "symbol": "TESTUSDT",
                "run_id": "l2-run",
                "l2_segment_id": "connection-segment-000001",
                "aggregate_trade_id": 500,
                "event_time_ms": int((T0 + timedelta(seconds=18)).timestamp() * 1000),
                "trade_time_ms": int((T0 + timedelta(seconds=18)).timestamp() * 1000),
                "price": "100",
                "quantity": "0.5",
                "aggressive_side": "SELL",
                "buyer_is_maker": True,
                "notional_usdt": "50",
                "first_trade_id": 700,
                "last_trade_id": 700,
                "connection_id": "connection",
                "wire_sequence": 5,
                "received_at_utc": (T0 + timedelta(seconds=18)).isoformat(),
                "received_monotonic_ns": 5,
                "aggregate_id_discontinuity_observed": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "symbol": "TESTUSDT",
                "run_id": "l2-run",
                "status": "running",
                "current_phase": "live",
                "updated_at_utc": (T0 + timedelta(seconds=45)).isoformat(),
                "collect_agg_trades": True,
                "trade_subscription_acknowledged": True,
                "counters": {
                    "public_agg_trade_id_discontinuities": 0,
                    "public_agg_trade_duplicates_dropped": 0,
                    "public_agg_trade_out_of_order_dropped": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    return L2StreamRef(
        feature_path=risk_path,
        public_trade_path=public_path,
        manifest_path=manifest_path,
        symbol="TESTUSDT",
        strategy_id="strategy-1",
        run_id="l2-run",
        max_age_seconds=60,
        history_window_seconds=120,
        deterioration_min_duration_seconds=20,
        deterioration_min_observations=2,
        deterioration_fraction=1.0,
    )


def _write_private(tmp_path: Path) -> PrivateEventStreamRef:
    event_path = tmp_path / "private_events.jsonl"
    fill_at = T0 + timedelta(seconds=11)
    records = [
        {
            "schema_version": PRIVATE_EVENT_SCHEMA_VERSION,
            "event_type": "trade_fill",
            "symbol": "TESTUSDT",
            "strategy_id": "strategy-1",
            "run_id": "private-run",
            "event_time_utc": fill_at.isoformat(),
            "trade_id": "1",
            "order_id": "10",
            "side": "SELL",
            "price": 99.4,
            "qty": 1.0,
            "maker": True,
        },
        {
            "schema_version": PRIVATE_EVENT_SCHEMA_VERSION,
            "event_type": "order_update",
            "symbol": "TESTUSDT",
            "strategy_id": "strategy-1",
            "run_id": "private-run",
            "event_time_utc": (fill_at + timedelta(seconds=1)).isoformat(),
            "order_id": "11",
            "status": "CANCELED",
            "executed_qty": 0.0,
        },
    ]
    event_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "private_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": PRIVATE_EVENT_MANIFEST_SCHEMA_VERSION,
                "run_id": "private-run",
                "symbol": "TESTUSDT",
                "strategy_id": "strategy-1",
                "status": "running",
                "updated_at_utc": (T0 + timedelta(seconds=45)).isoformat(),
                "event_path": str(event_path),
                "capture_mode": "binance_user_data_stream",
                "event_completeness": "event_complete",
                "source_scopes": ["orders", "trades"],
                "total_records": len(records),
                "duplicate_records_dropped": 0,
                "rejected_records": 0,
            }
        ),
        encoding="utf-8",
    )
    return PrivateEventStreamRef(
        event_path=event_path,
        manifest_path=manifest_path,
        symbol="TESTUSDT",
        strategy_id="strategy-1",
        run_id="private-run",
        max_age_seconds=60,
        history_window_seconds=120,
    )


def test_event_linkage_derives_sustained_liquidity_and_execution_evidence(
    tmp_path: Path,
) -> None:
    records = [
        _l2_record(T0, wire=1, bid="99.9", ask="100.1", bid_qty="10", removed=0, added=0),
        _l2_record(T0 + timedelta(seconds=10), wire=2, bid="99.9", ask="100.1", bid_qty="9", removed=0, added=0),
        _l2_record(T0 + timedelta(seconds=20), wire=3, bid="99.5", ask="100.5", bid_qty="2", removed=100, added=20),
        _l2_record(T0 + timedelta(seconds=45), wire=4, bid="99.4", ask="100.6", bid_qty="1", removed=40, added=50),
    ]
    l2_ref = _write_l2(tmp_path, records)
    private_ref = _write_private(tmp_path)

    evidence = derive_execution_risk_evidence(
        l2_ref,
        private_ref=private_ref,
        now=T0 + timedelta(seconds=45),
        position_size_base=2.0,
        position_size_usdt=200.0,
    )

    assert evidence.position_side == "long"
    assert evidence.strategy_id == "strategy-1"
    assert evidence.public_trade_identity_status == "exact_collector_target"
    assert evidence.exit_book_side == "bids"
    assert evidence.sustained_joint_deterioration is True
    assert evidence.sustained_spread_deterioration is True
    assert evidence.sustained_exit_depth_deterioration is True
    assert evidence.temporary_joint_deterioration is False
    assert evidence.joint_deterioration_trailing_duration_seconds == 25.0
    assert evidence.liquidity_state == "sustained_joint_deterioration"
    assert evidence.public_trade_status == "available"
    assert evidence.public_trade_count == 1
    assert evidence.aggressive_exit_side_trade_notional_usdt == 50.0
    assert evidence.exit_side_removed_notional_usdt == 140.0
    assert evidence.exit_side_removed_to_position_ratio == 0.7
    assert evidence.exit_side_added_to_position_ratio == 0.35
    assert evidence.exit_side_net_withdrawal_notional_usdt == 70.0
    assert evidence.exit_side_net_withdrawal_to_position_ratio == 0.35
    assert evidence.trade_aligned_removal_proxy_usdt == 50.0
    assert evidence.unexplained_removal_proxy_usdt == 90.0
    assert evidence.unexplained_removal_to_position_ratio == 0.45
    assert evidence.refill_proxy_usdt == 60.0
    assert evidence.refill_to_position_ratio == 0.3
    assert evidence.sweep_proxy_interval_count == 1
    assert evidence.private_fill_count == 1
    assert evidence.private_cancel_update_count == 1
    assert evidence.private_order_update_count == 1
    assert evidence.private_cancel_update_fraction == 1.0
    assert evidence.fill_l2_linked_count == 1
    assert evidence.fill_l2_unlinked_count == 0
    assert evidence.mean_estimated_slippage_bps is not None
    assert evidence.mean_estimated_slippage_bps > 0
    assert evidence.mean_adverse_selection_5s_bps is not None
    assert evidence.mean_adverse_selection_30s_bps is not None
    assert evidence.latest_fill_estimate is not None
    assert evidence.latest_fill_estimate.reference_lag_seconds == 1.0
    assert evidence.latest_fill_estimate.adverse_selection_5s_actual_horizon_seconds == 9.0
    assert evidence.latest_fill_estimate.adverse_selection_30s_actual_horizon_seconds == 34.0


def test_current_only_joint_deterioration_is_temporary_not_sustained(
    tmp_path: Path,
) -> None:
    records = [
        _l2_record(T0, wire=1, bid="99.9", ask="100.1", bid_qty="10", removed=0, added=0),
        _l2_record(T0 + timedelta(seconds=10), wire=2, bid="99.9", ask="100.1", bid_qty="9", removed=0, added=0),
        _l2_record(T0 + timedelta(seconds=20), wire=3, bid="99.9", ask="100.1", bid_qty="8", removed=0, added=0),
        _l2_record(T0 + timedelta(seconds=45), wire=4, bid="99.4", ask="100.6", bid_qty="1", removed=40, added=5),
    ]
    evidence = derive_execution_risk_evidence(
        _write_l2(tmp_path, records),
        private_ref=None,
        now=T0 + timedelta(seconds=45),
        position_size_base=2.0,
        position_size_usdt=200.0,
    )

    assert evidence.sustained_joint_deterioration is False
    assert evidence.temporary_joint_deterioration is True
    assert evidence.joint_deterioration_trailing_duration_seconds == 0.0
    assert evidence.liquidity_state == "temporary_joint_deterioration"
    assert evidence.private_event_status == "unavailable:no_private_event_stream"


def test_public_trade_manifest_anomalies_remain_visible_but_observational(
    tmp_path: Path,
) -> None:
    records = [
        _l2_record(T0, wire=1, bid="99.9", ask="100.1", bid_qty="10", removed=0, added=0),
        _l2_record(T0 + timedelta(seconds=45), wire=2, bid="99.4", ask="100.6", bid_qty="1", removed=50, added=5),
    ]
    ref = _write_l2(tmp_path, records)
    assert ref.manifest_path is not None
    manifest = json.loads(ref.manifest_path.read_text(encoding="utf-8"))
    manifest["counters"]["public_agg_trade_id_discontinuities"] = 1
    ref.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    evidence = derive_execution_risk_evidence(
        ref,
        private_ref=None,
        now=T0 + timedelta(seconds=45),
        position_size_base=2.0,
        position_size_usdt=200.0,
    )

    assert evidence.public_trade_status == "available_with_observed_transport_anomalies"
    assert evidence.public_trade_count == 1
    assert evidence.trade_aligned_removal_proxy_usdt == 50.0


def test_corrupt_public_trade_is_quarantined_without_losing_l2_evidence(
    tmp_path: Path,
) -> None:
    records = [
        _l2_record(T0, wire=1, bid="99.9", ask="100.1", bid_qty="10", removed=0, added=0),
        _l2_record(T0 + timedelta(seconds=45), wire=2, bid="99.4", ask="100.6", bid_qty="1", removed=50, added=5),
    ]
    ref = _write_l2(tmp_path, records)
    assert ref.public_trade_path is not None
    trade = json.loads(ref.public_trade_path.read_text(encoding="utf-8"))
    trade["notional_usdt"] = "51"
    ref.public_trade_path.write_text(json.dumps(trade) + "\n", encoding="utf-8")

    evidence = derive_execution_risk_evidence(
        ref,
        private_ref=None,
        now=T0 + timedelta(seconds=45),
        position_size_base=2.0,
        position_size_usdt=200.0,
    )

    assert evidence.l2_snapshot_count == 2
    assert evidence.public_trade_status.endswith("public_trade_notional_mismatch")
    assert evidence.public_trade_count == 0
    assert evidence.trade_aligned_removal_proxy_usdt is None
