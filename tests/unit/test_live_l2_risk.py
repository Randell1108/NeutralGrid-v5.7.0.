from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from neutralgrid.data.diff_depth import (
    MANIFEST_SCHEMA_VERSION,
    DiffDepthEvent,
    LocalOrderBook,
)
from neutralgrid.live.decision.l2_risk import (
    L2IntervalAccumulator,
    L2RiskError,
    L2StreamRef,
    build_l2_risk_record,
    load_position_normalized_l2_risk,
)


UTC = timezone.utc


def _book() -> LocalOrderBook:
    return LocalOrderBook(
        bids={
            Decimal("100.0"): Decimal("2"),
            Decimal("99.5"): Decimal("3"),
        },
        asks={
            Decimal("100.5"): Decimal("1"),
            Decimal("101.0"): Decimal("4"),
        },
    )


def _event() -> DiffDepthEvent:
    return DiffDepthEvent(
        symbol="TESTUSDT",
        event_time_ms=1,
        transaction_time_ms=1,
        first_update_id=11,
        final_update_id=11,
        previous_final_update_id=10,
        bids=((Decimal("100.0"), Decimal("1")),),
        asks=((Decimal("100.5"), Decimal("3")),),
        connection_id="connection",
        wire_sequence=7,
        received_at_utc="2026-08-02T01:00:00+00:00",
        received_monotonic_ns=1,
    )


def _record(at: datetime, *, bid_qty: float, wire: int) -> dict[str, object]:
    book = LocalOrderBook(
        bids={Decimal("100.0"): Decimal(str(bid_qty))},
        asks={Decimal("100.5"): Decimal("10")},
    )
    accumulator = L2IntervalAccumulator()
    return build_l2_risk_record(
        book=book,
        accumulator=accumulator,
        symbol="TESTUSDT",
        run_id="run-1",
        connection_id="connection",
        segment_id="segment-1",
        captured_at_utc=at.isoformat(),
        wire_sequence=wire,
        final_update_id=100 + wire,
        top_n=10,
    )


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_accumulator_and_record_preserve_book_update_proxies() -> None:
    book = _book()
    event = _event()
    accumulator = L2IntervalAccumulator()

    accumulator.observe(book, event)
    book.apply(event)
    record = build_l2_risk_record(
        book=book,
        accumulator=accumulator,
        symbol="TESTUSDT",
        run_id="run-1",
        connection_id="connection",
        segment_id="segment-1",
        captured_at_utc=event.received_at_utc,
        wire_sequence=event.wire_sequence,
        final_update_id=event.final_update_id,
        top_n=10,
    )

    proxies = record["interval_book_update_proxies"]
    assert isinstance(proxies, dict)
    assert proxies["bid_removed_notional_usdt"] == 100.0
    assert proxies["ask_added_notional_usdt"] == 201.0
    assert record["bid_depth_notional_usdt"] == pytest.approx(398.5)
    assert record["ask_depth_notional_usdt"] == pytest.approx(705.5)
    assert len(record["top_bids"]) == 2


def test_reader_calculates_position_normalized_exit_and_persistence(tmp_path: Path) -> None:
    now = datetime(2026, 8, 2, 1, 0, 10, tzinfo=UTC)
    path = tmp_path / "l2_risk_snapshots.jsonl"
    _write_records(
        path,
        [
            _record(now - timedelta(seconds=10), bid_qty=3.0, wire=1),
            _record(now - timedelta(seconds=5), bid_qty=2.0, wire=2),
            _record(now, bid_qty=1.0, wire=3),
        ],
    )

    evidence = load_position_normalized_l2_risk(
        L2StreamRef(
            feature_path=path,
            symbol="TESTUSDT",
            run_id="run-1",
            max_age_seconds=15,
            history_window_seconds=60,
        ),
        now=now,
        position_size_base=2.0,
        position_size_usdt=200.0,
    )

    assert evidence.position_side == "long"
    assert evidence.exit_book_side == "bids"
    assert evidence.exit_depth_notional_usdt == 100.0
    assert evidence.exit_depth_to_position_ratio == 0.5
    assert evidence.exit_fill_ratio == 0.5
    assert evidence.expected_exit_vwap == 100.0
    assert evidence.expected_exit_impact_bps == pytest.approx(24.93765586)
    assert evidence.exit_depth_current_to_median == 0.5
    assert evidence.snapshot_count == 3
    assert evidence.history_coverage_seconds == 10.0


def test_reader_rejects_stale_or_wrong_symbol(tmp_path: Path) -> None:
    now = datetime(2026, 8, 2, 1, 0, 10, tzinfo=UTC)
    path = tmp_path / "l2_risk_snapshots.jsonl"
    _write_records(path, [_record(now - timedelta(seconds=20), bid_qty=1.0, wire=1)])

    with pytest.raises(L2RiskError, match="stale"):
        load_position_normalized_l2_risk(
            L2StreamRef(feature_path=path, symbol="TESTUSDT", max_age_seconds=5),
            now=now,
            position_size_base=None,
            position_size_usdt=None,
        )

    with pytest.raises(L2RiskError, match="symbol mismatch"):
        load_position_normalized_l2_risk(
            L2StreamRef(feature_path=path, symbol="OTHERUSDT", max_age_seconds=30),
            now=now,
            position_size_base=None,
            position_size_usdt=None,
        )


def test_reader_rejects_non_live_collector_manifest(tmp_path: Path) -> None:
    now = datetime(2026, 8, 2, 1, 0, 10, tzinfo=UTC)
    path = tmp_path / "l2_risk_snapshots.jsonl"
    manifest_path = tmp_path / "manifest.json"
    _write_records(path, [_record(now, bid_qty=1.0, wire=1)])
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "symbol": "TESTUSDT",
                "run_id": "run-1",
                "status": "running",
                "current_phase": "buffering",
                "updated_at_utc": now.isoformat(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(L2RiskError, match="not in a running contiguous live segment"):
        load_position_normalized_l2_risk(
            L2StreamRef(
                feature_path=path,
                manifest_path=manifest_path,
                symbol="TESTUSDT",
                run_id="run-1",
                max_age_seconds=15,
            ),
            now=now,
            position_size_base=None,
            position_size_usdt=None,
        )


def test_reader_rejects_regressing_exchange_clock(tmp_path: Path) -> None:
    now = datetime(2026, 8, 2, 1, 0, 10, tzinfo=UTC)
    path = tmp_path / "l2_risk_snapshots.jsonl"
    first = _record(now - timedelta(seconds=5), bid_qty=2.0, wire=2)
    second = _record(now, bid_qty=1.0, wire=1)
    first["exchange_event_time_ms"] = int((now - timedelta(seconds=4)).timestamp() * 1000)
    second["exchange_event_time_ms"] = int((now - timedelta(seconds=6)).timestamp() * 1000)
    _write_records(path, [first, second])

    with pytest.raises(L2RiskError, match="exchange observation timestamps regress"):
        load_position_normalized_l2_risk(
            L2StreamRef(feature_path=path, symbol="TESTUSDT", max_age_seconds=30),
            now=now,
            position_size_base=1.0,
            position_size_usdt=100.0,
        )

def test_reader_rejects_regressing_wire_sequence(tmp_path: Path) -> None:
    now = datetime(2026, 8, 2, 1, 0, 10, tzinfo=UTC)
    path = tmp_path / "l2_risk_snapshots.jsonl"
    _write_records(
        path,
        [
            _record(now - timedelta(seconds=5), bid_qty=2.0, wire=2),
            _record(now, bid_qty=1.0, wire=1),
        ],
    )

    with pytest.raises(L2RiskError, match="wire sequence regresses"):
        load_position_normalized_l2_risk(
            L2StreamRef(feature_path=path, symbol="TESTUSDT", max_age_seconds=30),
            now=now,
            position_size_base=1.0,
            position_size_usdt=100.0,
        )
