from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from neutralgrid.live.decision.private_events import (
    PRIVATE_EVENT_MANIFEST_SCHEMA_VERSION,
    PRIVATE_EVENT_SCHEMA_VERSION,
    PrivateEventError,
    PrivateEventStreamRef,
    load_private_event_evidence,
)


UTC = timezone.utc
NOW = datetime(2026, 8, 2, 2, 0, tzinfo=UTC)


def _fill(*, trade_id: str = "7", at: datetime = NOW) -> dict[str, object]:
    return {
        "schema_version": PRIVATE_EVENT_SCHEMA_VERSION,
        "event_type": "trade_fill",
        "symbol": "BANDUSDT",
        "strategy_id": "413549698",
        "run_id": "private-run-1",
        "event_time_utc": at.isoformat(),
        "trade_id": trade_id,
        "order_id": "70",
        "side": "BUY",
        "price": 0.16,
        "qty": 100.0,
        "maker": True,
        "commission_usdt": -0.0032,
        "realized_pnl_usdt": 0.15,
    }


def _write_capture(
    tmp_path: Path,
    records: list[dict[str, object]],
    *,
    updated_at: datetime = NOW,
    strategy_id: str = "413549698",
) -> PrivateEventStreamRef:
    event_path = tmp_path / "private_events.jsonl"
    event_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": PRIVATE_EVENT_MANIFEST_SCHEMA_VERSION,
                "run_id": "private-run-1",
                "symbol": "BANDUSDT",
                "strategy_id": strategy_id,
                "status": "running",
                "updated_at_utc": updated_at.isoformat(),
                "event_path": str(event_path),
                "capture_mode": "binance_user_data_stream",
                "event_completeness": "event_complete",
                "source_scopes": ["orders", "trades", "income"],
                "total_records": len(records),
                "duplicate_records_dropped": 2,
                "rejected_records": 1,
            }
        ),
        encoding="utf-8",
    )
    return PrivateEventStreamRef(
        event_path=event_path,
        manifest_path=manifest_path,
        symbol="BANDUSDT",
        strategy_id="413549698",
        run_id="private-run-1",
        max_age_seconds=60,
        history_window_seconds=3600,
    )


def test_private_event_evidence_summarizes_exact_strategy_events(tmp_path: Path) -> None:
    records = [
        {
            "schema_version": PRIVATE_EVENT_SCHEMA_VERSION,
            "event_type": "order_update",
            "symbol": "BANDUSDT",
            "strategy_id": "413549698",
            "run_id": "private-run-1",
            "event_time_utc": (NOW - timedelta(seconds=10)).isoformat(),
            "order_id": "70",
            "status": "FILLED",
            "executed_qty": 100.0,
        },
        _fill(),
        {
            "schema_version": PRIVATE_EVENT_SCHEMA_VERSION,
            "event_type": "income",
            "symbol": "BANDUSDT",
            "strategy_id": "413549698",
            "run_id": "private-run-1",
            "event_time_utc": NOW.isoformat(),
            "transaction_id": "900",
            "income_type": "FUNDING_FEE",
            "income_usdt": -0.02,
        },
    ]
    evidence = load_private_event_evidence(_write_capture(tmp_path, records), now=NOW)

    assert evidence.event_completeness == "event_complete"
    assert evidence.records_in_window == 3
    assert evidence.order_status_counts == {"FILLED": 1}
    assert evidence.trade_fill_count == 1
    assert evidence.maker_fill_count == 1
    assert evidence.buy_fill_notional_usdt == pytest.approx(16.0)
    assert evidence.realized_pnl_usdt == pytest.approx(0.15)
    assert evidence.commission_usdt == pytest.approx(-0.0032)
    assert evidence.funding_fee_usdt == pytest.approx(-0.02)
    assert evidence.duplicate_records_dropped == 2
    assert evidence.rejected_records == 1


def test_private_event_evidence_rejects_duplicate_exchange_identity(
    tmp_path: Path,
) -> None:
    ref = _write_capture(tmp_path, [_fill(), _fill()])

    with pytest.raises(PrivateEventError, match="duplicate canonical"):
        load_private_event_evidence(ref, now=NOW)


def test_private_event_evidence_rejects_wrong_strategy_and_stale_manifest(
    tmp_path: Path,
) -> None:
    wrong = _write_capture(tmp_path, [_fill()], strategy_id="other-strategy")
    with pytest.raises(PrivateEventError, match="identity mismatch"):
        load_private_event_evidence(wrong, now=NOW)

    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()
    stale = _write_capture(
        stale_dir,
        [_fill()],
        updated_at=NOW - timedelta(minutes=5),
    )
    with pytest.raises(PrivateEventError, match="stale"):
        load_private_event_evidence(stale, now=NOW)


def test_snapshot_scope_is_preserved_without_claiming_event_completeness(
    tmp_path: Path,
) -> None:
    ref = _write_capture(tmp_path, [_fill()])
    assert ref.manifest_path is not None
    manifest = json.loads(ref.manifest_path.read_text(encoding="utf-8"))
    manifest["capture_mode"] = "binance_rest_history"
    manifest["event_completeness"] = "snapshot_only"
    ref.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    evidence = load_private_event_evidence(ref, now=NOW)

    assert evidence.capture_mode == "binance_rest_history"
    assert evidence.event_completeness == "snapshot_only"
