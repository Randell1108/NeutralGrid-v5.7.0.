from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from scripts.collect_private_grid_telemetry import (
    DrawerSnapshot,
    TelemetryError,
    _commit_cycle,
    _mark_one_shot_finished,
    _parse_expected_count,
    _validate_drawer_text,
    parse_args,
)


def test_private_scheduler_default_matches_approved_three_minute_cadence() -> None:
    assert parse_args([]).interval_seconds == 180.0


def test_parse_expected_count_requires_exactly_one_um_grid_tab() -> None:
    assert _parse_expected_count(["Spot Grid", "UM Grid (7)", "CM Grid"]) == 7


@pytest.mark.parametrize(
    "values",
    [
        [],
        ["UM Grid"],
        ["UM Grid (2)", "UM Grid (3)"],
        "UM Grid (7)",
    ],
)
def test_parse_expected_count_fails_closed(values: object) -> None:
    with pytest.raises(TelemetryError):
        _parse_expected_count(values)


def test_validate_drawer_text_extracts_strategy_and_created_time() -> None:
    text = "\n".join(
        [
            "BTCUSDT",
            "Working",
            "Time Created",
            "2026-07-28 17:30:30",
            "Total Profit (USDT)",
            "0.00 (0.00%)",
            "Positions",
            "no position",
            "Pending Order",
            "Grid Details",
            "Order History",
            "Strategy Number",
            "413468963",
        ]
    )

    strategy_id, created_at = _validate_drawer_text("BTCUSDT", text)

    assert strategy_id == "413468963"
    assert created_at == "2026-07-28 17:30:30"


def test_validate_drawer_text_rejects_partial_or_wrong_symbol() -> None:
    with pytest.raises(TelemetryError, match="symbol"):
        _validate_drawer_text("ETHUSDT", "BTCUSDT\nStrategy Number\n413468963")

    with pytest.raises(TelemetryError, match="incomplete"):
        _validate_drawer_text("BTCUSDT", "BTCUSDT\nStrategy Number\n413468963")


def test_commit_cycle_routes_live_snapshot_and_manifest_by_lima_date(tmp_path) -> None:
    live_root = tmp_path / "Live"
    audit_dir = tmp_path / "audit"
    snapshot = DrawerSnapshot(
        symbol="BTCUSDT",
        strategy_id="413468963",
        created_at_lima="2026-07-28 17:30:30",
        captured_at_utc="2026-07-28T22:39:07+00:00",
        captured_at_lima="2026-07-28T17:39:07-05:00",
        raw_text="BTCUSDT\ncomplete drawer",
    )

    manifest = _commit_cycle(
        [snapshot],
        live_root=live_root,
        audit_dir=audit_dir,
        cycle_started_at=datetime(2026, 7, 28, 22, 39, 7, tzinfo=timezone.utc),
    )

    text_path = (
        live_root
        / "2026-07-28"
        / "BTCUSDT"
        / "private_telemetry_413468963_20260728_173907_lima.txt"
    )
    json_path = text_path.with_suffix(".json")
    cycle_path = audit_dir / "cycles" / "cycle_20260728_173907_lima.json"
    assert text_path.read_text(encoding="utf-8") == "BTCUSDT\ncomplete drawer\n"
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["data_class"] == "live_bot_telemetry"
    assert metadata["raw_text"] is None
    assert json.loads(cycle_path.read_text(encoding="utf-8"))["status"] == "complete"
    assert manifest["active_bot_count"] == 1


def test_mark_one_shot_finished_replaces_transient_status() -> None:
    heartbeat = {
        "status": "waiting_for_valid_browser_state",
        "last_error": "Chrome unavailable",
    }

    _mark_one_shot_finished(
        heartbeat,
        succeeded=False,
        finished_at=datetime(2026, 8, 6, 8, 30, tzinfo=timezone.utc),
    )

    assert heartbeat == {
        "status_before_finish": "waiting_for_valid_browser_state",
        "status": "finished",
        "finished_at_utc": "2026-08-06T08:30:00+00:00",
        "finish_reason": "one_shot_failed",
        "last_error": "Chrome unavailable",
    }
