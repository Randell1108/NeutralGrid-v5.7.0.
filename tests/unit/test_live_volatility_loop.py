from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import run_live_volatility_loop as loop


UTC = timezone.utc


def _cycle_manifest(path: Path, *, completed: datetime) -> Path:
    payload = {
        "schema_version": loop.PLUGIN_CYCLE_SCHEMA,
        "status": "complete",
        "source": "chrome_plugin",
        "page_identity": (
            "Authenticated Binance Futures Grid page; UM Grid selected; "
            "Running selected"
        ),
        "source_url": "https://www.binance.bh/en/trading-bots/futures/grid/BTCUSDT",
        "cycle_completed_at_utc": completed.isoformat(),
        "active_bot_count": 1,
        "working_row_count": 1,
        "files": [{"symbol": "BTCUSDT", "strategy_id": "413000001"}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_roster_requires_fresh_authenticated_complete_cycle(tmp_path: Path) -> None:
    now = datetime(2026, 8, 21, 18, 0, tzinfo=UTC)
    path = _cycle_manifest(tmp_path / "cycles" / "cycle_ok.json", completed=now)

    roster = loop.load_roster(
        manifest_path=path,
        audit_root=tmp_path,
        max_age_seconds=180.0,
        now=now + timedelta(seconds=30),
    )

    assert roster.entries == (loop.RosterEntry("BTCUSDT", "413000001"),)


def test_load_roster_rejects_stale_cycle(tmp_path: Path) -> None:
    now = datetime(2026, 8, 21, 18, 0, tzinfo=UTC)
    path = _cycle_manifest(tmp_path / "cycles" / "cycle_old.json", completed=now)

    with pytest.raises(loop.LiveVolatilityLoopError, match="stale"):
        loop.load_roster(
            manifest_path=path,
            audit_root=tmp_path,
            max_age_seconds=60.0,
            now=now + timedelta(seconds=61),
        )


def test_load_roster_rejects_chrome_loss_without_manifest(tmp_path: Path) -> None:
    with pytest.raises(loop.LiveVolatilityLoopError, match="no Chrome-plugin cycle"):
        loop.load_roster(
            manifest_path=None,
            audit_root=tmp_path,
            max_age_seconds=60.0,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("page_identity", "", "page identity is missing"),
        ("source_url", "http://www.binance.com/grid", "not trusted Binance HTTPS"),
        ("source_url", "https://binance.com.evil.example/grid", "not trusted Binance HTTPS"),
    ],
)
def test_load_roster_rejects_missing_identity_or_untrusted_source(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    now = datetime(2026, 8, 21, 18, 0, tzinfo=UTC)
    path = _cycle_manifest(tmp_path / "cycles" / "cycle_bad.json", completed=now)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(loop.LiveVolatilityLoopError, match=message):
        loop.load_roster(
            manifest_path=path,
            audit_root=tmp_path,
            max_age_seconds=180.0,
            now=now + timedelta(seconds=30),
        )


def test_raw_kline_frame_excludes_non_final_and_rejects_short_rows() -> None:
    current = [60_000, "1", "2", "0.5", "1.5", "3", 120_000]
    final = [0, "1", "2", "0.5", "1.5", "3", 59_999]
    frame = loop._raw_kline_frame([final, current], now_ms=120_000)
    assert frame["open_time_ms"].tolist() == [0]
    with pytest.raises(loop.VolatilityError, match="seven"):
        loop._raw_kline_frame([[1, 2]], now_ms=120_000)


def test_unavailable_record_is_explicitly_verdict_inert() -> None:
    contract = loop.load_volatility_contract(
        loop.ROOT / "config" / "live_volatility_forecast_v1.json"
    )
    record = loop.unavailable_record(
        entry=loop.RosterEntry("BTCUSDT", "413000001"),
        contract=contract,
        requested_horizon_minutes=360,
        failure_class="ineligible_or_corrupt_artifact",
        reason="no eligible artifact",
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    assert record["status"] == "unavailable"
    assert record["eligibility"] is False
    assert record["latest_source_timestamp_utc"] is None
    assert record["verdict_influence"] is False
    assert record["runtime_effect"] == "none"


@pytest.mark.parametrize(
    ("status", "expected"),
    [(404, "http_404"), (429, "rate_limited"), (503, "http_5xx")],
)
def test_binance_http_failures_have_explicit_classes(
    status: int,
    expected: str,
) -> None:
    error = loop.BinanceAPIError(status, None, "failed", "/endpoint", "{}")
    assert loop._failure_class(error) == expected


def test_atomic_output_failure_removes_implementation_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "forecast.json"

    def fail_replace(_source: str, _destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(loop.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        loop._atomic_json(destination, {"status": "test"}, replace=False)

    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_lock_contention_then_clean_restart(tmp_path: Path) -> None:
    lock_path = tmp_path / "collector.lock"
    loop._acquire_lock(lock_path)
    with pytest.raises(loop.LiveVolatilityLoopError, match="already running"):
        loop._acquire_lock(lock_path)
    lock_path.unlink()

    loop._acquire_lock(lock_path)
    lock_path.unlink()


def test_lock_create_race_is_explicit_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "collector.lock"

    def fail_open(*_args: object, **_kwargs: object) -> int:
        raise FileExistsError("injected create race")

    monkeypatch.setattr(loop.os, "open", fail_open)
    with pytest.raises(loop.LiveVolatilityLoopError, match="appeared concurrently"):
        loop._acquire_lock(lock_path)


def test_one_cycle_persists_unavailable_when_artifact_is_ineligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    cycle = _cycle_manifest(tmp_path / "roster" / "cycles" / "cycle.json", completed=now)
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "neutralgrid_shadow_volatility_artifact_v1",
                "forecast_eligible": False,
                "verdict_influence": False,
            }
        ),
        encoding="utf-8",
    )

    async def fake_reconcile(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"mark": {}, "last": {}}

    monkeypatch.setattr(loop, "reconcile_symbol_tail", fake_reconcile)
    monkeypatch.setattr(loop, "reconcile_symbol_gaps", fake_reconcile)
    args = argparse.Namespace(
        contract=loop.ROOT / "config" / "live_volatility_forecast_v1.json",
        cycle_manifest=cycle,
        roster_audit_root=tmp_path / "roster",
        max_roster_age_seconds=900.0,
        price_store=tmp_path / "prices",
        artifact_dir=artifact_dir,
        requested_horizon_minutes=360,
        max_data_age_seconds=180.0,
        max_gap_rest_requests=8,
        live_root=tmp_path / "Live",
    )

    report = asyncio.run(loop.run_cycle(args, scheduled_start_utc=now))

    assert report["status"] == "complete"
    assert report["available_count"] == 0
    assert report["unavailable_count"] == 1
    forecast_path = Path(report["forecast_paths"][0])
    persisted = json.loads(forecast_path.read_text(encoding="utf-8"))
    assert persisted["failure_class"] == "ineligible_or_corrupt_artifact"
    assert persisted["verdict_influence"] is False
