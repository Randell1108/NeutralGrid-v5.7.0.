from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from neutralgrid.live.decision.private_telemetry import (
    PrivateTelemetryParseError,
    parse_private_telemetry_text,
)
from scripts import collect_depth_shadow as depth
from scripts import finalize_stale_telemetry_manifests as cleanup


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "private_telemetry"


@pytest.mark.parametrize(
    "name,prices",
    [
        ("BANKUSDT_synthetic", [101.00, 102.00, 103.00, 104.00, 105.00, 106.00, 107.00, 108.00]),
        ("STRKUSDT_synthetic", [201.00, 202.00, 203.00, 204.00, 205.00, 206.00, 207.00, 208.00, 209.00, 210.00, 211.00]),
    ],
)
def test_exact_sell_only_capture(name, prices):
    drawer = (FIXTURES / f"{name}.txt").read_text(encoding="utf-8")
    ladder = parse_private_telemetry_text(drawer)["open_order_ladder"]
    assert ladder["buy"] == []
    assert [row["price"] for row in ladder["sell"]] == prices
    assert [row["level"] for row in ladder["sell"]] == list(range(1, len(prices) + 1))
    for row in ladder["sell"]:
        assert row["side"] == "sell"
        assert row["pct_to_fill"] == pytest.approx((row["price"] / ladder["last_price"] - 1) * 100)


@pytest.mark.parametrize("mutation", ["count", "index", "price", "extra_price"])
def test_sell_only_capture_still_rejects_incomplete_or_ambiguous_rows(mutation):
    drawer = (FIXTURES / "BANKUSDT_synthetic.txt").read_text(encoding="utf-8")
    drawer = {
        "count": lambda s: s.replace("Sell(8)", "Sell(9)"),
        "index": lambda s: s.replace("\n4\n5\n", "\n4\n6\n"),
        "price": lambda s: s.replace("\n102.00\n", "\n-1.00\n"),
        "extra_price": lambda s: s.replace("\n108.00\n", "\n108.00\n109.00\n"),
    }[mutation](drawer)
    with pytest.raises(PrivateTelemetryParseError):
        parse_private_telemetry_text(drawer)


@pytest.mark.parametrize(
    "instant,expected_date",
    [
        ("2026-09-10T23:59:59+00:00", "2026-09-10"),
        ("2026-09-11T00:00:00+00:00", "2026-09-10"),
        ("2026-09-11T04:59:59+00:00", "2026-09-10"),
        ("2026-09-11T05:00:00+00:00", "2026-09-11"),
    ],
)
def test_depth_capture_uses_one_lima_ingestion_date(tmp_path, monkeypatch, instant, expected_date):
    frozen = datetime.fromisoformat(instant)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen.astimezone(tz)

    class FakeClient:
        async def get_order_book(self, symbol, *, limit):
            return {"lastUpdateId": 1, "bids": [["99", "2"]], "asks": [["101", "3"]]}

        async def get_premium_index(self, symbol):
            return {"lastFundingRate": "0.0001", "markPrice": "100", "indexPrice": "100"}

        async def close(self):
            pass

    source = tmp_path / "candidates.csv"
    source.write_text("symbol,candidate_id,position_size_usdt\nBTCUSDT,c1,1000\nETHUSDT,c2,1000\n")
    monkeypatch.setattr(depth, "ROOT", tmp_path)
    monkeypatch.setattr(depth, "datetime", FrozenDateTime)
    monkeypatch.setattr(depth, "BinanceClient", FakeClient)
    monkeypatch.setattr(depth, "_git_output", lambda args: None)
    audit = tmp_path / "audit"
    args = argparse.Namespace(input=str(source), symbols=None, output_dir=str(audit),
        max_candidates=2, duration_seconds=0, interval_seconds=60, iteration_timeout_seconds=2,
        limit=20, top_n=1, participation_rate=0.1, fallback_position_usdt=None,
        concurrency=1, max_scan_age_seconds=900, allow_stale_targets=True, dry_run=False)
    assert asyncio.run(depth.collect_depth_shadow(args)) == 0
    manifest = json.loads((audit / "depth_shadow_manifest.json").read_text())
    expected = tmp_path / "Live" / expected_date
    assert Path(manifest["live_root"]) == expected
    assert manifest["ingestion_timezone"] == "America/Lima"
    assert manifest["ingestion_date"] == expected_date
    for symbol in ("BTCUSDT", "ETHUSDT"):
        assert len(list((expected / symbol).glob("depth_shadow_*.jsonl"))) == 1


@pytest.mark.parametrize("handle,error,success,code,expected", [
    (123, 0, 1, 259, True), (123, 0, 1, 0, False),
    (0, 87, 0, 0, False), (0, 5, 0, 0, True),
    (0, 8, 0, 0, True), (123, 0, 0, 0, True),
])
def test_windows_cleanup_never_signals_process(monkeypatch, handle, error, success, code, expected):
    kernel = SimpleNamespace(OpenProcess=Mock(return_value=handle), CloseHandle=Mock())

    def get_exit(_handle, output):
        output._obj.value = code
        return success

    kernel.GetExitCodeProcess = Mock(side_effect=get_exit)
    # Patch the module reference, not global os.name (pytest/pathlib depend on it).
    monkeypatch.setattr(cleanup, "os", SimpleNamespace(name="nt", kill=Mock(side_effect=AssertionError("must not signal"))))
    monkeypatch.setattr(ctypes, "WinDLL", Mock(return_value=kernel), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: error, raising=False)
    assert cleanup._pid_is_alive(321) is expected
    kernel.CloseHandle.assert_called_once_with(handle) if handle else kernel.CloseHandle.assert_not_called()


@pytest.mark.parametrize("pid", [0, -1, True, "123", None])
def test_cleanup_rejects_invalid_pid_without_os_calls(monkeypatch, pid):
    monkeypatch.setattr(cleanup, "os", SimpleNamespace(name="nt", kill=Mock(side_effect=AssertionError("must not signal"))))
    assert cleanup._pid_is_alive(pid) is False
