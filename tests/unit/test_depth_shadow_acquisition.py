from __future__ import annotations

import json
import argparse
import asyncio
from pathlib import Path

import pandas as pd

import scripts.collect_depth_shadow as collect_module
from scripts.collect_depth_shadow import collect_depth_shadow
from scripts.run_depth_shadow_acquisition import main as acquisition_main


def test_acquisition_runner_blocks_stale_candidate_rows(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(
        "\n".join(
            [
                "symbol,candidate_id,position_size_usdt",
                "BTCUSDT,BTCUSDT_20260625_120000_deadbeef,1000",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "audit"

    exit_code = acquisition_main(
        [
            "--input",
            str(candidates),
            "--output-dir",
            str(output_dir),
            "--now-utc",
            "2026-06-25T13:00:00+00:00",
        ]
    )

    assert exit_code == 2
    manifest = json.loads((output_dir / "acquisition_manifest.json").read_text(encoding="utf-8"))
    rejected = pd.read_csv(output_dir / "rejected_targets.csv")
    targets = pd.read_csv(output_dir / "targets.csv")
    assert manifest["status"] == "blocked_no_fresh_targets"
    assert manifest["original_target_count"] == 1
    assert manifest["target_count"] == 0
    assert manifest["rejected_target_count"] == 1
    assert rejected.loc[0, "reason"] == "stale_scan_time"
    assert targets.empty


def test_acquisition_runner_writes_ready_manifest_without_starting(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(
        "\n".join(
            [
                "symbol,candidate_id,position_size_usdt",
                "BTCUSDT,BTCUSDT_20260625_120000_deadbeef,1000",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "audit"

    exit_code = acquisition_main(
        [
            "--input",
            str(candidates),
            "--output-dir",
            str(output_dir),
            "--now-utc",
            "2026-06-25T12:05:00+00:00",
            "--duration-seconds",
            "420",
            "--interval-seconds",
            "60",
            "--iteration-timeout-seconds",
            "90",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "acquisition_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ready_not_started"
    assert manifest["target_count"] == 1
    assert manifest["expected_snapshots_per_candidate"] == 8
    assert manifest["iteration_timeout_seconds"] == 90.0
    assert manifest["started"] is False
    assert "--iteration-timeout-seconds" in manifest["collector_command"]
    assert "--duration-seconds" in manifest["collector_command"]
    assert not (output_dir / "collector").exists()


def test_acquisition_runner_can_require_positive_position_targets(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(
        "\n".join(
            [
                "symbol,candidate_id,scan_time_utc,position_size_usdt,deploy_margin_usdt,leverage",
                "BTCUSDT,c1,2026-06-25T12:00:00Z,1000,,",
                "ETHUSDT,c2,2026-06-25T12:00:00Z,,0,10",
                "SOLUSDT,c3,2026-06-25T12:00:00Z,,,",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "audit"

    exit_code = acquisition_main(
        [
            "--input",
            str(candidates),
            "--output-dir",
            str(output_dir),
            "--now-utc",
            "2026-06-25T12:05:00+00:00",
            "--require-positive-position",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output_dir / "acquisition_manifest.json").read_text(encoding="utf-8"))
    rejected = pd.read_csv(output_dir / "rejected_targets.csv")
    targets = pd.read_csv(output_dir / "targets.csv")
    assert manifest["status"] == "ready_not_started"
    assert manifest["require_positive_position"] is True
    assert manifest["original_target_count"] == 3
    assert manifest["target_count"] == 1
    assert manifest["rejected_target_count"] == 2
    assert targets["candidate_id"].tolist() == ["c1"]
    assert sorted(rejected["reason"].tolist()) == [
        "missing_position_notional",
        "non_positive_position_notional",
    ]
    command = manifest["collector_command"]
    assert str(output_dir / "targets.csv") in command
    assert str(candidates) not in command


def test_collector_stops_by_elapsed_wall_clock_when_iteration_overruns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeClient:
        async def get_order_book(self, symbol: str, *, limit: int = 100):
            return {"lastUpdateId": 1, "bids": [["99", "2"]], "asks": [["101", "3"]]}

        async def get_premium_index(self, symbol: str):
            return {
                "lastFundingRate": "0.0001",
                "markPrice": "100",
                "indexPrice": "100",
                "nextFundingTime": 0,
            }

        async def close(self) -> None:
            return None

    candidates = tmp_path / "candidates.csv"
    candidates.write_text(
        "\n".join(
            [
                "symbol,candidate_id,position_size_usdt",
                "BTCUSDT,BTCUSDT_20260625_120000_deadbeef,1000",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "collector"
    ticks = iter([0.0, 121.0, 121.5, 122.0])
    last_tick = 122.0

    def fake_monotonic() -> float:
        nonlocal last_tick
        try:
            last_tick = next(ticks)
        except StopIteration:
            pass
        return last_tick

    monkeypatch.setattr(collect_module, "ROOT", tmp_path)
    monkeypatch.setattr(collect_module, "BinanceClient", FakeClient)
    monkeypatch.setattr(collect_module, "_monotonic", fake_monotonic)

    args = argparse.Namespace(
        input=str(candidates),
        symbols=None,
        output_dir=str(output_dir),
        max_candidates=1,
        duration_seconds=120,
        interval_seconds=60,
        iteration_timeout_seconds=120.0,
        limit=20,
        top_n=1,
        participation_rate=0.10,
        fallback_position_usdt=None,
        concurrency=1,
        max_scan_age_seconds=900.0,
        allow_stale_targets=True,
        dry_run=False,
    )

    exit_code = asyncio.run(collect_depth_shadow(args))

    assert exit_code == 0
    manifest = json.loads((output_dir / "depth_shadow_manifest.json").read_text(encoding="utf-8"))
    records = (output_dir / "depth_shadow_records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert manifest["status"] == "complete_depth_window_incomplete"
    assert manifest["planned_snapshot_count_per_candidate"] == 3
    assert manifest["actual_snapshot_count_per_candidate"] == 1
    assert manifest["iteration_count"] == 1
    assert manifest["records_written"] == 1
    assert 0.0 <= manifest["depth_window_seconds"] < 60.0
    assert manifest["required_depth_window_seconds"] == 60.0
    assert len(records) == 1
