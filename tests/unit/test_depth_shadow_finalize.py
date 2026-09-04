from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.finalize_depth_shadow_acquisition import (
    _oracle_min_window_hours,
    _required_depth_window_seconds,
    main as finalize_main,
)


def test_finalize_depth_shadow_oracle_window_matches_collector_requirement() -> None:
    required_seconds = _required_depth_window_seconds(
        {"duration_seconds": 25200, "interval_seconds": 60}
    )

    assert required_seconds == 25140.0
    assert _oracle_min_window_hours(required_seconds) == pytest.approx(6.983333333333333)


def test_finalize_depth_shadow_blocks_incomplete_collector(tmp_path: Path) -> None:
    acquisition_dir = tmp_path / "acquisition" / "collector"
    acquisition_dir.mkdir(parents=True)
    candidates = tmp_path / "candidates.csv"
    candidates.write_text("symbol,candidate_id\nBTCUSDT,BTCUSDT_20260626_120000_a\n", encoding="utf-8")
    (acquisition_dir / "depth_shadow_manifest.json").write_text(
        json.dumps({"status": "running", "top_n": 20, "participation_rate": 0.1}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "finalized"

    exit_code = finalize_main(
        [
            "--acquisition-dir",
            str(tmp_path / "acquisition"),
            "--candidates",
            str(candidates),
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "depth_shadow_finalize_manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 2
    assert manifest["status"] == "blocked_collector_incomplete"
    assert manifest["collector_status"] == "running"


def test_finalize_depth_shadow_blocks_short_depth_window(tmp_path: Path) -> None:
    acquisition_dir = tmp_path / "acquisition" / "collector"
    acquisition_dir.mkdir(parents=True)
    candidates = tmp_path / "candidates.csv"
    candidates.write_text("symbol,candidate_id\nBTCUSDT,BTCUSDT_20260626_120000_a\n", encoding="utf-8")
    (acquisition_dir / "depth_shadow_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "started_at_utc": "2026-06-26T12:00:00+00:00",
                "last_capture_time_utc": "2026-06-26T13:00:00+00:00",
                "duration_seconds": 25200,
                "interval_seconds": 60,
                "top_n": 20,
                "participation_rate": 0.1,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "finalized"

    exit_code = finalize_main(
        [
            "--acquisition-dir",
            str(tmp_path / "acquisition"),
            "--candidates",
            str(candidates),
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "depth_shadow_finalize_manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 2
    assert manifest["status"] == "blocked_depth_window_incomplete"
    assert manifest["depth_window_seconds"] == 3600.0
    assert manifest["required_depth_window_seconds"] == 25140.0


def test_finalize_depth_shadow_reports_collector_depth_window_status(tmp_path: Path) -> None:
    acquisition_dir = tmp_path / "acquisition" / "collector"
    acquisition_dir.mkdir(parents=True)
    candidates = tmp_path / "candidates.csv"
    candidates.write_text("symbol,candidate_id\nBTCUSDT,BTCUSDT_20260626_120000_a\n", encoding="utf-8")
    (acquisition_dir / "depth_shadow_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete_depth_window_incomplete",
                "started_at_utc": "2026-06-26T12:00:00+00:00",
                "last_capture_time_utc": "2026-06-26T13:00:00+00:00",
                "duration_seconds": 25200,
                "interval_seconds": 60,
                "top_n": 20,
                "participation_rate": 0.1,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "finalized"

    exit_code = finalize_main(
        [
            "--acquisition-dir",
            str(tmp_path / "acquisition"),
            "--candidates",
            str(candidates),
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "depth_shadow_finalize_manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 2
    assert manifest["status"] == "blocked_depth_window_incomplete"
    assert manifest["collector_status"] == "complete_depth_window_incomplete"
