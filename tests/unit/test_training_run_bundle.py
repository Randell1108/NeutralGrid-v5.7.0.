from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import run_full_pipeline
from scripts.validate_training_run_bundle import validate_run_bundle


def _write_bundle(
    run_dir: Path,
    *,
    snapshot_ids: list[str],
    training_ids: list[str],
    result_ids: list[str] | None = None,
) -> None:
    snapshot_dir = run_dir / "snapshots"
    backtest_dir = run_dir / "backtest_candidates"
    snapshot_dir.mkdir(parents=True)
    backtest_dir.mkdir(parents=True)

    pd.DataFrame({
        "candidate_id": snapshot_ids,
        "symbol": ["BTCUSDT"] * len(snapshot_ids),
        "start_time_utc": ["2026-05-22T00:00:00+00:00"] * len(snapshot_ids),
    }).to_parquet(snapshot_dir / "snapshots_2026-05-22.parquet", index=False)

    pd.DataFrame({
        "candidate_id": training_ids,
        "symbol": ["BTCUSDT"] * len(training_ids),
        "y": [1] * len(training_ids),
        "net_pnl_pct": [3.2] * len(training_ids),
    }).to_csv(backtest_dir / "training_data_20260522.csv", index=False)

    if result_ids is not None:
        pd.DataFrame({
            "candidate_id": result_ids,
            "symbol": ["BTCUSDT"] * len(result_ids),
            "net_pnl_pct": [3.2] * len(result_ids),
        }).to_csv(backtest_dir / "backtest_results_20260522.csv", index=False)


def test_run_dir_resolves_deployment_and_snapshot_paths(tmp_path: Path) -> None:
    args = run_full_pipeline.parse_args(["--run-dir", str(tmp_path / "run")])

    output_path, snapshot_dir = run_full_pipeline._resolve_pipeline_output_paths(
        args,
        "20260522_120000",
    )

    assert output_path == tmp_path / "run" / "deployment_ready_20260522_120000.csv"
    assert snapshot_dir == tmp_path / "run" / "snapshots"


def test_run_dir_plus_output_fails_closed() -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_full_pipeline.parse_args([
            "--run-dir",
            "data/training_runs/test",
            "--output",
            "results/out.csv",
        ])

    assert exc_info.value.code == 2


def test_validator_passes_when_backtested_ids_have_snapshots(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_bundle(
        run_dir,
        snapshot_ids=["cid-a", "cid-b", "cid-extra"],
        training_ids=["cid-a", "cid-b"],
        result_ids=["cid-a", "cid-b"],
    )

    payload = validate_run_bundle(run_dir)

    assert payload["passed"] is True
    assert payload["metrics"]["backtest_snapshot_match_rate"] == 1.0
    assert payload["classifications"]["backtested_missing_snapshot"] == []
    assert payload["classifications"]["snapshot_without_backtest"] == ["cid-extra"]


def test_validator_fails_when_backtested_id_lacks_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_bundle(
        run_dir,
        snapshot_ids=["cid-a"],
        training_ids=["cid-a", "cid-missing"],
        result_ids=["cid-a", "cid-missing"],
    )

    payload = validate_run_bundle(run_dir)

    assert payload["passed"] is False
    assert "backtested_missing_snapshot" in payload["errors"]
    assert payload["classifications"]["backtested_missing_snapshot"] == ["cid-missing"]


def test_validator_rejects_blank_backtest_candidate_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_bundle(
        run_dir,
        snapshot_ids=["cid-a"],
        training_ids=["cid-a", ""],
        result_ids=["cid-a"],
    )

    payload = validate_run_bundle(run_dir)

    assert payload["passed"] is False
    assert "blank_backtest_candidate_id" in payload["errors"]
    assert payload["classifications"]["blank_candidate_id"]["backtest_count"] == 1
