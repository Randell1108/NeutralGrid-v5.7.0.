from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from neutralgrid.data.depth_oracle import (
    DepthOracleConfig,
    build_depth_oracle_labels,
    summarize_depth_windows,
)
from scripts.build_depth_oracle_labels import main as build_depth_oracle_main


def _depth_records(*, hours: float = 7.0, fill: float = 1.0, capacity: float = 1.2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "candidate_id": "c1",
                "capture_time_utc": "2026-06-25T00:00:00+00:00",
                "top_n_depth_min_usdt": 1500.0,
                "min_side_fill_ratio": fill,
                "partial_fill_capacity_ratio": capacity,
                "spread_pct": 0.02,
                "max_side_impact_bps": 4.0,
                "round_trip_fee_pct": 0.04,
                "estimated_abs_funding_pct": 0.01,
                "position_notional_usdt": 1000.0,
            },
            {
                "symbol": "BTCUSDT",
                "candidate_id": "c1",
                "capture_time_utc": pd.Timestamp("2026-06-25T00:00:00Z")
                + pd.Timedelta(hours=hours),
                "top_n_depth_min_usdt": 1400.0,
                "min_side_fill_ratio": fill,
                "partial_fill_capacity_ratio": capacity,
                "spread_pct": 0.03,
                "max_side_impact_bps": 5.0,
                "round_trip_fee_pct": 0.04,
                "estimated_abs_funding_pct": 0.01,
                "position_notional_usdt": 1000.0,
            },
        ]
    )


def _outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "net_pnl_pct": 3.2,
                "time_to_target_hours": 6.5,
                "min_pnl_pct": -3.0,
            }
        ]
    )


def test_summarize_depth_windows_requires_forward_depth_coverage() -> None:
    diagnostics = summarize_depth_windows(_depth_records(hours=2.0), DepthOracleConfig())

    assert diagnostics.loc[0, "snapshot_count"] == 2
    assert diagnostics.loc[0, "window_hours"] == pytest.approx(2.0)
    assert diagnostics.loc[0, "required_window_hours"] == pytest.approx(7.0)


def test_build_depth_oracle_labels_produces_positive_when_all_evidence_passes() -> None:
    result = build_depth_oracle_labels(_depth_records(), _outcomes())

    assert result.manifest["status"] == "complete"
    assert result.manifest["labelable_rows"] == 1
    assert result.manifest["positive_rows"] == 1
    assert result.labels.loc[0, "depth_oracle_label"] == 1
    assert result.labels.loc[0, "label_status"] == "positive"
    assert result.labels.loc[0, "depth_adjusted_pnl_pct"] == pytest.approx(3.1)


def test_build_depth_oracle_labels_accepts_collector_complete_window() -> None:
    config = DepthOracleConfig(min_window_hours=(25200.0 - 60.0) / 3600.0)

    result = build_depth_oracle_labels(
        _depth_records(hours=6.998611111111111),
        _outcomes(),
        config=config,
    )

    assert result.manifest["status"] == "complete"
    assert result.labels.loc[0, "depth_oracle_label"] == 1
    assert result.labels.loc[0, "window_hours"] == pytest.approx(6.998611111111111)


def test_build_depth_oracle_labels_leaves_incomplete_windows_unlabeled() -> None:
    result = build_depth_oracle_labels(_depth_records(hours=2.0), _outcomes())

    assert result.manifest["status"] == "blocked_no_labelable_rows"
    assert pd.isna(result.labels.loc[0, "depth_oracle_label"])
    assert "insufficient_depth_window" in result.labels.loc[0, "label_reason"]


def test_build_depth_oracle_labels_marks_partial_fills_negative_when_evidence_complete() -> None:
    result = build_depth_oracle_labels(_depth_records(fill=0.8), _outcomes())

    assert result.manifest["labelable_rows"] == 1
    assert result.labels.loc[0, "depth_oracle_label"] == 0
    assert result.labels.loc[0, "failed_fill_reason"] == "partial_fill"


def test_build_depth_oracle_labels_marks_explicit_non_hit_negative_without_time_to_target() -> None:
    outcomes = pd.DataFrame(
        [
            {
                "candidate_id": "c1",
                "net_pnl_pct": -12.0,
                "time_to_target_hours": None,
                "tail_pnl_pct": -5.0,
                "target_hit": False,
            }
        ]
    )

    result = build_depth_oracle_labels(_depth_records(), outcomes)

    assert result.manifest["status"] == "complete"
    assert result.manifest["labelable_rows"] == 1
    assert result.manifest["negative_rows"] == 1
    assert result.labels.loc[0, "depth_oracle_label"] == 0
    assert result.labels.loc[0, "label_status"] == "negative"
    assert "missing_time_to_target" not in result.labels.loc[0, "label_reason"]
    assert "target_not_met" in result.labels.loc[0, "label_reason"]


def test_build_depth_oracle_cli_persists_fail_closed_artifacts(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth.csv"
    outcomes_path = tmp_path / "outcomes.csv"
    output_dir = tmp_path / "oracle"
    _depth_records(hours=1.0).to_csv(depth_path, index=False)
    _outcomes().to_csv(outcomes_path, index=False)

    exit_code = build_depth_oracle_main(
        [
            "--depth-input",
            str(depth_path),
            "--outcomes",
            str(outcomes_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 2
    manifest = json.loads((output_dir / "depth_oracle_manifest.json").read_text(encoding="utf-8"))
    labels = pd.read_csv(output_dir / "depth_labels.csv")
    assert manifest["status"] == "blocked_no_labelable_rows"
    assert manifest["unlabeled_rows"] == 1
    assert "insufficient_depth_window" in labels.loc[0, "label_reason"]
    assert (output_dir / "replay_diagnostics.csv").exists()
    assert (output_dir / "failed_fill_reasons.csv").exists()
