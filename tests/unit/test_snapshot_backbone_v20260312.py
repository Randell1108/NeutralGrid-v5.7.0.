from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neutralgrid.training.data_generator import LabelConfig, TrainingDataBuilder


def test_snapshot_builder_prefers_exact_candidate_and_backward_time_match(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()

    snapshots = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "start_time_utc": "2026-03-01T09:00:00+00:00",
                "candidate_id": "BTCUSDT_20260301_090000_abcd1234",
                "range_prob": 0.55,
                "trend_prob": 0.20,
            },
            {
                "symbol": "BTCUSDT",
                "start_time_utc": "2026-03-01T09:05:00+00:00",
                "candidate_id": "BTCUSDT_20260301_090000_abcd1234",
                "range_prob": 0.72,
                "trend_prob": 0.10,
                "utility_score": 1.50,
                "profit_per_grid_pct": 0.80,
                "primary_pipeline_score": 82.0,
            },
            {
                "symbol": "ETHUSDT",
                "start_time_utc": "2026-03-01 08:55:00+00:00",
                "candidate_id": "",
                "range_prob": 0.91,
                "trend_prob": 0.03,
                "utility_score": 2.0,
            },
            {
                "symbol": "ETHUSDT",
                "start_time_utc": "2026-03-01 09:20:00+00:00",
                "candidate_id": "",
                "range_prob": 0.11,
                "trend_prob": 0.70,
                "utility_score": -1.0,
            },
        ]
    )
    snapshots.to_parquet(snapshot_dir / "snapshots_2026-03-01.parquet", index=False)

    outcomes = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "candidate_id": "BTCUSDT_20260301_090000_abcd1234",
                "start_time_utc": "2026-03-01T09:10:00+00:00",
                "pnl_pct": 6.0,
            },
            {
                "symbol": "ETHUSDT",
                "candidate_id": "",
                "start_time_utc": "2026-03-01 09:10:00+00:00",
                "pnl_pct": -1.0,
            },
        ]
    )

    builder = TrainingDataBuilder(LabelConfig())
    training_df, meta = builder.build_from_snapshots(
        snapshot_dir=snapshot_dir,
        outcome_df=outcomes,
        tolerance=pd.Timedelta(hours=1),
    )

    assert meta["exact_candidate_matches"] == 1
    assert meta["time_matches"] == 1

    btc_row = training_df.loc[training_df["symbol"] == "BTCUSDT"].iloc[0]
    assert btc_row["snapshot_match_method"] == "candidate_id"
    assert btc_row["range_prob"] == pytest.approx(0.72)
    assert btc_row["utility_score"] == pytest.approx(1.50)
    assert str(btc_row["snapshot_matched_at_utc"]) == "2026-03-01 09:05:00+00:00"

    eth_row = training_df.loc[training_df["symbol"] == "ETHUSDT"].iloc[0]
    assert eth_row["snapshot_match_method"] == "time"
    assert eth_row["range_prob"] == pytest.approx(0.91)
    assert eth_row["utility_score"] == pytest.approx(2.0)
    assert str(eth_row["snapshot_matched_at_utc"]) == "2026-03-01 08:55:00+00:00"


def test_snapshot_builder_exact_candidate_match_is_time_causal(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()

    snapshots = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "start_time_utc": "2026-03-01T09:05:00+00:00",
                "candidate_id": "BTCUSDT_20260301_090000_dupe1234",
                "range_prob": 0.61,
            },
            {
                "symbol": "BTCUSDT",
                "start_time_utc": "2026-03-01T09:15:00+00:00",
                "candidate_id": "BTCUSDT_20260301_090000_dupe1234",
                "range_prob": 0.99,
            },
        ]
    )
    snapshots.to_parquet(snapshot_dir / "snapshots_2026-03-01.parquet", index=False)

    outcomes = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "candidate_id": "BTCUSDT_20260301_090000_dupe1234",
                "start_time_utc": "2026-03-01T09:10:00+00:00",
                "pnl_pct": 1.0,
            },
        ]
    )

    builder = TrainingDataBuilder(LabelConfig())
    training_df, meta = builder.build_from_snapshots(
        snapshot_dir=snapshot_dir,
        outcome_df=outcomes,
        tolerance=pd.Timedelta(hours=1),
    )

    assert meta["exact_candidate_matches"] == 1
    row = training_df.iloc[0]
    assert row["snapshot_match_method"] == "candidate_id"
    assert row["range_prob"] == pytest.approx(0.61)
    assert str(row["snapshot_matched_at_utc"]) == "2026-03-01 09:05:00+00:00"


def test_snapshot_builder_handles_mixed_utc_time_units(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()

    snapshots = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "start_time_utc": pd.Timestamp("2026-03-01T09:05:00+00:00"),
                "candidate_id": "BTCUSDT_20260301_090000_unit1234",
                "range_prob": 0.66,
            },
        ]
    )
    snapshots["start_time_utc"] = snapshots["start_time_utc"].astype("datetime64[us, UTC]")
    snapshots.to_parquet(snapshot_dir / "snapshots_2026-03-01.parquet", index=False)

    outcomes = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "candidate_id": "BTCUSDT_20260301_090000_unit1234",
                "start_time_utc": pd.Timestamp("2026-03-01T09:10:00+00:00"),
                "pnl_pct": 1.0,
            },
        ]
    )
    outcomes["start_time_utc"] = outcomes["start_time_utc"].astype("datetime64[ns, UTC]")

    builder = TrainingDataBuilder(LabelConfig())
    training_df, meta = builder.build_from_snapshots(
        snapshot_dir=snapshot_dir,
        outcome_df=outcomes,
        tolerance=pd.Timedelta(hours=1),
    )

    assert meta["exact_candidate_matches"] == 1
    row = training_df.iloc[0]
    assert row["snapshot_match_method"] == "candidate_id"
    assert row["range_prob"] == pytest.approx(0.66)
    assert str(row["snapshot_matched_at_utc"]) == "2026-03-01 09:05:00+00:00"
