from __future__ import annotations

from datetime import datetime, timezone

import pytest

from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES
from neutralgrid.grid.formulas import GEOMETRIC, grid_spacing_pct
from neutralgrid.training.data_generator import FeatureSnapshot
from neutralgrid.training.scanner_integration import FeatureCollector, build_feature_snapshot


def test_build_feature_snapshot_treats_nan_alias_as_missing_and_stamps_hmm_lineage() -> None:
    row = {
        "candidate_id": "BTC_1",
        "range_prob": float("nan"),
        "hmm_range_prob": 0.73,
        "trend_prob": float("nan"),
        "hmm_trend_prob": 0.27,
        "hmm_artifact_version": "rolling_180d_20260312_192602",
        "hmm_pipeline_version": "6.5.4",
        "hmm_trained_at_utc": "2026-03-12T19:27:33.786932+00:00",
        "hmm_calibration_status": "disabled_self_supervised",
    }

    snapshot = build_feature_snapshot("BTCUSDT", row)

    assert snapshot.range_prob == 0.73
    assert snapshot.trend_prob == 0.27
    assert snapshot.hmm_artifact_version == "rolling_180d_20260312_192602"
    assert snapshot.hmm_pipeline_version == "6.5.4"
    assert snapshot.hmm_trained_at_utc == "2026-03-12T19:27:33.786932+00:00"
    assert snapshot.hmm_calibration_status == "disabled_self_supervised"


def test_build_feature_snapshot_preserves_missing_active_profile_values() -> None:
    snapshot = build_feature_snapshot(
        "BTCUSDT",
        {
            "candidate_id": "BTC_2",
            "micro_osc_score": 0.42,
        },
    )

    snapshot_dict = snapshot.to_dict()
    missing = [
        feature for feature in ACTIVE_SNAPSHOT_META_FEATURES
        if snapshot_dict.get(feature) is None
    ]

    assert missing
    assert snapshot.micro_osc_score == 0.42
    assert snapshot.grid_spacing_pct is None


def test_build_feature_snapshot_derives_grid_spacing_pct_from_geometry() -> None:
    snapshot = build_feature_snapshot(
        "BTCUSDT",
        {
            "candidate_id": "BTC_2",
            "grid_lower": 95.0,
            "grid_upper": 105.0,
            "num_grids": 10,
            "mode": GEOMETRIC,
        },
    )

    assert snapshot.grid_spacing_pct == pytest.approx(
        grid_spacing_pct(95.0, 105.0, 10, GEOMETRIC)
    )


def test_build_feature_snapshot_does_not_derive_spacing_without_mode() -> None:
    snapshot = build_feature_snapshot(
        "BTCUSDT",
        {
            "candidate_id": "BTC_2",
            "grid_lower": 95.0,
            "grid_upper": 105.0,
            "num_grids": 10,
        },
    )

    assert snapshot.grid_spacing_pct is None


def test_build_feature_snapshot_does_not_alias_ev_24h_to_ev_score() -> None:
    snapshot = build_feature_snapshot(
        "BTCUSDT",
        {
            "candidate_id": "BTC_5",
            "ev_24h": 3.5,
        },
    )

    assert snapshot.ev_score is None


def test_feature_collector_validate_snapshot_reports_missing_active_profile_features() -> None:
    collector = FeatureCollector(enabled=False)
    snapshot = FeatureSnapshot(
        symbol="BTCUSDT",
        start_time_utc=datetime.now(timezone.utc),
        candidate_id="BTC_3",
    )

    issues = collector._validate_snapshot(
        snapshot,
        min_features=len(ACTIVE_SNAPSHOT_META_FEATURES),
    )

    assert any(issue.startswith("insufficient_features(") for issue in issues)
    assert any(issue.startswith("missing_profile_features(") for issue in issues)


def test_build_feature_snapshot_preserves_large_open_interest_values() -> None:
    snapshot = build_feature_snapshot(
        "PUMPUSDT",
        {
            "candidate_id": "PUMP_1",
            "open_interest": 3.236831e10,
        },
    )

    assert snapshot.open_interest == pytest.approx(3.236831e10)


def test_build_feature_snapshot_preserves_profile_provenance_fields() -> None:
    values = {
        "parkinson_vol_ratio_4h_24h_pre": 1.25,
        "variance_ratio_1m_15m_pre_2h": 0.91,
        "funding_carry_expected_next_7h": -0.004,
        "liquidity_stability_z_1h": 0.37,
    }

    snapshot = build_feature_snapshot("BTCUSDT", values)

    assert {key: snapshot.to_dict()[key] for key in values} == values
