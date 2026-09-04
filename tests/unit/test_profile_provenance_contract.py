from __future__ import annotations

from typing import Any, cast

from neutralgrid.backtest.candidate_pipeline import convert_to_training_row
from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES
from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES
from neutralgrid.training.unified_training_builder import ALL_META_FEATURES


def test_backtest_training_row_preserves_profile_provenance_without_meta_promotion() -> None:
    profile_values = {
        "parkinson_vol_ratio_4h_24h_pre": 1.25,
        "variance_ratio_1m_15m_pre_2h": 0.91,
        "funding_carry_expected_next_7h": -0.004,
        "liquidity_stability_z_1h": 0.37,
    }
    candidate: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "candidate_id": "BTCUSDT_20260701_120000_a1b2c3d4",
        **profile_values,
    }
    result: dict[str, Any] = {
        "net_pnl_pct": 4.0,
        "duration_hours": 1.0,
        "label_positive_by_horizon": True,
        "is_authoritative": True,
    }

    row = convert_to_training_row(
        cast(dict[str, Any], result),
        cast(dict[str, Any], candidate),
    )

    assert {feature: row[feature] for feature in DEFAULT_FEATURES} == profile_values
    assert set(DEFAULT_FEATURES).isdisjoint(ACTIVE_SNAPSHOT_META_FEATURES)
    assert set(DEFAULT_FEATURES).isdisjoint(ALL_META_FEATURES)
