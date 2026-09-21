from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import retrain_meta_labeler
from neutralgrid.calibration import utility_calibrator
from neutralgrid.grid.formulas import grid_spacing_pct
from neutralgrid.grid.spacing_profile import build_winner_iqr_profile
from neutralgrid.models.meta_labeler import MetaLabeler


@pytest.mark.parametrize("field", ["oof_auc_ci_low", "oof_auc_ci_high", "n_pos", "oof_ece"])
@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), float("-inf")])
def test_promotion_gate_rejects_missing_or_nonfinite_evidence(field: str, invalid: Any) -> None:
    metrics: dict[str, Any] = {
        "oof_auc_ci_low": 0.60,
        "oof_auc_ci_high": 0.80,
        "n_pos": 100,
        "oof_ece": 0.05,
    }
    metrics[field] = invalid
    status, reasons = MetaLabeler._evaluate_promotion_gate(**metrics)
    assert status == "fail"
    assert reasons


@pytest.mark.parametrize(
    "low,high,n_pos,ece,expected",
    [
        (0.60, 0.80, 70, 0.10, "pass"),
        (0.50, 0.80, 70, 0.10, "fail"),
        (0.60, 0.80, 69, 0.10, "fail"),
        (0.60, 0.80, 70, np.nextafter(0.10, 1.0), "fail"),
        (0.6322750729893257, 0.6949017408984314, 526, 0.02808024618719856, "pass"),
    ],
)
def test_promotion_gate_preserves_finite_policy_boundaries(
    low: float, high: float, n_pos: int, ece: float, expected: str
) -> None:
    status, _ = MetaLabeler._evaluate_promotion_gate(
        oof_auc_ci_low=low, oof_auc_ci_high=high, n_pos=n_pos, oof_ece=ece
    )
    assert status == expected


@pytest.mark.parametrize(
    "duration,winner_increment,utility_included",
    [(float(np.nextafter(7.0, 0.0)), 1, True), (7.0, 0, True), (float(np.nextafter(7.0, 8.0)), 0, False)],
)
def test_existing_summary_cohort_endpoints_remain_distinct(
    monkeypatch: pytest.MonkeyPatch, duration: float, winner_increment: int, utility_included: bool
) -> None:
    """Pin observed consumer contracts; this does not approve their equivalence."""
    frame = pd.DataFrame(
        {
            "strategy_id": ["baseline-a", "baseline-b", "boundary", "loser"],
            "symbol": ["AUSDT", "BUSDT", "CUSDT", "DUSDT"],
            "duration_hours": [6.0, 6.0, duration, 6.0],
            "pnl_pct": [2.0, 2.0, 2.0, -1.0],
            "grids_count": [20] * 4,
            "num_grids": [20] * 4,
            "price_range_low": [100.0] * 4,
            "price_range_high": [110.0, 120.0, 130.0, 140.0],
            "range_size_pct": [10.0, 20.0, 30.0, 40.0],
            "grid_spacing_pct": [grid_spacing_pct(100.0, high, 20, mode="geometric") for high in [110.0, 120.0, 130.0, 140.0]],
            "profit_per_grid_pct": [0.8] * 4,
            "start_time_utc": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "backfill_status": ["ok"] * 4,
            "range_prob": [0.7] * 4,
            "trend_prob": [0.2] * 4,
            "persistence_prob": [0.8] * 4,
            "hmm_artifact_version": ["rolling_180d_20260101_000000"] * 4,
        }
    )
    monkeypatch.setattr(utility_calibrator, "_read_calibration_sources", lambda _: (frame.copy(), frame.copy(), "fixture"))
    utility_pool = utility_calibrator._load_calibration_pool(
        Path("unused.xlsx"), expected_hmm_artifact_version="rolling_180d_20260101_000000"
    )
    profile = build_winner_iqr_profile(frame, min_pool_n=2, deciles=2)
    assert ("boundary" in set(utility_pool["strategy_id"])) is utility_included
    assert profile.winner_rows == 2 + winner_increment
    assert profile.winner_filter == "duration_hours < 7.0 AND pnl_pct > 1.0"


def test_meta_target_uses_time_to_target_instead_of_summary_duration() -> None:
    frame = pd.DataFrame(
        {
            "duration_hours": [24.0, 24.0, 1.0, 1.0],
            "net_pnl_pct": [-1.0, -1.0, 10.0, 10.0],
            "time_to_target_hours": [np.nextafter(7.0, 0.0), 7.0, np.nextafter(7.0, 8.0), np.nan],
        }
    )
    summary, eligible, target = retrain_meta_labeler._compute_fast_target_population(frame)
    assert summary["label_basis"] == "endogenous_time_to_target"
    assert eligible is not None and eligible.tolist() == [True] * 4
    assert target is not None and target.tolist() == [1, 1, 0, 0]
