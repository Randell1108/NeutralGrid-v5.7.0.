from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from neutralgrid.live.decision.pnl_history import (
    append_pnl_observation,
    build_pnl_observation,
)
from scripts import audit_pnl_training_readiness as readiness


UTC = timezone.utc


def _write_observation(
    live_root: Path,
    *,
    captured_at: datetime,
    total_profit: float,
) -> dict[str, Any]:
    evaluated_at = captured_at + timedelta(seconds=2)
    scanner_row: dict[str, Any] = {
        "ts": evaluated_at.isoformat(),
        "symbol": "GRVTUSDT",
        "strategy_id": "strategy-123",
        "candidate_id": "candidate-abc",
        "verdict": "CONTINUE",
        "reasons": [],
        "execution_telemetry": {
            "captured_at": captured_at.isoformat(),
            "pnl": {
                "total_profit_usdt": total_profit,
                "matched_profit_usdt": 1.0,
                "unmatched_pnl_usdt": total_profit - 1.0,
            },
            "position_inventory": {
                "size_usdt": -5.0,
                "entry_price": 0.04,
                "mark_price": 0.05,
                "position_pnl_usdt": total_profit - 1.0,
            },
        },
        "evaluation": {
            "evaluated_at_utc": evaluated_at.isoformat(),
            "price": 0.05,
            "range_prob": 0.6,
            "trend_prob": 0.2,
            "persistence_prob": 0.7,
            "l2_risk": {},
            "execution_risk": {
                "public_trade_status": "available",
                "private_event_status": "available",
            },
        },
    }
    record = build_pnl_observation(
        scanner_row,
        deploy_ts=datetime(2026, 8, 13, 14, 0, tzinfo=UTC),
        source_cycle_manifest="cycle.json",
        source_snapshot_path="snapshot.txt",
        source_snapshot_sha256="a" * 64,
    )
    append_pnl_observation(record, live_root=live_root)
    return record


def test_readiness_without_contract_keeps_acquisition_complete(tmp_path: Path) -> None:
    live_root = tmp_path / "Live"
    _write_observation(
        live_root,
        captured_at=datetime(2026, 8, 13, 15, 0, tzinfo=UTC),
        total_profit=1.0,
    )

    report = readiness.audit_training_readiness(
        live_root=live_root,
        forecast_contract_path=None,
    )

    assert report["cycle_status"] == "complete"
    assert report["training_readiness"] == "blocked_missing_explicit_forecast_contract"
    assert report["counts"]["physical_observation_files"] == 1
    assert report["counts"]["globally_unique_observations"] == 1
    assert report["counts"]["labelable"] is None


def test_readiness_with_contract_classifies_same_bot_forward_labels(
    tmp_path: Path,
) -> None:
    live_root = tmp_path / "Live"
    start = datetime(2026, 8, 13, 15, 0, tzinfo=UTC)
    _write_observation(live_root, captured_at=start, total_profit=1.0)
    _write_observation(
        live_root,
        captured_at=start + timedelta(minutes=5),
        total_profit=2.0,
    )
    contract_path = tmp_path / "forecast_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": readiness.FORECAST_CONTRACT_SCHEMA,
                "status": "approved",
                "forecast_horizon_minutes": 5.0,
                "label_tolerance_minutes": 0.5,
                "direction_label_definition": "delta_total_profit_usdt_gt_zero",
                "pnl_target_definition": "forward_delta_total_profit",
                "pnl_unit": "USDT",
                "fit_fraction": 0.6,
                "calibration_fraction": 0.2,
                "prediction_interval_coverage": 0.8,
                "min_fit_bots": 1,
                "min_calibration_bots": 1,
                "min_test_bots": 1,
                "min_fit_samples": 1,
                "min_calibration_samples": 1,
                "min_test_samples": 1,
                "missing_value_policy": "reject_feature_incomplete",
            }
        ),
        encoding="utf-8",
    )

    report = readiness.audit_training_readiness(
        live_root=live_root,
        forecast_contract_path=contract_path,
    )

    assert report["cycle_status"] == "complete"
    assert report["counts"]["labelable"] == 1
    assert report["counts"]["direction_positive"] == 1
    assert report["counts"]["direction_non_positive"] == 0
    assert report["counts"]["not_yet_matured_or_censored"] == 1
    assert report["training_readiness"] == "blocked_feature_incomplete"
    assert report["bot_disjoint_chronological_split_feasible"] is False
