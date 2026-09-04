from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from neutralgrid.live.decision.pnl_forecast import (
    PnlForecastConfig,
    PnlForecastError,
    build_forecast_examples,
    predict_shadow_pnl,
    train_evaluate_shadow_forecaster,
)
from neutralgrid.live.decision.pnl_history import build_pnl_observation
from scripts import train_live_pnl_forecaster


UTC = timezone.utc


def _observation(
    *,
    bot_index: int,
    tick: int,
    captured_at: datetime,
    pnl: float,
    signal: float,
) -> dict[str, object]:
    strategy_id = f"strategy-{bot_index:02d}"
    deploy_ts = captured_at.replace(hour=0, minute=0, second=0, microsecond=0)
    row = {
        "ts": (captured_at + timedelta(seconds=1)).isoformat(),
        "symbol": f"BOT{bot_index:02d}USDT",
        "strategy_id": strategy_id,
        "candidate_id": f"candidate-{bot_index:02d}",
        "verdict": "CONTINUE",
        "reasons": [],
        "execution_telemetry": {
            "captured_at": captured_at.isoformat(),
            "pnl": {
                "total_profit_usdt": pnl,
                "total_profit_pct": pnl,
                "matched_profit_usdt": pnl * 0.4,
                "unmatched_pnl_usdt": pnl * 0.6,
            },
            "position_inventory": {
                "size_base": 10.0,
                "size_usdt": 100.0,
                "position_pnl_usdt": pnl * 0.6,
                "mark_price": 10.0,
            },
        },
        "profit_deterioration": {
            "peak_total_profit_usdt": max(pnl, 0.0),
            "giveback_usdt": max(-pnl, 0.0),
            "giveback_pct_of_positive_peak": 0.0,
        },
        "evaluation": {
            "evaluated_at_utc": (captured_at + timedelta(seconds=1)).isoformat(),
            "price": 10.0,
            "range_prob": 0.7,
            "trend_prob": 0.2,
            "persistence_prob": 0.8,
            "l2_risk": {
                "expected_exit_impact_bps": max(-signal, 0.0),
                "exit_depth_to_position_ratio": 10.0 + signal,
                "spread_bps": 1.0 - signal * 0.1,
                "spread_current_to_median": 1.0 - signal * 0.05,
                "book_imbalance": signal,
                "exit_side_removal_to_addition_ratio": 1.0 - signal * 0.2,
            },
            "execution_risk": {
                "liquidity_state": "joint_deterioration_not_observed",
                "current_spread_bps": 1.0 - signal * 0.1,
                "baseline_spread_median_bps": 1.0,
                "exit_depth_current_to_baseline": 1.0 + signal * 0.2,
                "exit_side_imbalance": signal,
                "recent_spread_worse_fraction": max(-signal, 0.0),
                "recent_exit_depth_worse_fraction": max(-signal, 0.0),
                "aggressive_exit_side_trade_notional_usdt": 10.0,
                "trade_aligned_removal_proxy_usdt": 5.0,
                "unexplained_removal_proxy_usdt": max(-signal, 0.0) * 10.0,
                "refill_proxy_usdt": max(signal, 0.0) * 10.0,
                "mean_estimated_slippage_bps": max(-signal, 0.0),
                "p90_estimated_slippage_bps": max(-signal, 0.0) * 1.5,
                "mean_adverse_selection_5s_bps": -signal,
                "mean_adverse_selection_30s_bps": -signal,
                "sustained_joint_deterioration": signal < 0,
                "temporary_joint_deterioration": False,
                "public_trade_status": "available",
                "private_event_status": "available",
                "l2_run_id": f"l2-{bot_index}",
                "l2_segment_id": "segment-1",
            },
        },
    }
    return build_pnl_observation(
        row,
        deploy_ts=deploy_ts,
        source_cycle_manifest=f"cycle-{bot_index}-{tick}.json",
        source_snapshot_path=f"snapshot-{bot_index}-{tick}.txt",
        source_snapshot_sha256=f"{bot_index * 100 + tick + 1:064x}",
    )


def _synthetic_observations(bot_count: int = 10) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    for bot_index in range(bot_count):
        bot_start = origin + timedelta(hours=bot_index * 2)
        pnl = float(bot_index % 3) * 0.01
        signals = [1.0 if ((tick * 7 + bot_index * 3) % 5) < 2 else -1.0 for tick in range(13)]
        for tick, signal in enumerate(signals):
            captured_at = bot_start + timedelta(minutes=5 * tick)
            observations.append(
                _observation(
                    bot_index=bot_index,
                    tick=tick,
                    captured_at=captured_at,
                    pnl=pnl,
                    signal=signal,
                )
            )
            pnl += 0.8 * signal
    return observations


def _config() -> PnlForecastConfig:
    return PnlForecastConfig(
        horizon_minutes=5.0,
        label_tolerance_minutes=0.5,
        fit_fraction=0.6,
        calibration_fraction=0.2,
        prediction_interval_coverage=0.8,
        min_fit_bots=4,
        min_calibration_bots=2,
        min_test_bots=2,
        min_fit_samples=30,
        min_calibration_samples=10,
        min_test_samples=10,
    )


def test_forward_labels_use_only_same_bot_and_explicit_horizon() -> None:
    observations = _synthetic_observations(bot_count=2)

    examples, audit = build_forecast_examples(observations, config=_config())

    assert len(examples) == 24
    assert audit["duplicate_observations_dropped"] == 0
    assert set(examples["actual_horizon_minutes"].round(6)) == {5.0}
    assert (examples["bot_identity"] == examples["label_bot_identity"]).all()
    assert (examples["label_captured_at_utc"] > examples["captured_at_utc"]).all()


def test_training_is_bot_disjoint_temporal_and_beats_simple_baselines(
    tmp_path: Path,
) -> None:
    result = train_evaluate_shadow_forecaster(
        _synthetic_observations(),
        config=_config(),
        output_dir=tmp_path / "artifact",
    )

    assert result["status"] == "shadow_oos_validated"
    assert result["forecast_eligible"] is True
    split = result["split_audit"]
    assert split["fit_calibration_overlap_count"] == 0
    assert split["fit_test_overlap_count"] == 0
    assert split["calibration_test_overlap_count"] == 0
    assert split["strict_temporal_order"] is True
    assert result["gates"]["dollar_non_degradation"] is True
    assert result["gates"]["direction_non_degradation"] is True
    assert Path(result["artifact_paths"]["model"]).is_file()
    assert Path(result["artifact_paths"]["metadata"]).is_file()

    forecast = predict_shadow_pnl(
        tmp_path / "artifact",
        _synthetic_observations(bot_count=1),
    )
    assert forecast["status"] == "available"
    assert forecast["horizon_minutes"] == 5.0
    assert forecast["runtime_effect"] == "none"
    assert forecast["prediction_interval_lower_usdt"] <= forecast["predicted_delta_pnl_usdt"]
    assert forecast["prediction_interval_upper_usdt"] >= forecast["predicted_delta_pnl_usdt"]
    assert 0.0 <= forecast["probability_positive_delta"] <= 1.0


def test_insufficient_bot_count_fails_closed_without_eligible_artifact(
    tmp_path: Path,
) -> None:
    result = train_evaluate_shadow_forecaster(
        _synthetic_observations(bot_count=1),
        config=_config(),
        output_dir=tmp_path / "insufficient",
    )

    assert result["status"] == "insufficient_bot_count"
    assert result["forecast_eligible"] is False
    assert result["runtime_effect"] == "none"
    assert not (tmp_path / "insufficient" / "model.joblib").exists()


def test_training_cli_requires_explicit_horizon_and_label_tolerance() -> None:
    try:
        train_live_pnl_forecaster.parse_args(["--output-dir", "artifact"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover - argparse must reject the ambiguous contract
        raise AssertionError("CLI accepted an implicit forecast horizon")


def test_training_refuses_to_overwrite_prior_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "immutable-artifact"
    first = train_evaluate_shadow_forecaster(
        _synthetic_observations(),
        config=_config(),
        output_dir=output_dir,
    )
    metadata_before = (output_dir / "metadata.json").read_bytes()
    model_before = (output_dir / "model.joblib").read_bytes()

    with pytest.raises(PnlForecastError, match="refusing to overwrite"):
        train_evaluate_shadow_forecaster(
            _synthetic_observations(),
            config=_config(),
            output_dir=output_dir,
        )

    assert first["dataset_fingerprint"]
    assert (output_dir / "metadata.json").read_bytes() == metadata_before
    assert (output_dir / "model.joblib").read_bytes() == model_before
