from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from neutralgrid.live.decision import pnl_history
from neutralgrid.live.decision.pnl_history import (
    PNL_OBSERVATION_SCHEMA_VERSION,
    PnlHistoryError,
    append_pnl_observation,
    build_pnl_observation,
    load_pnl_observations,
)


UTC = timezone.utc
DEPLOYED_AT = datetime(2026, 8, 10, 10, 0, tzinfo=UTC)


def _scanner_row(*, total_profit: float, evaluated_at: datetime) -> dict[str, object]:
    return {
        "ts": evaluated_at.isoformat(),
        "symbol": "GRVTUSDT",
        "strategy_id": "strategy-123",
        "candidate_id": "candidate-abc",
        "verdict": "CONTINUE",
        "reasons": [],
        "execution_telemetry": {
            "captured_at": (evaluated_at - timedelta(seconds=2)).isoformat(),
            "pnl": {
                "total_profit_usdt": total_profit,
                "total_profit_pct": total_profit / 10.0,
                "matched_profit_usdt": 2.0,
                "unmatched_pnl_usdt": total_profit - 2.0,
                "transaction_fee_usdt": -0.1,
            },
            "position_inventory": {
                "size_base": -100.0,
                "size_usdt": -5.0,
                "mark_price": 0.05,
                "position_pnl_usdt": total_profit - 2.0,
            },
        },
        "profit_deterioration": {
            "peak_total_profit_usdt": 4.0,
            "giveback_usdt": max(4.0 - total_profit, 0.0),
        },
        "evaluation": {
            "evaluated_at_utc": evaluated_at.isoformat(),
            "price": 0.05,
            "range_prob": 0.6,
            "trend_prob": 0.2,
            "persistence_prob": 0.7,
            "l2_risk": {
                "expected_exit_impact_bps": 2.5,
                "exit_depth_to_position_ratio": 8.0,
            },
            "execution_risk": {
                "liquidity_state": "joint_deterioration_not_observed",
                "current_spread_bps": 1.2,
                "exit_depth_current_to_baseline": 0.9,
                "exit_side_imbalance": -0.25,
                "public_trade_status": "available",
                "private_event_status": "available",
                "mean_estimated_slippage_bps": 0.4,
                "mean_adverse_selection_5s_bps": 0.2,
                "mean_adverse_selection_30s_bps": 0.3,
            },
        },
    }


def _observation(
    *,
    total_profit: float,
    captured_at: datetime,
    evaluated_at: datetime | None = None,
    source_hash: str = "a" * 64,
) -> dict[str, object]:
    evaluated = evaluated_at or captured_at + timedelta(seconds=2)
    row = _scanner_row(total_profit=total_profit, evaluated_at=evaluated)
    telemetry = row["execution_telemetry"]
    assert isinstance(telemetry, dict)
    telemetry["captured_at"] = captured_at.isoformat()
    return build_pnl_observation(
        row,
        deploy_ts=DEPLOYED_AT,
        source_cycle_manifest="cycle.json",
        source_snapshot_path="snapshot.txt",
        source_snapshot_sha256=source_hash,
    )


def test_build_observation_preserves_exact_identity_and_microstructure() -> None:
    captured_at = datetime(2026, 8, 10, 18, 0, tzinfo=UTC)
    observation = _observation(total_profit=3.0, captured_at=captured_at)

    assert observation["schema_version"] == PNL_OBSERVATION_SCHEMA_VERSION
    assert observation["symbol"] == "GRVTUSDT"
    assert observation["strategy_id"] == "strategy-123"
    assert observation["deploy_ts_utc"] == DEPLOYED_AT.isoformat()
    assert observation["telemetry_captured_at_utc"] == captured_at.isoformat()
    assert observation["pnl"]["total_profit_usdt"] == 3.0
    assert observation["position"]["size_base"] == -100.0
    assert observation["features"]["expected_exit_impact_bps"] == 2.5
    assert observation["features"]["public_trade_status"] == "available"
    assert len(str(observation["bot_identity"])) == 64
    assert len(str(observation["observation_id"])) == 64


def test_atomic_history_spans_lima_dates_and_loads_chronologically(
    tmp_path: Path,
) -> None:
    first_at = datetime(2026, 8, 10, 4, 59, tzinfo=UTC)  # Aug 9 in Lima.
    second_at = datetime(2026, 8, 10, 5, 1, tzinfo=UTC)  # Aug 10 in Lima.
    first = _observation(total_profit=1.0, captured_at=first_at)
    second = _observation(total_profit=1.5, captured_at=second_at)

    first_result = append_pnl_observation(first, live_root=tmp_path)
    second_result = append_pnl_observation(second, live_root=tmp_path)
    loaded = load_pnl_observations(
        live_root=tmp_path,
        symbol="GRVTUSDT",
        strategy_id="strategy-123",
        deploy_ts=DEPLOYED_AT,
    )

    assert first_result.status == "appended"
    assert second_result.status == "appended"
    assert "2026-08-09" in first_result.path.parts
    assert "2026-08-10" in second_result.path.parts
    assert [row["pnl"]["total_profit_usdt"] for row in loaded] == [1.0, 1.5]
    assert first_result.path.read_text(encoding="utf-8").endswith("\n")


def test_atomic_history_supports_windows_length_final_path(tmp_path: Path) -> None:
    captured_at = datetime(2026, 8, 10, 18, 0, tzinfo=UTC)
    observation = _observation(total_profit=1.0, captured_at=captured_at)
    relative_target = (
        Path("2026-08-10")
        / "GRVTUSDT"
        / "pnl_history"
        / str(observation["bot_identity"])
        / "observations"
        / (
            f"{captured_at.strftime('%Y%m%dT%H%M%S%fZ')}_"
            f"{str(observation['observation_id'])[:16]}.json"
        )
    )
    # A final path of 249 characters is valid on legacy Windows, while the old
    # `.{target_name}.{random}.tmp` staging name would have reached 263.
    padding = 249 - len(str(tmp_path / relative_target)) - 1
    assert padding > 0
    live_root = tmp_path / ("x" * padding)
    expected_target = live_root / relative_target
    assert len(str(expected_target)) == 249

    result = append_pnl_observation(observation, live_root=live_root)

    assert result.path == expected_target
    assert result.path.is_file()
    assert not list(live_root.rglob(".tmp-*"))


def test_atomic_create_refuses_to_replace_existing_observation(tmp_path: Path) -> None:
    target = tmp_path / "observation.json"
    target.write_text('{"original":true}\n', encoding="utf-8")

    with pytest.raises(PnlHistoryError, match="refusing to overwrite"):
        pnl_history._atomic_create_json(target, {"replacement": True})

    assert target.read_text(encoding="utf-8") == '{"original":true}\n'
    assert not list(tmp_path.glob(".tmp-*"))


def test_exact_snapshot_duplicate_is_a_noop_even_if_scanner_time_changes(
    tmp_path: Path,
) -> None:
    captured_at = datetime(2026, 8, 10, 18, 0, tzinfo=UTC)
    first = _observation(total_profit=2.0, captured_at=captured_at)
    repeated = _observation(
        total_profit=2.0,
        captured_at=captured_at,
        evaluated_at=captured_at + timedelta(minutes=1),
    )

    assert append_pnl_observation(first, live_root=tmp_path).status == "appended"
    duplicate = append_pnl_observation(repeated, live_root=tmp_path)

    assert duplicate.status == "duplicate"
    loaded = load_pnl_observations(
        live_root=tmp_path,
        symbol="GRVTUSDT",
        strategy_id="strategy-123",
        deploy_ts=DEPLOYED_AT,
    )
    assert len(loaded) == 1


def test_same_identity_and_capture_time_with_changed_snapshot_fails_closed(
    tmp_path: Path,
) -> None:
    captured_at = datetime(2026, 8, 10, 18, 0, tzinfo=UTC)
    append_pnl_observation(
        _observation(total_profit=2.0, captured_at=captured_at),
        live_root=tmp_path,
    )

    with pytest.raises(PnlHistoryError, match="conflicting observation"):
        append_pnl_observation(
            _observation(
                total_profit=2.1,
                captured_at=captured_at,
                source_hash="b" * 64,
            ),
            live_root=tmp_path,
        )


def test_corrupt_existing_record_blocks_append_without_overwriting(
    tmp_path: Path,
) -> None:
    first = _observation(
        total_profit=1.0,
        captured_at=datetime(2026, 8, 10, 18, 0, tzinfo=UTC),
    )
    written = append_pnl_observation(first, live_root=tmp_path)
    written.path.write_text("{broken", encoding="utf-8")
    corrupt_before = written.path.read_text(encoding="utf-8")

    with pytest.raises(PnlHistoryError, match="invalid PnL observation JSON"):
        append_pnl_observation(
            _observation(
                total_profit=1.1,
                captured_at=datetime(2026, 8, 10, 18, 5, tzinfo=UTC),
            ),
            live_root=tmp_path,
        )

    assert written.path.read_text(encoding="utf-8") == corrupt_before
    assert not list(tmp_path.rglob("*.tmp"))


def test_loader_rejects_tampered_derived_identity(tmp_path: Path) -> None:
    observation = _observation(
        total_profit=1.0,
        captured_at=datetime(2026, 8, 10, 18, 0, tzinfo=UTC),
    )
    result = append_pnl_observation(observation, live_root=tmp_path)
    payload = json.loads(result.path.read_text(encoding="utf-8"))
    payload["bot_identity"] = "0" * 64
    integrity_payload = {
        key: value
        for key, value in payload.items()
        if key != "record_integrity_sha256"
    }
    payload["record_integrity_sha256"] = hashlib.sha256(
        json.dumps(
            integrity_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    result.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(PnlHistoryError, match="bot_identity"):
        load_pnl_observations(
            live_root=tmp_path,
            symbol="GRVTUSDT",
            strategy_id="strategy-123",
            deploy_ts=DEPLOYED_AT,
        )


def test_loader_rejects_tampered_forecast_feature(tmp_path: Path) -> None:
    observation = _observation(
        total_profit=1.0,
        captured_at=datetime(2026, 8, 10, 18, 0, tzinfo=UTC),
    )
    result = append_pnl_observation(observation, live_root=tmp_path)
    payload = json.loads(result.path.read_text(encoding="utf-8"))
    payload["features"]["trend_prob"] = 0.99
    result.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(PnlHistoryError, match="record_integrity_sha256"):
        load_pnl_observations(
            live_root=tmp_path,
            symbol="GRVTUSDT",
            strategy_id="strategy-123",
            deploy_ts=DEPLOYED_AT,
        )


def test_invalid_symbol_cannot_escape_live_root(tmp_path: Path) -> None:
    row = _scanner_row(
        total_profit=1.0,
        evaluated_at=datetime(2026, 8, 10, 18, 0, tzinfo=UTC),
    )
    row["symbol"] = "../GRVTUSDT"

    with pytest.raises(PnlHistoryError, match="symbol"):
        build_pnl_observation(
            row,
            deploy_ts=DEPLOYED_AT,
            source_cycle_manifest="cycle.json",
            source_snapshot_path="snapshot.txt",
            source_snapshot_sha256="a" * 64,
        )

    assert not list(tmp_path.rglob("*"))
