from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import pytest

from scripts import run_live_telemetry_controller as controller
from neutralgrid.live.decision.loader import load_bot_specs
from neutralgrid.live.decision.private_events import (
    PRIVATE_EVENT_MANIFEST_SCHEMA_VERSION,
)


UTC = timezone.utc


def _drawer_text(
    *,
    symbol: str = "BTCUSDT",
    strategy_id: str = "413500001",
    lower: float = 1.0,
    upper: float = 2.0,
) -> str:
    return "\n".join(
        [
            symbol,
            "Perp",
            "Futures Grid",
            "Working",
            "Time Created",
            "2026-07-30 10:00:00",
            "Total Profit (USDT)",
            "1.23 (1.23%)",
            "Invested Margin (USDT)",
            "100.00",
            "Matched Profit (USDT)",
            "1.00 (1.00%)",
            "Unmatched PNL (USDT)",
            "0.23 (0.23%)",
            "Funding Fee (USDT)",
            "0.00 (0.00%)",
            "Positions",
            "no position",
            "Pending Order",
            "Grid Details",
            "Mode",
            "Geometric",
            "Price Range",
            f"{lower} - {upper} USDT",
            "Number of Grids",
            "10",
            "Initial Leverage",
            "10x",
            "Order History",
            "Strategy Number",
            strategy_id,
        ]
    )


def _write_cycle(
    tmp_path: Path,
    *,
    completed_at: datetime,
    symbol: str = "BTCUSDT",
    strategy_id: str = "413500001",
    lower: float = 1.0,
    upper: float = 2.0,
) -> Path:
    symbol_dir = tmp_path / "Live" / "2026-07-30" / symbol
    symbol_dir.mkdir(parents=True)
    text_path = symbol_dir / "snapshot.txt"
    metadata_path = symbol_dir / "snapshot.json"
    text_path.write_text(
        _drawer_text(
            symbol=symbol,
            strategy_id=strategy_id,
            lower=lower,
            upper=upper,
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "strategy_id": strategy_id,
                "captured_at_utc": completed_at.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    cycle_dir = tmp_path / "audit" / "cycles"
    cycle_dir.mkdir(parents=True)
    manifest_path = cycle_dir / "cycle_20260730_120000_lima.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "cycle_started_at_utc": (
                    completed_at - timedelta(seconds=5)
                ).isoformat(),
                "cycle_completed_at_utc": completed_at.isoformat(),
                "active_bot_count": 1,
                "symbols": [symbol],
                "files": [
                    {
                        "symbol": symbol,
                        "strategy_id": strategy_id,
                        "text_path": str(text_path),
                        "json_path": str(metadata_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_load_complete_cycle_builds_exact_scanner_entry(tmp_path: Path) -> None:
    completed_at = datetime(2026, 7, 30, 17, 0, tzinfo=UTC)
    manifest_path = _write_cycle(tmp_path, completed_at=completed_at)

    cycle = controller.load_complete_cycle(manifest_path)

    assert cycle.symbols == ("BTCUSDT",)
    assert cycle.strategy_ids == ("413500001",)
    entry = cycle.bots[0].scanner_entry
    assert entry["grid_lower"] == 1.0
    assert entry["grid_upper"] == 2.0
    assert entry["num_grids"] == 10
    assert entry["leverage"] == 10.0
    assert entry["capital_usdt"] == 100.0
    assert entry["execution_telemetry"]["captured_at"] == completed_at.isoformat()


def test_attach_l2_streams_round_trips_through_scanner_registry(tmp_path: Path) -> None:
    completed_at = datetime(2026, 7, 30, 17, 0, tzinfo=UTC)
    cycle = controller.load_complete_cycle(
        _write_cycle(tmp_path, completed_at=completed_at)
    )
    run_dir = tmp_path / "diff_depth" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "l2_risk_snapshots.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps({"symbol": "BTCUSDT", "run_id": "run-1"}),
        encoding="utf-8",
    )
    manifest = tmp_path / "collector_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "symbol_run_dirs": {"BTCUSDT": str(run_dir)},
            }
        ),
        encoding="utf-8",
    )

    attached = controller.attach_l2_streams(
        cycle,
        manifest_paths=[manifest],
        max_age_seconds=12.0,
        history_window_seconds=240.0,
    )
    registry = controller.write_scanner_registry(attached, runtime_dir=tmp_path / "runtime")
    specs, warnings = load_bot_specs(registry, now=completed_at + timedelta(seconds=1))

    assert warnings == []
    assert len(specs) == 1
    assert specs[0].l2_stream is not None
    assert specs[0].l2_stream.run_id == "run-1"
    assert specs[0].l2_stream.max_age_seconds == 12.0
    assert specs[0].l2_stream.history_window_seconds == 240.0


def test_public_trade_attachment_requires_exact_collector_strategy_target(
    tmp_path: Path,
) -> None:
    completed_at = datetime(2026, 7, 30, 17, 0, tzinfo=UTC)
    cycle = controller.load_complete_cycle(
        _write_cycle(tmp_path, completed_at=completed_at)
    )
    run_dir = tmp_path / "diff_depth" / "run-trades"
    run_dir.mkdir(parents=True)
    (run_dir / "l2_risk_snapshots.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "public_agg_trades.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "symbol": "BTCUSDT",
                "run_id": "run-trades",
                "target": {
                    "symbol": "BTCUSDT",
                    "strategy_id": "413500001",
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "collector_manifest.json"
    payload = {
        "run_id": "run-trades",
        "targets": [{"symbol": "BTCUSDT", "strategy_id": "wrong-strategy"}],
        "symbol_run_dirs": {"BTCUSDT": str(run_dir)},
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(controller.ControllerError, match="collector target mismatch"):
        controller.attach_l2_streams(
            cycle,
            manifest_paths=[manifest],
            max_age_seconds=12.0,
            history_window_seconds=240.0,
        )

    payload["targets"][0]["strategy_id"] = "413500001"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    attached = controller.attach_l2_streams(
        cycle,
        manifest_paths=[manifest],
        max_age_seconds=12.0,
        history_window_seconds=240.0,
    )
    registry = controller.write_scanner_registry(
        attached,
        runtime_dir=tmp_path / "runtime-public-trades",
    )
    specs, warnings = load_bot_specs(
        registry,
        now=completed_at + timedelta(seconds=1),
    )

    assert warnings == []
    assert specs[0].l2_stream is not None
    assert specs[0].l2_stream.strategy_id == "413500001"


def test_attach_private_events_requires_exact_identity_and_round_trips(
    tmp_path: Path,
) -> None:
    completed_at = datetime(2026, 7, 30, 17, 0, tzinfo=UTC)
    cycle = controller.load_complete_cycle(
        _write_cycle(tmp_path, completed_at=completed_at)
    )
    event_path = tmp_path / "private_events.jsonl"
    event_path.write_text("", encoding="utf-8")
    manifest = tmp_path / "private_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": PRIVATE_EVENT_MANIFEST_SCHEMA_VERSION,
                "run_id": "private-run",
                "symbol": "BTCUSDT",
                "strategy_id": "413500001",
                "event_path": str(event_path),
            }
        ),
        encoding="utf-8",
    )

    attached = controller.attach_private_event_streams(
        cycle,
        manifest_paths=[manifest],
        max_age_seconds=300.0,
        history_window_seconds=3600.0,
    )
    registry = controller.write_scanner_registry(
        attached,
        runtime_dir=tmp_path / "runtime-private",
    )
    specs, warnings = load_bot_specs(
        registry,
        now=completed_at + timedelta(seconds=1),
    )

    assert warnings == []
    assert specs[0].private_event_stream is not None
    assert specs[0].private_event_stream.strategy_id == "413500001"
    assert specs[0].private_event_stream.run_id == "private-run"

    bad_payload = json.loads(manifest.read_text(encoding="utf-8"))
    bad_payload["strategy_id"] = "wrong"
    manifest.write_text(json.dumps(bad_payload), encoding="utf-8")
    with pytest.raises(controller.ControllerError, match="not active"):
        controller.attach_private_event_streams(
            cycle,
            manifest_paths=[manifest],
            max_age_seconds=300.0,
            history_window_seconds=3600.0,
        )


def test_persist_cycle_pnl_history_is_cross_iteration_and_deduplicated(
    tmp_path: Path,
) -> None:
    completed_at = datetime(2026, 7, 30, 17, 0, tzinfo=UTC)
    cycle = controller.load_complete_cycle(
        _write_cycle(tmp_path, completed_at=completed_at)
    )
    row = {
        "ts": (completed_at + timedelta(seconds=2)).isoformat(),
        "symbol": "BTCUSDT",
        "strategy_id": "413500001",
        "candidate_id": None,
        "verdict": "CONTINUE",
        "reasons": [],
        "execution_telemetry": cycle.bots[0].scanner_entry["execution_telemetry"],
        "profit_deterioration": {
            "peak_total_profit_usdt": 1.23,
            "giveback_usdt": 0.0,
        },
        "evaluation": {
            "evaluated_at_utc": (completed_at + timedelta(seconds=2)).isoformat(),
            "price": 1.5,
            "range_prob": 0.6,
            "trend_prob": 0.2,
            "persistence_prob": 0.7,
            "l2_risk": None,
            "execution_risk": None,
            "private_event_evidence": None,
        },
    }

    first = controller.persist_cycle_pnl_history(
        cycle,
        [row],
        live_root=tmp_path / "Live",
    )
    second = controller.persist_cycle_pnl_history(
        cycle,
        [dict(row, ts=(completed_at + timedelta(minutes=1)).isoformat())],
        live_root=tmp_path / "Live",
    )

    assert first[0]["status"] == "appended"
    assert second[0]["status"] == "duplicate"
    assert first[0]["history_count"] == 1
    assert second[0]["history_count"] == 1
    assert Path(first[0]["path"]).is_file()
    assert len(list((tmp_path / "Live").rglob("pnl_history/*/observations/*.json"))) == 1

    unavailable = controller.build_cycle_shadow_forecasts(
        cycle,
        live_root=tmp_path / "Live",
        artifact_dir=tmp_path / "missing-artifact",
    )
    assert unavailable[0]["status"] == "unavailable"
    assert unavailable[0]["runtime_effect"] == "none"
    assert "cannot read forecast metadata" in unavailable[0]["reason"]


def test_shadow_forecast_is_explicitly_not_configured_without_artifact(
    tmp_path: Path,
) -> None:
    cycle = controller.load_complete_cycle(
        _write_cycle(
            tmp_path,
            completed_at=datetime(2026, 7, 30, 17, 0, tzinfo=UTC),
        )
    )

    forecasts = controller.build_cycle_shadow_forecasts(
        cycle,
        live_root=tmp_path / "Live",
        artifact_dir=None,
    )

    assert forecasts == [
        {
            "symbol": "BTCUSDT",
            "strategy_id": "413500001",
            "status": "not_configured",
            "runtime_effect": "none",
        }
    ]


def test_load_complete_cycle_rejects_manifest_drawer_identity_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path = _write_cycle(
        tmp_path,
        completed_at=datetime(2026, 7, 30, 17, 0, tzinfo=UTC),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["files"][0]["strategy_id"] = "413500999"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(controller.ControllerError, match="metadata strategy mismatch"):
        controller.load_complete_cycle(manifest_path)


def test_validate_cycle_freshness_rejects_stale_cycle(tmp_path: Path) -> None:
    completed_at = datetime(2026, 7, 30, 17, 0, tzinfo=UTC)
    cycle = controller.load_complete_cycle(
        _write_cycle(tmp_path, completed_at=completed_at)
    )

    with pytest.raises(controller.ControllerError, match="stale"):
        controller.validate_cycle_freshness(
            cycle,
            now=completed_at + timedelta(seconds=901),
            max_age_seconds=900,
        )


def test_acquire_cycle_uses_only_fresh_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed_at = datetime(2026, 7, 30, 17, 0, tzinfo=UTC)
    cycle = controller.load_complete_cycle(
        _write_cycle(tmp_path, completed_at=completed_at)
    )
    args = Namespace(
        telemetry_audit_dir=tmp_path / "audit",
        max_telemetry_age_seconds=900.0,
    )

    def fail_fetch(_args: Namespace) -> controller.TelemetryCycle:
        raise controller.ControllerError("Chrome unavailable")

    monkeypatch.setattr(controller, "fetch_private_cycle", fail_fetch)
    loaded, source = controller.acquire_cycle(
        args,
        now=completed_at + timedelta(seconds=60),
    )

    assert source == "fresh_cache"
    assert loaded.strategy_ids == cycle.strategy_ids


def test_build_action_intents_requires_valid_adjust_bounds() -> None:
    rows = [
        {
            "ts": "2026-07-30T17:00:00+00:00",
            "symbol": "BTCUSDT",
            "strategy_id": "413500001",
            "verdict": "ADJUST",
            "reasons": ["price_near_lower"],
            "suggested_grid_lower": 0.9,
            "suggested_grid_upper": 1.9,
        },
        {
            "ts": "2026-07-30T17:00:00+00:00",
            "symbol": "ETHUSDT",
            "strategy_id": "413500002",
            "verdict": "CONTINUE",
            "reasons": [],
        },
    ]

    intents = controller.build_action_intents(rows, iteration_id="iteration")

    assert len(intents) == 1
    assert intents[0]["action"] == "ADJUST"
    assert intents[0]["suggested_grid_lower"] == 0.9
    assert len(intents[0]["idempotency_key"]) == 64

    bad_row = dict(rows[0], suggested_grid_lower=None)
    with pytest.raises(controller.ControllerError, match="lacks valid"):
        controller.build_action_intents([bad_row], iteration_id="iteration")


def test_route_actions_blocks_without_explicit_authority(tmp_path: Path) -> None:
    args = Namespace(
        controller_audit_dir=tmp_path,
        allow_actions=False,
        action_executable=None,
        action_arg=[],
    )
    intent = {
        "schema_version": "neutralgrid_action_intent_v1",
        "iteration_id": "iteration",
        "idempotency_key": "abc",
        "symbol": "BTCUSDT",
        "strategy_id": "413500001",
        "action": "END",
        "suggested_grid_lower": None,
        "suggested_grid_upper": None,
        "reasons": ["test"],
        "decision_ts": "2026-07-30T17:00:00+00:00",
    }

    outcomes = controller.route_actions(args, intents=[intent])

    assert outcomes[0]["status"] == "blocked_allow_actions_not_set"
    ledger = (tmp_path / "action_ledger.jsonl").read_text(encoding="utf-8")
    assert "blocked_allow_actions_not_set" in ledger


def test_route_actions_auto_adjust_requires_exact_external_approval(
    tmp_path: Path,
) -> None:
    approval_dir = tmp_path / "approvals"
    approval_dir.mkdir()
    args = Namespace(
        controller_audit_dir=tmp_path / "audit",
        allow_actions=False,
        action_executable=None,
        action_arg=["--browser-transport", "extension"],
        action_approval_dir=approval_dir,
        python=Path("python"),
    )
    intent = {
        "schema_version": "neutralgrid_action_intent_v1",
        "iteration_id": "iteration",
        "idempotency_key": "a" * 64,
        "symbol": "BTCUSDT",
        "strategy_id": "413500001",
        "action": "ADJUST",
        "suggested_grid_lower": 60000.0,
        "suggested_grid_upper": 70000.0,
        "reasons": ["test"],
        "decision_ts": "2026-07-30T17:00:00+00:00",
    }

    outcomes = controller.route_actions(args, intents=[intent])

    assert outcomes[0]["status"] == "blocked_exact_approval_unavailable"
    assert "exact unexpired" in outcomes[0]["approval_error"]


def test_route_actions_auto_invokes_canonical_adjust_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = controller._utc_now()
    approval_dir = tmp_path / "approvals"
    approval_dir.mkdir()
    approval = {
        "schema_version": "neutralgrid_action_approval_v1",
        "idempotency_key": "b" * 64,
        "symbol": "BTCUSDT",
        "strategy_id": "413500001",
        "action": "ADJUST",
        "suggested_grid_lower": 60000.0,
        "suggested_grid_upper": 70000.0,
        "preserve_current_position": True,
        "approved_at_utc": (now - timedelta(seconds=10)).isoformat(),
        "expires_at_utc": (now + timedelta(seconds=120)).isoformat(),
    }
    approval_path = approval_dir / "btc.json"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    args = Namespace(
        controller_audit_dir=tmp_path / "audit",
        allow_actions=False,
        action_executable=None,
        action_arg=["--browser-transport", "extension"],
        action_approval_dir=approval_dir,
        action_timeout_seconds=30.0,
        python=Path("python"),
    )
    intent = {
        "schema_version": "neutralgrid_action_intent_v1",
        "iteration_id": "iteration",
        "idempotency_key": "b" * 64,
        "symbol": "BTCUSDT",
        "strategy_id": "413500001",
        "action": "ADJUST",
        "suggested_grid_lower": 60000.0,
        "suggested_grid_upper": 70000.0,
        "reasons": ["test"],
        "decision_ts": now.isoformat(),
    }
    response = {
        "schema_version": "neutralgrid_action_execution_v1",
        "idempotency_key": "b" * 64,
        "status": "executed",
        "preserve_current_position": True,
        "additional_investment": 0.0,
        "executed_grid_lower": 60000.0,
        "executed_grid_upper": 70000.0,
        "price_decimals": 2,
    }
    observed: dict[str, Any] = {}

    def fake_run_process(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed["command"] = list(command)
        observed["stdin_text"] = kwargs["stdin_text"]
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

    monkeypatch.setattr(controller, "_run_process", fake_run_process)
    monkeypatch.setattr(
        controller,
        "fetch_private_cycle",
        lambda _args: Namespace(manifest_path=tmp_path / "post-cycle.json"),
    )
    monkeypatch.setattr(controller, "_verify_action_effect", lambda _intent, _cycle: None)

    outcomes = controller.route_actions(args, intents=[intent])

    assert outcomes[0]["status"] == "verified"
    assert observed["command"][:4] == [
        "python",
        str(controller.CANONICAL_ADJUST_EXECUTOR),
        "--approval-file",
        str(approval_path.resolve()),
    ]
    assert json.loads(str(observed["stdin_text"]))["idempotency_key"] == "b" * 64


def test_verify_action_effect_checks_end_and_adjust(tmp_path: Path) -> None:
    completed_at = datetime(2026, 7, 30, 17, 0, tzinfo=UTC)
    active_cycle = controller.load_complete_cycle(
        _write_cycle(
            tmp_path,
            completed_at=completed_at,
            lower=0.9,
            upper=1.9,
        )
    )
    adjust = {
        "symbol": "BTCUSDT",
        "strategy_id": "413500001",
        "action": "ADJUST",
        "suggested_grid_lower": 0.9,
        "suggested_grid_upper": 1.9,
    }
    controller._verify_action_effect(adjust, active_cycle)

    end = {
        "symbol": "ETHUSDT",
        "strategy_id": "413500002",
        "action": "END",
        "suggested_grid_lower": None,
        "suggested_grid_upper": None,
    }
    controller._verify_action_effect(end, active_cycle)

    with pytest.raises(controller.ControllerError, match="remains active"):
        controller._verify_action_effect(
            dict(end, symbol="BTCUSDT", strategy_id="413500001"),
            active_cycle,
        )


def test_executor_bounds_accept_only_single_quantum_outward_rounding() -> None:
    intent = {
        "symbol": "SKYAIUSDT",
        "strategy_id": "413527269",
        "action": "ADJUST",
        "suggested_grid_lower": 0.023595,
        "suggested_grid_upper": 0.025825,
    }
    response = {
        "schema_version": "neutralgrid_action_execution_v1",
        "preserve_current_position": True,
        "additional_investment": 0.0,
        "executed_grid_lower": 0.02359,
        "executed_grid_upper": 0.02583,
        "price_decimals": 5,
    }

    verified = controller._verification_intent_from_executor(intent, response)

    assert verified["suggested_grid_lower"] == 0.02359
    assert verified["suggested_grid_upper"] == 0.02583

    with pytest.raises(controller.ControllerError, match="outward rounding"):
        controller._verification_intent_from_executor(
            intent,
            dict(response, executed_grid_lower=0.02360),
        )


def test_executor_bounds_require_position_preservation_and_zero_investment() -> None:
    intent = {
        "symbol": "SKYAIUSDT",
        "strategy_id": "413527269",
        "action": "ADJUST",
        "suggested_grid_lower": 0.023595,
        "suggested_grid_upper": 0.025825,
    }
    response = {
        "schema_version": "neutralgrid_action_execution_v1",
        "preserve_current_position": False,
        "additional_investment": 0.0,
        "executed_grid_lower": 0.02359,
        "executed_grid_upper": 0.02583,
        "price_decimals": 5,
    }

    with pytest.raises(controller.ControllerError, match="preservation"):
        controller._verification_intent_from_executor(intent, response)

    with pytest.raises(controller.ControllerError, match="additional investment"):
        controller._verification_intent_from_executor(
            intent,
            dict(response, preserve_current_position=True, additional_investment=1.0),
        )


def test_parse_args_requires_allow_actions_for_executor() -> None:
    with pytest.raises(SystemExit):
        controller.parse_args(["--action-executable", "executor.exe"])


def test_parse_args_requires_manifest_and_forbids_debug_endpoint_in_plugin_mode() -> None:
    with pytest.raises(SystemExit):
        controller.parse_args(["--acquisition-mode", "plugin-manifest"])
    with pytest.raises(SystemExit):
        controller.parse_args(
            [
                "--acquisition-mode",
                "plugin-manifest",
                "--cycle-manifest",
                "cycle.json",
                "--observational-only",
                "--debug-endpoint",
                "http://127.0.0.1:9222",
            ]
        )
    with pytest.raises(SystemExit):
        controller.parse_args(
            [
                "--acquisition-mode",
                "plugin-manifest",
                "--cycle-manifest",
                "cycle.json",
            ]
        )
    with pytest.raises(SystemExit):
        controller.parse_args(
            [
                "--acquisition-mode",
                "plugin-manifest",
                "--cycle-manifest",
                "cycle.json",
                "--observational-only",
                "--allow-actions",
            ]
        )


def test_observational_mode_never_routes_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = Namespace(observational_only=True)
    intents = [
        {
            "symbol": "BTCUSDT",
            "strategy_id": "413500001",
            "action": "ADJUST",
            "idempotency_key": "a" * 64,
        }
    ]

    def forbidden_route(_args: Namespace, *, intents: Sequence[dict[str, Any]]) -> None:
        raise AssertionError(f"route_actions called for {intents}")

    monkeypatch.setattr(controller, "route_actions", forbidden_route)
    outcomes = controller.route_or_observe_actions(args, intents=intents)

    assert outcomes == [
        {
            "symbol": "BTCUSDT",
            "strategy_id": "413500001",
            "action": "ADJUST",
            "idempotency_key": "a" * 64,
            "status": "observational_not_executed",
            "runtime_effect": "none",
        }
    ]


def test_shadow_volatility_not_configured_is_explicitly_verdict_inert(
    tmp_path: Path,
) -> None:
    completed_at = datetime(2026, 7, 30, 17, 0, tzinfo=UTC)
    cycle = controller.load_complete_cycle(
        _write_cycle(tmp_path, completed_at=completed_at)
    )

    forecasts = controller.build_cycle_shadow_volatility_forecasts(
        cycle,
        artifact_dir=None,
        contract_path=tmp_path / "unused.json",
        price_store=tmp_path / "prices",
        requested_horizon_minutes=360,
        max_data_age_seconds=180.0,
        asof_utc=completed_at,
    )

    assert forecasts == [
        {
            "symbol": "BTCUSDT",
            "strategy_id": "413500001",
            "status": "not_configured",
            "eligibility": False,
            "verdict_influence": False,
            "runtime_effect": "none",
        }
    ]


def test_shadow_volatility_failure_cannot_change_action_intent(
    tmp_path: Path,
) -> None:
    completed_at = datetime(2026, 7, 30, 17, 0, tzinfo=UTC)
    cycle = controller.load_complete_cycle(
        _write_cycle(tmp_path, completed_at=completed_at)
    )
    bad_contract = tmp_path / "bad_contract.json"
    bad_contract.write_text("{}", encoding="utf-8")
    rows = [
        {
            "symbol": "BTCUSDT",
            "strategy_id": "413500001",
            "verdict": "ADJUST",
            "suggested_grid_lower": 1.1,
            "suggested_grid_upper": 1.9,
        }
    ]
    before = controller.build_action_intents(rows, iteration_id="iter-1")

    forecasts = controller.build_cycle_shadow_volatility_forecasts(
        cycle,
        artifact_dir=tmp_path / "artifact",
        contract_path=bad_contract,
        price_store=tmp_path / "prices",
        requested_horizon_minutes=360,
        max_data_age_seconds=180.0,
        asof_utc=completed_at,
    )
    after = controller.build_action_intents(rows, iteration_id="iter-1")

    assert forecasts[0]["status"] == "unavailable"
    assert forecasts[0]["verdict_influence"] is False
    assert after == before
