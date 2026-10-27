"""Transport-free controller and adapter fault experiments."""
from __future__ import annotations

import copy
import json
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts import run_live_telemetry_controller as controller
from neutralgrid.live.decision import action_shadow_io as adapter
from neutralgrid.live.decision.pnl_history import PnlHistoryError, build_pnl_observation
from neutralgrid.live.decision.state_action_reward import BotIdentity, SourceRef, State, file_hash, load_events


T0 = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)


def cycle(tmp_path):
    bot = controller.TelemetryBot(symbol="BTCUSDT", strategy_id="123", captured_at_utc=T0,
                                  raw_text_path=tmp_path / "raw.txt",
                                  scanner_entry={"deploy_ts": (T0 - timedelta(days=1)).isoformat()})
    return controller.TelemetryCycle(manifest_path=tmp_path / "manifest.json",
                                     started_at_utc=T0, completed_at_utc=T0, bots=(bot,))


def row(verdict="ADJUST") -> dict[str, Any]:
    return dict(symbol="BTCUSDT", strategy_id="123", ts=T0.isoformat(), verdict=verdict,
                suggested_grid_lower=95, suggested_grid_upper=115, reasons=["incumbent"])


def fail(*args, **kwargs):
    raise OSError("synthetic unavailable transport/model")


def test_unconfigured_shadow_does_not_load_or_route(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "load_pnl_observations", fail)
    monkeypatch.setattr(controller, "route_or_observe_actions", fail)
    monkeypatch.setattr(adapter, "report_shadow_row", fail)
    assert controller.build_cycle_action_shadows(Namespace(), cycle(tmp_path), [row()]) == []


def test_action_intent_states_capital_priority_without_trusting_row_claims():
    supplied = row()
    supplied["capital_preservation_policy"] = {"execution_authorized": True}
    policy = controller.build_action_intents([supplied], iteration_id="fixture")[0]["capital_preservation_policy"]
    assert policy["primary_objective"] == "preserve_invested_capital"
    assert policy["execution_authorized"] is False
    assert policy["minimum_liquidation_buffer_pct"] is None


@pytest.mark.parametrize("verdict", ["CONTINUE", "ADJUST", "END"])
def test_shadow_failure_cannot_change_rows_or_incumbent_intents(tmp_path, monkeypatch, verdict):
    args = Namespace(shadow_action_artifact_dir=tmp_path / "missing", shadow_action_limits=None,
                     record_shadow_decisions=False, live_root=tmp_path / "Live", observational_only=True)
    incumbent = row(verdict)
    before = copy.deepcopy(incumbent)
    intents = controller.build_action_intents([incumbent], iteration_id="fixture")
    monkeypatch.setattr(controller, "load_pnl_observations", lambda **kwargs: [])
    monkeypatch.setattr(controller, "route_or_observe_actions", fail)
    monkeypatch.setattr(adapter, "report_shadow_row", fail)
    reports = controller.build_cycle_action_shadows(args, cycle(tmp_path), [incumbent])
    assert reports[0]["verdict"] == verdict and not reports[0]["available"]
    assert not reports[0]["execution_eligible"] and reports[0]["observed_reward"] is None
    assert incumbent == before
    assert controller.build_action_intents([incumbent], iteration_id="fixture") == intents


def test_invalid_incumbent_uses_controller_error_boundary(tmp_path):
    args = Namespace(shadow_action_artifact_dir=tmp_path, shadow_action_limits=None,
                     record_shadow_decisions=False)
    with pytest.raises(controller.ControllerError):
        controller.build_cycle_action_shadows(args, cycle(tmp_path), [row("UNKNOWN")])


def test_optional_import_failure_does_not_abort_controller_iteration(tmp_path, monkeypatch):
    args = controller.parse_args(["--observational-only"])
    args.controller_audit_dir = tmp_path / "audit"
    args.runtime_dir = tmp_path / "runtime"
    args.live_root = tmp_path / "Live"
    snapshot = cycle(tmp_path)
    incumbent = row("CONTINUE")
    monkeypatch.setattr(controller, "acquire_cycle", lambda *args, **kwargs: (snapshot, "fixture"))
    monkeypatch.setattr(controller, "attach_l2_streams", lambda cycle, **kwargs: cycle)
    monkeypatch.setattr(controller, "attach_private_event_streams", lambda cycle, **kwargs: cycle)
    monkeypatch.setattr(controller, "write_scanner_registry", lambda *args, **kwargs: tmp_path / "registry")
    monkeypatch.setattr(controller, "run_scanner_tick", lambda *args, **kwargs: ([incumbent], {"returncode": 0}))
    monkeypatch.setattr(controller, "persist_cycle_pnl_history", lambda *args, **kwargs: [])
    monkeypatch.setattr(controller, "build_cycle_shadow_forecasts", lambda *args, **kwargs: [])
    monkeypatch.setattr(controller, "build_cycle_shadow_volatility_forecasts", lambda *args, **kwargs: [])
    def unavailable_import(*args):
        raise ImportError("synthetic optional dependency failure")
    monkeypatch.setattr(controller, "build_cycle_action_shadows", unavailable_import)
    monkeypatch.setattr(controller, "route_actions", fail)
    report = controller.run_iteration(args)
    assert report["status"] == "complete" and report["verdicts"][0]["verdict"] == "CONTINUE"
    assert report["action_outcomes"] == []
    assert report["shadow_actions"]["diagnostics"] == ["shadow_controller_failure:ImportError"]


def test_end_decision_is_recorded_without_loading_model(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text("{}")
    state = State(bot=BotIdentity(symbol="BTCUSDT", strategy_id="123", deployed_at=T0 - timedelta(days=1)),
                  captured_at=T0, available_at=T0, feature_window_start=T0,
                  sources=(SourceRef(path=str(source), sha256=file_hash(source)),))
    monkeypatch.setattr(adapter, "state_from_observations", lambda *args: state)
    monkeypatch.setattr(adapter, "read_model", fail)
    result = adapter.report_shadow_row(row=row("END"), observations=[], model_path=tmp_path / "missing",
                                       limits_payload=None, live_root=tmp_path / "Live")
    events, rejected = load_events(tmp_path / "Live")
    assert result["verdict"] == "END" and result["decision_event_path"]
    assert len(events) == 1 and not rejected
    assert events[0].payload["incumbent"]["verdict"] == events[0].payload["shadow"]["verdict"] == "END"


def test_corrupt_optional_model_preserves_adjust(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text("{}")
    state = State(bot=BotIdentity(symbol="BTCUSDT", strategy_id="123", deployed_at=T0 - timedelta(days=1)),
                  captured_at=T0, available_at=T0, feature_window_start=T0,
                  sources=(SourceRef(path=str(source), sha256=file_hash(source)),))
    monkeypatch.setattr(adapter, "state_from_observations", lambda *args: state)
    model = tmp_path / "model.json"
    model.write_text('{"truncated":')
    result = adapter.report_shadow_row(row=row(), observations=[], model_path=model, limits_payload=None)
    assert result["verdict"] == "ADJUST" and not result["available"]
    assert result["observed_reward"] is None


def test_adapter_binds_full_history_and_ignores_future_observations(tmp_path):
    history = []
    incumbent = row("CONTINUE")
    for i in range(3):
        at = T0 + timedelta(minutes=5 * (i - 1))
        raw = tmp_path / f"raw-{i}.txt"
        raw.write_text(f"synthetic drawer {i}")
        r = dict(incumbent, ts=at.isoformat(), evaluation={"evaluated_at_utc": at.isoformat(), "price": 105},
                 execution_telemetry={"captured_at": at.isoformat(), "pnl": {"total_profit_usdt": i},
                                      "position_inventory": {"size_base": 1},
                                      "grid": {"price_range_lower": 90, "price_range_upper": 110,
                                               "num_grids": 12, "invested_margin_usdt": 100}})
        observation = build_pnl_observation(r, deploy_ts=T0 - timedelta(days=1),
                                            source_cycle_manifest=str(tmp_path / "manifest"),
                                            source_snapshot_path=str(raw), source_snapshot_sha256=file_hash(raw))
        path = tmp_path / f"observation-{i}.json"
        path.write_text(json.dumps(observation))
        observation["_source_path"] = str(path)
        history.append(observation)
    incumbent["evaluation"] = {"price": 105}
    state = adapter.state_from_observations(history, incumbent)
    assert len(state.sources) == 4 and state.feature_window_start == T0 - timedelta(minutes=5)
    assert state.features["previous_delta_pnl_usdt"] == 1
    future = copy.deepcopy(history[-1])
    future["pnl"]["total_profit_usdt"] = 1e9
    assert state == adapter.state_from_observations(history[:2] + [future], incumbent)
    assert state.equity_usdt is None and state.peak_equity_usdt is None and state.symbol_available is None
    # A scanner row is not a source-verified exact-bot equity fact. Even finite
    # values must not silently bypass the native adapter's accounting boundary.
    incumbent["execution_telemetry"] = {
        "pnl": {"equity_usdt": 9999}, "risk": {"peak_equity_usdt": 10000}}
    unverified = adapter.state_from_observations(history, incumbent)
    assert unverified.equity_usdt is None and unverified.peak_equity_usdt is None
    path = Path(history[0]["_source_path"])
    path.write_text("{}")
    with pytest.raises(PnlHistoryError):
        adapter.state_from_observations(history, incumbent)
