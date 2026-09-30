from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import live_decision_scanner as scanner
from neutralgrid.data.diff_depth import MANIFEST_SCHEMA_VERSION
from neutralgrid.live.decision.l2_risk import L2StreamRef, load_validated_l2_manifest
from scripts import run_live_telemetry_controller as controller


NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_slow_first_bot_does_not_make_second_live_heartbeat_future_dated(tmp_path, monkeypatch):
    current = [NOW]
    manifest = tmp_path / "manifest.json"
    ref = L2StreamRef(feature_path=tmp_path / "l2.jsonl", manifest_path=manifest,
                      symbol="BTCUSDT", run_id="run-1", max_age_seconds=15)
    specs = [SimpleNamespace(state_key="first"), SimpleNamespace(state_key="second")]
    monkeypatch.setattr(scanner, "_load_specs_from_yaml_paths", lambda *a, **kw: (specs, []))
    monkeypatch.setattr(scanner, "load_history", lambda *a, **kw: None)
    monkeypatch.setattr(scanner, "save_history", lambda *a, **kw: None)
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return current[0]
    monkeypatch.setattr(scanner, "datetime", Clock)
    observed = []
    async def evaluate(spec, *, now, **kw):
        manifest.write_text(json.dumps({
            "schema_version": MANIFEST_SCHEMA_VERSION, "symbol": "BTCUSDT",
            "run_id": "run-1", "status": "running", "current_phase": "live",
            "updated_at_utc": current[0].isoformat(),
        }), encoding="utf-8")
        load_validated_l2_manifest(ref, now=now)
        observed.append(now)
        current[0] += timedelta(seconds=12)
        return SimpleNamespace(evaluated_at_utc=now)
    monkeypatch.setattr(scanner, "evaluate_bot", evaluate)
    decision_times = []
    def decide(evaluation, history, cfg, now):
        decision_times.append(now)
        return SimpleNamespace(new_history=None, recommendation=None, should_emit=False)
    monkeypatch.setattr(scanner, "decide", decide)
    await scanner._run_tick(argparse.Namespace(bots=Path("bots.yaml"), state_dir=tmp_path),
                            [], context=Mock(), client=Mock(), cfg=Mock(), now=NOW)
    assert observed == [NOW, NOW + timedelta(seconds=12)]
    assert decision_times == observed


@pytest.fixture
def scanner_output(tmp_path, monkeypatch):
    args = controller.parse_args([])
    args.controller_audit_dir = tmp_path / "controller"
    args.scanner_state_dir = tmp_path / "state"
    args.require_l2_evidence = True
    ref = {"run_id": "run-1", "max_age_seconds": 15.0}
    cycle = controller.TelemetryCycle(tmp_path / "cycle.json", NOW, NOW, (
        controller.TelemetryBot("BTCUSDT", "123", NOW, tmp_path / "raw.txt", {"l2_stream": ref}),
    ))
    evaluation = {"evaluated_at_utc": NOW.isoformat(), "diagnostics": [],
                  "l2_risk": {"run_id": "run-1", "captured_at_utc": NOW.isoformat(), "age_seconds": 0.0}}
    row = {"symbol": "BTCUSDT", "strategy_id": "123", "verdict": "CONTINUE", "evaluation": evaluation}
    def run(command, **kwargs):
        log_dir = Path(command[command.index("--logs-dir") + 1])
        (log_dir / "live_decisions_20260926.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(controller, "_run_process", run)
    return args, cycle, row


@pytest.mark.parametrize("failure", ["missing", "stale", "future", "nan", "wrong_run", "diagnostic", "bad_diagnostics", "missing_evaluation"])
def test_successful_process_cannot_publish_invalid_required_l2(scanner_output, failure):
    args, cycle, row = scanner_output
    evidence = row["evaluation"]["l2_risk"]
    if failure == "missing":
        row["evaluation"]["l2_risk"] = None
    elif failure in {"stale", "future", "nan"}:
        age = {"stale": 16, "future": -6, "nan": float("nan")}[failure]
        evidence["age_seconds"] = age
        if failure != "nan":
            evidence["captured_at_utc"] = (NOW - timedelta(seconds=age)).isoformat()
    elif failure == "wrong_run":
        evidence["run_id"] = "old-run"
    elif failure == "diagnostic":
        row["evaluation"]["diagnostics"] = ["l2_stream_unavailable:heartbeat is stale"]
    elif failure == "bad_diagnostics":
        row["evaluation"]["diagnostics"] = "not a list"
    else:
        row.pop("evaluation")
    with pytest.raises(controller.ControllerError, match="required L2"):
        controller.run_scanner_tick(args, cycle=cycle, registry_path=Path("registry.yaml"), iteration_id="test")


def test_valid_required_l2_preserves_nonblocking_source_warning(scanner_output):
    args, cycle, row = scanner_output
    row["evaluation"]["diagnostics"] = ["utility_calibrator_unavailable", "candidate_link_missing"]
    rows, _ = controller.run_scanner_tick(args, cycle=cycle, registry_path=Path("registry.yaml"), iteration_id="test")
    assert rows == [row]


def test_gate_is_opt_in_for_legacy_controller(scanner_output):
    args, cycle, row = scanner_output
    args.require_l2_evidence = False
    row["evaluation"]["l2_risk"] = None
    rows, _ = controller.run_scanner_tick(args, cycle=cycle, registry_path=Path("registry.yaml"), iteration_id="test")
    assert rows == [row]
