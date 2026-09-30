from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import pytest

from scripts import run_live_telemetry_controller as controller
from scripts import run_drawer_verdict_cycle as recurring


NOW = datetime(2026, 9, 25, 18, tzinfo=timezone.utc)


def cycle_at(age: float) -> controller.TelemetryCycle:
    captured = NOW - timedelta(seconds=age)
    return controller.TelemetryCycle(
        manifest_path=Path("cycle.json"),
        started_at_utc=captured - timedelta(seconds=1),
        completed_at_utc=NOW,
        bots=(controller.TelemetryBot("BTCUSDT", "123", captured, Path("raw.txt"), {}),),
    )


def test_old_drawer_cannot_be_refreshed_by_new_completion_timestamp() -> None:
    with pytest.raises(controller.ControllerError, match="BTCUSDT.*stale"):
        controller.validate_cycle_freshness(cycle_at(901), now=NOW, max_age_seconds=900)


def test_future_drawer_is_rejected() -> None:
    with pytest.raises(controller.ControllerError, match="BTCUSDT.*future"):
        controller.validate_cycle_freshness(cycle_at(-6), now=NOW, max_age_seconds=900)


@pytest.mark.parametrize("limit", [float("nan"), float("inf"), 0, -1])
def test_invalid_freshness_limit_cannot_disable_gate(limit: float) -> None:
    with pytest.raises(controller.ControllerError, match="max_age_seconds"):
        controller.validate_cycle_freshness(cycle_at(1), now=NOW, max_age_seconds=limit)


def test_freshness_boundary_is_inclusive() -> None:
    controller.validate_cycle_freshness(cycle_at(900), now=NOW, max_age_seconds=900)


def test_capture_outside_cycle_is_rejected() -> None:
    cycle = replace(cycle_at(1), started_at_utc=NOW)
    with pytest.raises(controller.ControllerError, match="outside.*cycle"):
        controller.validate_cycle_freshness(cycle, now=NOW, max_age_seconds=900)


@pytest.fixture
def admitted_cycle(tmp_path, monkeypatch):
    raw = tmp_path / "drawer.txt"
    raw.write_text("original capture", encoding="utf-8")
    manifest = tmp_path / "cycles" / "cycle.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"source": "chrome_plugin"}), encoding="utf-8")
    cycle = replace(cycle_at(10), manifest_path=manifest,
                    bots=(replace(cycle_at(10).bots[0], raw_text_path=raw),))
    args = recurring.parse_args(["--cycle-manifest", str(manifest), "--audit-dir", str(tmp_path / "audit"),
                                 "--diff-depth-manifest", str(tmp_path / "public.json")])
    monkeypatch.setattr(recurring, "_utc_now", lambda: NOW)
    monkeypatch.setattr(controller, "acquire_cycle", lambda *a, **kw: (cycle, "chrome_plugin_manifest"))
    return args, cycle


def complete_report():
    return {"status": "complete", "required_evidence_validated": True,
            "verdicts": [{"symbol": "BTCUSDT", "strategy_id": "123", "verdict": "CONTINUE"}]}


def test_duplicate_capture_cannot_advance_scanner_twice(admitted_cycle, monkeypatch):
    args, _ = admitted_cycle
    calls = []
    monkeypatch.setattr(controller, "run_iteration", lambda cfg: calls.append(cfg) or complete_report())
    first = recurring.consume(args)
    again = recurring.consume(args)
    assert first["status"] == "complete"
    assert again["status"] == "duplicate"
    assert again["verdicts"] == []
    assert len(calls) == 1
    assert calls[0].observational_only is True
    assert calls[0].allow_actions is False
    assert first["training_eligible"] is False
    assert first["data_class"] == "live_bot_telemetry"
    assert calls[0].require_l2_evidence is True


def test_same_observation_in_renamed_manifest_has_same_key(admitted_cycle):
    _, cycle = admitted_cycle
    assert recurring.capture_key(cycle) == recurring.capture_key(replace(cycle, manifest_path=Path("alias.json")))


def test_new_timestamp_is_new_observation_even_if_pnl_unchanged(admitted_cycle):
    _, cycle = admitted_cycle
    newer = replace(cycle, bots=(replace(cycle.bots[0], captured_at_utc=NOW),))
    assert recurring.capture_key(cycle) != recurring.capture_key(newer)


def test_changed_roster_cannot_reuse_an_already_consumed_bot_observation(admitted_cycle, monkeypatch):
    args, cycle = admitted_cycle
    calls = []
    monkeypatch.setattr(controller, "run_iteration", lambda cfg: calls.append(cfg) or complete_report())
    assert recurring.consume(args)["status"] == "complete"
    # A different complete-roster bundle must not count this BTC observation
    # again just because another newly deployed symbol has been added.
    other = replace(cycle.bots[0], symbol="ETHUSDT", strategy_id="124")
    expanded = replace(cycle, bots=(*cycle.bots, other))
    monkeypatch.setattr(controller, "acquire_cycle", lambda *a, **kw: (expanded, "chrome_plugin_manifest"))
    with pytest.raises(controller.ControllerError, match="observation already claimed"):
        recurring.consume(args)
    assert len(calls) == 1


@pytest.mark.parametrize("failure", [OSError("disk unavailable"), ValueError("invalid result"), RuntimeError("unexpected")])
def test_failure_is_blocked_and_same_capture_is_not_retried(admitted_cycle, monkeypatch, failure):
    args, _ = admitted_cycle
    calls = []
    def fail(cfg):
        calls.append(cfg)
        raise failure
    monkeypatch.setattr(controller, "run_iteration", fail)
    result = recurring.consume(args)
    assert result["status"] == "blocked"
    assert result["failure_class"] == type(failure).__name__
    assert result["verdicts"] == []
    assert recurring.consume(args)["status"] == "blocked"
    assert len(calls) == 1


def test_process_interruption_keeps_claim_and_prevents_replay(admitted_cycle, monkeypatch):
    args, _ = admitted_cycle
    def interrupted(cfg):
        raise KeyboardInterrupt
    monkeypatch.setattr(controller, "run_iteration", interrupted)
    with pytest.raises(KeyboardInterrupt):
        recurring.consume(args)
    receipts = list((args.audit_dir / "receipts").glob("*.json"))
    assert json.loads(receipts[0].read_text())["status"] == "started"
    assert recurring.consume(args)["status"] == "blocked"


def test_scan_that_expires_capture_does_not_publish_current_verdict(admitted_cycle, monkeypatch):
    args, _ = admitted_cycle
    def slow_scan(cfg):
        monkeypatch.setattr(recurring, "_utc_now", lambda: NOW + timedelta(seconds=901))
        return complete_report()
    monkeypatch.setattr(controller, "run_iteration", slow_scan)
    result = recurring.consume(args)
    assert result["status"] == "blocked"
    assert result["verdicts"] == []


@pytest.mark.parametrize("target", ["raw", "manifest"])
def test_mid_scan_mutation_is_not_published(admitted_cycle, monkeypatch, target):
    args, cycle = admitted_cycle
    def mutate(cfg):
        path = cycle.bots[0].raw_text_path if target == "raw" else cycle.manifest_path
        path.write_text("modified", encoding="utf-8")
        return complete_report()
    monkeypatch.setattr(controller, "run_iteration", mutate)
    assert recurring.consume(args)["status"] == "blocked"


def test_controller_block_is_preserved(admitted_cycle, monkeypatch):
    args, _ = admitted_cycle
    monkeypatch.setattr(controller, "run_iteration", lambda cfg: {"status": "blocked", "error": "missing L2"})
    result = recurring.consume(args)
    assert result["status"] == "blocked"
    assert result["error"] == "missing L2"


def test_health_expires_saved_verdict_without_another_run(admitted_cycle, monkeypatch):
    args, _ = admitted_cycle
    monkeypatch.setattr(controller, "run_iteration", lambda cfg: complete_report())
    recurring.consume(args)
    assert recurring.health(args.audit_dir, now=NOW)["status"] == "current"
    old = recurring.health(args.audit_dir, now=NOW + timedelta(seconds=901))
    assert old["status"] == "stale"
    assert old["verdicts"] == []


def test_corrupt_receipt_blocks_instead_of_replaying(admitted_cycle, monkeypatch):
    args, _ = admitted_cycle
    monkeypatch.setattr(controller, "run_iteration", lambda cfg: complete_report())
    recurring.consume(args)
    next((args.audit_dir / "receipts").glob("*.json")).write_text("broken", encoding="utf-8")
    with pytest.raises(controller.ControllerError, match="cannot read"):
        recurring.consume(args)


def test_missing_health_has_no_verdict(tmp_path):
    assert recurring.health(tmp_path)["verdicts"] == []


def test_command_line_has_no_action_or_training_switch():
    with pytest.raises(SystemExit):
        recurring.parse_args(["--status", "--allow-actions"])


def test_consumer_lock_rejects_overlap_then_recovers(tmp_path):
    with recurring.exclusive_run_guard(tmp_path):
        with pytest.raises(controller.ControllerError, match="already running"):
            with recurring.exclusive_run_guard(tmp_path):
                pytest.fail("overlapping owner admitted")
    with recurring.exclusive_run_guard(tmp_path):
        assert (tmp_path / "consumer.guard").exists()


def test_capture_failure_immediately_invalidates_prior_health(admitted_cycle, monkeypatch, capsys):
    args, _ = admitted_cycle
    monkeypatch.setattr(controller, "run_iteration", lambda cfg: complete_report())
    recurring.consume(args)
    assert recurring.health(args.audit_dir, now=NOW)["status"] == "current"
    assert recurring.main(["--capture-failure", "browser disconnected", "--audit-dir", str(args.audit_dir)]) == 2
    assert recurring.health(args.audit_dir, now=NOW) == {
        "status": "blocked", "error": "browser disconnected", "verdicts": []}
    assert json.loads(capsys.readouterr().out)["failure_class"] == "CaptureFailure"


def test_claim_write_failure_does_not_start_scanner(admitted_cycle, monkeypatch, capsys):
    args, _ = admitted_cycle
    calls = []
    monkeypatch.setattr(controller, "run_iteration", lambda cfg: calls.append(cfg))
    def no_disk(*a, **kw):
        raise OSError("storage unavailable")
    monkeypatch.setattr(controller, "_atomic_write_json", no_disk)
    assert recurring.main(["--cycle-manifest", str(args.cycle_manifest),
                           "--audit-dir", str(args.audit_dir)]) == 2
    assert calls == []
    assert json.loads(capsys.readouterr().out)["verdicts"] == []


def test_wrapper_matches_controller_default_age(admitted_cycle):
    args, _ = admitted_cycle
    defaults = controller.parse_args([])
    assert recurring.controller_args(args).max_telemetry_age_seconds == defaults.max_telemetry_age_seconds
    assert defaults.interval_seconds == 600


def test_missing_l2_manifest_blocks_before_scanner(admitted_cycle, monkeypatch):
    args, _ = admitted_cycle
    args.diff_depth_manifest = []
    calls = []
    monkeypatch.setattr(controller, "run_iteration", lambda cfg: calls.append(cfg))
    with pytest.raises(controller.ControllerError, match="P0 public L2"):
        recurring.consume(args)
    assert calls == []


def test_complete_without_required_evidence_is_not_published(admitted_cycle, monkeypatch):
    args, _ = admitted_cycle
    report = complete_report()
    report.pop("required_evidence_validated")
    monkeypatch.setattr(controller, "run_iteration", lambda cfg: report)
    assert recurring.consume(args)["status"] == "blocked"
    assert recurring.health(args.audit_dir, now=NOW)["verdicts"] == []
    assert recurring.consume(args)["status"] == "blocked"


def test_pre_repair_receipt_cannot_appear_current(admitted_cycle, monkeypatch):
    args, _ = admitted_cycle
    monkeypatch.setattr(controller, "run_iteration", lambda cfg: complete_report())
    recurring.consume(args)
    path = args.audit_dir / "manifest.json"
    payload = json.loads(path.read_text())
    payload.pop("required_evidence_validated")
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert recurring.health(args.audit_dir, now=NOW)["status"] == "blocked"
    assert recurring.health(args.audit_dir, now=NOW)["verdicts"] == []
