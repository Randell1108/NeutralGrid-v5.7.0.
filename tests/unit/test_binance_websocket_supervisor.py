from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import supervise_binance_websocket_services as supervisor


def test_windows_launcher_uses_validated_full_roster_fsync_batch() -> None:
    launcher = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "start_binance_websocket_services.ps1"
    ).read_text(encoding="utf-8")

    assert "[int]$FsyncEvery = 100" in launcher


@pytest.mark.skipif(os.name != "nt", reason="Windows PID-probe regression")
def test_windows_lock_probe_does_not_call_os_kill(monkeypatch) -> None:
    def fail_if_called(_pid: int, _signal: int) -> None:
        raise AssertionError("os.kill(pid, 0) is mutating on Windows")

    monkeypatch.setattr(supervisor.os, "kill", fail_if_called)

    assert supervisor._pid_is_running(os.getpid()) is True


def _roster(path: Path) -> Path:
    path.write_text(
        "symbol,strategy_id\nBTCUSDT,413500001\nETHUSDT,413500002\n",
        encoding="utf-8",
    )
    return path


def test_service_commands_have_exclusive_route_ownership(tmp_path: Path) -> None:
    roster = _roster(tmp_path / "targets.csv")
    args = supervisor.parse_args(
        [
            "--target-csv",
            str(roster),
            "--audit-root",
            str(tmp_path / "audit"),
            "--live-root",
            str(tmp_path / "Live"),
        ]
    )
    targets = supervisor.load_private_targets(roster)

    commands = supervisor.build_service_commands(args, targets)

    assert set(commands) == {"public", "market", "private"}
    public = commands["public"]
    market = commands["market"]
    private = commands["private"]
    assert "collect_diff_depth.py" in public[1]
    assert "--no-agg-trades" in public
    assert "--no-mark-price-updates" in public
    assert args.public_ws_base in public
    assert "collect_market_streams.py" in market[1]
    assert args.market_ws_base in market
    assert "--no-agg-trades" not in market
    assert "collect_private_user_stream.py" in private[1]
    assert args.private_ws_base in private
    assert args.private_lifecycle_ws_url in private
    assert "--lifecycle-ws-url" in private
    assert str(roster.resolve()) in private
    assert "BINANCE_API_KEY" not in " ".join(private)
    assert public[public.index("--symbols") + 1 : public.index("--duration-seconds")] == [
        "BTCUSDT",
        "ETHUSDT",
    ]
    for command in commands.values():
        assert command[command.index("--ingestion-date") + 1] == args.ingestion_date


def test_supervisor_restarts_failed_market_and_quarantines_missing_private_key(
    tmp_path: Path, monkeypatch
) -> None:
    roster = _roster(tmp_path / "targets.csv")
    audit_root = tmp_path / "audit"
    args = supervisor.parse_args(
        [
            "--target-csv",
            str(roster),
            "--audit-root",
            str(audit_root),
            "--live-root",
            str(tmp_path / "Live"),
            "--log-root",
            str(tmp_path / "logs"),
            "--duration-seconds",
            "0.4",
            "--manifest-heartbeat-seconds",
            "0.05",
            "--restart-base-seconds",
            "0.05",
            "--restart-max-seconds",
            "0.05",
            "--shutdown-timeout-seconds",
            "0.05",
        ]
    )

    class FakeProcess:
        next_pid = 9000

        def __init__(self, command, **_kwargs) -> None:
            self.command = command
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self.return_code = {
                "public": None,
                "market": 2,
                "private": 3,
            }[command[0]]

        def poll(self):
            return self.return_code

        def terminate(self) -> None:
            self.return_code = -15

        def kill(self) -> None:
            self.return_code = -9

        def wait(self, timeout=None):
            del timeout
            return self.return_code

    def fake_commands(_args, _targets):
        return {
            "public": ["public"],
            "market": ["market"],
            "private": ["private"],
        }

    monkeypatch.setattr(supervisor, "build_service_commands", fake_commands)
    monkeypatch.setattr(supervisor, "_git_output", lambda _args: None)
    monkeypatch.setattr(supervisor.subprocess, "Popen", FakeProcess)

    assert supervisor.supervise(args) == 0

    manifest = json.loads(
        (audit_root / "manifest.json").read_text(encoding="utf-8")
    )
    services = manifest["services"]
    assert manifest["status"] == "complete_private_blocked"
    assert manifest["stop_reason"] == "duration_elapsed"
    assert services["private"]["starts"] == 1
    assert services["private"]["blocked_reason"] == (
        "missing_api_key_restart_supervisor_after_configuration"
    )
    assert services["market"]["starts"] >= 2
    assert services["market"]["unexpected_exits"] >= 1
    assert services["public"]["last_exit_code"] is not None
    assert not (audit_root / "supervisor.lock").exists()


@pytest.mark.parametrize("failure_stage", ["start", "shutdown"])
def test_catchable_failures_record_terminal_reason_and_release_lock(tmp_path, monkeypatch, failure_stage):
    roster = _roster(tmp_path / "targets.csv")
    audit_root = tmp_path / "audit"
    args = supervisor.parse_args([
        "--target-csv", str(roster), "--audit-root", str(audit_root),
        "--live-root", str(tmp_path / "Live"), "--log-root", str(tmp_path / "logs"),
        "--duration-seconds", "0.001",
    ])

    def start_service(_service, **_kwargs):
        if failure_stage == "start":
            raise OSError("fixture start failure")

    def stop_services(services, _timeout):
        if failure_stage == "shutdown":
            raise OSError("fixture shutdown failure")
        for service in services.values():
            service.last_exit_code = 17

    monkeypatch.setattr(supervisor, "_git_output", lambda _args: None)
    monkeypatch.setattr(supervisor, "_start_service", start_service)
    monkeypatch.setattr(supervisor, "_stop_services", stop_services)
    assert supervisor.supervise(args) == 2
    manifest = json.loads((audit_root / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["exit_code"] == 2
    assert manifest["stop_reason"] == ("supervisor_exception" if failure_stage == "start" else "shutdown_exception")
    assert f"fixture {failure_stage} failure" in manifest["error"]
    assert "completed_at_utc" in manifest
    if failure_stage == "start":
        assert all(item["last_exit_code"] == 17 for item in manifest["services"].values())
    assert not (audit_root / "supervisor.lock").exists()


def test_competing_start_cannot_remove_existing_stop_marker(tmp_path, monkeypatch):
    roster = _roster(tmp_path / "targets.csv")
    audit_root = tmp_path / "audit"
    audit_root.mkdir()
    (audit_root / "STOP").write_text("existing owner's pending stop")
    (audit_root / "supervisor.lock").write_text(str(os.getpid()))
    args = supervisor.parse_args(["--target-csv", str(roster), "--audit-root", str(audit_root)])
    monkeypatch.setattr(supervisor, "_git_output", lambda _args: None)
    with pytest.raises(supervisor.WebSocketSupervisorError, match="already running"):
        supervisor.supervise(args)
    assert (audit_root / "STOP").read_text() == "existing owner's pending stop"


@pytest.mark.parametrize("option", ["--duration-seconds", "--shutdown-timeout-seconds", "--manifest-heartbeat-seconds"])
@pytest.mark.parametrize("value", ["nan", "inf"])
def test_supervisor_rejects_nonfinite_lifetime_bounds(tmp_path, option, value):
    with pytest.raises(SystemExit):
        supervisor.parse_args(["--target-csv", str(tmp_path / "targets.csv"), option, value])


@pytest.mark.parametrize("worker_stops", [True, False])
def test_supervisor_shutdown_handles_worker_after_launcher_exit(tmp_path, monkeypatch, worker_stops):
    from neutralgrid.core.process_identity import ProcessObservation

    service = supervisor.ServiceProcess(
        "public", ["unused"], tmp_path / "public", tmp_path / "out", tmp_path / "err"
    )
    service.audit_dir.mkdir()
    identity = ProcessObservation(123, "running", "worker_birth", "python.exe")
    (service.audit_dir / "manifest.json").write_text(json.dumps({
        "collector_pid": 123, "process_identity": identity.identity(), "audit_dir": str(service.audit_dir),
    }))

    class ExitedLauncher:
        def poll(self):
            return 0

    service.process = ExitedLauncher()
    monkeypatch.setattr(supervisor, "query_process", lambda pid: (
        ProcessObservation(pid, "exited") if worker_stops and (service.audit_dir / "STOP").exists() else identity
    ))
    if worker_stops:
        supervisor._stop_services({"public": service}, timeout=0)
    else:
        with pytest.raises(supervisor.WebSocketSupervisorError, match="worker still running"):
            supervisor._stop_services({"public": service}, timeout=0)
    assert (service.audit_dir / "STOP").is_file()


def test_failed_stop_marker_does_not_skip_other_owned_services(tmp_path, monkeypatch):
    services = {
        name: supervisor.ServiceProcess(name, [], tmp_path / name, tmp_path / "out", tmp_path / "err")
        for name in ("public", "market", "private")
    }
    requested = []

    def request_stop(service):
        requested.append(service.name)
        if service.name == "public":
            raise PermissionError("fixture write failure")

    monkeypatch.setattr(supervisor, "_request_child_stop", request_stop)
    with pytest.raises(supervisor.WebSocketSupervisorError, match="fixture write failure"):
        supervisor._stop_services(services, timeout=0)
    assert requested == ["public", "market", "private"]
