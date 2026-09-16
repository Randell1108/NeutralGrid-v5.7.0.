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
