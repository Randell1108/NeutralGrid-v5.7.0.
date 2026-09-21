from __future__ import annotations

import json
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import request_binance_websocket_stop as stopper
from neutralgrid.core.process_identity import ProcessObservation, matches_identity, query_process
from neutralgrid.core import process_identity as identity_module


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(stopper, "ROOT", tmp_path)
    root = tmp_path / "audit"
    observations = {}
    services = {}
    owner = ProcessObservation(500, "running", "owner_birth", sys.executable)
    observations[500] = ProcessObservation(500, "exited")
    for index, (name, (script, schema, service_name)) in enumerate(stopper.SERVICES.items()):
        worker = ProcessObservation(600 + index, "running", f"worker_birth_{index}", sys.executable)
        launcher = ProcessObservation(700 + index, "running", f"launcher_birth_{index}", sys.executable)
        observations[worker.pid] = worker
        observations[launcher.pid] = launcher
        service_dir = root / name
        services[name] = {
            "audit_dir": str(service_dir), "manifest_path": str(service_dir / "manifest.json"),
            "command": [sys.executable, str(tmp_path / "scripts" / script), "--audit-dir", str(service_dir)],
            "running": True, "pid": launcher.pid, "process_identity": launcher.identity(),
        }
        _write(service_dir / "manifest.json", {
            "schema_version": schema, "service": service_name,
            "collector_pid": worker.pid, "process_identity": worker.identity(), "status": "running",
            "audit_dir": str(service_dir), "run_id": f"fixture_{name}",
        })
    _write(root / "manifest.json", {
        "schema_version": "neutralgrid_binance_ws_supervisor_v1", "audit_root": str(root),
        "supervisor_pid": owner.pid, "process_identity": owner.identity(), "services": services,
    })
    return root, observations


def test_missing_supervisor_stops_three_exact_orphans_and_writes_receipt(tmp_path, monkeypatch):
    root, states = _fixture(tmp_path, monkeypatch)

    def observe(pid):
        if all((root / name / "STOP").exists() for name in stopper.SERVICES):
            return ProcessObservation(pid, "exited")
        return states[pid]

    report = stopper.request_stop(root, timeout=0.01, observe=observe)
    assert report["status"] == "stopped"
    assert len(report["markers_written"]) == 3
    assert not (root / "STOP").exists()
    assert json.loads((root / "stop_receipt.json").read_text()) == report
    assert all(item["final_state"] == "exited" for item in report["processes"].values())


def test_verified_living_supervisor_receives_normal_stop_request(tmp_path, monkeypatch):
    root, states = _fixture(tmp_path, monkeypatch)
    states[500] = ProcessObservation(500, "running", "owner_birth", sys.executable)

    def observe(pid):
        if all((root / name / "STOP").exists() for name in stopper.SERVICES):
            return ProcessObservation(pid, "exited")
        return states[pid]

    report = stopper.request_stop(root, timeout=0, observe=observe)
    assert report["status"] == "stopped"
    assert (root / "STOP").exists()
    assert len(report["markers_written"]) == 4


@pytest.mark.parametrize("bad_state", ["reused", "unknown", "unrelated_executable"])
def test_ambiguous_or_reused_child_prevents_all_marker_writes(tmp_path, monkeypatch, bad_state):
    root, states = _fixture(tmp_path, monkeypatch)
    original = states[600]
    states[600] = (
        ProcessObservation(600, "unknown") if bad_state == "unknown"
        else ProcessObservation(600, "running", "new_birth", original.executable) if bad_state == "reused"
        else ProcessObservation(600, "running", original.creation_token, "unrelated.exe")
    )
    report = stopper.request_stop(root, timeout=0, observe=states.__getitem__)
    assert report["status"] == "blocked"
    assert "unknown or PID was reused" in report["error"]
    assert report["markers_written"] == []
    assert not list(root.rglob("STOP"))


@pytest.mark.parametrize("corruption", ["outside_path", "other_script", "legacy_identity", "wrong_schema", "launcher_pid"])
def test_authority_mismatch_is_rejected_before_writing(tmp_path, monkeypatch, corruption):
    root, states = _fixture(tmp_path, monkeypatch)
    path = root / "manifest.json"
    payload = json.loads(path.read_text())
    if corruption == "outside_path":
        payload["services"]["public"]["audit_dir"] = str(tmp_path / "another_run")
    elif corruption == "other_script":
        payload["services"]["public"]["command"][1] = str(tmp_path / "scripts" / "other.py")
    elif corruption == "legacy_identity":
        payload.pop("process_identity")
    elif corruption == "launcher_pid":
        payload["services"]["public"]["pid"] = 999
    else:
        payload["schema_version"] = "other_schema"
    _write(path, payload)
    before = path.read_bytes()
    report = stopper.request_stop(root, timeout=0, observe=states.__getitem__)
    assert report["status"] == "blocked"
    assert not list(root.rglob("STOP"))
    assert path.read_bytes() == before


@pytest.mark.parametrize("field,value", [("audit_dir", "elsewhere"), ("run_id", ""), ("collector_pid", 999)])
def test_child_manifest_binding_is_required(tmp_path, monkeypatch, field, value):
    root, states = _fixture(tmp_path, monkeypatch)
    path = root / "public" / "manifest.json"
    payload = json.loads(path.read_text())
    payload[field] = value
    _write(path, payload)
    report = stopper.request_stop(root, timeout=0, observe=states.__getitem__)
    assert report["status"] == "blocked"
    assert not list(root.rglob("STOP"))


def test_preexisting_live_legacy_run_is_explicitly_blocked_without_markers(tmp_path, monkeypatch):
    root, states = _fixture(tmp_path, monkeypatch)
    path = root / "manifest.json"
    payload = json.loads(path.read_text())
    payload.pop("process_identity")
    payload["status"] = "running"
    _write(path, payload)
    states[500] = ProcessObservation(500, "running", "legacy_birth", sys.executable)
    report = stopper.request_stop(root, timeout=0, observe=states.__getitem__)
    assert report["status"] == "blocked"
    assert "legacy PID alone is insufficient" in report["error"]
    assert not list(root.rglob("STOP"))


def test_manifest_generation_change_prevents_stop(tmp_path, monkeypatch):
    root, states = _fixture(tmp_path, monkeypatch)

    def observe(pid):
        if pid == 602:
            path = root / "manifest.json"
            payload = json.loads(path.read_text())
            payload["new_run"] = True
            _write(path, payload)
        return states[pid]

    report = stopper.request_stop(root, timeout=0, observe=observe)
    assert report["status"] == "blocked"
    assert "manifest changed" in report["error"]
    assert not list(root.rglob("STOP"))


def test_timeout_reports_living_processes_without_termination(tmp_path, monkeypatch):
    root, states = _fixture(tmp_path, monkeypatch)
    report = stopper.request_stop(root, timeout=0, observe=states.__getitem__)
    assert report["status"] == "incomplete"
    assert set(report["pending"]) == {"public", "market", "private", "public_launcher", "market_launcher", "private_launcher"}
    assert all(states[pid].state == "running" for pid in states if pid != 500)


def test_unknown_owner_after_stop_is_not_success(tmp_path, monkeypatch):
    root, states = _fixture(tmp_path, monkeypatch)

    def observe(pid):
        if all((root / name / "STOP").exists() for name in stopper.SERVICES):
            return ProcessObservation(pid, "unknown")
        return states[pid]

    report = stopper.request_stop(root, timeout=0, observe=observe)
    assert report["status"] == "incomplete"
    assert all(item["final_state"] == "identity_changed_or_unknown" for item in report["processes"].values())


def test_query_current_process_is_read_only_and_birth_stable(monkeypatch):
    monkeypatch.setattr(os, "kill", lambda *_args: pytest.fail("identity query must not signal"))
    observed = query_process(os.getpid())
    assert observed.state == "running"
    assert matches_identity(query_process(os.getpid()), observed.identity())
    assert not matches_identity(ProcessObservation(observed.pid, "running", "reused", observed.executable), observed.identity())


@pytest.mark.parametrize("handle,error,success,exit_code,expected", [
    (0, 87, 0, 0, "exited"), (0, 5, 0, 0, "unknown"),
    (0, 8, 0, 0, "unknown"), (123, 0, 0, 0, "unknown"),
    (123, 0, 1, 0, "exited"),
])
def test_native_query_fails_closed_and_always_closes_handles(monkeypatch, handle, error, success, exit_code, expected):
    kernel = SimpleNamespace(
        OpenProcess=Mock(return_value=handle), CloseHandle=Mock(),
        GetProcessTimes=Mock(), QueryFullProcessImageNameW=Mock(),
    )

    def get_exit(_handle, output):
        output._obj.value = exit_code
        return success

    kernel.GetExitCodeProcess = Mock(side_effect=get_exit)
    monkeypatch.setattr(ctypes, "WinDLL", Mock(return_value=kernel), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: error, raising=False)
    assert identity_module._query_windows(123).state == expected
    if handle:
        kernel.CloseHandle.assert_called_once_with(handle)
    else:
        kernel.CloseHandle.assert_not_called()


@pytest.mark.parametrize("stage", ["times", "executable"])
def test_partial_native_identity_is_unknown(monkeypatch, stage):
    def get_exit(_handle, output):
        output._obj.value = 259
        return 1

    kernel = SimpleNamespace(
        OpenProcess=Mock(return_value=123), CloseHandle=Mock(),
        GetExitCodeProcess=Mock(side_effect=get_exit),
        GetProcessTimes=Mock(return_value=0 if stage == "times" else 1),
        QueryFullProcessImageNameW=Mock(return_value=0),
    )
    monkeypatch.setattr(ctypes, "WinDLL", Mock(return_value=kernel), raising=False)
    assert identity_module._query_windows(123).state == "unknown"
    kernel.CloseHandle.assert_called_once_with(123)


def test_real_isolated_orphan_workers_exit_on_exact_markers(tmp_path, monkeypatch):
    """Three harmless marker watchers; no network access or production scripts run."""
    real_root = stopper.ROOT
    root, _ = _fixture(tmp_path, monkeypatch)
    supervisor = json.loads((root / "manifest.json").read_text())
    supervisor["supervisor_pid"] = 2147483647
    supervisor["process_identity"]["pid"] = 2147483647
    processes = []
    try:
        for name, (script, schema, service_name) in stopper.SERVICES.items():
            path = tmp_path / "scripts" / script
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "import json, os, sys, time\nfrom pathlib import Path\n"
                f"sys.path.insert(0, {str(real_root)!r})\n"
                "from scripts.websocket_process_identity import query_process\n"
                "root = Path(sys.argv[2])\n"
                f"payload = {{'schema_version': {schema!r}, 'service': {service_name!r}, 'collector_pid': os.getpid(), 'process_identity': query_process(os.getpid()).identity(), 'status': 'running', 'audit_dir': str(root), 'run_id': 'isolated_{name}'}}\n"
                "(root / 'manifest.json').write_text(json.dumps(payload))\n"
                "(root / 'READY').write_text('ready')\n"
                "deadline = time.monotonic() + 20\n"
                "while not (root / 'STOP').exists() and time.monotonic() < deadline: time.sleep(.02)\n"
                "payload['status'] = 'complete'\n"
                "(root / 'manifest.json').write_text(json.dumps(payload))\n",
                encoding="utf-8",
            )
            command = supervisor["services"][name]["command"]
            process = subprocess.Popen(command, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            processes.append(process)
            supervisor["services"][name]["pid"] = process.pid
            supervisor["services"][name]["process_identity"] = query_process(process.pid).identity()
        deadline = time.monotonic() + 10
        while not all((root / name / "READY").exists() for name in stopper.SERVICES):
            assert time.monotonic() < deadline, "fixture workers failed to start"
            time.sleep(0.02)
        _write(root / "manifest.json", supervisor)
        report = stopper.request_stop(root, timeout=5, poll_seconds=0.02)
        assert report["status"] == "stopped", report
        assert all(process.wait(timeout=5) == 0 for process in processes)
        assert len(report["markers_written"]) == 3
        assert all(json.loads((root / name / "manifest.json").read_text())["status"] == "complete" for name in stopper.SERVICES)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
