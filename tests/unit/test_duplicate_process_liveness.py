from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from neutralgrid.live.decision import pnl_history
from scripts import collect_private_grid_telemetry as private_grid
from scripts import run_live_telemetry_controller as controller
from scripts import run_live_volatility_loop as volatility


@pytest.mark.parametrize("module,function", [(pnl_history, "_pid_is_running"), (private_grid, "_pid_is_running"), (volatility, "_pid_is_alive")])
@pytest.mark.parametrize("state,expected", [("running", True), ("unknown", True), ("exited", False)])
def test_duplicate_liveness_queries_are_read_only_and_keep_unknown_owners(
    monkeypatch: pytest.MonkeyPatch, module: Any, function: str, state: str, expected: bool
) -> None:
    query = Mock(return_value=SimpleNamespace(state=state))
    monkeypatch.setattr(module, "query_process", query, raising=False)
    monkeypatch.setattr(os, "kill", lambda *_: pytest.fail("a liveness probe must not signal its owner"))
    # Exercise the old Windows failed-query branch safely if a regression
    # reintroduces it; no native process APIs are called by this fake.
    kernel = SimpleNamespace(
        OpenProcess=Mock(return_value=123), GetExitCodeProcess=Mock(return_value=0), CloseHandle=Mock()
    )
    monkeypatch.setattr(ctypes, "WinDLL", Mock(return_value=kernel), raising=False)
    assert getattr(module, function)(123) is expected
    query.assert_called_once_with(123)


@pytest.mark.parametrize("state", ["running", "unknown"])
def test_controller_preserves_existing_owner_lock_without_signalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    path = tmp_path / "controller.lock"
    previous = b"123\nowner provenance\n"
    path.write_bytes(previous)
    query = Mock(return_value=SimpleNamespace(state=state))
    monkeypatch.setattr(controller, "query_process", query, raising=False)
    monkeypatch.setattr(os, "kill", lambda *_: pytest.fail("controller lock checks must not signal"))
    with pytest.raises(controller.ControllerError, match="already running"):
        controller._acquire_lock(path)
    query.assert_called_once_with(123)
    assert path.read_bytes() == previous


def test_controller_reclaims_only_confirmed_exited_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "controller.lock"
    path.write_text("123\nprevious owner\n", encoding="ascii")
    query = Mock(return_value=SimpleNamespace(state="exited"))
    monkeypatch.setattr(controller, "query_process", query, raising=False)
    monkeypatch.setattr(os, "kill", lambda *_: pytest.fail("controller lock checks must not signal"))
    descriptor = controller._acquire_lock(path)
    try:
        assert path.read_text(encoding="ascii").splitlines()[0] == str(os.getpid())
        query.assert_called_once_with(123)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("module,function", [(pnl_history, "_pid_is_running"), (private_grid, "_pid_is_running"), (volatility, "_pid_is_alive")])
@pytest.mark.parametrize("pid", [0, -1])
def test_nonpositive_probe_does_not_query_or_signal(
    monkeypatch: pytest.MonkeyPatch, module: Any, function: str, pid: int
) -> None:
    query = Mock(side_effect=AssertionError("nonpositive PID must not be queried"))
    monkeypatch.setattr(module, "query_process", query, raising=False)
    monkeypatch.setattr(os, "kill", lambda *_: pytest.fail("invalid PID must not be signalled"))
    assert getattr(module, function)(pid) is False
    query.assert_not_called()


def test_real_isolated_child_survives_all_duplicate_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disposable idle child exits via stdin; production services are untouched."""
    monkeypatch.setattr(os, "kill", lambda *_: pytest.fail("a liveness probe must never signal"))
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.readline()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        assert pnl_history._pid_is_running(process.pid)
        assert private_grid._pid_is_running(process.pid)
        assert volatility._pid_is_alive(process.pid)
        path = tmp_path / "child.lock"
        path.write_text(str(process.pid), encoding="ascii")
        with pytest.raises(controller.ControllerError, match="already running"):
            controller._acquire_lock(path)
        assert process.poll() is None
        assert process.stdin is not None
        process.stdin.write("finish\n")
        process.stdin.flush()
        assert process.wait(timeout=10) == 0
        deadline = time.monotonic() + 2.0
        while pnl_history._pid_is_running(process.pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not pnl_history._pid_is_running(process.pid)
        assert not private_grid._pid_is_running(process.pid)
        assert not volatility._pid_is_alive(process.pid)
    finally:
        if process.stdin is not None:
            process.stdin.close()
        if process.poll() is None:
            # Closing stdin releases this fixture without a termination signal.
            process.wait(timeout=10)
