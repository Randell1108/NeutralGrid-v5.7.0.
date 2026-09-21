from __future__ import annotations

import os

import pytest

from scripts import collect_diff_depth as public
from scripts import collect_private_user_stream as private
from scripts.websocket_process_identity import ProcessObservation


@pytest.mark.parametrize("collector", [public, private])
def test_collector_lock_probe_never_signals_current_process(monkeypatch, collector):
    monkeypatch.setattr(os, "kill", lambda *_args: pytest.fail("PID query attempted os.kill"))
    assert collector._pid_is_running(os.getpid()) is True


@pytest.mark.parametrize("collector", [public, private])
def test_unknown_collector_owner_blocks_competing_start(monkeypatch, collector):
    monkeypatch.setattr(collector, "query_process", lambda pid: ProcessObservation(pid, "unknown"))
    assert collector._pid_is_running(123) is True
