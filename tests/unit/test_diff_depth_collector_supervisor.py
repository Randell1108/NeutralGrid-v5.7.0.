from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import supervise_diff_depth_collector as supervisor


UTC = timezone.utc


def _write_manifests(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "Live" / "2026-08-13" / "BTCUSDT" / "diff_depth" / "run"
    run_dir.mkdir(parents=True)
    symbol_manifest = run_dir / "manifest.json"
    symbol_manifest.write_text(
        json.dumps(
            {
                "status": "running",
                "target": {"symbol": "BTCUSDT", "strategy_id": "413500001"},
                "updated_at_utc": "2026-08-13T15:00:58+00:00",
                "last_error": None,
                "market_connection_error": None,
                "collect_agg_trades": True,
                "collect_mark_price_updates": True,
                "trade_subscription_acknowledged": True,
                "mark_price_subscription_acknowledged": True,
                "counters": {
                    "wire_events": 20,
                    "snapshots": 1,
                    "events_applied": 10,
                    "sequence_gaps": 0,
                    "parse_errors": 0,
                    "public_agg_trades": 3,
                    "public_mark_price_updates": 2,
                    "market_connections": 1,
                    "market_coverage_gaps": 0,
                    "market_parse_errors": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    run_manifest = tmp_path / "audit" / "manifest.json"
    run_manifest.parent.mkdir(parents=True)
    run_manifest.write_text(
        json.dumps(
            {
                "schema_version": "neutralgrid_diff_depth_run_v1",
                "status": "running",
                "collector_pid": 1234,
                "updated_at_utc": "2026-08-13T15:00:59+00:00",
                "targets": [
                    {"symbol": "BTCUSDT", "strategy_id": "413500001"}
                ],
                "symbol_run_dirs": {"BTCUSDT": str(run_dir)},
                "symbol_counters": {
                    "BTCUSDT": {
                        "wire_events": 20,
                        "snapshots": 1,
                        "events_applied": 10,
                        "sequence_gaps": 0,
                        "parse_errors": 0,
                        "public_agg_trades": 3,
                        "public_mark_price_updates": 2,
                        "market_connections": 1,
                        "market_coverage_gaps": 0,
                        "market_parse_errors": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return run_manifest, symbol_manifest


def test_validate_collector_health_accepts_exact_fresh_zero_gap_roster(
    tmp_path: Path,
) -> None:
    run_manifest, _symbol_manifest = _write_manifests(tmp_path)

    report = supervisor.validate_collector_health(
        run_manifest,
        expected_identities=[("BTCUSDT", "413500001")],
        expected_pids=[1234],
        now=datetime(2026, 8, 13, 15, 1, tzinfo=UTC),
        max_age_seconds=10.0,
    )

    assert report["healthy"] is True
    assert report["totals"] == {
        "wire_events": 20,
        "snapshots": 1,
        "events_applied": 10,
        "sequence_gaps": 0,
        "parse_errors": 0,
        "public_agg_trades": 3,
        "public_mark_price_updates": 2,
        "market_connections": 1,
        "market_coverage_gaps": 0,
        "market_parse_errors": 0,
    }


def test_validate_collector_health_rejects_gap_or_identity_mismatch(
    tmp_path: Path,
) -> None:
    run_manifest, symbol_manifest = _write_manifests(tmp_path)
    payload = json.loads(symbol_manifest.read_text(encoding="utf-8"))
    payload["counters"]["sequence_gaps"] = 1
    symbol_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(supervisor.CollectorSupervisorError, match="sequence gaps"):
        supervisor.validate_collector_health(
            run_manifest,
            expected_identities=[("BTCUSDT", "413500001")],
            expected_pids=[1234],
            now=datetime(2026, 8, 13, 15, 1, tzinfo=UTC),
            max_age_seconds=10.0,
        )

    with pytest.raises(supervisor.CollectorSupervisorError, match="target mismatch"):
        supervisor.validate_collector_health(
            run_manifest,
            expected_identities=[("BTCUSDT", "wrong")],
            expected_pids=[1234],
            now=datetime(2026, 8, 13, 15, 1, tzinfo=UTC),
            max_age_seconds=10.0,
        )


def test_validate_collector_health_requires_observed_market_stream(
    tmp_path: Path,
) -> None:
    run_manifest, symbol_manifest = _write_manifests(tmp_path)
    payload = json.loads(symbol_manifest.read_text(encoding="utf-8"))
    payload["counters"]["public_mark_price_updates"] = 0
    symbol_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(supervisor.CollectorSupervisorError, match="no mark-price"):
        supervisor.validate_collector_health(
            run_manifest,
            expected_identities=[("BTCUSDT", "413500001")],
            expected_pids=[1234],
            now=datetime(2026, 8, 13, 15, 1, tzinfo=UTC),
            max_age_seconds=10.0,
        )


def test_collector_replacement_reasons_bind_roster_and_code_hash() -> None:
    owner = {
        "target_sha256": "a" * 64,
        "collector_script_sha256": "b" * 64,
    }

    assert supervisor.collector_replacement_reasons(
        owner,
        target_sha256="a" * 64,
        collector_script_sha256="b" * 64,
    ) == []
    assert supervisor.collector_replacement_reasons(
        owner,
        target_sha256="c" * 64,
        collector_script_sha256="b" * 64,
    ) == ["fresh roster target mismatch"]
    assert supervisor.collector_replacement_reasons(
        owner,
        target_sha256="c" * 64,
        collector_script_sha256="d" * 64,
    ) == ["fresh roster target mismatch", "collector script hash mismatch"]

def test_classify_owner_requires_exact_recorded_pid_and_command(tmp_path: Path) -> None:
    command = [
        str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        str(tmp_path / "scripts" / "collect_diff_depth.py"),
        "--input",
        str(tmp_path / "targets.csv"),
    ]
    owner = {
        "schema_version": supervisor.OWNER_SCHEMA,
        "pid": 1234,
        "command": command,
        "workspace": str(tmp_path.resolve()),
        "target_sha256": "a" * 64,
    }
    process = {
        "pid": 1234,
        "executable_path": command[0],
        "command_line": supervisor.windows_command_line(command),
    }

    assert supervisor.owner_matches_process(
        owner,
        process,
        workspace_root=tmp_path,
    )
    assert not supervisor.owner_matches_process(
        dict(owner, pid=9999),
        process,
        workspace_root=tmp_path,
    )


def test_repo_collector_filter_ignores_powershell_query_process(tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "collect_diff_depth.py"
    processes = [
        {
            "pid": 10,
            "executable_path": "powershell.exe",
            "command_line": f"powershell.exe -Command match {script}",
        }
    ]

    assert supervisor._repo_collector_processes(
        processes,
        workspace_root=tmp_path,
    ) == []


def test_owner_chain_accepts_exact_windows_venv_child(tmp_path: Path) -> None:
    command = [
        str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        str(tmp_path / "scripts" / "collect_diff_depth.py"),
        "--input",
        str(tmp_path / "targets.csv"),
    ]
    command_line = supervisor.windows_command_line(command)
    owner = {
        "schema_version": supervisor.OWNER_SCHEMA,
        "pid": 100,
        "command": command,
        "workspace": str(tmp_path.resolve()),
    }
    processes = [
        {
            "pid": 100,
            "parent_process_id": 50,
            "executable_path": command[0],
            "command_line": command_line,
        },
        {
            "pid": 101,
            "parent_process_id": 100,
            "executable_path": command[0],
            "command_line": command_line,
        },
    ]

    assert [
        process["pid"]
        for process in supervisor.owned_collector_process_chain(
            owner,
            processes,
            workspace_root=tmp_path,
        )
    ] == [100, 101]

    unrelated = dict(processes[1], pid=102, command_line="python unrelated.py")
    chain = supervisor.owned_collector_process_chain(
        owner,
        [*processes, unrelated],
        workspace_root=tmp_path,
    )
    assert [process["pid"] for process in chain] == [100, 101]


def test_validate_collector_health_accepts_manifest_child_pid(tmp_path: Path) -> None:
    run_manifest, _symbol_manifest = _write_manifests(tmp_path)
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    payload["collector_pid"] = 5678
    run_manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = supervisor.validate_collector_health(
        run_manifest,
        expected_identities=[("BTCUSDT", "413500001")],
        expected_pids=[1234, 5678],
        now=datetime(2026, 8, 13, 15, 1, tzinfo=UTC),
        max_age_seconds=10.0,
    )

    assert report["collector_pid"] == 5678
