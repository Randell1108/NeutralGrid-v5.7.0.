from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from neutralgrid.data.diff_depth import MANIFEST_SCHEMA_VERSION
from neutralgrid.data.market_stream import MARKET_MANIFEST_SCHEMA_VERSION
from neutralgrid.data.private_user_stream import PRIVATE_SERVICE_MANIFEST_SCHEMA_VERSION
from scripts.collect_diff_depth import DEFAULT_WS_BASE as DEFAULT_PUBLIC_WS_BASE
from scripts.collect_market_streams import DEFAULT_MARKET_WS_BASE
from scripts.collect_private_user_stream import (
    DEFAULT_LIFECYCLE_WS_URL,
    DEFAULT_PRIVATE_WS_BASE,
)
from scripts.supervise_binance_websocket_services import SUPERVISOR_SCHEMA_VERSION
from scripts import verify_binance_websocket_services as verifier
from scripts.verify_binance_websocket_services import parse_args, verify


@pytest.mark.skipif(os.name != "nt", reason="Windows PID-probe regression")
def test_windows_pid_probe_does_not_call_os_kill(monkeypatch) -> None:
    def fail_if_called(_pid: int, _signal: int) -> None:
        raise AssertionError("os.kill(pid, 0) is mutating on Windows")

    monkeypatch.setattr(verifier.os, "kill", fail_if_called)

    assert verifier._is_running(os.getpid()) is True


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _service_record(name: str, audit_root: Path) -> dict[str, object]:
    return {
        "name": name,
        "audit_dir": str(audit_root / name),
        "manifest_path": str(audit_root / name / "manifest.json"),
        "pid": None,
        "running": False,
        "starts": 1,
        "unexpected_exits": 0,
        "last_exit_code": 0,
        "blocked_reason": None,
    }


def _complete_manifests(tmp_path: Path) -> tuple[Path, Path, Path]:
    audit_root = tmp_path / "audit"
    live_root = tmp_path / "Live"
    roster = tmp_path / "targets.csv"
    roster.write_text("symbol,strategy_id\nBTCUSDT,413500001\n", encoding="utf-8")
    target = {"symbol": "BTCUSDT", "strategy_id": "413500001"}
    public_target = {"symbol": "BTCUSDT", "strategy_id": None}
    ingestion_date = "2026-09-10"
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    public_dir = live_root / ingestion_date / "BTCUSDT" / "diff_depth" / "run"
    market_dir = live_root / ingestion_date / "BTCUSDT" / "market_stream" / "run"
    private_dir = (
        live_root
        / ingestion_date
        / "BTCUSDT"
        / "private_user_stream"
        / "413500001"
        / "run"
    )
    _write_json(
        audit_root / "manifest.json",
        {
            "schema_version": SUPERVISOR_SCHEMA_VERSION,
            "status": "complete",
            "updated_at_utc": now,
            "supervisor_pid": 1,
            "target_csv": str(roster.resolve()),
            "target_sha256": hashlib.sha256(roster.read_bytes()).hexdigest(),
            "targets": [target],
            "unique_symbols": ["BTCUSDT"],
            "ingestion_date_lima": ingestion_date,
            "ingestion_timezone": "America/Lima",
            "live_root": str(live_root.resolve()),
            "runtime_effect": "observational_only",
            "services": {
                name: _service_record(name, audit_root)
                for name in ("public", "market", "private")
            },
        },
    )
    common = {
        "status": "complete",
        "updated_at_utc": now,
        "live_date_lima": ingestion_date,
        "ingestion_timezone": "America/Lima",
    }
    _write_json(
        audit_root / "public" / "manifest.json",
        {
            **common,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "service": "public_diff_depth",
            "traffic_class": "public",
            "targets": [public_target],
            "ws_base": DEFAULT_PUBLIC_WS_BASE,
            "collect_agg_trades": False,
            "collect_mark_price_updates": False,
            "symbol_run_dirs": {"BTCUSDT": str(public_dir.resolve())},
            "symbol_counters": {
                "BTCUSDT": {
                    "public_agg_trades": 0,
                    "public_mark_price_updates": 0,
                }
            },
        },
    )
    _write_json(
        audit_root / "market" / "manifest.json",
        {
            **common,
            "schema_version": MARKET_MANIFEST_SCHEMA_VERSION,
            "service": "market",
            "traffic_class": "market",
            "targets": [public_target],
            "ws_base": DEFAULT_MARKET_WS_BASE,
            "symbol_run_dirs": {"BTCUSDT": str(market_dir.resolve())},
        },
    )
    _write_json(
        audit_root / "private" / "manifest.json",
        {
            **common,
            "schema_version": PRIVATE_SERVICE_MANIFEST_SCHEMA_VERSION,
            "service": "private",
            "traffic_class": "private",
            "targets": [target],
            "ws_route": DEFAULT_PRIVATE_WS_BASE,
            "lifecycle_ws_url": DEFAULT_LIFECYCLE_WS_URL,
            "event_completeness": "unknown",
            "collector": {
                "status": "running",
                "last_error": None,
                "service_counters": {"connections": 1},
            },
            "symbol_strategy_run_dirs": {
                "BTCUSDT:413500001": str(private_dir.resolve())
            },
        },
    )
    _write_json(
        private_dir / "manifest.json",
        {
            "status": "running",
            "last_error": None,
            "service_counters": {"connections": 1},
        },
    )
    return roster, audit_root, live_root


def test_live_verifier_accepts_exact_three_service_contract(tmp_path: Path) -> None:
    roster, audit_root, _live_root = _complete_manifests(tmp_path)
    report = tmp_path / "verification.json"
    args = parse_args(
        [
            "--target-csv",
            str(roster),
            "--audit-root",
            str(audit_root),
            "--wait-seconds",
            "0",
            "--report",
            str(report),
        ]
    )

    assert verify(args) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert {item["name"] for item in payload["checks"]} >= {
        "static_contract",
        "roster_identity",
        "shared_lima_date",
        "exclusive_routes",
        "storage_boundary",
        "public_market_separation",
    }


def test_live_verifier_fails_route_crossover(tmp_path: Path) -> None:
    roster, audit_root, _live_root = _complete_manifests(tmp_path)
    public_path = audit_root / "public" / "manifest.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    public["ws_base"] = DEFAULT_MARKET_WS_BASE
    _write_json(public_path, public)
    report = tmp_path / "verification.json"
    args = parse_args(
        [
            "--target-csv",
            str(roster),
            "--audit-root",
            str(audit_root),
            "--wait-seconds",
            "0",
            "--report",
            str(report),
        ]
    )

    assert verify(args) == 2
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert "route ownership mismatch" in payload["last_error"]


def test_live_verifier_fails_private_lifecycle_route_mismatch(
    tmp_path: Path,
) -> None:
    roster, audit_root, _live_root = _complete_manifests(tmp_path)
    private_path = audit_root / "private" / "manifest.json"
    private = json.loads(private_path.read_text(encoding="utf-8"))
    private["lifecycle_ws_url"] = "wss://example.invalid/ws-fapi/v1"
    _write_json(private_path, private)
    report = tmp_path / "verification.json"
    args = parse_args(
        [
            "--target-csv",
            str(roster),
            "--audit-root",
            str(audit_root),
            "--wait-seconds",
            "0",
            "--report",
            str(report),
        ]
    )

    assert verify(args) == 2
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert "private lifecycle route mismatch" in payload["last_error"]


def test_live_verifier_fails_when_private_never_connected(tmp_path: Path) -> None:
    roster, audit_root, _live_root = _complete_manifests(tmp_path)
    private_path = audit_root / "private" / "manifest.json"
    private = json.loads(private_path.read_text(encoding="utf-8"))
    private["status"] = "running"
    private["collector"] = {
        "status": "running",
        "last_error": (
            "ValueError('User-data stream lifecycle userDataStream.start failed "
            "with status=401, code=-2015')"
        ),
        "service_counters": {"connections": 0},
    }
    _write_json(private_path, private)
    private_run_dir = Path(
        next(iter(private["symbol_strategy_run_dirs"].values()))
    )
    _write_json(
        private_run_dir / "manifest.json",
        {
            "status": "running",
            "last_error": private["collector"]["last_error"],
            "service_counters": {"connections": 0},
        },
    )
    report = tmp_path / "verification.json"
    args = parse_args(
        [
            "--target-csv",
            str(roster),
            "--audit-root",
            str(audit_root),
            "--wait-seconds",
            "0",
            "--report",
            str(report),
        ]
    )

    assert verify(args) == 2
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert "has not established a user-data connection" in payload["last_error"]
    assert "status=401, code=-2015" in payload["last_error"]
