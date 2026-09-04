from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.finalize_stale_telemetry_manifests import normalize_manifest, run_cleanup


NOW = datetime(2026, 8, 6, 8, 30, tzinfo=UTC)


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_private_stale_manifest_dry_run_then_apply(tmp_path: Path) -> None:
    path = tmp_path / "private" / "manifest.json"
    original = {
        "status": "waiting_for_valid_browser_state",
        "pid": 101,
        "last_error": "Chrome unavailable",
    }
    _write_manifest(path, original)

    dry_run = normalize_manifest(
        path,
        now=NOW,
        apply=False,
        pid_is_alive=lambda _pid: False,
    )

    assert dry_run["outcome"] == "would_normalize"
    assert json.loads(path.read_text(encoding="utf-8")) == original

    applied = normalize_manifest(
        path,
        now=NOW,
        apply=True,
        pid_is_alive=lambda _pid: False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert applied["outcome"] == "normalized"
    assert payload["status"] == "finished"
    assert payload["finished_at_utc"] == "2026-08-06T08:30:00+00:00"
    assert payload["status_before_cleanup"] == "waiting_for_valid_browser_state"
    assert payload["last_error"] == "Chrome unavailable"


def test_diff_depth_stale_manifest_normalizes_to_stopped(tmp_path: Path) -> None:
    path = tmp_path / "depth" / "manifest.json"
    _write_manifest(path, {"status": "running", "collector_pid": 202})

    result = normalize_manifest(
        path,
        now=NOW,
        apply=True,
        pid_is_alive=lambda _pid: False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert result["status"] == "stopped"
    assert payload["stopped_at_utc"] == "2026-08-06T08:30:00+00:00"


def test_cleanup_skips_live_owner_and_missing_pid(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "live" / "manifest.json", {"status": "running", "pid": 1})
    _write_manifest(
        tmp_path / "unknown" / "manifest.json",
        {"status": "waiting_for_valid_browser_state"},
    )

    report = run_cleanup(
        tmp_path,
        now=NOW,
        apply=True,
        pid_is_alive=lambda pid: pid == 1,
    )

    assert report["outcome_counts"] == {
        "skipped_missing_owner_pid": 1,
        "skipped_owner_alive": 1,
    }
    assert json.loads((tmp_path / "live" / "manifest.json").read_text())["status"] == "running"


def test_cleanup_normalizes_exact_aggregate_linked_symbol_manifest(
    tmp_path: Path,
) -> None:
    symbol_dir = tmp_path / "Live" / "2026-08-14" / "BTCUSDT" / "diff_depth" / "run-1"
    symbol_manifest = symbol_dir / "manifest.json"
    aggregate_manifest = tmp_path / "outputs" / "audits" / "collector" / "manifest.json"
    _write_manifest(
        symbol_manifest,
        {
            "schema_version": "binance_usdm_diff_depth_manifest_v1",
            "status": "running",
            "run_id": "run-1",
            "symbol": "BTCUSDT",
        },
    )
    _write_manifest(
        aggregate_manifest,
        {
            "schema_version": "binance_usdm_diff_depth_manifest_v1",
            "status": "running",
            "run_id": "run-1",
            "collector_pid": 202,
            "symbol_run_dirs": {"BTCUSDT": str(symbol_dir)},
        },
    )

    dry_run = run_cleanup(
        tmp_path,
        now=NOW,
        apply=False,
        pid_is_alive=lambda _pid: False,
    )

    assert dry_run["outcome_counts"] == {"would_normalize": 2}
    assert json.loads(symbol_manifest.read_text(encoding="utf-8"))["status"] == "running"

    applied = run_cleanup(
        tmp_path,
        now=NOW,
        apply=True,
        pid_is_alive=lambda _pid: False,
    )
    payload = json.loads(symbol_manifest.read_text(encoding="utf-8"))
    symbol_record = next(
        record for record in applied["records"] if record["path"] == str(symbol_manifest)
    )
    assert applied["outcome_counts"] == {"normalized": 2}
    assert symbol_record["owner_source_manifest"] == str(aggregate_manifest)
    assert payload["status"] == "stopped"
    assert payload["stopped_at_utc"] == "2026-08-06T08:30:00+00:00"


def test_cleanup_rejects_mismatched_aggregate_symbol_link(tmp_path: Path) -> None:
    symbol_dir = tmp_path / "Live" / "2026-08-14" / "BTCUSDT" / "diff_depth" / "run-1"
    symbol_manifest = symbol_dir / "manifest.json"
    _write_manifest(
        symbol_manifest,
        {
            "schema_version": "binance_usdm_diff_depth_manifest_v1",
            "status": "running",
            "run_id": "run-1",
            "symbol": "ETHUSDT",
        },
    )
    _write_manifest(
        tmp_path / "outputs" / "audits" / "collector" / "manifest.json",
        {
            "schema_version": "binance_usdm_diff_depth_manifest_v1",
            "status": "running",
            "run_id": "run-1",
            "collector_pid": 202,
            "symbol_run_dirs": {"BTCUSDT": str(symbol_dir)},
        },
    )

    report = run_cleanup(
        tmp_path,
        now=NOW,
        apply=True,
        pid_is_alive=lambda _pid: False,
    )

    assert report["outcome_counts"] == {
        "normalized": 1,
        "skipped_missing_owner_pid": 1,
    }
    assert json.loads(symbol_manifest.read_text(encoding="utf-8"))["status"] == "running"
