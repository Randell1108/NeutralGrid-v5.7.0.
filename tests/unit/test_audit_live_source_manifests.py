"""Manifest inventory remains read-only and refuses symbol-only attribution."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.audit_live_source_manifests import audit_manifests


def _manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manifest_audit_requires_exact_identity_and_time_overlap(tmp_path: Path) -> None:
    expired = tmp_path / "expired.csv"
    pd.DataFrame([{
        "strategy_id": "123", "symbol": "BTCUSDT", "status": "expired",
        "start_time_utc": "2026-09-01T00:00:00Z",
        "end_time_utc": "2026-09-02T00:00:00Z",
    }]).to_csv(expired, index=False)
    live = tmp_path / "Live"
    _manifest(
        live / "2026-09-01" / "BTCUSDT" / "diff_depth" / "run-1" / "manifest.json",
        {"symbol": "BTCUSDT", "target": {"strategy_id": "123"}, "run_id": "run-1",
         "started_at_utc": "2026-09-01T01:00:00Z",
         "completed_at_utc": "2026-09-01T02:00:00Z", "status": "complete_contiguous"},
    )
    _manifest(
        live / "2026-09-01" / "BTCUSDT" / "market_stream" / "run-2" / "manifest.json",
        {"symbol": "BTCUSDT", "target": {"strategy_id": None}, "run_id": "run-2",
         "started_at_utc": "2026-09-01T01:00:00Z", "status": "complete_contiguous"},
    )
    _manifest(
        live / "2026-09-03" / "BTCUSDT" / "private_user_stream" / "123"
        / "run-3" / "manifest.json",
        {"symbol": "BTCUSDT", "strategy_id": "123", "run_id": "run-3",
         "started_at_utc": "2026-09-03T01:00:00Z",
         "updated_at_utc": "2026-09-03T02:00:00Z", "status": "running"},
    )
    bad = live / "2026-09-03" / "BTCUSDT" / "diff_depth" / "run-4" / "manifest.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{", encoding="utf-8")

    rows, summary = audit_manifests(root=tmp_path, live_root=live, expired_bots=expired)

    assert summary["manifest_count"] == 4
    classifications = {row["run_id"]: row["classification"] for row in rows if row.get("run_id")}
    assert classifications == {
        "run-1": "overlap_valid_window",
        "run-2": "symbol_only",
        "run-3": "outside_life_window",
    }
    assert sum(row["classification"] == "invalid_manifest" for row in rows) == 1
    assert summary["manifest_errors"][0]["reason"].startswith("invalid_manifest:")
    assert all(row["training_eligible"] is False for row in rows)
