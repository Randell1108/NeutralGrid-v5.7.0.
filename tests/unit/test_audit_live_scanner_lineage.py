"""Exact-identity, deduplication and outcome-window checks for saved evidence."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.audit_live_scanner_lineage import audit


def _snapshot(
    path: Path, *, strategy_id: str, symbol: str, ts: str, raw_hash: str,
    deploy_ts: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "strategy_id": strategy_id, "symbol": symbol,
        "captured_at_utc": ts, "raw_text_sha256": raw_hash,
        "data_class": "live_bot_telemetry",
        "structured_telemetry": {"deploy_ts": deploy_ts} if deploy_ts else {},
    }), encoding="utf-8")


def test_audit_links_only_exact_identity_within_life_window(tmp_path: Path) -> None:
    expired = tmp_path / "expired.csv"
    pd.DataFrame([
        {"strategy_id": "123", "symbol": "BTCUSDT", "status": "Canceled", "start_time_utc": "2026-09-01T00:00:00Z", "end_time_utc": "2026-09-02T00:00:00Z", "pnl_pct": 2.0},
    ]).to_csv(expired, index=False)
    live = tmp_path / "Live"
    base = live / "2026-09-01" / "BTCUSDT"
    _snapshot(base / "private_telemetry_1.json", strategy_id="123", symbol="BTCUSDT", ts="2026-09-01T12:00:00Z", raw_hash="a", deploy_ts="2026-09-01T00:00:00Z")
    _snapshot(base / "private_telemetry_2.json", strategy_id="123", symbol="BTCUSDT", ts="2026-09-01T12:00:00Z", raw_hash="a")
    _snapshot(base / "private_telemetry_3.json", strategy_id="123", symbol="BTCUSDT", ts="2026-09-03T12:00:00Z", raw_hash="b")
    _snapshot(base / "private_telemetry_4.json", strategy_id="999", symbol="BTCUSDT", ts="2026-09-01T12:00:00Z", raw_hash="c")
    _snapshot(base / "private_telemetry_6.json", strategy_id="123", symbol="BTCUSDT", ts="2026-09-01T13:00:00Z", raw_hash="e", deploy_ts="2026-08-31T00:00:00Z")
    other = live / "2026-09-01" / "ETHUSDT"
    _snapshot(other / "private_telemetry_5.json", strategy_id="123", symbol="ETHUSDT", ts="2026-09-01T12:00:00Z", raw_hash="d")
    decision = tmp_path / "decisions.jsonl"
    decision.write_text("\n".join(json.dumps(row) for row in [
        {"strategy_id": "123", "symbol": "BTCUSDT", "ts": "2026-09-01T12:00:00Z", "verdict": "ADJUST"},
        {"strategy_id": "999", "symbol": "BTCUSDT", "ts": "2026-09-01T12:00:00Z", "verdict": "END"},
    ]) + "\n", encoding="utf-8")

    snapshots, pnl_observations, decisions, bots, summary = audit(
        root=tmp_path, live_root=live, expired_bots=expired, decision_paths=[decision]
    )

    assert summary["snapshot_classifications"] == {
        "deploy_time_conflict": 1, "duplicate_observation": 1, "matched_finalized": 1,
        "no_finalized_strategy": 1, "outside_life_window": 1,
        "strategy_symbol_conflict": 1,
    }
    assert summary["decision_classifications"] == {
        "matched_finalized": 1, "no_finalized_strategy": 1,
    }
    assert len(snapshots) == 6
    assert summary["snapshot_deploy_evidence"]["exact"] == 1
    assert pnl_observations == []
    assert len(decisions) == 2
    assert len(bots) == 1
    assert bots[0]["snapshot_count"] == 1
    assert bots[0]["pnl_pct"] == 2.0
    assert any(error["reason"] == "deploy_time_conflict" for error in summary["source_errors"])
    assert all(row["training_eligible"] is False for row in snapshots + decisions + bots)


def test_audit_keeps_bad_source_window_visible_without_linking(tmp_path: Path) -> None:
    expired = tmp_path / "expired.csv"
    pd.DataFrame([
        {"strategy_id": "123", "symbol": "BTCUSDT", "status": "expired", "start_time_utc": "2026-09-15T00:00:00Z", "end_time_utc": "1970-01-01T00:00:46Z"},
    ]).to_csv(expired, index=False)
    live = tmp_path / "Live"
    _snapshot(live / "2026-09-15" / "BTCUSDT" / "private_telemetry_1.json", strategy_id="123", symbol="BTCUSDT", ts="2026-09-15T12:00:00Z", raw_hash="a")

    snapshots, pnl_observations, decisions, bots, summary = audit(
        root=tmp_path, live_root=live, expired_bots=expired, decision_paths=[]
    )

    assert snapshots[0]["classification"] == "invalid_finalized_life_window"
    assert pnl_observations == []
    assert decisions == []
    assert bots == []
    assert summary["source_errors"][0]["reason"] == "invalid_finalized_life_window"


def test_audit_preserves_invalid_pnl_observation_as_source_error(tmp_path: Path) -> None:
    expired = tmp_path / "expired.csv"
    pd.DataFrame([
        {"strategy_id": "123", "symbol": "BTCUSDT", "status": "expired", "start_time_utc": "2026-09-15T00:00:00Z", "end_time_utc": "2026-09-16T00:00:00Z"},
    ]).to_csv(expired, index=False)
    pnl_path = (
        tmp_path / "Live" / "2026-09-15" / "BTCUSDT" / "pnl_history"
        / "bot-key" / "observations" / "bad.json"
    )
    pnl_path.parent.mkdir(parents=True)
    pnl_path.write_text("{}", encoding="utf-8")

    snapshots, pnl_observations, decisions, bots, summary = audit(
        root=tmp_path, live_root=tmp_path / "Live", expired_bots=expired,
        decision_paths=[],
    )

    assert snapshots == [] and decisions == [] and bots == []
    assert pnl_observations[0]["classification"] == "invalid_pnl_observation"
    assert summary["source_errors"][0]["reason"].startswith("invalid_pnl_observation:")
