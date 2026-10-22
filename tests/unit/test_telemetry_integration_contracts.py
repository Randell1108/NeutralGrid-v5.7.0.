"""Regression proofs for live evidence identity and adapter boundaries."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from neutralgrid.live.decision import action_shadow_io as adapter
from neutralgrid.live.decision import meta_shadow_analysis as meta_analysis
from neutralgrid.live.decision.meta_shadow_analysis import (
    ShadowAnalysisConfig, _split_fit_oos_by_join_key, _decision_identity_frame, _prepare_eligible, analyze_joined_shadow,
    join_shadow_to_outcomes, load_decision_jsonl,
)
from neutralgrid.live.decision.pnl_history import build_pnl_observation
from neutralgrid.live.decision.state_action_reward import file_hash


def _outcomes() -> pd.DataFrame:
    return pd.DataFrame([
        {"candidate_id": "scan-1", "strategy_id": "bot-1", "symbol": "BTCUSDT",
         "start_time_utc": "2026-10-01T00:00:00Z", "end_time_utc": "2026-10-01T02:00:00Z", "pnl_pct": 2.0},
        {"candidate_id": "scan-1", "strategy_id": "bot-2", "symbol": "BTCUSDT",
         "start_time_utc": "2026-10-02T00:00:00Z", "end_time_utc": "2026-10-02T02:00:00Z", "pnl_pct": -3.0},
    ])


def test_reused_candidate_matches_each_exact_deployment_once() -> None:
    decisions = pd.DataFrame([
        {"candidate_id": "scan-1", "strategy_id": "bot-1", "symbol": "BTCUSDT"},
        {"candidate_id": "scan-1", "strategy_id": "bot-2", "symbol": "BTCUSDT"},
        {"candidate_id": "scan-1", "strategy_id": "", "symbol": "BTCUSDT"},
        {"candidate_id": "wrong-scan", "strategy_id": "bot-1", "symbol": "BTCUSDT"},
    ])
    joined = join_shadow_to_outcomes(decisions, _outcomes())
    assert list(joined["pnl_pct"]) == [2.0, -3.0]
    assert list(joined["join_key"]) == ["bot-1|BTCUSDT", "bot-2|BTCUSDT"]


def test_candidate_and_legacy_rows_use_the_same_bot_split_key() -> None:
    decisions = pd.DataFrame([
        {"candidate_id": "scan-1", "strategy_id": "bot-1", "symbol": "BTCUSDT"},
        {"candidate_id": "", "strategy_id": "bot-1", "symbol": "BTCUSDT"},
    ])
    joined = join_shadow_to_outcomes(decisions, _outcomes().iloc[:1])
    assert joined["join_key"].nunique() == 1


def test_pending_reused_candidates_remain_distinct_bots() -> None:
    frame = _decision_identity_frame(pd.DataFrame([
        {"candidate_id": "scan-1", "strategy_id": "bot-1", "symbol": "BTCUSDT"},
        {"candidate_id": "scan-1", "strategy_id": "bot-2", "symbol": "BTCUSDT"},
    ]))
    assert set(frame["join_key"]) == {"bot-1|BTCUSDT", "bot-2|BTCUSDT"}


@pytest.mark.parametrize("probability,pnl", [(float("inf"), 1), (1.1, 1), (-.1, 1), (.4, float("inf"))])
def test_invalid_numeric_evidence_never_enters_calibration(probability: float, pnl: float) -> None:
    frame = pd.DataFrame([{"join_key": "bot-1|BTCUSDT", "meta_proba": probability, "pnl_pct": pnl,
        "meta_authoritative": True, "meta_full_fidelity": True, "verdict": "ADJUST",
        "ts_utc": "2026-10-01T01:00:00Z", "start_time_utc": "2026-10-01T00:00:00Z",
        "end_time_utc": "2026-10-01T02:00:00Z"}])
    assert _prepare_eligible(frame, config=ShadowAnalysisConfig()).empty


def test_exact_outcome_copies_are_idempotent_but_conflicts_fail_closed() -> None:
    decisions = pd.DataFrame([{"candidate_id": "scan-1", "strategy_id": "bot-1", "symbol": "BTCUSDT"}])
    same = pd.concat([_outcomes().iloc[:1]] * 2, ignore_index=True)
    assert len(join_shadow_to_outcomes(decisions, same)) == 1
    same.loc[1, "pnl_pct"] = -10.0
    with pytest.raises(ValueError, match="duplicate strategy_id/symbol"):
        join_shadow_to_outcomes(decisions, same)


def test_explicit_deployment_conflict_cannot_join() -> None:
    decisions = pd.DataFrame([{"candidate_id": "scan-1", "strategy_id": "bot-1", "symbol": "BTCUSDT",
                               "deploy_ts_utc": "2026-10-02T00:00:00Z"}])
    assert join_shadow_to_outcomes(decisions, _outcomes().iloc[:1]).empty


def test_copied_decision_logs_do_not_multiply_observations(tmp_path: Path) -> None:
    record = {"ts": "2026-10-01T01:00:00Z", "strategy_id": "bot-1", "symbol": "BTCUSDT",
              "verdict": "ADJUST", "evaluation": {"meta_proba": 0.3}}
    paths = [tmp_path / "one.jsonl", tmp_path / "copy.jsonl"]
    for path in paths:
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert len(load_decision_jsonl(paths + paths)) == 1
    record["verdict"] = "END"
    paths[1].write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting decision"):
        load_decision_jsonl(paths)


def test_analysis_cli_contains_invalid_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(**kwargs):
        raise ValueError("conflicting evidence")
    monkeypatch.setattr(meta_analysis, "analyze_from_files", reject)
    output = tmp_path / "blocked.json"
    code = meta_analysis.main(["--decisions", str(tmp_path / "input.jsonl"), "--output", str(output)])
    result = json.loads(output.read_text(encoding="utf-8"))
    assert code == 2
    assert result["status"] == "analysis_blocked" and result["ready_to_enable"] is False
    assert result["recommended_live_config"] == {"meta_tilt_enabled": False}


def _observation(tmp_path: Path) -> tuple[dict, dict]:
    at = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    evaluated = at + timedelta(seconds=30)
    raw = tmp_path / "raw.txt"
    raw.write_text("synthetic exact-bot evidence", encoding="utf-8")
    row = {"ts": at.isoformat(), "symbol": "BTCUSDT", "strategy_id": "123", "verdict": "CONTINUE",
           "evaluation": {"evaluated_at_utc": evaluated.isoformat(), "price": 105,
                          "symbol_unavailable": False, "transient_fetch_error": False},
           "execution_telemetry": {"captured_at": at.isoformat(), "pnl": {"total_profit_usdt": 1.0},
                                   "position_inventory": {"size_base": 1},
                                   "grid": {"price_range_lower": 90, "price_range_upper": 110,
                                            "num_grids": 12, "invested_margin_usdt": 100}}}
    observation = build_pnl_observation(row, deploy_ts=at - timedelta(days=1),
                                        source_cycle_manifest=str(tmp_path / "manifest.json"),
                                        source_snapshot_path=str(raw), source_snapshot_sha256=file_hash(raw))
    path = tmp_path / "observation.json"
    path.write_text(json.dumps(observation), encoding="utf-8")
    observation["_source_path"] = str(path)
    return row, observation


def test_adapter_uses_per_bot_evaluation_time_and_actual_availability_field(tmp_path: Path) -> None:
    row, observation = _observation(tmp_path)
    state = adapter.state_from_observations([observation], row)
    assert state.available_at.isoformat() == row["evaluation"]["evaluated_at_utc"]
    assert state.symbol_available is True
    assert state.equity_usdt is None and state.peak_equity_usdt is None
    row["evaluation"]["transient_fetch_error"] = True
    assert adapter.state_from_observations([observation], row).symbol_available is None


def test_adapter_never_falls_back_from_invalid_per_bot_time(tmp_path: Path) -> None:
    row, observation = _observation(tmp_path)
    row["evaluation"]["evaluated_at_utc"] = "invalid"
    with pytest.raises(ValueError):
        adapter.state_from_observations([observation], row)


def test_adapter_rejects_conflicting_explicit_deployment(tmp_path: Path) -> None:
    row, observation = _observation(tmp_path)
    row["deploy_ts"] = "2026-09-01T00:00:00Z"
    with pytest.raises(ValueError, match="deployment"):
        adapter.state_from_observations([observation], row)


def test_adapter_contains_malformed_state_without_changing_incumbent(tmp_path: Path) -> None:
    row, observation = _observation(tmp_path)
    row["evaluation"]["evaluated_at_utc"] = "invalid"
    original = copy.deepcopy(row)
    result = adapter.report_shadow_row(row=row, observations=[observation], model_path=None, limits_payload=None)
    assert result["available"] is False
    assert result["verdict"] == "CONTINUE"
    assert result["execution_eligible"] is False and result["influence_enabled"] is False
    assert result["diagnostics"] == ["shadow_adapter_failure:ValueError"]
    assert row == original


def test_adapter_still_excludes_genuinely_future_evidence(tmp_path: Path) -> None:
    row, observation = _observation(tmp_path)
    future = copy.deepcopy(observation)
    future["evaluated_at_utc"] = "2026-10-01T12:01:00+00:00"
    assert adapter.state_from_observations([observation, future], row) == adapter.state_from_observations([observation], row)


def test_recorded_shadow_decision_uses_the_same_available_time(tmp_path: Path) -> None:
    row, observation = _observation(tmp_path)
    result = adapter.report_shadow_row(row=row, observations=[observation], model_path=None,
                                       limits_payload=None, live_root=tmp_path / "Live")
    assert result["verdict"] == "CONTINUE" and result["influence_enabled"] is False
    assert result["decision_event_path"]
    event = json.loads(Path(result["decision_event_path"]).read_text())
    assert event["payload"]["decided_at"] == row["evaluation"]["evaluated_at_utc"].replace("+00:00", "Z")


def test_temporal_split_purges_fit_outcomes_not_known_before_oos() -> None:
    frame = pd.DataFrame([
        {"join_key": "a", "ts_utc": pd.Timestamp("2026-10-01T01:00:00Z"),
         "end_time_utc": "2026-10-03T00:00:00Z"},
        {"join_key": "b", "ts_utc": pd.Timestamp("2026-10-02T01:00:00Z"),
         "end_time_utc": "2026-10-02T02:00:00Z"},
    ])
    fit, oos = _split_fit_oos_by_join_key(frame, fit_fraction=0.5)
    assert fit.empty and list(oos["join_key"]) == ["b"]


def test_mixed_hmm_cohorts_cannot_claim_calibration_readiness() -> None:
    rows = []
    for i in range(40):
        at = datetime(2026, 10, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        rows.append({"join_key": str(i), "ts_utc": at.isoformat(),
                     "start_time_utc": at.isoformat(), "end_time_utc": (at + timedelta(minutes=30)).isoformat(),
                     "verdict": "ADJUST", "meta_proba": 0.4 if i % 2 == 0 else 0.8,
                     "pnl_pct": -1 if i % 2 == 0 else 2,
                     "meta_authoritative": True, "meta_full_fidelity": True,
                     "active_hmm_version": "old" if i < 20 else "new",
                     "linked_hmm_version": "old" if i < 20 else "new",
                     "meta_feature_profile": "profile", "contract_version": "1.1"})
    result = analyze_joined_shadow(pd.DataFrame(rows), config=ShadowAnalysisConfig(thresholds=(0.5,)))
    assert result["ready_to_enable"] is False
    assert result["status"] == "unverified_calibration_cohort"


def test_recurring_options_are_explicit_and_remain_observational(tmp_path: Path) -> None:
    from scripts import run_drawer_verdict_cycle as recurring
    args = recurring.parse_args([
        "--cycle-manifest", str(tmp_path / "cycle.json"),
        "--scanner-config-file", str(tmp_path / "scanner.yaml"),
        "--private-event-manifest", str(tmp_path / "private.json"),
        "--pnl-forecast-artifact-dir", str(tmp_path / "pnl"),
        "--volatility-forecast-artifact-dir", str(tmp_path / "volatility"),
        "--shadow-action-artifact-dir", str(tmp_path / "action"),
        "--shadow-action-limits", str(tmp_path / "limits.json"),
        "--record-shadow-decisions",
    ])
    forwarded = recurring.controller_args(args)
    assert forwarded.observational_only and forwarded.require_l2_evidence
    assert forwarded.private_event_manifest == [tmp_path / "private.json"]
    assert forwarded.scanner_config_file == tmp_path / "scanner.yaml"
    assert forwarded.pnl_forecast_artifact_dir == tmp_path / "pnl"
    assert forwarded.volatility_forecast_artifact_dir == tmp_path / "volatility"
    assert forwarded.shadow_action_artifact_dir == tmp_path / "action"
    assert forwarded.shadow_action_limits == tmp_path / "limits.json"
    assert forwarded.record_shadow_decisions
    defaults = recurring.controller_args(recurring.parse_args(["--cycle-manifest", str(tmp_path / "cycle.json")]))
    assert not defaults.record_shadow_decisions and defaults.scanner_config_file is None
    assert defaults.pnl_forecast_artifact_dir is None and defaults.private_event_manifest == []
