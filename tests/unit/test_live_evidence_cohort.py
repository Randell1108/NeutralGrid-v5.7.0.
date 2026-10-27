"""Censoring, provenance and failure-boundary proofs for evidence collection."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from neutralgrid.live.decision.action_shadow_io import report_shadow_row
from neutralgrid.live.decision.pnl_history import build_pnl_observation
from neutralgrid.live.decision.recommender import Verdict
from neutralgrid.live.decision.state_action_reward import file_hash


T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


def observation(minutes: float, pnl: float, *, margin: float = 100, strategy: str = "123") -> dict:
    at = (T0 + timedelta(minutes=minutes)).isoformat()
    return build_pnl_observation(
        {"symbol": "BTCUSDT", "strategy_id": strategy, "ts": at, "verdict": "CONTINUE",
         "execution_telemetry": {"captured_at": at, "pnl": {"total_profit_usdt": pnl},
                                 "grid": {"invested_margin_usdt": margin}}},
        deploy_ts=T0, source_cycle_manifest="unused", source_snapshot_path="unused",
        source_snapshot_sha256="0" * 64,
    )


def test_invalid_incumbent_is_contained_without_inventing_an_action():
    result = report_shadow_row(row={"verdict": "UNKNOWN"}, observations=[], model_path=None, limits_payload=None)
    assert result["available"] is False
    assert result["verdict"] is None
    assert result["execution_eligible"] is False
    assert result["observed_reward"] is None


def test_target_deadline_is_deployment_plus_seven_hours():
    from neutralgrid.live.decision.evidence_cohort import target_cohort
    result = target_cohort([observation(60, 0), observation(419, 3), observation(421, 4)])
    assert result["bots"][0]["classification"] == "observed_target_hit"
    assert result["origins"][0]["target_label"] == 1
    assert result["origins"][0]["remaining_minutes"] == 360
    assert result["origins"][1]["classification"] == "already_observed_at_target"
    assert result["origins"][2]["classification"] == "outside_target_horizon"


@pytest.mark.parametrize("rows", [
    [observation(60, -2), observation(420, -1)],
    [observation(60, -2), observation(421, 5)],
    [observation(60, -2), observation(419, -1)],
])
def test_sparse_nonhits_are_never_negative_labels(rows):
    from neutralgrid.live.decision.evidence_cohort import target_cohort
    result = target_cohort(rows)
    assert result["bots"][0]["classification"] == "censored_target_path"
    assert all(row["target_label"] is None for row in result["origins"])
    assert result["fit_eligible"] is False


def test_deadline_boundary_and_post_hit_giveback_are_not_mislabelled():
    from neutralgrid.live.decision.evidence_cohort import target_cohort
    result = target_cohort([observation(1, 0), observation(419, 3), observation(420, -10)])
    assert result["bots"][0]["classification"] == "observed_target_hit"
    assert result["origins"][-1]["classification"] == "already_observed_at_target"
    assert result["origins"][-1]["target_label"] is None
    assert result["origins"][-1]["reported_pnl_pct"] == -10


def test_capital_change_blocks_target_cohort_and_duplicates_do_not_add_support():
    from neutralgrid.live.decision.evidence_cohort import target_cohort
    one = observation(1, 0)
    result = target_cohort([one, copy.deepcopy(one), observation(420, 6, margin=200)])
    assert len(result["origins"]) == 2
    assert result["bots"][0]["classification"] == "capital_basis_changed_or_missing"
    assert result["positive_origins"] == 0


def test_conflicting_feature_copies_quarantine_the_bot():
    from neutralgrid.live.decision.evidence_cohort import target_cohort
    one = observation(1, 0)
    other = observation(1, 0)
    other["record_integrity_sha256"] = "1" * 64
    with pytest.raises(RuntimeError):
        target_cohort([one, other])


def test_cross_bot_future_hit_never_labels_another_bot():
    from neutralgrid.live.decision.evidence_cohort import target_cohort
    result = target_cohort([observation(1, 0), observation(420, 3, strategy="456")])
    assert result["origins"][0]["target_label"] is None


def test_tick_size_comes_from_price_filter_not_display_precision():
    from neutralgrid.live.decision.evidence_cohort import exchange_ticks
    raw = {"symbols": [{"symbol": "BTCUSDT", "pricePrecision": 2,
                        "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.05"}]}]}
    assert exchange_ticks(raw) == {"BTCUSDT": "0.05"}
    raw["symbols"].append(copy.deepcopy(raw["symbols"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        exchange_ticks(raw)


def test_packet_commit_is_immutable_and_cross_date_idempotent(tmp_path):
    from neutralgrid.live.decision.evidence_cohort import publish_packet
    packet = {"symbol": "BTCUSDT", "bot_identity": "a" * 64, "facts": [], "training_eligible": False}
    first = publish_packet(packet, live_root=tmp_path, now=T0)
    second = publish_packet(packet, live_root=tmp_path, now=T0 + timedelta(days=1))
    assert first == second
    assert len(list(tmp_path.rglob("*.json"))) == 1
    assert first.parts[-4] == "BTCUSDT"
    assert json.loads(first.read_text())["training_eligible"] is False


def test_packet_rejects_path_escape_and_naive_ingestion_time(tmp_path):
    from neutralgrid.live.decision.evidence_cohort import publish_packet
    for packet, now in [({"symbol": "../bad", "bot_identity": "a" * 64}, T0),
                        ({"symbol": "BTCUSDT", "bot_identity": "a" * 64}, T0.replace(tzinfo=None))]:
        with pytest.raises(ValueError):
            publish_packet(packet, live_root=tmp_path, now=now)
    assert not list(tmp_path.rglob("*.json"))


def test_bounded_json_rejects_duplicate_keys_and_nonfinite_values(tmp_path):
    from neutralgrid.live.decision.evidence_cohort import read_json
    source = tmp_path / "bad.json"
    for text in ['{"a":1,"a":2}', '{"a":NaN}', '[]']:
        source.write_text(text)
        with pytest.raises(ValueError):
            read_json(source)


def native_observation(tmp_path: Path, *, pnl: float = 1, verdict=None) -> tuple[Path, dict]:
    symbol_dir = tmp_path / "2026-10-01" / "BTCUSDT"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    raw = symbol_dir / "native.txt"
    raw.write_text("\n".join(["BTCUSDT", "Time Created", "2026-09-30 19:00:00",
                              "Total Profit (USDT)", "1.00 (1.00%)", "Positions", "no position",
                              "Pending Order", "Grid Details", "Invested Margin", "100.00",
                              "Advanced (Optional)", "Strategy Number", "123"]), encoding="utf-8")
    at = (T0 + timedelta(minutes=1)).isoformat()
    raw.with_suffix(".json").write_text(json.dumps({"symbol": "BTCUSDT", "strategy_id": "123",
        "raw_text_sha256": file_hash(raw), "captured_at_utc": at}), encoding="utf-8")
    record = build_pnl_observation({"symbol": "BTCUSDT", "strategy_id": "123", "ts": at, "verdict": verdict,
        "execution_telemetry": {"captured_at": at, "pnl": {"total_profit_usdt": pnl},
                                "grid": {"invested_margin_usdt": 100}}}, deploy_ts=T0,
        source_cycle_manifest="unused", source_snapshot_path=str(raw), source_snapshot_sha256=file_hash(raw))
    filename = (T0 + timedelta(minutes=1)).strftime("%Y%m%dT%H%M%S%fZ") + "_" + record["observation_id"][:16] + ".json"
    path = symbol_dir / "pnl_history" / record["bot_identity"] / "observations" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path, record


def test_collector_reparses_native_measurement_and_contains_missing_verdict(tmp_path):
    from neutralgrid.live.decision.evidence_cohort import collect_observations, evidence_packets, publish_packet
    native_observation(tmp_path)
    rows, rejected, refs = collect_observations(tmp_path)
    assert len(rows) == 1 and not rejected and len(refs) == 3
    packet = evidence_packets(rows, transitions=[])[0]
    assert packet["verdict_counts"] == {"MISSING": 1}
    assert packet["eligibility_evidence"]["observed_action_count"] == 0
    assert packet["missing_action_link"] is True
    publish_packet(packet, live_root=tmp_path, now=T0)
    after, _, _ = collect_observations(tmp_path)
    assert len(after) == 1  # Evidence indexes cannot multiply training observations.


def test_forged_pnl_with_valid_record_hash_is_rejected_against_original(tmp_path):
    from neutralgrid.live.decision.evidence_cohort import collect_observations
    native_observation(tmp_path, pnl=20)
    rows, rejected, _ = collect_observations(tmp_path)
    assert not rows
    assert "native drawer pnl.total_profit_usdt mismatch" in rejected[0]["reason"]


def test_native_capture_time_mismatch_and_mutation_are_rejected(tmp_path):
    from neutralgrid.live.decision.evidence_cohort import collect_observations
    _, record = native_observation(tmp_path)
    raw = Path(record["source_snapshot_path"])
    envelope = raw.with_suffix(".json")
    payload = json.loads(envelope.read_text())
    payload["captured_at_utc"] = T0.isoformat()
    envelope.write_text(json.dumps(payload))
    assert "capture envelope time mismatch" in collect_observations(tmp_path)[1][0]["reason"]
    raw.write_text("tampered")
    assert "raw snapshot hash mismatch" in collect_observations(tmp_path)[1][0]["reason"]


def test_malformed_record_quarantines_other_records_for_same_deployment(tmp_path):
    from neutralgrid.live.decision.evidence_cohort import collect_observations
    path, _ = native_observation(tmp_path)
    (path.parent / "broken.json").write_text("not-json")
    rows, rejected, _ = collect_observations(tmp_path)
    assert not rows and len(rejected) == 1


def test_missing_or_boolean_capital_is_censored_not_imputed():
    from neutralgrid.live.decision.evidence_cohort import target_cohort
    # Builder enforces numeric fields; its missing-value path is admissible.
    row = observation(1, 4)
    at = row["evaluated_at_utc"]
    missing = build_pnl_observation({"symbol": "BTCUSDT", "strategy_id": "123", "ts": at,
        "execution_telemetry": {"captured_at": at, "pnl": {"total_profit_usdt": 4}}},
        deploy_ts=T0, source_cycle_manifest="unused", source_snapshot_path="unused", source_snapshot_sha256="0" * 64)
    assert target_cohort([missing])["bots"][0]["classification"] == "capital_basis_changed_or_missing"
    with pytest.raises(RuntimeError):
        observation(1, 4, margin=True)


def test_current_ticks_never_become_historical_decision_precision(tmp_path):
    from neutralgrid.live.decision.evidence_cohort import collect_observations, evidence_packets
    native_observation(tmp_path)
    rows, _, _ = collect_observations(tmp_path)
    packets = evidence_packets(rows, transitions=[], tick_capture={"source": {}, "received_at_utc": T0.isoformat(),
                                                                "ticks": {"BTCUSDT": "0.05", "ETHUSDT": "0.01"}})
    assert packets[0]["exchange_metadata_capture"]["tick_size"] == "0.05"
    assert packets[0]["exchange_metadata_is_historical"] is False
    assert packets[0]["training_eligible"] is False


def test_research_output_cannot_overwrite_models_or_existing_reports(tmp_path):
    from scripts.capture_live_evidence_cohort import ROOT, research_output
    for path in (ROOT / "models/new", ROOT / "Live/new", ROOT / "reports", ROOT / "reports/live_telemetry_evidence_20261022"):
        with pytest.raises(ValueError):
            research_output(path, ROOT / "Live")


def test_oversized_packet_cannot_commit_an_unreadable_index(tmp_path):
    from neutralgrid.live.decision.evidence_cohort import MAX_BYTES, publish_packet
    with pytest.raises(ValueError, match="bounded read size"):
        publish_packet({"symbol": "BTCUSDT", "bot_identity": "a" * 64, "data": "x" * MAX_BYTES},
                       live_root=tmp_path, now=T0)
    assert not list(tmp_path.rglob("*.json"))


def test_ingestion_batch_checks_sources_before_any_publication(tmp_path):
    from scripts.capture_live_evidence_cohort import validate_ingestion
    from neutralgrid.live.decision.state_action_reward import (
        Action, BotIdentity, Decision, SourceRef, State, configuration_hash,
        feature_definition_hash, make_event,
    )
    raw = tmp_path / "source.json"
    raw.write_text("{}")
    bot = BotIdentity(symbol="BTCUSDT", strategy_id="123", deployed_at=T0)
    state = State(bot=bot, captured_at=T0, available_at=T0, feature_window_start=T0,
                  sources=(SourceRef(path=str(raw), sha256=file_hash(raw)),))
    decision = Decision(decided_at=T0, pre_state=state, incumbent=Action(verdict=Verdict.CONTINUE),
                        shadow=Action(verdict=Verdict.CONTINUE), origin="live", limits=None,
                        config_hash=configuration_hash(None), feature_hash=feature_definition_hash())
    event = make_event(bot=bot, decision_id="fixture", kind="decision", occurred_at=T0,
                       payload=decision.model_dump(mode="json"))
    envelope = tmp_path / "event.json"
    envelope.write_text(event.model_dump_json())
    assert validate_ingestion([envelope]) == [event]
    raw.write_text('{"changed": true}')
    with pytest.raises(RuntimeError, match="source missing or changed"):
        validate_ingestion([envelope])
    assert not list(tmp_path.glob("*/BTCUSDT/sar/*/*.json"))


def test_public_metadata_network_failure_is_contained(monkeypatch, tmp_path):
    from scripts import capture_live_evidence_cohort as collector
    def unavailable(*args, **kwargs):
        raise OSError("fixture network unavailable")
    monkeypatch.setattr(collector.urllib.request, "urlopen", unavailable)
    monkeypatch.setattr(collector, "research_output", lambda path, live_root: path)
    native_observation(tmp_path / "Live")
    report = collector.run(live_root=tmp_path / "Live", output=tmp_path / "report", fetch_ticks=True)
    assert report["valid_observations"] == 1
    assert report["tick_capture_failure"] == "OSError: fixture network unavailable"
    assert report["captured_packet_paths"] == []
    assert report["fitted"] is False


def test_terminal_cohort_cannot_supply_missing_action_or_first_passage_labels(monkeypatch, tmp_path):
    from scripts import capture_live_evidence_cohort as collector
    monkeypatch.setattr(collector, "research_output", lambda path, live_root: path)
    terminal = {"scores": [{"reported_total_pnl_usdt": "-56.32", "training_eligible": False}],
                "rejected": [{"path": "fixture", "reason": "source_changed"}]}
    monkeypatch.setattr(collector, "audit_terminal_outcomes", lambda live_root: terminal)
    native_observation(tmp_path / "Live")
    output = tmp_path / "report"
    report = collector.run(live_root=tmp_path / "Live", output=output)
    assert report["terminal_trajectory_outcomes"] == 1
    assert report["rejected_terminal_outcomes"] == terminal["rejected"]
    assert report["negative_target_origins"] == 0
    assert report["verified_action_examples"] == 0
    assert not report["fitted"] and not report["promotion_eligible"]
    assert json.loads((output / "terminal_outcome_cohort.json").read_text()) == terminal
