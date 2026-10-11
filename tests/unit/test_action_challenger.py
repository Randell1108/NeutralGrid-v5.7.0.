"""Synthetic model/fault experiments. No fixture is claimed as a live trade."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from neutralgrid.live.decision.action_challenger import (
    ActionArtifact, evaluate, predict_shadow, temporal_split, unavailable,
)
from neutralgrid.live.decision.action_path_features import PricePoint, path_features, signature_level_two
from neutralgrid.live.decision.recommender import Verdict
from neutralgrid.live.decision.state_action_reward import (
    Action, BotIdentity, Decision, Projection, RiskLimits, SarError, SourceRef, State,
    configuration_hash, feature_definition_hash, file_hash, fingerprint,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
D = Decimal


@pytest.fixture
def rows(tmp_path):
    source = tmp_path / "synthetic-source.json"
    source.write_text('{}')
    ref = SourceRef(path=str(source), sha256=file_hash(source))
    limits = RiskLimits(min_liquidation_buffer_pct=D(10), max_drawdown_pct=D(20))
    result = []
    for i in range(120):
        at = T0 + timedelta(hours=3*i)
        x = float((i % 10) - 5)
        s = State(bot=BotIdentity(symbol="BTCUSDT", strategy_id=str(i), deployed_at=at - timedelta(days=1)),
                  captured_at=at, available_at=at, feature_window_start=at, sources=(ref,),
                  grid_lower=D(90), grid_upper=D(110), num_grids=12, capital_usdt=D(100),
                  equity_usdt=D(100), unrealized_usdt=D(0), position_quantity=D(1),
                  market_price=D(105), market_captured_at=at, symbol_available=True,
                  liquidation_buffer_pct=D(30), peak_equity_usdt=D(100), hmm_lineage="synthetic-hmm",
                  features={"current_total_profit_usdt": x, "range_prob": .6})
        action = Action(verdict=Verdict.CONTINUE) if i % 2 == 0 else Action(verdict=Verdict.ADJUST, lower=D(95), upper=D(115))
        dec = Decision(decided_at=at, pre_state=s, incumbent=Action(verdict=Verdict.CONTINUE), shadow=action,
                       limits=limits, origin="live", config_hash=configuration_hash(limits), feature_hash=feature_definition_hash())
        reward = D(str(2*x + (1 if i % 2 else 0)))
        result.append(Projection(f"synthetic-{i}", dec, action, "verified_behavior", at+timedelta(hours=1),
                                 reward, reward/D(100), True, ()))
    return result


def test_small_model_roundtrip_and_holdout_report(rows):
    report, model = evaluate(rows)
    assert report["evaluated"] and model is not None
    assert report["promotion_eligible"] is False and report["policy_value"] is None
    assert "inventory_equity_order_replay_not_proven" in report["blockers"]
    assert set(report["behavior_support_fraction"]) == {"validation", "test"}
    restored = ActionArtifact.model_validate_json(model.model_dump_json())
    assert restored.artifact_id == model.artifact_id
    raw = json.loads(model.model_dump_json())
    raw["ridge_intercept"] += 1
    with pytest.raises(ValueError, match="checksum"):
        ActionArtifact.model_validate(raw)


def test_training_deduplicates_and_rejects_conflicting_transitions(rows):
    report, model = evaluate(rows + [rows[0]])
    assert model is not None and report["deduplicated_count"] == 1
    assert report["eligible_count"] == len(rows)
    changed = Projection(rows[0].transition_id, rows[0].decision, rows[0].behavior,
                         "verified_behavior", rows[0].endpoint, D(999), D(9), True, ())
    report, model = evaluate(rows + [changed])
    assert model is None and "conflicting_transition_projection" in report["blockers"]


def test_simulated_origin_cannot_enter_executed_training(rows):
    simulated = [Projection(r.transition_id, r.decision.model_copy(update={"origin": "simulation"}),
                            r.behavior, "verified_behavior", r.endpoint, r.net_usdt,
                            r.normalized_return, True, ()) for r in rows]
    report, model = evaluate(simulated)
    assert report["eligible_count"] == 0 and model is None


def test_holdout_requires_both_observed_actions(rows):
    _, _, test = temporal_split(rows)
    test_ids = {r.transition_id for r in test}
    changed = [Projection(r.transition_id, r.decision, Action(verdict=Verdict.CONTINUE),
                          r.evidence_class, r.endpoint, r.net_usdt, r.normalized_return, True, ())
               if r.transition_id in test_ids else r for r in rows]
    report, model = evaluate(changed)
    assert model is None and "test_action_missing" in report["blockers"]


def test_temporal_split_is_bot_disjoint_purged_and_embargoed(rows):
    fit, validation, test = temporal_split(rows)
    groups = [{(r.decision.pre_state.bot.symbol, r.decision.pre_state.bot.strategy_id) for r in group}
              for group in (fit, validation, test)]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert max(r.endpoint for r in fit) < min(r.decision.pre_state.feature_window_start for r in validation)
    assert max(r.endpoint for r in validation) < min(r.decision.pre_state.feature_window_start for r in test)
    # A related strategy spanning time cutoffs is removed from all cohorts.
    related = []
    for r in rows:
        bot = r.decision.pre_state.bot.model_copy(update={"strategy_id": "same-strategy"})
        s = r.decision.pre_state.model_copy(update={"bot": bot})
        related.append(Projection(r.transition_id, r.decision.model_copy(update={"pre_state": s}), r.behavior,
                                  r.evidence_class, r.endpoint, r.net_usdt, r.normalized_return, True, ()))
    assert temporal_split(related) == ([], [], [])


def test_unverified_rows_never_fit_and_mixed_lineage_is_blocked(rows):
    bad = [Projection(r.transition_id, r.decision, None, "uncontrolled", r.endpoint, None, None, False,
                      ("verified_behavior_missing",)) for r in rows]
    report, model = evaluate(bad)
    assert model is None and report["eligible_count"] == 0
    s = rows[-1].decision.pre_state.model_copy(update={"hmm_lineage": "other"})
    changed = rows[-1].decision.model_copy(update={"pre_state": s})
    rows[-1] = Projection(rows[-1].transition_id, changed, rows[-1].behavior, "verified_behavior",
                          rows[-1].endpoint, rows[-1].net_usdt, rows[-1].normalized_return, True, ())
    assert "mixed_or_stale_lineage_profile_features" in evaluate(rows)[0]["blockers"]


def test_unavailable_and_failed_model_preserve_incumbent(rows):
    s = rows[0].decision.pre_state
    for v in Verdict:
        result = unavailable(v, "missing")
        assert result["verdict"] == v.value and not result["influence_enabled"]
    result = predict_shadow(incumbent=Action(verdict=Verdict.CONTINUE), state=s, limits=None,
                            quantum=D('.1'), artifact=None, now=s.captured_at)
    assert result["verdict"] == "CONTINUE" and "missing_risk_limits" in result["diagnostics"]
    result = predict_shadow(incumbent=Action(verdict=Verdict.END), state=s,
                            limits=rows[0].decision.limits, quantum=None, artifact=None, now=s.captured_at)
    assert result["verdict"] == "END" and not result["available"]


def test_supported_shadow_prediction_and_unsupported_state(rows):
    _, model = evaluate(rows)
    assert model is not None
    # Widen only uncertainty/support in this explicit synthetic runtime fixture.
    body = model.model_dump(mode="json", exclude={"artifact_id"})
    body.update(holdout_passed=True, interval_radius_usdt=0.0,
                support_radii={"CONTINUE": 100.0, "ADJUST": 100.0})
    model = ActionArtifact.model_validate(body | {"artifact_id": fingerprint(body)})
    s = rows[0].decision.pre_state
    result = predict_shadow(incumbent=Action(verdict=Verdict.CONTINUE), state=s,
                            limits=rows[0].decision.limits, quantum=D('.1'), artifact=model, now=s.captured_at)
    assert result["available"] and result["verdict"] in {"CONTINUE", "ADJUST"}
    assert result["execution_eligible"] is False and result["observed_reward"] is None
    unsupported = s.model_copy(update={"features": {"current_total_profit_usdt": 1e9, "range_prob": .6}})
    result = predict_shadow(incumbent=Action(verdict=Verdict.CONTINUE), state=unsupported,
                            limits=rows[0].decision.limits, quantum=D('.1'), artifact=model, now=s.captured_at)
    assert result["verdict"] == "CONTINUE" and not result["available"]
    assert any("behavior_support_missing" in reason for reason in result["diagnostics"])


def test_path_signature_signed_components_and_subdivision():
    up = signature_level_two([(0,0), (.5,1), (1,0)])
    down = signature_level_two([(0,0), (.5,-1), (1,0)])
    assert up != down and np.linalg.norm(up[2:]) == np.linalg.norm(down[2:])
    assert signature_level_two([(0,0), (1,2)]) == signature_level_two([(0,0), (.5,1), (1,2)])


def test_future_path_perturbation_and_gaps(rows):
    s = rows[0].decision.pre_state
    ref = s.sources[0]
    points = [PricePoint(captured_at=s.captured_at - timedelta(minutes=60-5*i),
                         available_at=s.captured_at - timedelta(minutes=60-5*i), price=100+i,
                         source_path=ref.path, source_sha256=ref.sha256) for i in range(13)]
    original = path_features(points, asof=s.captured_at)
    future = PricePoint(captured_at=s.captured_at + timedelta(minutes=5),
                        available_at=s.captured_at + timedelta(minutes=5), price=1e9,
                        source_path=ref.path, source_sha256=ref.sha256)
    assert original == path_features(points + [future], asof=s.captured_at)
    with pytest.raises(ValueError, match="endpoints"):
        path_features(points[1:], asof=s.captured_at)
    with pytest.raises(ValueError):
        path_features(points[:6] + points[7:], asof=s.captured_at)


def test_offline_cli_audit_isolated_output_and_no_invented_reward(tmp_path):
    from scripts.evaluate_shadow_actions import run_audit
    live = tmp_path / "Live"
    live.mkdir()
    output = tmp_path / "research"
    report = run_audit(live_root=live, output_dir=output, path_ablation=True)
    assert report["verified_training_examples"] == 0 and report["observed_policy_gain"] is None
    assert report["prespecified_variants"] == ["snapshot", "summaries", "signature"]
    assert not list(output.rglob("model.json"))
    with pytest.raises(SarError):
        run_audit(live_root=live, output_dir=output)
    with pytest.raises(SarError):
        run_audit(live_root=live, output_dir=live / "experiment")


def test_cli_and_optional_controller_flags_cannot_enable_execution():
    from scripts import run_live_telemetry_controller as controller
    with pytest.raises(SystemExit):
        controller.parse_args(["--record-shadow-decisions"])
    args = controller.parse_args(["--record-shadow-decisions", "--observational-only"])
    assert args.observational_only
