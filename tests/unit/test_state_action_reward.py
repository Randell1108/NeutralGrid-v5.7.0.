"""Synthetic fixtures verify evidence rules; none represent executed live returns."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from neutralgrid.core import constants
from neutralgrid.live.decision.recommender import Verdict
from neutralgrid.live.decision.state_action_reward import (
    Action, Approval, BotIdentity, CostEvent, Decision, Execution, LedgerEvent,
    Finalized, Maintenance, Outcome, RiskLimits, SarError, Slippage, SourceRef, State,
    append_event, atomic_create_json, configuration_hash, contract_schema_catalog, feature_definition_hash,
    file_hash, load_events, make_event, project_transition, recenter, shield_reasons,
)

T0 = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
D = Decimal


@pytest.fixture
def evidence(tmp_path):
    path = tmp_path / "source.json"
    path.write_text('{}', encoding="utf-8")
    return SourceRef(path=str(path), sha256=file_hash(path))


def state(evidence, **changes):
    values = dict(bot=BotIdentity(symbol="BTCUSDT", strategy_id="123", deployed_at=T0 - timedelta(days=1)),
                  captured_at=T0, available_at=T0, feature_window_start=T0,
                  sources=(evidence,), grid_lower=D(90), grid_upper=D(110), num_grids=12,
                  capital_usdt=D(100), equity_usdt=D(100), unrealized_usdt=D(0),
                  position_quantity=D(1), market_price=D(105), market_captured_at=T0,
                  symbol_available=True, liquidation_buffer_pct=D(30), peak_equity_usdt=D(100),
                  hmm_lineage="test-hmm", features={"range_prob": .7})
    return State(**(values | changes))


def limits():
    return RiskLimits(min_liquidation_buffer_pct=D(10), max_drawdown_pct=D(20))


def decision(pre, **changes):
    values = dict(decided_at=T0, pre_state=pre,
                  incumbent=Action(verdict=Verdict.CONTINUE), shadow=Action(verdict=Verdict.CONTINUE),
                  limits=limits(), origin="live", config_hash=configuration_hash(limits()),
                  feature_hash=feature_definition_hash())
    return Decision(**(values | changes))


def event(kind, payload, at=None):
    if isinstance(payload, Decision):
        bot = payload.pre_state.bot
    elif isinstance(payload, Execution):
        bot = payload.pre_state.bot
    elif isinstance(payload, (Outcome, Maintenance)):
        bot = payload.state.bot
    else:
        bot = BotIdentity(symbol="BTCUSDT", strategy_id="123", deployed_at=T0 - timedelta(days=1))
    return make_event(bot=bot, decision_id="test-decision", kind=kind,
                      occurred_at=at or T0, payload=payload.model_dump(mode="json"))


def cost(evidence, kind, amount, identity=None):
    return CostEvent(exchange_id=identity or kind, kind=kind, occurred_at=T0 + timedelta(minutes=1),
                     source_asset="USDT", source_amount=D(amount), amount_usdt=D(amount), source=evidence)


def observed(evidence, *, adjust=False, endpoint_minutes=60, **outcome_changes):
    pre = state(evidence)
    action = recenter(pre, quantum=D('.1')) if adjust else Action(verdict=Verdict.CONTINUE)
    dec = decision(pre, shadow=action)
    events = [event("decision", dec)]
    end = state(evidence, captured_at=T0 + timedelta(minutes=endpoint_minutes),
                available_at=T0 + timedelta(minutes=endpoint_minutes), equity_usdt=D(101),
                unrealized_usdt=D('-.2'), grid_lower=action.lower if adjust else D(90),
                grid_upper=action.upper if adjust else D(110))
    if adjust:
        approval_path = Path(evidence.path).with_name("approval.json")
        approval_raw = dict(schema_version=constants.ACTION_APPROVAL_SCHEMA, action="ADJUST", symbol=pre.bot.symbol,
                            strategy_id=pre.bot.strategy_id, preserve_current_position=True,
                            suggested_grid_lower=str(action.lower), suggested_grid_upper=str(action.upper),
                            approved_at_utc=(T0 + timedelta(seconds=15)).isoformat(),
                            expires_at_utc=(T0 + timedelta(seconds=120)).isoformat(), idempotency_key="1" * 64)
        approval_path.write_text(json.dumps(approval_raw))
        approval_source = SourceRef(path=str(approval_path), sha256=file_hash(approval_path))
        approval = Approval(action=action, approved_at=T0 + timedelta(seconds=15),
                            expires_at=T0 + timedelta(seconds=120), source=approval_source)
        approved = event("approval", approval, approval.approved_at)
        before = state(evidence, captured_at=T0 + timedelta(seconds=20), available_at=T0 + timedelta(seconds=20))
        after = state(evidence, captured_at=T0 + timedelta(seconds=40), available_at=T0 + timedelta(seconds=40),
                      grid_lower=action.lower, grid_upper=action.upper)
        receipt_path = Path(evidence.path).with_name("execution.json")
        receipt_path.write_text(json.dumps(dict(schema_version=constants.ACTION_EXECUTION_SCHEMA, status="executed",
            action="ADJUST", symbol=pre.bot.symbol, strategy_id=pre.bot.strategy_id, preserve_current_position=True,
            additional_investment="0", num_grids=12, requested_grid_lower=str(action.lower),
            requested_grid_upper=str(action.upper), executed_grid_lower=str(action.lower),
            executed_grid_upper=str(action.upper), price_decimals=1, idempotency_key="1" * 64,
            pre_state=dict(symbol=pre.bot.symbol, strategy_id=pre.bot.strategy_id, status="Working",
                           lower="90", upper="110", num_grids=12, position_summary="fixture"),
            post_state=dict(symbol=pre.bot.symbol, strategy_id=pre.bot.strategy_id, status="Working",
                            lower=str(action.lower), upper=str(action.upper), num_grids=12,
                            position_summary="fixture"))))
        receipt_source = SourceRef(path=str(receipt_path), sha256=file_hash(receipt_path))
        executed = Execution(requested=action, applied=action, approval_event_id=approved.event_id,
                             pre_state=before, post_state=after, submitted_at=T0 + timedelta(seconds=30),
                             verified_at=T0 + timedelta(seconds=41), receipt=receipt_source, quantum=D('.1'),
                             preserve_position=True, additional_investment_usdt=D(0))
        events += [approved, event("execution", executed, executed.verified_at)]
    else:
        maintenance = Maintenance(state=end, coverage_start=T0, coverage_end=end.captured_at,
                                  intervention_coverage_complete=True, source=evidence)
        events.append(event("maintenance", maintenance, end.available_at))
    values = dict(state=end, coverage_start=T0, coverage_end=end.captured_at,
                  event_completeness="event_complete", intervention_coverage_complete=True,
                  realized_includes_commission=False, realized_includes_funding=False,
                  costs=(cost(evidence, "realized", '1.5'), cost(evidence, "commission", '.2'),
                         cost(evidence, "funding", '-.1')))
    outcome = Outcome(**(values | outcome_changes))
    events.append(event("outcome", outcome, outcome.state.available_at))
    return events


def test_versions_and_documentation_are_pinned():
    assert constants.DECISION_CONTRACT_VERSION == "1.1"
    assert constants.STATE_ACTION_REWARD_CONTRACT_VERSION == "1.0"
    root = Path(__file__).resolve().parents[2]
    doc = (root / "STATE_ACTION_REWARD_CONTRACT.md").read_text(encoding="utf-8")
    for name in ("DECISION_CONTRACT_VERSION", "ACTION_INTENT_SCHEMA", "ACTION_APPROVAL_SCHEMA",
                 "ACTION_EXECUTION_SCHEMA", "STATE_ACTION_REWARD_CONTRACT_VERSION", "STATE_ACTION_REWARD_SCHEMA"):
        assert f"| {name} | {getattr(constants, name)} |" in doc
    assert "Status: Contract v1.1 implemented" in (root / "LIVE_DECISION.md").read_text(encoding="utf-8")
    assert {v.value for v in Verdict} == {"CONTINUE", "ADJUST", "END"}
    catalog = json.loads((root / "docs" / "state_action_reward_v1.schema.catalog.json").read_text())
    assert catalog == contract_schema_catalog()


@pytest.mark.parametrize("adjust", [False, True])
def test_observed_behavior_reconciles_without_fabrication(evidence, adjust):
    result = project_transition(observed(evidence, adjust=adjust))
    assert result.training_eligible
    assert result.net_usdt == D(1)
    assert result.normalized_return == D('.01')
    assert result.behavior.verdict == (Verdict.ADJUST if adjust else Verdict.CONTINUE)


def test_unexecuted_adjust_never_inherits_later_returns(evidence):
    events = observed(evidence, adjust=True)
    result = project_transition([e for e in events if e.kind != "execution"])
    assert result.net_usdt is None and result.behavior is None
    assert "verified_behavior_missing" in result.reasons


@pytest.mark.parametrize("adjust", [False, True])
def test_later_capital_change_censors_action_association(evidence, adjust):
    events = observed(evidence, adjust=adjust)
    outcome = Outcome.model_validate(events[-1].payload)
    end = outcome.state.model_copy(update={"capital_usdt": D(200)})
    outcome = outcome.model_copy(update={"state": end})
    events[-1] = event("outcome", outcome, end.available_at)
    result = project_transition(events)
    assert result.net_usdt is None and not result.training_eligible


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("idempotency_key", None), ("schema_version", "unknown"),
    ("symbol", "ETHUSDT"), ("executed_grid_lower", "94"),
    ("price_decimals", True), ("pre_state", {}),
])
def test_native_receipt_cannot_be_relabelled_as_execution(evidence, field, value):
    events = observed(evidence, adjust=True)
    execution_event = next(e for e in events if e.kind == "execution")
    execution = Execution.model_validate(execution_event.payload)
    path = Path(execution.receipt.path)
    raw = json.loads(path.read_text())
    raw[field] = value
    path.write_text(json.dumps(raw))
    execution = execution.model_copy(update={"receipt": SourceRef(path=str(path), sha256=file_hash(path))})
    events = [event("execution", execution, execution.verified_at) if e.kind == "execution" else e for e in events]
    with pytest.raises((SarError, ValueError)):
        project_transition(events)


def test_native_approval_and_receipt_must_share_nonempty_identity(evidence):
    events = observed(evidence, adjust=True)
    execution = Execution.model_validate(next(e.payload for e in events if e.kind == "execution"))
    path = Path(execution.receipt.path)
    raw = json.loads(path.read_text())
    raw["idempotency_key"] = "2" * 64
    path.write_text(json.dumps(raw))
    execution = execution.model_copy(update={"receipt": SourceRef(path=str(path), sha256=file_hash(path))})
    events = [event("execution", execution, execution.verified_at) if e.kind == "execution" else e for e in events]
    result = project_transition(events)
    assert result.net_usdt is None and "approval_receipt_identity_mismatch" in result.reasons


def test_event_cannot_precede_verified_action(evidence):
    events = observed(evidence, adjust=True)
    execution = Execution.model_validate(next(e.payload for e in events if e.kind == "execution"))
    with pytest.raises(ValueError, match="lifecycle"):
        event("execution", execution, execution.verified_at - timedelta(seconds=1))


def test_verified_continue_is_distinct_from_unexecuted_shadow_adjust(evidence):
    events = observed(evidence)
    dec = Decision.model_validate(events[0].payload)
    dec = dec.model_copy(update={"shadow": recenter(dec.pre_state, quantum=D('.1'))})
    events[0] = event("decision", dec)
    result = project_transition(events)
    assert result.decision.shadow.verdict == Verdict.ADJUST
    assert result.behavior.verdict == Verdict.CONTINUE and result.net_usdt == D(1)
    result = project_transition([e for e in events if e.kind != "maintenance"])
    assert result.behavior is None and result.net_usdt is None


def test_first_endpoint_is_not_replaced_by_later_better_evidence(evidence):
    events = observed(evidence, event_completeness="unknown")
    outcome = Outcome.model_validate(events[-1].payload)
    end = outcome.state.model_copy(update={"captured_at": T0 + timedelta(minutes=65),
                                           "available_at": T0 + timedelta(minutes=65)})
    later = outcome.model_copy(update={"state": end, "coverage_end": end.captured_at,
                                       "event_completeness": "event_complete"})
    events.append(event("outcome", later, end.available_at))
    result = project_transition(events)
    assert result.endpoint == T0 + timedelta(minutes=60) and result.net_usdt is None


def test_finalized_lifetime_is_preserved_without_replacing_missing_hour(evidence, tmp_path):
    events = observed(evidence)
    lifetime = Outcome.model_validate(events[-1].payload)
    terminal = Finalized(terminal_at=lifetime.state.captured_at, terminal_status="ended", source=evidence,
                         lifetime=lifetime)
    events = [e for e in events if e.kind != "outcome"]
    events.append(event("finalized", terminal, terminal.terminal_at))
    root = tmp_path / "Live"
    for item in events:
        append_event(item, live_root=root)
    loaded, rejected = load_events(root)
    assert not rejected and any(e.kind == "finalized" for e in loaded)
    assert project_transition(loaded).net_usdt is None


def test_relative_sources_and_boolean_limits_are_rejected():
    with pytest.raises(ValueError, match="absolute"):
        SourceRef(path="relative.json", sha256="0" * 64)
    with pytest.raises(ValueError, match="boolean"):
        RiskLimits(min_liquidation_buffer_pct=D(10), max_drawdown_pct=D(20), max_market_age_seconds=True)


def test_parallel_writers_publish_one_event_or_report_busy(evidence, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    item = event("decision", decision(state(evidence)))
    live = tmp_path / "Live"
    def write():
        try:
            return append_event(item, live_root=live)
        except SarError as exc:
            assert "busy" in str(exc)
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: write(), range(12)))
    assert any(paths) and len(set(p for p in paths if p is not None)) == 1
    loaded, rejected = load_events(live)
    assert loaded == [item] and not rejected


@pytest.mark.parametrize("change,reason", [
    ({"event_completeness": "snapshot_only"}, "incomplete_accounting_or_interventions"),
    ({"intervention_coverage_complete": False}, "incomplete_accounting_or_interventions"),
    ({"interventions": (T0 + timedelta(minutes=30),)}, "intervening_action"),
    ({"realized_includes_commission": None}, "pnl_inclusion_semantics_unknown"),
])
def test_incomplete_or_intervened_outcome_is_not_a_reward(evidence, change, reason):
    result = project_transition(observed(evidence, **change))
    assert result.net_usdt is None and not result.training_eligible
    assert reason in result.reasons


@pytest.mark.parametrize("minutes,eligible", [(59, False), (60, True), (65, True), (65.01, False)])
def test_endpoint_boundaries(evidence, minutes, eligible):
    assert project_transition(observed(evidence, endpoint_minutes=minutes)).training_eligible == eligible


def test_fee_funding_inclusions_are_undone_once(evidence):
    result = project_transition(observed(evidence, realized_includes_commission=True,
                                        realized_includes_funding=True,
                                        costs=(cost(evidence, "realized", '1.2'), cost(evidence, "commission", '.2'),
                                               cost(evidence, "funding", '-.1'))))
    assert result.net_usdt == D(1)


def test_equity_reconciliation_and_conversion_fail_closed(evidence):
    events = observed(evidence, costs=(cost(evidence, "realized", '100'),))
    assert project_transition(events).net_usdt is None
    foreign = cost(evidence, "commission", '.2').model_copy(update={"source_asset": "BNB"})
    assert "cost_conversion_unknown" in project_transition(observed(evidence, costs=(foreign,))).reasons


def test_exchange_duplicates_are_deduplicated_and_conflicts_rejected(evidence):
    duplicate = cost(evidence, "commission", '.2')
    values = (cost(evidence, "realized", '1.5'), duplicate, duplicate, cost(evidence, "funding", '-.1'))
    assert project_transition(observed(evidence, costs=values)).net_usdt == D(1)
    changed = cost(evidence, "commission", '.3')
    assert "conflicting_exchange_event" in project_transition(observed(evidence, costs=values + (changed,))).reasons


def test_external_flow_removed_from_equity_delta(evidence):
    end = state(evidence, captured_at=T0 + timedelta(hours=1), available_at=T0 + timedelta(hours=1),
                equity_usdt=D(121), unrealized_usdt=D('-.2'))
    result = project_transition(observed(evidence, state=end, costs=(cost(evidence, "realized", '1.5'),
        cost(evidence, "commission", '.2'), cost(evidence, "funding", '-.1'), cost(evidence, "external_flow", '20'))))
    assert result.net_usdt == D(1)


def test_slippage_is_separate_and_cannot_use_future_benchmark(evidence):
    slip = Slippage(fill_id="1", fill_at=T0 + timedelta(minutes=1), side="BUY", quantity=D(1),
                    fill_price=D(101), benchmark_price=D(100), benchmark_at=T0 + timedelta(seconds=59),
                    estimated=False, source=evidence, benchmark_source=evidence)
    assert slip.cost_usdt == D(1)
    assert project_transition(observed(evidence, slippage=(slip,))).net_usdt == D(1)
    with pytest.raises(ValidationError):
        Slippage(**(slip.model_dump() | {"benchmark_at": T0 + timedelta(minutes=2)}))
    assert project_transition(observed(evidence, slippage=(slip, slip))).net_usdt == D(1)
    changed = slip.model_copy(update={"fill_price": D(102)})
    assert "conflicting_slippage_fill" in project_transition(observed(evidence, slippage=(slip, changed))).reasons


def test_approval_mismatch_and_expiry_are_not_execution(evidence):
    events = observed(evidence, adjust=True)
    approved = next(e for e in events if e.kind == "approval")
    source_path = Path(approved.payload["source"]["path"])
    source_raw = json.loads(source_path.read_text())
    source_raw["expires_at_utc"] = (T0 + timedelta(seconds=16)).isoformat()
    source_path.write_text(json.dumps(source_raw))
    payload = approved.payload | {"expires_at": T0 + timedelta(seconds=16),
                                  "source": {"path": str(source_path), "sha256": file_hash(source_path)}}
    changed = make_event(bot=approved.bot, decision_id=approved.decision_id, kind="approval",
                         occurred_at=approved.occurred_at, payload=payload)
    result = project_transition([changed if e.kind == "approval" else e for e in events])
    assert result.net_usdt is None and "approval_expired_or_out_of_order" in result.reasons


@pytest.mark.parametrize("changes,reason", [
    ({"liquidation_buffer_pct": None}, "liquidation_buffer_missing_or_breached"),
    ({"peak_equity_usdt": None}, "drawdown_unknown"),
    ({"peak_equity_usdt": D(200)}, "drawdown_breached"),
    ({"symbol_available": False}, "symbol_unavailable_or_unknown"),
    ({"market_captured_at": T0 - timedelta(minutes=1)}, "stale_market_data"),
    ({"hmm_lineage": None}, "hmm_lineage_unknown"),
])
def test_shields_are_deterministic(evidence, changes, reason):
    assert reason in shield_reasons(state(evidence, **changes), limits(), now=T0)
    assert shield_reasons(state(evidence), None, now=T0) == ["missing_risk_limits"]


def test_recenter_cap_and_rounding(evidence):
    action = recenter(state(evidence, market_price=D(1000)), quantum=D('.1'))
    assert action.lower == D(95) and action.upper == D(115)
    with pytest.raises(SarError, match="rounding"):
        recenter(state(evidence, grid_lower=D('90.04'), grid_upper=D('110.04'), market_price=D(1000)), quantum=D('.1'))


def test_future_features_boolean_finance_and_unknown_verdict_rejected(evidence):
    with pytest.raises(ValidationError):
        state(evidence, features={"hlabel": 3})
    with pytest.raises(ValidationError):
        state(evidence, capital_usdt=True)
    with pytest.raises(ValidationError):
        Action(verdict="UNKNOWN")
    with pytest.raises(ValidationError):
        decision(state(evidence, available_at=T0 + timedelta(seconds=1)))


def test_end_cannot_be_downgraded(evidence):
    with pytest.raises(ValidationError, match="END"):
        decision(state(evidence), incumbent=Action(verdict=Verdict.END))


def test_immutable_publication_and_cross_date_dedup(evidence, tmp_path):
    root = tmp_path / "Live"
    e = event("decision", decision(state(evidence)))
    path = append_event(e, live_root=root, ingested_at=T0)
    assert append_event(e, live_root=root, ingested_at=T0 + timedelta(days=1)) == path
    assert len(load_events(root)[0]) == 1
    assert "/2026-10-01/" in path.as_posix()
    changed = event("decision", decision(state(evidence), diagnostics=("changed",)))
    with pytest.raises(SarError, match="conflicting"):
        append_event(changed, live_root=root)
    raw = json.loads(path.read_text())
    raw["contract_version"] = "99"
    with pytest.raises(ValidationError):
        LedgerEvent.model_validate(raw)


def test_tampered_source_quarantines_entire_transition(evidence, tmp_path):
    root = tmp_path / "Live"
    for e in observed(evidence):
        append_event(e, live_root=root)
    Path(evidence.path).write_text('{"changed":true}')
    events, rejected = load_events(root)
    assert events == [] and len(rejected) == 3


def test_misrouted_symbol_directory_is_rejected(evidence, tmp_path):
    root = tmp_path / "Live"
    item = event("decision", decision(state(evidence)))
    path = append_event(item, live_root=root)
    wrong = path.parents[3] / "ETHUSDT" / "sar" / path.parent.name / path.name
    wrong.parent.mkdir(parents=True)
    wrong.write_bytes(path.read_bytes())
    loaded, rejected = load_events(root)
    assert loaded == [] and len(rejected) == 1


def test_interrupted_publication_never_creates_committed_event(tmp_path, monkeypatch):
    target = tmp_path / "event.json"
    monkeypatch.setattr("neutralgrid.live.decision.state_action_reward.os.link",
                        lambda *_: (_ for _ in ()).throw(OSError("simulated publication failure")))
    with pytest.raises(OSError):
        atomic_create_json(target, {"proof": True})
    assert not target.exists() and list(tmp_path.iterdir()) == []
