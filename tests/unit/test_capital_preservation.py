"""Synthetic accounting checks; no fixture establishes live action safety."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
from pathlib import Path

import pytest
from pydantic import ValidationError

from neutralgrid.live.decision.capital_preservation import assess_current_capital, policy_summary
from neutralgrid.live.decision.exact_bot_ledger import PrivateFact
from neutralgrid.live.decision.state_action_reward import BotIdentity, SourceRef, State, file_hash

AT = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)


def evidence(tmp_path: Path, *, equity="100", peak=None, **changes):
    bot = BotIdentity(symbol="BTCUSDT", strategy_id="123", deployed_at=AT - timedelta(days=1))
    body = dict(bot=bot.model_dump(mode="json"), account_scope="test_account", kind="equity", native_id="equity1",
                occurred_at=AT.isoformat(), available_at=AT.isoformat(),
                fields=dict(equity_usdt=equity, invested_capital_usdt="100", unrealized_usdt="0", peak_equity_usdt=peak))
    body.update(changes)
    path = tmp_path / "equity.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    ref = SourceRef(path=str(path), sha256=file_hash(path))
    fact = PrivateFact.model_validate(body | {"source": ref.model_dump()})
    state = State(bot=bot, captured_at=AT, available_at=AT, feature_window_start=AT,
                  sources=(ref,), capital_usdt=Decimal("100"), unrealized_usdt=Decimal("0"),
                  features={"current_total_profit_usdt": 999999})
    return state, fact, path


@pytest.mark.parametrize("equity,status,headroom", [("99.99", "breached", "-0.01"), ("100", "observed_preserved", "0"),
                                                   ("101", "observed_preserved", "1"), ("-2", "breached", "-102")])
def test_exact_equity_comparison_is_not_future_guarantee(tmp_path, equity, status, headroom):
    state, fact, _ = evidence(tmp_path, equity=equity)
    result = assess_current_capital(state, fact, live_root=tmp_path)
    assert result.status == status and result.headroom_usdt == Decimal(headroom)
    assert result.primary_objective == "preserve_invested_capital"
    assert result.execution_authorized is False and result.training_eligible is False
    assert result.future_preservation_guaranteed is False
    assert result.to_dict() == result.model_dump(mode="json")


def test_missing_equity_never_uses_large_reported_profit_or_row_equity(tmp_path):
    state, _, _ = evidence(tmp_path)
    state = state.model_copy(update={"equity_usdt": Decimal("999999")})
    result = assess_current_capital(state, None, live_root=tmp_path)
    assert result.status == "missing" and result.equity_usdt is None
    assert "capital_floor_evidence_missing" in result.diagnostics


@pytest.mark.parametrize("field,value,reason", [
    ("capital_usdt", None, "capital_floor_unknown"),
    ("capital_usdt", Decimal("101"), "capital_floor_accounting_mismatch"),
    ("unrealized_usdt", None, "capital_floor_accounting_mismatch"),
    ("unrealized_usdt", Decimal("1"), "capital_floor_accounting_mismatch"),
    ("equity_usdt", Decimal("999"), "capital_floor_accounting_mismatch"),
    ("captured_at", AT - timedelta(seconds=1), "capital_floor_time_mismatch"),
])
def test_native_state_consistency_required(tmp_path, field, value, reason):
    state, fact, _ = evidence(tmp_path)
    if field == "captured_at":
        state = state.model_copy(update={"feature_window_start": value})
    state = state.model_copy(update={field: value})
    result = assess_current_capital(state, fact, live_root=tmp_path)
    assert result.status == "missing" and reason in result.diagnostics


def test_different_deployment_is_rejected(tmp_path):
    state, fact, _ = evidence(tmp_path)
    state = state.model_copy(update={"bot": state.bot.model_copy(update={"deployed_at": AT - timedelta(days=2)})})
    result = assess_current_capital(state, fact, live_root=tmp_path)
    assert result.status == "missing" and "capital_floor_identity_mismatch" in result.diagnostics


@pytest.mark.parametrize("asof", [AT - timedelta(seconds=1), AT.replace(tzinfo=None)])
def test_future_or_naive_asof_fails_closed(tmp_path, asof):
    state, fact, _ = evidence(tmp_path)
    assert assess_current_capital(state, fact, live_root=tmp_path, asof=asof).status == "missing"


def test_fact_unavailable_at_native_snapshot_fails_closed(tmp_path):
    state, fact, _ = evidence(tmp_path, available_at=(AT + timedelta(seconds=1)).isoformat())
    result = assess_current_capital(state, fact, live_root=tmp_path, asof=AT + timedelta(seconds=2))
    assert result.status == "missing" and "capital_floor_unavailable" in result.diagnostics


@pytest.mark.parametrize("mutation", ["missing", "changed", "malformed", "oversized", "duplicate_fields"])
def test_source_failures_are_diagnostics_not_runtime_crashes(tmp_path, mutation):
    state, fact, path = evidence(tmp_path)
    if mutation == "missing":
        path.unlink()
    elif mutation == "oversized":
        path.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    else:
        path.write_text({"changed": "{}", "malformed": "{", "duplicate_fields": '{"x":1,"x":2}'}[mutation])
    result = assess_current_capital(state, fact, live_root=tmp_path)
    assert result.status == "missing" and "capital_floor_source_invalid" in result.diagnostics


def test_source_pointer_failures_are_contained(tmp_path):
    state, fact, _ = evidence(tmp_path)
    result = assess_current_capital(state, fact.model_copy(update={"source_pointer": "/absent"}), live_root=tmp_path)
    assert result.status == "missing" and "capital_floor_source_invalid" in result.diagnostics


def test_rehashed_unrelated_source_does_not_prove_supplied_equity(tmp_path):
    state, fact, path = evidence(tmp_path)
    raw = json.loads(path.read_text())
    raw["fields"]["equity_usdt"] = "999"
    path.write_text(json.dumps(raw))
    fact = fact.model_copy(update={"source": SourceRef(path=str(path), sha256=file_hash(path))})
    result = assess_current_capital(state, fact, live_root=tmp_path)
    assert result.status == "missing" and "capital_floor_source_invalid" in result.diagnostics


def test_other_fact_kind_does_not_establish_equity(tmp_path):
    state, fact, _ = evidence(tmp_path, kind="income", native_id="1",
                              fields={"transaction_id": "1", "income_type": "REALIZED_PNL", "amount": "100", "asset": "USDT"})
    result = assess_current_capital(state, fact, live_root=tmp_path)
    assert result.status == "missing" and "capital_floor_equity_fact_required" in result.diagnostics


@pytest.mark.parametrize("equity", [True, "NaN", "Infinity"])
def test_unvalidated_financial_values_are_revalidated(tmp_path, equity):
    state, fact, _ = evidence(tmp_path)
    fact = fact.model_copy(update={"fields": fact.fields | {"equity_usdt": equity}})
    assert assess_current_capital(state, fact, live_root=tmp_path).status == "missing"


def test_peak_and_liquidation_limits_are_independent(tmp_path):
    state, fact, _ = evidence(tmp_path, equity="100", peak="1000")
    result = assess_current_capital(state, fact, live_root=tmp_path)
    assert result.status == "observed_preserved"
    assert policy_summary()["minimum_liquidation_buffer_pct"] is None
    assert result.execution_authorized is False


def test_expected_account_is_checked_when_supplied(tmp_path):
    state, fact, _ = evidence(tmp_path)
    assert assess_current_capital(state, fact, live_root=tmp_path, expected_account_scope="other").status == "missing"
    assert assess_current_capital(state, fact, live_root=tmp_path, expected_account_scope="test_account").status == "observed_preserved"


def test_policy_is_fresh_and_objective_cannot_be_changed(tmp_path):
    one = policy_summary()
    one["primary_objective"] = "maximize_profit"
    assert policy_summary()["primary_objective"] == "preserve_invested_capital"
    state, fact, _ = evidence(tmp_path)
    result = assess_current_capital(state, fact, live_root=tmp_path)
    with pytest.raises(ValidationError):
        result.primary_objective = "maximize_profit"


def test_old_observation_is_not_claimed_fresh_or_lifetime_safe(tmp_path):
    state, fact, _ = evidence(tmp_path)
    result = assess_current_capital(state, fact, live_root=tmp_path, asof=AT + timedelta(days=10))
    assert result.status == "observed_preserved" and result.scope == "current_observation_only"
    assert result.observed_at == AT and result.future_preservation_guaranteed is False


@pytest.mark.parametrize("equity", ["1E1000000", "-1E1000000"])
def test_source_bound_extreme_exponent_is_controlled(tmp_path, equity):
    state, fact, _ = evidence(tmp_path, equity=equity)
    result = assess_current_capital(state, fact, live_root=tmp_path)
    assert result.status == "missing" and result.headroom_usdt is None
    assert result.diagnostics == ("capital_floor_arithmetic_unavailable",)
    assert result.execution_authorized is result.training_eligible is False


def test_headroom_does_not_inherit_caller_decimal_precision(tmp_path):
    state, fact, _ = evidence(tmp_path, equity="100.01")
    expected = assess_current_capital(state, fact, live_root=tmp_path)
    with localcontext() as context:
        context.prec = 2
        context.Emax = 2
        context.Emin = -2
        result = assess_current_capital(state, fact, live_root=tmp_path)
    assert result == expected and result.headroom_usdt == Decimal("0.01")
