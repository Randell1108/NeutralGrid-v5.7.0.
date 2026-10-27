"""Synthetic source files exercise conversion, not live performance evidence."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from neutralgrid.core.constants import ACTION_APPROVAL_SCHEMA, ACTION_EXECUTION_SCHEMA
from neutralgrid.live.decision.exact_bot_ledger import PrivateFact, append_fact
from neutralgrid.live.decision.native_sar import (
    NativeRecord, ledger_costs, ledger_outcome, native_approval, native_decision,
    native_execution, native_state,
)
from neutralgrid.live.decision.pnl_history import build_pnl_observation
from neutralgrid.live.decision.state_action_reward import (
    BotIdentity, Decision, Outcome, SarError, SourceRef, file_hash, project_transition,
)

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
BOT = BotIdentity(symbol="BTCUSDT", strategy_id="123", deployed_at=T0)


def source(path: Path, value: dict) -> NativeRecord:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return NativeRecord(SourceRef(path=str(path), sha256=file_hash(path)))


def observation(root: Path, seconds=60, *, verdict="CONTINUE", lower=90, upper=110):
    directory = root / "2026-10-01" / "BTCUSDT"
    directory.mkdir(parents=True, exist_ok=True)
    at = (T0 + timedelta(seconds=seconds)).isoformat()
    raw = directory / f"drawer{seconds}.txt"
    raw.write_text("\n".join(["BTCUSDT", "Time Created", "2026-09-30 19:00:00",
        "Total Profit (USDT)", "1.00 (1.00%)", "Positions", "no position", "Pending Order",
        "Grid Details", "Invested Margin", "100.00", "Advanced (Optional)", "Strategy Number", "123"]), encoding="utf-8")
    source(raw.with_suffix(".json"), {"symbol": "BTCUSDT", "strategy_id": "123",
        "raw_text_sha256": file_hash(raw), "captured_at_utc": at})
    row = {"symbol": "BTCUSDT", "strategy_id": "123", "ts": at, "deploy_ts": T0.isoformat(), "verdict": verdict,
        "suggested_grid_lower": 95, "suggested_grid_upper": 115,
        "execution_telemetry": {"captured_at": at, "pnl": {"total_profit_usdt": 1, "equity_usdt": 9999},
            "risk": {"peak_equity_usdt": 10000},
            "grid": {"invested_margin_usdt": 100, "price_range_lower": lower, "price_range_upper": upper, "num_grids": 12},
            "position_inventory": {"size_base": 1, "position_pnl_usdt": 0}}}
    record = build_pnl_observation(row, deploy_ts=T0, source_cycle_manifest="fixture", source_snapshot_path=str(raw),
                                   source_snapshot_sha256=file_hash(raw))
    filename = (T0 + timedelta(seconds=seconds)).strftime("%Y%m%dT%H%M%S%fZ") + "_" + record["observation_id"][:16] + ".json"
    record_source = source(directory / "pnl_history" / record["bot_identity"] / "observations" / filename, record)
    record["_source_path"] = record_source.source.path
    return source(directory / f"decision{seconds}.json", row), record


def fact(root: Path, kind="income", *, native_id="20", fields=None, account="account_A", ownership=None):
    fields = fields or {"transaction_id": native_id, "income_type": "FUNDING_FEE", "amount": "-.2", "asset": "USDT"}
    body = dict(bot=BOT.model_dump(mode="json"), account_scope=account, kind=kind, native_id=native_id,
                occurred_at=(T0 + timedelta(seconds=120)).isoformat(),
                available_at=(T0 + timedelta(seconds=125)).isoformat(), fields=fields)
    ref = source(root / f"{kind}-{native_id}-{account}.json", body)
    return PrivateFact.model_validate(body | {"source": ref.source.model_dump(), "ownership_fact": ownership})


def costs(root, facts):
    return ledger_costs(facts, bot=BOT, account_scope="account_A", start=T0 + timedelta(seconds=60),
                        end=T0 + timedelta(seconds=3660), available_at=T0 + timedelta(seconds=3660), live_root=root)


def test_decision_preserves_proposal_and_unknown_equity_and_limits(tmp_path):
    row, obs = observation(tmp_path)
    event = native_decision(row, [obs], live_root=tmp_path)
    parsed = Decision.model_validate(event.payload)
    assert parsed.limits is None and parsed.pre_state.equity_usdt is None and parsed.pre_state.peak_equity_usdt is None
    assert "primary_objective:preserve_invested_capital" in parsed.diagnostics
    assert "capital_floor_evidence_missing" in parsed.diagnostics
    transition = project_transition([event])
    assert transition.behavior is None
    assert not transition.training_eligible


@pytest.mark.parametrize("verdict", ["CONTINUE", "ADJUST", "END"])
def test_all_native_verdicts_convert_without_manufacturing_actions(tmp_path, verdict):
    row, obs = observation(tmp_path, verdict=verdict)
    event = native_decision(row, [obs], live_root=tmp_path)
    assert event.payload["incumbent"]["verdict"] == verdict
    assert project_transition([event]).behavior is None


def test_changed_original_snapshot_is_rejected(tmp_path):
    row, obs = observation(tmp_path)
    Path(obs["source_snapshot_path"]).write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        native_decision(row, [obs], live_root=tmp_path)


def test_caller_cannot_override_persisted_observation_using_unchanged_checksum(tmp_path):
    row, obs = observation(tmp_path)
    obs["grid"]["invested_margin_usdt"] = 999
    event = native_decision(row, [obs], live_root=tmp_path)
    assert Decision.model_validate(event.payload).pre_state.capital_usdt == Decimal("100")


def test_capture_envelope_stays_bound_after_conversion(tmp_path):
    row, obs = observation(tmp_path)
    event = native_decision(row, [obs], live_root=tmp_path)
    Path(obs["source_snapshot_path"]).with_suffix(".json").write_text("{}")
    with pytest.raises(SarError, match="source missing or changed"):
        project_transition([event])


def test_future_observations_are_not_attached_to_asof_decision_provenance(tmp_path):
    row, obs = observation(tmp_path)
    _, future = observation(tmp_path, 3660)
    decision = native_decision(row, [obs, future], live_root=tmp_path)
    paths = {ref.path for ref in Decision.model_validate(decision.payload).pre_state.sources}
    assert future["source_snapshot_path"] not in paths and future["_source_path"] not in paths


def test_native_jsonl_is_bound_to_line_and_file_hash(tmp_path):
    path = tmp_path / "decisions.jsonl"
    path.write_text('{"a":1}\n{"a":2}\n')
    record = NativeRecord(SourceRef(path=str(path), sha256=file_hash(path)), 2)
    assert record.read() == {"a": 2}
    path.write_text('{"a":1}\n{"a":3}\n')
    with pytest.raises(SarError, match="changed"):
        record.read()


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '[]'])
def test_ambiguous_native_jsonl_rejected(tmp_path, text):
    path = tmp_path / "bad.jsonl"
    path.write_text(text)
    with pytest.raises(SarError):
        NativeRecord(SourceRef(path=str(path), sha256=file_hash(path)), 1).read()


def test_costs_deduplicate_exact_fact_and_keep_signed_funding(tmp_path):
    one = fact(tmp_path)
    result = costs(tmp_path, [one, one])
    assert len(result) == 1 and result[0].kind == "funding" and result[0].amount_usdt == Decimal("-.2")


def test_exact_time_equity_fact_can_populate_equity_without_inventing_peak(tmp_path):
    row, obs = observation(tmp_path, 120)
    payload = row.read()
    payload["ts"] = (T0 + timedelta(seconds=125)).isoformat()
    row = source(Path(row.source.path), payload)
    equity = fact(tmp_path, "equity", fields={"equity_usdt": "101", "invested_capital_usdt": "100",
        "unrealized_usdt": "0", "peak_equity_usdt": None})
    state = native_state(row, [obs], live_root=tmp_path, equity_fact=equity)
    assert state.equity_usdt == Decimal("101") and state.peak_equity_usdt is None


@pytest.mark.parametrize("seconds,capital", [(119, "100"), (120, "101")])
def test_equity_requires_exact_time_and_same_capital(tmp_path, seconds, capital):
    row, obs = observation(tmp_path, seconds)
    payload = row.read()
    payload["ts"] = (T0 + timedelta(seconds=125)).isoformat()
    row = source(Path(row.source.path), payload)
    equity = fact(tmp_path, "equity", fields={"equity_usdt": "101", "invested_capital_usdt": capital,
        "unrealized_usdt": "0", "peak_equity_usdt": None})
    with pytest.raises(SarError):
        native_state(row, [obs], live_root=tmp_path, equity_fact=equity)


def test_boundary_costs_at_start_are_excluded(tmp_path):
    one = fact(tmp_path)
    result = ledger_costs([one], bot=BOT, account_scope="account_A", start=one.occurred_at,
                          end=one.available_at, available_at=one.available_at, live_root=tmp_path)
    assert result == ()


def test_signed_income_commission_becomes_positive_cost(tmp_path):
    one = fact(tmp_path, fields={"transaction_id": "20", "income_type": "COMMISSION", "amount": "-.3", "asset": "USDT"})
    assert costs(tmp_path, [one])[0].amount_usdt == Decimal(".3")


def test_non_usdt_fees_remain_unconverted(tmp_path):
    one = fact(tmp_path, fields={"transaction_id": "20", "income_type": "COMMISSION", "amount": "-.3", "asset": "BNB"})
    assert costs(tmp_path, [one])[0].amount_usdt is None


def test_account_mismatch_rejected(tmp_path):
    with pytest.raises(SarError, match="identity"):
        costs(tmp_path, [fact(tmp_path, account="account_B")])


def test_conflicting_native_identity_rejected(tmp_path):
    one = fact(tmp_path)
    second_root = tmp_path / "second"
    two = fact(second_root, fields={"transaction_id": "20", "income_type": "FUNDING_FEE", "amount": "-.4", "asset": "USDT"})
    with pytest.raises(SarError, match="conflicting"):
        costs(tmp_path, [one, two])


def test_fill_and_income_overlap_cannot_double_count_realized_pnl(tmp_path):
    owner = fact(tmp_path, "order_ownership", native_id="10", fields={"order_id": "10", "ownership_basis": "native_bot_order_receipt"})
    owner_path = append_fact(owner, live_root=tmp_path, now=T0 + timedelta(seconds=125))
    fill = fact(tmp_path, "trade_fill", native_id="30", fields={"order_id": "10", "trade_id": "30", "side": "BUY",
        "price": "100", "quantity": "1", "realized_pnl_usdt": "1", "commission": ".1", "commission_asset": "USDT"},
        ownership={"path": str(owner_path), "sha256": file_hash(owner_path)})
    income = fact(tmp_path, fields={"transaction_id": "20", "income_type": "REALIZED_PNL", "amount": "1", "asset": "USDT"})
    with pytest.raises(SarError, match="overlap"):
        costs(tmp_path, [fill, income])
    result = costs(tmp_path, [fill, fill])
    assert len(result) == 2 and {cost.kind for cost in result} == {"realized", "commission"}


def test_partial_ledger_outcome_is_stored_but_never_a_reward_label(tmp_path):
    row, obs = observation(tmp_path)
    dec = native_decision(row, [obs], live_root=tmp_path)
    end_row, end_obs = observation(tmp_path, 3660)
    event = ledger_outcome(end_row, [end_obs], dec, facts=[fact(tmp_path)], account_scope="account_A",
                           live_root=tmp_path, ingested_at=T0 + timedelta(seconds=3660))
    outcome = Outcome.model_validate(event.payload)
    assert outcome.event_completeness == "unknown" and not outcome.intervention_coverage_complete
    projected = project_transition([dec, event])
    assert projected.behavior is None and projected.net_usdt is None and not projected.training_eligible


def adjustment(root):
    row, obs = observation(root, verdict="ADJUST")
    intent = source(root / "intent.json", {"symbol": BOT.symbol, "strategy_id": BOT.strategy_id, "action": "ADJUST",
        "decision_ts": (T0 + timedelta(seconds=60)).isoformat(), "idempotency_key": "1" * 64,
        "suggested_grid_lower": 95, "suggested_grid_upper": 115})
    decision = native_decision(row, [obs], live_root=root, intent=intent)
    approval = source(root / "approval.json", {"schema_version": ACTION_APPROVAL_SCHEMA, "symbol": BOT.symbol,
        "strategy_id": BOT.strategy_id, "action": "ADJUST", "idempotency_key": "1" * 64,
        "preserve_current_position": True, "suggested_grid_lower": 95, "suggested_grid_upper": 115,
        "approved_at_utc": (T0 + timedelta(seconds=65)).isoformat(),
        "expires_at_utc": (T0 + timedelta(seconds=180)).isoformat()})
    approved = native_approval(approval, decision, intent=intent)
    pre, pre_obs = observation(root, 70, verdict="ADJUST")
    post, post_obs = observation(root, 90, verdict="ADJUST", lower=95, upper=115)
    raw = {"schema_version": ACTION_EXECUTION_SCHEMA, "status": "executed", "symbol": BOT.symbol,
        "strategy_id": BOT.strategy_id, "action": "ADJUST", "idempotency_key": "1" * 64,
        "decision_ts": (T0 + timedelta(seconds=60)).isoformat(), "submitted_at_utc": (T0 + timedelta(seconds=80)).isoformat(),
        "verified_at_utc": (T0 + timedelta(seconds=95)).isoformat(), "preserve_current_position": True,
        "additional_investment": 0, "num_grids": 12, "price_decimals": 1,
        "requested_grid_lower": 95, "requested_grid_upper": 115, "executed_grid_lower": 95, "executed_grid_upper": 115,
        "pre_state": {"symbol": BOT.symbol, "strategy_id": BOT.strategy_id, "status": "Working", "lower": 90, "upper": 110, "num_grids": 12},
        "post_state": {"symbol": BOT.symbol, "strategy_id": BOT.strategy_id, "status": "Working", "lower": 95, "upper": 115, "num_grids": 12}}
    tick = source(root / "tick.json", {"requested_at_utc": (T0 + timedelta(seconds=61)).isoformat(),
        "received_at_utc": (T0 + timedelta(seconds=62)).isoformat(), "exchange_info": {"symbols": [{"symbol": BOT.symbol,
            "pricePrecision": 8, "filters": [{"filterType": "PRICE_FILTER", "tickSize": ".1"}]}]}})
    return raw, decision, approved, dict(pre_record=pre, post_record=post, pre_observations=[pre_obs], post_observations=[post_obs], live_root=root, exchange_info=tick)


def test_native_approval_execution_converts_with_original_exact_states_and_tick(tmp_path):
    raw, dec, approval, kwargs = adjustment(tmp_path)
    event = native_execution(source(tmp_path / "receipt.json", raw), dec, approval, **kwargs)
    assert event.kind == "execution" and event.payload["quantum"] == "0.1"
    assert not project_transition([dec, approval, event]).training_eligible


def test_native_intent_mutation_invalidates_later_sar_replay(tmp_path):
    _, decision, approval, _ = adjustment(tmp_path)
    (tmp_path / "intent.json").write_text("{}")
    with pytest.raises(SarError, match="source missing or changed"):
        project_transition([decision, approval])


@pytest.mark.parametrize("field,value", [("submitted_at_utc", None), ("verified_at_utc", None),
    ("decision_ts", T0.isoformat()), ("idempotency_key", "2" * 64), ("price_decimals", 2),
    ("preserve_current_position", False), ("status", "failed")])
def test_legacy_missing_or_mismatched_receipts_are_rejected(tmp_path, field, value):
    raw, dec, approval, kwargs = adjustment(tmp_path)
    raw[field] = value
    with pytest.raises((SarError, ValueError)):
        native_execution(source(tmp_path / "receipt.json", raw), dec, approval, **kwargs)


def test_future_tick_cannot_be_backfilled_as_historical_execution_precision(tmp_path):
    raw, dec, approval, kwargs = adjustment(tmp_path)
    tick_record = kwargs["exchange_info"]
    tick = tick_record.read()
    tick["received_at_utc"] = (T0 + timedelta(days=1)).isoformat()
    kwargs["exchange_info"] = source(Path(tick_record.source.path), tick)
    with pytest.raises(SarError, match="tick evidence"):
        native_execution(source(tmp_path / "receipt.json", raw), dec, approval, **kwargs)


def test_cli_report_only_and_capture_are_explicit_and_idempotent(tmp_path, monkeypatch):
    from scripts import capture_native_sar as cli
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    row, _ = observation(tmp_path / "Live")
    args = ["--decision-source", row.source.path, "--output-dir", str(tmp_path / "reports" / "read")]
    assert cli.main(args) == 0
    assert not list((tmp_path / "Live").glob("*/BTCUSDT/sar/*/*.json"))
    report = json.loads((tmp_path / "reports" / "read" / "report.json").read_text())
    assert report["observed_action"] is None and report["net_reward_usdt"] is None
    for name in ("write1", "write2"):
        assert cli.main(["--decision-source", row.source.path, "--output-dir", str(tmp_path / "reports" / name), "--capture"]) == 0
    assert len(list((tmp_path / "Live").glob("*/BTCUSDT/sar/*/*.json"))) == 1


def test_cli_missing_receipt_prerequisites_cannot_partially_publish(tmp_path, monkeypatch):
    from scripts import capture_native_sar as cli
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    row, _ = observation(tmp_path / "Live")
    assert cli.main(["--decision-source", row.source.path, "--receipt-source", row.source.path,
        "--output-dir", str(tmp_path / "reports" / "failed"), "--capture"]) == 2
    assert not list((tmp_path / "Live").glob("*/BTCUSDT/sar/*/*.json"))


def test_cli_partial_outcome_does_not_invent_continue_or_reward(tmp_path, monkeypatch):
    from scripts import capture_native_sar as cli
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    row, _ = observation(tmp_path / "Live")
    end, _ = observation(tmp_path / "Live", 3660)
    output = tmp_path / "reports" / "outcome"
    assert cli.main(["--decision-source", row.source.path, "--outcome-source", end.source.path,
        "--account-scope", "account_A", "--output-dir", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["events"] == 2 and report["observed_action"] is None and report["net_reward_usdt"] is None
    assert not report["projection_training_eligible"] and not report["influence_enabled"]


@pytest.mark.parametrize("changed_source", ["response", "mapping", "original"])
def test_authenticated_producer_originals_remain_bound_after_sar_conversion(tmp_path, changed_source):
    from neutralgrid.live.decision.authenticated_evidence import admit_fills, collect_history, reviewed_ownership
    from neutralgrid.live.decision.exact_bot_ledger import audit_ledger
    live = tmp_path / "Live"
    original = source(tmp_path / "original.json", {"symbol": BOT.symbol, "strategyId": BOT.strategy_id,
        "orderId": 10, "bot": BOT.model_dump(mode="json"), "account_scope": "account_A",
        "time": int(T0.timestamp() * 1000), "available_at": (T0 + timedelta(seconds=1)).isoformat()})
    mapping = source(tmp_path / "mapping.json", {"schema_version": "neutralgrid_reviewed_native_ownership_v1",
        "bot": BOT.model_dump(mode="json"), "account_scope": "account_A", "reviewer": "synthetic_test",
        "reviewed_at": (T0 + timedelta(seconds=60)).isoformat(), "orders": [{"source": original.source.model_dump()}]})
    now = T0 + timedelta(seconds=200)
    owners = reviewed_ownership(source=mapping.source, bot=BOT, account_scope="account_A", live_root=live, now=now)

    async def fetch(lower, upper):
        return [{"symbol": BOT.symbol, "orderId": 10, "id": 20, "time": int((T0 + timedelta(seconds=120)).timestamp() * 1000),
            "price": "100", "qty": "1", "realizedPnl": "2", "commission": ".1", "commissionAsset": "USDT",
            "marginAsset": "USDT", "side": "BUY"}]

    history = asyncio.run(collect_history(fetch=fetch, endpoint="/fapi/v1/userTrades", bot=BOT,
        account_scope="account_A", start=T0, end=now, live_root=live, now=lambda: now))
    admit_fills(history, bot=BOT, account_scope="account_A", ownership_paths=owners, live_root=live, now=now)
    facts = [PrivateFact.model_validate(value) for value in audit_ledger(live)["facts"]]
    assert any(f.kind == "trade_fill" for f in facts)
    row, obs = observation(live)
    decision = native_decision(row, [obs], live_root=live)
    end, end_obs = observation(live, 3660)
    outcome = ledger_outcome(end, [end_obs], decision, facts=facts, account_scope="account_A",
                             live_root=live, ingested_at=T0 + timedelta(seconds=3660))
    target = {"response": Path(history["sources"][0]["path"]), "mapping": Path(mapping.source.path),
              "original": Path(original.source.path)}[changed_source]
    target.write_text("{}")
    with pytest.raises(SarError, match="source missing or changed"):
        project_transition([decision, outcome])
