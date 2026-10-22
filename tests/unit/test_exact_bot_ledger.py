"""Exact-bot ledger invariants: registration, ownership, provenance and accounting."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from neutralgrid.live.decision.exact_bot_ledger import (
    CapitalProtectionPolicy, PrivateFact, append_fact, audit_ledger, register_bot, verify_fact,
)
from neutralgrid.live.decision.state_action_reward import BotIdentity, SourceRef, file_hash

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
BOT = BotIdentity(symbol="BTCUSDT", strategy_id="123", deployed_at=T0)


def sourced_fact(tmp_path: Path, *, kind="order_ownership", bot=BOT, fields=None,
                 account="account_A", native_id="10", ownership=None) -> PrivateFact:
    path = tmp_path / f"{kind}_{native_id}_{bot.strategy_id}_{account}.json"
    if fields is None:
        fields = {"order_id": "10", "ownership_basis": "native_bot_order_receipt"}
    fact = PrivateFact(bot=bot, account_scope=account, kind=kind, native_id=native_id,
                       occurred_at=T0 + timedelta(minutes=1), available_at=T0 + timedelta(minutes=2),
                       fields=fields, source=SourceRef(path=str(path), sha256="0" * 64),
                       source_pointer="/records/0", ownership_fact=ownership)
    body = {k: v for k, v in fact.model_dump(mode="json").items() if k not in {"source", "source_pointer", "ownership_fact", "schema_version"}}
    path.write_text(json.dumps({"records": [body]}), encoding="utf-8")
    return PrivateFact.model_validate(fact.model_dump(mode="json") | {"source": {"path": str(path), "sha256": file_hash(path)}})


def test_capital_policy_preserves_investment_and_does_not_infer_liquidation_buffer():
    policy = CapitalProtectionPolicy()
    assert policy.reference == "invested_capital" and policy.maximum_capital_loss_pct == 0
    assert policy.minimum_liquidation_buffer_pct is None and policy.influence_enabled is False
    with pytest.raises(ValueError):
        CapitalProtectionPolicy(minimum_liquidation_buffer_pct=1)


def test_exact_fact_is_idempotent_across_ingestion_dates(tmp_path):
    fact = sourced_fact(tmp_path)
    first = append_fact(fact, live_root=tmp_path / "Live", now=T0 + timedelta(days=1))
    second = append_fact(fact, live_root=tmp_path / "Live", now=T0 + timedelta(days=2))
    assert first == second
    report = audit_ledger(tmp_path / "Live")
    assert report["source_bound_facts"] == 1 and not report["rejected"]
    assert report["training_eligible"] is False


def test_same_order_cannot_be_owned_by_two_deployments(tmp_path):
    first = sourced_fact(tmp_path)
    second = sourced_fact(tmp_path, bot=BotIdentity(symbol="BTCUSDT", strategy_id="456", deployed_at=T0))
    append_fact(first, live_root=tmp_path / "Live", now=T0 + timedelta(days=1))
    with pytest.raises(ValueError, match="cross-bot ownership"):
        append_fact(second, live_root=tmp_path / "Live", now=T0 + timedelta(days=1))


def test_exchange_ids_are_scoped_to_account(tmp_path):
    for account in ("account_A", "account_B"):
        append_fact(sourced_fact(tmp_path, account=account), live_root=tmp_path / "Live", now=T0 + timedelta(days=1))
    assert audit_ledger(tmp_path / "Live")["source_bound_facts"] == 2


def test_unrelated_hashed_source_is_not_fact_proof(tmp_path):
    fact = sourced_fact(tmp_path)
    source = Path(fact.source.path)
    raw = json.loads(source.read_text())
    raw["records"][0]["fields"]["order_id"] = "999"
    source.write_text(json.dumps(raw))
    changed = PrivateFact.model_validate(fact.model_dump(mode="json") | {"source": {"path": str(source), "sha256": file_hash(source)}})
    with pytest.raises(ValueError, match="does not prove"):
        append_fact(changed, live_root=tmp_path / "Live", now=T0 + timedelta(days=1))
    assert not list((tmp_path / "Live").glob("*/BTCUSDT/exact_bot_ledger/*/facts/*.json"))


@pytest.mark.parametrize("order", [True, 10.9, "10.9", "-1", "", None])
def test_noninteger_order_ids_are_not_truncated(tmp_path, order):
    with pytest.raises(ValueError):
        sourced_fact(tmp_path, fields={"order_id": order, "ownership_basis": "native_bot_order_receipt"})


def trade_fields(**changes):
    return {"order_id": "10", "trade_id": "20", "side": "BUY", "price": "100", "quantity": "1",
            "realized_pnl_usdt": "0", "commission": "0.1", "commission_asset": "USDT"} | changes


def test_fills_require_committed_exact_order_ownership(tmp_path):
    owner = sourced_fact(tmp_path)
    path = append_fact(owner, live_root=tmp_path / "Live", now=T0 + timedelta(days=1))
    link = SourceRef(path=str(path), sha256=file_hash(path))
    trade = sourced_fact(tmp_path, kind="trade_fill", native_id="20", fields=trade_fields(), ownership=link)
    append_fact(trade, live_root=tmp_path / "Live", now=T0 + timedelta(days=1))
    assert audit_ledger(tmp_path / "Live")["fact_classes"] == {"order_ownership": 1, "trade_fill": 1}
    wrong = sourced_fact(tmp_path, kind="trade_fill", native_id="21", fields=trade_fields(trade_id="21", order_id="99"), ownership=link)
    with pytest.raises(ValueError, match="ownership/deployment mismatch"):
        append_fact(wrong, live_root=tmp_path / "Live", now=T0 + timedelta(days=1))


def test_missing_fees_or_realized_profit_are_not_filled_with_zero(tmp_path):
    for absent in ("commission", "realized_pnl_usdt"):
        fields = trade_fields()
        del fields[absent]
        with pytest.raises(ValueError, match="exactly match"):
            sourced_fact(tmp_path, kind="trade_fill", native_id="20", fields=fields)


def test_misrouted_ownership_copy_is_not_a_committed_exact_bot_reference(tmp_path):
    owner = sourced_fact(tmp_path)
    path = append_fact(owner, live_root=tmp_path / "Live", now=T0 + timedelta(days=1))
    copied = path.parents[2] / "wrong_deployment" / "facts" / path.name
    copied.parent.mkdir(parents=True)
    copied.write_bytes(path.read_bytes())
    trade = sourced_fact(tmp_path, kind="trade_fill", native_id="20", fields=trade_fields(),
                         ownership=SourceRef(path=str(copied), sha256=file_hash(copied)))
    with pytest.raises(ValueError, match="canonical deployment path"):
        append_fact(trade, live_root=tmp_path / "Live", now=T0 + timedelta(days=1))


@pytest.mark.parametrize("equity,capital,expected", [(101, 100, True), (99, 100, False), (100, 100, True)])
def test_equity_checks_capital_floor_without_claiming_future_safety(tmp_path, equity, capital, expected):
    fact = sourced_fact(tmp_path, kind="equity", native_id="equity-1", fields={
        "equity_usdt": equity, "invested_capital_usdt": capital, "unrealized_usdt": 0, "peak_equity_usdt": 110})
    append_fact(fact, live_root=tmp_path / "Live", now=T0 + timedelta(days=1))
    check = audit_ledger(tmp_path / "Live")["capital_preservation_checks"][0]
    assert check["preserves_reported_invested_capital_at_observation"] is expected
    assert check["guarantees_future_capital_preservation"] is False


def test_boolean_finance_future_facts_and_fake_continue_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="boolean"):
        sourced_fact(tmp_path, kind="equity", fields={"equity_usdt": True, "invested_capital_usdt": 100,
                                                    "unrealized_usdt": 0, "peak_equity_usdt": None})
    with pytest.raises(ValueError, match="future"):
        append_fact(sourced_fact(tmp_path), live_root=tmp_path / "Live", now=T0)
    with pytest.raises(ValueError, match="never invented CONTINUE"):
        sourced_fact(tmp_path, kind="intervention", fields={"action": "CONTINUE", "receipt": {}})


def test_coverage_claim_remains_unproven_and_gaps_cannot_claim_complete(tmp_path):
    fields = {"start_at": T0.isoformat(), "end_at": (T0 + timedelta(minutes=1)).isoformat(),
              "scopes": ["trades"], "gaps": [], "producer_claims_complete": True}
    fact = sourced_fact(tmp_path, kind="coverage", native_id="window-1", fields=fields)
    path = append_fact(fact, live_root=tmp_path / "Live", now=T0 + timedelta(days=1))
    assert json.loads(path.read_text())["event_completeness"] == "unproven"
    fields["gaps"] = [{"start_at": T0.isoformat(), "end_at": T0.isoformat()}]
    with pytest.raises(ValueError, match="gapped coverage"):
        sourced_fact(tmp_path, kind="coverage", native_id="window-2", fields=fields)


def test_funding_and_realized_income_are_kept_separate_without_double_counting(tmp_path):
    for kind in ("FUNDING_FEE", "REALIZED_PNL"):
        fact = sourced_fact(tmp_path, kind="income", native_id="42", fields={"transaction_id": "42",
            "income_type": kind, "amount": "-0.1", "asset": "USDT"})
        # Preserve distinct original files even when exchange transaction IDs overlap.
        source = Path(fact.source.path)
        dedicated = source.with_name(kind + ".json")
        dedicated.write_bytes(source.read_bytes())
        fact = PrivateFact.model_validate(fact.model_dump(mode="json") | {"source": {"path": str(dedicated), "sha256": file_hash(dedicated)}})
        append_fact(fact, live_root=tmp_path / "Live", now=T0 + timedelta(days=1))
    audit = audit_ledger(tmp_path / "Live")
    assert audit["fact_classes"] == {"income": 2}
    assert audit["training_eligible"] is False


def test_order_cancellation_requires_the_same_exact_ownership_as_fills(tmp_path):
    owner = sourced_fact(tmp_path)
    path = append_fact(owner, live_root=tmp_path / "Live", now=T0 + timedelta(days=1))
    update = sourced_fact(tmp_path, kind="order_update", native_id="update-1", fields={
        "order_id": "10", "status": "CANCELED", "executed_quantity": "0"},
        ownership=SourceRef(path=str(path), sha256=file_hash(path)))
    append_fact(update, live_root=tmp_path / "Live", now=T0 + timedelta(days=1))
    assert audit_ledger(tmp_path / "Live")["fact_classes"]["order_update"] == 1


def test_changed_original_quarantines_linked_fills(tmp_path):
    owner = sourced_fact(tmp_path)
    path = append_fact(owner, live_root=tmp_path / "Live", now=T0 + timedelta(days=1))
    trade = sourced_fact(tmp_path, kind="trade_fill", native_id="20", fields=trade_fields(),
                         ownership=SourceRef(path=str(path), sha256=file_hash(path)))
    append_fact(trade, live_root=tmp_path / "Live", now=T0 + timedelta(days=1))
    Path(owner.source.path).write_text("{}")
    audit = audit_ledger(tmp_path / "Live")
    assert audit["source_bound_facts"] == 0 and len(audit["rejected"]) == 2


def test_concurrent_publication_cannot_duplicate_facts(tmp_path):
    fact = sourced_fact(tmp_path)
    def write(_):
        try:
            return str(append_fact(fact, live_root=tmp_path / "Live", now=T0 + timedelta(days=1)))
        except RuntimeError as exc:
            assert "busy" in str(exc)
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(write, range(8)))
    assert len({p for p in results if p is not None}) == 1
    assert audit_ledger(tmp_path / "Live")["source_bound_facts"] == 1


def test_registration_uses_native_identity_and_does_not_fabricate_accounting(tmp_path):
    from test_live_evidence_cohort import native_observation
    from neutralgrid.live.decision.evidence_cohort import collect_observations, evidence_packets, publish_packet
    native_observation(tmp_path / "Live")
    observations, _, _ = collect_observations(tmp_path / "Live")
    packet = evidence_packets(observations, transitions=[])[0]
    path = publish_packet(packet, live_root=tmp_path / "Live", now=T0)
    source = SourceRef(path=str(path), sha256=file_hash(path))
    first = register_bot(BOT, source=source, live_root=tmp_path / "Live", now=T0)
    assert register_bot(BOT, source=source, live_root=tmp_path / "Live", now=T0 + timedelta(days=1)) == first
    registered = json.loads(first.read_text())
    assert registered["account_scope"] is None and registered["protected_capital_usdt"] is None
    assert registered["capital_policy"]["reference"] == "invested_capital"
    assert audit_ledger(tmp_path / "Live")["source_bound_facts"] == 0
