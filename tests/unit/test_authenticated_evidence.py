from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from neutralgrid.live.decision.authenticated_evidence import (
    admit_fills, capture_pnl_path, collect_history, credential_scope, integer_id, reviewed_ownership, source_ref,
)
from neutralgrid.live.decision.exact_bot_ledger import audit_ledger
from neutralgrid.live.decision.state_action_reward import BotIdentity

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
NOW = T0 + timedelta(hours=3)
BOT = BotIdentity(symbol="BTCUSDT", strategy_id="123", deployed_at=T0)
SCOPE = credential_scope("fixture-key-not-a-real-key")
MS = int(T0.timestamp() * 1000)


def trade(**changes):
    return {"symbol": BOT.symbol, "orderId": 10, "id": 20, "time": MS + 1000,
            "price": "100", "qty": "1", "realizedPnl": "2", "commission": "0.1",
            "commissionAsset": "USDT", "marginAsset": "USDT", "side": "BUY"} | changes


def history(tmp_path, fetch, **changes):
    args = dict(fetch=fetch, endpoint="/fapi/v1/userTrades", bot=BOT, account_scope=SCOPE,
                start=T0, end=T0 + timedelta(seconds=2), live_root=tmp_path / "Live", now=lambda: NOW)
    return asyncio.run(collect_history(**(args | changes)))


def write(path: Path, body):
    path.write_text(json.dumps(body), encoding="utf-8")
    return source_ref(path)


def ownership(tmp_path, **changes):
    original = {"symbol": BOT.symbol, "strategyId": BOT.strategy_id, "orderId": 10,
                "bot": BOT.model_dump(mode="json"), "account_scope": SCOPE,
                "time": MS, "available_at": (T0 + timedelta(seconds=1)).isoformat()} | changes
    original_ref = write(tmp_path / "original.json", original)
    mapping = {"schema_version": "neutralgrid_reviewed_native_ownership_v1", "bot": BOT.model_dump(mode="json"),
               "account_scope": SCOPE, "reviewer": "test-original-review", "reviewed_at": (T0 + timedelta(minutes=1)).isoformat(),
               "orders": [{"source": original_ref.model_dump()}]}
    return reviewed_ownership(source=write(tmp_path / "mapping.json", mapping), bot=BOT,
                              account_scope=SCOPE, live_root=tmp_path / "Live", now=NOW)


@pytest.mark.parametrize("value", [True, False, 1.2, 1.0, "1.0", "01", "-1", "", None])
def test_identifiers_never_truncated(value):
    with pytest.raises(ValueError):
        integer_id(value)


def test_raw_unlinked_capture_does_not_manufacture_bot_facts(tmp_path):
    async def fetch(a, b):
        return [trade()]
    result = history(tmp_path, fetch)
    admission = admit_fills(result, bot=BOT, account_scope=SCOPE, ownership_paths=[], live_root=tmp_path / "Live", now=NOW)
    assert result["retrieval_exhausted"] and result["event_completeness"] == "unproven"
    assert admission["unlinked_rows"] == 1 and not admission["fact_paths"]
    assert not audit_ledger(tmp_path / "Live")["facts"]


def test_authenticated_fill_ownership_accounting_replay_and_source_mutation(tmp_path):
    owners = ownership(tmp_path)
    async def fetch(a, b):
        return [trade(), trade()]
    result = history(tmp_path, fetch)
    assert len(result["rows"]) == 1
    first = admit_fills(result, bot=BOT, account_scope=SCOPE, ownership_paths=owners, live_root=tmp_path / "Live", now=NOW)
    second = admit_fills(result, bot=BOT, account_scope=SCOPE, ownership_paths=owners, live_root=tmp_path / "Live", now=NOW)
    assert first == second
    audited = audit_ledger(tmp_path / "Live")
    assert audited["source_bound_facts"] == 2 and not audited["rejected"]
    fill = next(f for f in audited["facts"] if f["kind"] == "trade_fill")
    assert fill["fields"]["realized_pnl_usdt"] == "2" and fill["fields"]["commission"] == "0.1"
    (tmp_path / "original.json").write_text("{}")
    audited = audit_ledger(tmp_path / "Live")
    assert audited["source_bound_facts"] == 0 and audited["rejected"]


def test_order_trade_update_si_is_not_ownership(tmp_path):
    with pytest.raises(ValueError):
        ownership(tmp_path, strategyId=None, si=BOT.strategy_id)
    assert not audit_ledger(tmp_path / "Live")["facts"]


@pytest.mark.parametrize("field", ["realizedPnl", "commission", "commissionAsset"])
def test_missing_accounting_is_not_zero_imputed(tmp_path, field):
    owners = ownership(tmp_path)
    raw = trade()
    del raw[field]
    async def fetch(a, b):
        return [raw]
    result = history(tmp_path, fetch)
    with pytest.raises((KeyError, ValueError)):
        admit_fills(result, bot=BOT, account_scope=SCOPE, ownership_paths=owners, live_root=tmp_path / "Live", now=NOW)
    assert audit_ledger(tmp_path / "Live")["fact_classes"] == {"order_ownership": 1}


def test_conflicting_trade_identity_quarantines_entire_response(tmp_path):
    async def fetch(a, b):
        return [trade(), trade(price="101")]
    result = history(tmp_path, fetch)
    assert not result["rows"] and result["rejected"] == ["conflicting_native_identity"]


@pytest.mark.parametrize("asset", [None, "USDC", "BNFCR"])
def test_realized_pnl_denomination_cannot_be_inferred(tmp_path, asset):
    owners = ownership(tmp_path)
    async def fetch(a, b):
        return [trade(marginAsset=asset)]
    result = history(tmp_path, fetch)
    with pytest.raises(ValueError, match="settlement denomination"):
        admit_fills(result, bot=BOT, account_scope=SCOPE, ownership_paths=owners,
                    live_root=tmp_path / "Live", now=NOW)
    assert audit_ledger(tmp_path / "Live")["fact_classes"] == {"order_ownership": 1}


def test_saturated_millisecond_never_claims_complete(tmp_path):
    async def fetch(a, b):
        return [trade(time=a)]
    result = history(tmp_path, fetch, end=T0, limit=1)
    assert result["gaps"][0]["reason"] == "saturated_millisecond"
    assert not result["retrieval_exhausted"]


def test_saturated_windows_partition_to_millisecond_and_preserve_raw(tmp_path):
    calls = []
    async def fetch(a, b):
        calls.append((a, b))
        return [trade(time=a, id=a)] if a != b else []
    result = history(tmp_path, fetch, end=T0 + timedelta(milliseconds=3), limit=1)
    assert len(calls) == 7 and len(result["sources"]) == 7
    # Rows first seen in a parent response remain accounted, despite later
    # empty leaves. Retrieval is not promoted into interval completeness.
    assert result["event_completeness"] == "unproven"


def test_budget_exhaustion_exposes_unfetched_intervals(tmp_path):
    async def fetch(a, b):
        return [trade(time=a)]
    result = history(tmp_path, fetch, limit=1, max_requests=1)
    assert len(result["gaps"]) == 2 and not result["retrieval_exhausted"]


def test_transport_failure_sanitizes_exception_and_stops(tmp_path):
    calls = []
    async def fetch(a, b):
        calls.append(1)
        raise RuntimeError("signature=secret&apiKey=secret")
    result = history(tmp_path, fetch)
    assert len(calls) == 1 and "secret" not in json.dumps(result)
    assert result["gaps"][0]["reason"] == "transport_failure"


def test_income_identity_includes_type_without_inventing_attribution(tmp_path):
    async def fetch(a, b):
        return [{"symbol": BOT.symbol, "tranId": 1, "time": MS, "incomeType": kind,
                 "income": "1", "asset": "USDT"} for kind in ("COMMISSION", "REALIZED_PNL")]
    result = history(tmp_path, fetch, endpoint="/fapi/v1/income")
    assert len(result["rows"]) == 2 and result["ownership"] == "unproven"


@pytest.mark.parametrize("changes", [{"symbol": "ETHUSDT"}, {"time": MS - 1}, {"id": 1.5}])
def test_bad_exchange_identity_or_time_quarantines(tmp_path, changes):
    async def fetch(a, b):
        return [trade(**changes)]
    result = history(tmp_path, fetch)
    assert result["rejected"] and not result["rows"]


def test_native_response_mutation_quarantines_fill(tmp_path):
    owners = ownership(tmp_path)
    async def fetch(a, b):
        return [trade()]
    result = history(tmp_path, fetch)
    admit_fills(result, bot=BOT, account_scope=SCOPE, ownership_paths=owners, live_root=tmp_path / "Live", now=NOW)
    Path(result["rows"][0]["source"]["path"]).write_text("{}")
    assert audit_ledger(tmp_path / "Live")["source_bound_facts"] == 0


def test_later_refetch_retains_original_fact_availability_and_source(tmp_path):
    owners = ownership(tmp_path)
    async def fetch(a, b):
        return [trade()]
    first = history(tmp_path, fetch)
    one = admit_fills(first, bot=BOT, account_scope=SCOPE, ownership_paths=owners, live_root=tmp_path / "Live", now=NOW)
    before = Path(one["fact_paths"][0]).read_bytes()
    later = history(tmp_path, fetch, now=lambda: NOW + timedelta(minutes=1))
    two = admit_fills(later, bot=BOT, account_scope=SCOPE, ownership_paths=owners,
                      live_root=tmp_path / "Live", now=NOW + timedelta(minutes=1))
    assert one == two and Path(two["fact_paths"][0]).read_bytes() == before
    assert first["sources"] != later["sources"]


def test_cross_capture_conflict_preflight_commits_no_fills(tmp_path):
    owners = ownership(tmp_path)
    async def first(a, b):
        return [trade()]
    async def conflicting(a, b):
        return [trade(price="101")]
    one, two = history(tmp_path, first), history(tmp_path, conflicting)
    one["rows"].extend(two["rows"])
    with pytest.raises(ValueError, match="admission batch"):
        admit_fills(one, bot=BOT, account_scope=SCOPE, ownership_paths=owners, live_root=tmp_path / "Live", now=NOW)
    assert audit_ledger(tmp_path / "Live")["fact_classes"] == {"order_ownership": 1}


def test_damaged_ownership_envelope_rejected_before_producer_write(tmp_path):
    owners = ownership(tmp_path)
    envelope = json.loads(owners[0].read_text())
    envelope["record_sha256"] = "0" * 64
    owners[0].write_text(json.dumps(envelope))
    async def fetch(a, b):
        return [trade()]
    result = history(tmp_path, fetch)
    with pytest.raises(ValueError, match="intact committed"):
        admit_fills(result, bot=BOT, account_scope=SCOPE, ownership_paths=owners, live_root=tmp_path / "Live", now=NOW)
    assert len(list((tmp_path / "Live").glob("*/BTCUSDT/authenticated_evidence/*/producer_records/*.json"))) == 1


def test_same_symbol_other_deployment_is_not_owner(tmp_path):
    with pytest.raises(ValueError):
        ownership(tmp_path, bot=BotIdentity(symbol=BOT.symbol, strategy_id="456", deployed_at=T0).model_dump(mode="json"))


def test_reviewed_mapping_replay_across_dates_retains_committed_fact(tmp_path):
    first = ownership(tmp_path)
    mapping_ref = source_ref(tmp_path / "mapping.json")
    second = reviewed_ownership(source=mapping_ref, bot=BOT, account_scope=SCOPE,
                                live_root=tmp_path / "Live", now=NOW + timedelta(days=1))
    assert first == second
    assert audit_ledger(tmp_path / "Live")["source_bound_facts"] == 1


def test_pnl_path_replays_originals_and_exports_inert_forecast(tmp_path):
    from test_live_evidence_cohort import native_observation
    from neutralgrid.live.decision.forecast_contract import forecast_from_reported_pnl_path
    path, _ = native_observation(tmp_path / "Live")
    result = capture_pnl_path([path, path], bot=BOT, live_root=tmp_path / "Live", now=NOW)
    raw = json.loads(result.read_text())
    assert len(raw["points"]) == 1 and raw["path_coverage"] == "path_unobserved_between_snapshots"
    contract = forecast_from_reported_pnl_path(result, live_root=tmp_path / "Live")
    assert contract.label.value is None and contract.prior_target_state == "unknown"
    assert not contract.training_eligible and not contract.influence_enabled


@pytest.mark.parametrize("seconds", [-1, 20000])
def test_pnl_availability_before_capture_or_after_now_rejected(tmp_path, seconds):
    from test_live_evidence_cohort import native_observation
    from neutralgrid.live.decision.pnl_history import _record_integrity_payload, _sha256_payload
    path, row = native_observation(tmp_path / "Live")
    row["evaluated_at_utc"] = (T0 + timedelta(minutes=1, seconds=seconds)).isoformat()
    row["record_integrity_sha256"] = _sha256_payload(_record_integrity_payload(row))
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="causal availability"):
        capture_pnl_path([path], bot=BOT, live_root=tmp_path / "Live", now=NOW)


def test_offline_cli_does_not_construct_network_client(tmp_path, monkeypatch):
    import argparse
    from scripts import capture_authenticated_bot_evidence as cli
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    def forbidden():
        raise AssertionError("offline command initialized network client")
    monkeypatch.setattr(cli, "BinanceClient", forbidden)
    args = argparse.Namespace(output_dir=tmp_path / "reports" / "offline", registration=[],
                              all_registered=False, capture_pnl=False, fetch=False, ownership_mapping=[])
    report = asyncio.run(cli.capture(args))
    assert not report["errors"] and (args.output_dir / "report.json").exists()
