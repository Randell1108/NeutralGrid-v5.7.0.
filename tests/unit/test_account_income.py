from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path

import pytest

from neutralgrid.live.decision.account_income import collect_account_income, fetch_income, publish_account_income
from neutralgrid.live.decision.authenticated_evidence import credential_scope
from neutralgrid.live.decision.state_action_reward import fingerprint

START = datetime(2026, 10, 1, tzinfo=timezone.utc)
NOW = START + timedelta(days=1)
MS = int(START.timestamp() * 1000)
SCOPE = credential_scope("synthetic-test-key")


def row(**changes):
    return {"symbol": "", "incomeType": "STRATEGY_UMFUTURES_TRANSFER", "tranId": 12,
            "time": MS, "income": "-5.5", "asset": "USDT", "tradeId": "", "info": ""} | changes


def collect(tmp_path, fetch, **changes):
    args = dict(fetch=fetch, account_scope=SCOPE, storage_symbol="BTCUSDT", start=START,
                end=START + timedelta(seconds=1), live_root=tmp_path / "Live", now=lambda: NOW)
    return asyncio.run(collect_account_income(**(args | changes)))


def records(result):
    return [record for ref in result["record_shards"] for record in json.loads(Path(ref["path"]).read_text())["records"]]


def test_blank_symbol_transfer_retained_without_bot_attribution(tmp_path):
    async def fetch(a, b, limit):
        return [row(), row(symbol="ETHUSDT", tranId=13, incomeType="FUNDING_FEE")]
    result = collect(tmp_path, fetch)
    assert result["accepted_records"] == 2 and result["blank_symbol_records"] == 1
    assert result["retrieval_exhausted"] and result["event_completeness"] == "unknown"
    assert result["bot_attribution"] == "none" and not result["training_eligible"]
    assert {r["raw"]["symbol"] for r in records(result)} == {"", "ETHUSDT"}
    capture = json.loads(Path(result["sources"][0]["path"]).read_text())
    assert capture["payload"][0] == row()
    assert "symbol" not in capture["parameters"]


def test_authenticated_transport_omits_symbol_and_income_type():
    calls = []
    class Client:
        async def _request(self, endpoint, params, **kwargs):
            calls.append((endpoint, params, kwargs))
            return []
    assert asyncio.run(fetch_income(Client(), MS, MS + 1, 1000)) == []
    assert calls == [("/fapi/v1/income", {"startTime": MS, "endTime": MS + 1, "limit": 1000},
                      {"signed": True, "max_retries": 0})]


def test_identical_duplicates_collapse_but_income_types_are_distinct(tmp_path):
    async def fetch(a, b, limit):
        return [row(), row(), row(incomeType="TRANSFER")]
    result = collect(tmp_path, fetch)
    assert result["accepted_records"] == 2 and not result["rejected"]


def test_conflicting_duplicate_quarantines_entire_capture(tmp_path):
    async def fetch(a, b, limit):
        return [row(), row(income="-9")]
    result = collect(tmp_path, fetch)
    assert result["accepted_records"] == 0 and not result["record_shards"]
    assert "conflicting_income_identity" in result["rejected"]
    assert len(result["sources"]) == 1 and not result["retrieval_exhausted"]


def test_saturated_millisecond_is_explicit_gap(tmp_path):
    async def fetch(a, b, limit):
        return [row()]
    result = collect(tmp_path, fetch, end=START, limit=1)
    assert result["accepted_records"] == 1
    assert result["gaps"][0]["reason"] == "saturated_millisecond"


def test_partition_preserves_full_originals_and_budget(tmp_path):
    calls = []
    async def fetch(a, b, limit):
        calls.append((a, b))
        return [row()] if a <= MS <= b else []
    result = collect(tmp_path, fetch, limit=1, max_requests=2)
    assert len(calls) == 2 and result["requests"] == 2
    assert {g["reason"] for g in result["gaps"]} == {"request_budget"}
    assert result["accepted_records"] == 1 and not result["retrieval_exhausted"]


def test_transport_failure_stops_and_redacts_exception(tmp_path):
    async def fetch(a, b, limit):
        raise RuntimeError("secret-signed-url")
    result = collect(tmp_path, fetch)
    assert result["requests"] == 1 and result["gaps"][0]["reason"] == "transport_failure"
    assert "secret-signed-url" not in json.dumps(result)


@pytest.mark.parametrize("changes", [{"tranId": True}, {"tranId": 1.2}, {"tranId": "01"},
    {"income": "NaN"}, {"income": None}, {"time": MS + 5000}, {"symbol": None}, {"asset": ""}])
def test_malformed_records_never_enter_normalized_shards(tmp_path, changes):
    async def fetch(a, b, limit):
        return [row(**changes)]
    result = collect(tmp_path, fetch)
    assert result["rejected"] and result["accepted_records"] == 0


def test_record_budget_explicitly_quarantines(tmp_path):
    async def fetch(a, b, limit):
        return [row(), row(tranId=13)]
    result = collect(tmp_path, fetch, max_records=1)
    assert result["rejected"] == ["record_budget_exceeded"]


@pytest.mark.parametrize("changes", [{"start": START.replace(tzinfo=None)}, {"end": NOW + timedelta(seconds=1)},
    {"end": START - timedelta(seconds=1)}, {"start": START - timedelta(days=91)},
    {"start": START + timedelta(microseconds=1)}, {"storage_symbol": "../BAD"}, {"max_requests": True}])
def test_invalid_intervals_or_context_fail_before_fetch(tmp_path, changes):
    async def fetch(a, b, limit):
        pytest.fail("invalid input reached transport")
    with pytest.raises(ValueError):
        collect(tmp_path, fetch, **changes)


def test_content_replay_is_idempotent_across_ingestion_dates(tmp_path):
    args = dict(account_scope=SCOPE, storage_symbol="BTCUSDT", live_root=tmp_path, category="manifests")
    first = publish_account_income({"test": "fixture"}, at=NOW, **args)
    second = publish_account_income({"test": "fixture"}, at=NOW + timedelta(days=1), **args)
    assert first == second


def test_refetch_across_storage_context_preserves_one_identity_and_first_availability(tmp_path):
    async def fetch(a, b, limit):
        return [row()]
    first = collect(tmp_path, fetch)
    second = collect(tmp_path, fetch, now=lambda: NOW + timedelta(days=1), storage_symbol="ETHUSDT")
    assert first["new_canonical_identities"] == 1 and second["new_canonical_identities"] == 0
    assert records(first) == records(second)
    assert len(list((tmp_path / "Live").glob("????-??-??/*/account_income/*/identities/*.json"))) == 1


def test_conflict_across_acquisitions_quarantines_new_normalization(tmp_path):
    async def original(a, b, limit):
        return [row()]
    async def changed(a, b, limit):
        return [row(income="99")]
    first = collect(tmp_path, original)
    second = collect(tmp_path, changed)
    assert second["accepted_records"] == 0 and second["rejected"]
    assert records(first)[0]["raw"]["income"] == "-5.5"


def test_committed_original_mutation_rejects_refetch(tmp_path):
    async def fetch(a, b, limit):
        return [row()]
    first = collect(tmp_path, fetch)
    Path(first["sources"][0]["path"]).write_text("{}")
    second = collect(tmp_path, fetch, now=lambda: NOW + timedelta(hours=1))
    assert second["accepted_records"] == 0 and second["rejected"]


@pytest.mark.parametrize("field", ["available_at", "training_eligible", "bot_attribution"])
def test_recomputed_canonical_checksum_cannot_override_original_semantics(tmp_path, field):
    async def fetch(a, b, limit):
        return [row()]
    collect(tmp_path, fetch)
    path = next((tmp_path / "Live").glob("????-??-??/*/account_income/*/identities/*.json"))
    record = json.loads(path.read_text())
    record.pop("record_sha256")
    if field == "available_at":
        record["record"][field] = START.isoformat()
    elif field == "training_eligible":
        record[field] = True
    else:
        record["record"][field] = "inferred_bot"
    record["record_sha256"] = fingerprint(record)
    path.write_text(json.dumps(record))
    result = collect(tmp_path, fetch, now=lambda: NOW + timedelta(hours=1))
    assert result["accepted_records"] == 0 and result["rejected"]


@pytest.mark.parametrize("category,reason", [("records", "record_shard_not_persisted"), ("manifests", "manifest_not_persisted")])
def test_post_response_io_failure_is_reported_without_completeness(tmp_path, monkeypatch, category, reason):
    import neutralgrid.live.decision.account_income as module
    original_publish = module.publish_account_income
    def failing_publish(body, **kwargs):
        if kwargs["category"] == category:
            raise OSError("sensitive-private-disk-path")
        return original_publish(body, **kwargs)
    monkeypatch.setattr(module, "publish_account_income", failing_publish)
    async def fetch(a, b, limit):
        return [row()]
    result = collect(tmp_path, fetch)
    assert reason in result["rejected"] and not result["retrieval_exhausted"]
    assert result["sources"] and "sensitive-private-disk-path" not in json.dumps(result)
    if category == "records":
        assert result["accepted_records"] == 0 and result["canonical_records_validated"] == 1
    else:
        assert result["manifest"] is None


def test_partial_canonical_commit_is_retryable_and_explicit(tmp_path, monkeypatch):
    import neutralgrid.live.decision.account_income as module
    original_create = module.atomic_create_json
    count = 0
    def failing_create(path, body):
        nonlocal count
        if path.parent.name == "identities":
            count += 1
            if count == 2:
                raise OSError("fixture disk fault")
        return original_create(path, body)
    monkeypatch.setattr(module, "atomic_create_json", failing_create)
    async def fetch(a, b, limit):
        return [row(), row(tranId=13)]
    partial = collect(tmp_path, fetch)
    assert partial["canonical_commit_status"] == "partial_or_unknown"
    assert partial["new_canonical_identities"] is None and not partial["retrieval_exhausted"]
    monkeypatch.setattr(module, "atomic_create_json", original_create)
    retried = collect(tmp_path, fetch)
    assert retried["accepted_records"] == 2 and retried["new_canonical_identities"] == 1


def cli():
    path = Path(__file__).resolve().parents[2] / "scripts" / "capture_account_income_evidence.py"
    spec = importlib.util.spec_from_file_location("capture_account_income_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_authentication_failure_writes_safe_report_and_closes(tmp_path, monkeypatch):
    module = cli()
    calls = []
    class Client:
        is_authenticated = False
        base_url = "https://fapi.binance.com"
        async def close(self):
            calls.append("closed")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "BinanceClient", Client)
    monkeypatch.setattr(module, "now", lambda: NOW)
    output = tmp_path / "reports" / "capture"
    code = module.main(["--start-at", START.isoformat(), "--end-at", START.isoformat(),
                        "--storage-symbol", "BTCUSDT", "--output-dir", str(output)])
    report = json.loads((output / "report.json").read_text())
    assert code == 2 and calls == ["closed"] and report["status"] == "partial_or_blocked"
    assert report["submitted_exchange_orders"] == 0


def test_cli_rejects_naive_time_before_network():
    with pytest.raises(SystemExit):
        cli().main(["--start-at", "2026-10-01T00:00:00", "--end-at", START.isoformat(),
                    "--storage-symbol", "BTCUSDT", "--output-dir", "unused"])


@pytest.mark.parametrize("changes", [{"storage_symbol": "../BAD"}, {"max_requests": 0}, {"max_records": True}])
def test_programmatic_cli_invalid_context_never_initializes_client(tmp_path, monkeypatch, changes):
    import argparse
    module = cli()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "now", lambda: NOW)
    def forbidden():
        pytest.fail("invalid input initialized authenticated client")
    monkeypatch.setattr(module, "BinanceClient", forbidden)
    args = argparse.Namespace(**(dict(start_at=START, end_at=START, storage_symbol="BTCUSDT", max_requests=32,
                                     max_records=4096, output_dir=tmp_path / "reports" / "bad") | changes))
    with pytest.raises(ValueError):
        asyncio.run(module.capture(args))
    assert not args.output_dir.exists()
