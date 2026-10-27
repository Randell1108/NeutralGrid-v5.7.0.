"""Collector orchestration tests: injected clients never reach the network."""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import capture_authenticated_bot_evidence as collector
from neutralgrid.live.decision.state_action_reward import BotIdentity

DEPLOYED = datetime(2026, 10, 1, tzinfo=timezone.utc)
CUTOFF = DEPLOYED + timedelta(hours=12)
BOT = BotIdentity(symbol="BTCUSDT", strategy_id="123", deployed_at=DEPLOYED)


class FakeClient:
    """Only read interfaces exist; order submission is intentionally absent."""

    base_url = "https://fapi.binance.com"
    is_authenticated = True
    api_key = "synthetic-credential-never-transmitted"

    def __init__(self):
        self.calls = []
        self.closed = False

    async def sync_time(self):
        return None

    async def get_exchange_info(self):
        return {"symbols": [{"symbol": "BTCUSDT", "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.1"}]}]}

    async def _request(self, endpoint, parameters, *, signed, max_retries):
        assert signed is True and max_retries == 0
        assert endpoint in {"/fapi/v1/userTrades", "/fapi/v1/income"}
        self.calls.append((endpoint, dict(parameters)))
        at = int((DEPLOYED + timedelta(hours=9)).timestamp() * 1000)
        if endpoint.endswith("userTrades") and parameters["startTime"] <= at <= parameters["endTime"]:
            return [{"symbol": "BTCUSDT", "orderId": 10, "id": 20, "time": at, "price": "100", "qty": "1",
                "realizedPnl": "2", "commission": "0.1", "commissionAsset": "USDT", "marginAsset": "USDT", "side": "BUY"}]
        return []

    async def close(self):
        self.closed = True


def setup(tmp_path, monkeypatch, *, cutoff=CUTOFF):
    monkeypatch.setattr(collector, "ROOT", tmp_path)
    monkeypatch.setattr(collector, "now", lambda: cutoff)
    # Registration proof is separately tested by the ledger suite. These tests
    # isolate interval orchestration with a synthetic reviewed identity.
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"bot": BOT.model_dump(mode="json")}))
    monkeypatch.setattr(collector, "verify_registration", lambda raw, path, live: BotIdentity.model_validate(raw["bot"]))
    client = FakeClient()
    constructed = []

    def factory():
        constructed.append(True)
        return client

    monkeypatch.setattr(collector, "BinanceClient", factory)
    args = argparse.Namespace(output_dir=tmp_path / "reports" / "capture", registration=[registration],
        all_registered=False, capture_pnl=False, fetch=True, ownership_mapping=[], max_requests_per_endpoint=2,
        start_at=None, end_at=None)
    return args, client, constructed


def test_default_history_includes_activity_after_seven_hours(tmp_path, monkeypatch):
    args, client, _ = setup(tmp_path, monkeypatch)
    report = asyncio.run(collector.capture(args))
    assert not report["errors"]
    assert report["acquisitions"][0]["userTrades"]["rows"] == 1
    assert all(p["startTime"] == int(DEPLOYED.timestamp() * 1000) for _, p in client.calls)
    assert all(p["endTime"] == int(CUTOFF.timestamp() * 1000) for _, p in client.calls)
    assert report["submitted_exchange_orders"] == 0 and not report["influence_enabled"]
    assert client.closed


def test_one_run_cutoff_is_fixed_before_fetching(tmp_path, monkeypatch):
    args, client, _ = setup(tmp_path, monkeypatch)
    clock_calls = 0

    def advancing_clock():
        nonlocal clock_calls
        clock_calls += 1
        return CUTOFF if clock_calls == 1 else CUTOFF + timedelta(hours=1)

    monkeypatch.setattr(collector, "now", advancing_clock)
    report = asyncio.run(collector.capture(args))
    assert all(p["endTime"] == int(CUTOFF.timestamp() * 1000) for _, p in client.calls)
    assert report["api_history_window"]["run_cutoff_at"] == CUTOFF.isoformat()
    assert report["finished_at"] == (CUTOFF + timedelta(hours=1)).isoformat()


def test_explicit_aware_bounds_apply_to_api_history(tmp_path, monkeypatch):
    args, client, _ = setup(tmp_path, monkeypatch)
    args.start_at = "2026-10-01T03:00:00-05:00"
    args.end_at = "2026-10-01T10:00:00Z"
    report = asyncio.run(collector.capture(args))
    assert not report["errors"] and len(client.calls) == 2
    assert all(p["startTime"] == int((DEPLOYED + timedelta(hours=8)).timestamp() * 1000) for _, p in client.calls)
    assert all(p["endTime"] == int((DEPLOYED + timedelta(hours=10)).timestamp() * 1000) for _, p in client.calls)


@pytest.mark.parametrize("start,end", [
    ("bad", None), ("2026-10-01T08:00:00", None), (None, "2026-10-01T10:00:00"),
    ("2026-10-01T11:00:00Z", "2026-10-01T10:00:00Z"),
    ("2026-10-01T13:00:00Z", None), (None, "2026-10-01T13:00:00Z"),
])
def test_invalid_api_bounds_fail_before_client_or_output_creation(tmp_path, monkeypatch, start, end):
    args, client, constructed = setup(tmp_path, monkeypatch)
    args.start_at, args.end_at = start, end
    with pytest.raises(ValueError):
        asyncio.run(collector.capture(args))
    assert not constructed and not client.calls and not args.output_dir.exists()


def test_requested_start_before_deployment_is_intersected_explicitly(tmp_path, monkeypatch):
    args, client, _ = setup(tmp_path, monkeypatch)
    args.start_at = (DEPLOYED - timedelta(days=1)).isoformat()
    report = asyncio.run(collector.capture(args))
    assert all(p["startTime"] == int(DEPLOYED.timestamp() * 1000) for _, p in client.calls)
    assert report["acquisitions"][0]["history_start_at"] == DEPLOYED.isoformat()
    assert report["api_history_window"]["requested_start_at"] == args.start_at


def test_nonoverlapping_deployment_is_skipped_without_inventing_terminal_date(tmp_path, monkeypatch):
    args, client, constructed = setup(tmp_path, monkeypatch)
    args.end_at = (DEPLOYED - timedelta(seconds=1)).isoformat()
    report = asyncio.run(collector.capture(args))
    assert not report["acquisitions"] and not client.calls and not constructed
    assert report["skipped_api_deployments"][0]["reason"] == "deployment_after_requested_end"


def test_prior_history_manifest_and_raw_bytes_remain_immutable(tmp_path, monkeypatch):
    args, client, _ = setup(tmp_path, monkeypatch)
    args.end_at = (DEPLOYED + timedelta(hours=7)).isoformat()
    first = asyncio.run(collector.capture(args))
    manifest_ref = first["acquisitions"][0]["userTrades"]["manifest"]
    manifest = Path(manifest_ref["path"])
    original_manifest = manifest.read_bytes()
    old_body = json.loads(original_manifest)
    original_sources = {Path(ref["path"]): Path(ref["path"]).read_bytes() for ref in old_body["sources"]}
    args.output_dir = tmp_path / "reports" / "lifetime"
    args.end_at = None
    second = asyncio.run(collector.capture(args))
    assert second["acquisitions"][0]["userTrades"]["rows"] == 1
    assert manifest.read_bytes() == original_manifest
    assert all(path.read_bytes() == content for path, content in original_sources.items())
    assert second["acquisitions"][0]["userTrades"]["manifest"] != manifest_ref


def test_api_investigation_bounds_do_not_filter_stored_pnl_history(tmp_path, monkeypatch):
    args, client, constructed = setup(tmp_path, monkeypatch)
    args.fetch, args.capture_pnl = False, True
    args.start_at, args.end_at = "2026-10-01T08:00:00Z", "2026-10-01T10:00:00Z"
    rows = [{"symbol": BOT.symbol, "strategy_id": BOT.strategy_id, "deploy_ts": DEPLOYED,
             "_source_path": str(tmp_path / f"observation_{hour}.json")} for hour in (1, 9, 11)]
    monkeypatch.setattr(collector, "collect_observations", lambda live: (rows, [], []))
    indexed = []
    output = tmp_path / "synthetic_pnl_path.json"
    output.write_text("{}")

    def capture_path(paths, **kwargs):
        indexed.extend(paths)
        return output

    def blocked_forecast(path, **kwargs):
        raise ValueError("synthetic fixture has no forecast evidence")

    monkeypatch.setattr(collector, "capture_pnl_path", capture_path)
    monkeypatch.setattr(collector, "forecast_from_reported_pnl_path", blocked_forecast)
    report = asyncio.run(collector.capture(args))
    assert indexed == [Path(row["_source_path"]) for row in rows]
    assert not constructed and not client.calls and len(report["pnl_paths"]) == 1


def test_unchanged_request_budget_records_lifetime_gaps(tmp_path, monkeypatch):
    args, client, _ = setup(tmp_path, monkeypatch, cutoff=DEPLOYED + timedelta(days=20))
    args.max_requests_per_endpoint = 1
    report = asyncio.run(collector.capture(args))
    assert len(client.calls) == 2  # One bounded request for each approved endpoint.
    for key in ("userTrades", "income"):
        endpoint = report["acquisitions"][0][key]
        assert endpoint["requests"] == 1 and not endpoint["retrieval_exhausted"]
        assert any(gap["reason"] == "request_budget" for gap in endpoint["gaps"])
    assert not report["training_eligible"] and not report["promotion_eligible"]


def test_future_deployment_is_blocked_before_constructing_client(tmp_path, monkeypatch):
    args, client, constructed = setup(tmp_path, monkeypatch, cutoff=DEPLOYED - timedelta(seconds=1))
    report = asyncio.run(collector.capture(args))
    assert not constructed and not client.calls and not report["acquisitions"]
    assert report["errors"][0]["error_type"] == "FutureDeployment"


def test_inclusive_single_instant_window_is_supported(tmp_path, monkeypatch):
    args, client, _ = setup(tmp_path, monkeypatch)
    args.start_at = args.end_at = "2026-10-01T09:00:00Z"
    report = asyncio.run(collector.capture(args))
    assert report["acquisitions"][0]["userTrades"]["rows"] == 1
    assert all(p["startTime"] == p["endTime"] for _, p in client.calls)


def test_cli_invalid_future_bound_returns_error_without_network(tmp_path, monkeypatch):
    _, client, constructed = setup(tmp_path, monkeypatch)
    result = collector.main(["--fetch", "--end-at", "2026-10-01T13:00:00Z",
                             "--output-dir", str(tmp_path / "reports" / "invalid")])
    assert result == 2 and not client.calls and not constructed
