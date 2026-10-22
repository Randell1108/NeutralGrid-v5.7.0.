from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from neutralgrid.data.funding_observation import (
    capture_funding_observation, persist_funding_observation,
    premium_funding_info, replay_funding_observation,
    premium_rate_from_market_data,
)

T = datetime(2026, 10, 21, 4, 0, tzinfo=timezone.utc)


def payload():
    return {"symbol": "BTCUSDT", "time": int(T.timestamp()*1000),
            "lastFundingRate": "-0.0002", "nextFundingTime": int((T+timedelta(hours=2)).timestamp()*1000)}


def test_enrichment_uses_observed_premium_even_when_history_is_nonempty():
    market = {"funding_rate": [{"fundingRate": "0.99", "fundingTime": 1}], "premium_index": payload()}
    assert premium_rate_from_market_data(market, symbol="BTCUSDT", cutoff=T) == pytest.approx(-0.0002)
    del market["premium_index"]
    assert premium_rate_from_market_data(market, symbol="BTCUSDT", cutoff=T) is None
    market["premium_index"] = payload()
    assert premium_rate_from_market_data(market, symbol="BTCUSDT", cutoff=T+timedelta(hours=3)) is None


def test_receipt_roundtrip_units_zero_boundary_and_explicit_age(tmp_path):
    path = persist_funding_observation(payload(), symbol="BTCUSDT", received_at=T, live_root=tmp_path)
    assert path.parent.parent.parent.name == "2026-10-20"  # Lima ingestion date
    value, proof = replay_funding_observation(path, symbol="BTCUSDT", cutoff=T+timedelta(seconds=1), max_age_seconds=1)
    assert value == pytest.approx(-0.02)
    assert proof["observed_at_ms"] == payload()["time"]
    assert proof["cutoff_utc"] != proof["received_at_utc"]
    with pytest.raises(ValueError, match="too old"):
        replay_funding_observation(path, symbol="BTCUSDT", cutoff=T+timedelta(seconds=2), max_age_seconds=1)
    p = payload(); p["lastFundingRate"] = "0"
    zero = persist_funding_observation(p, symbol="BTCUSDT", received_at=T, live_root=tmp_path)
    assert replay_funding_observation(zero, symbol="BTCUSDT", cutoff=T, max_age_seconds=0)[0] == 0


@pytest.mark.parametrize("change", [
    {"time": int(T.timestamp()*1000)+1}, {"time": True}, {"time": None},
    {"nextFundingTime": int(T.timestamp()*1000)-1}, {"lastFundingRate": "nan"},
    {"lastFundingRate": True}, {"symbol": "ETHUSDT"},
])
def test_invalid_and_stale_sources_fail_closed(change):
    p = payload(); p.update(change)
    with pytest.raises((ValueError, TypeError)):
        premium_funding_info(p, symbol="BTCUSDT", cutoff=T)


def test_receipt_rejects_post_entry_receipt_tampering_and_other_strategy(tmp_path):
    path = persist_funding_observation(payload(), symbol="BTCUSDT", received_at=T+timedelta(seconds=1), live_root=tmp_path, strategy_id="42")
    with pytest.raises(ValueError, match="unavailable"):
        replay_funding_observation(path, symbol="BTCUSDT", cutoff=T, max_age_seconds=60, strategy_id="42")
    with pytest.raises(ValueError, match="strategy"):
        replay_funding_observation(path, symbol="BTCUSDT", cutoff=T+timedelta(seconds=2), max_age_seconds=60, strategy_id="43")
    record = json.loads(path.read_text()); record["payload"]["lastFundingRate"] = "1"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash"):
        replay_funding_observation(path, symbol="BTCUSDT", cutoff=T+timedelta(seconds=2), max_age_seconds=60, strategy_id="42")


@pytest.mark.asyncio
async def test_capture_persists_original_exchange_time_and_distinct_receipt(tmp_path):
    now = datetime.now(timezone.utc)
    p = payload(); p["time"] = int(now.timestamp()*1000)-1
    p["nextFundingTime"] = int((now+timedelta(hours=1)).timestamp()*1000)
    client = SimpleNamespace(get_premium_index=AsyncMock(return_value=p))
    info, path = await capture_funding_observation(client, "BTCUSDT", live_root=tmp_path)
    assert info["observed_at_ms"] == p["time"]
    assert json.loads(path.read_text())["payload"] == p
    client.get_premium_index.assert_awaited_once_with("BTCUSDT")
