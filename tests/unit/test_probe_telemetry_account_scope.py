from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json

import pytest

from scripts import probe_telemetry_account_scope as probe
from neutralgrid.api.binance_client import BinanceAPIError


def source(tmp_path, **changes):
    path = tmp_path / "Live" / "2026-10-22" / "BTCUSDT" / "ui.json"
    path.parent.mkdir(parents=True)
    body = {"schema_version": "neutralgrid_ui_account_identity_capture_v1",
            "url": "https://www.binance.com/en/my/dashboard",
            "captured_at_utc": "2026-01-01T00:00:00Z",
            "scope": "account_identity_only_not_bot_ownership", "uid_text": 'UID 123'} | changes
    path.write_text(json.dumps(body))
    return path


@pytest.mark.parametrize("uid_text", ["UID 123", '- generic: UID\n- generic: "123"'])
def test_ui_extracts_only_unambiguous_identifier(tmp_path, uid_text):
    path = source(tmp_path, uid_text=uid_text)
    assert probe.ui_identity(path, tmp_path / "Live", datetime.now(timezone.utc))[0] == "123"


@pytest.mark.parametrize("changes", [
    {"uid_text": "UID 123 UID 124"}, {"uid_text": "UID 00123"}, {"uid_text": "UID 1.2"},
    {"uid_text": None}, {"url": "https://example.org"}, {"captured_at_utc": "2026-01-01T00:00:00"},
    {"captured_at_utc": None}, {"captured_at_utc": []}, {"captured_at_utc": 123},
    {"captured_at_utc": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()},
])
def test_bad_ui_fails_before_any_network_or_output(tmp_path, changes, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("network must not be created")
    monkeypatch.setattr(probe, "BinanceClient", forbidden)
    with pytest.raises(ValueError):
        asyncio.run(probe.probe(source=source(tmp_path, **changes), symbol="BTCUSDT",
                               live_root=tmp_path / "Live", report_dir=tmp_path / "report"))
    assert not (tmp_path / "report").exists()


def test_ui_mutation_between_parse_and_reference_fails(tmp_path, monkeypatch):
    path = source(tmp_path)
    original = probe.read_json
    def mutating_read(p):
        result = original(p)
        p.write_text(p.read_text().replace("UID 123", "UID 124"))
        return result
    monkeypatch.setattr(probe, "read_json", mutating_read)
    with pytest.raises(ValueError, match="changed"):
        probe.ui_identity(path, tmp_path / "Live", datetime.now(timezone.utc))


@pytest.mark.parametrize("response,status", [({"uid": 123}, "match"), ({"uid": 124}, "mismatch"),
    ({}, "unverified"), ({"uid": True}, "unverified"), ({"uid": 1.2}, "unverified"),
    ({"uid": "0123"}, "unverified"), ({"uid": 0}, "unverified")])
def test_probe_is_get_only_separates_identity_from_ownership(tmp_path, monkeypatch, response, status):
    calls, closed = [], []
    class Client:
        def __init__(self, **kwargs):
            self.base_url = kwargs.get("base_url", probe.HOST)
            self.api_key = "test-key"
            self.api_secret = "never-store-secret"
            self.is_authenticated = True
        async def _request(self, endpoint, **kwargs):
            calls.append((self.base_url, endpoint, kwargs))
            return {"serverTime": 1790000000000} if endpoint.endswith("/time") else response
        async def close(self):
            closed.append(self.base_url)
        async def sync_time(self):
            return await self._request("/api/v3/time", signed=False, max_retries=0)
    monkeypatch.setattr(probe, "BinanceClient", Client)
    monkeypatch.setattr(probe, "SpotIdentityClient", Client)
    result = asyncio.run(probe.probe(source=source(tmp_path), symbol="BTCUSDT",
                                    live_root=tmp_path / "Live", report_dir=tmp_path / "report"))
    assert result["identity_status"] == status
    assert [c[1] for c in calls] == ["/api/v3/time", "/api/v3/account"]
    assert all(c[0] == probe.SPOT_HOST for c in calls)
    assert calls[1][2] == {"signed": True, "max_retries": 0}
    assert len(closed) == 2
    assert not result["bot_ownership_proven"] and not result["influence_enabled"]
    assert not result["trading_bots_rest_visibility_proven"] and not result["training_eligible"]
    assert "never-store-secret" not in json.dumps(result)


def test_error_message_and_signed_url_never_persist(tmp_path, monkeypatch):
    class Client:
        base_url = probe.HOST
        api_key = "test-key"
        api_secret = "secret"
        is_authenticated = True
        def __init__(self, **kwargs):
            pass
        async def _request(self, *args, **kwargs):
            raise BinanceAPIError(401, -2015, "secret signature=secret", "secret-url", "secret")
        async def close(self):
            pass
        async def sync_time(self):
            return await self._request("/api/v3/time", signed=False, max_retries=0)
    monkeypatch.setattr(probe, "BinanceClient", Client)
    monkeypatch.setattr(probe, "SpotIdentityClient", Client)
    result = asyncio.run(probe.probe(source=source(tmp_path), symbol="BTCUSDT",
                                    live_root=tmp_path / "Live", report_dir=tmp_path / "report"))
    assert result["identity_status"] == "unverified"
    assert result["errors"] == [{"error_type": "BinanceAPIError", "http_status": 401, "exchange_error_code": -2015}]
    assert "secret" not in (tmp_path / "report" / "report.json").read_text()


def test_spot_clock_never_uses_futures_route(monkeypatch):
    calls = []
    async def request(self, endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        return {"serverTime": 1790000000000}
    monkeypatch.setattr(probe.SpotIdentityClient, "_request", request)
    client = probe.SpotIdentityClient(api_key="fake", api_secret="fake", base_url=probe.SPOT_HOST)
    asyncio.run(client.sync_time())
    assert calls == [("/api/v3/time", {"signed": False, "max_retries": 0})]
    assert client._ENDPOINT_WEIGHTS["/api/v3/account"] == 20
