from __future__ import annotations

import socket
from typing import Any

import pytest
from aiohttp import web

from neutralgrid.api.binance_client import BinanceClient


@pytest.mark.asyncio
async def test_listen_key_lifecycle_uses_current_websocket_api(
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    calls: list[dict[str, Any]] = []
    listen_key = "secret-listen-key"

    async def lifecycle(request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        payload = await websocket.receive_json()
        calls.append(payload)
        method = payload["method"]
        result = (
            {"listenKey": listen_key}
            if method in {"userDataStream.start", "userDataStream.ping"}
            else {}
        )
        await websocket.send_json(
            {"id": payload["id"], "status": 200, "result": result}
        )
        await websocket.close()
        return websocket

    app = web.Application()
    app.router.add_get("/ws-fapi/v1", lifecycle)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    client = BinanceClient(
        api_key="api-key",
        api_secret="api-secret",
        user_data_ws_api_url=f"ws://127.0.0.1:{port}/ws-fapi/v1",
    )
    try:
        observed_key = await client.start_user_data_stream()
        await client.keepalive_user_data_stream(observed_key)
        await client.close_user_data_stream(observed_key)
    finally:
        await client.close()
        await runner.cleanup()

    assert observed_key == listen_key
    assert [item["method"] for item in calls] == [
        "userDataStream.start",
        "userDataStream.ping",
        "userDataStream.stop",
    ]
    assert all(item["params"] == {"apiKey": "api-key"} for item in calls)
    assert len({item["id"] for item in calls}) == 3


@pytest.mark.asyncio
async def test_listen_key_lifecycle_fails_closed_without_api_key() -> None:
    client = BinanceClient(api_key="", api_secret="")
    client.api_key = ""

    with pytest.raises(ValueError, match="API key"):
        await client.start_user_data_stream()
    with pytest.raises(ValueError, match="API key"):
        await client.keepalive_user_data_stream("listen-key")
    with pytest.raises(ValueError, match="API key"):
        await client.close_user_data_stream("listen-key")


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"listenKey": ""}, [], None])
async def test_start_user_data_stream_rejects_invalid_response(
    payload: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = BinanceClient(api_key="api-key", api_secret="")

    async def fake_ws_call(method: str) -> Any:
        assert method == "userDataStream.start"
        return payload

    monkeypatch.setattr(client, "_user_data_stream_ws_call", fake_ws_call)

    with pytest.raises(ValueError, match="response"):
        await client.start_user_data_stream()


@pytest.mark.asyncio
async def test_keepalive_rejects_changed_listen_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = BinanceClient(api_key="api-key", api_secret="")

    async def fake_ws_call(method: str) -> dict[str, str]:
        assert method == "userDataStream.ping"
        return {"listenKey": "different-key"}

    monkeypatch.setattr(client, "_user_data_stream_ws_call", fake_ws_call)

    with pytest.raises(ValueError, match="different listenKey"):
        await client.keepalive_user_data_stream("expected-key")
