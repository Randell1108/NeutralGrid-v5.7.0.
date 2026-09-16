from __future__ import annotations

import csv
import json
import socket
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.data.private_user_stream import LINKAGE_SCHEMA_VERSION
from scripts import collect_private_user_stream as private_collector


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_roster(path: Path, rows: list[tuple[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "strategy_id"])
        writer.writeheader()
        for symbol, strategy_id in rows:
            writer.writerow({"symbol": symbol, "strategy_id": strategy_id})
    return path


def _write_linkage(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": LINKAGE_SCHEMA_VERSION,
                "symbol": "BTCUSDT",
                "strategy_id": "413500001",
                "order_ids": [70],
                "provenance": "reviewed_test_exchange_export",
            }
        ),
        encoding="utf-8",
    )
    return path


def _order_event() -> dict[str, Any]:
    return {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1_700_000_000_100,
        "T": 1_700_000_000_090,
        "o": {
            "s": "BTCUSDT",
            "c": "grid-order-1",
            "S": "BUY",
            "x": "TRADE",
            "X": "FILLED",
            "i": 70,
            "l": "2",
            "z": "2",
            "L": "100.5",
            "N": "USDT",
            "n": "0.01",
            "T": 1_700_000_000_090,
            "t": 7,
            "m": True,
            "rp": "0.25",
        },
    }


def test_private_roster_allows_multiple_exact_strategies_per_symbol(
    tmp_path: Path,
) -> None:
    path = _write_roster(
        tmp_path / "targets.csv",
        [("BTCUSDT", "413500001"), ("BTCUSDT", "413500002")],
    )

    targets = private_collector.load_private_targets(path)

    assert [(item.symbol, item.strategy_id) for item in targets] == [
        ("BTCUSDT", "413500001"),
        ("BTCUSDT", "413500002"),
    ]


def test_private_collector_rejects_unapproved_nonproduction_endpoints(
    tmp_path: Path,
) -> None:
    roster = _write_roster(
        tmp_path / "targets.csv", [("BTCUSDT", "413500001")]
    )

    with pytest.raises(SystemExit):
        private_collector.parse_args(
            [
                "--input",
                str(roster),
                "--lifecycle-ws-url",
                "ws://127.0.0.1:1/ws-fapi/v1",
            ]
        )


@pytest.mark.asyncio
async def test_private_collector_lifecycle_redaction_and_exact_linkage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port = _unused_tcp_port()
    listen_key = "test-listen-key-never-durable"
    lifecycle: list[str] = []
    app = web.Application()

    async def lifecycle_stream(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        payload = await ws.receive_json()
        assert payload["params"] == {"apiKey": "test-api-key"}
        method = payload["method"]
        lifecycle.append(method)
        result = (
            {"listenKey": listen_key}
            if method in {"userDataStream.start", "userDataStream.ping"}
            else {}
        )
        await ws.send_json(
            {"id": payload["id"], "status": 200, "result": result}
        )
        await ws.close()
        return ws

    async def private_stream(request: web.Request) -> web.WebSocketResponse:
        assert request.match_info["listen_key"] == listen_key
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json(_order_event())
        await ws.send_json(
            {
                "e": "ACCOUNT_UPDATE",
                "E": 1_700_000_000_200,
                "T": 1_700_000_000_190,
                "listenKey": listen_key,
                "a": {"m": "ORDER", "P": [{"s": "BTCUSDT", "pa": "2"}]},
            }
        )
        async for _message in ws:
            pass
        return ws

    app.router.add_get("/ws-fapi/v1", lifecycle_stream)
    app.router.add_get("/private/ws/{listen_key}", private_stream)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()

    def make_client(*, user_data_ws_api_url: str) -> BinanceClient:
        return BinanceClient(
            api_key="test-api-key",
            api_secret="",
            user_data_ws_api_url=user_data_ws_api_url,
        )

    monkeypatch.setattr(private_collector, "BinanceClient", make_client)
    roster = _write_roster(
        tmp_path / "targets.csv", [("BTCUSDT", "413500001")]
    )
    linkage = _write_linkage(tmp_path / "linkage.json")
    audit_dir = tmp_path / "audit"
    live_root = tmp_path / "Live"
    args = private_collector.parse_args(
        [
            "--input",
            str(roster),
            "--linkage-file",
            str(linkage),
            "--duration-seconds",
            "1.5",
            "--audit-dir",
            str(audit_dir),
            "--live-root",
            str(live_root),
            "--ingestion-date",
            "2026-09-10",
            "--ws-base",
            f"ws://127.0.0.1:{port}/private/ws",
            "--lifecycle-ws-url",
            f"ws://127.0.0.1:{port}/ws-fapi/v1",
            "--allow-nonproduction-endpoints",
            "--keepalive-seconds",
            "0.05",
            "--heartbeat-seconds",
            "1",
            "--rotation-seconds",
            "10",
            "--fsync-every",
            "0",
        ]
    )
    try:
        exit_code = await private_collector.collect_private_user_stream(args)
    finally:
        await runner.cleanup()

    manifest = json.loads(
        (audit_dir / "manifest.json").read_text(encoding="utf-8")
    )
    run_dir = Path(
        manifest["symbol_strategy_run_dirs"]["BTCUSDT:413500001"]
    )
    target_manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    canonical = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert exit_code == 0
    assert manifest["service"] == "private"
    assert manifest["traffic_class"] == "private"
    assert manifest["status"] == "complete"
    assert manifest["event_completeness"] == "unknown"
    assert manifest["live_date_lima"] == "2026-09-10"
    assert manifest["ingestion_date_basis"] == "supervisor_frozen"
    assert "2026-09-10" in run_dir.parts
    assert target_manifest["event_completeness"] == "unknown"
    assert [item["event_type"] for item in canonical] == [
        "order_update",
        "trade_fill",
    ]
    assert canonical[1]["order_id"] == "70"
    assert canonical[1]["trade_id"] == "7"
    assert lifecycle[0] == "userDataStream.start"
    assert "userDataStream.ping" in lifecycle
    assert lifecycle[-1] == "userDataStream.stop"
    assert not (audit_dir / "collector.lock").exists()
    for path in [*audit_dir.rglob("*.json"), *live_root.rglob("*.json*")]:
        assert listen_key not in path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_private_collector_blocks_cleanly_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MissingKeyClient:
        api_key = ""

        def __init__(self, *, user_data_ws_api_url: str) -> None:
            self.user_data_ws_api_url = user_data_ws_api_url

        async def close(self) -> None:
            return None

    monkeypatch.setattr(private_collector, "BinanceClient", MissingKeyClient)
    roster = _write_roster(
        tmp_path / "targets.csv", [("BTCUSDT", "413500001")]
    )
    audit_dir = tmp_path / "audit"
    live_root = tmp_path / "Live"
    args = private_collector.parse_args(
        [
            "--input",
            str(roster),
            "--audit-dir",
            str(audit_dir),
            "--live-root",
            str(live_root),
        ]
    )

    exit_code = await private_collector.collect_private_user_stream(args)

    manifest = json.loads(
        (audit_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert exit_code == 3
    assert manifest["status"] == "blocked_missing_api_key"
    assert manifest["event_completeness"] == "unknown"
    assert not live_root.exists()
