from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from neutralgrid.data.market_stream import (
    MARKET_KLINE_SCHEMA_VERSION,
    MARKET_MANIFEST_SCHEMA_VERSION,
    MarketKline,
    MarketPayloadError,
    MarketStreamError,
    MarketStreamStorage,
    parse_market_kline,
)
from scripts.collect_market_streams import collect_market_streams, parse_args


def _kline(*, closed: bool = True) -> dict[str, object]:
    return {
        "stream": "btcusdt@kline_1m",
        "data": {
            "e": "kline",
            "E": 1_700_000_060_001,
            "s": "BTCUSDT",
            "k": {
                "t": 1_700_000_000_000,
                "T": 1_700_000_059_999,
                "s": "BTCUSDT",
                "i": "1m",
                "o": "100.0",
                "h": "102.0",
                "l": "99.0",
                "c": "101.0",
                "v": "12.5",
                "q": "1250.0",
                "n": 17,
                "V": "7.0",
                "Q": "700.0",
                "x": closed,
            },
        },
    }


def _parse(payload: dict[str, object]) -> MarketKline:
    return parse_market_kline(
        payload,
        expected_symbol="BTCUSDT",
        expected_intervals={"1m"},
        connection_id="connection-1",
        wire_sequence=1,
        received_at_utc="2026-09-11T01:00:00+00:00",
        received_monotonic_ns=123,
    )


def test_parse_market_kline_preserves_final_bar() -> None:
    parsed = _parse(_kline())
    record = parsed.to_record()

    assert record["schema_version"] == MARKET_KLINE_SCHEMA_VERSION
    assert record["record_type"] == "closed_kline"
    assert record["symbol"] == "BTCUSDT"
    assert record["interval"] == "1m"
    assert record["closed"] is True
    assert record["open"] == "100.0"
    assert record["close"] == "101.0"
    assert record["trade_count"] == 17


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("data", "s", "ETHUSDT"), "symbol mismatch"),
        (("data", "k", "i", "5m"), "unexpected kline interval"),
        (("data", "k", "x", "true"), "closed flag"),
        (("data", "k", "h", "98"), "high"),
        (("data", "k", "l", "103"), "low"),
        (("data", "k", "v", "NaN"), "finite"),
    ],
)
def test_parse_market_kline_rejects_schema_and_price_invariants(
    mutation: tuple[str, ...],
    message: str,
) -> None:
    payload = _kline()
    target: dict[str, Any] = payload
    for key in mutation[:-2]:
        nested = target[key]
        if not isinstance(nested, dict):
            raise AssertionError(f"fixture path {key!r} is not an object")
        target = nested
    target[mutation[-2]] = mutation[-1]

    with pytest.raises(MarketPayloadError, match=message):
        _parse(payload)


def test_market_storage_persists_wire_before_structured_and_closes(
    tmp_path: Path,
) -> None:
    storage = MarketStreamStorage(
        tmp_path / "run",
        symbol="BTCUSDT",
        run_id="market-run",
    )
    parsed = parse_market_kline(
        _kline(),
        expected_symbol="BTCUSDT",
        expected_intervals={"1m"},
        connection_id="connection-1",
        wire_sequence=1,
        received_at_utc="2026-09-11T01:00:00+00:00",
        received_monotonic_ns=123,
    )
    storage.append_wire(
        connection_id="connection-1",
        wire_sequence=1,
        received_at_utc="2026-09-11T01:00:00+00:00",
        received_monotonic_ns=123,
        raw_text=json.dumps(_kline()),
        raw_sha256="a" * 64,
    )
    storage.append_closed_kline(parsed)
    storage.write_manifest(
        {"status": "running", "updated_at_utc": "2026-09-11T01:00:00+00:00"}
    )
    storage.close()

    wire = (tmp_path / "run" / "wire_events.jsonl").read_text(encoding="utf-8")
    kline = (tmp_path / "run" / "closed_klines.jsonl").read_text(encoding="utf-8")
    manifest = json.loads(
        (tmp_path / "run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert '"raw_text"' in wire
    assert MARKET_KLINE_SCHEMA_VERSION in kline
    assert manifest["schema_version"] == MARKET_MANIFEST_SCHEMA_VERSION
    assert manifest["counters"]["wire_events"] == 1
    assert manifest["counters"]["closed_klines"] == 1


def test_market_storage_refuses_non_final_kline(tmp_path: Path) -> None:
    storage = MarketStreamStorage(
        tmp_path / "run",
        symbol="BTCUSDT",
        run_id="market-run",
    )
    parsed = parse_market_kline(
        _kline(closed=False),
        expected_symbol="BTCUSDT",
        expected_intervals={"1m"},
        connection_id="connection-1",
        wire_sequence=1,
        received_at_utc="2026-09-11T01:00:00+00:00",
        received_monotonic_ns=123,
    )

    with pytest.raises(MarketStreamError, match="non-final"):
        storage.append_closed_kline(parsed)
    storage.close()


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _agg_trade() -> dict[str, object]:
    return {
        "e": "aggTrade",
        "E": 1_700_000_000_100,
        "s": "BTCUSDT",
        "a": 7,
        "p": "100.5",
        "q": "2",
        "f": 10,
        "l": 11,
        "T": 1_700_000_000_090,
        "m": False,
    }


def _mark_price() -> dict[str, object]:
    return {
        "e": "markPriceUpdate",
        "E": 1_700_000_000_200,
        "s": "BTCUSDT",
        "p": "100.25",
        "i": "100.00",
        "P": "100.10",
        "r": "0.0001",
        "T": 1_700_028_800_000,
    }


@pytest.mark.asyncio
async def test_market_collector_isolated_service_persists_all_streams(
    tmp_path: Path,
) -> None:
    port = _unused_tcp_port()
    app = web.Application()

    async def market_stream(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        subscription = await ws.receive_json()
        assert subscription["method"] == "SUBSCRIBE"
        assert subscription["params"] == [
            "btcusdt@kline_1m",
            "btcusdt@aggTrade",
            "btcusdt@markPrice@1s",
        ]
        assert isinstance(subscription["id"], int)
        await ws.send_json({"result": None, "id": subscription["id"]})
        await ws.send_json(_agg_trade())
        await ws.send_json(_mark_price())
        await ws.send_json(_kline(closed=False)["data"])
        await ws.send_json(_kline(closed=True)["data"])
        async for _message in ws:
            pass
        return ws

    app.router.add_get("/market/stream", market_stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    audit_dir = tmp_path / "audit"
    args = parse_args(
        [
            "--symbols",
            "BTCUSDT",
            "--duration-seconds",
            "0.35",
            "--audit-dir",
            str(audit_dir),
            "--live-root",
            str(tmp_path / "Live"),
            "--ingestion-date",
            "2026-09-10",
            "--ws-base",
            f"ws://127.0.0.1:{port}/market/stream",
            "--intervals",
            "1m",
            "--fsync-every",
            "0",
            "--heartbeat-seconds",
            "1",
            "--rotation-seconds",
            "10",
        ]
    )
    try:
        exit_code = await collect_market_streams(args)
    finally:
        await runner.cleanup()

    manifest = json.loads(
        (audit_dir / "manifest.json").read_text(encoding="utf-8")
    )
    symbol_dir = Path(manifest["symbol_run_dirs"]["BTCUSDT"])
    symbol_manifest = json.loads(
        (symbol_dir / "manifest.json").read_text(encoding="utf-8")
    )
    counters = symbol_manifest["counters"]

    assert exit_code == 0
    assert manifest["service"] == "market"
    assert manifest["traffic_class"] == "market"
    assert manifest["status"] == "complete"
    assert manifest["live_date_lima"] == "2026-09-10"
    assert manifest["ingestion_date_basis"] == "supervisor_frozen"
    assert "2026-09-10" in symbol_dir.parts
    assert symbol_manifest["status"] == "complete_contiguous"
    assert symbol_manifest["subscription_acknowledged"] is True
    assert counters["wire_events"] == 5
    assert counters["public_agg_trades"] == 1
    assert counters["public_mark_price_updates"] == 1
    assert counters["non_final_kline_updates"] == 1
    assert counters["closed_klines"] == 1
    assert counters["parse_errors"] == 0
    assert not (audit_dir / "collector.lock").exists()
    assert len(
        (symbol_dir / "wire_events.jsonl").read_text(encoding="utf-8").splitlines()
    ) == 5
    assert len(
        (symbol_dir / "closed_klines.jsonl").read_text(encoding="utf-8").splitlines()
    ) == 1
