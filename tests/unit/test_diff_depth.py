from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path

import pytest
from aiohttp import web

from neutralgrid.data.diff_depth import (
    DepthSequenceEngine,
    PayloadValidationError,
    SequenceBufferOverflow,
    SymbolCaptureStorage,
    atomic_write_json,
    parse_depth_snapshot,
    parse_diff_depth_event,
    parse_public_agg_trade,
    parse_public_mark_price,
    replay_symbol_capture,
)
from scripts.collect_diff_depth import _load_targets, collect_diff_depth, parse_args


@pytest.mark.parametrize("strategy_column", ["strategy_id", "strategy_number"])
def test_load_targets_preserves_exact_strategy_identity(
    tmp_path: Path,
    strategy_column: str,
) -> None:
    target_csv = tmp_path / "targets.csv"
    target_csv.write_text(
        f"symbol,{strategy_column},candidate_id\n"
        "BTCUSDT,413500001,candidate-1\n",
        encoding="utf-8",
    )

    targets = _load_targets(symbols=None, input_path=str(target_csv))

    assert len(targets) == 1
    assert targets[0].symbol == "BTCUSDT"
    assert targets[0].strategy_id == "413500001"
    assert targets[0].candidate_id == "candidate-1"


def _event_payload(
    *,
    first_update_id: int,
    final_update_id: int,
    previous_final_update_id: int,
    bid_price: str = "100",
    bid_quantity: str = "2",
    ask_price: str = "101",
    ask_quantity: str = "3",
) -> dict[str, object]:
    return {
        "e": "depthUpdate",
        "E": 1_700_000_000_000 + final_update_id,
        "T": 1_700_000_000_000 + final_update_id,
        "s": "BTCUSDT",
        "U": first_update_id,
        "u": final_update_id,
        "pu": previous_final_update_id,
        "b": [[bid_price, bid_quantity]],
        "a": [[ask_price, ask_quantity]],
    }


def _event(
    *,
    first_update_id: int,
    final_update_id: int,
    previous_final_update_id: int,
    wire_sequence: int,
    connection_id: str = "connection-a",
    bid_price: str = "100",
    ask_price: str = "101",
):
    return parse_diff_depth_event(
        _event_payload(
            first_update_id=first_update_id,
            final_update_id=final_update_id,
            previous_final_update_id=previous_final_update_id,
            bid_price=bid_price,
            ask_price=ask_price,
        ),
        expected_symbol="BTCUSDT",
        connection_id=connection_id,
        wire_sequence=wire_sequence,
        received_at_utc="2026-07-29T01:00:00+00:00",
        received_monotonic_ns=wire_sequence,
    )


def _agg_trade_payload(
    *,
    aggregate_trade_id: int = 200,
    price: str = "100.5",
    quantity: str = "2",
    buyer_is_maker: bool = False,
) -> dict[str, object]:
    return {
        "e": "aggTrade",
        "E": 1_700_000_000_000 + aggregate_trade_id,
        "s": "BTCUSDT",
        "a": aggregate_trade_id,
        "p": price,
        "q": quantity,
        "f": aggregate_trade_id * 10,
        "l": aggregate_trade_id * 10 + 1,
        "T": 1_700_000_000_000 + aggregate_trade_id,
        "m": buyer_is_maker,
    }


def test_public_agg_trade_parser_preserves_direction_and_exchange_ids() -> None:
    event = parse_public_agg_trade(
        _agg_trade_payload(),
        expected_symbol="BTCUSDT",
        connection_id="connection-a",
        wire_sequence=4,
        received_at_utc="2026-08-02T01:00:00+00:00",
        received_monotonic_ns=42,
    )

    assert event.aggregate_trade_id == 200
    assert event.aggressive_side == "BUY"
    assert event.to_record()["notional_usdt"] == "201.0"
    assert event.first_trade_id == 2000
    assert event.last_trade_id == 2001

    with pytest.raises(PayloadValidationError, match="BUY|SELL|boolean|symbol"):
        parse_public_agg_trade(
            {**_agg_trade_payload(), "m": "false"},
            expected_symbol="BTCUSDT",
            connection_id="connection-a",
            wire_sequence=4,
            received_at_utc="2026-08-02T01:00:00+00:00",
            received_monotonic_ns=42,
        )


def _mark_price_payload() -> dict[str, object]:
    return {
        "e": "markPriceUpdate",
        "E": 1_700_000_000_100,
        "s": "BTCUSDT",
        "p": "100.25",
        "i": "100.20",
        "P": "0",
        "r": "-0.0001",
        "T": 1_700_003_600_000,
    }


def test_public_mark_price_parser_preserves_signed_funding_and_prices() -> None:
    event = parse_public_mark_price(
        _mark_price_payload(),
        expected_symbol="BTCUSDT",
        connection_id="connection-a",
        wire_sequence=5,
        received_at_utc="2026-08-21T01:00:00+00:00",
        received_monotonic_ns=43,
    )

    record = event.to_record()
    assert record["mark_price"] == "100.25"
    assert record["index_price"] == "100.20"
    assert record["funding_rate"] == "-0.0001"

    with pytest.raises(PayloadValidationError, match="positive"):
        parse_public_mark_price(
            {**_mark_price_payload(), "p": "0"},
            expected_symbol="BTCUSDT",
            connection_id="connection-a",
            wire_sequence=5,
            received_at_utc="2026-08-21T01:00:00+00:00",
            received_monotonic_ns=43,
        )


def _snapshot(
    *,
    last_update_id: int,
    connection_id: str = "connection-a",
    wire_sequence_seen: int = 0,
):
    return parse_depth_snapshot(
        {
            "lastUpdateId": last_update_id,
            "bids": [["99", "5"]],
            "asks": [["101", "5"]],
        },
        symbol="BTCUSDT",
        connection_id=connection_id,
        request_started_at_utc="2026-07-29T01:00:00+00:00",
        received_at_utc="2026-07-29T01:00:00+00:00",
        received_monotonic_ns=10,
        wire_sequence_seen=wire_sequence_seen,
        request_latency_ms=1.0,
    )


def test_bootstrap_bridge_and_contiguous_update_are_applied() -> None:
    engine = DepthSequenceEngine(
        "BTCUSDT",
        segment_prefix="connection-a",
    )
    first_actions = engine.offer_event(
        _event(
            first_update_id=100,
            final_update_id=102,
            previous_final_update_id=99,
            wire_sequence=1,
        )
    )
    bootstrap_actions = engine.offer_snapshot(
        _snapshot(last_update_id=101, wire_sequence_seen=1)
    )
    next_actions = engine.offer_event(
        _event(
            first_update_id=103,
            final_update_id=104,
            previous_final_update_id=102,
            wire_sequence=2,
        )
    )

    assert [action.kind for action in first_actions] == ["event_buffered"]
    assert [action.kind for action in bootstrap_actions] == [
        "snapshot_offered",
        "segment_started",
        "event_applied",
    ]
    assert [action.kind for action in next_actions] == ["event_applied"]
    assert engine.phase == "live"
    assert engine.last_u == 104
    assert engine.book is not None
    assert str(engine.book.best_bid) == "100"
    assert str(engine.book.best_ask) == "101"


def test_sequence_gap_is_labelled_and_fresh_snapshot_starts_new_segment() -> None:
    engine = DepthSequenceEngine(
        "BTCUSDT",
        segment_prefix="connection-a",
    )
    engine.offer_event(
        _event(
            first_update_id=100,
            final_update_id=101,
            previous_final_update_id=99,
            wire_sequence=1,
        )
    )
    engine.offer_snapshot(_snapshot(last_update_id=100, wire_sequence_seen=1))
    gap_actions = engine.offer_event(
        _event(
            first_update_id=104,
            final_update_id=105,
            previous_final_update_id=103,
            wire_sequence=2,
        )
    )
    resync_actions = engine.offer_snapshot(
        _snapshot(last_update_id=104, wire_sequence_seen=2)
    )

    assert [action.kind for action in gap_actions] == [
        "sequence_gap",
        "segment_closed",
        "event_buffered",
    ]
    gap = gap_actions[0].details
    assert gap["expected_pu"] == 101
    assert gap["received_pu"] == 103
    assert gap_actions[1].details["reason"] == "sequence_gap"
    assert [action.kind for action in resync_actions] == [
        "snapshot_offered",
        "segment_started",
        "event_applied",
    ]
    assert engine.last_u == 105
    assert resync_actions[1].details["segment_id"].endswith("000002")


def test_snapshot_behind_buffer_requests_another_snapshot() -> None:
    engine = DepthSequenceEngine(
        "BTCUSDT",
        segment_prefix="connection-a",
    )
    engine.offer_event(
        _event(
            first_update_id=200,
            final_update_id=201,
            previous_final_update_id=199,
            wire_sequence=1,
        )
    )

    actions = engine.offer_snapshot(_snapshot(last_update_id=150))

    assert [action.kind for action in actions] == [
        "snapshot_offered",
        "snapshot_behind_buffer",
    ]
    assert engine.phase == "buffering"
    assert engine.snapshot is None
    assert len(engine.buffer) == 1


def test_crossed_book_event_is_labelled_and_not_committed() -> None:
    engine = DepthSequenceEngine(
        "BTCUSDT",
        segment_prefix="connection-a",
    )
    engine.offer_event(
        _event(
            first_update_id=100,
            final_update_id=101,
            previous_final_update_id=99,
            wire_sequence=1,
        )
    )
    engine.offer_snapshot(_snapshot(last_update_id=100))

    actions = engine.offer_event(
        _event(
            first_update_id=102,
            final_update_id=103,
            previous_final_update_id=101,
            wire_sequence=2,
            bid_price="102",
        )
    )

    assert [action.kind for action in actions] == [
        "book_invariant_failure",
        "segment_closed",
        "event_buffered",
    ]
    assert engine.phase == "buffering"
    assert engine.book is None
    assert len(engine.buffer) == 1


def test_buffer_overflow_fails_closed() -> None:
    engine = DepthSequenceEngine(
        "BTCUSDT",
        segment_prefix="connection-a",
        max_buffer_events=1,
    )
    engine.offer_event(
        _event(
            first_update_id=100,
            final_update_id=101,
            previous_final_update_id=99,
            wire_sequence=1,
        )
    )

    with pytest.raises(SequenceBufferOverflow):
        engine.offer_event(
            _event(
                first_update_id=102,
                final_update_id=103,
                previous_final_update_id=101,
                wire_sequence=2,
            )
        )


def test_wire_frame_is_durable_before_parse_failure(tmp_path: Path) -> None:
    storage = SymbolCaptureStorage(
        tmp_path,
        symbol="BTCUSDT",
        run_id="test",
        fsync_every=1,
    )
    raw_text = "{not-json"
    storage.append_wire(
        connection_id="connection-a",
        wire_sequence=1,
        received_at_utc="2026-07-29T01:00:00+00:00",
        received_monotonic_ns=1,
        raw_text=raw_text,
    )

    with pytest.raises(PayloadValidationError):
        parse_diff_depth_event(
            raw_text,
            expected_symbol="BTCUSDT",
            connection_id="connection-a",
            wire_sequence=1,
            received_at_utc="2026-07-29T01:00:00+00:00",
            received_monotonic_ns=1,
        )
    storage.append_control("parse_error", {"wire_sequence": 1})
    storage.write_manifest({"status": "failed_parse"})
    storage.close()

    wire_records = [
        json.loads(line)
        for line in (tmp_path / "wire_events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(wire_records) == 1
    assert wire_records[0]["raw_text"] == raw_text
    assert storage.counters["parse_errors"] == 1


def test_atomic_manifest_write_retries_transient_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "manifest.json"
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(source: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(13, "simulated OneDrive lock")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)

    atomic_write_json(destination, {"status": "running", "events": 42})

    assert attempts == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "events": 42,
        "status": "running",
    }
    assert not list(tmp_path.glob(".*.tmp"))


def test_stored_timeline_replays_to_identical_engine_actions(
    tmp_path: Path,
) -> None:
    storage = SymbolCaptureStorage(
        tmp_path,
        symbol="BTCUSDT",
        run_id="test",
        fsync_every=0,
    )
    connection_id = "connection-a"
    engine = DepthSequenceEngine(
        "BTCUSDT",
        segment_prefix=connection_id,
    )
    storage.append_connection_start(connection_id)
    payload = _event_payload(
        first_update_id=100,
        final_update_id=101,
        previous_final_update_id=99,
    )
    raw_text = json.dumps(payload)
    storage.append_wire(
        connection_id=connection_id,
        wire_sequence=1,
        received_at_utc="2026-07-29T01:00:00+00:00",
        received_monotonic_ns=1,
        raw_text=raw_text,
    )
    event = parse_diff_depth_event(
        raw_text,
        expected_symbol="BTCUSDT",
        connection_id=connection_id,
        wire_sequence=1,
        received_at_utc="2026-07-29T01:00:00+00:00",
        received_monotonic_ns=1,
    )
    storage.append_event_timeline(event)
    storage.append_actions(engine.offer_event(event))
    snapshot = _snapshot(last_update_id=100, wire_sequence_seen=1)
    storage.append_snapshot_evidence(snapshot)
    storage.append_snapshot_timeline(snapshot)
    storage.append_actions(engine.offer_snapshot(snapshot))
    storage.append_connection_end(connection_id, "capture_complete")
    storage.append_actions(engine.end_connection("capture_complete"))
    storage.write_manifest({"status": "complete_contiguous"})
    storage.close()

    result = replay_symbol_capture(tmp_path)

    assert result.passed
    assert result.raw_events == 1
    assert result.raw_hash_failures == 0
    assert result.expected_actions == result.replayed_actions


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _assert_case_insensitive_json_keys_are_unique(value: object) -> None:
    if isinstance(value, dict):
        keys = [str(key).casefold() for key in value]
        assert len(keys) == len(set(keys))
        for nested in value.values():
            _assert_case_insensitive_json_keys_are_unique(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_case_insensitive_json_keys_are_unique(nested)


@pytest.mark.asyncio
async def test_isolated_rest_and_websocket_capture_is_replayable(
    tmp_path: Path,
) -> None:
    port = _unused_tcp_port()
    app = web.Application()

    async def depth_snapshot(_request: web.Request) -> web.Response:
        await asyncio.sleep(0.05)
        return web.json_response(
            {
                "lastUpdateId": 100,
                "bids": [["99", "5"]],
                "asks": [["101", "5"]],
            }
        )

    async def depth_stream(request: web.Request) -> web.WebSocketResponse:
        assert request.match_info["stream"] == "btcusdt@depth@100ms"
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(
            json.dumps(
                _event_payload(
                    first_update_id=100,
                    final_update_id=101,
                    previous_final_update_id=99,
                )
            )
        )
        await asyncio.sleep(0.10)
        await ws.send_str(
            json.dumps(
                _event_payload(
                    first_update_id=102,
                    final_update_id=103,
                    previous_final_update_id=101,
                )
            )
        )
        async for _message in ws:
            pass
        return ws

    async def market_stream(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        subscription = await ws.receive_json()
        assert subscription["method"] == "SUBSCRIBE"
        assert subscription["params"] == [
            "btcusdt@aggTrade",
            "btcusdt@markPrice@1s",
        ]
        await ws.send_json({"result": None, "id": subscription["id"]})
        await ws.send_str(json.dumps(_agg_trade_payload()))
        await ws.send_str(json.dumps(_mark_price_payload()))
        async for _message in ws:
            pass
        return ws

    app.router.add_get("/fapi/v1/depth", depth_snapshot)
    app.router.add_get("/public/ws/{stream}", depth_stream)
    app.router.add_get("/market/stream", market_stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    audit_dir = tmp_path / "audit"
    live_root = tmp_path / "Live"
    args = parse_args(
        [
            "--symbols",
            "BTCUSDT",
            "--duration-seconds",
            "0.5",
            "--audit-dir",
            str(audit_dir),
            "--live-root",
            str(live_root),
            "--ws-base",
            f"ws://127.0.0.1:{port}/public/ws",
            "--market-ws-base",
            f"ws://127.0.0.1:{port}/market/stream",
            "--rest-base",
            f"http://127.0.0.1:{port}",
            "--fsync-every",
            "0",
            "--heartbeat-seconds",
            "1",
            "--rotation-seconds",
            "10",
            "--snapshot-timeout-seconds",
            "1",
            "--risk-snapshot-seconds",
            "0.01",
        ]
    )
    try:
        exit_code = await collect_diff_depth(args)
    finally:
        await runner.cleanup()

    manifest = json.loads(
        (audit_dir / "manifest.json").read_text(encoding="utf-8")
    )
    symbol_run_dir = Path(manifest["symbol_run_dirs"]["BTCUSDT"])
    symbol_manifest = json.loads(
        (symbol_run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    verification = replay_symbol_capture(symbol_run_dir)

    assert exit_code == 0
    assert manifest["status"] == "complete"
    assert symbol_manifest["status"] == "complete_contiguous"
    assert symbol_manifest["counters"]["wire_events"] == 5
    assert symbol_manifest["counters"]["events_applied"] == 2
    assert symbol_manifest["counters"]["sequence_gaps"] == 0
    assert symbol_manifest["counters"]["risk_snapshots"] >= 1
    assert symbol_manifest["counters"]["public_agg_trades"] == 1
    assert symbol_manifest["counters"]["public_mark_price_updates"] == 1
    assert symbol_manifest["counters"]["market_connections"] == 1
    assert symbol_manifest["counters"]["market_coverage_gaps"] == 0
    assert symbol_manifest["counters"]["market_parse_errors"] == 0
    assert symbol_manifest["trade_subscription_acknowledged"] is True
    assert symbol_manifest["mark_price_subscription_acknowledged"] is True
    public_trades = [
        json.loads(line)
        for line in (symbol_run_dir / "public_agg_trades.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert public_trades[0]["aggressive_side"] == "BUY"
    assert public_trades[0]["aggregate_id_discontinuity_observed"] is False
    public_mark_prices = [
        json.loads(line)
        for line in (symbol_run_dir / "public_mark_prices.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert public_mark_prices[0]["mark_price"] == "100.25"
    risk_records = [
        json.loads(line)
        for line in (symbol_run_dir / "l2_risk_snapshots.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert risk_records[-1]["symbol"] == "BTCUSDT"
    assert risk_records[-1]["segment_id"]
    assert risk_records[-1]["final_update_id"] == 103
    assert verification.passed


@pytest.mark.asyncio
async def test_missing_combined_stream_ack_blocks_capture(
    tmp_path: Path,
) -> None:
    port = _unused_tcp_port()
    app = web.Application()

    async def depth_snapshot(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "lastUpdateId": 100,
                "bids": [["99", "5"]],
                "asks": [["101", "5"]],
            }
        )

    async def depth_stream(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(
            json.dumps(
                _event_payload(
                    first_update_id=100,
                    final_update_id=101,
                    previous_final_update_id=99,
                )
            )
        )
        async for _message in ws:
            pass
        return ws

    async def market_stream(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive_json()
        async for _message in ws:
            pass
        return ws

    app.router.add_get("/fapi/v1/depth", depth_snapshot)
    app.router.add_get("/public/ws/{stream}", depth_stream)
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
            "0.3",
            "--audit-dir",
            str(audit_dir),
            "--live-root",
            str(tmp_path / "Live"),
            "--ws-base",
            f"ws://127.0.0.1:{port}/public/ws",
            "--market-ws-base",
            f"ws://127.0.0.1:{port}/market/stream",
            "--rest-base",
            f"http://127.0.0.1:{port}",
            "--fsync-every",
            "0",
            "--heartbeat-seconds",
            "1",
            "--rotation-seconds",
            "10",
            "--snapshot-timeout-seconds",
            "1",
        ]
    )
    try:
        exit_code = await collect_diff_depth(args)
    finally:
        await runner.cleanup()

    manifest = json.loads(
        (audit_dir / "manifest.json").read_text(encoding="utf-8")
    )
    symbol_run_dir = Path(manifest["symbol_run_dirs"]["BTCUSDT"])
    symbol_manifest = json.loads(
        (symbol_run_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert exit_code == 2
    assert manifest["status"] == "incomplete_required_streams"
    assert symbol_manifest["counters"]["events_applied"] == 1
    assert symbol_manifest["status"] == (
        "complete_public_stream_subscription_unacknowledged"
    )
    assert symbol_manifest["trade_subscription_acknowledged"] is False
    assert symbol_manifest["mark_price_subscription_acknowledged"] is False


@pytest.mark.asyncio
async def test_isolated_sequence_gap_is_labelled_and_resynchronized(
    tmp_path: Path,
) -> None:
    port = _unused_tcp_port()
    app = web.Application()
    snapshot_requests = 0

    async def depth_snapshot(_request: web.Request) -> web.Response:
        nonlocal snapshot_requests
        snapshot_requests += 1
        last_update_id = 100 if snapshot_requests == 1 else 104
        return web.json_response(
            {
                "lastUpdateId": last_update_id,
                "bids": [["99", "5"]],
                "asks": [["101", "5"]],
            }
        )

    async def depth_stream(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(
            json.dumps(
                _event_payload(
                    first_update_id=100,
                    final_update_id=101,
                    previous_final_update_id=99,
                )
            )
        )
        await asyncio.sleep(0.10)
        await ws.send_str(
            json.dumps(
                _event_payload(
                    first_update_id=104,
                    final_update_id=105,
                    previous_final_update_id=103,
                )
            )
        )
        async for _message in ws:
            pass
        return ws

    app.router.add_get("/fapi/v1/depth", depth_snapshot)
    app.router.add_get("/public/ws/{stream}", depth_stream)
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
            "0.6",
            "--audit-dir",
            str(audit_dir),
            "--live-root",
            str(tmp_path / "Live"),
            "--ws-base",
            f"ws://127.0.0.1:{port}/public/ws",
            "--rest-base",
            f"http://127.0.0.1:{port}",
            "--fsync-every",
            "0",
            "--heartbeat-seconds",
            "1",
            "--rotation-seconds",
            "10",
            "--snapshot-timeout-seconds",
            "1",
            "--no-agg-trades",
            "--no-mark-price-updates",
        ]
    )
    try:
        exit_code = await collect_diff_depth(args)
    finally:
        await runner.cleanup()

    manifest = json.loads(
        (audit_dir / "manifest.json").read_text(encoding="utf-8")
    )
    symbol_run_dir = Path(manifest["symbol_run_dirs"]["BTCUSDT"])
    symbol_manifest = json.loads(
        (symbol_run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    actions = [
        json.loads(line)
        for line in (symbol_run_dir / "engine_actions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    verification = replay_symbol_capture(symbol_run_dir)

    assert exit_code == 0
    assert symbol_manifest["status"] == "complete_with_labelled_gaps"
    assert symbol_manifest["counters"]["sequence_gaps"] == 1
    assert symbol_manifest["counters"]["snapshots"] >= 2
    assert symbol_manifest["counters"]["segments_started"] == 2
    assert symbol_manifest["counters"]["segments_closed"] == 2
    assert any(action["kind"] == "sequence_gap" for action in actions)
    for action in actions:
        _assert_case_insensitive_json_keys_are_unique(action)
    assert verification.passed


@pytest.mark.asyncio
async def test_connection_gap_ends_only_after_websocket_reconnects(
    tmp_path: Path,
) -> None:
    port = _unused_tcp_port()
    app = web.Application()
    websocket_attempts = 0

    async def depth_snapshot(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "lastUpdateId": 100,
                "bids": [["99", "5"]],
                "asks": [["101", "5"]],
            }
        )

    async def depth_stream(request: web.Request) -> web.StreamResponse:
        nonlocal websocket_attempts
        websocket_attempts += 1
        if websocket_attempts == 1:
            raise web.HTTPServiceUnavailable(text="forced first-attempt failure")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(
            json.dumps(
                _event_payload(
                    first_update_id=100,
                    final_update_id=101,
                    previous_final_update_id=99,
                )
            )
        )
        async for _message in ws:
            pass
        return ws

    app.router.add_get("/fapi/v1/depth", depth_snapshot)
    app.router.add_get("/public/ws/{stream}", depth_stream)
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
            "0.5",
            "--audit-dir",
            str(audit_dir),
            "--live-root",
            str(tmp_path / "Live"),
            "--ws-base",
            f"ws://127.0.0.1:{port}/public/ws",
            "--rest-base",
            f"http://127.0.0.1:{port}",
            "--fsync-every",
            "0",
            "--heartbeat-seconds",
            "1",
            "--rotation-seconds",
            "10",
            "--snapshot-timeout-seconds",
            "1",
            "--reconnect-base-seconds",
            "0.05",
            "--reconnect-max-seconds",
            "0.05",
            "--no-agg-trades",
            "--no-mark-price-updates",
        ]
    )
    try:
        exit_code = await collect_diff_depth(args)
    finally:
        await runner.cleanup()

    manifest = json.loads(
        (audit_dir / "manifest.json").read_text(encoding="utf-8")
    )
    symbol_run_dir = Path(manifest["symbol_run_dirs"]["BTCUSDT"])
    control_records = [
        json.loads(line)
        for line in (symbol_run_dir / "control.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    control_kinds = [record["kind"] for record in control_records]
    started_index = control_kinds.index("coverage_gap_started")
    ended_index = control_kinds.index("coverage_gap_ended")

    assert exit_code == 0
    assert websocket_attempts == 2
    assert control_kinds.count("coverage_gap_started") == 1
    assert control_kinds.count("coverage_gap_ended") == 1
    assert started_index < ended_index
    assert control_records[ended_index]["details"]["connection_id"] != (
        control_records[started_index - 1]["details"]["connection_id"]
    )
