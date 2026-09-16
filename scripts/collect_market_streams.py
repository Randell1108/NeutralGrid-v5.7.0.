"""Collect Binance USD-M regular market streams as a dedicated service.

The service owns only Binance's ``/market`` traffic: aggregate trades,
one-second mark prices, and kline updates.  Diff-depth remains isolated in
``collect_diff_depth.py`` on the ``/public`` route.  Every text frame is made
durable before JSON or schema parsing.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import aiohttp


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutralgrid.data.diff_depth import (  # noqa: E402
    PayloadValidationError,
    PublicMarkPrice,
    atomic_write_json,
    parse_public_agg_trade,
    parse_public_mark_price,
)
from neutralgrid.data.market_stream import (  # noqa: E402
    MARKET_MANIFEST_SCHEMA_VERSION,
    MarketKline,
    MarketPayloadError,
    MarketStreamStorage,
    parse_market_kline,
)
from scripts.collect_diff_depth import (  # noqa: E402
    CaptureTarget,
    _acquire_lock,
    _git_output,
    _load_targets,
)


logger = logging.getLogger(__name__)
LIMA = ZoneInfo("America/Lima")
DEFAULT_MARKET_WS_BASE = "wss://fstream.binance.com/market/stream"
DEFAULT_INTERVALS = ("1m", "5m", "15m", "1h")


@dataclass(frozen=True)
class MarketRunResult:
    symbol: str
    status: str
    connections: int
    wire_events: int
    parsed_records: int
    parse_errors: int
    coverage_gaps: int
    run_dir: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().replace(microsecond=0).isoformat()


def _same_mark(left: PublicMarkPrice, right: PublicMarkPrice) -> bool:
    return (
        left.symbol,
        left.event_time_ms,
        left.mark_price,
        left.index_price,
        left.estimated_settle_price,
        left.funding_rate,
        left.next_funding_time_ms,
    ) == (
        right.symbol,
        right.event_time_ms,
        right.mark_price,
        right.index_price,
        right.estimated_settle_price,
        right.funding_rate,
        right.next_funding_time_ms,
    )


def _same_kline(left: MarketKline, right: MarketKline) -> bool:
    return (
        left.symbol,
        left.open_time_ms,
        left.close_time_ms,
        left.interval,
        left.open_price,
        left.high_price,
        left.low_price,
        left.close_price,
        left.base_volume,
        left.quote_volume,
        left.trade_count,
        left.taker_buy_base_volume,
        left.taker_buy_quote_volume,
        left.closed,
    ) == (
        right.symbol,
        right.open_time_ms,
        right.close_time_ms,
        right.interval,
        right.open_price,
        right.high_price,
        right.low_price,
        right.close_price,
        right.base_volume,
        right.quote_volume,
        right.trade_count,
        right.taker_buy_base_volume,
        right.taker_buy_quote_volume,
        right.closed,
    )


class SymbolMarketCollector:
    """One symbol's reconnecting market connection and durable state."""

    def __init__(
        self,
        *,
        target: CaptureTarget,
        storage: MarketStreamStorage,
        session: aiohttp.ClientSession,
        args: argparse.Namespace,
        stop_event: asyncio.Event,
        deadline_monotonic: float | None,
    ) -> None:
        self.target = target
        self.symbol = target.symbol
        self.storage = storage
        self.session = session
        self.args = args
        self.stop_event = stop_event
        self.deadline_monotonic = deadline_monotonic
        self.started_at_utc = _utc_now_iso()
        self.current_connection_id: str | None = None
        self.last_error: str | None = None
        self.last_event_at_utc: str | None = None
        self.subscription_acknowledged = False
        self.last_manifest_write_monotonic = float("-inf")
        self.last_trade_id: int | None = None
        self.last_mark: PublicMarkPrice | None = None
        self.last_closed_kline: dict[str, MarketKline] = {}
        self.gap_started_at_utc: str | None = None

    def _time_remaining(self) -> float | None:
        if self.deadline_monotonic is None:
            return None
        return self.deadline_monotonic - time.monotonic()

    def _should_stop(self) -> bool:
        remaining = self._time_remaining()
        return self.stop_event.is_set() or (
            remaining is not None and remaining <= 0
        )

    def _manifest(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "target": asdict(self.target),
            "started_at_utc": self.started_at_utc,
            "updated_at_utc": _utc_now_iso(),
            "last_event_at_utc": self.last_event_at_utc,
            "current_connection_id": self.current_connection_id,
            "subscription_acknowledged": self.subscription_acknowledged,
            "last_error": self.last_error,
            "ws_base": self.args.ws_base,
            "intervals": list(self.args.intervals),
            "collect_agg_trades": self.args.collect_agg_trades,
            "collect_mark_prices": self.args.collect_mark_prices,
            "fsync_every": self.args.fsync_every,
            "traffic_class": "market",
            "runtime_effect": "observational_only",
            "scope_note": (
                "Binance USD-M regular market events. Public aggregate trades "
                "are never classified as private fills; only final klines are "
                "written to closed_klines.jsonl."
            ),
        }

    async def write_manifest(self, *, status: str = "running", force: bool = False) -> None:
        now = time.monotonic()
        if (
            not force
            and now - self.last_manifest_write_monotonic
            < float(self.args.manifest_heartbeat_seconds)
        ):
            return
        await asyncio.to_thread(self.storage.write_manifest, self._manifest(status))
        self.last_manifest_write_monotonic = time.monotonic()

    def _streams(self) -> list[str]:
        symbol = self.symbol.lower()
        streams = [f"{symbol}@kline_{interval}" for interval in self.args.intervals]
        if self.args.collect_agg_trades:
            streams.append(f"{symbol}@aggTrade")
        if self.args.collect_mark_prices:
            streams.append(f"{symbol}@markPrice@1s")
        return streams

    async def run(self) -> MarketRunResult:
        backoff = float(self.args.reconnect_base_seconds)
        await self.write_manifest(force=True)
        while not self._should_stop():
            connection_id = f"market-{uuid.uuid4().hex}"
            self.current_connection_id = connection_id
            self.subscription_acknowledged = False
            try:
                reason = await self._run_connection(connection_id)
                backoff = float(self.args.reconnect_base_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reason = "connection_exception"
                self.last_error = repr(exc)
                self.storage.append_control(
                    reason,
                    {
                        "symbol": self.symbol,
                        "connection_id": connection_id,
                        "at_utc": _utc_now_iso(),
                        "error": repr(exc),
                    },
                )
                logger.exception("%s market connection failed", self.symbol)
            self.current_connection_id = None
            await self.write_manifest(force=True)
            if self._should_stop() or reason == "capture_complete":
                break
            if self.gap_started_at_utc is None:
                self.gap_started_at_utc = _utc_now_iso()
                self.storage.counters["coverage_gaps"] += 1
                self.storage.append_control(
                    "coverage_gap_started",
                    {
                        "symbol": self.symbol,
                        "at_utc": self.gap_started_at_utc,
                        "reason": reason,
                    },
                )
            sleep_seconds = min(
                backoff, float(self.args.reconnect_max_seconds)
            )
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(), timeout=sleep_seconds
                )
            except TimeoutError:
                pass
            backoff = min(
                backoff * 2.0, float(self.args.reconnect_max_seconds)
            )

        counters = self.storage.counters
        parsed_records = (
            counters["public_agg_trades"]
            + counters["public_mark_price_updates"]
            + counters["closed_klines"]
        )
        if not self.subscription_acknowledged:
            status = "complete_subscription_unacknowledged"
        elif counters["parse_errors"] or counters["coverage_gaps"]:
            status = "complete_with_labelled_gaps"
        elif parsed_records == 0:
            status = "complete_no_structured_records"
        else:
            status = "complete_contiguous"
        await self.write_manifest(status=status, force=True)
        return MarketRunResult(
            symbol=self.symbol,
            status=status,
            connections=counters["connections"],
            wire_events=counters["wire_events"],
            parsed_records=parsed_records,
            parse_errors=counters["parse_errors"],
            coverage_gaps=counters["coverage_gaps"],
            run_dir=str(self.storage.run_dir),
        )

    async def _run_connection(self, connection_id: str) -> str:
        ws = await self.session.ws_connect(
            self.args.ws_base,
            heartbeat=float(self.args.heartbeat_seconds),
            autoping=True,
            max_msg_size=int(self.args.max_message_bytes),
        )
        self.storage.counters["connections"] += 1
        request_id = int(time.time_ns() % 2_147_483_647)
        await ws.send_json(
            {"method": "SUBSCRIBE", "params": self._streams(), "id": request_id}
        )
        connected_monotonic = time.monotonic()
        wire_sequence = 0
        reason = "market_connection_closed"
        try:
            while not self._should_stop():
                if (
                    time.monotonic() - connected_monotonic
                    >= float(self.args.rotation_seconds)
                ):
                    reason = "scheduled_rotation"
                    break
                receive_timeout = 1.0
                remaining = self._time_remaining()
                if remaining is not None:
                    receive_timeout = max(0.01, min(receive_timeout, remaining))
                try:
                    message = await ws.receive(timeout=receive_timeout)
                except TimeoutError:
                    await self.write_manifest()
                    continue
                if message.type == aiohttp.WSMsgType.TEXT:
                    wire_sequence += 1
                    received_at = _utc_now_iso()
                    received_monotonic_ns = time.monotonic_ns()
                    raw_text = str(message.data)
                    self.storage.append_wire(
                        connection_id=connection_id,
                        wire_sequence=wire_sequence,
                        received_at_utc=received_at,
                        received_monotonic_ns=received_monotonic_ns,
                        raw_text=raw_text,
                        raw_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                    )
                    try:
                        decoded = json.loads(raw_text)
                    except json.JSONDecodeError as exc:
                        self._parse_failure(
                            connection_id, wire_sequence, received_at, repr(exc)
                        )
                        reason = "json_parse_error"
                        break
                    if not isinstance(decoded, dict):
                        self._parse_failure(
                            connection_id,
                            wire_sequence,
                            received_at,
                            "payload root is not an object",
                        )
                        reason = "payload_parse_error"
                        break
                    if "result" in decoded and "id" in decoded:
                        if decoded.get("id") != request_id or decoded.get("result") is not None:
                            self._parse_failure(
                                connection_id,
                                wire_sequence,
                                received_at,
                                "subscription acknowledgement mismatch",
                            )
                            reason = "subscription_rejected"
                            break
                        self.subscription_acknowledged = True
                        self.last_error = None
                        if self.gap_started_at_utc is not None:
                            self.storage.append_control(
                                "coverage_gap_ended",
                                {
                                    "symbol": self.symbol,
                                    "gap_started_at_utc": self.gap_started_at_utc,
                                    "at_utc": received_at,
                                },
                            )
                            self.gap_started_at_utc = None
                        self.storage.append_control(
                            "subscription_acknowledged",
                            {
                                "symbol": self.symbol,
                                "connection_id": connection_id,
                                "at_utc": received_at,
                                "streams": self._streams(),
                            },
                        )
                        await self.write_manifest(force=True)
                        continue
                    nested = decoded.get("data", decoded)
                    event_type = nested.get("e") if isinstance(nested, Mapping) else None
                    try:
                        accepted = self._process_event(
                            decoded,
                            event_type=event_type,
                            connection_id=connection_id,
                            wire_sequence=wire_sequence,
                            received_at_utc=received_at,
                            received_monotonic_ns=received_monotonic_ns,
                        )
                    except (PayloadValidationError, MarketPayloadError) as exc:
                        self._parse_failure(
                            connection_id, wire_sequence, received_at, str(exc)
                        )
                        reason = "schema_parse_error"
                        break
                    if accepted:
                        self.last_event_at_utc = received_at
                    await self.write_manifest()
                    continue
                if message.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED}:
                    reason = f"websocket_closed_{ws.close_code}"
                    break
                if message.type == aiohttp.WSMsgType.ERROR:
                    reason = f"websocket_error_{ws.exception()!r}"
                    break
                if message.type in {aiohttp.WSMsgType.PING, aiohttp.WSMsgType.PONG}:
                    continue
                reason = "unexpected_websocket_frame"
                break
            if self._should_stop():
                reason = "capture_complete"
        finally:
            self.storage.append_control(
                "connection_ended",
                {
                    "symbol": self.symbol,
                    "connection_id": connection_id,
                    "at_utc": _utc_now_iso(),
                    "reason": reason,
                    "wire_events": wire_sequence,
                },
            )
            if not ws.closed:
                await ws.close()
        return reason

    def _parse_failure(
        self,
        connection_id: str,
        wire_sequence: int,
        received_at_utc: str,
        error: str,
    ) -> None:
        self.storage.counters["parse_errors"] += 1
        self.last_error = error
        self.storage.append_control(
            "parse_error",
            {
                "symbol": self.symbol,
                "connection_id": connection_id,
                "wire_sequence": wire_sequence,
                "at_utc": received_at_utc,
                "error": error,
            },
        )

    def _process_event(
        self,
        payload: Mapping[str, Any],
        *,
        event_type: Any,
        connection_id: str,
        wire_sequence: int,
        received_at_utc: str,
        received_monotonic_ns: int,
    ) -> bool:
        if event_type == "aggTrade":
            trade = parse_public_agg_trade(
                payload,
                expected_symbol=self.symbol,
                connection_id=connection_id,
                wire_sequence=wire_sequence,
                received_at_utc=received_at_utc,
                received_monotonic_ns=received_monotonic_ns,
            )
            if self.last_trade_id is not None and trade.aggregate_trade_id <= self.last_trade_id:
                counter = (
                    "agg_trade_duplicates_dropped"
                    if trade.aggregate_trade_id == self.last_trade_id
                    else "agg_trade_out_of_order_dropped"
                )
                self.storage.counters[counter] += 1
                return False
            if (
                self.last_trade_id is not None
                and trade.aggregate_trade_id != self.last_trade_id + 1
            ):
                self.storage.counters["agg_trade_id_discontinuities"] += 1
            self.storage.append_trade(trade)
            self.last_trade_id = trade.aggregate_trade_id
            return True
        if event_type == "markPriceUpdate":
            mark = parse_public_mark_price(
                payload,
                expected_symbol=self.symbol,
                connection_id=connection_id,
                wire_sequence=wire_sequence,
                received_at_utc=received_at_utc,
                received_monotonic_ns=received_monotonic_ns,
            )
            if self.last_mark is not None and mark.event_time_ms <= self.last_mark.event_time_ms:
                if mark.event_time_ms == self.last_mark.event_time_ms and _same_mark(mark, self.last_mark):
                    self.storage.counters["mark_price_duplicates_dropped"] += 1
                    return False
                if mark.event_time_ms < self.last_mark.event_time_ms:
                    self.storage.counters["mark_price_out_of_order_dropped"] += 1
                    return False
                raise MarketPayloadError("conflicting mark-price duplicate")
            self.storage.append_mark_price(mark)
            self.last_mark = mark
            return True
        if event_type == "kline":
            kline = parse_market_kline(
                payload,
                expected_symbol=self.symbol,
                expected_intervals=set(self.args.intervals),
                connection_id=connection_id,
                wire_sequence=wire_sequence,
                received_at_utc=received_at_utc,
                received_monotonic_ns=received_monotonic_ns,
            )
            if not kline.closed:
                self.storage.counters["non_final_kline_updates"] += 1
                return False
            previous = self.last_closed_kline.get(kline.interval)
            if previous is not None and kline.open_time_ms <= previous.open_time_ms:
                if kline.open_time_ms == previous.open_time_ms and _same_kline(kline, previous):
                    self.storage.counters["closed_kline_duplicates_dropped"] += 1
                    return False
                if kline.open_time_ms < previous.open_time_ms:
                    self.storage.append_control(
                        "closed_kline_out_of_order_dropped",
                        {
                            "symbol": self.symbol,
                            "interval": kline.interval,
                            "open_time_ms": kline.open_time_ms,
                            "last_open_time_ms": previous.open_time_ms,
                        },
                    )
                    return False
                self.storage.counters["closed_kline_conflicts"] += 1
                raise MarketPayloadError("conflicting closed-kline duplicate")
            self.storage.append_closed_kline(kline)
            self.last_closed_kline[kline.interval] = kline
            return True
        self.storage.counters["unexpected_events"] += 1
        self.storage.append_control(
            "unexpected_event_preserved",
            {
                "symbol": self.symbol,
                "connection_id": connection_id,
                "wire_sequence": wire_sequence,
                "at_utc": received_at_utc,
                "event_type": str(event_type),
            },
        )
        return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--symbols", nargs="+")
    source.add_argument("--input", help="CSV containing symbol and optional IDs")
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--live-root", type=Path, default=ROOT / "Live")
    parser.add_argument(
        "--ingestion-date",
        help="Supervisor-frozen America/Lima date in YYYY-MM-DD form",
    )
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--ws-base", default=DEFAULT_MARKET_WS_BASE)
    parser.add_argument("--intervals", nargs="+", default=list(DEFAULT_INTERVALS))
    parser.add_argument("--no-agg-trades", action="store_false", dest="collect_agg_trades", default=True)
    parser.add_argument("--no-mark-prices", action="store_false", dest="collect_mark_prices", default=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=180.0)
    parser.add_argument("--manifest-heartbeat-seconds", type=float, default=5.0)
    parser.add_argument("--rotation-seconds", type=float, default=23 * 3600)
    parser.add_argument("--reconnect-base-seconds", type=float, default=1.0)
    parser.add_argument("--reconnect-max-seconds", type=float, default=30.0)
    parser.add_argument("--max-message-bytes", type=int, default=2_000_000)
    parser.add_argument("--fsync-every", type=int, default=1)
    args = parser.parse_args(argv)
    if args.duration_seconds < 0:
        parser.error("--duration-seconds must be >= 0")
    if args.ingestion_date is not None:
        try:
            parsed_date = date.fromisoformat(args.ingestion_date)
        except ValueError:
            parser.error("--ingestion-date must be YYYY-MM-DD")
        if parsed_date.isoformat() != args.ingestion_date:
            parser.error("--ingestion-date must use canonical YYYY-MM-DD form")
    if not args.intervals or any(not str(item).strip() for item in args.intervals):
        parser.error("--intervals must contain non-empty values")
    if len(set(args.intervals)) != len(args.intervals):
        parser.error("--intervals must not contain duplicates")
    for field in (
        "heartbeat_seconds",
        "manifest_heartbeat_seconds",
        "rotation_seconds",
        "reconnect_base_seconds",
        "reconnect_max_seconds",
        "max_message_bytes",
    ):
        if float(getattr(args, field)) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.fsync_every < 0:
        parser.error("--fsync-every must be >= 0")
    return args


async def collect_market_streams(args: argparse.Namespace) -> int:
    targets = _load_targets(symbols=args.symbols, input_path=args.input)
    started_at = _utc_now()
    run_id = started_at.strftime("market_stream_%Y%m%d_%H%M%S_%f") + f"_{os.getpid()}"
    live_date = args.ingestion_date or started_at.astimezone(LIMA).strftime("%Y-%m-%d")
    audit_dir = args.audit_dir or ROOT / "outputs" / "audits" / run_id
    audit_dir.mkdir(parents=True, exist_ok=True)
    stop_path = audit_dir / "STOP"
    stop_path.unlink(missing_ok=True)
    lock_path = audit_dir / "collector.lock"
    lock_descriptor = _acquire_lock(lock_path)
    stop_event = asyncio.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    live_root = args.live_root.resolve()
    storages: dict[str, MarketStreamStorage] = {}
    for target in targets:
        run_dir = (live_root / live_date / target.symbol / "market_stream" / run_id).resolve()
        try:
            run_dir.relative_to(live_root)
        except ValueError as exc:
            raise RuntimeError(f"market stream path escapes live root: {run_dir}") from exc
        storages[target.symbol] = MarketStreamStorage(
            run_dir,
            symbol=target.symbol,
            run_id=run_id,
            fsync_every=int(args.fsync_every),
        )

    run_manifest = audit_dir / "manifest.json"
    manifest_last_write = float("-inf")
    git_head = _git_output(["rev-parse", "--short", "HEAD"])
    git_status_short = _git_output(["status", "--short"])

    async def write_run_manifest(status: str, *, force: bool = False, **extra: Any) -> None:
        nonlocal manifest_last_write
        now = time.monotonic()
        if (
            not force
            and now - manifest_last_write < float(args.manifest_heartbeat_seconds)
        ):
            return
        payload = {
            "schema_version": MARKET_MANIFEST_SCHEMA_VERSION,
            "status": status,
            "service": "market",
            "traffic_class": "market",
            "run_id": run_id,
            "started_at_utc": started_at.replace(microsecond=0).isoformat(),
            "updated_at_utc": _utc_now_iso(),
            "live_date_lima": live_date,
            "ingestion_timezone": "America/Lima",
            "ingestion_date_basis": (
                "supervisor_frozen" if args.ingestion_date else "collector_start"
            ),
            "live_root": str(live_root),
            "collector_pid": os.getpid(),
            "targets": [asdict(target) for target in targets],
            "symbol_run_dirs": {
                symbol: str(storage.run_dir) for symbol, storage in storages.items()
            },
            "symbol_counters": {
                symbol: dict(storage.counters) for symbol, storage in storages.items()
            },
            "ws_base": args.ws_base,
            "intervals": list(args.intervals),
            "git_head": git_head,
            "git_status_short": git_status_short,
            "runtime_effect": "observational_only",
            **extra,
        }
        await asyncio.to_thread(atomic_write_json, run_manifest, payload)
        manifest_last_write = time.monotonic()

    deadline = (
        time.monotonic() + float(args.duration_seconds)
        if args.duration_seconds > 0
        else None
    )
    results: list[MarketRunResult] = []
    await write_run_manifest("starting", force=True)
    try:
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            collectors = {
                target.symbol: SymbolMarketCollector(
                    target=target,
                    storage=storages[target.symbol],
                    session=session,
                    args=args,
                    stop_event=stop_event,
                    deadline_monotonic=deadline,
                )
                for target in targets
            }
            tasks = {
                symbol: asyncio.create_task(collector.run())
                for symbol, collector in collectors.items()
            }
            while tasks and not all(task.done() for task in tasks.values()):
                if stop_path.exists():
                    stop_event.set()
                await write_run_manifest("running")
                await asyncio.sleep(0.5)
            gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
            failures: list[dict[str, str]] = []
            for symbol, result in zip(tasks, gathered):
                if isinstance(result, BaseException):
                    failures.append({"symbol": symbol, "error": repr(result)})
                else:
                    results.append(result)
            status = (
                "failed"
                if failures
                else (
                    "complete"
                    if all(result.status == "complete_contiguous" for result in results)
                    else "complete_with_labelled_gaps"
                )
            )
            await write_run_manifest(
                status,
                force=True,
                completed_at_utc=_utc_now_iso(),
                results=[asdict(result) for result in results],
                failures=failures,
            )
            return 2 if failures else 0
    finally:
        stop_event.set()
        for storage in storages.values():
            storage.close()
        os.close(lock_descriptor)
        lock_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    return asyncio.run(collect_market_streams(args))


if __name__ == "__main__":
    raise SystemExit(main())
