"""Collect prospective, sequence-verified Binance USD-M diff-depth events.

This collector starts the documented public depth connection and a separate
documented market connection per symbol. The market connection subscribes to
``<symbol>@aggTrade`` and ``<symbol>@markPrice@1s``. It persists every text
frame before parsing, bootstraps a local order book from
``/fapi/v1/depth``, detects every ``pu`` sequence break, labels every connection
coverage boundary, and resynchronizes from a fresh snapshot.

It does not reconstruct events from before startup and it does not collect
private order/fill events. Public aggregate trades are stored separately and
never treated as private fills.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, cast
from zoneinfo import ZoneInfo

import aiohttp


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutralgrid.data.diff_depth import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    DepthSequenceEngine,
    DepthSnapshot,
    DiffDepthError,
    EngineAction,
    PayloadValidationError,
    SequenceBufferOverflow,
    SymbolCaptureStorage,
    atomic_write_json,
    parse_depth_snapshot,
    parse_diff_depth_event,
    parse_public_mark_price,
    parse_public_agg_trade,
)
from neutralgrid.live.decision.l2_risk import (  # noqa: E402
    L2IntervalAccumulator,
    build_l2_risk_record,
)


logger = logging.getLogger(__name__)
LIMA = ZoneInfo("America/Lima")
SYMBOL_RE = re.compile(r"^[A-Z0-9]+USDT$")
DEFAULT_WS_BASE = "wss://fstream.binance.com/public/ws"
DEFAULT_MARKET_WS_BASE = "wss://fstream.binance.com/market/stream"
DEFAULT_REST_BASE = "https://fapi.binance.com"
DEFAULT_UPDATE_SPEED = "100ms"


@dataclass(frozen=True)
class CaptureTarget:
    """One explicitly registered symbol and optional bot identifiers."""

    symbol: str
    strategy_id: str | None = None
    candidate_id: str | None = None
    source_path: str | None = None
    source_row_index: int | None = None


@dataclass(frozen=True)
class SymbolRunResult:
    symbol: str
    status: str
    applied_events: int
    sequence_gaps: int
    parse_errors: int
    connections: int
    public_streams_ready: bool
    run_dir: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().replace(microsecond=0).isoformat()


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _validate_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not SYMBOL_RE.fullmatch(symbol):
        raise ValueError(f"invalid Binance USD-M symbol: {value!r}")
    return symbol


def _load_targets(
    *,
    symbols: Sequence[str] | None,
    input_path: str | None,
) -> list[CaptureTarget]:
    targets: list[CaptureTarget] = []
    if symbols:
        targets = [CaptureTarget(symbol=_validate_symbol(value)) for value in symbols]
    elif input_path:
        path = Path(input_path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "symbol" not in reader.fieldnames:
                raise ValueError(f"{path} must contain a symbol column")
            for row_index, row in enumerate(reader):
                symbol = _validate_symbol(str(row.get("symbol", "")))
                strategy = (
                    str(row.get("strategy_id", "")).strip()
                    or str(row.get("strategy_number", "")).strip()
                    or None
                )
                candidate = str(row.get("candidate_id", "")).strip() or None
                targets.append(
                    CaptureTarget(
                        symbol=symbol,
                        strategy_id=strategy,
                        candidate_id=candidate,
                        source_path=str(path),
                        source_row_index=row_index,
                    )
                )
    if not targets:
        raise ValueError("at least one --symbols value or --input row is required")
    seen: set[str] = set()
    unique: list[CaptureTarget] = []
    for target in targets:
        if target.symbol in seen:
            raise ValueError(f"duplicate capture symbol: {target.symbol}")
        seen.add(target.symbol)
        unique.append(target)
    return unique


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_lock(lock_path: Path) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            existing_pid = int(lock_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            existing_pid = -1
        if _pid_is_running(existing_pid):
            raise DiffDepthError(
                f"diff-depth collector already running with PID {existing_pid}"
            )
        lock_path.unlink(missing_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    os.fsync(descriptor)
    return descriptor


def _actions_need_snapshot(actions: Sequence[EngineAction]) -> bool:
    return any(
        action.kind
        in {
            "sequence_gap",
            "snapshot_behind_buffer",
            "book_invariant_failure",
        }
        for action in actions
    )


class SymbolDiffDepthCollector:
    """Network lifecycle and durable evidence for one symbol."""

    def __init__(
        self,
        *,
        target: CaptureTarget,
        storage: SymbolCaptureStorage,
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
        self.last_error: str | None = None
        self.last_connection_ended_at_utc: str | None = None
        self.current_connection_id: str | None = None
        self.current_phase = "starting"
        self.connection_opened_for_attempt = False
        self.last_manifest_write_monotonic = float("-inf")
        self.current_trade_subscription_acknowledged = False
        self.current_mark_price_subscription_acknowledged = False
        self.market_connection_error: str | None = None

    def _time_remaining(self) -> float | None:
        if self.deadline_monotonic is None:
            return None
        return self.deadline_monotonic - time.monotonic()

    def _should_stop(self) -> bool:
        if self.stop_event.is_set():
            return True
        remaining = self._time_remaining()
        return remaining is not None and remaining <= 0

    def _manifest_payload(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "target": asdict(self.target),
            "started_at_utc": self.started_at_utc,
            "updated_at_utc": _utc_now_iso(),
            "current_connection_id": self.current_connection_id,
            "current_phase": self.current_phase,
            "last_error": self.last_error,
            "update_speed": self.args.update_speed,
            "snapshot_limit": self.args.snapshot_limit,
            "ws_base": self.args.ws_base,
            "market_ws_base": self.args.market_ws_base,
            "rest_base": self.args.rest_base,
            "max_buffer_events": self.args.max_buffer_events,
            "fsync_every": self.args.fsync_every,
            "manifest_heartbeat_seconds": self.args.manifest_heartbeat_seconds,
            "shutdown_bootstrap_grace_seconds": (
                self.args.shutdown_bootstrap_grace_seconds
            ),
            "risk_snapshot_seconds": self.args.risk_snapshot_seconds,
            "risk_top_levels": self.args.risk_top_levels,
            "collect_agg_trades": self.args.collect_agg_trades,
            "collect_mark_price_updates": self.args.collect_mark_price_updates,
            "trade_subscription_acknowledged": (
                self.current_trade_subscription_acknowledged
            ),
            "mark_price_subscription_acknowledged": (
                self.current_mark_price_subscription_acknowledged
            ),
            "market_connection_error": self.market_connection_error,
            "event_complete_scope": (
                "Every successfully received Binance diff-depth frame is persisted "
                "before parsing. Completeness is per contiguous, bootstrapped "
                "sequence segment; connection and sequence gaps are labelled."
            ),
            "public_trade_scope": (
                "Every successfully received aggregate-trade frame on the "
                "documented market WebSocket is persisted before parsing. "
                "Aggregate-ID discontinuities are observations, not proof of "
                "missing events."
                if self.args.collect_agg_trades
                else "Aggregate-trade subscription disabled for this run."
            ),
            "public_mark_price_scope": (
                "Every successfully received one-second mark-price frame on the "
                "documented market WebSocket is persisted before parsing and as "
                "a validated structured record."
                if self.args.collect_mark_price_updates
                else "One-second mark-price subscription disabled for this run."
            ),
        }

    async def write_heartbeat(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if (
            not force
            and now - self.last_manifest_write_monotonic
            < float(self.args.manifest_heartbeat_seconds)
        ):
            return
        await asyncio.to_thread(
            self.storage.write_manifest,
            self._manifest_payload("running"),
        )
        self.last_manifest_write_monotonic = time.monotonic()

    async def run(self) -> SymbolRunResult:
        backoff = float(self.args.reconnect_base_seconds)
        coverage_gap_started_at: str | None = None
        await self.write_heartbeat(force=True)
        market_task: asyncio.Task[None] | None = None
        if self.args.collect_agg_trades or self.args.collect_mark_price_updates:
            market_task = asyncio.create_task(self._run_market_events())
        while not self._should_stop():
            connection_id = uuid.uuid4().hex
            self.current_connection_id = connection_id
            self.connection_opened_for_attempt = False
            try:
                reason = await self._run_connection(
                    connection_id,
                    coverage_gap_started_at,
                )
                backoff = float(self.args.reconnect_base_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reason = "connection_exception"
                self.last_error = repr(exc)
                self.storage.append_control(
                    "connection_exception",
                    {
                        "symbol": self.symbol,
                        "connection_id": connection_id,
                        "at_utc": _utc_now_iso(),
                        "error": repr(exc),
                    },
                )
                logger.exception("%s diff-depth connection failed", self.symbol)
            if self.connection_opened_for_attempt:
                coverage_gap_started_at = None

            self.last_connection_ended_at_utc = _utc_now_iso()
            self.current_connection_id = None
            self.current_phase = "between_connections"
            await self.write_heartbeat(force=True)
            if self._should_stop() or reason == "capture_complete":
                break
            if coverage_gap_started_at is None:
                coverage_gap_started_at = self.last_connection_ended_at_utc
                self.storage.append_control(
                    "coverage_gap_started",
                    {
                        "symbol": self.symbol,
                        "reason": reason,
                        "gap_started_at_utc": coverage_gap_started_at,
                    },
                )
            else:
                self.storage.append_control(
                    "coverage_gap_continues",
                    {
                        "symbol": self.symbol,
                        "connection_id": connection_id,
                        "reason": reason,
                        "gap_started_at_utc": coverage_gap_started_at,
                        "attempt_ended_at_utc": (
                            self.last_connection_ended_at_utc
                        ),
                    },
                )
            sleep_seconds = min(
                backoff,
                float(self.args.reconnect_max_seconds),
            )
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=sleep_seconds,
                )
            except TimeoutError:
                pass
            backoff = min(
                backoff * 2.0,
                float(self.args.reconnect_max_seconds),
            )

        if market_task is not None:
            try:
                await market_task
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.market_connection_error = repr(exc)
                self.storage.append_control(
                    "market_connection_task_failed",
                    {
                        "symbol": self.symbol,
                        "at_utc": _utc_now_iso(),
                        "error": repr(exc),
                    },
                )

        counters = self.storage.counters
        public_streams_ready = (
            self.market_connection_error is None
            and (
                not self.args.collect_agg_trades
                or self.current_trade_subscription_acknowledged
            )
            and (
                not self.args.collect_mark_price_updates
                or (
                    self.current_mark_price_subscription_acknowledged
                    and counters["public_mark_price_updates"] > 0
                )
            )
        )
        if counters["events_applied"] <= 0:
            status = "complete_no_valid_segment"
        elif not public_streams_ready:
            status = "complete_public_stream_subscription_unacknowledged"
        elif (
            counters["sequence_gaps"] > 0
            or counters["parse_errors"] > 0
            or counters["market_coverage_gaps"] > 0
            or counters["market_parse_errors"] > 0
            or counters["connections"] > 1
        ):
            status = "complete_with_labelled_gaps"
        else:
            status = "complete_contiguous"
        self.current_phase = "stopped"
        final_payload = self._manifest_payload(status)
        final_payload["completed_at_utc"] = _utc_now_iso()
        await asyncio.to_thread(self.storage.write_manifest, final_payload)
        return SymbolRunResult(
            symbol=self.symbol,
            status=status,
            applied_events=counters["events_applied"],
            sequence_gaps=counters["sequence_gaps"],
            parse_errors=counters["parse_errors"],
            connections=counters["connections"],
            public_streams_ready=public_streams_ready,
            run_dir=str(self.storage.run_dir),
        )

    async def _run_market_events(self) -> None:
        """Collect trade and mark events from Binance's market endpoint."""

        backoff = float(self.args.reconnect_base_seconds)
        coverage_gap_started_at: str | None = None
        while not self._should_stop():
            connection_id = f"market-{uuid.uuid4().hex}"
            try:
                reason = await self._run_market_connection(connection_id)
                backoff = float(self.args.reconnect_base_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reason = "market_connection_exception"
                self.market_connection_error = repr(exc)
                self.storage.append_control(
                    reason,
                    {
                        "symbol": self.symbol,
                        "connection_id": connection_id,
                        "at_utc": _utc_now_iso(),
                        "error": repr(exc),
                    },
                )
            if self._should_stop() or reason == "capture_complete":
                break
            ended_at_utc = _utc_now_iso()
            if coverage_gap_started_at is None:
                coverage_gap_started_at = ended_at_utc
                self.storage.counters["market_coverage_gaps"] += 1
                self.storage.append_control(
                    "market_coverage_gap_started",
                    {
                        "symbol": self.symbol,
                        "connection_id": connection_id,
                        "reason": reason,
                        "gap_started_at_utc": coverage_gap_started_at,
                    },
                )
            else:
                self.storage.append_control(
                    "market_coverage_gap_continues",
                    {
                        "symbol": self.symbol,
                        "connection_id": connection_id,
                        "reason": reason,
                        "gap_started_at_utc": coverage_gap_started_at,
                        "attempt_ended_at_utc": ended_at_utc,
                    },
                )
            sleep_seconds = min(
                backoff,
                float(self.args.reconnect_max_seconds),
            )
            remaining = self._time_remaining()
            if remaining is not None:
                sleep_seconds = max(0.0, min(sleep_seconds, remaining))
            if sleep_seconds > 0:
                try:
                    await asyncio.wait_for(
                        self.stop_event.wait(),
                        timeout=sleep_seconds,
                    )
                except TimeoutError:
                    pass
            backoff = min(
                backoff * 2.0,
                float(self.args.reconnect_max_seconds),
            )

    async def _run_market_connection(self, connection_id: str) -> str:
        url = self.args.market_ws_base.rstrip("/")
        self.storage.append_control(
            "market_connection_opening",
            {
                "symbol": self.symbol,
                "connection_id": connection_id,
                "at_utc": _utc_now_iso(),
                "url": url,
            },
        )
        ws = await self.session.ws_connect(
            url,
            heartbeat=float(self.args.heartbeat_seconds),
            autoping=True,
            max_msg_size=int(self.args.max_message_bytes),
        )
        self.storage.counters["market_connections"] += 1
        request_id = f"market-{connection_id[-17:]}"
        streams: list[str] = []
        if self.args.collect_agg_trades:
            streams.append(f"{self.symbol.lower()}@aggTrade")
        if self.args.collect_mark_price_updates:
            streams.append(f"{self.symbol.lower()}@markPrice@1s")
        self.current_trade_subscription_acknowledged = False
        self.current_mark_price_subscription_acknowledged = False
        await ws.send_json(
            {"method": "SUBSCRIBE", "params": streams, "id": request_id}
        )
        self.storage.append_control(
            "market_event_subscription_requested",
            {
                "symbol": self.symbol,
                "connection_id": connection_id,
                "at_utc": _utc_now_iso(),
                "request_id": request_id,
                "streams": streams,
            },
        )
        connected_monotonic = time.monotonic()
        wire_sequence = 0
        last_public_agg_trade_id: int | None = None
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
                    await self.write_heartbeat()
                    continue
                if message.type == aiohttp.WSMsgType.TEXT:
                    wire_sequence += 1
                    received_at_utc = _utc_now_iso()
                    received_monotonic_ns = time.monotonic_ns()
                    raw_text = str(message.data)
                    self.storage.append_wire(
                        connection_id=connection_id,
                        wire_sequence=wire_sequence,
                        received_at_utc=received_at_utc,
                        received_monotonic_ns=received_monotonic_ns,
                        raw_text=raw_text,
                    )
                    try:
                        decoded = json.loads(raw_text)
                    except json.JSONDecodeError as exc:
                        self.storage.counters["market_parse_errors"] += 1
                        self.storage.append_control(
                            "market_parse_error",
                            {
                                "symbol": self.symbol,
                                "connection_id": connection_id,
                                "wire_sequence": wire_sequence,
                                "at_utc": received_at_utc,
                                "error": f"WebSocket frame is not valid JSON: {exc}",
                            },
                        )
                        reason = "market_parse_error"
                        break
                    if not isinstance(decoded, dict):
                        self.storage.counters["market_parse_errors"] += 1
                        reason = "market_parse_error"
                        self.storage.append_control(
                            reason,
                            {
                                "symbol": self.symbol,
                                "connection_id": connection_id,
                                "wire_sequence": wire_sequence,
                                "at_utc": received_at_utc,
                                "error": "WebSocket payload is not an object",
                            },
                        )
                        break
                    if "result" in decoded and "id" in decoded:
                        if decoded.get("id") != request_id or decoded.get("result") is not None:
                            reason = "market_event_subscription_rejected"
                            self.storage.append_control(
                                reason,
                                {
                                    "symbol": self.symbol,
                                    "connection_id": connection_id,
                                    "wire_sequence": wire_sequence,
                                    "at_utc": received_at_utc,
                                    "payload": decoded,
                                },
                            )
                            break
                        self.current_trade_subscription_acknowledged = bool(
                            self.args.collect_agg_trades
                        )
                        self.current_mark_price_subscription_acknowledged = bool(
                            self.args.collect_mark_price_updates
                        )
                        self.market_connection_error = None
                        self.storage.append_control(
                            "market_event_subscription_acknowledged",
                            {
                                "symbol": self.symbol,
                                "connection_id": connection_id,
                                "wire_sequence": wire_sequence,
                                "at_utc": received_at_utc,
                                "request_id": request_id,
                            },
                        )
                        await self.write_heartbeat(force=True)
                        continue
                    nested = decoded.get("data", decoded)
                    event_type = nested.get("e") if isinstance(nested, dict) else None
                    if event_type == "markPriceUpdate":
                        try:
                            mark_price = parse_public_mark_price(
                                decoded,
                                expected_symbol=self.symbol,
                                connection_id=connection_id,
                                wire_sequence=wire_sequence,
                                received_at_utc=received_at_utc,
                                received_monotonic_ns=received_monotonic_ns,
                            )
                        except PayloadValidationError as exc:
                            self.storage.counters["market_parse_errors"] += 1
                            reason = "public_mark_price_parse_error"
                            self.storage.append_control(
                                reason,
                                {
                                    "symbol": self.symbol,
                                    "connection_id": connection_id,
                                    "wire_sequence": wire_sequence,
                                    "at_utc": received_at_utc,
                                    "error": str(exc),
                                },
                            )
                            break
                        self.storage.append_public_mark_price(mark_price)
                        await self.write_heartbeat()
                        continue
                    if event_type == "aggTrade":
                        try:
                            public_trade = parse_public_agg_trade(
                                decoded,
                                expected_symbol=self.symbol,
                                connection_id=connection_id,
                                wire_sequence=wire_sequence,
                                received_at_utc=received_at_utc,
                                received_monotonic_ns=received_monotonic_ns,
                            )
                        except PayloadValidationError as exc:
                            self.storage.counters["market_parse_errors"] += 1
                            reason = "public_trade_parse_error"
                            self.storage.append_control(
                                reason,
                                {
                                    "symbol": self.symbol,
                                    "connection_id": connection_id,
                                    "wire_sequence": wire_sequence,
                                    "at_utc": received_at_utc,
                                    "error": str(exc),
                                },
                            )
                            break
                        current_id = public_trade.aggregate_trade_id
                        if last_public_agg_trade_id is not None and current_id <= last_public_agg_trade_id:
                            kind = (
                                "public_trade_duplicate_dropped"
                                if current_id == last_public_agg_trade_id
                                else "public_trade_out_of_order_dropped"
                            )
                            self.storage.append_control(
                                kind,
                                {
                                    "symbol": self.symbol,
                                    "connection_id": connection_id,
                                    "wire_sequence": wire_sequence,
                                    "at_utc": received_at_utc,
                                    "aggregate_trade_id": current_id,
                                    "last_accepted_aggregate_trade_id": last_public_agg_trade_id,
                                },
                            )
                            await self.write_heartbeat()
                            continue
                        discontinuity = (
                            last_public_agg_trade_id is not None
                            and current_id != last_public_agg_trade_id + 1
                        )
                        self.storage.append_public_agg_trade(
                            public_trade,
                            id_discontinuity=discontinuity,
                            segment_id=None,
                        )
                        last_public_agg_trade_id = current_id
                        await self.write_heartbeat()
                        continue
                    reason = "unexpected_market_event"
                    self.storage.append_control(
                        reason,
                        {
                            "symbol": self.symbol,
                            "connection_id": connection_id,
                            "wire_sequence": wire_sequence,
                            "at_utc": received_at_utc,
                            "event_type": event_type,
                        },
                    )
                    break
                if message.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED}:
                    reason = f"market_websocket_closed_{ws.close_code}"
                    break
                if message.type == aiohttp.WSMsgType.ERROR:
                    reason = f"market_websocket_error_{ws.exception()!r}"
                    break
                if message.type in {aiohttp.WSMsgType.PING, aiohttp.WSMsgType.PONG}:
                    continue
                reason = "unexpected_market_websocket_frame"
                break
            if self._should_stop():
                reason = "capture_complete"
        finally:
            self.storage.append_control(
                "market_connection_ended",
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

    async def _run_connection(
        self,
        connection_id: str,
        coverage_gap_started_at: str | None,
    ) -> str:
        url = (
            f"{self.args.ws_base.rstrip('/')}/"
            f"{self.symbol.lower()}@depth@{self.args.update_speed}"
        )
        self.current_phase = "connecting"
        self.storage.append_control(
            "connection_opening",
            {
                "symbol": self.symbol,
                "connection_id": connection_id,
                "at_utc": _utc_now_iso(),
                "url": url,
            },
        )
        ws = await self.session.ws_connect(
            url,
            heartbeat=float(self.args.heartbeat_seconds),
            autoping=True,
            max_msg_size=int(self.args.max_message_bytes),
        )
        self.connection_opened_for_attempt = True
        connected_at_utc = _utc_now_iso()
        if coverage_gap_started_at is not None:
            self.storage.append_control(
                "coverage_gap_ended",
                {
                    "symbol": self.symbol,
                    "connection_id": connection_id,
                    "gap_started_at_utc": coverage_gap_started_at,
                    "gap_ended_at_utc": connected_at_utc,
                },
            )
        connected_monotonic = time.monotonic()
        wire_sequence = 0
        last_public_agg_trade_id: int | None = None
        engine = DepthSequenceEngine(
            self.symbol,
            segment_prefix=connection_id,
            max_buffer_events=int(self.args.max_buffer_events),
        )
        risk_accumulator = L2IntervalAccumulator()
        last_risk_snapshot_monotonic = float("-inf")
        self.storage.append_connection_start(connection_id)
        self.storage.append_control(
            "connection_opened",
            {
                "symbol": self.symbol,
                "connection_id": connection_id,
                "at_utc": connected_at_utc,
            },
        )
        self.current_phase = "buffering"
        snapshot_task: asyncio.Task[DepthSnapshot] | None = asyncio.create_task(
            self._fetch_snapshot(connection_id, lambda: wire_sequence)
        )
        snapshot_retry_at = 0.0
        reason = "connection_closed"
        try:
            while not self._should_stop():
                if (
                    time.monotonic() - connected_monotonic
                    >= float(self.args.rotation_seconds)
                ):
                    reason = "scheduled_rotation"
                    break

                if snapshot_task is not None and snapshot_task.done():
                    try:
                        snapshot = snapshot_task.result()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        self.last_error = repr(exc)
                        self.storage.append_control(
                            "snapshot_retry",
                            {
                                "symbol": self.symbol,
                                "connection_id": connection_id,
                                "at_utc": _utc_now_iso(),
                                "error": repr(exc),
                            },
                        )
                        snapshot_retry_at = (
                            time.monotonic()
                            + float(self.args.snapshot_retry_seconds)
                        )
                    else:
                        self.storage.append_snapshot_timeline(snapshot)
                        actions = engine.offer_snapshot(snapshot)
                        self.storage.append_actions(actions)
                        self.current_phase = engine.phase
                        applied_wires = [
                            int(action.details["wire_sequence"])
                            for action in actions
                            if action.kind == "event_applied"
                        ]
                        if (
                            engine.phase == "live"
                            and engine.book is not None
                            and engine.current_segment_id is not None
                            and engine.last_u is not None
                            and applied_wires
                        ):
                            risk_record = build_l2_risk_record(
                                book=engine.book,
                                accumulator=risk_accumulator,
                                symbol=self.symbol,
                                run_id=self.storage.run_id,
                                connection_id=connection_id,
                                segment_id=engine.current_segment_id,
                                captured_at_utc=snapshot.received_at_utc,
                                wire_sequence=max(applied_wires),
                                final_update_id=engine.last_u,
                                top_n=int(self.args.risk_top_levels),
                                exchange_event_time_ms=engine.last_event_time_ms,
                                exchange_transaction_time_ms=(
                                    engine.last_transaction_time_ms
                                ),
                            )
                            self.storage.append_risk_snapshot(risk_record)
                            risk_accumulator.reset()
                            last_risk_snapshot_monotonic = time.monotonic()
                        if _actions_need_snapshot(actions):
                            snapshot_retry_at = time.monotonic()
                    snapshot_task = None

                if (
                    engine.phase == "buffering"
                    and snapshot_task is None
                    and time.monotonic() >= snapshot_retry_at
                ):
                    snapshot_task = asyncio.create_task(
                        self._fetch_snapshot(connection_id, lambda: wire_sequence)
                    )

                receive_timeout = 1.0
                remaining = self._time_remaining()
                if remaining is not None:
                    receive_timeout = max(0.01, min(receive_timeout, remaining))
                try:
                    message = await ws.receive(timeout=receive_timeout)
                except TimeoutError:
                    await self.write_heartbeat()
                    continue

                if message.type == aiohttp.WSMsgType.TEXT:
                    wire_sequence += 1
                    received_at_utc = _utc_now_iso()
                    received_monotonic_ns = time.monotonic_ns()
                    raw_text = str(message.data)
                    self.storage.append_wire(
                        connection_id=connection_id,
                        wire_sequence=wire_sequence,
                        received_at_utc=received_at_utc,
                        received_monotonic_ns=received_monotonic_ns,
                        raw_text=raw_text,
                    )
                    try:
                        decoded = json.loads(raw_text)
                    except json.JSONDecodeError as exc:
                        self.storage.append_control(
                            "parse_error",
                            {
                                "symbol": self.symbol,
                                "connection_id": connection_id,
                                "wire_sequence": wire_sequence,
                                "at_utc": received_at_utc,
                                "error": f"WebSocket frame is not valid JSON: {exc}",
                            },
                        )
                        reason = "parse_error"
                        break
                    if not isinstance(decoded, dict):
                        self.storage.append_control(
                            "parse_error",
                            {
                                "symbol": self.symbol,
                                "connection_id": connection_id,
                                "wire_sequence": wire_sequence,
                                "at_utc": received_at_utc,
                                "error": "WebSocket payload is not an object",
                            },
                        )
                        reason = "parse_error"
                        break
                    nested = decoded.get("data", decoded)
                    event_type = nested.get("e") if isinstance(nested, dict) else None
                    if "result" in decoded and "id" in decoded:
                        self.storage.append_control(
                            "unexpected_depth_subscription_response",
                            {
                                "symbol": self.symbol,
                                "connection_id": connection_id,
                                "wire_sequence": wire_sequence,
                                "at_utc": received_at_utc,
                                "payload": decoded,
                            },
                        )
                        reason = "unexpected_depth_subscription_response"
                        break
                    if event_type == "markPriceUpdate":
                        try:
                            mark_price = parse_public_mark_price(
                                decoded,
                                expected_symbol=self.symbol,
                                connection_id=connection_id,
                                wire_sequence=wire_sequence,
                                received_at_utc=received_at_utc,
                                received_monotonic_ns=received_monotonic_ns,
                            )
                        except PayloadValidationError as exc:
                            self.storage.append_control(
                                "public_mark_price_parse_error",
                                {
                                    "symbol": self.symbol,
                                    "connection_id": connection_id,
                                    "wire_sequence": wire_sequence,
                                    "at_utc": received_at_utc,
                                    "error": str(exc),
                                },
                            )
                            reason = "public_mark_price_parse_error"
                            break
                        self.storage.append_public_mark_price(mark_price)
                        await self.write_heartbeat()
                        continue
                    if event_type == "aggTrade":
                        try:
                            public_trade = parse_public_agg_trade(
                                decoded,
                                expected_symbol=self.symbol,
                                connection_id=connection_id,
                                wire_sequence=wire_sequence,
                                received_at_utc=received_at_utc,
                                received_monotonic_ns=received_monotonic_ns,
                            )
                        except PayloadValidationError as exc:
                            self.storage.append_control(
                                "public_trade_parse_error",
                                {
                                    "symbol": self.symbol,
                                    "connection_id": connection_id,
                                    "wire_sequence": wire_sequence,
                                    "at_utc": received_at_utc,
                                    "error": str(exc),
                                },
                            )
                            reason = "public_trade_parse_error"
                            break
                        current_id = public_trade.aggregate_trade_id
                        if (
                            last_public_agg_trade_id is not None
                            and current_id <= last_public_agg_trade_id
                        ):
                            kind = (
                                "public_trade_duplicate_dropped"
                                if current_id == last_public_agg_trade_id
                                else "public_trade_out_of_order_dropped"
                            )
                            self.storage.append_control(
                                kind,
                                {
                                    "symbol": self.symbol,
                                    "connection_id": connection_id,
                                    "wire_sequence": wire_sequence,
                                    "at_utc": received_at_utc,
                                    "aggregate_trade_id": current_id,
                                    "last_accepted_aggregate_trade_id": (
                                        last_public_agg_trade_id
                                    ),
                                },
                            )
                            await self.write_heartbeat()
                            continue
                        discontinuity = (
                            last_public_agg_trade_id is not None
                            and current_id != last_public_agg_trade_id + 1
                        )
                        self.storage.append_public_agg_trade(
                            public_trade,
                            id_discontinuity=discontinuity,
                            segment_id=(
                                engine.current_segment_id
                                if engine.phase == "live"
                                else None
                            ),
                        )
                        if discontinuity:
                            self.storage.append_control(
                                "public_trade_id_discontinuity_observed",
                                {
                                    "symbol": self.symbol,
                                    "connection_id": connection_id,
                                    "wire_sequence": wire_sequence,
                                    "at_utc": received_at_utc,
                                    "previous_aggregate_trade_id": (
                                        last_public_agg_trade_id
                                    ),
                                    "current_aggregate_trade_id": current_id,
                                    "scope_note": (
                                        "Observed ID discontinuity; Binance's stream "
                                        "contract does not make this alone proof of "
                                        "a missed event."
                                    ),
                                },
                            )
                        last_public_agg_trade_id = current_id
                        await self.write_heartbeat()
                        continue
                    if event_type != "depthUpdate":
                        self.storage.append_control(
                            "unexpected_public_event",
                            {
                                "symbol": self.symbol,
                                "connection_id": connection_id,
                                "wire_sequence": wire_sequence,
                                "at_utc": received_at_utc,
                                "event_type": event_type,
                            },
                        )
                        reason = "unexpected_public_event"
                        break
                    try:
                        event = parse_diff_depth_event(
                            decoded,
                            expected_symbol=self.symbol,
                            connection_id=connection_id,
                            wire_sequence=wire_sequence,
                            received_at_utc=received_at_utc,
                            received_monotonic_ns=received_monotonic_ns,
                        )
                    except PayloadValidationError as exc:
                        self.storage.append_control(
                            "parse_error",
                            {
                                "symbol": self.symbol,
                                "connection_id": connection_id,
                                "wire_sequence": wire_sequence,
                                "at_utc": received_at_utc,
                                "error": str(exc),
                            },
                        )
                        reason = "parse_error"
                        break
                    self.storage.append_event_timeline(event)
                    if engine.phase == "live" and engine.book is not None:
                        risk_accumulator.observe(engine.book, event)
                    try:
                        actions = engine.offer_event(event)
                    except SequenceBufferOverflow as exc:
                        self.storage.append_control(
                            "buffer_overflow",
                            {
                                "symbol": self.symbol,
                                "connection_id": connection_id,
                                "wire_sequence": wire_sequence,
                                "at_utc": received_at_utc,
                                "error": str(exc),
                            },
                        )
                        reason = "buffer_overflow"
                        break
                    self.storage.append_actions(actions)
                    self.current_phase = engine.phase
                    if _actions_need_snapshot(actions):
                        risk_accumulator.reset()
                    if (
                        engine.phase == "live"
                        and engine.book is not None
                        and engine.current_segment_id is not None
                        and engine.last_u is not None
                        and any(action.kind == "event_applied" for action in actions)
                        and time.monotonic() - last_risk_snapshot_monotonic
                        >= float(self.args.risk_snapshot_seconds)
                    ):
                        risk_record = build_l2_risk_record(
                            book=engine.book,
                            accumulator=risk_accumulator,
                            symbol=self.symbol,
                            run_id=self.storage.run_id,
                            connection_id=connection_id,
                            segment_id=engine.current_segment_id,
                            captured_at_utc=received_at_utc,
                            wire_sequence=wire_sequence,
                            final_update_id=engine.last_u,
                            top_n=int(self.args.risk_top_levels),
                            exchange_event_time_ms=engine.last_event_time_ms,
                            exchange_transaction_time_ms=(
                                engine.last_transaction_time_ms
                            ),
                        )
                        self.storage.append_risk_snapshot(risk_record)
                        risk_accumulator.reset()
                        last_risk_snapshot_monotonic = time.monotonic()
                    if _actions_need_snapshot(actions) and snapshot_task is None:
                        snapshot_retry_at = time.monotonic()
                    await self.write_heartbeat()
                elif message.type in {
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                }:
                    reason = f"websocket_closed_{ws.close_code}"
                    break
                elif message.type == aiohttp.WSMsgType.ERROR:
                    reason = f"websocket_error_{ws.exception()!r}"
                    break
                elif message.type in {
                    aiohttp.WSMsgType.PING,
                    aiohttp.WSMsgType.PONG,
                }:
                    continue
                else:
                    self.storage.append_control(
                        "unexpected_websocket_frame",
                        {
                            "symbol": self.symbol,
                            "connection_id": connection_id,
                            "at_utc": _utc_now_iso(),
                            "message_type": str(message.type),
                        },
                    )
                    reason = "unexpected_websocket_frame"
                    break
            if (
                self._should_stop()
                and engine.phase == "buffering"
                and engine.buffer
                and snapshot_task is not None
            ):
                try:
                    snapshot = await asyncio.wait_for(
                        snapshot_task,
                        timeout=float(
                            self.args.shutdown_bootstrap_grace_seconds
                        ),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.storage.append_control(
                        "shutdown_bootstrap_incomplete",
                        {
                            "symbol": self.symbol,
                            "connection_id": connection_id,
                            "at_utc": _utc_now_iso(),
                            "buffered_event_count": len(engine.buffer),
                            "error": repr(exc),
                        },
                    )
                else:
                    self.storage.append_snapshot_timeline(snapshot)
                    actions = engine.offer_snapshot(snapshot)
                    self.storage.append_actions(actions)
                    self.current_phase = engine.phase
                    applied_wires = [
                        int(action.details["wire_sequence"])
                        for action in actions
                        if action.kind == "event_applied"
                    ]
                    if (
                        engine.phase == "live"
                        and engine.book is not None
                        and engine.current_segment_id is not None
                        and engine.last_u is not None
                        and applied_wires
                    ):
                        risk_record = build_l2_risk_record(
                            book=engine.book,
                            accumulator=risk_accumulator,
                            symbol=self.symbol,
                            run_id=self.storage.run_id,
                            connection_id=connection_id,
                            segment_id=engine.current_segment_id,
                            captured_at_utc=snapshot.received_at_utc,
                            wire_sequence=max(applied_wires),
                            final_update_id=engine.last_u,
                            top_n=int(self.args.risk_top_levels),
                            exchange_event_time_ms=engine.last_event_time_ms,
                            exchange_transaction_time_ms=(
                                engine.last_transaction_time_ms
                            ),
                        )
                        self.storage.append_risk_snapshot(risk_record)
                        risk_accumulator.reset()
                        last_risk_snapshot_monotonic = time.monotonic()
                snapshot_task = None
            if self._should_stop():
                reason = "capture_complete"
        finally:
            if snapshot_task is not None and not snapshot_task.done():
                snapshot_task.cancel()
                try:
                    await snapshot_task
                except asyncio.CancelledError:
                    pass
            actions = engine.end_connection(reason)
            self.storage.append_connection_end(connection_id, reason)
            self.storage.append_actions(actions)
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

    async def _fetch_snapshot(
        self,
        connection_id: str,
        wire_sequence_supplier: Any,
    ) -> DepthSnapshot:
        requested_at_utc = _utc_now_iso()
        started_monotonic = time.monotonic()
        url = f"{self.args.rest_base.rstrip('/')}/fapi/v1/depth"
        timeout = aiohttp.ClientTimeout(
            total=float(self.args.snapshot_timeout_seconds)
        )
        async with self.session.get(
            url,
            params={
                "symbol": self.symbol,
                "limit": int(self.args.snapshot_limit),
            },
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            decoded = await response.json()
        received_monotonic_ns = time.monotonic_ns()
        received_at_utc = _utc_now_iso()
        if not isinstance(decoded, dict):
            raise PayloadValidationError("REST snapshot response is not an object")
        snapshot = parse_depth_snapshot(
            cast(dict[str, Any], decoded),
            symbol=self.symbol,
            connection_id=connection_id,
            request_started_at_utc=requested_at_utc,
            received_at_utc=received_at_utc,
            received_monotonic_ns=received_monotonic_ns,
            wire_sequence_seen=int(wire_sequence_supplier()),
            request_latency_ms=(time.monotonic() - started_monotonic) * 1000.0,
        )
        self.storage.append_snapshot_evidence(snapshot)
        return snapshot


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--symbols", nargs="+")
    source.add_argument("--input", help="CSV containing symbol and optional IDs")
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--live-root", default=str(ROOT / "Live"))
    parser.add_argument("--audit-dir")
    parser.add_argument("--ws-base", default=DEFAULT_WS_BASE)
    parser.add_argument("--market-ws-base", default=DEFAULT_MARKET_WS_BASE)
    parser.add_argument("--rest-base", default=DEFAULT_REST_BASE)
    parser.add_argument(
        "--update-speed",
        choices=["100ms", "500ms"],
        default=DEFAULT_UPDATE_SPEED,
    )
    parser.add_argument("--snapshot-limit", type=int, default=1000)
    parser.add_argument("--snapshot-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--snapshot-retry-seconds", type=float, default=1.0)
    parser.add_argument(
        "--shutdown-bootstrap-grace-seconds",
        type=float,
        default=2.0,
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=180.0)
    parser.add_argument(
        "--manifest-heartbeat-seconds",
        type=float,
        default=5.0,
        help=(
            "minimum seconds between derivative manifest rewrites; raw event "
            "durability is controlled independently by --fsync-every"
        ),
    )
    parser.add_argument("--rotation-seconds", type=float, default=23 * 3600)
    parser.add_argument("--reconnect-base-seconds", type=float, default=1.0)
    parser.add_argument("--reconnect-max-seconds", type=float, default=30.0)
    parser.add_argument("--max-buffer-events", type=int, default=100_000)
    parser.add_argument("--max-message-bytes", type=int, default=2_000_000)
    parser.add_argument(
        "--risk-snapshot-seconds",
        type=float,
        default=5.0,
        help="seconds between bounded scanner-facing L2 derivatives",
    )
    parser.add_argument(
        "--risk-top-levels",
        type=int,
        default=50,
        help="book levels per side retained in each L2 risk derivative",
    )
    parser.add_argument(
        "--fsync-every",
        type=int,
        default=1,
        help="fsync every N records per file; 1 is strongest durability",
    )
    parser.add_argument(
        "--no-agg-trades",
        action="store_false",
        dest="collect_agg_trades",
        default=True,
        help="disable same-connection public aggregate-trade subscription",
    )
    parser.add_argument(
        "--no-mark-price-updates",
        action="store_false",
        dest="collect_mark_price_updates",
        default=True,
        help="disable same-connection one-second mark-price subscription",
    )
    args = parser.parse_args(argv)
    if args.duration_seconds < 0:
        parser.error("--duration-seconds must be >= 0")
    if args.snapshot_limit not in {5, 10, 20, 50, 100, 500, 1000}:
        parser.error("--snapshot-limit must be a Binance-supported depth limit")
    for name in (
        "snapshot_timeout_seconds",
        "snapshot_retry_seconds",
        "shutdown_bootstrap_grace_seconds",
        "heartbeat_seconds",
        "manifest_heartbeat_seconds",
        "rotation_seconds",
        "reconnect_base_seconds",
        "reconnect_max_seconds",
        "risk_snapshot_seconds",
    ):
        if float(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if args.max_buffer_events <= 0:
        parser.error("--max-buffer-events must be > 0")
    if args.max_message_bytes <= 0:
        parser.error("--max-message-bytes must be > 0")
    if args.risk_top_levels <= 0 or args.risk_top_levels > args.snapshot_limit:
        parser.error("--risk-top-levels must be > 0 and <= --snapshot-limit")
    if args.fsync_every < 0:
        parser.error("--fsync-every must be >= 0")
    return args


async def collect_diff_depth(args: argparse.Namespace) -> int:
    targets = _load_targets(symbols=args.symbols, input_path=args.input)
    started_at = _utc_now()
    run_id = started_at.strftime("diff_depth_%Y%m%d_%H%M%S")
    live_date = started_at.astimezone(LIMA).strftime("%Y-%m-%d")
    audit_dir = (
        Path(args.audit_dir)
        if args.audit_dir
        else ROOT / "outputs" / "audits" / run_id
    )
    audit_dir.mkdir(parents=True, exist_ok=True)
    stop_path = audit_dir / "STOP"
    stop_path.unlink(missing_ok=True)
    # The requested duration measures active capture time, not storage setup or
    # initial manifest provenance collection (which includes ``git status`` and
    # can be slow on a large/OneDrive-backed worktree).
    deadline: float | None = None
    capture_started_at_utc: str | None = None
    stop_event = asyncio.Event()

    def _request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_stop)

    storages: dict[str, SymbolCaptureStorage] = {}
    collectors: dict[str, SymbolDiffDepthCollector] = {}
    for target in targets:
        symbol_run_dir = (
            Path(args.live_root)
            / live_date
            / target.symbol
            / "diff_depth"
            / run_id
        )
        storage = SymbolCaptureStorage(
            symbol_run_dir,
            symbol=target.symbol,
            run_id=run_id,
            fsync_every=int(args.fsync_every),
        )
        storages[target.symbol] = storage

    run_manifest_path = audit_dir / "manifest.json"

    run_manifest_last_write_monotonic = float("-inf")

    async def _write_run_manifest(status: str, **extra: Any) -> None:
        nonlocal run_manifest_last_write_monotonic
        now = time.monotonic()
        if (
            status == "running"
            and now - run_manifest_last_write_monotonic
            < float(args.manifest_heartbeat_seconds)
        ):
            return
        payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": status,
            "run_id": run_id,
            "started_at_utc": started_at.replace(microsecond=0).isoformat(),
            "updated_at_utc": _utc_now_iso(),
            "live_date_lima": live_date,
            "audit_dir": str(audit_dir),
            "targets": [asdict(target) for target in targets],
            "symbol_run_dirs": {
                symbol: str(storage.run_dir)
                for symbol, storage in storages.items()
            },
            "symbol_counters": {
                symbol: dict(storage.counters)
                for symbol, storage in storages.items()
            },
            "duration_seconds": float(args.duration_seconds),
            "capture_started_at_utc": capture_started_at_utc,
            "update_speed": args.update_speed,
            "snapshot_limit": int(args.snapshot_limit),
            "fsync_every": int(args.fsync_every),
            "manifest_heartbeat_seconds": float(
                args.manifest_heartbeat_seconds
            ),
            "shutdown_bootstrap_grace_seconds": float(
                args.shutdown_bootstrap_grace_seconds
            ),
            "git_head": _git_output(["rev-parse", "--short", "HEAD"]),
            "git_status_short": _git_output(["status", "--short"]),
            "collector_pid": os.getpid(),
            "scope_note": (
                "Prospective Binance-published diff-depth on the public endpoint "
                "plus aggregate trades and mark prices on the market endpoint. "
                "Depth sequence completeness is per labelled contiguous segment; "
                "this is not historical queue reconstruction or private fill data."
            ),
            **extra,
        }
        await asyncio.to_thread(atomic_write_json, run_manifest_path, payload)
        run_manifest_last_write_monotonic = time.monotonic()

    await _write_run_manifest("starting")
    capture_started_at_utc = _utc_now_iso()
    deadline = (
        time.monotonic() + float(args.duration_seconds)
        if args.duration_seconds > 0
        else None
    )
    session_timeout = aiohttp.ClientTimeout(total=None)
    results: list[SymbolRunResult] = []
    try:
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            collectors = {
                target.symbol: SymbolDiffDepthCollector(
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
                await _write_run_manifest("running")
                await asyncio.sleep(0.5)
            gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
            failures: list[dict[str, str]] = []
            for symbol, result in zip(tasks, gathered):
                if isinstance(result, BaseException):
                    failures.append({"symbol": symbol, "error": repr(result)})
                else:
                    results.append(result)
            if failures:
                await _write_run_manifest(
                    "failed",
                    failures=failures,
                    results=[asdict(result) for result in results],
                    completed_at_utc=_utc_now_iso(),
                )
                return 2
    finally:
        for storage in storages.values():
            storage.close()

    all_required_streams_complete = all(
        result.applied_events > 0 and result.public_streams_ready
        for result in results
    )
    status = (
        "complete"
        if all_required_streams_complete
        else "incomplete_required_streams"
    )
    await _write_run_manifest(
        status,
        results=[asdict(result) for result in results],
        completed_at_utc=_utc_now_iso(),
    )
    return 0 if all_required_streams_complete else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit_dir = (
        Path(args.audit_dir)
        if args.audit_dir
        else ROOT / "outputs" / "audits" / "diff_depth_lock"
    )
    lock_path = audit_dir / "collector.lock"
    descriptor = _acquire_lock(lock_path)
    try:
        return asyncio.run(collect_diff_depth(args))
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
