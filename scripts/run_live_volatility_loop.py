"""Run the fail-closed three-minute shadow volatility inference loop.

The loop discovers an immutable authenticated telemetry roster, reconciles the
archive-to-live finalized one-minute mark/last tail through Binance REST, and
emits one verdict-inert forecast or explicit unavailable record per active bot.
It never writes active model lineage or calls an execution path.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import logging
import math
import os
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neutralgrid.api.binance_client import BinanceAPIError, BinanceClient
from neutralgrid.data.price_series.ps_store import PriceStore
from neutralgrid.data.price_series.ps_types import Candle, SeriesKind
from neutralgrid.live.decision.volatility import (
    VOLATILITY_OUTPUT_SCHEMA,
    VolatilityContract,
    VolatilityError,
    load_price_store_frame,
    load_volatility_contract,
    validate_price_frame,
)
from neutralgrid.live.decision.volatility_forecast import predict_shadow_volatility
from neutralgrid.live.monotonic_schedule import advance_nominal_start


TRUSTED_BINANCE_ROOT_HOSTNAMES = frozenset({"binance.com", "binance.bh"})


UTC = timezone.utc
LIMA = ZoneInfo("America/Lima")
MINUTE_MS = 60_000
PLUGIN_CYCLE_SCHEMA = "neutralgrid_private_telemetry_cycle_v2"
logger = logging.getLogger(__name__)


class LiveVolatilityLoopError(RuntimeError):
    """A roster, scheduler, data, or persistence invariant failed."""


@dataclass(frozen=True)
class RosterEntry:
    symbol: str
    strategy_id: str


@dataclass(frozen=True)
class Roster:
    manifest_path: Path
    completed_at_utc: datetime
    entries: tuple[RosterEntry, ...]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise LiveVolatilityLoopError(f"non-finite JSON constant {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except LiveVolatilityLoopError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LiveVolatilityLoopError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LiveVolatilityLoopError(f"{path}: JSON root must be an object")
    return payload


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise LiveVolatilityLoopError(f"{field} must be ISO-8601 text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LiveVolatilityLoopError(f"{field} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise LiveVolatilityLoopError(f"{field} lacks a timezone")
    return parsed.astimezone(UTC)


def _latest_cycle_manifest(audit_root: Path) -> Path:
    cycle_dir = audit_root.resolve() / "cycles"
    candidates = sorted(cycle_dir.glob("cycle_*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise LiveVolatilityLoopError(f"no Chrome-plugin cycle exists under {cycle_dir}")
    return candidates[-1]


def load_roster(
    *,
    manifest_path: Path | None,
    audit_root: Path,
    max_age_seconds: float,
    now: datetime | None = None,
) -> Roster:
    """Load an exact, complete, fresh authenticated Working roster."""

    path = manifest_path.resolve() if manifest_path is not None else _latest_cycle_manifest(audit_root)
    payload = _strict_json(path)
    if payload.get("schema_version") != PLUGIN_CYCLE_SCHEMA:
        raise LiveVolatilityLoopError("unsupported roster manifest schema")
    if payload.get("status") != "complete" or payload.get("source") != "chrome_plugin":
        raise LiveVolatilityLoopError("roster is not a complete Chrome-plugin cycle")
    page_identity = payload.get("page_identity")
    if not isinstance(page_identity, str) or not page_identity.strip():
        raise LiveVolatilityLoopError("roster page identity is missing")
    source_url = payload.get("source_url")
    if not isinstance(source_url, str) or not source_url.strip():
        raise LiveVolatilityLoopError("roster source URL is missing")
    parsed_source = urlparse(source_url)
    hostname = (parsed_source.hostname or "").lower()
    trusted_hostname = any(
        hostname == root_hostname or hostname.endswith(f".{root_hostname}")
        for root_hostname in TRUSTED_BINANCE_ROOT_HOSTNAMES
    )
    if parsed_source.scheme != "https" or not trusted_hostname:
        raise LiveVolatilityLoopError("roster source URL is not trusted Binance HTTPS")
    completed = _parse_utc(payload.get("cycle_completed_at_utc"), field="cycle_completed_at_utc")
    reference = _utc_now() if now is None else now.astimezone(UTC)
    age_seconds = (reference - completed).total_seconds()
    if age_seconds < 0.0 or age_seconds > max_age_seconds:
        raise LiveVolatilityLoopError(
            f"authenticated roster is stale: age_seconds={age_seconds:.3f}, "
            f"limit={max_age_seconds:.3f}"
        )
    files = payload.get("files")
    if not isinstance(files, list):
        raise LiveVolatilityLoopError("roster files must be a list")
    entries: list[RosterEntry] = []
    for item in files:
        if not isinstance(item, Mapping):
            raise LiveVolatilityLoopError("roster entry must be an object")
        symbol = str(item.get("symbol", "")).strip().upper()
        strategy_id = str(item.get("strategy_id", "")).strip()
        if not symbol or not strategy_id:
            raise LiveVolatilityLoopError("roster identity is incomplete")
        entries.append(RosterEntry(symbol=symbol, strategy_id=strategy_id))
    expected_count = payload.get("active_bot_count")
    identities = {(item.symbol, item.strategy_id) for item in entries}
    if expected_count != len(entries) or len(identities) != len(entries) or not entries:
        raise LiveVolatilityLoopError("roster cardinality or identity uniqueness failed")
    if payload.get("working_row_count") != expected_count:
        raise LiveVolatilityLoopError("Working-row count does not match active count")
    return Roster(path, completed, tuple(entries))


def _atomic_json(path: Path, payload: Mapping[str, Any], *, replace: bool) -> None:
    text = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and path.exists():
        raise LiveVolatilityLoopError(f"refusing to overwrite immutable output {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if not replace and path.exists():
            raise LiveVolatilityLoopError(f"output appeared during write: {path}")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _raw_kline_frame(rows: Sequence[Sequence[Any]], *, now_ms: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        if len(row) < 7:
            raise VolatilityError("REST kline has fewer than seven fields")
        close_time_ms = int(row[6])
        if close_time_ms >= now_ms:
            continue
        records.append(
            {
                "open_time_ms": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "close_time_ms": close_time_ms,
                "is_final": True,
            }
        )
    return pd.DataFrame.from_records(records)


def _latest_open_time_ms(frame: pd.DataFrame) -> int | None:
    if frame.empty:
        return None
    values = cast(pd.Series, pd.to_numeric(cast(pd.Series, frame["open_time_ms"]), errors="coerce"))
    if bool(values.isna().any()):
        raise VolatilityError("stored candle contains an invalid open_time_ms")
    return int(values.max())


def _append_audited_frame(
    store: PriceStore,
    *,
    price_store_root: Path,
    symbol: str,
    kind: SeriesKind,
    incoming: pd.DataFrame,
) -> dict[str, Any]:
    existing = load_price_store_frame(
        price_store_root,
        symbol=symbol,
        series_kind=kind.value,
    )
    if incoming.empty:
        _, audit = validate_price_frame(existing, symbol=symbol, series_kind=kind.value)
        return {
            "existing_rows": len(existing),
            "incoming_rows": 0,
            "appended_rows": 0,
            "exact_duplicates": audit.exact_duplicates,
            "gap_count": audit.gap_count,
            "missing_minutes": audit.missing_minutes,
        }
    combined = incoming if existing.empty else pd.concat([existing, incoming], ignore_index=True)
    combined = combined.sort_values("open_time_ms", kind="stable").reset_index(drop=True)
    _, audit = validate_price_frame(combined, symbol=symbol, series_kind=kind.value)
    existing_times: set[int] = set()
    if not existing.empty:
        times = cast(pd.Series, pd.to_numeric(cast(pd.Series, existing["open_time_ms"]), errors="coerce"))
        existing_times = set(times.dropna().astype("int64").tolist())
    appended = 0
    for row in incoming.to_dict(orient="records"):
        open_time_ms = int(row["open_time_ms"])
        if open_time_ms in existing_times:
            continue
        store.append_candle(
            symbol,
            kind,
            "1m",
            Candle(
                open_time_ms=open_time_ms,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                close_time_ms=int(row["close_time_ms"]),
                is_final=True,
            ),
        )
        appended += 1
    flushed = store.flush_to_disk()
    return {
        "existing_rows": len(existing),
        "incoming_rows": len(incoming),
        "appended_rows": appended,
        "flushed_rows": flushed,
        "exact_duplicates": audit.exact_duplicates,
        "conflicting_duplicates": audit.conflicting_duplicates,
        "gap_count": audit.gap_count,
        "missing_minutes": audit.missing_minutes,
    }


async def reconcile_symbol_tail(
    client: BinanceClient,
    store: PriceStore,
    *,
    price_store_root: Path,
    symbol: str,
    now: datetime,
) -> dict[str, Any]:
    """Fetch only the bounded archive-to-live tail and conflict-audit before write."""

    now_ms = int(now.timestamp() * 1000)
    end_ms = (now_ms // MINUTE_MS) * MINUTE_MS - 1
    mark_existing = load_price_store_frame(price_store_root, symbol=symbol, series_kind="mark_kline")
    last_existing = load_price_store_frame(price_store_root, symbol=symbol, series_kind="last_kline")
    latest_candidates = [
        value
        for value in (_latest_open_time_ms(mark_existing), _latest_open_time_ms(last_existing))
        if value is not None
    ]
    latest_common = min(latest_candidates) if len(latest_candidates) == 2 else None
    start_ms = (
        max(end_ms - 1499 * MINUTE_MS, latest_common + MINUTE_MS)
        if latest_common is not None
        else end_ms - 499 * MINUTE_MS
    )
    if start_ms > end_ms:
        start_ms = end_ms
    limit = min(1500, max(1, int((end_ms - start_ms) // MINUTE_MS) + 1))
    mark_rows, last_rows = await asyncio.gather(
        client.get_mark_price_klines(
            symbol,
            "1m",
            limit=limit,
            start_time=start_ms,
            end_time=end_ms,
            include_current=False,
        ),
        client.get_klines(
            symbol,
            "1m",
            limit=limit,
            start_time=start_ms,
            end_time=end_ms,
            include_current=False,
        ),
    )
    mark_frame = _raw_kline_frame(mark_rows, now_ms=now_ms)
    last_frame = _raw_kline_frame(last_rows, now_ms=now_ms)
    if mark_frame.empty or last_frame.empty:
        raise VolatilityError(f"{symbol}: REST reconciliation returned no finalized mark/last pair")
    return {
        "requested_start_ms": start_ms,
        "requested_end_ms": end_ms,
        "request_limit": limit,
        "mark": _append_audited_frame(
            store,
            price_store_root=price_store_root,
            symbol=symbol,
            kind=SeriesKind.MARK_KLINE,
            incoming=mark_frame,
        ),
        "last": _append_audited_frame(
            store,
            price_store_root=price_store_root,
            symbol=symbol,
            kind=SeriesKind.LAST_KLINE,
            incoming=last_frame,
        ),
    }


def _missing_ranges(
    frame: pd.DataFrame,
    *,
    symbol: str,
    series_kind: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[tuple[int, int]]:
    normalized, _ = validate_price_frame(
        frame,
        symbol=symbol,
        series_kind=series_kind,
    )
    open_times = cast(
        pd.Series,
        pd.to_numeric(cast(pd.Series, normalized["open_time_ms"]), errors="coerce"),
    )
    if start_ms is not None:
        normalized = normalized.loc[open_times >= start_ms]
        open_times = open_times.loc[normalized.index]
    if end_ms is not None:
        normalized = normalized.loc[open_times <= end_ms]
    if len(normalized) < 2:
        return []
    times = np.asarray(normalized["open_time_ms"], dtype=np.int64)
    gaps: list[tuple[int, int]] = []
    for previous, current in zip(times[:-1], times[1:], strict=True):
        if int(current - previous) > MINUTE_MS:
            gaps.append((int(previous + MINUTE_MS), int(current - 1)))
    return gaps


async def reconcile_symbol_gaps(
    client: BinanceClient,
    store: PriceStore,
    *,
    price_store_root: Path,
    symbol: str,
    max_requests: int,
    now: datetime,
    scope_start_ms: int | None = None,
    scope_end_ms: int | None = None,
) -> dict[str, Any]:
    """REST-reconcile only explicitly observed internal one-minute gaps."""

    if max_requests <= 0:
        raise VolatilityError("max_requests must be positive")
    now_ms = int(now.timestamp() * 1000)
    requests_used = 0
    report: dict[str, Any] = {"requests": 0, "series": {}}
    for kind in (SeriesKind.MARK_KLINE, SeriesKind.LAST_KLINE):
        existing = load_price_store_frame(
            price_store_root,
            symbol=symbol,
            series_kind=kind.value,
        )
        ranges = _missing_ranges(
            existing,
            symbol=symbol,
            series_kind=kind.value,
            start_ms=scope_start_ms,
            end_ms=scope_end_ms,
        )
        series_report: dict[str, Any] = {
            "gaps_before": len(ranges),
            "requested_ranges": [],
        }
        for gap_start, gap_end in ranges:
            cursor = gap_start
            last_missing_open = (gap_end // MINUTE_MS) * MINUTE_MS
            while cursor <= last_missing_open:
                if requests_used >= max_requests:
                    raise VolatilityError(
                        f"{symbol}: audited gap reconciliation exceeded "
                        f"{max_requests} REST requests"
                    )
                page_last_open = min(
                    last_missing_open,
                    cursor + 1499 * MINUTE_MS,
                )
                request_end = page_last_open + MINUTE_MS - 1
                limit = int((page_last_open - cursor) // MINUTE_MS) + 1
                if kind == SeriesKind.MARK_KLINE:
                    rows = await client.get_mark_price_klines(
                        symbol,
                        "1m",
                        limit=limit,
                        start_time=cursor,
                        end_time=request_end,
                        include_current=False,
                    )
                else:
                    rows = await client.get_klines(
                        symbol,
                        "1m",
                        limit=limit,
                        start_time=cursor,
                        end_time=request_end,
                        include_current=False,
                    )
                requests_used += 1
                incoming = _raw_kline_frame(rows, now_ms=now_ms)
                if incoming.empty:
                    raise VolatilityError(
                        f"{symbol}/{kind.value}: REST returned no finalized candles "
                        f"for audited gap {cursor}-{request_end}"
                    )
                append_report = _append_audited_frame(
                    store,
                    price_store_root=price_store_root,
                    symbol=symbol,
                    kind=kind,
                    incoming=incoming,
                )
                series_report["requested_ranges"].append(
                    {
                        "start_ms": cursor,
                        "end_ms": request_end,
                        "rows": len(incoming),
                        "append": append_report,
                    }
                )
                cursor = page_last_open + MINUTE_MS
        after = load_price_store_frame(
            price_store_root,
            symbol=symbol,
            series_kind=kind.value,
        )
        series_report["gaps_after"] = len(
            _missing_ranges(
                after,
                symbol=symbol,
                series_kind=kind.value,
                start_ms=scope_start_ms,
                end_ms=scope_end_ms,
            )
        )
        report["series"][kind.value] = series_report
    report["requests"] = requests_used
    return report


def unavailable_record(
    *,
    entry: RosterEntry,
    contract: VolatilityContract,
    requested_horizon_minutes: int,
    failure_class: str,
    reason: str,
    created_at: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": VOLATILITY_OUTPUT_SCHEMA,
        "status": "unavailable",
        "symbol": entry.symbol,
        "strategy_id": entry.strategy_id,
        "created_at_utc": created_at.isoformat(),
        "cutoff_utc": None,
        "latest_source_timestamp_utc": None,
        "requested_horizon_minutes": requested_horizon_minutes,
        "contract_sha256": contract.contract_sha256,
        "model_sha256": None,
        "data_manifest_sha256": None,
        "failure_class": failure_class,
        "reason": reason,
        "freshness_seconds": None,
        "gaps": None,
        "source_quality": "unavailable",
        "eligibility": False,
        "verdict_influence": False,
        "runtime_effect": "none",
    }


def _failure_class(exc: BaseException) -> str:
    if isinstance(exc, BinanceAPIError):
        if exc.status in (418, 429):
            return "rate_limited"
        if exc.status == 404:
            return "http_404"
        if exc.status >= 500:
            return "http_5xx"
        return "binance_http_error"
    if isinstance(exc, VolatilityError):
        message = str(exc).lower()
        if "gap" in message:
            return "price_sequence_gap"
        if "artifact" in message or "model" in message or "contract" in message:
            return "ineligible_or_corrupt_artifact"
        return "invalid_price_evidence"
    if isinstance(exc, (OSError, TimeoutError)):
        return "network_or_filesystem_error"
    return "unexpected_failure"


def _persist_forecast(live_root: Path, record: Mapping[str, Any], *, created_at: datetime) -> Path:
    symbol = str(record["symbol"])
    date_folder = created_at.astimezone(LIMA).date().isoformat()
    stamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    path = live_root.resolve() / date_folder / symbol / "volatility" / f"forecast_{stamp}.json"
    _atomic_json(path, record, replace=False)
    return path


async def run_cycle(args: argparse.Namespace, *, scheduled_start_utc: datetime) -> dict[str, Any]:
    actual_start = _utc_now()
    contract = load_volatility_contract(args.contract.resolve())
    max_roster_age_seconds = (
        float(contract.maximum_roster_age_seconds)
        if args.max_roster_age_seconds is None
        else float(args.max_roster_age_seconds)
    )
    max_data_age_seconds = (
        float(contract.maximum_source_age_seconds)
        if args.max_data_age_seconds is None
        else float(args.max_data_age_seconds)
    )
    if max_roster_age_seconds > contract.maximum_roster_age_seconds:
        raise LiveVolatilityLoopError(
            "--max-roster-age-seconds cannot weaken the approved contract"
        )
    if max_data_age_seconds > contract.maximum_source_age_seconds:
        raise LiveVolatilityLoopError(
            "--max-data-age-seconds cannot weaken the approved contract"
        )
    try:
        roster = load_roster(
            manifest_path=args.cycle_manifest,
            audit_root=args.roster_audit_root,
            max_age_seconds=max_roster_age_seconds,
            now=actual_start,
        )
    except LiveVolatilityLoopError:
        # Without a trustworthy exact active identity, no symbol-specific
        # output can be safely routed into Live/.  Persist only the cycle audit.
        raise
    client = BinanceClient()
    store = PriceStore(store_dir=args.price_store.resolve())
    records: list[dict[str, Any]] = []
    paths: list[str] = []
    reconciliation: dict[str, Any] = {}
    try:
        for entry in roster.entries:
            try:
                tail_report = await reconcile_symbol_tail(
                    client,
                    store,
                    price_store_root=args.price_store.resolve(),
                    symbol=entry.symbol,
                    now=actual_start,
                )
                end_ms = (
                    int(actual_start.timestamp() * 1000) // MINUTE_MS
                ) * MINUTE_MS - 1
                gap_report = await reconcile_symbol_gaps(
                    client,
                    store,
                    price_store_root=args.price_store.resolve(),
                    symbol=entry.symbol,
                    max_requests=args.max_gap_rest_requests,
                    now=actual_start,
                    scope_start_ms=end_ms - 48 * 60 * MINUTE_MS,
                    scope_end_ms=end_ms,
                )
                reconciliation[entry.symbol] = {
                    "tail": tail_report,
                    "gaps": gap_report,
                }
                mark_frame = load_price_store_frame(
                    args.price_store.resolve(),
                    symbol=entry.symbol,
                    series_kind=contract.primary_series,
                )
                last_frame = load_price_store_frame(
                    args.price_store.resolve(),
                    symbol=entry.symbol,
                    series_kind=contract.diagnostic_series,
                )
                record = predict_shadow_volatility(
                    args.artifact_dir.resolve(),
                    contract=contract,
                    symbol=entry.symbol,
                    strategy_id=entry.strategy_id,
                    mark_frame=mark_frame,
                    last_frame=last_frame,
                    requested_horizon_minutes=args.requested_horizon_minutes,
                    asof_utc=cast(pd.Timestamp, pd.Timestamp(actual_start)),
                )
                freshness = float(record["freshness_seconds"])
                if not math.isfinite(freshness) or freshness > max_data_age_seconds:
                    raise VolatilityError(
                        f"{entry.symbol}: finalized mark data is stale: {freshness:.3f}s"
                    )
                record["created_at_utc"] = actual_start.isoformat()
                record["eligibility"] = True
            except Exception as exc:
                record = unavailable_record(
                    entry=entry,
                    contract=contract,
                    requested_horizon_minutes=args.requested_horizon_minutes,
                    failure_class=_failure_class(exc),
                    reason=str(exc),
                    created_at=actual_start,
                )
            records.append(record)
            paths.append(str(_persist_forecast(args.live_root, record, created_at=actual_start)))
    finally:
        await client.close()
    actual_end = _utc_now()
    return {
        "schema_version": "neutralgrid_live_volatility_loop_cycle_v1",
        "status": "complete",
        "scheduled_start_utc": scheduled_start_utc.isoformat(),
        "actual_start_utc": actual_start.isoformat(),
        "actual_end_utc": actual_end.isoformat(),
        "duration_seconds": (actual_end - actual_start).total_seconds(),
        "roster_manifest": str(roster.manifest_path),
        "active_bot_count": len(roster.entries),
        "available_count": sum(record["status"] == "available" for record in records),
        "unavailable_count": sum(record["status"] == "unavailable" for record in records),
        "records": records,
        "forecast_paths": paths,
        "reconciliation": reconciliation,
        "verdict_influence": False,
    }


def _acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            pid = int(path.read_text(encoding="ascii").splitlines()[0])
        except (OSError, ValueError, IndexError):
            path.unlink(missing_ok=True)
        else:
            if not _pid_is_alive(pid):
                path.unlink(missing_ok=True)
            else:
                raise LiveVolatilityLoopError(
                    f"volatility loop already running with PID {pid}"
                )
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise LiveVolatilityLoopError(
            f"volatility loop lock appeared concurrently: {path}"
        ) from exc
    try:
        os.write(
            descriptor,
            f"{os.getpid()}\n{_utc_now().isoformat()}\n".encode("ascii"),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _pid_is_alive(pid: int) -> bool:
    """Cross-platform non-mutating PID liveness check."""

    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            # Access denied is evidence that the PID exists but is not
            # queryable by this token.  Treat it as alive and fail closed.
            return ctypes.get_last_error() == 5
        exit_code = ctypes.c_ulong()
        try:
            success = kernel32.GetExitCodeProcess(
                handle,
                ctypes.byref(exit_code),
            )
            return bool(success) and exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=float, default=180.0)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "config" / "live_volatility_forecast_v1.json",
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--cycle-manifest", type=Path, default=None)
    parser.add_argument(
        "--roster-audit-root",
        type=Path,
        default=ROOT / "outputs" / "audits" / "chrome_plugin_ingest",
    )
    parser.add_argument("--max-roster-age-seconds", type=float, default=None)
    parser.add_argument("--max-data-age-seconds", type=float, default=None)
    parser.add_argument("--requested-horizon-minutes", type=int, default=360)
    parser.add_argument("--max-gap-rest-requests", type=int, default=8)
    parser.add_argument("--price-store", type=Path, default=ROOT / "data" / "price_store")
    parser.add_argument("--live-root", type=Path, default=ROOT / "Live")
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=ROOT / "outputs" / "audits" / "live_volatility_loop_current",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    for field in ("interval_seconds", "max_roster_age_seconds", "max_data_age_seconds"):
        value = getattr(args, field)
        if value is not None and (
            not math.isfinite(float(value)) or float(value) <= 0.0
        ):
            parser.error(f"--{field.replace('_', '-')} must be finite and positive")
    if not 0 < args.requested_horizon_minutes <= 360:
        parser.error("--requested-horizon-minutes must lie in (0, 360]")
    if args.max_gap_rest_requests <= 0:
        parser.error("--max-gap-rest-requests must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    lock_path = args.audit_dir.resolve() / "collector.lock"
    stop_path = args.audit_dir.resolve() / "STOP"
    try:
        _acquire_lock(lock_path)
    except LiveVolatilityLoopError as exc:
        logger.error("cannot start volatility loop: %s", exc)
        return 2
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)
    scheduled_monotonic = time.monotonic()
    scheduled_utc = _utc_now()
    total_skipped = 0
    exit_code = 0
    try:
        while True:
            actual_start_monotonic = time.monotonic()
            lateness = max(0.0, actual_start_monotonic - scheduled_monotonic)
            try:
                report = asyncio.run(run_cycle(args, scheduled_start_utc=scheduled_utc))
            except Exception as exc:
                report = {
                    "schema_version": "neutralgrid_live_volatility_loop_cycle_v1",
                    "status": "unavailable",
                    "scheduled_start_utc": scheduled_utc.isoformat(),
                    "actual_start_utc": _utc_now().isoformat(),
                    "failure_class": _failure_class(exc),
                    "reason": str(exc),
                    "verdict_influence": False,
                }
                exit_code = 2
            completed_monotonic = time.monotonic()
            scheduled_monotonic, skipped = advance_nominal_start(
                scheduled_monotonic,
                interval_seconds=args.interval_seconds,
                completed_at=completed_monotonic,
            )
            total_skipped += skipped
            wait_seconds = max(0.0, scheduled_monotonic - completed_monotonic)
            scheduled_utc = _utc_now() + timedelta(seconds=wait_seconds)
            report["lateness_seconds"] = lateness
            report["skipped_slots"] = skipped
            report["total_skipped_slots"] = total_skipped
            report["next_scheduled_start_utc"] = scheduled_utc.isoformat()
            stamp = _utc_now().strftime("%Y%m%dT%H%M%S%fZ")
            _atomic_json(args.audit_dir.resolve() / "cycles" / f"cycle_{stamp}.json", report, replace=False)
            _atomic_json(args.audit_dir.resolve() / "manifest.json", report, replace=True)
            print(json.dumps(report, indent=2, sort_keys=True))
            if args.once:
                return exit_code
            if stop_requested or stop_path.exists():
                return exit_code
            while time.monotonic() < scheduled_monotonic:
                if stop_requested or stop_path.exists():
                    return exit_code
                time.sleep(min(1.0, max(0.0, scheduled_monotonic - time.monotonic())))
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
