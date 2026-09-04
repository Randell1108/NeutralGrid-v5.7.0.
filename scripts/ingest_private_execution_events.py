"""Ingest exact-order-linked Binance user trades as canonical private events.

The static importer cannot prove a continuous user-data stream and therefore
cannot emit ``event_complete``.  Strategy ownership must be supplied as a
reviewed ``neutralgrid_strategy_order_linkage_v1`` allowlist of exchange order
IDs; symbol and time proximity are deliberately insufficient.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neutralgrid.live.decision.private_events import (
    PRIVATE_EVENT_MANIFEST_SCHEMA_VERSION,
    PRIVATE_EVENT_SCHEMA_VERSION,
    PrivateEventStreamRef,
    load_validated_private_event_window,
)


LINKAGE_SCHEMA_VERSION = "neutralgrid_strategy_order_linkage_v1"
SYMBOL_RE = re.compile(r"^[A-Z0-9]{5,24}$")
IDENTITY_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PrivateExecutionIngestError(RuntimeError):
    """Private execution input is malformed or cannot be linked exactly."""


def _symbol(value: Any, *, field: str) -> str:
    normalized = str(value).strip().upper()
    if not SYMBOL_RE.fullmatch(normalized):
        raise PrivateExecutionIngestError(f"{field} is not a valid symbol")
    return normalized


def _identity_component(value: Any, *, field: str) -> str:
    normalized = str(value).strip()
    if not IDENTITY_COMPONENT_RE.fullmatch(normalized):
        raise PrivateExecutionIngestError(
            f"{field} must be a safe non-empty identity component"
        )
    return normalized


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None:
        raise PrivateExecutionIngestError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _lima_timezone() -> timezone | ZoneInfo:
    try:
        return ZoneInfo("America/Lima")
    except ZoneInfoNotFoundError:  # pragma: no cover - Windows tzdata fallback
        return timezone(-timedelta(hours=5))


def _finite(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise PrivateExecutionIngestError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PrivateExecutionIngestError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise PrivateExecutionIngestError(f"{field} must be {qualifier}")
    return number


def _integer_text(value: Any, *, field: str) -> str:
    if isinstance(value, bool):
        raise PrivateExecutionIngestError(f"{field} must be an integer identifier")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise PrivateExecutionIngestError(
            f"{field} must be an integer identifier"
        ) from exc
    if number < 0 or str(value).strip() == "":
        raise PrivateExecutionIngestError(
            f"{field} must be a non-negative integer identifier"
        )
    return str(number)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise PrivateExecutionIngestError(
                f"refusing to overwrite immutable private-event artifact {path}"
            )
        os.rename(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except OSError:
            pass


def normalize_strategy_trades(
    trades: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    strategy_id: str,
    run_id: str,
    allowed_order_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Normalize only trades whose exchange order ID is exactly allowlisted."""

    symbol_upper = _symbol(symbol, field="symbol")
    strategy = _identity_component(strategy_id, field="strategy_id")
    canonical_run_id = _identity_component(run_id, field="run_id")
    normalized_allowed = {
        _integer_text(value, field="allowed_order_ids") for value in allowed_order_ids
    }
    if not normalized_allowed:
        raise PrivateExecutionIngestError("strategy linkage order_ids must be non-empty")
    records: list[dict[str, Any]] = []
    seen_trade_ids: set[str] = set()
    unlinked = 0
    duplicates = 0
    for index, trade in enumerate(trades):
        trade_symbol = str(trade.get("symbol", "")).strip().upper()
        if trade_symbol != symbol_upper:
            raise PrivateExecutionIngestError(
                f"trades[{index}].symbol mismatch: expected {symbol_upper}, got {trade_symbol or 'missing'}"
            )
        order_id = _integer_text(trade.get("orderId"), field=f"trades[{index}].orderId")
        if order_id not in normalized_allowed:
            unlinked += 1
            continue
        trade_id = _integer_text(trade.get("id"), field=f"trades[{index}].id")
        if trade_id in seen_trade_ids:
            duplicates += 1
            continue
        seen_trade_ids.add(trade_id)
        event_ms = int(_finite(trade.get("time"), field=f"trades[{index}].time"))
        if event_ms <= 0:
            raise PrivateExecutionIngestError(f"trades[{index}].time must be positive")
        side = str(trade.get("side", "")).strip().upper()
        if side not in {"BUY", "SELL"}:
            raise PrivateExecutionIngestError(f"trades[{index}].side must be BUY or SELL")
        maker = trade.get("maker")
        if not isinstance(maker, bool):
            raise PrivateExecutionIngestError(f"trades[{index}].maker must be boolean")
        record: dict[str, Any] = {
            "schema_version": PRIVATE_EVENT_SCHEMA_VERSION,
            "event_type": "trade_fill",
            "symbol": symbol_upper,
            "strategy_id": strategy,
            "run_id": canonical_run_id,
            "event_time_utc": datetime.fromtimestamp(
                event_ms / 1000.0,
                tz=timezone.utc,
            ).isoformat(),
            "trade_id": trade_id,
            "order_id": order_id,
            "side": side,
            "price": _finite(
                trade.get("price"),
                field=f"trades[{index}].price",
                positive=True,
            ),
            "qty": _finite(
                trade.get("qty"),
                field=f"trades[{index}].qty",
                positive=True,
            ),
            "realized_pnl_usdt": _finite(
                trade.get("realizedPnl", 0.0),
                field=f"trades[{index}].realizedPnl",
            ),
            "maker": maker,
            "position_side": str(trade.get("positionSide", "")).strip().upper() or None,
            "source": "binance_fapi_user_trades_export",
        }
        commission = _finite(
            trade.get("commission", 0.0),
            field=f"trades[{index}].commission",
        )
        commission_asset = str(trade.get("commissionAsset", "")).strip().upper()
        if commission_asset == "USDT":
            record["commission_usdt"] = commission
        else:
            record["commission_original"] = commission
            record["commission_asset"] = commission_asset or None
        records.append(record)
    records.sort(key=lambda record: (str(record["event_time_utc"]), str(record["trade_id"])))
    return records, {
        "input_trades": len(trades),
        "linked_trades": len(records),
        "unlinked_trades_filtered": unlinked,
        "duplicate_trades_dropped": duplicates,
    }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivateExecutionIngestError(f"cannot read JSON {path}: {exc}") from exc


def ingest_private_execution_export(
    *,
    source_path: Path,
    linkage_path: Path,
    live_root: Path,
    run_id: str,
    event_completeness: str,
    window_start_utc: datetime | None,
    window_end_utc: datetime | None,
    observed_at_utc: datetime,
) -> dict[str, Any]:
    """Commit one immutable strategy-scoped private execution artifact."""

    observed_at = _aware_utc(observed_at_utc, field="observed_at_utc")
    canonical_run_id = _identity_component(run_id, field="run_id")
    if event_completeness == "event_complete":
        raise PrivateExecutionIngestError(
            "a static import cannot claim event_complete; use snapshot_only or history_complete"
        )
    if event_completeness not in {"snapshot_only", "history_complete"}:
        raise PrivateExecutionIngestError("unsupported static-import completeness")
    if event_completeness == "history_complete":
        if window_start_utc is None or window_end_utc is None:
            raise PrivateExecutionIngestError(
                "history_complete requires explicit window_start_utc and window_end_utc"
            )
    linkage_raw = _load_json(linkage_path)
    if not isinstance(linkage_raw, dict):
        raise PrivateExecutionIngestError("strategy-order linkage must be an object")
    linkage = cast(dict[str, Any], linkage_raw)
    if linkage.get("schema_version") != LINKAGE_SCHEMA_VERSION:
        raise PrivateExecutionIngestError("unsupported strategy-order linkage schema")
    symbol = _symbol(linkage.get("symbol"), field="linkage.symbol")
    strategy_id = _identity_component(
        linkage.get("strategy_id"),
        field="linkage.strategy_id",
    )
    provenance = str(linkage.get("provenance", "")).strip()
    order_ids_raw = linkage.get("order_ids")
    if not provenance:
        raise PrivateExecutionIngestError(
            "linkage provenance is required"
        )
    if provenance.casefold() == "symbol_and_time_only":
        raise PrivateExecutionIngestError(
            "symbol/time-only linkage is ambiguous and not accepted"
        )
    if not isinstance(order_ids_raw, list):
        raise PrivateExecutionIngestError("linkage order_ids must be a list")
    allowed_order_ids = {
        _integer_text(value, field="linkage.order_ids") for value in order_ids_raw
    }
    if not allowed_order_ids:
        raise PrivateExecutionIngestError("strategy linkage order_ids must be non-empty")

    source_raw = _load_json(source_path)
    if isinstance(source_raw, dict):
        trades_raw = source_raw.get("trades")
    else:
        trades_raw = source_raw
    if not isinstance(trades_raw, list) or not all(isinstance(item, dict) for item in trades_raw):
        raise PrivateExecutionIngestError("source must be a trade list or an object with trades")
    trades = [cast(dict[str, Any], item) for item in trades_raw]
    records, audit = normalize_strategy_trades(
        trades,
        symbol=symbol,
        strategy_id=strategy_id,
        run_id=canonical_run_id,
        allowed_order_ids=allowed_order_ids,
    )

    start = (
        _aware_utc(window_start_utc, field="window_start_utc")
        if window_start_utc is not None
        else None
    )
    end = (
        _aware_utc(window_end_utc, field="window_end_utc")
        if window_end_utc is not None
        else None
    )
    if start is not None and end is not None:
        if start > end:
            raise PrivateExecutionIngestError("history window start is after end")
        if end > observed_at:
            raise PrivateExecutionIngestError("history window end is after observed_at_utc")
        for record in records:
            event_at = datetime.fromisoformat(str(record["event_time_utc"]))
            if not start <= event_at <= end:
                raise PrivateExecutionIngestError(
                    "linked trade lies outside the declared complete history window"
                )
    for record in records:
        event_at = datetime.fromisoformat(str(record["event_time_utc"]))
        if event_at > observed_at:
            raise PrivateExecutionIngestError(
                "linked trade event is after observed_at_utc"
            )

    live_date = observed_at.astimezone(_lima_timezone()).date().isoformat()
    run_parent = live_root / live_date / symbol / "private_events" / strategy_id
    run_dir = run_parent / canonical_run_id
    if run_dir.exists():
        raise PrivateExecutionIngestError(
            f"private execution run already exists and is immutable: {run_dir}"
        )
    staging_dir = run_parent / f".{canonical_run_id}.{os.getpid()}.staging"
    if staging_dir.exists():
        raise PrivateExecutionIngestError(
            f"private execution staging directory already exists: {staging_dir}"
        )
    staging_dir.mkdir(parents=True)
    staging_event_path = staging_dir / "events.jsonl"
    staging_manifest_path = staging_dir / "manifest.json"
    event_path = run_dir / "events.jsonl"
    manifest_path = run_dir / "manifest.json"
    event_text = "".join(
        json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        for record in records
    )
    linkage_hash = _sha256_file(linkage_path)
    source_hash = _sha256_file(source_path)
    manifest = {
        "schema_version": PRIVATE_EVENT_MANIFEST_SCHEMA_VERSION,
        "run_id": canonical_run_id,
        "symbol": symbol,
        "strategy_id": strategy_id,
        "status": "complete",
        "updated_at_utc": observed_at.isoformat(),
        "event_path": "events.jsonl",
        "capture_mode": "reviewed_binance_user_trades_export",
        "event_completeness": event_completeness,
        "source_scopes": ["trades", "exact_order_id_linkage"],
        "total_records": len(records),
        "duplicate_records_dropped": audit["duplicate_trades_dropped"],
        "rejected_records": audit["unlinked_trades_filtered"],
        "input_trade_count": audit["input_trades"],
        "linked_order_id_count": len(allowed_order_ids),
        "source_path": str(source_path.resolve()),
        "source_sha256": source_hash,
        "strategy_order_linkage_path": str(linkage_path.resolve()),
        "strategy_order_linkage_sha256": linkage_hash,
        "strategy_order_linkage_provenance": provenance,
        "window_start_utc": start.isoformat() if start is not None else None,
        "window_end_utc": end.isoformat() if end is not None else None,
        "scope_note": (
            "Only exchange order IDs from the reviewed exact-strategy allowlist are included. "
            "This static artifact does not claim continuous user-data-stream completeness."
        ),
    }
    history_seconds = max(
        60.0,
        (observed_at - start).total_seconds() + 60.0 if start is not None else 86400.0,
    )
    try:
        _atomic_write(staging_event_path, event_text)
        _atomic_write(
            staging_manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        load_validated_private_event_window(
            PrivateEventStreamRef(
                event_path=staging_event_path,
                manifest_path=staging_manifest_path,
                symbol=symbol,
                strategy_id=strategy_id,
                run_id=canonical_run_id,
                max_age_seconds=60.0,
                history_window_seconds=history_seconds,
            ),
            now=observed_at,
        )
        if run_dir.exists():
            raise PrivateExecutionIngestError(
                f"private execution run appeared concurrently: {run_dir}"
            )
        os.rename(staging_dir, run_dir)
    finally:
        if staging_dir.exists():
            staging_manifest_path.unlink(missing_ok=True)
            staging_event_path.unlink(missing_ok=True)
            try:
                staging_dir.rmdir()
            except OSError:
                pass
    return {
        "status": "complete",
        **audit,
        "symbol": symbol,
        "strategy_id": strategy_id,
        "run_id": canonical_run_id,
        "event_completeness": event_completeness,
        "event_path": str(event_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "source_sha256": source_hash,
        "strategy_order_linkage_sha256": linkage_hash,
    }


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--strategy-order-linkage", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--live-root", type=Path, default=ROOT / "Live")
    parser.add_argument(
        "--event-completeness",
        choices=("snapshot_only", "history_complete"),
        required=True,
    )
    parser.add_argument("--window-start-utc", type=_parse_utc, default=None)
    parser.add_argument("--window-end-utc", type=_parse_utc, default=None)
    parser.add_argument("--observed-at-utc", type=_parse_utc, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = ingest_private_execution_export(
            source_path=args.source,
            linkage_path=args.strategy_order_linkage,
            live_root=args.live_root,
            run_id=str(args.run_id),
            event_completeness=str(args.event_completeness),
            window_start_utc=args.window_start_utc,
            window_end_utc=args.window_end_utc,
            observed_at_utc=args.observed_at_utc or datetime.now(timezone.utc),
        )
    except (PrivateExecutionIngestError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
