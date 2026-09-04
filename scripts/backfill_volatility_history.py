"""Backfill checksum-verified one-minute mark/last history for live symbols.

The command writes only to the canonical PriceSeries store and a run-scoped
audit directory.  Missing or unverifiable Binance Vision archives are recorded
as blockers; they are never replaced with interpolated candles.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.data.binance_vision.downloader import (
    download_kline_batch,
    download_mark_price_kline_batch,
)
from neutralgrid.data.binance_vision.ingest import KLINE_COLUMNS, parse_zip
from neutralgrid.data.binance_vision.urls import daily_dates
from neutralgrid.data.price_series.ps_store import PriceStore
from neutralgrid.data.price_series.ps_types import Candle, SeriesKind
from neutralgrid.live.decision.volatility import (
    VolatilityError,
    audit_single_symbol_readiness,
    build_rv_examples,
    load_price_store_frame,
    load_volatility_contract,
    validate_price_frame,
)
from scripts.run_live_volatility_loop import (
    LiveVolatilityLoopError,
    load_roster,
    reconcile_symbol_gaps,
    reconcile_symbol_tail,
)


UTC = timezone.utc
logger = logging.getLogger(__name__)


class VolatilityBackfillError(RuntimeError):
    """The requested historical acquisition cannot satisfy its contract."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


async def _verified_listing_boundary(symbol: str) -> tuple[date, dict[str, Any]]:
    """Read the exchange-reported USD-M onboard date without inferring it."""

    client = BinanceClient()
    try:
        payload = await client.get_exchange_info(symbol)
    finally:
        await client.close()
    symbols = payload.get("symbols") if isinstance(payload, dict) else None
    entries = symbols if isinstance(symbols, list) else [payload]
    matches = [
        item
        for item in entries
        if isinstance(item, dict) and str(item.get("symbol", "")).upper() == symbol.upper()
    ]
    if len(matches) != 1:
        raise VolatilityBackfillError(
            f"{symbol}: exchangeInfo did not return one exact symbol record"
        )
    onboard_value = matches[0].get("onboardDate")
    if isinstance(onboard_value, bool) or not isinstance(onboard_value, (int, float)):
        raise VolatilityBackfillError(f"{symbol}: exchangeInfo onboardDate is unavailable")
    onboard_ms = int(onboard_value)
    if onboard_ms <= 0:
        raise VolatilityBackfillError(f"{symbol}: exchangeInfo onboardDate is invalid")
    onboard_utc = datetime.fromtimestamp(onboard_ms / 1000.0, tz=UTC)
    return onboard_utc.date(), {
        "source": "GET /fapi/v1/exchangeInfo",
        "symbol": symbol.upper(),
        "onboard_date_ms": onboard_ms,
        "onboard_at_utc": onboard_utc.isoformat(),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _archive_frame(paths: Sequence[Path], *, symbol: str, series_kind: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    for path in paths:
        checksum_path = path.with_name(path.name + ".CHECKSUM")
        if not checksum_path.is_file():
            raise VolatilityBackfillError(f"checksum file is missing for {path}")
        raw = parse_zip(path)
        if list(raw.columns) != KLINE_COLUMNS:
            raise VolatilityBackfillError(f"{path}: unexpected kline columns")
        frame = raw.copy()
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        open_raw = cast(pd.Series, pd.to_numeric(frame["open_time"], errors="coerce"))
        close_raw = cast(pd.Series, pd.to_numeric(frame["close_time"], errors="coerce"))
        open_values = open_raw.to_numpy(dtype=float, na_value=np.nan)
        close_values = close_raw.to_numpy(dtype=float, na_value=np.nan)
        if not bool(np.isfinite(open_values).all()) or not bool(np.isfinite(close_values).all()):
            raise VolatilityBackfillError(f"{path}: non-finite archive timestamps")
        # Binance archives may encode epoch timestamps in milliseconds or
        # microseconds.  The canonical PriceSeries store is milliseconds.
        frame["open_time_ms"] = np.where(
            open_values >= 100_000_000_000_000.0,
            np.floor(open_values / 1000.0),
            open_values,
        ).astype("int64")
        frame["close_time_ms"] = np.where(
            close_values >= 100_000_000_000_000.0,
            np.floor(close_values / 1000.0),
            close_values,
        ).astype("int64")
        frame["is_final"] = True
        frames.append(
            frame.loc[
                :,
                [
                    "open_time_ms",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time_ms",
                    "is_final",
                ],
            ]
        )
    if not frames:
        return pd.DataFrame(), {"archive_files": 0, "rows": 0}
    combined = pd.concat(frames, ignore_index=True)
    _, audit = validate_price_frame(
        combined,
        symbol=symbol,
        series_kind=series_kind,
    )
    combined = (
        combined.sort_values("open_time_ms")
        .drop_duplicates(subset="open_time_ms", keep="first")
        .reset_index(drop=True)
    )
    return combined, {
        "archive_files": len(paths),
        "rows": len(combined),
        "exact_duplicates": audit.exact_duplicates,
        "gap_count": audit.gap_count,
        "missing_minutes": audit.missing_minutes,
    }


def _append_audited(
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
    combined = incoming if existing.empty else pd.concat([existing, incoming], ignore_index=True)
    combined = combined.sort_values("open_time_ms", kind="stable").reset_index(drop=True)
    _, audit = validate_price_frame(
        combined,
        symbol=symbol,
        series_kind=kind.value,
    )
    existing_times: set[int] = set()
    if not existing.empty:
        existing_open_times = cast(
            pd.Series,
            pd.to_numeric(cast(pd.Series, existing["open_time_ms"]), errors="coerce"),
        )
        existing_times = set(existing_open_times.dropna().astype("int64").tolist())
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


async def _download_symbol(
    *,
    symbol: str,
    archive_dates: list[date],
    cache_dir: Path,
    max_concurrency: int,
    force: bool,
) -> tuple[list[Path], list[Path]]:
    mark_task = download_mark_price_kline_batch(
        symbol=symbol,
        interval="1m",
        dates=archive_dates,
        granularity="daily",
        cache_dir=cache_dir,
        max_concurrency=max_concurrency,
        force=force,
    )
    last_task = download_kline_batch(
        symbol=symbol,
        interval="1m",
        dates=archive_dates,
        granularity="daily",
        cache_dir=cache_dir / "regular",
        max_concurrency=max_concurrency,
        force=force,
        require_checksum=True,
    )
    return await asyncio.gather(mark_task, last_task)


async def run_backfill(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_volatility_contract(args.contract)
    symbols = tuple(sorted({str(symbol).upper() for symbol in args.symbol}))
    if args.cycle_manifest is not None:
        roster = load_roster(
            manifest_path=args.cycle_manifest.resolve(),
            audit_root=args.cycle_manifest.resolve().parent.parent,
            max_age_seconds=float(contract.maximum_roster_age_seconds),
        )
        manifest_symbols = tuple(entry.symbol for entry in roster.entries)
        if symbols and symbols != tuple(sorted(manifest_symbols)):
            raise VolatilityBackfillError(
                "explicit symbols do not exactly match the supplied active roster"
            )
        symbols = tuple(sorted(manifest_symbols))
    if not symbols:
        raise VolatilityBackfillError("at least one symbol or --cycle-manifest is required")
    run_id = _utc_now().strftime("%Y%m%dT%H%M%S%fZ")
    audit_dir = args.audit_root.resolve() / run_id
    store = PriceStore(store_dir=args.price_store.resolve())
    results: list[dict[str, Any]] = []
    yesterday = _utc_now().date() - timedelta(days=1)
    for symbol in symbols:
        listing_boundary, listing_evidence = await _verified_listing_boundary(symbol)
        history_days = max(contract.minimum_history_days, args.initial_days)
        symbol_report: dict[str, Any] = {
            "symbol": symbol,
            "listing_boundary": listing_evidence,
            "attempts": [],
        }
        while history_days <= args.max_days:
            requested_start = yesterday - timedelta(days=history_days - 1)
            start = max(requested_start, listing_boundary)
            dates = daily_dates(start, yesterday)
            mark_paths, last_paths = await _download_symbol(
                symbol=symbol,
                archive_dates=dates,
                cache_dir=args.cache_dir.resolve(),
                max_concurrency=args.max_concurrency,
                force=args.force,
            )
            missing_mark = len(dates) - len(mark_paths)
            missing_last = len(dates) - len(last_paths)
            attempt: dict[str, Any] = {
                "history_days_requested": history_days,
                "requested_start_date": requested_start.isoformat(),
                "start_date": start.isoformat(),
                "end_date": yesterday.isoformat(),
                "clamped_to_verified_listing_boundary": start > requested_start,
                "requested_archives_per_series": len(dates),
                "mark_archives": len(mark_paths),
                "last_archives": len(last_paths),
                "missing_mark_archives": missing_mark,
                "missing_last_archives": missing_last,
            }
            mark_frame, mark_archive_audit = _archive_frame(
                mark_paths,
                symbol=symbol,
                series_kind="mark_kline",
            )
            last_frame, last_archive_audit = _archive_frame(
                last_paths,
                symbol=symbol,
                series_kind="last_kline",
            )
            attempt["mark_archive_audit"] = mark_archive_audit
            attempt["last_archive_audit"] = last_archive_audit
            attempt["mark_store"] = _append_audited(
                store,
                price_store_root=args.price_store.resolve(),
                symbol=symbol,
                kind=SeriesKind.MARK_KLINE,
                incoming=mark_frame,
            )
            attempt["last_store"] = _append_audited(
                store,
                price_store_root=args.price_store.resolve(),
                symbol=symbol,
                kind=SeriesKind.LAST_KLINE,
                incoming=last_frame,
            )
            rest_client = BinanceClient()
            try:
                attempt["rest_gap_reconciliation"] = await reconcile_symbol_gaps(
                    rest_client,
                    store,
                    price_store_root=args.price_store.resolve(),
                    symbol=symbol,
                    max_requests=args.max_gap_rest_requests,
                    now=_utc_now(),
                    scope_start_ms=int(
                        datetime.combine(start, datetime.min.time(), tzinfo=UTC).timestamp()
                        * 1000
                    ),
                    scope_end_ms=int(
                        datetime.combine(
                            yesterday,
                            datetime.max.time(),
                            tzinfo=UTC,
                        ).timestamp()
                        * 1000
                    ),
                )
                attempt["rest_tail_reconciliation"] = await reconcile_symbol_tail(
                    rest_client,
                    store,
                    price_store_root=args.price_store.resolve(),
                    symbol=symbol,
                    now=_utc_now(),
                )
            finally:
                await rest_client.close()
            stored_mark = load_price_store_frame(
                args.price_store.resolve(),
                symbol=symbol,
                series_kind="mark_kline",
            )
            scope_start_ms = int(
                datetime.combine(start, datetime.min.time(), tzinfo=UTC).timestamp()
                * 1000
            )
            if "open_time_ms" in stored_mark.columns:
                stored_open_times = cast(
                    pd.Series,
                    pd.to_numeric(
                        cast(pd.Series, stored_mark["open_time_ms"]),
                        errors="coerce",
                    ),
                )
                scoped_mark = stored_mark.loc[
                    stored_open_times >= scope_start_ms
                ].copy()
            else:
                scoped_mark = stored_mark.copy()
            normalized_mark, stored_mark_audit = validate_price_frame(
                scoped_mark,
                symbol=symbol,
                series_kind="mark_kline",
            )
            calendar_days = (
                0
                if normalized_mark.empty
                else int(
                    cast(pd.Series, normalized_mark["open_time"])
                    .dt.floor("D")
                    .nunique()
                )
            )
            examples, example_audit = build_rv_examples(
                scoped_mark,
                symbol=symbol,
                contract=contract,
            )
            readiness = audit_single_symbol_readiness(
                examples,
                symbol=symbol,
                contract=contract,
            )
            readiness["calendar_days"] = calendar_days
            readiness["minimum_calendar_days"] = contract.minimum_history_days
            readiness["history_gate"] = calendar_days >= contract.minimum_history_days
            if not readiness["history_gate"]:
                readiness["status"] = "blocked"
                readiness["reason"] = "minimum_calendar_history_not_met"
            attempt["example_audit"] = example_audit
            attempt["readiness_scope_start_utc"] = datetime.combine(
                start,
                datetime.min.time(),
                tzinfo=UTC,
            ).isoformat()
            attempt["stored_mark_audit"] = stored_mark_audit.__dict__
            attempt["readiness"] = readiness
            symbol_report["attempts"].append(attempt)
            if readiness["status"] == "ready":
                symbol_report["status"] = "ready"
                break
            if start == listing_boundary and requested_start <= listing_boundary:
                symbol_report["status"] = "blocked_at_verified_listing_boundary"
                symbol_report["blocker"] = (
                    "origin floors remain insufficient at exchangeInfo onboardDate"
                )
                break
            history_days += args.extension_days
        if symbol_report.get("status") != "ready":
            symbol_report.setdefault("status", "blocked_history_or_origin_floor")
        results.append(symbol_report)
    report = {
        "schema_version": "neutralgrid_volatility_history_backfill_v1",
        "run_id": run_id,
        "created_at_utc": _utc_now().isoformat(),
        "status": "ready" if all(item["status"] == "ready" for item in results) else "blocked",
        "contract_path": str(contract.path),
        "contract_sha256": contract.contract_sha256,
        "symbols": list(symbols),
        "price_store": str(args.price_store.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "results": results,
    }
    _atomic_json(audit_dir / "manifest.json", report)
    _atomic_json(args.audit_root.resolve() / "manifest.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append", default=[])
    parser.add_argument("--cycle-manifest", type=Path, default=None)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "config" / "live_volatility_forecast_v1.json",
    )
    parser.add_argument("--price-store", type=Path, default=ROOT / "data" / "price_store")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "cache" / "volatility_archives",
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=ROOT / "outputs" / "audits" / "live_volatility_backfill",
    )
    parser.add_argument("--initial-days", type=int, default=90)
    parser.add_argument("--extension-days", type=int, default=30)
    parser.add_argument("--max-days", type=int, default=365)
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--max-gap-rest-requests", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    for field in (
        "initial_days",
        "extension_days",
        "max_days",
        "max_concurrency",
        "max_gap_rest_requests",
    ):
        if int(getattr(args, field)) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.max_days < args.initial_days:
        parser.error("--max-days must be at least --initial-days")
    if not args.symbol and args.cycle_manifest is None:
        parser.error("provide --symbol or --cycle-manifest")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    try:
        report = asyncio.run(run_backfill(args))
    except (
        LiveVolatilityLoopError,
        VolatilityBackfillError,
        VolatilityError,
        OSError,
        ValueError,
    ) as exc:
        logger.error("volatility history backfill failed: %s", exc)
        return 2
    logger.info("volatility history backfill status=%s", report["status"])
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
