"""Collect candidate-keyed live order-book depth shadow snapshots.

This script starts the data-acquisition path for the depth-aware deployable
winner loop. It captures current Binance USDT-M order-book snapshots for
candidate rows and persists them without modifying gates, configs, or models.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutralgrid.api.binance_client import BinanceClient  # noqa: E402
from neutralgrid.core.config import get_config  # noqa: E402
from neutralgrid.data.depth_shadow import (  # noqa: E402
    DepthShadowTarget,
    append_jsonl,
    filter_fresh_depth_shadow_targets,
    load_depth_shadow_targets,
    make_depth_shadow_record,
    utc_now_iso,
)


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


def _latest_deployment_csv() -> Path:
    paths = sorted((ROOT / "results").glob("deployment_ready_*.csv"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError("No results/deployment_ready_*.csv files found; pass --input explicitly")
    return paths[-1]


def _targets_from_symbols(symbols: list[str], fallback_position_usdt: float | None) -> list[DepthShadowTarget]:
    scan_time_utc = utc_now_iso()
    return [
        DepthShadowTarget(
            symbol=symbol.upper(),
            candidate_id=f"manual_{symbol.upper()}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            scan_time_utc=scan_time_utc,
            position_notional_usdt=fallback_position_usdt,
            source_row_index=None,
            source_path="--symbols",
        )
        for symbol in symbols
    ]


async def _fetch_books(
    client: BinanceClient,
    symbols: list[str],
    *,
    limit: int,
    concurrency: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    semaphore = asyncio.Semaphore(concurrency)
    books: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    async def _one(symbol: str) -> None:
        async with semaphore:
            try:
                books[symbol] = await client.get_order_book(symbol, limit=limit)
            except Exception as exc:  # pragma: no cover - exercised only against live API
                errors[symbol] = repr(exc)

    await asyncio.gather(*(_one(symbol) for symbol in symbols))
    return books, errors


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _fee_context() -> dict[str, Any]:
    cfg = get_config()
    maker_fee = float(cfg.grid.maker_fee)
    taker_fee = float(cfg.grid.taker_fee)
    close_fee_mode = str(getattr(cfg.grid, "close_fee_mode", "maker")).lower()
    close_fee_rate = maker_fee if close_fee_mode == "maker" else taker_fee
    horizon_hours = float(getattr(cfg.barrier, "time_hours", 7.0))
    return {
        "maker_fee_rate": maker_fee,
        "taker_fee_rate": taker_fee,
        "close_fee_mode": close_fee_mode,
        "close_fee_rate": close_fee_rate,
        "round_trip_fee_pct": (maker_fee + close_fee_rate) * 100.0,
        "funding_horizon_hours": horizon_hours,
        "funding_interval_hours": 8.0,
    }


def _monotonic() -> float:
    """Return a monotonic timestamp; isolated for deterministic collector tests."""
    return time.monotonic()


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _depth_window_coverage_seconds(manifest: dict[str, Any]) -> float | None:
    started_at = _parse_utc(manifest.get("started_at_utc"))
    last_capture = _parse_utc(
        manifest.get("last_successful_capture_time_utc")
        or manifest.get("last_capture_time_utc")
    )
    if started_at is None or last_capture is None:
        return None
    return max(0.0, float((last_capture - started_at).total_seconds()))


def _required_depth_window_seconds(duration_seconds: float, interval_seconds: float) -> float | None:
    if duration_seconds <= 0:
        return None
    return max(0.0, duration_seconds - max(0.0, interval_seconds))


async def _fetch_premium_context(
    client: BinanceClient,
    symbols: list[str],
    *,
    concurrency: int,
    fee_context: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    semaphore = asyncio.Semaphore(concurrency)
    contexts: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    async def _one(symbol: str) -> None:
        async with semaphore:
            context = dict(fee_context)
            try:
                premium = await client.get_premium_index(symbol)
                funding_rate = _float_or_none(premium.get("lastFundingRate"))
                mark_price = _float_or_none(premium.get("markPrice"))
                index_price = _float_or_none(premium.get("indexPrice"))
                basis_pct = (
                    (mark_price - index_price) / index_price * 100.0
                    if mark_price is not None and index_price is not None and index_price > 0
                    else None
                )
                context.update(
                    {
                        "funding_rate": funding_rate,
                        "estimated_abs_funding_pct": (
                            abs(funding_rate) * (float(fee_context["funding_horizon_hours"]) / 8.0) * 100.0
                            if funding_rate is not None
                            else None
                        ),
                        "mark_price": mark_price,
                        "index_price": index_price,
                        "basis_pct": basis_pct,
                        "next_funding_time_ms": premium.get("nextFundingTime"),
                    }
                )
            except Exception as exc:  # pragma: no cover - exercised only against live API
                errors[symbol] = repr(exc)
                context.update(
                    {
                        "funding_rate": None,
                        "estimated_abs_funding_pct": None,
                        "mark_price": None,
                        "index_price": None,
                        "basis_pct": None,
                        "next_funding_time_ms": None,
                    }
                )
            contexts[symbol] = context

    await asyncio.gather(*(_one(symbol) for symbol in symbols))
    return contexts, errors


def _write_targets(path: Path, targets: list[DepthShadowTarget]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "symbol",
        "candidate_id",
        "scan_time_utc",
        "position_notional_usdt",
        "source_row_index",
        "source_path",
    ]
    pd.DataFrame([target.__dict__ for target in targets], columns=pd.Index(columns)).to_csv(path, index=False)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _write_rejected_targets(path: Path, rejected: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([target.__dict__ for target in rejected]).to_csv(path, index=False)


async def collect_depth_shadow(args: argparse.Namespace) -> int:
    input_path = Path(args.input) if args.input else _latest_deployment_csv()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    audit_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / "audits" / f"depth_shadow_{run_id}"
    live_root = ROOT / "Live" / datetime.now(timezone.utc).strftime("%Y-%m-%d")

    symbols = [s.upper() for s in args.symbols] if args.symbols else None
    if args.symbols and not args.input:
        targets = _targets_from_symbols(args.symbols, args.fallback_position_usdt)
    else:
        targets = load_depth_shadow_targets(
            input_path,
            max_candidates=args.max_candidates,
            symbols=symbols,
            fallback_position_usdt=args.fallback_position_usdt,
        )
    if not targets:
        raise ValueError("No depth-shadow targets resolved")

    original_target_count = len(targets)
    rejected_targets: list[Any] = []
    if not args.allow_stale_targets:
        now = datetime.now(timezone.utc)
        targets, rejected_targets = filter_fresh_depth_shadow_targets(
            targets,
            now=now,
            max_scan_age_seconds=float(args.max_scan_age_seconds),
        )

    unique_symbols = sorted({target.symbol for target in targets})
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "git_head": _git_output(["rev-parse", "--short", "HEAD"]),
        "git_status_short": _git_output(["status", "--short"]),
        "input_path": str(input_path) if not args.symbols or args.input else "--symbols",
        "audit_dir": str(audit_dir),
        "live_root": str(live_root),
        "original_target_count": original_target_count,
        "target_count": len(targets),
        "rejected_target_count": len(rejected_targets),
        "symbol_count": len(unique_symbols),
        "symbols": unique_symbols,
        "duration_seconds": args.duration_seconds,
        "interval_seconds": args.interval_seconds,
        "iteration_timeout_seconds": args.iteration_timeout_seconds,
        "planned_snapshot_count_per_candidate": (
            max(1, int(args.duration_seconds // args.interval_seconds) + 1)
            if args.duration_seconds > 0
            else 1
        ),
        "limit": args.limit,
        "top_n": args.top_n,
        "participation_rate": args.participation_rate,
        "fallback_position_usdt": args.fallback_position_usdt,
        "fee_context": _fee_context(),
        "freshness_guard": {
            "enabled": not args.allow_stale_targets,
            "max_scan_age_seconds": args.max_scan_age_seconds,
            "rejected_targets_csv": str(audit_dir / "rejected_targets.csv"),
        },
        "command": " ".join(sys.argv),
        "note": (
            "Current REST depth snapshots only; not historical reconstruction. "
            "Freshness guard prevents stale scan rows from becoming ex-ante evidence."
        ),
    }
    if rejected_targets:
        _write_rejected_targets(audit_dir / "rejected_targets.csv", rejected_targets)
    if not targets:
        manifest["status"] = "blocked_no_fresh_targets"
        _write_targets(audit_dir / "targets.csv", targets)
        _write_manifest(audit_dir / "depth_shadow_manifest.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 2

    if args.dry_run:
        print(json.dumps({**manifest, "dry_run": True}, indent=2))
        return 0

    _write_targets(audit_dir / "targets.csv", targets)
    _write_manifest(audit_dir / "depth_shadow_manifest.json", manifest)

    client = BinanceClient()
    fee_context = _fee_context()
    all_records: list[dict[str, Any]] = []
    errors_by_iteration: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc)
    started_monotonic = _monotonic()
    deadline_monotonic = started_monotonic + max(0.0, float(args.duration_seconds))
    manifest["started_at_utc"] = started_at.replace(microsecond=0).isoformat()
    manifest["planned_completed_at_utc"] = (
        started_at + timedelta(seconds=max(0.0, float(args.duration_seconds)))
    ).replace(microsecond=0).isoformat()
    manifest["status"] = "running"
    _write_manifest(audit_dir / "depth_shadow_manifest.json", manifest)

    iteration = 0
    try:
        while True:
            capture_ts = utc_now_iso()
            try:
                books, errors = await asyncio.wait_for(
                    _fetch_books(
                        client,
                        unique_symbols,
                        limit=args.limit,
                        concurrency=args.concurrency,
                    ),
                    timeout=float(args.iteration_timeout_seconds),
                )
                contexts, context_errors = await asyncio.wait_for(
                    _fetch_premium_context(
                        client,
                        unique_symbols,
                        concurrency=args.concurrency,
                        fee_context=fee_context,
                    ),
                    timeout=float(args.iteration_timeout_seconds),
                )
            except TimeoutError:
                errors_by_iteration.append(
                    {
                        "iteration": iteration,
                        "capture_time_utc": capture_ts,
                        "errors": {
                            "__iteration_timeout__": (
                                f"iteration exceeded {float(args.iteration_timeout_seconds):.3f}s"
                            )
                        },
                    }
                )
                manifest["last_attempt_time_utc"] = capture_ts
                manifest["iteration_count"] = iteration + 1
                manifest["records_written_so_far"] = len(all_records)
                manifest["error_iterations_so_far"] = len(errors_by_iteration)
                manifest["elapsed_seconds_so_far"] = round(_monotonic() - started_monotonic, 3)
                _write_manifest(audit_dir / "depth_shadow_manifest.json", manifest)
                iteration += 1
                if args.duration_seconds <= 0:
                    break
                remaining = deadline_monotonic - _monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(float(args.interval_seconds), remaining))
                continue
            errors = {**errors, **{f"{symbol}:premium": err for symbol, err in context_errors.items()}}
            by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for target in targets:
                book = books.get(target.symbol)
                if book is None:
                    continue
                record = make_depth_shadow_record(
                    target,
                    book,
                    capture_time_utc=capture_ts,
                    top_n=args.top_n,
                    participation_rate=args.participation_rate,
                    market_context=contexts.get(target.symbol, fee_context),
                )
                record["iteration"] = iteration
                all_records.append(record)
                by_symbol[target.symbol].append(record)

            new_count = sum(len(v) for v in by_symbol.values())
            if new_count:
                append_jsonl(audit_dir / "depth_shadow_records.jsonl", all_records[-new_count:])
            for symbol, records in by_symbol.items():
                append_jsonl(live_root / symbol / f"depth_shadow_{run_id}.jsonl", records)
            if errors:
                errors_by_iteration.append({"iteration": iteration, "capture_time_utc": capture_ts, "errors": errors})
            manifest["last_capture_time_utc"] = capture_ts
            if new_count:
                manifest["last_successful_capture_time_utc"] = capture_ts
            manifest["iteration_count"] = iteration + 1
            manifest["records_written_so_far"] = len(all_records)
            manifest["error_iterations_so_far"] = len(errors_by_iteration)
            manifest["elapsed_seconds_so_far"] = round(_monotonic() - started_monotonic, 3)
            _write_manifest(audit_dir / "depth_shadow_manifest.json", manifest)

            iteration += 1
            if args.duration_seconds <= 0:
                break
            remaining = deadline_monotonic - _monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(float(args.interval_seconds), remaining))
    finally:
        await client.close()

    summary_path = audit_dir / "depth_shadow_summary.csv"
    if all_records:
        summary_cols = [col for col in all_records[0].keys() if col not in {"bids", "asks"}]
        pd.DataFrame([{col: record.get(col) for col in summary_cols} for record in all_records]).to_csv(
            summary_path,
            index=False,
        )
    _write_manifest(audit_dir / "depth_shadow_errors.json", {"errors_by_iteration": errors_by_iteration})
    manifest["completed_at_utc"] = utc_now_iso()
    manifest["records_written"] = len(all_records)
    manifest["error_iterations"] = len(errors_by_iteration)
    manifest["iteration_count"] = iteration
    manifest["elapsed_seconds"] = round(_monotonic() - started_monotonic, 3)
    manifest["actual_snapshot_count_per_candidate"] = (
        int(len(all_records) / len(targets)) if targets and len(all_records) % len(targets) == 0 else None
    )
    depth_window_seconds = _depth_window_coverage_seconds(manifest)
    required_window_seconds = _required_depth_window_seconds(
        float(args.duration_seconds),
        float(args.interval_seconds),
    )
    manifest["depth_window_seconds"] = depth_window_seconds
    manifest["required_depth_window_seconds"] = required_window_seconds
    depth_window_complete = (
        required_window_seconds is None
        or (depth_window_seconds is not None and depth_window_seconds >= required_window_seconds)
    )
    if not all_records:
        manifest["status"] = "blocked_no_records"
    elif depth_window_complete:
        manifest["status"] = "complete"
    else:
        manifest["status"] = "complete_depth_window_incomplete"
    _write_manifest(audit_dir / "depth_shadow_manifest.json", manifest)
    print(json.dumps({"audit_dir": str(audit_dir), "records_written": len(all_records)}, indent=2))
    return 0 if all_records else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Candidate CSV/Parquet/XLSX. Defaults to latest results/deployment_ready_*.csv")
    parser.add_argument("--symbols", nargs="*", help="Manual symbol list; used when no --input is supplied")
    parser.add_argument("--output-dir", help="Audit output directory")
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--duration-seconds", type=int, default=0)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--iteration-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--participation-rate", type=float, default=0.10)
    parser.add_argument("--fallback-position-usdt", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument(
        "--max-scan-age-seconds",
        type=float,
        default=900.0,
        help="Reject candidate rows whose scan timestamp is older than this. Default 900 seconds.",
    )
    parser.add_argument(
        "--allow-stale-targets",
        action="store_true",
        help="Collect anyway for diagnostics; output is not valid ex-ante training evidence.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be > 0")
    if args.iteration_timeout_seconds <= 0:
        parser.error("--iteration-timeout-seconds must be > 0")
    if args.max_scan_age_seconds <= 0:
        parser.error("--max-scan-age-seconds must be > 0")
    if args.top_n <= 0 or args.limit <= 0:
        parser.error("--top-n and --limit must be > 0")
    if args.top_n > args.limit:
        parser.error("--top-n cannot exceed --limit")
    if args.max_candidates is not None and args.max_candidates <= 0:
        parser.error("--max-candidates must be > 0")
    return args


def main() -> None:
    raise SystemExit(asyncio.run(collect_depth_shadow(parse_args())))


if __name__ == "__main__":
    main()
