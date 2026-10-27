"""Capture unfiltered account income; the required symbol names storage context only."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.live.decision.account_income import (
    collect_account_income, fetch_income, validate_interval, validate_limits, validate_storage_symbol,
)
from neutralgrid.live.decision.authenticated_evidence import HOST, credential_scope
from neutralgrid.live.decision.state_action_reward import atomic_create_json, utc

logger = logging.getLogger(__name__)


def now() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(value: str) -> datetime:
    try:
        return utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("explicit timezone-aware ISO-8601 timestamp required") from exc


async def capture(args: argparse.Namespace) -> dict[str, Any]:
    validate_interval(args.start_at, args.end_at, now())
    validate_storage_symbol(args.storage_symbol)
    validate_limits(args.max_requests, 1000, args.max_records)
    # HTTP client INFO output can contain signed request URLs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    output = args.output_dir.resolve()
    if ROOT / "reports" not in output.parents or output.exists():
        raise ValueError("fresh output directory under reports required")
    output.mkdir(parents=True)
    report: dict[str, Any] = {"started_at": now().isoformat(), "errors": [], "result": None,
                              "bot_attribution": "none", "training_eligible": False,
                              "promotion_eligible": False, "influence_enabled": False, "submitted_exchange_orders": 0}
    client: BinanceClient | None = None
    try:
        client = BinanceClient()
        if not client.is_authenticated or client.base_url.rstrip("/") != HOST:
            raise ValueError("production authentication required")
        scope = credential_scope(client.api_key)
        await client.sync_time()

        async def fetch(lower: int, upper: int, limit: int) -> Any:
            assert client is not None
            return await fetch_income(client, lower, upper, limit)

        report["result"] = await collect_account_income(fetch=fetch, account_scope=scope,
            storage_symbol=args.storage_symbol, start=args.start_at, end=args.end_at,
            live_root=ROOT / "Live", now=now, max_requests=args.max_requests, max_records=args.max_records)
    except Exception as exc:
        report["errors"].append({"stage": "capture", "error_type": type(exc).__name__})
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception as exc:
                report["errors"].append({"stage": "close", "error_type": type(exc).__name__})
    result = report["result"]
    blocked = bool(report["errors"] or result is None or result["rejected"] or result["gaps"])
    report.update(finished_at=now().isoformat(), status="partial_or_blocked" if blocked else "captured_account_evidence_only")
    atomic_create_json(output / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-at", type=parse_time, required=True)
    parser.add_argument("--end-at", type=parse_time, required=True)
    parser.add_argument("--storage-symbol", required=True, help="folder context only; never sent as an API filter")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=32)
    parser.add_argument("--max-records", type=int, default=4096)
    args = parser.parse_args(argv)
    if not 1 <= args.max_requests <= 1024 or not 1 <= args.max_records <= 10000:
        parser.error("acquisition budget outside supported bounds")
    try:
        report = asyncio.run(capture(args))
    except (ValueError, OSError, RuntimeError) as exc:
        logger.error("Account evidence capture failed: %s", type(exc).__name__)
        return 2
    logger.warning("Account income capture: %s; report=%s", report["status"], args.output_dir / "report.json")
    return 2 if report["status"] == "partial_or_blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
