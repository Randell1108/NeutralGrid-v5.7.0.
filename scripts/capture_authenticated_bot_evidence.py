"""Read-only authenticated evidence collection; no trading or model activation."""
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
from neutralgrid.live.decision.authenticated_evidence import (
    HOST, PRODUCER_SHA256, SCHEMA, admit_fills, capture_pnl_path, collect_history, credential_scope,
    publish, reviewed_ownership, source_ref,
)
from neutralgrid.live.decision.evidence_cohort import collect_observations, exchange_ticks, read_json
from neutralgrid.live.decision.exact_bot_ledger import verify_registration
from neutralgrid.live.decision.forecast_contract import forecast_from_reported_pnl_path
from neutralgrid.live.decision.state_action_reward import BotIdentity, atomic_create_json, utc

logger = logging.getLogger(__name__)


def now() -> datetime:
    return datetime.now(timezone.utc)


def _api_bound(value: Any, *, name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be an aware ISO-8601 timestamp")
    try:
        return utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError as exc:
        raise ValueError(f"{name} must be an aware ISO-8601 timestamp") from exc


async def capture(args: argparse.Namespace) -> dict[str, Any]:
    # This is an acquisition cutoff, not a forecast deadline or terminal date.
    # Freeze it before any I/O so every deployment has the same requested end.
    run_cutoff = utc(now())
    requested_start = _api_bound(getattr(args, "start_at", None), name="start-at")
    requested_end = _api_bound(getattr(args, "end_at", None), name="end-at")
    api_end = requested_end if requested_end is not None else run_cutoff
    if api_end > run_cutoff or (requested_start is not None and requested_start > run_cutoff):
        raise ValueError("API history bounds cannot be later than the immutable run cutoff")
    if requested_start is not None and requested_start > api_end:
        raise ValueError("API history start must not follow its end")
    max_requests = getattr(args, "max_requests_per_endpoint", 32)
    if isinstance(max_requests, bool) or not isinstance(max_requests, int) or not 1 <= max_requests <= 1024:
        raise ValueError("request budget must be between 1 and 1024")
    live = (ROOT / "Live").resolve()
    output = args.output_dir.resolve()
    if ROOT / "reports" not in output.parents or output.exists():
        raise ValueError("output must be a fresh directory under reports")
    output.mkdir(parents=True)
    report: dict[str, Any] = {"schema_version": SCHEMA, "started_at": run_cutoff.isoformat(),
                              "producer_sha256": PRODUCER_SHA256,
                              "pnl_paths": [], "forecast_contracts": [], "forecast_blocks": [],
                              "acquisitions": [], "errors": [], "skipped_api_deployments": [],
                              "api_history_window": {"run_cutoff_at": run_cutoff.isoformat(),
                                  "requested_start_at": requested_start.isoformat() if requested_start is not None else None,
                                  "requested_end_at": requested_end.isoformat() if requested_end is not None else None,
                                  "effective_end_at": api_end.isoformat(),
                                  "start_rule": "max(deployment, requested_start)",
                                  "scope": "authenticated_api_histories_only"},
                              "training_eligible": False, "promotion_eligible": False,
                              "influence_enabled": False, "submitted_exchange_orders": 0}
    registrations = args.registration or (sorted(live.glob("????-??-??/*/exact_bot_ledger/*/registration.json"))
                                           if args.all_registered else [])
    bots: dict[str, BotIdentity] = {}
    for path in registrations:
        try:
            raw, _ = read_json(path)
            bot = verify_registration(raw, path.resolve(), live)
            bots[bot.key] = bot
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            report["errors"].append({"stage": "registration", "path": str(path), "error_type": type(exc).__name__})
    if args.capture_pnl:
        observations, rejected, _ = collect_observations(live)
        report["rejected_pnl_observations"] = rejected
        grouped: dict[str, tuple[BotIdentity, list[Path]]] = {}
        for row in observations:
            bot = BotIdentity(symbol=row["symbol"], strategy_id=row["strategy_id"], deployed_at=row["deploy_ts"])
            grouped.setdefault(bot.key, (bot, []))[1].append(Path(row["_source_path"]))
        for bot, paths in grouped.values():
            try:
                path = capture_pnl_path(paths, bot=bot, live_root=live, now=now())
                report["pnl_paths"].append(source_ref(path).model_dump())
                try:
                    contract = forecast_from_reported_pnl_path(path, live_root=live)
                    contract_path = publish(contract.model_dump(mode="json"), bot=bot, live_root=live,
                                            at=now(), category="forecast_contracts")
                    report["forecast_contracts"].append(source_ref(contract_path).model_dump())
                except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                    report["forecast_blocks"].append({"bot": bot.key, "reason": str(exc)[:300]})
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                report["errors"].append({"stage": "pnl_path", "bot": bot.key, "error_type": type(exc).__name__})
    acquisition_windows: list[tuple[BotIdentity, datetime, datetime]] = []
    if args.fetch:
        if not bots:
            report["errors"].append({"stage": "fetch", "error_type": "VerifiedRegistrationsRequired"})
        for bot in sorted(bots.values(), key=lambda b: (b.deployed_at, b.key), reverse=True):
            if bot.deployed_at > run_cutoff:
                report["errors"].append({"stage": "fetch", "bot": bot.key, "error_type": "FutureDeployment"})
            elif bot.deployed_at > api_end:
                report["skipped_api_deployments"].append({"bot": bot.key,
                    "reason": "deployment_after_requested_end"})
            else:
                start = max(bot.deployed_at, requested_start) if requested_start is not None else bot.deployed_at
                acquisition_windows.append((bot, start, api_end))
    client: BinanceClient | None = None
    try:
        if args.fetch and acquisition_windows:
            client = BinanceClient()
            if not client.is_authenticated or client.base_url.rstrip("/") != HOST:
                raise ValueError("configured production authentication required")
            scope = credential_scope(client.api_key)
            report["account_scope"] = scope
            report["scope_basis"] = "credential_fingerprint_not_verified_account_uid"
            await client.sync_time()
            tick_requested = now()
            exchange_info = await client.get_exchange_info()
            tick_received = now()
            ticks = exchange_ticks(exchange_info)
            report["verified_registrations_selected"] = len(bots)
            for bot, start, end in acquisition_windows:
                tick_path = publish({"schema_version": SCHEMA, "record_type": "current_exchange_filter",
                                     "producer_sha256": PRODUCER_SHA256,
                                     "requested_at_utc": tick_requested.isoformat(), "received_at_utc": tick_received.isoformat(),
                                     "symbol": bot.symbol, "tick_size": ticks.get(bot.symbol),
                                     "exchange_info": {"symbols": [s for s in exchange_info.get("symbols", []) if s.get("symbol") == bot.symbol]},
                                     "historical_precision_proven": False},
                                    bot=bot, live_root=live, at=now(), category="responses")
                owners: list[Path] = []
                for mapping_path in args.ownership_mapping:
                    mapping, _ = read_json(mapping_path)
                    if mapping.get("bot") == bot.model_dump(mode="json"):
                        owners.extend(reviewed_ownership(source=source_ref(mapping_path), bot=bot,
                                                         account_scope=scope, live_root=live, now=now()))

                async def trades(lower: int, upper: int) -> Any:
                    assert client is not None
                    return await client._request("/fapi/v1/userTrades", {"symbol": bot.symbol,
                                                 "startTime": lower, "endTime": upper, "limit": 1000},
                                                 signed=True, max_retries=0)

                async def income(lower: int, upper: int) -> Any:
                    # Current endpoint docs do not specify page. Bound time
                    # intervals explicitly instead of relying on ignored pages.
                    assert client is not None
                    return await client._request("/fapi/v1/income", {"symbol": bot.symbol,
                                                 "startTime": lower, "endTime": upper, "limit": 1000},
                                                 signed=True, max_retries=0)

                item: dict[str, Any] = {"bot": bot.model_dump(mode="json"), "tick_source": source_ref(tick_path).model_dump(),
                                        "history_start_at": start.isoformat(), "history_end_at": end.isoformat()}
                abort = False
                for endpoint, fetch in (("/fapi/v1/userTrades", trades), ("/fapi/v1/income", income)):
                    result = await collect_history(fetch=fetch, endpoint=endpoint, bot=bot, account_scope=scope,
                                                   start=start, end=end, live_root=live, now=now,
                                                   max_requests=max_requests)
                    item[endpoint.rsplit("/", 1)[1]] = {key: result[key] for key in
                        ("requests", "manifest", "gaps", "rejected", "retrieval_exhausted")}
                    item[endpoint.rsplit("/", 1)[1]]["rows"] = len(result["rows"])
                    if endpoint.endswith("userTrades") and not result["rejected"]:
                        item["attribution"] = admit_fills(result, bot=bot, account_scope=scope, ownership_paths=owners,
                                                         live_root=live, now=now())
                    if any(g["reason"] == "transport_failure" for g in result["gaps"]):
                        abort = True
                        break
                report["acquisitions"].append(item)
                if abort:
                    report["errors"].append({"stage": "fetch", "error_type": "TransportFailureRemainingBotsNotFetched"})
                    break
    except Exception as exc:
        # No exception text/traceback in persisted output: transports may carry
        # signed URLs or private payloads. Original successful captures persist.
        report["errors"].append({"stage": "acquisition", "error_type": type(exc).__name__})
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception as exc:
                report["errors"].append({"stage": "client_close", "error_type": type(exc).__name__})
    report["finished_at"] = now().isoformat()
    report["status"] = "partial_or_blocked" if report["errors"] else "captured_with_eligibility_gates_preserved"
    report["remaining_boundaries"] = ["native_order_ownership_requires_original_mapping",
        "credential_scope_is_not_account_uid", "income_funding_and_flows_not_automatically_bot_attributed",
        "no_verified_equity_or_peak_equity_from_account_totals", "snapshot_paths_do_not_prove_nonhits",
        "full_intervention_and_accounting_coverage_unproven", "liquidation_limit_unspecified"]
    atomic_create_json(output / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registration", type=Path, action="append", default=[])
    parser.add_argument("--all-registered", action="store_true")
    parser.add_argument("--capture-pnl", action="store_true")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--ownership-mapping", type=Path, action="append", default=[])
    parser.add_argument("--max-requests-per-endpoint", type=int, default=32)
    parser.add_argument("--start-at", help="Aware ISO-8601 API history start; intersected with each deployment. PnL path indexing is unchanged.")
    parser.add_argument("--end-at", help="Aware ISO-8601 API history end; defaults to the immutable run-start cutoff.")
    args = parser.parse_args(argv)
    if not 1 <= args.max_requests_per_endpoint <= 1024:
        parser.error("request budget must be between 1 and 1024")
    try:
        report = asyncio.run(capture(args))
    except (ValueError, OSError) as exc:
        logger.error("Evidence capture failed: %s", type(exc).__name__)
        return 2
    logger.warning("Evidence capture: %s; report=%s", report["status"], args.output_dir / "report.json")
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
