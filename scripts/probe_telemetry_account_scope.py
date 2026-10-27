"""Compare a captured dashboard UID to the configured API credential, read-only."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import logging
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neutralgrid.api.binance_client import BinanceAPIError, BinanceClient
from neutralgrid.live.decision.authenticated_evidence import HOST, credential_scope, integer_id, source_ref
from neutralgrid.live.decision.evidence_cohort import LIMA, read_json
from neutralgrid.live.decision.state_action_reward import atomic_create_json, utc

logger = logging.getLogger(__name__)
SPOT_HOST = "https://api.binance.com"


class SpotIdentityClient(BinanceClient):
    """Use the documented Spot clock even during inherited timestamp recovery."""

    _ENDPOINT_WEIGHTS = {**BinanceClient._ENDPOINT_WEIGHTS, "/api/v3/account": 20, "/api/v3/time": 1}

    async def sync_time(self) -> int:
        clock = await self._request("/api/v3/time", signed=False, max_retries=0)
        server_ms = int(integer_id(clock["serverTime"]))
        self._time_offset_ms = server_ms - int(datetime.now(timezone.utc).timestamp() * 1000)
        return self._time_offset_ms


def ui_identity(source: Path, live_root: Path, at: datetime) -> tuple[str, dict[str, str]]:
    """Validate a visible UI excerpt; it is not an exchange-signed assertion."""
    path = source.resolve()
    if live_root.resolve() not in path.parents:
        raise ValueError("UI identity must be an original stored under Live")
    body, digest = read_json(path)
    if (body.get("schema_version") != "neutralgrid_ui_account_identity_capture_v1"
            or body.get("url") != "https://www.binance.com/en/my/dashboard"
            or body.get("scope") != "account_identity_only_not_bot_ownership"):
        raise ValueError("unsupported UI identity source")
    stamp = body.get("captured_at_utc")
    if not isinstance(stamp, str) or not stamp.strip():
        raise ValueError("UI capture timestamp missing")
    captured = utc(datetime.fromisoformat(stamp.replace("Z", "+00:00")))
    if captured > utc(at):
        raise ValueError("UI capture is future dated")
    text = body.get("uid_text")
    if not isinstance(text, str):
        raise ValueError("UI UID missing")
    match = re.fullmatch(r'(?:UID\s+|\- generic: UID\s+\- generic: ")([1-9][0-9]*)(?:")?', text)
    if match is None:
        raise ValueError("UI UID is missing or ambiguous")
    ref = source_ref(path).model_dump()
    if ref["sha256"] != digest:
        raise ValueError("UI identity changed during verification")
    return integer_id(match.group(1)), ref


async def probe(*, source: Path, symbol: str, live_root: Path, report_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    if not re.fullmatch(r"[A-Z0-9]{2,30}USDT", symbol):
        raise ValueError("invalid storage context symbol")
    uid, ref = ui_identity(source, live_root, started)
    if report_dir.exists():
        raise ValueError("report directory must be fresh")
    report_dir.mkdir(parents=True)
    result: dict[str, Any] = {
        "schema_version": "neutralgrid_account_scope_probe_v1", "started_at": started.isoformat(),
        "ui_source": ref, "identity_status": "unverified", "errors": [],
        "training_eligible": False, "promotion_eligible": False, "influence_enabled": False,
        "bot_ownership_proven": False, "trading_bots_rest_visibility_proven": False,
        "scope": "account_identity_only_storage_symbol_is_not_attribution",
    }
    futures: BinanceClient | None = None
    spot: BinanceClient | None = None
    try:
        futures = BinanceClient()
        if not futures.is_authenticated or futures.base_url.rstrip("/") != HOST:
            raise ValueError("configured production credential required")
        result["credential_scope"] = credential_scope(futures.api_key)
        spot = SpotIdentityClient(api_key=futures.api_key, api_secret=futures.api_secret, base_url=SPOT_HOST)
        # Futures client's sync_time route is Futures-specific. Use the documented
        # Spot clock here, with no mutation of the shared scanner configuration.
        await spot.sync_time()
        requested = datetime.now(timezone.utc)
        response = await spot._request("/api/v3/account", signed=True, max_retries=0)
        received = datetime.now(timezone.utc)
        if not isinstance(response, dict) or "uid" not in response:
            raise ValueError("API UID missing")
        api_uid = integer_id(response["uid"])
        if api_uid == "0":
            raise ValueError("invalid API UID")
        evidence = {
            "schema_version": "neutralgrid_authenticated_account_identity_v1",
            "host": SPOT_HOST, "endpoint": "/api/v3/account", "method": "GET",
            "requested_at_utc": requested.isoformat(), "received_at_utc": received.isoformat(),
            "credential_scope": result["credential_scope"], "uid": api_uid,
            "capture_kind": "allowlisted_uid_projection_of_authenticated_json_response",
            "authentication_basis": "local_authenticated_capture_not_server_signed",
            "training_eligible": False, "bot_ownership_proven": False,
        }
        destination = (live_root.resolve() / received.astimezone(LIMA).date().isoformat() / symbol /
                       "account_scope_probe" / received.strftime("%Y%m%dT%H%M%S%fZ") / "api_identity.json")
        atomic_create_json(destination, evidence)
        result["api_source"] = source_ref(destination).model_dump()
        result["identity_status"] = "match" if api_uid == uid else "mismatch"
    except Exception as exc:
        error: dict[str, Any] = {"error_type": type(exc).__name__}
        if isinstance(exc, BinanceAPIError):
            error.update(http_status=exc.status, exchange_error_code=exc.code)
        result["errors"].append(error)
    finally:
        for client in (spot, futures):
            if client is not None:
                try:
                    await client.close()
                except Exception as exc:
                    result["errors"].append({"stage": "close", "error_type": type(exc).__name__})
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_create_json(report_dir / "report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ui-source", type=Path, required=True)
    parser.add_argument("--storage-symbol", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if ROOT / "reports" not in args.output_dir.resolve().parents:
        parser.error("output must be under reports")
    # HTTP debug logs can contain signed URLs. Only safe result fields are logged.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    try:
        result = asyncio.run(probe(source=args.ui_source, symbol=args.storage_symbol,
                                  live_root=ROOT / "Live", report_dir=args.output_dir.resolve()))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        logger.error("Account identity probe blocked: %s", type(exc).__name__)
        return 2
    logger.warning("Account identity: %s; errors=%d", result["identity_status"], len(result["errors"]))
    return 0 if result["identity_status"] == "match" and not result["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
