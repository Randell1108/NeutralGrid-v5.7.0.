"""Train and frozen-holdout evaluate the shadow realized-volatility model.

The command is deliberately separate from active model lineage.  It accepts an
explicit active roster or symbol list, audits the canonical one-minute price
store, enforces the approved history and origin floors, and writes only to a
run-scoped ``outputs/audits/live_volatility`` directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast
from urllib.parse import urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neutralgrid.live.decision.volatility import (
    VolatilityError,
    audit_single_symbol_readiness,
    build_rv_examples,
    load_price_store_frame,
    load_volatility_contract,
    validate_price_frame,
)
from neutralgrid.live.decision.volatility_forecast import (
    train_evaluate_shadow_volatility,
)


UTC = timezone.utc
TRUSTED_BINANCE_ROOT_HOSTNAMES = frozenset({"binance.com", "binance.bh"})
logger = logging.getLogger(__name__)


class VolatilityTrainingError(RuntimeError):
    """The stored evidence cannot enter shadow volatility training."""


def _load_roster(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VolatilityTrainingError(f"cannot read cycle manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise VolatilityTrainingError("cycle manifest must have status=complete")
    if payload.get("schema_version") != "neutralgrid_private_telemetry_cycle_v2":
        raise VolatilityTrainingError("unsupported cycle manifest schema")
    if payload.get("source") != "chrome_plugin":
        raise VolatilityTrainingError("cycle manifest source is not chrome_plugin")
    page_identity = payload.get("page_identity")
    if not isinstance(page_identity, str) or not page_identity.strip():
        raise VolatilityTrainingError("cycle manifest page identity is missing")
    source_url = payload.get("source_url")
    if not isinstance(source_url, str) or not source_url.strip():
        raise VolatilityTrainingError("cycle manifest source URL is missing")
    parsed_source = urlparse(source_url)
    hostname = (parsed_source.hostname or "").lower()
    trusted_hostname = any(
        hostname == root_hostname or hostname.endswith(f".{root_hostname}")
        for root_hostname in TRUSTED_BINANCE_ROOT_HOSTNAMES
    )
    if parsed_source.scheme != "https" or not trusted_hostname:
        raise VolatilityTrainingError("cycle manifest source URL is not trusted Binance HTTPS")
    files = payload.get("files")
    if not isinstance(files, list):
        raise VolatilityTrainingError("cycle manifest files must be a list")
    roster: dict[str, str] = {}
    for item in files:
        if not isinstance(item, dict):
            raise VolatilityTrainingError("cycle manifest file entry must be an object")
        symbol = str(item.get("symbol", "")).strip().upper()
        strategy_id = str(item.get("strategy_id", "")).strip()
        if not symbol or not strategy_id:
            raise VolatilityTrainingError("cycle manifest identity is incomplete")
        if symbol in roster:
            raise VolatilityTrainingError(f"duplicate cycle symbol: {symbol}")
        roster[symbol] = strategy_id
    if payload.get("active_bot_count") != len(roster) or not roster:
        raise VolatilityTrainingError("cycle active count does not match identities")
    if payload.get("working_row_count") != len(roster):
        raise VolatilityTrainingError("cycle Working-row count does not match identities")
    return roster


def _load_backfill_scopes(
    path: Path,
    *,
    symbols: Sequence[str],
    contract_sha256: str,
) -> dict[str, date]:
    """Bind training to checksum-audited acquisition windows, not old store rows."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VolatilityTrainingError(
            f"cannot read backfill manifest {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "neutralgrid_volatility_history_backfill_v1"
    ):
        raise VolatilityTrainingError("unsupported volatility backfill manifest")
    if payload.get("contract_sha256") != contract_sha256:
        raise VolatilityTrainingError("backfill manifest contract hash mismatch")
    results = payload.get("results")
    if not isinstance(results, list):
        raise VolatilityTrainingError("backfill manifest results must be a list")
    scopes: dict[str, date] = {}
    for result in results:
        if not isinstance(result, Mapping):
            raise VolatilityTrainingError("backfill result must be an object")
        symbol = str(result.get("symbol", "")).strip().upper()
        attempts = result.get("attempts")
        if not symbol or not isinstance(attempts, list) or not attempts:
            raise VolatilityTrainingError("backfill result lacks an acquisition attempt")
        latest = attempts[-1]
        if not isinstance(latest, Mapping):
            raise VolatilityTrainingError("backfill attempt must be an object")
        raw_start = latest.get("start_date")
        if not isinstance(raw_start, str):
            raise VolatilityTrainingError("backfill attempt lacks start_date")
        try:
            scopes[symbol] = date.fromisoformat(raw_start)
        except ValueError as exc:
            raise VolatilityTrainingError(
                f"backfill start_date is invalid for {symbol}"
            ) from exc
    expected = {str(symbol).upper() for symbol in symbols}
    if set(scopes) != expected:
        raise VolatilityTrainingError(
            "backfill manifest symbols do not exactly match the training roster"
        )
    return scopes


def _calendar_day_count(normalized: pd.DataFrame) -> int:
    if normalized.empty:
        return 0
    times = cast(pd.Series, normalized["open_time"])
    return int(times.dt.floor("D").nunique())


def assemble_training_examples(
    *,
    symbols: Sequence[str],
    price_store: Path,
    contract_path: Path,
    scope_start_dates: Mapping[str, date] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load, audit, and combine only symbols satisfying the frozen evidence floors."""

    contract = load_volatility_contract(contract_path)
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for raw_symbol in sorted(set(symbols)):
        symbol = raw_symbol.upper()
        raw_mark = load_price_store_frame(
            price_store,
            symbol=symbol,
            series_kind=contract.primary_series,
        )
        scope_start = (
            None if scope_start_dates is None else scope_start_dates.get(symbol)
        )
        if scope_start_dates is not None and scope_start is None:
            raise VolatilityTrainingError(f"no audited acquisition scope for {symbol}")
        if scope_start is not None and "open_time_ms" in raw_mark.columns:
            scope_start_ms = int(
                datetime.combine(
                    scope_start,
                    datetime.min.time(),
                    tzinfo=UTC,
                ).timestamp()
                * 1000
            )
            open_times = cast(
                pd.Series,
                pd.to_numeric(
                    cast(pd.Series, raw_mark["open_time_ms"]),
                    errors="coerce",
                ),
            )
            raw_mark = (
                raw_mark.loc[open_times >= scope_start_ms]
                .copy()
                .reset_index(drop=True)
            )
        normalized, price_audit = validate_price_frame(
            raw_mark,
            symbol=symbol,
            series_kind=contract.primary_series,
        )
        calendar_days = _calendar_day_count(normalized)
        examples, example_audit = build_rv_examples(
            raw_mark,
            symbol=symbol,
            contract=contract,
        )
        readiness = audit_single_symbol_readiness(
            examples,
            symbol=symbol,
            contract=contract,
        )
        history_gate = calendar_days >= contract.minimum_history_days
        ready = history_gate and readiness.get("status") == "ready"
        audit = {
            "symbol": symbol,
            "acquisition_scope_start_date": (
                None if scope_start is None else scope_start.isoformat()
            ),
            "calendar_days": calendar_days,
            "minimum_calendar_days": contract.minimum_history_days,
            "history_gate": history_gate,
            "price_audit": price_audit.__dict__,
            "example_audit": example_audit,
            "origin_readiness": readiness,
            "training_admitted": ready,
        }
        audits.append(audit)
        if not ready:
            continue
        else:
            frames.append(examples)
    if not frames:
        raise VolatilityTrainingError(
            "no symbol satisfies the frozen history and origin evidence floors"
        )
    return pd.concat(frames, ignore_index=True), audits


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append", default=[])
    parser.add_argument("--cycle-manifest", type=Path, default=None)
    parser.add_argument("--backfill-manifest", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "config" / "live_volatility_forecast_v1.json",
    )
    parser.add_argument("--price-store", type=Path, default=ROOT / "data" / "price_store")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "audits" / "live_volatility",
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    if not args.symbol and args.cycle_manifest is None:
        parser.error("provide --symbol or --cycle-manifest")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        explicit = {str(value).strip().upper() for value in args.symbol if str(value).strip()}
        roster: dict[str, str] = {}
        if args.cycle_manifest is not None:
            roster = _load_roster(args.cycle_manifest.resolve())
            if explicit and explicit != set(roster):
                raise VolatilityTrainingError(
                    "explicit symbols do not exactly match the supplied active roster"
                )
        symbols = sorted(explicit or set(roster))
        contract = load_volatility_contract(args.contract.resolve())
        scopes = _load_backfill_scopes(
            args.backfill_manifest.resolve(),
            symbols=symbols,
            contract_sha256=contract.contract_sha256,
        )
        examples, audits = assemble_training_examples(
            symbols=symbols,
            price_store=args.price_store.resolve(),
            contract_path=args.contract.resolve(),
            scope_start_dates=scopes,
        )
        run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise VolatilityTrainingError("--run-id must be a single safe path component")
        output_dir = args.output_root.resolve() / run_id
        if output_dir.exists() and any(output_dir.iterdir()):
            raise VolatilityTrainingError(f"run directory is not empty: {output_dir}")
        result = train_evaluate_shadow_volatility(
            examples,
            contract=contract,
            output_dir=output_dir,
            source_audits=audits,
        )
        # The model function has already committed the frozen artifacts.  This
        # stdout record is intentionally not used as an artifact of authority.
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result.get("forecast_eligible") is True else 2
    except (VolatilityError, VolatilityTrainingError, OSError, ValueError) as exc:
        logger.error("shadow volatility training blocked: %s", exc)
        return 2


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
