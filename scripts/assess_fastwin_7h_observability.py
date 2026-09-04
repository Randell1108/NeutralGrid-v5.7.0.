#!/usr/bin/env python3
"""Run a bounded, isolated seven-hour target-observability assessment.

The active fast-winner target is ``time_to_target_hours <= 7``.  Historical
candidate backtests may instead have a six-hour observation window.  This tool
selects only source rows that were fully observed for exactly six hours without
reaching the target, replays the same configuration at both 360 and 420 bars,
and reports whether a previously negative target label flips in the additional
hour.  It never writes a training pool or model artifact.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Awaitable, Callable, cast

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.backtest.candidate_pipeline import (
    fetch_historical_klines,
    run_single_backtest,
)


logger = logging.getLogger(__name__)

BASELINE_HOURS = 6
EXTENDED_HOURS = 7
BASELINE_BARS = BASELINE_HOURS * 60
EXTENDED_BARS = EXTENDED_HOURS * 60
_DEFAULT_LEVERAGE = 10
_FLOAT_RTOL = 1e-8
_FLOAT_ATOL = 1e-8


@dataclass(frozen=True)
class AssessmentConfig:
    sample_size: int
    leverage: int


def _strict_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _parse_start(value: Any) -> datetime | None:
    parsed = pd.to_datetime(pd.Series([value]), utc=True, errors="coerce", format="mixed").iloc[0]
    if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed):
        return parsed.to_pydatetime()
    return None


def _stable_rank(candidate_id: str) -> str:
    return hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()


def select_isolated_cohort(source: pd.DataFrame, *, sample_size: int) -> pd.DataFrame:
    """Return a deterministic cohort whose sixth-hour labels can change at hour 7."""
    required = {
        "candidate_id",
        "symbol",
        "backtest_start_ts_utc",
        "duration_hours",
        "target_reached",
        "time_to_target_hours",
        "net_pnl_pct",
        "grid_lower",
        "grid_upper",
        "num_grids",
        "capital_base",
        "capital_fraction",
        "funding_mode",
        "mode",
        "realism_profile",
        "is_authoritative",
        "max_holding_bars",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"source results lack required replay columns: {missing}")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    candidate_ids = cast(pd.Series, source["candidate_id"]).fillna("").astype(str).str.strip()
    start_valid = cast(pd.Series, source["backtest_start_ts_utc"]).map(_parse_start).notna()
    duration = cast(pd.Series, pd.to_numeric(cast(pd.Series, source["duration_hours"]), errors="coerce"))
    capital_fraction = cast(pd.Series, pd.to_numeric(cast(pd.Series, source["capital_fraction"]), errors="coerce"))
    num_grids = cast(pd.Series, pd.to_numeric(cast(pd.Series, source["num_grids"]), errors="coerce"))
    lower = cast(pd.Series, pd.to_numeric(cast(pd.Series, source["grid_lower"]), errors="coerce"))
    upper = cast(pd.Series, pd.to_numeric(cast(pd.Series, source["grid_upper"]), errors="coerce"))
    target_reached = cast(pd.Series, source["target_reached"]).map(_strict_bool)
    authoritative = cast(pd.Series, source["is_authoritative"]).map(_strict_bool)
    target_time = cast(pd.Series, pd.to_numeric(cast(pd.Series, source["time_to_target_hours"]), errors="coerce"))
    max_holding = cast(pd.Series, pd.to_numeric(cast(pd.Series, source["max_holding_bars"]), errors="coerce"))

    mask = (
        candidate_ids.ne("")
        & start_valid
        & np.isclose(np.asarray(duration, dtype=float), BASELINE_HOURS, equal_nan=False)
        & target_reached.eq(False)
        & target_time.isna()
        & authoritative.eq(True)
        & cast(pd.Series, source["mode"]).fillna("").astype(str).str.lower().eq("geometric")
        & cast(pd.Series, source["realism_profile"]).fillna("").astype(str).str.lower().eq("legacy")
        & max_holding.eq(BASELINE_BARS)
        & lower.gt(0)
        & upper.gt(lower)
        & num_grids.gt(1)
        & capital_fraction.gt(0)
        & capital_fraction.le(1)
    )
    eligible = cast(pd.DataFrame, source.loc[mask].copy())
    if eligible.empty:
        raise ValueError("no source rows satisfy the isolated observability cohort contract")
    eligible["_stable_rank"] = candidate_ids.loc[mask].map(_stable_rank)
    cohort = cast(
        pd.DataFrame,
        eligible.sort_values(["_stable_rank", "candidate_id"], kind="stable")
        .head(sample_size)
        .drop(columns=["_stable_rank"])
        .reset_index(drop=True),
    )
    if bool(cast(pd.Series, cohort["candidate_id"]).duplicated().any()):
        raise ValueError("isolated cohort has duplicate candidate_id values")
    return cohort


async def assess_cohort(
    cohort: pd.DataFrame,
    *,
    config: AssessmentConfig,
    fetcher: Callable[[Any, str, datetime, int, str], Awaitable[pd.DataFrame]] = fetch_historical_klines,
    runner: Callable[..., dict[str, Any]] = run_single_backtest,
    client_factory: Callable[[], Any] = BinanceClient,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay each row independently, retaining failures as explicit evidence."""
    records: list[dict[str, Any]] = []
    client = client_factory()
    try:
        for _, row in cohort.iterrows():
            candidate_id = str(row["candidate_id"])
            record: dict[str, Any] = {
                "candidate_id": candidate_id,
                "symbol": str(row["symbol"]),
                "source_target_reached": False,
                "source_time_to_target_hours": None,
                "status": "pending",
                "baseline_reproduced": False,
                "seven_hour_target_reached": None,
                "seven_hour_time_to_target_hours": None,
                "label_flip_at_hour_7": False,
                "error": "",
            }
            try:
                start = _parse_start(row["backtest_start_ts_utc"])
                capital = _finite_float(row["capital_base"])
                if start is None or capital is None or capital <= 0:
                    raise ValueError("invalid backtest_start_ts_utc or capital_base")
                price_source = str(row.get("price_source", "last") or "last").lower()
                if price_source != "last":
                    raise ValueError(f"unsupported legacy price_source={price_source!r}")
                klines = await fetcher(client, str(row["symbol"]), start, EXTENDED_HOURS, price_source)
                if len(klines) < EXTENDED_BARS:
                    record["status"] = "insufficient_seven_hour_bars"
                    record["bars_fetched"] = int(len(klines))
                    records.append(record)
                    continue

                candidate_payload = row.to_dict()
                baseline = runner(
                    candidate_row=candidate_payload,
                    klines_df=klines.iloc[:BASELINE_BARS].copy(),
                    capital=capital,
                    leverage=config.leverage,
                    max_holding_bars=BASELINE_BARS,
                    funding_mode=str(row["funding_mode"]),
                    realism_profile="legacy",
                )
                source_pnl = _finite_float(row["net_pnl_pct"])
                baseline_pnl = _finite_float(baseline.get("net_pnl_pct"))
                baseline_target = _strict_bool(baseline.get("target_reached"))
                baseline_time = _finite_float(baseline.get("time_to_target_hours"))
                pnl_matches = (
                    source_pnl is not None
                    and baseline_pnl is not None
                    and bool(np.isclose(source_pnl, baseline_pnl, rtol=_FLOAT_RTOL, atol=_FLOAT_ATOL))
                )
                baseline_matches = baseline_target is False and baseline_time is None and pnl_matches
                record.update(
                    {
                        "bars_fetched": int(len(klines)),
                        "baseline_reproduced": baseline_matches,
                        "baseline_net_pnl_pct": baseline_pnl,
                        "source_net_pnl_pct": source_pnl,
                    }
                )
                if not baseline_matches:
                    record["status"] = "baseline_not_reproducible"
                    records.append(record)
                    continue

                extended = runner(
                    candidate_row=candidate_payload,
                    klines_df=klines.iloc[:EXTENDED_BARS].copy(),
                    capital=capital,
                    leverage=config.leverage,
                    max_holding_bars=EXTENDED_BARS,
                    funding_mode=str(row["funding_mode"]),
                    realism_profile="legacy",
                )
                extended_target = _strict_bool(extended.get("target_reached"))
                extended_time = _finite_float(extended.get("time_to_target_hours"))
                flip = extended_target is True and extended_time is not None and extended_time <= EXTENDED_HOURS
                record.update(
                    {
                        "status": "comparable",
                        "seven_hour_target_reached": extended_target,
                        "seven_hour_time_to_target_hours": extended_time,
                        "seven_hour_net_pnl_pct": _finite_float(extended.get("net_pnl_pct")),
                        "label_flip_at_hour_7": flip,
                    }
                )
            except Exception as exc:  # Keep one bad symbol from aborting the audit.
                logger.warning("7h observability replay failed for %s: %s", candidate_id, exc)
                record["status"] = "replay_error"
                record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
    finally:
        await client.close()

    detail = pd.DataFrame(records)
    comparable = cast(pd.Series, detail["status"]).eq("comparable")
    flips = cast(pd.Series, detail["label_flip_at_hour_7"]).fillna(False).astype(bool)
    summary = {
        "target_contract": "fast_winner_time_to_3pct_le_7h",
        "baseline_hours": BASELINE_HOURS,
        "extended_hours": EXTENDED_HOURS,
        "baseline_bars": BASELINE_BARS,
        "extended_bars": EXTENDED_BARS,
        "requested_rows": int(len(cohort)),
        "comparable_rows": int(comparable.sum()),
        "noncomparable_rows": int((~comparable).sum()),
        "label_flips": int((flips & comparable).sum()),
        "status_counts": {
            str(key): int(value)
            for key, value in cast(pd.Series, detail["status"]).value_counts(dropna=False).items()
        },
        "calibration_input_impact": (
            "affected_in_isolated_cohort" if bool((flips & comparable).any())
            else "not_detected_in_isolated_cohort"
        ),
    }
    return detail, summary


async def _main_async(args: argparse.Namespace) -> int:
    source = pd.read_csv(args.source, low_memory=False)
    cohort = select_isolated_cohort(source, sample_size=args.sample_size)
    detail, summary = await assess_cohort(
        cohort,
        config=AssessmentConfig(sample_size=args.sample_size, leverage=args.leverage),
    )
    print(json.dumps({"summary": summary, "rows": detail.to_dict(orient="records")}, indent=2, default=str))
    return 0 if summary["comparable_rows"] > 0 else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument(
        "--leverage",
        type=int,
        default=_DEFAULT_LEVERAGE,
        help="Source-run leverage to reproduce; defaults to the recorded CLI default.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if args.leverage <= 0:
        raise ValueError("leverage must be positive")
    raise SystemExit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
