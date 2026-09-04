#!/usr/bin/env python3
"""Compare candidate-ID-matched configurations with observed bot lifecycles.

This is an audit-only runner. It uses the candidate's grid geometry and the
observed bot's start window, duration, margin, and leverage so the simulated
result is comparable with the recorded terminal PnL. It does not create
training rows or alter any deployment artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.backtest.candidate_pipeline import (
    fetch_historical_klines,
    run_single_backtest,
)
from neutralgrid.backtest.realism_governance import (
    CANDIDATE_TIME_GEOMETRIC_PROFILE,
    validate_realism_output_path,
)


logger = logging.getLogger("matched_candidate_comparison")


def _finite_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    number = _finite_float_or_none(value)
    return None if number is None else int(number)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay candidate geometry over observed bot lifecycles.",
    )
    parser.add_argument(
        "--expired-bots-path",
        default="data/new_expired_bots.xlsx",
        help="Canonical workbook containing observed terminal bot records.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing deployment-ready CSV artifacts.",
    )
    parser.add_argument(
        "--intake-manifest",
        default="data/manual_input/2026-07-29/manifest.csv",
        help="Manifest that bounds the observed-bot cohort for this audit.",
    )
    parser.add_argument(
        "--output",
        default="outputs/audits/matched_candidate_live_window_20260729.csv",
        help="Isolated CSV path for the shadow comparison output.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between Binance requests in seconds.",
    )
    args = parser.parse_args()
    try:
        validate_realism_output_path(CANDIDATE_TIME_GEOMETRIC_PROFILE, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def load_exact_candidate_rows(results_dir: Path, candidate_ids: set[str]) -> pd.DataFrame:
    """Load each supplied candidate ID once and retain artifact provenance."""
    frames: list[pd.DataFrame] = []
    for path in sorted(results_dir.glob("deployment_ready_*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            logger.warning("Skipping unreadable candidate artifact %s: %s", path.name, exc)
            continue
        if "candidate_id" not in frame.columns:
            continue
        matched = frame.loc[
            cast(pd.Series, frame["candidate_id"]).isin(sorted(candidate_ids))
        ].copy()
        if not matched.empty:
            matched["candidate_source_file"] = path.name
            frames.append(matched)

    if not frames:
        return pd.DataFrame()
    candidates = pd.concat(frames, ignore_index=True)
    return candidates.drop_duplicates(subset=["candidate_id"], keep="last")


def load_observed_rows(path: Path, intake_manifest: Path) -> pd.DataFrame:
    """Load completed bot records eligible for an exact candidate comparison."""
    rows = pd.read_excel(path)
    required = {
        "strategy_id",
        "symbol",
        "candidate_id",
        "start_time_utc",
        "duration_hours",
        "invested_margin_usdt",
        "leverage",
        "total_profit_usdt",
        "pnl_pct",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Observed bot workbook is missing required columns: {missing}")

    eligible = rows.loc[
        rows["candidate_id"].notna()
        & rows["start_time_utc"].notna()
        & rows["duration_hours"].notna()
        & rows["invested_margin_usdt"].notna()
        & rows["leverage"].notna()
    ].copy()
    eligible["candidate_id"] = eligible["candidate_id"].astype(str)
    manifest = pd.read_csv(intake_manifest)
    if "strategy_id" not in manifest.columns:
        raise ValueError("Intake manifest is missing strategy_id")
    strategy_ids = cast(
        pd.Series,
        pd.to_numeric(cast(pd.Series, manifest["strategy_id"]), errors="raise"),
    )
    eligible_strategy_ids = cast(
        pd.Series,
        pd.to_numeric(
            cast(pd.Series, eligible["strategy_id"]),
            errors="coerce",
        ),
    )
    eligible = eligible.loc[
        eligible_strategy_ids.isin(strategy_ids)
    ].copy()
    return eligible.rename(
        columns={
            "symbol": "observed_symbol",
            "start_time_utc": "observed_start_time_utc",
            "duration_hours": "observed_duration_hours",
            "invested_margin_usdt": "observed_margin_usdt",
            "leverage": "observed_leverage",
            "total_profit_usdt": "observed_total_profit_usdt",
            "pnl_pct": "observed_pnl_pct",
        }
    )


async def run_comparison(args: argparse.Namespace) -> pd.DataFrame:
    observed = load_observed_rows(
        Path(args.expired_bots_path), Path(args.intake_manifest)
    )
    candidates = load_exact_candidate_rows(
        Path(args.results_dir), set(observed["candidate_id"]),
    )
    if candidates.empty:
        raise ValueError("No observed candidate IDs were found in deployment-ready artifacts")

    cohort = observed.merge(
        candidates,
        on="candidate_id",
        how="inner",
        suffixes=("_observed", "_candidate"),
        validate="one_to_one",
    )
    if cohort.empty:
        raise ValueError("No exact candidate IDs matched the observed bot records")

    logger.info("Replaying %d exact candidate-to-bot matches", len(cohort))
    client = BinanceClient()
    output_rows: list[dict[str, Any]] = []
    try:
        for position, (_, row) in enumerate(cohort.iterrows(), start=1):
            candidate_id = str(row.get("candidate_id", ""))
            symbol = str(row.get("symbol", ""))
            try:
                start_timestamp = pd.to_datetime(
                    row["observed_start_time_utc"], utc=True, errors="coerce"
                )
                if not isinstance(start_timestamp, pd.Timestamp) or bool(
                    pd.isna(start_timestamp)
                ):
                    raise ValueError(
                        "observed_start_time_utc is not a valid scalar timestamp for "
                        f"candidate_id={candidate_id}"
                    )
                start_time = start_timestamp.to_pydatetime()
                observed_duration_hours = float(row["observed_duration_hours"])
                observed_margin_usdt = float(row["observed_margin_usdt"])
                observed_leverage = int(row["observed_leverage"])
                observed_total_profit_usdt = float(row["observed_total_profit_usdt"])
                observed_pnl_pct = float(row["observed_pnl_pct"])
                duration_bars = max(1, int(round(observed_duration_hours * 60)))
                fetch_hours = max(1, int((duration_bars + 59) // 60))
                logger.info(
                    "[%d/%d] %s %s: %.0f-minute observed lifecycle",
                    position,
                    len(cohort),
                    symbol,
                    candidate_id,
                    duration_bars,
                )
                klines = await fetch_historical_klines(
                    client,
                    symbol,
                    start_time,
                    hours=fetch_hours,
                )
                if len(klines) < duration_bars:
                    raise ValueError(
                        f"only {len(klines)} historical bars returned for {duration_bars}-bar window"
                    )

                candidate_payload = dict(row)
                # Match the observed allocation; candidate capital_fraction would
                # otherwise model a smaller notional than the real bot used.
                candidate_payload["capital_fraction"] = 1.0
                result = run_single_backtest(
                    candidate_row=candidate_payload,
                    klines_df=klines,
                    capital=observed_margin_usdt,
                    leverage=observed_leverage,
                    max_holding_bars=duration_bars,
                    duration_bars=duration_bars,
                    realism_profile=CANDIDATE_TIME_GEOMETRIC_PROFILE,
                )
                result.update(
                    {
                        "strategy_id": row["strategy_id"],
                        "candidate_id": candidate_id,
                        "symbol": symbol,
                        "candidate_source_file": row["candidate_source_file"],
                        "observed_start_time_utc": start_time.isoformat(),
                        "observed_duration_hours": observed_duration_hours,
                        "observed_duration_bars": duration_bars,
                        "observed_margin_usdt": observed_margin_usdt,
                        "observed_leverage": observed_leverage,
                        "observed_total_profit_usdt": observed_total_profit_usdt,
                        "observed_pnl_pct": observed_pnl_pct,
                        "comparison_anchor": "observed_bot_start_and_duration",
                        "sizing_anchor": "observed_margin_and_leverage",
                        "candidate_match_type": "configuration_matched_candidate_id",
                        "candidate_capital_fraction_overridden": True,
                        "klines_fetched": len(klines),
                    }
                )
                result["pnl_pct_difference"] = (
                    float(result["net_pnl_pct"]) - observed_pnl_pct
                )
                result["pnl_direction_matches"] = bool(
                    (float(result["net_pnl_pct"]) >= 0)
                    == (observed_pnl_pct >= 0)
                )
                output_rows.append(result)
            except Exception as exc:
                logger.warning("Skipping %s: %s", candidate_id, exc)
                output_rows.append(
                    {
                        "strategy_id": row.get("strategy_id"),
                        "candidate_id": candidate_id,
                        "symbol": symbol,
                        "candidate_source_file": row.get("candidate_source_file"),
                        "observed_start_time_utc": str(
                            row.get("observed_start_time_utc", "")
                        ),
                        "observed_duration_hours": _finite_float_or_none(
                            row.get("observed_duration_hours")
                        ),
                        "observed_margin_usdt": _finite_float_or_none(
                            row.get("observed_margin_usdt")
                        ),
                        "observed_leverage": _int_or_none(
                            row.get("observed_leverage")
                        ),
                        "observed_total_profit_usdt": _finite_float_or_none(
                            row.get("observed_total_profit_usdt")
                        ),
                        "observed_pnl_pct": _finite_float_or_none(
                            row.get("observed_pnl_pct")
                        ),
                        "comparison_anchor": "observed_bot_start_and_duration",
                        "sizing_anchor": "observed_margin_and_leverage",
                        "candidate_match_type": "configuration_matched_candidate_id",
                        "backtest_status": "skipped",
                        "backtest_error_type": type(exc).__name__,
                        "backtest_error": str(exc),
                    }
                )
            await asyncio.sleep(args.delay)
    finally:
        try:
            await client.close()
        except Exception as exc:
            logger.warning("Binance client close failed after comparison: %s", exc)

    return pd.DataFrame(output_rows)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    output = await run_comparison(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    net_pnl = cast(
        pd.Series,
        output.get("net_pnl_pct", pd.Series(dtype=float)),
    )
    succeeded = int(net_pnl.notna().sum())
    logger.info("Wrote %d comparison rows (%d successful) to %s", len(output), succeeded, output_path)


if __name__ == "__main__":
    asyncio.run(main())
