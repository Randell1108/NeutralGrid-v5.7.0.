#!/usr/bin/env python3
"""
Backtest Undeployed Scanner Candidates.

Loads deployment_ready CSVs, filters to high-scoring undeployed candidates,
runs RealisticGridBacktester against historical 1m klines fetched from
Binance, and outputs backtest results + meta-labeler training rows.

Usage:
    python backtest_candidates.py --dry-run
    python backtest_candidates.py --max-candidates 3
    python backtest_candidates.py --min-score 75 --hours 6 --max-candidates 50
    python backtest_candidates.py --latest-only --max-candidates 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd

# Ensure local `src/` package is used when running this script directly
# (ERR-077: a stale editable install elsewhere must not shadow this tree).
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from neutralgrid.core.config import get_config
from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.backtest.candidate_pipeline import (
    _parse_scan_timestamp,
    attach_mark_close,
    convert_to_training_row,
    extract_exchange_filter_settings,
    fetch_historical_klines,
    filter_backtest_candidates,
    funding_rate_series_from_rows,
    load_all_scanner_csvs,
    resolve_backtest_start_timestamp,
    run_single_backtest,
)
from neutralgrid.backtest.realism_governance import validate_realism_output_path
from backtest.btk_seed_state import (
    CANDIDATE_TIME_PUBLIC_MARKET_PROFILE,
    LEGACY_REALISM_PROFILE,
    REALISM_PROFILES,
)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backtest_candidates")
# Suppress noisy backtest debug logs
logging.getLogger("backtest_realistic").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest undeployed scanner candidates against historical klines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--min-score", type=float, default=45.0,
        help="Minimum scanner score to include (default: 45)",
    )
    parser.add_argument(
        "--bypass-only",
        action="store_true",
        default=False,
        help=(
            "Backtest micro-osc bypass-only candidates "
            "(micro_osc_score >= 0.45, survival_prob >= 0.60, grid_is_valid != True)"
        ),
    )
    parser.add_argument(
        "--max-candidates", type=int, default=250,
        help="Max candidates to backtest per run (default: 250)",
    )
    parser.add_argument(
        "--fastwin-full-pool",
        action="store_true",
        help=(
            "Generate the fresh active-FASTWIN source: unbounded candidates, "
            "seven-hour/420-bar outcomes, legacy authority, and a run manifest. "
            "Requires an explicit fresh --output directory."
        ),
    )
    parser.add_argument(
        "--hours", type=int, default=6,
        help="Hours of kline data to fetch per candidate (default: 6)",
    )
    parser.add_argument(
        "--min-bars", type=int, default=360,
        help="Minimum kline bars required to run backtest (default: 360 = 6h)",
    )
    parser.add_argument(
        "--capital", type=float, default=400.0,
        help="Backtest capital per bot (default: 400)",
    )
    parser.add_argument(
        "--leverage", type=int, default=10,
        help="Backtest leverage (default: 10)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory containing deployment_ready CSVs (default: results/)",
    )
    parser.add_argument(
        "--linkage-path", type=str,
        default="data/linkage/deploy_linkage_log.csv",
        help="Path to deploy linkage CSV (DeployLinker or legacy format)",
    )
    parser.add_argument(
        "--output", type=str, default="data/backtest_candidates",
        help="Output directory for results (default: data/backtest_candidates/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show candidate selection counts without running backtests",
    )
    parser.add_argument(
        "--latest-only", action="store_true",
        help="Only use the latest scanner CSV file",
    )
    parser.add_argument(
        "--delay", type=float, default=0.2,
        help="Delay between API requests in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--match-live-duration", action="store_true",
        help="Truncate klines to match live bot duration from expired bots file",
    )
    parser.add_argument(
        "--realism-profile",
        choices=REALISM_PROFILES,
        default=LEGACY_REALISM_PROFILE,
        help=(
            "Explicit backtest realism profile. Default legacy preserves the "
            "canonical training baseline; non-legacy profiles require an "
            "isolated --output path and remain shadow-only."
        ),
    )
    parser.add_argument(
        "--expired-bots-path", type=str,
        default="data/new_expired_bots.xlsx",
        help="Path to expired bots Excel file (for --match-live-duration)",
    )
    parser.add_argument(
        "--authoritative-pool-path", type=str,
        default="data/new_expired_bots.xlsx",
        help="Path to authoritative candidate pool (used for valid_indicator)",
    )
    parser.add_argument(
        "--champion-candidate-pool-path", type=str,
        default="",
        help=(
            "Optional historical FASTWIN candidate-universe CSV. IDs meeting "
            "--min-score are fresh-backtested from current scanner rows; old "
            "outcomes and features are never reused."
        ),
    )
    parser.add_argument(
        "--pool-start-date",
        type=str,
        default="2026-06-08",
        help="Earliest candidate scan date to include (YYYY-MM-DD), default 2026-06-08",
    )
    parser.add_argument(
        "--pool-end-date",
        type=str,
        default="",
        help=(
            "Latest candidate scan date to include (YYYY-MM-DD). Empty means "
            "today (UTC)."
        ),
    )
    # Step 15 (Plan v6.0): Wire alignment audit
    parser.add_argument(
        "--audit", action="store_true",
        help="Run alignment audit after backtesting (compares backtest vs live outcomes)",
    )
    args = parser.parse_args()
    if args.fastwin_full_pool:
        if args.output == "data/backtest_candidates":
            parser.error(
                "--fastwin-full-pool requires a fresh explicit --output directory"
            )
        if args.realism_profile != LEGACY_REALISM_PROFILE:
            parser.error("--fastwin-full-pool requires --realism-profile legacy")
        args.max_candidates = None
        args.hours = 7
        args.min_bars = 420
    try:
        validate_realism_output_path(args.realism_profile, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _parse_window_date(value: str, label: str) -> datetime:
    if not value:
        raise ValueError(f"{label} is required")
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"{label} must use YYYY-MM-DD format: {value!r}") from exc


def _load_authoritative_candidate_ids(
    workbook_path: Path,
    window_start: datetime,
    window_end: datetime,
) -> set[str]:
    if not workbook_path.exists():
        logger.warning("Authoritative pool file not found: %s", workbook_path)
        return set()

    try:
        suffix = workbook_path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            pool_df = pd.read_csv(
                workbook_path,
                sep="\t" if suffix == ".tsv" else ",",
                low_memory=False,
            )
        elif suffix in {".xlsx", ".xls"}:
            with pd.ExcelFile(workbook_path) as workbook:
                if "General" in workbook.sheet_names:
                    pool_df = pd.read_excel(workbook, sheet_name="General")
                elif len(workbook.sheet_names) == 1:
                    pool_df = pd.read_excel(
                        workbook, sheet_name=workbook.sheet_names[0]
                    )
                else:
                    logger.warning(
                        "Authoritative pool has ambiguous workbook layout "
                        "without a General sheet: %s (%s)",
                        workbook_path,
                        workbook.sheet_names,
                    )
                    return set()
        else:
            logger.warning(
                "Unsupported authoritative pool extension %r: %s",
                suffix,
                workbook_path,
            )
            return set()
    except Exception as exc:
        logger.warning("Cannot read authoritative pool %s: %s", workbook_path, exc)
        return set()

    if "candidate_id" not in pool_df.columns:
        logger.warning("Authoritative pool missing candidate_id column: %s", workbook_path)
        return set()

    authoritative: set[str] = set()
    for row_id in pool_df["candidate_id"]:
        if row_id is None or pd.isna(row_id):
            continue
        candidate_id = str(row_id).strip()
        if not candidate_id or candidate_id.lower() == "nan":
            continue
        parsed = _parse_scan_timestamp(candidate_id)
        if parsed is None:
            continue
        if window_start <= parsed <= window_end:
            authoritative.add(candidate_id)

    logger.info(
        "Loaded %d candidate IDs from authoritative pool %s for window [%s, %s]",
        len(authoritative),
        workbook_path,
        window_start.date().isoformat(),
        window_end.date().isoformat(),
    )
    return authoritative


def _load_champion_candidate_ids(path: Path, min_score: float) -> set[str]:
    """Load historical champion IDs meeting the current score floor."""
    if not path.exists():
        raise FileNotFoundError(f"Champion candidate pool not found: {path}")
    header = pd.read_csv(path, nrows=0)
    required = {"candidate_id", "primary_pipeline_score"}
    missing = required.difference(header.columns)
    if missing:
        raise ValueError(
            f"Champion candidate pool lacks required columns: {sorted(missing)}"
        )

    selected: set[str] = set()
    total = 0
    for chunk in pd.read_csv(
        path,
        usecols=lambda column: column in {"candidate_id", "primary_pipeline_score"},
        chunksize=100_000,
        low_memory=False,
    ):
        total += len(chunk)
        score = cast(
            pd.Series,
            pd.to_numeric(
                cast(pd.Series, chunk["primary_pipeline_score"]), errors="coerce"
            ),
        )
        ids = cast(pd.Series, chunk["candidate_id"]).astype("string").str.strip()
        selected.update(
            ids.loc[score.ge(min_score) & ids.notna() & ids.ne("")]
            .astype(str)
            .tolist()
        )
    logger.info(
        "Loaded %d/%d champion candidate IDs meeting score >= %.1f from %s",
        len(selected), total, min_score, path,
    )
    return selected


async def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    # Backtest sampling window — explicit historical slice of candidate IDs.
    try:
        pool_start = _parse_window_date(args.pool_start_date, "pool-start-date")
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    if args.pool_end_date:
        try:
            pool_end = _parse_window_date(args.pool_end_date, "pool-end-date")
        except ValueError as exc:
            logger.error("%s", exc)
            sys.exit(1)
    else:
        pool_end = datetime.now(timezone.utc)
    # Make end date inclusive when passed as YYYY-MM-DD.
    pool_end_inclusive = (
        datetime.combine(pool_end.date(), datetime.max.time(), tzinfo=timezone.utc)
        if pool_end.time().hour == 0
        and pool_end.time().minute == 0
        and pool_end.time().second == 0
        and pool_end.time().microsecond == 0
        else pool_end
    )

    authoritative_pool_ids = _load_authoritative_candidate_ids(
        Path(args.authoritative_pool_path),
        pool_start,
        pool_end_inclusive,
    )
    champion_pool_ids: set[str] = set()
    if args.champion_candidate_pool_path:
        champion_pool_ids = _load_champion_candidate_ids(
            Path(args.champion_candidate_pool_path), args.min_score,
        )

    logger.info("=" * 70)
    logger.info("CANDIDATE BACKTESTING PIPELINE")
    logger.info("=" * 70)

    # ── Step 1: Load scanner CSVs ───────────────────────────────────────
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        sys.exit(1)

    logger.info("Loading scanner CSVs from %s ...", results_dir)

    if args.latest_only:
        csvs = sorted(
            results_dir.glob("deployment_ready_*.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        if not csvs:
            logger.error("No deployment_ready CSVs found")
            sys.exit(1)
        # Load only the latest file
        latest = csvs[-1]
        logger.info("Latest-only mode: using %s", latest.name)
        all_candidates = load_all_scanner_csvs(
            results_dir, filename_pattern=latest.name
        )
    else:
        all_candidates = load_all_scanner_csvs(results_dir)

    logger.info("Total candidates loaded: %d", len(all_candidates))

    # ── Step 2: Filter to backtestable candidates ───────────────────────
    linkage_path = Path(args.linkage_path)
    if not linkage_path.exists():
        legacy_path = Path("data/candidate_execution_linkage.csv")
        if legacy_path.exists():
            logger.info("DeployLinker CSV not found, falling back to legacy linkage: %s", legacy_path)
            linkage_path = legacy_path
    candidates = filter_backtest_candidates(
        all_candidates,
        linkage_path=linkage_path,
        min_score=args.min_score,
        bypass_only=args.bypass_only,
    )
    scanner_filtered_count = len(candidates)
    candidates = candidates.copy()
    candidates["candidate_source"] = (
        "bypass_route" if args.bypass_only else "standard_route"
    )

    # The active FASTWIN cohort is the union of the scanner route, the optional
    # historical champion candidate universe, and explicit authoritative IDs.
    # Champion rows bypass scanner-only eligibility filters but still require
    # current score and current grid geometry; old outcomes/features are never
    # reused.
    candidate_ids = cast(pd.Series, all_candidates["candidate_id"]).astype("string")
    authoritative_rows = cast(
        pd.DataFrame,
        all_candidates.loc[candidate_ids.isin(list(authoritative_pool_ids))].copy(),
    )
    if not authoritative_rows.empty:
        authoritative_rows = authoritative_rows.drop_duplicates(
            subset=["candidate_id"], keep="first"
        )
        authoritative_rows["candidate_source"] = "authoritative_workbook"

    champion_rows = cast(pd.DataFrame, all_candidates.iloc[0:0].copy())
    if champion_pool_ids:
        champion_rows = cast(
            pd.DataFrame,
            all_candidates.loc[candidate_ids.isin(list(champion_pool_ids))].copy(),
        )
        champion_score = cast(
            pd.Series,
            pd.to_numeric(cast(pd.Series, champion_rows["score"]), errors="coerce"),
        )
        champion_rows = cast(
            pd.DataFrame,
            champion_rows.loc[champion_score.ge(args.min_score)].copy(),
        )
        for col in ("grid_lower", "grid_upper", "num_grids"):
            if col in champion_rows.columns:
                champion_rows = cast(
                    pd.DataFrame,
                    champion_rows.loc[champion_rows[col].notna()].copy(),
                )
        champion_rows = cast(
            pd.DataFrame,
            champion_rows.sort_values("score", ascending=False)
            .drop_duplicates(subset=["candidate_id"], keep="first")
            .reset_index(drop=True),
        )
        champion_rows["candidate_source"] = "champion_candidate_pool"
        missing_champion = champion_pool_ids.difference(
            set(champion_rows["candidate_id"].astype(str))
        )
        if missing_champion:
            raise ValueError(
                "Champion IDs are missing current scanner configuration or current "
                f"score/geometry: {len(missing_champion)} rows"
            )

    standard_ids = set(candidates["candidate_id"].astype(str))
    authoritative_ids_selected = set(authoritative_rows["candidate_id"].astype(str))
    champion_ids_selected = set(champion_rows["candidate_id"].astype(str))
    candidates = cast(
        pd.DataFrame,
        pd.concat([authoritative_rows, champion_rows, candidates], ignore_index=True)
        .drop_duplicates(subset=["candidate_id"], keep="first")
        .reset_index(drop=True),
    )
    authoritative_added_count = len(
        authoritative_ids_selected.difference(standard_ids | champion_ids_selected)
    )
    champion_added_count = len(
        champion_ids_selected.difference(standard_ids | authoritative_ids_selected)
    )
    logger.info(
        "Candidate union: %d workbook IDs, %d champion IDs, %d scanner-filtered rows, "
        "%d authoritative rows added, %d champion rows added, %d union rows before window",
        len(authoritative_pool_ids),
        len(champion_pool_ids),
        scanner_filtered_count,
        authoritative_added_count,
        champion_added_count,
        len(candidates),
    )

    # Restrict candidate selection by explicit scan-date window.
    scan_timestamps = candidates["candidate_id"].map(
        lambda x: _parse_scan_timestamp(str(x)),
    )
    candidate_id_mask = cast(pd.Series, candidates["candidate_id"]).astype("string")
    champion_exception_mask = candidate_id_mask.isin(list(champion_pool_ids))
    window_mask = (
        cast(pd.Series, scan_timestamps).notna()
        & (
            (cast(pd.Series, scan_timestamps) >= pool_start)
            | champion_exception_mask
        )
        & (cast(pd.Series, scan_timestamps) <= pool_end_inclusive)
    )
    if not isinstance(candidates, pd.DataFrame):
        candidates = pd.DataFrame(candidates)
    pre_window_count = len(candidates)
    candidates = cast(
        pd.DataFrame,
        candidates.loc[window_mask].reset_index(drop=True),
    )
    dropped_unparseable = int(
        (cast(pd.Series, scan_timestamps).isna()).sum()
    )
    if dropped_unparseable:
        logger.info("Dropped %d rows with unparseable candidate_id timestamps", dropped_unparseable)
    logger.info(
        "After pool window filter [%s, %s]: %d / %d",
        pool_start.date().isoformat(),
        pool_end_inclusive.date().isoformat(),
        len(candidates),
        pre_window_count,
    )
    union_window_count = len(candidates)

    if candidates.empty:
        logger.warning("No backtestable candidates found. Exiting.")
        sys.exit(0)

    # Prioritisation is mode-dependent:
    # - standard mode: scan-time composite score orders already valid grids
    # - bypass-only mode: micro_osc_score is the primary archetype signal
    if args.bypass_only and "micro_osc_score" in candidates.columns:
        candidates = candidates.sort_values(
            ["micro_osc_score", "score"], ascending=[False, False]
        ).head(args.max_candidates)
        micro_osc = cast(pd.Series, candidates["micro_osc_score"])
        score_series = cast(pd.Series, candidates["score"])
        logger.info(
            "Selected %d bypass candidates for backtesting "
            "(micro_osc_score range: %.4f - %.4f, score range: %.1f - %.1f)",
            len(candidates),
            float(micro_osc.min()),
            float(micro_osc.max()),
            float(score_series.min()),
            float(score_series.max()),
        )
    else:
        candidates = candidates.sort_values("score", ascending=False).head(
            args.max_candidates
        )
        score_series = cast(pd.Series, candidates["score"])
        logger.info(
            "Selected %d candidates for backtesting (score range: %.1f - %.1f)",
            len(candidates),
            float(score_series.min()),
            float(score_series.max()),
        )

    # ── Dry run: just report and exit ───────────────────────────────────
    if args.dry_run:
        logger.info("")
        logger.info("DRY RUN — candidate summary:")
        logger.info("-" * 70)
        for _, row in candidates.head(20).iterrows():
            logger.info(
                "  %s | score=%.1f | grid=[%.4f, %.4f] x%d | file=%s",
                row["symbol"],
                float(cast(Any, row["score"])),
                float(cast(Any, row["grid_lower"])),
                float(cast(Any, row["grid_upper"])),
                int(float(cast(Any, row["num_grids"]))),
                row.get("scan_file", "?"),
            )
        if len(candidates) > 20:
            logger.info("  ... and %d more", len(candidates) - 20)
        logger.info("-" * 70)
        logger.info("Total: %d candidates would be backtested", len(candidates))
        return

    # ── P4: Load live durations if requested ───────────────────────────
    live_duration_map: dict[str, int] = {}
    if args.match_live_duration:
        expired_path = Path(args.expired_bots_path)
        if expired_path.exists():
            try:
                expired_df = pd.read_excel(expired_path)
                for _, erow in expired_df.iterrows():
                    cid = str(erow.get("candidate_id", ""))
                    dur_h = erow.get("duration_hours")
                    if cid and dur_h is not None and pd.notna(dur_h):
                        live_duration_map[cid] = int(float(dur_h) * 60)
                logger.info(
                    "Loaded %d live durations from %s",
                    len(live_duration_map), expired_path,
                )
            except Exception as exc:
                logger.warning("Cannot load expired bots: %s", exc)
        else:
            logger.warning("Expired bots file not found: %s", expired_path)

    # ── Step 3: Run backtests ───────────────────────────────────────────
    logger.info("")
    logger.info("Connecting to Binance API...")
    client = BinanceClient()

    backtest_results: list[dict] = []
    training_rows: list[dict] = []
    skipped = 0
    exchange_info: dict[str, Any] | None = None

    try:
        if args.realism_profile == CANDIDATE_TIME_PUBLIC_MARKET_PROFILE:
            try:
                exchange_info = await client.get_exchange_info()
            except Exception as exc:
                logger.warning("Public-market profile cannot load exchangeInfo: %s", exc)
                exchange_info = None

        total = len(candidates)
        for idx, (_, row) in enumerate(candidates.iterrows()):
            symbol = str(row["symbol"])
            candidate_id = str(row["candidate_id"])
            score = float(cast(Any, row["score"]))

            logger.info(
                "[%d/%d] %s (score=%.1f, id=%s)",
                idx + 1, total, symbol, score, candidate_id,
            )

            # Preserve the scan timestamp as metadata, but start the backtest
            # from the deployment_ready CSV output time.
            scan_ts = _parse_scan_timestamp(candidate_id)
            if scan_ts is None:
                logger.warning("  Cannot parse scan timestamp from %s — skipping", candidate_id)
                skipped += 1
                continue
            try:
                backtest_start_ts = resolve_backtest_start_timestamp(dict(row))
            except ValueError as exc:
                logger.warning("  Cannot resolve backtest start for %s: %s — skipping", candidate_id, exc)
                skipped += 1
                continue

            # Fetch klines
            try:
                klines_df = await fetch_historical_klines(
                    client, symbol, backtest_start_ts, hours=args.hours,
                )
            except Exception as exc:
                logger.warning("  Kline fetch failed for %s: %s — skipping", symbol, exc)
                skipped += 1
                continue

            if len(klines_df) < args.min_bars:
                logger.warning(
                    "  Only %d bars for %s (need %d) — skipping",
                    len(klines_df), symbol, args.min_bars,
                )
                skipped += 1
                continue

            candidate_payload = dict(row)
            valid_indicator = 1.0 if candidate_id in authoritative_pool_ids else 0.5
            candidate_payload["valid_indicator"] = valid_indicator
            candidate_payload["sample_weight_override"] = valid_indicator
            if args.realism_profile == CANDIDATE_TIME_PUBLIC_MARKET_PROFILE:
                if exchange_info is None:
                    logger.warning(
                        "  exchangeInfo unavailable for public-market profile — skipping %s",
                        symbol,
                    )
                    skipped += 1
                    continue

                candidate_payload.update(
                    extract_exchange_filter_settings(exchange_info, symbol)
                )
                candidate_payload["historical_depth_source"] = "missing"

                window_end = backtest_start_ts + timedelta(hours=args.hours)
                start_ms = int(backtest_start_ts.timestamp() * 1000)
                end_ms = int(window_end.timestamp() * 1000)

                try:
                    mark_klines = await fetch_historical_klines(
                        client,
                        symbol,
                        backtest_start_ts,
                        hours=args.hours,
                        price_source="mark",
                    )
                    if not mark_klines.empty:
                        klines_df = attach_mark_close(klines_df, mark_klines)
                        candidate_payload["mark_price_source"] = "binance_mark_price_klines"
                        candidate_payload["mark_price_rows"] = len(mark_klines)
                    else:
                        candidate_payload["mark_price_source"] = "missing"
                        candidate_payload["mark_price_rows"] = 0
                except Exception as exc:
                    logger.warning("  Mark-price kline fetch failed for %s: %s", symbol, exc)
                    candidate_payload["mark_price_source"] = "missing"
                    candidate_payload["mark_price_rows"] = 0

                try:
                    funding_rows = await client.get_funding_rate(
                        symbol,
                        limit=1000,
                        start_time=start_ms,
                        end_time=end_ms,
                    )
                    funding_series, funding_status, funding_count = (
                        funding_rate_series_from_rows(funding_rows)
                    )
                    if funding_series is not None:
                        candidate_payload["funding_rate_series"] = funding_series
                    candidate_payload["funding_series_status"] = funding_status
                    candidate_payload["funding_series_source"] = "binance_funding_rate"
                    candidate_payload["funding_series_rows"] = funding_count
                except Exception as exc:
                    logger.warning("  Funding-rate fetch failed for %s: %s", symbol, exc)
                    candidate_payload["funding_series_status"] = "missing"
                    candidate_payload["funding_series_source"] = "missing"
                    candidate_payload["funding_series_rows"] = 0

            # Run backtest
            try:
                # P4: Duration override from live bot data
                duration_bars = live_duration_map.get(candidate_id)
                result = run_single_backtest(
                    candidate_row=candidate_payload,
                    klines_df=klines_df,
                    capital=args.capital,
                    leverage=args.leverage,
                    max_holding_bars=args.hours * 60,
                    duration_bars=duration_bars,
                    realism_profile=args.realism_profile,
                )
            except Exception as exc:
                logger.warning("  Backtest failed for %s: %s — skipping", symbol, exc)
                skipped += 1
                continue

            logger.info(
                "  PnL: %+.2f%% | DD: %.2f%% | RTs: %d | WR: %.1f%% | TIR: %.1f%%",
                result["net_pnl_pct"],
                result["max_drawdown_pct"],
                result["round_trips"],
                result["win_rate"],
                result["time_in_range_pct"],
            )

            # Merge scanner features into result for the full output
            for feat_col in (
                             # Base micro/range/volatility features
                             "range_prob", "trend_prob", "hmm_range_prob",
                             "hmm_trend_prob", "survival_prob", "hurst_exponent",
                             "ou_halflife", "funding_rate", "adx_1h", "adx_15m",
                             "adx_5m", "rsi_15m", "ev_score", "range_size_pct",
                             "meta_prob", "micro_osc_score", "micro_osc_bypass",
                             "ema_crosses_5m", "vwap_crosses_5m", "atr_pct_15m",
                             "bb_width", "quote_volume_24h", "open_interest",
                             # afml_enriched_v20260318: enriched features
                             "hmm_tail_cvar_95", "tos_gcf", "persistence_prob",
                             "micro_round_trip_cost_pct",
                             "regime_conf",
                             # Tier 4 context features
                             "long_short_ratio", "funding_rate_zscore",
                             "open_interest_change_pct",
                             "bb_width_ratio_1h_15m",
                             # Binance-derived crowding / flow context (v20260407)
                             "top_account_lsr_log", "top_position_lsr_log",
                             "top_position_vs_account_delta", "global_lsr_log",
                             "taker_imbalance", "taker_buy_sell_log", "basis_pct",
                             # Binance-derived execution / liquidity context (v20260407)
                             "spread_pct", "top20_bid_depth_usdt",
                             "top20_ask_depth_usdt", "top20_depth_min_usdt",
                             "book_imbalance_top20", "oi_notional", "oi_to_volume",
                             ):
                val = row.get(feat_col)
                if val is not None and pd.notna(val):
                    result[feat_col] = val

            result["scan_timestamp"] = scan_ts.isoformat()
            result["backtest_start_ts_utc"] = backtest_start_ts.isoformat()
            result["backtest_time_anchor"] = str(
                row.get("candidate_available_source", "deployment_ready_csv_mtime")
            )
            backtest_results.append(result)

            # Convert to training row
            training_row = convert_to_training_row(
                backtest_result=result,
                candidate_row=candidate_payload,
            )
            training_row["valid_indicator"] = valid_indicator
            # Step 4: Copy hierarchical y onto result for conformal/MI alignment
            result["y"] = training_row["y"]
            training_rows.append(training_row)

            # Rate limiting
            await asyncio.sleep(args.delay)

    finally:
        await client.close()

    # ── Step 4: Save outputs ────────────────────────────────────────────
    if not backtest_results:
        logger.warning("No successful backtests. Nothing to save.")
        sys.exit(0)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backtest results CSV
    # ── Dedup rule (alignment-v1): keep latest backtest per candidate_id ──
    # Same candidate_id can appear across multiple daily runs.  When multiple
    # rows share a candidate_id, keep the one with the latest backtest_timestamp
    # (produced by the latest engine version).
    results_path = output_dir / f"backtest_results_{timestamp}.csv"
    results_df = pd.DataFrame(backtest_results)
    if "candidate_id" in results_df.columns and "backtest_timestamp" in results_df.columns:
        pre_dedup = len(results_df)
        results_df = results_df.sort_values("backtest_timestamp")
        results_df = results_df.drop_duplicates(subset=["candidate_id"], keep="last")
        deduped = pre_dedup - len(results_df)
        if deduped > 0:
            logger.info("Dedup: removed %d duplicate candidate_id rows (kept latest)", deduped)
    results_df.to_csv(results_path, index=False)
    logger.info("Backtest results saved to: %s (%d rows)", results_path, len(results_df))

    # Training data CSV
    training_path = output_dir / f"training_data_{timestamp}.csv"
    training_df = pd.DataFrame(training_rows)
    training_df.to_csv(training_path, index=False)
    logger.info("Training data saved to: %s (%d rows)", training_path, len(training_df))

    run_manifest = {
        "generation_mode": "fresh_full_pool" if args.fastwin_full_pool else "fresh_candidate_backtest",
        "full_pool": bool(args.fastwin_full_pool),
        "realism_profile": args.realism_profile,
        "results_dir": str(Path(args.results_dir).resolve()),
        "authoritative_pool_path": str(Path(args.authoritative_pool_path).resolve()),
        "authoritative_pool_id_count": len(authoritative_pool_ids),
        "authoritative_rows_added": authoritative_added_count,
        "champion_candidate_pool_path": (
            str(Path(args.champion_candidate_pool_path).resolve())
            if args.champion_candidate_pool_path
            else ""
        ),
        "champion_candidate_id_count": len(champion_pool_ids),
        "champion_rows_added": champion_added_count,
        "scanner_filtered_count_before_union": scanner_filtered_count,
        "union_count_before_window": pre_window_count,
        "union_count_after_window": union_window_count,
        "authoritative_ids_in_selected_union": int(
            cast(pd.Series, candidates["candidate_id"])
            .astype("string")
            .isin(list(authoritative_pool_ids))
            .sum()
        ),
        "pool_start_date": args.pool_start_date,
        "pool_end_date": args.pool_end_date or datetime.now(timezone.utc).date().isoformat(),
        "max_candidates": args.max_candidates,
        "hours": args.hours,
        "min_bars": args.min_bars,
        "leverage": args.leverage,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_file": str(results_path.resolve()),
        "training_file": str(training_path.resolve()),
        "successful_rows": int(len(training_df)),
    }
    (output_dir / "backtest_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Phase 3: Fit conformal risk controller on backtest results
    try:
        from neutralgrid.calibration.conformal_risk_control_v20260311 import (
            ConformalRiskController,
            ConformalConfig,
        )

        # Load results we just saved
        if results_path.exists():
            results_df = pd.read_csv(results_path)
            _pnl_col = "net_pnl_pct" if "net_pnl_pct" in results_df.columns else "pnl_pct"
            if "meta_prob" in results_df.columns and _pnl_col in results_df.columns:
                import numpy as _np
                _probs_arr = _np.asarray(pd.to_numeric(pd.Series(results_df["meta_prob"]), errors="coerce"), dtype=float)
                _pnl_arr = _np.asarray(pd.to_numeric(pd.Series(results_df[_pnl_col]), errors="coerce"), dtype=float)
                _valid = _np.isfinite(_probs_arr) & _np.isfinite(_pnl_arr)
                if int(_np.sum(_valid)) >= 20:
                    _probs_clean = _probs_arr[_valid]
                    # Step 4B: Use hierarchical y for conformal (fallback to hurdle for legacy CSVs)
                    if "y" in results_df.columns:
                        _labels = _np.asarray(
                            pd.to_numeric(pd.Series(results_df["y"]), errors="coerce"),
                            dtype=float,
                        )[_valid]
                    else:
                        _labels = (_pnl_arr[_valid] > get_config().barrier.meta_hurdle_pct).astype(float)
                    ctrl = ConformalRiskController(ConformalConfig())
                    quantile = ctrl.fit(_labels, _probs_clean, gate_type="meta_prob")
                    ctrl.save(quantile)
                    logger.info(
                        "Conformal quantile saved (q_hat=%.4f, coverage=%.4f, n=%d)",
                        quantile.quantile_value,
                        quantile.coverage_guarantee,
                        quantile.n_calibration,
                    )
                else:
                    logger.info("Not enough samples for conformal calibration (%d < 20)", int(_np.sum(_valid)))
            else:
                logger.info("Conformal calibration skipped: missing meta_prob or net_pnl_pct/pnl_pct columns")
    except ImportError:
        logger.debug("Conformal risk control module not available")
    except Exception as _crc_err:
        logger.warning("Conformal calibration failed (non-fatal): %s", _crc_err)

    # Phase 2: Compute MI weights from backtest signals → data/cache/mi_weights.json
    try:
        from neutralgrid.scanner.mi_weighted_scorer_v20260311 import (
            compute_mi_weights,
            MIWeightConfig,
        )
        import numpy as _mi_np

        _mi_artifact_generation_enabled = False
        if not _mi_artifact_generation_enabled:
            logger.info(
                "MI weight artifact generation skipped: runtime MI remains disabled "
                "until source-column and raw-MI provenance are implemented."
            )
        elif results_path.exists():
            _mi_df = pd.read_csv(results_path)
            _pnl_col_mi = "net_pnl_pct" if "net_pnl_pct" in _mi_df.columns else "pnl_pct"
            # Step 4C: 3-tier label precedence for MI — y → label_positive_by_horizon → hurdle
            if "y" in _mi_df.columns:
                _mi_target = _mi_np.asarray(
                    pd.to_numeric(pd.Series(_mi_df["y"]), errors="coerce"),
                    dtype=float,
                )
            elif "label_positive_by_horizon" in _mi_df.columns:
                _mi_target = _mi_np.asarray(
                    pd.to_numeric(pd.Series(_mi_df["label_positive_by_horizon"]), errors="coerce"),
                    dtype=float,
                )
            elif _pnl_col_mi in _mi_df.columns:
                _mi_pnl = _mi_np.asarray(
                    pd.to_numeric(pd.Series(_mi_df[_pnl_col_mi]), errors="coerce"),
                    dtype=float,
                )
                _mi_target = (_mi_pnl > get_config().barrier.meta_hurdle_pct).astype(float)
            else:
                _mi_target = _mi_np.array([])

            if len(_mi_target) >= 15 and _mi_np.isfinite(_mi_target).sum() >= 15:
                # Extract the 4 MI signals: similarity_score, profile_proba, ev_score, meta_prob
                # From backtest results we have: score (=similarity), ev_score, range_prob, meta_prob
                _mi_signals: dict[str, _mi_np.ndarray] = {}
                _signal_map = {
                    "similarity_score": "score",       # scan score maps to similarity
                    "profile_proba": "range_prob",     # closest proxy in backtest results
                    "ev_score": "ev_score",
                    "meta_prob": "meta_prob",
                }
                for mi_name, csv_col in _signal_map.items():
                    if csv_col in _mi_df.columns:
                        _mi_signals[mi_name] = _mi_np.asarray(
                            pd.to_numeric(pd.Series(_mi_df[csv_col]), errors="coerce"),
                            dtype=float,
                        )

                if len(_mi_signals) >= 2:
                    _mi_weights_result, _mi_raw = compute_mi_weights(
                        _mi_signals, _mi_target, MIWeightConfig()
                    )
                    # Save to cache for run_full_pipeline.py to load
                    import json as _mi_json
                    _mi_cache_path = Path("data/cache/mi_weights.json")
                    _mi_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    _mi_artifact = {
                        **_mi_weights_result,
                        "_raw_mi": _mi_raw,
                        "_n_samples": int(_mi_np.isfinite(_mi_target).sum()),
                        "_source": str(results_path),
                    }
                    _mi_cache_path.write_text(
                        _mi_json.dumps(_mi_artifact, indent=2),
                        encoding="utf-8",
                    )
                    logger.info(
                        "MI weights saved to %s: %s (N=%d)",
                        _mi_cache_path,
                        {k: f"{v:.4f}" for k, v in _mi_weights_result.items()},
                        _mi_artifact["_n_samples"],
                    )
                else:
                    logger.info("MI weights: fewer than 2 signal columns available, skipping")
            else:
                logger.info("MI weights: insufficient samples (%d), skipping",
                           int(_mi_np.isfinite(_mi_target).sum()) if len(_mi_target) > 0 else 0)
    except ImportError:
        logger.debug("MI weighted scorer module not available")
    except Exception as _mi_err:
        logger.warning("MI weight computation failed (non-fatal): %s", _mi_err)

    # ── Summary ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("BACKTESTING SUMMARY")
    logger.info("=" * 70)
    logger.info("  Candidates tested:   %d", len(backtest_results))
    logger.info("  Skipped:             %d", skipped)

    pnls = [r["net_pnl_pct"] for r in backtest_results]
    _hurdle = get_config().barrier.meta_hurdle_pct
    positive = sum(1 for p in pnls if p >= _hurdle)
    logger.info("  Mean PnL:            %+.2f%%", sum(pnls) / len(pnls))
    logger.info("  Median PnL:          %+.2f%%", sorted(pnls)[len(pnls) // 2])
    logger.info("  Best PnL:            %+.2f%%", max(pnls))
    logger.info("  Worst PnL:           %+.2f%%", min(pnls))
    logger.info("  Above hurdle (%.1f%%%%):  %d / %d (%.0f%%)",
                _hurdle, positive, len(pnls), positive / len(pnls) * 100)
    logger.info("")
    logger.info("  Results:  %s", results_path)
    logger.info("  Training: %s", training_path)
    logger.info("=" * 70)

    # ── Step 15 (Plan v6.0): Run alignment audit if --audit flag is set ──
    if args.audit:
        try:
            from neutralgrid.training.btk_alignment_audit_v20260316 import AlignmentAuditor
            from neutralgrid.training.live_outcome_ingestor import LiveOutcomeIngestor

            ingestor = LiveOutcomeIngestor(
                expired_bots_path=Path(args.expired_bots_path),
            )
            auditor = AlignmentAuditor(ingestor=ingestor)
            audit_report = auditor.compute_alignment_report(
                backtest_results_dir=Path(args.output),
                required_formula_version=None,  # Accept all versions
            )
            if not audit_report.empty:
                audit_path = Path(args.output) / f"alignment_audit_{timestamp}.csv"
                audit_report.to_csv(audit_path, index=False)
                eligible_series = cast(pd.Series, audit_report["calibration_eligible"])
                eligible = int(eligible_series.sum())
                logger.info(
                    "Alignment audit complete: %d pairs, %d eligible. Saved to %s",
                    len(audit_report), eligible, audit_path,
                )
            else:
                logger.info("Alignment audit: no matched pairs (need both backtest + live data)")
        except ImportError as _audit_imp_err:
            logger.warning("Alignment audit module not available: %s", _audit_imp_err)
        except Exception as _audit_err:
            logger.warning("Alignment audit failed (non-fatal): %s", _audit_err)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)
