#!/usr/bin/env python3
"""PIPELINE_FIX v2 — at-scale fast-winner dataset generation.

Backtests EXISTING scanner candidates (ex-ante features already computed) FORWARD
from their real scan timestamp over a long observation window so the ENDOGENOUS
time-to-target (first bar net MTM PnL reaches +3%) can be measured across the 7h
boundary. This is the data the fast-winner label needs and that the 6h pool could
never express.

Why this bypasses ``load_all_scanner_csvs``: that loader overwrites
``candidate_available_ts_utc`` with the CSV FILE mtime (candidate_pipeline.py:266-269),
which on this checkout resolves the backtest start to ~now (no forward data). Here we
read the CSVs with pandas and anchor the start to the candidate_id scan timestamp,
which ``resolve_backtest_start_timestamp`` reads from the row column directly.

READ-ONLY w.r.t. models/artifacts: it only writes a training-data CSV. No promotion.

Usage::

    python scripts/generate_fastwin_dataset.py --hours 24 --min-bars 1440 \
        --min-score 45 --max-candidates 400 \
        --output outputs/audits/fastwin_shadow
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# backtest/ is not an installed package; ensure the repo root is importable when
# this script is run from scripts/ (mirrors the test-suite sys.path pattern).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from neutralgrid.backtest.realism_governance import (
    CANDIDATE_TIME_GEOMETRIC_PROFILE,
    validate_realism_output_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("backtest_realistic").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("generate_fastwin_dataset")

_GRID_COLS = ("grid_lower", "grid_upper", "num_grids")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate fast-winner labeled backtest dataset")
    p.add_argument("--results-dir", default="results", help="dir with deployment_ready_*.csv")
    p.add_argument("--hours", type=int, default=24, help="observation window hours (default 24)")
    p.add_argument("--min-bars", type=int, default=1440, help="min 1m bars required (default 1440)")
    p.add_argument("--min-score", type=float, default=45.0)
    p.add_argument("--min-age-hours", type=float, default=25.0,
                   help="only candidates whose scan ts is >= this many hours in the past "
                        "(so the forward window exists). Default 25 (margin over 24).")
    p.add_argument("--max-candidates", type=int, default=400)
    p.add_argument("--capital", type=float, default=400.0)
    p.add_argument("--leverage", type=int, default=10)
    p.add_argument("--delay", type=float, default=0.12, help="seconds between candidates")
    p.add_argument("--output", default="outputs/audits/fastwin_shadow")
    p.add_argument("--symbol-cap", type=int, default=0,
                   help="if >0, max candidates per symbol (diversify across symbols)")
    args = p.parse_args()
    try:
        validate_realism_output_path(CANDIDATE_TIME_GEOMETRIC_PROFILE, args.output)
    except ValueError as exc:
        p.error(str(exc))
    return args


def load_candidates(results_dir: str, min_score: float, min_age_hours: float,
                    max_candidates: int, symbol_cap: int) -> pd.DataFrame:
    from neutralgrid.backtest.candidate_pipeline import _parse_scan_timestamp

    files = sorted(glob.glob(str(Path(results_dir) / "deployment_ready_*.csv")))
    if not files:
        logger.error("No deployment_ready_*.csv in %s", results_dir)
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as exc:
            logger.warning("read failed %s: %s", f, exc)
    df = pd.concat(frames, ignore_index=True)
    if "candidate_id" not in df.columns:
        logger.error("no candidate_id column")
        return pd.DataFrame()

    df["_scan_ts"] = df["candidate_id"].map(lambda c: _parse_scan_timestamp(str(c)))
    df = df[df["_scan_ts"].notna()].copy()
    # NOTE: uses the harness clock; real Binance data is at the same wall-clock.
    now = datetime.now(timezone.utc)
    df["_age_h"] = df["_scan_ts"].map(lambda t: (now - t).total_seconds() / 3600.0)
    df = df[df["_age_h"] >= float(min_age_hours)]
    if "score" in df.columns:
        df = df[pd.to_numeric(df["score"], errors="coerce") >= float(min_score)]
    for c in _GRID_COLS:
        if c not in df.columns:
            logger.error("missing grid column %s", c)
            return pd.DataFrame()
        df = df[pd.to_numeric(df[c], errors="coerce").notna()]
    # one row per candidate_id (keep latest scan), then optional per-symbol cap
    df = df.sort_values("_scan_ts").drop_duplicates(subset=["candidate_id"], keep="last")
    df = df.sort_values(["symbol", "_scan_ts"])
    if symbol_cap and symbol_cap > 0:
        df = df.groupby("symbol", group_keys=False).head(symbol_cap)
    # Keep symbol-diverse, score-MIXED order (do NOT sort by score, which would
    # bias the head() toward only the highest scores = mostly fast winners).
    if max_candidates and max_candidates > 0:
        df = df.head(max_candidates)
    return df.reset_index(drop=True)


async def main() -> None:
    args = parse_args()
    from neutralgrid.api.binance_client import BinanceClient
    from neutralgrid.backtest.candidate_pipeline import (
        fetch_historical_klines,
        run_single_backtest,
        convert_to_training_row,
    )
    cand = load_candidates(args.results_dir, args.min_score, args.min_age_hours,
                           args.max_candidates, args.symbol_cap)
    if cand.empty:
        logger.error("No eligible candidates."); sys.exit(1)
    logger.info("Eligible candidates: %d across %d symbols (scan>=%.0fh old, score>=%.0f)",
                len(cand), cand["symbol"].nunique(), args.min_age_hours, args.min_score)

    client = BinanceClient()
    rows: List[Dict[str, Any]] = []
    ok = skipped = 0
    try:
        for i, (_, row) in enumerate(cand.iterrows()):
            sym = str(row.get("symbol", ""))
            start = row["_scan_ts"].to_pydatetime() if hasattr(row["_scan_ts"], "to_pydatetime") else row["_scan_ts"]
            payload = {k: v for k, v in row.items() if not str(k).startswith("_")}
            # Anchor the backtest start to the real scan time (read by
            # resolve_backtest_start_timestamp; NOT overwritten because we bypass
            # load_all_scanner_csvs).
            payload["candidate_available_ts_utc"] = start.isoformat()
            payload["candidate_available_source"] = "fastwin_scan_time"
            try:
                klines = await fetch_historical_klines(client, sym, start, hours=args.hours)
            except Exception as exc:
                logger.warning("[%d/%d] %s kline fetch failed: %s", i + 1, len(cand), sym, exc)
                skipped += 1; continue
            if klines is None or len(klines) < args.min_bars:
                skipped += 1; continue
            try:
                result = run_single_backtest(
                    candidate_row=payload,
                    klines_df=klines,
                    capital=args.capital,
                    leverage=args.leverage,
                    max_holding_bars=args.hours * 60,
                    realism_profile=CANDIDATE_TIME_GEOMETRIC_PROFILE,
                )
                training_row = convert_to_training_row(
                    backtest_result=result, candidate_row=payload,
                )
            except Exception as exc:
                logger.warning("[%d/%d] %s backtest failed: %s", i + 1, len(cand), sym, exc)
                skipped += 1; continue
            rows.append(training_row)
            ok += 1
            if ok % 25 == 0:
                logger.info("  ... %d backtested (%d skipped), latest %s pnl=%.2f t2t=%s",
                            ok, skipped, sym, result.get("net_pnl_pct", float("nan")),
                            result.get("time_to_target_hours"))
            await asyncio.sleep(args.delay)
    finally:
        try:
            await client.close()
        except Exception:
            pass

    if not rows:
        logger.error("No successful backtests."); sys.exit(1)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"training_data_fastwin_{ts}.csv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    logger.info("WROTE %d rows -> %s (skipped %d)", len(df_out), out_path, skipped)
    # quick label summary
    if "time_to_target_hours" in df_out.columns:
        ttt = pd.to_numeric(df_out["time_to_target_hours"], errors="coerce")
        reached = ttt.notna()
        fast = reached & (ttt <= 7.0)
        logger.info("LABEL: target_reached=%d/%d, fast(<=7h)=%d, slow(>7h)=%d, never=%d; "
                    "t2t hours min/med/max=%.2f/%.2f/%.2f",
                    int(reached.sum()), len(df_out), int(fast.sum()),
                    int((reached & (ttt > 7.0)).sum()), int((~reached).sum()),
                    float(ttt[reached].min()) if reached.any() else float("nan"),
                    float(ttt[reached].median()) if reached.any() else float("nan"),
                    float(ttt[reached].max()) if reached.any() else float("nan"))


if __name__ == "__main__":
    asyncio.run(main())
