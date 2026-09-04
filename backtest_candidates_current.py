#!/usr/bin/env python3
"""
Realistic backtest of candidate grid configs against CURRENT market data.

Reads the backtest_results CSV produced by backtest_candidates.py, deduplicates
to the best-scoring config per symbol, fetches the most recent N hours of 1m
klines (not scan-time), and runs RealisticGridBacktester.

This is an out-of-sample validation: the grids were designed for a past moment,
but we test them against today's price action.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path for backtest/ imports
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from neutralgrid.core.config import get_config
from neutralgrid.api.binance_client import BinanceClient
from backtest.btk_unified_runner import build_training_config, run_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backtest_current")
logging.getLogger("backtest_realistic").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def fetch_recent_klines(
    client: BinanceClient, symbol: str, hours: int = 6,
) -> pd.DataFrame:
    """Fetch the most recent ``hours`` hours of 1m klines."""
    limit = min(hours * 60, 1500)
    klines = await client.get_klines(symbol=symbol, interval="1m", limit=limit)
    if not klines:
        return pd.DataFrame()
    df = pd.DataFrame(
        klines,
        columns=pd.Index([
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ]),
    )
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Realistic backtest of candidate grids against current market data",
    )
    parser.add_argument(
        "--input", type=str,
        default=None,
        help="Path to backtest_results CSV from backtest_candidates.py",
    )
    parser.add_argument("--hours", type=int, default=6, help="Hours of recent data")
    parser.add_argument("--capital", type=float, default=400.0)
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument(
        "--output", type=str,
        default=None,
    )
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    # ── Resolve input path (auto-detect latest if not specified) ────────
    if args.input is None:
        candidates_dir = Path("data/backtest_candidates")
        bt_files = sorted(candidates_dir.glob("backtest_results_*.csv"))
        if not bt_files:
            logger.error("No backtest_results_*.csv found in %s", candidates_dir)
            sys.exit(1)
        args.input = str(bt_files[-1])  # latest by filename sort
        logger.info("Auto-detected input: %s", args.input)

    # ── Resolve output path (auto-detect latest if not specified) ─────
    if args.output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d")
        args.output = f"data/backtest_candidates/current_market_backtest_{ts}.csv"

    # ── Load previous results ───────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    prev = pd.read_csv(input_path)
    logger.info("Loaded %d rows from %s", len(prev), input_path)

    # Deduplicate: keep highest-scoring config per symbol
    prev_sorted = prev.sort_values("score", ascending=False)
    best_per_symbol = prev_sorted.drop_duplicates(subset=["symbol"], keep="first")
    logger.info(
        "Deduplicated to %d unique symbols (from %d configs)",
        len(best_per_symbol), len(prev),
    )

    # ── Run backtests against current market data ───────────────────────
    logger.info("=" * 70)
    logger.info("REALISTIC BACKTEST vs CURRENT MARKET DATA (%dh)", args.hours)
    logger.info("=" * 70)

    client = BinanceClient()
    results = []
    skipped = 0
    total = len(best_per_symbol)

    try:
        for idx, (_, row) in enumerate(best_per_symbol.iterrows()):
            symbol = str(row["symbol"])
            grid_lower = float(row["grid_lower"])
            grid_upper = float(row["grid_upper"])
            num_grids = int(float(row["num_grids"]))
            candidate_id = str(row.get("candidate_id", ""))
            prev_pnl = float(row["net_pnl_pct"])

            logger.info(
                "[%d/%d] %s  grid=[%.4f, %.4f] x%d  (scan PnL: %+.2f%%)",
                idx + 1, total, symbol, grid_lower, grid_upper, num_grids, prev_pnl,
            )

            # Fetch recent klines
            try:
                klines_df = await fetch_recent_klines(client, symbol, hours=args.hours)
            except Exception as exc:
                logger.warning("  Kline fetch failed: %s — skipping", exc)
                skipped += 1
                continue

            if len(klines_df) < 355:
                logger.warning(
                    "  Only %d bars (need 355) — skipping", len(klines_df),
                )
                skipped += 1
                continue

            # Check if current price is even in range
            current_price = klines_df.iloc[-1]["close"]
            in_range = grid_lower <= current_price <= grid_upper
            price_vs_range = "IN RANGE" if in_range else "OUT OF RANGE"

            # Funding rate from candidate if available
            # Step 13 (Plan v6.0): Preserve funding rate sign — negative rates
            # mean shorts pay longs, which is material for grid bot PnL.
            # abs() was stripping this information.
            funding_rate = 0.0001
            fr = row.get("funding_rate")
            if fr is not None and pd.notna(fr):
                fr_val = float(fr)
                funding_rate = fr_val if fr_val != 0 else 0.0001

            cfg = build_training_config(
                symbol=symbol,
                lower=grid_lower,
                upper=grid_upper,
                num_grids=num_grids,
                capital=args.capital,
                leverage=args.leverage,
                max_holding_bars=args.hours * 60,
                funding_rate=funding_rate,
            )

            try:
                result = run_backtest(cfg, klines_df)
            except Exception as exc:
                logger.warning("  Backtest error: %s — skipping", exc)
                skipped += 1
                continue

            result["candidate_id"] = candidate_id
            result["scan_pnl_pct"] = prev_pnl
            result["scan_score"] = float(row.get("score") or 0)
            result["current_price"] = current_price
            result["price_in_range"] = in_range
            result["klines_fetched"] = len(klines_df)
            results.append(result)

            # Log result
            pnl_change = result["net_pnl_pct"] - prev_pnl
            logger.info(
                "  PnL: %+.2f%% | DD: %.2f%% | RTs: %d | WR: %.1f%% | "
                "TIR: %.1f%% | %s | vs scan: %+.2f%%",
                result["net_pnl_pct"],
                result["max_drawdown_pct"],
                result["round_trips"],
                result["win_rate"],
                result["time_in_range_pct"],
                price_vs_range,
                pnl_change,
            )

            await asyncio.sleep(args.delay)

    finally:
        await client.close()

    if not results:
        logger.warning("No successful backtests.")
        sys.exit(0)

    # ── Save results ────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logger.info("Results saved to: %s (%d rows)", output_path, len(results_df))

    # ── Summary comparison ──────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("CURRENT MARKET BACKTEST SUMMARY")
    logger.info("=" * 70)
    logger.info("  Symbols tested:    %d", len(results))
    logger.info("  Skipped:           %d", skipped)
    logger.info("")

    pnls = [r["net_pnl_pct"] for r in results]
    scan_pnls = [r["scan_pnl_pct"] for r in results]
    in_range_count = sum(1 for r in results if r["price_in_range"])

    logger.info("  %-25s %-15s %-15s", "", "SCAN-TIME", "CURRENT")
    logger.info("  %-25s %-15s %-15s", "-" * 25, "-" * 15, "-" * 15)
    logger.info(
        "  %-25s %+13.2f%%  %+13.2f%%",
        "Mean PnL", sum(scan_pnls) / len(scan_pnls), sum(pnls) / len(pnls),
    )
    logger.info(
        "  %-25s %+13.2f%%  %+13.2f%%",
        "Median PnL",
        sorted(scan_pnls)[len(scan_pnls) // 2],
        sorted(pnls)[len(pnls) // 2],
    )
    logger.info(
        "  %-25s %+13.2f%%  %+13.2f%%",
        "Best PnL", max(scan_pnls), max(pnls),
    )
    logger.info(
        "  %-25s %+13.2f%%  %+13.2f%%",
        "Worst PnL", min(scan_pnls), min(pnls),
    )

    above_hurdle = sum(1 for p in pnls if p >= get_config().barrier.meta_hurdle_pct)
    positive = sum(1 for p in pnls if p > 0)
    negative = sum(1 for p in pnls if p <= 0)
    logger.info("")
    logger.info("  Price in range now:  %d / %d (%.0f%%)",
                in_range_count, len(results), in_range_count / len(results) * 100)
    logger.info("  Positive PnL:        %d / %d (%.0f%%)",
                positive, len(results), positive / len(results) * 100)
    logger.info("  Negative PnL:        %d / %d (%.0f%%)",
                negative, len(results), negative / len(results) * 100)
    logger.info("  Above hurdle (3%%):   %d / %d (%.0f%%)",
                above_hurdle, len(results), above_hurdle / len(results) * 100)
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)
