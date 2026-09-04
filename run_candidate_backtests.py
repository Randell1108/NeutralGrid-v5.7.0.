#!/usr/bin/env python3
"""
Batch realistic backtest of all deployment_ready candidates with score > 45.

Loads CSVs from results/, filters to backtestable candidates, fetches 6h
of 1m klines starting from each candidate's scan timestamp, and runs the
realistic backtester via the unified runner.

Outputs a report of PnL by symbol and max holding time.
"""
from __future__ import annotations

import asyncio
import glob
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd

# Ensure backtest/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from backtest.btk_unified_runner import run_backtest, build_training_config
from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.core.config import get_config

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_scan_timestamp(candidate_id: str | None, scan_file: str | None = None) -> datetime | None:
    """Extract UTC datetime from candidate_id or fall back to scan_file name.

    Tries candidate_id first (e.g. ``BTCUSDT_20260224_100100``).
    If that fails or is empty, extracts the timestamp from the CSV filename
    (e.g. ``deployment_ready_20260217_170412.csv``).
    """
    # Try candidate_id first
    if candidate_id and str(candidate_id) not in ("", "nan"):
        parts = str(candidate_id).split("_", 1)
        if len(parts) >= 2:
            ts_str = parts[1]
            for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d%H%M%S"):
                try:
                    return datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

    # Fall back to scan_file name: deployment_ready_YYYYMMDD_HHMMSS.csv
    if scan_file:
        base = str(scan_file).replace(".csv", "")
        # Strip "deployment_ready_" prefix to get "YYYYMMDD_HHMMSS"
        prefix = "deployment_ready_"
        if base.startswith(prefix):
            ts_str = base[len(prefix):]
            for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d%H%M%S"):
                try:
                    return datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

    return None


def load_and_filter_candidates(results_dir: str, min_score: float = 45.0) -> pd.DataFrame:
    """Load all deployment_ready CSVs and filter to backtestable candidates."""
    files = sorted(glob.glob(os.path.join(results_dir, "deployment_ready_*.csv")))
    print(f"Found {len(files)} deployment_ready CSVs")

    all_candidates = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "score" not in df.columns or "symbol" not in df.columns:
                continue
            mask = df["score"] > min_score
            if "grid_is_valid" in df.columns:
                mask = mask & (df["grid_is_valid"] == True)  # noqa: E712
            for col in ("grid_lower", "grid_upper", "num_grids"):
                if col in df.columns:
                    mask = mask & df[col].notna()
            valid = df[mask].copy()
            if len(valid) > 0:
                valid["scan_file"] = os.path.basename(f)
                all_candidates.append(valid)
        except Exception:
            pass

    if not all_candidates:
        print("No valid candidates found!")
        return pd.DataFrame()

    combined = pd.concat(all_candidates, ignore_index=True)
    # Dedup: keep highest score per symbol
    deduped = combined.sort_values("score", ascending=False).drop_duplicates(
        subset=["symbol"], keep="first"
    )
    print(f"Total valid: {len(combined)}, unique symbols: {len(deduped)}")
    return deduped.reset_index(drop=True)


async def fetch_klines(client: BinanceClient, symbol: str, start_time: datetime, hours: int = 6) -> pd.DataFrame:
    """Fetch hours of 1m klines starting at start_time."""
    start_ms = int(start_time.timestamp() * 1000)
    limit = min(hours * 60, 1500)
    klines = await client.get_klines(
        symbol=symbol, interval="1m", limit=limit, start_time=start_ms,
    )
    if not klines:
        return pd.DataFrame()
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


async def run_all_backtests(candidates: pd.DataFrame) -> list[dict]:
    """Run backtests for all candidates."""
    client = BinanceClient()
    results = []
    total = len(candidates)

    for idx, (_, row) in enumerate(candidates.iterrows()):
        symbol = str(row["symbol"])
        candidate_id = str(row.get("candidate_id", ""))
        score = float(row["score"])
        grid_lower = float(row["grid_lower"])
        grid_upper = float(row["grid_upper"])
        num_grids = int(float(row["num_grids"]))
        leverage = int(row.get("leverage", 10)) if pd.notna(row.get("leverage")) else 10
        cap_frac = 1.0
        if pd.notna(row.get("capital_fraction")):
            try:
                cap_frac = float(row.get("capital_fraction"))
            except (TypeError, ValueError):
                cap_frac = 1.0
        cap_frac = max(0.0, min(1.0, cap_frac))
        vol_proxy_pct = None
        if pd.notna(row.get("range_size_pct")):
            try:
                rp = float(row.get("range_size_pct"))
                if rp > 0:
                    vol_proxy_pct = rp
            except (TypeError, ValueError):
                vol_proxy_pct = None

        # Get funding rate from candidate
        # Step 13 (Plan v6.0): Preserve funding rate sign — negative rates
        # mean shorts pay longs, which is material for grid bot PnL.
        funding_rate = 0.0001
        fr = row.get("funding_rate")
        if fr is not None and pd.notna(fr):
            fr_val = float(fr)
            funding_rate = fr_val if fr_val != 0 else 0.0001

        # Parse scan timestamp (candidate_id first, fall back to CSV filename)
        scan_file = str(row.get("scan_file", ""))
        scan_ts = parse_scan_timestamp(candidate_id, scan_file)
        if scan_ts is None:
            print(f"  [{idx+1}/{total}] {symbol}: SKIP (no parseable timestamp from id={candidate_id!r} or file={scan_file!r})")
            continue

        print(f"  [{idx+1}/{total}] {symbol} (score={score:.0f}, lev={leverage}x, grids={num_grids}) ...", end=" ", flush=True)

        try:
            klines_df = await fetch_klines(client, symbol, scan_ts, hours=6)
            if klines_df.empty or len(klines_df) < 10:
                print(f"SKIP (only {len(klines_df)} bars)")
                continue

            cfg = build_training_config(
                symbol=symbol,
                lower=grid_lower,
                upper=grid_upper,
                num_grids=num_grids,
                capital=float(get_config().grid.capital),
                capital_fraction=cap_frac,
                volatility_target_pct=float(get_config().position_sizing.volatility_target_pct),
                volatility_proxy_pct=vol_proxy_pct,
                leverage=leverage,
                funding_rate=funding_rate,
                funding_mode="continuous",
            )
            result = run_backtest(cfg, klines_df)

            result["candidate_id"] = candidate_id
            result["score"] = score
            result["scan_file"] = row.get("scan_file", "")
            result["scan_timestamp"] = scan_ts.isoformat()
            result["klines_fetched"] = len(klines_df)
            results.append(result)

            status = "LIQUIDATED" if result.get("liquidated") else ("WIN" if result["net_pnl"] > 0 else "LOSS")
            print(f"{status} | PnL={result['net_pnl_pct']:+.2f}% | equity=${result['final_equity']:.2f} | trades={result['round_trips']} | maxhold={result['max_hold_bars']}bars")

        except Exception as e:
            print(f"ERROR: {e}")

        # Small delay to respect rate limits
        await asyncio.sleep(0.3)

    await client.close()
    return results


def generate_report(results: list[dict]) -> str:
    """Generate a formatted report from backtest results."""
    if not results:
        return "No results to report."

    df = pd.DataFrame(results)
    total = len(df)

    # Basic stats
    winners = df[df["net_pnl"] > 0]
    losers = df[df["net_pnl"] <= 0]
    liquidated = df[df["liquidated"] == True]  # noqa: E712

    lines = []
    lines.append("=" * 100)
    lines.append("REALISTIC BACKTEST REPORT — ALL CANDIDATES (score > 45, valid grid)")
    lines.append("=" * 100)
    lines.append(f"Engine: {df.iloc[0].get('engine_version', 'N/A')} | Funding: continuous | Fee mode: maker")
    lines.append(f"Total candidates backtested: {total}")
    lines.append(f"Winners (PnL > 0):  {len(winners)} ({len(winners)/total*100:.1f}%)")
    lines.append(f"Losers (PnL <= 0):  {len(losers)} ({len(losers)/total*100:.1f}%)")
    lines.append(f"Liquidated:         {len(liquidated)} ({len(liquidated)/total*100:.1f}%)")
    lines.append("")

    # PnL distribution
    lines.append("--- PnL Distribution ---")
    lines.append(f"Mean PnL%:    {df['net_pnl_pct'].mean():+.2f}%")
    lines.append(f"Median PnL%:  {df['net_pnl_pct'].median():+.2f}%")
    lines.append(f"Best PnL%:    {df['net_pnl_pct'].max():+.2f}%")
    lines.append(f"Worst PnL%:   {df['net_pnl_pct'].min():+.2f}%")
    lines.append(f"Std PnL%:     {df['net_pnl_pct'].std():.2f}%")
    lines.append("")

    # High PnL symbols (> 3% net PnL)
    high_pnl = df[df["net_pnl_pct"] > 3.0].sort_values("net_pnl_pct", ascending=False)
    lines.append(f"--- HIGH PnL SYMBOLS (net_pnl > 3%) --- [{len(high_pnl)} symbols]")
    if len(high_pnl) > 0:
        lines.append(f"{'Symbol':<18} {'Score':>6} {'PnL%':>8} {'Equity':>10} {'Trades':>7} {'MaxHold':>9} {'AvgHold':>9} {'MaxDD%':>7} {'Sharpe':>7} {'Liq':>4}")
        lines.append("-" * 100)
        for _, r in high_pnl.iterrows():
            lines.append(
                f"{r['symbol']:<18} {r['score']:>6.0f} {r['net_pnl_pct']:>+7.2f}% "
                f"${r['final_equity']:>8.2f} {r['round_trips']:>7} "
                f"{r['max_hold_bars']:>6}bar {r['avg_hold_hours']:>6.1f}h "
                f"{r['max_drawdown_pct']:>6.2f}% {float(r['sharpe_ratio']):>7.1f} "
                f"{'Y' if r.get('liquidated') else 'N':>4}"
            )
    else:
        lines.append("  (none)")
    lines.append("")

    # Medium PnL symbols (0% to 3%)
    med_pnl = df[(df["net_pnl_pct"] > 0) & (df["net_pnl_pct"] <= 3.0)].sort_values("net_pnl_pct", ascending=False)
    lines.append(f"--- MODERATE PnL SYMBOLS (0% < PnL <= 3%) --- [{len(med_pnl)} symbols]")
    if len(med_pnl) > 0:
        lines.append(f"{'Symbol':<18} {'Score':>6} {'PnL%':>8} {'Equity':>10} {'Trades':>7} {'MaxHold':>9} {'AvgHold':>9} {'MaxDD%':>7}")
        lines.append("-" * 100)
        for _, r in med_pnl.iterrows():
            lines.append(
                f"{r['symbol']:<18} {r['score']:>6.0f} {r['net_pnl_pct']:>+7.2f}% "
                f"${r['final_equity']:>8.2f} {r['round_trips']:>7} "
                f"{r['max_hold_bars']:>6}bar {r['avg_hold_hours']:>6.1f}h "
                f"{r['max_drawdown_pct']:>6.2f}%"
            )
    else:
        lines.append("  (none)")
    lines.append("")

    # Negative PnL
    neg_pnl = df[df["net_pnl_pct"] <= 0].sort_values("net_pnl_pct", ascending=True)
    lines.append(f"--- NEGATIVE PnL SYMBOLS --- [{len(neg_pnl)} symbols]")
    if len(neg_pnl) > 0:
        lines.append(f"{'Symbol':<18} {'Score':>6} {'PnL%':>8} {'Equity':>10} {'Trades':>7} {'MaxHold':>9} {'MaxDD%':>7} {'Liq':>4}")
        lines.append("-" * 100)
        for _, r in neg_pnl.iterrows():
            lines.append(
                f"{r['symbol']:<18} {r['score']:>6.0f} {r['net_pnl_pct']:>+7.2f}% "
                f"${r['final_equity']:>8.2f} {r['round_trips']:>7} "
                f"{r['max_hold_bars']:>6}bar "
                f"{r['max_drawdown_pct']:>6.2f}% "
                f"{'Y' if r.get('liquidated') else 'N':>4}"
            )
    lines.append("")

    # Max holding time analysis for high-PnL symbols
    lines.append("--- MAX HOLDING TIME ANALYSIS (High PnL symbols) ---")
    if len(high_pnl) > 0:
        lines.append(f"{'Symbol':<18} {'PnL%':>8} {'MaxHold':>10} {'MaxHoldH':>10} {'AvgHold':>10} {'StalePct':>9}")
        lines.append("-" * 70)
        for _, r in high_pnl.iterrows():
            max_h = r["max_hold_bars"] / 60
            lines.append(
                f"{r['symbol']:<18} {r['net_pnl_pct']:>+7.2f}% "
                f"{r['max_hold_bars']:>7}bar "
                f"{max_h:>8.1f}h "
                f"{r['avg_hold_hours']:>8.1f}h "
                f"{r['stale_close_pct']:>7.1f}%"
            )
        lines.append("")
        avg_max_hold = high_pnl["max_hold_bars"].mean()
        lines.append(f"Avg max hold (high PnL): {avg_max_hold:.0f} bars ({avg_max_hold/60:.1f}h)")
    lines.append("")

    # Summary
    lines.append("=" * 100)
    lines.append("SUMMARY")
    lines.append("=" * 100)
    lines.append(f"Total backtested:     {total}")
    lines.append(f"Win rate:             {len(winners)/total*100:.1f}%")
    lines.append(f"High PnL (>3%):       {len(high_pnl)} symbols")
    lines.append(f"Liquidated:           {len(liquidated)} symbols")
    lines.append(f"Avg PnL%:             {df['net_pnl_pct'].mean():+.2f}%")
    lines.append(f"Avg max DD%:          {df['max_drawdown_pct'].mean():.2f}%")
    if len(winners) > 0:
        lines.append(f"Avg winner PnL%:      {winners['net_pnl_pct'].mean():+.2f}%")
        lines.append(f"Avg winner max hold:  {winners['max_hold_bars'].mean():.0f} bars ({winners['max_hold_bars'].mean()/60:.1f}h)")
    if len(losers) > 0:
        lines.append(f"Avg loser PnL%:       {losers['net_pnl_pct'].mean():+.2f}%")
    lines.append("=" * 100)

    return "\n".join(lines)


async def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    print("Loading candidates from results/ ...")
    candidates = load_and_filter_candidates(results_dir, min_score=45.0)

    if candidates.empty:
        print("No candidates to backtest.")
        return

    print(f"\nRunning realistic backtests on {len(candidates)} candidates...")
    print("(6h horizon, continuous funding, maker close fees, 2-bar delay, 360-bar max hold)\n")

    results = await run_all_backtests(candidates)

    if results:
        report = generate_report(results)
        print("\n")
        print(report)

        # Save results
        out_dir = os.path.join(os.path.dirname(__file__), "data", "backtest_candidates")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(out_dir, f"backtest_results_{ts}.csv")
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

        report_path = os.path.join(out_dir, f"backtest_report_{ts}.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Report saved to:  {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
