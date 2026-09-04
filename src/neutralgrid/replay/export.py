"""
Export replay results to CSV files (events, snapshots, levels, trades).

Produces per-symbol output directories with deterministic, auditable
records that can reconstruct the open order book at any point in time.
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from neutralgrid.replay.schema import (
    Anomaly,
    BookLevel,
    BookSnapshot,
    Event,
    TradeRow,
)
from neutralgrid.replay.sweep import SweepResult

logger = logging.getLogger(__name__)


_MAX_SAFE_TS_S = 32503680000  # 3000-01-01 00:00:00 UTC


def _ms_to_utc_str(ms: int) -> str:
    """Format epoch ms as ISO-8601 UTC string.

    Timestamps beyond year 3000 (e.g. OPEN_SENTINEL_MS) are capped to
    avoid platform-specific overflow on Windows.
    """
    ts_s = ms / 1000
    if ts_s > _MAX_SAFE_TS_S:
        return "9999-12-31 23:59:59"
    return datetime.fromtimestamp(
        ts_s, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── per-symbol exports ───────────────────────────────────────────────────

def export_events(events: list[Event], out_dir: Path) -> Path:
    """Write events.csv — canonical lossless event record."""
    path = out_dir / "events.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "time_ms", "time_utc", "event_type", "order_id",
            "symbol", "side", "price", "orig_qty", "executed_qty",
        ])
        for e in events:
            w.writerow([
                e.time_ms,
                _ms_to_utc_str(e.time_ms),
                e.event_type,
                e.order_id,
                e.symbol,
                e.side,
                f"{e.price:.8f}",
                f"{e.orig_qty:.8f}",
                f"{e.executed_qty:.8f}",
            ])
    return path


def export_snapshots(snapshots: list[BookSnapshot], out_dir: Path) -> Path:
    """Write snapshots.csv — aggregate book state at each event boundary."""
    path = out_dir / "snapshots.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "time_ms", "time_utc", "open_order_count",
            "open_buy_count", "open_sell_count",
            "buy_levels_count", "sell_levels_count",
            "best_bid", "best_ask",
            "total_open_buy_qty", "total_open_sell_qty",
        ])
        for s in snapshots:
            w.writerow([
                s.time_ms,
                _ms_to_utc_str(s.time_ms),
                s.open_order_count,
                s.open_buy_count,
                s.open_sell_count,
                s.buy_levels_count,
                s.sell_levels_count,
                f"{s.best_bid:.8f}" if s.best_bid is not None else "",
                f"{s.best_ask:.8f}" if s.best_ask is not None else "",
                f"{s.total_open_buy_qty:.8f}",
                f"{s.total_open_sell_qty:.8f}",
            ])
    return path


def export_levels(levels: list[BookLevel], out_dir: Path) -> Path:
    """Write levels.csv — per-price-level ladder at each snapshot time."""
    path = out_dir / "levels.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "time_ms", "time_utc", "side", "price",
            "order_count", "total_orig_qty",
        ])
        for lv in levels:
            w.writerow([
                lv.time_ms,
                _ms_to_utc_str(lv.time_ms),
                lv.side,
                f"{lv.price:.8f}",
                lv.order_count,
                f"{lv.total_orig_qty:.8f}",
            ])
    return path


def export_trades(
    trades: list[TradeRow],
    out_dir: Path,
) -> Path:
    """Write trades.csv — trade stream for linking fills to book shape."""
    path = out_dir / "trades.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "t_trade_ms", "time_utc", "order_id", "side",
            "price", "qty", "maker", "fee", "realized_profit",
        ])
        for t in trades:
            w.writerow([
                t.t_trade_ms,
                _ms_to_utc_str(t.t_trade_ms),
                t.order_id,
                t.side,
                f"{t.price:.8f}",
                f"{t.qty:.8f}",
                str(t.maker).lower() if t.maker is not None else "",
                f"{t.fee:.8f}" if t.fee is not None else "",
                (f"{t.realized_profit:.8f}"
                 if t.realized_profit is not None else ""),
            ])
    return path


# ── anomalies export ─────────────────────────────────────────────────────

def export_anomalies(anomalies: list[Anomaly], out_dir: Path) -> Path:
    """Write anomalies.csv — data-quality flags (never auto-corrected)."""
    path = out_dir / "anomalies.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "order_id", "symbol", "anomaly_type",
            "expected", "actual", "detail",
        ])
        for a in anomalies:
            w.writerow([
                a.order_id,
                a.symbol,
                a.anomaly_type,
                f"{a.expected:.8f}",
                f"{a.actual:.8f}",
                a.detail,
            ])
    return path


# ── full export orchestrator ─────────────────────────────────────────────

def export_all(
    results: dict[str, SweepResult],
    trades: list[TradeRow],
    anomalies: list[Anomaly],
    output_dir: Path,
) -> dict[str, Path]:
    """Export all replay results to per-symbol subdirectories.

    Directory structure:
        output_dir/
            anomalies.csv
            <SYMBOL>/
                events.csv
                snapshots.csv
                levels.csv
                trades.csv

    Returns a dict of symbol -> output directory path.
    """
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)

    # group trades by symbol
    trades_by_symbol: dict[str, list[TradeRow]] = defaultdict(list)
    for t in trades:
        trades_by_symbol[t.symbol].append(t)

    # sort trades within each symbol
    for sym_trades in trades_by_symbol.values():
        sym_trades.sort(key=lambda t: t.t_trade_ms)

    symbol_dirs: dict[str, Path] = {}

    for symbol, result in sorted(results.items()):
        sym_dir = _ensure_dir(output_dir / symbol)
        symbol_dirs[symbol] = sym_dir

        export_events(result.events, sym_dir)
        export_snapshots(result.snapshots, sym_dir)
        export_levels(result.levels, sym_dir)

        sym_trades = trades_by_symbol.get(symbol, [])
        if sym_trades:
            export_trades(sym_trades, sym_dir)

        snap_count = len(result.snapshots)
        logger.info(
            "  Exported %s: %d events, %d snapshots, %d levels, %d trades",
            symbol, len(result.events), snap_count,
            len(result.levels), len(sym_trades),
        )

    # global anomalies
    if anomalies:
        export_anomalies(anomalies, output_dir)
        logger.info("  Exported %d anomalies", len(anomalies))

    logger.info(
        "Export complete: %d symbols -> %s",
        len(symbol_dirs), output_dir,
    )
    return symbol_dirs
