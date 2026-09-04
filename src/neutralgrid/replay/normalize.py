"""
CSV loading, normalization, and 90-day window filtering.

Parses Binance Order History and Trade History exports into typed
OrderRow / TradeRow lists.  Timestamps are converted to UTC epoch
milliseconds; string fields are uppercased; numerics are parsed as float.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from neutralgrid.replay.schema import (
    OrderRow,
    TradeRow,
    BotSession,
    OverlapInterval,
)

logger = logging.getLogger(__name__)

_ORDER_STATUS_RANK = {
    "NEW": 0,
    "PARTIALLY_FILLED": 1,
    "FILLED": 2,
    "CANCELED": 2,
    "CANCELLED": 2,
    "EXPIRED": 2,
    "REJECTED": 2,
}


# ── timestamp helpers ────────────────────────────────────────────────────

def _parse_ts_ms(value: str) -> int:
    """Parse a UTC timestamp string into epoch milliseconds.

    Handles common Binance export formats:
      '2025-01-15 08:30:00'
      '2025-01-15T08:30:00.000Z'
      '2025-01-15 08:30:00+00:00'
    """
    value = str(value).strip()

    # Strip timezone offset suffix (e.g. '+00:00', '-05:00') and treat as UTC
    # Binance exports always use UTC; execution_telemetry uses +00:00 suffix
    cleaned = value
    if len(value) > 6 and value[-6] in ('+', '-') and value[-3] == ':':
        cleaned = value[:-6]
    elif value.endswith('Z'):
        cleaned = value[:-1]

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
    ):
        try:
            dt = datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"Unparseable timestamp: {value!r}")


import re

_NUMERIC_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convert a value to float, returning *default* on failure.

    Handles Binance export quirks like '-0.00847350 USDT' (strips
    currency suffix) and '0E-8' (scientific notation).
    """
    s = str(value).strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    # strip currency suffix (e.g. ' USDT', ' BNB')
    m = _NUMERIC_RE.match(s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return default


def _safe_bool(value: object) -> bool | None:
    """Convert 'TRUE'/'FALSE'/1/0 to bool, None on failure."""
    s = str(value).strip().upper()
    if s in ("TRUE", "1", "YES"):
        return True
    if s in ("FALSE", "0", "NO"):
        return False
    return None


def _order_row_rank(row: OrderRow) -> tuple[int, float, int, int]:
    """Rank order rows so later, more-complete records win."""
    return (
        row.t_update_ms,
        row.executed_qty,
        _ORDER_STATUS_RANK.get(row.status, 1),
        row.t_create_ms,
    )


def _dedupe_orders(rows: list[OrderRow]) -> list[OrderRow]:
    """Collapse duplicate order IDs, keeping the best available record."""
    if not rows:
        return []

    best_by_id: dict[int, OrderRow] = {}
    duplicate_rows = 0
    replacements = 0

    for row in rows:
        existing = best_by_id.get(row.order_id)
        if existing is None:
            best_by_id[row.order_id] = row
            continue

        duplicate_rows += 1
        if _order_row_rank(row) > _order_row_rank(existing):
            best_by_id[row.order_id] = row
            replacements += 1

    deduped = sorted(
        best_by_id.values(),
        key=lambda row: (row.t_create_ms, row.order_id, row.t_update_ms),
    )
    logger.info(
        "Order dedupe: %d -> %d rows (dropped=%d, replaced_with_better=%d)",
        len(rows), len(deduped), duplicate_rows, replacements,
    )
    return deduped


def _trade_row_key(row: TradeRow) -> tuple[object, ...]:
    """Stable exact-duplicate key for normalized trade rows."""
    return (
        row.order_id,
        row.symbol,
        row.side,
        row.price,
        row.qty,
        row.t_trade_ms,
        row.maker,
        row.fee,
        row.realized_profit,
    )


def _dedupe_trades(rows: list[TradeRow]) -> list[TradeRow]:
    """Drop exact duplicate trades while preserving first-seen order."""
    if not rows:
        return []

    seen: set[tuple[object, ...]] = set()
    deduped: list[TradeRow] = []
    duplicate_rows = 0

    for row in rows:
        key = _trade_row_key(row)
        if key in seen:
            duplicate_rows += 1
            continue
        seen.add(key)
        deduped.append(row)

    logger.info(
        "Trade dedupe: %d -> %d rows (dropped=%d)",
        len(rows), len(deduped), duplicate_rows,
    )
    return deduped


# ── order loading ────────────────────────────────────────────────────────

def load_orders(paths: list[Path]) -> list[OrderRow]:
    """Load and normalize order rows from one or more CSV files."""
    rows: list[OrderRow] = []
    for path in paths:
        df = pd.read_csv(path, dtype=str)
        df.columns = df.columns.str.strip()
        logger.info("Loading orders from %s (%d rows)", path.name, len(df))

        for _, r in df.iterrows():
            try:
                row = OrderRow(
                    order_id=int(str(r["Order No"])),
                    symbol=str(r["Symbol"]).strip().upper(),
                    side=str(r["Side"]).strip().upper(),
                    type=str(r.get("Type", "")).strip().upper(),
                    price=_safe_float(r["Price"]),
                    orig_qty=_safe_float(r["Amount"]),
                    executed_qty=_safe_float(r.get("Executed Amount",
                                                    r.get("Filled", 0))),
                    status=str(r["Status"]).strip().upper(),
                    t_create_ms=_parse_ts_ms(str(r["Time(UTC)"])),
                    t_update_ms=_parse_ts_ms(
                        str(r.get("Update Time", r.get("Time(UTC)", "")))
                    ),
                    client_order_id=str(r.get("Client Order Id", "")).strip(),
                )
                rows.append(row)
            except Exception as exc:
                logger.warning("Skipping malformed order row in %s: %s",
                               path.name, exc)
    rows = _dedupe_orders(rows)
    logger.info("Total orders loaded: %d", len(rows))
    return rows


# ── trade loading ────────────────────────────────────────────────────────

def load_trades(paths: list[Path]) -> list[TradeRow]:
    """Load and normalize trade rows from one or more CSV files."""
    rows: list[TradeRow] = []
    for path in paths:
        df = pd.read_csv(path, dtype=str)
        df.columns = df.columns.str.strip()
        logger.info("Loading trades from %s (%d rows)", path.name, len(df))

        for _, r in df.iterrows():
            try:
                row = TradeRow(
                    order_id=int(str(r["Order Id"])),
                    symbol=str(r["Symbol"]).strip().upper(),
                    side=str(r.get("Side", "")).strip().upper(),
                    price=_safe_float(r["Price"]),
                    qty=_safe_float(r["Quantity"]),
                    t_trade_ms=_parse_ts_ms(str(r["Time(UTC)"])),
                    maker=_safe_bool(r.get("Maker")),
                    fee=_safe_float(r.get("Fee")) if "Fee" in r else None,
                    realized_profit=(
                        _safe_float(r.get("Realized Profit"))
                        if "Realized Profit" in r else None
                    ),
                )
                rows.append(row)
            except Exception as exc:
                logger.warning("Skipping malformed trade row in %s: %s",
                               path.name, exc)
    rows = _dedupe_trades(rows)
    logger.info("Total trades loaded: %d", len(rows))
    return rows


# ── 90-day window filter ─────────────────────────────────────────────────

def apply_window_filter(
    orders: list[OrderRow],
    trades: list[TradeRow],
    window_days: int = 90,
) -> tuple[list[OrderRow], list[TradeRow], int, int]:
    """Filter orders and trades to the last *window_days*.

    The window end is the maximum timestamp observed across both datasets.
    Returns (filtered_orders, filtered_trades, t_start_ms, t_end_ms).
    """
    if not orders and not trades:
        return [], [], 0, 0

    all_ts: list[int] = []
    for o in orders:
        all_ts.extend([o.t_create_ms, o.t_update_ms])
    for t in trades:
        all_ts.append(t.t_trade_ms)

    t_end_ms = max(all_ts)
    t_start_ms = t_end_ms - (window_days * 24 * 60 * 60 * 1000)

    filtered_orders = [
        o for o in orders
        if o.t_create_ms >= t_start_ms or o.t_update_ms >= t_start_ms
    ]
    filtered_trades = [
        t for t in trades
        if t.t_trade_ms >= t_start_ms
    ]

    logger.info(
        "Window filter (%d days): %s to %s",
        window_days,
        datetime.fromtimestamp(t_start_ms / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M"),
        datetime.fromtimestamp(t_end_ms / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M"),
    )
    logger.info(
        "  Orders: %d -> %d  |  Trades: %d -> %d",
        len(orders), len(filtered_orders),
        len(trades), len(filtered_trades),
    )
    return filtered_orders, filtered_trades, t_start_ms, t_end_ms


# ── overlap detection ────────────────────────────────────────────────────

def load_sessions(path: Path) -> list[BotSession]:
    """Load bot sessions from execution telemetry CSV.

    Expects columns: symbol, start_time_utc, end_time_utc
    (and optionally strategy_id).
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    sessions: list[BotSession] = []
    for _, r in df.iterrows():
        try:
            sessions.append(BotSession(
                symbol=str(r["symbol"]).strip().upper(),
                t_start_ms=_parse_ts_ms(str(r["start_time_utc"])),
                t_end_ms=_parse_ts_ms(str(r["end_time_utc"])),
                strategy_id=str(r.get("strategy_id", "")).strip(),
            ))
        except Exception as exc:
            logger.warning("Skipping malformed session row: %s", exc)
    logger.info("Loaded %d bot sessions from %s", len(sessions), path.name)
    return sessions


def detect_overlaps(sessions: list[BotSession]) -> list[OverlapInterval]:
    """Detect temporal overlaps per symbol.

    For each symbol with >1 session, sort by start time and check for
    pairwise overlap.  Returns intervals where two bots ran simultaneously.
    """
    from collections import defaultdict

    by_symbol: dict[str, list[BotSession]] = defaultdict(list)
    for s in sessions:
        by_symbol[s.symbol].append(s)

    overlaps: list[OverlapInterval] = []
    for symbol, syms in by_symbol.items():
        if len(syms) < 2:
            continue
        syms.sort(key=lambda s: s.t_start_ms)
        for i in range(len(syms) - 1):
            a = syms[i]
            for j in range(i + 1, len(syms)):
                b = syms[j]
                # overlap exists if a ends after b starts
                if a.t_end_ms > b.t_start_ms:
                    overlap_start = b.t_start_ms
                    overlap_end = min(a.t_end_ms, b.t_end_ms)
                    overlaps.append(OverlapInterval(
                        symbol=symbol,
                        t_start_ms=overlap_start,
                        t_end_ms=overlap_end,
                    ))
                    logger.warning(
                        "Overlap detected: %s from %s to %s",
                        symbol,
                        datetime.fromtimestamp(
                            overlap_start / 1000, tz=timezone.utc
                        ).strftime("%Y-%m-%d %H:%M"),
                        datetime.fromtimestamp(
                            overlap_end / 1000, tz=timezone.utc
                        ).strftime("%Y-%m-%d %H:%M"),
                    )
    return overlaps


def filter_overlapping_orders(
    orders: list[OrderRow],
    overlaps: list[OverlapInterval],
) -> tuple[list[OrderRow], int]:
    """Remove orders whose lifecycle intersects any overlap interval.

    An order is discarded if its [t_create_ms, t_update_ms] range
    intersects any overlap interval for the same symbol.

    Returns (filtered_orders, num_discarded).
    """
    from collections import defaultdict

    overlap_by_symbol: dict[str, list[OverlapInterval]] = defaultdict(list)
    for ov in overlaps:
        overlap_by_symbol[ov.symbol].append(ov)

    kept: list[OrderRow] = []
    discarded = 0
    for o in orders:
        if o.symbol not in overlap_by_symbol:
            kept.append(o)
            continue
        hit = False
        for ov in overlap_by_symbol[o.symbol]:
            if o.t_create_ms < ov.t_end_ms and o.t_update_ms >= ov.t_start_ms:
                hit = True
                break
        if hit:
            discarded += 1
        else:
            kept.append(o)

    if discarded:
        logger.info("Discarded %d orders in overlap intervals", discarded)
    return kept, discarded
