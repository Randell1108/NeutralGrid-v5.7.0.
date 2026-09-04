"""
Join trades to orders and compute deterministic close times.

For each order, the close time is determined by:
  - FILLED + trades exist  -> max(t_trade_ms) for that order_id
  - FILLED / CANCELED      -> t_update_ms
  - NEW (still open)       -> OPEN_SENTINEL_MS (capped at replay end later)

Quantity mismatches between order.executed_qty and sum(trade.qty) are
flagged as anomalies but never "fixed".
"""
from __future__ import annotations

import logging
from collections import defaultdict

from neutralgrid.replay.schema import (
    OrderRow,
    TradeRow,
    OrderLifecycle,
    Anomaly,
    OPEN_SENTINEL_MS,
)

logger = logging.getLogger(__name__)

_QTY_TOLERANCE = 1e-8


def _aggregate_trades(
    trades: list[TradeRow],
) -> tuple[dict[int, int], dict[int, int], dict[int, float]]:
    """Aggregate trade data per order_id.

    Returns:
        (last_trade_time, trade_count, filled_qty) dicts keyed by order_id.
    """
    last_trade_time: dict[int, int] = {}
    trade_count: dict[int, int] = defaultdict(int)
    filled_qty: dict[int, float] = defaultdict(float)

    for t in trades:
        oid = t.order_id
        trade_count[oid] += 1
        filled_qty[oid] += t.qty
        if oid not in last_trade_time or t.t_trade_ms > last_trade_time[oid]:
            last_trade_time[oid] = t.t_trade_ms

    return dict(last_trade_time), dict(trade_count), dict(filled_qty)


def build_lifecycles(
    orders: list[OrderRow],
    trades: list[TradeRow],
) -> tuple[list[OrderLifecycle], list[Anomaly]]:
    """Build deterministic lifecycles for all orders.

    Returns (lifecycles, anomalies).
    """
    last_trade_time, trade_counts, trade_filled = _aggregate_trades(trades)

    lifecycles: list[OrderLifecycle] = []
    anomalies: list[Anomaly] = []

    for o in orders:
        t_open = o.t_create_ms
        tc = trade_counts.get(o.order_id, 0)
        tf = trade_filled.get(o.order_id, 0.0)
        ltt = last_trade_time.get(o.order_id)

        # determine close time
        if o.status == "NEW":
            t_close = OPEN_SENTINEL_MS
            source = "open"
        elif o.status == "FILLED" and ltt is not None:
            t_close = ltt
            source = "trade"
        else:
            # FILLED without trades, or CANCELED
            t_close = o.t_update_ms
            source = "update"

        # qty mismatch check (only when trades exist)
        mismatch = False
        if tc > 0 and abs(o.executed_qty - tf) > _QTY_TOLERANCE:
            mismatch = True
            anomalies.append(Anomaly(
                order_id=o.order_id,
                symbol=o.symbol,
                anomaly_type="qty_mismatch",
                expected=o.executed_qty,
                actual=tf,
                detail=(
                    f"order.executed_qty={o.executed_qty:.8f} vs "
                    f"sum(trade.qty)={tf:.8f}"
                ),
            ))

        lifecycles.append(OrderLifecycle(
            order=o,
            t_open_ms=t_open,
            t_close_ms=t_close,
            close_source=source,
            trade_count=tc,
            filled_qty_from_trades=tf,
            qty_mismatch=mismatch,
        ))

    # summary logging
    still_open = sum(1 for lc in lifecycles if lc.is_still_open)
    by_source = defaultdict(int)
    for lc in lifecycles:
        by_source[lc.close_source] += 1

    logger.info(
        "Built %d lifecycles: trade-closed=%d, update-closed=%d, "
        "still-open=%d, anomalies=%d",
        len(lifecycles),
        by_source.get("trade", 0),
        by_source.get("update", 0),
        still_open,
        len(anomalies),
    )
    return lifecycles, anomalies
