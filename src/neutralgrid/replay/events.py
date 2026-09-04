"""
Convert OrderLifecycles into a deterministic, sorted event stream.

Events are sorted by (time_ms, priority) where REMOVE (0) always
precedes ADD (1) at the same timestamp.  This avoids an artificial
moment where a removed order still appears open.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from neutralgrid.replay.schema import Event, OrderLifecycle, OPEN_SENTINEL_MS

logger = logging.getLogger(__name__)


def build_events(
    lifecycles: list[OrderLifecycle],
    t_end_ms: int | None = None,
) -> list[Event]:
    """Generate ADD/REMOVE events from lifecycles.

    Parameters
    ----------
    lifecycles : list[OrderLifecycle]
        All order lifecycles to convert.
    t_end_ms : int, optional
        If provided, still-open orders (OPEN_SENTINEL_MS) get a REMOVE
        capped at this timestamp.  Typically the replay window end.

    Returns
    -------
    list[Event]
        Chronologically sorted event stream.
    """
    events: list[Event] = []

    for lc in lifecycles:
        o = lc.order

        # ADD event at open time
        events.append(Event(
            time_ms=lc.t_open_ms,
            event_type="ADD",
            symbol=o.symbol,
            order_id=o.order_id,
            side=o.side,
            price=o.price,
            orig_qty=o.orig_qty,
            executed_qty=o.executed_qty,
        ))

        # REMOVE event at close time (if not still-open, or capped)
        close_ms = lc.t_close_ms
        if close_ms == OPEN_SENTINEL_MS:
            if t_end_ms is not None:
                close_ms = t_end_ms
            else:
                # no cap: skip REMOVE for still-open orders
                continue

        events.append(Event(
            time_ms=close_ms,
            event_type="REMOVE",
            symbol=o.symbol,
            order_id=o.order_id,
            side=o.side,
            price=o.price,
            orig_qty=o.orig_qty,
            executed_qty=o.executed_qty,
        ))

    # deterministic sort: (time_ms, priority) where REMOVE < ADD
    events.sort(key=lambda e: e.sort_key)

    logger.info(
        "Generated %d events from %d lifecycles",
        len(events), len(lifecycles),
    )
    return events


def split_events_by_symbol(
    events: list[Event],
) -> dict[str, list[Event]]:
    """Partition an event stream by symbol.

    Events within each symbol list retain their sorted order.
    """
    by_symbol: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        by_symbol[e.symbol].append(e)
    return dict(by_symbol)
