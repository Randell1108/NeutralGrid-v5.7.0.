"""
Sweep-line replay to reconstruct the open order book over time.

Processes a chronologically sorted event stream per symbol, maintaining
the full open-order state at each event boundary.  Emits snapshots and
price-level records for every unique timestamp.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from neutralgrid.replay.schema import (
    Event,
    BookSnapshot,
    BookLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class _OpenBookState:
    """Mutable state for a single symbol's open order book."""
    open_orders: dict[int, Event] = field(default_factory=dict)
    buy_levels: dict[float, set[int]] = field(default_factory=lambda: defaultdict(set))
    sell_levels: dict[float, set[int]] = field(default_factory=lambda: defaultdict(set))

    def add(self, event: Event) -> None:
        self.open_orders[event.order_id] = event
        levels = self.buy_levels if event.side == "BUY" else self.sell_levels
        levels[event.price].add(event.order_id)

    def remove(self, event: Event) -> None:
        self.open_orders.pop(event.order_id, None)
        levels = self.buy_levels if event.side == "BUY" else self.sell_levels
        price_set = levels.get(event.price)
        if price_set is not None:
            price_set.discard(event.order_id)
            if not price_set:
                del levels[event.price]

    def snapshot(self, time_ms: int) -> BookSnapshot:
        buys = [e for e in self.open_orders.values() if e.side == "BUY"]
        sells = [e for e in self.open_orders.values() if e.side == "SELL"]

        best_bid = max(self.buy_levels.keys()) if self.buy_levels else None
        best_ask = min(self.sell_levels.keys()) if self.sell_levels else None

        return BookSnapshot(
            time_ms=time_ms,
            open_order_count=len(self.open_orders),
            open_buy_count=len(buys),
            open_sell_count=len(sells),
            buy_levels_count=len(self.buy_levels),
            sell_levels_count=len(self.sell_levels),
            best_bid=best_bid,
            best_ask=best_ask,
            total_open_buy_qty=sum(e.orig_qty for e in buys),
            total_open_sell_qty=sum(e.orig_qty for e in sells),
        )

    def levels(self, time_ms: int) -> list[BookLevel]:
        result: list[BookLevel] = []
        for price, oids in sorted(self.buy_levels.items(), reverse=True):
            orders = [self.open_orders[oid] for oid in oids
                      if oid in self.open_orders]
            if orders:
                result.append(BookLevel(
                    time_ms=time_ms,
                    side="BUY",
                    price=price,
                    order_count=len(orders),
                    total_orig_qty=sum(o.orig_qty for o in orders),
                ))
        for price, oids in sorted(self.sell_levels.items()):
            orders = [self.open_orders[oid] for oid in oids
                      if oid in self.open_orders]
            if orders:
                result.append(BookLevel(
                    time_ms=time_ms,
                    side="SELL",
                    price=price,
                    order_count=len(orders),
                    total_orig_qty=sum(o.orig_qty for o in orders),
                ))
        return result


@dataclass
class SweepResult:
    """Full replay output for a single symbol."""
    symbol: str
    events: list[Event]
    snapshots: list[BookSnapshot]
    levels: list[BookLevel]


def replay_symbol(symbol: str, events: list[Event]) -> SweepResult:
    """Run sweep-line replay for a single symbol.

    Parameters
    ----------
    symbol : str
        Symbol name.
    events : list[Event]
        Pre-sorted events for this symbol (must already be sorted by
        ``Event.sort_key``).

    Returns
    -------
    SweepResult
        Events, snapshots, and levels for the symbol.
    """
    state = _OpenBookState()
    snapshots: list[BookSnapshot] = []
    all_levels: list[BookLevel] = []

    # group events by timestamp to emit one snapshot per unique time
    i = 0
    n = len(events)
    while i < n:
        current_ts = events[i].time_ms
        # process all events at this timestamp
        while i < n and events[i].time_ms == current_ts:
            e = events[i]
            if e.event_type == "REMOVE":
                state.remove(e)
            else:
                state.add(e)
            i += 1

        # emit snapshot and levels after all events at this ts
        snapshots.append(state.snapshot(current_ts))
        all_levels.extend(state.levels(current_ts))

    logger.debug(
        "Replay %s: %d events -> %d snapshots, %d level records",
        symbol, len(events), len(snapshots), len(all_levels),
    )
    return SweepResult(
        symbol=symbol,
        events=events,
        snapshots=snapshots,
        levels=all_levels,
    )


def replay_all(
    events_by_symbol: dict[str, list[Event]],
) -> dict[str, SweepResult]:
    """Run sweep-line replay for all symbols.

    Returns a dict of symbol -> SweepResult.
    """
    results: dict[str, SweepResult] = {}
    for symbol in sorted(events_by_symbol):
        results[symbol] = replay_symbol(symbol, events_by_symbol[symbol])

    total_snaps = sum(len(r.snapshots) for r in results.values())
    logger.info(
        "Replay complete: %d symbols, %d total snapshots",
        len(results), total_snaps,
    )
    return results
