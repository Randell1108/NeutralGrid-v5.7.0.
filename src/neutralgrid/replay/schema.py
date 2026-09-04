"""
Typed data models for the offline order-book replay pipeline.

All numeric fields use float (matching repo convention) with explicit
rounding at serialization boundaries.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

OPEN_SENTINEL_MS: int = sys.maxsize


# ── raw rows (parsed from CSV) ──────────────────────────────────────────

@dataclass
class OrderRow:
    """Normalized row from the Binance Order History export."""
    order_id: int
    symbol: str
    side: str        # BUY | SELL
    type: str        # LIMIT | MARKET | …
    price: float
    orig_qty: float
    executed_qty: float
    status: str      # NEW | FILLED | CANCELED
    t_create_ms: int
    t_update_ms: int
    client_order_id: str = ""


@dataclass
class TradeRow:
    """Normalized row from the Binance Trade History export."""
    order_id: int
    symbol: str
    side: str
    price: float
    qty: float
    t_trade_ms: int
    maker: bool | None = None
    fee: float | None = None
    realized_profit: float | None = None


# ── derived lifecycle ────────────────────────────────────────────────────

@dataclass
class OrderLifecycle:
    """Deterministic lifecycle derived from order + trade data."""
    order: OrderRow
    t_open_ms: int
    t_close_ms: int      # OPEN_SENTINEL_MS if still open
    close_source: str     # 'trade' | 'update' | 'open'
    trade_count: int = 0
    filled_qty_from_trades: float = 0.0
    qty_mismatch: bool = False

    @property
    def is_still_open(self) -> bool:
        return self.t_close_ms == OPEN_SENTINEL_MS

    @property
    def symbol(self) -> str:
        return self.order.symbol


# ── event stream ─────────────────────────────────────────────────────────

@dataclass
class Event:
    """ADD or REMOVE event for the sweep-line replay."""
    time_ms: int
    event_type: str     # ADD | REMOVE
    symbol: str
    order_id: int
    side: str
    price: float
    orig_qty: float
    executed_qty: float

    @property
    def sort_key(self) -> tuple[int, int]:
        """REMOVE (0) sorts before ADD (1) at the same timestamp."""
        priority = 0 if self.event_type == "REMOVE" else 1
        return (self.time_ms, priority)


# ── book state snapshots ─────────────────────────────────────────────────

@dataclass
class BookSnapshot:
    """Aggregate open-book state at a single event-boundary timestamp."""
    time_ms: int
    open_order_count: int
    open_buy_count: int
    open_sell_count: int
    buy_levels_count: int
    sell_levels_count: int
    best_bid: float | None
    best_ask: float | None
    total_open_buy_qty: float
    total_open_sell_qty: float


@dataclass
class BookLevel:
    """Single price level in the reconstructed open-book ladder."""
    time_ms: int
    side: str
    price: float
    order_count: int
    total_orig_qty: float


# ── anomaly record ───────────────────────────────────────────────────────

@dataclass
class Anomaly:
    """Data-quality anomaly detected during join_trades."""
    order_id: int
    symbol: str
    anomaly_type: str
    expected: float
    actual: float
    detail: str = ""


# ── overlap detection ────────────────────────────────────────────────────

@dataclass
class BotSession:
    """A single bot session from execution telemetry."""
    symbol: str
    t_start_ms: int
    t_end_ms: int
    strategy_id: str = ""


@dataclass
class OverlapInterval:
    """Time interval where two bots ran on the same symbol."""
    symbol: str
    t_start_ms: int
    t_end_ms: int
