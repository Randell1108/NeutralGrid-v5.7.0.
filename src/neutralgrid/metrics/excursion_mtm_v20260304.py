"""Mark-to-market excursion metrics (MAE/MFE) utilities.

This module implements a single-pass O(n) event engine that computes:
- Maximum Adverse Excursion (MAE): max peak-to-trough equity drop
- Maximum Favorable Excursion (MFE): max trough-to-peak equity rise

All excursion metrics are computed on a mark-to-market (MTM) equity path:
    equity_t = cash_pnl_t + unrealized_t(mark_t)
where:
    cash_pnl_t = realized_pnl - commissions + funding
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MtmExcursionMetrics:
    """Result of MTM excursion computation."""

    mae_usdt: float
    mfe_usdt: float
    max_equity_usdt: float
    min_equity_usdt: float
    equity_points: int
    mark_points_used: int
    used_mark_prices: bool


@dataclass
class _HedgePositionState:
    """Hedge-mode state: independent long and short books."""

    long_qty: float = 0.0
    long_entry: float = 0.0
    short_qty: float = 0.0  # stored as absolute quantity
    short_entry: float = 0.0


@dataclass
class _NetPositionState:
    """One-way mode state: single signed position."""

    qty: float = 0.0  # positive=long, negative=short
    entry: float = 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion with finite-value guard."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not isfinite(parsed):
        return default
    return parsed


def _safe_int(value: Any, default: int = 0) -> int:
    """Best-effort int conversion."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _update_abs_position(qty: float, entry: float, delta: float, trade_price: float) -> tuple[float, float]:
    """Update an absolute-quantity position (used by long/short hedge books).

    Parameters
    ----------
    qty : float
        Current absolute quantity (>= 0).
    entry : float
        Current average entry price.
    delta : float
        Signed quantity change; positive adds, negative reduces.
    trade_price : float
        Fill price.
    """

    if delta > 0:
        total = qty + delta
        if qty <= 0:
            return total, trade_price
        new_entry = (qty * entry + delta * trade_price) / total
        return total, new_entry

    if delta < 0:
        remaining = qty + delta
        if remaining <= 0:
            return 0.0, 0.0
        return remaining, entry

    return qty, entry


def _update_net_position(qty: float, entry: float, delta: float, trade_price: float) -> tuple[float, float]:
    """Update one-way (net) position state using average-price accounting."""

    if delta == 0:
        return qty, entry

    if qty == 0:
        return delta, trade_price

    same_side = (qty > 0 and delta > 0) or (qty < 0 and delta < 0)
    if same_side:
        total = qty + delta
        new_entry = (
            abs(qty) * entry + abs(delta) * trade_price
        ) / (abs(qty) + abs(delta))
        return total, new_entry

    # Opposite side: reduce, close, or flip.
    if abs(delta) < abs(qty):
        return qty + delta, entry
    if abs(delta) == abs(qty):
        return 0.0, 0.0
    return qty + delta, trade_price


def calculate_mtm_excursions(
    trades: Sequence[Mapping[str, Any]],
    funding_fees: Sequence[Mapping[str, Any]] | None = None,
    mark_klines: Sequence[Sequence[Any]] | None = None,
) -> MtmExcursionMetrics:
    """Compute MTM MAE and MFE from trade, funding, and mark-price events.

    Notes
    -----
    - Trades contribute realized PnL and commission cash flows.
    - Funding records contribute funding cash flows.
    - Mark klines contribute MTM price updates for unrealized PnL.
    - If no mark series is available, the first observed trade price is used as
      an initialization anchor, then unrealized PnL evolves only when new mark
      points arrive (or remains static when none exist).
    """

    if not trades and not funding_fees and not mark_klines:
        return MtmExcursionMetrics(
            mae_usdt=0.0,
            mfe_usdt=0.0,
            max_equity_usdt=0.0,
            min_equity_usdt=0.0,
            equity_points=0,
            mark_points_used=0,
            used_mark_prices=False,
        )

    events: list[tuple[int, int, str, Any]] = []

    # priority: mark (0) -> funding (1) -> trade (2)
    if mark_klines:
        for row in mark_klines:
            if len(row) < 7:
                continue
            ts_ms = _safe_int(row[6], 0)
            price = _safe_float(row[4], 0.0)
            if ts_ms <= 0 or price <= 0:
                continue
            events.append((ts_ms, 0, "mark", price))

    if funding_fees:
        for record in funding_fees:
            ts_ms = _safe_int(record.get("time", 0), 0)
            income = _safe_float(record.get("income", 0.0), 0.0)
            if ts_ms <= 0:
                continue
            events.append((ts_ms, 1, "funding", income))

    for trade in trades:
        ts_ms = _safe_int(trade.get("time", 0), 0)
        side = str(trade.get("side", "")).upper()
        if ts_ms <= 0 or side not in {"BUY", "SELL"}:
            continue
        qty = _safe_float(trade.get("qty", 0.0), 0.0)
        price = _safe_float(trade.get("price", 0.0), 0.0)
        if qty <= 0 or price <= 0:
            continue
        payload = {
            "side": side,
            "position_side": str(trade.get("positionSide", "BOTH")).upper(),
            "qty": qty,
            "price": price,
            "realized_pnl": _safe_float(trade.get("realizedPnl", 0.0), 0.0),
            "commission": abs(_safe_float(trade.get("commission", 0.0), 0.0)),
        }
        events.append((ts_ms, 2, "trade", payload))

    if not events:
        return MtmExcursionMetrics(
            mae_usdt=0.0,
            mfe_usdt=0.0,
            max_equity_usdt=0.0,
            min_equity_usdt=0.0,
            equity_points=0,
            mark_points_used=0,
            used_mark_prices=False,
        )

    events.sort(key=lambda item: (item[0], item[1]))

    hedge = _HedgePositionState()
    net = _NetPositionState()

    cash_pnl = 0.0
    current_mark: float | None = None
    mark_points_used = 0
    equity_points = 0

    # Include baseline equity=0.0 as the starting point.
    peak_equity = 0.0
    trough_equity = 0.0
    max_drawdown = 0.0
    max_runup = 0.0

    for _, _, kind, payload in events:
        if kind == "mark":
            current_mark = _safe_float(payload, 0.0)
            if current_mark > 0:
                mark_points_used += 1
            else:
                current_mark = None

        elif kind == "funding":
            cash_pnl += _safe_float(payload, 0.0)

        elif kind == "trade":
            side = payload["side"]
            position_side = payload["position_side"]
            qty = _safe_float(payload["qty"], 0.0)
            price = _safe_float(payload["price"], 0.0)
            realized_pnl = _safe_float(payload["realized_pnl"], 0.0)
            commission = _safe_float(payload["commission"], 0.0)

            if current_mark is None and price > 0:
                current_mark = price

            cash_pnl += realized_pnl - commission

            if position_side == "LONG":
                delta = qty if side == "BUY" else -qty
                hedge.long_qty, hedge.long_entry = _update_abs_position(
                    hedge.long_qty, hedge.long_entry, delta, price,
                )
            elif position_side == "SHORT":
                delta = qty if side == "SELL" else -qty
                hedge.short_qty, hedge.short_entry = _update_abs_position(
                    hedge.short_qty, hedge.short_entry, delta, price,
                )
            else:
                delta = qty if side == "BUY" else -qty
                net.qty, net.entry = _update_net_position(
                    net.qty, net.entry, delta, price,
                )

        # If a mark is unavailable and there is open exposure, we cannot MTM.
        has_open_exposure = bool(
            hedge.long_qty > 0 or hedge.short_qty > 0 or net.qty != 0
        )
        if current_mark is None and has_open_exposure:
            continue

        unrealized = 0.0
        if current_mark is not None:
            unrealized += hedge.long_qty * (current_mark - hedge.long_entry)
            unrealized += hedge.short_qty * (hedge.short_entry - current_mark)
            unrealized += net.qty * (current_mark - net.entry)

        equity = cash_pnl + unrealized
        equity_points += 1

        if equity > peak_equity:
            peak_equity = equity
        drawdown = peak_equity - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        if equity < trough_equity:
            trough_equity = equity
        runup = equity - trough_equity
        if runup > max_runup:
            max_runup = runup

    return MtmExcursionMetrics(
        mae_usdt=max_drawdown,
        mfe_usdt=max_runup,
        max_equity_usdt=peak_equity,
        min_equity_usdt=trough_equity,
        equity_points=equity_points,
        mark_points_used=mark_points_used,
        used_mark_prices=mark_points_used > 0,
    )
