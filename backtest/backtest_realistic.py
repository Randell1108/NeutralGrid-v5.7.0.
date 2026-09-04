#!/usr/bin/env python3
"""
Realistic Neutral Grid Bot Backtester — bidirectional (long + short).

Uses close-to-close crossing only, matching real Binance Neutral Grid
execution patterns:
- Price crosses UP through level: close long from below (TP), open short
- Price crosses DOWN through level: close short from above (TP), open long
"""
from __future__ import annotations

import asyncio
import argparse
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, TYPE_CHECKING, cast
import numpy as np
import pandas as pd

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.grid.equity_circuit_breaker import (
    CircuitBreakerConfig,
    EquityCircuitBreaker,
)
from neutralgrid.grid.formulas import (
    ARITHMETIC,
    BINANCE_DISPLAYED_INTERVALS,
    GEOMETRIC,
    grid_spacing_pct as _canonical_grid_spacing_pct,
    grid_interval_count,
    grid_level_count,
)

try:
    from neutralgrid.core.constants import BOT_HORIZON_HOURS as BOT_HORIZON_HOURS
except ImportError:  # pragma: no cover — editable install expected
    BOT_HORIZON_HOURS: float = 6.0

if TYPE_CHECKING:
    from backtest.btk_seed_state import SeedState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backtest_realistic")


@dataclass
class GridConfig:
    """Grid bot configuration.

    WARNING: Default values for ``funding_mode`` and ``close_fee_mode`` differ from
    TRAINING_ENGINE_DEFAULTS in btk_label_contract.py.  For training label
    generation, always use ``build_training_config()`` from
    btk_unified_runner.py, which applies consistent training-standard defaults.

    Step 20 (Plan v6.0): fields whose GridConfig defaults differ from
    TRAINING_ENGINE_DEFAULTS — these divergences are by design (backward
    compat for ad-hoc research runs vs strict training physics):

    ============ ================ ================
    Field        GridConfig       Training Default
    ============ ================ ================
    funding_mode snapshot         continuous
    close_fee_mode taker          maker
    cb_trailing_activate_pct 2.0  999.0
    cb_trailing_offset_pct 1.0   999.0
    cb_inventory_imbalance_ratio 0.80 0.85
    cb_inventory_imbalance_dd_pct 2.0 3.0
    ============ ================ ================
    """
    symbol: str
    lower: float
    upper: float
    num_grids: int
    mode: str = GEOMETRIC
    grid_count_semantics: str = BINANCE_DISPLAYED_INTERVALS
    capital: float = 400.0
    capital_fraction: float = 1.0
    leverage: int = 10
    volatility_target_pct: float | None = None
    volatility_proxy_pct: float | None = None
    volatility_scale_min: float = 0.05
    volatility_scale_max: float = 1.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    # Order lifecycle delay: after a fill triggers a new order at a grid
    # level, it takes this many bars for the order to be placed in the book
    # and become fillable.  Binance grid bots use resting limit orders that
    # fill on touch, but the *replacement* order after a fill needs time to
    # be detected, placed via API, and queued.  Calibrated from live data
    # (ADAUSDT/LTCUSDT Feb 8 2026): minimum observed gap ≈ 2-3 minutes.
    order_delay_bars: int = 2
    funding_rate: float = 0.0001  # Default funding rate per 8h period
    funding_interval_bars: int = 480  # 8h * 60 min = 480 bars for 1m data
    slippage_bps: float = 0.0  # Slippage in basis points per fill
    # Inventory lifecycle: max holding time for any single position (in bars)
    # Positions exceeding this are force-closed at current price.
    # Default: 6h = 360 bars for 1m data
    max_holding_bars: int = int(BOT_HORIZON_HOURS * 60)
    # Funding accrual mode:
    #   "snapshot"   — charge only at exact funding_interval_bars ticks (legacy)
    #   "continuous"  — prorate per bar based on open notional (more realistic)
    funding_mode: str = "snapshot"
    # Fee charged on grid-crossing close (sell) fills.
    # "taker" = aggressive/market-like (default, conservative)
    # "maker" = resting limit (matches live grid bot TP orders)
    close_fee_mode: str = "taker"
    # P0-A: Liquidation — Binance bracket 1 maintenance margin rate (0-50k notional)
    maintenance_margin_rate: float = 0.004
    # P0-B: Exchange filter rounding (0 = no rounding, backward compatible)
    tick_size: float = 0.0   # PRICE_FILTER: round grid levels to this increment
    step_size: float = 0.0   # LOT_SIZE: round position qty to this increment
    # P1-A: Price source — "last" (default) or "mark" (for PnL/liquidation realism)
    price_source: str = "last"
    valuation_price_source: str = "last"
    # P1-B: Historical funding rate series — list of per-8h rates indexed by window
    # None = use static funding_rate (backward compatible)
    funding_rate_series: list | None = None
    # P1-C: Margin mode — "isolated" (default) or "cross" (documentation/serialization)
    margin_mode: str = "isolated"
    # P2-B: Bid-ask spread model — half-spread in basis points per side (0 = backward compat)
    spread_bps: float = 0.0
    # P3-A: Fill mode — "close" (close-to-close only, default) or "wick" (use high/low)
    fill_mode: str = "close"
    # P0+P1: Global cooldown — after ANY fill, ALL levels blocked for N bars
    # Models real bot cancel → recompute → replace cycle (~1-2 min)
    global_cooldown_bars: int = 0   # 0 = disabled (backward compatible)
    # ── Equity Circuit Breaker — loss truncation + profit preservation ──
    # Solves asymmetry: winners +3-5% but losers -10%.
    # Without CB at 60% WR: EV = 0.60×4 + 0.40×(-10) = -1.6% (negative)
    # With CB at 5%  at 60% WR: EV = 0.60×4 + 0.40×(-5) = +0.4% (positive)
    cb_enabled: bool = False          # Off by default (backward compat)
    cb_max_dd_pct: float = 5.0        # Exit if loss from initial capital > X%
    cb_trailing_activate_pct: float = 2.0  # Activate trailing lock after +X% profit
    cb_trailing_offset_pct: float = 1.0    # Trail X% below peak profit
    cb_inventory_imbalance_ratio: float = 0.80  # Exit if >80% on one side
    cb_inventory_imbalance_dd_pct: float = 2.0   # Min DD for imbalance trigger


@dataclass
class Trade:
    """A single grid trade."""
    timestamp: datetime
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    pnl: float = 0.0


class RealisticGridBacktester:
    """
    Realistic neutral grid bot simulator — bidirectional (long + short).

    Matches Binance Neutral Grid Bot behavior:
    1. Close-to-close crossing with optional wick fills
    2. One position per grid level (long below price, short above)
    3. Realistic execution timing with order delay
    4. Longs pay funding, shorts receive (net notional funding)
    """

    def __init__(self, config: GridConfig):
        if config.num_grids < 2:
            raise ValueError(
                f"num_grids must be >= 2; got {config.num_grids}"
            )
        self.config = config
        self.grid_count_semantics = str(
            getattr(config, "grid_count_semantics", BINANCE_DISPLAYED_INTERVALS)
        ).strip().lower()
        self.grid_intervals = grid_interval_count(config.num_grids, self.grid_count_semantics)
        self.grid_levels = self._compute_grid_levels()
        self._level_to_idx = {level: i for i, level in enumerate(self.grid_levels)}
        self.grid_spacing = self.grid_levels[1] - self.grid_levels[0]
        self.grid_spacing_pct = _canonical_grid_spacing_pct(
            config.lower,
            config.upper,
            config.num_grids,
            str(config.mode).lower(),
            self.grid_count_semantics,
        )
        self._effective_capital, self._volatility_scale_applied = self._resolve_effective_capital()

        # Position tracking: grid_level -> {'entry_price': float, 'qty': float, 'side': str, 'entry_time': datetime, 'entry_bar': int}
        self.positions: Dict[float, dict] = {}
        self.trades: List[Trade] = []
        self.equity_curve = []

        # Order lifecycle: tracks the earliest bar a newly placed order at
        # each grid level becomes fillable.  Models the Binance order
        # placement cycle (fill detection -> API call -> order in book).
        self._level_available_bar: Dict[float, int] = {}
        self._seed_active_sides_by_level: Dict[float, set[str]] = {}
        self._seed_restrict_active_ladder = False
        self._seeded_active_level_count = 0
        self._seed_position_size_override: float | None = None
        self._seed_qty_source = "none"
        self._seed_evidence_class = "missing"
        self._seed_source = ""

        # Inventory lifecycle metrics
        self._stale_closes = 0       # Positions force-closed for exceeding max hold
        self._hold_durations: List[int] = []  # Bars held per closed position

        # P0-A: Liquidation state
        self._liquidated = False
        self._liquidation_bar: int = -1
        self._liquidation_equity: float = 0.0

        # P0+P1: Global cooldown state
        self._global_cooldown_until: int = 0  # bar index when cooldown expires

        # ── Equity Circuit Breaker state ───────────────────────────────────
        self._circuit_breaker = EquityCircuitBreaker(
            CircuitBreakerConfig(
                enabled=config.cb_enabled,
                max_dd_pct=config.cb_max_dd_pct,
                trailing_activate_pct=config.cb_trailing_activate_pct,
                trailing_offset_pct=config.cb_trailing_offset_pct,
                inventory_imbalance_ratio=config.cb_inventory_imbalance_ratio,
                inventory_imbalance_dd_pct=config.cb_inventory_imbalance_dd_pct,
            ),
            initial_capital=0.0,  # Set in run() after effective capital resolved
        )

    def _resolve_effective_capital(self) -> tuple[float, float]:
        """Resolve deployed capital after capital_fraction and vol targeting."""
        cfg = self.config
        base_capital = max(float(cfg.capital), 0.0)
        fraction = float(np.clip(float(getattr(cfg, "capital_fraction", 1.0)), 0.0, 1.0))
        deployed = base_capital * fraction

        vol_scale = 1.0
        target = getattr(cfg, "volatility_target_pct", None)
        proxy = getattr(cfg, "volatility_proxy_pct", None)
        if target is not None and proxy is not None:
            try:
                target_f = float(target)
                proxy_f = float(proxy)
                if target_f > 0 and proxy_f > 0:
                    vol_scale = float(
                        np.clip(
                            target_f / proxy_f,
                            float(getattr(cfg, "volatility_scale_min", 0.05)),
                            float(getattr(cfg, "volatility_scale_max", 1.0)),
                        )
                    )
                    deployed *= vol_scale
            except Exception:
                vol_scale = 1.0

        deployed = max(float(deployed), 0.0)
        return deployed, float(vol_scale)

    def _unrealized_pnl(self, mark_price: float) -> float:
        """Compute total unrealized PnL across all open positions at *mark_price*.

        Long positions profit when mark_price > entry_price.
        Short positions profit when mark_price < entry_price.
        """
        total = 0.0
        for pos in self.positions.values():
            if pos['side'] == 'long':
                total += (mark_price - pos['entry_price']) * pos['qty']
            else:  # short
                total += (pos['entry_price'] - mark_price) * pos['qty']
        return total

    def seed_from_state(self, seed: "SeedState", bar0_idx: int = 0) -> None:
        """Optionally seed the backtester with a replay-derived ladder state.

        Sets ``_level_available_bar`` and side-specific active sets for each
        seeded BUY/SELL level so the simulator recognises them as active from
        bar 0 with the verified order side.  Does **not**
        fabricate positions — only marks which grid levels already had
        resting orders in the book at *t0*.

        Parameters
        ----------
        seed : SeedState
            Seed dataclass produced by :func:`backtest.btk_replay_seed_loader.load_seed_from_replay`.
        bar0_idx : int
            Bar index to treat as the start (default 0).
        """
        self.positions.clear()
        self._level_available_bar.clear()
        self._seed_active_sides_by_level.clear()
        self._seed_restrict_active_ladder = False
        self._seeded_active_level_count = 0
        self._seed_position_size_override = None
        self._seed_qty_source = getattr(seed, "qty_source", "none")
        self._seed_evidence_class = getattr(seed, "evidence_class", "missing")
        self._seed_source = getattr(seed, "source", "")

        # Snap each seeded buy level to the nearest grid level
        for sl in seed.open_buy_levels:
            best_level = min(self.grid_levels, key=lambda g: abs(g - sl.price))
            if sl.price > 0 and abs(best_level - sl.price) / sl.price < 0.02:  # 2% tolerance
                self._level_available_bar[best_level] = bar0_idx
                self._seed_active_sides_by_level.setdefault(best_level, set()).add("BUY")

        for sl in seed.open_sell_levels:
            best_level = min(self.grid_levels, key=lambda g: abs(g - sl.price))
            if sl.price > 0 and abs(best_level - sl.price) / sl.price < 0.02:
                self._level_available_bar[best_level] = bar0_idx
                self._seed_active_sides_by_level.setdefault(best_level, set()).add("SELL")

        self._seeded_active_level_count = len(self._level_available_bar)
        self._seed_restrict_active_ladder = self._seeded_active_level_count > 0

        qty_override = getattr(seed, "qty_per_order", None)
        if qty_override is not None:
            try:
                qty = float(qty_override)
                if qty > 0 and np.isfinite(qty):
                    self._seed_position_size_override = qty
            except (TypeError, ValueError):
                self._seed_position_size_override = None

    def _is_level_available(self, level: float, bar_idx: int, side: str | None = None) -> bool:
        """Return whether *level* has an active order at *bar_idx*."""
        if self._seed_restrict_active_ladder and level not in self._level_available_bar:
            return False
        if self._seed_restrict_active_ladder and side is not None:
            allowed_sides = self._seed_active_sides_by_level.get(level, set())
            if side.strip().upper() not in allowed_sides:
                return False
        return bar_idx >= self._level_available_bar.get(level, 0)

    def _compute_grid_levels(self) -> List[float]:
        """Compute grid price levels, optionally rounded to tick_size.

        ``grid_count_semantics`` makes the convention explicit:
        ``legacy_line_count`` creates ``num_grids`` levels, while
        ``binance_displayed_intervals`` creates ``num_grids + 1`` levels per
        Binance's displayed grid-count formula.
        """
        cfg = self.config
        mode = str(getattr(cfg, "mode", GEOMETRIC)).strip().lower()
        intervals = grid_interval_count(cfg.num_grids, self.grid_count_semantics)
        level_count = grid_level_count(cfg.num_grids, self.grid_count_semantics)
        if mode == GEOMETRIC:
            ratio = (cfg.upper / cfg.lower) ** (1.0 / intervals)
            levels = [cfg.lower * (ratio ** i) for i in range(level_count)]
            levels[0] = cfg.lower
            levels[-1] = cfg.upper
        elif mode == ARITHMETIC:
            step = (cfg.upper - cfg.lower) / intervals
            levels = [cfg.lower + i * step for i in range(level_count)]
        else:
            raise ValueError(f"mode must be one of {(ARITHMETIC, GEOMETRIC)!r}, got {mode!r}")
        if cfg.tick_size > 0:
            levels = [round(round(l / cfg.tick_size) * cfg.tick_size, 10) for l in levels]
        if len(set(levels)) != len(levels):
            raise ValueError(
                "tick_size rounding collapsed grid levels; refusing to run "
                f"{cfg.symbol} with {cfg.num_grids} levels"
            )
        if any(levels[i] >= levels[i + 1] for i in range(len(levels) - 1)):
            raise ValueError("grid levels must be strictly increasing after rounding")
        return levels

    def _get_position_size(self) -> float:
        """Calculate position size per grid, optionally rounded to step_size."""
        if self._seed_position_size_override is not None:
            return self._seed_position_size_override
        cfg = self.config
        capital_per_grid = (self._effective_capital * cfg.leverage) / cfg.num_grids
        avg_price = (cfg.lower + cfg.upper) / 2
        if avg_price <= 0:
            raise ValueError(f"Invalid grid range: lower={cfg.lower}, upper={cfg.upper}, avg_price must be > 0")
        qty = capital_per_grid / avg_price
        if cfg.step_size > 0:
            qty = round(round(qty / cfg.step_size) * cfg.step_size, 10)
        return qty

    def _force_close_stale_positions(
        self,
        bar_idx: int,
        curr_close: float,
        ts,
    ) -> tuple[float, float, int, int]:
        """
        Force-close any position held longer than max_holding_bars.

        Inventory lifecycle control: stale positions are closed at market
        (current close price) with taker fee + slippage to prevent
        indefinite exposure and capital lock-up.

        Returns:
            (total_pnl_delta, total_fee_delta, round_trips_delta, wins_delta)
        """
        cfg = self.config
        if cfg.max_holding_bars <= 0:
            return (0.0, 0.0, 0, 0)

        pnl_delta = 0.0
        fee_delta = 0.0
        rt_delta = 0
        win_delta = 0
        stale_levels = []

        for level, pos in list(self.positions.items()):
            hold_bars = bar_idx - pos.get('entry_bar', bar_idx)
            if hold_bars >= cfg.max_holding_bars:
                stale_levels.append(level)

        for level in stale_levels:
            pos = self.positions[level]
            hold_bars = bar_idx - pos.get('entry_bar', bar_idx)
            # Force close at current market price (taker execution)
            if pos['side'] == 'long':
                pnl = (curr_close - pos['entry_price']) * pos['qty']
            else:  # short
                pnl = (pos['entry_price'] - curr_close) * pos['qty']
            fee = curr_close * pos['qty'] * cfg.taker_fee
            slippage = curr_close * pos['qty'] * (cfg.slippage_bps / 10000)
            net = pnl - fee - slippage

            close_side = 'sell' if pos['side'] == 'long' else 'buy'
            self.trades.append(Trade(ts, close_side, curr_close, pos['qty'], net))
            pnl_delta += pnl
            fee_delta += fee + slippage
            rt_delta += 1
            if net > 0:
                win_delta += 1

            self._hold_durations.append(hold_bars)
            self._stale_closes += 1
            del self.positions[level]
            # Level becomes available again after delay
            self._level_available_bar[level] = bar_idx + cfg.order_delay_bars
            logger.debug(
                f"STALE CLOSE at {curr_close:.4f} (held {hold_bars} bars, "
                f"entry {pos['entry_price']:.4f}, PnL: {net:.2f})"
            )

        return (pnl_delta, fee_delta, rt_delta, win_delta)

    def run(self, df: pd.DataFrame) -> dict:
        """
        Run realistic backtest using close-to-close crossing only.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        """
        cfg = self.config
        position_size = self._get_position_size()
        starting_capital = self._effective_capital
        # ERR-072: a zero-qty ladder (capital_fraction below the position-size
        # minimum, or qty rounded to 0 by step_size) must not open positions —
        # a non-deployment records 0 trades, not phantom zero-qty fills.
        can_open = position_size > 0

        total_pnl = 0.0
        total_fees = 0.0
        total_funding = 0.0
        round_trips = 0
        wins = 0

        if len(df) == 0:
            raise ValueError("Cannot backtest empty DataFrame")

        prev_close = df.iloc[0]['close']
        equity = starting_capital     # realized-only running equity
        peak_equity = equity
        trough_equity = equity
        max_dd = 0.0
        max_mae_usdt = 0.0
        max_mfe_usdt = 0.0

        time_in_range = 0
        total_bars = len(df)

        # Reset circuit breaker with resolved effective capital
        self._circuit_breaker.reset(starting_capital)

        delay = cfg.order_delay_bars
        close_mode = str(cfg.close_fee_mode).lower()
        close_fee_rate = cfg.taker_fee if close_mode == "taker" else cfg.maker_fee

        # Continuous funding: prorate the per-8h rate to per-bar
        # (overridden per bar when funding_rate_series is provided)
        funding_per_bar = (
            cfg.funding_rate / cfg.funding_interval_bars
            if cfg.funding_mode == "continuous" and cfg.funding_interval_bars > 0 else 0.0
        )

        # P2-B: Half-spread cost factor per side
        half_spread_factor = cfg.spread_bps / 10000

        logger.info(f"Starting backtest with {len(df)} candles (order delay: {delay} bars, funding_mode: {cfg.funding_mode})")
        logger.info(f"Grid levels: {[f'{g:.4f}' for g in self.grid_levels]}")

        _closes = df['close'].to_numpy()
        _timestamps = (df['timestamp'] if 'timestamp' in df.columns else df.index).to_numpy()
        # P3-A: Extract high/low for wick fill mode
        _highs = df['high'].to_numpy() if cfg.fill_mode == "wick" else None
        _lows = df['low'].to_numpy() if cfg.fill_mode == "wick" else None

        last_sim_bar = len(df) - 1  # Track actual simulation endpoint

        for bar_idx in range(len(df)):
            curr_close = float(_closes[bar_idx])
            ts: datetime = cast(datetime, pd.Timestamp(_timestamps[bar_idx]).to_pydatetime())

            # Check if price is in range
            if cfg.lower <= curr_close <= cfg.upper:
                time_in_range += 1

            # ── Funding fee accrual ──────────────────────────────────────
            # P1-B: Determine current funding rate (series or static)
            if cfg.funding_rate_series is not None:
                window_idx = bar_idx // cfg.funding_interval_bars
                if window_idx < len(cfg.funding_rate_series):
                    current_funding_rate = cfg.funding_rate_series[window_idx]
                else:
                    current_funding_rate = cfg.funding_rate_series[-1]
                current_funding_per_bar = current_funding_rate / cfg.funding_interval_bars
            else:
                current_funding_rate = cfg.funding_rate
                current_funding_per_bar = funding_per_bar

            if cfg.funding_mode == "continuous" and self.positions:
                # Binance futures: longs pay funding, shorts receive (rate > 0)
                # Net funding cost = (long_notional - short_notional) * rate
                long_notional = sum(
                    curr_close * pos['qty'] for pos in self.positions.values()
                    if pos['side'] == 'long'
                )
                short_notional = sum(
                    curr_close * pos['qty'] for pos in self.positions.values()
                    if pos['side'] == 'short'
                )
                net_notional = long_notional - short_notional
                bar_funding = net_notional * current_funding_per_bar
                total_funding += bar_funding
                equity -= bar_funding
            elif cfg.funding_mode == "snapshot":
                # Legacy: charge only at exact interval ticks
                if (
                    bar_idx > 0
                    and bar_idx % cfg.funding_interval_bars == 0
                    and self.positions
                ):
                    long_notional = sum(
                        curr_close * pos['qty'] for pos in self.positions.values()
                        if pos['side'] == 'long'
                    )
                    short_notional = sum(
                        curr_close * pos['qty'] for pos in self.positions.values()
                        if pos['side'] == 'short'
                    )
                    net_notional = long_notional - short_notional
                    funding_cost = net_notional * current_funding_rate
                    total_funding += funding_cost
                    equity -= funding_cost

            # ── Inventory lifecycle: force-close stale positions ─────────
            stale_pnl, stale_fees, stale_rt, stale_wins = (
                self._force_close_stale_positions(bar_idx, curr_close, ts)
            )
            if stale_rt > 0:
                total_pnl += stale_pnl
                total_fees += stale_fees
                round_trips += stale_rt
                wins += stale_wins
                equity += (stale_pnl - stale_fees)

            # ── P0+P1: Global cooldown gate ──────────────────────────────
            _fills_allowed = (cfg.global_cooldown_bars == 0 or bar_idx >= self._global_cooldown_until)
            _trades_before_fills = len(self.trades)

            # Track levels filled this bar to prevent wick double-fills
            _filled_this_bar: set[float] = set()

            # ── REALISTIC: close-to-close grid level crossing ────────────
            for level in self.grid_levels:
                # P0+P1: Global cooldown — skip all fills during cooldown
                if not _fills_allowed:
                    break

                # Price crossed UP through this level
                if prev_close < level <= curr_close:
                    _filled_this_bar.add(level)
                    # CLOSE: sell existing long from below_level (take profit)
                    level_idx = self._level_to_idx[level]
                    if level_idx > 0:
                        below_level = self.grid_levels[level_idx - 1]
                        if below_level in self.positions and self.positions[below_level].get('side') == 'long':
                            pos = self.positions[below_level]
                            # P2-B: Sell fill at bid (level - half_spread)
                            sell_price = level * (1 - half_spread_factor)
                            pnl = (sell_price - pos['entry_price']) * pos['qty']
                            fee = sell_price * pos['qty'] * close_fee_rate
                            slippage = sell_price * pos['qty'] * (cfg.slippage_bps / 10000)
                            net = pnl - fee - slippage

                            self.trades.append(Trade(ts, 'sell', sell_price, pos['qty'], net))
                            total_pnl += pnl
                            total_fees += fee + slippage
                            round_trips += 1
                            if net > 0:
                                wins += 1

                            equity += net

                            self._hold_durations.append(bar_idx - pos.get('entry_bar', 0))
                            del self.positions[below_level]
                            self._level_available_bar[below_level] = bar_idx + delay
                            logger.debug(f"Closed long at {sell_price:.4f}, PnL: {net:.2f}")

                    # OPEN: place SELL (short) order at this level (gated by cooldown)
                    if cfg.lower < level <= cfg.upper and level not in self.positions:
                        if can_open and self._is_level_available(level, bar_idx, side="SELL"):
                            # P2-B: Short entry at bid (level - half_spread)
                            short_entry_price = level * (1 - half_spread_factor)
                            fee = short_entry_price * position_size * cfg.maker_fee
                            slippage = short_entry_price * position_size * (cfg.slippage_bps / 10000)
                            self.positions[level] = {
                                'entry_price': short_entry_price,
                                'qty': position_size,
                                'side': 'short',
                                'entry_time': ts,
                                'entry_bar': bar_idx,
                            }
                            self.trades.append(Trade(ts, 'sell', short_entry_price, position_size, -(fee + slippage)))
                            total_fees += fee + slippage
                            equity -= fee + slippage
                            self._level_available_bar[level] = bar_idx + delay
                            logger.debug(f"Opened short at {short_entry_price:.4f}")

                # Price crossed DOWN through this level
                elif prev_close > level >= curr_close:
                    _filled_this_bar.add(level)
                    # CLOSE: buy back existing short from above_level (take profit)
                    level_idx = self._level_to_idx[level]
                    if level_idx < len(self.grid_levels) - 1:
                        above_level = self.grid_levels[level_idx + 1]
                        if above_level in self.positions and self.positions[above_level].get('side') == 'short':
                            pos = self.positions[above_level]
                            # P2-B: Buy fill at ask (level + half_spread)
                            buy_price = level * (1 + half_spread_factor)
                            pnl = (pos['entry_price'] - buy_price) * pos['qty']
                            fee = buy_price * pos['qty'] * close_fee_rate
                            slippage = buy_price * pos['qty'] * (cfg.slippage_bps / 10000)
                            net = pnl - fee - slippage

                            self.trades.append(Trade(ts, 'buy', buy_price, pos['qty'], net))
                            total_pnl += pnl
                            total_fees += fee + slippage
                            round_trips += 1
                            if net > 0:
                                wins += 1

                            equity += net

                            self._hold_durations.append(bar_idx - pos.get('entry_bar', 0))
                            del self.positions[above_level]
                            self._level_available_bar[above_level] = bar_idx + delay
                            logger.debug(f"Closed short at {buy_price:.4f}, PnL: {net:.2f}")

                    # OPEN: place BUY (long) order at this level (gated by cooldown)
                    if cfg.lower <= level < cfg.upper and level not in self.positions:
                        if can_open and self._is_level_available(level, bar_idx, side="BUY"):
                            # P2-B: Long entry at ask (level + half_spread)
                            buy_price = level * (1 + half_spread_factor)
                            fee = buy_price * position_size * cfg.maker_fee
                            slippage = buy_price * position_size * (cfg.slippage_bps / 10000)
                            self.positions[level] = {
                                'entry_price': buy_price,
                                'qty': position_size,
                                'side': 'long',
                                'entry_time': ts,
                                'entry_bar': bar_idx,
                            }
                            self.trades.append(Trade(ts, 'buy', buy_price, position_size, -(fee + slippage)))
                            total_fees += fee + slippage
                            equity -= fee + slippage
                            self._level_available_bar[level] = bar_idx + delay
                            logger.debug(f"Opened long at {buy_price:.4f}")

            # ── P3-A: Intrabar wick fills ─────────────────────────────────
            if _fills_allowed and cfg.fill_mode == "wick" and _highs is not None and _lows is not None:
                bar_high = float(_highs[bar_idx])
                bar_low = float(_lows[bar_idx])
                bar_direction_up = curr_close >= prev_close

                for level in self.grid_levels:
                    # Skip levels already handled by close-to-close crossing
                    if prev_close < level <= curr_close:
                        continue
                    if prev_close > level >= curr_close:
                        continue

                    # Skip levels already filled this bar (prevent double-fill)
                    if level in _filled_this_bar:
                        continue

                    # Wick touched this level?
                    if bar_low <= level <= bar_high:
                        if bar_direction_up:
                            # Upward bar: treat wick touch as UP crossing
                            # Close long from below_level (take profit)
                            level_idx = self._level_to_idx[level]
                            if level_idx > 0:
                                below_level = self.grid_levels[level_idx - 1]
                                if below_level in self.positions and self.positions[below_level].get('side') == 'long':
                                    pos = self.positions[below_level]
                                    sell_price = level * (1 - half_spread_factor)
                                    pnl = (sell_price - pos['entry_price']) * pos['qty']
                                    fee = sell_price * pos['qty'] * close_fee_rate
                                    slip = sell_price * pos['qty'] * (cfg.slippage_bps / 10000)
                                    net = pnl - fee - slip
                                    self.trades.append(Trade(ts, 'sell', sell_price, pos['qty'], net))
                                    total_pnl += pnl
                                    total_fees += fee + slip
                                    round_trips += 1
                                    if net > 0:
                                        wins += 1
                                    equity += net
                                    self._hold_durations.append(bar_idx - pos.get('entry_bar', 0))
                                    del self.positions[below_level]
                                    self._level_available_bar[below_level] = bar_idx + delay

                            # Open short at this level
                            if cfg.lower < level <= cfg.upper and level not in self.positions:
                                if can_open and self._is_level_available(level, bar_idx, side="SELL"):
                                    short_entry = level * (1 - half_spread_factor)
                                    fee = short_entry * position_size * cfg.maker_fee
                                    slip = short_entry * position_size * (cfg.slippage_bps / 10000)
                                    self.positions[level] = {
                                        'entry_price': short_entry,
                                        'qty': position_size,
                                        'side': 'short',
                                        'entry_time': ts,
                                        'entry_bar': bar_idx,
                                    }
                                    self.trades.append(Trade(ts, 'sell', short_entry, position_size, -(fee + slip)))
                                    total_fees += fee + slip
                                    equity -= fee + slip
                                    self._level_available_bar[level] = bar_idx + delay
                        else:
                            # Downward bar: treat wick touch as DOWN crossing
                            # Close short from above_level (take profit)
                            level_idx = self._level_to_idx[level]
                            if level_idx < len(self.grid_levels) - 1:
                                above_level = self.grid_levels[level_idx + 1]
                                if above_level in self.positions and self.positions[above_level].get('side') == 'short':
                                    pos = self.positions[above_level]
                                    buy_price = level * (1 + half_spread_factor)
                                    pnl = (pos['entry_price'] - buy_price) * pos['qty']
                                    fee = buy_price * pos['qty'] * close_fee_rate
                                    slip = buy_price * pos['qty'] * (cfg.slippage_bps / 10000)
                                    net = pnl - fee - slip
                                    self.trades.append(Trade(ts, 'buy', buy_price, pos['qty'], net))
                                    total_pnl += pnl
                                    total_fees += fee + slip
                                    round_trips += 1
                                    if net > 0:
                                        wins += 1
                                    equity += net
                                    self._hold_durations.append(bar_idx - pos.get('entry_bar', 0))
                                    del self.positions[above_level]
                                    self._level_available_bar[above_level] = bar_idx + delay

                            # Open long at this level
                            if cfg.lower <= level < cfg.upper and level not in self.positions:
                                if can_open and self._is_level_available(level, bar_idx, side="BUY"):
                                    buy_price = level * (1 + half_spread_factor)
                                    fee = buy_price * position_size * cfg.maker_fee
                                    slip = buy_price * position_size * (cfg.slippage_bps / 10000)
                                    self.positions[level] = {
                                        'entry_price': buy_price,
                                        'qty': position_size,
                                        'side': 'long',
                                        'entry_time': ts,
                                        'entry_bar': bar_idx,
                                    }
                                    self.trades.append(Trade(ts, 'buy', buy_price, position_size, -(fee + slip)))
                                    total_fees += fee + slip
                                    equity -= fee + slip
                                    self._level_available_bar[level] = bar_idx + delay

            # P0+P1: Set global cooldown if any fills occurred this bar
            if len(self.trades) > _trades_before_fills and cfg.global_cooldown_bars > 0:
                self._global_cooldown_until = bar_idx + cfg.global_cooldown_bars

            prev_close = curr_close

            # ── Per-bar mark-to-market equity + drawdown ─────────────────
            valuation_price = curr_close
            if (
                str(getattr(cfg, "valuation_price_source", "last")).lower() == "mark"
                and "mark_close" in df.columns
            ):
                try:
                    valuation_price = float(df.iloc[bar_idx]["mark_close"])
                except (TypeError, ValueError):
                    valuation_price = curr_close
            unrealized = self._unrealized_pnl(valuation_price)
            total_equity = equity + unrealized
            self.equity_curve.append(total_equity)
            peak_equity = max(peak_equity, total_equity)
            trough_equity = min(trough_equity, total_equity)
            mae_usdt_now = peak_equity - total_equity
            mfe_usdt_now = total_equity - trough_equity
            max_mae_usdt = max(max_mae_usdt, mae_usdt_now)
            max_mfe_usdt = max(max_mfe_usdt, mfe_usdt_now)
            dd = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0.0
            max_dd = max(max_dd, dd)

            # ── Equity Circuit Breaker check ────────────────────────────
            # Neutral grid: compute actual long/short inventory counts.
            # Imbalance detector fires when one side dominates AND DD
            # exceeds threshold. See btk_label_contract.py for Branch A/B.
            if not self._liquidated and self.positions and cfg.cb_enabled:
                _long_count = sum(1 for p in self.positions.values() if p['side'] == 'long')
                _short_count = sum(1 for p in self.positions.values() if p['side'] == 'short')
                cb_exit, cb_reason = self._circuit_breaker.check(
                    current_equity=total_equity,
                    long_count=_long_count,
                    short_count=_short_count,
                    bar_idx=bar_idx,
                )
                if cb_exit:
                    # Force-close all positions at market (taker fee + slippage)
                    cb_pnl = 0.0
                    cb_fees = 0.0
                    for level, pos in list(self.positions.items()):
                        if pos['side'] == 'long':
                            pnl = (curr_close - pos['entry_price']) * pos['qty']
                        else:  # short
                            pnl = (pos['entry_price'] - curr_close) * pos['qty']
                        fee = curr_close * pos['qty'] * cfg.taker_fee
                        slip = curr_close * pos['qty'] * (cfg.slippage_bps / 10000)
                        net = pnl - fee - slip
                        close_side = 'sell' if pos['side'] == 'long' else 'buy'
                        self.trades.append(Trade(ts, close_side, curr_close, pos['qty'], net))
                        cb_pnl += pnl
                        cb_fees += fee + slip
                        round_trips += 1
                        if net > 0:
                            wins += 1
                        self._hold_durations.append(bar_idx - pos.get('entry_bar', 0))
                    total_pnl += cb_pnl
                    total_fees += cb_fees
                    equity += (cb_pnl - cb_fees)
                    self.positions.clear()
                    self.equity_curve[-1] = equity
                    logger.info(
                        f"CIRCUIT BREAKER [{cb_reason}] at bar {bar_idx}, "
                        f"equity={equity:.2f}"
                    )
                    last_sim_bar = bar_idx
                    break  # Simulation ends — bot is dead

            # ── P0-A: Liquidation check (isolated margin) ────────────────
            if not self._liquidated and self.positions:
                position_notional = sum(
                    curr_close * pos['qty'] for pos in self.positions.values()
                )
                maintenance_margin = position_notional * cfg.maintenance_margin_rate
                # ERR-072: require positive notional so a zero-notional book
                # cannot register 0 <= 0 as a maintenance-margin liquidation.
                if position_notional > 0 and total_equity <= maintenance_margin:
                    self._liquidated = True
                    self._liquidation_bar = bar_idx
                    self._liquidation_equity = total_equity
                    # Close all positions at market (no PnL recovery)
                    self.positions.clear()
                    equity = total_equity  # Realize the loss
                    self.equity_curve[-1] = equity
                    last_sim_bar = bar_idx
                    break  # Simulation ends

        # ── Final metrics ────────────────────────────────────────────────
        actual_bars = last_sim_bar + 1

        price_start = df.iloc[0]['close']
        price_end = float(_closes[last_sim_bar])
        valuation_price_end = price_end
        if (
            str(getattr(cfg, "valuation_price_source", "last")).lower() == "mark"
            and "mark_close" in df.columns
        ):
            try:
                valuation_price_end = float(df.iloc[last_sim_bar]["mark_close"])
            except (TypeError, ValueError):
                valuation_price_end = price_end

        # Mark-to-market final equity includes unrealized PnL of open inventory
        final_unrealized = self._unrealized_pnl(valuation_price_end)
        final_equity = equity + final_unrealized

        # Net PnL is defined as final MTM equity minus starting capital
        net_pnl = final_equity - starting_capital
        net_pnl_pct = (net_pnl / starting_capital) * 100 if starting_capital > 0 else 0.0

        # ── Binance-style PnL decomposition ───────────────────────────────
        # ``total_funding`` is tracked as a cost: positive values reduce equity.
        # Binance reports funding as a signed contribution to total profit, so
        # invert the sign before applying the documented identity:
        # total = net_realized_ex_funding + open_pnl + funding_fee.
        funding_fee_usdt = -total_funding
        gross_realized_profit_usdt = total_pnl
        net_realized_ex_funding = gross_realized_profit_usdt - total_fees
        open_pnl_usdt = final_unrealized
        total_profit_identity_residual_usdt = (
            net_pnl - (net_realized_ex_funding + open_pnl_usdt + funding_fee_usdt)
        )
        matched_profit_usdt = gross_realized_profit_usdt
        unmatched_pnl_usdt = net_pnl - matched_profit_usdt - funding_fee_usdt
        unmatched_identity_residual_usdt = (
            unmatched_pnl_usdt - (net_pnl - matched_profit_usdt - funding_fee_usdt)
        )

        # ── Realized PnL decomposition (alignment-v1) ─────────────────────
        # equity tracks realized-only (fills − fees − funding).
        # realized_net_pnl = equity − starting_capital ≡ net_pnl − open_pnl
        realized_net_pnl = equity - starting_capital
        realized_net_pnl_pct = (realized_net_pnl / starting_capital) * 100 if starting_capital > 0 else 0.0
        _abs_unreal = abs(final_unrealized)
        _abs_real = abs(realized_net_pnl)
        unrealized_fraction = (
            _abs_unreal / (_abs_unreal + _abs_real)
            if (_abs_unreal + _abs_real) > 0 else 0.0
        )
        win_rate = (wins / round_trips * 100) if round_trips > 0 else 0

        eq_arr = np.array(self.equity_curve) if len(self.equity_curve) > 1 else np.array([starting_capital])
        if len(eq_arr) > 1:
            nonzero_eq = np.where(eq_arr[:-1] == 0, 1e-12, eq_arr[:-1])
            returns = np.diff(eq_arr) / nonzero_eq
        else:
            returns = np.array([0.0])
        # Crypto trades 24/7: 365 days * 24 hours * 60 minutes = 525,600 1-min bars/year
        # Adjust annualization for actual sample size
        ret_std = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
        n_bars = len(returns)
        annualization_factor = np.sqrt(525600 / max(1, n_bars))
        sharpe = (float(np.mean(returns)) / ret_std * annualization_factor) if ret_std > 0 else 0.0

        # Return distribution moments (higher-order risk diagnostics)
        from scipy import stats as _scipy_stats
        return_skewness = float(_scipy_stats.skew(returns)) if n_bars > 2 else 0.0
        return_kurtosis = float(_scipy_stats.kurtosis(returns, fisher=False)) if n_bars > 3 else 3.0

        price_change_pct = (price_end - price_start) / price_start * 100 if price_start > 0 else 0.0
        duration_hours = actual_bars / 60.0

        # --- Inventory lifecycle metrics ---
        avg_hold_bars = float(np.mean(self._hold_durations)) if self._hold_durations else 0.0
        max_hold_bars = int(max(self._hold_durations)) if self._hold_durations else 0
        avg_hold_hours = avg_hold_bars / 60.0
        stale_close_pct = (self._stale_closes / round_trips * 100) if round_trips > 0 else 0.0
        open_positions = len(self.positions)
        long_positions = sum(1 for p in self.positions.values() if p['side'] == 'long')
        short_positions = sum(1 for p in self.positions.values() if p['side'] == 'short')

        # Horizon label: would the bot have ended with positive PnL?
        # P0-A: If liquidated, label is always False
        # CB: If circuit breaker triggered, label based on actual final equity
        label_positive_by_horizon = False if self._liquidated else (final_equity > starting_capital)

        return {
            'symbol': cfg.symbol,
            'duration_hours': duration_hours,
            'grid_lower': cfg.lower,
            'grid_upper': cfg.upper,
            'num_grids': cfg.num_grids,
            'mode': str(cfg.mode).lower(),
            'grid_count_semantics': self.grid_count_semantics,
            'grid_interval_count': self.grid_intervals,
            'grid_level_count': len(self.grid_levels),
            'grid_spacing_pct': self.grid_spacing_pct,
            'total_trades': len(self.trades),
            'round_trips': round_trips,
            'win_rate': win_rate,
            'gross_pnl': total_pnl,
            'gross_realized_profit_usdt': gross_realized_profit_usdt,
            'matched_grid_profit_usdt': gross_realized_profit_usdt,
            'matched_profit_usdt': matched_profit_usdt,
            'net_realized_profit_usdt': net_realized_ex_funding,
            'open_pnl_usdt': open_pnl_usdt,
            'unmatched_pnl_usdt': unmatched_pnl_usdt,
            'total_profit_usdt': net_pnl,
            'funding_fee_usdt': funding_fee_usdt,
            'total_profit_identity_residual_usdt': total_profit_identity_residual_usdt,
            'unmatched_identity_residual_usdt': unmatched_identity_residual_usdt,
            'fees_paid': total_fees,
            'net_pnl': net_pnl,
            'net_pnl_pct': net_pnl_pct,
            'mae': max_mae_usdt,
            'mfe': max_mfe_usdt,
            'mae_pct_initial': (max_mae_usdt / starting_capital) * 100 if starting_capital > 0 else 0.0,
            'mfe_pct_initial': (max_mfe_usdt / starting_capital) * 100 if starting_capital > 0 else 0.0,
            'max_drawdown_pct': max_dd * 100,
            'max_runup_pct': (max_mfe_usdt / starting_capital) * 100 if starting_capital > 0 else 0.0,
            'sharpe_ratio': sharpe,
            'return_skewness': return_skewness,
            'return_kurtosis': return_kurtosis,
            'price_start': price_start,
            'price_end': price_end,
            'valuation_price_end': valuation_price_end,
            'price_change_pct': price_change_pct,
            'time_in_range_pct': (time_in_range / total_bars) * 100,
            'funding_fees': total_funding,
            # Inventory lifecycle
            'avg_hold_bars': round(avg_hold_bars, 1),
            'avg_hold_hours': round(avg_hold_hours, 2),
            'max_hold_bars': max_hold_bars,
            'stale_closes': self._stale_closes,
            'stale_close_pct': round(stale_close_pct, 1),
            # H*-censoring flag: True iff ≥1 position hit max_holding_bars.
            # See btk_label_contract.REQUIRED_LABEL_FIELDS (DURATION_FIX §13 / D14.1).
            'horizon_censored': bool(self._stale_closes > 0),
            'open_positions_at_end': open_positions,
            'long_positions_at_end': long_positions,
            'short_positions_at_end': short_positions,
            'max_holding_bars': cfg.max_holding_bars,
            # Phase 1+4: MTM equity and horizon label
            'unrealized_pnl_at_end': final_unrealized,
            'final_equity': final_equity,
            'peak_equity': peak_equity,
            # ── Realized PnL decomposition (alignment-v1) ──
            'realized_net_pnl': realized_net_pnl,
            'realized_net_pnl_pct': realized_net_pnl_pct,
            'unrealized_fraction': unrealized_fraction,
            'label_positive_by_horizon': label_positive_by_horizon,
            'funding_mode': cfg.funding_mode,
            # P0-A: Liquidation
            'liquidated': self._liquidated,
            'liquidation_bar': self._liquidation_bar,
            'liquidation_equity': self._liquidation_equity,
            # P0-B: Exchange filter rounding
            'tick_size': cfg.tick_size,
            'step_size': cfg.step_size,
            # P1-A: Price source
            'price_source': cfg.price_source,
            'fill_price_source': cfg.price_source,
            'valuation_price_source': cfg.valuation_price_source,
            # P1-B: Funding rate series
            'funding_rate_series_len': len(cfg.funding_rate_series) if cfg.funding_rate_series else 0,
            # P1-C: Margin mode
            'margin_mode': cfg.margin_mode,
            # P2-B: Bid-ask spread
            'spread_bps': cfg.spread_bps,
            # P3-A: Fill mode
            'fill_mode': cfg.fill_mode,
            # P0+P1: Global cooldown
            'global_cooldown_bars': cfg.global_cooldown_bars,
            # P3: Termination penalty (unrealized PnL decomposition)
            'termination_pnl_pct': (final_unrealized / starting_capital) * 100 if starting_capital > 0 else 0.0,
            'exit_penalty_pct': abs(min(0.0, (final_unrealized / starting_capital) * 100)) if starting_capital > 0 else 0.0,
            # ── Equity Circuit Breaker ──
            'cb_enabled': cfg.cb_enabled,
            'cb_triggered': self._circuit_breaker.triggered,
            'cb_trigger_reason': self._circuit_breaker.trigger_reason,
            'cb_trigger_bar': self._circuit_breaker.trigger_bar,
            'cb_max_dd_pct': cfg.cb_max_dd_pct,
            'cb_trailing_activate_pct': cfg.cb_trailing_activate_pct,
            'cb_trailing_offset_pct': cfg.cb_trailing_offset_pct,
            # Sizing diagnostics (capital_fraction + vol targeting)
            'capital_base': cfg.capital,
            'capital_fraction': getattr(cfg, "capital_fraction", 1.0),
            'capital_used': starting_capital,
            'volatility_target_pct': getattr(cfg, "volatility_target_pct", None),
            'volatility_proxy_pct': getattr(cfg, "volatility_proxy_pct", None),
            'volatility_scale_applied': self._volatility_scale_applied,
            'position_size': position_size,
            'position_size_source': (
                self._seed_qty_source
                if self._seed_position_size_override is not None
                else "model"
            ),
            'seed_restrict_active_ladder': self._seed_restrict_active_ladder,
            'seeded_active_level_count': self._seeded_active_level_count,
            'seed_evidence_class': self._seed_evidence_class,
            'seed_source': self._seed_source,
        }


async def fetch_klines(client: BinanceClient, symbol: str, interval: str, hours: int) -> pd.DataFrame:
    """Fetch historical klines."""
    limit = min(int(hours * 60), 1500)
    klines = await client.get_klines(symbol, interval, limit=limit)

    df = pd.DataFrame(klines, columns=pd.Index([
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ]))

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def print_results(result: dict):
    """Print backtest results."""
    print()
    print("=" * 80)
    print(f"REALISTIC BACKTEST RESULTS: {result['symbol']}")
    print("=" * 80)
    print()
    print(f"Duration:                 {result['duration_hours']:.1f} hours")
    print()
    print("--- Grid Parameters ---")
    print(f"Grid Range:               ${result['grid_lower']:.4f} - ${result['grid_upper']:.4f}")
    print(f"Number of Grids:          {result['num_grids']}")
    print(f"Grid Spacing:             {result['grid_spacing_pct']:.2f}%")
    print()
    print("--- Performance ---")
    print(f"Total Trades:             {result['total_trades']}")
    print(f"Round Trips:              {result['round_trips']}")
    print(f"Win Rate:                 {result['win_rate']:.1f}%")
    print(f"Gross PnL:                ${result['gross_pnl']:.2f}")
    print(f"Fees Paid:                ${result['fees_paid']:.2f}")
    print(f"Net PnL:                  ${result['net_pnl']:.2f} ({result['net_pnl_pct']:+.2f}%)")
    print()
    print("--- Risk Metrics ---")
    print(f"MAE (USDT):               ${result.get('mae', 0):.2f}")
    print(f"MFE (USDT):               ${result.get('mfe', 0):.2f}")
    print(f"Max Drawdown (MTM):       {result['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio:             {result['sharpe_ratio']:.2f}")
    print(f"Final Equity (MTM):       ${result.get('final_equity', 0):.2f}")
    print(f"Unrealized PnL at End:    ${result.get('unrealized_pnl_at_end', 0):.2f}")
    print(f"Peak Equity:              ${result.get('peak_equity', 0):.2f}")
    print(f"Horizon Label (+PnL):     {result.get('label_positive_by_horizon', 'N/A')}")
    print(f"Funding Mode:             {result.get('funding_mode', 'snapshot')}")
    print()
    print("--- Inventory Lifecycle ---")
    print(f"Avg Hold Time:            {result.get('avg_hold_hours', 0):.2f}h ({result.get('avg_hold_bars', 0):.0f} bars)")
    print(f"Max Hold Time:            {result.get('max_hold_bars', 0)} bars")
    print(f"Stale Closes (>max hold): {result.get('stale_closes', 0)} ({result.get('stale_close_pct', 0):.1f}%)")
    print(f"Open Positions at End:    {result.get('open_positions_at_end', 0)}")
    print(f"Max Holding Limit:        {result.get('max_holding_bars', 0)} bars ({result.get('max_holding_bars', 0)/60:.0f}h)")
    print()
    print("--- Price Action ---")
    print(f"Price Start:              ${result['price_start']:.4f}")
    print(f"Price End:                ${result['price_end']:.4f}")
    print(f"Price Change:             {result['price_change_pct']:+.2f}%")
    print(f"Time in Range:            {result['time_in_range_pct']:.1f}%")
    print()
    print("=" * 80)
    print()


async def main():
    """Step 12 (Plan v6.0): CLI now routes through run_backtest() from
    btk_unified_runner.py instead of calling RealisticGridBacktester directly.
    This ensures label contract validation, engine settings serialization,
    and consistent provenance stamping even for ad-hoc CLI runs.
    """
    parser = argparse.ArgumentParser(description="Realistic Grid Bot Backtester")
    parser.add_argument("--symbol", required=True, help="Trading symbol")
    parser.add_argument("--lower", type=float, required=True, help="Grid lower bound")
    parser.add_argument("--upper", type=float, required=True, help="Grid upper bound")
    parser.add_argument("--grids", type=int, required=True, help="Number of grids")
    parser.add_argument("--hours", type=int, default=6, help="Backtest period in hours")
    parser.add_argument("--capital", type=float, default=400.0, help="Capital per position")
    parser.add_argument("--leverage", type=int, default=10, help="Leverage")
    parser.add_argument("--order-delay", type=int, default=2,
                        help="Order lifecycle delay in bars (default: 2, matches training default)")
    args = parser.parse_args()

    logger.info(f"Fetching {args.hours}h of 1m klines for {args.symbol}...")

    client = BinanceClient()
    df = await fetch_klines(client, args.symbol, '1m', args.hours)
    await client.close()

    logger.info(f"Fetched {len(df)} candles")
    logger.info(f"Running REALISTIC backtest (close-to-close only)...")
    logger.info(f"  Grid: ${args.lower:.4f} - ${args.upper:.4f}")
    logger.info(f"  Grids: {args.grids}")
    logger.info(f"  Capital: ${args.capital} @ {args.leverage}x")

    # Route through run_backtest() for contract compliance.
    # CLI uses GridConfig directly (not build_training_config) so this is an
    # ad-hoc research run — tagged is_authoritative=False by Step 10 logic
    # when physics fields diverge from TRAINING_ENGINE_DEFAULTS.
    from backtest.btk_unified_runner import run_backtest
    config = GridConfig(
        symbol=args.symbol,
        lower=args.lower,
        upper=args.upper,
        num_grids=args.grids,
        capital=args.capital,
        leverage=args.leverage,
        order_delay_bars=args.order_delay,
    )

    result = run_backtest(config, df)
    print_results(result)


if __name__ == "__main__":
    asyncio.run(main())
