"""
Equity Circuit Breaker — per-bot loss truncation and profit preservation.

Solves the core asymmetry problem in neutral grid bots:
    Winners average +3% to +5%, but losers take -10%.
    Without loss truncation, you need >75% win rate just to break even.

Mathematical basis:
    EV = win_rate × avg_win + (1 - win_rate) × avg_loss

    Without CB (avg_loss = -10%):
        60% WR → EV = 0.60×4 + 0.40×(-10) = -1.6%   (NEGATIVE)

    With CB at -5% max drawdown:
        60% WR → EV = 0.60×4 + 0.40×(-5) = +0.4%    (POSITIVE)

Three independent exit signals:

1. **Max Drawdown Stop** — Exit if unrealized loss from initial capital
   exceeds threshold.  Truncates tail losses.

2. **Trailing Profit Lock** — Once profit reaches activation threshold,
   trail a stop below peak profit.  Preserves winners that start to reverse.

3. **Inventory Imbalance Detector** — Exit if too many positions
   accumulate on one side while in drawdown.  Detects trending regimes
   early before full damage.

All three are independently configurable and can be disabled individually.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Configuration for the equity circuit breaker.

    All percentage values are expressed as positive floats
    (e.g., 5.0 means 5%).
    """
    enabled: bool = False                      # Off by default (backward compat)
    max_dd_pct: float = 5.0                    # Max drawdown from initial capital
    trailing_activate_pct: float = 2.0         # Activate trailing after +X% profit
    trailing_offset_pct: float = 1.0           # Trail X% below peak profit
    inventory_imbalance_ratio: float = 0.80    # Exit if >80% positions on one side
    inventory_imbalance_dd_pct: float = 2.0    # Min drawdown to trigger imbalance exit


class EquityCircuitBreaker:
    """Per-bot equity circuit breaker for loss truncation and profit lock.

    Usage (live trading)::

        cb = EquityCircuitBreaker(
            CircuitBreakerConfig(enabled=True, max_dd_pct=5.0),
            initial_capital=400.0,
        )

        # Each bar:
        should_exit, reason = cb.check(
            current_equity=total_equity,
            long_count=len(long_positions),
            short_count=len(short_positions),
        )
        if should_exit:
            close_all_positions()
            log(f"Circuit breaker: {reason}")

    Usage (backtest engine)::

        Integrated directly into RealisticGridBacktester.run() loop.
        See backtest_realistic.py for integration.
    """

    def __init__(
        self,
        config: CircuitBreakerConfig,
        initial_capital: float,
    ) -> None:
        self.cfg = config
        self.initial_capital = initial_capital
        self.peak_equity = initial_capital

        # Trailing profit lock state
        self._trailing_active = False

        # Result state
        self.triggered = False
        self.trigger_reason = ""
        self.trigger_bar = -1

    def check(
        self,
        current_equity: float,
        long_count: int = 0,
        short_count: int = 0,
        bar_idx: int = 0,
    ) -> tuple[bool, str]:
        """Check all circuit breaker conditions.

        Parameters
        ----------
        current_equity : float
            Current mark-to-market equity (realized + unrealized).
        long_count : int
            Number of open long (buy-side) positions.
        short_count : int
            Number of open short (sell-side) positions.
            For neutral grid bots: positions below current price are long,
            positions above are short.  The imbalance detector fires when
            one side dominates (e.g. trending market accumulates longs
            while shorts close, or vice versa).
        bar_idx : int
            Current bar index (for logging).

        Returns
        -------
        (should_exit, reason) : tuple[bool, str]
            If should_exit is True, the bot should close all positions
            and stop trading.  ``reason`` describes which condition fired.
        """
        if not self.cfg.enabled or self.triggered:
            return False, ""

        # Update peak equity (only upward)
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # ── 1. Max drawdown stop ──────────────────────────────────────────
        # Drawdown relative to INITIAL CAPITAL (not peak) — this measures
        # absolute loss, which is what matters for your account balance.
        dd_from_initial_pct = (
            (self.initial_capital - current_equity) / self.initial_capital * 100
            if self.initial_capital > 0 else 0.0
        )

        if dd_from_initial_pct >= self.cfg.max_dd_pct:
            self.triggered = True
            self.trigger_reason = f"max_dd_{dd_from_initial_pct:.1f}pct"
            self.trigger_bar = bar_idx
            return True, self.trigger_reason

        # ── 2. Trailing profit lock ───────────────────────────────────────
        profit_pct = (
            (current_equity - self.initial_capital) / self.initial_capital * 100
            if self.initial_capital > 0 else 0.0
        )

        if profit_pct >= self.cfg.trailing_activate_pct:
            self._trailing_active = True

        if self._trailing_active:
            peak_profit_pct = (
                (self.peak_equity - self.initial_capital) / self.initial_capital * 100
                if self.initial_capital > 0 else 0.0
            )
            trail_level_pct = peak_profit_pct - self.cfg.trailing_offset_pct

            if profit_pct <= trail_level_pct and trail_level_pct > 0:
                self.triggered = True
                self.trigger_reason = (
                    f"trailing_lock_at_{profit_pct:.1f}pct"
                    f"_peak_{peak_profit_pct:.1f}pct"
                )
                self.trigger_bar = bar_idx
                return True, self.trigger_reason

        # ── 3. Inventory imbalance detector ───────────────────────────────
        total_positions = long_count + short_count
        if total_positions >= 3 and dd_from_initial_pct >= self.cfg.inventory_imbalance_dd_pct:
            max_side = max(long_count, short_count)
            imbalance = max_side / total_positions
            if imbalance >= self.cfg.inventory_imbalance_ratio:
                self.triggered = True
                side = "long" if long_count >= short_count else "short"
                self.trigger_reason = (
                    f"inventory_imbalance_{side}"
                    f"_{imbalance:.0%}_dd_{dd_from_initial_pct:.1f}pct"
                )
                self.trigger_bar = bar_idx
                return True, self.trigger_reason

        return False, ""

    def reset(self, initial_capital: float | None = None) -> None:
        """Reset circuit breaker state for a new bot run."""
        if initial_capital is not None:
            self.initial_capital = initial_capital
        self.peak_equity = self.initial_capital
        self._trailing_active = False
        self.triggered = False
        self.trigger_reason = ""
        self.trigger_bar = -1
