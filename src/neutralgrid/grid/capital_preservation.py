"""Capital-preservation exit policy for neutral grid bot simulations.

The policy is intentionally separate from the canonical equity circuit breaker.
It models the operator-specified manual interventions without changing existing
training or production defaults.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapitalPreservationConfig:
    """Explicit per-bot loss and recovery limits, expressed as percentages."""

    enabled: bool = False
    manual_cancel_loss_pct: float = 5.0
    accelerated_drop_pct_per_bar: float = 2.0
    automatic_stop_loss_pct: float = 10.0
    absolute_drawdown_ceiling_pct: float = 15.0
    recovery_horizon_bars: int = 360


class CapitalPreservationPolicy:
    """Fail-closed per-bot exit policy evaluated on each one-minute bar.

    A loss episode is armed when equity reaches the manual-cancel threshold.
    Returning above that threshold is the only recovery signal. The bot exits
    immediately when the same bar also carries an accelerated loss, when the
    automatic stop is reached, or when the absolute ceiling is reached. A loss
    episode that stays armed for the full recovery horizon is manually closed.
    """

    def __init__(
        self,
        config: CapitalPreservationConfig,
        initial_capital: float,
    ) -> None:
        self.cfg = config
        self.initial_capital = float(initial_capital)
        self.triggered = False
        self.trigger_reason = ""
        self.trigger_bar = -1
        self.recovery_armed_bar = -1
        self.one_bar_drop_pct = 0.0
        self._previous_equity = float(initial_capital)
        self._validate_config()

    def _validate_config(self) -> None:
        cfg = self.cfg
        if cfg.manual_cancel_loss_pct <= 0:
            raise ValueError("manual_cancel_loss_pct must be positive")
        if cfg.accelerated_drop_pct_per_bar <= 0:
            raise ValueError("accelerated_drop_pct_per_bar must be positive")
        if cfg.automatic_stop_loss_pct <= cfg.manual_cancel_loss_pct:
            raise ValueError(
                "automatic_stop_loss_pct must exceed manual_cancel_loss_pct"
            )
        if cfg.absolute_drawdown_ceiling_pct <= cfg.automatic_stop_loss_pct:
            raise ValueError(
                "absolute_drawdown_ceiling_pct must exceed automatic_stop_loss_pct"
            )
        if cfg.recovery_horizon_bars <= 0:
            raise ValueError("recovery_horizon_bars must be positive")

    def reset(self, initial_capital: float | None = None) -> None:
        if initial_capital is not None:
            self.initial_capital = float(initial_capital)
        self.triggered = False
        self.trigger_reason = ""
        self.trigger_bar = -1
        self.recovery_armed_bar = -1
        self.one_bar_drop_pct = 0.0
        self._previous_equity = self.initial_capital

    def _trigger(self, reason: str, bar_idx: int) -> tuple[bool, str]:
        self.triggered = True
        self.trigger_reason = reason
        self.trigger_bar = bar_idx
        return True, reason

    def check(
        self,
        current_equity: float,
        long_count: int = 0,
        short_count: int = 0,
        bar_idx: int = 0,
    ) -> tuple[bool, str]:
        """Evaluate the explicit capital-preservation rules for one bar."""
        del long_count, short_count
        if not self.cfg.enabled or self.triggered:
            return False, ""

        capital = self.initial_capital
        if capital <= 0:
            raise ValueError("initial_capital must be positive")

        equity = float(current_equity)
        loss_pct = max((capital - equity) / capital * 100.0, 0.0)
        self.one_bar_drop_pct = max(
            (self._previous_equity - equity) / capital * 100.0,
            0.0,
        )
        self._previous_equity = equity

        if loss_pct >= self.cfg.absolute_drawdown_ceiling_pct:
            return self._trigger(
                f"absolute_drawdown_ceiling_{loss_pct:.2f}pct",
                bar_idx,
            )

        if loss_pct >= self.cfg.automatic_stop_loss_pct:
            return self._trigger(
                f"automatic_stop_loss_{loss_pct:.2f}pct",
                bar_idx,
            )

        if loss_pct < self.cfg.manual_cancel_loss_pct:
            self.recovery_armed_bar = -1
            return False, ""

        if self.one_bar_drop_pct >= self.cfg.accelerated_drop_pct_per_bar:
            return self._trigger(
                "manual_accelerated_loss_"
                f"{loss_pct:.2f}pct_drop_{self.one_bar_drop_pct:.2f}pct",
                bar_idx,
            )

        if self.recovery_armed_bar < 0:
            self.recovery_armed_bar = bar_idx

        elapsed = bar_idx - self.recovery_armed_bar + 1
        if elapsed >= self.cfg.recovery_horizon_bars:
            return self._trigger(
                f"manual_no_recovery_{loss_pct:.2f}pct_{elapsed}bars",
                bar_idx,
            )

        return False, ""
