"""
Unified Triple-Barrier Configuration for AFML-Compliant Labeling.

This module provides a single source of truth for barrier parameters used across:
- TripleBarrierLabeler (src/neutralgrid/models/triple_barrier.py)
- BarrierLabelGenerator (src/neutralgrid/training/data_generator.py)
- MetaLabeler (src/neutralgrid/models/meta_labeler.py)
- CPCV purging (src/neutralgrid/backtest/cpcv.py)

AFML Design Principles:
- Consistent barrier definitions across training and inference
- Optional volatility-scaled barriers for regime-adaptive thresholds
- Clear separation between PT/SL barriers and meta-labeling hurdle

Reference: AFML Chapter 3 (Triple-Barrier Method)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import logging

from neutralgrid.core.constants import BOT_HORIZON_HOURS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnifiedBarrierConfig:
    """
    Single source of truth for triple-barrier parameters.

    Attributes:
        pt_pct: Profit-taking barrier as percentage (e.g., 15.0 = +15%)
        sl_pct: Stop-loss barrier as percentage (e.g., -10.0 = -10%)
        time_barrier_hours: Vertical barrier in hours (default 6h)
        timeframe_minutes: Primary timeframe for bar counting (default 15m)
        meta_hurdle_pct: Meta-labeling success threshold (e.g., 3.0 = +3%)
            This is the PnL threshold for binary y=1 label in meta-labeling.
            Can differ from pt_pct for more conservative success definition.

    Volatility Scaling (optional):
        use_volatility_scaling: If True, barriers are scaled by daily volatility
        k_pt: Multiplier for PT barrier (PT = k_pt * daily_vol)
        k_sl: Multiplier for SL barrier (SL = k_sl * daily_vol)

    Example:
        >>> config = UnifiedBarrierConfig()
        >>> config.pt_pct  # 15.0
        >>> config.time_barrier_bars  # 24 (6h * 60 / 15)
    """

    # Horizontal barriers (percentages)
    pt_pct: float = 15.0           # Profit-taking: +15%
    sl_pct: float = -12.0          # Stop-loss: -12%

    # Vertical barrier (time)
    time_barrier_hours: float = BOT_HORIZON_HOURS
    timeframe_minutes: int = 15    # Primary timeframe for bar counting

    # Meta-labeling threshold (can differ from PT)
    meta_hurdle_pct: float = 3.0   # Binary success threshold: +3%

    # Volatility scaling (AFML-recommended, optional)
    use_volatility_scaling: bool = False
    k_pt: float = 2.0              # PT = k_pt * daily_vol (when scaling enabled)
    k_sl: float = 1.5              # SL = k_sl * daily_vol (when scaling enabled)

    @property
    def time_barrier_bars(self) -> int:
        """Convert time barrier hours to number of bars."""
        return int(self.time_barrier_hours * 60 / self.timeframe_minutes)

    @property
    def pt_decimal(self) -> float:
        """PT barrier as decimal (e.g., 0.15 for 15%)."""
        return self.pt_pct / 100.0

    @property
    def sl_decimal(self) -> float:
        """SL barrier as decimal (e.g., -0.12 for -12%)."""
        return self.sl_pct / 100.0

    @property
    def meta_hurdle_decimal(self) -> float:
        """Meta hurdle as decimal (e.g., 0.03 for 3%)."""
        return self.meta_hurdle_pct / 100.0

    def compute_scaled_barriers(
        self,
        daily_vol_pct: float,
    ) -> Tuple[float, float]:
        """
        Compute volatility-scaled barriers if enabled.

        AFML recommends scaling barriers by volatility so that label semantics
        remain consistent across different volatility regimes and symbols.

        Args:
            daily_vol_pct: Daily volatility as percentage (e.g., 5.0 for 5%)

        Returns:
            Tuple of (pt_pct, sl_pct) - either fixed or volatility-scaled
        """
        if not self.use_volatility_scaling:
            return (self.pt_pct, self.sl_pct)

        scaled_pt = self.k_pt * daily_vol_pct
        scaled_sl = -self.k_sl * daily_vol_pct

        # Clamp to sane range to prevent absurd barriers from bad vol estimates
        min_pt_pct, max_pt_pct = 2.0, 30.0
        min_sl_pct, max_sl_pct = -30.0, -2.0
        clamped_pt = max(min_pt_pct, min(scaled_pt, max_pt_pct))
        clamped_sl = max(min_sl_pct, min(scaled_sl, max_sl_pct))

        if clamped_pt != scaled_pt or clamped_sl != scaled_sl:
            logger.warning(
                f"Volatility-scaled barriers clamped: PT {scaled_pt:.2f}%->{clamped_pt:.2f}%, "
                f"SL {scaled_sl:.2f}%->{clamped_sl:.2f}% (daily_vol={daily_vol_pct:.2f}%)"
            )

        logger.debug(
            f"Volatility-scaled barriers: PT={clamped_pt:.2f}%, SL={clamped_sl:.2f}% "
            f"(daily_vol={daily_vol_pct:.2f}%)"
        )

        return (clamped_pt, clamped_sl)

    def get_barrier_prices(
        self,
        entry_price: float,
        daily_vol_pct: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute actual barrier price levels from entry price.

        Args:
            entry_price: Entry price for the position
            daily_vol_pct: Daily volatility (required if use_volatility_scaling=True)

        Returns:
            Tuple of (pt_price, sl_price)
        """
        if self.use_volatility_scaling:
            if daily_vol_pct is None:
                raise ValueError(
                    "daily_vol_pct required when use_volatility_scaling=True"
                )
            pt_pct, sl_pct = self.compute_scaled_barriers(daily_vol_pct)
        else:
            pt_pct, sl_pct = self.pt_pct, self.sl_pct

        pt_price = entry_price * (1.0 + pt_pct / 100.0)
        sl_price = entry_price * (1.0 + sl_pct / 100.0)

        return (pt_price, sl_price)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_barrier_config() -> UnifiedBarrierConfig:
    """
    Factory function to load barrier config from global settings.

    This function reads from the unified config and constructs a UnifiedBarrierConfig.
    Use this to ensure consistent configuration across all modules.

    Returns:
        UnifiedBarrierConfig instance with values from config
    """
    try:
        from neutralgrid.core.config import get_config

        _cfg = get_config()
        return UnifiedBarrierConfig(
            pt_pct=_cfg.barrier.pt_pct,
            sl_pct=_cfg.barrier.sl_pct,
            time_barrier_hours=_cfg.barrier.time_hours,
            timeframe_minutes=_cfg.barrier.timeframe_minutes,
            meta_hurdle_pct=_cfg.barrier.meta_hurdle_pct,
            use_volatility_scaling=_cfg.barrier.use_vol_scaling,
            k_pt=_cfg.barrier.k_pt,
            k_sl=_cfg.barrier.k_sl,
        )
    except ImportError:
        logger.warning("config module not found, using default barrier config")
        return UnifiedBarrierConfig()


# =============================================================================
# DEFAULT INSTANCE
# =============================================================================

# Default config for direct import
DEFAULT_BARRIER_CONFIG = UnifiedBarrierConfig()
