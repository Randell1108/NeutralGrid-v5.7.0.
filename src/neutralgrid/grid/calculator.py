"""
Grid Bot Parameter Calculator.

Calculates optimal grid parameters for NEUTRAL grid bots
using ATR-normalized spacing and risk-validated leverage.
"""

from dataclasses import dataclass
from typing import Any, Optional
import logging
import math

from neutralgrid.core.config import get_config

logger = logging.getLogger(__name__)
from neutralgrid.validation.regime_validator import ValidationResult
from neutralgrid.grid.spacing_profile import get_target_spacing_pct


@dataclass
class GridParams:
    """Complete grid bot parameters for deployment."""
    # Symbol & Status
    symbol: str
    is_valid: bool
    reason: Optional[str] = None

    # Grid Structure
    grid_lower: Optional[float] = None
    grid_upper: Optional[float] = None
    num_grids: Optional[int] = None
    grid_spacing_pct: Optional[float] = None
    profit_per_grid_pct: Optional[float] = None  # Uniform per-grid profit (geometric identity)

    # Capital & Leverage
    capital: Optional[float] = None
    leverage: Optional[int] = None
    total_notional: Optional[float] = None

    # Risk Controls
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_holding_time: Optional[str] = None

    # Expected Performance
    expected_net_return_pct: Optional[float] = None

    # Fees
    maker_fee: Optional[float] = None
    taker_fee: Optional[float] = None

    # Final live sizing is determined later in the enrichment pipeline.
    capital_fraction: Optional[float] = None
    regime_confidence: Optional[float] = None
    sizing_reason: Optional[str] = None

    def __post_init__(self):
        """Populate None defaults from live config."""
        cfg = get_config()
        if self.capital is None:
            self.capital = cfg.grid.capital
        if self.stop_loss_pct is None:
            self.stop_loss_pct = cfg.grid.stop_loss_pct
        if self.take_profit_pct is None:
            self.take_profit_pct = cfg.grid.take_profit_pct
        if self.max_holding_time is None:
            self.max_holding_time = cfg.grid.max_holding_time
        if self.maker_fee is None:
            self.maker_fee = cfg.grid.maker_fee
        if self.taker_fee is None:
            self.taker_fee = cfg.grid.taker_fee

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "is_valid": self.is_valid,
            "reason": self.reason,
            "grid_lower": self.grid_lower,
            "grid_upper": self.grid_upper,
            "num_grids": self.num_grids,
            "grid_spacing_pct": round(self.grid_spacing_pct, 4) if self.grid_spacing_pct is not None else None,
            "profit_per_grid_pct": round(self.profit_per_grid_pct, 4) if self.profit_per_grid_pct is not None else None,
            "capital": self.capital,
            "leverage": self.leverage,
            "total_notional": round(self.total_notional, 2) if self.total_notional is not None else None,
            "stop_loss_pct": round(self.stop_loss_pct * 100, 2) if self.stop_loss_pct is not None else None,
            "take_profit_pct": round(self.take_profit_pct * 100, 2) if self.take_profit_pct is not None else None,
            "max_holding_time": self.max_holding_time,
            "expected_net_return_pct": round(self.expected_net_return_pct, 2) if self.expected_net_return_pct is not None else None,
            "maker_fee_pct": round(self.maker_fee * 100, 4) if self.maker_fee is not None else None,
            "taker_fee_pct": round(self.taker_fee * 100, 4) if self.taker_fee is not None else None,
            # Live sizing is computed later in enrichment.
            "capital_fraction": round(self.capital_fraction, 4) if self.capital_fraction is not None else None,
            "regime_confidence": round(self.regime_confidence, 4) if self.regime_confidence is not None else None,
            "sizing_reason": self.sizing_reason,
        }


class GridCalculator:
    """
    Grid parameter optimizer for NEUTRAL grid bots (geometric grid identity).

    Spacing target = empirical winner-derived geometric subpool median, capped
    at cfg.grid.max_spacing_pct (winner q95). ATR-scaling is not used —
    empirical winners do not align spacing with realized volatility at any
    consistent multiplier.
    Grid range is derived from the validated 15M range.
    Profit/grid follows Binance's geometric formula: (1 - c) * r - 1 - c
    where r = (Upper / Lower) ** (1 / num_grids).
    """
    
    def calculate_grid_range(
        self,
        range_high: float,
        range_low: float,
        margin_pct: float = 0.05
    ) -> tuple[float, float]:
        """
        Calculate grid range with safety margin.
        
        Uses validated 15m range but adds small buffer to avoid
        boundary executions at range extremes.
        
        Args:
            range_high: Upper bound from 15M validation
            range_low: Lower bound from 15M validation
            margin_pct: Safety margin percentage (default 5%)
        
        Returns:
            (grid_lower, grid_upper)
        """
        range_size = range_high - range_low
        margin = range_size * margin_pct
        
        grid_lower = range_low + margin
        grid_upper = range_high - margin
        
        return grid_lower, grid_upper
    
    def calculate_grid_spacing(
        self,
        atr_1m: float,
        current_price: float,
        range_size: float,
        min_profit_pct_override: float | None = None,
    ) -> float:
        """
        Calculate winner-derived grid spacing under geometric grid identity.

        Spacing is the larger of the empirical winner target (geometric subpool
        median) and the post-fee profit floor, capped at the winner q95
        (cfg.grid.max_spacing_pct). ATR-driven scaling has been removed: the
        empirical spacing% / atr_pct ratio in the geometric winner subpool has
        no consistent multiplier (median 0.186, q25/q75 = 0.154/0.292), so a
        constant ATR-multiplier rule is provably miscalibrated and is dropped.

        Args:
            atr_1m: retained for API compatibility; no longer consumed.
            current_price: retained for API compatibility; no longer consumed.
            range_size: retained for API compatibility; no longer consumed.
            min_profit_pct_override: optional override for the profit floor (percent).

        Returns:
            Grid spacing as decimal fraction (e.g. 0.0072 for 0.72%).
        """
        cfg = get_config()
        max_spacing_pct = float(cfg.grid.max_spacing_pct)

        # --- Empirical target (winner-like) spacing — geometric subpool median ---
        try:
            target_pct = float(get_target_spacing_pct()) / 100.0  # convert percent -> decimal
        except Exception as e:
            logger.warning("Failed to load target spacing from profile, using fallback: %s", e)
            target_pct = 0.0072  # geometric subpool median fallback (0.72%)

        # --- Profit floor (after fees) ---
        # Net profit per grid (decimal) = spacing - maker_fee - avg_exit_fee.
        avg_exit_fee = (cfg.grid.maker_fee + cfg.grid.taker_fee) / 2
        _profit_floor = min_profit_pct_override if min_profit_pct_override is not None else float(cfg.grid.profit_grid_min_pct_static_fallback)
        min_net = _profit_floor / 100.0
        min_for_profit = min_net + cfg.grid.maker_fee + avg_exit_fee

        spacing_pct = max(target_pct, min_for_profit)
        spacing_pct = min(spacing_pct, max_spacing_pct)
        return float(spacing_pct)
    
    def calculate_num_grids(
        self,
        grid_lower: float,
        grid_upper: float,
        spacing_pct: float,
        min_grids: int | None = None,
        max_grids: int | None = None,
    ) -> int:
        """
        Number of grid lines under the geometric grid identity.

        Geometric: r = (Upper / Lower) ** (1 / n) where n = num_grids - 1
        intervals between adjacent levels (matches Binance docs and the
        extractor convention at _bot_data_extractor_core.py:98). Inverting:
            n = ln(Upper / Lower) / ln(1 + spacing_pct)
        and num_grids = n + 1.

        Args:
            grid_lower: Lower grid boundary (price).
            grid_upper: Upper grid boundary (price).
            spacing_pct: Geometric spacing per interval as a decimal (e.g. 0.0072).
            min_grids/max_grids: clamp bounds (defaults from GridConfig).

        Returns:
            Number of grid levels (int >= min_grids).
        """
        cfg = get_config()
        if min_grids is None:
            min_grids = cfg.grid.min_grids
        if max_grids is None:
            max_grids = cfg.grid.max_grids

        if not (math.isfinite(grid_lower) and math.isfinite(grid_upper) and math.isfinite(spacing_pct)):
            return min_grids

        if grid_lower >= grid_upper or spacing_pct <= 0 or grid_lower <= 0:
            return min_grids

        r = 1.0 + spacing_pct
        if r <= 1.0:
            return min_grids

        n_intervals = int(math.log(grid_upper / grid_lower) / math.log(r))
        num_grids = n_intervals + 1

        return max(min_grids, min(num_grids, max_grids))

    def calculate_profit_per_grid(
        self,
        grid_lower: float,
        grid_upper: float,
        num_grids: int,
        maker_fee: float | None = None,
        taker_fee: float | None = None,
    ) -> tuple[float, float, float]:
        """
        Net profit per grid under Binance's geometric grid identity.

        Geometric grids have a constant price ratio r between adjacent levels,
        so the per-grid percentage profit is uniform across the range.

        Binance Geometric Grid Formulas:
            r          = (Upper / Lower) ** (1 / n)         # n = displayed grid intervals
            Profit/Grid = (1 - c) * r - 1 - c

        c is the per-side fee rate. To keep scanner EV consistent with the
        backtest engine, we use c = (maker_fee + close_fee_rate) / 2, where
        close_fee_rate follows cfg.grid.close_fee_mode.

        Args:
            grid_lower, grid_upper: price boundaries.
            num_grids: Binance-displayed interval count.
            maker_fee, taker_fee: fee rates (default from GridConfig).

        Returns:
            (profit, profit, profit) — uniform under geometric. Tuple shape
            preserved for caller compatibility with prior arithmetic API.
            Profit values are decimals (e.g. 0.0072 for 0.72%).
        """
        if maker_fee is None or taker_fee is None:
            cfg = get_config()
            if maker_fee is None:
                maker_fee = cfg.grid.maker_fee
            if taker_fee is None:
                taker_fee = cfg.grid.taker_fee

        if num_grids <= 1 or grid_lower <= 0 or grid_lower >= grid_upper:
            return (0.0, 0.0, 0.0)

        from neutralgrid.grid.formulas import (
            BINANCE_DISPLAYED_INTERVALS,
            GEOMETRIC,
            profit_per_grid_pct,
        )

        cfg = get_config()
        close_fee_mode = str(getattr(cfg.grid, "close_fee_mode", "maker")).lower()
        close_fee_rate = maker_fee if close_fee_mode == "maker" else taker_fee
        c = max(0.0, (float(maker_fee) + float(close_fee_rate)) / 2.0)

        # GRID_SYNCH Step 2: delegate to shared formulas module.
        profit_pct = profit_per_grid_pct(
            float(grid_lower),
            float(grid_upper),
            int(num_grids),
            GEOMETRIC,
            c,
            BINANCE_DISPLAYED_INTERVALS,
        )
        profit = profit_pct / 100.0
        return (profit, profit, profit)
    
    def select_leverage(
        self,
        range_size_pct: float,
        stop_loss_pct: float,
        current_price: float
    ) -> int:
        """
        Select lowest safe leverage within permitted range.
        
        Ensures that:
        1. Full range movement doesn't trigger liquidation
        2. Stop loss can be hit without liquidation
        
        Args:
            range_size_pct: Range size as percentage
            stop_loss_pct: Stop loss percentage (negative)
            current_price: Current price
        
        Returns:
            Selected leverage (5x-10x)
        """
        # Liquidation occurs at ~(1 / leverage) adverse move
        # We want stop loss to trigger well before liquidation
        
        # Maximum adverse move we need to survive
        max_adverse = abs(stop_loss_pct) + range_size_pct / 2
        
        # Leverage should be such that liquidation distance > max_adverse * 1.5
        # Liquidation at ~(1/leverage) adverse move
        # So leverage < 1 / (max_adverse * 1.5)
        
        cfg = get_config()
        denominator = max_adverse * 1.5
        safe_leverage = int(1 / denominator) if denominator > 0 else cfg.grid.leverage_max

        # Clamp to permitted range
        leverage = max(cfg.grid.leverage_min, min(safe_leverage, cfg.grid.leverage_max))

        return leverage

    def compute_regime_adjusted_grids(
        self,
        num_grids: int,
        grid_spacing_pct: float,
        trend_prob: float,
        hurst_exponent: Optional[float] = None,
        min_grids: int | None = None,
        max_grids: int | None = None,
        grid_lower: float | None = None,
        grid_upper: float | None = None,
        max_spacing_pct: float = 2.0,
    ) -> tuple[int, float]:
        """
        Adjust grid count and spacing based on regime.

        AFML principle: Widen spacing when trend_prob rises (reduce overtrading).

        When grid_lower and grid_upper are provided, the returned spacing is
        derived from final geometry (bounds / count) rather than stored
        independently.  The max_spacing_pct cap is re-enforced on the final
        geometry — if exceeded, grid count is increased until satisfied.

        Args:
            num_grids: Original number of grids
            grid_spacing_pct: Original grid spacing percentage
            trend_prob: HMM posterior probability of trend regime
            hurst_exponent: Hurst exponent (optional)
            min_grids: Minimum allowed grids (default: from GridConfig)
            max_grids: Maximum allowed grids (default: from GridConfig)
            grid_lower: Lower price boundary (enables geometry-derived spacing)
            grid_upper: Upper price boundary (enables geometry-derived spacing)
            max_spacing_pct: Maximum spacing percentage cap (default: 2.0%)

        Returns:
            Tuple of (adjusted_num_grids, adjusted_spacing_pct)
        """
        # Get threshold from config
        cfg = get_config()
        if min_grids is None:
            min_grids = cfg.grid.min_grids
        if max_grids is None:
            max_grids = cfg.grid.max_grids
        trend_prob_max = float(cfg.hmm.trend_prob_max)

        # Base case: no adjustment needed
        if trend_prob <= 0.20:
            # Derive spacing from geometric identity when bounds are available
            if grid_lower is not None and grid_upper is not None and grid_lower > 0 and num_grids > 1:
                adj_grids = num_grids
                implied = ((grid_upper / grid_lower) ** (1.0 / max(1, adj_grids - 1)) - 1.0) * 100
                # Enforce max_spacing_pct cap (int truncation can widen spacing)
                while implied > max_spacing_pct and adj_grids < max_grids:
                    adj_grids += 1
                    implied = ((grid_upper / grid_lower) ** (1.0 / max(1, adj_grids - 1)) - 1.0) * 100
                return (adj_grids, round(implied, 4))
            return (num_grids, grid_spacing_pct)

        # Calculate spacing multiplier based on trend probability.
        # Keep 1.0x-1.3x bounds but use a convex curve so widening is modest
        # in weak trend and stronger near the veto threshold.
        if trend_prob <= trend_prob_max:
            intensity = (trend_prob - 0.20) / (trend_prob_max - 0.20)
            intensity = max(0.0, min(1.0, intensity))
            spacing_mult = 1.0 + (intensity ** 1.35) * 0.3
        else:
            spacing_mult = 1.3

        # Adjust for Hurst if available
        if hurst_exponent is not None and hurst_exponent > 0.50:
            hurst_max = float(cfg.stochastic.hurst_max_trending)
            if hurst_exponent <= hurst_max and hurst_max > 0.50:
                # Additional 10% widening as Hurst approaches threshold
                hurst_mult = 1.0 + (hurst_exponent - 0.50) / (hurst_max - 0.50) * 0.1
                spacing_mult *= hurst_mult

        # Recalculate grid count with wider spacing
        # Fewer grids = less overtrading in borderline regimes
        adjusted_grids = int(num_grids / spacing_mult)
        adjusted_grids = max(min_grids, min(adjusted_grids, max_grids))

        # Derive spacing from geometric identity when bounds are available (single source of truth)
        if grid_lower is not None and grid_upper is not None and grid_lower > 0 and adjusted_grids > 1:
            implied_spacing = ((grid_upper / grid_lower) ** (1.0 / max(1, adjusted_grids - 1)) - 1.0) * 100
            # Re-apply max_spacing_pct cap: if exceeded, increase grid count
            while implied_spacing > max_spacing_pct and adjusted_grids < max_grids:
                adjusted_grids += 1
                implied_spacing = ((grid_upper / grid_lower) ** (1.0 / max(1, adjusted_grids - 1)) - 1.0) * 100
            adjusted_spacing = round(implied_spacing, 4)
        else:
            # Fallback: multiplicative (backward compat for callers without bounds)
            adjusted_spacing = round(grid_spacing_pct * spacing_mult, 4)

        return (adjusted_grids, adjusted_spacing)

    def estimate_expected_return(
        self,
        num_grids: int,
        profit_per_grid: float,
        fills_per_hour: float = 2.0,
        hours: Optional[float] = None  # Uses MAX_HOLDING_SECONDS from config
    ) -> float:
        """
        Estimate expected net return percentage.
        
        Simple model: assume steady-state grid operation
        with average fill rate based on volatility.
        
        Args:
            num_grids: Number of grid levels
            profit_per_grid: Net profit per grid (percentage)
            fills_per_hour: Estimated fills per hour
            hours: Total operation hours
        
        Returns:
            Expected return as percentage
        """
        # Default to max holding time from config
        if hours is None:
            hours = get_config().grid.max_holding_seconds / 3600  # Convert seconds to hours
        
        # Total expected fills (both entry and exit)
        total_round_trips = fills_per_hour * hours
        
        # Keep active-grid utilization aligned with PnLRanker/EV profile logic.
        active_fraction = min(0.75, (max(float(num_grids), 0.0) / 50.0) ** 0.5 * 0.5)
        
        expected_return = total_round_trips * profit_per_grid * active_fraction
        
        return expected_return
    
    def generate_params(
        self,
        validation_result: ValidationResult,
        range_prob: Optional[float] = None,
        trend_prob: Optional[float] = None,
        survival_prob: Optional[float] = None,
        hurst_exponent: Optional[float] = None,
        min_profit_pct: Optional[float] = None,
        capital_base: Optional[float] = None,
    ) -> GridParams:
        """
        Generate complete grid parameters from validation result.

        Only generates if validation passed.
        Uses 1M ATR for execution-level calibration.

        AFML Enhancement: applies regime-aware grid geometry adjustment when
        regime metrics are provided. Final live sizing is handled later in the
        enrichment pipeline.

        Args:
            validation_result: Result from regime validation
            range_prob: HMM range probability (for geometry context)
            trend_prob: HMM trend probability (for grid widening logic)
            survival_prob: Stochastic survival probability (optional)
            hurst_exponent: Hurst exponent (optional)

        Returns:
            GridParams object with grid geometry and profitability fields
        """
        if not validation_result.is_valid:
            return GridParams(
                symbol=validation_result.symbol,
                is_valid=False,
                reason="regime_validation_failed",
            )
        
        # Extract validated values
        range_high = validation_result.range_high
        range_low = validation_result.range_low
        current_price = validation_result.current_price
        atr_1m = validation_result.atr_1m
        
        if range_high is None or range_low is None or current_price is None or atr_1m is None:
            return GridParams(
                symbol=validation_result.symbol,
                is_valid=False,
                reason="missing_range_or_price",
            )
        
        # Calculate grid range (no margin — range already robust via quantile/ATR logic)
        grid_lower, grid_upper = self.calculate_grid_range(range_high, range_low, margin_pct=0.0)
        range_size = grid_upper - grid_lower
        range_size_pct = range_size / current_price if current_price > 0 else 0.0
        
        # Calculate spacing
        spacing_pct = self.calculate_grid_spacing(
            atr_1m, current_price, range_size,
            min_profit_pct_override=min_profit_pct,
        )
        
        # Calculate initial number of grids by inverting the geometric identity
        num_grids = self.calculate_num_grids(grid_lower, grid_upper, spacing_pct)

        cfg = get_config()

        # Select leverage
        effective_capital = (
            float(capital_base)
            if capital_base is not None
            else float(cfg.grid.capital)
        )

        leverage = self.select_leverage(
            range_size_pct,
            cfg.grid.stop_loss_pct,
            current_price
        )

        # AFML Enhancement: Apply regime-aware grid adjustment only.
        # Final spacing is always derived from the geometric identity (bounds / count).
        sizing_reason = "deferred_to_live_sizers"
        _max_spacing_pct_pct = float(cfg.grid.max_spacing_pct) * 100.0  # cap in percentage form

        if range_prob is not None and trend_prob is not None:
            adjusted_num_grids, adjusted_spacing_pct = self.compute_regime_adjusted_grids(
                num_grids=num_grids,
                grid_spacing_pct=spacing_pct * 100,
                trend_prob=trend_prob,
                hurst_exponent=hurst_exponent,
                grid_lower=grid_lower,
                grid_upper=grid_upper,
                max_spacing_pct=_max_spacing_pct_pct,
            )
        else:
            adjusted_num_grids = num_grids
            # Derive spacing from geometric identity (single source of truth)
            if grid_lower > 0 and adjusted_num_grids > 1:
                adjusted_spacing_pct = round(
                    ((grid_upper / grid_lower) ** (1.0 / max(1, adjusted_num_grids - 1)) - 1.0) * 100, 4
                )
                # Enforce cap even without regime adjustment (int truncation can widen spacing)
                cfg_max_grids = cfg.grid.max_grids
                while adjusted_spacing_pct > _max_spacing_pct_pct and adjusted_num_grids < cfg_max_grids:
                    adjusted_num_grids += 1
                    adjusted_spacing_pct = round(
                        ((grid_upper / grid_lower) ** (1.0 / max(1, adjusted_num_grids - 1)) - 1.0) * 100, 4
                    )
            else:
                adjusted_spacing_pct = spacing_pct * 100

        # Compute profitability ONCE from final geometry. Under geometric grid
        # identity profit per grid is uniform; the calculate_profit_per_grid tuple
        # returns (p, p, p). We assign once to keep the API stable.
        _ppg_uniform, _, _ = self.calculate_profit_per_grid(
            grid_lower, grid_upper, adjusted_num_grids
        )
        profit_per_grid_pct = _ppg_uniform * 100

        # Enforce minimum net profit per grid on FINAL geometry
        min_profit_threshold = min_profit_pct if min_profit_pct is not None else float(cfg.grid.profit_grid_min_pct_static_fallback)
        if profit_per_grid_pct < min_profit_threshold:
            # ERR-091: detect the structurally-unsatisfiable configuration.
            # calculate_grid_spacing silently clamps spacing to max_spacing_pct
            # (line ~185); when the demanded floor exceeds the profit achievable
            # at that cap, NO geometry can pass -- report it as such instead of
            # a generic below-min rejection. Ceiling mirrors
            # calculate_profit_per_grid: profit = (1-c)*r - 1 - c at r = 1+cap.
            _close_fee_mode = str(getattr(cfg.grid, "close_fee_mode", "maker")).lower()
            _close_fee = cfg.grid.maker_fee if _close_fee_mode == "maker" else cfg.grid.taker_fee
            _c = max(0.0, (float(cfg.grid.maker_fee) + float(_close_fee)) / 2.0)
            _cap_r = 1.0 + float(cfg.grid.max_spacing_pct)
            _ceiling_pct = ((1.0 - _c) * _cap_r - 1.0 - _c) * 100.0
            if min_profit_threshold > _ceiling_pct:
                return GridParams(
                    symbol=validation_result.symbol,
                    is_valid=False,
                    reason=(
                        f"profit_floor_exceeds_spacing_cap"
                        f"(floor={min_profit_threshold:.2f}%>ceiling={_ceiling_pct:.4f}%"
                        f"@max_spacing={float(cfg.grid.max_spacing_pct) * 100:.2f}%)"
                    ),
                )
            return GridParams(
                symbol=validation_result.symbol,
                is_valid=False,
                reason=f"profit_per_grid_below_min({profit_per_grid_pct:.4f}%<{min_profit_threshold:.2f}%)",
            )

        # Grid geometry uses the caller's effective base capital. Final live
        # capital_fraction is applied later by Kelly and PositionSizer.
        adjusted_capital = effective_capital
        adjusted_notional = adjusted_capital * leverage

        # Estimate expected return (uniform profit per grid under geometric identity)
        expected_return = self.estimate_expected_return(
            adjusted_num_grids, _ppg_uniform
        )

        return GridParams(
            symbol=validation_result.symbol,
            is_valid=True,
            reason="ok",
            grid_lower=round(grid_lower, 6),
            grid_upper=round(grid_upper, 6),
            num_grids=adjusted_num_grids,
            grid_spacing_pct=adjusted_spacing_pct,
            profit_per_grid_pct=profit_per_grid_pct,
            capital=adjusted_capital,
            leverage=leverage,
            total_notional=adjusted_notional,
            stop_loss_pct=cfg.grid.stop_loss_pct,
            take_profit_pct=cfg.grid.take_profit_pct,
            max_holding_time=cfg.grid.max_holding_time,
            expected_net_return_pct=expected_return * 100,
            maker_fee=cfg.grid.maker_fee,
            taker_fee=cfg.grid.taker_fee,
            # Live sizing is deferred to enrichment-time Kelly/PositionSizer.
            capital_fraction=None,
            regime_confidence=range_prob,
            sizing_reason=sizing_reason,
        )
