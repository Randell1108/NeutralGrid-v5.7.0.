"""
Microstructure-Aware Penalties for AFML-Compliant Grid Validation.

Neutral grids are highly sensitive to:
- Spread / slippage
- Fill quality
- Fee mode (maker vs taker mix)
- Funding drift

This module estimates effective cost per round trip and applies penalties
to filter "paper PnL" candidates that fail net of microstructure.

Design principles:
- Conservative cost estimation (overestimate rather than underestimate)
- Dynamic spread estimation from order book
- Funding regime penalties
- PROFIT_GRID_MIN must exceed cost * safety_multiple
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from neutralgrid.core.constants import BOT_HORIZON_HOURS


@dataclass(frozen=True)
class MicrostructureConfig:
    """Configuration for microstructure cost estimation."""

    # Fee rates (from config)
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0005  # 0.05%

    # Expected maker/taker mix for grid fills
    # Grids often fill as maker, but exits can be taker
    maker_fill_ratio: float = 0.7  # 70% maker, 30% taker

    # Slippage multiplier on spread
    # Actual execution often worse than mid-price
    slippage_mult: float = 1.5

    # Safety multiple for profit requirement
    # profit_per_grid must exceed cost * safety_multiple
    safety_multiple: float = 1.5

    # Funding rate extremes
    funding_extreme_threshold: float = 0.003  # 0.3%
    funding_penalty_mult: float = 2.0  # Double the penalty for extreme funding

    # Minimum order book depth for valid spread estimate
    min_depth_usdt: float = 10000.0  # $10k on each side


@dataclass
class MicrostructureCosts:
    """Decomposed microstructure cost components."""

    # Per-trade costs
    spread_cost_pct: float  # Half-spread as percentage
    slippage_cost_pct: float  # Estimated slippage
    fee_cost_pct: float  # Blended maker/taker fee

    # Round-trip costs
    round_trip_cost_pct: float  # Total cost for one grid cycle

    # Position costs
    funding_cost_pct: float  # Expected funding cost over horizon
    funding_penalty: float  # Additional penalty for extreme funding

    # Derived thresholds
    min_profit_required_pct: float  # Minimum profit per grid to be viable

    # Quality flags
    sufficient_liquidity: bool
    extreme_funding: bool

    # Input metadata
    inputs: Dict[str, Any] = field(default_factory=dict)


class MicrostructureEstimator:
    """
    Estimate microstructure costs for grid trading viability.

    Usage:
        estimator = MicrostructureEstimator(MicrostructureConfig())
        costs = estimator.estimate_costs(
            order_book={"bids": [...], "asks": [...]},
            funding_rate=0.001,
            volatility_pct=2.0,
        )
        if costs.min_profit_required_pct > profit_per_grid:
            print("Grid not viable after microstructure costs")
    """

    def __init__(self, config: Optional[MicrostructureConfig] = None):
        """
        Initialize MicrostructureEstimator.

        Args:
            config: MicrostructureConfig with fee rates, thresholds, etc.
        """
        self.config = config or MicrostructureConfig()

    def estimate_spread_from_orderbook(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        depth_usdt: Optional[float] = None,
    ) -> tuple[float, float, bool]:
        """
        Estimate effective spread from order book.

        Args:
            bids: List of [price, quantity] for bids
            asks: List of [price, quantity] for asks
            depth_usdt: Required depth in USDT for valid estimate

        Returns:
            Tuple of (mid_price, spread_pct, sufficient_liquidity)
        """
        if depth_usdt is None:
            depth_usdt = self.config.min_depth_usdt

        if not bids or not asks:
            return (0.0, 0.01, False)  # 1% default spread, insufficient liquidity

        try:
            # Best bid and ask
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])

            if best_bid <= 0 or best_ask <= 0:
                return (0.0, 0.01, False)

            mid_price = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
            spread_pct = spread / mid_price

            # Check liquidity depth
            bid_depth = sum(float(b[0]) * float(b[1]) for b in bids[:10])
            ask_depth = sum(float(a[0]) * float(a[1]) for a in asks[:10])
            sufficient_liquidity = min(bid_depth, ask_depth) >= depth_usdt

            return (mid_price, spread_pct, sufficient_liquidity)

        except (IndexError, ValueError, TypeError):
            return (0.0, 0.01, False)

    def estimate_slippage(
        self,
        spread_pct: float,
        volatility_pct: float,
    ) -> float:
        """
        Estimate slippage based on spread and volatility.

        Higher volatility = more slippage (order book thins out).

        Args:
            spread_pct: Current spread as percentage
            volatility_pct: Recent volatility as percentage

        Returns:
            Estimated slippage as percentage
        """
        # Base slippage is a fraction of spread
        base_slippage = spread_pct * 0.5

        # Volatility adjustment (higher vol = worse fills)
        vol_adjustment = 1.0 + (volatility_pct / 5.0) ** 0.5  # Sublinear scaling

        slippage = base_slippage * vol_adjustment * self.config.slippage_mult

        return slippage

    def estimate_blended_fee(self) -> float:
        """
        Estimate blended fee based on expected maker/taker mix.

        Returns:
            Blended fee rate as decimal
        """
        blended = (
            self.config.maker_fee * self.config.maker_fill_ratio +
            self.config.taker_fee * (1.0 - self.config.maker_fill_ratio)
        )
        return blended

    def estimate_funding_cost(
        self,
        funding_rate: float,
        horizon_hours: float = BOT_HORIZON_HOURS,
        funding_interval_hours: float = 8.0,
    ) -> tuple[float, bool]:
        """
        Estimate funding cost over horizon.

        Args:
            funding_rate: Current funding rate per interval
            horizon_hours: Position holding horizon
            funding_interval_hours: Funding interval (usually 8h)

        Returns:
            Tuple of (funding_cost_pct, is_extreme)
        """
        if funding_rate is None:
            return (0.0, False)

        # Number of funding periods in horizon
        num_periods = horizon_hours / funding_interval_hours

        # Total funding cost (can be positive or negative)
        # For grid bots, we're typically net long/short around mid-range
        # Assume average 50% position exposure
        avg_exposure = 0.5
        funding_cost = abs(funding_rate) * num_periods * avg_exposure

        # Check if extreme
        is_extreme = abs(funding_rate) > self.config.funding_extreme_threshold

        return (funding_cost * 100.0, is_extreme)  # Return as percentage

    def estimate_costs(
        self,
        order_book: Optional[Dict[str, List]] = None,
        funding_rate: Optional[float] = None,
        volatility_pct: float = 2.0,
        horizon_hours: float = BOT_HORIZON_HOURS,
    ) -> MicrostructureCosts:
        """
        Estimate full microstructure costs.

        Args:
            order_book: Dict with "bids" and "asks" lists
            funding_rate: Current funding rate
            volatility_pct: Recent volatility as percentage
            horizon_hours: Position holding horizon

        Returns:
            MicrostructureCosts with decomposed cost components
        """
        # Spread estimation
        if order_book:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            _mid_price, spread_pct, sufficient_liquidity = self.estimate_spread_from_orderbook(
                bids, asks
            )
        else:
            # Fallback: use volatility-based spread estimate
            spread_pct = volatility_pct * 0.01  # 1% of volatility as spread
            sufficient_liquidity = True  # Assume OK without order book

        # Half-spread cost (pay spread on entry)
        spread_cost = spread_pct / 2.0

        # Slippage estimation
        slippage_cost = self.estimate_slippage(spread_pct, volatility_pct)

        # Fee cost
        fee_cost = self.estimate_blended_fee()

        # Round-trip cost: entry spread + exit spread + 2x fees + slippage
        round_trip_cost = spread_cost * 2 + fee_cost * 2 + slippage_cost * 2

        # Funding cost
        funding_cost, extreme_funding = self.estimate_funding_cost(
            funding_rate if funding_rate is not None else 0.0, horizon_hours
        )

        # Funding penalty (additional cost for extreme funding)
        funding_penalty = 0.0
        if extreme_funding:
            funding_penalty = funding_cost * self.config.funding_penalty_mult

        # Minimum profit required (with safety multiple)
        min_profit_required = round_trip_cost * self.config.safety_multiple * 100.0

        return MicrostructureCosts(
            spread_cost_pct=spread_cost * 100.0,
            slippage_cost_pct=slippage_cost * 100.0,
            fee_cost_pct=fee_cost * 100.0,
            round_trip_cost_pct=round_trip_cost * 100.0,
            funding_cost_pct=funding_cost,
            funding_penalty=funding_penalty,
            min_profit_required_pct=min_profit_required,
            sufficient_liquidity=sufficient_liquidity,
            extreme_funding=extreme_funding,
            inputs={
                "spread_pct": spread_pct * 100.0,
                "volatility_pct": volatility_pct,
                "funding_rate": funding_rate,
                "horizon_hours": horizon_hours,
            },
        )

    def compute_dynamic_profit_floor(
        self,
        costs: MicrostructureCosts,
        volatility_pct: float = 2.0,
        trend_prob: float = 0.0,
        survival_prob: float = 0.7,
        leverage: int = 10,
        horizon_hours: float = BOT_HORIZON_HOURS,
    ) -> tuple[float, str, dict]:
        """
        Compute dynamic minimum profit-per-grid covering all-in expected costs.

        The correct minimum profit-per-grid covers:
        1. Fee floor: blended maker/taker round-trip fees
        2. Spread/slippage floor: order book spread + execution slippage
        3. Adverse selection premium: increases with vol and trend risk
        4. Safety margin: increases with leverage, decreases with survivability

        The floor increases when:
        - Spread is wider (worse fills)
        - Volatility is higher (more adverse selection risk)
        - Liquidity is worse (bigger slippage)
        - Trend probability is higher (directional risk)
        - Leverage is higher (liquidation proximity, funding effects)
        - Survival probability is lower (range breakout risk)

        Args:
            costs: MicrostructureCosts from estimate_costs()
            volatility_pct: Recent volatility as percentage
            trend_prob: HMM posterior probability of trend regime [0, 1]
            survival_prob: P(price stays in range) over horizon [0, 1]
            leverage: Current leverage multiplier
            horizon_hours: Holding horizon in hours

        Returns:
            Tuple of (dynamic_floor_pct, tier_name, components_dict)
        """
        from neutralgrid.core.config import get_config
        _grid = get_config().grid

        # --- Component 1: Fee floor (round-trip) ---
        fee_rt_pct = self.estimate_blended_fee() * 2.0 * 100.0  # as %

        # --- Component 2: Spread / slippage floor ---
        spread_rt_pct = costs.spread_cost_pct * 2.0  # round-trip half-spread
        slippage_pct = costs.slippage_cost_pct * 2.0  # round-trip slippage
        spread_slippage_floor = spread_rt_pct + slippage_pct

        # --- Component 3: Adverse selection premium ---
        # Higher vol → fills get picked off by informed flow
        # Scale: ~5% of vol (e.g. 2% vol → 0.10% premium)
        vol_premium = volatility_pct * 0.05
        # Higher trend_prob → directional adverse selection
        # Scale: up to 0.30% at trend_prob=1.0
        trend_premium = trend_prob * 0.30
        adverse_selection = vol_premium + trend_premium

        # --- Component 4: Safety margin (multiplicative) ---
        # Base safety: 20% buffer above breakeven
        base_safety = 1.20

        # Leverage factor: +2% per unit of leverage above 5x
        # At 10x: 1.0 + 5*0.02 = 1.10  (10% extra)
        # At 20x: 1.0 + 15*0.02 = 1.30 (30% extra)
        leverage_factor = 1.0 + max(0, leverage - 5) * 0.02

        # Survival factor: penalize when breakout risk is high
        # survival=0.7+ → 1.0 (no extra), survival=0.3 → 1.20 (20% extra)
        survival_factor = 1.0 + max(0.0, 0.70 - survival_prob) * 0.50

        # Horizon factor: longer holding → more cumulative risk
        # Baseline = H* (BOT_HORIZON_HOURS); each extra H* adds 5%
        horizon_factor = 1.0 + max(0.0, horizon_hours - BOT_HORIZON_HOURS) / BOT_HORIZON_HOURS * 0.05

        safety_mult = base_safety * leverage_factor * survival_factor * horizon_factor

        # --- Combine ---
        raw_cost_pct = fee_rt_pct + spread_slippage_floor + adverse_selection
        dynamic_floor_pct = raw_cost_pct * safety_mult

        # Absolute minimum: never below fee-only floor × 1.5
        absolute_min = fee_rt_pct * 1.5
        # Reference static fallback (not enforced; dynamic floor is primary)
        static_fallback = float(_grid.profit_grid_min_pct_static_fallback)
        dynamic_floor_pct = max(dynamic_floor_pct, absolute_min)

        # Determine tier label for logging
        spread = costs.spread_cost_pct
        lo, hi = _grid.liquidity_tier_spread_thresholds
        if spread <= lo:
            tier = "high"
        elif spread <= hi:
            tier = "mid"
        else:
            tier = "low"

        components = {
            "fee_rt_pct": round(fee_rt_pct, 4),
            "spread_slippage_pct": round(spread_slippage_floor, 4),
            "adverse_selection_pct": round(adverse_selection, 4),
            "vol_premium_pct": round(vol_premium, 4),
            "trend_premium_pct": round(trend_premium, 4),
            "safety_mult": round(safety_mult, 4),
            "leverage_factor": round(leverage_factor, 4),
            "survival_factor": round(survival_factor, 4),
            "horizon_factor": round(horizon_factor, 4),
            "raw_cost_pct": round(raw_cost_pct, 4),
            "dynamic_floor_pct": round(dynamic_floor_pct, 4),
            "static_fallback_pct": static_fallback,
        }

        return (round(dynamic_floor_pct, 4), tier, components)

    def is_viable(
        self,
        costs: MicrostructureCosts,
        profit_per_grid_pct: float,
    ) -> tuple[bool, str]:
        """
        Check if grid is viable after microstructure costs.

        Args:
            costs: MicrostructureCosts from estimate_costs()
            profit_per_grid_pct: Expected profit per grid

        Returns:
            Tuple of (is_viable, reason)
        """
        # Check profit covers round-trip costs with safety margin
        if profit_per_grid_pct < costs.min_profit_required_pct:
            return (
                False,
                f"profit({profit_per_grid_pct:.2f}%)<min_required({costs.min_profit_required_pct:.2f}%)"
            )

        # Check liquidity
        if not costs.sufficient_liquidity:
            return (False, "insufficient_liquidity")

        # Check extreme funding (warning, not rejection)
        if costs.extreme_funding:
            # Still viable but with warning
            return (True, f"warning:extreme_funding({costs.funding_cost_pct:.2f}%)")

        return (True, "ok")
