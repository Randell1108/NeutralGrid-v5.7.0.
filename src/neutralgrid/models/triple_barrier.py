"""
Triple Barrier Method for AFML-Compliant Labeling.

Implements the triple barrier labeling from AFML Chapter 3:
- Profit-taking barrier: +15% (PT)
- Stop-loss barrier: -12% (SL)
- Time barrier: 6h

Labels:
- 1 (positive): Hit PT before SL or time
- 0 (neutral): Hit time barrier before PT or SL
- -1 (negative): Hit SL before PT or time

This provides more informative labels than simple binary (win/lose)
by distinguishing between:
- Clear winners (hit profit target)
- Time decay (neither PT nor SL, grid timed out)
- Clear losers (hit stop loss)

AFML Compliance Updates:
- Uses HIGH prices for PT detection (barrier may be touched intrabar)
- Uses LOW prices for SL detection (barrier may be touched intrabar)
- Stores t1 (vertical barrier timestamp) for proper purging
- Handles same-bar double-touch with conservative tie-break

Design principles:
- Aligned with grid bot constraints (PT=+15%, SL=-12%, Time=6h)
- No look-ahead bias (uses future prices for labeling only during training)
- Deterministic labeling for reproducibility
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple, Literal
import numpy as np
import pandas as pd
import logging

from neutralgrid.core.constants import BOT_HORIZON_HOURS
from .barrier_config import UnifiedBarrierConfig, get_barrier_config

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class TripleBarrierConfig:
    """
    Configuration for triple barrier labeling.

    NOTE: For new code, prefer using UnifiedBarrierConfig from barrier_config.py.
    This class is maintained for backwards compatibility.
    """

    # Profit-taking barrier (positive return threshold)
    pt_pct: float = 15.0  # +15%

    # Stop-loss barrier (negative return threshold)
    sl_pct: float = -12.0  # -12%

    # Time barrier (maximum holding period in hours)
    time_barrier_hours: float = BOT_HORIZON_HOURS

    # Timeframe for price data (for bar counting)
    timeframe_minutes: int = 15  # 15-minute bars

    # Dynamic volatility-scaled barriers (AFML Ch 3)
    # When set, PT = pt_vol_multiplier * daily_vol, SL = -sl_vol_multiplier * daily_vol
    # Falls back to fixed pt_pct/sl_pct when daily_vol is not provided at labeling time
    pt_vol_multiplier: Optional[float] = None
    sl_vol_multiplier: Optional[float] = None

    @property
    def time_barrier_bars(self) -> int:
        """Convert time barrier to number of bars."""
        return int(self.time_barrier_hours * 60 / self.timeframe_minutes)

    @classmethod
    def from_unified(cls, unified: UnifiedBarrierConfig) -> "TripleBarrierConfig":
        """Create from UnifiedBarrierConfig."""
        return cls(
            pt_pct=unified.pt_pct,
            sl_pct=unified.sl_pct,
            time_barrier_hours=unified.time_barrier_hours,
            timeframe_minutes=unified.timeframe_minutes,
            pt_vol_multiplier=unified.k_pt if unified.use_volatility_scaling else None,
            sl_vol_multiplier=unified.k_sl if unified.use_volatility_scaling else None,
        )


# =============================================================================
# LABEL RESULT
# =============================================================================

@dataclass
class TripleBarrierLabel:
    """
    Result of triple barrier labeling for a single entry.

    Attributes:
        label: Triple-barrier label (1=PT hit, 0=time, -1=SL hit)
        barrier_hit: Which barrier was hit ("pt", "sl", "time")
        bars_to_hit: Number of bars until barrier was hit
        return_pct: Gross return percentage at exit (before funding)
        net_return_pct: Net return percentage (after funding drag)
        funding_cost_pct: Estimated funding cost over holding period
        entry_price: Price at entry
        exit_price: Price at exit (barrier touch price)
        entry_bar: Index of entry bar
        exit_bar: Index of exit bar
        t1: Timestamp of exit (vertical barrier end time)
        double_touch: True if both PT and SL were touched in same bar
    """

    # Label: 1 (profit), 0 (time), -1 (loss)
    label: int

    # Which barrier was hit
    barrier_hit: str  # "pt", "sl", "time"

    # Bars until barrier hit
    bars_to_hit: int

    # Return at barrier hit (gross, before funding)
    return_pct: float

    # Entry and exit info (required)
    entry_price: float
    exit_price: float
    entry_bar: int
    exit_bar: int

    # Net return after funding drag (AFML-compliant)
    net_return_pct: float = 0.0

    # Funding cost estimate over holding period
    funding_cost_pct: float = 0.0

    # AFML additions
    t1: Optional[datetime] = None  # Exit timestamp (vertical barrier time)
    double_touch: bool = False  # Same-bar PT and SL touch


# =============================================================================
# DOUBLE TOUCH HANDLING
# =============================================================================

DoubleTouchPolicy = Literal["drop", "sl_first", "pt_first"]


# =============================================================================
# LABELER CLASS
# =============================================================================

class TripleBarrierLabeler:
    """
    Apply triple barrier method to generate labels.

    AFML Compliance:
    - Uses HIGH prices for PT barrier detection (catches intrabar touches)
    - Uses LOW prices for SL barrier detection (catches intrabar touches)
    - Stores t1 (exit timestamp) for proper purging/embargo
    - Handles same-bar double-touch with configurable policy

    Usage:
        labeler = TripleBarrierLabeler(TripleBarrierConfig())

        # Label a single entry (close-only, backwards compatible)
        label = labeler.label_entry(
            closes=close_prices,
            entry_bar=100,
        )

        # Label with high/low for accurate barrier detection (AFML-compliant)
        label = labeler.label_entry(
            closes=close_prices,
            entry_bar=100,
            highs=high_prices,
            lows=low_prices,
        )

        # Label entire DataFrame with OHLC
        labels = labeler.label_dataframe_ohlc(
            df=ohlc_df,
            entry_indices=[100, 200, 300],
        )
    """

    def __init__(
        self,
        config: Optional[TripleBarrierConfig] = None,
        double_touch_policy: DoubleTouchPolicy = "sl_first",
    ):
        """
        Initialize TripleBarrierLabeler.

        Args:
            config: TripleBarrierConfig with barrier levels. If None, uses
                    defaults from UnifiedBarrierConfig.
            double_touch_policy: How to handle same-bar PT and SL touches:
                - "drop": Skip the sample (returns None)
                - "sl_first": Assume SL hit first (conservative, recommended)
                - "pt_first": Assume PT hit first (optimistic)
        """
        if config is None:
            unified = get_barrier_config()
            config = TripleBarrierConfig.from_unified(unified)
        self.config = config
        self.double_touch_policy = double_touch_policy

    def label_entry(
        self,
        closes: np.ndarray,
        entry_bar: int,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        funding_rate: Optional[float] = None,
        funding_interval_hours: float = 8.0,
        daily_vol: Optional[float] = None,
    ) -> Optional[TripleBarrierLabel]:
        """
        Apply triple barrier to a single entry point.

        AFML Compliance:
        - If highs/lows provided, uses them for barrier detection
        - PT is checked against highs (catches intrabar touches on wicks)
        - SL is checked against lows (catches intrabar touches on wicks)
        - If highs/lows not provided, falls back to close-only (backwards compatible)
        - If funding_rate provided, computes net return after funding drag
        - If daily_vol provided and vol multipliers configured, uses dynamic barriers

        Args:
            closes: Array of close prices
            entry_bar: Index of entry bar
            highs: Array of high prices (optional, for AFML-compliant detection)
            lows: Array of low prices (optional, for AFML-compliant detection)
            timestamps: Array of timestamps (optional, for t1 computation)
            funding_rate: Current funding rate per interval (optional, for AFML-compliant net returns)
            funding_interval_hours: Funding interval in hours (default 8h for Binance)
            daily_vol: Daily volatility (e.g. ewm std of returns) for dynamic barrier scaling (AFML Ch 3)

        Returns:
            TripleBarrierLabel or None if insufficient data or dropped
        """
        n = len(closes)

        if entry_bar < 0 or entry_bar >= n:
            return None

        entry_price = closes[entry_bar]
        if entry_price <= 0:
            return None

        # Calculate barrier prices
        # Use dynamic volatility-scaled barriers if daily_vol and multipliers are set
        if daily_vol is not None and self.config.pt_vol_multiplier is not None:
            pt_pct = self.config.pt_vol_multiplier * daily_vol * 100.0
        else:
            pt_pct = self.config.pt_pct

        if daily_vol is not None and self.config.sl_vol_multiplier is not None:
            sl_pct = -(self.config.sl_vol_multiplier * daily_vol * 100.0)
        else:
            sl_pct = self.config.sl_pct

        pt_price = entry_price * (1.0 + pt_pct / 100.0)
        sl_price = entry_price * (1.0 + sl_pct / 100.0)

        # Maximum bar to check (time barrier)
        max_bar = min(entry_bar + self.config.time_barrier_bars, n - 1)

        # Use high/low if provided, otherwise fall back to close
        check_high = highs if highs is not None else closes
        check_low = lows if lows is not None else closes

        # Track which barrier is hit first
        hit_bar = max_bar
        hit_price = closes[max_bar]  # Use close for exit price
        barrier_hit = "time"
        label = 0  # Default: time barrier
        double_touch = False

        # Check each bar for PT or SL hit
        for bar in range(entry_bar + 1, max_bar + 1):
            high = check_high[bar]
            low = check_low[bar]

            # Check for same-bar double touch
            pt_touched = high >= pt_price
            sl_touched = low <= sl_price

            if pt_touched and sl_touched:
                # Same bar touched both barriers - cannot determine order from OHLC
                double_touch = True

                if self.double_touch_policy == "drop":
                    logger.debug(
                        f"Double touch at bar {bar}: dropping sample "
                        f"(high={high:.2f} >= PT={pt_price:.2f}, "
                        f"low={low:.2f} <= SL={sl_price:.2f})"
                    )
                    return None

                elif self.double_touch_policy == "sl_first":
                    # Conservative: assume SL hit first
                    hit_bar = bar
                    hit_price = sl_price
                    barrier_hit = "sl"
                    label = -1
                    break

                else:  # pt_first
                    # Optimistic: assume PT hit first
                    hit_bar = bar
                    hit_price = pt_price
                    barrier_hit = "pt"
                    label = 1
                    break

            elif pt_touched:
                # PT barrier touched (use barrier level as fill price)
                hit_bar = bar
                hit_price = pt_price  # Exit at the barrier level, not the wick extreme
                barrier_hit = "pt"
                label = 1
                break

            elif sl_touched:
                # SL barrier touched (use barrier level as fill price)
                hit_bar = bar
                hit_price = sl_price  # Exit at the barrier level, not the wick extreme
                barrier_hit = "sl"
                label = -1
                break

        # Calculate return at exit (gross, before funding)
        return_pct = (hit_price - entry_price) / entry_price * 100.0

        # Compute t1 if timestamps provided
        t1 = None
        if timestamps is not None and len(timestamps) > hit_bar:
            raw_t1 = timestamps[hit_bar]
            parsed_t1 = pd.to_datetime(raw_t1, utc=True, errors="coerce")
            if isinstance(parsed_t1, pd.Timestamp) and not pd.isna(parsed_t1):
                t1 = parsed_t1.to_pydatetime()

        # Calculate funding cost if funding_rate provided (AFML-compliant)
        # Sign convention: for long positions, positive funding_rate = cost,
        # negative funding_rate = credit.  Grid bots are net-long on average,
        # so we preserve the sign: cost when rate > 0, credit when rate < 0.
        funding_cost_pct = 0.0
        if funding_rate is not None:
            bars_held = hit_bar - entry_bar
            hours_held = bars_held * self.config.timeframe_minutes / 60.0
            num_funding_periods = hours_held / funding_interval_hours
            # Assume average 50% position exposure (grid bots oscillate around midpoint)
            avg_exposure = 0.5
            # Factor in leverage: funding is paid on notional (capital * leverage)
            try:
                from neutralgrid.core.config import get_config
                leverage = float(get_config().grid.leverage_max)
            except (ImportError, AttributeError, ValueError, TypeError):
                leverage = 10.0
            funding_cost_pct = funding_rate * num_funding_periods * avg_exposure * leverage * 100.0

        # Net return after funding drag (positive cost reduces return,
        # negative cost i.e. credit increases return)
        net_return_pct = return_pct - funding_cost_pct

        return TripleBarrierLabel(
            label=label,
            barrier_hit=barrier_hit,
            bars_to_hit=hit_bar - entry_bar,
            return_pct=return_pct,
            net_return_pct=net_return_pct,
            funding_cost_pct=funding_cost_pct,
            entry_price=entry_price,
            exit_price=hit_price,
            entry_bar=entry_bar,
            exit_bar=hit_bar,
            t1=t1,
            double_touch=double_touch,
        )

    def label_entry_with_ohlc(
        self,
        df: pd.DataFrame,
        entry_bar: int,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        time_col: Optional[str] = None,
        funding_rate: Optional[float] = None,
        daily_vol: Optional[float] = None,
    ) -> Optional[TripleBarrierLabel]:
        """
        Label using full OHLC data for accurate barrier detection.

        This is the recommended method for AFML-compliant labeling as it
        uses highs for PT detection and lows for SL detection.

        Args:
            df: DataFrame with OHLC data
            entry_bar: Index of entry bar
            open_col: Column name for open prices
            high_col: Column name for high prices
            low_col: Column name for low prices
            close_col: Column name for close prices
            time_col: Column name for timestamps (or use index if DatetimeIndex)
            funding_rate: Current funding rate per interval (optional)
            daily_vol: Daily volatility for dynamic barrier scaling (optional, AFML Ch 3)

        Returns:
            TripleBarrierLabel or None if insufficient data
        """
        # Extract timestamps
        timestamps = None
        if time_col and time_col in df.columns:
            timestamps = np.asarray(df[time_col])
        elif isinstance(df.index, pd.DatetimeIndex):
            timestamps = np.asarray(df.index)

        return self.label_entry(
            closes=np.asarray(df[close_col]),
            entry_bar=entry_bar,
            highs=np.asarray(df[high_col]),
            lows=np.asarray(df[low_col]),
            timestamps=timestamps,
            funding_rate=funding_rate,
            daily_vol=daily_vol,
        )

    def label_dataframe(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        entry_indices: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Apply triple barrier to multiple entries in a DataFrame (close-only).

        NOTE: For AFML-compliant labeling with high/low detection, use
        label_dataframe_ohlc() instead.

        If entry_indices not provided, labels every bar as potential entry.

        Args:
            df: DataFrame with price data
            price_col: Column name for prices
            entry_indices: Optional list of entry bar indices

        Returns:
            DataFrame with triple barrier labels
        """
        prices = np.asarray(df[price_col])

        if entry_indices is None:
            # Label every bar (except last time_barrier_bars)
            entry_indices = list(range(len(df) - self.config.time_barrier_bars))

        results = []

        for entry_bar in entry_indices:
            label_result = self.label_entry(prices, entry_bar)

            if label_result:
                results.append({
                    "entry_bar": entry_bar,
                    "label": label_result.label,
                    "barrier_hit": label_result.barrier_hit,
                    "bars_to_hit": label_result.bars_to_hit,
                    "return_pct": label_result.return_pct,
                    "net_return_pct": label_result.net_return_pct,
                    "funding_cost_pct": label_result.funding_cost_pct,
                    "entry_price": label_result.entry_price,
                    "exit_price": label_result.exit_price,
                    "exit_bar": label_result.exit_bar,
                    "t1": label_result.t1,
                    "double_touch": label_result.double_touch,
                })

        return pd.DataFrame(results)

    def label_dataframe_ohlc(
        self,
        df: pd.DataFrame,
        entry_indices: Optional[List[int]] = None,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        time_col: Optional[str] = None,
        funding_rate: Optional[float] = None,
        daily_vol: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Apply triple barrier with OHLC data for AFML-compliant labeling.

        Uses highs for PT detection and lows for SL detection to catch
        intrabar barrier touches that would be missed with close-only.

        Args:
            df: DataFrame with OHLC data
            entry_indices: Optional list of entry bar indices
            open_col: Column name for open prices
            high_col: Column name for high prices
            low_col: Column name for low prices
            close_col: Column name for close prices
            time_col: Column name for timestamps (or use index if DatetimeIndex)
            funding_rate: Current funding rate per interval (optional, for AFML-compliant net returns)
            daily_vol: Daily volatility for dynamic barrier scaling (optional, AFML Ch 3)

        Returns:
            DataFrame with triple barrier labels including t1, net_return_pct, funding_cost_pct
        """
        closes = np.asarray(df[close_col])
        highs = np.asarray(df[high_col])
        lows = np.asarray(df[low_col])

        # Extract timestamps
        timestamps = None
        if time_col and time_col in df.columns:
            timestamps = np.asarray(df[time_col])
        elif isinstance(df.index, pd.DatetimeIndex):
            timestamps = np.asarray(df.index)

        if entry_indices is None:
            # Label every bar (except last time_barrier_bars)
            entry_indices = list(range(len(df) - self.config.time_barrier_bars))

        results = []

        for entry_bar in entry_indices:
            label_result = self.label_entry(
                closes=closes,
                entry_bar=entry_bar,
                highs=highs,
                lows=lows,
                timestamps=timestamps,
                funding_rate=funding_rate,
                daily_vol=daily_vol,
            )

            if label_result:
                # Get entry timestamp if available
                entry_time = None
                if timestamps is not None and entry_bar < len(timestamps):
                    entry_time = timestamps[entry_bar]
                    if isinstance(entry_time, (np.datetime64, pd.Timestamp)):
                        entry_time = pd.Timestamp(entry_time).to_pydatetime()

                results.append({
                    "entry_bar": entry_bar,
                    "entry_time": entry_time,
                    "label": label_result.label,
                    "barrier_hit": label_result.barrier_hit,
                    "bars_to_hit": label_result.bars_to_hit,
                    "return_pct": label_result.return_pct,
                    "net_return_pct": label_result.net_return_pct,
                    "funding_cost_pct": label_result.funding_cost_pct,
                    "entry_price": label_result.entry_price,
                    "exit_price": label_result.exit_price,
                    "exit_bar": label_result.exit_bar,
                    "t1": label_result.t1,
                    "double_touch": label_result.double_touch,
                })

        return pd.DataFrame(results)

    def compute_binary_label(
        self,
        label_result: TripleBarrierLabel,
        include_time_as_positive: bool = False,
        use_net_return: bool = True,
        hurdle_pct: float = 3.0,
    ) -> int:
        """
        Convert triple barrier label to binary for meta-labeling.

        Uses a configurable hurdle (default 3%) for triple-barrier labeling.
        Note: MetaLabelerConfig.hurdle_pct defaults to 3.0% — callers should
        pass the meta-labeler's hurdle explicitly for consistency.

        Args:
            label_result: TripleBarrierLabel
            include_time_as_positive: Treat time barrier hits as positive (0 or 1)
            use_net_return: Use net_return_pct (after funding) instead of gross return
            hurdle_pct: Minimum return % to count as success (default 3.0)

        Returns:
            Binary label (0 or 1)
        """
        return_to_check = label_result.net_return_pct if use_net_return else label_result.return_pct

        if label_result.label == 1:
            # PT hit — success only if return clears the hurdle
            return 1 if return_to_check >= hurdle_pct else 0

        if label_result.label == 0 and include_time_as_positive:
            # Time barrier — success only if return clears the hurdle
            return 1 if return_to_check >= hurdle_pct else 0

        return 0

    def label_for_meta_learning(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        entry_indices: Optional[List[int]] = None,
        include_time_positive: bool = True,
        use_net_return: bool = True,
        hurdle_pct: float = 3.0,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate binary labels for meta-learning.

        AFML meta-labeling uses binary labels:
        - 1: Would trade (expected return >= hurdle_pct)
        - 0: Would skip (expected return < hurdle_pct or SL hit)

        Args:
            df: DataFrame with price data
            price_col: Column name for prices
            entry_indices: Optional list of entry bar indices
            include_time_positive: Include time barrier + positive return as success
            use_net_return: Use net_return_pct (after funding) for time barrier decisions
            hurdle_pct: Minimum return % to count as success (default 3.0).
                        Note: MetaLabelerConfig.hurdle_pct defaults to 3.0%.

        Returns:
            Tuple of (binary_labels, details_df)
        """
        labels_df = self.label_dataframe(df, price_col, entry_indices)

        if len(labels_df) == 0:
            return (np.array([]), labels_df)

        # Convert to binary using unified hurdle (AFML compliance)
        return_col = "net_return_pct" if use_net_return else "return_pct"
        binary_labels = np.array([
            1 if (row["label"] == 1 and row[return_col] >= hurdle_pct) else (
                1 if (include_time_positive and row["label"] == 0 and row[return_col] >= hurdle_pct)
                else 0
            )
            for _, row in labels_df.iterrows()
        ])

        labels_df["binary_label"] = binary_labels

        return (binary_labels, labels_df)

    def label_for_meta_learning_ohlc(
        self,
        df: pd.DataFrame,
        entry_indices: Optional[List[int]] = None,
        include_time_positive: bool = True,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        time_col: Optional[str] = None,
        funding_rate: Optional[float] = None,
        use_net_return: bool = True,
        hurdle_pct: float = 3.0,
        daily_vol: Optional[float] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate binary labels for meta-learning using OHLC data.

        AFML-compliant version that uses high/low for barrier detection
        and net returns (after funding drag) for labeling decisions.

        Args:
            df: DataFrame with OHLC data
            entry_indices: Optional list of entry bar indices
            include_time_positive: Include time barrier + positive return as success
            open_col: Column name for open prices
            high_col: Column name for high prices
            low_col: Column name for low prices
            close_col: Column name for close prices
            time_col: Column name for timestamps
            funding_rate: Current funding rate per interval (optional)
            use_net_return: Use net_return_pct (after funding) for time barrier decisions
            hurdle_pct: Minimum return % to count as success (default 3.0).
                        Note: MetaLabelerConfig.hurdle_pct defaults to 3.0%.
            daily_vol: Daily volatility for dynamic barrier scaling (optional, AFML Ch 3)

        Returns:
            Tuple of (binary_labels, details_df with t1, net_return_pct, funding_cost_pct)
        """
        labels_df = self.label_dataframe_ohlc(
            df=df,
            entry_indices=entry_indices,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            time_col=time_col,
            funding_rate=funding_rate,
            daily_vol=daily_vol,
        )

        if len(labels_df) == 0:
            return (np.array([]), labels_df)

        # Convert to binary using unified hurdle (AFML compliance)
        return_col = "net_return_pct" if use_net_return else "return_pct"
        binary_labels = np.array([
            1 if (row["label"] == 1 and row[return_col] >= hurdle_pct) else (
                1 if (include_time_positive and row["label"] == 0 and row[return_col] >= hurdle_pct)
                else 0
            )
            for _, row in labels_df.iterrows()
        ])

        labels_df["binary_label"] = binary_labels

        return (binary_labels, labels_df)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_triple_barrier_labeler(
    pt_pct: float = 15.0,
    sl_pct: float = -12.0,
    time_barrier_hours: float = BOT_HORIZON_HOURS,
    timeframe_minutes: int = 15,
    double_touch_policy: DoubleTouchPolicy = "sl_first",
) -> TripleBarrierLabeler:
    """
    Factory function to create TripleBarrierLabeler.

    Args:
        pt_pct: Profit-taking barrier percentage
        sl_pct: Stop-loss barrier percentage (negative)
        time_barrier_hours: Time barrier in hours
        timeframe_minutes: Timeframe in minutes
        double_touch_policy: How to handle same-bar PT and SL touches

    Returns:
        Configured TripleBarrierLabeler instance
    """
    config = TripleBarrierConfig(
        pt_pct=pt_pct,
        sl_pct=sl_pct,
        time_barrier_hours=time_barrier_hours,
        timeframe_minutes=timeframe_minutes,
    )
    return TripleBarrierLabeler(config, double_touch_policy=double_touch_policy)


def create_labeler_from_config() -> TripleBarrierLabeler:
    """
    Create TripleBarrierLabeler from unified config (config.py).

    This is the recommended way to create a labeler for production use
    as it ensures consistent barrier parameters across the codebase.

    Returns:
        TripleBarrierLabeler with settings from config.py
    """
    unified = get_barrier_config()
    config = TripleBarrierConfig.from_unified(unified)
    return TripleBarrierLabeler(config)
