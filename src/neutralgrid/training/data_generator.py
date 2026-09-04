"""
Training Data Generator for Meta-Labeler.

Builds supervised learning tables with:
1. Feature snapshots at decision time
2. Forward outcome labels (barrier-based, AFML-compliant)

Design principles:
- No data leakage: features computed strictly before label horizon
- Deterministic: explicit random_state, reproducible transforms
- Training-serving alignment: uses same feature functions as inference

Key AFML concepts implemented:
- Triple-barrier labeling (profit-take, stop-loss, vertical barrier)
- Purged cross-validation support (timestamps preserved)
- Feature engineering consistent with inference pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, cast
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from neutralgrid.core.config import get_config
from neutralgrid.core.constants import BOT_HORIZON_HOURS
from neutralgrid.grid.formulas import (
    profit_per_grid_pct as _ppg_pct,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TRAINING TABLE SCHEMA
# =============================================================================

@dataclass(frozen=True)
class TrainingTableSchema:
    """
    Schema definition for meta-labeler training data.

    Each row = one candidate decision at time t.

    AFML Compliance:
    - t1 column stores vertical barrier timestamp for proper purging
    - barrier_touched tracks which barrier was hit first
    - These enable time-based purging and sample weighting
    """

    # Identifiers (required)
    identifiers: Tuple[str, ...] = ("symbol", "start_time_utc")

    # AFML event metadata (required for proper CV purging and sample weighting)
    event_metadata: Tuple[str, ...] = (
        "t1",               # Vertical barrier timestamp (event end time)
        "barrier_touched",  # Which barrier hit: "pt", "sl", "time"
        "barrier_price",    # Price at barrier touch
    )

    # Feature columns (base training schema — the active meta-labeler
    # feature set is defined by ACTIVE_SNAPSHOT_META_FEATURES in
    # meta_labeler.py).
    # NOTE: hlabel is NOT a feature — it deterministically maps to y and
    # would be information leakage.  It lives in the labels tuple below.
    features: Tuple[str, ...] = (
        "range_prob",
        "trend_prob",
        "survival_prob",
        "hurst_exponent",
        "ou_halflife",
        "profit_per_grid_pct",
        "num_grids",
        "grid_spacing_pct",
        "range_size_pct",
        "adx_1h",
        "adx_15m",
        "rsi_15m",
        "funding_rate",
        "ev_score",
        "primary_pipeline_score",
    )

    # Label columns (required for training).
    # y is the primary training target: from hlabel_meta (hierarchical
    # deploy decision) when available, else from PnL hurdle.
    labels: Tuple[str, ...] = (
        "pnl_pct",        # Realized PnL over horizon (percentage)
        "sl_hit",         # Whether stop-loss was hit
        "y",              # Binary label: 1 if success, 0 otherwise
        "y_horizon",      # Horizon label before hierarchical override
        "hlabel",         # Hierarchical level 0-3 (highest gate passed)
        "hlabel_meta",    # Hierarchical deploy decision (L1 & L2 & L3)
    )

    # Optional audit columns
    audit: Tuple[str, ...] = (
        "end_time_utc",
        "exit_reason",
        "duration_hours",
        "max_drawdown_pct",
        "max_runup_pct",
    )

    @property
    def all_columns(self) -> Tuple[str, ...]:
        """All columns in the schema."""
        return self.identifiers + self.event_metadata + self.features + self.labels + self.audit


# Default schema instance
DEFAULT_SCHEMA = TrainingTableSchema()
HMM_FEATURE_SEMANTICS_VERSION = "agg_regime_probs_v20260320"
TRAINING_DATASET_SCHEMA_VERSION = "training_dataset_v20260320"


@dataclass
class FeatureSnapshot:
    """
    Complete feature snapshot for a single candidate at decision time.

    All features computed BEFORE the decision (no forward information).
    """
    # Identifiers
    symbol: str
    start_time_utc: datetime
    candidate_id: Optional[str] = None

    # HMM/Regime features
    range_prob: Optional[float] = None
    trend_prob: Optional[float] = None
    hmm_trained_at_utc: Optional[str] = None
    hmm_artifact_version: Optional[str] = None
    hmm_pipeline_version: Optional[str] = None
    hmm_feature_semantics_version: Optional[str] = None
    hmm_feature_source: Optional[str] = None
    hmm_calibration_status: Optional[str] = None

    # Stochastic/survival features
    survival_prob: Optional[float] = None
    hurst_exponent: Optional[float] = None
    ou_halflife: Optional[float] = None

    # Grid parameters
    profit_per_grid_pct: Optional[float] = None
    num_grids: Optional[int] = None
    grid_spacing_pct: Optional[float] = None
    mode: Optional[str] = None  # "arithmetic" | "geometric" | None — bot-side metadata, NOT a model feature (see GRID_SYNCH §6/C3)
    range_size_pct: Optional[float] = None

    # Technical indicators
    adx_1h: Optional[float] = None
    adx_15m: Optional[float] = None
    rsi_15m: Optional[float] = None
    adx_5m: Optional[float] = None
    ema_slope_1h: Optional[float] = None
    ema_crosses_5m: Optional[int] = None
    vwap_crosses_5m: Optional[int] = None
    bb_width: Optional[float] = None
    atr_pct_15m: Optional[float] = None
    quote_volume_24h: Optional[float] = None
    open_interest: Optional[float] = None
    regime_conf: Optional[float] = None

    # Market micro-cost
    funding_rate: Optional[float] = None
    micro_round_trip_cost_pct: Optional[float] = None

    # afml_enriched_v20260318: HMM tail risk, fill velocity, regime duration
    hmm_tail_cvar_95: Optional[float] = None
    tos_gcf: Optional[float] = None
    persistence_prob: Optional[float] = None

    # Tier 4 context features
    long_short_ratio: Optional[float] = None
    funding_rate_zscore: Optional[float] = None
    open_interest_change_pct: Optional[float] = None
    bb_width_ratio_1h_15m: Optional[float] = None

    # Micro-oscillator archetype
    micro_osc_score: Optional[float] = None

    # Binance-derived crowding / flow context
    top_account_lsr_log: Optional[float] = None
    top_position_lsr_log: Optional[float] = None
    top_position_vs_account_delta: Optional[float] = None
    global_lsr_log: Optional[float] = None
    taker_imbalance: Optional[float] = None
    taker_buy_sell_log: Optional[float] = None
    basis_pct: Optional[float] = None

    # Binance-derived execution / liquidity context
    spread_pct: Optional[float] = None
    top20_bid_depth_usdt: Optional[float] = None
    top20_ask_depth_usdt: Optional[float] = None
    top20_depth_min_usdt: Optional[float] = None
    book_imbalance_top20: Optional[float] = None
    oi_notional: Optional[float] = None
    oi_to_volume: Optional[float] = None

    # Pattern-profile inputs (scan-time provenance, not active meta features)
    parkinson_vol_ratio_4h_24h_pre: Optional[float] = None
    variance_ratio_1m_15m_pre_2h: Optional[float] = None
    funding_carry_expected_next_7h: Optional[float] = None
    liquidity_stability_z_1h: Optional[float] = None

    # Composite score
    ev_score: Optional[float] = None
    ev_contract_fingerprint: Optional[str] = None
    primary_pipeline_score: Optional[float] = None

    # Optional: range bounds (for label computation)
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    entry_price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "symbol": self.symbol,
            "start_time_utc": self.start_time_utc,
            "candidate_id": self.candidate_id,
            "range_prob": self.range_prob,
            "trend_prob": self.trend_prob,
            "hmm_trained_at_utc": self.hmm_trained_at_utc,
            "hmm_artifact_version": self.hmm_artifact_version,
            "hmm_pipeline_version": self.hmm_pipeline_version,
            "hmm_feature_semantics_version": self.hmm_feature_semantics_version,
            "hmm_feature_source": self.hmm_feature_source,
            "hmm_calibration_status": self.hmm_calibration_status,
            "survival_prob": self.survival_prob,
            "hurst_exponent": self.hurst_exponent,
            "ou_halflife": self.ou_halflife,
            "profit_per_grid_pct": self.profit_per_grid_pct,
            "num_grids": self.num_grids,
            "grid_spacing_pct": self.grid_spacing_pct,
            "mode": self.mode,
            "range_size_pct": self.range_size_pct,
            "adx_1h": self.adx_1h,
            "adx_15m": self.adx_15m,
            "rsi_15m": self.rsi_15m,
            "adx_5m": self.adx_5m,
            "ema_slope_1h": self.ema_slope_1h,
            "ema_crosses_5m": self.ema_crosses_5m,
            "vwap_crosses_5m": self.vwap_crosses_5m,
            "bb_width": self.bb_width,
            "atr_pct_15m": self.atr_pct_15m,
            "quote_volume_24h": self.quote_volume_24h,
            "open_interest": self.open_interest,
            "regime_conf": self.regime_conf,
            "funding_rate": self.funding_rate,
            "micro_round_trip_cost_pct": self.micro_round_trip_cost_pct,
            "hmm_tail_cvar_95": self.hmm_tail_cvar_95,
            "tos_gcf": self.tos_gcf,
            "persistence_prob": self.persistence_prob,
            "long_short_ratio": self.long_short_ratio,
            "funding_rate_zscore": self.funding_rate_zscore,
            "open_interest_change_pct": self.open_interest_change_pct,
            "bb_width_ratio_1h_15m": self.bb_width_ratio_1h_15m,
            "micro_osc_score": self.micro_osc_score,
            # Binance-derived crowding / flow context
            "top_account_lsr_log": self.top_account_lsr_log,
            "top_position_lsr_log": self.top_position_lsr_log,
            "top_position_vs_account_delta": self.top_position_vs_account_delta,
            "global_lsr_log": self.global_lsr_log,
            "taker_imbalance": self.taker_imbalance,
            "taker_buy_sell_log": self.taker_buy_sell_log,
            "basis_pct": self.basis_pct,
            # Binance-derived execution / liquidity context
            "spread_pct": self.spread_pct,
            "top20_bid_depth_usdt": self.top20_bid_depth_usdt,
            "top20_ask_depth_usdt": self.top20_ask_depth_usdt,
            "top20_depth_min_usdt": self.top20_depth_min_usdt,
            "book_imbalance_top20": self.book_imbalance_top20,
            "oi_notional": self.oi_notional,
            "oi_to_volume": self.oi_to_volume,
            "parkinson_vol_ratio_4h_24h_pre": self.parkinson_vol_ratio_4h_24h_pre,
            "variance_ratio_1m_15m_pre_2h": self.variance_ratio_1m_15m_pre_2h,
            "funding_carry_expected_next_7h": self.funding_carry_expected_next_7h,
            "liquidity_stability_z_1h": self.liquidity_stability_z_1h,
            "ev_score": self.ev_score,
            "ev_contract_fingerprint": self.ev_contract_fingerprint,
            "primary_pipeline_score": self.primary_pipeline_score,
            "range_high": self.range_high,
            "range_low": self.range_low,
            "entry_price": self.entry_price,
        }



@dataclass
class OutcomeLabel:
    """
    Forward outcome label for a candidate.

    Computed from price data AFTER the decision time.
    Uses triple-barrier method from AFML.

    AFML Compliance:
    - t1 stores vertical barrier timestamp for proper purging
    - barrier_touched tracks which barrier was hit first (not final PnL)
    - barrier_price stores the price at barrier touch
    - funding_cost_pct and net_pnl_pct track funding drag for AFML-compliant labels
    """
    # Identifiers (must match FeatureSnapshot)
    symbol: str
    start_time_utc: datetime

    # Core labels
    pnl_pct: float              # Realized PnL percentage (gross, before funding)
    sl_hit: bool                # Whether stop-loss was triggered
    tp_hit: bool                # Whether take-profit was triggered

    # Derived binary label
    y: int                      # 1 if success (pnl >= hurdle and not sl_hit), 0 otherwise

    # AFML event metadata (required for proper CV purging)
    t1: Optional[datetime] = None           # Vertical barrier timestamp (event end)
    barrier_touched: Optional[str] = None   # "pt", "sl", "time"
    barrier_price: Optional[float] = None   # Price at barrier touch

    # AFML funding drag tracking (Station 4 compliance)
    funding_cost_pct: Optional[float] = None   # Estimated funding cost over holding period
    net_pnl_pct: Optional[float] = None        # Net PnL after funding drag (pnl_pct - funding_cost_pct)
    funding_rate: Optional[float] = None       # Funding rate at entry time (for reference)

    # Audit information
    end_time_utc: Optional[datetime] = None
    exit_reason: Optional[str] = None  # "tp_hit", "sl_hit", "vertical", "manual"
    duration_hours: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    max_runup_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "symbol": self.symbol,
            "start_time_utc": self.start_time_utc,
            "pnl_pct": self.pnl_pct,
            "sl_hit": self.sl_hit,
            "tp_hit": self.tp_hit,
            "y": self.y,
            "t1": self.t1,
            "barrier_touched": self.barrier_touched,
            "barrier_price": self.barrier_price,
            "funding_cost_pct": self.funding_cost_pct,
            "net_pnl_pct": self.net_pnl_pct,
            "funding_rate": self.funding_rate,
            "end_time_utc": self.end_time_utc,
            "exit_reason": self.exit_reason,
            "duration_hours": self.duration_hours,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_runup_pct": self.max_runup_pct,
        }


# =============================================================================
# BARRIER-BASED LABEL GENERATOR (AFML-Compliant)
# =============================================================================

@dataclass(frozen=True)
class LabelConfig:
    """Configuration for triple-barrier label generation."""

    # Horizontal barriers
    pt_pct: float = 15.0            # Take-profit barrier: +15% (AFML triple-barrier PT)
    sl_pct: float = -12.0           # Stop-loss barrier: -12%
    meta_hurdle_pct: float = 3.0    # Meta-labeling success threshold (binary y=1 if PnL >= this)

    # Vertical barrier
    horizon_hours: float = BOT_HORIZON_HOURS     # Maximum holding time (canonical H*)

    # Grid-specific adjustments
    use_grid_pnl: bool = True       # If True, uses grid-specific PnL logic
    include_funding: bool = True    # Include funding costs in PnL


class BarrierLabelGenerator:
    """
    Triple-barrier label generator for grid bot candidates.

    Implements AFML's event labeling framework:
    - Upper barrier: Take-profit target (pt_pct)
    - Lower barrier: Stop-loss threshold (sl_pct)
    - Vertical barrier: Maximum holding time (horizon_hours)

    The label y = 1 if:
    - PnL >= pt_pct before hitting sl_pct or vertical barrier

    The label y = 0 if:
    - PnL <= sl_pct before hurdle (stop-loss hit)
    - Vertical barrier reached without hitting hurdle
    """

    def __init__(self, config: Optional[LabelConfig] = None):
        self.config = config or LabelConfig()


    def compute_label_from_final_pnl(

        self,

        *,

        symbol: str,

        start_time_utc: datetime,

        pnl_pct: float,

        duration_hours: Optional[float] = None,

        end_time_utc: Optional[datetime] = None,

    ) -> OutcomeLabel:

        """Compute label from terminal PnL when the full price path is unavailable.


        This remains AFML-consistent by producing an event span (t0=start_time_utc,

        t1=end_time_utc or start_time_utc + horizon) so that purged/CPCV validation and

        concurrency/uniqueness sample-weighting can be applied.


        Limitation (explicit): Without intrahorizon prices we cannot know the first barrier

        touch. In that case, barrier_touched is inferred from the terminal PnL only.


        Args:

            symbol: Trading symbol for the event.

            start_time_utc: Event start timestamp (t0).

            pnl_pct: Terminal realized PnL percentage (e.g., 5.0 for 5%).

            duration_hours: Observed duration in hours (optional).

            end_time_utc: Observed event end timestamp (optional).


        Returns:

            OutcomeLabel including t1 for purging/weighting.

        """

        # Barrier hits inferred from terminal outcome (cannot know first-touch without path)

        sl_hit = pnl_pct <= self.config.sl_pct

        tp_hit = pnl_pct >= self.config.meta_hurdle_pct


        # Event end time (t1): prefer observed end_time_utc; otherwise fall back to

        # start_time_utc + duration_hours; otherwise vertical barrier horizon.

        if end_time_utc is not None and isinstance(end_time_utc, datetime):

            t1 = end_time_utc

        elif duration_hours is not None and duration_hours > 0:

            # Clamp duration to horizon to prevent unreasonably long event spans
            clamped_hours = min(float(duration_hours), float(self.config.horizon_hours))
            t1 = start_time_utc + timedelta(hours=clamped_hours)

        else:

            t1 = start_time_utc + timedelta(hours=float(self.config.horizon_hours))


        # Exit reason (audit)

        if sl_hit:

            exit_reason = "sl_hit"

            barrier_touched = "sl"

        elif tp_hit:

            exit_reason = "tp_hit"

            barrier_touched = "pt"

        else:

            # If duration suggests the horizon was reached, mark as vertical barrier.

            if duration_hours is not None and duration_hours >= self.config.horizon_hours:

                exit_reason = "vertical"

                barrier_touched = "time"

            else:

                exit_reason = "manual"

                barrier_touched = None


        # Binary label: success if hurdle reached and not stopped out

        y = 1 if (tp_hit and not sl_hit) else 0


        return OutcomeLabel(

            symbol=symbol,

            start_time_utc=start_time_utc,

            pnl_pct=float(pnl_pct),

            sl_hit=bool(sl_hit),

            tp_hit=bool(tp_hit),

            y=int(y),

            t1=t1,

            barrier_touched=barrier_touched,

            barrier_price=None,

            exit_reason=exit_reason,

            duration_hours=float(duration_hours) if duration_hours is not None else None,

            end_time_utc=end_time_utc,

        )


    def compute_label_from_price_path(
        self,
        entry_price: float,
        prices: np.ndarray,
        timestamps: np.ndarray,
        start_time: datetime,
        entry_side: str = "long",
    ) -> OutcomeLabel:
        """
        Compute label from price path using triple-barrier method.

        This is the full AFML-compliant version with barrier touch detection.

        Args:
            entry_price: Entry price at start_time
            prices: Price array (close prices, e.g., 15M bars)
            timestamps: Corresponding timestamps
            start_time: Decision time
            entry_side: "long" or "short" (grids typically long-biased)

        Returns:
            OutcomeLabel with barrier touch detection
        """
        logger.warning(
            "compute_label_from_price_path is deprecated; prefer TripleBarrierLabeler "
            "from neutralgrid.models.triple_barrier for OHLC-aware barrier labeling."
        )

        if len(prices) == 0:
            return OutcomeLabel(
                symbol="",
                start_time_utc=start_time,
                pnl_pct=0.0,
                sl_hit=False,
                tp_hit=False,
                y=0,
                exit_reason="no_data",
            )

        # Compute returns relative to entry
        if entry_side == "long":
            returns_pct = (prices / entry_price - 1) * 100
        else:
            returns_pct = (1 - prices / entry_price) * 100

        # Track max drawdown and runup
        max_runup = np.max(returns_pct)
        max_drawdown = np.min(returns_pct)

        # Find first barrier touch
        tp_idx = np.where(returns_pct >= self.config.pt_pct)[0]
        sl_idx = np.where(returns_pct <= self.config.sl_pct)[0]

        # Determine which barrier was hit first
        tp_first = tp_idx[0] if len(tp_idx) > 0 else len(prices)
        sl_first = sl_idx[0] if len(sl_idx) > 0 else len(prices)

        # Vertical barrier (horizon)
        horizon_dt = timedelta(hours=self.config.horizon_hours)
        vertical_idx = len(prices)  # Default: full series

        for i, ts in enumerate(timestamps):
            if isinstance(ts, (int, float)):
                ts = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            elif not isinstance(ts, datetime):
                ts = pd.to_datetime(ts, utc=True)

            if ts - start_time >= horizon_dt:
                vertical_idx = i
                break

        # Determine exit (handle double-touch with sl_first policy)
        if tp_first == sl_first and tp_first < len(prices):
            # Double touch on same bar -- conservative sl_first policy
            exit_idx = sl_first
            exit_reason = "sl_hit"
            tp_hit = False
            sl_hit = True
            barrier_touched = "sl"
            final_pnl = returns_pct[sl_first]
        else:
            exit_idx = min(tp_first, sl_first, vertical_idx)

            if exit_idx == tp_first and tp_first < sl_first:
                exit_reason = "tp_hit"
                tp_hit = True
                sl_hit = False
                barrier_touched = "pt"
                final_pnl = returns_pct[tp_first]
            elif exit_idx == sl_first and sl_first < tp_first:
                exit_reason = "sl_hit"
                tp_hit = False
                sl_hit = True
                barrier_touched = "sl"
                final_pnl = returns_pct[sl_first]
            else:
                exit_reason = "vertical"
                tp_hit = False
                sl_hit = False
                barrier_touched = "time"
                final_pnl = returns_pct[min(vertical_idx, len(returns_pct) - 1)]

        # Binary label
        y = 1 if (tp_hit and not sl_hit) else 0

        # Populate t1 (vertical barrier timestamp for AFML purging)
        t1 = None
        if exit_idx < len(timestamps):
            t1_raw = timestamps[exit_idx]
            if isinstance(t1_raw, (int, float)):
                t1 = datetime.fromtimestamp(t1_raw / 1000, tz=timezone.utc)
            else:
                t1 = pd.to_datetime(t1_raw)
                if t1.tzinfo is None:
                    t1 = t1.replace(tzinfo=timezone.utc)
                t1 = t1.to_pydatetime() if isinstance(t1, pd.Timestamp) else t1

        # Duration
        if exit_idx < len(timestamps):
            end_ts = timestamps[exit_idx]
            if isinstance(end_ts, (int, float)):
                end_time = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc)
            else:
                end_time = pd.to_datetime(end_ts, utc=True)
            if hasattr(start_time, 'tzinfo') and start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            duration = (end_time - start_time).total_seconds() / 3600
        else:
            end_time = None
            duration = self.config.horizon_hours

        # Barrier price: price at the exit index (AFML event metadata)
        _barrier_price: Optional[float] = None
        if exit_idx < len(prices):
            _barrier_price = float(prices[exit_idx])

        return OutcomeLabel(
            symbol="",
            start_time_utc=start_time,
            pnl_pct=float(final_pnl),
            sl_hit=sl_hit,
            tp_hit=tp_hit,
            y=y,
            t1=t1,
            barrier_touched=barrier_touched,
            barrier_price=_barrier_price,
            end_time_utc=end_time,
            exit_reason=exit_reason,
            duration_hours=duration,
            max_drawdown_pct=float(max_drawdown),
            max_runup_pct=float(max_runup),
        )


# =============================================================================
# EXISTING DATA MAPPER (for new_expired_bots.xlsx)
# =============================================================================

class ExistingDataMapper:
    """
    Maps existing historical bot data to training table format.

    Handles the gap between available columns and expected features.
    """

    # Column mapping: existing -> expected
    COLUMN_MAP = {
        "grids_count": "num_grids",
        "grid_spacing_pct": "grid_spacing_pct",
        "mode": "mode",  # GRID_SYNCH §1.3 — bot-side metadata; consumed by map_dataframe to branch arithmetic vs geometric profit-per-grid; must NOT enter model X-matrix
        # Technical indicators already match
        "adx_1h": "adx_1h",
        "adx_15m": "adx_15m",
        "rsi_15m": "rsi_15m",
        "range_size_pct": "range_size_pct",
        "primary_pipeline_score": "primary_pipeline_score",
        # Outcome columns
        "pnl_pct": "pnl_pct",
        "duration_hours": "duration_hours",
        "start_time_utc": "start_time_utc",
        "end_time_utc": "end_time_utc",
        "symbol": "symbol",
    }

    # Features that cannot be computed from existing data
    MISSING_FEATURES = [
        "range_prob",       # Needs historical 1H OHLCV for HMM
        "trend_prob",       # Needs historical 1H OHLCV for HMM
        "survival_prob",    # Needs historical 15M OHLCV + range bounds
        "hurst_exponent",   # Needs historical 15M log-prices
        "ou_halflife",      # Needs historical 15M log-prices
        "funding_rate",     # Not captured in historical data
        "ev_score",         # Needs survival_prob + other components
    ]

    def __init__(self, label_config: Optional[LabelConfig] = None):
        self.label_config = label_config or LabelConfig()
        self.label_generator = BarrierLabelGenerator(self.label_config)

    def compute_profit_per_grid(
        self,
        grid_lower: float,
        grid_upper: float,
        num_grids: int,
        mode: str,
        maker_fee: Optional[float] = None,
        taker_fee: Optional[float] = None,
        close_fee_mode: Optional[str] = None,
    ) -> float:
        """
        Per-grid profit percentage routed through the canonical formula.

        Delegates to ``neutralgrid.grid.formulas.profit_per_grid_pct`` (single
        source of truth introduced by GRID_SYNCH.md Step 2).

        ``mode`` ("arithmetic" | "geometric") is REQUIRED — no default.
        Pass per-row mode from the workbook (`General` sheet, GRID_SYNCH §1
        adds the column). Per GRID_SYNCH §2.1 / GRIDFIX-001, this removes
        the prior silent flip where the wrapper hardcoded GEOMETRIC and
        produced wrong-mode values for arithmetic past bots.

        Fee constant ``c`` mirrors F2/F3 cross-site equivalence
        (GRID_SYNCH §2.2 invariant C10):
          ``c = (maker_fee + close_fee_rate) / 2``
          ``close_fee_rate = maker_fee if close_fee_mode == "maker" else taker_fee``
        Defaults are pulled from ``get_config().grid``; explicit kwargs
        override (used by test fixtures).
        """
        if num_grids <= 1 or grid_lower <= 0 or grid_lower >= grid_upper:
            return float("nan")
        cfg_grid = get_config().grid
        eff_maker = float(cfg_grid.maker_fee) if maker_fee is None else float(maker_fee)
        eff_taker = float(cfg_grid.taker_fee) if taker_fee is None else float(taker_fee)
        eff_close_mode = (
            str(getattr(cfg_grid, "close_fee_mode", "maker")).lower()
            if close_fee_mode is None
            else str(close_fee_mode).lower()
        )
        close_fee_rate = eff_maker if eff_close_mode == "maker" else eff_taker
        c = max(0.0, (eff_maker + close_fee_rate) / 2.0)
        return _ppg_pct(
            float(grid_lower),
            float(grid_upper),
            int(num_grids),
            mode,
            c,
        )

    @staticmethod
    def _is_missing_scalar(value: Any) -> bool:
        """Return True when value is a scalar missing value (None/NaN/NaT)."""
        if value is None:
            return True
        if isinstance(value, (pd.Series, pd.DataFrame, np.ndarray, list, tuple, dict, set)):
            return False
        return bool(pd.isna(value))

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        """Safely coerce a scalar value to float with default fallback."""
        if ExistingDataMapper._is_missing_scalar(value):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def map_dataframe(
        self,
        df: pd.DataFrame,
        price_paths: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> pd.DataFrame:
        """
        Map existing data to training table format.

        Args:
            df: DataFrame from new_expired_bots.xlsx
            price_paths: Optional dict mapping "{symbol}_{start_time}" ->
                         (prices_array, timestamps_array) for price-path labeling.
                         When provided, uses AFML triple-barrier method instead of
                         final-PnL inference.

        Returns:
            DataFrame with mapped columns and computed fields
        """
        result = pd.DataFrame()

        # Copy identifier columns
        result["symbol"] = df["symbol"]
        if "candidate_id" in df.columns:
            result["candidate_id"] = df["candidate_id"].astype(str)
        if "strategy_id" in df.columns:
            result["strategy_id"] = df["strategy_id"].astype(str)
        if "start_time_utc" in df.columns:
            result["start_time_utc"] = pd.to_datetime(df["start_time_utc"], utc=True)
        else:
            # Use index as pseudo-time for ordering
            result["start_time_utc"] = pd.date_range(
                start="2025-01-01", periods=len(df), freq="h", tz="UTC"
            )

        # Map existing feature columns
        grids_source = (
            cast(pd.Series, df["grids_count"])
            if "grids_count" in df.columns
            else pd.Series(0, index=df.index, dtype="float64")
        )
        num_grids_numeric = cast(pd.Series, pd.to_numeric(grids_source, errors="coerce"))
        result["num_grids"] = num_grids_numeric.fillna(0).astype(int)
        result["grid_spacing_pct"] = pd.to_numeric(df.get("grid_spacing_pct"), errors="coerce")
        result["adx_1h"] = pd.to_numeric(df.get("adx_1h"), errors="coerce")
        result["adx_15m"] = pd.to_numeric(df.get("adx_15m"), errors="coerce")
        result["rsi_15m"] = pd.to_numeric(df.get("rsi_15m"), errors="coerce")
        result["range_size_pct"] = pd.to_numeric(df.get("range_size_pct"), errors="coerce")
        if "primary_pipeline_score" in df.columns:
            result["primary_pipeline_score"] = pd.to_numeric(
                df["primary_pipeline_score"], errors="coerce"
            )
        optional_live_plus = (
            "adx_5m",
            "ema_slope_1h",
            "ema_crosses_5m",
            "vwap_crosses_5m",
            "bb_width",
            "atr_pct_15m",
            "quote_volume_24h",
            "open_interest",
            "regime_conf",
            "micro_round_trip_cost_pct",
        )
        for col in optional_live_plus:
            if col in df.columns:
                result[col] = pd.to_numeric(df[col], errors="coerce")

        # Preserve an explicit decision-time profit-per-grid feature.  Compact
        # authoritative rows intentionally omit grid geometry, so replacing a
        # recorded finite value with NaN would make a valid training snapshot
        # appear incomplete in the compatibility availability report.
        stored_profit_source = df.get("profit_per_grid_pct")
        if isinstance(stored_profit_source, pd.Series):
            stored_profit_per_grid = cast(
                pd.Series, pd.to_numeric(stored_profit_source, errors="coerce")
            )
        else:
            stored_profit_per_grid = pd.Series(np.nan, index=df.index)

        # Compute profit_per_grid_pct from grid params only where the stored
        # decision-time value is unavailable.
        # GRID_SYNCH §2.3 / GRIDFIX-001: thread per-row `mode` through the
        # wrapper. Pre-check `pd.notna(row["mode"])` (§2.4) so a NaN-mode row
        # yields NaN PPG instead of raising. Replaces the prior `else 0.0`
        # silent default — `0.0` is a valid PPG value and would collide with
        # real data; NaN is the project's canonical missing sentinel.
        if all(c in df.columns for c in ["price_range_low", "price_range_high", "grids_count", "mode"]):
            derived_profit_per_grid = df.apply(
                lambda row: self.compute_profit_per_grid(
                    float(row["price_range_low"]),
                    float(row["price_range_high"]),
                    int(row["grids_count"]),
                    mode=str(row["mode"]),
                ) if (
                    pd.notna(row["price_range_low"])
                    and pd.notna(row["price_range_high"])
                    and pd.notna(row["grids_count"])
                    and pd.notna(row["mode"])
                ) else float("nan"),
                axis=1,
            )
        else:
            derived_profit_per_grid = pd.Series(np.nan, index=df.index)
        result["profit_per_grid_pct"] = stored_profit_per_grid.where(
            stored_profit_per_grid.notna(), derived_profit_per_grid
        )

        # Map outcome columns
        result["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
        result["net_pnl_pct"] = result["pnl_pct"]
        if "net_pnl_pct" in df.columns:
            result["net_pnl_pct"] = pd.to_numeric(df["net_pnl_pct"], errors="coerce")
        elif all(c in df.columns for c in ["total_profit_usdt", "commission_usdt", "invested_margin_usdt"]):
            tp = cast(pd.Series, pd.to_numeric(df["total_profit_usdt"], errors="coerce"))
            cm = cast(pd.Series, pd.to_numeric(df["commission_usdt"], errors="coerce")).fillna(0.0)
            mg = cast(pd.Series, pd.to_numeric(df["invested_margin_usdt"], errors="coerce"))
            with np.errstate(divide="ignore", invalid="ignore"):
                net_pct = ((tp - cm) / mg) * 100.0
            result["net_pnl_pct"] = cast(pd.Series, net_pct).replace([np.inf, -np.inf], np.nan).fillna(result["pnl_pct"])

        if "duration_hours" in df.columns:
            result["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce")

        if "end_time_utc" in df.columns:
            result["end_time_utc"] = pd.to_datetime(df["end_time_utc"], utc=True)

        # GRID_SYNCH §2.5 / GRIDFIX-001 — fail-closed exclusion at the
        # data-quality boundary. Drop rows where BOTH grid_spacing_pct AND
        # profit_per_grid_pct are NaN (no usable grid geometry).
        # Empirically (data-curator + backtest-evaluator probes 2026-05-04):
        # zero rows dropped on the current xlsx because grid_spacing_pct is
        # fully populated. Gate is defensive against future re-extracts;
        # locked-pool size (`feedback_sample_pool_fixed.md`) is mechanically
        # preserved on the current data.
        if (
            "grid_spacing_pct" in result.columns
            and "profit_per_grid_pct" in result.columns
        ):
            drop_mask = (
                result["grid_spacing_pct"].isna()
                & result["profit_per_grid_pct"].isna()
            )
            if bool(drop_mask.any()):
                if "symbol" in result.columns:
                    dropped_symbols = result.loc[drop_mask, "symbol"].tolist()
                else:
                    dropped_symbols = []
                logger.warning(
                    "[map_dataframe] dropping %d rows with both grid_spacing_pct "
                    "and profit_per_grid_pct missing: symbols=%s",
                    int(drop_mask.sum()),
                    dropped_symbols,
                )
                result = result.loc[~drop_mask].reset_index(drop=True)

        # Compute labels (AFML-consistent event spans)
        # Prefer price-path labeling (AFML Ch 3) over final-PnL inference
        labels = []
        price_path_used = 0
        for _, row in result.iterrows():
            sym = str(row.get("symbol", ""))
            t0 = row.get("start_time_utc")
            if self._is_missing_scalar(t0):
                labels.append({
                    "sl_hit": False, "tp_hit": False, "y": 0,
                    "exit_reason": "invalid_start", "t1": pd.NaT,
                    "barrier_touched": "time", "barrier_price": None,
                })
                continue
            t0_dt = cast(datetime, t0.to_pydatetime() if isinstance(t0, pd.Timestamp) else t0)
            t_end = row.get("end_time_utc") if "end_time_utc" in row else None
            t_end_dt: Optional[datetime] = None
            if not self._is_missing_scalar(t_end):
                t_end_ts = pd.to_datetime(cast(Any, t_end), utc=True, errors="coerce")
                if isinstance(t_end_ts, pd.Timestamp) and not pd.isna(t_end_ts):
                    # Python datetime supports microseconds, while pandas may
                    # retain nanoseconds.  Make the existing truncation
                    # explicit so conversion is deterministic and warning-free.
                    t_end_dt = t_end_ts.floor("us").to_pydatetime()
            label = None

            # Try price-path labeling first (AFML Ch 3 triple barrier)
            if price_paths is not None:
                key = f"{sym}_{t0}"
                if key in price_paths:
                    prices, timestamps = price_paths[key]
                    if len(prices) > 10:
                        entry_price = float(prices[0])
                        label = self.label_generator.compute_label_from_price_path(
                            entry_price=entry_price,
                            prices=prices,
                            timestamps=timestamps,
                            start_time=t0_dt,
                        )
                        label = OutcomeLabel(
                            symbol=sym,
                            start_time_utc=t0_dt,
                            pnl_pct=label.pnl_pct,
                            sl_hit=label.sl_hit,
                            tp_hit=label.tp_hit,
                            y=label.y,
                            t1=label.t1,
                            barrier_touched=label.barrier_touched,
                            barrier_price=label.barrier_price,
                            end_time_utc=label.end_time_utc,
                            exit_reason=label.exit_reason,
                            duration_hours=label.duration_hours,
                            max_drawdown_pct=label.max_drawdown_pct,
                            max_runup_pct=label.max_runup_pct,
                        )
                        price_path_used += 1

            # Fallback: final-PnL inference
            if label is None:
                pnl_pct = self._as_float(row.get("pnl_pct"), 0.0)
                duration_raw = row.get("duration_hours", self.label_config.horizon_hours)
                duration_hours: Optional[float] = None
                if not self._is_missing_scalar(duration_raw):
                    duration_hours = self._as_float(
                        duration_raw, float(self.label_config.horizon_hours)
                    )
                label = self.label_generator.compute_label_from_final_pnl(
                    symbol=sym,
                    start_time_utc=t0_dt,
                    pnl_pct=pnl_pct,
                    duration_hours=duration_hours,
                    end_time_utc=t_end_dt,
                )

            labels.append({
                "sl_hit": label.sl_hit,
                "tp_hit": label.tp_hit,
                "y": label.y,
                "exit_reason": label.exit_reason,
                "t1": label.t1,
                "barrier_touched": label.barrier_touched,
                "barrier_price": label.barrier_price,
            })

        if price_paths is not None:
            logger.info(
                f"Label method: {price_path_used}/{len(labels)} used price-path (AFML Ch 3), "
                f"{len(labels) - price_path_used} used final-PnL fallback"
            )

        label_df = pd.DataFrame(labels)
        for col in label_df.columns:
            result[col] = np.asarray(label_df[col])

        # Add backfilled or placeholder features
        # If the feature exists in source data, use it; otherwise add NaN placeholder
        for feat in self.MISSING_FEATURES:
            if feat in df.columns:
                # Feature was backfilled - use actual data
                result[feat] = pd.to_numeric(df[feat], errors="coerce")
            else:
                # Feature truly missing - add NaN placeholder
                result[feat] = np.nan

        return result

    def get_feature_availability_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Report on feature availability after mapping.

        Returns statistics on available vs missing features.
        """
        mapped_df = self.map_dataframe(df)

        feature_cols = list(DEFAULT_SCHEMA.features)
        report = {
            "total_samples": len(mapped_df),
            "available_features": [],
            "missing_features": [],
            "partial_features": [],
        }

        for feat in feature_cols:
            if feat not in mapped_df.columns:
                report["missing_features"].append(feat)
            else:
                non_null_pct = mapped_df[feat].notna().mean() * 100
                if non_null_pct == 0:
                    report["missing_features"].append(feat)
                elif non_null_pct < 100:
                    report["partial_features"].append({
                        "feature": feat,
                        "coverage_pct": round(non_null_pct, 1),
                    })
                else:
                    report["available_features"].append(feat)

        # Label statistics
        if "y" in mapped_df.columns:
            report["positive_label_rate"] = round(mapped_df["y"].mean() * 100, 1)

        return report


# =============================================================================
# FEATURE SNAPSHOT LOGGER (for live integration)
# =============================================================================

class FeatureSnapshotLogger:
    """
    Logs feature snapshots at decision time for future training.

    This is the bridge between live scanner and training pipeline.
    Each time a candidate is evaluated (whether accepted or rejected),
    its features are logged for potential training use.

    File format: Parquet with daily partitions for efficient reads.
    """

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_size = 100  # Flush after this many snapshots

    def _get_partition_path(self, timestamp: datetime) -> Path:
        """Get path for daily partition file."""
        date_str = timestamp.strftime("%Y-%m-%d")
        return self.log_dir / f"snapshots_{date_str}.parquet"

    def log_snapshot(self, snapshot: FeatureSnapshot) -> None:
        """
        Log a feature snapshot to buffer.

        Flushes to disk when buffer is full.
        """
        self._buffer.append(snapshot.to_dict())

        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def flush(self) -> None:
        """Flush buffered snapshots to disk."""
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer)

        # Group by date and append to partition files
        df["_date"] = pd.to_datetime(df["start_time_utc"]).dt.date

        for date_key, group in df.groupby("_date"):
            _date = date_key if isinstance(date_key, datetime) else datetime.combine(date_key, datetime.min.time())  # type: ignore[arg-type]
            partition_path = self._get_partition_path(_date)

            # Remove temporary column
            group = group.drop(columns=["_date"])

            if partition_path.exists():
                # Append to existing file
                existing = pd.read_parquet(partition_path)
                combined = pd.concat([existing, group], ignore_index=True)
                combined.to_parquet(partition_path, index=False)
            else:
                group.to_parquet(partition_path, index=False)

        self._buffer.clear()
        logger.debug(f"Flushed {len(df)} snapshots to disk")

    def load_snapshots(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load snapshots from disk for a date range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            DataFrame with all snapshots in range
        """
        # Find partition files
        partition_files = sorted(self.log_dir.glob("snapshots_*.parquet"))

        if not partition_files:
            return pd.DataFrame()

        dfs = []
        for path in partition_files:
            # Extract date from filename
            date_str = path.stem.replace("snapshots_", "")
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue

            # Filter by date range (strip tzinfo to avoid naive vs aware comparison)
            if start_date and file_date < start_date.replace(hour=0, minute=0, second=0, tzinfo=None):
                continue
            if end_date and file_date > end_date.replace(hour=23, minute=59, second=59, tzinfo=None):
                continue

            dfs.append(pd.read_parquet(path))

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.flush()


# =============================================================================
# TRAINING DATA BUILDER
# =============================================================================

class TrainingDataBuilder:
    """
    High-level API for building training datasets.

    Combines:
    - Feature snapshot loading
    - Label generation
    - Data validation
    - Missing feature handling
    """

    def __init__(
        self,
        label_config: Optional[LabelConfig] = None,
        fill_missing_with: float = 0.0,
    ):
        """
        Initialize builder.

        Args:
            label_config: Configuration for label generation
            fill_missing_with: Value to fill missing features (default 0.0)
        """
        self.label_config = label_config or LabelConfig()
        self.label_generator = BarrierLabelGenerator(self.label_config)
        self.data_mapper = ExistingDataMapper(self.label_config)
        self.fill_missing_with = fill_missing_with

    def build_from_existing_data(
        self,
        data_path: Path,
        validate: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Build training dataset from existing historical data.

        Args:
            data_path: Path to historical data file (CSV or Excel)
            validate: Whether to validate the resulting dataset

        Returns:
            Tuple of (training_df, metadata)
        """
        # Load data
        data_path = Path(data_path)
        if data_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(data_path)
        else:
            df = pd.read_csv(data_path)

        logger.info(f"Loaded {len(df)} samples from {data_path}")

        # Map to training format
        training_df = self.data_mapper.map_dataframe(df)

        # Get availability report
        availability = self.data_mapper.get_feature_availability_report(df)

        # Handle missing features (AFML: never silently fill with 0.0)
        feature_cols = list(DEFAULT_SCHEMA.features)
        present_cols = [
            f for f in feature_cols
            if f in training_df.columns and cast(pd.Series, training_df[f]).notna().any()
        ]
        missing_cols = [f for f in feature_cols if f not in training_df.columns]
        if missing_cols:
            logger.warning(
                "Missing %d feature column(s) in training data, skipping: %s",
                len(missing_cols), missing_cols,
            )
        nan_before = len(training_df)
        if present_cols:
            nan_mask = cast(pd.Series, training_df[present_cols].isna().any(axis=1))
            nan_count = int(nan_mask.sum())
            if nan_count > 0:
                logger.warning(
                    "Dropping %d / %d rows with NaN feature values (%.1f%%)",
                    nan_count, nan_before, 100.0 * nan_count / max(nan_before, 1),
                )
                training_df = cast(
                    pd.DataFrame, training_df.loc[~nan_mask].reset_index(drop=True)
                )

        # Validate if requested
        validation_issues = []
        if validate:
            validation_issues = self._validate_training_data(training_df)

        metadata: Dict[str, Any] = {
            "source_path": str(data_path),
            "total_samples": len(training_df),
            "feature_availability": availability,
            "label_config": {
                "pt_pct": self.label_config.pt_pct,
                "meta_hurdle_pct": self.label_config.meta_hurdle_pct,
                "sl_pct": self.label_config.sl_pct,
                "horizon_hours": self.label_config.horizon_hours,
            },
            "positive_rate": float(cast(pd.Series, training_df["y"]).mean()) if "y" in training_df.columns else None,
            "validation_issues": validation_issues,
            "nan_handling": "drop",
            "missing_feature_cols": missing_cols,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        return cast(pd.DataFrame, training_df), metadata

    def build_from_snapshots(
        self,
        snapshot_dir: Path,
        outcome_df: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tolerance: Optional[pd.Timedelta] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Build training dataset by joining logged snapshots with outcomes.

        Args:
            snapshot_dir: Directory containing snapshot parquet files
            outcome_df: DataFrame with outcomes (symbol, start_time_utc, pnl_pct, ...)
            start_date: Filter snapshots from this date
            end_date: Filter snapshots to this date

        Returns:
            Tuple of (training_df, metadata)
        """
        def _to_datetime_utc_mixed_local(values: Any) -> pd.Series:
            idx = values.index if isinstance(values, pd.Series) else None
            try:
                parsed = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
            except TypeError:
                parsed = pd.to_datetime(values, utc=True, errors="coerce")
            series = parsed if isinstance(parsed, pd.Series) else pd.Series(parsed, index=idx)
            # ``merge_asof`` requires identical datetime units on both join keys.
            # Snapshot parquet loads can surface ``datetime64[us, UTC]`` while
            # live outcomes often arrive as ``datetime64[ns, UTC]``.
            return cast(pd.Series, series.dt.as_unit("ns"))

        tolerance_value = cast(pd.Timedelta, tolerance if isinstance(tolerance, pd.Timedelta) else pd.Timedelta(hours=6))

        snapshot_logger = FeatureSnapshotLogger(snapshot_dir)
        snapshots_df = snapshot_logger.load_snapshots(start_date, end_date)
        if snapshots_df.empty:
            return pd.DataFrame(), {"error": "No snapshots found"}

        outcomes = outcome_df.copy().reset_index(drop=True)
        outcomes["_row_id"] = np.arange(len(outcomes), dtype=int)
        snapshots_df = snapshots_df.copy()

        outcomes["start_time_utc"] = _to_datetime_utc_mixed_local(outcomes["start_time_utc"])
        snapshots_df["start_time_utc"] = _to_datetime_utc_mixed_local(
            snapshots_df["start_time_utc"]
        )

        outcomes = cast(
            pd.DataFrame,
            outcomes.dropna(subset=["symbol", "start_time_utc"]).reset_index(drop=True),
        )
        snapshots_df = cast(
            pd.DataFrame,
            snapshots_df.dropna(subset=["symbol", "start_time_utc"]).reset_index(drop=True),
        )

        if "candidate_id" not in outcomes.columns:
            outcomes["candidate_id"] = ""
        if "candidate_id" not in snapshots_df.columns:
            snapshots_df["candidate_id"] = ""

        outcomes["candidate_id"] = outcomes["candidate_id"].fillna("").astype(str)
        snapshots_df["candidate_id"] = snapshots_df["candidate_id"].fillna("").astype(str)

        reserved_columns = {"_row_id", "symbol", "start_time_utc"}
        snapshot_value_columns = [
            col for col in snapshots_df.columns if col not in reserved_columns
        ]
        if "candidate_id" in snapshot_value_columns:
            snapshot_value_columns.remove("candidate_id")

        snapshots_df["_snapshot_feature_count"] = snapshots_df[snapshot_value_columns].notna().sum(
            axis=1
        )
        if "primary_pipeline_score" in snapshots_df.columns:
            primary_score_series = pd.to_numeric(
                snapshots_df["primary_pipeline_score"],
                errors="coerce",
            )
        else:
            primary_score_series = pd.Series(np.nan, index=snapshots_df.index, dtype=float)
        snapshots_df["_snapshot_has_primary_score"] = cast(pd.Series, primary_score_series).notna().astype(int)

        training_df = outcomes.copy()
        training_df["_snapshot_match_method"] = ""
        training_df["_matched_snapshot_start_time_utc"] = pd.Series(
            pd.NaT,
            index=training_df.index,
            dtype="datetime64[ns, UTC]",
        )
        exact_match_count = 0
        time_match_count = 0

        exact_candidate_rows = cast(
            pd.DataFrame,
            snapshots_df.loc[snapshots_df["candidate_id"].str.strip() != ""].copy(),
        )
        if not exact_candidate_rows.empty:
            exact_join = cast(
                pd.DataFrame,
                training_df.loc[
                    training_df["candidate_id"].str.strip() != "",
                    ["_row_id", "candidate_id", "start_time_utc"],
                ]
                .merge(
                    cast(pd.DataFrame, exact_candidate_rows[
                        [
                            "candidate_id",
                            "start_time_utc",
                            "_snapshot_feature_count",
                            "_snapshot_has_primary_score",
                        ]
                        + snapshot_value_columns
                    ]).rename(columns={"start_time_utc": "snapshot_start_time_utc"}),
                    on="candidate_id",
                    how="left",
                ),
            )
            exact_join = cast(
                pd.DataFrame,
                exact_join.loc[
                    exact_join["snapshot_start_time_utc"].notna()
                    & (exact_join["snapshot_start_time_utc"] <= exact_join["start_time_utc"])
                ],
            )
            if not exact_join.empty:
                exact_join = cast(
                    pd.DataFrame,
                    exact_join.sort_values(
                        [
                            "_row_id",
                            "_snapshot_feature_count",
                            "_snapshot_has_primary_score",
                            "snapshot_start_time_utc",
                        ],
                        ascending=[True, False, False, False],
                    ).drop_duplicates(subset=["_row_id"], keep="first"),
                )
            matched_mask = cast(
                pd.Series,
                exact_join[snapshot_value_columns].notna().any(axis=1),
            )
            if bool(matched_mask.any()):
                exact_match_count = int(matched_mask.sum())
                matched_rows = cast(pd.DataFrame, exact_join.loc[matched_mask])
                for col in snapshot_value_columns:
                    training_df.loc[matched_rows["_row_id"], col] = matched_rows[col].to_numpy()
                training_df.loc[
                    matched_rows["_row_id"], "_snapshot_match_method"
                ] = "candidate_id"
                training_df.loc[
                    matched_rows["_row_id"], "_matched_snapshot_start_time_utc"
                ] = pd.to_datetime(
                    matched_rows["snapshot_start_time_utc"],
                    utc=True,
                    errors="coerce",
                ).array

        unmatched_mask = cast(
            pd.Series, training_df["_snapshot_match_method"].eq("")
        )
        if bool(unmatched_mask.any()):
            left = cast(
                pd.DataFrame,
                training_df.loc[unmatched_mask, ["_row_id", "symbol", "start_time_utc"]]
                .sort_values(["start_time_utc", "symbol"])
                .reset_index(drop=True),
            )
            right = cast(
                pd.DataFrame,
                snapshots_df.sort_values(["start_time_utc", "symbol"]).reset_index(drop=True),
            )
            if not left.empty and not right.empty:
                joined = cast(
                    pd.DataFrame,
                    pd.merge_asof(
                        left,
                        cast(pd.DataFrame, right[
                            ["symbol", "start_time_utc"] + snapshot_value_columns
                        ]).rename(columns={"start_time_utc": "snapshot_start_time_utc"}),
                        left_on="start_time_utc",
                        right_on="snapshot_start_time_utc",
                        by="symbol",
                        direction="backward",
                        tolerance=tolerance_value,
                    ),
                )
                matched_mask = cast(
                    pd.Series,
                    joined[snapshot_value_columns].notna().any(axis=1),
                )
                if bool(matched_mask.any()):
                    time_match_count = int(matched_mask.sum())
                    matched_rows = cast(pd.DataFrame, joined.loc[matched_mask])
                    for col in snapshot_value_columns:
                        training_df.loc[matched_rows["_row_id"], col] = matched_rows[
                            col
                        ].to_numpy()
                    training_df.loc[
                        matched_rows["_row_id"], "_snapshot_match_method"
                    ] = "time"
                    training_df.loc[
                        matched_rows["_row_id"], "_matched_snapshot_start_time_utc"
                    ] = pd.to_datetime(
                        matched_rows["snapshot_start_time_utc"],
                        utc=True,
                        errors="coerce",
                    ).array

        # Compute labels from outcomes
        if "pnl_pct" in training_df.columns:
            labels = []
            for _, row in training_df.iterrows():
                pnl = ExistingDataMapper._as_float(row.get("pnl_pct"), 0.0)
                start_ts = row.get("start_time_utc")
                end_ts = row.get("end_time_utc") if "end_time_utc" in training_df.columns else None
                _dur = row.get("duration_hours")
                duration: Optional[float] = None
                if not ExistingDataMapper._is_missing_scalar(_dur):
                    duration = ExistingDataMapper._as_float(
                        _dur, float(self.label_config.horizon_hours)
                    )

                if ExistingDataMapper._is_missing_scalar(start_ts):
                    labels.append({"sl_hit": False, "tp_hit": False, "y": 0, "t1": pd.NaT, "barrier_touched": "time"})
                    continue

                start_dt = pd.to_datetime(cast(Any, start_ts), utc=True, errors="coerce")
                if not isinstance(start_dt, pd.Timestamp) or pd.isna(start_dt):
                    # Cannot form an event span; default to NaT and a negative label.
                    labels.append({"sl_hit": False, "tp_hit": False, "y": 0, "t1": pd.NaT, "barrier_touched": "time"})
                    continue

                end_dt: Optional[datetime] = None
                if not ExistingDataMapper._is_missing_scalar(end_ts):
                    parsed_end = pd.to_datetime(cast(Any, end_ts), utc=True, errors="coerce")
                    if isinstance(parsed_end, pd.Timestamp) and not pd.isna(parsed_end):
                        end_dt = parsed_end.to_pydatetime()

                label = self.label_generator.compute_label_from_final_pnl(
                    symbol=str(row.get("symbol", "")),
                    start_time_utc=start_dt.to_pydatetime(),
                    pnl_pct=pnl,
                    duration_hours=duration,
                    end_time_utc=end_dt,
                )

                labels.append({
                    "sl_hit": label.sl_hit,
                    "tp_hit": label.tp_hit,
                    "y": label.y,
                    "t1": label.t1,
                    "barrier_touched": label.barrier_touched,
                })

            label_df = pd.DataFrame(labels)
            for col in label_df.columns:
                training_df[col] = np.asarray(label_df[col])

        training_df["snapshot_match_method"] = training_df.pop("_snapshot_match_method")
        training_df["snapshot_matched_at_utc"] = training_df.pop(
            "_matched_snapshot_start_time_utc"
        )
        training_df["dataset_schema_version"] = TRAINING_DATASET_SCHEMA_VERSION
        matched_rows = cast(
            pd.Series, training_df["snapshot_match_method"].ne("")
        )

        metadata = {
            "snapshot_dir": str(snapshot_dir),
            "total_snapshots": len(snapshots_df),
            "matched_samples": int(matched_rows.sum()),
            "exact_candidate_matches": exact_match_count,
            "time_matches": time_match_count,
            "unmatched_rows": int((~matched_rows).sum()),
            "tolerance_hours": float(tolerance_value.total_seconds() / 3600.0),
            "date_range": {
                "start": str(start_date) if start_date else None,
                "end": str(end_date) if end_date else None,
            },
        }

        return cast(pd.DataFrame, training_df.drop(columns=["_row_id"])), metadata

    def _validate_training_data(self, df: pd.DataFrame) -> List[str]:
        """Validate training data for common issues."""
        issues = []

        # Check required columns
        required = ["symbol", "start_time_utc", "t1", "pnl_pct", "y"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            issues.append(f"Missing required columns: {missing}")

        # Check for NaN in labels
        if "y" in df.columns:
            nan_labels = df["y"].isna().sum()
            if nan_labels > 0:
                issues.append(f"Found {nan_labels} NaN values in label column 'y'")

        # Check positive rate
        if "y" in df.columns:
            pos_rate = df["y"].mean()
            if pos_rate < 0.05:
                issues.append(f"Very low positive rate ({pos_rate:.1%}) - model may not learn effectively")
            elif pos_rate > 0.95:
                issues.append(f"Very high positive rate ({pos_rate:.1%}) - labels may be too easy")

        # Check for sufficient samples
        if len(df) < 30:
            issues.append(f"Very few samples ({len(df)}) - CV may not be reliable")

        # Check timestamp ordering
        if "start_time_utc" in df.columns:
            ts = pd.to_datetime(df["start_time_utc"])
            if ts.isna().any():
                issues.append("Found NaN timestamps - CV ordering will be affected")

        return issues


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_and_prepare_training_data(
    data_path: Path,
    hurdle_pct: float = 3.0,
    sl_pct: float = -12.0,
    horizon_hours: float = BOT_HORIZON_HOURS,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to load and prepare training data.

    Args:
        data_path: Path to historical data file
        hurdle_pct: PnL hurdle for success label
        sl_pct: Stop-loss threshold
        horizon_hours: Prediction horizon

    Returns:
        Tuple of (training_df, metadata)
    """
    config = LabelConfig(
        meta_hurdle_pct=hurdle_pct,
        sl_pct=sl_pct,
        horizon_hours=horizon_hours,
    )
    builder = TrainingDataBuilder(label_config=config)
    return builder.build_from_existing_data(data_path)


def get_missing_features_report(data_path: Path) -> Dict[str, Any]:
    """
    Get a report on which features are missing from historical data.

    This helps identify what data needs to be collected for proper training.
    """
    mapper = ExistingDataMapper()

    data_path = Path(data_path)
    if data_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    return mapper.get_feature_availability_report(df)
