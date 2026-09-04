"""
Deterministic configuration management.

This module provides a single source of truth for configuration with:
- Environment variable overrides
- YAML/JSON file loading
- Type-safe defaults
- Validation

Design principles:
- Explicit defaults (no hidden configuration)
- Environment variables override file config
- Type safety with validation
- Clear error messages for invalid config
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import warnings

from neutralgrid.core.constants import BOT_HORIZON_BARS_15M, BOT_HORIZON_HOURS, BOT_HORIZON_SECONDS

_PROJECT_ROOT = Path(
    os.getenv("NEUTRALGRID_BASE_DIR", str(Path(__file__).resolve().parents[3]))
).resolve()

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    _env_path = _PROJECT_ROOT / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed, use system env vars


# =========================================================================
# SUB-CONFIG DATACLASSES
# =========================================================================


@dataclass
class GridConfig:
    """Grid calculation and risk parameters."""

    capital: float = 400.0
    leverage_min: int = 10
    leverage_max: int = 10
    max_holding_time: str = "6h"
    max_holding_seconds: int = BOT_HORIZON_SECONDS

    # Profit floor
    profit_grid_min_pct_static_fallback: float = 0.59
    profit_grid_min_pct: float = 0.59  # backward-compat alias

    # Liquidity tiers
    liquidity_tier_spread_thresholds: Tuple[float, float] = (0.02, 0.06)
    liquidity_tier_high: Tuple[float, float] = (0.25, 1.2)
    liquidity_tier_mid: Tuple[float, float] = (0.35, 1.4)
    liquidity_tier_low: Tuple[float, float] = (0.59, 1.7)

    # Grid count limits (Binance futures allows up to 175)
    min_grids: int = 7
    max_grids: int = 175

    # Grid spacing
    grid_spacing_profile_path: str = "data/new_expired_bots.xlsx"
    grid_spacing_top_quantile: float = 0.80
    # Cap on geometric per-interval spacing (decimal). Default = empirical q95
    # of the geometric winner subpool (n=29 unique symbols) = 1.1322% → 0.0115.
    max_spacing_pct: float = 0.0115

    # Fees
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    close_fee_mode: str = "maker"  # Keep scanner profit math aligned with training/backtest defaults

    # Risk controls (derived from barrier in __post_init__)
    stop_loss_pct: float = -0.12
    take_profit_pct: float = 0.15


@dataclass
class BarrierConfig:
    """Triple-barrier labeling parameters."""

    pt_pct: float = 15.0
    sl_pct: float = -12.0
    time_hours: float = BOT_HORIZON_HOURS
    timeframe_minutes: int = 15
    meta_hurdle_pct: float = 3.0
    use_vol_scaling: bool = False
    k_pt: float = 2.0
    k_sl: float = 1.5


@dataclass
class IndicatorConfig:
    """Technical indicator parameters."""

    ema_fast: int = 20
    ema_medium: int = 50
    ema_slow: int = 200
    bb_period: int = 20
    bb_std_dev: float = 2.0
    donchian_period: int = 20
    atr_period: int = 14
    rsi_period: int = 14
    adx_period: int = 14


@dataclass
class BinanceConfig:
    """Binance API configuration."""

    futures_base_url: str = "https://fapi.binance.com"
    api_key: str = ""
    api_secret: str = ""
    endpoints: Dict[str, str] = field(default_factory=lambda: {
        "klines": "/fapi/v1/klines",
        "open_interest": "/fapi/v1/openInterest",
        "funding_rate": "/fapi/v1/fundingRate",
        "premium_index": "/fapi/v1/premiumIndex",
        "depth": "/fapi/v1/depth",
        "ticker": "/fapi/v1/ticker/24hr",
        "global_long_short_ratio": "/futures/data/globalLongShortAccountRatio",
        "top_long_short_account": "/futures/data/topLongShortAccountRatio",
        "top_long_short_position": "/futures/data/topLongShortPositionRatio",
        "taker_long_short": "/futures/data/takerlongshortRatio",
        "open_interest_hist": "/futures/data/openInterestHist",
        "account": "/fapi/v2/account",
        "positions": "/fapi/v2/positionRisk",
        "exchange_info": "/fapi/v1/exchangeInfo",
        "user_trades": "/fapi/v1/userTrades",
        "income": "/fapi/v1/income",
        "open_orders": "/fapi/v1/openOrders",
        "all_orders": "/fapi/v1/allOrders",
        "mark_price_klines": "/fapi/v1/markPriceKlines",
    })


@dataclass
class HMMConfig:
    """HMM model configuration."""

    # Model architecture
    n_components: int = 4
    covariance_type: str = "diag"
    n_iter: int = 200
    tol: float = 1e-3
    random_state: int = 7

    # Feature computation
    vol_window: int = 20
    ema_period: int = 20

    # Training data
    train_symbols: int = 60
    train_limit_1h: int = 500
    infer_limit: int = 800

    # Decision thresholds
    range_prob_min: float = 0.30
    trend_prob_max: float = 0.40

    # Dominance gating
    range_dominance_min: float = 0.95
    dominance_eps: float = 1e-10

    # Pass mode
    pass_mode: str = "dominance"

    # Posterior smoothing
    smooth_k: int = 5
    smooth_alpha: Optional[float] = None
    adaptive_transitions: bool = True
    adaptive_transition_window: int = 48
    adaptive_transition_weight: float = 0.35

    # Feature schema
    feature_schema: str = "v2"
    exogenous_signal_columns: Tuple[str, ...] = (
        "news_risk",
        "event_risk",
        "macro_risk",
        "political_risk",
    )

    # Hysteresis
    hysteresis_bars: int = 3

    # Direction-aware
    direction_aware: bool = True

    # Soft gating
    soft_gating: bool = True
    soft_gate_lower: float = 0.80
    soft_gate_upper: float = 0.95
    soft_gate_spread: float = 0.15

    # Tail-adjusted posterior correction (non-stationary enhancement)
    tail_correction_enabled: bool = True
    tail_correction_max_weight: float = 3.0
    tail_correction_threshold_sigma: float = 2.0



@dataclass
class StochasticConfig:
    """Stochastic regime check parameters."""

    enable: bool = True
    survival_horizon_bars: int = BOT_HORIZON_BARS_15M
    survival_mc_paths: int = 10000
    hurst_max_trending: float = 0.65
    ou_halflife_min_bars: int = 4
    ou_halflife_max_bars: int = 48
    # ERR-092 (2026-07-12): the OU half-life window is ADVISORY by default.
    # The half-life is still computed (de-trended fit, see
    # StochasticRegimeChecker.estimate_ou_params_detrended) and reported in
    # metrics/telemetry, but it no longer hard-rejects: measured on the
    # 35-symbol engine-verified cohort (session 981a7d2c) the window had
    # rho=-0.011/AUC 0.47 raw and rho=0.096/AUC 0.44 de-trended vs realized
    # grid PnL — no discrimination — while rejecting 47% of the universe and
    # 76%/47% of engine-profitable symbols. Precedent: survival-gate demotion
    # 2026-05-22 (stochastic.py). The Hurst gate remains hard. Set True to
    # restore the hard window.
    ou_halflife_gate_hard: bool = False


@dataclass
class EdgeTierConfig:
    """Adaptive edge-per-grid spacing policy (feasible-first, then optimize).

    Defines MEDIUM and BIG edge tiers anchored to microstructure cost basis.
    net_edge_pct = profit_per_grid_pct - micro_min_profit_required_pct

    MEDIUM: net_edge_pct in [medium_buffer_pct, big_buffer_pct)
    BIG:    net_edge_pct >= big_buffer_pct + upgrade_margin_pct
    """

    enable: bool = True
    medium_buffer_pct: float = 0.20   # net edge above micro floor for MEDIUM tier
    big_buffer_pct: float = 0.60      # net edge above micro floor for BIG tier
    upgrade_margin_pct: float = 0.05  # Big must exceed target by this extra margin
    ev_upgrade_min_delta: float = 0.05  # Minimum EV score delta required to upgrade


@dataclass
class PositionSizingConfig:
    """
    Position sizing controls for generalized Kelly and risk targeting.
    """

    # ERR-090 (2026-07-12): generalized Kelly DISABLED by default -- demoted to
    # opt-in research sizing. Its inputs are incoherent as measured on the live
    # pool: p = meta_prob = P(net MTM >= +3% in 7h) (calibrated base ~0.47 on
    # the meta pool) while b = avg_win/avg_loss from net>0/net<0 events on the
    # unconditional backtest_candidates pool (b=0.5814 -> break-even p=0.632);
    # the ALIGNED pairing (E = mfe_pct_initial>=3.0 on the same pool) measures
    # b_E=0.437 -> break-even p=0.696 vs pool P(E)=0.766, i.e. the model's
    # probability scale and the sizing pool's event rate disagree so badly that
    # every observed meta_prob (max 0.563) produced kelly_raw<0, max(0,raw)
    # floored capital_fraction to 0, and Stage B position_too_small rejected
    # every candidate from 2026-06-24 onward (yield 38-59/run -> 0). Sizing
    # authority reverts to the risk-budget PositionSizer (pre-06-24 behavior).
    # Re-enable ONLY with a re-derived coherent (p, b) pairing: either add
    # time_to_target_hours + harvest semantics to the profile pool, or
    # recalibrate p on the sizing pool.
    enable_generalized_kelly: bool = False
    fractional_kelly: float = 0.50
    optimize_fractional_kelly: bool = True
    fractional_kelly_sweep_min: float = 0.10
    fractional_kelly_sweep_max: float = 1.00
    fractional_kelly_sweep_step: float = 0.05
    drawdown_tolerance_pct: float = 10.0
    volatility_target_pct: float = 3.0
    reject_non_positive_edge: bool = True
    min_fraction: float = 0.0
    max_fraction: float = 1.0


@dataclass
class MicrostructureGateConfig:
    """Hard microstructure gate thresholds.

    All candidates must pass every check or be immediately rejected.
    """

    max_round_trip_pct: float = 1.20        # reject if RT cost > 1.20%
    max_spread_to_profit_ratio: float = 0.60  # reject if spread/profit > 60%
    max_funding_drag_pct: float = 0.80      # reject if |funding| > 0.80%


@dataclass
class OscillationScorerConfig:
    """Tradable Oscillation Score (TOS) weights and normalisation caps."""

    # Weights for the 7 sub-signals (order: gcf, mrs, rc, spr, fd, ds, lp)
    weights: Tuple[float, ...] = (0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10)

    # Saturation constant for grid-cross frequency
    gcf_saturation_k: float = 0.30

    # Cap for spread-to-profit inversion
    spr_cap: float = 0.80

    # Cap for funding drag inversion (as %)
    funding_drag_cap_pct: float = 1.0

    # Cap for liquidation proximity inversion
    liq_proximity_cap: float = 2.0


@dataclass
class TwoStageConfig:
    """Stage B deployment-approval thresholds."""

    min_tos: float = 40.0               # minimum Tradable Oscillation Score
    min_position_fraction: float = 0.05  # position sizer must allocate > 5%
    min_range_prob: float = 0.45         # stricter than scan-time threshold
    # FASTWIN-01 soft meta gate. ENABLED 2026-06-22 (ERR-059) now that the promoted
    # meta-labeler (artifact 20260622_230203, OOF AUC 0.770 [0.758,0.782]) is made
    # AUTHORITATIVE at the enrich decision by the ev_score-ordering fix. Stage B
    # rejects on a missing meta_prob (data_missing:meta, fail-closed) and on
    # meta_prob < min_meta_prob. min_meta_prob was originally the calibrated OOF
    # operating point for ~precision>=0.60 (tau=0.3727, recall=0.883) from that
    # retrain. Re-derived 2026-07-07 against the FASTWIN-02 artifact
    # (20260704_214452, OOF AUC 0.808): on the current model's promotion-gate OOF,
    # 0.37 sits at precision 0.625 / recall 0.888, and the precision>=0.60 point
    # would be tau=0.3378 (seed-stable [0.3374, 0.3403] over seeds 42/1/7/13/123).
    # 0.37 is RETAINED deliberately: on the fresh 62-candidate cohort
    # (outputs/audits/selection_experiments_20260707/) the current model's 5
    # sub-0.37 candidates went 0/5 on the 6h fast-winner target (mean net -4.08%
    # vs +1.86% selected), so the extra strictness costs nothing observed.
    meta_gate_enabled: bool = True
    min_meta_prob: float = 0.37


@dataclass
class RiskBudgetConfig:
    """Hard Risk Budget Position Sizer parameters."""

    min_fraction: float = 0.0         # allow hard-reject (fraction=0)
    max_fraction: float = 1.0         # absolute maximum allocation
    factor_floor: float = 0.30        # minimum for any single factor

    # Regime confidence
    regime_range_floor: float = 0.35  # range_prob below this → floor

    # Survival defaults when data missing
    default_survival_scale: float = 0.70
    default_micro_scale: float = 0.70
    default_vol_scale: float = 0.80

    # Volatility target
    vol_target_pct: float = 3.0


@dataclass
class HierarchicalLabelConfig:
    """Hierarchical training label thresholds."""

    # L1: Geometry Survival
    max_drawdown_floor_pct: float = -25.0  # reject if drawdown worse than -25%

    # L2: Execution Viability
    min_fills: int = 3                     # at least 3 grid fills
    min_duration_hours: float = 1.0        # grid must survive at least 1 hour

    # L2: Unrealized fraction noise gate (alignment-v1)
    max_unrealized_fraction: float = 0.50  # fail L2 if > 50% PnL is unrealized

    # L3: Net Return Hurdle — aligned with BarrierConfig.meta_hurdle_pct (3.0%)
    hurdle_pct: float = 3.0               # net PnL must exceed 3%


@dataclass(frozen=True)
class CPCVConfig:
    """
    Combinatorial Purged Cross-Validation parameters.

    AFML Compliance:
    - Supports time-based purging (recommended) aligned to event horizons
    - Maintains backwards compatibility with percent-based purging
    - Supports true holdout period (Station 4 compliance)
    """

    n_groups: int = 6
    n_test_groups: int = 2
    purge_pct: float = 0.02
    embargo_pct: float = 0.01
    purge_hours: float = 12.0
    embargo_hours: float = 1.5
    horizon_hours: float | None = BOT_HORIZON_HOURS  # Training horizon for t1 synthesis; defaults to purge_hours
    holdout_pct: float = 0.20
    sweep_target_pass_rate: Tuple[float, float] = (0.15, 0.25)
    auto_threshold: bool = True
    annual_trading_days: int = 365
    risk_free_rate: float = 0.0
    default_trial_count: int = 10  # conservative fallback when trial tracker unavailable

    @property
    def use_time_based_purging(self) -> bool:
        return self.purge_hours is not None

    @property
    def has_holdout(self) -> bool:
        return self.holdout_pct > 0

    def __post_init__(self):
        if self.n_groups < 3:
            raise ValueError("n_groups must be >= 3")
        if self.n_test_groups >= self.n_groups:
            raise ValueError("n_test_groups must be < n_groups")
        if not 0 <= self.holdout_pct < 1:
            raise ValueError("holdout_pct must be in [0, 1)")


@dataclass
class ValidationConfig:
    """Validation pipeline configuration."""

    # Feature flags
    enable_profile_gate: bool = False
    enable_profile_model: bool = True

    # Timeframe limits
    kline_limits: Dict[str, int] = field(default_factory=lambda: {
        "1h": 200,
        "15m": 800,
        "5m": 300,
        "1m": 500,
    })

    # Range quality
    range_lookback_bars_15m: int = 48
    range_quantile_low: float = 0.05
    range_quantile_high: float = 0.95

    # PnL-aware ranking
    pnl_hurdle_pct: float = 0.03
    # Operator validation window — intentionally local, NOT BOT_HORIZON_HOURS (D14.5-item5)
    pnl_horizon_hours: int = 6
    stop_loss_hurdle_pct: float = -0.10
    funding_extreme_threshold: float = 0.003

    # Data quality
    data_quality_strict: bool = False


@dataclass
class UtilityConfig:
    """Utility-based scoring parameters."""

    lambda_risk: Optional[float] = None
    min_threshold: float = 0.0

    # Baseline assumptions for provisional regime utility scoring
    # (used before actual grid params are known)
    provisional_profit_per_grid_pct: float = 0.8
    provisional_num_grids: int = 20
    provisional_range_size_pct: float = 3.0


@dataclass
class FeatureLoggingConfig:
    """Feature snapshot logging parameters."""

    enable: bool = True
    log_dir: str = "data/training_snapshots"
    min_features: int = 3


@dataclass
class DatabaseConfig:
    """Database configuration."""

    path: str = "data/validator.db"


@dataclass
class ArtifactConfig:
    """Model artifact configuration."""

    artifacts_dir: str = "artifacts"
    models_dir: str = "models"
    cache_dir: str = "data/cache"
    training_sets_dir: str = "data/training_sets"
    profile_dir: str = "data/profile"

    auto_version: bool = True


@dataclass
class PriceSeriesConfig:
    """PriceSeries streaming and storage parameters."""

    ring_buffer_size: int = 1500          # ~25h of 1m candles
    store_dir: str = "data/price_store"   # on-disk Parquet dir
    ws_reconnect_base_s: float = 1.0      # initial backoff
    ws_reconnect_max_s: float = 60.0      # max backoff
    ws_ping_interval_s: float = 180.0     # keep-alive ping
    gap_threshold_factor: float = 1.5     # gap = interval * factor
    backfill_bars: int = 500              # bars to backfill on startup
    ws_base_url: str = "wss://fstream.binance.com"


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    log_dir: str = "logs"

    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 3

    include_timestamp: bool = True
    include_run_id: bool = True
    include_artifact_version: bool = True


# =========================================================================
# MICRO-OSCILLATOR CONFIG
# =========================================================================


@dataclass
class MicroOscConfig:
    """Configuration for trending micro-oscillator archetype detection."""

    enabled: bool = True
    min_score: float = 0.45
    min_survival_prob: float = 0.60


# =========================================================================
# ROOT CONFIG
# =========================================================================


@dataclass
class Config:
    """
    Complete application configuration.

    This is the single source of truth for all configuration.
    """

    grid: GridConfig = field(default_factory=GridConfig)
    barrier: BarrierConfig = field(default_factory=BarrierConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    stochastic: StochasticConfig = field(default_factory=StochasticConfig)
    edge_tier: EdgeTierConfig = field(default_factory=EdgeTierConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    cpcv: CPCVConfig = field(default_factory=CPCVConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    feature_logging: FeatureLoggingConfig = field(default_factory=FeatureLoggingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    price_series: PriceSeriesConfig = field(default_factory=PriceSeriesConfig)
    microstructure_gate: MicrostructureGateConfig = field(default_factory=MicrostructureGateConfig)
    oscillation_scorer: OscillationScorerConfig = field(default_factory=OscillationScorerConfig)
    two_stage: TwoStageConfig = field(default_factory=TwoStageConfig)
    risk_budget: RiskBudgetConfig = field(default_factory=RiskBudgetConfig)
    hierarchical_label: HierarchicalLabelConfig = field(default_factory=HierarchicalLabelConfig)
    micro_osc: MicroOscConfig = field(default_factory=MicroOscConfig)
    # Phase 1-3 calibration configuration (v20260311)
    temperature_scaling_enabled: bool = True
    conformal_alpha: float = 0.20
    conformal_min_calibration_samples: int = 20
    mi_weight_shrinkage: float = 0.30
    adaptive_micro_gate_enabled: bool = False
    base_dir: Path = field(default_factory=lambda: _PROJECT_ROOT)

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a config path against the project base directory."""
        p = Path(relative_path)
        return p if p.is_absolute() else self.base_dir / p

    def __post_init__(self):
        """Compute derived values and validate configuration."""
        # Derive grid risk controls from barrier config
        self.grid.stop_loss_pct = self.barrier.sl_pct / 100.0
        self.grid.take_profit_pct = self.barrier.pt_pct / 100.0
        self.grid.close_fee_mode = str(self.grid.close_fee_mode).lower()

        # Load API keys from env
        if not self.binance.api_key:
            self.binance.api_key = os.getenv("BINANCE_API_KEY", "")
        if not self.binance.api_secret:
            self.binance.api_secret = os.getenv("BINANCE_API_SECRET", "")

        self._validate()

    def _validate(self):
        """Validate configuration values."""
        # ── Grid / Barrier / Fee validation ──────────────────────────────
        if self.grid.capital <= 0:
            raise ValueError(f"grid.capital must be > 0, got {self.grid.capital}")
        if self.grid.leverage_min <= 0 or self.grid.leverage_max <= 0:
            raise ValueError("grid.leverage_min and leverage_max must be > 0")
        if self.grid.leverage_min > self.grid.leverage_max:
            raise ValueError("grid.leverage_min must be <= leverage_max")
        if not 0 <= self.grid.maker_fee < 0.01:
            raise ValueError(f"grid.maker_fee must be in [0, 0.01), got {self.grid.maker_fee}")
        if not 0 <= self.grid.taker_fee < 0.01:
            raise ValueError(f"grid.taker_fee must be in [0, 0.01), got {self.grid.taker_fee}")
        if self.grid.close_fee_mode not in ("maker", "taker"):
            raise ValueError("grid.close_fee_mode must be 'maker' or 'taker'")

        if self.barrier.pt_pct <= 0:
            raise ValueError(f"barrier.pt_pct must be > 0, got {self.barrier.pt_pct}")
        if self.barrier.sl_pct >= 0:
            raise ValueError(f"barrier.sl_pct must be < 0, got {self.barrier.sl_pct}")
        if self.barrier.time_hours <= 0:
            raise ValueError(f"barrier.time_hours must be > 0, got {self.barrier.time_hours}")
        if self.barrier.k_pt <= 0 or self.barrier.k_sl <= 0:
            raise ValueError("barrier.k_pt and k_sl must be > 0")

        # ── HMM validation ──────────────────────────────────────────────
        if self.hmm.n_components < 2:
            raise ValueError("HMM n_components must be >= 2")
        if self.hmm.covariance_type not in ["diag", "full", "spherical", "tied"]:
            raise ValueError(f"Invalid HMM covariance_type: {self.hmm.covariance_type}")
        if not 0 < self.hmm.range_prob_min < 1:
            raise ValueError("HMM range_prob_min must be in (0, 1)")
        if not 0 < self.hmm.trend_prob_max < 1:
            raise ValueError("HMM trend_prob_max must be in (0, 1)")
        if self.hmm.adaptive_transition_window < 3:
            raise ValueError("HMM adaptive_transition_window must be >= 3")
        if not 0 <= self.hmm.adaptive_transition_weight <= 1:
            raise ValueError("HMM adaptive_transition_weight must be in [0, 1]")

        # ── Logging validation ───────────────────────────────────────────
        if self.logging.level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid logging level: {self.logging.level}")

        # ── Cross-config consistency ─────────────────────────────────────
        kline_15m = self.validation.kline_limits.get("15m", 0)
        if kline_15m < self.hmm.infer_limit:
            raise ValueError(
                f"kline_limits['15m'] ({kline_15m}) must be >= "
                f"hmm.infer_limit ({self.hmm.infer_limit})"
            )

        # ── Hurdle threshold synchronization (alignment-v1) ───────────
        if self.hierarchical_label.hurdle_pct != self.barrier.meta_hurdle_pct:
            raise ValueError(
                f"Config drift: HierarchicalLabelConfig.hurdle_pct "
                f"({self.hierarchical_label.hurdle_pct}) != "
                f"BarrierConfig.meta_hurdle_pct ({self.barrier.meta_hurdle_pct}). "
                f"These must be synchronized to prevent label/deployment divergence."
            )


# =========================================================================
# LOADING FUNCTIONS
# =========================================================================


def load_from_env(config: Optional[Config] = None) -> Config:
    """
    Apply environment variable overrides to a config.

    Args:
        config: Base config to apply overrides to (default: fresh Config)

    Returns:
        Config object with environment overrides
    """
    if config is None:
        config = Config()

    # HMM overrides
    if val := os.getenv("HMM_N_COMPONENTS"):
        config.hmm.n_components = int(val)
    if val := os.getenv("HMM_COVARIANCE_TYPE"):
        config.hmm.covariance_type = val
    if val := os.getenv("HMM_N_ITER"):
        config.hmm.n_iter = int(val)
    if val := os.getenv("HMM_ADAPTIVE_TRANSITIONS"):
        config.hmm.adaptive_transitions = val.lower() in ("true", "1", "yes")
    if val := os.getenv("HMM_ADAPTIVE_TRANSITION_WINDOW"):
        config.hmm.adaptive_transition_window = int(val)
    if val := os.getenv("HMM_ADAPTIVE_TRANSITION_WEIGHT"):
        config.hmm.adaptive_transition_weight = float(val)
    if val := os.getenv("HMM_EXOGENOUS_SIGNAL_COLUMNS"):
        cols = tuple(x.strip() for x in val.split(",") if x.strip())
        if cols:
            config.hmm.exogenous_signal_columns = cols
    if val := os.getenv("HMM_RANGE_PROB_MIN"):
        config.hmm.range_prob_min = float(val)
    if val := os.getenv("HMM_TREND_PROB_MAX"):
        config.hmm.trend_prob_max = float(val)
    if val := os.getenv("HMM_TAIL_CORRECTION_ENABLED"):
        config.hmm.tail_correction_enabled = val.lower() in ("true", "1", "yes")
    if val := os.getenv("HMM_TAIL_CORRECTION_MAX_WEIGHT"):
        config.hmm.tail_correction_max_weight = float(val)
    if val := os.getenv("HMM_TAIL_CORRECTION_THRESHOLD_SIGMA"):
        config.hmm.tail_correction_threshold_sigma = float(val)

    # Validation flags
    if val := os.getenv("ENABLE_PROFILE_GATE"):
        config.validation.enable_profile_gate = val.lower() in ("true", "1", "yes")
    if val := os.getenv("ENABLE_PROFILE_MODEL"):
        config.validation.enable_profile_model = val.lower() in ("true", "1", "yes")
    if val := os.getenv("ENABLE_STOCHASTIC_CHECKS"):
        config.stochastic.enable = val.lower() in ("true", "1", "yes")

    # Artifact configuration
    if val := os.getenv("ARTIFACTS_DIR"):
        config.artifacts.artifacts_dir = val
    if val := os.getenv("CACHE_DIR"):
        config.artifacts.cache_dir = val

    # Logging configuration
    if val := os.getenv("LOGGING_LEVEL"):
        config.logging.level = val.upper()
    if val := os.getenv("LOG_DIR"):
        config.logging.log_dir = val

    # Binance API (also loaded in __post_init__, but env takes precedence)
    if val := os.getenv("BINANCE_API_KEY"):
        config.binance.api_key = val
    if val := os.getenv("BINANCE_API_SECRET"):
        config.binance.api_secret = val

    # Phase 1-3 calibration overrides
    if val := os.getenv("TEMPERATURE_SCALING_ENABLED"):
        config.temperature_scaling_enabled = val.lower() in ("true", "1", "yes")
    if val := os.getenv("CONFORMAL_ALPHA"):
        config.conformal_alpha = float(val)
    if val := os.getenv("MI_WEIGHT_SHRINKAGE"):
        config.mi_weight_shrinkage = float(val)
    if val := os.getenv("ADAPTIVE_MICRO_GATE_ENABLED"):
        config.adaptive_micro_gate_enabled = val.lower() in ("true", "1", "yes")
    if val := os.getenv("CONFORMAL_MIN_CALIBRATION_SAMPLES"):
        config.conformal_min_calibration_samples = int(val)

    # Micro-oscillator archetype
    if val := os.getenv("MICRO_OSC_ENABLED"):
        config.micro_osc.enabled = val.lower() in ("1", "true", "yes")

    # Base directory override
    if val := os.getenv("NEUTRALGRID_BASE_DIR"):
        config.base_dir = Path(val).resolve()

    return config


def load_from_file(config_path: Path) -> Config:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to config file (.yaml, .yml, or .json)

    Returns:
        Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix in [".yaml", ".yml"]:
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise ValueError("pyyaml not installed. Install with: pip install pyyaml")
    elif config_path.suffix == ".json":
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    return _dict_to_config(data)


def _dict_to_config(data: Dict[str, Any]) -> Config:
    """Convert dictionary to Config object."""
    config = Config()

    # Apply nested dict values to matching sub-config fields
    _sub_configs = {
        "grid": config.grid,
        "barrier": config.barrier,
        "indicators": config.indicators,
        "binance": config.binance,
        "hmm": config.hmm,
        "stochastic": config.stochastic,
        "edge_tier": config.edge_tier,
        "position_sizing": config.position_sizing,
        "cpcv": config.cpcv,
        "validation": config.validation,
        "utility": config.utility,
        "feature_logging": config.feature_logging,
        "database": config.database,
        "artifacts": config.artifacts,
        "logging": config.logging,
        "price_series": config.price_series,
        "microstructure_gate": config.microstructure_gate,
        "oscillation_scorer": config.oscillation_scorer,
        "two_stage": config.two_stage,
        "risk_budget": config.risk_budget,
        "hierarchical_label": config.hierarchical_label,
    }

    for section_name, sub_cfg in _sub_configs.items():
        if section_name in data:
            from dataclasses import fields as dc_fields
            if getattr(type(sub_cfg), "__dataclass_params__", None) and type(sub_cfg).__dataclass_params__.frozen:
                # Frozen dataclass: reconstruct with overrides
                field_names = {f.name for f in dc_fields(sub_cfg)}
                kwargs = {f.name: getattr(sub_cfg, f.name) for f in dc_fields(sub_cfg)}
                for key, value in data[section_name].items():
                    if key in field_names:
                        kwargs[key] = value
                new_sub = type(sub_cfg)(**kwargs)
                setattr(config, section_name, new_sub)
            else:
                for key, value in data[section_name].items():
                    if hasattr(sub_cfg, key):
                        setattr(sub_cfg, key, value)

    # ERR-081: overrides above are applied AFTER Config() construction, so the
    # __post_init__ validation has not seen them. Re-validate so a config file
    # cannot bypass the startup invariants (e.g. the hurdle_pct cross-check).
    config._validate()

    return config


def load_config(
    config_path: Optional[Path] = None,
    use_env: bool = True,
) -> Config:
    """
    Load configuration with precedence: env vars > file > defaults.

    Args:
        config_path: Optional path to config file
        use_env: Whether to override with environment variables

    Returns:
        Config object
    """
    config = Config()

    if config_path is not None:
        try:
            config = load_from_file(config_path)
        except (FileNotFoundError, ValueError, ImportError, KeyError, TypeError) as e:
            warnings.warn(f"Failed to load config file {config_path}: {e}")

    if use_env:
        config = load_from_env(config)

    return config


# =========================================================================
# GLOBAL SINGLETON
# =========================================================================


_config: Optional[Config] = None
_config_lock = threading.Lock()


def get_config() -> Config:
    """
    Get global config instance.

    Lazy-loads on first access. Config is loaded once and cached.
    Thread-safe via double-checked locking.

    Returns:
        Config object
    """
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = load_config()
    return _config


def reset_config():
    """Reset global config instance (useful for testing)."""
    global _config
    with _config_lock:
        _config = None
