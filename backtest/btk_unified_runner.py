"""
Unified backtest runner API — single source of truth for label generation.

This module is THE entry point for all backtest execution in the pipeline.
It wraps the realistic backtester as the sole engine and enforces:

1. **Label contract**: every result is validated before returning
2. **Engine settings serialization**: every result is self-describing
3. **Training defaults**: ``build_training_config()`` applies consistent
   settings for label generation

Usage::

    from backtest.btk_unified_runner import run_backtest, build_training_config

    # Training label generation (uses training defaults)
    cfg = build_training_config("BTCUSDT", 40000, 42000, 20, capital=400)
    result = run_backtest(cfg, klines_df)

    # Ad-hoc / out-of-sample backtesting (custom config)
    from backtest.backtest_realistic import GridConfig
    cfg = GridConfig(symbol="BTCUSDT", lower=40000, upper=42000, num_grids=20)
    result = run_backtest(cfg, klines_df)

The pluggable engine design means all pipeline code goes through this runner.
If the engine ever needs to change, it changes in one place.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

import pandas as pd

from backtest.backtest_realistic import GridConfig, RealisticGridBacktester
from backtest.btk_seed_state import (
    CANDIDATE_TIME_GEOMETRIC_PROFILE,
    CANDIDATE_TIME_PUBLIC_MARKET_PROFILE,
    LEGACY_REALISM_PROFILE,
    build_candidate_time_geometry_seed,
    validate_realism_profile,
)
from backtest.btk_label_contract import (
    FORMULA_VERSION,
    LABEL_CONTRACT_VERSION,
    TRAINING_ENGINE_DEFAULTS,
    extract_engine_settings,
    validate_engine_result,
)
# Step 4 (Plan v6.0): Import ENGINE_VERSION from single source of truth.
try:
    from neutralgrid.core.constants import ENGINE_VERSION as ENGINE_VERSION
except ImportError:  # pragma: no cover — editable install expected
    # backtest/ lives outside the installed package. Bootstrap the repo's
    # src/ layout onto sys.path and re-import. Never hardcode version
    # literals here — they are single-sourced in core.constants
    # (safety-invariants.md, Version Constants). A hard ImportError here is
    # preferable to silently drifting stale literals.
    import sys
    from pathlib import Path

    _SRC_DIR = Path(__file__).resolve().parents[1] / "src"
    if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from neutralgrid.core.constants import ENGINE_VERSION as ENGINE_VERSION

if TYPE_CHECKING:
    from backtest.btk_seed_state import SeedState

logger = logging.getLogger(__name__)


def build_training_config(
    symbol: str,
    lower: float,
    upper: float,
    num_grids: int,
    **overrides: Any,
) -> GridConfig:
    """Build a GridConfig with training-standard defaults for label generation.

    Applies ``TRAINING_ENGINE_DEFAULTS`` as the base, then merges any
    caller-provided overrides (non-None values only).  This ensures training
    labels are always generated with consistent, documented settings.

    Parameters
    ----------
    symbol, lower, upper, num_grids : required grid geometry
    **overrides : keyword arguments
        Any ``GridConfig`` field can be overridden.  Common overrides:
        ``capital``, ``leverage``, ``funding_rate``.
        ``None`` values are ignored (training default is used).

    Returns
    -------
    GridConfig
        Ready-to-use config with training defaults merged.
    """
    settings: Dict[str, Any] = dict(TRAINING_ENGINE_DEFAULTS)

    # Step 9 (Plan v6.0): Physics lock — funding_mode is physics-locked.
    # Overriding it changes label semantics (continuous vs snapshot accrual).
    # Log a warning for any physics field override that differs from training
    # defaults, and tag the result as non-authoritative research.
    _PHYSICS_LOCKED_FIELDS = frozenset({
        "mode", "grid_count_semantics", "funding_mode", "close_fee_mode", "order_delay_bars", "slippage_bps",
        "maker_fee", "taker_fee", "funding_interval_bars",
        "maintenance_margin_rate", "tick_size", "step_size", "price_source",
        "valuation_price_source",
        "spread_bps", "fill_mode", "margin_mode", "global_cooldown_bars",
        "cb_enabled", "cb_max_dd_pct", "cb_trailing_activate_pct",
        "cb_trailing_offset_pct", "cb_inventory_imbalance_ratio",
        "cb_inventory_imbalance_dd_pct",
    })
    _physics_overridden = False
    _overridden_fields: list[str] = []
    for k, v in overrides.items():
        if v is None:
            continue
        if k in _PHYSICS_LOCKED_FIELDS and k in TRAINING_ENGINE_DEFAULTS:
            default_val = TRAINING_ENGINE_DEFAULTS[k]
            if v != default_val:
                _physics_overridden = True
                _overridden_fields.append(k)
                logger.warning(
                    "Physics override: %s=%r (training default: %r). "
                    "Result will be tagged is_authoritative=False.",
                    k, v, default_val,
                )

    settings.update({k: v for k, v in overrides.items() if v is not None})

    # Stamp physics override metadata for runner-derived status (Step 10)
    config = GridConfig(
        symbol=symbol,
        lower=lower,
        upper=upper,
        num_grids=num_grids,
        **settings,
    )
    # Attach metadata as runtime attributes (not dataclass fields)
    object.__setattr__(config, "_physics_overridden", _physics_overridden)
    object.__setattr__(config, "_overridden_fields", _overridden_fields)
    return config


def run_backtest(
    config: GridConfig,
    klines_df: pd.DataFrame,
    *,
    seed_state: Optional["SeedState"] = None,
    realism_profile: str = LEGACY_REALISM_PROFILE,
) -> Dict[str, Any]:
    """Run a single realistic backtest and return a validated, annotated result.

    This is THE single entry point for all backtest execution in the system.
    Every result is:

    1. Produced by ``RealisticGridBacktester`` (the sole engine)
    2. Annotated with engine settings for reproducibility
    3. Validated against the label contract

    Parameters
    ----------
    config : GridConfig
        Grid configuration (use ``build_training_config()`` for training).
    klines_df : pd.DataFrame
        1-minute OHLCV DataFrame with columns: timestamp, open, high, low,
        close, volume.
    seed_state : SeedState, optional
        Replay-derived ladder state to seed the backtester at bar 0.
    realism_profile : str
        Explicit realism profile. ``legacy`` preserves existing behavior.
        Candidate-time profiles build a partial geometry seed from the
        configured grid and first replay price when no external seed is
        supplied.

    Returns
    -------
    dict
        Backtest result dict with all standard fields plus engine settings
        metadata.  Guaranteed to pass ``validate_engine_result()``.

    Raises
    ------
    ValueError
        If the engine produces a result missing required label fields
        (should never happen with the realistic engine — indicates a bug).
    """
    profile = validate_realism_profile(realism_profile)
    if seed_state is not None and profile != LEGACY_REALISM_PROFILE:
        raise ValueError(
            "External seed_state cannot be combined with "
            f"realism_profile={profile!r}; use legacy seeded mode or the "
            "profile without external seed data."
        )

    backtester = RealisticGridBacktester(config)

    active_seed = seed_state
    if active_seed is None and profile in {
        CANDIDATE_TIME_GEOMETRIC_PROFILE,
        CANDIDATE_TIME_PUBLIC_MARKET_PROFILE,
    }:
        if not klines_df.empty and "close" in klines_df.columns and "timestamp" in klines_df.columns:
            first_row = klines_df.iloc[0]
            first_close = float(first_row["close"])
            first_ts = pd.Timestamp(first_row["timestamp"])
            if not pd.isna(first_ts):
                first_dt = first_ts.to_pydatetime()
                if not isinstance(first_dt, datetime):
                    first_dt = None
            else:
                first_dt = None
            if first_dt is not None:
                active_seed = build_candidate_time_geometry_seed(
                    symbol=config.symbol,
                    grid_levels=backtester.grid_levels,
                    start_price=first_close,
                    t0=first_dt,
                )

    if active_seed is not None:
        backtester.seed_from_state(active_seed)

    result = backtester.run(klines_df)

    # ── PnL curve feature extraction ──────────────────────────────────────
    # Default endogenous time-to-target fields so EVERY result carries them
    # (PIPELINE_FIX v2): the fast-winner label keys on time_to_target_hours.
    result.setdefault("time_to_target_hours", float("nan"))
    result.setdefault("target_reached", False)
    equity_curve = backtester.equity_curve
    if equity_curve and len(equity_curve) >= 3:
        starting_capital = result.get("capital_used", equity_curve[0])
        pnl_curve = [eq - starting_capital for eq in equity_curve]
        try:
            from neutralgrid.metrics.pnl_curve_features_v20260310 import (
                extract_pnl_curve_features,
                extract_time_to_threshold,
            )

            curve_features = extract_pnl_curve_features(pnl_curve)
            if curve_features is not None:
                result.update(curve_features.to_dict())
                logger.debug(
                    "PnL curve features: shape=%s, points=%d",
                    curve_features.shape_class,
                    curve_features.points,
                )
            # PIPELINE_FIX v2: endogenous time-to-target (first 1m bar where the
            # net MTM PnL curve reaches +3% of capital). Same net basis as
            # net_pnl_pct; this is the label measure, never a feature.
            ttt = extract_time_to_threshold(
                pnl_curve,
                float(starting_capital),
                threshold_pct=3.0,
                bar_minutes=1.0,
            )
            result["time_to_target_hours"] = ttt["time_to_target_hours"]
            result["target_reached"] = ttt["target_reached"]
        except Exception as exc:
            logger.debug("PnL curve feature extraction skipped: %s", exc)

    # ── Annotate with engine settings for reproducibility ─────────────────
    result["engine_version"] = ENGINE_VERSION
    result["label_contract_version"] = LABEL_CONTRACT_VERSION
    result["formula_version"] = FORMULA_VERSION
    result["backtest_run_id"] = str(uuid.uuid4())
    result["realism_profile"] = profile
    result["seed_state_source"] = active_seed.source if active_seed is not None else "none"
    result.update(extract_engine_settings(config))

    # ── Step 10 (Plan v6.0): Runner-derived authoritative status ─────────
    # A result is authoritative IFF ALL physics fields match
    # TRAINING_ENGINE_DEFAULTS AND funding_rate_series is None.
    # This check is independent of how the config was built — it compares
    # actual config values against the canonical training physics.
    _PHYSICS_FIELDS_TO_CHECK = (
        "mode", "grid_count_semantics", "funding_mode", "close_fee_mode", "order_delay_bars", "slippage_bps",
        "maker_fee", "taker_fee", "funding_interval_bars",
        "maintenance_margin_rate", "tick_size", "step_size", "price_source",
        "valuation_price_source",
        "spread_bps", "fill_mode", "margin_mode", "global_cooldown_bars",
        "cb_enabled", "cb_max_dd_pct", "cb_trailing_activate_pct",
        "cb_trailing_offset_pct", "cb_inventory_imbalance_ratio",
        "cb_inventory_imbalance_dd_pct",
    )
    _mismatched_fields: list[str] = []
    for _field in _PHYSICS_FIELDS_TO_CHECK:
        config_val = getattr(config, _field, None)
        default_val = TRAINING_ENGINE_DEFAULTS.get(_field)
        if config_val is not None and default_val is not None and config_val != default_val:
            _mismatched_fields.append(_field)

    _has_funding_series = getattr(config, "funding_rate_series", None) is not None
    is_authoritative = len(_mismatched_fields) == 0 and not _has_funding_series
    result["is_authoritative"] = is_authoritative

    if not is_authoritative:
        reasons = []
        if _mismatched_fields:
            reasons.append(f"physics_mismatches={_mismatched_fields}")
        if _has_funding_series:
            reasons.append("funding_rate_series=non-None")
        logger.info(
            "Result tagged is_authoritative=False: %s",
            ", ".join(reasons),
        )

    # ── Validate label contract ───────────────────────────────────────────
    validate_engine_result(result)

    return result
