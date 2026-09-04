"""
Label contract for the NEUTRAL Grid Bot backtesting system.

Defines the authoritative contract for training labels produced by the
realistic backtester — the single source of truth for meta-labeler training.

Contract guarantees:
- Horizon is always 6h (standard)
- Label is always: label_positive_by_horizon = final_equity > capital_used
- Equity is always mark-to-market (unrealized PnL included)
- Funding is always explicit (default: continuous for training)
- Fees and slippage are always included

All training labels MUST originate from ``btk_unified_runner.run_backtest()``.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Tuple, Type

import numpy as np

# Step 4 (Plan v6.0): Import version constants from single source of truth.
# Never hardcode version strings — always import from constants.py.
# try/except because backtest/ is outside the installed package.
try:
    from neutralgrid.core.constants import (
        BOT_HORIZON_HOURS as BOT_HORIZON_HOURS,
        FORMULA_VERSION as FORMULA_VERSION,
        LABEL_CONTRACT_VERSION as LABEL_CONTRACT_VERSION,
    )
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
    from neutralgrid.core.constants import (
        BOT_HORIZON_HOURS as BOT_HORIZON_HOURS,
        FORMULA_VERSION as FORMULA_VERSION,
        LABEL_CONTRACT_VERSION as LABEL_CONTRACT_VERSION,
    )

# ── Standard horizon ─────────────────────────────────────────────────────────
STANDARD_HORIZON_HOURS = BOT_HORIZON_HOURS

# ── Required fields in every engine result used for label generation ─────────
# If any of these are missing, the result MUST be rejected.
REQUIRED_LABEL_FIELDS: FrozenSet[str] = frozenset({
    "label_positive_by_horizon",
    "final_equity",
    "capital_used",
    "max_drawdown_pct",
    "mae",
    "mfe",
    "mae_pct_initial",
    "mfe_pct_initial",
    "net_pnl_pct",
    "mode",
    "grid_count_semantics",
    "funding_mode",
    "funding_fees",
    "fees_paid",
    "liquidated",
    "exit_penalty_pct",
    "cb_triggered",
    # ── alignment-v1: realized PnL decomposition ──
    "realized_net_pnl",
    "realized_net_pnl_pct",
    "unrealized_fraction",
    # ── H*-censoring flag (DURATION_FIX §13 / D14.1 + D14.2) ──
    # True if at least one position was force-closed at max_holding_bars
    # (i.e. the bot reached H* with open inventory). Enables downstream
    # consumers to distinguish truly-completed bots from H*-truncated ones.
    "horizon_censored",
})

# ── Engine settings fields serialized into each output row ───────────────────
# These make every backtest result self-describing and reproducible.
ENGINE_SETTINGS_FIELDS: List[str] = [
    "engine_version",
    "label_contract_version",
    "formula_version",
    "mode",
    "grid_count_semantics",
    "funding_mode",
    "close_fee_mode",
    "slippage_bps",
    "order_delay_bars",
    "maker_fee",
    "taker_fee",
    "max_holding_bars",
    "funding_rate",
    "funding_interval_bars",
    "maintenance_margin_rate",
    "tick_size",
    "step_size",
    "price_source",
    "valuation_price_source",
    "funding_rate_series_len",
    "margin_mode",
    "spread_bps",
    "fill_mode",
    "global_cooldown_bars",
    "capital_fraction",
    "volatility_target_pct",
    "volatility_proxy_pct",
    "volatility_scale_min",
    "volatility_scale_max",
    "cb_enabled",
    "cb_max_dd_pct",
    "cb_trailing_activate_pct",
    "cb_trailing_offset_pct",
    "cb_inventory_imbalance_ratio",
    "cb_inventory_imbalance_dd_pct",
]

# ── Training-standard engine defaults ────────────────────────────────────────
# Applied by ``btk_unified_runner.build_training_config()`` for consistent
# label generation.  Callers can override individual values.
TRAINING_ENGINE_DEFAULTS: Dict[str, Any] = {
    "mode": "geometric",               # Future training labels use geometric grid identity
    "grid_count_semantics": "binance_displayed_intervals",
    "funding_mode": "continuous",       # Per-bar prorated funding (most realistic)
    "close_fee_mode": "maker",          # Live data shows 100% maker fills on TP closes
    "order_delay_bars": 2,              # Calibrated from live data
    "slippage_bps": 0.0,               # No slippage by default
    "max_holding_bars": int(BOT_HORIZON_HOURS * 60),  # H* inventory lifecycle (6h @ 1m = 360)
    "maker_fee": 0.0002,               # Binance USDT-M maker fee
    "taker_fee": 0.0005,               # Binance USDT-M taker fee
    "funding_rate": 0.0001,            # Standard 0.01% per 8h period
    "funding_interval_bars": 480,       # 8h * 60 = 480 1-min bars
    "maintenance_margin_rate": 0.004,   # Binance bracket 1 (0-50k notional)
    "tick_size": 0.0,                   # No rounding (backward compat)
    "step_size": 0.0,                   # No rounding (backward compat)
    "price_source": "last",             # Last-price klines (default)
    "valuation_price_source": "last",      # MTM valuation source (last unless explicit mark series supplied)
    "spread_bps": 0.0,                  # No spread (backward compat)
    "fill_mode": "wick",                # Intrabar high/low touches approximate resting grid limit fills
    "margin_mode": "isolated",          # Isolated margin (default)
    "global_cooldown_bars": 0,          # Resting grid orders should not globally pause all levels after one fill
    # Equity Circuit Breaker — loss truncation for positive EV
    # Calibrated 2026-03-10 A/B test (30 candidates, 3 configs):
    #   DD-only Sharpe 1.22, DD+trail(5/3) Sharpe 1.41, CB-OFF Sharpe 1.40
    #   Trailing lock fires too often on grid bot equity chop — disabled.
    #   DD stop capped 3/30 bots at ~-5% that would have hit -10%+ in bear.
    #
    # CB firing semantics for neutral grid bots (bidirectional):
    #   Branch A (imbalance-first): total_positions >= 3 AND
    #     dd_from_initial_pct >= cb_inventory_imbalance_dd_pct (3%) AND
    #     imbalance_ratio >= cb_inventory_imbalance_ratio (0.85)
    #     → fires when one side (long or short) dominates AND DD >= 3%.
    #     In trending markets, one side accumulates while the other closes,
    #     producing genuine directional imbalance.
    #   Branch B (max-dd-first): total_positions < 3 OR DD < 3%
    #     → only max_dd_pct at 5% can fire.
    "cb_enabled": False,                  # Off unless an explicit live stop/circuit rule is modeled
    "cb_max_dd_pct": 5.0,                # Exit if loss from capital > 5%
    "cb_trailing_activate_pct": 999.0,    # DISABLED — grid equity too choppy
    "cb_trailing_offset_pct": 999.0,      # DISABLED — see trailing_activate
    "cb_inventory_imbalance_ratio": 0.85, # Exit if >85% on one side + DD > 3%
    "cb_inventory_imbalance_dd_pct": 3.0, # Raised from 2.0: less sensitive
}


def validate_engine_result(result: Dict[str, Any]) -> None:
    """Validate that an engine result meets the label contract.

    Raises ``ValueError`` if required label fields are missing.
    Raises ``TypeError`` if critical fields have wrong types.

    This is called automatically by ``btk_unified_runner.run_backtest()``
    on every result before returning to the caller.
    """
    missing = REQUIRED_LABEL_FIELDS - set(result.keys())
    if missing:
        raise ValueError(
            f"Engine result missing required label fields: {sorted(missing)}. "
            "Ensure the result comes from RealisticGridBacktester.run()."
        )

    # Type checks on critical label fields.
    # Accept both Python bool and numpy.bool_ (numpy 2.4+ reports __name__
    # as "bool" but isinstance(np.bool_(True), bool) is False).
    _bool_types: Tuple[Type[Any], ...] = (bool, np.bool_)
    _numeric_types: Tuple[Type[Any], ...] = (int, float, np.integer, np.floating)

    label = result["label_positive_by_horizon"]
    if not isinstance(label, _bool_types):
        raise TypeError(
            f"label_positive_by_horizon must be bool, got {type(label).__name__}"
        )

    liquidated = result["liquidated"]
    if not isinstance(liquidated, _bool_types):
        raise TypeError(
            f"liquidated must be bool, got {type(liquidated).__name__}"
        )

    cb_triggered = result["cb_triggered"]
    if not isinstance(cb_triggered, _bool_types):
        raise TypeError(
            f"cb_triggered must be bool, got {type(cb_triggered).__name__}"
        )

    horizon_censored = result["horizon_censored"]
    if not isinstance(horizon_censored, _bool_types):
        raise TypeError(
            f"horizon_censored must be bool, got {type(horizon_censored).__name__}"
        )

    mode = str(result["mode"]).strip().lower()
    if mode not in {"arithmetic", "geometric"}:
        raise ValueError(f"mode must be arithmetic or geometric, got {result['mode']!r}")

    grid_count_semantics = str(result["grid_count_semantics"]).strip().lower()
    if grid_count_semantics not in {"legacy_line_count", "binance_displayed_intervals"}:
        raise ValueError(
            "grid_count_semantics must be legacy_line_count or "
            f"binance_displayed_intervals, got {result['grid_count_semantics']!r}"
        )

    for field in (
        "final_equity",
        "capital_used",
        "max_drawdown_pct",
        "mae",
        "mfe",
        "mae_pct_initial",
        "mfe_pct_initial",
        "net_pnl_pct",
        "exit_penalty_pct",
    ):
        val = result[field]
        if not isinstance(val, _numeric_types):
            raise TypeError(
                f"{field} must be numeric, got {type(val).__name__}"
            )


def extract_engine_settings(config: Any) -> Dict[str, Any]:
    """Extract reproducibility-critical engine settings from a GridConfig.

    Returns a dict suitable for merging into the backtest result row so that
    every output is self-describing.

    Parameters
    ----------
    config : GridConfig
        The config object used for the backtest run.  Accepts any object with
        the expected attributes (duck-typed for testability).
    """
    frs = getattr(config, "funding_rate_series", None)
    return {
        "funding_mode": getattr(config, "funding_mode", "unknown"),
        "mode": getattr(config, "mode", "unknown"),
        "grid_count_semantics": getattr(config, "grid_count_semantics", "unknown"),
        "close_fee_mode": getattr(config, "close_fee_mode", "unknown"),
        "slippage_bps": getattr(config, "slippage_bps", 0.0),
        "order_delay_bars": getattr(config, "order_delay_bars", 0),
        "maker_fee": getattr(config, "maker_fee", 0.0),
        "taker_fee": getattr(config, "taker_fee", 0.0),
        "max_holding_bars": getattr(config, "max_holding_bars", 0),
        "funding_rate": getattr(config, "funding_rate", 0.0),
        "funding_interval_bars": getattr(config, "funding_interval_bars", 480),
        "maintenance_margin_rate": getattr(config, "maintenance_margin_rate", 0.004),
        "tick_size": getattr(config, "tick_size", 0.0),
        "step_size": getattr(config, "step_size", 0.0),
        "price_source": getattr(config, "price_source", "last"),
        "valuation_price_source": getattr(config, "valuation_price_source", "last"),
        "funding_rate_series_len": len(frs) if frs else 0,
        "margin_mode": getattr(config, "margin_mode", "isolated"),
        "spread_bps": getattr(config, "spread_bps", 0.0),
        "fill_mode": getattr(config, "fill_mode", "close"),
        "global_cooldown_bars": getattr(config, "global_cooldown_bars", 0),
        "capital_fraction": getattr(config, "capital_fraction", 1.0),
        "volatility_target_pct": getattr(config, "volatility_target_pct", None),
        "volatility_proxy_pct": getattr(config, "volatility_proxy_pct", None),
        "volatility_scale_min": getattr(config, "volatility_scale_min", 0.05),
        "volatility_scale_max": getattr(config, "volatility_scale_max", 1.0),
        "cb_enabled": getattr(config, "cb_enabled", False),
        "cb_max_dd_pct": getattr(config, "cb_max_dd_pct", 5.0),
        "cb_trailing_activate_pct": getattr(config, "cb_trailing_activate_pct", 2.0),
        "cb_trailing_offset_pct": getattr(config, "cb_trailing_offset_pct", 1.0),
        "cb_inventory_imbalance_ratio": getattr(config, "cb_inventory_imbalance_ratio", 0.85),
        "cb_inventory_imbalance_dd_pct": getattr(config, "cb_inventory_imbalance_dd_pct", 3.0),
    }
