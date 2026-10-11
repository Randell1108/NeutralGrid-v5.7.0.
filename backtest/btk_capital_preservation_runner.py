"""Capital-preservation specialization of the canonical unified backtester.

This module keeps the existing runner as the execution authority while applying
the operator's explicit per-bot manual and automatic exit rules. Results are
diagnostic research evidence because the policy changes canonical physics.
"""
from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import pandas as pd

from backtest.backtest_realistic import GridConfig
from backtest.btk_seed_state import LEGACY_REALISM_PROFILE
from backtest.btk_unified_runner import build_training_config, run_backtest
from neutralgrid.core.constants import BOT_HORIZON_HOURS

if TYPE_CHECKING:
    from backtest.btk_seed_state import SeedState


CAPITAL_PRESERVATION_PROFILE_VERSION = "capital_preservation_v1"
MANUAL_CANCEL_LOSS_PCT = 5.0
ACCELERATED_DROP_PCT_PER_MINUTE = 2.0
AUTOMATIC_STOP_LOSS_PCT = 10.0
ABSOLUTE_DRAWDOWN_CEILING_PCT = 15.0
RECOVERY_HORIZON_BARS = int(BOT_HORIZON_HOURS * 60)

_LOCKED_POLICY_FIELDS = frozenset(
    {
        "capital_preservation_enabled",
        "manual_cancel_loss_pct",
        "accelerated_drop_pct_per_bar",
        "automatic_stop_loss_pct",
        "absolute_drawdown_ceiling_pct",
        "recovery_horizon_bars",
        "cb_enabled",
    }
)


def build_capital_preservation_config(
    symbol: str,
    lower: float,
    upper: float,
    num_grids: int,
    **overrides: Any,
) -> GridConfig:
    """Build the fixed capital-preservation policy on canonical base physics."""
    conflicts = sorted(_LOCKED_POLICY_FIELDS.intersection(overrides))
    if conflicts:
        raise ValueError(
            "capital-preservation policy fields are fixed; remove overrides: "
            + ", ".join(conflicts)
        )
    return build_training_config(
        symbol,
        lower,
        upper,
        num_grids,
        **overrides,
        cb_enabled=False,
        capital_preservation_enabled=True,
        manual_cancel_loss_pct=MANUAL_CANCEL_LOSS_PCT,
        accelerated_drop_pct_per_bar=ACCELERATED_DROP_PCT_PER_MINUTE,
        automatic_stop_loss_pct=AUTOMATIC_STOP_LOSS_PCT,
        absolute_drawdown_ceiling_pct=ABSOLUTE_DRAWDOWN_CEILING_PCT,
        recovery_horizon_bars=RECOVERY_HORIZON_BARS,
    )


def run_capital_preservation_backtest(
    config: GridConfig,
    klines_df: pd.DataFrame,
    *,
    seed_state: Optional["SeedState"] = None,
) -> dict[str, Any]:
    """Execute the specialized policy through the canonical runner."""
    if not config.capital_preservation_enabled:
        raise ValueError("capital-preservation policy is not enabled")
    result = run_backtest(
        config,
        klines_df,
        seed_state=seed_state,
        realism_profile=LEGACY_REALISM_PROFILE,
    )
    result["capital_preservation_profile_version"] = (
        CAPITAL_PRESERVATION_PROFILE_VERSION
    )
    result["capital_preservation_scope"] = "per_bot"
    return result
