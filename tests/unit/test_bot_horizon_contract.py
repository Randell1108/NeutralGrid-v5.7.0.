"""
Horizon contract test — DURATION_FIX.md Step 3 / §13.

Single narrow gate: every bot-lifetime field resolves to H* (6h) via the
canonical constants module. Updated values here must be mirrored by
`BOT_HORIZON_*` in `neutralgrid.core.constants`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from neutralgrid.core.constants import (
    BOT_HORIZON_BARS_15M,
    BOT_HORIZON_HOURS,
    BOT_HORIZON_SECONDS,
)


def test_grid_max_holding_seconds_equals_bot_horizon():
    from neutralgrid.core.config import GridConfig

    assert GridConfig().max_holding_seconds == BOT_HORIZON_SECONDS


def test_barrier_time_hours_equals_bot_horizon():
    from neutralgrid.core.config import BarrierConfig

    assert BarrierConfig().time_hours == BOT_HORIZON_HOURS


def test_meta_labeler_horizon_equals_bot_horizon():
    from neutralgrid.models.meta_labeler import MetaLabelerConfig

    assert MetaLabelerConfig().horizon_hours == BOT_HORIZON_HOURS


def test_triple_barrier_time_hours_equals_bot_horizon():
    from neutralgrid.models.triple_barrier import TripleBarrierConfig

    assert TripleBarrierConfig().time_barrier_hours == BOT_HORIZON_HOURS


def test_backtest_max_holding_bars_equals_h_star_minutes():
    from backtest.backtest_realistic import GridConfig as BTGridConfig

    cfg = BTGridConfig(symbol="TESTUSDT", lower=100.0, upper=110.0, num_grids=5)
    assert cfg.max_holding_bars == int(BOT_HORIZON_HOURS * 60)


def test_survival_horizon_equals_bot_horizon_bars():
    from neutralgrid.core.config import StochasticConfig as CoreStoch
    from neutralgrid.validation.stochastic import StochasticConfig as ValStoch

    assert CoreStoch().survival_horizon_bars == BOT_HORIZON_BARS_15M
    assert ValStoch().survival_horizon == BOT_HORIZON_BARS_15M


def test_btk_label_contract_horizon_equals_bot_horizon():
    from backtest.btk_label_contract import STANDARD_HORIZON_HOURS, TRAINING_ENGINE_DEFAULTS

    assert STANDARD_HORIZON_HOURS == BOT_HORIZON_HOURS
    assert TRAINING_ENGINE_DEFAULTS["max_holding_bars"] == int(BOT_HORIZON_HOURS * 60)
