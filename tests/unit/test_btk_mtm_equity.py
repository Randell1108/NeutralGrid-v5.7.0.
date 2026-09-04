"""Unit tests for Phase 1: MTM equity curve and drawdown accounting.

Validates that the RealisticGridBacktester computes equity curves using
mark-to-market (unrealized + realized) PnL, and that drawdown is derived
from per-bar MTM equity rather than realized-only snapshots.
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# backtest/ lives at the repo root, not inside src/neutralgrid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtest.backtest_realistic import GridConfig, RealisticGridBacktester


def _make_klines(
    closes: list[float],
    start: datetime | None = None,
) -> pd.DataFrame:
    """Build a minimal 1-min klines DataFrame from a list of close prices."""
    if start is None:
        start = datetime(2026, 1, 1)
    n = len(closes)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * n,
        }
    )


@pytest.fixture()
def grid_config() -> GridConfig:
    """Standard grid config for tests: lower=100, upper=110, 5 grids, capital=1000."""
    return GridConfig(
        symbol="TESTUSDT",
        lower=100.0,
        upper=110.0,
        num_grids=5,
        capital=1000.0,
        leverage=10,
        maker_fee=0.0,
        taker_fee=0.0,
        order_delay_bars=0,
        funding_rate=0.0,
        slippage_bps=0.0,
        max_holding_bars=0,  # disabled
        funding_mode="snapshot",
    )


# ── Test (a): equity_curve values reflect unrealized PnL ──────────────

def test_equity_curve_includes_unrealized(grid_config: GridConfig):
    """When price rises through levels, short positions open (bidirectional).
    The equity curve should reflect the unrealized loss — not stay flat."""
    # Grid levels for lower=100, upper=110, num_grids=5:
    #   100, 102, 104, 106, 108, 110
    # Spacing = 2.0

    # Price starts below all levels, then rises through 102 (triggers shorts),
    # then stays at 103 for a few bars (unrealized loss on shorts).
    closes = [99.0, 102.5, 103.0, 103.0, 103.0]
    df = _make_klines(closes)
    bt = RealisticGridBacktester(grid_config)
    result = bt.run(df)

    # After bar 1 (close=102.5): crossed UP through 100 and 102.
    # Shorts opened at 100 and 102 (bidirectional neutral grid).
    # At bars 2-4 (close=103), shorts are underwater → unrealized loss.
    curve = bt.equity_curve

    # The curve should NOT be flat after positions are opened.
    # With MTM, it should reflect unrealized PnL (negative for shorts above entry).
    assert len(curve) == len(closes)

    # Last value should differ from capital because of unrealized PnL on shorts
    assert curve[-1] != grid_config.capital, (
        f"Expected equity != {grid_config.capital} due to unrealized PnL, got {curve[-1]}"
    )


# ── Test (b): max_drawdown_pct > 0 when there is an unrealized dip ───

def test_drawdown_reflects_unrealized_dip(grid_config: GridConfig):
    """Price oscillates opening both long and short positions.
    max_drawdown_pct should be positive to reflect MTM drawdown."""
    # Bidirectional grid: UP crossings open shorts, DOWN crossings open longs.
    # The oscillation creates both realized round trips and open positions
    # whose unrealized PnL generates drawdown.
    closes = (
        [99.0]
        + [102.5]  # cross up through 100, 102 -> short at 100, 102
        + [104.5]  # cross up through 104 -> close short@102 at 104, short at 104
        + [106.5]  # cross up through 106 -> close short@104 at 106, short at 106
        + [103.0]  # drop: cross down through 106, 104 -> close short@106, long at 106/104
        + [101.0]  # drop further: cross down through 102 -> long at 102
        + [101.0]  # stay low
    )
    df = _make_klines(closes)
    bt = RealisticGridBacktester(grid_config)
    result = bt.run(df)

    # There should be a measurable drawdown from the unrealized dip
    assert result["max_drawdown_pct"] > 0, (
        f"Expected positive drawdown from unrealized dip, got {result['max_drawdown_pct']}"
    )


# ── Test (c): unrealized_pnl_at_end is correct for open positions ─────

def test_unrealized_pnl_at_end(grid_config: GridConfig):
    """Verify unrealized_pnl_at_end matches manual computation for open positions."""
    closes = [99.0, 102.5, 105.0, 105.0]
    df = _make_klines(closes)
    bt = RealisticGridBacktester(grid_config)
    result = bt.run(df)

    # Compute expected unrealized manually from remaining open positions (side-aware)
    final_price = closes[-1]
    expected_unrealized = 0.0
    for pos in bt.positions.values():
        if pos["side"] == "long":
            expected_unrealized += (final_price - pos["entry_price"]) * pos["qty"]
        else:  # short
            expected_unrealized += (pos["entry_price"] - final_price) * pos["qty"]

    assert abs(result["unrealized_pnl_at_end"] - expected_unrealized) < 1e-6, (
        f"unrealized_pnl_at_end={result['unrealized_pnl_at_end']} != expected {expected_unrealized}"
    )

    # Sanity: there should be at least one open position
    assert len(bt.positions) > 0, "Expected at least one open position at end"


# ── Test (d): final_equity = capital + net_pnl ────────────────────────

def test_final_equity_equals_capital_plus_net_pnl(grid_config: GridConfig):
    """final_equity should equal capital + net_pnl (both MTM-based)."""
    closes = [99.0, 102.5, 105.0, 103.0, 107.0, 105.0]
    df = _make_klines(closes)
    bt = RealisticGridBacktester(grid_config)
    result = bt.run(df)

    assert abs(result["final_equity"] - (grid_config.capital + result["net_pnl"])) < 1e-6, (
        f"final_equity={result['final_equity']} != capital({grid_config.capital}) + net_pnl({result['net_pnl']})"
    )


# ── Test (e): label_positive_by_horizon is consistent with final equity ──

def test_label_positive_by_horizon_consistency(grid_config: GridConfig):
    """label_positive_by_horizon should be True iff final_equity > capital."""
    # Scenario 1: price rises — should be positive
    closes_up = [99.0, 102.5, 105.0, 107.0, 109.0]
    df_up = _make_klines(closes_up)
    bt_up = RealisticGridBacktester(grid_config)
    res_up = bt_up.run(df_up)

    # Use == (not `is`) because numpy comparisons return np.bool_ not Python bool
    assert res_up["label_positive_by_horizon"] == (res_up["final_equity"] > grid_config.capital)

    # Scenario 2: price drops — should be negative
    closes_down = [109.0, 106.0, 103.0, 100.0, 98.0]
    df_down = _make_klines(closes_down)
    bt_down = RealisticGridBacktester(grid_config)
    res_down = bt_down.run(df_down)

    assert res_down["label_positive_by_horizon"] == (res_down["final_equity"] > grid_config.capital)


# ── Test: _unrealized_pnl method directly ─────────────────────────────

def test_unrealized_pnl_method(grid_config: GridConfig):
    """Directly test _unrealized_pnl computes correctly for both sides."""
    bt = RealisticGridBacktester(grid_config)

    # No positions => 0
    assert bt._unrealized_pnl(105.0) == 0.0

    # Manually inject long positions
    bt.positions[100.0] = {"entry_price": 100.0, "qty": 10.0, "side": "long", "entry_bar": 0}
    bt.positions[102.0] = {"entry_price": 102.0, "qty": 10.0, "side": "long", "entry_bar": 0}

    # Mark at 105: unrealized = (105-100)*10 + (105-102)*10 = 50 + 30 = 80
    assert abs(bt._unrealized_pnl(105.0) - 80.0) < 1e-9

    # Mark at 99: unrealized = (99-100)*10 + (99-102)*10 = -10 + -30 = -40
    assert abs(bt._unrealized_pnl(99.0) - (-40.0)) < 1e-9

    # Add a short position and verify bidirectional calculation
    bt.positions.clear()
    bt.positions[100.0] = {"entry_price": 100.0, "qty": 10.0, "side": "long", "entry_bar": 0}
    bt.positions[108.0] = {"entry_price": 108.0, "qty": 10.0, "side": "short", "entry_bar": 0}

    # Mark at 105: long = (105-100)*10 = 50, short = (108-105)*10 = 30 → total 80
    assert abs(bt._unrealized_pnl(105.0) - 80.0) < 1e-9

    # Mark at 110: long = (110-100)*10 = 100, short = (108-110)*10 = -20 → total 80
    assert abs(bt._unrealized_pnl(110.0) - 80.0) < 1e-9


# ── Test: DD is computed per-bar from MTM, not per-event ──────────────

def test_dd_computed_per_bar_not_per_event(grid_config: GridConfig):
    """Equity curve length must equal the number of bars (one entry per bar),
    confirming DD is computed once per bar, not per trade event."""
    closes = [99.0, 102.5, 104.5, 106.5, 103.0, 101.0, 105.0]
    df = _make_klines(closes)
    bt = RealisticGridBacktester(grid_config)
    bt.run(df)

    assert len(bt.equity_curve) == len(closes), (
        f"equity_curve length {len(bt.equity_curve)} != bar count {len(closes)}"
    )
