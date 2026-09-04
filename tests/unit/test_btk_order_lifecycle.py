"""
Unit tests for R5: order lifecycle — grid crossing loop refactor.

Validates that:
 - CLOSE operations are never gated by cooldown
 - OPEN operations respect ``_level_available_bar`` cooldown
 - Cooldown is set consistently after both UP and DOWN close events
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtest.backtest_realistic import GridConfig, RealisticGridBacktester
from neutralgrid.grid.formulas import LEGACY_LINE_COUNT


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_klines(
    closes: list[float],
    start: datetime | None = None,
) -> pd.DataFrame:
    """Build a minimal 1-min klines DataFrame from a list of close prices."""
    if start is None:
        start = datetime(2026, 1, 1)
    n = len(closes)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1.0] * n,
    })


def _config(delay: int = 3) -> GridConfig:
    """Grid config with levels at [100, 102, 104, 106, 108, 110].

    GRIDFIX-001 / GRID_SYNCH §3.1: num_grids is grid LINES (Binance
    convention). 6 lines over [100, 110] yields step=2.0 and the same
    integer-spaced level set the test scenarios were written against.
    """
    return GridConfig(
        symbol="TESTUSDT",
        lower=100.0,
        upper=110.0,
        num_grids=6,
        capital=1000.0,
        leverage=10,
        maker_fee=0.0,
        taker_fee=0.0,
        order_delay_bars=delay,
        funding_rate=0.0,
        slippage_bps=0.0,
        max_holding_bars=0,  # disabled
        funding_mode="snapshot",
        # Default grid identity is now geometric (commit 04a15b4). These
        # lifecycle scenarios are hand-computed against arithmetic integer
        # levels [100,102,...,110], so pin arithmetic mode explicitly.
        mode="arithmetic",
        grid_count_semantics=LEGACY_LINE_COUNT,
    )


# ── Test 1 & 2: Close never blocked by cooldown ──────────────────────────────

class TestCloseNotBlockedByCooldown:
    """R5 bug #1 & #2: CLOSE operations must never be gated by cooldown."""

    def test_up_crossing_close_not_blocked(self):
        """UP crossing: close of long from below_level proceeds despite cooldown.
        Bidirectional: DOWN opens longs, UP closes them."""
        # Levels: [100, 102, 104, ...]
        # Bar 0: 103.0
        # Bar 1: 99.5 -> DOWN through 102, 100
        #   Long at 102 (cooldown 102=4), long at 100 (cooldown 100=4)
        # Bar 2: 99.5 -> no crossing
        # Bar 3: 102.5 -> UP through 100, 102
        #   At level 102: close long from 100 (below_level), cooldown 100=4, bar=3<4.
        #   Close proceeds (not gated by cooldown).
        cfg = _config(delay=3)
        df = _make_klines([103.0, 99.5, 99.5, 102.5])
        bt = RealisticGridBacktester(cfg)
        bt.run(df)

        # Long at 100 should have been closed by UP crossing through 102
        assert 100.0 not in bt.positions, (
            "Long at 100 should be closed -- UP close must not be blocked "
            "by cooldown"
        )
        # A sell trade at price ~102 should exist (closing the long)
        sell_trades = [
            t for t in bt.trades
            if t.side == "sell" and abs(t.price - 102.0) < 0.1
        ]
        assert len(sell_trades) >= 1

    def test_down_crossing_close_not_blocked(self):
        """DOWN crossing: close of short from above_level proceeds despite cooldown.
        Bidirectional: UP opens shorts, DOWN closes them."""
        # Bar 0: 101.0
        # Bar 1: 104.5 -> UP through 102, 104
        #   Short at 102 (cooldown 102=4), short at 104 (cooldown 104=4)
        # Bar 2: 104.5 -> no crossing
        # Bar 3: 101.5 -> DOWN through 104, 102
        #   At level 102: close short from 104 (above_level), cooldown 104=4, bar=3<4.
        #   Close proceeds (not gated by cooldown).
        cfg = _config(delay=3)
        df = _make_klines([101.0, 104.5, 104.5, 101.5])
        bt = RealisticGridBacktester(cfg)
        bt.run(df)

        # Short at 104 should be closed by DOWN crossing through 102
        assert 104.0 not in bt.positions, (
            "Short at 104 should be closed -- DOWN close must not be "
            "blocked by cooldown"
        )


# ── Test 3: Buy blocked by cooldown ──────────────────────────────────────────

class TestBuyBlockedByCooldown:
    """OPEN operations must be gated by _level_available_bar."""

    def test_open_blocked_by_cooldown(self):
        """Open at a level with active cooldown should not proceed.
        Bidirectional: DOWN opens longs. Verify cooldown gates reopening."""
        # Levels: [100, 102, 104, ...]
        # Bar 0: 105.0
        # Bar 1: 101.5 -> DOWN through 104, 102: long@104, long@102. Cooldown 102=4.
        # Bar 2: 104.5 -> UP through 102, 104:
        #   At 104: close long@102 (below). Cooldown 102=5.
        #   Short at 104: blocked (long there).
        # Bar 3: 101.5 -> DOWN through 104, 102:
        #   At 102: open long. bar=3 < cooldown 5 -> BLOCKED.
        cfg = _config(delay=3)
        df = _make_klines([105.0, 101.5, 104.5, 101.5])
        bt = RealisticGridBacktester(cfg)
        bt.run(df)

        # Long at 102 should NOT exist (open was blocked by cooldown)
        assert 102.0 not in bt.positions, (
            "Open at 102 should be blocked by cooldown"
        )


# ── Test 4 & 5: Sell and buy independent ──────────────────────────────────────

class TestCloseAndOpenIndependent:
    """Close and open are independent operations within the same crossing."""

    def test_up_crossing_close_proceeds_open_blocked(self):
        """UP: close of long from below proceeds; short open at level blocked
        by its own cooldown."""
        # Levels: [100, 102, 104, ...]
        # Bar 0: 105.0
        # Bar 1: 101.5 -> DOWN through 104, 102: long@104 (cd=4), long@102 (cd=4)
        # Bar 2: 104.5 -> UP through 102, 104:
        #   At 104: close long@102 (below). Cooldown 102=5.
        #   Short at 104: blocked (long there).
        # Bar 3: 101.5 -> DOWN through 104, 102:
        #   Long at 104: still there. At 102: cd=5, bar=3<5 → BLOCKED.
        # Bar 4: 104.5 -> UP through 102, 104:
        #   At 104: close long@102? No pos at 102 (blocked). Nothing to close.
        #   Short at 104: blocked (long still there).
        cfg = _config(delay=3)
        df = _make_klines([105.0, 101.5, 104.5, 101.5, 104.5])
        bt = RealisticGridBacktester(cfg)
        bt.run(df)

        # Long@102 was closed at bar 2, new long blocked by cooldown at bar 3
        assert 102.0 not in bt.positions

    def test_down_crossing_close_proceeds_open_blocked(self):
        """DOWN: close of short from above proceeds; long open at level blocked
        by its own cooldown."""
        # Levels: [100, 102, 104, 106, 108, 110]
        # Bar 0: 101.0
        # Bar 1: 104.5 -> UP through 102, 104: short@102 (cd=6), short@104 (cd=6)
        # Bar 2: 101.5 -> DOWN through 104, 102:
        #   At 102: close short@104 (above). Cooldown 104=7.
        #   Long at 102: cd=6, bar=2<6 → BLOCKED.
        cfg = _config(delay=5)
        df = _make_klines([101.0, 104.5, 101.5])
        bt = RealisticGridBacktester(cfg)
        bt.run(df)

        # Short@104 was closed at bar 2 (close not blocked by cooldown)
        assert 104.0 not in bt.positions
        # Short@102 from bar 1 persists (not closed by this crossing)
        # Long open at 102 was blocked by cooldown → short remains
        assert bt.positions[102.0]["side"] == "short"


# ── Test 6 & 7: Replacement order delay set after close ──────────────────────

class TestReplacementOrderDelay:
    """R5 bug #3: cooldown set after close events on the replacement level."""

    def test_cooldown_set_after_up_close(self):
        """After UP close of long, _level_available_bar[below_level] is set."""
        # Bar 0: 103.0
        # Bar 1: 99.5 -> DOWN through 102, 100: long@102 (cd=4), long@100 (cd=4)
        # Bar 2-3: 99.5 -> no crossing
        # Bar 4: 102.5 -> UP through 100, 102:
        #   At 102: close long@100 (below). Cooldown 100 = 4+3 = 7.
        cfg = _config(delay=3)
        df = _make_klines([103.0, 99.5, 99.5, 99.5, 102.5])
        bt = RealisticGridBacktester(cfg)
        bt.run(df)

        # Cooldown on 100.0 (below_level) should be bar 4 + 3 = 7
        assert bt._level_available_bar.get(100.0) == 7, (
            f"Expected cooldown on 100.0 = 7, "
            f"got {bt._level_available_bar.get(100.0)}"
        )

    def test_cooldown_set_after_down_close(self):
        """After DOWN close of short, _level_available_bar[above_level] is set."""
        # Bar 0: 101.0
        # Bar 1: 104.5 -> UP through 102, 104: short@102 (cd=4), short@104 (cd=4)
        # Bar 2-4: 104.5 -> no crossing (wait for cooldown to expire)
        # Bar 5: 101.5 -> DOWN through 104, 102:
        #   At 102: close short@104 (above). Cooldown 104 = 5+3 = 8.
        cfg = _config(delay=3)
        df = _make_klines([101.0, 104.5, 104.5, 104.5, 104.5, 101.5])
        bt = RealisticGridBacktester(cfg)
        bt.run(df)

        # Cooldown on 104.0 (above_level) should be bar 5 + 3 = 8
        assert bt._level_available_bar.get(104.0) == 8, (
            f"Expected cooldown on 104.0 = 8, "
            f"got {bt._level_available_bar.get(104.0)}"
        )
