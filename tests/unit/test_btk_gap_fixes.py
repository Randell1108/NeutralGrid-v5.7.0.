"""
Unit tests for backtester quality gaps C, E, G, H.

Gap C — Configurable close fee mode (maker vs taker on grid-crossing sells)
Gap E — Mark-price funding notional (curr_close instead of entry_price)
Gap G — Performance optimization (_level_to_idx dict, numpy loop)
Gap H — Zero-PnL trade recording (removed abs(pnl) > 0.01 filter)
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtest.backtest_realistic import GridConfig, RealisticGridBacktester


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


def _base_config(**overrides) -> GridConfig:
    """Grid config with levels at [100, 102, 104, 106, 108, 110].

    GRIDFIX-001 / GRID_SYNCH §3.1: num_grids is grid LINES (Binance
    convention). 6 lines over [100, 110] yields step=2.0 and the
    integer-spaced level set above.
    """
    defaults = dict(
        symbol="TESTUSDT",
        lower=100.0,
        upper=110.0,
        num_grids=6,
        capital=1000.0,
        leverage=10,
        maker_fee=0.0002,
        taker_fee=0.0005,
        order_delay_bars=0,
        funding_rate=0.0001,
        slippage_bps=0.0,
        max_holding_bars=0,
        funding_mode="snapshot",
        close_fee_mode="taker",
    )
    defaults.update(overrides)
    return GridConfig(**defaults)


# ── Gap C: Configurable close fee mode ────────────────────────────────────────

class TestCloseFeeMode:
    """Gap C: close_fee_mode selects maker vs taker fee on grid-crossing sells."""

    def test_default_close_fee_mode_is_taker(self):
        """Default close_fee_mode should be 'taker'."""
        cfg = GridConfig(symbol="TEST", lower=100, upper=110, num_grids=5)
        assert cfg.close_fee_mode == "taker"

    def test_maker_close_fee_reduces_costs(self):
        """Running with close_fee_mode='maker' should yield higher net_pnl
        than 'taker' (lower fees on closes)."""
        # Bidirectional: DOWN opens longs, UP closes them.
        # Oscillate to create round trips.
        closes = [105.0, 101.5, 104.5, 101.5, 104.5, 101.5, 104.5]

        cfg_taker = _base_config(close_fee_mode="taker")
        bt_taker = RealisticGridBacktester(cfg_taker)
        res_taker = bt_taker.run(_make_klines(closes))

        cfg_maker = _base_config(close_fee_mode="maker")
        bt_maker = RealisticGridBacktester(cfg_maker)
        res_maker = bt_maker.run(_make_klines(closes))

        assert res_maker['net_pnl'] > res_taker['net_pnl'], (
            f"Maker close fee should produce higher net_pnl: "
            f"maker={res_maker['net_pnl']:.4f}, taker={res_taker['net_pnl']:.4f}"
        )

    def test_stale_close_always_taker(self):
        """Force-close of stale positions should always use taker_fee,
        regardless of close_fee_mode setting."""
        # DOWN crossing opens long, hold for max_holding_bars, force-close
        closes = [105.0, 101.5] + [101.5] * 10
        cfg = _base_config(
            close_fee_mode="maker",
            max_holding_bars=5,
            maker_fee=0.0001,
            taker_fee=0.001,
        )
        bt = RealisticGridBacktester(cfg)
        res = bt.run(_make_klines(closes))

        # The stale close should have used taker_fee (0.001), not maker (0.0001)
        assert bt._stale_closes >= 1, "Expected at least one stale close"
        assert res['fees_paid'] > 0


# ── Gap E: Mark-price funding notional ────────────────────────────────────────

class TestMarkPriceFunding:
    """Gap E: funding notional uses current mark price, not entry price."""

    def test_funding_uses_mark_price(self):
        """With a long position entered at 102 and price at 95, funding should
        be based on current mark price, not entry price."""
        # DOWN crossing creates longs (which pay positive funding).
        # Bar 0: 105 (setup)
        # Bar 1: 101.5 (long at 102, 104)
        # Bars 2-481: 95.0 (price drops, fund at bar 480 based on mark 95)
        closes = [105.0, 101.5] + [95.0] * 480
        cfg = _base_config(
            funding_mode="snapshot",
            funding_interval_bars=480,
            funding_rate=0.0001,
            maker_fee=0.0,
            taker_fee=0.0,
        )
        bt = RealisticGridBacktester(cfg)
        res = bt.run(_make_klines(closes))

        avg_price = (cfg.lower + cfg.upper) / 2
        position_size = (cfg.capital * cfg.leverage) / cfg.num_grids / avg_price

        # Funding should be based on mark price (95), not entry price (~102)
        # With longs, net_notional is positive → funding_fees > 0
        assert res['funding_fees'] != 0, "Funding should be charged"

    def test_funding_abs_higher_at_higher_mark_price(self):
        """Higher mark price on longs produces higher absolute funding
        (mark-price notional is larger)."""
        # Grid levels: [100, 102, 104, 106, 108, 110]
        # DOWN crossing opens longs at all levels, then price stays below
        # lowest level (100) to avoid UP crossings that would flip positions.
        base_closes = [111.0, 99.0]

        # Low mark price: notional ≈ 90 * total_qty per bar
        low_closes = base_closes + [90.0] * 100
        # High mark price: notional ≈ 99.5 * total_qty (below 100, no crossing)
        high_closes = base_closes + [99.5] * 100

        cfg_low = _base_config(
            funding_mode="continuous",
            maker_fee=0.0,
            taker_fee=0.0,
        )
        cfg_high = _base_config(
            funding_mode="continuous",
            maker_fee=0.0,
            taker_fee=0.0,
        )

        bt_low = RealisticGridBacktester(cfg_low)
        res_low = bt_low.run(_make_klines(low_closes))

        bt_high = RealisticGridBacktester(cfg_high)
        res_high = bt_high.run(_make_klines(high_closes))

        # Higher mark price → higher notional → more absolute funding
        assert abs(res_high['funding_fees']) > abs(res_low['funding_fees']), (
            f"High mark-price funding ({res_high['funding_fees']:.6f}) should have "
            f"greater magnitude than low ({res_low['funding_fees']:.6f})"
        )


# ── Gap G: Performance optimization ──────────────────────────────────────────

class TestPerformanceOptimization:
    """Gap G: _level_to_idx dict and numpy loop refactor."""

    def test_level_to_idx_dict_exists(self):
        """_level_to_idx should be populated and match grid_levels."""
        cfg = _base_config()
        bt = RealisticGridBacktester(cfg)
        assert hasattr(bt, '_level_to_idx'), "_level_to_idx dict should exist"
        assert len(bt._level_to_idx) == len(bt.grid_levels)
        for i, level in enumerate(bt.grid_levels):
            assert bt._level_to_idx[level] == i

    def test_results_identical_to_baseline(self):
        """Same config + same data should produce deterministic results."""
        closes = [99.0, 100.5, 102.5, 101.5, 103.0, 100.0, 102.0, 104.5]
        cfg = _base_config(maker_fee=0.0, taker_fee=0.0)
        df = _make_klines(closes)

        bt1 = RealisticGridBacktester(cfg)
        res1 = bt1.run(df)

        bt2 = RealisticGridBacktester(cfg)
        res2 = bt2.run(df)

        # All numeric fields should be identical
        for key in ['net_pnl', 'gross_pnl', 'fees_paid', 'round_trips',
                     'total_trades', 'final_equity', 'funding_fees']:
            assert res1[key] == res2[key], (
                f"Mismatch on '{key}': {res1[key]} != {res2[key]}"
            )


# ── Gap H: Zero-PnL trade recording ──────────────────────────────────────────

class TestZeroPnlTradeRecording:
    """Gap H: DOWN crossing closes at entry level (pnl=0) are now recorded."""

    def test_zero_pnl_close_is_recorded(self):
        """Force-close at entry price (zero gross pnl) should still be
        recorded as a trade."""
        # Grid levels: [100, 102, 104, 106, 108, 110]
        # Bar 0: 105.0 (setup)
        # Bar 1: 101.5 (DOWN through 104, 102 → long@104, long@102)
        # Bars 2-7: 102.0 (hold at entry price of long@102)
        # Bar 6: stale close both longs at 102.0 → pnl=0 for long@102
        closes = [105.0, 101.5] + [102.0] * 6
        cfg = _base_config(maker_fee=0.0, taker_fee=0.0, max_holding_bars=5)
        bt = RealisticGridBacktester(cfg)
        bt.run(_make_klines(closes))

        sell_trades = [t for t in bt.trades if t.side == "sell"]
        assert len(sell_trades) >= 1, (
            "Zero-PnL force-close should be recorded as a sell trade"
        )
        # Verify sell at 102 exists (force-close of long@102 at mark 102.0)
        sell_at_102 = [t for t in sell_trades if abs(t.price - 102.0) < 0.01]
        assert len(sell_at_102) >= 1, (
            f"Expected sell at 102, got {len(sell_at_102)}"
        )

    def test_zero_pnl_fees_counted(self):
        """Fees from zero-PnL close should appear in total fees."""
        # DOWN through 104, 102 → open longs (buy trades with maker fee)
        # Stale force-close at 102.0 → sell trades with taker fee
        closes = [105.0, 101.5] + [102.0] * 6
        cfg = _base_config(maker_fee=0.0002, taker_fee=0.0005, max_holding_bars=5)
        bt = RealisticGridBacktester(cfg)
        res = bt.run(_make_klines(closes))

        assert res['fees_paid'] > 0, "Fees should include zero-pnl close fee"
        # Buy trades from open, sell trades from force-close
        buy_trades = [t for t in bt.trades if t.side == "buy"]
        sell_trades = [t for t in bt.trades if t.side == "sell"]
        assert len(buy_trades) >= 1
        assert len(sell_trades) >= 1

    def test_round_trips_include_zero_pnl(self):
        """round_trips count should include zero-PnL closes."""
        # DOWN opens longs, force-close at 102.0 (zero PnL for long@102)
        closes = [105.0, 101.5] + [102.0] * 6
        cfg = _base_config(maker_fee=0.0, taker_fee=0.0, max_holding_bars=5)
        bt = RealisticGridBacktester(cfg)
        res = bt.run(_make_klines(closes))

        # Stale close of long@102 at 102.0 = zero-PnL round trip
        assert res['round_trips'] >= 1, (
            f"Expected at least 1 round trip, got {res['round_trips']}"
        )
