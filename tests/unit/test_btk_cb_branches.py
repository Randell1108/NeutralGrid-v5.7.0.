"""
Step 18 (Plan v6.0): Circuit breaker branch tests.

Tests both firing branches of the equity circuit breaker for neutral
grid bots with bidirectional (long + short) positions:

Branch A (imbalance-first):
    total_positions >= 3 AND dd >= cb_inventory_imbalance_dd_pct AND
    imbalance >= cb_inventory_imbalance_ratio → fires at DD 3-5%

Branch B (max-dd-first):
    total_positions < 3 OR DD < imbalance_dd_pct
    → only max_dd_pct at 5% can fire

Note: Tests below use short_count=0 to validate the CB component in
isolation. In the live backtester, actual long/short counts are computed
from the position inventory.
"""
from __future__ import annotations

import pytest

from neutralgrid.grid.equity_circuit_breaker import (
    CircuitBreakerConfig,
    EquityCircuitBreaker,
)


@pytest.fixture
def training_cb_config() -> CircuitBreakerConfig:
    """CB config matching TRAINING_ENGINE_DEFAULTS from btk_label_contract.py."""
    return CircuitBreakerConfig(
        enabled=True,
        max_dd_pct=5.0,
        trailing_activate_pct=999.0,     # DISABLED
        trailing_offset_pct=999.0,       # DISABLED
        inventory_imbalance_ratio=0.85,
        inventory_imbalance_dd_pct=3.0,
    )


class TestBranchA_ImbalanceFirst:
    """Branch A: positions >= 3, DD >= 3% → imbalance fires before max_dd."""

    def test_imbalance_fires_at_3pct_dd_with_3_positions(
        self, training_cb_config: CircuitBreakerConfig
    ) -> None:
        """With 3+ positions and DD >= 3%, imbalance branch fires."""
        cb = EquityCircuitBreaker(training_cb_config, initial_capital=400.0)
        # Simulate 3.5% drawdown: equity = 400 * (1 - 0.035) = 386
        should_exit, reason = cb.check(
            current_equity=386.0,
            long_count=3,
            short_count=0,  # Grid bot: always 0
            bar_idx=100,
        )
        assert should_exit is True
        assert "inventory_imbalance" in reason
        assert cb.triggered is True

    def test_imbalance_fires_at_exactly_3pct_dd(
        self, training_cb_config: CircuitBreakerConfig
    ) -> None:
        """DD exactly at imbalance threshold (3%) should fire."""
        cb = EquityCircuitBreaker(training_cb_config, initial_capital=400.0)
        # 3% DD: equity = 400 * (1 - 0.03) = 388
        should_exit, reason = cb.check(
            current_equity=388.0,
            long_count=5,
            short_count=0,
            bar_idx=50,
        )
        assert should_exit is True
        assert "inventory_imbalance" in reason

    def test_imbalance_short_count_zero_means_imbalance_always_1(
        self, training_cb_config: CircuitBreakerConfig
    ) -> None:
        """With short_count=0, imbalance = long/(long+0) = 1.0 > 0.85."""
        cb = EquityCircuitBreaker(training_cb_config, initial_capital=400.0)
        # 4% DD with 10 positions — imbalance = 10/10 = 1.0
        should_exit, reason = cb.check(
            current_equity=384.0,
            long_count=10,
            short_count=0,
            bar_idx=200,
        )
        assert should_exit is True
        assert "inventory_imbalance" in reason

    def test_imbalance_does_not_fire_with_2_positions(
        self, training_cb_config: CircuitBreakerConfig
    ) -> None:
        """With < 3 positions, imbalance branch is skipped even at 4% DD."""
        cb = EquityCircuitBreaker(training_cb_config, initial_capital=400.0)
        # 4% DD but only 2 positions — below position count gate
        should_exit, reason = cb.check(
            current_equity=384.0,
            long_count=2,
            short_count=0,
            bar_idx=100,
        )
        # DD is 4%, below max_dd (5%), and positions < 3, so nothing fires
        assert should_exit is False

    def test_imbalance_does_not_fire_at_2pct_dd(
        self, training_cb_config: CircuitBreakerConfig
    ) -> None:
        """With 3+ positions but DD < 3%, imbalance branch doesn't fire."""
        cb = EquityCircuitBreaker(training_cb_config, initial_capital=400.0)
        # 2% DD with 5 positions — below DD threshold
        should_exit, reason = cb.check(
            current_equity=392.0,
            long_count=5,
            short_count=0,
            bar_idx=100,
        )
        assert should_exit is False


class TestBranchB_MaxDDFirst:
    """Branch B: positions < 3 OR DD < imbalance_dd_pct → only max_dd fires."""

    def test_max_dd_fires_at_5pct_with_0_positions(
        self, training_cb_config: CircuitBreakerConfig
    ) -> None:
        """Max DD fires regardless of position count."""
        cb = EquityCircuitBreaker(training_cb_config, initial_capital=400.0)
        # 5.5% DD: equity = 400 * (1 - 0.055) = 378
        should_exit, reason = cb.check(
            current_equity=378.0,
            long_count=1,
            short_count=0,
            bar_idx=300,
        )
        assert should_exit is True
        assert "max_dd" in reason

    def test_max_dd_fires_at_exactly_5pct(
        self, training_cb_config: CircuitBreakerConfig
    ) -> None:
        """DD exactly at max_dd_pct (5%) should fire."""
        cb = EquityCircuitBreaker(training_cb_config, initial_capital=400.0)
        # 5% DD: equity = 400 * (1 - 0.05) = 380
        should_exit, reason = cb.check(
            current_equity=380.0,
            long_count=2,
            short_count=0,
            bar_idx=200,
        )
        assert should_exit is True
        assert "max_dd" in reason

    def test_max_dd_does_not_fire_at_4_5pct(
        self, training_cb_config: CircuitBreakerConfig
    ) -> None:
        """DD at 4.5% (below 5%) with < 3 positions should not fire."""
        cb = EquityCircuitBreaker(training_cb_config, initial_capital=400.0)
        # 4.5% DD: equity = 400 * (1 - 0.045) = 382
        should_exit, reason = cb.check(
            current_equity=382.0,
            long_count=2,
            short_count=0,
            bar_idx=150,
        )
        assert should_exit is False


class TestBranchInteraction:
    """Tests that verify the interaction between both branches."""

    def test_imbalance_fires_before_max_dd_with_many_positions(
        self, training_cb_config: CircuitBreakerConfig
    ) -> None:
        """With 3+ positions, imbalance fires at DD 3-5% before max_dd at 5%."""
        cb = EquityCircuitBreaker(training_cb_config, initial_capital=400.0)
        # 3.5% DD, 5 positions → imbalance should fire (not max_dd)
        should_exit, reason = cb.check(
            current_equity=386.0,
            long_count=5,
            short_count=0,
            bar_idx=100,
        )
        assert should_exit is True
        assert "inventory_imbalance" in reason
        # Confirm it's NOT max_dd
        assert "max_dd" not in reason

    def test_max_dd_fires_for_lone_position(
        self, training_cb_config: CircuitBreakerConfig
    ) -> None:
        """With 1 position, only max_dd can fire (imbalance needs >= 3)."""
        cb = EquityCircuitBreaker(training_cb_config, initial_capital=400.0)
        # 6% DD, 1 position → max_dd fires
        should_exit, reason = cb.check(
            current_equity=376.0,
            long_count=1,
            short_count=0,
            bar_idx=300,
        )
        assert should_exit is True
        assert "max_dd" in reason

    def test_trailing_disabled_at_999(
        self, training_cb_config: CircuitBreakerConfig
    ) -> None:
        """With trailing_activate_pct=999, trailing lock never fires."""
        cb = EquityCircuitBreaker(training_cb_config, initial_capital=400.0)
        # Simulate profit peak then reversal
        cb.check(current_equity=450.0, long_count=3, short_count=0, bar_idx=10)
        should_exit, reason = cb.check(
            current_equity=410.0,  # 10% drop from peak
            long_count=3,
            short_count=0,
            bar_idx=100,
        )
        # DD from initial is only 2.5%, positions < 3 imbalance gate
        # But here we have 3 positions so imbalance checks...
        # DD from initial = (400-410)/400 = -2.5%, but equity > initial
        # so dd_from_initial_pct = 0 (no loss from initial)
        # → no fire
        assert should_exit is False

    def test_no_fire_when_disabled(self) -> None:
        """CB disabled entirely should never fire."""
        cfg = CircuitBreakerConfig(enabled=False)
        cb = EquityCircuitBreaker(cfg, initial_capital=400.0)
        # Even at 50% DD
        should_exit, reason = cb.check(
            current_equity=200.0,
            long_count=10,
            short_count=0,
            bar_idx=500,
        )
        assert should_exit is False
        assert reason == ""

    def test_cb_only_fires_once(
        self, training_cb_config: CircuitBreakerConfig
    ) -> None:
        """Once triggered, subsequent check() returns (False, '') to prevent
        double-exit.  The backtester won't re-enter the CB path because
        positions are empty after force-close."""
        cb = EquityCircuitBreaker(training_cb_config, initial_capital=400.0)
        first_exit, _ = cb.check(
            current_equity=375.0,
            long_count=5,
            short_count=0,
            bar_idx=100,
        )
        assert first_exit is True
        assert cb.triggered is True
        # Second call — already triggered → returns (False, "") early
        second_exit, second_reason = cb.check(
            current_equity=390.0,
            long_count=5,
            short_count=0,
            bar_idx=200,
        )
        assert second_exit is False
        assert second_reason == ""
