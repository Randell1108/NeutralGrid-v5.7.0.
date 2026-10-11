from __future__ import annotations

import pytest
import pandas as pd

from backtest.btk_capital_preservation_runner import (
    ABSOLUTE_DRAWDOWN_CEILING_PCT,
    ACCELERATED_DROP_PCT_PER_MINUTE,
    AUTOMATIC_STOP_LOSS_PCT,
    MANUAL_CANCEL_LOSS_PCT,
    RECOVERY_HORIZON_BARS,
    build_capital_preservation_config,
    run_capital_preservation_backtest,
)
from neutralgrid.grid.capital_preservation import (
    CapitalPreservationConfig,
    CapitalPreservationPolicy,
)


@pytest.fixture
def policy() -> CapitalPreservationPolicy:
    return CapitalPreservationPolicy(
        CapitalPreservationConfig(
            enabled=True,
            manual_cancel_loss_pct=5.0,
            accelerated_drop_pct_per_bar=2.0,
            automatic_stop_loss_pct=10.0,
            absolute_drawdown_ceiling_pct=15.0,
            recovery_horizon_bars=360,
        ),
        initial_capital=400.0,
    )


def test_accelerated_loss_cancels_at_five_percent(policy: CapitalPreservationPolicy) -> None:
    assert policy.check(current_equity=392.0, bar_idx=0) == (False, "")

    should_exit, reason = policy.check(current_equity=379.0, bar_idx=1)

    assert should_exit is True
    assert reason.startswith("manual_accelerated_loss_")
    assert policy.one_bar_drop_pct == pytest.approx(3.25)


def test_slow_five_percent_loss_gets_recovery_window(policy: CapitalPreservationPolicy) -> None:
    for bar_idx, equity in enumerate((396.0, 392.0, 388.0, 384.0, 380.0)):
        assert policy.check(current_equity=equity, bar_idx=bar_idx) == (False, "")

    assert policy.recovery_armed_bar == 4
    assert policy.check(current_equity=380.0, bar_idx=362) == (False, "")

    should_exit, reason = policy.check(current_equity=380.0, bar_idx=363)

    assert should_exit is True
    assert reason.startswith("manual_no_recovery_")


def test_recovery_above_five_percent_resets_window(policy: CapitalPreservationPolicy) -> None:
    for bar_idx, equity in enumerate((396.0, 392.0, 388.0, 384.0, 380.0)):
        assert policy.check(current_equity=equity, bar_idx=bar_idx) == (False, "")
    assert policy.recovery_armed_bar == 4

    assert policy.check(current_equity=381.0, bar_idx=100) == (False, "")
    assert policy.recovery_armed_bar == -1

    assert policy.check(current_equity=380.0, bar_idx=359) == (False, "")
    assert policy.check(current_equity=381.0, bar_idx=718) == (False, "")


def test_automatic_stop_loss_triggers_at_ten_percent(policy: CapitalPreservationPolicy) -> None:
    for bar_idx, equity in enumerate((396.0, 392.0, 388.0, 384.0, 380.0)):
        assert policy.check(current_equity=equity, bar_idx=bar_idx) == (False, "")

    should_exit, reason = policy.check(current_equity=360.0, bar_idx=5)

    assert should_exit is True
    assert reason.startswith("automatic_stop_loss_")


def test_absolute_ceiling_has_highest_priority(policy: CapitalPreservationPolicy) -> None:
    should_exit, reason = policy.check(current_equity=339.0, bar_idx=0)

    assert should_exit is True
    assert reason.startswith("absolute_drawdown_ceiling_")


def test_disabled_policy_is_inert() -> None:
    policy = CapitalPreservationPolicy(
        CapitalPreservationConfig(enabled=False),
        initial_capital=400.0,
    )

    assert policy.check(current_equity=100.0, bar_idx=0) == (False, "")


def test_specialized_config_pins_operator_policy() -> None:
    config = build_capital_preservation_config(
        "BTCUSDT",
        90.0,
        110.0,
        20,
        capital=400.0,
    )

    assert config.capital_preservation_enabled is True
    assert config.cb_enabled is False
    assert config.manual_cancel_loss_pct == MANUAL_CANCEL_LOSS_PCT
    assert config.accelerated_drop_pct_per_bar == ACCELERATED_DROP_PCT_PER_MINUTE
    assert config.automatic_stop_loss_pct == AUTOMATIC_STOP_LOSS_PCT
    assert config.absolute_drawdown_ceiling_pct == ABSOLUTE_DRAWDOWN_CEILING_PCT
    assert config.recovery_horizon_bars == RECOVERY_HORIZON_BARS == 360
    assert getattr(config, "_physics_overridden") is True


def test_specialized_config_rejects_policy_override() -> None:
    with pytest.raises(ValueError, match="policy fields are fixed"):
        build_capital_preservation_config(
            "BTCUSDT",
            90.0,
            110.0,
            20,
            automatic_stop_loss_pct=11.0,
        )


def test_specialized_engine_enforces_absolute_equity_floor() -> None:
    config = build_capital_preservation_config(
        "TESTUSDT",
        90.0,
        110.0,
        20,
        capital=400.0,
        leverage=10,
        maker_fee=0.0,
        taker_fee=0.0,
        funding_rate=0.0,
        order_delay_bars=0,
        max_holding_bars=0,
        fill_mode="close",
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="1min"),
            "open": [100.0, 110.0, 200.0],
            "high": [100.0, 110.0, 200.0],
            "low": [100.0, 110.0, 200.0],
            "close": [100.0, 110.0, 200.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )

    result = run_capital_preservation_backtest(config, frame)

    assert result["cb_triggered"] is True
    assert result["capital_preservation_floor_applied"] is True
    assert result["capital_preservation_floor_adjustment_usdt"] > 0
    assert result["final_equity"] == pytest.approx(340.0)
    assert result["net_pnl_pct"] == pytest.approx(-15.0)
    assert result["total_profit_identity_residual_usdt"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("manual_cancel_loss_pct", 0.0),
        ("accelerated_drop_pct_per_bar", 0.0),
        ("automatic_stop_loss_pct", 5.0),
        ("absolute_drawdown_ceiling_pct", 10.0),
        ("recovery_horizon_bars", 0),
    ],
)
def test_invalid_policy_ordering_fails_closed(field: str, value: float) -> None:
    values: dict[str, float | int | bool] = {
        "enabled": True,
        "manual_cancel_loss_pct": 5.0,
        "accelerated_drop_pct_per_bar": 2.0,
        "automatic_stop_loss_pct": 10.0,
        "absolute_drawdown_ceiling_pct": 15.0,
        "recovery_horizon_bars": 360,
    }
    values[field] = value

    with pytest.raises(ValueError):
        CapitalPreservationPolicy(CapitalPreservationConfig(**values), 400.0)
