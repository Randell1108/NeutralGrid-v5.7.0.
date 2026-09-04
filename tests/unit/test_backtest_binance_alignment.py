from __future__ import annotations

import math
import os
from datetime import datetime, timezone

import pandas as pd
import pytest

from backtest.backtest_realistic import GridConfig, RealisticGridBacktester
from neutralgrid.grid.calculator import GridCalculator
from neutralgrid.backtest.candidate_pipeline import (
    load_all_scanner_csvs,
    resolve_backtest_start_timestamp,
)
from neutralgrid.grid.formulas import (
    BINANCE_DISPLAYED_INTERVALS,
    GEOMETRIC,
    profit_per_grid_pct,
)


def _binance_ui_trunc2(value: float) -> float:
    return math.floor((value + 1e-12) * 100.0) / 100.0


def _klines(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-26T14:11:07Z", periods=len(closes), freq="1min"),
            "open": closes,
            "high": [c * 1.002 for c in closes],
            "low": [c * 0.998 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
        }
    )


@pytest.mark.parametrize(
    (
        "symbol",
        "lower",
        "upper",
        "num_grids",
        "precise_profit",
        "display_profit",
        "rounded_levels",
    ),
    [
        (
            "IPUSDT",
            0.4740,
            0.4970,
            7,
            0.639056056101881,
            0.63,
            [0.4740, 0.4772, 0.4804, 0.4837, 0.4870, 0.4903, 0.4936, 0.4970],
        ),
        (
            "PNUTUSDT",
            0.05492,
            0.05922,
            11,
            0.6475045081948827,
            0.64,
            [0.05492, 0.05529, 0.05567, 0.05606, 0.05644, 0.05683,
             0.05722, 0.05761, 0.05801, 0.05841, 0.05881, 0.05922],
        ),
        (
            "ARKMUSDT",
            0.1368,
            0.1530,
            13,
            0.8244505851842088,
            0.82,
            [0.1368, 0.1379, 0.1391, 0.1403, 0.1415, 0.1428, 0.1440,
             0.1452, 0.1465, 0.1478, 0.1490, 0.1503, 0.1516, 0.1530],
        ),
    ],
)
def test_binance_grid_count_reconstructs_live_profit_per_grid(
    symbol: str,
    lower: float,
    upper: float,
    num_grids: int,
    precise_profit: float,
    display_profit: float,
    rounded_levels: list[float],
) -> None:
    cfg = GridConfig(symbol=symbol, lower=lower, upper=upper, num_grids=num_grids)
    bt = RealisticGridBacktester(cfg)

    assert bt.grid_count_semantics == BINANCE_DISPLAYED_INTERVALS
    assert len(bt.grid_levels) == num_grids + 1

    tick_tolerance = 1.1e-5 if symbol == "PNUTUSDT" else 1.1e-4
    assert bt.grid_levels == pytest.approx(rounded_levels, abs=tick_tolerance)

    reconstructed = profit_per_grid_pct(
        lower, upper, num_grids, GEOMETRIC, cfg.maker_fee, BINANCE_DISPLAYED_INTERVALS
    )
    assert reconstructed == pytest.approx(precise_profit, abs=1e-12)
    assert _binance_ui_trunc2(reconstructed) == pytest.approx(display_profit)


def test_grid_calculator_outputs_precise_binance_interval_profit_per_grid() -> None:
    profit, profit_min, profit_max = GridCalculator().calculate_profit_per_grid(
        0.4740,
        0.4970,
        7,
        maker_fee=0.0002,
        taker_fee=0.0002,
    )
    precise_pct = profit * 100.0
    legacy_pct = profit_per_grid_pct(0.4740, 0.4970, 7, GEOMETRIC, 0.0002)

    assert profit == pytest.approx(profit_min)
    assert profit == pytest.approx(profit_max)
    assert precise_pct == pytest.approx(0.639056056101881, abs=1e-12)
    assert legacy_pct == pytest.approx(0.7528381877434942, abs=1e-12)
    assert precise_pct < legacy_pct


def test_pnl_decomposition_identities_are_explicit() -> None:
    cfg = GridConfig(
        symbol="TESTUSDT",
        lower=100.0,
        upper=110.0,
        num_grids=5,
        mode="arithmetic",
        capital=100.0,
        maker_fee=0.0,
        taker_fee=0.0,
        funding_mode="continuous",
        funding_rate=0.0,
        order_delay_bars=0,
        fill_mode="close",
    )
    result = RealisticGridBacktester(cfg).run(_klines([105.0, 102.0, 106.0, 104.0]))

    total_identity = (
        result["net_realized_profit_usdt"]
        + result["open_pnl_usdt"]
        + result["funding_fee_usdt"]
    )
    unmatched_identity = (
        result["total_profit_usdt"]
        - result["matched_profit_usdt"]
        - result["funding_fee_usdt"]
    )
    assert result["total_profit_usdt"] == pytest.approx(total_identity)
    assert result["unmatched_pnl_usdt"] == pytest.approx(unmatched_identity)
    assert result["total_profit_identity_residual_usdt"] == pytest.approx(0.0)
    assert result["unmatched_identity_residual_usdt"] == pytest.approx(0.0)


def test_matched_profit_is_gross_realized_not_fee_net() -> None:
    cfg = GridConfig(
        symbol="TESTUSDT",
        lower=100.0,
        upper=110.0,
        num_grids=5,
        mode="arithmetic",
        capital=100.0,
        maker_fee=0.001,
        taker_fee=0.001,
        funding_mode="continuous",
        funding_rate=0.0,
        order_delay_bars=0,
        fill_mode="close",
    )
    result = RealisticGridBacktester(cfg).run(_klines([105.0, 102.0, 106.0, 104.0]))

    assert result["fees_paid"] > 0.0
    assert result["matched_profit_usdt"] == pytest.approx(result["gross_realized_profit_usdt"])
    assert result["matched_grid_profit_usdt"] == pytest.approx(result["gross_realized_profit_usdt"])
    assert result["net_realized_profit_usdt"] == pytest.approx(
        result["gross_realized_profit_usdt"] - result["fees_paid"]
    )


def _csv_number(value: object) -> float:
    text = str(value).strip().replace(",", "").replace(" USDT", "")
    if not text:
        return 0.0
    return float(text)


def test_history11_reconciles_pnl_and_paired_matched_trade_counts() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    exports = os.path.join(repo_root, "data", "manual_exports")
    order_path = os.path.join(exports, "Order history 11.csv")
    trade_path = os.path.join(exports, "Trade History 11.csv")
    tx_path = os.path.join(exports, "Transaction history 11.csv")
    if not all(os.path.exists(path) for path in (order_path, trade_path, tx_path)):
        pytest.skip("History 11 manual exports are not available")

    orders = pd.read_csv(order_path, dtype=str)
    trades = pd.read_csv(trade_path, dtype=str)
    tx = pd.read_csv(tx_path, dtype=str)
    orders["_order_id"] = orders["Order No"].astype(str).str.strip()
    trades["_order_id"] = trades["Order Id"].astype(str).str.strip()
    trades["_realized"] = trades["Realized Profit"].map(_csv_number)
    trades["_fee"] = trades["Fee"].map(_csv_number)
    tx["_time"] = pd.to_datetime(tx["Date(UTC)"], utc=True, errors="coerce")
    tx["_amount"] = tx["Amount"].map(_csv_number)

    cases = [
        ("IPUSDT", "412235216", "2026-05-26T14:23:48Z", 9, 0.30, 0.29687745),
        ("PNUTUSDT", "412235095", "2026-05-26T14:17:57Z", 8, 0.23, 0.23761746),
    ]
    for symbol, strategy_id, deploy_ts_text, matched_trades, live_total, expected_net in cases:
        bot_orders = orders[
            (orders["Symbol"] == symbol)
            & (orders["Strategy Id"].astype(str).str.strip() == strategy_id)
        ].copy()
        assert not bot_orders.empty

        order_ids = set(bot_orders["_order_id"])
        bot_trades = trades[
            (trades["Symbol"] == symbol) & trades["_order_id"].isin(order_ids)
        ].copy()
        assert not bot_trades.empty

        buy_fills = len(
            bot_orders[
                (bot_orders["Type"] == "LIMIT")
                & (bot_orders["Side"] == "BUY")
                & (bot_orders["Status"] == "FILLED")
            ]
        )
        sell_fills = len(
            bot_orders[
                (bot_orders["Type"] == "LIMIT")
                & (bot_orders["Side"] == "SELL")
                & (bot_orders["Status"] == "FILLED")
            ]
        )
        assert min(buy_fills, sell_fills) == matched_trades

        market_close = bot_orders[
            (bot_orders["Type"] == "MARKET") & (bot_orders["Status"] == "FILLED")
        ].copy()
        assert len(market_close) == 1
        close_ts = pd.to_datetime(market_close.iloc[0]["Update Time"], utc=True)
        deploy_ts = pd.Timestamp(deploy_ts_text)
        tx_window = tx[
            (tx["Symbol"] == symbol)
            & (tx["_time"] >= deploy_ts - pd.Timedelta(minutes=2))
            & (tx["_time"] <= close_ts + pd.Timedelta(minutes=2))
        ]
        funding = float(tx_window.loc[tx_window["type"] == "FUNDING_FEE", "_amount"].sum())
        net = float(bot_trades["_realized"].sum() + bot_trades["_fee"].sum() + funding)

        assert net == pytest.approx(expected_net, abs=1e-8)
        assert abs(net - live_total) < 0.01


def test_deployment_ready_output_time_is_backtest_start(tmp_path) -> None:
    csv_path = tmp_path / "deployment_ready_20260526_131351.csv"
    csv_path.write_text(
        "symbol,score,grid_is_valid,grid_lower,grid_upper,num_grids,candidate_id\n"
        "IPUSDT,80,true,0.4740,0.4970,7,IPUSDT_20260526_131351_e1d0f21e\n",
        encoding="utf-8",
    )
    output_epoch = datetime(2026, 5, 26, 14, 11, 7, tzinfo=timezone.utc).timestamp()
    csv_path.touch()
    os.utime(csv_path, (output_epoch, output_epoch))

    loaded = load_all_scanner_csvs(tmp_path, require_meta_features=False)
    row = loaded.iloc[0].to_dict()

    assert row["scan_file"] == csv_path.name
    assert row["candidate_available_source"] == "deployment_ready_csv_mtime"
    assert resolve_backtest_start_timestamp(row) == datetime(
        2026, 5, 26, 14, 11, 7, tzinfo=timezone.utc
    )


def test_backtest_start_missing_provenance_fails_loudly() -> None:
    with pytest.raises(ValueError, match="candidate_available_ts_utc"):
        resolve_backtest_start_timestamp(
            {"candidate_id": "IPUSDT_20260526_131351_e1d0f21e"}
        )
