from __future__ import annotations

import pytest

from neutralgrid.live.decision.private_telemetry import (
    PrivateTelemetryParseError,
    parse_private_telemetry_text,
)


_BAND_DRAWER = """BANDUSDT
Perp
Futures Grid
Working
Isolated
Neutral 10x
Time Created
2026-08-01 13:46:47
PNL
Total Profit (USDT)
0.90 (0.30%)
Invested Margin (USDT)
300.00
Matched Profit (USDT)
6.37 (2.12%)
Unmatched PNL (USDT)
-5.48 (-1.82%)
Funding Fee (USDT)
0.00 (0.00%)
Annualized Yield
737.56%
Transaction Fee (USDT)
-0.48914617
Positions
BANDUSDT
Perp
0.1588
Isolated Margin Balance
300.9051 USDT
Maintenance Margin
7.1168 USDT
Bots Risk Ratio
2.1 Low Risk
Size
474.4467 USDT (2,987.7 BAND)
Margin (USDT)
47.44
Entry Price (USDT)
0.1606
PNL(ROE)
-5.38 (-11.33%)
Margin Ratio
2.37%
Est. Liq. Price
0.1250
Mark Price (USDT)
0.1588
Pending Order
Qty Per Order
995.9 BAND
Last Price
0.1588 USDT
Buy(2)
Sell(2)
% to Fill
Price (USDT)
% to Fill
-0.31%
0.1583
1
0.1606
1.13%
-0.94%
0.1573
2
0.1617
1.82%
Grid Details
Realized Profit
+6.28 USDT
Mode
Geometric
Price Range
0.1530 - 0.1686 USDT
Number of Grids
14
Profit Per Grid
0.65%
Invested Margin
300.00 USDT
Qty Per Order
995.9 BAND
Initial Leverage
10x
Current Leverage
10x
Grid Start Price
0.1627 USDT
Position Margin
47.44 USDT (Isolated)
Margin used by open orders
93.00 USDT
Total Current Margin
140.45 USDT
Strategy Number
413549698
Advanced (Optional)
Stop Loss
-30.00 USDT ≈ -10.00% (Last)
Take Profit
+45.00 USDT ≈ +15.00% (Last)
Close all positions on stop
Enabled
Close all positions on TP/SL stop
Enabled
History
Total Matched Profit
6.37 USDT
24H Matched Trades
6
Total Matched Trades
6
Grid Orders
Sort by:
All
2026-08-01 15:09:30
1.03041790 USDT
2026-08-01 15:47:02
Pending Sell
"""


def test_parse_private_telemetry_preserves_complete_drawer_fields() -> None:
    parsed = parse_private_telemetry_text(_BAND_DRAWER)

    assert parsed["pnl"] == {
        "total_profit_usdt": 0.90,
        "total_profit_pct": 0.30,
        "matched_profit_usdt": 6.37,
        "matched_profit_pct": 2.12,
        "realized_profit_usdt": 6.28,
        "unmatched_pnl_usdt": -5.48,
        "unmatched_pnl_pct": -1.82,
        "funding_fee_usdt": 0.0,
        "funding_fee_pct": 0.0,
        "annualized_yield_pct": 737.56,
        "transaction_fee_usdt": -0.48914617,
    }
    assert parsed["position_inventory"]["size_usdt"] == 474.4467
    assert parsed["position_inventory"]["size_base"] == 2987.7
    assert parsed["position_inventory"]["position_pnl_usdt"] == -5.38
    assert parsed["risk"]["risk_ratio"] == 2.1
    assert parsed["risk"]["risk_label"] == "Low Risk"
    assert parsed["risk"]["liquidation_distance_to_mark_pct"] == pytest.approx(
        21.2846347607
    )

    ladder = parsed["open_order_ladder"]
    assert ladder["qty_per_order_base"] == 995.9
    assert [row["price"] for row in ladder["buy"]] == [0.1583, 0.1573]
    assert [row["price"] for row in ladder["sell"]] == [0.1606, 0.1617]

    assert parsed["grid"]["mode"] == "geometric"
    assert parsed["grid"]["margin_used_by_open_orders_usdt"] == 93.0
    assert parsed["tp_sl"]["stop_loss"]["pnl_usdt"] == -30.0
    assert parsed["tp_sl"]["take_profit"]["roi_pct"] == 15.0
    assert parsed["tp_sl"]["close_all_positions_on_stop"] is True

    history = parsed["order_history"]
    assert history["total_matched_trades"] == 6
    assert history["entries"][0]["matched_profit_usdt"] == 1.0304179
    assert history["entries"][1]["status"] == "Pending Sell"


def test_parse_private_telemetry_fails_closed_on_ladder_count_mismatch() -> None:
    with pytest.raises(PrivateTelemetryParseError, match="ladder count mismatch"):
        parse_private_telemetry_text(_BAND_DRAWER.replace("Sell(2)", "Sell(3)"))


def test_parse_private_telemetry_preserves_declared_side_after_price_crossing() -> None:
    drawer = _BAND_DRAWER.replace(
        "Last Price\n0.1588 USDT\nBuy(2)",
        "Last Price\n0.1580 USDT\nBuy(2)",
    )

    parsed = parse_private_telemetry_text(drawer)

    ladder = parsed["open_order_ladder"]
    assert [row["price"] for row in ladder["buy"]] == [0.1583, 0.1573]
    assert [row["price"] for row in ladder["sell"]] == [0.1606, 0.1617]


def test_parse_private_telemetry_preserves_column_major_all_buy_ladder() -> None:
    pending_start = _BAND_DRAWER.index("Pending Order\n")
    grid_start = _BAND_DRAWER.index("Grid Details\n", pending_start)
    pending = """Pending Order
Qty Per Order
995.9 BAND
Last Price
0.1588 USDT
Buy(3)
Sell(0)
% to Fill
Price (USDT)
% to Fill
-0.31%
0.1583
-0.94%
0.1573
-1.57%
0.1563
1
2
3
No pending orders.
"""
    drawer = _BAND_DRAWER[:pending_start] + pending + _BAND_DRAWER[grid_start:]

    parsed = parse_private_telemetry_text(drawer)

    ladder = parsed["open_order_ladder"]
    assert [row["price"] for row in ladder["buy"]] == [0.1583, 0.1573, 0.1563]
    assert ladder["sell"] == []


def test_parse_private_telemetry_preserves_integer_valued_decimal_price() -> None:
    pending_start = _BAND_DRAWER.index("Pending Order\n")
    grid_start = _BAND_DRAWER.index("Grid Details\n", pending_start)
    pending = """Pending Order
Qty Per Order
2.56 ETC
Last Price
8.827 USDT
Buy(14)
Sell(2)
% to Fill
Price (USDT)
% to Fill
-0.94%
8.744
1
8.874
0.53%
-1.67%
8.679
2
8.941
1.29%
-2.40%
8.615
3
-3.11%
8.552
4
-3.84%
8.488
5
-4.54%
8.426
6
-5.25%
8.363
7
-5.94%
8.302
8
-6.65%
8.240
9
-7.34%
8.179
10
-8.02%
8.119
11
-8.70%
8.059
12
-9.36%
8.000
13
-10.03%
7.941
14
"""
    drawer = (
        _BAND_DRAWER[:pending_start]
        + pending
        + _BAND_DRAWER[grid_start:].replace(
            "0.1530 - 0.1686 USDT",
            "7.941 - 8.941 USDT",
        )
    )

    parsed = parse_private_telemetry_text(drawer)

    ladder = parsed["open_order_ladder"]
    assert [row["price"] for row in ladder["buy"]] == [
        8.744,
        8.679,
        8.615,
        8.552,
        8.488,
        8.426,
        8.363,
        8.302,
        8.240,
        8.179,
        8.119,
        8.059,
        8.0,
        7.941,
    ]
    assert [row["price"] for row in ladder["sell"]] == [8.874, 8.941]
