"""Tests for MAE/MFE computation in new_bot_data_extractor.compute_trade_metrics."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from new_bot_data_extractor import compute_trade_metrics


def _ms(ts: datetime) -> int:
    return int(ts.timestamp() * 1000)


def _mark_row(ts: datetime, close: float) -> list[object]:
    return [
        _ms(ts),            # open_time
        str(close),         # open
        str(close),         # high
        str(close),         # low
        str(close),         # close
        "0",                # volume
        _ms(ts),            # close_time (used by excursion engine)
        "0", "0", "0", "0", "0",
    ]


def test_compute_trade_metrics_populates_mtm_mae_mfe():
    t0 = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 3, 1, 0, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 3, 1, 0, 2, tzinfo=timezone.utc)

    trades = pd.DataFrame(
        [
            {
                "Side": "Buy",
                "Price": "100.0 USDT",
                "Quantity": "1",
                "Realized Profit": "0.0 USDT",
                "Fee": "0.0 USDT",
                "Maker": "TRUE",
                "parsed_time": t0,
            },
                {
                    "Side": "Sell",
                    "Price": "110.0 USDT",
                    "Quantity": "1",
                    "Realized Profit": "10.0 USDT",
                    "Fee": "0.0 USDT",
                    "Maker": "TRUE",
                    "parsed_time": t2,
                },
            ]
        )

    mark_klines = [
        _mark_row(t0, 100.0),
        _mark_row(t1, 95.0),
        _mark_row(t2, 110.0),
    ]

    metrics = compute_trade_metrics(
        trades,
        mark_klines=mark_klines,
        funding_fee_events=[],
        investment_margin_usdt=400.0,
    )

    assert metrics.mae == pytest.approx(5.0)
    assert metrics.mfe == pytest.approx(15.0)
    assert metrics.mae_pct_initial == pytest.approx(1.25)
    assert metrics.mfe_pct_initial == pytest.approx(3.75)


def test_compute_trade_metrics_without_mark_klines_uses_trade_path():
    t0 = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 3, 1, 0, 1, tzinfo=timezone.utc)

    trades = pd.DataFrame(
        [
            {
                "Side": "Buy",
                "Price": "100.0 USDT",
                "Quantity": "1",
                "Realized Profit": "0.0 USDT",
                "Fee": "0.0 USDT",
                "Maker": "TRUE",
                "parsed_time": t0,
            },
            {
                "Side": "Sell",
                "Price": "100.0 USDT",
                "Quantity": "1",
                "Realized Profit": "10.0 USDT",
                "Fee": "0.0 USDT",
                "Maker": "TRUE",
                "parsed_time": t1,
            },
        ]
    )

    metrics = compute_trade_metrics(trades, mark_klines=[], funding_fee_events=[])
    assert metrics.mae == pytest.approx(0.0)
    assert metrics.mfe == pytest.approx(10.0)
