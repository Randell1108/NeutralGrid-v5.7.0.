"""Unit tests for BinanceClient pagination helpers (v20260304)."""

from __future__ import annotations

import pytest

from neutralgrid.api.binance_client import BinanceClient


@pytest.mark.asyncio
async def test_get_user_trades_all_splits_saturated_windows():
    client = BinanceClient(api_key="k", api_secret="s")
    calls: list[tuple[int, int]] = []

    async def fake_get_user_trades(
        symbol: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
        from_id: int | None = None,
    ) -> list[dict]:
        assert symbol == "BTCUSDT"
        assert from_id is None
        assert start_time is not None and end_time is not None
        calls.append((start_time, end_time))
        width = end_time - start_time

        # Force split on wider windows by saturating the requested limit.
        if width > 5_000:
            return [
                {"id": i, "time": start_time + i, "orderId": i}
                for i in range(limit)
            ]

        base = int(start_time)
        return [{"id": base, "time": base, "orderId": base}]

    client.get_user_trades = fake_get_user_trades  # type: ignore[method-assign]

    rows = await client.get_user_trades_all(
        symbol="BTCUSDT",
        start_time=0,
        end_time=20_000,
        limit=1000,
    )

    assert len(calls) > 1
    assert len(rows) >= 2
    assert rows == sorted(rows, key=lambda r: (int(r["time"]), int(r["id"])))


@pytest.mark.asyncio
async def test_get_income_history_all_pages_and_dedupes():
    client = BinanceClient(api_key="k", api_secret="s")

    async def fake_get_income_history(
        symbol: str | None = None,
        income_type: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
        page: int = 1,
    ) -> list[dict]:
        assert symbol == "BTCUSDT"
        assert income_type == "FUNDING_FEE"
        assert limit == 2
        if page == 1:
            return [
                {"tranId": "1", "time": 10, "incomeType": "FUNDING_FEE", "tradeId": ""},
                {"tranId": "2", "time": 20, "incomeType": "FUNDING_FEE", "tradeId": ""},
            ]
        if page == 2:
            # Includes one duplicate of tranId=2 for dedupe validation.
            return [
                {"tranId": "2", "time": 20, "incomeType": "FUNDING_FEE", "tradeId": ""},
                {"tranId": "3", "time": 30, "incomeType": "FUNDING_FEE", "tradeId": ""},
            ]
        return []

    client.get_income_history = fake_get_income_history  # type: ignore[method-assign]

    rows = await client.get_income_history_all(
        symbol="BTCUSDT",
        income_type="FUNDING_FEE",
        start_time=0,
        end_time=100,
        limit=2,
        max_pages=5,
    )

    assert [int(r["time"]) for r in rows] == [10, 20, 30]
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_get_mark_price_klines_all_advances_by_close_time():
    client = BinanceClient(api_key="k", api_secret="s")

    async def fake_get_mark_price_klines(
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
        include_current: bool = False,
    ) -> list[list]:
        assert symbol == "BTCUSDT"
        assert interval == "1m"
        assert end_time == 3_000
        assert include_current is False

        st = int(start_time or 0)
        if st <= 0:
            return [
                [0, "0", "0", "0", "100", "0", 999],
                [1000, "0", "0", "0", "101", "0", 1999],
            ]
        if st <= 2000:
            return [
                [2000, "0", "0", "0", "102", "0", 2999],
            ]
        return []

    client.get_mark_price_klines = fake_get_mark_price_klines  # type: ignore[method-assign]

    rows = await client.get_mark_price_klines_all(
        symbol="BTCUSDT",
        interval="1m",
        start_time=0,
        end_time=3_000,
        limit=2,
        include_current=False,
    )

    assert len(rows) == 3
    assert [int(r[0]) for r in rows] == [0, 1000, 2000]
