"""
Deterministic URL construction for Binance Vision historical data.

Binance Vision provides free historical kline data as ZIP archives at:
    https://data.binance.vision/data/{market}/monthly/klines/{SYMBOL}/{INTERVAL}/
    https://data.binance.vision/data/{market}/daily/klines/{SYMBOL}/{INTERVAL}/

Each ZIP contains a single headerless CSV with 12 columns matching the
Binance kline REST API format.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Literal

BASE_URL = "https://data.binance.vision/"

# Market path segments
MARKET_PATHS = {
    "futures_um": "futures/um",
    "spot": "spot",
}

Granularity = Literal["monthly", "daily"]
Market = Literal["futures_um", "spot"]


def zip_filename(
    symbol: str,
    interval: str,
    dt: date,
    granularity: Granularity = "monthly",
) -> str:
    """Build the ZIP filename for a given date.

    Examples:
        monthly: BTCUSDT-1h-2024-06.zip
        daily:   BTCUSDT-1h-2024-06-15.zip
    """
    sym = symbol.upper()
    if granularity == "monthly":
        return f"{sym}-{interval}-{dt.year}-{dt.month:02d}.zip"
    return f"{sym}-{interval}-{dt.year}-{dt.month:02d}-{dt.day:02d}.zip"


def _kline_path(
    symbol: str,
    interval: str,
    dt: date,
    market: Market = "futures_um",
    granularity: Granularity = "monthly",
) -> str:
    """Build the relative path segment (without base URL)."""
    market_seg = MARKET_PATHS[market]
    fname = zip_filename(symbol, interval, dt, granularity)
    return f"data/{market_seg}/{granularity}/klines/{symbol.upper()}/{interval}/{fname}"


def kline_url(
    symbol: str,
    interval: str,
    dt: date,
    market: Market = "futures_um",
    granularity: Granularity = "monthly",
) -> str:
    """Full URL for a kline ZIP archive."""
    return BASE_URL + _kline_path(symbol, interval, dt, market, granularity)


def checksum_url(
    symbol: str,
    interval: str,
    dt: date,
    market: Market = "futures_um",
    granularity: Granularity = "monthly",
) -> str:
    """Full URL for the SHA256 checksum file (.CHECKSUM)."""
    return kline_url(symbol, interval, dt, market, granularity) + ".CHECKSUM"


def mark_price_kline_url(
    symbol: str,
    interval: str,
    dt: date,
    market: Market = "futures_um",
    granularity: Granularity = "monthly",
) -> str:
    """Full URL for a futures mark-price kline ZIP archive.

    Mark-price archives are a separate Binance Vision dataset.  Keeping this
    additive API avoids changing the existing regular-kline URL contract.
    """

    if market != "futures_um":
        raise ValueError("markPriceKlines are supported only for futures_um")
    market_seg = MARKET_PATHS[market]
    fname = zip_filename(symbol, interval, dt, granularity)
    relative = (
        f"data/{market_seg}/{granularity}/markPriceKlines/"
        f"{symbol.upper()}/{interval}/{fname}"
    )
    return BASE_URL + relative


def mark_price_checksum_url(
    symbol: str,
    interval: str,
    dt: date,
    market: Market = "futures_um",
    granularity: Granularity = "monthly",
) -> str:
    """Full URL for a mark-price kline archive checksum."""

    return mark_price_kline_url(
        symbol, interval, dt, market, granularity
    ) + ".CHECKSUM"


def monthly_dates(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> List[date]:
    """Generate first-of-month dates for each month in the inclusive range.

    >>> monthly_dates(2024, 1, 2024, 3)
    [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)]
    """
    dates: List[date] = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        dates.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates


def daily_dates(start: date, end: date) -> List[date]:
    """Generate a list of dates from *start* to *end* inclusive.

    >>> daily_dates(date(2024, 6, 1), date(2024, 6, 3))
    [date(2024, 6, 1), date(2024, 6, 2), date(2024, 6, 3)]
    """
    dates: List[date] = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates
