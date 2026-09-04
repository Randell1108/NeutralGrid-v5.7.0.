"""
Binance Vision historical data pipeline.

Public API::

    from neutralgrid.data.binance_vision import ensure_kline_store, load_parquet_range

    manifest = await ensure_kline_store("BTCUSDT", "1h", "futures_um", min_years=4.5)
    df = load_parquet_range(parquet_root("data/cache/klines", "futures_um", "BTCUSDT", "1h"))
"""

from .pipeline import ensure_kline_store
from .store import load_parquet_range, parquet_root

__all__ = [
    "ensure_kline_store",
    "load_parquet_range",
    "parquet_root",
]
