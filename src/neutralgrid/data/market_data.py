"""
Market data fetching and caching for reproducible training.

This module provides the infrastructure for building reproducible training sets:
- Fetch klines from Binance API with rate limiting
- Cache to disk for reproducibility
- Provide clean interface for training pipeline

Design principles:
- All data fetched is cached to disk (reproducibility)
- Cache includes metadata (timestamp, source, params)
- Training sets are auditable and version-controlled
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import pandas as pd
import warnings

logger = logging.getLogger(__name__)


def _cache_key(symbol: str, timeframe: str, start_time: int, end_time: int, limit: int) -> str:
    """
    Generate cache key for kline data.

    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe (e.g., "1h", "15m")
        start_time: Start timestamp (ms)
        end_time: End timestamp (ms)
        limit: Number of bars

    Returns:
        Cache key string
    """
    key_str = f"{symbol}_{timeframe}_{start_time}_{end_time}_{limit}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cache_path(cache_dir: Path, cache_key: str) -> Path:
    """Get cache file path for a cache key."""
    return cache_dir / f"{cache_key}.json"


KLINE_CACHE_TTL_SECONDS: int = 3600  # Default: 1 hour


async def fetch_klines_cached(
    client,
    symbol: str,
    timeframe: str,
    limit: int = 500,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
    cache_ttl_seconds: int = KLINE_CACHE_TTL_SECONDS,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Fetch klines with disk caching for reproducibility.

    Args:
        client: Binance API client
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        timeframe: Timeframe (e.g., "1h", "15m", "5m")
        limit: Number of bars to fetch
        start_time: Optional start timestamp (ms since epoch)
        end_time: Optional end timestamp (ms since epoch)
        cache_dir: Directory for cache files (default: data/cache/klines)
        force_refresh: Force refresh from API even if cached

    Returns:
        DataFrame with kline data

    Example:
        >>> from api.binance_client import BinanceClient
        >>> client = BinanceClient()
        >>> df = await fetch_klines_cached(
        ...     client, "BTCUSDT", "1h", limit=100
        ... )
        >>> print(df.shape)
        (100, 12)
    """
    # Default cache directory
    if cache_dir is None:
        from neutralgrid.core.config import get_config
        cfg = get_config()
        cache_dir = cfg.resolve_path(cfg.artifacts.cache_dir) / "klines"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use current time if end_time not specified
    if end_time is None:
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)

    # Generate cache key
    cache_key = _cache_key(symbol, timeframe, start_time or 0, end_time, limit)
    cache_path = _get_cache_path(cache_dir, cache_key)

    # Check cache
    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)

            # Validate cache metadata
            metadata = cached.get("metadata", {})
            fetched_at = metadata.get("fetched_at_utc")
            cache_expired = True
            if fetched_at:
                try:
                    fetched_dt = datetime.fromisoformat(fetched_at)
                    age_seconds = (datetime.now(timezone.utc) - fetched_dt).total_seconds()
                    cache_expired = age_seconds > cache_ttl_seconds
                except (ValueError, TypeError):
                    cache_expired = True
            if cache_expired:
                pass  # Fall through to API fetch
            elif (
                metadata.get("symbol") == symbol
                and metadata.get("timeframe") == timeframe
                and metadata.get("limit") == limit
            ):
                # Load DataFrame from cache
                df = pd.DataFrame(
                    cached["data"],
                    columns=pd.Index(
                        [
                            "open_time",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "close_time",
                            "quote_volume",
                            "trades",
                            "taker_buy_base",
                            "taker_buy_quote",
                            "ignore",
                        ]
                    ),
                )

                # Convert types
                for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

                return df
        except Exception as e:
            warnings.warn(f"Cache read failed for {symbol} {timeframe}: {e}")

    # Fetch from API
    klines = await client.get_klines(
        symbol=symbol,
        interval=timeframe,
        limit=limit,
        start_time=start_time,
        end_time=end_time,
        include_current=False,
    )

    # Parse to DataFrame
    df = pd.DataFrame(
        klines,
        columns=pd.Index(
            [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ]
        ),
    )

    # Convert types
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Deduplicate by open_time before caching (API may return overlapping bars)
    df = df.drop_duplicates(subset="open_time", keep="last").reset_index(drop=True)

    # Validate data quality (same gate as binance_vision pipeline)
    from neutralgrid.core.exceptions import ValidationPipelineError
    try:
        from neutralgrid.data.curator import DataCurator
        curator = DataCurator()
        quality = curator.validate_ohlcv(df, timeframe=timeframe, timestamp_col="open_time")
        if not quality.passed:
            msg = f"Data quality check FAILED for {symbol} {timeframe}: {quality.summary()}"
            logger.error(msg)
            if strict:
                raise ValidationPipelineError(
                    symbol=symbol,
                    stage="data_curation",
                    message=msg,
                )
    except ValidationPipelineError:
        raise
    except Exception as e:
        logger.error("Data quality validation skipped for %s: %s", symbol, e)
        if strict:
            raise ValidationPipelineError(
                symbol=symbol,
                stage="data_curation",
                message=f"Curation unavailable: {e}",
            ) from e

    # Cache to disk
    try:
        cache_data = {
            "metadata": {
                "symbol": symbol,
                "timeframe": timeframe,
                "limit": limit,
                "start_time": start_time,
                "end_time": end_time,
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "num_bars": len(df),
            },
            "data": df.assign(
                open_time=df["open_time"].astype("int64") // 10**6,
                close_time=df["close_time"].astype("int64") // 10**6,
            ).to_dict(orient="records"),
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, default=str)
    except Exception as e:
        warnings.warn(f"Cache write failed for {symbol} {timeframe}: {e}")

    return df


async def fetch_training_dataset(
    client,
    symbols: List[str],
    timeframe: str = "1h",
    limit: int = 500,
    cache_dir: Optional[Path] = None,
    max_concurrency: int = 5,
    force_refresh: bool = False,
    strict: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch training dataset for multiple symbols with caching.

    This is the main entry point for building reproducible training sets.
    All fetched data is cached to disk for reproducibility.

    Args:
        client: Binance API client
        symbols: List of symbols to fetch (e.g., ["BTCUSDT", "ETHUSDT"])
        timeframe: Timeframe to fetch (default: "1h")
        limit: Number of bars per symbol (default: 500)
        cache_dir: Directory for cache files
        max_concurrency: Maximum concurrent API requests
        force_refresh: Force refresh from API even if cached

    Returns:
        Dictionary mapping symbol to DataFrame

    Example:
        >>> from api.binance_client import BinanceClient
        >>> client = BinanceClient()
        >>> datasets = await fetch_training_dataset(
        ...     client,
        ...     symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],
        ...     timeframe="1h",
        ...     limit=500,
        ... )
        >>> print(len(datasets))
        3
        >>> print(datasets["BTCUSDT"].shape)
        (500, 12)
    """
    if cache_dir is None:
        from neutralgrid.core.config import get_config
        cfg = get_config()
        cache_dir = cfg.resolve_path(cfg.artifacts.cache_dir) / "klines"

    sem = asyncio.Semaphore(max_concurrency)

    async def _fetch_one(symbol: str) -> tuple[str, pd.DataFrame]:
        async with sem:
            try:
                df = await fetch_klines_cached(
                    client=client,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    cache_dir=cache_dir,
                    force_refresh=force_refresh,
                    strict=strict,
                )
                return (symbol, df)
            except Exception as e:
                warnings.warn(f"Failed to fetch {symbol}: {e}")
                return (symbol, pd.DataFrame())

    # Fetch all symbols concurrently
    results = await asyncio.gather(*[_fetch_one(s) for s in symbols])

    # Filter out failed fetches
    datasets = {symbol: df for symbol, df in results if not df.empty}

    return datasets


def save_training_dataset(
    datasets: Dict[str, pd.DataFrame],
    output_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save training dataset to disk with metadata.

    This creates a versioned, auditable training set that can be used
    for reproducible model training.

    Directory structure:
        output_dir/
          metadata.json          # Dataset metadata
          BTCUSDT.parquet         # Per-symbol data
          ETHUSDT.parquet
          ...

    Args:
        datasets: Dictionary mapping symbol to DataFrame
        output_dir: Output directory for dataset
        metadata: Optional metadata to include

    Returns:
        Path to output directory

    Example:
        >>> datasets = await fetch_training_dataset(...)
        >>> save_training_dataset(
        ...     datasets,
        ...     Path("data/training_sets/2025-01-01"),
        ...     metadata={"description": "Training set for HMM v1.0"}
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each symbol's data
    for symbol, df in datasets.items():
        if not df.empty:
            parquet_path = output_dir / f"{symbol}.parquet"
            df.to_parquet(parquet_path, index=False)

    # Infer timeframe from first dataset (safe for single-row DataFrames)
    timeframe_hours = None
    if datasets:
        first_key = list(datasets.keys())[0]
        first_df = datasets[first_key]
        if len(first_df) >= 2:
            timeframe_hours = (
                first_df["close_time"].iloc[-1] - first_df["close_time"].iloc[-2]
            ).total_seconds() / 3600
        else:
            timeframe_hours = None

    # Save metadata
    dataset_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_symbols": len(datasets),
        "symbols": list(datasets.keys()),
        "total_bars": sum(len(df) for df in datasets.values()),
        "timeframe": timeframe_hours,
    }

    # Merge with provided metadata
    if metadata is not None:
        dataset_metadata.update(metadata)

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, indent=2, default=str)

    return output_dir


def load_training_dataset(dataset_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load training dataset from disk.

    Args:
        dataset_dir: Directory containing saved dataset

    Returns:
        Dictionary mapping symbol to DataFrame

    Example:
        >>> datasets = load_training_dataset(
        ...     Path("data/training_sets/2025-01-01")
        ... )
        >>> print(len(datasets))
        30
    """
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    datasets = {}

    # Load all parquet files
    for parquet_path in dataset_dir.glob("*.parquet"):
        symbol = parquet_path.stem
        df = pd.read_parquet(parquet_path)
        datasets[symbol] = df

    return datasets
