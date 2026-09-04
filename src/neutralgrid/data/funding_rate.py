"""
Funding Rate Utilities for AFML Feature Extraction.

Provides current/live funding rate fetching from Binance Futures API.
Used for prospective logging and real-time feature enrichment.

NOTE: Historical funding rates ARE available via /fapi/v1/fundingRate with startTime/endTime.
The current/predicted rate is also available via /fapi/v1/premiumIndex.

AFML Compliance:
- Logs current funding rate at bot creation time
- Enables 14/14 feature coverage for future training data
- Addresses historical data gap identified in implementation

"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def expected_funding_carry_next_hours(
    funding_info: Mapping[str, Any] | None,
    *,
    horizon_hours: float = 7.0,
) -> Optional[float]:
    """Expected funding carry over the next horizon using live premiumIndex data."""
    if not funding_info:
        return None
    try:
        hours_to_next = float(funding_info.get("funding_interval_hours", float("inf")))
    except (TypeError, ValueError):
        return None
    try:
        funding_rate_pct = float(funding_info.get("funding_rate_pct", 0.0))
    except (TypeError, ValueError):
        return None
    if not (float("-inf") < hours_to_next < float("inf")):
        return None
    if not (float("-inf") < funding_rate_pct < float("inf")):
        return None
    if hours_to_next <= float(horizon_hours):
        return float(funding_rate_pct)
    return 0.0


def realized_funding_carry_proxy_next_hours(
    raw_records: List[Dict],
    entry_time_ms: int,
    *,
    horizon_hours: float = 7.0,
) -> float:
    """Historical proxy: first realized settlement in (entry, entry + horizon]."""
    if not raw_records:
        return 0.0
    window_end_ms = int(entry_time_ms + horizon_hours * 3600 * 1000)
    candidates: list[tuple[int, float]] = []
    for row in raw_records:
        try:
            funding_time = int(row["fundingTime"])
            funding_rate = float(row["fundingRate"]) * 100.0
        except (KeyError, TypeError, ValueError):
            continue
        if entry_time_ms < funding_time <= window_end_ms:
            candidates.append((funding_time, funding_rate))
    if not candidates:
        return 0.0
    candidates.sort(key=lambda item: item[0])
    return float(candidates[0][1])


class FundingRateProvider:
    """
    Fetch current funding rates from Binance Futures API.

    Uses the /fapi/v1/premiumIndex endpoint which returns:
    - lastFundingRate: Current/predicted funding rate for next funding event
    - nextFundingTime: Timestamp of next funding event (8h intervals)
    - markPrice: Current mark price
    - indexPrice: Current index price

    Usage:
        from neutralgrid.data.funding_rate import FundingRateProvider
        from api.binance_client import BinanceClient

        provider = FundingRateProvider(client)
        funding_pct = await provider.get_current_funding_rate('BTCUSDT')
        print(f"Current funding rate: {funding_pct:.4f}%")
    """

    def __init__(self, client):
        """
        Initialize FundingRateProvider.

        Args:
            client: BinanceClient instance (async)
        """
        self.client = client

    async def get_current_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Get current funding rate for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            Funding rate as percentage (e.g., 0.01 = 0.01%)
            Returns None if fetch fails

        Example:
            rate = await provider.get_current_funding_rate('BTCUSDT')
            # Returns: 0.0072 (meaning 0.0072% funding rate)
        """
        try:
            result = await self.client.get_premium_index(symbol)

            # Extract funding rate (returned as decimal, e.g., 0.0001 = 0.01%)
            funding_rate = float(result.get('lastFundingRate', 0))

            # Convert to percentage (multiply by 100)
            funding_rate_pct = funding_rate * 100

            logger.debug(
                f"{symbol} funding rate: {funding_rate_pct:.4f}% "
                f"(next funding: {result.get('nextFundingTime')})"
            )

            return funding_rate_pct

        except Exception as e:
            logger.error(f"Failed to fetch funding rate for {symbol}: {e}")
            return None

    async def get_funding_rates_batch(
        self,
        symbols: List[str]
    ) -> Dict[str, Optional[float]]:
        """
        Get current funding rates for multiple symbols.

        Args:
            symbols: List of trading pair symbols

        Returns:
            Dict mapping symbol -> funding rate percentage

        Example:
            rates = await provider.get_funding_rates_batch([
                'BTCUSDT', 'ETHUSDT', 'XRPUSDT'
            ])
            # Returns: {'BTCUSDT': 0.0072, 'ETHUSDT': 0.0050, ...}
        """
        results = {}

        for symbol in symbols:
            rate = await self.get_current_funding_rate(symbol)
            results[symbol] = rate

        return results

    async def get_funding_info(self, symbol: str) -> Optional[Dict]:
        """
        Get full funding rate information for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dict with:
            - funding_rate_pct: Current funding rate as percentage
            - next_funding_time: Timestamp of next funding event
            - mark_price: Current mark price
            - index_price: Current index price
            - funding_interval_hours: Time until next funding (hours)

        Example:
            info = await provider.get_funding_info('BTCUSDT')
            print(f"Funding in {info['funding_interval_hours']:.1f} hours")
        """
        try:
            result = await self.client.get_premium_index(symbol)

            # Extract fields
            funding_rate = float(result.get('lastFundingRate', 0)) * 100
            next_funding_ms = int(result.get('nextFundingTime', 0))
            mark_price = float(result.get('markPrice', 0))
            index_price = float(result.get('indexPrice', 0))

            # Calculate time until next funding
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            time_until_funding_ms = next_funding_ms - now_ms
            time_until_funding_hours = time_until_funding_ms / (1000 * 3600)

            return {
                'symbol': symbol,
                'funding_rate_pct': funding_rate,
                'next_funding_time': next_funding_ms,
                'mark_price': mark_price,
                'index_price': index_price,
                'funding_interval_hours': max(0, time_until_funding_hours),
                'fetched_at': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to fetch funding info for {symbol}: {e}")
            return None


# Convenience function
async def get_current_funding_rate(
    client,
    symbol: str
) -> Optional[float]:
    """
    Quick helper to get current funding rate.

    Args:
        client: BinanceClient instance
        symbol: Trading pair symbol

    Returns:
        Funding rate as percentage (e.g., 0.01 = 0.01%)

    Usage:
        from neutralgrid.data.funding_rate import get_current_funding_rate
        from api.binance_client import BinanceClient

        client = BinanceClient()
        rate = await get_current_funding_rate(client, 'BTCUSDT')
        await client.close()
    """
    provider = FundingRateProvider(client)
    return await provider.get_current_funding_rate(symbol)


# ============================================================================
# PROSPECTIVE LOGGING INTEGRATION
# ============================================================================

def log_funding_rate_to_training_data(
    symbol: str,
    funding_rate_pct: float,
    timestamp: datetime,
    output_file: str = "data/funding_rate_log.json"
):
    """
    Log funding rate for prospective training data collection.

    Call this function when:
    - Creating a new grid bot
    - Adding a bot record to training data
    - Enriching candidates in the pipeline

    Args:
        symbol: Trading pair symbol
        funding_rate_pct: Current funding rate as percentage
        timestamp: When the rate was fetched
        output_file: JSON log file path

    Example:
        # In bot creation script:
        funding_rate = await get_current_funding_rate(client, symbol)
        log_funding_rate_to_training_data(
            symbol=symbol,
            funding_rate_pct=funding_rate,
            timestamp=datetime.now()
        )
    """
    import json
    from pathlib import Path

    # Load existing log
    log_file = Path(output_file)
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = []

    # Append new entry
    entry = {
        'symbol': symbol,
        'funding_rate_pct': funding_rate_pct,
        'timestamp': timestamp.isoformat(),
    }
    log_data.append(entry)

    # Save
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    logger.info(f"Logged funding rate for {symbol}: {funding_rate_pct:.4f}%")


# ============================================================================
# HISTORICAL FUNDING RATE UTILITIES (alignment-v1)
# ============================================================================


class HistoricalFundingLoader:
    """Fetch and align historical funding rates for backtest consumption.

    Wraps :pymethod:`BinanceClient.get_funding_rate` (which already supports
    ``start_time`` / ``end_time`` range queries) and produces a bar-aligned
    ``funding_rate_series`` suitable for ``GridConfig.funding_rate_series``.

    The engine indexes the series by::

        window_idx = bar_idx // funding_interval_bars

    Each ``window_idx`` must map to the rate that was *active* during those
    bars.  A rate is active from its ``fundingTime`` until the next settlement.
    """

    def __init__(self, client) -> None:
        """
        Parameters
        ----------
        client : BinanceClient
            Async Binance client with ``get_funding_rate()`` method.
        """
        self.client = client

    async def fetch_raw(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> List[Dict]:
        """Fetch raw historical funding rate records from Binance.

        Returns list of ``{fundingRate, fundingTime, symbol, markPrice}`` dicts
        at 8-hour settlement boundaries.  Thin wrapper over the existing
        ``BinanceClient.get_funding_rate()`` endpoint.
        """
        return await self.client.get_funding_rate(
            symbol=symbol,
            limit=limit,
            start_time=start_time_ms,
            end_time=end_time_ms,
        )

    @staticmethod
    def align_to_bar_series(
        raw_records: List[Dict],
        backtest_start_ms: int,
        total_bars: int,
        funding_interval_bars: int = 480,
    ) -> List[float]:
        """Convert settlement-timestamped API records into a bar-aligned series.

        The engine consumes ``funding_rate_series[window_idx]`` where
        ``window_idx = bar_idx // funding_interval_bars``.  This method
        resolves which settlement rate was *active* at the midpoint of each
        window, ensuring correct attribution even when bar-0 does not
        coincide with a Binance settlement boundary.

        Parameters
        ----------
        raw_records
            Dicts from :pymeth:`fetch_raw`, each with ``fundingRate``
            (str or float) and ``fundingTime`` (int, milliseconds).
        backtest_start_ms
            Timestamp in ms of bar 0 (first candle open) of the backtest.
        total_bars
            Total number of 1-minute bars in the backtest kline DataFrame.
        funding_interval_bars
            Bars per funding window (default 480 = 8 h × 60 min).

        Returns
        -------
        list[float]
            One signed rate per window_idx.  ``series[0]`` is the rate active
            at bar 0.  Length = ``total_bars // funding_interval_bars + 1``
            to cover a partial final window.
        """
        if not raw_records:
            return []

        # Sort by settlement time ascending
        sorted_recs = sorted(raw_records, key=lambda r: int(r["fundingTime"]))
        settlements = [
            (int(r["fundingTime"]), float(r["fundingRate"]))
            for r in sorted_recs
        ]

        ms_per_bar = 60_000  # 1-minute bars
        num_windows = (total_bars // funding_interval_bars) + 1
        series: List[float] = []

        for w in range(num_windows):
            # Midpoint timestamp of this window
            bar_mid_idx = w * funding_interval_bars + funding_interval_bars // 2
            bar_mid_ms = backtest_start_ms + bar_mid_idx * ms_per_bar

            # Find the rate active at bar_mid_ms:
            # latest settlement whose fundingTime <= bar_mid_ms
            active_rate = settlements[0][1]  # fallback to earliest known
            for settle_ms, rate in settlements:
                if settle_ms <= bar_mid_ms:
                    active_rate = rate
                else:
                    break
            series.append(active_rate)

        return series
