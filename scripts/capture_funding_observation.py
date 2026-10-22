"""Capture public funding evidence before manual bot creation; places no orders."""
from __future__ import annotations

import argparse
import asyncio
import logging
import math

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.data.funding_observation import capture_funding_observation

logger = logging.getLogger(__name__)


async def capture(symbol: str, *, count: int, interval_seconds: float | None) -> None:
    if count < 1 or (count > 1 and (interval_seconds is None or not math.isfinite(interval_seconds) or interval_seconds <= 0)):
        raise ValueError("Positive --count and explicit positive interval for repeated captures required")
    client = BinanceClient()
    try:
        for index in range(count):
            _, path = await capture_funding_observation(client, symbol)
            logger.info("Saved original funding observation: %s", path)
            if index + 1 < count:
                assert interval_seconds is not None
                await asyncio.sleep(interval_seconds)
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--interval-seconds", type=float)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(capture(args.symbol, count=args.count, interval_seconds=args.interval_seconds))


if __name__ == "__main__":
    main()
