"""Backfill Binance Vision 1H kline store for HMM canonical retrain.

Ensures the top N symbols by volume have at least 180 days of 1H data
in the canonical Binance Vision parquet store.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add src to path
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.data.binance_vision.pipeline import ensure_kline_store
from neutralgrid.scanner.scan import get_top_usdtm_symbols_by_quote_volume

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_vision_store")


async def main() -> None:
    probe_count = 100
    min_years = 180 / 365.25  # exactly 180 days = 4320 bars

    logger.info("=" * 70)
    logger.info("BINANCE VISION 1H KLINE STORE BACKFILL")
    logger.info("=" * 70)
    logger.info("  Probe universe: top %d by volume", probe_count)
    logger.info("  Min history: %.1f years (%.0f days)", min_years, min_years * 365)
    logger.info("=" * 70)

    client = BinanceClient()
    try:
        ranked = await get_top_usdtm_symbols_by_quote_volume(client, top_n=probe_count)
        symbols = [s for s, _ in ranked]
        logger.info("Fetched %d symbols from exchange", len(symbols))
    except Exception as e:
        logger.error("Failed to fetch symbols: %s", e)
        await client.close()
        sys.exit(1)

    succeeded = 0
    failed = 0
    t0 = time.monotonic()

    for i, sym in enumerate(symbols, 1):
        logger.info("[%d/%d] Ensuring store for %s ...", i, len(symbols), sym)
        try:
            result = await ensure_kline_store(
                sym,
                interval="1h",
                market="futures_um",
                min_years=min_years,
                cache_root="data/cache/klines",
                mode="auto",
            )
            bar_count = result.get("bar_count", 0)
            logger.info("  %s: %d bars stored", sym, bar_count)
            succeeded += 1
        except Exception as exc:
            logger.warning("  %s: FAILED (%s)", sym, exc)
            failed += 1

    elapsed = time.monotonic() - t0
    await client.close()

    logger.info("")
    logger.info("=" * 70)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 70)
    logger.info("  Succeeded: %d / %d", succeeded, len(symbols))
    logger.info("  Failed:    %d / %d", failed, len(symbols))
    logger.info("  Elapsed:   %.1f seconds", elapsed)
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
