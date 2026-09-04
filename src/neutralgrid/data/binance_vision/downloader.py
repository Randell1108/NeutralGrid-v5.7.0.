"""
Async download with retries + SHA256 checksum verification.

Uses ``httpx.AsyncClient`` (matching ``api/binance_client.py`` pattern) with
a per-batch semaphore for rate control.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import ssl
import tempfile
from datetime import date
from pathlib import Path
from typing import List, Optional

import httpx
import truststore

from .urls import (
    Granularity,
    Market,
    checksum_url,
    kline_url,
    mark_price_checksum_url,
    mark_price_kline_url,
    zip_filename,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60.0       # larger than 30 s default (ZIP files)
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 2.0
DEFAULT_MAX_CONCURRENCY = 5
DEFAULT_MAX_BACKOFF = 300.0


def _new_download_client() -> httpx.AsyncClient:
    """Create a Binance Vision client using the operating system trust store."""
    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, verify=ssl_context)


# --- Low-level helpers -------------------------------------------------------

def _is_retryable_http_status(exc: httpx.HTTPStatusError) -> bool:
    """Return whether an HTTP failure can plausibly succeed on retry.

    Binance Vision uses 404 for archives that do not exist. Retrying a
    permanent missing-file response only delays callers and repeats the same
    warning. Request timeout/rate-limit responses and server errors remain
    retryable.
    """
    status = exc.response.status_code
    return status in {408, 425, 429} or 500 <= status <= 599


def _retry_delay(
    exc: Exception,
    *,
    backoff_base: float,
    attempt: int,
) -> float:
    """Return bounded exponential delay, honoring numeric Retry-After."""

    delay = min(float(backoff_base**attempt), DEFAULT_MAX_BACKOFF)
    if isinstance(exc, httpx.HTTPStatusError):
        header = exc.response.headers.get("Retry-After")
        if header is not None:
            try:
                retry_after = float(header)
            except ValueError:
                retry_after = 0.0
            if math.isfinite(retry_after) and retry_after > 0.0:
                delay = max(delay, retry_after)
    return delay

def _parse_checksum_file(content: str) -> str:
    """Extract SHA256 hex digest from Binance CHECKSUM file.

    Format: ``"<sha256>  <filename>"``
    """
    parts = content.strip().split()
    if len(parts) < 1:
        raise ValueError(f"Unexpected checksum format: {content!r}")
    return parts[0].lower()


def _sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# --- Core download -----------------------------------------------------------

async def download_file(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    expected_sha256: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
) -> Path:
    """Download *url* to *dest* with retries.

    If *expected_sha256* is given, the downloaded file is verified and
    re-downloaded on mismatch.

    Raises:
        httpx.HTTPStatusError: immediately for permanent 4xx responses, or
            after retries are exhausted for transient HTTP failures.
        ValueError: on checksum mismatch after retries exhausted.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    _last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            descriptor, temporary_name = tempfile.mkstemp(
                dir=str(dest.parent),
                prefix=f".{dest.name}.",
                suffix=".tmp",
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                await asyncio.to_thread(temporary.write_bytes, resp.content)

                # Verify checksum before the path can become authoritative.
                if expected_sha256 is not None:
                    actual = await asyncio.to_thread(_sha256_file, temporary)
                    if actual != expected_sha256:
                        raise ValueError(
                            f"SHA256 mismatch for {dest.name}: "
                            f"expected {expected_sha256[:16]}…, got {actual[:16]}…"
                        )
                with temporary.open("r+b") as stream:
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, dest)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise

            return dest

        except (httpx.HTTPStatusError, httpx.TransportError, OSError, ValueError) as exc:
            _last_exc = exc
            if isinstance(exc, httpx.HTTPStatusError) and not _is_retryable_http_status(exc):
                raise
            if attempt < max_retries:
                wait = _retry_delay(
                    exc,
                    backoff_base=backoff_base,
                    attempt=attempt,
                )
                logger.warning(
                    "Attempt %d/%d for %s failed (%s), retrying in %.1fs",
                    attempt, max_retries, url.split("/")[-1], exc, wait,
                )
                await asyncio.sleep(wait)
            else:
                raise

    if _last_exc is not None:
        raise _last_exc
    raise RuntimeError("download_file: no download attempts were executed")


async def download_kline_zip(
    client: httpx.AsyncClient,
    symbol: str,
    interval: str,
    dt: date,
    market: Market = "futures_um",
    granularity: Granularity = "monthly",
    cache_dir: Path | None = None,
    force: bool = False,
    require_checksum: bool = False,
) -> Path:
    """Download a single kline ZIP + verify its checksum.

    Skips download if the file already exists on disk and checksum matches
    (unless *force* is ``True``).

    Returns:
        Path to the verified ZIP file.
    """
    if cache_dir is None:
        from neutralgrid.core.config import get_config
        cfg = get_config()
        cache_dir = cfg.resolve_path(cfg.artifacts.cache_dir) / "klines"
    fname = zip_filename(symbol, interval, dt, granularity)
    sub_dir = (
        Path(cache_dir) / market / symbol.upper() / interval / granularity
    )
    dest = sub_dir / fname
    checksum_dest = sub_dir / f"{fname}.CHECKSUM"

    # Fetch checksum first
    cksum_url = checksum_url(symbol, interval, dt, market, granularity)
    try:
        await download_file(client, cksum_url, checksum_dest)
        checksum_text = await asyncio.to_thread(
            checksum_dest.read_text,
            encoding="utf-8",
        )
        expected_sha = _parse_checksum_file(checksum_text)
    except httpx.HTTPStatusError:
        if require_checksum:
            raise
        # Some very old months may not have checksums — proceed without
        expected_sha = None
        logger.debug("No checksum available for %s", fname)

    # Skip if already valid on disk
    if not force and dest.exists():
        if expected_sha is None or await asyncio.to_thread(_sha256_file, dest) == expected_sha:
            logger.debug("Skipping %s (already on disk and valid)", fname)
            return dest

    # Download ZIP
    url = kline_url(symbol, interval, dt, market, granularity)
    logger.info("Downloading %s", fname)
    return await download_file(client, url, dest, expected_sha)


async def download_kline_batch(
    symbol: str,
    interval: str,
    dates: List[date],
    market: Market = "futures_um",
    granularity: Granularity = "monthly",
    cache_dir: Path | None = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    force: bool = False,
    require_checksum: bool = False,
) -> List[Path]:
    """Download a batch of kline ZIPs with concurrency control.

    Creates its own ``httpx.AsyncClient`` for the batch.

    Returns:
        List of paths to successfully downloaded (or cached) ZIP files.
    """
    if not dates:
        return []

    if cache_dir is None:
        from neutralgrid.core.config import get_config
        cfg = get_config()
        cache_dir = cfg.resolve_path(cfg.artifacts.cache_dir) / "klines"

    sem = asyncio.Semaphore(max_concurrency)
    results: List[Optional[Path]] = [None] * len(dates)

    async def _download_one(idx: int, dt: date) -> None:
        async with sem:
            try:
                path = await download_kline_zip(
                    client=client,
                    symbol=symbol,
                    interval=interval,
                    dt=dt,
                    market=market,
                    granularity=granularity,
                    cache_dir=cache_dir,
                    force=force,
                    require_checksum=require_checksum,
                )
                results[idx] = path
            except Exception as exc:
                logger.warning("Failed to download %s %s: %s", symbol, dt, exc)

    async with _new_download_client() as client:
        tasks = [_download_one(i, dt) for i, dt in enumerate(dates)]
        await asyncio.gather(*tasks)

    return [p for p in results if p is not None]


async def download_mark_price_kline_zip(
    client: httpx.AsyncClient,
    symbol: str,
    interval: str,
    dt: date,
    market: Market = "futures_um",
    granularity: Granularity = "monthly",
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Download one checksum-verified Binance Vision mark-price archive."""

    if market != "futures_um":
        raise ValueError("markPriceKlines are supported only for futures_um")
    if cache_dir is None:
        from neutralgrid.core.config import get_config

        cfg = get_config()
        cache_dir = cfg.resolve_path(cfg.artifacts.cache_dir) / "mark_price_klines"
    fname = zip_filename(symbol, interval, dt, granularity)
    sub_dir = (
        Path(cache_dir)
        / "markPriceKlines"
        / market
        / symbol.upper()
        / interval
        / granularity
    )
    dest = sub_dir / fname
    checksum_dest = sub_dir / f"{fname}.CHECKSUM"
    checksum_endpoint = mark_price_checksum_url(
        symbol, interval, dt, market, granularity
    )
    try:
        await download_file(client, checksum_endpoint, checksum_dest)
        checksum_text = await asyncio.to_thread(
            checksum_dest.read_text,
            encoding="utf-8",
        )
        expected_sha = _parse_checksum_file(checksum_text)
    except httpx.HTTPStatusError:
        # The volatility contract requires checksum-verified provenance.  A
        # missing checksum is therefore an unavailable archive, not a reason
        # to ingest unverifiable bytes.
        raise
    if not force and dest.exists():
        if await asyncio.to_thread(_sha256_file, dest) == expected_sha:
            return dest
    endpoint = mark_price_kline_url(symbol, interval, dt, market, granularity)
    logger.info("Downloading mark-price archive %s", fname)
    return await download_file(client, endpoint, dest, expected_sha)


async def download_mark_price_kline_batch(
    symbol: str,
    interval: str,
    dates: List[date],
    market: Market = "futures_um",
    granularity: Granularity = "monthly",
    cache_dir: Path | None = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    force: bool = False,
) -> List[Path]:
    """Download mark-price archives with bounded concurrency and explicit misses."""

    if not dates:
        return []
    sem = asyncio.Semaphore(max_concurrency)
    results: List[Optional[Path]] = [None] * len(dates)

    async with _new_download_client() as client:
        async def download_one(index: int, archive_date: date) -> None:
            async with sem:
                try:
                    results[index] = await download_mark_price_kline_zip(
                        client=client,
                        symbol=symbol,
                        interval=interval,
                        dt=archive_date,
                        market=market,
                        granularity=granularity,
                        cache_dir=cache_dir,
                        force=force,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to download mark-price archive %s %s: %s",
                        symbol,
                        archive_date,
                        exc,
                    )

        await asyncio.gather(
            *(download_one(index, archive_date) for index, archive_date in enumerate(dates))
        )
    return [path for path in results if path is not None]
