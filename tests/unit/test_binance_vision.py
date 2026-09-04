"""
Unit tests for the Binance Vision data pipeline.

Covers: urls, ingest, validate, store, manifest.
No network calls — all tests are pure or use tmp_path fixtures.
"""

from __future__ import annotations

import io
import hashlib
import json
import ssl
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import httpx

# ---------------------------------------------------------------------------
# URL construction (pure — no I/O)
# ---------------------------------------------------------------------------

from neutralgrid.data.binance_vision.urls import (
    checksum_url,
    daily_dates,
    kline_url,
    mark_price_checksum_url,
    mark_price_kline_url,
    monthly_dates,
    zip_filename,
)
from neutralgrid.data.binance_vision.downloader import (
    download_file,
    download_kline_zip,
    download_mark_price_kline_zip,
)
from neutralgrid.data.binance_vision import downloader


def test_download_client_uses_system_trust_store() -> None:
    ssl_context = MagicMock()
    client = MagicMock()

    with (
        patch.object(
            downloader.truststore, "SSLContext", return_value=ssl_context
        ) as ssl_context_factory,
        patch.object(downloader.httpx, "AsyncClient", return_value=client) as client_factory,
    ):
        result = downloader._new_download_client()

    assert result is client
    ssl_context_factory.assert_called_once_with(ssl.PROTOCOL_TLS_CLIENT)
    client_factory.assert_called_once_with(
        timeout=downloader.DEFAULT_TIMEOUT,
        verify=ssl_context,
    )


@pytest.mark.asyncio
async def test_download_file_does_not_retry_permanent_404(tmp_path: Path) -> None:
    requests = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(404, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await download_file(
                client,
                "https://data.binance.vision/missing.zip",
                tmp_path / "missing.zip",
                max_retries=3,
                backoff_base=0.0,
            )

    assert requests == 1
    assert not (tmp_path / "missing.zip").exists()


@pytest.mark.asyncio
async def test_download_file_retries_transient_429(tmp_path: Path) -> None:
    requests = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests < 3:
            return httpx.Response(429, request=request)
        return httpx.Response(200, request=request, content=b"ok")

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        result = await download_file(
            client,
            "https://data.binance.vision/eventual.zip",
            tmp_path / "eventual.zip",
            max_retries=3,
            backoff_base=0.0,
        )

    assert requests == 3
    assert result.read_bytes() == b"ok"


@pytest.mark.asyncio
async def test_download_file_never_shortens_retry_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = 0
    delays: list[float] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(
                503,
                request=request,
                headers={"Retry-After": "400"},
            )
        return httpx.Response(200, request=request, content=b"ok")

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(
        "neutralgrid.data.binance_vision.downloader.asyncio.sleep",
        fake_sleep,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        await download_file(
            client,
            "https://data.binance.vision/transient.zip",
            tmp_path / "transient.zip",
            max_retries=2,
            backoff_base=2.0,
        )

    assert requests == 2
    assert delays == [400.0]


@pytest.mark.asyncio
async def test_download_file_atomic_replace_failure_leaves_no_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=b"complete-bytes")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(
        "neutralgrid.data.binance_vision.downloader.os.replace",
        fail_replace,
    )
    destination = tmp_path / "archive.zip"
    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        with pytest.raises(OSError, match="injected"):
            await download_file(
                client,
                "https://data.binance.vision/archive.zip",
                destination,
                max_retries=1,
            )

    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.asyncio
async def test_ordinary_archive_can_require_checksum(tmp_path: Path) -> None:
    requests = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(404, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await download_kline_zip(
                client,
                "BTCUSDT",
                "1m",
                date(2026, 8, 1),
                granularity="daily",
                cache_dir=tmp_path,
                require_checksum=True,
            )

    assert requests == 1
    assert not list(tmp_path.rglob("*.zip"))


@pytest.mark.asyncio
async def test_mark_price_archive_requires_checksum(tmp_path: Path) -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await download_mark_price_kline_zip(
                client,
                "BTCUSDT",
                "1m",
                date(2026, 8, 1),
                granularity="daily",
                cache_dir=tmp_path,
            )

    assert not list(tmp_path.rglob("*.zip"))


@pytest.mark.asyncio
async def test_mark_price_archive_is_verified_against_checksum(tmp_path: Path) -> None:
    archive = b"checksum-verified-archive"
    digest = hashlib.sha256(archive).hexdigest()

    def _handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith(".CHECKSUM"):
            return httpx.Response(
                200,
                request=request,
                text=f"{digest}  BTCUSDT-1m-2026-08-01.zip\n",
            )
        return httpx.Response(200, request=request, content=archive)

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        path = await download_mark_price_kline_zip(
            client,
            "BTCUSDT",
            "1m",
            date(2026, 8, 1),
            granularity="daily",
            cache_dir=tmp_path,
        )

    assert path.read_bytes() == archive
    assert path.with_name(path.name + ".CHECKSUM").is_file()


class TestUrls:
    """Deterministic URL construction."""

    def test_zip_filename_monthly(self):
        assert zip_filename("BTCUSDT", "1h", date(2024, 6, 1), "monthly") == "BTCUSDT-1h-2024-06.zip"

    def test_zip_filename_daily(self):
        assert zip_filename("BTCUSDT", "1h", date(2024, 6, 15), "daily") == "BTCUSDT-1h-2024-06-15.zip"

    def test_zip_filename_uppercase(self):
        assert zip_filename("btcusdt", "1h", date(2024, 1, 1), "monthly") == "BTCUSDT-1h-2024-01.zip"

    def test_kline_url_futures_monthly(self):
        url = kline_url("BTCUSDT", "1h", date(2024, 6, 1), "futures_um", "monthly")
        expected = (
            "https://data.binance.vision/data/futures/um/monthly/klines/"
            "BTCUSDT/1h/BTCUSDT-1h-2024-06.zip"
        )
        assert url == expected

    def test_kline_url_futures_daily(self):
        url = kline_url("ETHUSDT", "15m", date(2024, 6, 15), "futures_um", "daily")
        assert "/daily/klines/ETHUSDT/15m/ETHUSDT-15m-2024-06-15.zip" in url

    def test_checksum_url(self):
        url = checksum_url("BTCUSDT", "1h", date(2024, 6, 1))
        assert url.endswith(".CHECKSUM")
        assert "BTCUSDT-1h-2024-06.zip.CHECKSUM" in url

    def test_mark_price_daily_urls(self):
        url = mark_price_kline_url(
            "btcusdt", "1m", date(2026, 8, 1), granularity="daily"
        )
        assert url.endswith(
            "/futures/um/daily/markPriceKlines/BTCUSDT/1m/"
            "BTCUSDT-1m-2026-08-01.zip"
        )
        assert mark_price_checksum_url(
            "BTCUSDT", "1m", date(2026, 8, 1), granularity="daily"
        ) == url + ".CHECKSUM"

    def test_mark_price_rejects_spot_market(self):
        with pytest.raises(ValueError, match="futures_um"):
            mark_price_kline_url(
                "BTCUSDT", "1m", date(2026, 8, 1), market="spot"
            )

    def test_monthly_dates(self):
        dates = monthly_dates(2024, 1, 2024, 3)
        assert len(dates) == 3
        assert dates[0] == date(2024, 1, 1)
        assert dates[-1] == date(2024, 3, 1)

    def test_monthly_dates_cross_year(self):
        dates = monthly_dates(2023, 11, 2024, 2)
        assert len(dates) == 4
        assert dates[0] == date(2023, 11, 1)
        assert dates[-1] == date(2024, 2, 1)

    def test_daily_dates(self):
        dates = daily_dates(date(2024, 6, 1), date(2024, 6, 3))
        assert len(dates) == 3
        assert dates[0] == date(2024, 6, 1)
        assert dates[-1] == date(2024, 6, 3)

    def test_daily_dates_empty(self):
        dates = daily_dates(date(2024, 6, 5), date(2024, 6, 3))
        assert dates == []


# ---------------------------------------------------------------------------
# Ingest (uses tmp_path for ZIP fixtures)
# ---------------------------------------------------------------------------

from neutralgrid.data.binance_vision.ingest import (
    KLINE_COLUMNS,
    ingest_zips,
    normalize_dataframe,
    parse_zip,
)


def _make_zip(tmp_path: Path, name: str, rows: list[list]) -> Path:
    """Helper: create a Binance-style ZIP with a headerless CSV inside."""
    csv_name = name.replace(".zip", ".csv")
    buf = io.StringIO()
    for row in rows:
        buf.write(",".join(str(v) for v in row) + "\n")

    zip_path = tmp_path / name
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(csv_name, buf.getvalue())
    return zip_path


def _sample_kline_row(open_time_ms: int) -> list:
    """Generate a single valid kline row at *open_time_ms*."""
    close_time_ms = open_time_ms + 3_600_000 - 1
    return [
        open_time_ms,     # open_time
        "50000.0",        # open
        "50100.0",        # high
        "49900.0",        # low
        "50050.0",        # close
        "100.5",          # volume
        close_time_ms,    # close_time
        "5025000.0",      # quote_volume
        "1234",           # trades
        "55.0",           # taker_buy_base
        "2750000.0",      # taker_buy_quote
        "0",              # ignore
    ]


class TestIngest:
    """ZIP parsing and DataFrame normalisation."""

    def test_parse_zip_single_csv(self, tmp_path):
        rows = [_sample_kline_row(1_700_000_000_000 + i * 3_600_000) for i in range(5)]
        zp = _make_zip(tmp_path, "BTCUSDT-1h-2024-06.zip", rows)
        df = parse_zip(zp)
        assert list(df.columns) == KLINE_COLUMNS
        assert len(df) == 5

    def test_parse_zip_no_csv_raises(self, tmp_path):
        zp = tmp_path / "bad.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("readme.txt", "hello")
        with pytest.raises(ValueError, match="Expected 1 CSV"):
            parse_zip(zp)

    def test_normalize_converts_types(self, tmp_path):
        rows = [_sample_kline_row(1_700_000_000_000 + i * 3_600_000) for i in range(3)]
        zp = _make_zip(tmp_path, "test.zip", rows)
        raw = parse_zip(zp)
        df = normalize_dataframe(raw)

        assert pd.api.types.is_datetime64_any_dtype(df["open_time"])
        assert pd.api.types.is_datetime64_any_dtype(df["close_time"])
        assert pd.api.types.is_float_dtype(df["open"])
        assert pd.api.types.is_float_dtype(df["volume"])

    def test_normalize_deduplicates(self, tmp_path):
        ts = 1_700_000_000_000
        rows = [_sample_kline_row(ts), _sample_kline_row(ts)]  # duplicate open_time
        zp = _make_zip(tmp_path, "dup.zip", rows)
        raw = parse_zip(zp)
        df = normalize_dataframe(raw)
        assert len(df) == 1

    def test_ingest_zips_concatenates(self, tmp_path):
        base = 1_700_000_000_000
        z1 = _make_zip(tmp_path, "a.zip", [_sample_kline_row(base + i * 3_600_000) for i in range(3)])
        z2 = _make_zip(tmp_path, "b.zip", [_sample_kline_row(base + (i + 3) * 3_600_000) for i in range(3)])
        df, report = ingest_zips([z1, z2])
        assert len(df) == 6
        assert report["total_raw_rows"] == 6
        assert report["duplicates_removed"] == 0

    def test_ingest_zips_dedupes_overlap(self, tmp_path):
        base = 1_700_000_000_000
        # Overlapping: rows 0-2 in both
        z1 = _make_zip(tmp_path, "a.zip", [_sample_kline_row(base + i * 3_600_000) for i in range(3)])
        z2 = _make_zip(tmp_path, "b.zip", [_sample_kline_row(base + i * 3_600_000) for i in range(3)])
        df, report = ingest_zips([z1, z2])
        assert len(df) == 3
        assert report["duplicates_removed"] == 3

    def test_column_names_match_market_data(self):
        """Column names must exactly match market_data.py:116-129."""
        expected = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ]
        assert KLINE_COLUMNS == expected


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

from neutralgrid.data.binance_vision.validate import (
    check_ohlcv_integrity,
    check_row_count,
    check_time_cadence,
    validate_kline_store,
)


def _make_ohlcv_df(n: int = 100, start_ms: int = 1_700_000_000_000) -> pd.DataFrame:
    """Create a clean OHLCV DataFrame with *n* 1H bars.

    Uses ``utc=True`` to match production output from ``normalize_dataframe``.
    """
    interval_ms = 3_600_000
    times = [start_ms + i * interval_ms for i in range(n)]
    return pd.DataFrame({
        "open_time": pd.to_datetime(times, unit="ms", utc=True),
        "open": np.random.uniform(49000, 51000, n),
        "high": np.random.uniform(50500, 52000, n),
        "low": np.random.uniform(48000, 49500, n),
        "close": np.random.uniform(49000, 51000, n),
        "volume": np.random.uniform(10, 500, n),
        "close_time": pd.to_datetime([t + interval_ms - 1 for t in times], unit="ms", utc=True),
        "quote_volume": np.random.uniform(500000, 5000000, n),
        "trades": np.random.randint(100, 5000, n),
        "taker_buy_base": np.random.uniform(5, 250, n),
        "taker_buy_quote": np.random.uniform(250000, 2500000, n),
        "ignore": 0,
    })


class TestValidate:
    """Data quality gates."""

    def test_ohlcv_integrity_pass(self):
        df = pd.DataFrame({
            "open": [100.0], "high": [110.0], "low": [90.0],
            "close": [105.0], "volume": [50.0],
        })
        ok, metrics, issues = check_ohlcv_integrity(df)
        assert ok is True
        assert not issues

    def test_ohlcv_integrity_bad_high(self):
        df = pd.DataFrame({
            "open": [100.0], "high": [99.0], "low": [90.0],  # high < open
            "close": [105.0], "volume": [50.0],
        })
        ok, metrics, issues = check_ohlcv_integrity(df)
        assert ok is False
        assert metrics["bad_high"] == 1

    def test_ohlcv_integrity_negative_volume(self):
        df = pd.DataFrame({
            "open": [100.0], "high": [110.0], "low": [90.0],
            "close": [105.0], "volume": [-1.0],
        })
        ok, _, issues = check_ohlcv_integrity(df)
        assert ok is False
        assert any("negative volume" in i for i in issues)

    def test_time_cadence_pass(self):
        df = _make_ohlcv_df(50)
        ok, metrics, issues = check_time_cadence(df, interval="1h")
        assert ok is True
        assert metrics["non_monotonic"] == 0

    def test_time_cadence_gap(self):
        df = _make_ohlcv_df(50)
        # Insert a 12-hour gap
        df.loc[25:, "open_time"] = df.loc[25:, "open_time"] + pd.Timedelta(hours=12)
        ok, metrics, issues = check_time_cadence(
            df, interval="1h", max_single_gap_hours=6
        )
        assert ok is False
        assert metrics["max_gap_hours"] > 6

    def test_row_count_pass(self):
        ok, _, _ = check_row_count(pd.DataFrame({"a": range(100)}), min_rows=50)
        assert ok is True

    def test_row_count_fail(self):
        ok, _, issues = check_row_count(pd.DataFrame({"a": range(10)}), min_rows=50)
        assert ok is False
        assert issues

    def test_validate_kline_store_full_pass(self):
        df = _make_ohlcv_df(200)
        # Fix high/low to be valid
        df["high"] = df[["open", "close"]].max(axis=1) + 1
        df["low"] = df[["open", "close"]].min(axis=1) - 1
        result = validate_kline_store(df, interval="1h", min_rows=100)
        assert result.passed is True
        assert "ohlcv_integrity" in result.checks


# ---------------------------------------------------------------------------
# Store (Parquet round-trip with tmp_path)
# ---------------------------------------------------------------------------

from neutralgrid.data.binance_vision.store import (
    get_latest_open_time_ms,
    load_parquet_range,
    parquet_root,
    write_partitioned,
)


class TestStore:
    """Hive-partitioned Parquet read/write."""

    def test_parquet_root_format(self):
        p = parquet_root("data/cache/klines", "futures_um", "BTCUSDT", "1h")
        assert str(p).replace("\\", "/") == "data/cache/klines/futures_um/BTCUSDT/1h/parquet"

    def test_write_and_read_roundtrip(self, tmp_path):
        df = _make_ohlcv_df(50)
        root = tmp_path / "parquet"
        write_partitioned(df, root)

        loaded = load_parquet_range(root)
        assert len(loaded) == 50
        # Columns should NOT include year/month
        assert "year" not in loaded.columns
        assert "month" not in loaded.columns
        # open_time should be datetime
        assert pd.api.types.is_datetime64_any_dtype(loaded["open_time"])

    def test_range_filter(self, tmp_path):
        df = _make_ohlcv_df(100)
        root = tmp_path / "parquet"
        write_partitioned(df, root)

        mid_ms = int(df["open_time"].iloc[50].timestamp() * 1000)
        loaded = load_parquet_range(root, start_ms=mid_ms)
        assert len(loaded) <= 51  # 50..99 inclusive plus potential boundary

    def test_get_latest_open_time_ms(self, tmp_path):
        df = _make_ohlcv_df(10)
        root = tmp_path / "parquet"
        write_partitioned(df, root)

        latest = get_latest_open_time_ms(root)
        expected = int(df["open_time"].max().timestamp() * 1000)
        assert latest == expected

    def test_empty_root(self, tmp_path):
        root = tmp_path / "empty"
        df = load_parquet_range(root)
        assert df.empty


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

from neutralgrid.data.binance_vision.manifest import (
    create_manifest,
    get_last_open_time_ms,
    load_manifest,
    save_manifest,
)


class TestManifest:
    """Manifest create / save / load round-trip."""

    def test_create_manifest_fields(self):
        m = create_manifest(
            symbol="BTCUSDT",
            interval="1h",
            market="futures_um",
            parquet_root="/tmp/parquet",
            start_ms=1_000_000,
            end_ms=2_000_000,
            bar_count=500,
            files_downloaded=6,
        )
        assert m["schema_version"] == "1.0.0"
        assert m["symbol"] == "BTCUSDT"
        assert m["bar_count"] == 500

    def test_save_and_load_roundtrip(self, tmp_path):
        m = create_manifest(
            symbol="ETHUSDT", interval="1h", market="futures_um",
            parquet_root=str(tmp_path), start_ms=100, end_ms=200,
            bar_count=10, files_downloaded=1,
        )
        path = tmp_path / "manifest.json"
        save_manifest(m, path)

        loaded = load_manifest(path)
        assert loaded is not None
        assert loaded["symbol"] == "ETHUSDT"
        assert loaded["bar_count"] == 10

    def test_load_missing_returns_none(self, tmp_path):
        assert load_manifest(tmp_path / "nope.json") is None

    def test_get_last_open_time_ms(self):
        m = {"end_ms": 1_700_000_000_000}
        assert get_last_open_time_ms(m) == 1_700_000_000_000

    def test_get_last_open_time_ms_none(self):
        assert get_last_open_time_ms(None) is None
        assert get_last_open_time_ms({}) is None
