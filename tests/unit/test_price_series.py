"""
Unit tests for the PriceSeries subsystem.

Covers: ps_types, ps_quality, ps_store, ps_rest_backfill.
All tests are pure (no network calls).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest

from neutralgrid.data.price_series.ps_types import Candle, PriceTick, SeriesKind
from neutralgrid.data.price_series.ps_quality import (
    QualityReport,
    check_candle_monotonic,
    check_tick_monotonic,
    dedupe_candles,
    detect_gaps,
    filter_closed_only,
    validate_candle_series,
)
from neutralgrid.data.price_series.ps_rest_backfill import (
    _raw_kline_to_candle,
    interval_to_ms,
)


# =====================================================================
# Fixtures & helpers
# =====================================================================


def _candle(
    open_time_ms: int,
    close: float = 100.0,
    is_final: bool = True,
    *,
    high: float = 105.0,
    low: float = 95.0,
    volume: float = 10.0,
) -> Candle:
    """Convenience factory for a ``Candle`` with sensible defaults."""
    return Candle(
        open_time_ms=open_time_ms,
        open=close - 1.0,
        high=high,
        low=low,
        close=close,
        volume=volume,
        close_time_ms=open_time_ms + 59_999,
        is_final=is_final,
    )


def _tick(ts_ms: int, mark_price: float = 100.0) -> PriceTick:
    """Convenience factory for a ``PriceTick``."""
    return PriceTick(ts_ms=ts_ms, mark_price=mark_price)


def _make_config_stub(tmp_path: Path):
    """Return a lightweight Config-like object suitable for ``PriceStore``."""
    from neutralgrid.core.config import Config, PriceSeriesConfig

    cfg = Config.__new__(Config)
    # Minimal fields that PriceStore.__init__ touches
    object.__setattr__(cfg, "price_series", PriceSeriesConfig(
        ring_buffer_size=100,
        store_dir=str(tmp_path / "price_store"),
    ))
    object.__setattr__(cfg, "base_dir", tmp_path)

    # resolve_path must work
    def _resolve_path(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else tmp_path / p
    cfg.resolve_path = _resolve_path  # type: ignore[attr-defined]
    return cfg


@pytest.fixture()
def price_store(tmp_path: Path):
    """Yield a ``PriceStore`` backed by a temp directory."""
    from neutralgrid.data.price_series.ps_store import PriceStore

    cfg = _make_config_stub(tmp_path)
    with patch("neutralgrid.data.price_series.ps_store.get_config", return_value=cfg):
        yield PriceStore(ring_buffer_size=100, store_dir=tmp_path / "price_store")


# =====================================================================
# ps_types
# =====================================================================


class TestSeriesKind:
    """SeriesKind enum coverage."""

    def test_values(self):
        assert SeriesKind.LAST_KLINE.value == "last_kline"
        assert SeriesKind.MARK_KLINE.value == "mark_kline"
        assert SeriesKind.MARK_TICK.value == "mark_tick"

    def test_members_count(self):
        assert len(SeriesKind) == 3

    def test_lookup_by_value(self):
        assert SeriesKind("last_kline") is SeriesKind.LAST_KLINE
        assert SeriesKind("mark_kline") is SeriesKind.MARK_KLINE
        assert SeriesKind("mark_tick") is SeriesKind.MARK_TICK


class TestPriceTick:
    """PriceTick construction and immutability."""

    def test_construction_required_fields(self):
        t = PriceTick(ts_ms=1000, mark_price=50_000.0)
        assert t.ts_ms == 1000
        assert t.mark_price == 50_000.0
        assert t.index_price is None

    def test_construction_all_fields(self):
        t = PriceTick(ts_ms=2000, mark_price=50_000.0, index_price=49_990.0)
        assert t.index_price == 49_990.0

    def test_immutability(self):
        t = PriceTick(ts_ms=1000, mark_price=50_000.0)
        with pytest.raises(FrozenInstanceError):
            t.ts_ms = 9999  # type: ignore[misc]

    def test_equality(self):
        a = PriceTick(ts_ms=1, mark_price=1.0)
        b = PriceTick(ts_ms=1, mark_price=1.0)
        assert a == b

    def test_inequality(self):
        a = PriceTick(ts_ms=1, mark_price=1.0)
        b = PriceTick(ts_ms=2, mark_price=1.0)
        assert a != b


class TestCandle:
    """Candle construction and immutability."""

    def test_construction(self):
        c = Candle(
            open_time_ms=0,
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=500.0,
            close_time_ms=59_999,
            is_final=True,
        )
        assert c.open_time_ms == 0
        assert c.close == 105.0
        assert c.is_final is True

    def test_immutability(self):
        c = _candle(0)
        with pytest.raises(FrozenInstanceError):
            c.close = 999.0  # type: ignore[misc]

    def test_equality(self):
        a = _candle(1000, close=100.0)
        b = _candle(1000, close=100.0)
        assert a == b

    def test_inequality_different_open_time(self):
        a = _candle(1000)
        b = _candle(2000)
        assert a != b


# =====================================================================
# ps_quality
# =====================================================================


class TestCheckCandleMonotonic:

    def test_empty_series(self):
        assert check_candle_monotonic([]) is True

    def test_single_candle(self):
        assert check_candle_monotonic([_candle(1000)]) is True

    def test_strictly_increasing(self):
        candles = [_candle(t) for t in (1000, 2000, 3000)]
        assert check_candle_monotonic(candles) is True

    def test_equal_timestamps_not_monotonic(self):
        candles = [_candle(1000), _candle(1000)]
        assert check_candle_monotonic(candles) is False

    def test_decreasing_timestamps_not_monotonic(self):
        candles = [_candle(3000), _candle(2000), _candle(1000)]
        assert check_candle_monotonic(candles) is False

    def test_single_violation_in_middle(self):
        candles = [_candle(1000), _candle(3000), _candle(2000), _candle(4000)]
        assert check_candle_monotonic(candles) is False


class TestCheckTickMonotonic:

    def test_empty_series(self):
        assert check_tick_monotonic([]) is True

    def test_single_tick(self):
        assert check_tick_monotonic([_tick(100)]) is True

    def test_strictly_increasing(self):
        ticks = [_tick(t) for t in (100, 200, 300)]
        assert check_tick_monotonic(ticks) is True

    def test_equal_timestamps_allowed(self):
        """Tick monotonicity is non-decreasing (equal OK)."""
        ticks = [_tick(100), _tick(100), _tick(200)]
        assert check_tick_monotonic(ticks) is True

    def test_decreasing_timestamp_fails(self):
        ticks = [_tick(200), _tick(100)]
        assert check_tick_monotonic(ticks) is False


class TestDetectGaps:

    def test_no_gaps(self):
        """Consecutive candles exactly 1 interval apart produce no gaps."""
        candles = [_candle(t) for t in range(0, 300_000, 60_000)]
        gaps = detect_gaps(candles, interval_ms=60_000)
        assert gaps == []

    def test_single_gap(self):
        candles = [_candle(0), _candle(60_000), _candle(180_000)]
        gaps = detect_gaps(candles, interval_ms=60_000)
        assert len(gaps) == 1
        assert gaps[0] == (60_000, 180_000)

    def test_custom_threshold(self):
        """With a higher threshold factor, the gap disappears."""
        candles = [_candle(0), _candle(60_000), _candle(180_000)]
        # 120_000 delta needs > 60_000 * 2.5 = 150_000 to be a gap
        gaps = detect_gaps(candles, interval_ms=60_000, threshold_factor=2.5)
        assert gaps == []

    def test_multiple_gaps(self):
        candles = [_candle(0), _candle(200_000), _candle(500_000)]
        gaps = detect_gaps(candles, interval_ms=60_000)
        assert len(gaps) == 2

    def test_empty_series(self):
        assert detect_gaps([], interval_ms=60_000) == []

    def test_single_candle(self):
        assert detect_gaps([_candle(0)], interval_ms=60_000) == []


class TestDedupeCandles:

    def test_no_duplicates(self):
        candles = [_candle(1000), _candle(2000), _candle(3000)]
        result = dedupe_candles(candles)
        assert len(result) == 3

    def test_removes_duplicates_keeps_last(self):
        c1 = _candle(1000, close=100.0)
        c2 = _candle(1000, close=200.0)  # same open_time_ms, different close
        result = dedupe_candles([c1, c2])
        assert len(result) == 1
        assert result[0].close == 200.0  # last one wins

    def test_result_sorted_by_open_time(self):
        candles = [_candle(3000), _candle(1000), _candle(2000)]
        result = dedupe_candles(candles)
        times = [c.open_time_ms for c in result]
        assert times == [1000, 2000, 3000]

    def test_empty_series(self):
        assert dedupe_candles([]) == []


class TestFilterClosedOnly:

    def test_keeps_final(self):
        candles = [_candle(1000, is_final=True), _candle(2000, is_final=True)]
        assert len(filter_closed_only(candles)) == 2

    def test_removes_non_final(self):
        candles = [_candle(1000, is_final=True), _candle(2000, is_final=False)]
        result = filter_closed_only(candles)
        assert len(result) == 1
        assert result[0].open_time_ms == 1000

    def test_all_non_final(self):
        candles = [_candle(t, is_final=False) for t in (1000, 2000)]
        assert filter_closed_only(candles) == []

    def test_empty_series(self):
        assert filter_closed_only([]) == []


class TestValidateCandleSeries:

    def test_clean_series(self):
        candles = [_candle(t) for t in range(0, 300_000, 60_000)]
        report = validate_candle_series(candles, interval_ms=60_000)
        assert report.is_monotonic is True
        assert report.has_gaps is False
        assert report.gap_ranges == []
        assert report.duplicate_count == 0

    def test_short_series_defaults(self):
        """Series of 0 or 1 candles returns default (clean) report."""
        report_0 = validate_candle_series([], interval_ms=60_000)
        assert isinstance(report_0, QualityReport)
        assert report_0.is_monotonic is True

        report_1 = validate_candle_series([_candle(0)], interval_ms=60_000)
        assert report_1.is_monotonic is True

    def test_non_monotonic_detected(self):
        candles = [_candle(2000), _candle(1000)]
        report = validate_candle_series(candles, interval_ms=60_000)
        assert report.is_monotonic is False

    def test_gaps_detected(self):
        candles = [_candle(0), _candle(60_000), _candle(300_000)]
        report = validate_candle_series(candles, interval_ms=60_000)
        assert report.has_gaps is True
        assert len(report.gap_ranges) == 1

    def test_duplicates_counted(self):
        candles = [_candle(1000), _candle(1000), _candle(2000)]
        report = validate_candle_series(candles, interval_ms=60_000)
        assert report.duplicate_count == 1

    def test_custom_threshold_factor(self):
        """A lenient threshold may hide gaps that the default would catch."""
        candles = [_candle(0), _candle(60_000), _candle(180_000)]
        strict = validate_candle_series(candles, interval_ms=60_000, threshold_factor=1.5)
        lenient = validate_candle_series(candles, interval_ms=60_000, threshold_factor=3.0)
        assert strict.has_gaps is True
        assert lenient.has_gaps is False


# =====================================================================
# ps_store  (uses the ``price_store`` fixture)
# =====================================================================


class TestPriceStoreAppendAndGetCandles:

    def test_append_and_retrieve_single_candle(self, price_store):
        c = _candle(1000)
        price_store.append_candle("BTCUSDT", SeriesKind.LAST_KLINE, "1m", c)
        result = price_store.get_candles("BTCUSDT", SeriesKind.LAST_KLINE, "1m")
        assert len(result) == 1
        assert result[0] == c

    def test_symbol_normalised_to_upper(self, price_store):
        """Lower-case symbol input should still match upper-case queries."""
        c = _candle(1000)
        price_store.append_candle("btcusdt", SeriesKind.LAST_KLINE, "1m", c)
        result = price_store.get_candles("BTCUSDT", SeriesKind.LAST_KLINE, "1m")
        assert len(result) == 1

    def test_dedup_on_same_open_time(self, price_store):
        """Appending a candle with the same open_time_ms replaces the previous."""
        c1 = _candle(1000, close=100.0)
        c2 = _candle(1000, close=200.0)
        price_store.append_candle("BTCUSDT", SeriesKind.LAST_KLINE, "1m", c1)
        price_store.append_candle("BTCUSDT", SeriesKind.LAST_KLINE, "1m", c2)
        result = price_store.get_candles(
            "BTCUSDT", SeriesKind.LAST_KLINE, "1m", closed_only=False,
        )
        assert len(result) == 1
        assert result[0].close == 200.0

    def test_different_open_time_appends_new(self, price_store):
        price_store.append_candle(
            "BTCUSDT", SeriesKind.LAST_KLINE, "1m", _candle(1000),
        )
        price_store.append_candle(
            "BTCUSDT", SeriesKind.LAST_KLINE, "1m", _candle(2000),
        )
        result = price_store.get_candles("BTCUSDT", SeriesKind.LAST_KLINE, "1m")
        assert len(result) == 2

    def test_closed_only_filtering(self, price_store):
        price_store.append_candle(
            "BTCUSDT", SeriesKind.LAST_KLINE, "1m", _candle(1000, is_final=True),
        )
        price_store.append_candle(
            "BTCUSDT", SeriesKind.LAST_KLINE, "1m", _candle(2000, is_final=False),
        )
        closed = price_store.get_candles(
            "BTCUSDT", SeriesKind.LAST_KLINE, "1m", closed_only=True,
        )
        all_ = price_store.get_candles(
            "BTCUSDT", SeriesKind.LAST_KLINE, "1m", closed_only=False,
        )
        assert len(closed) == 1
        assert len(all_) == 2

    def test_limit_parameter(self, price_store):
        for t in range(0, 600_000, 60_000):
            price_store.append_candle(
                "BTCUSDT", SeriesKind.LAST_KLINE, "1m", _candle(t),
            )
        result = price_store.get_candles(
            "BTCUSDT", SeriesKind.LAST_KLINE, "1m", limit=3,
        )
        assert len(result) == 3
        # Should be the last 3 candles
        assert result[0].open_time_ms == 420_000

    def test_empty_buffer_returns_empty(self, price_store):
        result = price_store.get_candles("ETHUSDT", SeriesKind.LAST_KLINE, "1m")
        assert result == []

    def test_different_kinds_isolated(self, price_store):
        """LAST_KLINE and MARK_KLINE are separate buffers."""
        price_store.append_candle(
            "BTCUSDT", SeriesKind.LAST_KLINE, "1m", _candle(1000, close=100.0),
        )
        price_store.append_candle(
            "BTCUSDT", SeriesKind.MARK_KLINE, "1m", _candle(1000, close=200.0),
        )
        last = price_store.get_candles("BTCUSDT", SeriesKind.LAST_KLINE, "1m")
        mark = price_store.get_candles("BTCUSDT", SeriesKind.MARK_KLINE, "1m")
        assert len(last) == 1
        assert last[0].close == 100.0
        assert len(mark) == 1
        assert mark[0].close == 200.0

    def test_different_intervals_isolated(self, price_store):
        price_store.append_candle(
            "BTCUSDT", SeriesKind.LAST_KLINE, "1m", _candle(1000),
        )
        price_store.append_candle(
            "BTCUSDT", SeriesKind.LAST_KLINE, "5m", _candle(1000),
        )
        assert len(price_store.get_candles("BTCUSDT", SeriesKind.LAST_KLINE, "1m")) == 1
        assert len(price_store.get_candles("BTCUSDT", SeriesKind.LAST_KLINE, "5m")) == 1

    def test_final_candle_queued_for_flush(self, price_store):
        """A final candle should appear in the pending dict for disk flush."""
        c = _candle(1000, is_final=True)
        price_store.append_candle("BTCUSDT", SeriesKind.LAST_KLINE, "1m", c)
        key = ("BTCUSDT", SeriesKind.LAST_KLINE, "1m")
        assert key in price_store._pending_candles
        assert len(price_store._pending_candles[key]) == 1

    def test_non_final_candle_not_queued(self, price_store):
        c = _candle(1000, is_final=False)
        price_store.append_candle("BTCUSDT", SeriesKind.LAST_KLINE, "1m", c)
        key = ("BTCUSDT", SeriesKind.LAST_KLINE, "1m")
        assert key not in price_store._pending_candles


class TestPriceStoreAppendAndGetTicks:

    def test_append_and_get_latest(self, price_store):
        price_store.append_tick("BTCUSDT", _tick(1000, mark_price=50_000.0))
        latest = price_store.get_latest_tick("BTCUSDT")
        assert latest is not None
        assert latest.ts_ms == 1000
        assert latest.mark_price == 50_000.0

    def test_latest_tick_is_most_recent(self, price_store):
        price_store.append_tick("BTCUSDT", _tick(1000, mark_price=50_000.0))
        price_store.append_tick("BTCUSDT", _tick(2000, mark_price=51_000.0))
        latest = price_store.get_latest_tick("BTCUSDT")
        assert latest is not None
        assert latest.ts_ms == 2000
        assert latest.mark_price == 51_000.0

    def test_symbol_normalised(self, price_store):
        price_store.append_tick("btcusdt", _tick(1000))
        assert price_store.get_latest_tick("BTCUSDT") is not None

    def test_no_tick_returns_none(self, price_store):
        assert price_store.get_latest_tick("ETHUSDT") is None


class TestPriceStoreFlushAndLoad:

    def test_flush_writes_parquet(self, price_store):
        c = _candle(1_700_000_000_000, is_final=True)  # ~2023-11-14
        price_store.append_candle("BTCUSDT", SeriesKind.LAST_KLINE, "1m", c)
        count = price_store.flush_to_disk()
        assert count == 1

    def test_flush_clears_pending(self, price_store):
        c = _candle(1_700_000_000_000, is_final=True)
        price_store.append_candle("BTCUSDT", SeriesKind.LAST_KLINE, "1m", c)
        price_store.flush_to_disk()
        assert len(price_store._pending_candles) == 0

    def test_load_round_trip(self, price_store):
        """Write to disk and read back; values must survive the round trip."""
        c = _candle(1_700_000_000_000, close=42_000.0, is_final=True)
        price_store.append_candle("BTCUSDT", SeriesKind.LAST_KLINE, "1m", c)
        price_store.flush_to_disk()

        loaded = price_store.load_from_disk("BTCUSDT", SeriesKind.LAST_KLINE, "1m")
        assert len(loaded) == 1
        assert loaded[0].open_time_ms == c.open_time_ms
        assert loaded[0].close == c.close
        assert loaded[0].is_final is True

    def test_flush_deduplicates_on_disk(self, price_store):
        """Flushing the same open_time_ms twice should keep only the latest."""
        c1 = _candle(1_700_000_000_000, close=40_000.0, is_final=True)
        price_store.append_candle("BTCUSDT", SeriesKind.LAST_KLINE, "1m", c1)
        price_store.flush_to_disk()

        c2 = _candle(1_700_000_000_000, close=41_000.0, is_final=True)
        price_store.append_candle("BTCUSDT", SeriesKind.LAST_KLINE, "1m", c2)
        price_store.flush_to_disk()

        loaded = price_store.load_from_disk("BTCUSDT", SeriesKind.LAST_KLINE, "1m")
        assert len(loaded) == 1
        # The on-disk dedup keeps the row with the higher index (= last)
        assert loaded[0].close == 41_000.0

    def test_gap_fill_flush_preserves_monotonic_disk_order(self, price_store):
        base = 1_700_000_000_000
        price_store.append_candle(
            "BTCUSDT",
            SeriesKind.LAST_KLINE,
            "1m",
            _candle(base, is_final=True),
        )
        price_store.append_candle(
            "BTCUSDT",
            SeriesKind.LAST_KLINE,
            "1m",
            _candle(base + 120_000, is_final=True),
        )
        price_store.flush_to_disk()
        price_store.append_candle(
            "BTCUSDT",
            SeriesKind.LAST_KLINE,
            "1m",
            _candle(base + 60_000, is_final=True),
        )
        price_store.flush_to_disk()

        loaded = price_store.load_from_disk(
            "BTCUSDT",
            SeriesKind.LAST_KLINE,
            "1m",
        )
        assert [candle.open_time_ms for candle in loaded] == [
            base,
            base + 60_000,
            base + 120_000,
        ]

    def test_load_empty_dir_returns_empty(self, price_store):
        loaded = price_store.load_from_disk("XYZUSDT", SeriesKind.LAST_KLINE, "1m")
        assert loaded == []


# =====================================================================
# ps_rest_backfill
# =====================================================================


class TestIntervalToMs:

    @pytest.mark.parametrize(
        ("interval", "expected_ms"),
        [
            ("1m", 60_000),
            ("3m", 180_000),
            ("5m", 300_000),
            ("15m", 900_000),
            ("30m", 1_800_000),
            ("1h", 3_600_000),
            ("2h", 7_200_000),
            ("4h", 14_400_000),
            ("1d", 86_400_000),
        ],
    )
    def test_known_intervals(self, interval: str, expected_ms: int):
        assert interval_to_ms(interval) == expected_ms

    def test_unsupported_interval_raises(self):
        with pytest.raises(ValueError, match="Unsupported interval"):
            interval_to_ms("7m")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Unsupported interval"):
            interval_to_ms("")


class TestRawKlineToCandle:

    def test_basic_conversion(self):
        raw = [
            1_700_000_000_000,  # open_time_ms
            "42000.00",          # open
            "42500.00",          # high
            "41500.00",          # low
            "42100.00",          # close
            "1234.56",           # volume
            1_700_000_059_999,  # close_time_ms
            # Binance returns extra fields; they should be ignored
            "51000000.00",
            100,
            "600.00",
            "25000000.00",
            "0",
        ]
        c = _raw_kline_to_candle(raw, is_final=True)

        assert isinstance(c, Candle)
        assert c.open_time_ms == 1_700_000_000_000
        assert c.open == 42_000.0
        assert c.high == 42_500.0
        assert c.low == 41_500.0
        assert c.close == 42_100.0
        assert c.volume == 1_234.56
        assert c.close_time_ms == 1_700_000_059_999
        assert c.is_final is True

    def test_is_final_false(self):
        raw = [0, "1", "2", "3", "4", "5", 59_999]
        c = _raw_kline_to_candle(raw, is_final=False)
        assert c.is_final is False

    def test_string_numeric_coercion(self):
        """Binance returns most numeric values as strings."""
        raw = ["0", "1.1", "2.2", "3.3", "4.4", "5.5", "59999"]
        c = _raw_kline_to_candle(raw)
        assert c.open_time_ms == 0
        assert c.open == pytest.approx(1.1)
        assert c.volume == pytest.approx(5.5)
        assert c.close_time_ms == 59_999


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
