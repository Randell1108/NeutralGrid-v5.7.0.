from __future__ import annotations

import numpy as np
import pandas as pd

from neutralgrid.data.funding_rate import (
    expected_funding_carry_next_hours,
    realized_funding_carry_proxy_next_hours,
)
from neutralgrid.indicators.technical import (
    calc_liquidity_stability_zscore,
    calc_parkinson_vol_ratio,
    calc_variance_ratio,
)
from neutralgrid.scanner.feature_extractor import compute_features


def _make_df(
    periods: int,
    freq: str,
    *,
    base_price: float,
    step: float,
    quote_volume_base: float = 1000.0,
) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=periods, freq=freq, tz="UTC")
    close = base_price + np.arange(periods, dtype=float) * step
    high = close * 1.01
    low = close * 0.99
    volume = np.full(periods, 10.0)
    quote_volume = quote_volume_base + np.arange(periods, dtype=float) * 25.0
    return pd.DataFrame(
        {
            "open_time": idx,
            "open": close - 0.05,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": quote_volume,
        }
    )


def test_calc_parkinson_vol_ratio_returns_float_and_requires_full_windows():
    highs = np.linspace(101.0, 150.0, 96)
    lows = highs * 0.99
    ratio = calc_parkinson_vol_ratio(highs, lows, short_window=16, long_window=96)
    short_log_hl = np.log(highs[-16:] / lows[-16:])
    long_log_hl = np.log(highs[-96:] / lows[-96:])
    short_sigma = np.sqrt(np.mean(short_log_hl * short_log_hl) / (4.0 * np.log(2.0)))
    long_sigma = np.sqrt(np.mean(long_log_hl * long_log_hl) / (4.0 * np.log(2.0)))
    expected = float(short_sigma / long_sigma)
    assert ratio is not None
    assert abs(ratio - expected) < 1e-12
    assert calc_parkinson_vol_ratio(highs[:-1], lows[:-1], short_window=16, long_window=96) is None


def test_calc_variance_ratio_is_strict_on_1m_lookback():
    prices = 100.0 + np.cumsum(np.sin(np.arange(120) / 8.0) * 0.1 + 0.02)
    vr = calc_variance_ratio(prices, lookback=120, aggregation=15)
    log_p = np.log(prices[-120:])
    r1 = np.diff(log_p)
    mu = float(np.mean(r1))
    expected = float(
        np.var(log_p[15:] - log_p[:-15] - 15 * mu, ddof=1)
        / (15 * np.var(r1 - mu, ddof=1))
    )
    assert vr is not None
    assert abs(vr - expected) < 1e-12
    assert calc_variance_ratio(prices[:-1], lookback=120, aggregation=15) is None


def test_calc_liquidity_stability_zscore_matches_manual_window():
    quote_volume = np.arange(1.0, 101.0)
    zscore = calc_liquidity_stability_zscore(
        quote_volume,
        current_window_bars=4,
        trailing_windows=24,
    )
    hourly = quote_volume.reshape(25, 4).sum(axis=1)
    current = float(hourly[-1])
    trailing = hourly[:-1]
    expected = float((current - trailing.mean()) / trailing.std(ddof=0))
    assert zscore is not None
    assert abs(zscore - expected) < 1e-12


def test_expected_funding_carry_next_hours_uses_7h_horizon():
    assert expected_funding_carry_next_hours(
        {"funding_rate_pct": 0.12, "funding_interval_hours": 6.5},
        horizon_hours=7.0,
    ) == 0.12
    assert expected_funding_carry_next_hours(None, horizon_hours=7.0) is None
    assert expected_funding_carry_next_hours(
        {"funding_rate_pct": 0.12, "funding_interval_hours": 8.0},
        horizon_hours=7.0,
    ) == 0.0


def test_realized_funding_carry_proxy_picks_first_settlement_in_window():
    entry_ms = 1_000_000
    raw = [
        {"fundingTime": entry_ms + 2 * 3600 * 1000, "fundingRate": "0.0002"},
        {"fundingTime": entry_ms + 5 * 3600 * 1000, "fundingRate": "0.0004"},
        {"fundingTime": entry_ms + 9 * 3600 * 1000, "fundingRate": "0.0008"},
    ]
    carry = realized_funding_carry_proxy_next_hours(raw, entry_ms, horizon_hours=7.0)
    assert carry == 0.02


def test_compute_features_emits_new_profile_fields():
    df_1h = _make_df(60, "1h", base_price=100.0, step=0.2)
    df_15m = _make_df(120, "15min", base_price=100.0, step=0.05)
    df_5m = _make_df(90, "5min", base_price=100.0, step=0.02)
    df_1m = _make_df(150, "1min", base_price=100.0, step=0.005)

    features = compute_features(
        "BTCUSDT",
        klines_1h=df_1h,
        klines_15m=df_15m,
        klines_5m=df_5m,
        klines_1m=df_1m,
        funding_rate=0.0005,
        funding_info={"funding_rate_pct": 0.05, "funding_interval_hours": 4.0},
        quote_volume_24h=1_000_000.0,
    )

    assert features.parkinson_vol_ratio_4h_24h_pre is not None
    assert features.variance_ratio_1m_15m_pre_2h is not None
    assert features.funding_carry_expected_next_7h == 0.05
    assert features.liquidity_stability_z_1h is not None


def test_compute_features_preserves_missing_live_funding_as_none():
    df_1h = _make_df(60, "1h", base_price=100.0, step=0.2)
    df_15m = _make_df(120, "15min", base_price=100.0, step=0.05)
    df_5m = _make_df(90, "5min", base_price=100.0, step=0.02)
    df_1m = _make_df(150, "1min", base_price=100.0, step=0.005)

    features = compute_features(
        "BTCUSDT",
        klines_1h=df_1h,
        klines_15m=df_15m,
        klines_5m=df_5m,
        klines_1m=df_1m,
        funding_rate=0.0005,
        funding_info=None,
        quote_volume_24h=1_000_000.0,
    )

    assert features.funding_carry_expected_next_7h is None
