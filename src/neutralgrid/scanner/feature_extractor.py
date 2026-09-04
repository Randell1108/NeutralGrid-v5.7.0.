from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd

from neutralgrid.core.config import get_config
from neutralgrid.data.funding_rate import expected_funding_carry_next_hours

logger = logging.getLogger(__name__)

from neutralgrid.indicators.technical import (
    calc_ema,
    calc_bollinger_bands,
    calc_vwap,
    calc_atr,
    calc_rsi,
    calc_adx,
    calc_ema_slope,
    count_ema_crosses,
    count_vwap_crosses,
    calc_liquidity_stability_zscore,
    calc_parkinson_vol_ratio,
    calc_variance_ratio,
    detect_trend_structure,
)

def _get_indicator_config():
    """Lazily load indicator config (avoids import-time crash if config not initialized)."""
    cfg = get_config()
    return {
        "adx_period": cfg.indicators.adx_period,
        "rsi_period": cfg.indicators.rsi_period,
        "bb_period": cfg.indicators.bb_period,
        "bb_std_dev": cfg.indicators.bb_std_dev,
        "ema_fast": cfg.indicators.ema_fast,
        "ema_medium": cfg.indicators.ema_medium,
        "ema_slow": cfg.indicators.ema_slow,
    }


def _cfg_val(key: str):
    """Get indicator config value (fresh read each call — no stale cache)."""
    return _get_indicator_config()[key]


def klines_to_df(klines: list[list[Any]]) -> pd.DataFrame:
    """Convert raw Binance kline data to DataFrame.

    AFML: Preserves quote_volume for accurate liquidity features.
    Previously quote_volume was dropped, causing fallback to volume*close approximation.
    """
    if not klines:
        return pd.DataFrame(
            columns=pd.Index(
                ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]
            )
        )

    df = pd.DataFrame(
        klines,
        columns=pd.Index(
            [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "num_trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ]
        ),
    )
    # Convert numeric columns
    for c in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Rename quote_asset_volume to quote_volume for consistency
    df["quote_volume"] = df["quote_asset_volume"]

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")

    # AFML: Include quote_volume in output (was previously dropped)
    result_cols = ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]
    result_df = cast(pd.DataFrame, df[result_cols])
    # AFML: Log dropped rows instead of silent dropna
    dropped_rows = int(cast(pd.Series, result_df.isna().any(axis=1)).sum())
    if dropped_rows > 0:
        logger.warning(f"parse_klines: dropping {dropped_rows} rows with NaN values in OHLCV data")
    return cast(pd.DataFrame, result_df.dropna())


@dataclass
class SymbolFeatures:
    symbol: str

    parkinson_vol_ratio_4h_24h_pre: Optional[float] = None
    variance_ratio_1m_15m_pre_2h: Optional[float] = None
    funding_carry_expected_next_7h: Optional[float] = None
    liquidity_stability_z_1h: Optional[float] = None

    adx_1h: Optional[float] = None
    adx_15m: Optional[float] = None
    adx_5m: Optional[float] = None
    rsi_15m: Optional[float] = None
    ema_slope_1h: Optional[float] = None
    ema_crosses_5m: Optional[int] = None
    vwap_crosses_5m: Optional[int] = None
    range_size_pct: Optional[float] = None
    bb_width: Optional[float] = None
    bb_width_1h: Optional[float] = None
    bb_width_ratio_1h_15m: Optional[float] = None
    trend_structure: Optional[str] = None

    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None

    atr_15m: Optional[float] = None
    atr_pct_15m: Optional[float] = None
    last_price: Optional[float] = None
    quote_volume_24h: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def compute_profile_feature_block(
    *,
    klines_15m: pd.DataFrame,
    klines_1m: pd.DataFrame,
    funding_carry_expected_next_7h: float | None,
) -> Dict[str, Optional[float]]:
    """Compute the 4-field pattern-profile feature block."""
    result: Dict[str, Optional[float]] = {
        "parkinson_vol_ratio_4h_24h_pre": None,
        "variance_ratio_1m_15m_pre_2h": None,
        "funding_carry_expected_next_7h": None,
        "liquidity_stability_z_1h": None,
    }

    if len(klines_15m) >= 96:
        parkinson_ratio = calc_parkinson_vol_ratio(
            np.asarray(klines_15m["high"]),
            np.asarray(klines_15m["low"]),
            short_window=16,
            long_window=96,
        )
        if parkinson_ratio is not None:
            result["parkinson_vol_ratio_4h_24h_pre"] = float(parkinson_ratio)

    if len(klines_1m) >= 120:
        variance_ratio = calc_variance_ratio(
            np.asarray(klines_1m["close"]),
            lookback=120,
            aggregation=15,
        )
        if variance_ratio is not None:
            result["variance_ratio_1m_15m_pre_2h"] = float(variance_ratio)

    if funding_carry_expected_next_7h is not None and np.isfinite(funding_carry_expected_next_7h):
        result["funding_carry_expected_next_7h"] = float(funding_carry_expected_next_7h)

    if len(klines_15m) >= 100 and "quote_volume" in klines_15m.columns:
        quote_volume_series = cast(
            pd.Series,
            pd.to_numeric(cast(pd.Series, klines_15m["quote_volume"]), errors="coerce"),
        )
        liquidity_stability = calc_liquidity_stability_zscore(
            quote_volume_series.to_numpy(dtype=float),
            current_window_bars=4,
            trailing_windows=24,
        )
        if liquidity_stability is not None:
            result["liquidity_stability_z_1h"] = float(liquidity_stability)

    return result


def compute_features(
    symbol: str,
    *,
    klines_1h: pd.DataFrame,
    klines_15m: pd.DataFrame,
    klines_5m: pd.DataFrame,
    klines_1m: pd.DataFrame,
    funding_rate: float | None = None,
    funding_info: dict[str, Any] | None = None,
    open_interest: float | None = None,
    quote_volume_24h: float | None = None,
) -> SymbolFeatures:
    f = SymbolFeatures(
        symbol=symbol,
        funding_rate=funding_rate,
        open_interest=open_interest,
        quote_volume_24h=quote_volume_24h,
    )

    try:
        profile_block = compute_profile_feature_block(
            klines_15m=klines_15m,
            klines_1m=klines_1m,
            funding_carry_expected_next_7h=expected_funding_carry_next_hours(
                funding_info,
                horizon_hours=7.0,
            ),
        )
        f.parkinson_vol_ratio_4h_24h_pre = profile_block["parkinson_vol_ratio_4h_24h_pre"]
        f.variance_ratio_1m_15m_pre_2h = profile_block["variance_ratio_1m_15m_pre_2h"]
        f.funding_carry_expected_next_7h = profile_block["funding_carry_expected_next_7h"]
        f.liquidity_stability_z_1h = profile_block["liquidity_stability_z_1h"]

        # ADX on 1H — ADX needs 2*period bars for full warmup (period for
        # TR/DM smoothing + period for DX smoothing)
        if len(klines_1h) >= 2 * _cfg_val("adx_period"):
            adx, _, _ = calc_adx(np.asarray(klines_1h["high"]), np.asarray(klines_1h["low"]), np.asarray(klines_1h["close"]), _cfg_val("adx_period"))
            f.adx_1h = float(np.round(adx[-1], 2)) if len(adx) and np.isfinite(adx[-1]) else None

        # EMA slope and trend structure - need more bars
        if len(klines_1h) >= max(_cfg_val("ema_medium"), 50) + 10:
            ema = calc_ema(np.asarray(klines_1h["close"]), _cfg_val("ema_medium"))
            slopes = calc_ema_slope(ema, lookback=5)
            f.ema_slope_1h = float(slopes[-1]) if len(slopes) and np.isfinite(slopes[-1]) else None

            f.trend_structure = detect_trend_structure(np.asarray(klines_1h["high"]), np.asarray(klines_1h["low"]), lookback=10)

        # BB width on 1H (for cross-timeframe ratio)
        if len(klines_1h) >= _cfg_val("bb_period") + 10:
            _upper_1h, _mid_1h, _lower_1h, _bbw_1h = calc_bollinger_bands(
                np.asarray(klines_1h["close"]),
                _cfg_val("bb_period"),
                _cfg_val("bb_std_dev"),
            )
            if len(_bbw_1h) and np.isfinite(_bbw_1h[-1]):
                f.bb_width_1h = float(_bbw_1h[-1])

        if len(klines_15m) >= max(2 * _cfg_val("adx_period"), _cfg_val("rsi_period"), _cfg_val("bb_period")) + 10:
            adx, _, _ = calc_adx(np.asarray(klines_15m["high"]), np.asarray(klines_15m["low"]), np.asarray(klines_15m["close"]), _cfg_val("adx_period"))
            f.adx_15m = float(np.round(adx[-1], 2)) if len(adx) and np.isfinite(adx[-1]) else None

            rsi = calc_rsi(np.asarray(klines_15m["close"]), _cfg_val("rsi_period"))
            f.rsi_15m = float(np.round(rsi[-1], 2)) if len(rsi) and np.isfinite(rsi[-1]) else None

            upper, _middle, lower, bb_width = calc_bollinger_bands(np.asarray(klines_15m["close"]), _cfg_val("bb_period"), _cfg_val("bb_std_dev"))
            if len(upper) and len(lower):
                range_size = float(upper[-1] - lower[-1])
                last_close = float(np.asarray(klines_15m["close"])[-1])
                if last_close > 0 and np.isfinite(range_size):
                    f.range_size_pct = float(np.round((range_size / last_close) * 100.0, 4))
                f.bb_width = float(np.round(bb_width[-1], 4)) if np.isfinite(bb_width[-1]) else None

                # Cross-timeframe BB width ratio (1h / 15m)
                if (
                    f.bb_width_1h is not None
                    and f.bb_width is not None
                    and np.isfinite(f.bb_width_1h)
                    and np.isfinite(f.bb_width)
                    and f.bb_width > 0
                ):
                    f.bb_width_ratio_1h_15m = float(f.bb_width_1h / f.bb_width)

            atr = calc_atr(np.asarray(klines_15m["high"]), np.asarray(klines_15m["low"]), np.asarray(klines_15m["close"]), period=14)
            if len(atr) and np.isfinite(atr[-1]):
                f.atr_15m = float(atr[-1])
                last_close = float(np.asarray(klines_15m["close"])[-1])
                if last_close > 0:
                    f.atr_pct_15m = float(atr[-1] / last_close)

            # 24h quote-volume proxy (used for liquidity similarity)
            try:
                if "quote_volume" in klines_15m.columns:
                    qv = np.asarray(pd.to_numeric(klines_15m["quote_volume"], errors="coerce"))
                else:
                    vol_series = cast(
                        pd.Series, pd.to_numeric(klines_15m["volume"], errors="coerce")
                    )
                    close_series = cast(
                        pd.Series, pd.to_numeric(klines_15m["close"], errors="coerce")
                    )
                    qv = np.asarray(vol_series * close_series)
                qv = qv[np.isfinite(qv)]
                if qv.size:
                    wq = qv[-48:] if qv.size >= 48 else qv
                    if f.quote_volume_24h is None:
                        f.quote_volume_24h = float(np.nansum(wq))
            except (ValueError, KeyError) as e:
                logger.debug("Volume computation failed for %s: %s", symbol, e)

            f.last_price = float(np.asarray(klines_15m["close"])[-1])

        if len(klines_5m) >= max(_cfg_val("adx_period"), _cfg_val("ema_medium")) + 10:
            adx, _, _ = calc_adx(np.asarray(klines_5m["high"]), np.asarray(klines_5m["low"]), np.asarray(klines_5m["close"]), _cfg_val("adx_period"))
            f.adx_5m = float(np.round(adx[-1], 2)) if len(adx) and np.isfinite(adx[-1]) else None

            ema_fast = calc_ema(np.asarray(klines_5m["close"]), _cfg_val("ema_fast"))
            ema_med = calc_ema(np.asarray(klines_5m["close"]), _cfg_val("ema_medium"))
            f.ema_crosses_5m = int(count_ema_crosses(ema_fast, ema_med))

            vwap = calc_vwap(np.asarray(klines_5m["high"]), np.asarray(klines_5m["low"]), np.asarray(klines_5m["close"]), np.asarray(klines_5m["volume"]))
            f.vwap_crosses_5m = int(count_vwap_crosses(np.asarray(klines_5m["close"]), vwap))
    except Exception as e:
        logger.warning("compute_features failed for %s: %s", symbol, e, exc_info=True)

    return f
