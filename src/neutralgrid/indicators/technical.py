"""
Technical Indicators for Regime Validation.
Pure calculation functions using numpy/pandas.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, cast


def calc_ema(prices: np.ndarray | pd.Series, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average.
    
    Args:
        prices: Array of closing prices
        period: EMA period
    
    Returns:
        Array of EMA values
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    ema = prices.ewm(span=period, adjust=False).mean()
    ema.iloc[:period - 1] = np.nan
    return np.asarray(ema)


def calc_sma(prices: np.ndarray | pd.Series, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average.

    Uses explicit min_periods=period to ensure NaN during warmup (AFML compliance).
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    return np.asarray(prices.rolling(window=period, min_periods=period).mean())


def calc_bollinger_bands(
    prices: np.ndarray | pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.

    Uses explicit min_periods=period to ensure NaN during warmup (AFML compliance).

    Args:
        prices: Array of closing prices
        period: BB period (default 20)
        std_dev: Standard deviation multiplier (default 2)

    Returns:
        Tuple of (upper_band, middle_band, lower_band, bb_width)
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)

    middle = prices.rolling(window=period, min_periods=period).mean()
    std = prices.rolling(window=period, min_periods=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    # Bollinger Band Width (BBW) = (Upper - Lower) / Middle
    bb_width = np.where(middle != 0, (upper - lower) / middle, np.nan)

    return np.asarray(upper), np.asarray(middle), np.asarray(lower), np.asarray(bb_width)


def calc_donchian(
    highs: np.ndarray | pd.Series,
    lows: np.ndarray | pd.Series,
    period: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Donchian Channels.

    Uses explicit min_periods=period to ensure NaN during warmup (AFML compliance).

    Args:
        highs: Array of high prices
        lows: Array of low prices
        period: Channel period (default 20)

    Returns:
        Tuple of (upper_channel, lower_channel, middle_channel)
    """
    if isinstance(highs, np.ndarray):
        highs = pd.Series(highs)
    if isinstance(lows, np.ndarray):
        lows = pd.Series(lows)

    upper = highs.rolling(window=period, min_periods=period).max()
    lower = lows.rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2

    return np.asarray(upper), np.asarray(lower), np.asarray(middle)


def calc_vwap(
    highs: np.ndarray | pd.Series,
    lows: np.ndarray | pd.Series,
    closes: np.ndarray | pd.Series,
    volumes: np.ndarray | pd.Series
) -> np.ndarray:
    """
    Calculate Volume Weighted Average Price (session VWAP).
    
    Uses cumulative calculation from start of data (session start).
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        volumes: Array of volumes
    
    Returns:
        Array of VWAP values
    """
    # Typical price
    typical_price = (highs + lows + closes) / 3
    
    # Cumulative TPV and volume
    cum_tpv = np.cumsum(typical_price * volumes)
    cum_vol = np.cumsum(volumes)
    
    # VWAP (guard against zero cumulative volume)
    vwap = np.where(cum_vol != 0, cum_tpv / cum_vol, typical_price)
    
    return vwap if isinstance(vwap, np.ndarray) else np.asarray(vwap)


def calc_atr(
    highs: np.ndarray | pd.Series,
    lows: np.ndarray | pd.Series,
    closes: np.ndarray | pd.Series,
    period: int = 14
) -> np.ndarray:
    """
    Calculate Average True Range.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        period: ATR period (default 14)
    
    Returns:
        Array of ATR values
    """
    if isinstance(highs, np.ndarray):
        highs = pd.Series(highs)
    if isinstance(lows, np.ndarray):
        lows = pd.Series(lows)
    if isinstance(closes, np.ndarray):
        closes = pd.Series(closes)
    
    # True Range components
    high_low = highs - lows
    high_close = (highs - closes.shift(1)).abs()
    low_close = (lows - closes.shift(1)).abs()
    
    # True Range is max of the three
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # ATR uses Wilder's smoothing (RMA): alpha = 1/period (matches ADX, TradingView, Binance)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    atr.iloc[:period] = np.nan

    return np.asarray(atr)


def calc_rsi(
    prices: np.ndarray | pd.Series,
    period: int = 14
) -> np.ndarray:
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: Array of closing prices
        period: RSI period (default 14)
    
    Returns:
        Array of RSI values (0-100)
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    # Price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = (-delta).where(delta < 0, 0)
    
    # Wilder's smoothing (RMA) of gains and losses: alpha = 1/period
    # Matches TradingView/Binance RSI (not standard EMA which uses alpha = 2/(period+1))
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False).mean()
    
    # RS and RSI (safe division to avoid division by zero)
    avg_loss_safe = cast(pd.Series, avg_loss.replace(0, np.nan))
    rs = cast(pd.Series, avg_gain / avg_loss_safe)
    rs = cast(pd.Series, rs.replace([np.inf, -np.inf], np.nan))
    rsi = cast(pd.Series, 100.0 - (100.0 / (1.0 + rs)))
    rsi.iloc[:period] = np.nan

    return np.asarray(rsi)


def calc_adx(
    highs: np.ndarray | pd.Series,
    lows: np.ndarray | pd.Series,
    closes: np.ndarray | pd.Series,
    period: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Average Directional Index (ADX) with DI+ and DI-.
    
    Uses Wilder's smoothing (RMA) with alpha=1/period to match TradingView/Binance.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        period: ADX period (default 14)
    
    Returns:
        Tuple of (adx, plus_di, minus_di)
    """
    if isinstance(highs, np.ndarray):
        highs = pd.Series(highs)
    if isinstance(lows, np.ndarray):
        lows = pd.Series(lows)
    if isinstance(closes, np.ndarray):
        closes = pd.Series(closes)
    
    # Calculate True Range
    high_low = highs - lows
    high_close = (highs - closes.shift(1)).abs()
    low_close = (lows - closes.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate directional movement
    up_move = highs.diff()
    down_move = -lows.diff()
    
    # +DM and -DM conditions
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=highs.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=lows.index)
    
    # Wilder's smoothing (RMA): alpha = 1/period
    # EMA uses alpha = 2/(period+1), Wilder's uses alpha = 1/period
    alpha = 1.0 / period
    tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()
    
    # Calculate DI+ and DI-
    plus_di = cast(pd.Series, (plus_dm_smooth / tr_smooth) * 100.0)
    minus_di = cast(pd.Series, (minus_dm_smooth / tr_smooth) * 100.0)
    
    # Handle division by zero (let NaN propagate naturally for warmup)
    plus_di = plus_di.replace([np.inf, -np.inf], np.nan)
    minus_di = minus_di.replace([np.inf, -np.inf], np.nan)
    
    # Calculate DX
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = pd.Series(np.where(di_sum != 0, 100 * (di_diff / di_sum), 0.0), index=highs.index)
    
    # ADX is Wilder's smoothed DX
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    # Enforce warmup NaN for 2 * period bars (DI needs period, ADX needs another period)
    warmup = 2 * period
    adx.iloc[:warmup] = np.nan
    plus_di.iloc[:warmup] = np.nan
    minus_di.iloc[:warmup] = np.nan

    return np.asarray(adx), np.asarray(plus_di), np.asarray(minus_di)


def calc_ema_slope(ema: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Calculate the slope of an EMA line.
    
    Args:
        ema: Array of EMA values
        lookback: Number of periods to calculate slope over
    
    Returns:
        Array of EMA slopes (normalized by price level)
    """
    slopes = np.full(len(ema), np.nan)
    for i in range(lookback, len(ema)):
        # Linear regression slope
        x = np.arange(lookback)
        y = ema[i - lookback:i]
        if len(y) == lookback and np.all(np.isfinite(y)) and y.mean() != 0:
            slope = np.polyfit(x, y, 1)[0]
            # Normalize by current price level
            slopes[i] = slope / y[-1]
    return slopes


def count_ema_crosses(ema_fast: np.ndarray, ema_slow: np.ndarray) -> int:
    """
    Count the number of times two EMAs cross each other.
    
    Args:
        ema_fast: Fast EMA values
        ema_slow: Slow EMA values
    
    Returns:
        Number of crossovers
    """
    # Filter out NaN values
    valid_mask = ~(np.isnan(ema_fast) | np.isnan(ema_slow))
    fast = ema_fast[valid_mask]
    slow = ema_slow[valid_mask]
    
    if len(fast) < 2:
        return 0
    
    # Position: 1 if fast > slow, -1 otherwise
    position = np.sign(fast - slow)
    
    # Count changes in position (crosses)
    crosses = np.sum(np.abs(np.diff(position)) == 2)
    
    return int(crosses)


def count_vwap_crosses(prices: np.ndarray, vwap: np.ndarray) -> int:
    """
    Count the number of times price crosses VWAP.
    
    Args:
        prices: Close prices
        vwap: VWAP values
    
    Returns:
        Number of VWAP crossovers
    """
    # Filter out NaN values
    valid_mask = ~(np.isnan(prices) | np.isnan(vwap))
    p = prices[valid_mask]
    v = vwap[valid_mask]
    
    if len(p) < 2:
        return 0
    
    # Position relative to VWAP
    position = np.sign(p - v)
    
    # Count crosses
    crosses = np.sum(np.abs(np.diff(position)) == 2)
    
    return int(crosses)


def calc_parkinson_volatility(
    highs: np.ndarray | pd.Series,
    lows: np.ndarray | pd.Series,
    window: int,
) -> Optional[float]:
    """Estimate Parkinson volatility over the trailing window."""
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    if h.size < window or l.size < window or window <= 0:
        return None
    h = h[-window:]
    l = l[-window:]
    valid = np.isfinite(h) & np.isfinite(l) & (h > 0.0) & (l > 0.0)
    if int(valid.sum()) < window:
        return None
    log_hl = np.log(h[valid] / l[valid])
    variance = np.mean(log_hl * log_hl) / (4.0 * np.log(2.0))
    if not np.isfinite(variance) or variance < 0.0:
        return None
    return float(np.sqrt(variance))


def calc_parkinson_vol_ratio(
    highs: np.ndarray | pd.Series,
    lows: np.ndarray | pd.Series,
    short_window: int = 16,
    long_window: int = 96,
) -> Optional[float]:
    """Trailing Parkinson volatility ratio short_window / long_window."""
    short_sigma = calc_parkinson_volatility(highs, lows, short_window)
    long_sigma = calc_parkinson_volatility(highs, lows, long_window)
    if short_sigma is None or long_sigma is None or long_sigma <= 0.0:
        return None
    ratio = short_sigma / long_sigma
    if not np.isfinite(ratio):
        return None
    return float(ratio)


def calc_variance_ratio(
    prices: np.ndarray | pd.Series,
    lookback: int = 120,
    aggregation: int = 15,
) -> Optional[float]:
    """Lo-MacKinlay-style variance ratio on trailing prices."""
    p = np.asarray(prices, dtype=float)
    if p.size < lookback or aggregation <= 1:
        return None
    p = p[-lookback:]
    valid = np.isfinite(p) & (p > 0.0)
    if int(valid.sum()) < lookback:
        return None
    log_p = np.log(p[valid])
    r1 = np.diff(log_p)
    if r1.size <= aggregation:
        return None
    mu = float(np.mean(r1))
    var1 = float(np.var(r1 - mu, ddof=1))
    if not np.isfinite(var1) or var1 <= 0.0:
        return None
    rq = log_p[aggregation:] - log_p[:-aggregation]
    if rq.size < 2:
        return None
    denom = aggregation * np.var(r1 - mu, ddof=1)
    if not np.isfinite(denom) or denom <= 0.0:
        return None
    varq = float(np.var(rq - aggregation * mu, ddof=1))
    vr = varq / denom
    if not np.isfinite(vr):
        return None
    return float(vr)


def calc_liquidity_stability_zscore(
    quote_volume: np.ndarray | pd.Series,
    current_window_bars: int = 4,
    trailing_windows: int = 24,
) -> Optional[float]:
    """Current 1h quote-volume sum z-scored vs the prior hourly distribution.

    Uses non-overlapping windows so ``current_window_bars=4`` and
    ``trailing_windows=24`` means:
      - latest 4 x 15m bars form the current 1h sum
      - preceding 24 hourly sums come from the prior 24h of 15m quote volume
    """
    qv = np.asarray(quote_volume, dtype=float)
    needed = current_window_bars * (trailing_windows + 1)
    if qv.size < needed:
        return None
    windowed = qv[-needed:]
    if not np.all(np.isfinite(windowed)):
        return None
    hourly = windowed.reshape(trailing_windows + 1, current_window_bars).sum(axis=1)
    current = float(hourly[-1])
    trailing_arr = np.asarray(hourly[:-1], dtype=float)
    mean = float(np.mean(trailing_arr))
    std = float(np.std(trailing_arr, ddof=0))
    if not np.isfinite(std) or std <= 0.0:
        return None
    zscore = (current - mean) / std
    if not np.isfinite(zscore):
        return None
    return float(zscore)


def detect_trend_structure(
    highs: np.ndarray,
    lows: np.ndarray,
    lookback: int = 10
) -> str:
    """
    Detect if price shows trending structure (HH/HL or LL/LH).
    
    Args:
        highs: High prices
        lows: Low prices
        lookback: Number of periods to analyze
    
    Returns:
        "uptrend", "downtrend", or "range"
    """
    if len(highs) < lookback * 2:
        return "range"
    
    # Find local maxima and minima
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    older_highs = highs[-lookback * 2:-lookback]
    older_lows = lows[-lookback * 2:-lookback]
    
    recent_hh = np.max(recent_highs)
    recent_ll = np.min(recent_lows)
    _recent_hl = np.min(recent_highs)
    _recent_lh = np.max(recent_lows)
    
    older_hh = np.max(older_highs)
    older_ll = np.min(older_lows)
    
    # Check for uptrend (HH + HL)
    if recent_hh > older_hh and recent_ll > older_ll:
        return "uptrend"
    
    # Check for downtrend (LH + LL)
    if recent_hh < older_hh and recent_ll < older_ll:
        return "downtrend"
    
    return "range"
