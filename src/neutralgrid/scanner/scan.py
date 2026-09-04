from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd

from neutralgrid.core.config import get_config

logger = logging.getLogger(__name__)
from neutralgrid.api.binance_client import BinanceClient

from neutralgrid.core.candidate_id import make_candidate_id_from_row
from neutralgrid.scanner.feature_extractor import klines_to_df, compute_features, SymbolFeatures
from neutralgrid.scanner.pattern_profile import PatternProfile
from neutralgrid.scanner.profile_model import ProfileModel

# HMM regime inference for upfront probability computation
from neutralgrid.validation.hmm_regime import load_hmm_model, infer_regime, ensure_hmm_model

# AFML-compliant scoring components
try:
    __import__("neutralgrid.scanner.pnl_ranker")
    __import__("neutralgrid.models.meta_labeler")
    AFML_SCORING_AVAILABLE = True
except ImportError:
    AFML_SCORING_AVAILABLE = False

# AFML Training: Feature snapshot collection for meta-labeler training
try:
    from neutralgrid.training import (
        build_feature_snapshot,
    )
    FEATURE_LOGGING_AVAILABLE = True
except ImportError:
    FEATURE_LOGGING_AVAILABLE = False
    build_feature_snapshot = None

# Stochastic regime validation for complete feature set
try:
    from neutralgrid.validation.stochastic import StochasticRegimeChecker, StochasticConfig
    from neutralgrid.validation.utility import (
        compute_governed_provisional_utility,
        UtilityCalibratorUnavailable,
    )
    STOCHASTIC_VALIDATION_AVAILABLE = True
except ImportError:
    STOCHASTIC_VALIDATION_AVAILABLE = False
    StochasticRegimeChecker = None
    StochasticConfig = None
    compute_governed_provisional_utility = None
    UtilityCalibratorUnavailable = None  # type: ignore[assignment,misc]

# Module-level guard so the utility-unavailable warning is emitted once per
# process; subsequent occurrences log at debug level.
_utility_unavailable_warning_emitted = False

# Phase 2: MI-weighted scoring (replaces fixed weights when artifact available)
try:
    from neutralgrid.scanner.mi_weighted_scorer_v20260311 import (
        compute_mi_weighted_score,
        default_runtime_mi_weights,
    )
    MI_SCORER_AVAILABLE = True
except ImportError:
    MI_SCORER_AVAILABLE = False
    compute_mi_weighted_score = None  # type: ignore[assignment]
    default_runtime_mi_weights = None  # type: ignore[assignment]


@dataclass
class ScanResult:
    symbol: str
    score: float
    features: Dict[str, Any]
    notes: List[str]


class RateGate:
    def __init__(self, *, max_concurrency: int = 8, min_delay_s: float = 0.05):
        self._sem = asyncio.Semaphore(max_concurrency)
        self._min_delay_s = float(min_delay_s)
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def __aenter__(self):
        await self._sem.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._sem.release()

    async def throttle(self):
        if self._min_delay_s <= 0:
            return
        async with self._lock:
            now = asyncio.get_running_loop().time()
            wait = self._min_delay_s - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_running_loop().time()


async def get_top_usdtm_symbols_by_quote_volume(client: BinanceClient, *, top_n: int = 100) -> List[Tuple[str, float]]:
    exchange = await client.get_exchange_info()
    symbols = []
    for s in exchange.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("status") != "TRADING":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        symbols.append(s.get("symbol"))

    symbols_set = set(symbols)

    tickers = await client._request(get_config().binance.endpoints["ticker"], params={})
    rows: list[tuple[str, float]] = []
    for t in tickers:
        sym = t.get("symbol")
        if sym not in symbols_set:
            continue
        try:
            qv = float(t.get("quoteVolume"))
        except Exception:
            continue
        rows.append((sym, qv))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]


def _compute_stochastic_features(
    df_15m: pd.DataFrame,
    range_prob: Optional[float],
    trend_prob: Optional[float],
    range_size_pct: Optional[float],
) -> Dict[str, Any]:
    """
    Compute stochastic regime features (survival_prob, hurst, ou_halflife, utility).

    These features require 15M OHLCV data and are expensive to compute,
    so they're only computed when feature logging is enabled.

    Args:
        df_15m: 15-minute OHLCV DataFrame
        range_prob: HMM range probability
        trend_prob: HMM trend probability
        range_size_pct: Range size as percentage

    Returns:
        Dict with stochastic features (may be partial if computation fails)
    """
    result: Dict[str, Optional[float]] = {
        "survival_prob": None,
        "hurst_exponent": None,
        "ou_halflife": None,
        "utility_score": None,
    }

    if not STOCHASTIC_VALIDATION_AVAILABLE:
        return result

    if df_15m is None or len(df_15m) < 300:
        if df_15m is not None and len(df_15m) < 300:
            logger.debug(
                "Stochastic features skipped: %d bars < 300 minimum for stable estimation",
                len(df_15m),
            )
        return result

    try:
        import numpy as np

        closes = np.asarray(df_15m["close"])
        highs = np.asarray(df_15m["high"])
        lows = np.asarray(df_15m["low"])

        # Compute range bounds for stochastic analysis
        lookback = min(192, len(df_15m))
        range_high = float(np.nanpercentile(highs[-lookback:], 95))
        range_low = float(np.nanpercentile(lows[-lookback:], 5))

        if range_low <= 0 or range_high <= range_low:
            return result

        # Get log prices
        log_prices = np.log(closes[closes > 0])

        if len(log_prices) < 50:
            return result

        # Create stochastic checker
        _cfg = get_config()
        assert StochasticConfig is not None  # guarded by STOCHASTIC_VALIDATION_AVAILABLE
        stoch_config = StochasticConfig(
            survival_horizon=_cfg.stochastic.survival_horizon_bars,
            mc_paths=_cfg.stochastic.survival_mc_paths,
            hurst_max=_cfg.stochastic.hurst_max_trending,
            ou_halflife_min=_cfg.stochastic.ou_halflife_min_bars,
            ou_halflife_max=_cfg.stochastic.ou_halflife_max_bars,
            random_state=42,
        )
        assert StochasticRegimeChecker is not None  # guarded by STOCHASTIC_VALIDATION_AVAILABLE
        checker = StochasticRegimeChecker(stoch_config)

        # Analyze stochastic properties
        stoch_result = checker.analyze(
            log_prices=log_prices,
            range_high_log=np.log(range_high),
            range_low_log=np.log(range_low),
        )

        result["survival_prob"] = round(stoch_result.survival_prob, 4)
        result["hurst_exponent"] = round(stoch_result.hurst_exponent, 4)
        result["ou_theta"] = round(stoch_result.ou_theta, 6)
        result["ou_mu"] = round(stoch_result.ou_mu, 6)
        result["ou_sigma"] = round(stoch_result.ou_sigma, 6)
        if np.isfinite(stoch_result.ou_halflife):
            result["ou_halflife"] = round(stoch_result.ou_halflife, 2)

        # Compute provisional regime utility with the same fixed-geometry
        # semantics the validator/snapshot path uses, so meta-labeler
        # training and scan-time inference see one utility_score meaning.
        if range_prob is not None and trend_prob is not None:
            assert compute_governed_provisional_utility is not None
            try:
                utility_result = compute_governed_provisional_utility(
                    range_prob=range_prob,
                    trend_prob=trend_prob,
                    min_utility=0.0,
                )
                result["utility_score"] = round(utility_result.utility_score, 4)
            except UtilityCalibratorUnavailable as exc:  # type: ignore[misc]
                # Fail-closed soft: emit utility_score=None so downstream
                # consumers see "feature unavailable" rather than a silent
                # v0 substitution.
                global _utility_unavailable_warning_emitted
                if not _utility_unavailable_warning_emitted:
                    logger.warning(
                        "utility calibrator unavailable; scan utility_score will be None: %s",
                        exc,
                    )
                    _utility_unavailable_warning_emitted = True
                else:
                    logger.debug("utility calibrator unavailable: %s", exc)
                result["utility_score"] = None

    except Exception as e:
        logger.debug("Stochastic feature computation failed: %s", e)

    return result


def _candidate_notes(f: SymbolFeatures, profile: PatternProfile) -> list[str]:
    notes: list[str] = []

    def in_band(name: str, v: Optional[float]) -> bool:
        if v is None:
            return False
        lo = profile.q10.get(name)
        hi = profile.q90.get(name)
        if lo is None or hi is None:
            return False
        return (v >= lo) and (v <= hi)

    if in_band("adx_1h", f.adx_1h):
        notes.append("ADX_1h in winner band")
    if in_band("adx_15m", f.adx_15m):
        notes.append("ADX_15m in winner band")
    if in_band("bb_width", f.bb_width):
        notes.append("BB width in winner band")
    if in_band("range_size_pct", f.range_size_pct):
        notes.append("Range size in winner band")

    if f.ema_crosses_5m is not None and f.ema_crosses_5m >= 10:
        notes.append("High EMA cross count (choppy)")
    if f.vwap_crosses_5m is not None and f.vwap_crosses_5m >= 20:
        notes.append("High VWAP cross count (mean-reverting)")
    if f.funding_rate is not None and abs(f.funding_rate) <= 0.003:
        notes.append("Funding near flat")

    return notes


async def scan_top_symbols(
    *,
    client: BinanceClient,
    profile: PatternProfile,
    profile_model: ProfileModel | None = None,
    pnl_ranker: Optional[Any] = None,
    meta_labeler: Optional[Any] = None,
    feature_collector: Optional[Any] = None,
    compute_stochastic: Optional[bool] = None,  # Compute stochastic features (default: True if feature_collector)
    top_n: int = 100,
    max_concurrency: int = 8,
    min_delay_s: float = 0.05,
    mi_weights: Optional[Dict[str, float]] = None,
    scan_data_out: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    gate = RateGate(max_concurrency=max_concurrency, min_delay_s=min_delay_s)

    if profile_model is None:
        logger.warning(
            "profile_model is absent; scan will run in degraded similarity-only mode. "
            "Per-row scoring_flags will carry 'profile_model_absent'. "
            "Run neutralgrid-retrain and ensure either current.json points to a "
            "promoted artifact, or profile_model.json exists as a bootstrap "
            "candidate while current.json is absent."
        )

    # AFML: Load HMM model upfront for inline regime probability computation
    # This enables full scoring (not deferred EV)
    hmm_artifact = None
    try:
        hmm_artifact = load_hmm_model()
        if hmm_artifact is None:
            logger.warning(
                "HMM model not available. Scoring will proceed without regime "
                "probabilities. Run: neutralgrid-retrain --window 180d --top-n 60"
            )
    except Exception as e:
        logger.warning("Failed to load HMM model: %s", e)

    top = await get_top_usdtm_symbols_by_quote_volume(client, top_n=top_n)
    top_syms = [s for s, _ in top]
    qv_map = {s: qv for s, qv in top}

    # Validate HMM model exists if not already loaded
    if hmm_artifact is None and top_syms:
        try:
            await ensure_hmm_model(client=client, symbols=top_syms, force=False)
            hmm_artifact = load_hmm_model()
        except Exception as e:
            logger.warning("No trained HMM model found: %s - scoring will use deferred EV", e)

    # HMM inference limit (bars to use for regime detection)
    hmm_infer_limit = int(get_config().hmm.infer_limit)
    _scan_time = datetime.now(timezone.utc)
    _scan_ts = _scan_time.strftime("%Y%m%d_%H%M%S")

    # Determine if we should compute stochastic features
    # Default: always compute when available (survival_prob is required for EV ranking)
    should_compute_stochastic = compute_stochastic
    if should_compute_stochastic is None:
        should_compute_stochastic = STOCHASTIC_VALIDATION_AVAILABLE

    async def _gated_call(coro):
        """Execute a coroutine through the shared rate gate."""
        async with gate:
            await gate.throttle()
            return await coro

    async def fetch_one(symbol: str) -> Optional[ScanResult]:
        try:
            _kl = get_config().validation.kline_limits
            # Batch all API calls concurrently through the rate gate
            (
                kl_1h, kl_15m, kl_5m, kl_1m, prem, oi,
                _global_lsr, _top_acct_lsr, _top_pos_lsr, _taker_lsr,
                _depth, _oi_hist, _funding_hist,
            ) = await asyncio.gather(
                _gated_call(client.get_klines(symbol, "1h", _kl["1h"])),
                _gated_call(client.get_klines(symbol, "15m", _kl["15m"])),
                _gated_call(client.get_klines(symbol, "5m", _kl["5m"])),
                _gated_call(client.get_klines(symbol, "1m", _kl["1m"])),
                _gated_call(client.get_premium_index(symbol)),
                _gated_call(client.get_open_interest(symbol)),
                # Binance-derived context features
                _gated_call(client.get_global_long_short_ratio(symbol, "5m", 1)),
                _gated_call(client.get_top_long_short_account_ratio(symbol, "5m", 1)),
                _gated_call(client.get_top_long_short_position_ratio(symbol, "5m", 1)),
                _gated_call(client.get_taker_long_short_ratio(symbol, "5m", 1)),
                _gated_call(client.get_order_book(symbol, 20)),
                _gated_call(client.get_open_interest_hist(symbol, "5m", 12)),
                _gated_call(client.get_funding_rate(symbol, limit=20)),
            )

            # Cache raw kline data for downstream enrichment reuse
            if scan_data_out is not None:
                scan_data_out[symbol] = {
                    "klines_1h": kl_1h,
                    "klines_15m": kl_15m,
                    "klines_5m": kl_5m,
                    "klines_1m": kl_1m,
                    "cached_at_utc": datetime.now(timezone.utc),
                }

            funding_rate = None
            try:
                funding_rate = float(prem["lastFundingRate"])
            except Exception:
                funding_rate = None

            funding_info = None
            try:
                next_funding_ms = int(prem["nextFundingTime"])
                hours_to_next_funding = max(
                    0.0,
                    (next_funding_ms - int(datetime.now(timezone.utc).timestamp() * 1000))
                    / (1000 * 3600),
                )
                funding_info = {
                    "funding_rate_pct": float(prem["lastFundingRate"]) * 100.0,
                    "funding_interval_hours": hours_to_next_funding,
                }
            except Exception:
                funding_info = None

            open_interest = None
            try:
                open_interest = float(oi["openInterest"])
            except Exception:
                open_interest = None
            if open_interest is None:
                try:
                    if _oi_hist and len(_oi_hist) > 0:
                        _latest_oi = float(_oi_hist[-1].get("sumOpenInterest", 0))
                        if _latest_oi > 0:
                            open_interest = _latest_oi
                except Exception:
                    open_interest = None

            # ── Binance-derived context features ──────────────────────────
            import math as _math

            # Mark/index prices for basis
            _mark_price = None
            _index_price = None
            try:
                _mark_price = float(prem["markPrice"])
                _index_price = float(prem["indexPrice"])
            except Exception:
                pass

            # Global LSR
            _global_lsr_log = None
            try:
                if _global_lsr and len(_global_lsr) > 0:
                    _ratio = float(_global_lsr[0]["longShortRatio"])
                    if _ratio > 0:
                        _global_lsr_log = float(_math.log(_ratio))
            except Exception:
                pass

            # Top account LSR
            _top_account_lsr_log = None
            try:
                if _top_acct_lsr and len(_top_acct_lsr) > 0:
                    _ratio = float(_top_acct_lsr[0]["longShortRatio"])
                    if _ratio > 0:
                        _top_account_lsr_log = float(_math.log(_ratio))
            except Exception:
                pass

            # Top position LSR
            _top_position_lsr_log = None
            try:
                if _top_pos_lsr and len(_top_pos_lsr) > 0:
                    _ratio = float(_top_pos_lsr[0]["longShortRatio"])
                    if _ratio > 0:
                        _top_position_lsr_log = float(_math.log(_ratio))
            except Exception:
                pass

            # Top position vs account delta
            _top_pos_vs_acct_delta = None
            if _top_position_lsr_log is not None and _top_account_lsr_log is not None:
                _top_pos_vs_acct_delta = float(_top_position_lsr_log - _top_account_lsr_log)

            # Taker imbalance and buy/sell log
            _taker_imbalance = None
            _taker_buy_sell_log = None
            try:
                if _taker_lsr and len(_taker_lsr) > 0:
                    _buy_vol = float(_taker_lsr[0].get("buyVol", 0))
                    _sell_vol = float(_taker_lsr[0].get("sellVol", 0))
                    _total = _buy_vol + _sell_vol
                    if _total > 0:
                        _taker_imbalance = float((_buy_vol - _sell_vol) / _total)
                    _bs_ratio = float(_taker_lsr[0].get("buySellRatio", 0))
                    if _bs_ratio > 0:
                        _taker_buy_sell_log = float(_math.log(_bs_ratio))
            except Exception:
                pass

            # Basis pct
            _basis_pct = None
            if _mark_price is not None and _index_price is not None and _index_price > 0:
                _basis_pct = float((_mark_price - _index_price) / _index_price * 100.0)

            # Spread pct and order book depth
            _spread_pct = None
            _top20_bid_depth_usdt = None
            _top20_ask_depth_usdt = None
            _top20_depth_min_usdt = None
            _book_imbalance_top20 = None
            try:
                if _depth and "bids" in _depth and "asks" in _depth:
                    _bids = _depth["bids"]
                    _asks = _depth["asks"]
                    if _bids and _asks:
                        _best_bid = float(_bids[0][0])
                        _best_ask = float(_asks[0][0])
                        if _best_bid > 0:
                            _spread_pct = float((_best_ask - _best_bid) / _best_bid * 100.0)
                        _bid_depth = sum(float(b[0]) * float(b[1]) for b in _bids[:20])
                        _ask_depth = sum(float(a[0]) * float(a[1]) for a in _asks[:20])
                        _top20_bid_depth_usdt = float(_bid_depth)
                        _top20_ask_depth_usdt = float(_ask_depth)
                        _top20_depth_min_usdt = float(min(_bid_depth, _ask_depth))
                        _total_depth = _bid_depth + _ask_depth
                        if _total_depth > 0:
                            _book_imbalance_top20 = float((_bid_depth - _ask_depth) / _total_depth)
            except Exception:
                pass

            # OI notional
            _oi_notional = None
            if open_interest is not None and _mark_price is not None:
                _oi_notional = float(open_interest * _mark_price)

            # OI to volume ratio
            _oi_to_volume = None
            _qv24 = qv_map.get(symbol)
            if _oi_notional is not None and _qv24 is not None and _qv24 > 0:
                _oi_to_volume = float(_oi_notional / _qv24)

            # Funding rate z-score (from recent history)
            _funding_rate_zscore = None
            try:
                if _funding_hist and len(_funding_hist) >= 3:
                    _fr_values = [float(f["fundingRate"]) for f in _funding_hist]
                    _fr_mean = sum(_fr_values) / len(_fr_values)
                    _fr_var = sum((v - _fr_mean) ** 2 for v in _fr_values) / len(_fr_values)
                    _fr_std = _fr_var ** 0.5
                    if _fr_std > 1e-10 and funding_rate is not None:
                        _funding_rate_zscore = float((funding_rate - _fr_mean) / _fr_std)
            except Exception:
                pass

            # OI change pct (from history)
            _open_interest_change_pct = None
            try:
                if _oi_hist and len(_oi_hist) >= 2:
                    _first_oi = float(_oi_hist[0].get("sumOpenInterest", 0))
                    _last_oi = float(_oi_hist[-1].get("sumOpenInterest", 0))
                    if _first_oi > 0:
                        _open_interest_change_pct = float((_last_oi - _first_oi) / _first_oi * 100.0)
            except Exception:
                pass

            df1 = klines_to_df(kl_1h)
            df15 = klines_to_df(kl_15m)
            df5 = klines_to_df(kl_5m)
            df1m = klines_to_df(kl_1m)

            feats = compute_features(
                symbol,
                klines_1h=df1,
                klines_15m=df15,
                klines_5m=df5,
                klines_1m=df1m,
                funding_rate=funding_rate,
                funding_info=funding_info,
                open_interest=open_interest,
                quote_volume_24h=qv_map.get(symbol),
            )

            row = feats.as_dict()
            if row.get("quote_volume_24h") is None:
                try:
                    if "quote_volume" not in df15.columns:
                        raise KeyError("quote_volume")
                    qv_series = cast(pd.Series, pd.to_numeric(
                        cast(pd.Series, df15["quote_volume"]),
                        errors="coerce",
                    ))
                    qv_values = qv_series.dropna().to_numpy(dtype=float)
                    if qv_values.size:
                        qv_window = qv_values[-48:] if qv_values.size >= 48 else qv_values
                        row["quote_volume_24h"] = float(qv_window.sum())
                except Exception:
                    pass
            if row.get("open_interest") is None and open_interest is not None:
                row["open_interest"] = float(open_interest)

            # ── Inject Binance-derived context features ───────────────────
            row["top_account_lsr_log"] = _top_account_lsr_log
            row["top_position_lsr_log"] = _top_position_lsr_log
            row["top_position_vs_account_delta"] = _top_pos_vs_acct_delta
            row["global_lsr_log"] = _global_lsr_log
            row["taker_imbalance"] = _taker_imbalance
            row["taker_buy_sell_log"] = _taker_buy_sell_log
            row["basis_pct"] = _basis_pct
            row["funding_rate_zscore"] = _funding_rate_zscore
            row["open_interest_change_pct"] = _open_interest_change_pct
            row["spread_pct"] = _spread_pct
            row["top20_bid_depth_usdt"] = _top20_bid_depth_usdt
            row["top20_ask_depth_usdt"] = _top20_ask_depth_usdt
            row["top20_depth_min_usdt"] = _top20_depth_min_usdt
            row["book_imbalance_top20"] = _book_imbalance_top20
            row["oi_notional"] = _oi_notional
            row["oi_to_volume"] = _oi_to_volume
            oi_notional_value = row.get("oi_notional")
            quote_volume_value = row.get("quote_volume_24h")
            if (
                row.get("oi_to_volume") is None
                and oi_notional_value is not None
                and quote_volume_value not in (None, 0)
            ):
                try:
                    row["oi_to_volume"] = float(
                        float(oi_notional_value) / float(quote_volume_value)
                    )
                except Exception:
                    pass

            # AFML: Inline HMM regime inference - compute probabilities now, not deferred
            if hmm_artifact is not None:
                try:
                    regime_result = infer_regime(hmm_artifact, df15, infer_limit=hmm_infer_limit)
                    range_prob = regime_result.get("range_prob_agg", regime_result.get("range_prob"))
                    trend_prob = regime_result.get("trend_prob_agg", regime_result.get("trend_prob"))
                    row["range_prob"] = range_prob
                    row["trend_prob"] = trend_prob
                    row["hmm_range_prob"] = range_prob
                    row["hmm_trend_prob"] = trend_prob
                    row["hmm_range_prob_state"] = regime_result.get("range_prob")
                    row["hmm_trend_prob_state"] = regime_result.get("trend_prob")
                    row["hmm_trained_at_utc"] = regime_result.get("trained_at_utc")
                    row["hmm_artifact_version"] = regime_result.get("artifact_version")
                    row["hmm_pipeline_version"] = regime_result.get("pipeline_version")
                    _cal_prov = regime_result.get("calibration_provenance") or {}
                    if isinstance(_cal_prov, dict):
                        row["hmm_calibration_status"] = _cal_prov.get("status")
                    row["persistence_prob"] = regime_result.get("persistence_prob")
                    row["hmm_evaluated"] = True
                except Exception as hmm_err:
                    logger.debug("HMM inference failed for %s: %s", symbol, hmm_err)
                    row["hmm_evaluated"] = False

            # AFML Training: Compute stochastic features for complete training data
            # This includes survival_prob, hurst_exponent, ou_halflife, utility_score
            if should_compute_stochastic and df15 is not None:
                stochastic_features = _compute_stochastic_features(
                    df_15m=df15,
                    range_prob=row.get("range_prob"),
                    trend_prob=row.get("trend_prob"),
                    range_size_pct=row.get("range_size_pct"),
                )
                row["survival_prob"] = stochastic_features.get("survival_prob")
                row["hurst_exponent"] = stochastic_features.get("hurst_exponent")
                row["ou_halflife"] = stochastic_features.get("ou_halflife")
                row["ou_theta"] = stochastic_features.get("ou_theta")
                row["ou_mu"] = stochastic_features.get("ou_mu")
                row["ou_sigma"] = stochastic_features.get("ou_sigma")
                row["utility_score"] = stochastic_features.get("utility_score")
                row["stochastic_computed"] = True
            else:
                row["stochastic_computed"] = False

            # Micro-oscillator score: always computed for training data collection
            _vwap_c = row.get("vwap_crosses_5m")
            _ema_c = row.get("ema_crosses_5m")
            _hurst = row.get("hurst_exponent")
            if _vwap_c is not None and _ema_c is not None:
                _k_vwap, _k_ema = 25.0, 20.0
                _vwap_norm = _vwap_c / (_vwap_c + _k_vwap)
                _ema_norm = _ema_c / (_ema_c + _k_ema)
                if _hurst is not None:
                    _hurst_norm = max(0.0, 1.0 - 2.0 * abs(_hurst - 0.5))
                    row["micro_osc_score"] = round(
                        0.45 * _vwap_norm + 0.35 * _hurst_norm + 0.20 * _ema_norm, 4
                    )
                else:
                    row["micro_osc_score"] = round(
                        0.60 * _vwap_norm + 0.40 * _ema_norm, 4
                    )

            sim = profile.similarity_score(row)
            row["similarity_score"] = float(sim)
            p = profile_model.proba(row) if profile_model is not None else None

            # AFML-compliant scoring components
            # IMPORTANT: In strict AFML mode we do not impute missing inputs (e.g., default survival_prob,
            # estimated grid params). EV/meta scores are only computed when all required inputs are available.
            ev_score = None
            meta_prob = None

            # AFML C1: Track scoring flags for auditability
            scoring_flags: list[str] = []
            if profile_model is None:
                scoring_flags.append("profile_model_absent")

            # EV scoring requires grid params (profit_per_grid_pct, num_grids) which are
            # only available after enrichment. EV is computed in post-enrichment via
            # _apply_afml_post_scoring() in run_full_pipeline.py.

            if AFML_SCORING_AVAILABLE and meta_labeler is not None and meta_labeler.is_trained:
                try:
                    missing_meta = meta_labeler.get_missing_feature_names(row)
                    if missing_meta:
                        scoring_flags.append("meta_skipped_missing_features")
                    else:
                        meta_prob = float(meta_labeler.predict_proba(row))
                        # Scan-time meta inference is only provisional.  The
                        # authoritative final ``meta_prob`` is owned by the
                        # post-enrich contract once grid geometry is available.
                        row["scan_meta_prob"] = round(meta_prob, 4)
                except Exception as e:
                    meta_prob = None
                    scoring_flags.append(f"meta_inference_failed:{type(e).__name__}")

            # Compute final score
            # Scan-time path: MI-weighted score over currently available
            # signals only. Post-enrichment EV scoring happens later.
            if MI_SCORER_AVAILABLE and compute_mi_weighted_score is not None:
                mi_weights_in_use = (
                    dict(mi_weights)
                    if mi_weights is not None
                    else default_runtime_mi_weights()
                    if default_runtime_mi_weights is not None
                    else {}
                )
                _mi_features: dict[str, float | None] = {
                    "similarity_score": float(sim),
                    "profile_proba": float(100.0 * p) if p is not None else None,
                    "ev_score": float(100.0 * ev_score) if ev_score is not None else None,
                    "meta_prob": float(100.0 * meta_prob) if meta_prob is not None else None,
                }
                used_signal_count = 0
                available_weight = 0.0
                for name, weight in mi_weights_in_use.items():
                    value = _mi_features.get(name)
                    if (
                        math.isfinite(float(weight))
                        and float(weight) > 0.0
                        and value is not None
                        and math.isfinite(value)
                    ):
                        used_signal_count += 1
                        available_weight += float(weight)
                score = compute_mi_weighted_score(_mi_features, mi_weights_in_use)
                scoring_flags.append("mi_weighted")
                if mi_weights is None:
                    scoring_flags.append("mi_uniform_fallback")
                row["mi_used_signal_count"] = used_signal_count
                row["mi_available_weight"] = round(available_weight, 4)
                row["mi_weight_mode"] = "artifact" if mi_weights is not None else "uniform"
            elif p is not None:
                # Compatibility fallback when MI weights are unavailable.
                scoring_flags.append("afml_score_incomplete_fallback")
                score = float(0.65 * sim + 0.35 * (100.0 * p))
                row["mi_used_signal_count"] = 0
                row["mi_available_weight"] = 0.0
                row["mi_weight_mode"] = "unavailable"
            else:
                score = float(sim)
                row["mi_used_signal_count"] = 0
                row["mi_available_weight"] = 0.0
                row["mi_weight_mode"] = "unavailable"

            row["winner_proba"] = float(p) if p is not None else None
            # AFML C1: Store scoring flags for auditability
            row["scoring_flags"] = "; ".join(scoring_flags) if scoring_flags else ""
            notes = _candidate_notes(feats, profile)

            return ScanResult(symbol=symbol, score=score, features=row, notes=notes)
        except Exception as e:
            logger.warning("fetch_one failed for %s: %s", symbol, e, exc_info=True)
            return None

    gathered = await asyncio.gather(*[fetch_one(s) for s in top_syms], return_exceptions=True)
    results = [r for r in gathered if isinstance(r, ScanResult)]
    # Step 11A: aggregate scan observability
    logger.info("Scan gather complete: %d/%d symbols succeeded", len(results), len(top_syms))
    if not results:
        logger.warning("Zero symbols passed scan; returning empty DataFrame")
        return pd.DataFrame()

    rows = []
    for r in results:
        d = {"symbol": r.symbol, "score": r.score, "notes": "; ".join(r.notes)}
        d.update(r.features)
        # Gate 4 (Fix 1a): Preserve scan-time composite score under a distinct name
        d["scan_score"] = r.score
        # Gate 4 (Fix 2a): Rename scan-time BB-based range estimate to avoid
        # collision with the authoritative enrichment-time range_size_pct.
        if "range_size_pct" in d:
            d["scan_range_size_pct"] = d["range_size_pct"]
        # Gate 4 (Fix 3a): Rename scan-time provisional utility to avoid
        # collision with authoritative regime_utility from enrichment.
        if "utility_score" in d:
            d["scan_utility_score"] = d["utility_score"]
        # Gate 5 (Fix 4a): Generate a stable candidate_id at scan time.
        d["candidate_id"] = make_candidate_id_from_row(d, _scan_ts)
        rows.append(d)

        if feature_collector is not None and FEATURE_LOGGING_AVAILABLE and build_feature_snapshot is not None:
            try:
                snapshot = build_feature_snapshot(
                    symbol=r.symbol,
                    scanner_row=d,
                    validation_result=None,
                    timestamp=_scan_time,
                )
                feature_collector.log(snapshot)
            except Exception as log_err:
                logger.debug("Failed to log feature snapshot for %s: %s", r.symbol, log_err)

    df = pd.DataFrame(rows)
    df = df.sort_values(["score", "quote_volume_24h"], ascending=[False, False]).reset_index(drop=True)
    return df
