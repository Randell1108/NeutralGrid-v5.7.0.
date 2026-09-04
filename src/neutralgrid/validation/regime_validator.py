"""
Single orchestrator for validation - one place to understand "what is a valid candidate."

This module is the ONLY entry point for validation logic. All checks are
explicitly defined here with uniform structure.

Design principles:
- Each check returns: {name, executed, passed, metrics, reason_if_failed}
- Feature flags control which checks run
- No business logic in API layer - only here
- Clear separation: online (inference) vs offline (training)
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Any, List, cast
import numpy as np
import pandas as pd

from neutralgrid.core.config import get_config
from neutralgrid.models.artifacts import resolve_hmm_artifact_dir
from neutralgrid.indicators.technical import calc_atr, calc_adx, calc_rsi
from neutralgrid.validation.hmm_regime import load_hmm_model, infer_regime
from neutralgrid.validation.utility import (
    UtilityCalibratorUnavailable,
    compute_governed_provisional_utility,
)

# Module-level guard so the utility-unavailable warning is emitted once per
# process; subsequent symbol evaluations log at debug level.
_utility_unavailable_warning_emitted = False
from neutralgrid.validation.stochastic import StochasticRegimeChecker, StochasticConfig
from neutralgrid.core.exceptions import InferenceError, ModelLoadError, ValidationPipelineError

# AFML H1: Data quality validation
from neutralgrid.data.curator import DataCurator, DataQualityConfig

_logger = logging.getLogger(__name__)


def _load_cpcv_auto_threshold() -> Optional[dict]:
    """Load the CPCV-selected threshold artifact if it exists.

    Gate 2 alignment: validates that the artifact's ``pass_mode`` (if
    present) is compatible with the current runtime ``pass_mode`` from
    config.  If they don't match, logs a warning and returns ``None``
    so the caller falls back to config defaults rather than using
    potentially incompatible thresholds.

    Returns the artifact dict or None if unavailable / incompatible.
    """
    try:
        artifact_dir = resolve_hmm_artifact_dir()
    except Exception:
        return None
    threshold_path = artifact_dir / "cpcv_selected_threshold.json"
    if not threshold_path.exists():
        return None

    try:
        with open(threshold_path, "r", encoding="utf-8") as f:
            artifact = json.load(f)
    except Exception as e:
        _logger.warning("Failed to load CPCV threshold from %s: %s", threshold_path, e)
        return None

    # Gate 2: pass_mode compatibility check.
    # The threshold artifact may have been generated under a different
    # pass_mode (e.g. "dominance" thresholds are meaningless if runtime
    # uses "hard").  When the artifact records a pass_mode and it
    # differs from the current config, warn and fall back.
    _cfg = get_config()
    runtime_pass_mode = str(_cfg.hmm.pass_mode).lower()
    artifact_pass_mode = artifact.get("pass_mode")

    if artifact_pass_mode is not None and str(artifact_pass_mode).lower() != runtime_pass_mode:
        _logger.warning(
            "CPCV threshold artifact pass_mode=%r does not match "
            "runtime HMM_PASS_MODE=%r — ignoring artifact and "
            "falling back to config defaults to avoid incompatible "
            "thresholds (artifact: %s)",
            artifact_pass_mode,
            runtime_pass_mode,
            threshold_path,
        )
        return None

    return artifact


def compute_range_dominance(
    range_prob: float, trend_prob: float, eps: float = 1e-10,
) -> float:
    """State-count-invariant dominance metric.

    Normalizes range vs trend probability to the [0, 1] range,
    ignoring other states (e.g. bull_trend/bear_trend split).

    ``dominance = range_prob / (range_prob + trend_prob + eps)``

    A value close to 1.0 means range strongly dominates trend;
    a value close to 0.5 means they are roughly equal.
    """
    return float(range_prob / (range_prob + trend_prob + eps))


@dataclass
class CheckResult:
    """Uniform result structure for all validation checks."""

    name: str
    executed: bool
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None


@dataclass
class TimeframeResult:
    """Result of a validation block (for backward compatibility)."""

    timeframe: str
    is_valid: bool
    checks: dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None


@dataclass
class ValidationResult:
    """
    Final validation output consumed by GridCalculator.

    AFML-Compliant Design:
    - Direct regime metrics exposed at top level (not nested in checks dict)
    - All probabilities are explicit (no hidden defaults)
    - Data quality issues tracked for auditability
    """

    symbol: str
    is_valid: bool

    # For compatibility with existing enrichment logic
    tf_1h: Optional[TimeframeResult] = None
    tf_15m: Optional[TimeframeResult] = None
    tf_5m: Optional[TimeframeResult] = None

    # Derived values for grid calculation (only if valid)
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    current_price: Optional[float] = None
    atr_1m: Optional[float] = None

    # AFML: Direct regime metrics (C3 requirement)
    range_prob: Optional[float] = None
    trend_prob: Optional[float] = None
    hmm_artifact_version: Optional[str] = None
    hmm_pipeline_version: Optional[str] = None
    hmm_calibration_provenance: Optional[dict[str, Any]] = None
    volatility_tier: Optional[str] = None
    conditional_tail_risk: Optional[dict[str, Any]] = None
    conditional_tail_risk_enhanced: Optional[dict[str, Any]] = None
    tail_correction_applied: bool = False
    volatility_tier_kurtosis_aware: Optional[str] = None

    # AFML: HMM lineage fields (previously only in tf_1h.checks)
    regime_conf: Optional[float] = None
    posterior_mode: Optional[str] = None
    persistence_prob: Optional[float] = None
    hmm_trained_at_utc: Optional[str] = None
    posteriors: Optional[Any] = None

    # AFML: Stochastic regime metrics (when ENABLE_STOCHASTIC_CHECKS=True)
    survival_prob: Optional[float] = None
    hurst_exponent: Optional[float] = None
    ou_halflife: Optional[float] = None

    # AFML: Provisional regime utility (computed without final grid params)
    # Full utility is computed in GridCalculator after grid params are known
    regime_utility: Optional[float] = None

    # AFML: Data quality tracking (H1 requirement)
    data_quality_passed: bool = True
    data_quality_issues: List[str] = field(default_factory=list)

    # AFML: Scoring flags for auditability
    scoring_flags: List[str] = field(default_factory=list)


def parse_klines(klines: list) -> pd.DataFrame:
    """
    Parse raw Binance kline data into a DataFrame.

    AFML H1: Enforces UTC timezone on all timestamps for consistency.
    """
    cols = cast(
        Any,
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df = pd.DataFrame(klines, columns=cols)
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # AFML H1: Enforce UTC timezone for all timestamps
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


class RegimeValidator:
    """
    Single orchestrator for NEUTRAL grid bot validation.

    This is the ONLY place where validation logic lives. All checks are
    explicit methods that return CheckResult objects.

    Feature flags (from config.py):
    - ENABLE_STOCHASTIC_CHECKS: Enable stochastic regime checks (survival prob, Hurst, OU)
    - ENABLE_PROFILE_GATE: Enable profile gate checks (ADX/RSI bounds from winners)
    - ENABLE_PROFILE_MODEL: Enable probabilistic model checks (Gaussian discriminant)
    """

    def __init__(
        self,
        gate: Any = None,  # DEPRECATED - use ENABLE_PROFILE_GATE instead
        *,
        pattern_profile: Any = None,  # DEPRECATED - use ENABLE_PROFILE_GATE instead
        profile_model: Any = None,  # DEPRECATED - use ENABLE_PROFILE_MODEL instead
        min_profile_score: float | None = None,  # DEPRECATED
    ):
        """
        Initialize validator.

        Args:
            gate: DEPRECATED - retained for backward compatibility only
            pattern_profile: DEPRECATED - retained for backward compatibility only
            profile_model: DEPRECATED - retained for backward compatibility only
            min_profile_score: DEPRECATED - retained for backward compatibility only
        """
        # DEPRECATED: These are kept for backward compatibility but not used
        # unless feature flags are enabled
        self._gate = gate
        self._pattern_profile = pattern_profile
        self._profile_model = profile_model
        self._min_profile_score = float(min_profile_score) if min_profile_score is not None else None

        # Load HMM model artifact (core validation, always active)
        self._hmm_artifact = load_hmm_model()

        # AFML H1: Data quality curator
        self._curator = DataCurator(DataQualityConfig())

        # Hysteresis state: track consecutive pass/fail per symbol (bounded to 200 entries)
        self._hysteresis_max_size = 200
        self._hysteresis_counts: OrderedDict[str, int] = OrderedDict()  # symbol -> counter (+N = N consecutive passes, -N = fails)
        self._hysteresis_last_gate: dict[str, bool] = {}  # symbol -> last effective gate decision

    def check_hmm_regime(self, symbol: str, df: pd.DataFrame) -> CheckResult:
        """
        Check HMM-based regime classification.

        This is the PRIMARY validation check. Uses the active rolling HMM
        artifact trained on cross-symbol 15m data to classify RANGE vs TREND.

        Returns:
            CheckResult with:
            - executed: True if check ran
            - passed: True if range_prob >= threshold AND trend_prob <= threshold
            - metrics: {range_prob, trend_prob, posteriors, state_means, ...}
            - reason: Why it failed (if applicable)
        """
        # Refresh artifact handle to pick up retrained models without restart.
        latest = load_hmm_model()
        if latest is not None:
            self._hmm_artifact = latest

        if self._hmm_artifact is None:
            return CheckResult(
                name="hmm_regime",
                executed=False,
                passed=False,
                reason="hmm_model_not_trained",
            )

        _cfg = get_config()
        try:
            info = infer_regime(
                self._hmm_artifact,
                df,
                infer_limit=int(_cfg.hmm.infer_limit),
            )
        except (InferenceError, ModelLoadError, ValueError) as e:
            if isinstance(e, ValueError) and "Insufficient kline history for HMM inference" in str(e):
                # Expected for symbols with sparse/invalid kline history; keep logs concise.
                _logger.info("HMM inference skipped for %s: %s", symbol, e)
                reason = "hmm_insufficient_history"
            else:
                _logger.error("HMM inference crashed for %s: %s", symbol, e, exc_info=True)
                reason = f"hmm_inference_crashed:{type(e).__name__}"
            return CheckResult(
                name="hmm_regime",
                executed=False,
                passed=False,
                reason=reason,
            )
        except Exception as e:
            _logger.error("Unexpected error in HMM check: %s", e, exc_info=True)
            raise ValidationPipelineError(
                symbol=symbol, stage="hmm_regime",
                message=f"Unexpected: {type(e).__name__}: {e}",
            ) from e

        range_prob = float(info.get("range_prob", 0.0))
        trend_prob = float(info.get("trend_prob", 0.0))

        # Aggregated probabilities (sum of range-like / trend-like state groups)
        # These capture probability mass from "intermediate" states that single-
        # state range_prob/trend_prob miss (critical for >2-state HMMs).
        range_prob_agg = float(info.get("range_prob_agg", range_prob))
        trend_prob_agg = float(info.get("trend_prob_agg", trend_prob))

        range_prob_min = float(_cfg.hmm.range_prob_min)
        trend_prob_max = float(_cfg.hmm.trend_prob_max)

        # Hard threshold check (original behavior, uses single-state probs)
        hard_threshold_passed = (range_prob >= range_prob_min) and (trend_prob <= trend_prob_max)

        # Dominance gating (state-count-invariant, uses AGGREGATED probs)
        dominance_eps = float(_cfg.hmm.dominance_eps)
        range_dominance = compute_range_dominance(range_prob_agg, trend_prob_agg, dominance_eps)

        # Auto-threshold: load from CPCV utility sweep artifact if enabled
        range_dominance_min = float(_cfg.hmm.range_dominance_min)
        cpcv_auto = _cfg.cpcv.auto_threshold
        cpcv_artifact = None
        if cpcv_auto:
            cpcv_artifact = _load_cpcv_auto_threshold()
            if cpcv_artifact is not None:
                range_dominance_min = float(cpcv_artifact.get("dominance_threshold", range_dominance_min))

        dominance_passed = range_dominance >= range_dominance_min

        # Soft gating: compute regime_conf as a continuous [0, 1] confidence score
        soft_gating = _cfg.hmm.soft_gating
        soft_lower = float(_cfg.hmm.soft_gate_lower)
        soft_upper = float(_cfg.hmm.soft_gate_upper)

        # When CPCV auto-threshold is active, use it as the upper bound
        if cpcv_auto and cpcv_artifact is not None:
            soft_upper = float(cpcv_artifact.get("dominance_threshold", soft_upper))
            spread = float(_cfg.hmm.soft_gate_spread)
            soft_lower = soft_upper - spread

        if soft_upper > soft_lower:
            regime_conf = max(0.0, min(1.0, (range_dominance - soft_lower) / (soft_upper - soft_lower)))
        else:
            regime_conf = 1.0 if range_dominance >= soft_upper else 0.0

        # Utility-based scoring (AFML enhancement)
        min_utility = float(_cfg.utility.min_threshold)
        utility_overrides = {"min_utility": min_utility}
        if _cfg.utility.lambda_risk is not None:
            utility_overrides["lambda_risk"] = float(_cfg.utility.lambda_risk)

        # AFML C2: Compute PROVISIONAL regime utility using baseline grid assumptions
        # This is NOT the final utility score - it measures regime quality only.
        # Full utility (accounting for actual grid params) is computed in GridCalculator
        # after generate_params() determines real profit_per_grid_pct, num_grids, etc.
        utility_result = None
        try:
            utility_result = compute_governed_provisional_utility(
                range_prob=range_prob,
                trend_prob=trend_prob,
                artifact_path=None,
                lambda_risk=utility_overrides.get("lambda_risk"),
                min_utility=utility_overrides["min_utility"],
            )
        except UtilityCalibratorUnavailable as exc:
            # Fail-closed: when the calibrator artifact is missing/invalid we
            # do not silently substitute v0 defaults. The validator marks
            # utility_score as None and treats utility_passed as False so
            # any downstream `pass_mode == "utility"` or `"hybrid"` path
            # rejects rather than admits. One warning per process; subsequent
            # symbol evaluations debug-log to avoid flooding.
            global _utility_unavailable_warning_emitted
            if not _utility_unavailable_warning_emitted:
                _logger.warning(
                    "utility calibrator unavailable; regime_utility=None, "
                    "utility_passed=False: %s",
                    exc,
                )
                _utility_unavailable_warning_emitted = True
            else:
                _logger.debug("utility calibrator unavailable: %s", exc)
            utility_score: Optional[float] = None
            utility_passed = False
        else:
            # This is "regime_utility" - a provisional score for ranking/filtering
            # before grid parameters are known. Not to be confused with full utility.
            utility_score = utility_result.utility_score
            utility_passed = utility_result.utility_score >= float(utility_overrides["min_utility"])

        # Determine base pass/fail using HMM_PASS_MODE
        pass_mode = str(_cfg.hmm.pass_mode).lower()
        if soft_gating:
            # Soft gating: always pass; let regime_conf handle downstream weighting
            base_passed = True
        elif pass_mode == "dominance":
            # dominance gate + aggregated trend_prob hard veto
            base_passed = dominance_passed and (trend_prob_agg <= trend_prob_max)
        elif pass_mode == "utility":
            base_passed = utility_passed
        elif pass_mode == "hybrid":
            base_passed = hard_threshold_passed and utility_passed
        else:  # "hard"
            base_passed = hard_threshold_passed

        # Hysteresis: require N consecutive bars to confirm a regime flip
        hysteresis_bars = int(_cfg.hmm.hysteresis_bars)

        if hysteresis_bars > 1:
            # Update consecutive counter for this symbol
            count = self._hysteresis_counts.get(symbol, 0)
            last_gate = self._hysteresis_last_gate.get(symbol, base_passed)

            if base_passed:
                count = max(1, count + 1) if count > 0 else 1
            else:
                count = min(-1, count - 1) if count < 0 else -1

            self._hysteresis_counts[symbol] = count
            self._hysteresis_counts.move_to_end(symbol)  # LRU: mark as recently used

            # Evict least-recently-used entries if over max size
            if len(self._hysteresis_counts) > self._hysteresis_max_size:
                oldest = next(iter(self._hysteresis_counts))
                del self._hysteresis_counts[oldest]
                self._hysteresis_last_gate.pop(oldest, None)

            # Only flip if counter has reached threshold in the new direction
            if abs(count) >= hysteresis_bars:
                effective_passed = base_passed
                self._hysteresis_last_gate[symbol] = effective_passed
            else:
                effective_passed = last_gate  # hold previous decision

            passed = effective_passed
        else:
            passed = base_passed

        metrics = {
            "range_prob": round(range_prob, 4),
            "trend_prob": round(trend_prob, 4),
            "range_prob_agg": round(range_prob_agg, 4),
            "trend_prob_agg": round(trend_prob_agg, 4),
            "range_state": int(info.get("range_state", 0)),
            "trend_state": int(info.get("trend_state", 0)),
            "posteriors": info.get("posteriors"),
            "trained_at_utc": info.get("trained_at_utc"),
            "state_means": info.get("state_means"),
            "artifact_version": info.get("artifact_version"),
            "pipeline_version": info.get("pipeline_version"),
            "calibration_provenance": info.get("calibration_provenance", {}),
            "posterior_mode": info.get("posterior_mode"),
            "volatility_tier": info.get("volatility_tier"),
            "volatility_tier_probs": info.get("volatility_tier_probs"),
            "conditional_tail_risk": info.get("conditional_tail_risk"),
            "persistence_prob": info.get("persistence_prob"),
            "tail_correction_applied": info.get("tail_correction_applied", False),
            "conditional_tail_risk_enhanced": info.get("conditional_tail_risk_enhanced", {}),
            "volatility_tier_kurtosis_aware": info.get("volatility_tier_kurtosis_aware"),
            "thresholds": {
                "range_prob_min": range_prob_min,
                "trend_prob_max": trend_prob_max,
            },
            # Hysteresis state
            "hysteresis_bars": hysteresis_bars,
            "hysteresis_count": self._hysteresis_counts.get(symbol, 0),
            "hysteresis_effective_passed": passed,
            # Dominance gating (uses aggregated probs)
            "range_dominance": round(range_dominance, 4),
            "range_dominance_min": range_dominance_min,
            "dominance_passed": dominance_passed,
            # Soft gating confidence score [0, 1]
            "regime_conf": round(regime_conf, 4),
            "soft_gating": soft_gating,
            "soft_gate_lower": soft_lower,
            "soft_gate_upper": soft_upper,
            "cpcv_auto_threshold": cpcv_auto,
            # AFML utility scoring additions. When the utility calibrator is
            # unavailable, all three utility-derived fields are None so
            # downstream consumers see a transparent feature-missing signal.
            "utility_score": round(utility_score, 4) if utility_score is not None else None,
            "utility_passed": utility_passed,
            "hard_threshold_passed": hard_threshold_passed,
            "pass_mode": pass_mode,
            "expected_grid_return": (
                round(utility_result.expected_grid_return, 4)
                if utility_result is not None
                else None
            ),
            "expected_drawdown": (
                round(utility_result.expected_drawdown, 4)
                if utility_result is not None
                else None
            ),
        }

        reason = None
        if not passed:
            reason = (
                f"hmm_regime_fail(mode={pass_mode}, "
                f"rng_agg={metrics['range_prob_agg']}, trnd_agg={metrics['trend_prob_agg']}, "
                f"dominance={metrics['range_dominance']})"
            )

        return CheckResult(
            name="hmm_regime",
            executed=True,
            passed=passed,
            metrics=metrics,
            reason=reason,
        )

    def check_range_quality(self, df_15m: pd.DataFrame, range_prob: float = 0.5) -> CheckResult:
        """
        Extract range bounds from 15M data for grid parameterization.

        AFML-Compliant Range Estimation:
        - Uses configurable lookback (48 bars for 15M) instead of 100
        - Uses quantiles (95th/5th) instead of raw min/max (robust to wick spikes)
        - Adds ATR buffer scaled by HMM range_prob

        This is NOT a gating check - it's a data extraction step that computes
        the high/low bounds for grid placement.

        Args:
            df_15m: 15-minute OHLCV DataFrame
            range_prob: HMM posterior probability of range regime (used for ATR buffer scaling)

        Returns:
            CheckResult with:
            - executed: True if check ran
            - passed: True (always passes - this is not a gate)
            - metrics: {range_high, range_low, range_size_pct, current_price, ...}
        """
        if len(df_15m) < 50:
            return CheckResult(
                name="range_quality",
                executed=False,
                passed=False,
                reason="insufficient_15m_data",
            )

        highs_15 = np.asarray(df_15m["high"])
        lows_15 = np.asarray(df_15m["low"])
        closes_15 = np.asarray(df_15m["close"])

        # AFML improvement: Use configurable lookback (48 bars for 15M)
        _cfg = get_config()
        lookback_config = int(_cfg.validation.range_lookback_bars_15m)
        lookback = min(lookback_config, len(df_15m))

        # Get quantile settings from config
        quantile_low = float(_cfg.validation.range_quantile_low)
        quantile_high = float(_cfg.validation.range_quantile_high)

        # AFML improvement: Use quantiles instead of raw min/max (robust to wick spikes)
        range_high_raw = float(np.nanpercentile(highs_15[-lookback:], quantile_high * 100))
        range_low_raw = float(np.nanpercentile(lows_15[-lookback:], quantile_low * 100))

        # Compute ATR for buffer calculation
        atr_period = int(_cfg.indicators.atr_period)
        atr_values = calc_atr(
            np.asarray(df_15m["high"]),
            np.asarray(df_15m["low"]),
            np.asarray(df_15m["close"]),
            atr_period,
        )

        # Average ATR over recent bars
        atr_lookback = min(14, len(atr_values))
        atr_15m = float(np.nanmean(atr_values[-atr_lookback:])) if atr_lookback > 0 else 0.0

        # AFML improvement: ATR buffer based on HMM range_prob
        # k = 1.5 - 0.5 * range_prob (k in [1.0, 1.5])
        # Higher range_prob -> smaller buffer (more confident in range)
        # Lower range_prob -> larger buffer (less confident, widen range)
        k = 1.5 - 0.5 * max(0.0, min(1.0, range_prob))
        range_high = range_high_raw + k * atr_15m
        range_low = range_low_raw - k * atr_15m

        current_price = float(closes_15[-1])

        # Ensure range makes sense (high > low)
        if range_low >= range_high:
            range_high = range_high_raw
            range_low = range_low_raw

        metrics = {
            "range_high": round(range_high, 8),
            "range_low": round(range_low, 8),
            "range_high_raw": round(range_high_raw, 8),
            "range_low_raw": round(range_low_raw, 8),
            "range_size_pct": round(((range_high - range_low) / max(range_low, 1e-12)) * 100, 4),
            "current_price": round(current_price, 8),
            "lookback_bars": lookback,
            "atr_15m": round(atr_15m, 8),
            "atr_buffer_k": round(k, 4),
            "range_prob_input": round(range_prob, 4),
            "quantiles": {
                "low": quantile_low,
                "high": quantile_high,
            },
        }

        return CheckResult(
            name="range_quality",
            executed=True,
            passed=True,  # Always passes - this is not a gate
            metrics=metrics,
        )

    def check_volatility_bounds(self, df_1h: pd.DataFrame, df_15m: pd.DataFrame) -> CheckResult:
        """
        Check volatility bounds using ADX/RSI thresholds.

        ⚠️  FEATURE FLAG: Only runs if ENABLE_PROFILE_GATE is True in config

        This check uses learned ADX/RSI bounds from historical winners to
        filter out candidates with excessive trend or RSI extremes.

        ProfileGate bounds:
        - adx_1h <= gate.adx_1h_max (low trend strength on 1H)
        - adx_15m <= gate.adx_15m_max (low trend strength on 15M)
        - rsi_15m in [gate.rsi_15m_low, gate.rsi_15m_high] (not overbought/oversold)

        Returns:
            CheckResult with:
            - executed: False if feature flag disabled
            - passed: True if ADX/RSI within bounds
            - metrics: {adx_1h, adx_15m, rsi_15m, bounds}
        """
        _cfg = get_config()
        enabled = _cfg.validation.enable_profile_gate

        if not enabled:
            return CheckResult(
                name="volatility_bounds",
                executed=False,
                passed=True,  # Don't block validation if disabled
                reason="feature_disabled",
            )

        # If enabled but no gate was loaded, fail
        if self._gate is None:
            return CheckResult(
                name="volatility_bounds",
                executed=False,
                passed=False,
                reason="profile_gate_not_loaded",
            )

        # Validate data sufficiency
        adx_period = int(_cfg.indicators.adx_period)
        rsi_period = int(_cfg.indicators.rsi_period)

        if len(df_1h) < adx_period + 5 or len(df_15m) < max(adx_period, rsi_period) + 5:
            return CheckResult(
                name="volatility_bounds",
                executed=False,
                passed=False,
                reason="insufficient_data_for_indicators",
            )

        try:
            # Compute ADX on 1H timeframe
            adx_1h_arr, _, _ = calc_adx(
                np.asarray(df_1h["high"]),
                np.asarray(df_1h["low"]),
                np.asarray(df_1h["close"]),
                adx_period,
            )
            adx_1h = float(np.nanmean(adx_1h_arr[-5:])) if len(adx_1h_arr) >= 5 else float(adx_1h_arr[-1])

            # Compute ADX on 15M timeframe
            adx_15m_arr, _, _ = calc_adx(
                np.asarray(df_15m["high"]),
                np.asarray(df_15m["low"]),
                np.asarray(df_15m["close"]),
                adx_period,
            )
            adx_15m = float(np.nanmean(adx_15m_arr[-5:])) if len(adx_15m_arr) >= 5 else float(adx_15m_arr[-1])

            # Compute RSI on 15M timeframe
            rsi_15m_arr = calc_rsi(np.asarray(df_15m["close"]), rsi_period)
            rsi_15m = float(np.nanmean(rsi_15m_arr[-5:])) if len(rsi_15m_arr) >= 5 else float(rsi_15m_arr[-1])

        except Exception as e:
            return CheckResult(
                name="volatility_bounds",
                executed=True,
                passed=False,
                reason=f"indicator_calculation_failed:{type(e).__name__}",
            )

        # Check bounds against ProfileGate thresholds
        adx_1h_ok = adx_1h <= self._gate.adx_1h_max
        adx_15m_ok = adx_15m <= self._gate.adx_15m_max
        rsi_15m_ok = self._gate.rsi_15m_low <= rsi_15m <= self._gate.rsi_15m_high

        passed = adx_1h_ok and adx_15m_ok and rsi_15m_ok

        metrics = {
            "adx_1h": round(adx_1h, 2),
            "adx_15m": round(adx_15m, 2),
            "rsi_15m": round(rsi_15m, 2),
            "bounds": {
                "adx_1h_max": self._gate.adx_1h_max,
                "adx_15m_max": self._gate.adx_15m_max,
                "rsi_15m_low": self._gate.rsi_15m_low,
                "rsi_15m_high": self._gate.rsi_15m_high,
            },
            "checks": {
                "adx_1h_ok": adx_1h_ok,
                "adx_15m_ok": adx_15m_ok,
                "rsi_15m_ok": rsi_15m_ok,
            },
        }

        reason = None
        if not passed:
            failures = []
            if not adx_1h_ok:
                failures.append(f"adx_1h={adx_1h:.1f}>{self._gate.adx_1h_max}")
            if not adx_15m_ok:
                failures.append(f"adx_15m={adx_15m:.1f}>{self._gate.adx_15m_max}")
            if not rsi_15m_ok:
                failures.append(f"rsi_15m={rsi_15m:.1f} not in [{self._gate.rsi_15m_low},{self._gate.rsi_15m_high}]")
            reason = "volatility_bounds_fail(" + ", ".join(failures) + ")"

        return CheckResult(
            name="volatility_bounds",
            executed=True,
            passed=passed,
            metrics=metrics,
            reason=reason,
        )

    def check_stochastic_regime(self, df_15m: pd.DataFrame, range_high: float, range_low: float) -> CheckResult:
        """
        Check stochastic regime properties (survival prob, Hurst, OU half-life).

        ⚠️  FEATURE FLAG: Only runs if ENABLE_STOCHASTIC_CHECKS is True in config

        This check models log-price as an Ito diffusion and estimates:
        - Survival probability within range over horizon (Monte Carlo)
        - Hurst exponent (trending vs mean-reverting)
        - OU half-life (mean-reversion speed)

        Thresholds (from config):
        - hurst <= HURST_MAX_TRENDING (0.55): Not in trending regime
        - ou_halflife in [OU_HALFLIFE_MIN_BARS, OU_HALFLIFE_MAX_BARS]: Reasonable mean-reversion
        - survival_prob >= SURVIVAL_MIN: Configurable containment gate

        Returns:
            CheckResult with:
            - executed: False if feature flag disabled
            - passed: True if survival prob, Hurst, and OU half-life within bounds
            - metrics: {survival_prob, hurst, ou_halflife, ou_theta, ou_mu, ou_sigma}
        """
        _cfg = get_config()
        enabled = _cfg.stochastic.enable

        if not enabled:
            return CheckResult(
                name="stochastic_regime",
                executed=False,
                passed=True,  # Don't block validation if disabled
                reason="feature_disabled",
            )

        # Validate data sufficiency
        min_bars = 50
        if len(df_15m) < min_bars:
            return CheckResult(
                name="stochastic_regime",
                executed=False,
                passed=False,
                reason=f"insufficient_data({len(df_15m)}<{min_bars})",
            )

        # Get config parameters
        survival_horizon = int(_cfg.stochastic.survival_horizon_bars)
        mc_paths = int(_cfg.stochastic.survival_mc_paths)
        hurst_max = float(_cfg.stochastic.hurst_max_trending)
        ou_halflife_min = int(_cfg.stochastic.ou_halflife_min_bars)
        ou_halflife_max = int(_cfg.stochastic.ou_halflife_max_bars)

        try:
            # Create stochastic checker with config parameters
            stoch_config = StochasticConfig(
                survival_horizon=survival_horizon,
                mc_paths=mc_paths,
                hurst_max=hurst_max,
                ou_halflife_min=ou_halflife_min,
                ou_halflife_max=ou_halflife_max,
                random_state=int(_cfg.hmm.random_state),
            )
            checker = StochasticRegimeChecker(stoch_config)

            # Get log prices
            closes = np.asarray(df_15m["close"])
            log_prices = np.log(closes[closes > 0])

            if len(log_prices) < min_bars:
                return CheckResult(
                    name="stochastic_regime",
                    executed=False,
                    passed=False,
                    reason="insufficient_valid_prices",
                )

            # Analyze stochastic properties
            result = checker.analyze(
                log_prices=log_prices,
                range_high_log=np.log(range_high),
                range_low_log=np.log(range_low),
            )

        except Exception as e:
            return CheckResult(
                name="stochastic_regime",
                executed=True,
                passed=False,
                reason=f"stochastic_analysis_failed:{type(e).__name__}",
            )

        # ERR-092: the half-life window is advisory unless configured hard.
        # It is still computed (de-trended fit) and surfaced in metrics; only
        # its gating authority is controlled by ou_halflife_gate_hard.
        halflife_gate_hard = bool(
            getattr(_cfg.stochastic, "ou_halflife_gate_hard", True)
        )

        # Build metrics
        metrics = {
            "survival_prob": round(result.survival_prob, 4),
            "hurst_exponent": round(result.hurst_exponent, 4),
            "hurst_significant": result.hurst_significant,
            "hurst_stability": result.hurst_stability,
            "hurst_stderr": result.hurst_stderr,
            "ou_halflife": round(result.ou_halflife, 2) if np.isfinite(result.ou_halflife) else None,
            "ou_theta": round(result.ou_theta, 6),
            "ou_mu": round(result.ou_mu, 6),
            "ou_sigma": round(result.ou_sigma, 6),
            "thresholds": {
                "hurst_max": hurst_max,
                "ou_halflife_min": ou_halflife_min,
                "ou_halflife_max": ou_halflife_max,
            },
            "checks": {
                "hurst_ok": result.hurst_ok,
                "halflife_ok": result.halflife_ok,
                "halflife_gate_hard": halflife_gate_hard,
            },
        }

        passed = result.hurst_ok and (result.halflife_ok or not halflife_gate_hard)

        if not result.halflife_ok and not halflife_gate_hard:
            metrics["halflife_advisory"] = (
                f"halflife={result.ou_halflife:.1f} outside "
                f"[{ou_halflife_min},{ou_halflife_max}] (advisory, not gating)"
                if np.isfinite(result.ou_halflife)
                else "halflife=inf (advisory, not gating)"
            )

        reason = None
        if not passed:
            failures = []
            if not result.hurst_ok:
                failures.append(f"hurst={result.hurst_exponent:.3f}>{hurst_max}")
            if halflife_gate_hard and not result.halflife_ok:
                if np.isfinite(result.ou_halflife):
                    failures.append(f"halflife={result.ou_halflife:.1f} not in [{ou_halflife_min},{ou_halflife_max}]")
                else:
                    failures.append("halflife=inf (no mean-reversion)")
            reason = "stochastic_regime_fail(" + ", ".join(failures) + ")"

        return CheckResult(
            name="stochastic_regime",
            executed=True,
            passed=passed,
            metrics=metrics,
            reason=reason,
        )

    def check_data_quality(
        self,
        df: pd.DataFrame,
        timeframe: str,
    ) -> CheckResult:
        """
        AFML H1: Check data quality before feature extraction.

        Uses DataCurator to validate:
        - Missing bars
        - Stale data
        - NaN/Inf values
        - Outliers (warnings only)

        Args:
            df: Parsed kline DataFrame
            timeframe: Timeframe string ("1h", "15m", etc.)

        Returns:
            CheckResult with quality metrics
        """
        result = self._curator.validate_ohlcv(df, timeframe=timeframe)

        metrics = {
            "checks": result.checks,
            "metrics": result.metrics,
            "issues": result.issues,
            "warnings": result.warnings,
        }

        reason = None
        if not result.passed:
            reason = f"data_quality_fail:{'; '.join(result.issues)}"

        return CheckResult(
            name="data_quality",
            executed=True,
            passed=result.passed,
            metrics=metrics,
            reason=reason,
        )

    def validate(self, market_data: dict) -> ValidationResult:
        """
        Orchestrate all validation checks.

        This is the SINGLE entry point for validation. It runs all enabled
        checks in sequence and returns a unified ValidationResult.

        Current pipeline:
        0. Data quality check (ALWAYS ACTIVE - NEW for AFML H1)
        1. HMM regime check (ALWAYS ACTIVE - primary gate)
        2. Range quality extraction (ALWAYS ACTIVE - not a gate)
        3. Volatility bounds check (OPTIONAL - controlled by ENABLE_PROFILE_GATE)
        4. Stochastic regime check (OPTIONAL - controlled by ENABLE_STOCHASTIC_CHECKS)

        Args:
            market_data: Dict with keys: symbol, klines (dict of timeframe -> kline data)

        Returns:
            ValidationResult with is_valid=True only if all enabled checks pass
        """
        _cfg = get_config()
        symbol = str(market_data.get("symbol", "UNKNOWN"))
        klines = market_data.get("klines", {}) or {}

        # Track data quality issues for the ValidationResult
        data_quality_passed = True
        data_quality_issues: List[str] = []

        # =====================================================================
        # CHECK 0: Data Quality (AFML H1 - PRE-VALIDATION)
        # =====================================================================

        if "1h" not in klines:
            return ValidationResult(
                symbol=symbol,
                is_valid=False,
                tf_1h=TimeframeResult("1h", False, reason="missing_1h_data"),
            )

        df_1h = parse_klines(klines["1h"])

        # AFML H1: Validate 1h data quality
        dq_1h = self.check_data_quality(df_1h, "1h")
        if not dq_1h.passed:
            data_quality_passed = False
            data_quality_issues.extend(dq_1h.metrics.get("issues", []))
            if _cfg.validation.data_quality_strict:
                return ValidationResult(
                    symbol=symbol,
                    is_valid=False,
                    tf_1h=TimeframeResult(
                        "1h", False, checks=dq_1h.metrics, reason=dq_1h.reason,
                    ),
                    data_quality_passed=False,
                    data_quality_issues=data_quality_issues,
                )

        # =====================================================================
        # CHECK 1a: 15m data availability (required for HMM which now uses 15m)
        # =====================================================================

        if "15m" not in klines:
            return ValidationResult(
                symbol=symbol,
                is_valid=False,
                tf_1h=TimeframeResult("1h", True),
                tf_15m=TimeframeResult("15m", False, reason="missing_15m_data"),
            )

        df_15m = parse_klines(klines["15m"])

        # AFML H1: Validate 15m data quality
        dq_15m = self.check_data_quality(df_15m, "15m")
        if not dq_15m.passed:
            data_quality_passed = False
            data_quality_issues.extend(dq_15m.metrics.get("issues", []))
            if _cfg.validation.data_quality_strict:
                return ValidationResult(
                    symbol=symbol,
                    is_valid=False,
                    tf_1h=TimeframeResult("1h", True),
                    tf_15m=TimeframeResult(
                        "15m", False, checks=dq_15m.metrics, reason=dq_15m.reason,
                    ),
                    data_quality_passed=False,
                    data_quality_issues=data_quality_issues,
                )

        # =====================================================================
        # CHECK 1b: HMM Regime (PRIMARY GATE) — fed with 15m data
        # =====================================================================

        hmm_check = self.check_hmm_regime(symbol, df_15m)
        hmm_metrics = hmm_check.metrics
        hmm_result_fields: dict[str, Any] = {
            "range_prob": hmm_metrics.get("range_prob_agg", hmm_metrics.get("range_prob")),
            "trend_prob": hmm_metrics.get("trend_prob_agg", hmm_metrics.get("trend_prob")),
            "hmm_artifact_version": hmm_metrics.get("artifact_version"),
            "hmm_pipeline_version": hmm_metrics.get("pipeline_version"),
            "hmm_calibration_provenance": hmm_metrics.get("calibration_provenance"),
            "volatility_tier": hmm_metrics.get("volatility_tier"),
            "conditional_tail_risk": hmm_metrics.get("conditional_tail_risk"),
            "conditional_tail_risk_enhanced": hmm_metrics.get("conditional_tail_risk_enhanced"),
            "tail_correction_applied": bool(hmm_metrics.get("tail_correction_applied", False)),
            "volatility_tier_kurtosis_aware": hmm_metrics.get("volatility_tier_kurtosis_aware"),
            "regime_conf": hmm_metrics.get("regime_conf"),
            "posterior_mode": hmm_metrics.get("posterior_mode"),
            "persistence_prob": hmm_metrics.get("persistence_prob"),
            "hmm_trained_at_utc": hmm_metrics.get("trained_at_utc"),
            "posteriors": hmm_metrics.get("posteriors"),
            "regime_utility": hmm_metrics.get("utility_score"),
        }

        if not hmm_check.passed:
            return ValidationResult(
                symbol=symbol,
                is_valid=False,
                tf_1h=TimeframeResult(
                    "1h",
                    False,
                    checks=hmm_check.metrics,
                    reason=hmm_check.reason,
                ),
                **hmm_result_fields,
                data_quality_passed=data_quality_passed,
                data_quality_issues=data_quality_issues,
            )

        # =====================================================================
        # CHECK 2: Range Quality (DATA EXTRACTION - NOT A GATE)
        # =====================================================================

        # Pass aggregated range_prob from HMM check for ATR buffer scaling
        range_prob = hmm_check.metrics.get("range_prob_agg", hmm_check.metrics.get("range_prob", 0.5))
        range_check = self.check_range_quality(df_15m, range_prob=range_prob)

        if not range_check.passed:
            return ValidationResult(
                symbol=symbol,
                is_valid=False,
                tf_1h=TimeframeResult("1h", True, checks=hmm_check.metrics),
                tf_15m=TimeframeResult("15m", False, checks=range_check.metrics, reason=range_check.reason),
                **hmm_result_fields,
                data_quality_passed=data_quality_passed,
                data_quality_issues=data_quality_issues,
            )

        range_high = range_check.metrics["range_high"]
        range_low = range_check.metrics["range_low"]
        current_price = range_check.metrics["current_price"]

        # =====================================================================
        # CHECK 3: Volatility Bounds (OPTIONAL - FEATURE FLAG)
        # =====================================================================

        vol_check = self.check_volatility_bounds(df_1h, df_15m)
        if vol_check.executed and not vol_check.passed:
            return ValidationResult(
                symbol=symbol,
                is_valid=False,
                tf_1h=TimeframeResult("1h", True, checks=hmm_check.metrics),
                tf_15m=TimeframeResult("15m", False, checks=range_check.metrics, reason=vol_check.reason),
                range_high=range_high,
                range_low=range_low,
                current_price=current_price,
                **hmm_result_fields,
                data_quality_passed=data_quality_passed,
                data_quality_issues=data_quality_issues,
            )

        # =====================================================================
        # CHECK 4: Stochastic Regime (OPTIONAL - FEATURE FLAG)
        # =====================================================================

        stoch_check = self.check_stochastic_regime(df_15m, range_high, range_low)
        if stoch_check.executed and not stoch_check.passed:
            _stoch_metrics = stoch_check.metrics
            return ValidationResult(
                symbol=symbol,
                is_valid=False,
                tf_1h=TimeframeResult("1h", True, checks=hmm_check.metrics),
                tf_15m=TimeframeResult(
                    "15m",
                    False,
                    checks={**range_check.metrics, **_stoch_metrics},
                    reason=stoch_check.reason,
                ),
                range_high=range_high,
                range_low=range_low,
                current_price=current_price,
                **hmm_result_fields,
                survival_prob=_stoch_metrics.get("survival_prob"),
                hurst_exponent=_stoch_metrics.get("hurst_exponent"),
                ou_halflife=_stoch_metrics.get("ou_halflife"),
                data_quality_passed=data_quality_passed,
                data_quality_issues=data_quality_issues,
            )

        # =====================================================================
        # EXTRACT ATR FOR GRID SPACING (NON-GATING)
        # =====================================================================

        atr_1m = None
        if "1m" in klines:
            df_1m = parse_klines(klines["1m"])
            if len(df_1m) >= int(_cfg.indicators.atr_period):
                atr_values = calc_atr(
                    np.asarray(df_1m["high"]),
                    np.asarray(df_1m["low"]),
                    np.asarray(df_1m["close"]),
                    int(_cfg.indicators.atr_period),
                )
                if len(atr_values) > 0:
                    atr_1m = float(np.nanmean(atr_values[-5:]))

        # =====================================================================
        # ALL CHECKS PASSED - RETURN VALID
        # =====================================================================

        # AFML: Extract stochastic metrics if check was executed
        _survival_prob = None
        _hurst_exponent = None
        _ou_halflife = None
        if stoch_check.executed:
            _survival_prob = stoch_check.metrics.get("survival_prob")
            _hurst_exponent = stoch_check.metrics.get("hurst_exponent")
            _ou_halflife = stoch_check.metrics.get("ou_halflife")

        return ValidationResult(
            symbol=symbol,
            is_valid=True,
            tf_1h=TimeframeResult("1h", True, checks=hmm_check.metrics),
            tf_15m=TimeframeResult("15m", True, checks=range_check.metrics),
            tf_5m=None,  # 5m check not implemented
            range_high=range_high,
            range_low=range_low,
            current_price=current_price,
            atr_1m=atr_1m,
            # AFML: Direct regime metrics (C3 requirement)
            **hmm_result_fields,
            survival_prob=_survival_prob,
            hurst_exponent=_hurst_exponent,
            ou_halflife=_ou_halflife,
            # AFML H1: Data quality tracking
            data_quality_passed=data_quality_passed,
            data_quality_issues=data_quality_issues,
        )
