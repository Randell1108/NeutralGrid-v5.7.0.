from __future__ import annotations

import asyncio
import copy
import logging
import math
from dataclasses import dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, cast
from pathlib import Path

import pandas as pd
import numpy as np

from neutralgrid.core.config import get_config
from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.indicators.technical import calc_atr
from neutralgrid.validation.regime_validator import RegimeValidator, parse_klines
from neutralgrid.validation.profile_gate import ProfileGate
from neutralgrid.validation.hmm_regime import ensure_hmm_model
from neutralgrid.grid.calculator import GridCalculator

# AFML C4: Microstructure gating
from neutralgrid.validation.microstructure import (
    MicrostructureEstimator,
    MicrostructureConfig,
)
from neutralgrid.validation.microstructure_hard_gate import (
    MicrostructureHardGate,
)
from neutralgrid.scanner.tradable_oscillation import TradableOscillationScorer
from neutralgrid.scanner.two_stage_selector import TwoStageSelector
from neutralgrid.grid.position_sizer import PositionSizer
from neutralgrid.scanner.pnl_ranker import PnLRanker, RankingConfig
from neutralgrid.scanner.empirical_profile_v20260302 import (
    load_empirical_profile_cached,
    generalized_kelly_details,
)

# Phase 2: Adaptive microstructure gate
try:
    from neutralgrid.scanner.adaptive_microstructure_gate_v20260311 import (
        AdaptiveMicrostructureGate,
    )
    ADAPTIVE_GATE_AVAILABLE = True
except ImportError:
    ADAPTIVE_GATE_AVAILABLE = False
    AdaptiveMicrostructureGate = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


@dataclass
class _RegimeData:
    """Intermediate result from regime fetch and validation (Stages 1-6)."""
    market_data: dict
    vres: Any
    hmm_range_prob: Any = None
    hmm_trend_prob: Any = None
    survival_prob: Any = None
    hurst_exponent: Any = None
    ou_halflife: Any = None
    ou_halflife_feature: Any = None
    ou_halflife_feature_reason: Any = None
    regime_utility: Any = None
    volatility_tier: Any = None
    conditional_tail_risk: dict = field(default_factory=dict)
    conditional_tail_risk_enhanced: dict = field(default_factory=dict)
    tail_correction_applied: bool = False
    volatility_tier_ka: Any = None
    regime_conf: Any = None
    posterior_mode: Any = None
    hmm_posteriors: Any = None
    hmm_trained_at: Any = None
    hmm_artifact_version: Any = None
    hmm_pipeline_version: Any = None
    hmm_calibration_provenance: dict = field(default_factory=dict)
    funding_rate_decimal: Any = None
    persistence_prob: Any = None
    # Tier 4 context features
    long_short_ratio: float | None = None
    funding_rate_zscore: float | None = None
    open_interest_change_pct: float | None = None
    # Micro-oscillator archetype
    micro_osc_score: float | None = None
    micro_osc_bypass: bool = False
    # OU parameters propagated from scan-time stochastic analysis
    ou_theta: float | None = None
    ou_mu: float | None = None
    ou_sigma: float | None = None
    # Discovery-mode audit: non-gating geometry extraction after a regime reject
    discovery_geometry_filled: bool = False
    discovery_geometry_reason: str | None = None


@dataclass
class _MicroData:
    """Intermediate result from microstructure cost estimation (Stage 9)."""
    ms_payload: dict = field(default_factory=dict)
    ms_costs: Any = None
    adaptive_value: Any = None
    has_book: bool = False
    vol_proxy: float = 0.0


@dataclass
class _GridData:
    """Intermediate result from grid generation (Stage 10)."""
    g: Any = None
    edge_info: dict = field(default_factory=dict)
    range_size_pct: Any = None


def _grid_params_dict(g: Any) -> dict:
    """Extract grid parameters from GridParams for return dicts."""
    if g is None:
        return {}
    return {
        "grid_lower": getattr(g, "grid_lower", None),
        "grid_upper": getattr(g, "grid_upper", None),
        "num_grids": getattr(g, "num_grids", None),
        "grid_spacing_pct": getattr(g, "grid_spacing_pct", None),
        "profit_per_grid_pct": getattr(g, "profit_per_grid_pct", None),
        "profit_per_grid_min_pct": getattr(g, "profit_per_grid_min_pct", None),
        "profit_per_grid_max_pct": getattr(g, "profit_per_grid_max_pct", None),
        "leverage": getattr(g, "leverage", None),
        "stop_loss_pct": getattr(g, "stop_loss_pct", None),
    }


def _apply_ou_params_from_scan_row(rd: _RegimeData, scan_row: pd.Series) -> None:
    """Copy scan-time OU parameters into regime state when present."""
    for field in ("ou_theta", "ou_mu", "ou_sigma"):
        value = scan_row.get(field)
        if value is None or pd.isna(value):
            continue
        try:
            setattr(rd, field, float(value))
        except (TypeError, ValueError):
            continue


def _coerce_klines_frame(klines: Any) -> pd.DataFrame | None:
    """Convert Binance kline payloads to a DataFrame when possible."""
    if isinstance(klines, pd.DataFrame):
        return klines
    if isinstance(klines, list) and klines:
        try:
            return parse_klines(klines)
        except Exception:
            return None
    return None


def _fill_discovery_geometry_from_market_data(
    vres: Any,
    market_data: dict[str, Any],
    validator: RegimeValidator,
) -> tuple[bool, str]:
    """Run the validator's non-gating range extractor after a regime reject.

    RegimeValidator.validate() returns immediately when the HMM gate rejects,
    before running check_range_quality().  In discovery mode we still need grid
    geometry for offline triage, so this reuses the same non-gating extractor
    and the same 1m ATR spacing source used by the normal valid path.
    """
    klines = market_data.get("klines") or {}
    if not isinstance(klines, dict):
        return False, "missing_klines"

    klines_15m = klines.get("15m")
    if klines_15m is None:
        klines_15m = market_data.get("klines_15m")
    df_15m = _coerce_klines_frame(klines_15m)
    if df_15m is None:
        return False, "missing_15m_data"

    range_prob_raw = getattr(vres, "range_prob", None)
    try:
        range_prob = float(range_prob_raw) if range_prob_raw is not None else 0.5
    except (TypeError, ValueError):
        range_prob = 0.5

    try:
        range_check = validator.check_range_quality(df_15m, range_prob=range_prob)
    except Exception as exc:
        return False, f"range_quality_error:{type(exc).__name__}"

    if not getattr(range_check, "passed", False):
        return False, str(getattr(range_check, "reason", None) or "range_quality_failed")

    metrics = getattr(range_check, "metrics", {}) or {}
    try:
        vres.range_high = float(metrics["range_high"])
        vres.range_low = float(metrics["range_low"])
        vres.current_price = float(metrics["current_price"])
    except (KeyError, TypeError, ValueError):
        return False, "range_quality_missing_metrics"

    klines_1m = klines.get("1m")
    if klines_1m is None:
        klines_1m = market_data.get("klines_1m")
    df_1m = _coerce_klines_frame(klines_1m)
    if df_1m is None:
        return True, "range_quality_after_regime_reject;missing_1m_data"

    try:
        atr_period = int(get_config().indicators.atr_period)
        if len(df_1m) >= atr_period:
            atr_values = calc_atr(
                np.asarray(df_1m["high"]),
                np.asarray(df_1m["low"]),
                np.asarray(df_1m["close"]),
                atr_period,
            )
            if len(atr_values) > 0:
                atr_1m = float(np.nanmean(atr_values[-5:]))
                if np.isfinite(atr_1m):
                    vres.atr_1m = atr_1m
    except Exception as exc:
        return True, f"range_quality_after_regime_reject;atr_error:{type(exc).__name__}"

    if getattr(vres, "atr_1m", None) is None:
        return True, "range_quality_after_regime_reject;missing_1m_atr"
    return True, "range_quality_after_regime_reject"


def _has_grid_generation_geometry(vres: Any) -> bool:
    """Return True when grid generation has the minimum non-null geometry inputs."""
    for field in ("range_high", "range_low", "current_price", "atr_1m"):
        value = getattr(vres, field, None)
        if value is None or pd.isna(value):
            return False
    return True


def _grid_bounds_range_size_pct(grid_lower: Any, grid_upper: Any) -> float | None:
    """Derive range_size_pct from actual grid geometry."""
    try:
        import numpy as np

        low = float(grid_lower)
        high = float(grid_upper)
        if not np.isfinite(low) or not np.isfinite(high):
            return None
        if low <= 0 or high <= low:
            return None
        midpoint = (low + high) / 2.0
        if midpoint <= 0:
            return None
        return ((high - low) / midpoint) * 100.0
    except Exception:
        return None


def _recalculate_survival_prob_from_grid(
    rd: _RegimeData,
    grid: Any,
    *,
    sym: str | None = None,
) -> float | None:
    """Recompute survival probability from actual grid bounds.

    Returns the recalculated survival probability, or None if prerequisites
    are missing or the recalculation fails.
    """
    ou_theta = rd.ou_theta
    ou_mu = rd.ou_mu
    ou_sigma = rd.ou_sigma
    current_price = rd.vres.current_price
    if (
        ou_theta is None
        or ou_mu is None
        or ou_sigma is None
        or current_price is None
        or grid is None
    ):
        return None

    grid_lower = getattr(grid, "grid_lower", None)
    grid_upper = getattr(grid, "grid_upper", None)
    range_size_pct = _grid_bounds_range_size_pct(grid_lower, grid_upper)
    if range_size_pct is None:
        return None
    assert grid_lower is not None and grid_upper is not None

    try:
        import numpy as np
        from neutralgrid.validation.stochastic import (
            StochasticConfig as _SC,
            StochasticRegimeChecker as _SRC,
        )

        _cfg = get_config()
        _checker = _SRC(_SC(
            survival_horizon=_cfg.stochastic.survival_horizon_bars,
            mc_paths=_cfg.stochastic.survival_mc_paths,
        ))
        ou_theta_f = float(ou_theta)
        ou_mu_f = float(ou_mu)
        ou_sigma_f = float(ou_sigma)
        current_price_f = float(current_price)
        grid_lower_f = cast(float, grid_lower)
        grid_upper_f = cast(float, grid_upper)
        return float(_checker.compute_survival_probability(
            ou_theta=ou_theta_f,
            ou_mu=ou_mu_f,
            ou_sigma=ou_sigma_f,
            current_log_price=np.log(current_price_f),
            range_high_log=np.log(grid_upper_f),
            range_low_log=np.log(grid_lower_f),
            horizon=_cfg.stochastic.survival_horizon_bars,
            n_paths=_cfg.stochastic.survival_mc_paths,
        ))
    except Exception as e:
        if sym is not None:
            logger.debug("Post-enrichment survival_prob recalc failed for %s: %s", sym, e)
        return None


_SCAN_PRESERVE_COLUMNS = (
    "range_prob",
    "trend_prob",
    "persistence_prob",
    "survival_prob",
    "hurst_exponent",
    "ou_halflife",
    "regime_utility",
    "hmm_range_prob",
    "hmm_trend_prob",
    "micro_osc_score",
    "micro_round_trip_cost_pct",
    "micro_min_profit_required_pct",
    "micro_viable",
    "funding_rate",
    "last_price",
    "quote_volume_24h",
)


def _preserve_scan_inputs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in _SCAN_PRESERVE_COLUMNS:
        scan_column = f"scan_{column}"
        if column in out.columns and scan_column not in out.columns:
            out[scan_column] = out[column]
    return out


def _kline_open_time(row: Any) -> int | None:
    try:
        return int(row[0])
    except Exception:
        return None


def _horizon_realized_vol_pct(
    klines_15m: Any,
    horizon_bars: float,
    estimation_bars: int = 192,
) -> float | None:
    """Realized volatility over the bot horizon, in percent (ERR-091).

    Estimated as std of 15m log returns over the trailing ``estimation_bars``
    bars, scaled by sqrt(horizon_bars). This is the ~2%-scale short-horizon
    volatility the microstructure floor's vol_premium was designed for
    (microstructure.py compute_dynamic_profit_floor: "~5% of vol, e.g. 2% vol
    -> 0.10% premium"), replacing the half-range-width proxy that ran the
    premium at ~8x design scale. Returns None when the klines are unusable.
    """
    if not isinstance(klines_15m, list) or len(klines_15m) < 20:
        return None
    try:
        closes = np.array(
            [float(k[4]) for k in klines_15m[-(estimation_bars + 1):]],
            dtype=float,
        )
    except (IndexError, TypeError, ValueError):
        return None
    closes = closes[np.isfinite(closes) & (closes > 0)]
    if len(closes) < 20:
        return None
    rets = np.diff(np.log(closes))
    sigma = float(np.std(rets))
    if not math.isfinite(sigma) or sigma <= 0:
        return None
    return sigma * math.sqrt(max(horizon_bars, 1.0)) * 100.0


def _merge_kline_sets(cached: Any, refreshed: Any) -> Any:
    if not isinstance(cached, list) or not isinstance(refreshed, list):
        return refreshed if refreshed is not None else cached
    merged: dict[int, Any] = {}
    passthrough: list[Any] = []
    for row in cached:
        key = _kline_open_time(row)
        if key is None:
            passthrough.append(row)
        else:
            merged[key] = row
    for row in refreshed:
        key = _kline_open_time(row)
        if key is None:
            passthrough.append(row)
        else:
            merged[key] = row
    return passthrough + [merged[key] for key in sorted(merged)]


@dataclass
class EnrichConfig:
    score_threshold: float = 45.0
    max_symbols: int = 250          # safety cap to manage API weight
    concurrency: int = 5
    widen: float = 0.15
    score_column: str = "score"
    range_prob_threshold: float = 0.20  # Aligned with lowest entropy-adaptive tier
    capital_base_usdt: float | None = None
    discovery_mode: bool = False


async def enrich_with_grid_params(
    df_candidates: pd.DataFrame,
    *,
    client: BinanceClient,
    pattern_profile,
    cfg: EnrichConfig | None = None,
    scan_data_cache: Dict[str, Dict[str, Any]] | None = None,
    meta_labeler: Any | None = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = EnrichConfig()

    df = _preserve_scan_inputs(df_candidates)
    if cfg.score_column not in df.columns:
        return df

    # Select candidates by score threshold OR high range probability
    # This ensures high range_prob symbols get enriched even with lower scores
    score_mask = df[cfg.score_column] >= float(cfg.score_threshold)
    range_prob_col = "hmm_range_prob" if "hmm_range_prob" in df.columns else "range_prob"
    if range_prob_col in df.columns:
        range_mask = df[range_prob_col] >= float(cfg.range_prob_threshold)
    else:
        range_mask = pd.Series(False, index=df.index)

    # Step 2a: Micro-oscillator bypass — 3rd OR condition for eligibility
    _mosc = get_config().micro_osc
    if (
        _mosc.enabled
        and "micro_osc_score" in df.columns
        and "survival_prob" in df.columns
    ):
        micro_osc_mask = (
            df["micro_osc_score"].fillna(0) >= _mosc.min_score
        ) & (
            df["survival_prob"].fillna(0) >= _mosc.min_survival_prob
        )
    else:
        micro_osc_mask = pd.Series(False, index=df.index)

    eligible = cast(
        pd.DataFrame, df.loc[score_mask | range_mask | micro_osc_mask].copy()
    )
    eligible["micro_osc_bypass"] = micro_osc_mask.reindex(eligible.index).fillna(False)

    # Step 2b: Sort bypass rows to top, then range_prob desc, then score desc
    if range_prob_col in df.columns:
        eligible = eligible.sort_values(
            by=["micro_osc_bypass", range_prob_col, cfg.score_column],
            ascending=[False, False, False],
        ).head(int(cfg.max_symbols))
    else:
        eligible = eligible.sort_values(
            by=["micro_osc_bypass", cfg.score_column], ascending=[False, False]
        ).head(int(cfg.max_symbols))

    # Initialize enrichment columns before any early returns so that
    # below-threshold enforcement always runs regardless of eligible count.
    enrichment_defaults = [
        ("grid_is_valid", False),
        ("grid_lower", None),
        ("grid_upper", None),
        ("num_grids", None),
        ("grid_spacing_pct", None),
        ("profit_per_grid_pct", None),
        ("profit_per_grid_min_pct", None),
        ("profit_per_grid_max_pct", None),
        ("leverage", None),
        ("stop_loss_pct", None),
        ("grid_reason", None),
        ("rejection_reasons", None),
        ("failure_stage", None),
        ("hmm_range_prob", None),
        ("hmm_trend_prob", None),
        ("hmm_trained_at_utc", None),
        ("hmm_posterior_mode", None),
        ("hmm_volatility_tier", None),
        ("hmm_tail_var_95", None),
        ("hmm_tail_cvar_95", None),
        ("hmm_tail_var_99", None),
        ("hmm_tail_cvar_99", None),
        ("hmm_tail_prob_p01", None),
        # Cornish-Fisher tail risk (non-stationary enhancement)
        ("hmm_cf_var_95", None),
        ("hmm_cf_cvar_95", None),
        ("hmm_cf_var_99", None),
        ("hmm_cf_cvar_99", None),
        ("hmm_weighted_kurtosis", None),
        ("hmm_gaussian_divergence", None),
        ("hmm_tail_correction", False),
        ("hmm_volatility_tier_ka", None),
        ("scan_meta_prob", None),
        ("meta_prob_source", "missing"),
        # AFML C3: Direct regime metrics from ValidationResult
        ("survival_prob", None),
        ("hurst_exponent", None),
        ("ou_halflife", None),
        ("ou_halflife_raw", None),
        ("ou_halflife_feature_reason", None),
        ("regime_utility", None),
        # AFML C4: Microstructure gating columns
        ("micro_round_trip_cost_pct", None),
        ("micro_min_profit_required_pct", None),
        ("micro_sufficient_liquidity", None),
        ("micro_extreme_funding", None),
        ("micro_viable", None),
        ("micro_reason", None),
        # Soft gating: regime confidence score [0, 1]
        ("regime_conf", None),
        # afml_enriched_v20260318: HMM persistence probability
        ("persistence_prob", None),
        # Dynamic profit floor columns
        ("dynamic_profit_floor_pct", None),
        ("micro_floor_archetype", None),
        ("micro_floor_trend_prob_input", None),
        ("micro_floor_survival_prob_input", None),
        ("liquidity_tier", None),
        ("profit_floor_components", None),
        # AFML bet sizing from meta-labeler
        ("capital_fraction", None),
        ("sizing_reason", None),
        # Below-threshold tagging (enriched but not auto-valid)
        ("below_threshold_tag", False),
        # Adaptive edge-per-grid audit columns
        ("edge_tier_chosen", None),
        ("edge_tier_attempted", False),
        ("edge_upgrade_success", None),
        ("edge_fallback_reason", None),
        ("net_edge_pct", None),
        ("edge_medium_ev_score", None),
        ("edge_big_ev_score", None),
        # Generalized Kelly diagnostics
        ("kelly_payoff_b", None),
        ("kelly_avg_win_pct", None),
        ("kelly_avg_loss_pct", None),
        ("kelly_raw_fraction", None),
        ("kelly_fractional", None),
        ("kelly_fractional_multiplier", None),
        ("kelly_fractional_mode", None),
        ("kelly_drawdown_scale", None),
        ("kelly_volatility_scale", None),
        ("kelly_profile_samples", None),
        ("kelly_profile_scope", None),
        ("kelly_profile_scope_samples", None),
        ("kelly_sweep_growth", None),
        ("kelly_sweep_drawdown_pct", None),
        ("kelly_sweep_feasible", None),
        ("kelly_sweep_evaluated", None),
        # Hard microstructure gate (Enhancement 4)
        ("hard_gate_passed", None),
        ("hard_gate_reason", None),
        # Tradable Oscillation Score (Enhancement 2)
        ("tos", None),
        ("tos_gcf", None),
        ("tos_mrs", None),
        ("tos_rc", None),
        # Position Sizer (Enhancement 3)
        ("ps_fraction", None),
        ("ps_regime_scale", None),
        ("ps_survival_scale", None),
        ("ps_micro_scale", None),
        ("ps_vol_scale", None),
        ("ps_heat_scale", None),
        ("ps_reason", None),
        ("ps_archetype", None),
        ("ps_range_prob_input", None),
        ("ps_trend_prob_input", None),
        ("ps_volatility_pct_input", None),
        ("ps_volatility_source", None),
        # Stage B approval (Enhancement 1)
        ("stage_b_approved", None),
        ("stage_b_reason", None),
        ("discovery_mode", False),
        ("discovery_geometry_filled", False),
        ("discovery_geometry_reason", None),
        ("pre_reject_would_reject", False),
        ("pre_reject_reason", None),
        ("scan_cache_age_seconds", None),
        ("scan_cache_stale", False),
        ("regime_would_reject", False),
        ("regime_rejection_reasons", None),
        ("hard_gate_would_reject", False),
        ("hard_gate_rejection_reasons", None),
        # Tier 4 context features
        ("long_short_ratio", None),
        ("funding_rate_zscore", None),
        ("open_interest_change_pct", None),
        ("bb_width_ratio_1h_15m", None),
    ]
    missing_defaults = {
        col: default
        for col, default in enrichment_defaults
        if col not in df.columns
    }
    if missing_defaults:
        additions = pd.DataFrame(missing_defaults, index=df.index)
        df = cast(pd.DataFrame, pd.concat([df, additions], axis=1))
    else:
        df = df.copy()
    df["discovery_mode"] = bool(cfg.discovery_mode)

    # Step 2c: Propagate micro_osc_bypass from eligible into df before threshold tag
    df["micro_osc_bypass"] = False
    if "micro_osc_bypass" in eligible.columns:
        df.loc[eligible.index, "micro_osc_bypass"] = eligible["micro_osc_bypass"]

    # Step 2d: Tag below-threshold rows, excluding micro-osc bypass rows
    _bypass_col = df["micro_osc_bypass"].astype(bool)
    below_threshold_mask = (
        (df[cfg.score_column] < float(cfg.score_threshold)) & ~_bypass_col
    )
    df.loc[below_threshold_mask, "below_threshold_tag"] = True

    def _enforce_threshold_gate(frame: pd.DataFrame) -> pd.DataFrame:
        """Hard gate: below-threshold symbols that did NOT pass Stage B
        must never be auto-deployable.  Stage B entropy-adaptive approval
        overrides the below-threshold tag."""
        if cfg.discovery_mode:
            return frame
        bt_mask = frame["below_threshold_tag"] == True  # noqa: E712
        if bt_mask.any():
            # Stage B approval overrides the below-threshold tag:
            # if the entropy-adaptive gate approved this candidate, respect it.
            _sb_col = (
                frame["stage_b_approved"]
                if "stage_b_approved" in frame.columns
                else pd.Series(False, index=frame.index)
            )
            stage_b_override = (
                cast(pd.Series, _sb_col)
                .astype("boolean")
                .fillna(False)
                .astype(bool)
            )
            # Only invalidate rows that are below threshold AND not Stage B approved
            invalidate_mask = bt_mask & ~stage_b_override
            if invalidate_mask.any():
                frame.loc[invalidate_mask, "grid_is_valid"] = False
                stage_missing = frame["failure_stage"].isna()
                approved_mask = frame["failure_stage"] == "approved"
                apply_mask = invalidate_mask & (stage_missing | approved_mask)
                if apply_mask.any():
                    frame.loc[apply_mask, "grid_reason"] = "score_below_threshold"
                    frame.loc[apply_mask, "rejection_reasons"] = "score_below_threshold"
                    frame.loc[apply_mask, "failure_stage"] = "score_threshold"
                for col in [
                    "grid_lower", "grid_upper", "num_grids", "grid_spacing_pct",
                    "profit_per_grid_pct", "profit_per_grid_min_pct",
                    "profit_per_grid_max_pct", "leverage", "stop_loss_pct",
                ]:
                    if col in frame.columns:
                        frame.loc[apply_mask, col] = None
        return frame

    symbols = eligible["symbol"].astype(str).tolist()
    if not symbols:
        return _enforce_threshold_gate(df)

    # Fix 3: Early-reject symbols that clearly fail the low-range floor.
    # Low scan-phase survival is kept as audit context only; fresh regime
    # validation remains authoritative before Stage B can approve a row.
    range_prob_col = "hmm_range_prob" if "hmm_range_prob" in eligible.columns else "range_prob"
    pre_reject: set[str] = set()

    def _scan_cache_age_seconds(sym: str) -> float | None:
        if scan_data_cache is None:
            return None
        cached = scan_data_cache.get(sym)
        if not cached:
            return None
        cached_at = cached.get("cached_at_utc")
        if not isinstance(cached_at, datetime):
            return None
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - cached_at).total_seconds()

    for _, row in eligible.iterrows():
        sym = str(row.get("symbol", ""))
        rp = row.get(range_prob_col)
        sp = row.get("survival_prob")
        cache_age_s = _scan_cache_age_seconds(sym)
        cache_is_stale = cache_age_s is not None and cache_age_s > 300.0
        if cache_age_s is not None:
            df.loc[df["symbol"] == sym, "scan_cache_age_seconds"] = cache_age_s
            df.loc[df["symbol"] == sym, "scan_cache_stale"] = cache_is_stale
        # Step 2e: Bypass rows skip pre-reject (they pass via micro_osc criteria)
        _is_bypass = bool(row.get("micro_osc_bypass")) if "micro_osc_bypass" in eligible.columns else False
        # Conservative thresholds: only reject clearly-failing symbols
        if not _is_bypass and rp is not None and pd.notna(rp) and float(rp) < 0.20:
            reason = f"pre_reject:low_range_prob({float(rp):.3f}<0.20)"
            df.loc[df["symbol"] == sym, "pre_reject_would_reject"] = True
            df.loc[df["symbol"] == sym, "pre_reject_reason"] = reason
            if not cfg.discovery_mode and not cache_is_stale:
                pre_reject.add(sym)
                df.loc[df["symbol"] == sym, "grid_is_valid"] = False
                df.loc[df["symbol"] == sym, "grid_reason"] = reason
                df.loc[df["symbol"] == sym, "rejection_reasons"] = reason
                df.loc[df["symbol"] == sym, "failure_stage"] = "pre_reject"
                logger.debug("Early-reject %s: range_prob=%.3f < 0.20", sym, float(rp))
            elif cache_is_stale:
                logger.debug(
                    "Scan-phase low range for %s is based on stale cache %.0fs old; deferring to fresh validation",
                    sym,
                    cache_age_s,
                )
        elif not _is_bypass and sp is not None and pd.notna(sp) and float(sp) < 0.40:
            reason = f"pre_reject:low_survival({float(sp):.3f}<0.40)"
            df.loc[df["symbol"] == sym, "pre_reject_would_reject"] = True
            df.loc[df["symbol"] == sym, "pre_reject_reason"] = reason
            logger.debug(
                "Scan-phase low survival for %s: %.3f < 0.40; deferring to fresh validation",
                sym,
                float(sp),
            )
    if pre_reject:
        logger.info(
            "Early-reject: %d/%d symbols skipped (low range_prob)",
            len(pre_reject), len(symbols),
        )
        symbols = [s for s in symbols if s not in pre_reject]
        if not symbols:
            return _enforce_threshold_gate(df)

    gate = ProfileGate.from_pattern_profile(pattern_profile, widen=cfg.widen)
    # Ensure the active rolling HMM artifact is available before validation.
    # This is intentionally bounded (see config.HMM_TRAIN_SYMBOLS / HMM_TRAIN_LIMIT_1H)
    # to avoid excessive API weight.
    await ensure_hmm_model(client=client, symbols=symbols)

    validator = RegimeValidator(gate=gate)
    calculator = GridCalculator()

    sem = asyncio.Semaphore(int(cfg.concurrency))

    # AFML C4: Create microstructure estimator with config values
    ms_config = MicrostructureConfig(
        maker_fee=float(get_config().grid.maker_fee),
        taker_fee=float(get_config().grid.taker_fee),
        funding_extreme_threshold=float(get_config().validation.funding_extreme_threshold),
    )
    ms_estimator = MicrostructureEstimator(ms_config)
    hard_gate = MicrostructureHardGate()
    tos_scorer = TradableOscillationScorer()
    pos_sizer = PositionSizer()
    stage_b_selector = TwoStageSelector()

    # Phase 2: Adaptive microstructure gate (replaces fixed thresholds when enabled)
    _adaptive_gate = None
    _adaptive_enabled = getattr(get_config(), "adaptive_micro_gate_enabled", False)
    if ADAPTIVE_GATE_AVAILABLE and _adaptive_enabled and AdaptiveMicrostructureGate is not None:
        _adaptive_gate = AdaptiveMicrostructureGate()
        logger.info("Adaptive microstructure gate enabled — thresholds will be batch-adaptive")

    # Phase 2+3: Load conformal quantile artifact for Stage B Gate 5
    _conformal_quantile = None
    try:
        from neutralgrid.calibration.conformal_risk_control_v20260311 import (
            ConformalRiskController,
        )
        _conformal_quantile = ConformalRiskController.load()
        if _conformal_quantile is not None:
            logger.info("Conformal quantile loaded (q_hat=%.4f, gate=%s)",
                       _conformal_quantile.quantile_value,
                       _conformal_quantile.gate_type)
    except (ImportError, Exception) as _cq_err:
        logger.debug("Conformal quantile not available: %s", _cq_err)

    ev_ranker = PnLRanker(RankingConfig())
    empirical_profile = load_empirical_profile_cached(str(Path("data/backtest_candidates")))
    ps_cfg = get_config().position_sizing

    # -----------------------------------------------------------------
    # Extracted sub-functions (Phase 2 decomposition of run_one)
    # -----------------------------------------------------------------

    def _evaluate_gates(
        md: _MicroData, gd: _GridData, sym: str,
    ) -> tuple[dict, dict | None]:
        """Stages 13-14: Hard gate + adaptive gate.

        Returns (hg_payload, rejection_dict_or_None).
        """
        hg_payload: dict = {}
        try:
            # Call evaluate() directly so we can override
            # sufficient_liquidity when no order-book was available.
            hg_result = hard_gate.evaluate(
                round_trip_cost_pct=(
                    getattr(md.ms_costs, "round_trip_cost_pct", None)
                    if md.ms_costs else None
                ),
                spread_cost_pct=(
                    getattr(md.ms_costs, "spread_cost_pct", None)
                    if md.ms_costs else None
                ),
                funding_cost_pct=(
                    getattr(md.ms_costs, "funding_cost_pct", None)
                    if md.ms_costs else None
                ),
                sufficient_liquidity=(
                    md.ms_costs.sufficient_liquidity
                    if md.ms_costs else None
                ),
                profit_per_grid_pct=gd.g.profit_per_grid_pct,
                dynamic_profit_floor_pct=md.adaptive_value,
            )
            hg_payload["hard_gate_passed"] = hg_result.passed
            hg_payload["hard_gate_reason"] = hg_result.reason
            if not hg_result.passed:
                rejection_dict = {
                    "grid_is_valid": False,
                    "grid_reason": f"hard_gate_fail:{hg_result.reason}",
                    "rejection_reasons": hg_result.reason,
                    "grid_lower": gd.g.grid_lower,
                    "grid_upper": gd.g.grid_upper,
                    "num_grids": gd.g.num_grids,
                    "grid_spacing_pct": gd.g.grid_spacing_pct,
                    "profit_per_grid_pct": gd.g.profit_per_grid_pct,
                    "leverage": gd.g.leverage,
                    "stop_loss_pct": gd.g.stop_loss_pct,
                    **hg_payload,
                }
                return hg_payload, rejection_dict
        except Exception as e:
            logger.warning("%s: Hard gate evaluation failed: %s", sym, e)
            hg_payload["hard_gate_passed"] = False
            hg_payload["hard_gate_reason"] = f"evaluation_error:{type(e).__name__}"

        # Phase 2: Adaptive microstructure gate (supplementary to hard gate)
        if _adaptive_gate is not None and md.ms_costs is not None:
            try:
                _rt_pct = getattr(md.ms_costs, "round_trip_cost_pct", None)
                _sp_ratio = None
                _fd_pct = getattr(md.ms_costs, "funding_cost_pct", None)
                # Compute spread-to-profit ratio if we have the data
                _spread_pct = getattr(md.ms_costs, "spread_cost_pct", None)
                if _spread_pct is not None and gd.g.profit_per_grid_pct > 0:
                    _sp_ratio = float(_spread_pct) / float(gd.g.profit_per_grid_pct)
                # Build single-candidate batch for threshold computation
                import numpy as _ag_np
                _ag_batch: dict[str, _ag_np.ndarray] = {}
                if _rt_pct is not None:
                    _ag_batch["round_trip_pct"] = _ag_np.array([float(_rt_pct)])
                if _sp_ratio is not None:
                    _ag_batch["spread_to_profit_ratio"] = _ag_np.array([float(_sp_ratio)])
                if _fd_pct is not None:
                    _ag_batch["funding_drag_pct"] = _ag_np.array([float(_fd_pct)])

                _ag_thresholds = _adaptive_gate.compute_batch_thresholds(_ag_batch)
                _ag_candidate = {
                    "round_trip_pct": float(_rt_pct) if _rt_pct is not None else None,
                    "spread_to_profit_ratio": _sp_ratio,
                    "funding_drag_pct": float(_fd_pct) if _fd_pct is not None else None,
                }
                _ag_passed, _ag_reasons = _adaptive_gate.evaluate(
                    _ag_candidate, _ag_thresholds
                )
                hg_payload["adaptive_gate_passed"] = _ag_passed
                hg_payload["adaptive_gate_source"] = _ag_thresholds.source
                if not _ag_passed:
                    hg_payload["adaptive_gate_reasons"] = "|".join(_ag_reasons)
                    logger.info(
                        "%s: Adaptive micro gate REJECTED: %s",
                        sym, "|".join(_ag_reasons),
                    )
            except Exception as _ag_err:
                logger.debug("%s: Adaptive gate evaluation failed: %s", sym, _ag_err)
                hg_payload["adaptive_gate_passed"] = None
                hg_payload["adaptive_gate_source"] = f"error:{type(_ag_err).__name__}"

        return hg_payload, None

    def _compute_position_and_tos(
        rd: _RegimeData, gd: _GridData, md: _MicroData,
        capital_fraction: Any, sym: str,
    ) -> tuple[dict, dict, Any, str]:
        """Stages 15-16: Position sizing + TOS.

        Returns (ps_payload, tos_payload, capital_fraction, sizing_suffix).
        """
        # =============================================================
        # Enhancement 3: Hard Risk Budget Position Sizer
        # =============================================================
        ps_payload: dict = {}
        sizing_suffix = ""
        try:
            kelly_sizing_active = capital_fraction is not None
            vol_proxy_ps = 2.0
            vol_source = "position_sizer"
            if rd.vres.range_high and rd.vres.range_low and rd.vres.range_low > 0:
                vol_proxy_ps = ((rd.vres.range_high - rd.vres.range_low) / rd.vres.range_low) * 100 / 2
            if kelly_sizing_active:
                # Generalized Kelly already applies volatility targeting to its
                # final fraction. Feed the risk-budget target here so the
                # PositionSizer does not apply the same volatility penalty twice.
                vol_proxy_ps = float(get_config().risk_budget.vol_target_pct)
                vol_source = "kelly_owned"
            _rp = float(rd.hmm_range_prob) if rd.hmm_range_prob is not None else 0.0
            _tp = float(rd.hmm_trend_prob) if rd.hmm_trend_prob is not None else 0.5
            ps_archetype = "standard_hmm"
            if bool(rd.micro_osc_bypass) and rd.survival_prob is not None:
                _surv_for_regime = float(rd.survival_prob)
                _rp = float(np.clip(_surv_for_regime, 0.0, 1.0))
                _tp = float(np.clip(1.0 - _surv_for_regime, 0.0, 1.0))
                ps_archetype = "micro_osc_survival"
            ps_result = pos_sizer.compute(
                range_prob=_rp,
                trend_prob=_tp,
                survival_prob=float(rd.survival_prob) if rd.survival_prob is not None else None,
                round_trip_cost_pct=md.ms_costs.round_trip_cost_pct if md.ms_costs else None,
                profit_per_grid_pct=gd.g.profit_per_grid_pct,
                volatility_pct=vol_proxy_ps,
            )
            ps_payload = {
                "ps_fraction": ps_result.fraction,
                "ps_regime_scale": ps_result.regime_confidence_scale,
                "ps_survival_scale": ps_result.survival_scale,
                "ps_micro_scale": ps_result.microstructure_scale,
                "ps_vol_scale": ps_result.volatility_scale,
                "ps_heat_scale": ps_result.portfolio_heat_scale,
                "ps_reason": ps_result.sizing_reason,
                "ps_archetype": ps_archetype,
                "ps_range_prob_input": _rp,
                "ps_trend_prob_input": _tp,
                "ps_volatility_pct_input": vol_proxy_ps,
                "ps_volatility_source": vol_source,
            }
            # Integrate position sizer with Kelly sizing
            if capital_fraction is not None:
                capital_fraction = max(0.0, round(capital_fraction * ps_result.fraction, 4))
            else:
                capital_fraction = ps_result.fraction
            sizing_suffix = f"; pos_sizer(f={ps_result.fraction:.4f})"
        except Exception as e:
            logger.warning("%s: Position sizer failed: %s", sym, e)
            ps_payload["ps_fraction"] = None
            ps_payload["ps_reason"] = f"error:{type(e).__name__}"

        # =============================================================
        # Enhancement 2: Tradable Oscillation Score
        # =============================================================
        tos_payload: dict = {}
        try:
            # Use 5m closes for micro-oscillator rows, otherwise keep the
            # standard 15m TOS input.
            closes_for_tos = np.array([float(rd.vres.current_price or 0.0)])
            klines = rd.market_data.get("klines") or {}
            if bool(rd.micro_osc_bypass):
                klines_for_tos = klines.get("5m") if isinstance(klines, dict) else None
                if klines_for_tos is None:
                    klines_for_tos = rd.market_data.get("klines_5m")
            else:
                klines_for_tos = klines.get("15m") if isinstance(klines, dict) else None
                if klines_for_tos is None:
                    klines_for_tos = rd.market_data.get("klines_15m")
            if klines_for_tos is not None and hasattr(klines_for_tos, "__len__") and len(klines_for_tos) > 0:
                try:
                    closes_for_tos = np.array(
                        [float(k[4]) for k in klines_for_tos], dtype=float
                    )
                except (IndexError, ValueError, TypeError):
                    pass

            _ppg = gd.g.profit_per_grid_pct if gd.g.profit_per_grid_pct is not None else 0.0
            _gsp = gd.g.grid_spacing_pct if gd.g.grid_spacing_pct is not None else 1.0
            _spread = md.ms_costs.spread_cost_pct if md.ms_costs else 0.0
            _funding = md.ms_costs.funding_cost_pct if md.ms_costs else 0.0
            _depth = None
            order_book = rd.market_data.get("order_book", {})
            if order_book:
                bids = order_book.get("bids", [])
                asks = order_book.get("asks", [])
                bid_depth = sum(float(b[0]) * float(b[1]) for b in bids[:10]) if bids else 0.0
                ask_depth = sum(float(a[0]) * float(a[1]) for a in asks[:10]) if asks else 0.0
                _depth = min(bid_depth, ask_depth)
            capital_base_usdt = (
                float(cfg.capital_base_usdt)
                if cfg.capital_base_usdt is not None
                else float(get_config().grid.capital)
            )
            _pos_size = capital_base_usdt * float(gd.g.leverage or 10)

            tos_result = tos_scorer.compute(
                closes=closes_for_tos,
                grid_lower=float(gd.g.grid_lower or 0),
                grid_upper=float(gd.g.grid_upper or 0),
                grid_spacing_pct=float(_gsp),
                profit_per_grid_pct=float(_ppg),
                spread_cost_pct=float(_spread),
                funding_cost_pct=float(_funding),
                book_depth_usdt=_depth,
                position_size_usdt=_pos_size,
                current_price=float(rd.vres.current_price or 0),
                leverage=int(gd.g.leverage or 10),
                hurst_exponent=float(rd.hurst_exponent) if rd.hurst_exponent is not None else None,
                ou_halflife=float(rd.ou_halflife) if rd.ou_halflife is not None else None,
            )
            tos_payload = {
                "tos": tos_result.tos,
                "tos_gcf": tos_result.grid_cross_frequency,
                "tos_mrs": tos_result.mean_reversion_strength,
                "tos_rc": tos_result.range_containment,
            }
        except Exception as e:
            logger.warning("%s: TOS computation failed: %s", sym, e)
            tos_payload["tos"] = None

        return ps_payload, tos_payload, capital_fraction, sizing_suffix

    def _evaluate_stage_b(
        rd: _RegimeData, hg_payload: dict, tos_payload: dict,
        capital_fraction: Any, meta_prob: Any, sym: str,
    ) -> tuple[dict, bool, str, Any]:
        """Stage B deployment approval.

        Returns (sb_payload, grid_valid, grid_reason, reject_reasons).
        """
        sb_payload: dict = {}
        try:
            sb_result = stage_b_selector.approve(
                hard_gate_passed=bool(hg_payload.get("hard_gate_passed")),
                hard_gate_reason=hg_payload.get("hard_gate_reason") or "unknown",
                tos=tos_payload.get("tos"),
                # Use final capital_fraction (Kelly x sizer), not raw ps_fraction
                position_size_fraction=capital_fraction,
                range_prob=float(rd.hmm_range_prob) if rd.hmm_range_prob is not None else None,
                trend_prob=float(rd.hmm_trend_prob) if rd.hmm_trend_prob is not None else None,
                meta_prob=meta_prob,
                conformal_quantile=_conformal_quantile,
                symbol=sym,
                posteriors=rd.hmm_posteriors,
                micro_osc_score=float(rd.micro_osc_score) if rd.micro_osc_score is not None else 0.0,
                # ERR-082: pass None (not 0.0) when survival_prob is absent so
                # Gate 4 micro-osc mode emits data_missing:survival_prob.
                survival_prob=float(rd.survival_prob) if rd.survival_prob is not None else None,
                # ERR-093: Gate 4 survival mode only for true bypass rows.
                micro_osc_bypass=bool(rd.micro_osc_bypass),
            )
            sb_payload["stage_b_approved"] = sb_result.approved
            sb_payload["stage_b_reason"] = sb_result.reason
        except Exception as e:
            logger.warning("%s: Stage B evaluation failed: %s", sym, e)
            sb_payload["stage_b_approved"] = False
            sb_payload["stage_b_reason"] = f"error:{type(e).__name__}"

        # Stage B is the deployment gate -- controls grid_is_valid
        _sb_approved = sb_payload.get("stage_b_approved", False)
        _sb_reason = sb_payload.get("stage_b_reason", "stage_b_unknown")
        if cfg.discovery_mode:
            _grid_valid = True
            _grid_reason = "ok" if bool(_sb_approved) else f"discovery_mode:stage_b_rejected:{_sb_reason}"
            _reject_reasons = None if bool(_sb_approved) else _sb_reason
        else:
            _grid_valid = bool(_sb_approved)
            _grid_reason = "ok" if _grid_valid else f"stage_b_rejected:{_sb_reason}"
            _reject_reasons = None if _grid_valid else _sb_reason

        return sb_payload, _grid_valid, _grid_reason, _reject_reasons

    def _estimate_microstructure(rd: _RegimeData, sym: str) -> _MicroData:
        """Stage 9: Microstructure cost estimation."""
        ms_payload: dict = {}
        adaptive_value = None  # will hold per-symbol min profit %
        liquidity_tier = None
        ms_costs = None
        floor_components = None
        _has_book = False
        vol_proxy = 2.0  # fallback default

        try:
            order_book = rd.market_data.get("order_book", {})
            _has_book = bool(
                order_book
                and order_book.get("bids")
                and order_book.get("asks")
            )

            live_horizon_hours = float(get_config().grid.max_holding_seconds) / 3600.0

            # ERR-091: volatility input on the floor's DESIGN scale (~2% short-
            # horizon vol), estimated as bot-horizon realized vol from 15m
            # returns. The former proxy (half the validated range width, values
            # 14-17) ran vol_premium at ~8x design scale AND double-counted the
            # same statistic into the slippage vol_adjustment, pushing dynamic
            # floors (1.33-2.08%) above the max_spacing-capped profit ceiling
            # (~1.03-1.08%) -- guaranteed rejection for wide-range symbols.
            vol_proxy = 2.0  # design-scale fallback when klines unusable
            _klines_15m = (rd.market_data.get("klines") or {}).get("15m")
            if _klines_15m is None:
                _klines_15m = rd.market_data.get("klines_15m")
            _rv = _horizon_realized_vol_pct(
                _klines_15m, horizon_bars=live_horizon_hours * 4.0
            )
            if _rv is not None:
                vol_proxy = _rv

            ms_costs = ms_estimator.estimate_costs(
                order_book=order_book,
                funding_rate=rd.funding_rate_decimal,
                volatility_pct=vol_proxy,
                horizon_hours=live_horizon_hours,
            )

            # Dynamic profit floor: covers fees + spread + adverse selection + safety
            _surv = float(rd.survival_prob) if rd.survival_prob is not None else 0.70
            _trend = float(rd.hmm_trend_prob) if rd.hmm_trend_prob is not None else 0.0
            floor_archetype = "standard_hmm"
            if bool(rd.micro_osc_bypass) and rd.survival_prob is not None:
                _trend = float(np.clip(1.0 - _surv, 0.0, 1.0))
                floor_archetype = "micro_osc_survival"
            adaptive_value, liquidity_tier, floor_components = (
                ms_estimator.compute_dynamic_profit_floor(
                    costs=ms_costs,
                    volatility_pct=vol_proxy,
                    trend_prob=_trend,
                    survival_prob=_surv,
                    leverage=int(get_config().grid.leverage_max),
                    horizon_hours=live_horizon_hours,
                )
            )

            ms_payload = {
                "micro_round_trip_cost_pct": round(ms_costs.round_trip_cost_pct, 4),
                "micro_min_profit_required_pct": round(ms_costs.min_profit_required_pct, 4),
                "micro_sufficient_liquidity": ms_costs.sufficient_liquidity,
                "micro_extreme_funding": ms_costs.extreme_funding,
                "dynamic_profit_floor_pct": round(adaptive_value, 4) if adaptive_value is not None else None,
                "micro_floor_archetype": floor_archetype,
                "micro_floor_trend_prob_input": round(_trend, 4),
                "micro_floor_survival_prob_input": round(_surv, 4),
                "liquidity_tier": liquidity_tier,
                "profit_floor_components": str(floor_components) if floor_components else None,
            }
        except Exception as e:
            logger.warning(f"{sym}: Microstructure estimation failed: {e}")
            ms_payload = {
                "micro_viable": None,
                "micro_reason": f"estimation_failed:{type(e).__name__}",
            }

        return _MicroData(
            ms_payload=ms_payload,
            ms_costs=ms_costs,
            adaptive_value=adaptive_value,
            has_book=_has_book,
            vol_proxy=vol_proxy,
        )

    def _generate_grid(
        rd: _RegimeData, md: _MicroData, sym: str,
    ) -> tuple[_GridData, dict | None]:
        """Stage 10: Grid generation with edge tiers. Returns (grid_data, rejection_dict_or_None)."""
        edge_cfg = get_config().edge_tier
        raw_vres: Any = rd.vres
        grid_vres: Any = raw_vres
        should_bypass_regime_validity = (
            not bool(getattr(raw_vres, "is_valid", False))
            and (
                cfg.discovery_mode
                or (rd.micro_osc_bypass and _has_grid_generation_geometry(raw_vres))
            )
        )
        if should_bypass_regime_validity:
            if is_dataclass(raw_vres) and not isinstance(raw_vres, type):
                grid_vres = replace(raw_vres, is_valid=True)
            else:
                grid_vres = copy.copy(raw_vres)
                grid_vres.is_valid = True
        range_size_pct: float | None = None
        edge_info: dict = {
            "edge_tier_chosen": None,
            "edge_tier_attempted": False,
            "edge_upgrade_success": None,
            "edge_fallback_reason": None,
            "net_edge_pct": None,
            "edge_medium_ev_score": None,
            "edge_big_ev_score": None,
        }

        # Scan-time range size percentage for early EV comparisons
        scan_range_size_pct = None
        try:
            if (
                rd.vres.range_high is not None
                and rd.vres.range_low is not None
                and rd.vres.current_price is not None
                and float(rd.vres.current_price) > 0
            ):
                scan_range_size_pct = (
                    (float(rd.vres.range_high) - float(rd.vres.range_low))
                    / float(rd.vres.current_price)
                    * 100.0
                )
        except Exception:
            scan_range_size_pct = None

        # Microstructure cost basis for net_edge computation
        static_fallback = float(get_config().grid.profit_grid_min_pct_static_fallback)
        micro_floor = (
            md.ms_costs.min_profit_required_pct
            if md.ms_costs is not None
            else (md.adaptive_value or static_fallback)
        )
        base_floor = md.adaptive_value if md.adaptive_value is not None else static_fallback

        if edge_cfg.enable and rd.micro_osc_bypass:
            dense_target = md.adaptive_value if md.adaptive_value is not None else micro_floor
            g = calculator.generate_params(
                grid_vres,
                min_profit_pct=dense_target,
                range_prob=rd.hmm_range_prob,
                trend_prob=rd.hmm_trend_prob,
                survival_prob=rd.survival_prob,
                hurst_exponent=rd.hurst_exponent,
                capital_base=cfg.capital_base_usdt,
            )
            edge_info["edge_tier_attempted"] = True
            edge_info["edge_tier_chosen"] = "MICRO_OSC_DENSE"
            edge_info["edge_upgrade_success"] = False
            edge_info["edge_fallback_reason"] = "micro_osc_dense_grid_cost_floor"
            edge_info["net_edge_pct"] = round((g.profit_per_grid_pct or 0) - micro_floor, 4)
        elif edge_cfg.enable:
            # --- Stage 1: Medium edge (feasible-first) ---
            medium_target = max(base_floor, micro_floor + edge_cfg.medium_buffer_pct)
            g_medium = calculator.generate_params(
                grid_vres,
                min_profit_pct=medium_target,
                range_prob=rd.hmm_range_prob,
                trend_prob=rd.hmm_trend_prob,
                survival_prob=rd.survival_prob,
                hurst_exponent=rd.hurst_exponent,
                capital_base=cfg.capital_base_usdt,
            )
            edge_info["edge_tier_attempted"] = True

            if not g_medium.is_valid:
                # Medium failed -> not feasible at any edge tier
                grid_reason = getattr(g_medium, "reason", None) or "grid_invalid"
                reject_codes = []
                if "profit_per_grid_below_min" in (grid_reason or ""):
                    reject_codes.append(f"profit_per_grid_fail:{grid_reason}")
                elif "missing_range" in (grid_reason or ""):
                    reject_codes.append("missing_range_data")
                else:
                    reject_codes.append(grid_reason)
                edge_info["edge_fallback_reason"] = grid_reason
                return (
                    _GridData(g=g_medium, edge_info=edge_info, range_size_pct=scan_range_size_pct),
                    {
                        "grid_is_valid": False,
                        "grid_reason": grid_reason,
                        "rejection_reasons": "|".join(reject_codes),
                        "failure_stage": "grid_generation",
                    },
                )

            g = g_medium
            net_edge_med = (g_medium.profit_per_grid_pct or 0) - micro_floor
            edge_info["edge_tier_chosen"] = "MEDIUM"
            edge_info["edge_upgrade_success"] = False
            edge_info["net_edge_pct"] = round(net_edge_med, 4)

            # EV diagnostics for MEDIUM tier
            ev_medium = None
            if (
                rd.survival_prob is not None
                and rd.hmm_trend_prob is not None
                and scan_range_size_pct is not None
            ):
                try:
                    ev_medium = float(
                        ev_ranker.compute_score(
                            profit_per_grid_pct=float(g_medium.profit_per_grid_pct or 0.0),
                            num_grids=int(g_medium.num_grids or 0),
                            survival_prob=float(rd.survival_prob),
                            trend_prob=float(rd.hmm_trend_prob),
                            funding_rate=float(rd.funding_rate_decimal)
                            if rd.funding_rate_decimal is not None
                            else None,
                            range_size_pct=float(scan_range_size_pct),
                            symbol=sym,
                            leverage=int(g_medium.leverage or 10),
                        ).rank_score
                    )
                    edge_info["edge_medium_ev_score"] = round(ev_medium, 4)
                except Exception:
                    ev_medium = None

            # --- Stage 2: Try Big edge (optimize) ---
            big_target = max(base_floor, micro_floor + edge_cfg.big_buffer_pct)
            g_big = calculator.generate_params(
                grid_vres,
                min_profit_pct=big_target,
                range_prob=rd.hmm_range_prob,
                trend_prob=rd.hmm_trend_prob,
                survival_prob=rd.survival_prob,
                hurst_exponent=rd.hurst_exponent,
                capital_base=cfg.capital_base_usdt,
            )

            if g_big.is_valid:
                # Verify Big passes microstructure viability
                big_micro_ok = True
                big_fail_reason = None
                if md.ms_costs is not None:
                    big_profit = g_big.profit_per_grid_pct or 0.0
                    is_v, v_reason = ms_estimator.is_viable(md.ms_costs, big_profit)
                    if not is_v:
                        big_micro_ok = False
                        big_fail_reason = v_reason

                if big_micro_ok:
                    # Check edge and EV improvements before upgrade
                    net_edge_big = (g_big.profit_per_grid_pct or 0) - micro_floor
                    required_net = edge_cfg.big_buffer_pct + edge_cfg.upgrade_margin_pct
                    choose_big_by_edge = net_edge_big >= required_net

                    ev_big = None
                    if (
                        rd.survival_prob is not None
                        and rd.hmm_trend_prob is not None
                        and scan_range_size_pct is not None
                    ):
                        try:
                            ev_big = float(
                                ev_ranker.compute_score(
                                    profit_per_grid_pct=float(g_big.profit_per_grid_pct or 0.0),
                                    num_grids=int(g_big.num_grids or 0),
                                    survival_prob=float(rd.survival_prob),
                                    trend_prob=float(rd.hmm_trend_prob),
                                    funding_rate=float(rd.funding_rate_decimal)
                                    if rd.funding_rate_decimal is not None
                                    else None,
                                    range_size_pct=float(scan_range_size_pct),
                                    symbol=sym,
                                    leverage=int(g_big.leverage or 10),
                                ).rank_score
                            )
                            edge_info["edge_big_ev_score"] = round(ev_big, 4)
                        except Exception:
                            ev_big = None

                    # If EV is measurable for both tiers, require meaningful EV lift.
                    if ev_big is not None and ev_medium is not None:
                        choose_big_by_ev = ev_big >= (
                            ev_medium + float(edge_cfg.ev_upgrade_min_delta)
                        )
                    else:
                        # Fallback to edge-only decision when EV cannot be compared.
                        choose_big_by_ev = True

                    if choose_big_by_edge and choose_big_by_ev:
                        g = g_big
                        edge_info["edge_tier_chosen"] = "BIG"
                        edge_info["edge_upgrade_success"] = True
                        edge_info["net_edge_pct"] = round(net_edge_big, 4)
                    else:
                        g = g_medium
                        edge_info["edge_tier_chosen"] = "MEDIUM"
                        edge_info["edge_upgrade_success"] = False
                        if not choose_big_by_edge:
                            edge_info["edge_fallback_reason"] = (
                                f"upgrade_margin_not_met(net_edge={net_edge_big:.4f}"
                                f"<{required_net:.4f})"
                            )
                        else:
                            _delta = None
                            if ev_big is not None and ev_medium is not None:
                                _delta = ev_big - ev_medium
                            if _delta is not None:
                                edge_info["edge_fallback_reason"] = (
                                    f"ev_delta_not_met(delta={_delta:.4f}"
                                    f"<{float(edge_cfg.ev_upgrade_min_delta):.4f})"
                                )
                            else:
                                edge_info["edge_fallback_reason"] = "ev_unavailable_fallback"
                        edge_info["net_edge_pct"] = round(net_edge_med, 4)
                else:
                    g = g_medium
                    edge_info["edge_tier_chosen"] = "MEDIUM"
                    edge_info["edge_upgrade_success"] = False
                    edge_info["edge_fallback_reason"] = (
                        f"big_microstructure_fail:{big_fail_reason}"
                    )
                    net_edge_med = (g_medium.profit_per_grid_pct or 0) - micro_floor
                    edge_info["net_edge_pct"] = round(net_edge_med, 4)
            else:
                # Big grid generation failed -> keep Medium
                g = g_medium
                edge_info["edge_tier_chosen"] = "MEDIUM"
                edge_info["edge_upgrade_success"] = False
                big_reason = getattr(g_big, "reason", None) or "grid_invalid"
                edge_info["edge_fallback_reason"] = (
                    f"big_grid_fail:{big_reason}"
                )
                edge_info["net_edge_pct"] = round(net_edge_med, 4)
        else:
            # Edge tier disabled -> single-pass grid generation
            g = calculator.generate_params(
                grid_vres,
                min_profit_pct=md.adaptive_value,
                range_prob=rd.hmm_range_prob,
                trend_prob=rd.hmm_trend_prob,
                survival_prob=rd.survival_prob,
                hurst_exponent=rd.hurst_exponent,
                capital_base=cfg.capital_base_usdt,
            )

        range_size_pct = _grid_bounds_range_size_pct(g.grid_lower, g.grid_upper)
        if range_size_pct is None:
            range_size_pct = scan_range_size_pct
        _survival_prob_enriched = _recalculate_survival_prob_from_grid(rd, g, sym=sym)
        if _survival_prob_enriched is not None:
            rd.survival_prob = _survival_prob_enriched

        if not g.is_valid:
            grid_reason = getattr(g, "reason", None) or "grid_invalid"
            reject_codes = []
            if "profit_per_grid_below_min" in (grid_reason or ""):
                reject_codes.append(f"profit_per_grid_fail:{grid_reason}")
            elif "missing_range" in (grid_reason or ""):
                reject_codes.append("missing_range_data")
            else:
                reject_codes.append(grid_reason)
            return (
                _GridData(g=g, edge_info=edge_info, range_size_pct=range_size_pct),
                {
                    "grid_is_valid": False,
                    "grid_reason": grid_reason,
                    "rejection_reasons": "|".join(reject_codes),
                    "failure_stage": "grid_generation",
                },
            )

        return _GridData(g=g, edge_info=edge_info, range_size_pct=range_size_pct), None

    def _check_viability(
        gd: _GridData, md: _MicroData, sym: str,
    ) -> None:
        """Stage 11: Post-grid viability annotation."""
        try:
            if md.ms_costs is not None:
                profit_per_grid = gd.g.profit_per_grid_pct or 0.0
                is_viable, ms_reason = ms_estimator.is_viable(md.ms_costs, profit_per_grid)
                md.ms_payload["micro_viable"] = is_viable
                md.ms_payload["micro_reason"] = ms_reason

                if not is_viable:
                    logger.debug(f"{sym}: Microstructure fail - {ms_reason}")
        except Exception as e:
            logger.warning(f"{sym}: Microstructure viability check failed: {e}")
            md.ms_payload["micro_viable"] = None
            md.ms_payload["micro_reason"] = f"viability_check_failed:{type(e).__name__}"

    def _compute_kelly(
        sym: str, gd: _GridData, rd: _RegimeData, meta_prob: Any,
    ) -> tuple[dict, Any, str]:
        """Stage 12: Kelly sizing. Returns (kelly_info, capital_fraction, sizing_reason)."""
        capital_fraction = None
        sizing_reason = getattr(gd.g, "sizing_reason", None) or "standard_sizing"
        kelly_info: dict = {
            "kelly_payoff_b": None,
            "kelly_avg_win_pct": None,
            "kelly_avg_loss_pct": None,
            "kelly_raw_fraction": None,
            "kelly_fractional": None,
            "kelly_fractional_multiplier": None,
            "kelly_fractional_mode": None,
            "kelly_drawdown_scale": None,
            "kelly_volatility_scale": None,
            "kelly_profile_samples": None,
            "kelly_profile_scope": None,
            "kelly_profile_scope_samples": None,
            "kelly_sweep_growth": None,
            "kelly_sweep_drawdown_pct": None,
            "kelly_sweep_feasible": None,
            "kelly_sweep_evaluated": None,
        }

        if meta_prob is not None:
            if bool(ps_cfg.enable_generalized_kelly):
                kelly = generalized_kelly_details(
                    meta_prob=meta_prob,
                    profile=empirical_profile,
                    fractional_kelly=float(ps_cfg.fractional_kelly),
                    optimize_fractional=bool(
                        getattr(ps_cfg, "optimize_fractional_kelly", False)
                    ),
                    fractional_sweep_min=float(
                        getattr(ps_cfg, "fractional_kelly_sweep_min", 0.10)
                    ),
                    fractional_sweep_max=float(
                        getattr(ps_cfg, "fractional_kelly_sweep_max", 1.00)
                    ),
                    fractional_sweep_step=float(
                        getattr(ps_cfg, "fractional_kelly_sweep_step", 0.05)
                    ),
                    drawdown_tolerance_pct=float(ps_cfg.drawdown_tolerance_pct),
                    volatility_target_pct=float(ps_cfg.volatility_target_pct),
                    volatility_proxy_pct=float(gd.range_size_pct)
                    if gd.range_size_pct is not None
                    else None,
                    symbol=sym,
                    trend_prob=float(rd.hmm_trend_prob)
                    if rd.hmm_trend_prob is not None
                    else None,
                    min_samples=ev_ranker.config.empirical_alignment_min_samples,
                )
                kelly_info = {
                    "kelly_payoff_b": round(float(kelly["payoff_ratio_b"]), 4),
                    "kelly_avg_win_pct": round(float(kelly["avg_win_pct"]), 4),
                    "kelly_avg_loss_pct": round(float(kelly["avg_loss_pct"]), 4),
                    "kelly_raw_fraction": round(float(kelly["raw_fraction"]), 4),
                    "kelly_fractional": round(float(kelly["fractional_kelly"]), 4),
                    "kelly_fractional_multiplier": round(
                        float(kelly.get("fractional_multiplier", ps_cfg.fractional_kelly)),
                        4,
                    ),
                    "kelly_fractional_mode": str(kelly.get("fractional_mode", "fixed")),
                    "kelly_drawdown_scale": round(float(kelly["drawdown_scale"]), 4),
                    "kelly_volatility_scale": round(float(kelly["volatility_scale"]), 4),
                    "kelly_profile_samples": int(kelly["profile_samples"]),
                    "kelly_profile_scope": str(kelly.get("profile_scope", "global")),
                    "kelly_profile_scope_samples": round(
                        float(kelly.get("profile_scope_samples", kelly["profile_samples"])),
                        1,
                    ),
                    "kelly_sweep_growth": round(
                        float(kelly.get("sweep_objective_growth", 0.0)),
                        6,
                    ),
                    "kelly_sweep_drawdown_pct": round(
                        float(kelly.get("sweep_estimated_drawdown_pct", 0.0)),
                        4,
                    ),
                    "kelly_sweep_feasible": int(
                        round(float(kelly.get("sweep_feasible_points", 0.0)))
                    ),
                    "kelly_sweep_evaluated": int(
                        round(float(kelly.get("sweep_evaluated_points", 0.0)))
                    ),
                }

                kelly_edge = float(kelly["raw_fraction"])
                if bool(ps_cfg.reject_non_positive_edge) and kelly_edge <= 0:
                    sizing_reason += (
                        f"; negative_kelly_edge(raw={kelly_edge:.3f},meta_prob={meta_prob:.2f})"
                    )
                target_fraction = float(kelly["final_fraction"])
                target_fraction = max(float(ps_cfg.min_fraction), target_fraction)
                target_fraction = min(float(ps_cfg.max_fraction), target_fraction)
                capital_fraction = target_fraction
                sizing_reason += (
                    f"; gen_kelly(p={meta_prob:.2f},b={float(kelly['payoff_ratio_b']):.3f},"
                    f"f={target_fraction:.3f})"
                )
            else:
                sizing_reason += f"; gen_kelly_disabled(meta_prob={meta_prob:.2f})"

        return kelly_info, capital_fraction, sizing_reason

    def _normalize_ou_halflife_feature(vres: Any, raw_halflife: Any) -> tuple[float | None, str | None]:
        """Return the finite OU half-life value used by meta-labeler features.

        ERR-065: non-mean-reverting rows (halflife=inf) FAIL CLOSED (feature
        absent -> meta probe skipped -> Stage B `data_missing:meta`). An
        earlier draft capped them to `ou_halflife_max_bars` (48), but the
        training pool's ou_halflife is raw and heavy-tailed (median ~41,
        p99 ~1014, max ~9e5, never capped at 48) — feeding 48 would score a
        no-mean-reversion row as MEDIAN mean-reversion behavior, inflating
        meta_prob for exactly the rows the strategy should avoid, and would
        be silent imputation in live admission logic (the invariant the D1
        fix exists to prevent). The reason column preserves the audit trail.
        """
        if raw_halflife is not None:
            try:
                value = float(raw_halflife)
            except (TypeError, ValueError):
                value = np.nan
            if np.isfinite(value):
                return value, None

        for tf_result in (getattr(vres, "tf_1h", None), getattr(vres, "tf_15m", None), getattr(vres, "tf_5m", None)):
            reason = getattr(tf_result, "reason", None)
            if isinstance(reason, str) and "halflife=inf (no mean-reversion)" in reason:
                return None, "non_mean_reverting_fail_closed"
        return None, None

    async def _fetch_regime_data(sym: str) -> _RegimeData:
        """Stages 1-6: Fetch market data, validate regime, extract HMM posteriors, funding, trigger price."""
        async with sem:
            cached_klines: dict[str, Any] = {}
            if scan_data_cache is not None and sym in scan_data_cache:
                cached = scan_data_cache[sym]
                # Step 11B: warn if cached klines are stale (>300s)
                cached_at = cached.get("cached_at_utc")
                if cached_at is not None:
                    age_s = (datetime.now(timezone.utc) - cached_at).total_seconds()
                    if age_s > 300:
                        logger.warning(
                            "Scan cache for %s is %.0fs old (>300s); klines may be stale",
                            sym, age_s,
                        )
                if "klines_1h" in cached:
                    cached_klines["1h"] = cached["klines_1h"]
                if "klines_15m" in cached:
                    cached_klines["15m"] = cached["klines_15m"]
                if "klines_5m" in cached:
                    cached_klines["5m"] = cached["klines_5m"]
                if "klines_1m" in cached:
                    cached_klines["1m"] = cached["klines_1m"]

            market_data = await client.get_all_market_data(sym)
            if cached_klines:
                refreshed_klines = dict(market_data.get("klines") or {})
                for interval, cached_rows in cached_klines.items():
                    refreshed_klines[interval] = _merge_kline_sets(
                        cached_rows,
                        refreshed_klines.get(interval),
                    )
                market_data["klines"] = refreshed_klines
                market_data["scan_cache_refreshed"] = True

        vres = validator.validate(market_data)
        discovery_geometry_filled = False
        discovery_geometry_reason = None
        if (
            cfg.discovery_mode
            and not bool(getattr(vres, "is_valid", False))
            and (
                getattr(vres, "range_high", None) is None
                or getattr(vres, "range_low", None) is None
                or getattr(vres, "current_price", None) is None
                or getattr(vres, "atr_1m", None) is None
            )
        ):
            discovery_geometry_filled, discovery_geometry_reason = (
                _fill_discovery_geometry_from_market_data(
                    vres,
                    market_data,
                    validator,
                )
            )

        # AFML C3: Extract direct regime metrics from ValidationResult
        # These are now exposed at top level (not nested in checks dict)
        hmm_range_prob = vres.range_prob
        hmm_trend_prob = vres.trend_prob
        survival_prob = vres.survival_prob
        hurst_exponent = vres.hurst_exponent
        ou_halflife = vres.ou_halflife
        ou_halflife_feature, ou_halflife_feature_reason = _normalize_ou_halflife_feature(vres, ou_halflife)
        regime_utility = vres.regime_utility
        volatility_tier = vres.volatility_tier
        conditional_tail_risk = vres.conditional_tail_risk or {}
        if not isinstance(conditional_tail_risk, dict):
            conditional_tail_risk = {}
        conditional_tail_risk_enhanced = vres.conditional_tail_risk_enhanced or {}
        if not isinstance(conditional_tail_risk_enhanced, dict):
            conditional_tail_risk_enhanced = {}
        tail_correction_applied = getattr(vres, "tail_correction_applied", False)
        volatility_tier_ka = getattr(vres, "volatility_tier_kurtosis_aware", None)

        # Backward-compatible fallback for failed validations where
        # top-level fields may be unset but tf_1h metrics still exist.
        if vres.tf_1h and isinstance(vres.tf_1h.checks, dict):
            if volatility_tier is None:
                _vt = vres.tf_1h.checks.get("volatility_tier")
                if _vt is not None:
                    volatility_tier = str(_vt)
            if not conditional_tail_risk:
                _ctr = vres.tf_1h.checks.get("conditional_tail_risk")
                if isinstance(_ctr, dict):
                    conditional_tail_risk = _ctr

        # Extract regime_conf from top-level ValidationResult fields (populated
        # by the validator from HMM check metrics). Falls back to tf_1h.checks
        # for backward compatibility with older validation results.
        regime_conf: float | None = getattr(vres, "regime_conf", None)
        posterior_mode: str | None = getattr(vres, "posterior_mode", None)
        hmm_artifact_version: str | None = getattr(vres, "hmm_artifact_version", None)
        hmm_pipeline_version: str | None = getattr(vres, "hmm_pipeline_version", None)
        calibration_provenance: dict[str, Any] = getattr(vres, "hmm_calibration_provenance", None) or {}

        # Extract persistence_prob from top-level.
        persistence_prob: float | None = getattr(vres, "persistence_prob", None)

        # Fallback to tf_1h.checks for legacy validation results
        if vres.tf_1h and isinstance(vres.tf_1h.checks, dict):
            _checks = vres.tf_1h.checks
            if regime_conf is None:
                _rc = _checks.get("regime_conf")
                if _rc is not None:
                    regime_conf = float(_rc)
            if posterior_mode is None:
                _pm = _checks.get("posterior_mode")
                if _pm is not None:
                    posterior_mode = str(_pm)
            if hmm_artifact_version is None:
                _av = _checks.get("artifact_version")
                if _av is not None:
                    hmm_artifact_version = str(_av)
            if hmm_pipeline_version is None:
                _pv = _checks.get("pipeline_version")
                if _pv is not None:
                    hmm_pipeline_version = str(_pv)
            if not calibration_provenance:
                _cp = _checks.get("calibration_provenance")
                if isinstance(_cp, dict):
                    calibration_provenance = _cp
            if persistence_prob is None:
                _pp = _checks.get("persistence_prob")
                if _pp is not None:
                    persistence_prob = float(_pp)

        # Fallback for range/trend prob from nested checks
        if hmm_range_prob is None and vres.tf_1h and isinstance(vres.tf_1h.checks, dict):
            hmm_checks = vres.tf_1h.checks
            hmm_range_prob = hmm_checks.get("range_prob_agg", hmm_checks.get("range_prob"))
            hmm_trend_prob = hmm_checks.get("trend_prob_agg", hmm_checks.get("trend_prob"))

        # Canonical HMM posteriors — read from top-level first, fallback to checks.
        _hmm_posteriors = getattr(vres, "posteriors", None)
        if _hmm_posteriors is None and vres.tf_1h and isinstance(vres.tf_1h.checks, dict):
            _post = vres.tf_1h.checks.get("posteriors")
            if _post is not None:
                _hmm_posteriors = _post

        # HMM trained_at — read from top-level first, fallback to checks.
        hmm_trained_at = getattr(vres, "hmm_trained_at_utc", None)
        if hmm_trained_at is None and vres.tf_1h and isinstance(vres.tf_1h.checks, dict):
            hmm_trained_at = vres.tf_1h.checks.get("trained_at_utc")

        # Extract funding rate from premium_index (AFML: 14/14 feature coverage)
        # Kept in decimal form (e.g. 0.0001) to match PnLRanker expectations
        funding_rate_decimal = None
        try:
            fr_data = market_data.get("funding_rate") or market_data.get("premium_index", {})
            if isinstance(fr_data, dict):
                funding_rate_decimal = float(fr_data.get("lastFundingRate", 0))
            elif isinstance(fr_data, list) and len(fr_data) > 0:
                funding_rate_decimal = float(fr_data[0].get("fundingRate", 0))
        except Exception:
            funding_rate_decimal = None

        # --- Tier 4 context features ---
        # Long/Short ratio: last value from global L/S ratio history
        long_short_ratio: float | None = None
        try:
            ls_data = market_data.get("long_short_ratio")
            if isinstance(ls_data, list) and ls_data:
                long_short_ratio = float(ls_data[-1].get("longShortRatio", 0))
        except (ValueError, TypeError, KeyError):
            long_short_ratio = None

        # Funding rate z-score: standardized deviation of latest rate
        funding_rate_zscore: float | None = None
        try:
            import numpy as np
            fr_hist = market_data.get("funding_rate")
            if isinstance(fr_hist, list) and len(fr_hist) >= 10:
                rates = [float(r["fundingRate"]) for r in fr_hist]
                mu, sigma = float(np.mean(rates)), float(np.std(rates))
                funding_rate_zscore = (rates[-1] - mu) / sigma if sigma > 1e-10 else 0.0
        except (ValueError, TypeError, KeyError):
            funding_rate_zscore = None

        # Open interest change %: 24h change from OI history
        open_interest_change_pct: float | None = None
        try:
            oi_hist = market_data.get("open_interest_hist")
            if isinstance(oi_hist, list) and len(oi_hist) >= 2:
                current = float(oi_hist[-1].get("sumOpenInterestValue", 0))
                baseline = float(oi_hist[0].get("sumOpenInterestValue", 0))
                if baseline > 0:
                    open_interest_change_pct = ((current - baseline) / baseline) * 100.0
        except (ValueError, TypeError, KeyError):
            open_interest_change_pct = None

        return _RegimeData(
            market_data=market_data,
            vres=vres,
            hmm_range_prob=hmm_range_prob,
            hmm_trend_prob=hmm_trend_prob,
            survival_prob=survival_prob,
            hurst_exponent=hurst_exponent,
            ou_halflife=ou_halflife,
            ou_halflife_feature=ou_halflife_feature,
            ou_halflife_feature_reason=ou_halflife_feature_reason,
            regime_utility=regime_utility,
            volatility_tier=volatility_tier,
            conditional_tail_risk=conditional_tail_risk,
            conditional_tail_risk_enhanced=conditional_tail_risk_enhanced,
            tail_correction_applied=tail_correction_applied,
            volatility_tier_ka=volatility_tier_ka,
            regime_conf=regime_conf,
            posterior_mode=posterior_mode,
            hmm_posteriors=_hmm_posteriors,
            hmm_trained_at=hmm_trained_at,
            hmm_artifact_version=hmm_artifact_version,
            hmm_pipeline_version=hmm_pipeline_version,
            hmm_calibration_provenance=calibration_provenance,
            funding_rate_decimal=funding_rate_decimal,
            persistence_prob=persistence_prob,
            long_short_ratio=long_short_ratio,
            funding_rate_zscore=funding_rate_zscore,
            open_interest_change_pct=open_interest_change_pct,
            discovery_geometry_filled=discovery_geometry_filled,
            discovery_geometry_reason=discovery_geometry_reason,
        )

    def _build_base_payload(rd: _RegimeData) -> dict:
        """Stage 7: Assemble base payload from regime data."""
        return {
            "hmm_range_prob": rd.hmm_range_prob,
            "hmm_trend_prob": rd.hmm_trend_prob,
            # Canonical post-enrich contract: bare fields are authoritative
            # once enrichment completes. Compatibility aliases are retained
            # only for downstream provenance / backward compatibility.
            "range_prob": rd.hmm_range_prob,
            "trend_prob": rd.hmm_trend_prob,
            "hmm_trained_at_utc": rd.hmm_trained_at,
            "hmm_artifact_version": rd.hmm_artifact_version,
            "hmm_pipeline_version": rd.hmm_pipeline_version,
            "hmm_posterior_mode": rd.posterior_mode,
            "hmm_calibration_status": rd.hmm_calibration_provenance.get("status"),
            "hmm_volatility_tier": rd.volatility_tier,
            "hmm_tail_var_95": rd.conditional_tail_risk.get("var_95"),
            "hmm_tail_cvar_95": rd.conditional_tail_risk.get("cvar_95"),
            "hmm_tail_var_99": rd.conditional_tail_risk.get("var_99"),
            "hmm_tail_cvar_99": rd.conditional_tail_risk.get("cvar_99"),
            "hmm_tail_prob_p01": rd.conditional_tail_risk.get("p_loss_le_global_p01"),
            # Cornish-Fisher tail risk (non-stationary enhancement)
            "hmm_cf_var_95": rd.conditional_tail_risk_enhanced.get("cf_var_95"),
            "hmm_cf_cvar_95": rd.conditional_tail_risk_enhanced.get("cf_cvar_95"),
            "hmm_cf_var_99": rd.conditional_tail_risk_enhanced.get("cf_var_99"),
            "hmm_cf_cvar_99": rd.conditional_tail_risk_enhanced.get("cf_cvar_99"),
            "hmm_weighted_kurtosis": rd.conditional_tail_risk_enhanced.get("weighted_excess_kurtosis"),
            "hmm_gaussian_divergence": rd.conditional_tail_risk_enhanced.get("weighted_gaussian_divergence"),
            "hmm_tail_correction": rd.tail_correction_applied,
            "hmm_volatility_tier_ka": rd.volatility_tier_ka,
            # AFML C3: New direct regime metrics
            "survival_prob": rd.survival_prob,
            "hurst_exponent": rd.hurst_exponent,
            "ou_halflife": rd.ou_halflife_feature,
            "ou_halflife_raw": rd.ou_halflife,
            "ou_halflife_feature_reason": rd.ou_halflife_feature_reason,
            "utility_score": rd.regime_utility,
            "regime_utility": rd.regime_utility,
            # Soft gating: regime confidence [0, 1]
            "regime_conf": rd.regime_conf,
            # afml_enriched_v20260318: HMM dominant-state persistence
            "persistence_prob": rd.persistence_prob,
            # Funding rate in decimal (e.g. 0.0001) for PnLRanker compatibility
            "funding_rate": rd.funding_rate_decimal,
            # Tier 4 context features
            "long_short_ratio": rd.long_short_ratio,
            "funding_rate_zscore": rd.funding_rate_zscore,
            "open_interest_change_pct": rd.open_interest_change_pct,
            # OU parameters propagated from scan output
            "ou_theta": rd.ou_theta,
            "ou_mu": rd.ou_mu,
            "ou_sigma": rd.ou_sigma,
            # Discovery-mode audit trail
            "discovery_geometry_filled": rd.discovery_geometry_filled,
            "discovery_geometry_reason": rd.discovery_geometry_reason,
        }

    def _build_meta_feature_row(
        scan_row: pd.Series | None,
        rd: _RegimeData,
        gd: _GridData,
        ev_score: float | None = None,
        md: _MicroData | None = None,
    ) -> dict[str, Any]:
        """Assemble the canonical meta-labeler feature row after grid generation."""
        feature_row: dict[str, Any] = {}
        if scan_row is not None:
            feature_row.update(scan_row.to_dict())

        feature_row.update(
            {
                "range_prob": rd.hmm_range_prob,
                "trend_prob": rd.hmm_trend_prob,
                "survival_prob": rd.survival_prob,
                "hurst_exponent": rd.hurst_exponent,
                "ou_halflife": rd.ou_halflife_feature,
                "utility_score": rd.regime_utility,
                "regime_utility": rd.regime_utility,
                "profit_per_grid_pct": getattr(gd.g, "profit_per_grid_pct", None),
                "num_grids": getattr(gd.g, "num_grids", None),
                "grid_spacing_pct": getattr(gd.g, "grid_spacing_pct", None),
                "range_size_pct": gd.range_size_pct,
                "funding_rate": rd.funding_rate_decimal,
                "hmm_tail_cvar_95": rd.conditional_tail_risk.get("cvar_95"),
                "persistence_prob": rd.persistence_prob,
                "regime_conf": rd.regime_conf,
                # ERR-059: ev_score (PnLRanker rank_score) is the meta-labeler's #1
                # feature. Supplying it here (computed before the probe) makes
                # meta_prob full-fidelity and AUTHORITATIVE at the deployment
                # decision; previously it was absent so meta_prob was always None.
                "ev_score": ev_score,
                # ERR-059 follow-up: micro_round_trip_cost_pct is one of the 20
                # contract features. It is computed at Stage 9 (md.ms_payload)
                # BEFORE this probe; thread it in so meta_prob is computable at
                # enrich time. Without it get_missing_feature_names() flagged it
                # missing and meta_prob stayed None -> Stage B data_missing:meta.
                # .get() keeps it absent (fail-closed) when microstructure
                # estimation genuinely failed (no key in the exception branch).
                "micro_round_trip_cost_pct": (
                    md.ms_payload.get("micro_round_trip_cost_pct")
                    if md is not None
                    else None
                ),
            }
        )
        return feature_row

    def _check_regime_rejection(rd: _RegimeData) -> tuple[str, str] | None:
        """Stage 8: Check regime validity. Returns (reason, rejection_reasons) or None if valid."""
        if not rd.vres.is_valid:
            # Collect ALL rejection reasons (not just the first)
            reject_codes: list[str] = []
            for tf_result in [rd.vres.tf_1h, rd.vres.tf_15m, rd.vres.tf_5m]:
                if tf_result and not tf_result.is_valid and tf_result.reason:
                    reject_codes.append(tf_result.reason)
            # Classify specific rejection types from metrics
            if rd.hmm_range_prob is not None and rd.hmm_range_prob < 0.20:
                reject_codes.append(f"low_range_prob({rd.hmm_range_prob:.3f})")
            if rd.hmm_trend_prob is not None and rd.hmm_trend_prob > 0.40:
                reject_codes.append(f"high_trend_prob({rd.hmm_trend_prob:.3f})")
            if rd.survival_prob is not None and rd.survival_prob < 0.50:
                reject_codes.append(f"low_survival({rd.survival_prob:.3f})")
            # ERR-092: annotate against the CONFIGURED hurst gate (0.65), not
            # the stale hardcoded 0.55 that no longer matched the actual gate.
            _stoch_cfg = get_config().stochastic
            if (
                rd.hurst_exponent is not None
                and rd.hurst_exponent > _stoch_cfg.hurst_max_trending
            ):
                reject_codes.append(f"hurst_trending({rd.hurst_exponent:.3f})")
            if rd.ou_halflife is not None:
                if rd.ou_halflife < _stoch_cfg.ou_halflife_min_bars:
                    reject_codes.append(f"ou_too_fast({rd.ou_halflife:.1f})")
                elif rd.ou_halflife > _stoch_cfg.ou_halflife_max_bars:
                    reject_codes.append(f"ou_too_slow({rd.ou_halflife:.1f})")
            if not rd.vres.data_quality_passed:
                reject_codes.append("data_quality_fail")
            if not reject_codes:
                reject_codes.append("regime_invalid")
            reason = reject_codes[0]
            return reason, "|".join(reject_codes)
        return None

    async def run_one(sym: str) -> tuple[str, Dict[str, Any]]:
        try:
            # Stages 1-6: Fetch and validate regime
            rd = await _fetch_regime_data(sym)

            # Propagate micro-osc scan-phase data into regime data
            scan_row: pd.Series | None = None
            _sym_rows = eligible.loc[eligible["symbol"] == sym]
            if not _sym_rows.empty:
                _scan_row = cast(pd.Series, _sym_rows.iloc[0])
                scan_row = _scan_row
                _apply_ou_params_from_scan_row(rd, _scan_row)
                if "micro_osc_score" in eligible.columns:
                    _mos = _scan_row.get("micro_osc_score")
                    if _mos is not None and pd.notna(_mos):
                        rd.micro_osc_score = float(_mos)
                if "micro_osc_bypass" in eligible.columns:
                    rd.micro_osc_bypass = bool(_scan_row.get("micro_osc_bypass"))
                if (
                    rd.micro_osc_bypass
                    and not bool(getattr(rd.vres, "is_valid", False))
                    and (
                        getattr(rd.vres, "range_high", None) is None
                        or getattr(rd.vres, "range_low", None) is None
                        or getattr(rd.vres, "current_price", None) is None
                        or getattr(rd.vres, "atr_1m", None) is None
                    )
                ):
                    rd.discovery_geometry_filled, rd.discovery_geometry_reason = (
                        _fill_discovery_geometry_from_market_data(
                            rd.vres,
                            rd.market_data,
                            validator,
                        )
                    )

            # Stage 7: Base payload
            base_payload = _build_base_payload(rd)

            # Stage 8: Regime validity check
            rejection = _check_regime_rejection(rd)
            regime_payload: dict[str, Any] = {
                "regime_would_reject": False,
                "regime_rejection_reasons": None,
            }
            if rejection is not None:
                reason, reject_reasons = rejection
                regime_payload = {
                    "regime_would_reject": True,
                    "regime_rejection_reasons": reject_reasons,
                }
                if not cfg.discovery_mode and not rd.micro_osc_bypass:
                    return sym, {
                        "grid_is_valid": False,
                        "grid_reason": reason,
                        "rejection_reasons": reject_reasons,
                        "failure_stage": "regime",
                        **base_payload,
                        **regime_payload,
                    }

            # Stage 9: Microstructure
            md = _estimate_microstructure(rd, sym)

            # Stage 10: Grid generation
            gd, grid_rejection = _generate_grid(rd, md, sym)
            # Grid generation may recalculate survival_prob from actual grid bounds.
            # Refresh the payload so all downstream returns reflect the corrected value.
            base_payload = _build_base_payload(rd)
            if grid_rejection is not None:
                return sym, {**grid_rejection, **base_payload, **regime_payload, **md.ms_payload, **gd.edge_info}

            # Stage 11: Viability
            _check_viability(gd, md, sym)

            # Stage 12: Kelly sizing
            # ERR-059: compute ev_score (PnLRanker rank_score; the meta-labeler's
            # first feature) BEFORE the meta probe, mirroring run_full_pipeline's
            # canonical scorer and the live monitor. Without it, ev_score was absent
            # here so meta_prob was always None and the promoted model never
            # influenced sizing/gating. leverage matches run_full_pipeline
            # (candidate leverage or 10), so the serve-time ev_score is on the same
            # basis as the training/ranking ev_score. ev_ranker does NOT consume
            # meta_prob (verified) -> no circularity.
            _ev_score: float | None = None
            _ev_contract_fingerprint = getattr(
                ev_ranker, "ev_contract_fingerprint", None
            )
            try:
                _ppg = getattr(gd.g, "profit_per_grid_pct", None)
                _ng = getattr(gd.g, "num_grids", None)
                if all(
                    v is not None and pd.notna(v)
                    for v in (_ppg, _ng, rd.survival_prob, rd.hmm_trend_prob, gd.range_size_pct)
                ):
                    # Narrow Optionals for the type checker (guarded by all() above),
                    # mirroring run_full_pipeline's scorer.
                    assert _ppg is not None and _ng is not None
                    _ev_res = ev_ranker.compute_score(
                        profit_per_grid_pct=float(_ppg),
                        num_grids=int(_ng),
                        survival_prob=float(rd.survival_prob),
                        trend_prob=float(rd.hmm_trend_prob),
                        funding_rate=(
                            float(rd.funding_rate_decimal)
                            if rd.funding_rate_decimal is not None
                            and pd.notna(rd.funding_rate_decimal)
                            else None
                        ),
                        range_size_pct=float(gd.range_size_pct),
                        symbol=sym,
                        leverage=int(gd.g.leverage or 10),
                    )
                    _ev_score = round(float(_ev_res.rank_score), 4)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("ev_score compute failed for %s: %s", sym, exc)

            meta_prob = None
            meta_prob_source = "missing"
            if meta_labeler is not None and getattr(meta_labeler, "is_trained", False):
                meta_features = _build_meta_feature_row(scan_row, rd, gd, ev_score=_ev_score, md=md)
                if not meta_labeler.get_missing_feature_names(meta_features):
                    meta_prob = float(meta_labeler.predict_proba(meta_features))
                    meta_prob_source = "enrich"

            # FASTWIN-01: meta_prob may size positions / gate deployment ONLY when
            # the loaded meta-labeler passed the promotion gate
            # (eval_metrics.promotion_status == "pass", surfaced as
            # MetaLabeler.promotion_status on load). Until then it stays
            # diagnostic-only (deployment_meta_prob=None) so Kelly and Stage B
            # behave exactly as before promotion. The string comparison is robust
            # to test doubles (a Mock compares unequal to "pass") and fail-closed
            # for legacy models without the field (None != "pass").
            meta_authoritative = (
                getattr(meta_labeler, "promotion_status", None) == "pass"
            )
            deployment_meta_prob = (
                meta_prob if (meta_authoritative and meta_prob is not None) else None
            )
            kelly_info, capital_fraction, sizing_reason = _compute_kelly(
                sym, gd, rd, deployment_meta_prob,
            )

            # Stages 13-14: Hard gates
            hg_payload, gate_rejection = _evaluate_gates(md, gd, sym)
            if gate_rejection is not None:
                hg_payload["hard_gate_would_reject"] = True
                hg_payload["hard_gate_rejection_reasons"] = gate_rejection.get("rejection_reasons")
                if not cfg.discovery_mode:
                    # ERR-073: record the VALIDATED range_size_pct on the
                    # hard-gate rejection path (same Gate 4 Fix 2b override the
                    # Stage 18 return applies). The Stage 12 meta probe consumes
                    # gd.range_size_pct; without this override the row keeps the
                    # scan-time BB estimate and the recorded meta_prob is not
                    # reproducible from the row's own feature columns.
                    _hg_range_pct: dict[str, Any] = {}
                    if gd.range_size_pct is not None:
                        _hg_range_pct["range_size_pct"] = round(float(gd.range_size_pct), 6)
                    return sym, {
                        **gate_rejection,
                        **base_payload,
                        **regime_payload,
                        **md.ms_payload,
                        **gd.edge_info,
                        **kelly_info,
                        **hg_payload,
                        **_hg_range_pct,
                        "failure_stage": "hard_gate",
                        "meta_prob": meta_prob,
                        "ev_contract_fingerprint": _ev_contract_fingerprint,
                        "meta_prob_source": meta_prob_source,
                        "meta_prob_authority": (
                            "authoritative" if meta_authoritative else "diagnostic_only"
                        ),
                    }
            else:
                hg_payload["hard_gate_would_reject"] = False
                hg_payload["hard_gate_rejection_reasons"] = None

            # Stages 15-16: Position sizing + TOS
            ps_payload, tos_payload, capital_fraction, sizing_suffix = _compute_position_and_tos(rd, gd, md, capital_fraction, sym)
            if sizing_suffix:
                sizing_reason += sizing_suffix

            # Stage 17: Stage B approval
            sb_payload, _grid_valid, _grid_reason, _reject_reasons = _evaluate_stage_b(
                rd, hg_payload, tos_payload, capital_fraction, deployment_meta_prob, sym,
            )

            # Stage 18: Final return
            # Gate 4 (Fix 2b): Write the authoritative validated range_size_pct
            # into the output, overriding the scan-time BB-based estimate.
            # The scan-time value is preserved in scan_range_size_pct (set by scan.py).
            _range_pct_payload: dict[str, Any] = {}
            if gd.range_size_pct is not None:
                _range_pct_payload["range_size_pct"] = round(float(gd.range_size_pct), 6)
            _survival_payload: dict[str, Any] = {
                "survival_prob": rd.survival_prob,
                "ou_theta": rd.ou_theta,
                "ou_mu": rd.ou_mu,
                "ou_sigma": rd.ou_sigma,
            }

            return sym, {
                "grid_is_valid": _grid_valid,
                **_grid_params_dict(gd.g),
                "grid_reason": _grid_reason,
                "rejection_reasons": _reject_reasons,
                "failure_stage": (
                    "approved"
                    if bool(sb_payload.get("stage_b_approved"))
                    else ("discovery" if cfg.discovery_mode and _grid_valid else "stage_b")
                ),
                "capital_fraction": round(capital_fraction, 4) if capital_fraction is not None else None,
                "sizing_reason": sizing_reason,
                **base_payload,
                **regime_payload,
                **md.ms_payload,
                **gd.edge_info,
                **kelly_info,
                **hg_payload,
                **ps_payload,
                **tos_payload,
                **sb_payload,
                **_range_pct_payload,
                **_survival_payload,
                "meta_prob": meta_prob,
                "ev_contract_fingerprint": _ev_contract_fingerprint,
                "meta_prob_source": meta_prob_source,
                "meta_prob_authority": (
                    "authoritative" if meta_authoritative else "diagnostic_only"
                ),
            }

        except Exception as e:
            logger.warning("Enrichment failed for %s: %s", sym, e, exc_info=True)
            _etype = type(e).__name__
            _emsg = str(e)[:80]
            return sym, {
                "grid_is_valid": False,
                "grid_reason": f"exception:{_etype}",
                "rejection_reasons": f"exception:{_etype}:{_emsg}",
                "failure_stage": "exception",
            }

    results = await asyncio.gather(*[run_one(s) for s in symbols], return_exceptions=True)
    updates: dict[str, dict] = {}
    for r in results:
        if isinstance(r, BaseException):
            logger.error("Unexpected gather exception: %s", r, exc_info=r)
            continue
        result_tuple: tuple[str, dict] = r  # type: ignore[assignment]
        updates[result_tuple[0]] = result_tuple[1]

    for i, row in df.iterrows():
        sym = str(row.get("symbol", ""))
        payload = updates.get(sym)
        if not payload:
            continue
        for k, v in payload.items():
            df.at[i, k] = v

    return _enforce_threshold_gate(df)
