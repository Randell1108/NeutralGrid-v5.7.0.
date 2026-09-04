"""
Meta-labeler for AFML-compliant binary classification.

The model supports generic meta-label training, but the active production
contract in this repo is the fast-winner target learned from the unified
snapshot/backtest backbone:

- label column: ``fast_winner_target``
- positive class: net MTM PnL first reaches +3% within seven hours
- output: ``meta_prob`` = probability of that fast-horizon target

Key concepts from AFML:
- Binary "bet sizing" signal (go/no-go)
- Calibrated probabilities via purged CV
- Feature engineering from regime and grid metrics

Design principles:
- Deterministic training with explicit random_state
- Purged CV to prevent look-ahead bias
- Interpretable feature importance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal, Tuple, cast
import numpy as np
import os
import json
import pandas as pd
import pickle
from pathlib import Path
import logging

from neutralgrid.core.constants import BOT_HORIZON_HOURS, MIN_POSITIVE_RATE

logger = logging.getLogger(__name__)


# ── Snapshot-only feature profile (29 features, v20260407) ─────────────
# Combined Candidate Feature Set: retained existing scan-time features (13)
# + Binance-derived crowding/flow features (9)
# + Binance-derived execution/liquidity features (7).
#
# All features must be fetchable from Binance Futures public endpoints
# or derivable only from Binance Futures fields at scan time.
SNAPSHOT_META_FEATURES_V20260407: tuple[str, ...] = (
    # ── Existing micro/range/volatility base (13) ──────────────────────
    "micro_osc_score",             # Micro-oscillator composite score
    "ema_crosses_5m",              # 5m EMA cross density
    "vwap_crosses_5m",             # 5m VWAP cross density
    "atr_pct_15m",                 # Normalized 15m ATR
    "bb_width",                    # 15m Bollinger Band width
    "bb_width_ratio_1h_15m",       # Multi-TF volatility compression ratio
    "range_size_pct",              # Local range as % of price
    "adx_5m",                      # 5m trend strength
    "adx_15m",                     # 15m trend strength
    "rsi_15m",                     # 15m RSI (boundedness context)
    "quote_volume_24h",            # 24h quote volume (liquidity proxy)
    "funding_rate",                # Last funding rate (carry cost)
    "open_interest",               # Open interest (positioning depth)
    # ── Binance-derived crowding / flow context (9) ────────────────────
    "top_account_lsr_log",         # ln(top-trader account long/short ratio)
    "top_position_lsr_log",        # ln(top-trader position long/short ratio)
    "top_position_vs_account_delta",  # Position vs account LSR delta
    "global_lsr_log",              # ln(global long/short ratio)
    "taker_imbalance",             # (buyVol - sellVol) / total
    "taker_buy_sell_log",          # ln(buySellRatio)
    "basis_pct",                   # (markPrice - indexPrice) / indexPrice * 100
    "funding_rate_zscore",         # z-scored funding rate
    "open_interest_change_pct",    # OI change % (new money flow)
    # ── Binance-derived execution / liquidity context (7) ──────────────
    "spread_pct",                  # (bestAsk - bestBid) / bestBid * 100
    "top20_bid_depth_usdt",        # Sum(price*qty) top-20 bids
    "top20_ask_depth_usdt",        # Sum(price*qty) top-20 asks
    "top20_depth_min_usdt",        # min(bid_depth, ask_depth)
    "book_imbalance_top20",        # (bid_depth - ask_depth) / total
    "oi_notional",                 # openInterest * markPrice
    "oi_to_volume",                # oi_notional / quote_volume_24h
)

SNAPSHOT_META_FEATURES_V20260420: tuple[str, ...] = (
    # Scanner/micro range context retained from v20260407.
    "micro_osc_score",
    "ema_crosses_5m",
    "vwap_crosses_5m",
    "atr_pct_15m",
    "bb_width",
    "bb_width_ratio_1h_15m",
    "range_size_pct",
    "adx_5m",
    "adx_15m",
    "rsi_15m",
    "quote_volume_24h",
    "funding_rate",
    "open_interest",
    # H*/regime and grid-geometry features restored for fast-winner learning.
    "range_prob",
    "trend_prob",
    "survival_prob",
    "utility_score",
    "ou_halflife",
    "profit_per_grid_pct",
    "num_grids",
    "grid_spacing_pct",
    "adx_1h",
    # Binance-derived crowding / flow context.
    "top_account_lsr_log",
    "top_position_lsr_log",
    "top_position_vs_account_delta",
    "global_lsr_log",
    "taker_imbalance",
    "taker_buy_sell_log",
    "basis_pct",
    "funding_rate_zscore",
    "open_interest_change_pct",
    # Binance-derived execution / liquidity context.
    "spread_pct",
    "top20_bid_depth_usdt",
    "top20_ask_depth_usdt",
    "top20_depth_min_usdt",
    "book_imbalance_top20",
    "oi_notional",
    "oi_to_volume",
)

SNAPSHOT_META_FEATURES_V20260421_BOOTSTRAP: tuple[str, ...] = (
    "range_prob",
    "survival_prob",
    "ou_halflife",
    "profit_per_grid_pct",
    "num_grids",
    "grid_spacing_pct",
    "adx_1h",
)

# FASTWIN-01 (2026-05-30): ex-ante, leakage-free, NON-HMM, circularity-free fast-winner
# feature set. Verified out-of-sample on the authoritative geometric backtest pool
# (n=256, n_pos=111): calibrated L2-logistic OOF AUC 0.704 [0.641, 0.761], ECE 0.056.
# Deliberately EXCLUDES: range_prob/trend_prob/persistence_prob/survival_prob (owned by the
# HMM winner calibrator + HMM-lineage dependent) and primary_pipeline_score/score/scan_score
# (CIRCULAR: the scanner score blends meta_prob via mi_weighted_scorer; guarded by
# tests/test_afml_compliance.py). ev_score is independent (PnLRanker.rank_score) and retained.
SNAPSHOT_META_FEATURES_V20260530_FASTWIN: tuple[str, ...] = (
    "ev_score",
    "adx_1h",
    "adx_15m",
    "adx_5m",
    "atr_pct_15m",
    "ou_halflife",
    "rsi_15m",
    "ema_slope_1h",
    "ema_crosses_5m",
    "vwap_crosses_5m",
    "hurst_exponent",
    "range_size_pct",
    "bb_width",
    "grid_spacing_pct",
    "profit_per_grid_pct",
    "num_grids",
    "quote_volume_24h",
    "open_interest",
    "micro_round_trip_cost_pct",
    "funding_rate",
)

ACTIVE_SNAPSHOT_META_FEATURES: tuple[str, ...] = SNAPSHOT_META_FEATURES_V20260530_FASTWIN
PROMOTION_EVALUATION_CONTRACT = (
    "chronological_contiguous_group_purged_weighted_kfold_v2"
)

# ── Active label contract — white-box theory ─────────────────────────────────
# The active meta-label estimates whether the simulated net MTM PnL curve first
# reaches +3% of capital within seven hours. The endogenous
# `time_to_target_hours` path is authoritative when present; legacy rows use
# the documented duration-window AND-label fallback. Active feature identity is
# `SNAPSHOT_META_FEATURES_V20260530_FASTWIN`, not an HMM-regime feature bundle.
# Historical validation belongs in versioned artifact metadata and audit
# reports; it is intentionally not restated here as current evidence.
#
# D1 — utility_score removed from active feature contract (2026-05-06):
# `utility_score` was dropped from `SNAPSHOT_META_FEATURES_V20260421_BOOTSTRAP`
# after empirical evidence that it actively misleads the model.  Receipts:
#   - Pearson(utility_score, pnl_pct) = -0.32 on n=22 manual winners (21/22
#     scored *negative* utility despite being profitable).
#   - Holdout AUC bootstrap 95% CI = [0.21, 0.80] straddles 0.5 (n=85, May 4).
#   - Permutation p-value = 0.45 — null of zero discrimination cannot be
#     rejected at any reasonable alpha.
#   - Variance decomposition: regime/structural inputs explain R²=0.65 of
#     utility_score variance; profitability explains R²=0.10 (6.3× gap).
#   - Grid optimizer collapses both penalty coefficients to corners every
#     refit (kappa→0.05 floor, lambda→4.0 ceiling).
#   - Adding 12 new labels (2026-05-05 ingestion: pool 85 → 97) flipped G4
#     from True to False — signal is too weak to survive small data shifts.
#
# Compositional rationale: per Lopez de Prado AFML §3 and Hudson & Thames
# "Does Meta-Labeling Add to Signal Efficacy?", the meta-labeler emits the
# calibrated probability used for sizing — it does NOT consume one as input.
# Feeding `utility_score` (itself a calibrated-decision-surface scalar over
# range_prob/trend_prob and grid metrics) created a circular double-
# calibration: the meta-model was being asked to recalibrate over a
# competing, mis-specified decision rule.  D1 restores compositional
# discipline: only primitive regime/geometry features feed the meta-labeler.
#
# Mis-specification hypothesis (open): `U = E[R] - lambda*E[DD] - kappa*E[BL]`
# is likely structurally wrong at *grid-bot scale*.  In directional
# strategies, drawdown predicts capital impairment; in a grid bot, transient
# drawdown is constitutive of inventory accumulation (the entry phase from
# which winners emerge).  Penalizing E[DD] therefore penalizes the very
# state from which fast winners arise — explaining the negative-correlation
# sign flip.  Future revival should re-derive utility for inventory-bearing
# market-making (e.g., Avellaneda–Stoikov-style penalty on terminal inventory
# variance), not just refit on more data with the same formula.
#
# Calibrator infrastructure is KEPT DORMANT (not deleted): `validation/
# utility.py`, `calibration/utility_calibrator.py`, `scripts/recalibrate_
# utility.py`, `artifacts/utility/`.  Runtime is fail-closed:
# `regime_validator.py` raises `UtilityCalibratorUnavailable` when
# `current.json` is absent — but `pass_mode="dominance"` already bypasses
# the utility gate, so this branch is dead code at runtime.  V20260420
# legacy models continue to load via `_INFERENCE_FEATURE_ALIASES`
# (regime_utility/scan_utility_score → utility_score) and the
# `_FEATURE_MEDIAN_DEFAULTS["utility_score"] = 0.0` fallback.
ACTIVE_META_TARGET_LABEL_COLUMN = "fast_winner_target"
# Required-present guard column for the fast-target frame. NOTE: with the
# PIPELINE_FIX v2 endogenous label active, the FAST measure is the engine's
# `time_to_target_hours` (first bar the net MTM PnL curve reaches +3%), not this
# column. `duration_hours` is retained as the presence guard both the endogenous
# and the legacy duration-window fallback paths rely on; do not repurpose it.
ACTIVE_META_TARGET_DURATION_COLUMN = "duration_hours"
ACTIVE_META_TARGET_DURATION_MAX_HOURS = 7.0
ACTIVE_META_TARGET_PNL_THRESHOLD_PCT = 3.0
# PIPELINE_FIX v2 (2026-06-08): the active fast-winner target is the ENDOGENOUS
# time-to-target — a candidate is a fast winner iff its backtest reaches +3% net
# MTM PnL within 7h (`time_to_target_hours <= 7.0`). Legacy rows lacking
# `time_to_target_hours` fall back to the duration-window AND-label. Validation
# metrics are versioned with the model and holdout audit, not this source file.
ACTIVE_META_TARGET_CONTRACT = "fast_winner_time_to_3pct_le_7h"

META_FEATURE_PROFILES: dict[str, tuple[str, ...]] = {
    "snapshot_v20260407": SNAPSHOT_META_FEATURES_V20260407,
    "snapshot_v20260420": SNAPSHOT_META_FEATURES_V20260420,
    "snapshot_v20260421_bootstrap": SNAPSHOT_META_FEATURES_V20260421_BOOTSTRAP,
    "snapshot_v20260530_fastwin": SNAPSHOT_META_FEATURES_V20260530_FASTWIN,
}

# Runtime artifacts use a few synonymous field names. Normalize them here so
# inference sees the same canonical feature schema that training used.
_INFERENCE_FEATURE_ALIASES: dict[str, str] = {
    # Scan-prefixed aliases for features in snapshot_v20260407 profile
    "scan_range_size_pct": "range_size_pct",
    "scan_adx_15m": "adx_15m",
    "scan_adx_5m": "adx_5m",
    "scan_rsi_15m": "rsi_15m",
    "scan_ema_crosses_5m": "ema_crosses_5m",
    "scan_vwap_crosses_5m": "vwap_crosses_5m",
    "scan_atr_pct_15m": "atr_pct_15m",
    "scan_quote_volume_24h": "quote_volume_24h",
    "scan_open_interest": "open_interest",
    "scan_bb_width": "bb_width",
    "scan_funding_rate": "funding_rate",
    # Binance-derived feature aliases
    "scan_top_account_lsr_log": "top_account_lsr_log",
    "scan_top_position_lsr_log": "top_position_lsr_log",
    "scan_top_position_vs_account_delta": "top_position_vs_account_delta",
    "scan_global_lsr_log": "global_lsr_log",
    "scan_taker_imbalance": "taker_imbalance",
    "scan_taker_buy_sell_log": "taker_buy_sell_log",
    "scan_basis_pct": "basis_pct",
    "scan_funding_rate_zscore": "funding_rate_zscore",
    "scan_open_interest_change_pct": "open_interest_change_pct",
    "scan_spread_pct": "spread_pct",
    "scan_top20_bid_depth_usdt": "top20_bid_depth_usdt",
    "scan_top20_ask_depth_usdt": "top20_ask_depth_usdt",
    "scan_top20_depth_min_usdt": "top20_depth_min_usdt",
    "scan_book_imbalance_top20": "book_imbalance_top20",
    "scan_oi_notional": "oi_notional",
    "scan_oi_to_volume": "oi_to_volume",
    "scan_bb_width_ratio_1h_15m": "bb_width_ratio_1h_15m",
    "scan_micro_osc_score": "micro_osc_score",
    # PIPELINE_FIX C.2: scan_* aliases for the 8 active FASTWIN-profile features the
    # live-outcome ingestor emits scan-prefixed (live_outcome_ingestor.py:513) but
    # that previously lacked a canonical alias. Required so hybrid-pool live rows
    # expose the active features under canonical names BEFORE completeness filtering,
    # using the same shared normalizer as inference (no train/serve drift).
    "scan_ev_score": "ev_score",
    "scan_adx_1h": "adx_1h",
    "scan_ou_halflife": "ou_halflife",
    "scan_ema_slope_1h": "ema_slope_1h",
    "scan_hurst_exponent": "hurst_exponent",
    "scan_profit_per_grid_pct": "profit_per_grid_pct",
    "scan_num_grids": "num_grids",
    "scan_micro_round_trip_cost_pct": "micro_round_trip_cost_pct",
    "hmm_range_prob": "range_prob",
    "hmm_trend_prob": "trend_prob",
    "regime_utility": "utility_score",
    "scan_utility_score": "utility_score",
    "scan_grid_spacing_pct": "grid_spacing_pct",
}

# Per-feature median defaults for legacy models without a fitted imputer.
# Using domain-appropriate medians avoids the distortion of zero-imputation
# (e.g., range_prob=0.0 is an extreme bearish signal, not a neutral default).
_FEATURE_MEDIAN_DEFAULTS: dict[str, float] = {
    "range_prob": 0.5,
    "trend_prob": 0.25,
    "survival_prob": 0.5,
    "profit_per_grid_pct": 0.3,
    "num_grids": 10.0,
    "grid_spacing_pct": 0.75,
    "range_size_pct": 5.0,
    "adx_1h": 25.0,
    "adx_15m": 25.0,
    "adx_5m": 25.0,
    "rsi_15m": 50.0,
    "funding_rate": 0.0001,
    "ema_slope_1h": 0.0,
    "ema_crosses_5m": 0.0,
    "vwap_crosses_5m": 0.0,
    "atr_pct_15m": 0.005,
    "quote_volume_24h": 50_000_000.0,
    "utility_score": 0.0,
    "hurst_exponent": 0.5,
    "ou_halflife": 24.0,
    "bb_width": 0.03,
    "open_interest": 50_000_000.0,
    "micro_round_trip_cost_pct": 0.15,
    # FASTWIN-01: ev_score is the PnLRanker expected-value edge (rank_score),
    # roughly zero-centered; neutral default for legacy models without a fitted imputer.
    "ev_score": 0.0,
    "regime_conf": 0.5,
    "hmm_tail_cvar_95": -0.04,
    "tos_gcf": 0.3,
    "persistence_prob": 0.85,
    # Tier 4 context features
    "long_short_ratio": 1.0,
    "funding_rate_zscore": 0.0,
    "open_interest_change_pct": 0.0,
    "bb_width_ratio_1h_15m": 1.0,
    # Micro-oscillator archetype
    "micro_osc_score": 0.0,
    # Binance-derived crowding / flow context
    "top_account_lsr_log": 0.0,
    "top_position_lsr_log": 0.0,
    "top_position_vs_account_delta": 0.0,
    "global_lsr_log": 0.0,
    "taker_imbalance": 0.0,
    "taker_buy_sell_log": 0.0,
    "basis_pct": 0.0,
    # Binance-derived execution / liquidity context
    "spread_pct": 0.05,
    "top20_bid_depth_usdt": 500_000.0,
    "top20_ask_depth_usdt": 500_000.0,
    "top20_depth_min_usdt": 500_000.0,
    "book_imbalance_top20": 0.0,
    "oi_notional": 50_000_000.0,
    "oi_to_volume": 1.0,
}

# Columns that carry label information and must NEVER be used as model features.
# hlabel encodes the hierarchical label level (0-3) — hlabel==3 ⟺ y==1,
# so including it would be perfect information leakage.
_KNOWN_LABEL_COLUMNS: frozenset[str] = frozenset({
    "hlabel", "hlabel_meta", "hlabel_L1", "hlabel_L2", "hlabel_L3", "hlabel_reason",
    "y", "y_horizon", "label", "pnl_pct", "net_pnl_pct", "sl_hit",
    "realized_net_pnl_pct", "realized_pnl_per_margin_hour",
    "rotation_score", "unrealized_fraction",
    # PIPELINE_FIX A.1: the fast-winner target is derived from these outcome
    # columns (duration_hours <= 7h AND pnl >= 3). They are labels, never
    # features — guarded here so the AND-label retrain (Phase B) and any hybrid
    # live-outcome pool (Phase C) cannot leak the target into the feature matrix.
    "duration_hours", "fast_winner_target",
    # PIPELINE_FIX v2: endogenous time-to-target + reached flag are outcome/label
    # quantities (post-hoc, from the realized PnL curve) — NEVER ex-ante features.
    "time_to_target_hours", "target_reached",
    # ERR-079: remaining outcome-derived columns present in the fastwin training
    # CSVs. None is in any feature profile today; guarded so a future profile
    # naming one cannot silently leak the realized path into the feature matrix.
    "barrier_price", "barrier_touched", "horizon_censored",
    "mae", "mae_pct_initial", "mfe", "mfe_pct_initial",
    "realized_net_pnl",
    "t1", "t1_is_synthetic", "t1_truncated",
    "sample_weight_override",
})

# ERR-079: outcome-derived column FAMILIES guarded by prefix — every member is a
# post-hoc quantity from the realized PnL curve / hierarchical label detail and
# must never enter the feature matrix, including members added in the future.
_LABEL_COLUMN_PREFIXES: tuple[str, ...] = ("hlabel_detail_", "pnl_curve_")


def _is_label_column(name: str) -> bool:
    """True when a column carries label/outcome information (leakage guard)."""
    return name in _KNOWN_LABEL_COLUMNS or name.startswith(_LABEL_COLUMN_PREFIXES)


def normalize_inference_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fill canonical inference feature names from known runtime aliases."""
    Xdf = df.copy()
    for alias, canonical in _INFERENCE_FEATURE_ALIASES.items():
        if alias not in Xdf.columns:
            continue
        if canonical not in Xdf.columns:
            Xdf[canonical] = Xdf[alias]
            continue
        canonical_na = cast(pd.Series, Xdf[canonical].isna())
        alias_notna = cast(pd.Series, Xdf[alias].notna())
        fill_mask = canonical_na & alias_notna
        if bool(fill_mask.any()):
            Xdf.loc[fill_mask, canonical] = Xdf.loc[fill_mask, alias]
    return Xdf


def normalize_inference_feature_dict(values: Dict[str, Any]) -> Dict[str, Any]:
    """Return one runtime row with canonical inference feature names filled."""
    if not values:
        return {}
    normalized = normalize_inference_feature_frame(pd.DataFrame([dict(values)]))
    return cast(Dict[str, Any], normalized.iloc[0].to_dict())


def resolve_feature_profile(
    profile: str,
    available_columns: Optional[List[str]] = None,
) -> List[str]:
    """Resolve a named feature profile, optionally filtered to available columns."""
    if profile not in META_FEATURE_PROFILES:
        raise ValueError(
            f"Unknown feature profile '{profile}'. "
            f"Available: {sorted(META_FEATURE_PROFILES)}"
        )
    features = list(META_FEATURE_PROFILES[profile])
    if available_columns is None:
        return features
    available = set(available_columns)
    return [feat for feat in features if feat in available]


def apply_feature_profile_defaults(
    values: Dict[str, Any],
    *,
    feature_names: Optional[List[str]] = None,
) -> tuple[Dict[str, Any], List[str]]:
    """Fill missing scan-time feature values with the explicit model contract."""
    feature_list = list(feature_names) if feature_names is not None else list(ACTIVE_SNAPSHOT_META_FEATURES)
    filled = dict(values)
    defaulted: List[str] = []

    for feat in feature_list:
        value = filled.get(feat)
        if value is None or bool(pd.isna(value)):
            filled[feat] = _FEATURE_MEDIAN_DEFAULTS[feat]
            defaulted.append(feat)

    return filled, defaulted


class _PostProbabilityCalibrator:
    """
    Lightweight wrapper that applies optional post-model probability calibration.

    The base model can itself be calibrated (e.g., temporal holdout CalibratedClassifierCV).
    This wrapper adds an extra OOS calibrator learned from pooled CV predictions.
    """

    def __init__(
        self,
        base_model: Any,
        calibrator: Any = None,
        method: str | None = None,
    ):
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method

    def _apply(self, proba: np.ndarray) -> np.ndarray:
        p = np.asarray(proba, dtype=float).reshape(-1)
        if self.calibrator is None or self.method is None:
            return np.clip(p, 0.0, 1.0)
        if self.method == "isotonic_oos":
            p = np.asarray(self.calibrator.predict(p), dtype=float)
        elif self.method == "sigmoid_oos":
            p = np.asarray(
                self.calibrator.predict_proba(p.reshape(-1, 1)), dtype=float
            )[:, 1]
        elif self.method == "beta_oos":
            # BetaCalibrator uses predict_proba(X) → (N, 2) interface
            p = np.asarray(
                self.calibrator.predict_proba(p.reshape(-1, 1)), dtype=float
            )[:, 1]
        return np.clip(p, 0.0, 1.0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        base = np.asarray(self.base_model.predict_proba(X), dtype=float)[:, 1]
        cal = self._apply(base)
        return np.column_stack([1.0 - cal, cal])


@dataclass(frozen=True)
class MetaLabelerConfig:
    """
    Configuration for meta-labeler training.

    AFML Compliance:
    - Supports sample weighting based on concurrency/uniqueness
    - CV-aware imputation (fit on train fold only)
    - Probability calibration for reliable probability estimates
    """

    # Label definition
    hurdle_pct: float = 3.0       # 3% PnL hurdle for "success" (percentage units)
    sl_pct: float = -12.0         # -12% stop loss threshold (percentage units)
    horizon_hours: float = BOT_HORIZON_HOURS   # Prediction horizon (canonical H*)

    # Model hyperparameters
    n_estimators: int = 50
    max_depth: int = 3
    min_samples_split: int = 10
    min_samples_leaf: int = 15
    learning_rate: float = 0.05

    # CV settings
    cv_folds: int = 6
    purge_pct: float = 0.02
    embargo_pct: float = 0.01

    # Time-based CPCV controls (preferred when t1 is present)
    # If purge_hours is None, MetaLabeler defaults it to horizon_hours.
    purge_hours: float | None = 12.0
    embargo_hours: float = 1.5

    # AFML sample weighting
    use_sample_weights: bool = True  # Use concurrency-based sample weights
    use_time_decay: bool = True      # Use time-decay weights (N_eff<30 guard in sample_weights.py)
    time_decay_halflife_days: float = 30.0
    use_class_weights: bool = True   # Apply balanced class weights via sample_weight

    # Probability calibration (AFML recommended)
    calibrate_probabilities: bool = True
    calibration_method: str = "sigmoid"  # "isotonic" or "sigmoid"
    calibration_holdout_pct: float = 0.20  # Fraction of data held out for calibration
    oos_calibration_min_samples: int = 30
    oos_calibration_max_ece_degradation: float = 0.01
    oos_calibration_max_brier_degradation: float = 0.002
    oos_calibration_min_improvement: float = 0.0

    # Sequential bootstrap bagging (AFML Ch 6)
    use_sequential_bootstrap: bool = False  # When True, use DecisionTree on sequentially-bootstrapped data

    # Base estimator selection (FASTWIN-01). "gbm" = GradientBoostingClassifier
    # (legacy bootstrap profile), "logistic" = L2-regularized LogisticRegression
    # (calibrated; the verified best for the fast-winner profile at this N — complexity
    # is controlled by L2 regularization rather than feature truncation). Ignored when
    # use_sequential_bootstrap is True (DecisionTree path takes precedence).
    # "vote_logit_hgb" (FASTWIN-02, 2026-07-04) = soft-vote of the FASTWIN-01
    # logistic and a strongly-regularized HistGradientBoostingClassifier; the
    # per-fold sigmoid calibrates the BLENDED score, which is what keeps the
    # promotion-gate ECE <= 0.10 (raw HGB OOF fails it at ~0.13). Verified on
    # the 6048-row fast-winner pool: promotion-OOF AUC 0.807 vs 0.769 logistic
    # (paired dAUC +0.036..+0.041 across 5 fold seeds, CI-low > 0), and
    # +0.022 [0.017, 0.026] under time-blocked purged folds (5/5 eras >= 0).
    estimator_type: str = "gbm"

    # FASTWIN-02 HGB leg hyperparameters (vote_logit_hgb only). Deliberately
    # over-regularized for n~6k: gentle learning rate, few leaves, large leaf
    # minimum, strong L2. Do not loosen without re-running the paired
    # shuffled + time-blocked challenger evaluation.
    hgb_learning_rate: float = 0.03
    hgb_max_leaf_nodes: int = 10
    hgb_max_iter: int = 350
    hgb_min_samples_leaf: int = 60
    hgb_l2_regularization: float = 3.0
    hgb_max_features: float = 0.8

    # Random state for reproducibility
    random_state: int = 42

    # Feature completeness gate — prevents silent regression to imputed noise.
    # Per-feature NaN rates above warn_pct are logged; above block_pct halt
    # training with a clear error pointing to snapshot regeneration or
    # explicit profile reduction.
    feature_nan_warn_pct: float = 30.0   # warn if any feature > 30% NaN
    feature_nan_block_pct: float = 95.0  # block if any feature > 95% NaN

    # Feature list
    features: List[str] = field(default_factory=lambda: list(ACTIVE_SNAPSHOT_META_FEATURES))


@dataclass
class MetaLabelerMetrics:
    """Training metrics for meta-labeler."""

    auc_cv: float               # Cross-validated AUC
    precision_at_5: float       # Precision at top 5
    f1_threshold: float         # Optimal threshold maximizing F1
    f1_score: float             # F1 at optimal threshold
    feature_importance: Dict[str, Any]
    train_samples: int
    positive_rate: float        # Fraction of positive labels
    calibration_method: Optional[str] = None
    calibration_ece_raw: Optional[float] = None
    calibration_ece_calibrated: Optional[float] = None
    calibration_brier_raw: Optional[float] = None
    calibration_brier_calibrated: Optional[float] = None
    calibration_oos_accepted: Optional[bool] = None
    calibration_oos_reject_reason: Optional[str] = None
    label_pnl_column: str = "pnl_pct"
    # FASTWIN-01 promotion gate — computed by _evaluate_promotion_oof on a
    # study-faithful symbol-grouped, time-purged, all-rows-scored OOF with
    # calibrated per-fold predictions (the methodology that validated the
    # 0.50/0.10 thresholds), independent of the model's training CPCV.
    oof_auc: Optional[float] = None
    oof_auc_ci_low: Optional[float] = None
    oof_auc_ci_high: Optional[float] = None
    oof_ece: Optional[float] = None
    # ERR-075 decision-population telemetry.  These metrics are evaluated on
    # the same calibrated OOF probabilities as the headline promotion metric,
    # but only for rows with capital_fraction > 0.  They are diagnostic-only:
    # the promotion gate continues to use the all-row fields above.
    oof_deployable_auc: Optional[float] = None
    oof_deployable_ece: Optional[float] = None
    oof_deployable_n: Optional[int] = None
    oof_deployable_n_pos: Optional[int] = None
    oof_deployable_positive_rate: Optional[float] = None
    oof_deployable_definition: Optional[str] = None
    n_pos: Optional[int] = None          # positive training events (gate input b)
    promotion_status: Optional[str] = None      # "pass" | "fail" | None (untrained)
    promotion_reasons: Optional[List[str]] = None


class MetaLabeler:
    """
    Binary classifier for meta-labeling grid bot candidates.

    Predicts the active production meta target probability.

    In this repo's deployed contract, ``meta_prob`` means the probability that
    the candidate becomes a fast winner under the current retrain target
    (FASTWIN-01): ``duration_hours <= 7`` and ``net_pnl_pct >= 3`` (the
    effective-PnL fast-winner hurdle, ACTIVE_META_TARGET_PNL_THRESHOLD_PCT).

    The class still supports generic label construction when ``train()`` is
    called without an explicit ``y_col``, but production retraining uses the
    explicit ``fast_winner_target`` label built upstream.

    Training:
        labeler = MetaLabeler(MetaLabelerConfig())
        metrics = labeler.train(df_historical, timestamp_col="start_time_utc")

    Inference:
        proba = labeler.predict_proba(feature_dict)
    """

    def __init__(self, config: Optional[MetaLabelerConfig] = None):
        """
        Initialize MetaLabeler.

        Args:
            config: MetaLabelerConfig with hyperparameters
        """
        self.config = config or MetaLabelerConfig()
        self._model = None
        self._feature_names: List[str] = []
        self._is_trained = False
        self._is_calibrated = False
        self._metrics: Optional[MetaLabelerMetrics] = None
        self._imputer = None  # CV-aware imputer saved for inference
        self._oos_calibration_method: str | None = None
        self._oos_calibration_report: Dict[str, Any] = {}
        self._label_pnl_col_used: str = "pnl_pct"
        # FASTWIN-01 promotion authority. None until trained/loaded; set to
        # "pass"/"fail" by the promotion gate. Decision-time consumers must treat
        # anything other than "pass" (incl. None) as NOT authoritative.
        self.promotion_status: Optional[str] = None
        # FASTWIN-01 recall-tuning telemetry: OOF recall at fixed precision
        # floors + operating thresholds, recorded by _evaluate_promotion_oof.
        # Consumed offline to choose two_stage.min_meta_prob. None until trained.
        # Never affects the promotion verdict or any decision-time behavior.
        self._last_oof_operating_points: Optional[Dict[str, Any]] = None
        # ERR-075 diagnostic population metrics.  Kept separate from the
        # method's return tuple so callers cannot accidentally substitute them
        # for the governed all-row promotion inputs.
        self._last_oof_deployable_metrics: Optional[Dict[str, Any]] = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def is_promoted(self) -> bool:
        """True only when the FASTWIN-01 promotion gate passed (fail-closed)."""
        return self.promotion_status == "pass"

    @property
    def metrics(self) -> Optional[MetaLabelerMetrics]:
        return self._metrics

    def create_labels(
        self,
        df: pd.DataFrame,
        pnl_col: str = "pnl_pct",
        sl_hit_col: Optional[str] = "sl_hit",
    ) -> pd.Series:
        """
        Create binary labels for meta-labeling.

        Label = 1 if:
        - pnl_pct >= hurdle_pct AND
        - Did not hit stop loss (sl_hit != True)

        Args:
            df: DataFrame with performance data
            pnl_col: Column name for PnL percentage
            sl_hit_col: Column name for stop loss hit flag (optional)

        Returns:
            Binary label series (1 = success, 0 = failure)
        """
        label_col = pnl_col
        has_net_pnl = (
            "net_pnl_pct" in df.columns
            and bool(np.asarray(df["net_pnl_pct"].notna()).any())
        )
        if (
            pnl_col == "pnl_pct"
            and has_net_pnl
        ):
            label_col = "net_pnl_pct"
            logger.info(
                "Using net_pnl_pct for meta labels to align with post-cost profitability objective"
            )
        self._label_pnl_col_used = label_col

        # Sanity check: warn if values look like decimals rather than percentages.
        pnl_values = df[label_col].dropna()
        if len(pnl_values) > 0 and pnl_values.abs().max() < 1.0:
            logger.warning(
                f"pnl_col '{label_col}' max absolute value is {pnl_values.abs().max():.4f}. "
                f"Values appear to be in decimal format. Expected percentage format (e.g., 5.0 for 5%)."
            )

        # PnL above hurdle
        pnl_ok = df[label_col] >= self.config.hurdle_pct

        # Did not hit stop loss (if column exists)
        if sl_hit_col and sl_hit_col in df.columns:
            sl_ok = ~df[sl_hit_col].fillna(False).astype("boolean")
        else:
            # Infer from PnL: assume SL hit if pnl <= sl_pct
            sl_ok = df[label_col] > self.config.sl_pct

        return (pnl_ok & sl_ok).astype(int)

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix from DataFrame.

        Training-serving alignment:
        - During inference, uses the feature order from training.
        - Uses the fitted imputer when available (no ad-hoc填充).
        """
        if df is None or df.empty:
            raise ValueError('Empty features DataFrame')

        Xdf = self._normalize_inference_frame(df)

        # Determine feature set
        if self._feature_names:
            feats = list(self._feature_names)
        else:
            feats = [f for f in self.config.features if f in Xdf.columns]

        # Guard: strip any label columns that would cause information leakage
        leaked = {f for f in feats if _is_label_column(f)}
        if leaked:
            logger.warning("Dropping label columns from features: %s", sorted(leaked))
            feats = [f for f in feats if not _is_label_column(f)]

        if len(feats) < 3:
            raise ValueError(f'Not enough features available. Found: {feats}')

        # Build aligned frame
        for f in feats:
            if f not in Xdf.columns:
                Xdf[f] = np.nan

        Xdf = Xdf[feats].copy()
        for col in Xdf.columns:
            Xdf[col] = pd.to_numeric(Xdf[col], errors='coerce')

        Xdf = Xdf.replace([np.inf, -np.inf], np.nan)
        X_raw = np.asarray(Xdf)

        if self._imputer is not None:
            X = np.asarray(self._imputer.transform(X_raw))
        else:
            # Fallback for legacy models without fitted imputer: use per-feature
            # median defaults instead of zero (which distorts bounded features).
            logger.warning("Using median-default imputation (no fitted imputer)")
            fill_values = np.array(
                [_FEATURE_MEDIAN_DEFAULTS.get(f, 0.0) for f in feats],
                dtype=float,
            )
            X = X_raw.copy()
            for col_idx in range(X.shape[1]):
                mask = ~np.isfinite(X[:, col_idx])
                X[mask, col_idx] = fill_values[col_idx]

        return X, feats

    def _normalize_inference_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill canonical feature names from known runtime aliases."""
        return normalize_inference_feature_frame(df)

    def get_missing_feature_names(self, features: Dict[str, Any]) -> List[str]:
        """Return canonical feature names that are absent or NaN for inference."""
        names = list(self._feature_names) if self._feature_names else list(self.config.features)
        Xdf = self._normalize_inference_frame(pd.DataFrame([dict(features)]))
        row = Xdf.iloc[0]
        missing: List[str] = []
        for name in names:
            value = row.get(name, np.nan)
            if value is None:
                missing.append(name)
                continue
            # ERR-054: fail-closed on malformed (container / array-like) feature
            # values. A raw Binance funding-rate LIST or open-interest DICT makes
            # ``pd.isna(value)`` return an array, and ``x or <array>`` raises
            # "truth value of an array ... is ambiguous", aborting the whole probe.
            # A valid inference feature is a scalar; treat anything else as missing.
            if isinstance(value, (list, tuple, set, dict, np.ndarray, pd.Series)) or np.ndim(value) != 0:
                missing.append(name)
                continue
            if pd.isna(value):
                missing.append(name)
        return missing

    @staticmethod
    def _expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute expected calibration error (ECE)."""
        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_prob, dtype=float).reshape(-1)
        if len(yt) == 0:
            return 0.0
        bins = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
        idx = np.digitize(yp, bins, right=True) - 1
        idx = np.clip(idx, 0, len(bins) - 2)
        ece = 0.0
        n = float(len(yt))
        for b in range(len(bins) - 1):
            mask = idx == b
            if not np.any(mask):
                continue
            frac = float(np.sum(mask)) / n
            conf = float(np.mean(yp[mask]))
            acc = float(np.mean(yt[mask]))
            ece += abs(acc - conf) * frac
        return float(ece)

    @staticmethod
    def _bootstrap_auc_ci(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_boot: int = 1000,
        random_state: int = 42,
    ) -> tuple[Optional[float], Optional[float]]:
        """Percentile bootstrap 95% CI for a single-model ROC-AUC.

        FASTWIN-01 promotion gate input. Resamples (with replacement) the
        scored rows; resamples lacking both classes are skipped. Returns
        (None, None) — fail-closed — when fewer than 8 rows, a single class,
        or fewer than 50 valid resamples are available, so a degenerate CI can
        never be read as "excludes 0.50".
        """
        from sklearn.metrics import roc_auc_score

        yt = np.asarray(y_true, dtype=int).reshape(-1)
        yp = np.asarray(y_prob, dtype=float).reshape(-1)
        if len(yt) < 8 or len(np.unique(yt)) < 2:
            return None, None
        rng = np.random.default_rng(int(random_state))
        n = len(yt)
        idx = np.arange(n)
        aucs: List[float] = []
        for _ in range(int(n_boot)):
            bi = rng.choice(idx, n, replace=True)
            if len(np.unique(yt[bi])) < 2:
                continue
            aucs.append(float(roc_auc_score(yt[bi], yp[bi])))
        if len(aucs) < 50:
            return None, None
        return (
            float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)),
        )

    @staticmethod
    def _evaluate_promotion_gate(
        *,
        oof_auc_ci_low: Optional[float],
        oof_auc_ci_high: Optional[float],
        n_pos: Optional[int],
        oof_ece: Optional[float],
        min_n_pos: int = 70,
        max_ece: float = 0.10,
    ) -> tuple[str, List[str]]:
        """FASTWIN-01 promotion gate (fail-closed).

        Promote (status ``"pass"``) iff ALL three hold; otherwise ``"fail"``
        with a reason per failed/missing criterion:
          (a) the purged OOF-AUC 95% bootstrap CI excludes 0.50 — interpreted as
              the LOWER bound > 0.50 (a one-sided discrimination-improvement
              requirement; deliberate, do not "fix" to two-sided),
          (b) ``n_pos >= 70`` positive training events,
          (c) pooled OOF ECE <= 0.10.
        Any missing/non-finite metric appends a reason and forces ``"fail"`` —
        a model is never promoted on absent evidence.
        """
        reasons: List[str] = []
        if oof_auc_ci_low is None or oof_auc_ci_high is None:
            reasons.append("auc_ci_unavailable")
        elif not (oof_auc_ci_low > 0.50):
            reasons.append(f"auc_ci_includes_0.50(low={oof_auc_ci_low:.3f})")
        if n_pos is None:
            reasons.append("n_pos_unavailable")
        elif int(n_pos) < int(min_n_pos):
            reasons.append(f"n_pos_lt_{int(min_n_pos)}(n_pos={int(n_pos)})")
        if oof_ece is None or not np.isfinite(oof_ece):
            reasons.append("ece_unavailable")
        elif float(oof_ece) > float(max_ece):
            reasons.append(f"ece_gt_{max_ece:.2f}(ece={float(oof_ece):.3f})")
        return ("pass" if not reasons else "fail"), reasons

    def _evaluate_promotion_oof(
        self,
        X_raw: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray],
        start: Optional[np.ndarray],
        t1: Optional[np.ndarray],
        estimator_factory,
        deployable_mask: Optional[np.ndarray] = None,
        n_splits: int = 5,
        n_boot: int = 1000,
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Study-faithful promotion-gate OOF evaluation (FASTWIN-01).

        The promotion gate is judged with the SAME methodology that validated its
        thresholds (``scripts/meta_labeler_feature_study.py``): a symbol-grouped,
        time-purged K-fold that scores EVERY row, with per-fold CALIBRATED
        predictions (``CalibratedClassifierCV``, sigmoid). It therefore measures
        discrimination (AUC + bootstrap CI) and calibration (ECE) on the same
        footing the 0.50/0.10 cut-offs were derived on.

        This is deliberately SEPARATE from the model's training CPCV. CPCV is the
        conservative scheme the deployed model + OOS calibrator are built on, but
        on small symbol-diverse pools its symbol grouping degenerates (verified:
        256 rows / 100 symbols -> 3 splits, only 135 rows scored), which is a
        property of the CV, not the signal. This gate evaluation scores all rows
        so promotion reflects the model's true out-of-sample quality.

        Leakage safety is preserved: imputer / scaler / estimator are fit on the
        train fold only. Train rows whose event window ``[start, t1]`` overlaps
        the test window are purged before fitting WHEN the purge leaves > 15
        train rows; with shuffled stratified folds each test fold typically
        spans the full data time range, so the purge usually collapses and the
        fold trains unpurged (logged at WARNING with the affected fraction —
        symbol grouping remains the operative leakage guard; oof_auc may be
        modestly optimistic from cross-symbol temporal concurrency). Returns
        ``(oof_auc, ci_low, ci_high, oof_ece)``; all ``None`` (fail-closed) when
        the pool is too small / single-class to evaluate.  When
        ``deployable_mask`` is supplied, a decision-population AUC/ECE summary
        is recorded on ``_last_oof_deployable_metrics`` for metadata only.  It
        never changes this return tuple or the promotion verdict.
        """
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import roc_auc_score

        try:
            from sklearn.model_selection import (
                StratifiedGroupKFold,
                StratifiedKFold,
            )
        except Exception:  # pragma: no cover - sklearn always provides these
            return None, None, None, None

        self._last_oof_deployable_metrics = None
        yv = np.asarray(y, dtype=int).reshape(-1)
        Xv = np.asarray(X_raw, dtype=float)
        n = len(yv)
        if n < 30 or len(np.unique(yv)) < 2:
            return None, None, None, None

        rs = int(self.config.random_state)
        n_splits = int(max(2, min(n_splits, int(np.bincount(yv).min()))))
        if (
            groups is not None
            and len({str(g) for g in np.asarray(groups).tolist()}) >= n_splits
        ):
            split_iter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=rs
            ).split(Xv, yv, groups=np.asarray(groups))
        else:
            split_iter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=rs
            ).split(Xv, yv)

        oof = np.full(n, np.nan)
        unpurged_folds = 0
        total_folds = 0
        for tr, te in split_iter:
            total_folds += 1
            tri = tr
            if start is not None and t1 is not None:
                ts0 = np.nanmin(start[te])
                ts1 = np.nanmax(t1[te])
                keep = ~((t1[tr] >= ts0) & (start[tr] <= ts1))
                if int(keep.sum()) > 15:
                    tri = tr[keep]
                else:
                    unpurged_folds += 1
            if len(np.unique(yv[tri])) < 2:
                continue
            try:
                imp = SimpleImputer(strategy="median").fit(Xv[tri])
                scl = StandardScaler().fit(imp.transform(Xv[tri]))
                est = CalibratedClassifierCV(
                    estimator_factory(), method="sigmoid", cv=3
                )
                est.fit(scl.transform(imp.transform(Xv[tri])), yv[tri])
                oof[te] = np.asarray(
                    est.predict_proba(scl.transform(imp.transform(Xv[te]))),
                    dtype=float,
                )[:, 1]
            except Exception as exc:  # pragma: no cover - degenerate fold
                logger.debug("Promotion-gate OOF fold skipped: %s", exc)
                continue

        if unpurged_folds:
            logger.warning(
                "Promotion-gate OOF: time purge collapsed on %d/%d folds "
                "(shuffled folds span the full time range); those folds "
                "trained unpurged — symbol grouping remains the leakage "
                "guard, and oof_auc may be modestly optimistic from "
                "cross-symbol temporal concurrency.",
                unpurged_folds,
                total_folds,
            )

        mask = ~np.isnan(oof)
        if int(mask.sum()) < 30 or len(np.unique(yv[mask])) < 2:
            return None, None, None, None
        oof_clip = np.clip(oof[mask], 1e-6, 1 - 1e-6)
        auc = float(roc_auc_score(yv[mask], oof_clip))
        lo, hi = self._bootstrap_auc_ci(
            yv[mask], oof_clip, n_boot=n_boot, random_state=rs
        )
        ece = float(self._expected_calibration_error(yv[mask], oof_clip))
        if deployable_mask is not None:
            try:
                deployable = np.asarray(deployable_mask, dtype=bool).reshape(-1)
                if len(deployable) != n:
                    self._last_oof_deployable_metrics = {
                        "available": False,
                        "definition": "capital_fraction > 0",
                        "reason": (
                            "length_mismatch"
                            f"(mask={len(deployable)},rows={n})"
                        ),
                    }
                else:
                    deployable_scored = mask & deployable
                    deployable_y = yv[deployable_scored]
                    deployable_oof = np.clip(
                        oof[deployable_scored], 1e-6, 1 - 1e-6
                    )
                    if (
                        int(deployable_scored.sum()) < 30
                        or len(np.unique(deployable_y)) < 2
                    ):
                        self._last_oof_deployable_metrics = {
                            "available": False,
                            "definition": "capital_fraction > 0",
                            "reason": "insufficient_scored_rows_or_single_class",
                            "n": int(deployable_scored.sum()),
                            "n_pos": int(np.sum(deployable_y == 1)),
                        }
                    else:
                        self._last_oof_deployable_metrics = {
                            "available": True,
                            "definition": "capital_fraction > 0",
                            "n": int(deployable_scored.sum()),
                            "n_pos": int(np.sum(deployable_y == 1)),
                            "positive_rate": float(np.mean(deployable_y)),
                            "auc": float(
                                roc_auc_score(deployable_y, deployable_oof)
                            ),
                            "ece": float(
                                self._expected_calibration_error(
                                    deployable_y, deployable_oof
                                )
                            ),
                        }
            except Exception as exc:  # diagnostic only; promotion stays governed
                self._last_oof_deployable_metrics = {
                    "available": False,
                    "definition": "capital_fraction > 0",
                    "reason": f"diagnostic_error:{type(exc).__name__}",
                }
        # FASTWIN-01 recall-tuning telemetry (no behavior change): record OOF
        # recall at fixed precision floors and the operating threshold tau* on the
        # SAME calibrated OOF predictions the promotion gate uses. Consumed offline
        # to choose two_stage.min_meta_prob; does NOT affect the promotion verdict
        # (return tuple unchanged) or any decision-time behavior.
        try:
            self._last_oof_operating_points = self._oof_recall_at_precision(
                yv[mask], oof_clip
            )
            logger.info(
                "FASTWIN-01 OOF recall@precision: %s",
                self._last_oof_operating_points,
            )
        except Exception as _rp_exc:  # pragma: no cover - diagnostic only
            self._last_oof_operating_points = None
            logger.debug("OOF recall@precision computation skipped: %s", _rp_exc)
        return auc, lo, hi, ece

    @staticmethod
    def _oof_recall_at_precision(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        precision_floors: Tuple[float, ...] = (0.50, 0.55, 0.60, 0.65),
    ) -> Dict[str, Any]:
        """Max OOF recall achievable at each precision floor, and its threshold.

        For each ``floor`` returns ``{tau, recall, precision}`` for the threshold
        that MAXIMISES recall subject to ``precision >= floor`` over the unique OOF
        probabilities (no monotonicity assumption). Pure measurement helper used to
        select ``two_stage.min_meta_prob``; mutates no model state.
        """
        yt = np.asarray(y_true, dtype=int).reshape(-1)
        yp = np.asarray(y_prob, dtype=float).reshape(-1)
        n_pos = int(yt.sum())
        out: Dict[str, Any] = {"n": int(len(yt)), "n_pos": n_pos}
        # n_pos == 0 also covers the empty-array case (an empty int array sums to
        # 0). Always set "operating_points" so the return shape is uniform.
        if n_pos == 0:
            out["operating_points"] = {}
            return out
        cand = np.unique(yp)
        results: Dict[str, Any] = {}
        for floor in precision_floors:
            best: Optional[Dict[str, float]] = None
            for tau in cand:
                sel = yp >= tau
                tp = int(np.sum(sel & (yt == 1)))
                fp = int(np.sum(sel & (yt == 0)))
                if tp == 0:
                    continue
                prec = tp / (tp + fp)
                if prec >= floor:
                    rec = tp / n_pos
                    if best is None or rec > best["recall"]:
                        best = {
                            "tau": float(tau),
                            "recall": float(rec),
                            "precision": float(prec),
                        }
            results[f"precision>={floor:.2f}"] = best
        out["operating_points"] = results
        return out

    def _evaluate_oos_calibration_gate(
        self,
        report: Dict[str, float],
    ) -> tuple[bool, str | None]:
        """Accept OOS calibration only when quality is non-degrading."""
        samples = float(report.get("samples", 0.0))
        if samples < float(self.config.oos_calibration_min_samples):
            return False, "insufficient_oos_samples"

        ece_raw = float(report.get("ece_raw", float("nan")))
        ece_cal = float(report.get("ece_calibrated", float("nan")))
        brier_raw = float(report.get("brier_raw", float("nan")))
        brier_cal = float(report.get("brier_calibrated", float("nan")))
        if not np.isfinite([ece_raw, ece_cal, brier_raw, brier_cal]).all():
            return False, "non_finite_calibration_metrics"

        ece_delta = ece_raw - ece_cal
        brier_delta = brier_raw - brier_cal
        if ece_delta < -float(self.config.oos_calibration_max_ece_degradation):
            return False, "ece_degraded_beyond_tolerance"
        if brier_delta < -float(self.config.oos_calibration_max_brier_degradation):
            return False, "brier_degraded_beyond_tolerance"

        min_improvement = float(self.config.oos_calibration_min_improvement)
        if ece_delta < min_improvement and brier_delta < min_improvement:
            return False, "no_metric_improvement"
        return True, None

    def train(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "start_time_utc",
        pnl_col: str = "pnl_pct",
        t1_col: str = "t1",
        y_col: Optional[str] = None,
    ) -> MetaLabelerMetrics:
        """
        Train meta-labeler with purged cross-validation.

        AFML Compliance:
        - Uses sample weighting based on event concurrency (if t1 available)
        - CV-aware imputation (fit imputer on train fold only)
        - Probability calibration for reliable estimates

        Args:
            df: Historical data with features and performance
            timestamp_col: Column for time-ordering (t0)
            pnl_col: Column for PnL percentage
            t1_col: Column for event end time (for sample weighting)
            y_col: Optional pre-computed label column. When specified and
                present in *df* with at least one non-null value, this
                column is used directly as the training target instead of
                rebuilding labels from PnL.  This is the mechanism by
                which hierarchical labels (hlabel_meta) reach the model.

        Returns:
            Training metrics
        """
        # Create labels — prefer pre-computed column when available.
        if (
            y_col is not None
            and y_col in df.columns
            and bool(cast(pd.Series, df[y_col]).notna().any())
        ):
            y = df[y_col].fillna(0).astype(int)
            logger.info(
                "Using pre-computed label column '%s' (%d positive / %d total)",
                y_col, int(cast(pd.Series, y).sum()), len(y),
            )
            # Still record which pnl column would have been used for audit
            self._label_pnl_col_used = pnl_col
        else:
            y = self.create_labels(df, pnl_col=pnl_col)

        # C3 fix: guard against extreme class imbalance before training
        _global_pos_rate = float(np.mean(np.asarray(y) == 1))
        if _global_pos_rate < MIN_POSITIVE_RATE:
            raise ValueError(
                f"Positive rate {_global_pos_rate:.1%} is below 5%. "
                f"Training would produce an unstable model (class weight "
                f">{1.0 / (2.0 * max(_global_pos_rate, 1e-9)):.0f}x). "
                f"Fix data composition before retraining."
            )
        if _global_pos_rate < 0.15:
            logger.warning(
                "Positive rate %.1f%% is below recommended 15%%. "
                "Class weights will amplify positives by %.1fx. "
                "Consider raising --min-score or downsampling negatives.",
                _global_pos_rate * 100,
                1.0 / (2.0 * _global_pos_rate),
            )

        # Select available features (don't impute yet - do inside CV)
        available = [f for f in self.config.features if f in df.columns]

        # Guard: strip any label columns that would cause information leakage
        leaked = {f for f in available if _is_label_column(f)}
        if leaked:
            logger.warning("Dropping label columns from training features: %s", sorted(leaked))
            available = [f for f in available if not _is_label_column(f)]

        if len(available) < 3:
            raise ValueError(f"Not enough features available. Found: {available}")

        self._feature_names = available
        X_raw = cast(pd.DataFrame, df[available].copy())

        # Convert to numeric
        for col in X_raw.columns:
            X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")

        # Handle inf
        X_raw = cast(pd.DataFrame, X_raw.replace([np.inf, -np.inf], np.nan))

        # ── Feature completeness gate ────────────────────────────────
        # Prevents silent regression to imputed noise by checking
        # per-feature NaN rates before training proceeds.
        nan_rates_series = cast(pd.Series, X_raw.isna().mean())
        warn_threshold = self.config.feature_nan_warn_pct / 100.0
        block_threshold = self.config.feature_nan_block_pct / 100.0

        warned_features = [
            (str(feat), float(rate))
            for feat, rate in nan_rates_series.items()
            if float(rate) > warn_threshold
        ]
        if warned_features:
            logger.warning(
                "Feature completeness gate — %d feature(s) exceed %.0f%% NaN:",
                len(warned_features),
                self.config.feature_nan_warn_pct,
            )
            for feat, rate in warned_features:
                logger.warning("  %s: %.1f%% NaN", feat, rate * 100.0)

        blocked_features = [
            (str(feat), float(rate))
            for feat, rate in nan_rates_series.items()
            if float(rate) > block_threshold
        ]
        if blocked_features:
            blocked_msg = ", ".join(
                f"{feat} ({rate:.0%})" for feat, rate in blocked_features
            )
            raise ValueError(
                f"Feature completeness gate BLOCKED training: "
                f"{len(blocked_features)} feature(s) exceed "
                f"{self.config.feature_nan_block_pct:.0f}% NaN rate: "
                f"{blocked_msg}. "
                f"Regenerate incomplete snapshot rows from scan-time sources, or "
                f"remove dead features from the feature profile."
            )

        overall_nan_pct = float(nan_rates_series.mean()) * 100.0
        logger.info(
            "Feature completeness: %.1f%% overall fill rate across %d features",
            100.0 - overall_nan_pct,
            len(available),
        )

        # AFML Ch8: Feature multicollinearity diagnostics
        if len(available) >= 2:
            try:
                from neutralgrid.training.feature_diagnostics import compute_feature_diagnostics
                compute_feature_diagnostics(
                    cast(pd.DataFrame, X_raw), vif_threshold=5.0, corr_threshold=0.9
                )
            except Exception as e:
                logger.warning("Feature diagnostics skipped: %s", e)

        # Import sklearn components
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import roc_auc_score
            from sklearn.impute import SimpleImputer
            from sklearn.calibration import CalibratedClassifierCV
        except ImportError:
            logger.error("sklearn not installed, cannot train meta-labeler")
            raise

        # Import CPCV (Combinatorial Purged Cross-Validation)
        from neutralgrid.backtest.cpcv import CPCV, CPCVConfig

        # Helper: create the appropriate model based on config
        def _make_model():
            if self.config.use_sequential_bootstrap:
                return DecisionTreeClassifier(
                    max_depth=self.config.max_depth,
                    min_samples_split=self.config.min_samples_split,
                    min_samples_leaf=self.config.min_samples_leaf,
                    random_state=self.config.random_state,
                )
            elif self.config.estimator_type == "logistic":
                # FASTWIN-01: L2-regularized logistic; class_weight balances the
                # fast-winner positive class. Wrapped by CalibratedClassifierCV
                # downstream for probability calibration, exactly as the GBM path is.
                from sklearn.linear_model import LogisticRegression

                return LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=self.config.random_state,
                )
            elif self.config.estimator_type == "vote_logit_hgb":
                # FASTWIN-02: soft-vote of the FASTWIN-01 logistic and a
                # strongly-regularized HistGradientBoostingClassifier. The vote
                # output (not each leg) is what CalibratedClassifierCV
                # calibrates downstream — required for the ECE <= 0.10 gate.
                from sklearn.ensemble import (
                    HistGradientBoostingClassifier,
                    VotingClassifier,
                )
                from sklearn.linear_model import LogisticRegression

                return VotingClassifier(
                    estimators=[
                        (
                            "logit",
                            LogisticRegression(
                                penalty="l2",
                                C=1.0,
                                class_weight="balanced",
                                solver="liblinear",
                                random_state=self.config.random_state,
                            ),
                        ),
                        (
                            "hgb",
                            HistGradientBoostingClassifier(
                                learning_rate=self.config.hgb_learning_rate,
                                max_leaf_nodes=self.config.hgb_max_leaf_nodes,
                                max_iter=self.config.hgb_max_iter,
                                min_samples_leaf=self.config.hgb_min_samples_leaf,
                                l2_regularization=self.config.hgb_l2_regularization,
                                max_features=self.config.hgb_max_features,
                                # False, not "auto": auto would silently enable
                                # early stopping once the pool exceeds 10k rows,
                                # changing the validated estimator. cast() only
                                # papers over the sklearn stub typing it as str.
                                early_stopping=cast(Any, False),
                                random_state=self.config.random_state,
                            ),
                        ),
                    ],
                    voting="soft",
                    weights=[1.0, 1.0],
                )
            else:
                return GradientBoostingClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    min_samples_split=self.config.min_samples_split,
                    min_samples_leaf=self.config.min_samples_leaf,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state,
                )

        def _make_preprocessor():
            """CV-aware feature preprocessor (fit on the train fold only).

            FASTWIN-01: the L2 logistic estimator is scale-sensitive — the
            fast-winner feature set spans ~1e9 (quote_volume_24h, open_interest)
            to ~1e-4 (funding_rate, ema_slope_1h), so without standardization the
            L2 penalty crushes small-scale features and the model degenerates to
            ~0.5 AUC. We therefore chain a StandardScaler after the mean imputer
            for the logistic path (mirrors scripts/meta_labeler_feature_study.py,
            which standardizes and reproduces OOF AUC ~0.70). Tree estimators
            (GBM / sequential-bootstrap DecisionTree) are scale-invariant and
            keep the imputer alone. The fitted preprocessor is saved as
            ``self._imputer`` and applied at inference via ``_prepare_features``,
            so scaling travels with the model — no separate artifact slot.
            """
            if (
                self.config.estimator_type in ("logistic", "vote_logit_hgb")
                and not self.config.use_sequential_bootstrap
            ):
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler

                return make_pipeline(
                    SimpleImputer(strategy='mean'), StandardScaler()
                )
            return SimpleImputer(strategy='mean')

        # Compute sample weights if t1 available
        sample_weights = None
        _compute_weights_fn: Any = None
        weight_config: Any = None
        if self.config.use_sample_weights and t1_col in df.columns:
            try:
                from neutralgrid.training.sample_weights import (
                    compute_sample_weights as _csw_fn,
                    SampleWeightConfig,
                )
                _compute_weights_fn = _csw_fn
                weight_config = SampleWeightConfig(
                    use_uniqueness=True,
                    use_time_decay=self.config.use_time_decay,
                    time_decay_halflife_days=self.config.time_decay_halflife_days,
                )
                sample_weights = _compute_weights_fn(
                    df, config=weight_config, t0_col=timestamp_col, t1_col=t1_col
                )
                logger.info(
                    "Computed sample weights: mean=%.3f, std=%.3f",
                    sample_weights.mean(), sample_weights.std()
                )
            except Exception as e:
                logger.warning(f"Could not compute sample weights: {e}")
                sample_weights = None

        # Incorporate sample_weight_override from DataFrame (e.g., backtest=0.5, live=1.0)
        if "sample_weight_override" in df.columns:
            swo = np.nan_to_num(
                np.asarray(pd.to_numeric(
                    df["sample_weight_override"], errors="coerce"
                ), dtype=float),
                nan=1.0,
            )
            if sample_weights is not None:
                sample_weights = np.asarray(sample_weights, dtype=float) * np.asarray(swo, dtype=float)
            else:
                sample_weights = np.asarray(swo, dtype=float)
            logger.info(
                "Applied sample_weight_override: mean=%.3f, min=%.3f, max=%.3f",
                float(np.mean(swo)), float(np.min(swo)), float(np.max(swo)),
            )

        # Train with CPCV
        # Prefer time-based purging/embargo when event end times (t1) are available.
        purge_hours = (
            float(self.config.purge_hours) if self.config.purge_hours is not None else float(self.config.horizon_hours)
        )
        use_time_purge = False
        if t1_col in df.columns:
            try:
                use_time_purge = pd.to_datetime(df[t1_col], utc=True, errors='coerce').notna().any()
            except Exception:
                use_time_purge = False

        if use_time_purge:
            cpcv_config = CPCVConfig(
                n_groups=self.config.cv_folds,
                n_test_groups=2,
                purge_hours=purge_hours,
                embargo_hours=float(self.config.embargo_hours),
                horizon_hours=float(self.config.horizon_hours),
            )
        else:
            cpcv_config = CPCVConfig(
                n_groups=self.config.cv_folds,
                n_test_groups=2,
                purge_pct=self.config.purge_pct,
                embargo_pct=self.config.embargo_pct,
                horizon_hours=float(self.config.horizon_hours),
            )
        cv = CPCV(cpcv_config)

        # Ensure df has timestamp for CV
        if timestamp_col not in df.columns:
            # Add index as pseudo-timestamp
            df = df.copy()
            df[timestamp_col] = pd.date_range(start="2020-01-01", periods=len(df), freq="h")

        # PRE-SORT by timestamp before CPCV split (Gate 7 Fix 1).
        # CPCV.split() sorts internally on its copy, but the caller's df,
        # X_raw, y, and sample_weights must be in the same temporal order so
        # that index-based mask operations (df.index.isin) map correctly and
        # calibration holdout splits take the chronologically latest data.
        _ts_parsed = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
        _sort_order = _ts_parsed.argsort()
        df = df.iloc[_sort_order].reset_index(drop=True)
        X_raw = cast(pd.DataFrame, X_raw.iloc[_sort_order].reset_index(drop=True))
        y = y.iloc[_sort_order].reset_index(drop=True)
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype=float)[_sort_order]
        logger.info(
            "Pre-sorted %d training rows by %s before CPCV split",
            len(df), timestamp_col,
        )

        aucs = []
        p5s = []
        fold_perm_importances = []
        oos_y_true = []
        oos_y_proba = []

        _group_col = "symbol" if "symbol" in df.columns else None
        for train_idx, test_idx in cv.split(df, timestamp_col=timestamp_col, t1_col=t1_col, group_col=_group_col):
            # Map indices to positions
            train_mask = df.index.isin(train_idx)
            test_mask = df.index.isin(test_idx)

            X_train_raw = np.asarray(X_raw[train_mask])
            y_train = np.asarray(y[train_mask])
            X_test_raw = np.asarray(X_raw[test_mask])
            y_test = np.asarray(y[test_mask])

            if len(np.unique(y_train)) < 2:
                continue

            # CV-AWARE PREPROCESSING: fit imputer (+ scaler for logistic) on the
            # train fold only (AFML compliance — no test-fold leakage).
            imputer = _make_preprocessor()
            X_train = np.asarray(imputer.fit_transform(X_train_raw))
            X_test = np.asarray(imputer.transform(X_test_raw))  # Use train statistics

            # Recompute sample weights on this fold's training subset
            # so uniqueness reflects post-purge concurrency (AFML Ch 4)
            fold_weights = None
            if sample_weights is not None and _compute_weights_fn is not None and t1_col in df.columns:
                try:
                    fold_df = df.loc[train_mask]
                    fold_weights = _compute_weights_fn(
                        fold_df, config=weight_config, t0_col=timestamp_col, t1_col=t1_col
                    )
                except Exception:
                    fold_weights = None

            # Apply per-fold sample_weight_override (backtest=0.5, live=1.0)
            if "sample_weight_override" in df.columns:
                fold_swo = np.nan_to_num(
                    np.asarray(pd.to_numeric(
                        df.loc[train_mask, "sample_weight_override"], errors="coerce"
                    ), dtype=float),
                    nan=1.0,
                )
                if fold_weights is not None:
                    fold_weights = np.asarray(fold_weights, dtype=float) * np.asarray(fold_swo, dtype=float)
                else:
                    fold_weights = np.asarray(fold_swo, dtype=float)

            # Class rebalancing via sample_weight (sklearn balanced formula)
            # w_class(c) = n_total / (2 * n_c) — equalizes effective class sizes
            if getattr(self.config, "use_class_weights", True):
                n_pos = int(np.sum(y_train == 1))
                n_neg = len(y_train) - n_pos
                if n_pos > 0 and n_neg > 0:
                    cw_map = {0: len(y_train) / (2.0 * n_neg), 1: len(y_train) / (2.0 * n_pos)}
                    class_w = np.array([cw_map[int(lbl)] for lbl in y_train], dtype=float)
                    if fold_weights is not None:
                        fold_weights = np.asarray(fold_weights, dtype=float) * class_w
                    else:
                        fold_weights = class_w

            # Train fold model
            model = _make_model()
            if self.config.use_sequential_bootstrap and t1_col in df.columns:
                # Use sequential bootstrap indices for bagging (AFML Ch 6)
                try:
                    from neutralgrid.training.sample_weights import sequential_bootstrap_indices
                    fold_df = df.loc[train_mask]
                    t0_fold = pd.to_datetime(fold_df[timestamp_col])
                    t1_fold = pd.to_datetime(fold_df[t1_col])
                    seq_idx = sequential_bootstrap_indices(
                        t0_fold, t1_fold, n_samples=len(X_train),
                        random_state=self.config.random_state,
                    )
                    # Bounds-check: clamp indices to valid range
                    seq_idx = np.asarray(seq_idx)
                    seq_idx = seq_idx[seq_idx < len(X_train)]
                    if len(seq_idx) == 0:
                        raise ValueError("All sequential bootstrap indices out of bounds")
                    X_train_seq = X_train[seq_idx]
                    y_train_seq = y_train[seq_idx]
                    w_seq = fold_weights[seq_idx] if fold_weights is not None else None
                    model.fit(X_train_seq, y_train_seq, sample_weight=w_seq)
                except Exception as e:
                    logger.warning("Sequential bootstrap failed in fold, falling back: %s", e)
                    model.fit(X_train, y_train, sample_weight=fold_weights)
            else:
                model.fit(X_train, y_train, sample_weight=fold_weights)

            # Evaluate
            y_pred_proba = np.asarray(model.predict_proba(X_test), dtype=float)[:, 1]
            oos_y_true.extend(y_test.tolist())
            oos_y_proba.extend(y_pred_proba.tolist())

            # Extract test weights for weighted OOS metrics (AFML audit)
            test_weights = (
                np.asarray(sample_weights, dtype=float)[np.where(test_mask)[0]]
                if sample_weights is not None
                else None
            )

            if len(np.unique(y_test)) >= 2:
                auc = roc_auc_score(y_test, y_pred_proba, sample_weight=test_weights)
                aucs.append(auc)

            # Precision at 5
            if len(y_test) >= 5:
                top5_idx = np.argsort(y_pred_proba)[-5:]
                p5 = y_test[top5_idx].mean()
                p5s.append(p5)

            # OOS permutation importance per fold (AFML Ch 8)
            if len(np.unique(y_test)) >= 2:
                try:
                    from sklearn.inspection import permutation_importance as perm_imp
                    fold_perm = perm_imp(
                        model, X_test, y_test, n_repeats=5,
                        random_state=self.config.random_state, scoring="roc_auc",
                    )
                    fold_perm_importances.append(fold_perm["importances_mean"])
                except ImportError:
                    logger.debug("sklearn.inspection not available, skipping permutation importance")
                except Exception as e:
                    logger.debug("Permutation importance failed on fold: %s", e)

        # ERR-053: surface CPCV-OOF coverage. On small / symbol-diverse pools the
        # symbol-grouped CPCV degenerates (few valid splits) and scores only a
        # fraction of rows, so `auc_cv` (the per-fold mean computed later) is
        # ADVISORY-ONLY and gates NOTHING — promotion is judged separately by
        # `_evaluate_promotion_oof` (study-faithful, all-rows-scored, calibrated).
        # Warn when coverage is low so the auc_cv vs oof_auc divergence is
        # explainable from the logs. No model-output change (transparency only).
        _auc_cv_oof_coverage = len(oos_y_true) / max(len(df), 1)
        if _auc_cv_oof_coverage < 0.80:
            logger.warning(
                "CPCV-OOF coverage %.1f%% (%d/%d rows scored across %d valid "
                "folds); auc_cv is advisory-only at this coverage — rely on the "
                "promotion gate's study-faithful OOF (ERR-053).",
                _auc_cv_oof_coverage * 100.0, len(oos_y_true), len(df), len(aucs),
            )

        # Final preprocessing for full training (imputer + scaler for logistic).
        # Saved as self._imputer so _prepare_features applies the same transform
        # (incl. standardization) at inference — scaling rides with the model.
        final_imputer = _make_preprocessor()
        X = np.asarray(final_imputer.fit_transform(np.asarray(X_raw)))
        self._imputer = final_imputer  # Save for inference

        # Preserve pre-class-weight sample_weights for calibration holdout.
        # Calibration should reflect the NATURAL class distribution, not the
        # rebalanced one — class weights would inflate minority class frequency
        # in the calibration holdout and bias the calibrator's learned base rate.
        sample_weights_for_calibration = (
            np.array(sample_weights, dtype=float, copy=True)
            if sample_weights is not None else None
        )

        # Apply class rebalancing to full-dataset sample_weights (same formula as per-fold)
        # Weight order: uniqueness × sample_weight_override × class_weights
        if getattr(self.config, "use_class_weights", True):
            y_arr = np.asarray(y)
            n_pos_full = int(np.sum(y_arr == 1))
            n_neg_full = len(y_arr) - n_pos_full
            if n_pos_full > 0 and n_neg_full > 0:
                cw_full = {
                    0: len(y_arr) / (2.0 * n_neg_full),
                    1: len(y_arr) / (2.0 * n_pos_full),
                }
                class_w_full = np.array(
                    [cw_full[int(lbl)] for lbl in y_arr], dtype=float
                )
                if sample_weights is not None:
                    sample_weights = np.asarray(sample_weights, dtype=float) * class_w_full
                else:
                    sample_weights = class_w_full
                logger.info(
                    "Final model class weights: pos=%.3f (n=%d), neg=%.3f (n=%d)",
                    cw_full[1], n_pos_full, cw_full[0], n_neg_full,
                )

        # Train final model on all data
        base_model = _make_model()
        if self.config.use_sequential_bootstrap and t1_col in df.columns:
            try:
                from neutralgrid.training.sample_weights import sequential_bootstrap_indices
                t0_full = pd.to_datetime(df[timestamp_col])
                t1_full = pd.to_datetime(df[t1_col])
                seq_idx = sequential_bootstrap_indices(
                    t0_full, t1_full, n_samples=len(X),
                    random_state=self.config.random_state,
                )
                # Bounds-check: clamp indices to valid range
                seq_idx = np.asarray(seq_idx)
                seq_idx = seq_idx[seq_idx < len(X)]
                if len(seq_idx) == 0:
                    raise ValueError("All sequential bootstrap indices out of bounds")
                base_model.fit(X[seq_idx], np.asarray(y)[seq_idx],
                               sample_weight=sample_weights[seq_idx] if sample_weights is not None else None)
            except Exception as e:
                logger.warning("Sequential bootstrap failed for final model, falling back: %s", e)
                base_model.fit(X, np.asarray(y), sample_weight=sample_weights)
        else:
            base_model.fit(X, np.asarray(y), sample_weight=sample_weights)

        # PROBABILITY CALIBRATION (AFML-compliant temporal holdout)
        # Fix #7: Use temporal holdout instead of unpurged cv=3 which leaks
        # future information into calibration.  We hold out the most recent
        # calibration_holdout_pct of training data (by time order) for calibration fitting.
        if self.config.calibrate_probabilities:
            try:
                cal_split = int(len(X) * (1.0 - self.config.calibration_holdout_pct))
                # C2 fix: also guard against holdout with single class
                _cal_y = np.asarray(y)[cal_split:]
                _cal_has_both_classes = len(np.unique(_cal_y)) >= 2
                if not (cal_split >= 50 and (len(X) - cal_split) >= 20):
                    _cal_skip_reason = "not enough data for temporal calibration"
                elif not _cal_has_both_classes:
                    _cal_skip_reason = (
                        f"calibration holdout has only class {int(_cal_y[0])} "
                        f"({len(_cal_y)} samples)"
                    )
                else:
                    _cal_skip_reason = None

                if _cal_skip_reason is None:
                    # Train base model on older portion
                    cal_base = _make_model()
                    cal_weights = sample_weights[:cal_split] if sample_weights is not None else None
                    cal_base.fit(X[:cal_split], np.asarray(y)[:cal_split], sample_weight=cal_weights)

                    # Calibrate on newer 20% (temporal holdout, no leakage)
                    # Use pre-class-weight weights for calibration fitting so the
                    # calibrator learns the natural class distribution, not the
                    # rebalanced one.
                    cal_sw = (
                        sample_weights_for_calibration[cal_split:]
                        if sample_weights_for_calibration is not None else None
                    )
                    try:
                        calibrated_model = CalibratedClassifierCV(
                            cal_base,
                            method=cast(Literal["sigmoid", "isotonic"], self.config.calibration_method),
                            cv="prefit",  # Prefer classic prefit path when supported.
                        )
                        calibrated_model.fit(
                            X[cal_split:],
                            np.asarray(y)[cal_split:],
                            sample_weight=cal_sw,
                        )
                    except ValueError as prefit_err:
                        if "prefit" not in str(prefit_err):
                            raise
                        # sklearn>=1.6: prefit was replaced by FrozenEstimator flow.
                        try:
                            from sklearn.frozen import FrozenEstimator
                        except ImportError:
                            # sklearn < 1.6 lacks FrozenEstimator; re-raise the
                            # original prefit error so the outer handler can log it.
                            raise prefit_err from None
                        calibrated_model = CalibratedClassifierCV(
                            estimator=FrozenEstimator(cal_base),
                            method=cast(Literal["sigmoid", "isotonic"], self.config.calibration_method),
                            cv=None,
                        )
                        # sklearn FrozenEstimator path does not reliably propagate
                        # sample_weight semantics; calibrate unweighted to avoid
                        # misleading weighting behavior warnings.
                        calibrated_model.fit(X[cal_split:], np.asarray(y)[cal_split:])
                    self._model = calibrated_model
                    self._is_calibrated = True
                    logger.info(
                        "Applied %s calibration via temporal holdout (train=%d, cal=%d)",
                        self.config.calibration_method, cal_split, len(X) - cal_split,
                    )
                else:
                    logger.warning(
                        "Skipping temporal calibration: %s, using uncalibrated",
                        _cal_skip_reason,
                    )
                    self._model = base_model
                    self._is_calibrated = False
            except Exception as e:
                logger.warning(f"Calibration failed, using uncalibrated model: {e}")
                self._model = base_model
                self._is_calibrated = False
        else:
            self._model = base_model
            self._is_calibrated = False

        # OOS calibration on pooled CV predictions.
        # This maps predicted probabilities to observed frequencies using only
        # out-of-sample predictions, improving reliability of meta_prob.
        oos_calibrator = None
        oos_method = None
        oos_yc: Any = None
        calibration_report: Dict[str, Any] = {}
        if (
            self.config.calibrate_probabilities
            and len(oos_y_true) >= int(self.config.oos_calibration_min_samples)
            and len(set(oos_y_true)) >= 2
        ):
            try:
                from sklearn.metrics import brier_score_loss

                oos_yt = np.asarray(oos_y_true, dtype=float)
                oos_yp = np.clip(np.asarray(oos_y_proba, dtype=float), 1e-6, 1 - 1e-6)
                candidate_calibrator = None
                candidate_method = None

                if self.config.calibration_method == "isotonic":
                    from sklearn.isotonic import IsotonicRegression

                    candidate_calibrator = IsotonicRegression(
                        y_min=0.0,
                        y_max=1.0,
                        out_of_bounds="clip",
                    )
                    candidate_calibrator.fit(oos_yp, oos_yt)
                    oos_yc = np.asarray(candidate_calibrator.predict(oos_yp), dtype=float)
                    candidate_method = "isotonic_oos"
                else:
                    from sklearn.linear_model import LogisticRegression

                    candidate_calibrator = LogisticRegression(
                        solver="lbfgs",
                        random_state=self.config.random_state,
                    )
                    candidate_calibrator.fit(oos_yp.reshape(-1, 1), oos_yt.astype(int))
                    oos_yc = np.asarray(
                        candidate_calibrator.predict_proba(oos_yp.reshape(-1, 1)), dtype=float
                    )[:, 1]
                    candidate_method = "sigmoid_oos"

                calibration_report = {
                    "brier_raw": float(brier_score_loss(oos_yt, oos_yp)),
                    "brier_calibrated": float(brier_score_loss(oos_yt, oos_yc)),
                    "ece_raw": float(self._expected_calibration_error(oos_yt, oos_yp)),
                    "ece_calibrated": float(self._expected_calibration_error(oos_yt, oos_yc)),
                    "samples": float(len(oos_yt)),
                }
                calibration_report["ece_improvement"] = (
                    float(calibration_report["ece_raw"]) -
                    float(calibration_report["ece_calibrated"])
                )
                calibration_report["brier_improvement"] = (
                    float(calibration_report["brier_raw"]) -
                    float(calibration_report["brier_calibrated"])
                )
                accepted, reject_reason = self._evaluate_oos_calibration_gate(
                    cast(Dict[str, float], calibration_report)
                )
                calibration_report["accepted"] = 1.0 if accepted else 0.0
                if reject_reason:
                    calibration_report["reject_reason"] = reject_reason

                if accepted:
                    oos_calibrator = candidate_calibrator
                    oos_method = candidate_method
                    logger.info(
                        "Applied %s OOS calibration (samples=%d, ECE %.4f->%.4f, Brier %.4f->%.4f)",
                        oos_method,
                        len(oos_yt),
                        calibration_report["ece_raw"],
                        calibration_report["ece_calibrated"],
                        calibration_report["brier_raw"],
                        calibration_report["brier_calibrated"],
                    )
                else:
                    logger.warning(
                        "Rejected %s OOS calibration (reason=%s, ECE %.4f->%.4f, Brier %.4f->%.4f)",
                        candidate_method,
                        reject_reason,
                        calibration_report["ece_raw"],
                        calibration_report["ece_calibrated"],
                        calibration_report["brier_raw"],
                        calibration_report["brier_calibrated"],
                    )
            except Exception as e:
                logger.warning("OOS calibration failed, using base probabilities: %s", e)
                oos_calibrator = None
                oos_method = None
                calibration_report = {}

        # Phase 2: Conditional beta calibration upgrade
        # If sigmoid/isotonic calibration was accepted, check the reliability
        # diagnostic on the ACCEPTED calibrator's OUTPUT (residual
        # miscalibration) to decide whether asymmetric beta calibration is
        # warranted. ERR-062: this diagnostic previously ran on the RAW model
        # output, and beta replaced the gate-accepted calibrator on
        # convergence alone — deploying a measurably worse calibrator
        # (artifact 20260701_224916: sigmoid ECE 0.0181 silently replaced by
        # beta ECE 0.0652 on the same pooled OOS predictions). Beta must now
        # strictly beat the accepted calibrator's ECE to be deployed.
        if oos_calibrator is not None and oos_method is not None and oos_yc is not None:
            try:
                from neutralgrid.calibration.reliability_diagnostic_v20260311 import (
                    ReliabilityDiagnostic,
                )
                from neutralgrid.calibration.beta_calibrator_v20260311 import (
                    BetaCalibrator,
                )

                oos_yt_arr = np.asarray(oos_y_true, dtype=float)
                oos_yp_arr = np.clip(
                    np.asarray(oos_y_proba, dtype=float), 1e-6, 1 - 1e-6
                )
                oos_yc_arr = np.clip(
                    np.asarray(oos_yc, dtype=float), 1e-6, 1 - 1e-6
                )
                diag = ReliabilityDiagnostic()
                rel_report = diag.compute(oos_yt_arr, oos_yc_arr)

                if rel_report.needs_beta_upgrade:
                    logger.info(
                        "Reliability diagnostic: residual max_deviation=%.4f > 0.05 "
                        "on %s output — evaluating beta calibration upgrade",
                        rel_report.max_deviation,
                        oos_method,
                    )
                    beta_cal = BetaCalibrator()
                    beta_result = beta_cal.fit(oos_yt_arr, oos_yp_arr)
                    accepted_ece = float(calibration_report["ece_calibrated"])
                    if (
                        beta_result.converged
                        and float(beta_result.ece_after) < accepted_ece
                    ):
                        # Preserve the displaced calibrator's metrics for audit,
                        # then report the DEPLOYED calibrator's metrics in the
                        # canonical ece/brier_calibrated fields.
                        calibration_report["displaced_method"] = oos_method
                        calibration_report["displaced_ece_calibrated"] = accepted_ece
                        calibration_report["displaced_brier_calibrated"] = float(
                            calibration_report["brier_calibrated"]
                        )
                        oos_calibrator = beta_cal
                        oos_method = "beta_oos"
                        calibration_report["beta_upgrade"] = True
                        calibration_report["beta_ece_before"] = beta_result.ece_before
                        calibration_report["beta_ece_after"] = beta_result.ece_after
                        calibration_report["reliability_max_deviation"] = rel_report.max_deviation
                        calibration_report["ece_calibrated"] = float(beta_result.ece_after)
                        calibration_report["brier_calibrated"] = float(beta_result.brier_after)
                        logger.info(
                            "Beta calibration applied (ECE %.4f→%.4f beats %s ECE %.4f; a=%.4f, b=%.4f, c=%.4f)",
                            beta_result.ece_before,
                            beta_result.ece_after,
                            calibration_report["displaced_method"],
                            accepted_ece,
                            beta_result.a,
                            beta_result.b,
                            beta_result.c,
                        )
                    elif not beta_result.converged:
                        logger.warning("Beta calibration fit did not converge, keeping %s", oos_method)
                        calibration_report["beta_upgrade"] = False
                        calibration_report["beta_upgrade_reason"] = "not_converged"
                    else:
                        logger.info(
                            "Beta calibration rejected: ECE %.4f does not beat %s ECE %.4f — keeping %s",
                            float(beta_result.ece_after),
                            oos_method,
                            accepted_ece,
                            oos_method,
                        )
                        calibration_report["beta_upgrade"] = False
                        calibration_report["beta_upgrade_reason"] = "ece_not_better"
                        calibration_report["beta_ece_candidate"] = float(beta_result.ece_after)
                        calibration_report["reliability_max_deviation"] = rel_report.max_deviation
                else:
                    logger.info(
                        "Reliability diagnostic: residual max_deviation=%.4f ≤ 0.05 "
                        "on %s output — calibration sufficient",
                        rel_report.max_deviation,
                        oos_method,
                    )
                    calibration_report["beta_upgrade"] = False
                    calibration_report["reliability_max_deviation"] = rel_report.max_deviation
            except ImportError:
                logger.debug("Beta calibrator or reliability diagnostic not available")
            except Exception as beta_err:
                logger.warning("Beta calibration upgrade check failed: %s", beta_err)

        self._oos_calibration_method = oos_method
        self._oos_calibration_report = calibration_report
        # C1 fix: OOS calibrator was trained on raw CV predictions (line 833),
        # so it must receive raw model output — not temporally-calibrated output.
        # When OOS calibration did not apply, preserve temporal calibration.
        if oos_calibrator is not None:
            base_for_oos = base_model
        else:
            base_for_oos = self._model
        self._model = _PostProbabilityCalibrator(
            base_model=base_for_oos,
            calibrator=oos_calibrator,
            method=oos_method,
        )
        self._is_calibrated = bool(self._is_calibrated or (oos_calibrator is not None))
        self._is_trained = True

        # Feature importance: MDI (built-in) from base model
        if hasattr(base_model, 'feature_importances_'):
            importance = dict(zip(self._feature_names, getattr(base_model, 'feature_importances_')))
        elif hasattr(base_model, 'estimators_'):
            # BaggingClassifier: average feature importances across base estimators.
            # FASTWIN-02: VotingClassifier also exposes estimators_, but neither
            # leg (LogisticRegression, HistGradientBoosting) has
            # feature_importances_ — an empty list must fall back to zeros
            # instead of np.mean([]) (which would crash the zip below). MDI is
            # advisory here; the OOS permutation importance below is the
            # authoritative measure either way.
            _sub_imps = [
                getattr(est, 'feature_importances_')
                for est in getattr(base_model, 'estimators_')
                if hasattr(est, 'feature_importances_')
            ]
            if _sub_imps:
                importance = dict(zip(self._feature_names, np.mean(_sub_imps, axis=0)))
            else:
                importance = {f: 0.0 for f in self._feature_names}
        else:
            importance = {f: 0.0 for f in self._feature_names}

        # AFML Ch 8: OOS permutation importance averaged across CV folds
        permutation_importance = {}
        if fold_perm_importances:
            avg_perm = np.mean(fold_perm_importances, axis=0)
            permutation_importance = dict(zip(
                self._feature_names,
                [float(v) for v in avg_perm],
            ))
            logger.info(
                "OOS permutation importance (%d folds): top feature=%s (%.4f)",
                len(fold_perm_importances),
                max(permutation_importance, key=lambda k: permutation_importance[k]),
                max(permutation_importance.values()),
            )

        # Merge both importance types
        combined_importance = {
            feat: {
                "mdi": importance.get(feat, 0.0),
                "permutation": permutation_importance.get(feat, 0.0),
            }
            for feat in self._feature_names
        }

        # F1-based threshold tuning on pooled OOS predictions
        best_f1_threshold = 0.5
        best_f1 = 0.0
        if len(oos_y_true) >= 10:
            from sklearn.metrics import f1_score as sklearn_f1
            oos_yt = np.array(oos_y_true)
            oos_yp = np.array(oos_y_proba)
            # C5 fix: adaptive F1 threshold range based on OOS positive rate
            _oos_pos_rate = float(np.mean(oos_yt))
            _f1_lower = max(0.05, _oos_pos_rate * 0.5)
            _f1_upper = min(0.80, max(_oos_pos_rate * 4.0, _f1_lower + 0.10))
            for thr in np.arange(_f1_lower, _f1_upper, 0.02):
                f1 = float(
                    cast(
                        Any,
                        sklearn_f1(
                            oos_yt,
                            (oos_yp >= thr).astype(int),
                            zero_division=cast(Any, 0),
                        ),
                    )
                )
                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_threshold = float(thr)
            logger.info("F1-optimal threshold=%.2f (F1=%.3f)", best_f1_threshold, best_f1)

        self._f1_threshold = best_f1_threshold
        oos_accept_raw = self._oos_calibration_report.get("accepted")
        oos_accepted = bool(oos_accept_raw) if oos_accept_raw is not None else None
        oos_reject_reason = self._oos_calibration_report.get("reject_reason")
        if oos_reject_reason is not None:
            oos_reject_reason = str(oos_reject_reason)

        # ── FASTWIN-01 promotion gate metrics (study-faithful OOF) ───────────
        # Judged with the methodology that VALIDATED the thresholds: a
        # symbol-grouped, time-purged K-fold scoring ALL rows with calibrated
        # per-fold predictions (_evaluate_promotion_oof). This is independent of
        # the model's training CPCV, which on small symbol-diverse pools
        # degenerates and scores too few rows to judge promotion. n_pos is the
        # training-frame positive count (data-sufficiency criterion). All
        # fail-closed to None when the pool is too small / single-class.
        _gate_groups = (
            df["symbol"].astype(str).to_numpy() if "symbol" in df.columns else None
        )
        _gate_start = pd.to_datetime(
            df[timestamp_col], utc=True, errors="coerce"
        ).to_numpy()
        if t1_col in df.columns:
            _gate_t1 = pd.to_datetime(
                df[t1_col], utc=True, errors="coerce"
            ).to_numpy()
        else:
            _gate_t1 = (
                pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
                + pd.to_timedelta(float(self.config.horizon_hours), unit="h")
            ).to_numpy()
        _gate_deployable: Optional[np.ndarray] = None
        if "capital_fraction" in df.columns:
            _capital_fraction_series = cast(
                pd.Series,
                pd.to_numeric(
                    cast(pd.Series, df["capital_fraction"]), errors="coerce"
                ),
            )
            _capital_fraction = np.asarray(
                _capital_fraction_series, dtype=float
            )
            _gate_deployable = np.isfinite(_capital_fraction) & (
                _capital_fraction > 0.0
            )
        oof_auc, oof_auc_ci_low, oof_auc_ci_high, oof_ece = (
            self._evaluate_promotion_oof(
                np.asarray(X_raw),
                np.asarray(y),
                _gate_groups,
                _gate_start,
                _gate_t1,
                estimator_factory=_make_model,
                deployable_mask=_gate_deployable,
            )
        )
        _deployable_metrics = self._last_oof_deployable_metrics or {}
        _deployable_available = bool(_deployable_metrics.get("available"))
        train_n_pos = int(np.sum(np.asarray(y) == 1))
        promotion_status, promotion_reasons = self._evaluate_promotion_gate(
            oof_auc_ci_low=oof_auc_ci_low,
            oof_auc_ci_high=oof_auc_ci_high,
            n_pos=train_n_pos,
            oof_ece=oof_ece,
        )
        self.promotion_status = promotion_status
        logger.info(
            "FASTWIN-01 promotion gate: status=%s reasons=%s "
            "(oof_auc=%s ci=[%s, %s] oof_ece=%s n_pos=%d)",
            promotion_status, promotion_reasons,
            None if oof_auc is None else round(oof_auc, 3),
            None if oof_auc_ci_low is None else round(oof_auc_ci_low, 3),
            None if oof_auc_ci_high is None else round(oof_auc_ci_high, 3),
            None if oof_ece is None else round(oof_ece, 3),
            train_n_pos,
        )

        self._metrics = MetaLabelerMetrics(
            # ADVISORY-ONLY (ERR-053): per-fold CPCV mean; degenerates on small/
            # symbol-diverse pools (see CPCV-OOF coverage warning above). Gates
            # nothing — promotion uses eval_metrics.oof_auc (_evaluate_promotion_oof).
            auc_cv=float(np.mean(aucs)) if aucs else 0.5,
            precision_at_5=float(np.mean(p5s)) if p5s else 0.0,
            f1_threshold=best_f1_threshold,
            f1_score=best_f1,
            feature_importance=combined_importance,
            train_samples=len(y),
            positive_rate=float(y.mean()),
            calibration_method=self._oos_calibration_method
            or (self.config.calibration_method if self._is_calibrated else None),
            calibration_ece_raw=self._oos_calibration_report.get("ece_raw"),
            calibration_ece_calibrated=self._oos_calibration_report.get("ece_calibrated"),
            calibration_brier_raw=self._oos_calibration_report.get("brier_raw"),
            calibration_brier_calibrated=self._oos_calibration_report.get("brier_calibrated"),
            calibration_oos_accepted=oos_accepted,
            calibration_oos_reject_reason=oos_reject_reason,
            label_pnl_column=self._label_pnl_col_used,
            oof_auc=oof_auc,
            oof_auc_ci_low=oof_auc_ci_low,
            oof_auc_ci_high=oof_auc_ci_high,
            oof_ece=oof_ece,
            oof_deployable_auc=(
                float(_deployable_metrics["auc"])
                if _deployable_available
                else None
            ),
            oof_deployable_ece=(
                float(_deployable_metrics["ece"])
                if _deployable_available
                else None
            ),
            oof_deployable_n=(
                int(_deployable_metrics["n"])
                if "n" in _deployable_metrics
                else None
            ),
            oof_deployable_n_pos=(
                int(_deployable_metrics["n_pos"])
                if "n_pos" in _deployable_metrics
                else None
            ),
            oof_deployable_positive_rate=(
                float(_deployable_metrics["positive_rate"])
                if _deployable_available
                else None
            ),
            oof_deployable_definition=(
                str(_deployable_metrics["definition"])
                if "definition" in _deployable_metrics
                else None
            ),
            n_pos=train_n_pos,
            promotion_status=promotion_status,
            promotion_reasons=promotion_reasons,
        )

        logger.info(
            "Meta-labeler trained: AUC=%.3f, P@5=%.3f, F1=%.3f@%.2f, samples=%d, calibrated=%s",
            self._metrics.auc_cv,
            self._metrics.precision_at_5,
            self._metrics.f1_score,
            self._metrics.f1_threshold,
            self._metrics.train_samples,
            self._is_calibrated,
        )

        return self._metrics

    def predict_proba(self, features: Dict[str, Any]) -> float:
        """Predict the active production target probability.

        This uses the same feature ordering and imputation as training.
        """
        if not self._is_trained or self._model is None:
            raise ValueError("Model not trained. Call train() first.")

        import numpy as np
        import pandas as pd

        # Feature availability check: warn on excessive imputation (train/serve skew)
        n_missing = len(self.get_missing_feature_names(features))
        if n_missing > 2:
            logger.warning(
                "Meta-labeler inference: %d features being imputed "
                "(possible train/serve skew)",
                n_missing,
            )

        df = pd.DataFrame([dict(features)])

        X, _ = self._prepare_features(df)
        proba_arr = np.asarray(self._model.predict_proba(X), dtype=float)
        proba = float(proba_arr[0, 1])
        # Clip to [0,1] defensively
        return float(min(1.0, max(0.0, proba)))

    def predict_proba_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for batch of candidates.

        Args:
            df: DataFrame with feature columns

        Returns:
            Array of probabilities
        """
        if not self._is_trained or self._model is None:
            return np.full(len(df), 0.5)

        X, _ = self._prepare_features(df)
        return np.asarray(self._model.predict_proba(X), dtype=float)[:, 1]

    def save(self, path: Path):
        """Save trained model via artifact management system.

        AFML Fix #10: Routes through artifacts.py for metadata, versioning,
        and checksum tracking.  Falls back to raw pickle if artifact system
        is unavailable.
        """
        if not self._is_trained:
            raise ValueError("Cannot save untrained model")

        path = Path(path)

        # Try artifact-managed save first
        try:
            from neutralgrid import __version__ as PIPELINE_VERSION
            from neutralgrid.models.artifacts import (
                save_artifact,
                ArtifactMetadata,
                get_git_commit,
                resolve_hmm_artifact_dir,
            )
            from datetime import datetime, timezone

            artifact_dir = path.parent / "meta_labeler"
            lineage: Dict[str, Any] = {}
            try:
                hmm_dir = resolve_hmm_artifact_dir()
                lineage["hmm_artifact_version"] = hmm_dir.name
                hmm_meta_path = hmm_dir / "metadata.json"
                if hmm_meta_path.exists():
                    with open(hmm_meta_path, "r", encoding="utf-8") as f:
                        hmm_meta = json.load(f)
                    hmm_pipeline = hmm_meta.get("pipeline_version")
                    if hmm_pipeline:
                        lineage["hmm_pipeline_version"] = str(hmm_pipeline)
                    hmm_trained_at = hmm_meta.get("trained_at_utc")
                    if hmm_trained_at:
                        lineage["hmm_trained_at_utc"] = str(hmm_trained_at)
            except Exception as e:
                logger.debug("Could not attach HMM lineage metadata: %s", e)

            metadata = ArtifactMetadata(
                artifact_version=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
                trained_at_utc=datetime.now(timezone.utc).isoformat(),
                pipeline_version=PIPELINE_VERSION,
                code_commit=get_git_commit(),
                lineage=lineage or None,
                features=list(self._feature_names),
                # ERR-063: derive model_type from the actual estimator
                # configuration — this was hardcoded to GradientBoosting and
                # misreported the deployed LogisticRegression FASTWIN model.
                model_type=(
                    "DecisionTreeClassifier"
                    if self.config.use_sequential_bootstrap
                    else "LogisticRegression"
                    if self.config.estimator_type == "logistic"
                    else "VotingClassifier(LogisticRegression+HistGradientBoosting)"
                    if self.config.estimator_type == "vote_logit_hgb"
                    else "GradientBoostingClassifier"
                ),
                model_params={
                    "estimator_type": self.config.estimator_type,
                    "n_estimators": self.config.n_estimators,
                    "max_depth": self.config.max_depth,
                    "learning_rate": self.config.learning_rate,
                    # ERR-041 (PIPELINE_FIX 2026-04-29): when an active target
                    # column is set (e.g. ACTIVE_META_TARGET_LABEL_COLUMN =
                    # "fast_winner_target"), the CLI --hurdle-pct value is
                    # decorative — the actual label threshold is
                    # ACTIVE_META_TARGET_PNL_THRESHOLD_PCT. Reporting the
                    # actual threshold here keeps `model_params.hurdle_pct`
                    # consistent with `eval_metrics.target_positive_pnl_threshold_pct`
                    # per safety-invariants.md "Config Integrity".
                    "hurdle_pct": (
                        ACTIVE_META_TARGET_PNL_THRESHOLD_PCT
                        if ACTIVE_META_TARGET_LABEL_COLUMN
                        else self.config.hurdle_pct
                    ),
                    "use_class_weights": getattr(self.config, "use_class_weights", True),
                },
                eval_metrics={
                    "auc_cv": self._metrics.auc_cv if self._metrics else None,
                    "precision_at_5": self._metrics.precision_at_5 if self._metrics else None,
                    "positive_rate": self._metrics.positive_rate if self._metrics else None,
                    "target_contract": ACTIVE_META_TARGET_CONTRACT,
                    "target_label_column": ACTIVE_META_TARGET_LABEL_COLUMN,
                    "target_duration_column": ACTIVE_META_TARGET_DURATION_COLUMN,
                    "target_duration_hours_max": ACTIVE_META_TARGET_DURATION_MAX_HOURS,
                    "target_positive_pnl_threshold_pct": ACTIVE_META_TARGET_PNL_THRESHOLD_PCT,
                    "is_calibrated": self._is_calibrated,
                    "calibration_method": (
                        self._metrics.calibration_method if self._metrics else None
                    ),
                    "calibration_ece_raw": (
                        self._metrics.calibration_ece_raw if self._metrics else None
                    ),
                    "calibration_ece_calibrated": (
                        self._metrics.calibration_ece_calibrated if self._metrics else None
                    ),
                    "calibration_brier_raw": (
                        self._metrics.calibration_brier_raw if self._metrics else None
                    ),
                    "calibration_brier_calibrated": (
                        self._metrics.calibration_brier_calibrated if self._metrics else None
                    ),
                    "calibration_oos_accepted": (
                        self._metrics.calibration_oos_accepted if self._metrics else None
                    ),
                    "calibration_oos_reject_reason": (
                        self._metrics.calibration_oos_reject_reason if self._metrics else None
                    ),
                    "label_pnl_column": (
                        self._metrics.label_pnl_column if self._metrics else None
                    ),
                    # FASTWIN-01 promotion gate (fail-closed: None => not promoted)
                    "oof_auc": self._metrics.oof_auc if self._metrics else None,
                    "oof_auc_ci_low": (
                        self._metrics.oof_auc_ci_low if self._metrics else None
                    ),
                    "oof_auc_ci_high": (
                        self._metrics.oof_auc_ci_high if self._metrics else None
                    ),
                    "oof_ece": self._metrics.oof_ece if self._metrics else None,
                    # ERR-075: decision-population visibility only.  Promotion
                    # remains governed by the all-row OOF fields above.
                    "oof_deployable_auc": (
                        self._metrics.oof_deployable_auc if self._metrics else None
                    ),
                    "oof_deployable_ece": (
                        self._metrics.oof_deployable_ece if self._metrics else None
                    ),
                    "oof_deployable_n": (
                        self._metrics.oof_deployable_n if self._metrics else None
                    ),
                    "oof_deployable_n_pos": (
                        self._metrics.oof_deployable_n_pos if self._metrics else None
                    ),
                    "oof_deployable_positive_rate": (
                        self._metrics.oof_deployable_positive_rate
                        if self._metrics
                        else None
                    ),
                    "oof_deployable_definition": (
                        self._metrics.oof_deployable_definition
                        if self._metrics
                        else None
                    ),
                    "n_pos": self._metrics.n_pos if self._metrics else None,
                    "promotion_status": (
                        self._metrics.promotion_status if self._metrics else None
                    ),
                    "promotion_reasons": (
                        self._metrics.promotion_reasons if self._metrics else None
                    ),
                },
                total_samples=self._metrics.train_samples if self._metrics else None,
            )

            # Save model + imputer via artifact system
            save_artifact(
                artifact_dir=artifact_dir,
                model=self._model,
                scaler=self._imputer,
                metadata=metadata,
            )

            # Also save legacy pickle for backwards compatibility
            state = {
                "model": self._model,
                "feature_names": self._feature_names,
                "imputer": self._imputer,
                "is_calibrated": self._is_calibrated,
                "oos_calibration_method": self._oos_calibration_method,
                "oos_calibration_report": self._oos_calibration_report,
                "label_pnl_col_used": self._label_pnl_col_used,
                "config": self.config,
                "metrics": self._metrics,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            import tempfile
            tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix='.pkl.tmp')
            try:
                with os.fdopen(tmp_fd, 'wb') as f:
                    pickle.dump(state, f)
                os.replace(tmp_path, str(path))
            except Exception:
                os.unlink(tmp_path)
                raise

            logger.info("Meta-labeler saved via artifact system to %s", artifact_dir)

        except Exception as e:
            logger.error("Artifact save failed for meta-labeler: %s", e)
            raise

    @classmethod
    def load(cls, path: Path) -> "MetaLabeler":
        """Load trained model, preferring artifact system.

        AFML Fix #10: Tries artifact-managed load first for metadata
        validation, falls back to raw pickle for legacy models.
        """
        path = Path(path)

        # Try artifact-managed load first
        artifact_dir = path.parent / "meta_labeler"
        if artifact_dir.exists() and (artifact_dir / "model.joblib").exists():
            try:
                from neutralgrid.models.artifacts import load_artifact
                from neutralgrid.models.artifacts import resolve_hmm_artifact_dir
                from neutralgrid.models.artifact_compat import require_hmm_meta_labeler_compatibility
                artifact = load_artifact(artifact_dir, validate_schema=False)

                hmm_meta = None
                try:
                    hmm_meta_path = resolve_hmm_artifact_dir() / "metadata.json"
                    if hmm_meta_path.exists():
                        with open(hmm_meta_path, "r", encoding="utf-8") as f:
                            hmm_meta = json.load(f)
                except Exception as hmm_meta_exc:
                    logger.warning("Could not read active HMM metadata for compatibility check: %s", hmm_meta_exc)

                require_hmm_meta_labeler_compatibility(
                    hmm_metadata=hmm_meta,
                    meta_metadata=artifact["metadata"].to_dict(),
                )

                feature_schema = artifact.get("feature_schema")
                metadata_features = list(artifact["metadata"].features or [])
                if isinstance(feature_schema, dict):
                    schema_features = list(feature_schema.get("features", []) or [])
                    if schema_features != metadata_features:
                        raise ValueError(
                            "Meta-labeler feature schema mismatch: "
                            f"metadata={metadata_features}, schema={schema_features}"
                        )

                labeler = cls()
                labeler._model = artifact["model"]
                labeler._imputer = artifact.get("scaler")
                labeler._feature_names = artifact["metadata"].features or []
                labeler._is_calibrated = bool(
                    artifact["metadata"].eval_metrics.get("is_calibrated", False)
                    if artifact["metadata"].eval_metrics else False
                )
                if artifact["metadata"].eval_metrics:
                    labeler._oos_calibration_method = artifact["metadata"].eval_metrics.get(
                        "calibration_method"
                    )
                    labeler._label_pnl_col_used = (
                        artifact["metadata"].eval_metrics.get("label_pnl_column")
                        or labeler._label_pnl_col_used
                    )
                    # FASTWIN-01: carry promotion authority from metadata.
                    # Absent key => None => NOT promoted (fail-closed).
                    labeler.promotion_status = artifact["metadata"].eval_metrics.get(
                        "promotion_status"
                    )
                labeler._is_trained = True

                logger.info("Meta-labeler loaded via artifact system from %s", artifact_dir)
                return labeler
            except Exception as e:
                raise ValueError(f"Artifact-managed meta-labeler load failed: {e}") from e

        # Legacy pickle load
        import warnings
        warnings.warn(
            f"Loading meta-labeler from legacy pickle at {path}. "
            "Re-run `neutralgrid-retrain meta-labeler` to migrate to the "
            "artifact system with metadata validation.",
            DeprecationWarning,
            stacklevel=2,
        )
        with open(path, "rb") as f:
            state = pickle.load(f)

        labeler = cls(state["config"])
        labeler._model = state["model"]
        labeler._feature_names = state.get("feature_names")
        labeler._imputer = state.get("imputer")
        labeler._is_calibrated = bool(state.get("is_calibrated", False))
        labeler._oos_calibration_method = state.get("oos_calibration_method")
        labeler._oos_calibration_report = state.get("oos_calibration_report", {})
        labeler._label_pnl_col_used = state.get(
            "label_pnl_col_used", labeler._label_pnl_col_used
        )
        labeler._metrics = state.get("metrics")
        # FASTWIN-01: restore promotion authority from the pickled metrics if
        # present; legacy artifacts predating the gate => None => not promoted.
        labeler.promotion_status = getattr(labeler._metrics, "promotion_status", None)
        labeler._is_trained = True

        # Validate feature list consistency
        if labeler._feature_names:
            expected_features = set(MetaLabelerConfig().features)
            loaded_features = set(labeler._feature_names)
            missing = expected_features - loaded_features
            extra = loaded_features - expected_features
            if missing:
                logger.warning(
                    "Meta-labeler missing expected features: %s. "
                    "Model may need retraining.", sorted(missing)
                )
            if extra:
                logger.info("Meta-labeler has extra features not in default config: %s", sorted(extra))
        else:
            logger.warning("Meta-labeler loaded without feature_names - cannot validate feature schema")

        if labeler._imputer is not None and labeler._feature_names:
            expected = len(labeler._feature_names)
            actual = getattr(labeler._imputer, 'n_features_in_', None)
            if actual is not None and actual != expected:
                logger.warning(
                    "Imputer dimension mismatch: imputer expects %d features, "
                    "but feature list has %d. Model may produce errors at inference.",
                    actual, expected,
                )

        logger.info("Meta-labeler loaded from %s", path)
        return labeler


def create_meta_labeler(
    hurdle_pct: float = 3.0,
    sl_pct: float = -12.0,
    horizon_hours: float = BOT_HORIZON_HOURS,
    random_state: int = 42,
) -> MetaLabeler:
    """
    Factory function to create MetaLabeler with standard settings.

    Args:
        hurdle_pct: PnL hurdle for success
        sl_pct: Stop loss threshold
        horizon_hours: Prediction horizon
        random_state: Random seed

    Returns:
        Configured MetaLabeler instance
    """
    config = MetaLabelerConfig(
        hurdle_pct=hurdle_pct,
        sl_pct=sl_pct,
        horizon_hours=horizon_hours,
        random_state=random_state,
    )
    return MetaLabeler(config)
