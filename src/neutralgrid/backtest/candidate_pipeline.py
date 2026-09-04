"""
Candidate Backtesting Pipeline.

Loads scanner CSVs, filters to backtestable candidates, runs
RealisticGridBacktester on historical klines, and converts results
to meta-labeler training rows.

AFML safeguards:
- Features are taken from scan time (before backtest) -- no look-ahead.
- t1 (vertical barrier) is set for proper CPCV purging.
- Backtest rows are flagged source="backtest" with 0.5x sample weight.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd

from neutralgrid.core.config import get_config
from neutralgrid.core.constants import BOT_HORIZON_HOURS

logger = logging.getLogger(__name__)

# ── Scanner CSV column → training feature mapping ──────────────────────────
# The scanner CSV uses "hmm_range_prob" / "hmm_trend_prob" while the training
# schema expects "range_prob" / "trend_prob".  We also need "score" →
# "primary_pipeline_score".

_SCANNER_TO_FEATURE: Dict[str, str] = {
    "hmm_range_prob": "range_prob",
    "hmm_trend_prob": "trend_prob",
    "range_prob": "range_prob",
    "trend_prob": "trend_prob",
    "survival_prob": "survival_prob",
    "hurst_exponent": "hurst_exponent",
    "ou_halflife": "ou_halflife",
    "profit_per_grid_pct": "profit_per_grid_pct",
    "num_grids": "num_grids",
    "grid_spacing_pct": "grid_spacing_pct",
    "mode": "mode",  # GRID_SYNCH §1.4 — bot-side metadata, NOT a model feature
    "scan_mode": "mode",
    "range_size_pct": "range_size_pct",
    "adx_1h": "adx_1h",
    "adx_15m": "adx_15m",
    "adx_5m": "adx_5m",
    "rsi_15m": "rsi_15m",
    "ema_slope_1h": "ema_slope_1h",
    "ema_crosses_5m": "ema_crosses_5m",
    "vwap_crosses_5m": "vwap_crosses_5m",
    "bb_width": "bb_width",
    "atr_pct_15m": "atr_pct_15m",
    "quote_volume_24h": "quote_volume_24h",
    "regime_conf": "regime_conf",
    "funding_rate": "funding_rate",
    "open_interest": "open_interest",
    "micro_round_trip_cost_pct": "micro_round_trip_cost_pct",
    "ev_score": "ev_score",
    "scan_score": "primary_pipeline_score",
    "score": "primary_pipeline_score",  # backward-compat fallback
    "hlabel": "hlabel",
    # afml_enriched_v20260318 additions
    "hmm_tail_cvar_95": "hmm_tail_cvar_95",
    "tos_gcf": "tos_gcf",
    "persistence_prob": "persistence_prob",
    # Tier 4 context features
    "long_short_ratio": "long_short_ratio",
    "funding_rate_zscore": "funding_rate_zscore",
    "open_interest_change_pct": "open_interest_change_pct",
    "bb_width_ratio_1h_15m": "bb_width_ratio_1h_15m",
    # Micro-oscillator archetype
    "micro_osc_score": "micro_osc_score",
    "scan_micro_osc_score": "micro_osc_score",
    # Binance-derived crowding / flow context (v20260407)
    "top_account_lsr_log": "top_account_lsr_log",
    "top_position_lsr_log": "top_position_lsr_log",
    "top_position_vs_account_delta": "top_position_vs_account_delta",
    "global_lsr_log": "global_lsr_log",
    "taker_imbalance": "taker_imbalance",
    "taker_buy_sell_log": "taker_buy_sell_log",
    "basis_pct": "basis_pct",
    # Binance-derived execution / liquidity context (v20260407)
    "spread_pct": "spread_pct",
    "top20_bid_depth_usdt": "top20_bid_depth_usdt",
    "top20_ask_depth_usdt": "top20_ask_depth_usdt",
    "top20_depth_min_usdt": "top20_depth_min_usdt",
    "book_imbalance_top20": "book_imbalance_top20",
    "oi_notional": "oi_notional",
    "oi_to_volume": "oi_to_volume",
    # Pattern-profile inputs. These are scan-time provenance columns used by
    # scanner/profile challengers, not active meta-labeler features.
    "parkinson_vol_ratio_4h_24h_pre": "parkinson_vol_ratio_4h_24h_pre",
    "variance_ratio_1m_15m_pre_2h": "variance_ratio_1m_15m_pre_2h",
    "funding_carry_expected_next_7h": "funding_carry_expected_next_7h",
    "liquidity_stability_z_1h": "liquidity_stability_z_1h",
}

# Columns to copy into training CSVs.
# Model features + hlabel (audit/label column, NOT a model feature —
# hlabel deterministically maps to y and would be information leakage).
TRAINING_OUTPUT_COLUMNS: List[str] = [
    "range_prob",
    "trend_prob",
    "survival_prob",
    "hurst_exponent",
    "ou_halflife",
    "profit_per_grid_pct",
    "num_grids",
    "grid_spacing_pct",
    "range_size_pct",
    "adx_1h",
    "adx_15m",
    "rsi_15m",
    "funding_rate",
    "ev_score",
    "primary_pipeline_score",
    # live_plus_v20260312 additions
    "adx_5m",
    "ema_slope_1h",
    "ema_crosses_5m",
    "vwap_crosses_5m",
    "atr_pct_15m",
    "quote_volume_24h",
    "bb_width",
    "open_interest",
    "hmm_tail_cvar_95",
    "tos_gcf",
    "persistence_prob",
    "hlabel",  # hierarchical label level (0-3); label column, not model input
    # Step 3 (Plan v6.0): Provenance columns for ingestion gating
    "engine_version",
    "label_contract_version",
    "backtest_run_id",
    # Gate 4 (Fix 2c): Audit column — scan-time BB-based range estimate
    "scan_range_size_pct",
    # Micro-oscillator archetype
    "micro_osc_score",
    # ERR-083: ACTIVE meta-labeler feature — declared here, not carried by the
    # NaN fill-list literal below.
    "micro_round_trip_cost_pct",
    # Tier 4 context features (v20260407)
    "funding_rate_zscore",
    "open_interest_change_pct",
    "bb_width_ratio_1h_15m",
    # Binance-derived crowding / flow context (v20260407)
    "top_account_lsr_log",
    "top_position_lsr_log",
    "top_position_vs_account_delta",
    "global_lsr_log",
    "taker_imbalance",
    "taker_buy_sell_log",
    "basis_pct",
    # Binance-derived execution / liquidity context (v20260407)
    "spread_pct",
    "top20_bid_depth_usdt",
    "top20_ask_depth_usdt",
    "top20_depth_min_usdt",
    "book_imbalance_top20",
    "oi_notional",
    "oi_to_volume",
    # Pattern-profile provenance (excluded from the active meta feature list)
    "parkinson_vol_ratio_4h_24h_pre",
    "variance_ratio_1m_15m_pre_2h",
    "funding_carry_expected_next_7h",
    "liquidity_stability_z_1h",
    # GRID_SYNCH §1.4 — bot-side metadata; must NOT enter model X-matrix (see meta_labeler.py:138)
    "mode",
]

# Backward-compatible alias (deprecated — use TRAINING_OUTPUT_COLUMNS)
TRAINING_FEATURES = TRAINING_OUTPUT_COLUMNS

OPTIONAL_LIVE_PLUS_FIELDS_V20260312: List[str] = [
    "adx_5m",
    "ema_slope_1h",
    "ema_crosses_5m",
    "vwap_crosses_5m",
    "bb_width",
    "atr_pct_15m",
    "quote_volume_24h",
    "regime_conf",
]

# PnL curve structural features (produced by btk_unified_runner)
_PNL_CURVE_FIELDS = (
    "pnl_curve_fill_concentration",
    "pnl_curve_active_ratio",
    "pnl_curve_time_to_peak",
    "pnl_curve_flatline_pct",
    "pnl_curve_entropy",
    "pnl_curve_shape_class",
    "pnl_curve_points",
    "pnl_curve_raw_json",
)

# Regex to extract timestamp from deployment_ready filenames
_FILENAME_TS_RE = re.compile(r"deployment_ready_(\d{8}_\d{6})")

# Bypass thresholds mirror MicroOscConfig defaults at scan/enrichment time.
# This static CSV filter operates on already-emitted artifacts rather than
# runtime validator state, so the thresholds are kept explicit here.
_BYPASS_MIN_SCORE = 0.45
_BYPASS_MIN_SURVIVAL_PROB = 0.60

_PUBLIC_MARKET_PROFILE = "candidate_time_public_market_v1"


# ═════════════════════════════════════════════════════════════════════════════
# 1. Load all scanner CSVs
# ═════════════════════════════════════════════════════════════════════════════


def load_all_scanner_csvs(
    results_dir: Path,
    filename_pattern: str = "deployment_ready_*.csv",
    require_meta_features: bool = True,
) -> pd.DataFrame:
    """Load all deployment-ready CSVs and synthesise missing ``candidate_id``.

    Returns a unified DataFrame with a guaranteed ``candidate_id`` column and
    a ``scan_file`` column tracking the source filename.

    Args:
        results_dir: Directory containing scanner CSVs.
        filename_pattern: Glob pattern for CSV files.
        require_meta_features: When True (default), skip CSVs that lack the
            columns required by the active meta-labeler feature profile.
            This ensures only candidates with complete feature snapshots are
            loaded, avoiding 100%-NaN feature columns at training time.
    """
    csv_files = sorted(results_dir.glob(filename_pattern))
    if not csv_files:
        raise FileNotFoundError(
            f"No scanner CSVs matching '{filename_pattern}' in {results_dir}"
        )

    frames: List[pd.DataFrame] = []
    n_skipped_features = 0
    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            continue

        if df.empty or "symbol" not in df.columns:
            continue

        # Gate: skip CSVs that predate the active meta-labeler feature profile.
        # grid_spacing_pct remains the only accepted derivable exception.
        if require_meta_features:
            from neutralgrid.models.meta_labeler import (
                ACTIVE_SNAPSHOT_META_FEATURES,
                normalize_inference_feature_frame,
            )

            normalized = normalize_inference_feature_frame(df)
            required_features = set(ACTIVE_SNAPSHOT_META_FEATURES) - {"grid_spacing_pct"}
            missing = {
                feature for feature in required_features
                if feature not in normalized.columns
            }
            if missing:
                n_skipped_features += 1
                continue

        # Wide scanner artifacts can become fragmented once reloaded; take one
        # clean copy before adding loader metadata so pandas does not emit
        # PerformanceWarning during bulk selection.
        df = df.copy()

        # Extract file-level timestamp for candidate_id synthesis
        m = _FILENAME_TS_RE.search(path.stem)
        file_ts = m.group(1) if m else path.stem
        output_ts_utc = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

        df["scan_file"] = path.name
        df["candidate_available_ts_utc"] = output_ts_utc.isoformat()
        df["candidate_available_source"] = "deployment_ready_csv_mtime"

        # Synthesise candidate_id if missing
        if "candidate_id" not in df.columns or bool(df["candidate_id"].isna().all()):
            df["candidate_id"] = df["symbol"].astype(str) + "_" + file_ts

        frames.append(df)

    if not frames:
        raise ValueError(
            "All scanner CSVs were empty or unreadable"
            + (f" ({n_skipped_features} skipped: missing meta-labeler features)"
               if n_skipped_features else "")
        )

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d candidates from %d scanner CSVs (%d skipped: pre-feature-profile)",
        len(combined), len(frames), n_skipped_features,
    )
    return combined


# ═════════════════════════════════════════════════════════════════════════════
# 2. Filter to backtestable candidates
# ═════════════════════════════════════════════════════════════════════════════


def _load_deployed_candidate_ids(
    linkage_path: Path,
) -> tuple[set[str], set[str]]:
    """Return deployed candidate IDs as ``(exact_ids, prefix_ids)``.

    *exact_ids* come from a ``candidate_id`` column (DeployLinker format,
    includes config hash).  *prefix_ids* are ``SYMBOL_YYYYMMDD_HHMMSS``
    prefixes reconstructed from legacy ``symbol`` + ``cand_csv_timestamp``
    columns; they match the start of hashed candidate IDs.
    """
    if not linkage_path.exists():
        logger.warning("Linkage file not found: %s — skipping exclusion", linkage_path)
        return set(), set()

    try:
        link_df = pd.read_csv(linkage_path)
    except Exception as exc:
        logger.warning("Cannot read linkage file: %s", exc)
        return set(), set()

    exact_ids: set[str] = set()
    prefix_ids: set[str] = set()

    has_candidate_id = "candidate_id" in link_df.columns
    has_legacy = "symbol" in link_df.columns and "cand_csv_timestamp" in link_df.columns

    for _, row in link_df.iterrows():
        if has_candidate_id:
            cid = str(row.get("candidate_id", ""))
            if cid and cid != "nan":
                exact_ids.add(cid)

        if has_legacy:
            symbol = str(row.get("symbol", ""))
            csv_ts = str(row.get("cand_csv_timestamp", ""))
            if symbol and csv_ts and csv_ts != "nan":
                ts_clean = csv_ts.replace("-", "").replace(":", "").replace("T", "_")
                ts_clean = ts_clean[:15]  # YYYYMMDD_HHMMSS
                prefix_ids.add(f"{symbol}_{ts_clean}")

    logger.info(
        "Loaded deployed candidate IDs: %d exact, %d prefix from %s",
        len(exact_ids), len(prefix_ids), linkage_path,
    )
    return exact_ids, prefix_ids


def _meta_labeler_needs_bootstrap() -> bool:
    """Check if the deployed meta-labeler uses an outdated feature profile.

    Returns True when no model exists or the model's feature list does not
    match the current active snapshot meta-feature profile.  This
    signals a bootstrap phase where ``grid_is_valid`` (which depends on the
    meta-labeler's conformal gate) should be relaxed for training data
    generation.  Once a model trained on the current profile is deployed,
    this returns False and the standard gate resumes automatically.
    """
    from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES

    metadata_path = Path("models/meta_labeler/metadata.json")
    if not metadata_path.exists():
        return True

    try:
        import json
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        model_features = set(meta.get("features") or [])
        required_features = set(ACTIVE_SNAPSHOT_META_FEATURES)
        return not required_features.issubset(model_features)
    except Exception:
        return True


def _truthy_series(values: pd.Series) -> pd.Series:
    """Return a boolean view for CSV-loaded bool columns."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def filter_backtest_candidates(
    df: pd.DataFrame,
    linkage_path: Path,
    min_score: float = 45.0,
    bypass_only: bool = False,
) -> pd.DataFrame:
    """Filter scanner DataFrame to candidates eligible for backtesting.

    Criteria:
    - score >= min_score
    - grid_is_valid == True in standard mode (auto-relaxed during bootstrap)
    - grid_is_valid != True plus bypass thresholds in bypass-only mode
    - grid_lower, grid_upper, num_grids all non-null
    - Not already deployed (excluded via linkage CSV)
    - candidate_id present

    Bootstrap auto-detection: if the deployed meta-labeler does not match the
    current feature profile, ``grid_is_valid`` is relaxed because the conformal
    gate depends on a stale model.  Once retrained, the gate self-heals.
    """
    n_start = len(df)

    # Score threshold. Discovery-mode artifacts may intentionally emit
    # below-threshold rows with valid grid geometry for data generation; keep
    # those rows backtestable while preserving below_threshold_tag in the CSV.
    score_mask = cast(pd.Series, df["score"].astype(float) >= min_score)
    discovery_score_mask = pd.Series(False, index=df.index)
    if "discovery_mode" in df.columns and "grid_is_valid" in df.columns:
        discovery_score_mask = (
            _truthy_series(cast(pd.Series, df["discovery_mode"]))
            & _truthy_series(cast(pd.Series, df["grid_is_valid"]))
        )
        n_discovery_below_score = int((discovery_score_mask & ~score_mask).sum())
        if n_discovery_below_score > 0:
            logger.info(
                "Discovery mode: retaining %d grid-valid below-score candidate(s) for backtest data generation",
                n_discovery_below_score,
            )

    mask = score_mask | discovery_score_mask
    df_f = cast(pd.DataFrame, df.loc[mask].copy())
    logger.info("After score >= %s: %d / %d", min_score, len(df_f), n_start)

    # Auto-detect bootstrap: relax grid_is_valid if the deployed model is stale
    bootstrap = not bypass_only and _meta_labeler_needs_bootstrap()
    if bootstrap:
        logger.warning(
            "Bootstrap mode: deployed meta-labeler does not match current "
            "feature profile — relaxing grid_is_valid gate for training data "
            "generation. This will self-heal after retrain.",
        )

    # Valid grid in standard mode, bypass-only candidates in bypass mode
    if "grid_is_valid" in df_f.columns:
        grid_valid = _truthy_series(cast(pd.Series, df_f["grid_is_valid"]))
        if bypass_only:
            if "micro_osc_score" not in df_f.columns or "survival_prob" not in df_f.columns:
                logger.info(
                    "Bypass-only requested but micro_osc_score/survival_prob columns are missing"
                )
                return cast(pd.DataFrame, df_f.iloc[0:0].copy())

            micro_osc_score = cast(pd.Series, df_f["micro_osc_score"])
            survival_prob = cast(pd.Series, df_f["survival_prob"])
            bypass_mask = (
                (grid_valid != True)  # noqa: E712
                & (micro_osc_score.fillna(0.0) >= _BYPASS_MIN_SCORE)
                & (survival_prob.fillna(0.0) >= _BYPASS_MIN_SURVIVAL_PROB)
            )
            df_f = cast(pd.DataFrame, df_f.loc[bypass_mask].copy())
            logger.info(
                "After bypass-only qualifier filter (micro_osc_score >= %.2f, survival_prob >= %.2f): %d",
                _BYPASS_MIN_SCORE,
                _BYPASS_MIN_SURVIVAL_PROB,
                len(df_f),
            )
        elif not bootstrap:
            df_f = cast(pd.DataFrame, df_f.loc[grid_valid == True].copy())  # noqa: E712
            logger.info("After grid_is_valid: %d", len(df_f))
        else:
            logger.info(
                "Bootstrap: skipping grid_is_valid filter (%d candidates retained)",
                len(df_f),
            )
    elif bypass_only:
        logger.info("Bypass-only requested but grid_is_valid column missing; returning empty set")
        return cast(pd.DataFrame, df_f.iloc[0:0].copy())

    # Non-null grid params
    for col in ("grid_lower", "grid_upper", "num_grids"):
        if col in df_f.columns:
            col_values = cast(pd.Series, df_f[col])
            df_f = cast(pd.DataFrame, df_f.loc[col_values.notna()].copy())
    logger.info("After non-null grid params: %d", len(df_f))

    # Exclude deployed
    exact_ids, prefix_ids = _load_deployed_candidate_ids(linkage_path)
    if exact_ids or prefix_ids:
        candidate_ids = cast(pd.Series, df_f["candidate_id"])
        exclude_mask = pd.Series(False, index=df_f.index)
        if exact_ids:
            exclude_mask = exclude_mask | candidate_ids.isin(list(exact_ids))
        if prefix_ids:
            exclude_mask = exclude_mask | candidate_ids.apply(
                lambda cid: any(str(cid).startswith(p) for p in prefix_ids)
            )
        n_excluded = int(exclude_mask.sum())
        df_f = cast(pd.DataFrame, df_f.loc[~exclude_mask].copy())
        logger.info("After excluding deployed bots (%d matched): %d", n_excluded, len(df_f))

    # Require candidate_id
    candidate_ids = cast(pd.Series, df_f["candidate_id"])
    df_f = cast(pd.DataFrame, df_f.loc[candidate_ids.notna()].copy())

    # Deduplicate by candidate_id (keep highest score)
    sort_cols = ["score"]
    ascending = [False]
    if bypass_only and "micro_osc_score" in df_f.columns:
        sort_cols = ["micro_osc_score", "score"]
        ascending = [False, False]
    df_f = cast(
        pd.DataFrame,
        df_f.sort_values(by=sort_cols, ascending=ascending).drop_duplicates(
            subset=["candidate_id"], keep="first"
        ),
    )

    logger.info(
        "Final backtestable candidates: %d (from %d total)", len(df_f), n_start
    )
    return cast(pd.DataFrame, df_f.reset_index(drop=True))


# ═════════════════════════════════════════════════════════════════════════════
# 3. Fetch historical klines
# ═════════════════════════════════════════════════════════════════════════════


def _parse_scan_timestamp(candidate_id: str) -> Optional[datetime]:
    """Extract a UTC datetime from a candidate_id.

    Supports both legacy (``BTCUSDT_20260224_100100``) and new config-hashed
    (``BTCUSDT_20260224_100100_a1b2c3d4``) formats.
    """
    from neutralgrid.core.candidate_id import extract_parts
    parsed = extract_parts(candidate_id)
    ts_str = parsed["scan_ts"]
    if not ts_str:
        return None
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _parse_utc_timestamp(value: Any) -> Optional[datetime]:
    """Parse a timestamp-like value into a timezone-aware UTC datetime."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            parsed = pd.to_datetime(text, errors="coerce", utc=True)
            if pd.isna(parsed):
                return None
            return cast(pd.Timestamp, parsed).to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def resolve_backtest_start_timestamp(candidate_row: Dict[str, Any] | pd.Series) -> datetime:
    """Return the authoritative kline start timestamp for candidate backtests.

    Backtests start when the deploy-ready CSV was output, not when the symbol
    was scanned. The scan timestamp remains metadata only. Missing CSV-output
    provenance is a hard error so older rows cannot silently fall back to the
    candidate_id timestamp.
    """
    for col in ("backtest_start_ts_utc", "candidate_available_ts_utc"):
        ts = _parse_utc_timestamp(candidate_row.get(col))
        if ts is not None:
            return ts
    raise ValueError(
        "candidate row is missing candidate_available_ts_utc/backtest_start_ts_utc; "
        "cannot start backtest from candidate_id scan timestamp"
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _find_exchange_symbol(exchange_info: dict[str, Any], symbol: str) -> dict[str, Any] | None:
    symbols = exchange_info.get("symbols")
    if isinstance(symbols, list):
        for item in symbols:
            if isinstance(item, dict) and str(item.get("symbol", "")).upper() == symbol.upper():
                return item
    if str(exchange_info.get("symbol", "")).upper() == symbol.upper():
        return exchange_info
    return None


def extract_exchange_filter_settings(exchange_info: dict[str, Any], symbol: str) -> dict[str, Any]:
    """Extract candidate-time exchange filters without using precision fallbacks."""
    item = _find_exchange_symbol(exchange_info, symbol)
    if item is None:
        return {
            "exchange_filter_source": "missing",
            "tick_size_source": "missing",
            "step_size_source": "missing",
            "min_notional_source": "missing",
            "exchange_filter_validation_status": "missing_symbol",
            "exchange_filter_rejection_reason": "symbol_not_found_in_exchange_info",
        }

    filters = item.get("filters", [])
    filter_map = {
        str(flt.get("filterType", "")): flt
        for flt in filters
        if isinstance(flt, dict)
    } if isinstance(filters, list) else {}
    price_filter = filter_map.get("PRICE_FILTER", {})
    lot_size = filter_map.get("LOT_SIZE", {})
    min_notional = filter_map.get("MIN_NOTIONAL", {})

    tick_size = (
        _float_or_none(price_filter.get("tickSize"))
        if isinstance(price_filter, dict)
        else None
    )
    step_size = (
        _float_or_none(lot_size.get("stepSize"))
        if isinstance(lot_size, dict)
        else None
    )
    notional_raw = (
        min_notional.get("notional", min_notional.get("minNotional"))
        if isinstance(min_notional, dict)
        else None
    )
    min_notional_value = _float_or_none(notional_raw)
    missing: list[str] = []
    if tick_size is None or tick_size <= 0:
        missing.append("PRICE_FILTER.tickSize")
    if step_size is None or step_size <= 0:
        missing.append("LOT_SIZE.stepSize")
    if min_notional_value is None or min_notional_value < 0:
        missing.append("MIN_NOTIONAL.notional")

    status = "valid" if not missing else "missing_filter"
    return {
        "exchange_filter_source": "binance_exchange_info",
        "tick_size_source": "binance_exchange_info" if tick_size is not None else "missing",
        "step_size_source": "binance_exchange_info" if step_size is not None else "missing",
        "min_notional_source": "binance_exchange_info" if min_notional_value is not None else "missing",
        "tick_size": tick_size,
        "step_size": step_size,
        "min_notional": min_notional_value,
        "tick_size_value": tick_size,
        "step_size_value": step_size,
        "min_notional_value": min_notional_value,
        "exchange_filter_validation_status": status,
        "exchange_filter_rejection_reason": ",".join(missing),
    }


def attach_mark_close(last_klines: pd.DataFrame, mark_klines: pd.DataFrame) -> pd.DataFrame:
    """Attach mark close prices for explicit mark valuation diagnostics."""
    if last_klines.empty or mark_klines.empty or "timestamp" not in mark_klines.columns:
        return last_klines
    left = last_klines.copy()
    right = mark_klines[["timestamp", "close"]].copy()
    right["mark_close"] = right["close"]
    right = right.drop(columns=["close"])
    merged = pd.merge(left, right, on="timestamp", how="left")
    return cast(pd.DataFrame, merged)


def funding_rate_series_from_rows(
    funding_rows: list[dict[str, Any]],
) -> tuple[list[float] | None, str, int]:
    """Convert Binance funding rows into engine funding-rate series metadata."""
    if not funding_rows:
        return None, "no_event_in_window", 0
    rates: list[float] = []
    for row in sorted(funding_rows, key=lambda item: int(item.get("fundingTime", 0) or 0)):
        rate = _float_or_none(row.get("fundingRate"))
        if rate is not None:
            rates.append(rate)
    if not rates:
        return None, "missing", 0
    return rates, "binance_funding_rate", len(rates)


def _coerce_funding_series(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        rates = [_float_or_none(item) for item in value]
        clean = [float(item) for item in rates if item is not None]
        return clean or None
    return None


def _validate_public_market_filters(
    *,
    symbol: str,
    lower: float,
    upper: float,
    num_grids: int,
    capital: float,
    leverage: int,
    capital_fraction: float,
    tick_size: float | None,
    step_size: float | None,
    min_notional: float | None,
    status: str,
) -> tuple[str, str]:
    if status != "valid":
        return status, "exchange_filter_status_not_valid"
    if tick_size is None or tick_size <= 0:
        return "invalid", "missing_or_invalid_tick_size"
    if step_size is None or step_size <= 0:
        return "invalid", "missing_or_invalid_step_size"
    if min_notional is None or min_notional < 0:
        return "invalid", "missing_or_invalid_min_notional"

    rounded_lower = round(round(lower / tick_size) * tick_size, 10)
    rounded_upper = round(round(upper / tick_size) * tick_size, 10)
    if rounded_lower <= 0 or rounded_upper <= rounded_lower:
        return "invalid", "rounded_bounds_not_strictly_increasing"

    avg_price = (lower + upper) / 2.0
    if avg_price <= 0:
        return "invalid", "invalid_average_grid_price"
    raw_qty = (capital * capital_fraction * leverage) / num_grids / avg_price
    rounded_qty = round(round(raw_qty / step_size) * step_size, 10)
    if rounded_qty <= 0:
        return "invalid", "rounded_quantity_not_positive"
    if rounded_qty * rounded_lower < min_notional:
        return "invalid", "min_notional_violation"
    return "valid", ""


async def fetch_historical_klines(
    client: Any,
    symbol: str,
    start_time: datetime,
    hours: int = 6,
    price_source: str = "last",
) -> pd.DataFrame:
    """Fetch ``hours`` hours of 1m OHLCV starting at ``start_time``.

    Uses the Binance klines endpoint with ``startTime`` parameter so callers
    get data from their explicit candidate-availability timestamp forward
    (not the most recent N bars).

    Returns a DataFrame with columns: timestamp, open, high, low, close, volume.

    Args:
        client: Binance API client instance
        symbol: Trading pair symbol
        start_time: UTC datetime to start fetching from
        hours: Number of hours of data to fetch
        price_source: "last" (default) or "mark" — selects kline endpoint
    """
    start_ms = int(start_time.timestamp() * 1000)
    limit = min(hours * 60, 1500)

    if price_source == "mark":
        klines = await client.get_mark_price_klines(
            symbol=symbol,
            interval="1m",
            limit=limit,
            start_time=start_ms,
        )
    else:
        klines = await client.get_klines(
            symbol=symbol,
            interval="1m",
            limit=limit,
            start_time=start_ms,
        )

    if not klines:
        logger.warning("No klines returned for symbol, returning empty DataFrame")
        return pd.DataFrame()

    kline_columns = pd.Index(
        [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ]
    )
    df = pd.DataFrame(
        klines,
        columns=kline_columns,
    )
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # AFML Ch2: Validate kline data quality before backtesting
    try:
        from neutralgrid.data.curator import DataCurator
        curator = DataCurator()
        result = curator.validate_ohlcv(
            df,
            timestamp_col="timestamp",
            timeframe="1m",
            reference_time=cast(pd.Series, df["timestamp"]).max().tz_localize("UTC"),
        )
        # Step 2: Hard-fail on critical data quality issues
        if not all(result.checks.get(c, True) for c in
                   ("missing_bars", "nan_inf", "ohlcv_semantics", "duplicates")):
            logger.warning(
                "Kline data rejected for %s: %s",
                symbol,
                [c for c, v in result.checks.items() if not v],
            )
            return pd.DataFrame()
        if not result.passed:
            logger.warning(
                "Kline data quality check failed for %s: %s — proceeding with caution",
                symbol,
                [c for c, v in result.checks.items() if not v],
            )
    except Exception as exc:
        logger.debug("DataCurator validation skipped: %s", exc)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# 4. Run single backtest
# ═════════════════════════════════════════════════════════════════════════════


def run_single_backtest(
    candidate_row: Dict[str, Any],
    klines_df: pd.DataFrame,
    capital: float = 400.0,
    leverage: int = 10,
    max_holding_bars: int = int(BOT_HORIZON_HOURS * 60),
    funding_mode: str = "continuous",
    replay_seed_dir: Optional[Path] = None,
    duration_bars: int | None = None,
    realism_profile: str = "legacy",
) -> Dict[str, Any]:
    """Run the unified realistic backtester on a single candidate.

    Delegates to ``btk_unified_runner.run_backtest()`` — the single source
    of truth for backtest execution.  Uses ``build_training_config()`` to
    apply training-standard defaults for consistent label generation.

    Parameters
    ----------
    candidate_row : dict
        Row from a scanner CSV.
    klines_df : DataFrame
        1m OHLCV data for the candidate's time window.
    capital, leverage, max_holding_bars : float/int
        Grid config overrides.
    funding_mode : str
        ``"continuous"`` (default, training standard) or ``"snapshot"``
        (legacy).
    replay_seed_dir : Path | None
        If provided and ``<dir>/<SYMBOL>/levels.csv`` exists, the backtester
        will be seeded with the replay ladder state at scan time.
    duration_bars : int | None
        If provided, truncate klines to this many bars before running the
        backtest.  Used by ``--match-live-duration`` to align BT window
        with actual live bot duration.  ``None`` = use all available bars.
    realism_profile : str
        Explicit backtest realism profile. Default ``legacy`` preserves
        existing full-pipeline behavior.

    Returns the backtest result dict (validated against label contract)
    merged with candidate metadata.
    """
    # Lazy import to avoid hard top-level dep on the script-level backtest pkg
    from backtest.btk_unified_runner import build_training_config, run_backtest
    from backtest.btk_seed_state import validate_realism_profile

    realism_profile = validate_realism_profile(realism_profile)

    grid_lower = float(candidate_row["grid_lower"])
    grid_upper = float(candidate_row["grid_upper"])
    num_grids = int(float(candidate_row["num_grids"]))

    mode_source = "default_geometric"
    mode = "geometric"
    raw_mode = candidate_row.get("mode")
    if raw_mode is None or pd.isna(raw_mode):
        raw_mode = candidate_row.get("backtest_mode")
        if raw_mode is not None and pd.notna(raw_mode):
            mode_source = "backtest_mode"
    else:
        mode_source = "candidate_mode"
    if raw_mode is not None and pd.notna(raw_mode):
        mode = str(raw_mode).strip().lower()
        if mode not in {"arithmetic", "geometric"}:
            raise ValueError(f"Invalid grid mode for {candidate_row.get('symbol', '?')}: {raw_mode!r}")

    # Use candidate's funding rate if available
    funding_rate = 0.0001
    fr = candidate_row.get("funding_rate")
    if fr is not None and pd.notna(fr):
        try:
            fr_val = float(fr)
        except (TypeError, ValueError):
            logger.warning("Invalid funding_rate '%s', using default 0.0001", fr)
            fr_val = 0.0001
        # Scanner stores rate as signed decimal (e.g. 0.0001 or -0.0003).
        # Preserve sign — engine (equity -= bar_funding) handles both directions.
        # Negative rate = shorts pay longs = grid bot receives funding income.
        funding_rate = fr_val if fr_val != 0 else 0.0001
        # Guard against extreme/corrupt rates from API
        if abs(funding_rate) > 0.01:
            logger.warning(
                "Extreme funding rate %.6f for %s, clamping to +/-0.01",
                funding_rate,
                candidate_row.get("symbol", "?"),
            )
            funding_rate = max(-0.01, min(0.01, funding_rate))

    funding_series_status = str(candidate_row.get("funding_series_status", "static"))
    if (
        realism_profile == _PUBLIC_MARKET_PROFILE
        and funding_series_status == "no_event_in_window"
    ):
        # For the explicit public-market profile, a verified empty funding
        # event window should not accrue the legacy static proxy.
        funding_rate = 0.0

    # Capital fraction from scanner/Kelly sizing (enforced in engine sizing).
    cap_frac = 1.0
    cap_frac_raw = candidate_row.get("capital_fraction")
    if cap_frac_raw is not None and pd.notna(cap_frac_raw):
        try:
            cap_frac = float(cap_frac_raw)
        except (TypeError, ValueError):
            cap_frac = 1.0
    if not np.isfinite(cap_frac):
        cap_frac = 1.0
    cap_frac = max(0.0, min(1.0, cap_frac))

    # Volatility targeting must be explicit.  Do not derive it from
    # range_size_pct, because that silently changes deployed capital.
    vol_proxy_pct = None
    vol_proxy_raw = candidate_row.get("volatility_proxy_pct")
    if vol_proxy_raw is not None and pd.notna(vol_proxy_raw):
        try:
            vol_proxy_val = float(vol_proxy_raw)
            if vol_proxy_val > 0 and np.isfinite(vol_proxy_val):
                vol_proxy_pct = vol_proxy_val
        except (TypeError, ValueError):
            vol_proxy_pct = None

    vol_target_pct = None
    vol_target_raw = candidate_row.get("volatility_target_pct")
    if vol_target_raw is not None and pd.notna(vol_target_raw):
        try:
            vol_target_val = float(vol_target_raw)
            if vol_target_val > 0 and np.isfinite(vol_target_val):
                vol_target_pct = vol_target_val
        except (TypeError, ValueError):
            vol_target_pct = None

    tick_size = _float_or_none(candidate_row.get("tick_size"))
    if tick_size is None:
        tick_size = _float_or_none(candidate_row.get("tick_size_value"))
    step_size = _float_or_none(candidate_row.get("step_size"))
    if step_size is None:
        step_size = _float_or_none(candidate_row.get("step_size_value"))
    min_notional = _float_or_none(candidate_row.get("min_notional"))
    if min_notional is None:
        min_notional = _float_or_none(candidate_row.get("min_notional_value"))
    exchange_filter_status = str(
        candidate_row.get("exchange_filter_validation_status", "not_requested")
    )
    exchange_filter_reason = str(candidate_row.get("exchange_filter_rejection_reason", ""))
    if realism_profile == _PUBLIC_MARKET_PROFILE:
        exchange_filter_status, exchange_filter_reason = _validate_public_market_filters(
            symbol=str(candidate_row["symbol"]),
            lower=grid_lower,
            upper=grid_upper,
            num_grids=num_grids,
            capital=capital,
            leverage=leverage,
            capital_fraction=cap_frac,
            tick_size=tick_size,
            step_size=step_size,
            min_notional=min_notional,
            status=exchange_filter_status,
        )
        if exchange_filter_status != "valid":
            raise ValueError(
                "candidate-time public-market exchange filters invalid for "
                f"{candidate_row.get('symbol', '?')}: {exchange_filter_reason}"
            )

    funding_rate_series = _coerce_funding_series(candidate_row.get("funding_rate_series"))
    valuation_price_source = "last"
    if realism_profile == _PUBLIC_MARKET_PROFILE and "mark_close" in klines_df.columns:
        mark_non_null = cast(pd.Series, klines_df["mark_close"]).notna()
        if bool(mark_non_null.any()):
            valuation_price_source = "mark"

    cfg = build_training_config(
        symbol=str(candidate_row["symbol"]),
        lower=grid_lower,
        upper=grid_upper,
        num_grids=num_grids,
        mode=mode,
        capital=capital,
        capital_fraction=cap_frac,
        volatility_target_pct=vol_target_pct,
        volatility_proxy_pct=vol_proxy_pct,
        leverage=leverage,
        max_holding_bars=max_holding_bars,
        funding_rate=funding_rate,
        funding_mode=funding_mode,
        funding_rate_series=funding_rate_series,
        tick_size=tick_size,
        step_size=step_size,
        valuation_price_source=valuation_price_source,
    )

    # Optional: load replay seed state
    seed_state = None
    if replay_seed_dir is not None:
        if realism_profile != "legacy":
            raise ValueError(
                "replay_seed_dir cannot be combined with "
                f"realism_profile={realism_profile!r}"
            )
        symbol = str(candidate_row["symbol"])
        scan_ts = _parse_scan_timestamp(str(candidate_row.get("candidate_id", "")))
        if scan_ts is not None:
            try:
                from backtest.btk_replay_seed_loader import load_seed_from_replay
                seed = load_seed_from_replay(replay_seed_dir, symbol, scan_ts)
                if seed is not None:
                    seed_state = seed
                    logger.info("Seeded %s from replay at %s", symbol, scan_ts)
            except Exception as exc:
                logger.debug("Replay seed skipped for %s: %s", symbol, exc)

    # P4: Duration override — truncate klines to match live bot duration
    if duration_bars is not None and duration_bars >= 0 and len(klines_df) > duration_bars:
        klines_df = klines_df.iloc[:duration_bars].copy()

    # Delegate to unified runner (validates contract, serializes settings)
    result = run_backtest(
        cfg,
        klines_df,
        seed_state=seed_state,
        realism_profile=realism_profile,
    )

    # Merge candidate metadata (pipeline-specific, not engine concern)
    result["candidate_id"] = candidate_row.get("candidate_id", "")
    result["scan_file"] = candidate_row.get("scan_file", "")
    result["score"] = candidate_row.get("score")
    # Use the robust regex parser — the prior string split was lossy for hashed IDs
    _parsed_scan_ts = _parse_scan_timestamp(str(candidate_row.get("candidate_id", "")))
    result["scan_timestamp"] = _parsed_scan_ts.isoformat() if _parsed_scan_ts else ""
    _backtest_start_ts = _parse_utc_timestamp(candidate_row.get("backtest_start_ts_utc"))
    if _backtest_start_ts is None:
        _backtest_start_ts = _parse_utc_timestamp(candidate_row.get("candidate_available_ts_utc"))
    result["backtest_start_ts_utc"] = (
        _backtest_start_ts.isoformat() if _backtest_start_ts is not None else ""
    )
    result["backtest_time_anchor"] = (
        str(candidate_row.get("candidate_available_source", "deployment_ready_csv_mtime"))
        if _backtest_start_ts is not None else "missing"
    )
    result["klines_fetched"] = len(klines_df)
    result["backtest_timestamp"] = datetime.now(timezone.utc).isoformat()
    result["mode_source"] = mode_source
    result["capital_fraction_input"] = cap_frac
    result["volatility_target_pct_input"] = vol_target_pct
    result["volatility_proxy_pct_input"] = vol_proxy_pct
    result["exchange_filter_source"] = candidate_row.get("exchange_filter_source", "not_requested")
    result["tick_size_source"] = candidate_row.get("tick_size_source", "not_requested")
    result["step_size_source"] = candidate_row.get("step_size_source", "not_requested")
    result["min_notional_source"] = candidate_row.get("min_notional_source", "not_requested")
    result["min_notional"] = min_notional
    result["exchange_filter_validation_status"] = exchange_filter_status
    result["exchange_filter_rejection_reason"] = exchange_filter_reason
    result["funding_series_status"] = funding_series_status
    result["funding_series_source"] = candidate_row.get("funding_series_source", "static")
    result["funding_series_rows"] = candidate_row.get("funding_series_rows", 0)
    result["mark_price_source"] = candidate_row.get("mark_price_source", "not_requested")
    result["mark_price_rows"] = candidate_row.get("mark_price_rows", 0)
    result["historical_depth_source"] = candidate_row.get("historical_depth_source", "not_requested")

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 5. Build feature dict from scanner row
# ═════════════════════════════════════════════════════════════════════════════


def build_feature_dict_from_scanner_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map scanner CSV columns to training features via ``_SCANNER_TO_FEATURE``.

    Missing features are set to NaN.
    """
    features: Dict[str, Any] = {}
    seen: set[str] = set()

    for src_col, dst_col in _SCANNER_TO_FEATURE.items():
        if dst_col in seen:
            continue
        val = row.get(src_col)
        if val is not None and pd.notna(val):
            if dst_col == "mode":
                mode = str(val).strip().lower()
                if mode in {"arithmetic", "geometric"}:
                    features[dst_col] = mode
                    seen.add(dst_col)
                continue
            try:
                features[dst_col] = float(val)
            except (ValueError, TypeError):
                continue
            seen.add(dst_col)

    # Fill any missing training features with NaN
    for feat in TRAINING_OUTPUT_COLUMNS + OPTIONAL_LIVE_PLUS_FIELDS_V20260312 + [
        "long_short_ratio",
    ]:
        if feat not in features:
            features[feat] = float("nan")

    return features


# ═════════════════════════════════════════════════════════════════════════════
# 6. Convert backtest result to training row
# ═════════════════════════════════════════════════════════════════════════════


def convert_to_training_row(
    backtest_result: Dict[str, Any],
    candidate_row: Dict[str, Any],
    hurdle_pct: Optional[float] = None,
    pt_pct: float = 15.0,
    sl_pct: float = -12.0,
    horizon_hours: float = BOT_HORIZON_HOURS,
) -> Dict[str, Any]:
    """Convert a backtest result + scanner row into a meta-labeler training row.

    Labels are derived from the backtest net PnL using the same thresholds as
    the triple-barrier labeler:
    - y = 1 if net_pnl_pct >= hurdle_pct (success)
    - y = 0 otherwise
    - barrier_touched inferred from pnl magnitude
    """
    # Step 5D: resolve hurdle from config single source of truth
    if hurdle_pct is None:
        hurdle_pct = get_config().barrier.meta_hurdle_pct
    net_pnl_pct = float(backtest_result.get("net_pnl_pct", 0.0))

    # ── alignment-v1: extract realized decomposition ────────────────────
    _raw_realized = backtest_result.get("realized_net_pnl_pct")
    realized_net_pnl_pct: float | None = (
        float(_raw_realized) if _raw_realized is not None else None
    )
    _raw_uf = backtest_result.get("unrealized_fraction")
    unrealized_fraction: float | None = (
        float(_raw_uf) if _raw_uf is not None else None
    )

    # AFML Ch3 deviation: barrier inferred from terminal PnL, not first-touch
    # price path. AFML requires detecting which barrier was hit FIRST in the
    # path (PT, SL, or time). Current implementation uses terminal PnL as a
    # proxy because the backtest engine returns final state, not first-touch
    # events. This is a known approximation for grid-bot semantics where
    # the "barrier" concept maps to range containment rather than directional
    # price targeting.
    # Infer barrier touched
    if net_pnl_pct >= pt_pct:
        barrier = "pt"
    elif net_pnl_pct <= sl_pct:
        barrier = "sl"
    else:
        barrier = "time"

    # Binary label — require the MTM horizon label from validated engine output.
    # The label_positive_by_horizon field is guaranteed when results come from
    # btk_unified_runner.run_backtest().  The fallback path below handles legacy
    # result dicts that predate the label contract.
    # Gate 6 (Fix 2): Track label_source and is_authoritative based on
    # whether hierarchical labels are available or fallback was used.
    label_from_bt = backtest_result.get("label_positive_by_horizon")
    _label_source: str = "hierarchical"
    _label_is_authoritative: bool = True
    if label_from_bt is not None:
        y = 1 if label_from_bt else 0
    else:
        logger.warning(
            "Engine result for %s missing 'label_positive_by_horizon'. "
            "Falling back to hurdle comparison (%.1f%%). "
            "Use btk_unified_runner.run_backtest() for contract compliance.",
            backtest_result.get("symbol", "?"),
            hurdle_pct,
        )
        y = 1 if net_pnl_pct >= hurdle_pct else 0
        _label_source = "hurdle_fallback"
        _label_is_authoritative = False
    sl_hit = barrier == "sl"

    # Parse scan timestamp for t1
    # Gate 6 (Fix 3): Detect truncated backtests and synthesise t1 from
    # actual end time instead of the canonical horizon to avoid legitimising
    # truncated runs.
    scan_ts = _parse_scan_timestamp(str(candidate_row.get("candidate_id", "")))
    try:
        event_start_ts = resolve_backtest_start_timestamp(candidate_row)
    except ValueError:
        event_start_ts = scan_ts
    start_time_utc = event_start_ts.isoformat() if event_start_ts else ""

    _t1_is_synthetic: bool = False
    _t1_truncated: bool = False
    if event_start_ts:
        _bt_duration_hours = float(backtest_result.get("duration_hours", 0.0))
        _bt_sl_hit = barrier == "sl"
        # A backtest is truncated if it ended early due to SL hit or
        # its duration is materially shorter than the target horizon
        # (allow 5% tolerance for rounding).
        _is_truncated = (
            _bt_sl_hit
            or (_bt_duration_hours > 0 and _bt_duration_hours < horizon_hours * 0.95)
        )
        if _is_truncated and _bt_duration_hours > 0:
            # Use actual end time, not canonical horizon
            t1 = (event_start_ts + timedelta(hours=_bt_duration_hours)).isoformat()
            _t1_is_synthetic = True
            _t1_truncated = True
        else:
            t1 = (event_start_ts + timedelta(hours=horizon_hours)).isoformat()
            _t1_is_synthetic = True
    else:
        t1 = ""

    # Build feature snapshot from scanner row (scan-time features, no look-ahead)
    features = build_feature_dict_from_scanner_row(candidate_row)

    duration_hours = float(backtest_result.get("duration_hours", 0.0))
    hmm_artifact_version = candidate_row.get("hmm_artifact_version")
    if hmm_artifact_version is None or pd.isna(hmm_artifact_version):
        hmm_artifact_version = candidate_row.get("active_hmm_artifact_version")

    row: Dict[str, Any] = {
        # Identifiers
        "symbol": str(candidate_row.get("symbol", "")),
        "start_time_utc": start_time_utc,
        # AFML event metadata
        "t1": t1,
        "t1_is_synthetic": _t1_is_synthetic,
        "t1_truncated": _t1_truncated,
        "pnl_pct": net_pnl_pct,
        "y": y,
        "sl_hit": sl_hit,
        "barrier_touched": barrier,
        # Gate 6 (Fix 2): Label provenance — track how the label was derived
        "label_source": _label_source,
        # price_end = close price at final simulated bar. For grid bots this is
        # a proxy for AFML barrier_price since PnL comes from grid fills, not
        # directional price movement. Metadata only — not a model feature.
        "barrier_price": backtest_result.get("price_end"),
        "duration_hours": duration_hours,
        # PIPELINE_FIX v2: endogenous time-to-target (first bar the net MTM PnL
        # curve reaches +3% of capital) — the speed measure for the fast-winner
        # label. Outcome/label column, guarded in _KNOWN_LABEL_COLUMNS; NEVER a
        # model feature. NaN when the +3% target was never reached in the window.
        "time_to_target_hours": backtest_result.get("time_to_target_hours"),
        "target_reached": backtest_result.get("target_reached"),
        # H*-censoring flag (DURATION_FIX §21.4). True iff ≥1 position was
        # force-closed at max_holding_bars. Audit/provenance only — never
        # added to ALL_META_FEATURES (would leak label-process info).
        "horizon_censored": bool(backtest_result.get("horizon_censored", False)),
        # Provenance
        "source": "backtest",
        "hmm_feature_source": "backtest",  # Gate 5 (Fix 5b): explicit lineage
        "hmm_artifact_version": hmm_artifact_version,
        "sample_weight_override": 0.5,
        "candidate_id": candidate_row.get("candidate_id", ""),
        "backtest_timestamp": backtest_result.get("backtest_timestamp"),
        "ev_contract_fingerprint": candidate_row.get("ev_contract_fingerprint"),
        "capital_fraction": candidate_row.get("capital_fraction"),
        "capital_used": backtest_result.get("capital_used"),
        "volatility_scale_applied": backtest_result.get("volatility_scale_applied"),
        # Excursion outcomes (MTM path; contract v2.3+)
        "mae": backtest_result.get("mae"),
        "mfe": backtest_result.get("mfe"),
        "mae_pct_initial": backtest_result.get("mae_pct_initial"),
        "mfe_pct_initial": backtest_result.get("mfe_pct_initial"),
        # ── alignment-v1: realized PnL decomposition + rotation metrics ──
        "realized_net_pnl": backtest_result.get("realized_net_pnl"),
        "realized_net_pnl_pct": realized_net_pnl_pct,
        "unrealized_fraction": unrealized_fraction,
    }
    # Merge features
    row.update(features)

    # PnL curve structural features (if produced by unified runner)
    for field in _PNL_CURVE_FIELDS:
        val = backtest_result.get(field)
        if val is not None:
            row[field] = val

    # Enhancement 5: Hierarchical training labels — primary training target.
    # The hierarchical deploy decision (hlabel_meta) replaces the raw horizon
    # label as the training target.  It is stricter: requires geometry survival
    # (L1), execution viability (L2), AND net return hurdle (L3).
    # alignment-v1: pass realized PnL + unrealized fraction + range_breakout.
    # Step 3A: preserve engine horizon label before try (not lost on exception)
    row["y_horizon"] = row["y"]
    try:
        from neutralgrid.training.hierarchical_labels import HierarchicalLabeler
        h_labeler = HierarchicalLabeler()

        # Derive range_breakout from time_in_range_pct (Step 8 of plan)
        _tir = float(backtest_result.get("time_in_range_pct", 100.0))
        _range_breakout = _tir < 50.0

        h_result = h_labeler.label(
            liquidated=bool(backtest_result.get("liquidated", False)),
            max_drawdown_pct=float(backtest_result.get("max_drawdown_pct", 0.0)),
            range_breakout=_range_breakout,
            total_fills=int(backtest_result.get("total_trades", 0)),
            duration_hours=duration_hours,
            net_pnl_pct=net_pnl_pct,
            realized_net_pnl_pct=realized_net_pnl_pct,
            unrealized_fraction=unrealized_fraction,
            fees_paid=float(backtest_result.get("fees_paid", 0.0)),
            funding_fees=float(backtest_result.get("funding_fees", 0.0)),
        )
        h_dict = h_result.to_dict()
        # Hierarchical deploy decision becomes the primary training target
        hlabel_y = h_dict.pop("y", None)
        row.update(h_dict)
        if hlabel_y is not None:
            row["y"] = hlabel_y
            # Gate 6 (Fix 2): Hierarchical labels succeeded — authoritative
            row["label_source"] = "hierarchical"
            _label_is_authoritative = True
    except Exception as e:
        logger.warning("Hierarchical labeling failed for %s: %s",
                       candidate_row.get("symbol", "?"), e)
        row["hlabel"] = float("nan")
        row["hlabel_meta"] = float("nan")
        row["hlabel_L1"] = float("nan")
        row["hlabel_L2"] = float("nan")
        row["hlabel_L3"] = float("nan")
        row["hlabel_reason"] = ""
        # Gate 6 (Fix 2): Hierarchical labeling failed — not authoritative
        row["label_source"] = "hurdle_fallback"
        _label_is_authoritative = False

    # Gate 6 (Fix 2): Set is_authoritative based on label derivation path.
    # Only set if not already provided by the engine (Step 11 provenance
    # loop below may override from backtest_result).
    row["is_authoritative"] = _label_is_authoritative

    # ── alignment-v1: rotation metrics (derived training features) ───────
    _capital_used = float(backtest_result.get("capital_used", 1.0))
    if _capital_used > 0 and duration_hours > 0 and realized_net_pnl_pct is not None:
        realized_pnl_per_margin_hour = (realized_net_pnl_pct / 100.0) / duration_hours
        _uf = unrealized_fraction if unrealized_fraction is not None else 0.0
        rotation_score = realized_pnl_per_margin_hour * (1.0 - _uf)
    else:
        realized_pnl_per_margin_hour = 0.0
        rotation_score = 0.0
    row["realized_pnl_per_margin_hour"] = realized_pnl_per_margin_hour
    row["rotation_score"] = rotation_score

    # Preserve engine settings from the validated result so downstream
    # ingestion can gate stale physics, not only stale label dates.
    from backtest.btk_label_contract import ENGINE_SETTINGS_FIELDS

    for provenance_field in tuple(dict.fromkeys([
        *ENGINE_SETTINGS_FIELDS,
        "backtest_run_id",
        "formula_version",
        "is_authoritative",
        "mode_source",
        "realism_profile",
        "seed_state_source",
        "exchange_filter_source",
        "tick_size_source",
        "step_size_source",
        "min_notional_source",
        "min_notional",
        "exchange_filter_validation_status",
        "exchange_filter_rejection_reason",
        "funding_series_status",
        "funding_series_source",
        "funding_series_rows",
        "fill_price_source",
        "valuation_price_source",
        "mark_price_source",
        "mark_price_rows",
        "historical_depth_source",
    ])):
        prov_val = backtest_result.get(provenance_field)
        if prov_val is not None:
            row[provenance_field] = prov_val

    # Fallback: formula_version from constants if not in engine result
    if "formula_version" not in row:
        from neutralgrid.core.constants import FORMULA_VERSION
        row["formula_version"] = FORMULA_VERSION

    return row
