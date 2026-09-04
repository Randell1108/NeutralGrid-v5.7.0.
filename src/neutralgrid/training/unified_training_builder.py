"""
Unified training table builder — snapshot-only architecture.

Single authoritative path: snapshots (``data/training_snapshots/*.parquet``)
are the sole feature source.  Backtest CSVs supply outcomes only.  Features
are joined to outcomes by ``candidate_id``.

Legacy paths (expired_bots.xlsx, backfill, scanner-CSV backfill) have been
removed.  All training rows must originate from a point-in-time snapshot
captured at scan time.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd

from neutralgrid.backtest.candidate_pipeline import (
    OPTIONAL_LIVE_PLUS_FIELDS_V20260312,
)
from neutralgrid.backtest.realism_governance import (
    CANDIDATE_TIME_PUBLIC_MARKET_PROFILE,
    REALISM_PROFILES,
    SHADOW_REALISM_PROFILES,
)
from neutralgrid.core.config import get_config
from neutralgrid.core.constants import (
    ENGINE_VERSION,
    FORMULA_VERSION,
    LABEL_CONTRACT_VERSION,
    MIN_POSITIVE_RATE,
)
from neutralgrid.grid.formulas import ARITHMETIC, GEOMETRIC, grid_spacing_pct
from neutralgrid.scanner.pnl_ranker import PnLRanker, RankingConfig
from neutralgrid.training.data_generator import (
    DEFAULT_SCHEMA,
    BarrierLabelGenerator,
    HMM_FEATURE_SEMANTICS_VERSION,
    LabelConfig,
    TRAINING_DATASET_SCHEMA_VERSION,
)
from neutralgrid.validation.utility import (
    UtilityCalibratorUnavailable,
    compute_governed_provisional_utility,
)

# Module-level guard so the utility-unavailable warning is emitted once per
# training run; subsequent occurrences log at debug level to avoid flooding.
_utility_unavailable_warning_emitted = False

logger = logging.getLogger(__name__)

TRAINING_FEATURES = list(DEFAULT_SCHEMA.features)
EXTRA_META_FEATURES = list(OPTIONAL_LIVE_PLUS_FIELDS_V20260312) + [
    "open_interest",
    "micro_round_trip_cost_pct",
    # afml_enriched_v20260318 additions
    "hmm_tail_cvar_95",
    "tos_gcf",
    "persistence_prob",
    # Tier 4 context features
    "long_short_ratio",
    "funding_rate_zscore",
    "open_interest_change_pct",
    "bb_width_ratio_1h_15m",
    # Micro-oscillator archetype
    "micro_osc_score",
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
    # GRID_SYNCH §1.5 — `mode` is bot-side metadata carried for audit / training-time
    # branching only. It must NOT enter the model X-matrix (categorical string would
    # crash classifiers / mis-encode under label-encoding). See plan §6 invariant C3
    # and meta_labeler.py:138 (ACTIVE_SNAPSHOT_META_FEATURES does NOT contain it).
    "mode",
]
ALL_META_FEATURES = list(dict.fromkeys(TRAINING_FEATURES + EXTRA_META_FEATURES))
# FASTWIN v2 (2026-06-08): per-symbol row cap for build_meta_labeler_pool. The
# fast-winner signal is regime-dominated and validated on a symbol-grouped,
# time-purged K-fold; balancing per-symbol counts keeps the folds balanced and
# maximizes the leakage-safe AUC. 30 is the validated optimum (see the cap step
# in build_meta_labeler_pool). 0 disables.
META_POOL_MAX_ROWS_PER_SYMBOL = 30
_SNAPSHOT_GEOMETRY_BACKFILL_ALLOWED = frozenset({"grid_spacing_pct"})
_VALID_GRID_MODES = {ARITHMETIC, GEOMETRIC}

# ERR-083: the former _SCAN_TO_FEATURE mapping here was DEAD CODE
# (definition-only, no consumer). The LIVE scan_* alias normalization is
# neutralgrid.models.meta_labeler._INFERENCE_FEATURE_ALIASES, applied via
# normalize_inference_feature_frame in build_inference_frame below. The
# Feature Pipeline Update Rule (safety-invariants.md) trio site 3 now names
# that mapping.

_EXCURSION_OUTCOME_FIELDS = (
    "mae",
    "mfe",
    "mae_pct_initial",
    "mfe_pct_initial",
)

_PNL_CURVE_COLUMNS = (
    "pnl_curve_fill_concentration",
    "pnl_curve_active_ratio",
    "pnl_curve_time_to_peak",
    "pnl_curve_flatline_pct",
    "pnl_curve_entropy",
    "pnl_curve_shape_class",
    "pnl_curve_points",
    "pnl_curve_raw_json",
)

_LINEAGE_COLUMNS = [
    "snapshot_matched_at_utc",
    "hmm_trained_at_utc",
    "hmm_artifact_version",
    "hmm_pipeline_version",
    "hmm_feature_semantics_version",
    "hmm_feature_source",
    "hmm_calibration_status",
    "ev_contract_fingerprint",
    "label_source",
    "dataset_schema_version",
]

_PUBLIC_MARKET_PROVENANCE_COLUMNS = (
    "exchange_filter_source",
    "tick_size_source",
    "step_size_source",
    "min_notional_source",
    "exchange_filter_validation_status",
    "funding_series_status",
    "funding_series_source",
    "fill_price_source",
    "valuation_price_source",
    "mark_price_source",
    "historical_depth_source",
)


def _to_datetime_utc_mixed(values: Any) -> pd.Series:
    """Parse timestamps robustly across mixed source formats."""
    index = values.index if isinstance(values, pd.Series) else None
    try:
        parsed = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if isinstance(parsed, pd.Series):
        return parsed
    return pd.Series(parsed, index=index)


def _non_empty_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _as_float(value: Any) -> float | None:
    numeric = cast(pd.Series, pd.to_numeric(pd.Series([value]), errors="coerce")).iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _series_mean_float(values: pd.Series) -> float:
    if len(values) == 0:
        return 0.0
    return float(cast(Any, values.mean()))


def _as_int(value: Any) -> int | None:
    numeric = _as_float(value)
    if numeric is None or not np.isfinite(numeric):
        return None
    return int(round(numeric))


_BOOL_TRUE_STRINGS = frozenset({"true", "1", "yes"})
_BOOL_FALSE_STRINGS = frozenset({"false", "0", "no"})


def _as_bool_strict(value: Any) -> bool | None:
    """Parse a value that was produced as a Python bool but may have been
    round-tripped through CSV as an object-typed string.

    Returns ``None`` for missing values so the caller can omit the field.
    Raises ``ValueError`` for non-missing values that cannot be resolved to
    a boolean — silent coercion is the exact bug this helper prevents
    (bool("False") == True in vanilla Python).
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (int, np.integer)):
        ivalue = int(value)
        if ivalue in (0, 1):
            return bool(ivalue)
        raise ValueError(f"horizon_censored numeric value out of range: {value!r}")
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    lowered = text.lower()
    if lowered in _BOOL_TRUE_STRINGS:
        return True
    if lowered in _BOOL_FALSE_STRINGS:
        return False
    raise ValueError(f"horizon_censored value is not bool-like: {value!r}")


def _label_from_any(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1
    if text in {"false", "0", "no"}:
        return 0
    numeric = _as_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def _derive_excursion_from_engine_fields(
    row: Any,
    out: dict[str, Any],
    symbol: str,
) -> None:
    """Approximate excursion metrics when raw columns are absent."""
    max_dd_pct = _as_float(row.get("max_drawdown_pct"))
    peak_equity = _as_float(row.get("peak_equity"))
    max_runup_pct = _as_float(row.get("max_runup_pct"))
    net_pnl = _as_float(row.get("net_pnl"))
    net_pnl_pct = _as_float(row.get("net_pnl_pct"))
    capital_used = _as_float(row.get("capital_used"))

    starting_capital = capital_used
    if starting_capital is None and net_pnl is not None and net_pnl_pct not in (None, 0.0):
        starting_capital = abs(net_pnl * 100.0 / cast(float, net_pnl_pct))

    if "mae" not in out and max_dd_pct is not None and peak_equity is not None:
        out["mae"] = peak_equity * max_dd_pct / 100.0
        logger.debug("Derived mae for %s from peak_equity/max_drawdown_pct", symbol)

    if "mae_pct_initial" not in out:
        if "mae" in out and starting_capital not in (None, 0.0):
            out["mae_pct_initial"] = (out["mae"] / cast(float, starting_capital)) * 100.0
        elif max_dd_pct is not None:
            out["mae_pct_initial"] = max_dd_pct

    if "mfe" not in out and peak_equity is not None and starting_capital is not None:
        mfe = max(peak_equity - starting_capital, 0.0)
        if mfe > 0:
            out["mfe"] = mfe

    if "mfe_pct_initial" not in out:
        if "mfe" in out and starting_capital not in (None, 0.0):
            out["mfe_pct_initial"] = (out["mfe"] / cast(float, starting_capital)) * 100.0
        elif max_runup_pct is not None:
            out["mfe_pct_initial"] = max_runup_pct


class UnifiedTrainingBuilder:
    """Build a training table from snapshots (features) and backtests (outcomes).

    Architecture: snapshot-only.  The ``data/training_snapshots/*.parquet``
    files are the single authoritative feature source.  Backtest CSVs in
    ``data/backtest_candidates/`` supply outcomes (``net_pnl_pct``, ``y``,
    hierarchical labels, etc.).  Rows are joined by ``candidate_id``.
    """

    def __init__(
        self,
        backtest_results_dir: Optional[Path] = None,
        snapshot_dir: Optional[Path] = None,
        label_config: Optional[LabelConfig] = None,
        include_reconstruction: bool = False,
        # Kept for backward-compat with callers that still pass these; ignored.
        expired_bots_path: Optional[Path] = None,
        linkage_dir: Optional[Path] = None,
        scanner_results_dir: Optional[Path] = None,
        snapshot_tolerance: Optional[pd.Timedelta] = None,
        min_bot_date: str = "",
    ) -> None:
        self._bt_dir = Path(backtest_results_dir) if backtest_results_dir else Path("data/backtest_candidates")
        self._snapshot_dir = Path(snapshot_dir) if snapshot_dir else Path("data/training_snapshots")
        self._label_config = label_config or LabelConfig()
        self._label_gen = BarrierLabelGenerator(self._label_config)
        self._include_reconstruction = include_reconstruction
        self._ranker = PnLRanker(RankingConfig())

    def build(self) -> pd.DataFrame:
        """Build training table: snapshots (features) joined to backtests (outcomes).

        Snapshots are the sole authoritative feature source.  Backtest CSVs
        supply outcomes only.  Rows are inner-joined by ``candidate_id``.
        """

        # ── Step 1: Load snapshot features (authoritative feature source) ─
        snapshot_df = self._load_snapshots()
        if snapshot_df.empty:
            logger.warning("No snapshot data — training table will be empty")
            return pd.DataFrame()

        # ── Step 2: Load backtest outcome rows ───────────────────────
        backtest_df = self._load_backtest_rows()
        if backtest_df.empty:
            logger.warning("No backtest training data — training table will be empty")
            return pd.DataFrame()

        # ── Step 3: Join snapshot features to backtest outcomes ───────
        combined = self._join_snapshots_to_outcomes(snapshot_df, backtest_df)
        if combined.empty:
            logger.warning("No snapshot-outcome matches — training table will be empty")
            return pd.DataFrame()
        logger.info("Joined %d snapshot-outcome rows", len(combined))

        # ── Step 4: Ensure all meta-feature columns exist ─────────────
        # Validate the active feature profile against snapshot data.
        from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES
        _missing_profile = [
            f for f in ACTIVE_SNAPSHOT_META_FEATURES
            if f not in combined.columns
        ]
        if _missing_profile:
            logger.warning(
                "Snapshot data missing %d/%d profile features (will be NaN): %s",
                len(_missing_profile), len(ACTIVE_SNAPSHOT_META_FEATURES),
                ", ".join(_missing_profile),
            )

        for feature in ALL_META_FEATURES:
            if feature not in combined.columns:
                combined[feature] = np.nan

        required_cols = {
            "source": "snapshot",
            "sample_weight_override": 1.0,
            "candidate_id": "",
            "feature_source": "snapshot",
            "feature_count": 0,
        }
        for col, default in required_cols.items():
            if col not in combined.columns:
                combined[col] = default

        for col in _EXCURSION_OUTCOME_FIELDS + _PNL_CURVE_COLUMNS:
            if col not in combined.columns:
                combined[col] = np.nan

        for col in _LINEAGE_COLUMNS:
            if col not in combined.columns:
                combined[col] = ""

        if "start_time_utc" in combined.columns:
            combined["start_time_utc"] = _to_datetime_utc_mixed(combined["start_time_utc"])
            combined = cast(pd.DataFrame, combined.sort_values("start_time_utc").reset_index(drop=True))

        combined["feature_count"] = cast(
            pd.DataFrame, combined[ALL_META_FEATURES]
        ).notna().sum(axis=1)

        # ── Step 5: Authority and version filtering ───────────────────
        pre_filter_count = len(combined)

        # Filter reconstruction rows
        n_reconstruction = 0
        if "source_class" in combined.columns:
            reconstruction_mask = cast(pd.Series, combined["source_class"]).eq("reconstruction")
            n_reconstruction = int(reconstruction_mask.sum())
            if n_reconstruction > 0:
                if self._include_reconstruction:
                    logger.warning(
                        "Gate 6: including %d source_class=reconstruction rows "
                        "(include_reconstruction=True)",
                        n_reconstruction,
                    )
                else:
                    combined = cast(
                        pd.DataFrame,
                        combined.loc[~reconstruction_mask].reset_index(drop=True),
                    )
                    logger.info(
                        "Gate 6: excluded %d source_class=reconstruction rows "
                        "(use include_reconstruction=True to include them)",
                        n_reconstruction,
                    )

        # Filter version-gated rows
        n_version_gated = 0
        version_gated_rows = pd.DataFrame()
        if "version_gated" in combined.columns:
            vg_series = cast(pd.Series, combined["version_gated"])
            vg_mask = cast(pd.Series, vg_series.where(vg_series.notna(), False)).astype(bool)
            n_version_gated = int(vg_mask.sum())
            if n_version_gated > 0:
                version_gated_rows = cast(pd.DataFrame, combined.loc[vg_mask].copy())
                combined = cast(
                    pd.DataFrame,
                    combined.loc[~vg_mask].reset_index(drop=True),
                )
                logger.info(
                    "Gate 6: excluded %d version_gated rows",
                    n_version_gated,
                )

        if combined.empty and not version_gated_rows.empty:
            bootstrap_rows = version_gated_rows
            if "version_gate_reason" in bootstrap_rows.columns:
                contract_reason = (
                    cast(pd.Series, bootstrap_rows["version_gate_reason"])
                    .fillna("")
                    .astype(str)
                    .str.contains(
                        "engine_version|formula_version|mode|fill_mode|"
                        "global_cooldown_bars|cb_enabled|is_authoritative|"
                        "realism_profile",
                        regex=True,
                    )
                )
                bootstrap_rows = cast(pd.DataFrame, bootstrap_rows.loc[~contract_reason].copy())
            hstar_cols = {"y_horizon", "label_contract_version"}
            if not bootstrap_rows.empty and hstar_cols.issubset(bootstrap_rows.columns):
                combined = bootstrap_rows.reset_index(drop=True)
                combined["version_gate_relaxed_for_bootstrap"] = True
                logger.warning(
                    "Gate 6 bootstrap relaxation: no current-version rows remain; "
                    "including %d version_gated rows with H* horizon labels for "
                    "model recovery. version_gated remains True for audit.",
                    len(combined),
                )

        # Filter non-authoritative rows
        n_non_authoritative = 0
        if "is_authoritative" in combined.columns:
            non_auth_mask = cast(pd.Series, combined["is_authoritative"]).eq(False)
            n_non_authoritative = int(non_auth_mask.sum())
            if n_non_authoritative > 0:
                combined = cast(
                    pd.DataFrame,
                    combined.loc[~non_auth_mask].reset_index(drop=True),
                )
                logger.info(
                    "Gate 6: excluded %d is_authoritative=False rows",
                    n_non_authoritative,
                )

        total_excluded = pre_filter_count - len(combined)
        if total_excluded > 0:
            logger.info(
                "Gate 6: authority/version filter removed %d/%d rows "
                "(reconstruction=%d, version_gated=%d, non_authoritative=%d)",
                total_excluded, pre_filter_count,
                n_reconstruction if not self._include_reconstruction else 0,
                n_version_gated,
                n_non_authoritative,
            )

        if (
            not combined.empty
            and "y" in combined.columns
            and "y_horizon" in combined.columns
        ):
            y_current = cast(pd.Series, cast(pd.Series, combined["y"]).map(_label_from_any))
            y_hstar = cast(pd.Series, cast(pd.Series, combined["y_horizon"]).map(_label_from_any))
            current_non_null = cast(pd.Series, y_current.dropna())
            hstar_non_null = cast(pd.Series, y_hstar.dropna())
            current_pos_rate = _series_mean_float(current_non_null)
            hstar_pos_rate = _series_mean_float(hstar_non_null)
            if (
                current_pos_rate < MIN_POSITIVE_RATE
                and hstar_non_null.nunique() > 1
                and hstar_pos_rate >= MIN_POSITIVE_RATE
            ):
                combined["y"] = y_hstar
                combined["label_source"] = "y_horizon_hstar_bootstrap_post_filter"
                combined["hlabel_meta_bootstrap_below_min_positive_rate"] = True
                logger.warning(
                    "Post-filter label target positive rate %.1f%% is not "
                    "trainable; using H* horizon labels for bootstrap "
                    "(positive rate %.1f%%).",
                    current_pos_rate * 100.0,
                    hstar_pos_rate * 100.0,
                )

        logger.info(
            "Unified training table: %d rows",
            len(combined),
        )

        return combined

    def build_meta_labeler_pool(
        self,
        *,
        include_live_outcomes: bool = False,
        live_expired_bots_path: str = "data/new_expired_bots.xlsx",
        live_linkage_dir: str = "data/linkage",
        live_scanner_results_dir: str = "results",
        max_rows_per_symbol: int = META_POOL_MAX_ROWS_PER_SYMBOL,
    ) -> pd.DataFrame:
        """Build the authoritative fast-winner meta-labeler pool.

        FASTWIN-01 sourcing path. ``build()`` inner-joins decision-time snapshots
        to backtest outcomes on ``candidate_id`` and keeps only candidates that
        have BOTH a logged snapshot AND a backtest outcome (~60 rows). For the
        fast-winner meta-labeler we instead source rows directly from the backtest
        training CSVs: ``_load_backtest_rows`` reads ``training_data_*.csv`` with a
        plain ``pd.read_csv`` (no column filtering) and applies the ingestion +
        authority gate, so it retains the EX-ANTE feature columns written from the
        scan-time scanner row in ``candidate_pipeline.convert_to_training_row``
        (features come from the scan row; outcomes are appended post-hoc -> no
        look-ahead). Admission still applies the complete current authority
        contract, including realism-profile isolation; a source pool containing
        only shadow realism rows therefore returns empty rather than relaxing a
        gate.

        The authority filter is the SAME one ``build()`` Step 5 applies. We
        therefore replicate ``build()``'s exact exclusions:
        ``source_class == "reconstruction"`` out, ``version_gated == True`` out
        (this is what enforces ``mode == "geometric"`` per the Grid Mode Authority
        invariant and excludes shadow realism profiles),
        ``is_authoritative == False`` out. Filtering on ``is_authoritative``
        alone is insufficient because other contract failures are represented by
        ``version_gated``.

        Leakage safety is preserved independently: the meta-labeler consumes ONLY
        its explicit feature profile (none are outcome columns) and
        ``MetaLabeler.train`` strips ``_KNOWN_LABEL_COLUMNS`` before fitting. This
        does NOT modify ``build()`` or the snapshot-only provenance path other
        training consumers rely on; it is an opt-in pool for the meta retrain.
        """
        backtest = self._load_backtest_rows()
        if backtest.empty:
            logger.warning("Meta pool: no backtest rows available")
            return backtest

        out = backtest.copy()
        before = len(out)

        if "source_class" in out.columns:
            recon_mask = cast(pd.Series, out["source_class"]).eq("reconstruction")
            out = cast(pd.DataFrame, out.loc[~recon_mask].reset_index(drop=True))

        if "version_gated" in out.columns:
            vg = cast(pd.Series, out["version_gated"])
            vg_mask = cast(pd.Series, vg.where(vg.notna(), False)).astype(bool)
            out = cast(pd.DataFrame, out.loc[~vg_mask].reset_index(drop=True))
        else:
            logger.warning("Meta pool: version_gated column absent; fail-closed empty")
            return pd.DataFrame()

        if "is_authoritative" in out.columns:
            non_auth_mask = cast(pd.Series, out["is_authoritative"]).eq(False)
            out = cast(pd.DataFrame, out.loc[~non_auth_mask].reset_index(drop=True))
        else:
            logger.warning("Meta pool: is_authoritative column absent; fail-closed empty")
            return pd.DataFrame()

        logger.info(
            "Meta pool authoritative contract filter: %d -> %d rows (dropped %d)",
            before, len(out), before - len(out),
        )
        if out.empty:
            return out

        # PIPELINE_FIX C.1: hybrid pool — opt-in union of REAL bot outcomes. Scoped
        # to THIS pool path ONLY; build() and every geometric-only consumer are
        # untouched. Live rows are authoritative-by-provenance (real outcomes) and
        # bypass the geometric authority gate above by construction (they are added
        # here, after it). When include_live_outcomes is False this is a no-op.
        if include_live_outcomes:
            live = self._load_live_outcome_rows(
                expired_bots_path=live_expired_bots_path,
                linkage_dir=live_linkage_dir,
                scanner_results_dir=live_scanner_results_dir,
            )
            if live.empty:
                raise ValueError(
                    "include_live_outcomes was requested, but no governed direct-linked "
                    "live outcomes passed the identity and feature-completeness contract"
                )
            live = live.copy()
            live["source"] = "live"
            _before_union = len(out)
            out = cast(
                pd.DataFrame,
                pd.concat([out, live], ignore_index=True, sort=False),
            )
            logger.info(
                "Hybrid meta pool: unioned %d feature-complete direct-linked live row(s) "
                "(backtest=%d -> total=%d pre-dedup)",
                len(live), _before_union, len(out),
            )

        for feature in ALL_META_FEATURES:
            if feature not in out.columns:
                out[feature] = np.nan

        if "start_time_utc" in out.columns:
            out["start_time_utc"] = _to_datetime_utc_mixed(out["start_time_utc"])
            out = cast(
                pd.DataFrame,
                out.sort_values("start_time_utc").reset_index(drop=True),
            )

        # Data-health (AFML): a candidate_id is ONE labeled event. Re-backtesting
        # the same candidate (e.g. a repeated backtest_candidates run, which
        # excludes only DEPLOYED candidates, not already-backtested ones) appends a
        # second row carrying the SAME ex-ante features but a possibly different
        # outcome/label. Those contradictory (X, y) pairs split across CV folds
        # corrupt the model (verified: injecting such duplicates dropped study OOF
        # AUC 0.70 -> 0.44). Deduplicate by candidate_id, keeping the most recent
        # backtest. Recency is keyed on backtest_timestamp (the run time) -- the
        # SAME key the write-time dedup in backtest_candidates.py and build()'s
        # snapshot dedup use. start_time_utc is the candidate's market-event time
        # (resolve_backtest_start_timestamp / scan_ts) and is IDENTICAL across
        # re-backtests, so keying recency on it would make keep="last" pick an
        # arbitrary duplicate. Rows without a real candidate_id are preserved as-is
        # (a missing id is not an identity and must not collapse distinct events).
        if "candidate_id" in out.columns:
            # Normalize the dedup KEY exactly as the presence mask does (strip
            # whitespace; an id differing only by surrounding whitespace is the
            # same event). Do NOT lowercase: candidate_ids embed the case-
            # significant symbol, so case-folding could merge distinct events.
            _cid = cast(pd.Series, out["candidate_id"]).astype("string").str.strip()
            _has_cid = _cid.notna() & (_cid != "") & (_cid.str.lower() != "nan")
            if "backtest_timestamp" in out.columns:
                _bt_ts = pd.to_datetime(
                    out["backtest_timestamp"], errors="coerce", utc=True
                )
            else:
                _bt_ts = pd.Series(
                    pd.NaT, index=out.index, dtype="datetime64[ns, UTC]"
                )
            # PIPELINE_FIX C.4: provenance precedence — a real live outcome outranks
            # a simulated backtest for the same candidate_id. Backtest-only pools
            # have a constant rank (0), so this reduces to the recency dedup below
            # (verified no-op for the backtest-only path).
            if "source" in out.columns:
                _prov_rank = (
                    cast(pd.Series, out["source"]).astype("string").str.lower().eq("live").astype(int)
                )
            else:
                _prov_rank = pd.Series(0, index=out.index, dtype=int)
            _keyed = cast(
                pd.DataFrame,
                out.loc[_has_cid].assign(
                    _cid_key=_cid.loc[_has_cid],
                    _bt_ts_key=_bt_ts.loc[_has_cid],
                    _prov_rank=_prov_rank.loc[_has_cid],
                ),
            )
            _before = len(_keyed)
            # Stable sort by [provenance, run time] (NaT first) so live > backtest,
            # then latest within a source; keep="last" retains the winner.
            _keyed = _keyed.sort_values(
                ["_prov_rank", "_bt_ts_key"], kind="stable", na_position="first"
            )
            _keyed = _keyed.drop_duplicates(subset="_cid_key", keep="last")
            _keyed = cast(
                pd.DataFrame,
                _keyed.drop(columns=["_cid_key", "_bt_ts_key", "_prov_rank"]),
            )
            _removed = _before - len(_keyed)
            if _removed > 0:
                logger.info(
                    "Meta pool candidate_id dedup: removed %d duplicate backtest row(s)",
                    _removed,
                )
            out = cast(
                pd.DataFrame,
                pd.concat([_keyed, out.loc[~_has_cid]], ignore_index=True),
            )
            if "start_time_utc" in out.columns:
                out = cast(
                    pd.DataFrame,
                    out.sort_values("start_time_utc").reset_index(drop=True),
                )

        # FASTWIN v2 (2026-06-08) per-symbol balancing: the fast-winner signal is
        # REGIME-dominated and the leakage-safe validation is a SYMBOL-grouped,
        # time-purged K-fold. Extra rows concentrated in a few high-volume symbols
        # unbalance those folds and DEGRADE both the AUC and its stability
        # (verified on the 6,726-candidate pool: uncapped -> OOF AUC 0.748 +/- 0.034,
        # floor 0.668, ECE 0.106; capped at 30/symbol -> 0.780 +/- 0.002, floor
        # 0.776, ECE 0.066, 13/13 seeds >= 0.70). Cap per symbol AFTER the
        # candidate_id dedup so each symbol contributes a balanced number of
        # independent events. Deterministic (candidate_id sort) for reproducibility.
        # max_rows_per_symbol <= 0 disables (e.g. small in-memory test fixtures are
        # unaffected because their per-symbol counts are below the cap).
        if (
            max_rows_per_symbol
            and max_rows_per_symbol > 0
            and "symbol" in out.columns
            and len(out) > 0
        ):
            _sort_key = "candidate_id" if "candidate_id" in out.columns else "symbol"
            _before_cap = len(out)
            out = cast(
                pd.DataFrame,
                out.sort_values(_sort_key, kind="stable")
                .groupby("symbol", group_keys=False)
                .head(int(max_rows_per_symbol))
                .reset_index(drop=True),
            )
            if len(out) < _before_cap:
                logger.info(
                    "Meta pool per-symbol cap (%d): %d -> %d rows (%d symbols)",
                    max_rows_per_symbol, _before_cap, len(out),
                    int(cast(pd.Series, out["symbol"]).nunique()),
                )
            if "start_time_utc" in out.columns:
                out = cast(
                    pd.DataFrame,
                    out.sort_values("start_time_utc").reset_index(drop=True),
                )

        logger.info("Meta-labeler geometric pool: %d rows", len(out))
        return out

    def _load_live_outcome_rows(
        self,
        *,
        expired_bots_path: str,
        linkage_dir: str,
        scanner_results_dir: str,
    ) -> pd.DataFrame:
        """Load REAL bot outcomes for the hybrid fast-winner pool (PIPELINE_FIX C).

        Scoped to ``build_meta_labeler_pool`` only. READ-ONLY: ingests the expired-bot
        workbook + deploy linkage via ``LiveOutcomeIngestor`` (which never mutates the
        workbook), normalizes ``scan_*`` feature aliases to canonical names using the
        SAME shared inference normalizer (no train/serve drift), and keeps ONLY rows
        with direct deployment linkage that are feature-complete on the ACTIVE
        meta-feature profile (label-only and forensic rows are excluded). Returns an
        empty frame if ingestion or direct-link provenance is unavailable.
        """
        try:
            from neutralgrid.training.live_outcome_ingestor import LiveOutcomeIngestor
            from neutralgrid.models.meta_labeler import (
                ACTIVE_SNAPSHOT_META_FEATURES,
                normalize_inference_feature_frame,
            )
        except Exception as exc:  # pragma: no cover - optional path
            logger.warning("Hybrid meta pool: live-outcome ingest unavailable: %s", exc)
            return pd.DataFrame()

        try:
            ingestor = LiveOutcomeIngestor(
                expired_bots_path=Path(expired_bots_path),
                linkage_dir=Path(linkage_dir),
                scanner_results_dir=Path(scanner_results_dir),
            )
            live = ingestor.ingest()
        except Exception as exc:
            logger.warning("Hybrid meta pool: live ingest failed: %s", exc)
            return pd.DataFrame()

        if live is None or live.empty:
            logger.info("Hybrid meta pool: no live rows ingested")
            return pd.DataFrame()

        raw_n = int(len(live))
        if "match_method" not in live.columns:
            logger.warning(
                "Hybrid meta pool: match_method provenance is absent; fail-closed empty"
            )
            return pd.DataFrame()
        match_method = (
            cast(pd.Series, live["match_method"])
            .astype("string")
            .str.strip()
            .str.lower()
        )
        direct_mask = match_method.eq("linkage").fillna(False)
        direct_n = int(direct_mask.sum())
        if direct_n < raw_n:
            logger.warning(
                "Hybrid meta pool: excluded %d/%d live row(s) without governed "
                "direct deployment linkage",
                raw_n - direct_n,
                raw_n,
            )
        live = cast(pd.DataFrame, live.loc[direct_mask].copy())
        if live.empty:
            logger.warning("Hybrid meta pool: no governed direct-linked live rows")
            return live

        if "candidate_id" in live.columns:
            _cid = cast(pd.Series, live["candidate_id"]).astype("string").str.strip()
            matched_n = int((_cid.notna() & (_cid != "") & (_cid.str.lower() != "nan")).sum())
        else:
            matched_n = 0

        # Normalize scan_* -> canonical BEFORE completeness (shared with inference).
        live = normalize_inference_feature_frame(live)

        # The realized live pnl_pct IS the net outcome; expose it under the column
        # the fast-target label uses (net_pnl_pct) so live rows are labelable
        # alongside backtest. Both are label columns (stripped from features), not
        # predictors. Without this, live rows carry pnl only under pnl_pct and the
        # net_pnl_pct-keyed label would treat every live row as missing-pnl.
        if "pnl_pct" in live.columns:
            _live_pnl = cast(pd.Series, pd.to_numeric(live["pnl_pct"], errors="coerce"))
            if "net_pnl_pct" in live.columns:
                live["net_pnl_pct"] = cast(
                    pd.Series, pd.to_numeric(live["net_pnl_pct"], errors="coerce")
                ).fillna(_live_pnl)
            else:
                live["net_pnl_pct"] = _live_pnl

        active = list(ACTIVE_SNAPSHOT_META_FEATURES)
        present_active = [f for f in active if f in live.columns]
        absent_active = [f for f in active if f not in live.columns]
        if absent_active:
            logger.warning(
                "Hybrid meta pool: %d active feature(s) absent from live frame after "
                "normalization (%s); affected rows cannot be feature-complete",
                len(absent_active), absent_active,
            )
        if len(present_active) < len(active) or not present_active:
            complete_mask = pd.Series(False, index=live.index)
        else:
            complete_mask = cast(pd.Series, live[present_active].notna().all(axis=1))
        complete = cast(pd.DataFrame, live.loc[complete_mask].copy())

        logger.info(
            "Hybrid meta pool live ingest: raw=%d, direct_linked=%d, "
            "matched_candidate_id=%d, feature_complete=%d "
            "(active_features=%d, absent=%d)",
            raw_n, direct_n, matched_n, len(complete), len(active), len(absent_active),
        )
        return complete

    # ── Snapshot loading ──────────────────────────────────────────────────

    def _load_snapshots(self) -> pd.DataFrame:
        """Load all snapshot parquet files as the authoritative feature source."""
        if not self._snapshot_dir.exists():
            logger.warning("Snapshot directory does not exist: %s", self._snapshot_dir)
            return pd.DataFrame()

        parquet_files = sorted(self._snapshot_dir.glob("snapshots_*.parquet"))
        if not parquet_files:
            logger.warning("No snapshot parquet files in %s", self._snapshot_dir)
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        for path in parquet_files:
            try:
                frames.append(pd.read_parquet(path))
            except Exception as exc:
                logger.warning("Could not read snapshot %s: %s", path.name, exc)

        if not frames:
            return pd.DataFrame()

        snapshot_df = cast(pd.DataFrame, pd.concat(frames, ignore_index=True))

        # Normalize candidate_id
        if "candidate_id" in snapshot_df.columns:
            snapshot_df["candidate_id"] = (
                cast(pd.Series, snapshot_df["candidate_id"])
                .fillna("")
                .astype(str)
                .str.strip()
            )
        else:
            logger.warning("Snapshots missing candidate_id column")
            return pd.DataFrame()

        # Deduplicate: keep the latest snapshot per candidate_id
        if "start_time_utc" in snapshot_df.columns:
            snapshot_df["start_time_utc"] = _to_datetime_utc_mixed(snapshot_df["start_time_utc"])
            snapshot_df = cast(
                pd.DataFrame,
                snapshot_df.sort_values("start_time_utc", ascending=False)
                .drop_duplicates(subset=["candidate_id"], keep="first")
                .reset_index(drop=True),
            )

        logger.info("Snapshots: %d unique candidates from %d files", len(snapshot_df), len(parquet_files))
        return snapshot_df

    def _join_snapshots_to_outcomes(
        self,
        snapshot_df: pd.DataFrame,
        backtest_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Join snapshot features to backtest outcomes by candidate_id."""
        # Normalize candidate_ids in backtest
        if "candidate_id" not in backtest_df.columns:
            logger.warning("Backtest data missing candidate_id — cannot join")
            return pd.DataFrame()

        bt_cids = (
            cast(pd.Series, backtest_df["candidate_id"])
            .fillna("")
            .astype(str)
            .str.strip()
        )
        backtest_df = backtest_df.copy()
        backtest_df["candidate_id"] = bt_cids

        # Determine which columns are outcomes (from backtest) vs features (from snapshot)
        _snapshot_feature_cols = [
            c for c in snapshot_df.columns
            if c not in {"candidate_id"}
        ]
        _outcome_cols = [
            c for c in backtest_df.columns
            if (
                c not in _snapshot_feature_cols
                or c == "candidate_id"
                or c in _SNAPSHOT_GEOMETRY_BACKFILL_ALLOWED
            )
        ]

        # Dedup backtest rows: prefer rows produced by the current complete
        # geometric-realism contract, then the newest run timestamp.
        bt_dedup = cast(pd.DataFrame, backtest_df[_outcome_cols].copy())
        sort_cols: list[str] = []
        ascending: list[bool] = []
        for col, expected in (
            ("engine_version", ENGINE_VERSION),
            ("formula_version", FORMULA_VERSION),
            ("mode", "geometric"),
            ("fill_mode", "wick"),
        ):
            if col in bt_dedup.columns:
                flag_col = f"_{col}_current_sort"
                bt_dedup[flag_col] = (
                    cast(pd.Series, bt_dedup[col])
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .eq(str(expected).lower())
                )
                sort_cols.append(flag_col)
                ascending.append(False)
        if "label_contract_version" in bt_dedup.columns:
            bt_dedup["_lcv_sort"] = pd.to_datetime(
                bt_dedup["label_contract_version"], errors="coerce",
            )
            sort_cols.append("_lcv_sort")
            ascending.append(False)
        if "backtest_timestamp" in bt_dedup.columns:
            bt_dedup["_bt_ts_sort"] = pd.to_datetime(
                bt_dedup["backtest_timestamp"], errors="coerce", utc=True,
            )
            sort_cols.append("_bt_ts_sort")
            ascending.append(False)
        if sort_cols:
            bt_dedup = bt_dedup.sort_values(
                by=sort_cols,
                ascending=ascending,
                na_position="last",
            )
            bt_dedup = bt_dedup.drop(columns=sort_cols)
        bt_dedup = bt_dedup.drop_duplicates(subset=["candidate_id"], keep="first")

        # Inner join: only rows with both snapshot features and backtest outcomes
        merged = pd.merge(
            snapshot_df,
            bt_dedup,
            on="candidate_id",
            how="inner",
            suffixes=("", "_bt"),
        )

        # For any overlapping columns, prefer snapshot values (features)
        for col in merged.columns:
            if col.endswith("_bt"):
                base_col = col[:-3]
                if base_col in merged.columns:
                    # Snapshot features are authoritative. Never backfill an
                    # active training feature from backtest rows, except for
                    # narrow candidate-geometry repairs that are already
                    # derivable from the scan-time candidate contract.
                    if (
                        base_col not in ALL_META_FEATURES
                        or base_col in _SNAPSHOT_GEOMETRY_BACKFILL_ALLOWED
                    ):
                        mask = cast(pd.Series, merged[base_col].isna())
                        if bool(mask.any()):
                            merged.loc[mask, base_col] = merged.loc[mask, col]
                else:
                    merged[base_col] = merged[col]
                merged = merged.drop(columns=[col])

        merged["feature_source"] = "snapshot"
        merged["source"] = "snapshot"

        logger.info(
            "Snapshot-outcome join: %d matched / %d snapshots / %d backtest rows",
            len(merged), len(snapshot_df), len(backtest_df),
        )
        return merged

    # Legacy methods (_build_live_rows, _load_legacy_feature_rows,
    # _select_legacy_live_features, _load_pnl_curve_sheet) removed in
    # snapshot-only architecture redesign (2026-04-07).

    def _derive_profit_per_grid_pct(
        self,
        grid_lower: float | None,
        grid_upper: float | None,
        num_grids: int | None,
    ) -> float | None:
        if grid_lower is None or grid_upper is None or num_grids is None:
            return None
        if grid_lower <= 0 or grid_upper <= grid_lower or num_grids <= 0:
            return None
        grid_step = (grid_upper - grid_lower) / num_grids
        return float((grid_step / grid_lower) * 100.0)

    def _derive_grid_spacing_pct(
        self,
        grid_lower: float | None,
        grid_upper: float | None,
        num_grids: int | None,
        mode: str | None,
    ) -> float | None:
        if grid_lower is None or grid_upper is None or num_grids is None:
            return None
        if grid_lower <= 0 or grid_upper <= grid_lower or num_grids < 2:
            return None
        if mode is None:
            return None
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in _VALID_GRID_MODES:
            return None
        return float(grid_spacing_pct(grid_lower, grid_upper, num_grids, normalized_mode))

    def _derive_range_size_pct(
        self,
        grid_lower: float | None,
        grid_upper: float | None,
    ) -> float | None:
        if grid_lower is None or grid_upper is None:
            return None
        reference_price = (grid_lower + grid_upper) / 2.0
        if grid_lower <= 0 or grid_upper <= grid_lower or reference_price <= 0:
            return None
        return float(((grid_upper - grid_lower) / reference_price) * 100.0)

    def _derive_utility_score(self, row: Dict[str, Any]) -> float | None:
        range_prob = _as_float(row.get("range_prob"))
        trend_prob = _as_float(row.get("trend_prob"))
        if None in (range_prob, trend_prob):
            return None
        try:
            utility = compute_governed_provisional_utility(
                range_prob=cast(float, range_prob),
                trend_prob=cast(float, trend_prob),
            )
        except UtilityCalibratorUnavailable as exc:
            # Offline training: missing calibrator yields utility_score=None
            # so the meta-labeler sees a transparent feature-missing signal
            # rather than a silent v0 substitution. Emit one warning per
            # training run to avoid log flooding when iterating per-row.
            global _utility_unavailable_warning_emitted
            if not _utility_unavailable_warning_emitted:
                logger.warning(
                    "utility calibrator unavailable; training utility_score=None: %s",
                    exc,
                )
                _utility_unavailable_warning_emitted = True
            else:
                logger.debug("utility calibrator unavailable: %s", exc)
            return None
        return float(utility.utility_score)

    def _derive_ev_contract(
        self, row: Dict[str, Any]
    ) -> Dict[str, float | str | None]:
        profit_per_grid_pct = _as_float(row.get("profit_per_grid_pct"))
        num_grids = _as_int(row.get("num_grids"))
        survival_prob = _as_float(row.get("survival_prob"))
        trend_prob = _as_float(row.get("trend_prob"))
        range_size_pct = _as_float(row.get("range_size_pct"))
        if None in (
            profit_per_grid_pct,
            num_grids,
            survival_prob,
            trend_prob,
            range_size_pct,
        ):
            return {
                "ev_score": None,
                "ev_24h": None,
                "ev_raw": None,
                "ev_aligned": None,
                "ev_contract_fingerprint": None,
            }
        funding_rate = _as_float(row.get("funding_rate"))
        symbol = _non_empty_str(row.get("symbol")) or None
        leverage = _as_int(row.get("leverage"))
        if leverage is None:
            leverage = 10
        result = self._ranker.compute_score(
            profit_per_grid_pct=cast(float, profit_per_grid_pct),
            num_grids=cast(int, num_grids),
            survival_prob=cast(float, survival_prob),
            trend_prob=cast(float, trend_prob),
            funding_rate=funding_rate,
            range_size_pct=cast(float, range_size_pct),
            symbol=symbol,
            leverage=cast(int, leverage),
        )
        return {
            "ev_score": float(result.rank_score),
            "ev_24h": float(result.ev_24h),
            "ev_raw": float(result.ev_raw),
            "ev_aligned": float(result.ev_aligned),
            "ev_contract_fingerprint": getattr(
                self._ranker, "ev_contract_fingerprint", None
            ),
        }

    def _derive_ev_score(self, row: Dict[str, Any]) -> float | None:
        return cast(float | None, self._derive_ev_contract(row)["ev_score"])

    def _derive_ev_24h(self, row: Dict[str, Any]) -> float | None:
        return cast(float | None, self._derive_ev_contract(row)["ev_24h"])

    def _derive_regime_conf(self, row: Dict[str, Any]) -> float | None:
        range_prob = _as_float(row.get("range_prob"))
        trend_prob = _as_float(row.get("trend_prob"))
        if range_prob is None or trend_prob is None:
            return None
        dominance = range_prob / max(range_prob + trend_prob, 1e-10)
        cfg = get_config()
        soft_lower = float(cfg.hmm.soft_gate_lower)
        soft_upper = float(cfg.hmm.soft_gate_upper)
        if soft_upper <= soft_lower:
            return float(1.0 if dominance >= soft_upper else 0.0)
        return float(max(0.0, min(1.0, (dominance - soft_lower) / (soft_upper - soft_lower))))

    # _set_numeric_if_missing, _map_live_outcome_to_training removed in
    # snapshot-only architecture redesign (2026-04-07).

    def _normalize_backtest_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "net_pnl_pct" in out.columns and "pnl_pct" not in out.columns:
            out["pnl_pct"] = pd.to_numeric(out["net_pnl_pct"], errors="coerce")
        elif "pnl_pct" in out.columns:
            out["pnl_pct"] = pd.to_numeric(out["pnl_pct"], errors="coerce")

        if "net_pnl_pct" in out.columns:
            out["net_pnl_pct"] = pd.to_numeric(out["net_pnl_pct"], errors="coerce")
        elif "pnl_pct" in out.columns:
            out["net_pnl_pct"] = pd.to_numeric(out["pnl_pct"], errors="coerce")

        # Step 1 (Plan v6.0): Target precedence fix — hierarchical y is
        # authoritative when present.  Previous order ("y_horizon", ..., "y")
        # silently overwrote strict hierarchical labels with lenient horizon
        # labels, corrupting the training target.
        assigned_target = False
        hlabel_y = None
        if "hlabel_meta" in out.columns and cast(pd.Series, out["hlabel_meta"]).notna().any():
            hlabel_y = cast(pd.Series, cast(pd.Series, out["hlabel_meta"]).map(_label_from_any))
            if "y" in out.columns:
                existing_y = cast(pd.Series, cast(pd.Series, out["y"]).map(_label_from_any))
                conflict_mask = hlabel_y.notna() & existing_y.notna() & (hlabel_y != existing_y)
                if bool(conflict_mask.any()):
                    logger.warning(
                        "Label contract mismatch: %d row(s) have y != hlabel_meta; "
                        "using hlabel_meta unless bootstrap fallback applies.",
                        int(conflict_mask.sum()),
                    )

            hlabel_non_null = cast(pd.Series, hlabel_y.dropna())
            hlabel_pos_rate = _series_mean_float(hlabel_non_null)
            if hlabel_non_null.nunique() <= 1 or hlabel_pos_rate < MIN_POSITIVE_RATE:
                for fallback_col in ("y_horizon", "label_positive_by_horizon"):
                    if fallback_col in out.columns and cast(pd.Series, out[fallback_col]).notna().any():
                        fallback_y = cast(pd.Series, cast(pd.Series, out[fallback_col]).map(_label_from_any))
                        fallback_non_null = cast(pd.Series, fallback_y.dropna())
                        fallback_pos_rate = _series_mean_float(fallback_non_null)
                        if fallback_non_null.nunique() > 1 and fallback_pos_rate >= MIN_POSITIVE_RATE:
                            out["y"] = fallback_y
                            out["label_source"] = f"{fallback_col}_hstar_bootstrap"
                            out["hlabel_meta_bootstrap_below_min_positive_rate"] = hlabel_pos_rate < MIN_POSITIVE_RATE
                            out["hlabel_meta_degenerate"] = hlabel_non_null.nunique() <= 1
                            assigned_target = True
                            logger.warning(
                                "Hierarchical label target positive rate %.1f%% is not "
                                "trainable; using %s for bootstrap H* training labels "
                                "(fallback positive rate %.1f%%).",
                                hlabel_pos_rate * 100.0,
                                fallback_col,
                                fallback_pos_rate * 100.0,
                            )
                            break

            if not assigned_target:
                out["y"] = hlabel_y
                out["label_source"] = "hlabel_meta"
                assigned_target = True

        if not assigned_target:
            # ERR-021 fix: extend the degeneracy bypass already used by the
            # hlabel_meta path above (lines 793-815) to the secondary
            # label-precedence loop. notna().any() accepts a column of all
            # zeros (zero is not NA) as valid, which historically blocked the
            # net_pnl_pct >= meta_hurdle_pct fallback when backtest outcome
            # CSVs carry an all-zero `y`. Reuse `_series_mean_float` and the
            # 5% positive-rate threshold from meta_labeler.py:725-732.
            preferred_target = None
            preferred_y: pd.Series | None = None
            _y_precedence_skipped: list[str] = []  # ERR-036 audit trail
            for col in ("y", "y_horizon", "label_positive_by_horizon"):
                if col not in out.columns:
                    continue
                candidate_y = cast(pd.Series, cast(pd.Series, out[col]).map(_label_from_any))
                candidate_non_null = cast(pd.Series, candidate_y.dropna())
                if candidate_non_null.empty:
                    continue
                candidate_pos_rate = _series_mean_float(candidate_non_null)
                if candidate_non_null.nunique() <= 1 or candidate_pos_rate < MIN_POSITIVE_RATE:
                    _y_precedence_skipped.append(str(col))  # ERR-036 audit trail
                    logger.warning(
                        "Label-precedence candidate '%s' is degenerate "
                        "(nunique=%d, positive_rate=%.1f%%); falling through to next source.",
                        col, int(candidate_non_null.nunique()), candidate_pos_rate * 100.0,
                    )
                    continue
                preferred_target = col
                preferred_y = candidate_y
                break
            # ERR-036: mirror the hlabel_meta path's audit columns (above) so an
            # operator can reconstruct from the output dataframe which y-candidate
            # columns were rejected (degenerate / below MIN_POSITIVE_RATE) and why.
            # Additive, write-only audit evidence; never consumed as a feature.
            out["y_precedence_skipped_columns"] = ",".join(_y_precedence_skipped)
            out["y_precedence_degenerate"] = bool(_y_precedence_skipped)
            if preferred_target is not None and preferred_y is not None:
                out["y"] = preferred_y
                out["label_source"] = preferred_target
                assigned_target = True

        if not assigned_target and "net_pnl_pct" in out.columns:
            out["y"] = (
                cast(pd.Series, pd.to_numeric(out["net_pnl_pct"], errors="coerce"))
                >= float(self._label_config.meta_hurdle_pct)
            ).astype(int)
            out["label_source"] = "net_pnl_pct_hurdle"

        if "y_horizon" not in out.columns and "y" in out.columns:
            out["y_horizon"] = out["y"]

        if "dataset_schema_version" not in out.columns:
            out["dataset_schema_version"] = TRAINING_DATASET_SCHEMA_VERSION
        if "hmm_feature_semantics_version" not in out.columns:
            out["hmm_feature_semantics_version"] = np.where(
                cast(pd.Series, out.get("range_prob", pd.Series(index=out.index))).notna()
                | cast(pd.Series, out.get("trend_prob", pd.Series(index=out.index))).notna(),
                HMM_FEATURE_SEMANTICS_VERSION,
                "",
            )
        if "label_source" not in out.columns:
            out["label_source"] = ""

        if "sl_hit" not in out.columns and "net_pnl_pct" in out.columns:
            out["sl_hit"] = cast(pd.Series, pd.to_numeric(out["net_pnl_pct"], errors="coerce")) <= float(
                self._label_config.sl_pct
            )

        if "barrier_touched" not in out.columns and "net_pnl_pct" in out.columns:
            pnl = cast(pd.Series, pd.to_numeric(out["net_pnl_pct"], errors="coerce"))
            out["barrier_touched"] = np.where(
                pnl >= float(self._label_config.pt_pct),
                "pt",
                np.where(pnl <= float(self._label_config.sl_pct), "sl", "time"),
            )

        if "start_time_utc" in out.columns:
            out["start_time_utc"] = _to_datetime_utc_mixed(out["start_time_utc"])
        else:
            out["start_time_utc"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")

        missing_start = cast(pd.Series, out["start_time_utc"].isna())
        if bool(missing_start.any()) and "candidate_id" in out.columns:
            parsed_times = []
            for candidate_id in out["candidate_id"].fillna(""):
                parsed_times.append(self._parse_start_time(_non_empty_str(candidate_id), None))
            parsed_series = pd.to_datetime(pd.Series(parsed_times), utc=True, errors="coerce")
            out.loc[missing_start, "start_time_utc"] = parsed_series.loc[missing_start].to_numpy()

        if "t1" not in out.columns:
            out["t1"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
        # Gate 6 (Fix 3): Ensure t1_is_synthetic and t1_truncated columns exist
        if "t1_is_synthetic" not in out.columns:
            out["t1_is_synthetic"] = False
        if "t1_truncated" not in out.columns:
            out["t1_truncated"] = False

        missing_t1 = cast(pd.Series, pd.to_datetime(out["t1"], utc=True, errors="coerce").isna())
        valid_start = _to_datetime_utc_mixed(out["start_time_utc"])
        if bool(missing_t1.any()):
            _horizon_hours = float(self._label_config.horizon_hours)
            # Gate 6 (Fix 3): For truncated backtests, use actual duration
            # rather than the canonical horizon for t1 synthesis.
            _has_duration = "duration_hours" in out.columns
            _has_sl_hit = "sl_hit" in out.columns
            for idx in out.index[missing_t1]:
                _start = valid_start.loc[idx]
                if pd.isna(_start):
                    continue
                _dur = _as_float(out.at[idx, "duration_hours"]) if _has_duration else None
                _sl = bool(out.at[idx, "sl_hit"]) if _has_sl_hit else False
                _is_truncated = (
                    _dur is not None
                    and _dur > 0
                    and (_sl or _dur < _horizon_hours * 0.95)
                )
                if _is_truncated:
                    out.at[idx, "t1"] = _start + pd.Timedelta(hours=cast(float, _dur))
                    out.at[idx, "t1_truncated"] = True
                else:
                    out.at[idx, "t1"] = _start + pd.Timedelta(hours=_horizon_hours)
                out.at[idx, "t1_is_synthetic"] = True

        return out

    def _parse_start_time(self, candidate_id: str, fallback: Any) -> Optional[datetime]:
        match = re.search(r"_(\d{8}_\d{6})", candidate_id)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                pass
        parsed = _to_datetime_utc_mixed(pd.Series([fallback])).iloc[0]
        if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed):
            return parsed.to_pydatetime()
        return None

    # _backfill_from_scanner_csvs removed in snapshot-only architecture
    # redesign (2026-04-07).

    def _enrich_backtest_feature_coverage(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        out = self._normalize_backtest_targets(df)
        for feature in ALL_META_FEATURES:
            if feature not in out.columns:
                out[feature] = np.nan

        for col in ALL_META_FEATURES + ["score", "grid_lower", "grid_upper", "price_start", "price_end"]:
            if col in out.columns and col != "candidate_id":
                if col == "mode":
                    continue
                out[col] = pd.to_numeric(out[col], errors="coerce")

        if "primary_pipeline_score" in out.columns and "score" in out.columns:
            missing = cast(pd.Series, out["primary_pipeline_score"].isna())
            out.loc[missing, "primary_pipeline_score"] = out.loc[missing, "score"]

        if "num_grids" in out.columns:
            num_grids = pd.to_numeric(out["num_grids"], errors="coerce")
        elif "grids_count" in out.columns:
            num_grids = pd.to_numeric(out["grids_count"], errors="coerce")
        else:
            num_grids = pd.Series(np.nan, index=out.index, dtype=float)
        out["num_grids"] = num_grids

        if {"grid_lower", "grid_upper", "num_grids"}.issubset(set(out.columns)):
            missing_profit = cast(pd.Series, out["profit_per_grid_pct"].isna())
            for idx in out.index[missing_profit]:
                profit = self._derive_profit_per_grid_pct(
                    _as_float(out.at[idx, "grid_lower"]),
                    _as_float(out.at[idx, "grid_upper"]),
                    _as_int(out.at[idx, "num_grids"]),
                )
                if profit is not None:
                    out.at[idx, "profit_per_grid_pct"] = profit

            missing_spacing = cast(pd.Series, out["grid_spacing_pct"].isna())
            derived_spacing_count = 0
            for idx in out.index[missing_spacing]:
                spacing = self._derive_grid_spacing_pct(
                    _as_float(out.at[idx, "grid_lower"]),
                    _as_float(out.at[idx, "grid_upper"]),
                    _as_int(out.at[idx, "num_grids"]),
                    None if "mode" not in out.columns else str(out.at[idx, "mode"]),
                )
                if spacing is not None:
                    out.at[idx, "grid_spacing_pct"] = spacing
                    derived_spacing_count += 1
            if derived_spacing_count:
                out["grid_spacing_source"] = "derived_from_candidate_geometry"
                logger.info(
                    "Derived grid_spacing_pct for %d rows from candidate geometry",
                    derived_spacing_count,
                )

        if {"grid_lower", "grid_upper"}.issubset(set(out.columns)):
            missing_range = cast(pd.Series, out["range_size_pct"].isna())
            for idx in out.index[missing_range]:
                derived = self._derive_range_size_pct(
                    _as_float(out.at[idx, "grid_lower"]),
                    _as_float(out.at[idx, "grid_upper"]),
                )
                if derived is not None:
                    out.at[idx, "range_size_pct"] = derived

        if "utility_score" in out.columns:
            missing_utility = cast(pd.Series, out["utility_score"].isna())
            for idx in out.index[missing_utility]:
                utility = self._derive_utility_score(cast(dict[str, Any], out.loc[idx].to_dict()))
                if utility is not None:
                    out.at[idx, "utility_score"] = utility

        if "ev_score" in out.columns:
            missing_ev = cast(pd.Series, out["ev_score"].isna())
            for idx in out.index[missing_ev]:
                ev_contract = self._derive_ev_contract(cast(dict[str, Any], out.loc[idx].to_dict()))
                if ev_contract["ev_score"] is not None:
                    out.at[idx, "ev_score"] = ev_contract["ev_score"]
                if "ev_24h" in out.columns and pd.isna(out.at[idx, "ev_24h"]) and ev_contract["ev_24h"] is not None:
                    out.at[idx, "ev_24h"] = ev_contract["ev_24h"]
                if ev_contract["ev_score"] is not None:
                    if "ev_contract_fingerprint" not in out.columns:
                        out["ev_contract_fingerprint"] = pd.Series(
                            pd.NA, index=out.index, dtype="object"
                        )
                    out.at[idx, "ev_contract_fingerprint"] = ev_contract[
                        "ev_contract_fingerprint"
                    ]

        if "ev_24h" in out.columns:
            missing_ev_24h = cast(pd.Series, out["ev_24h"].isna())
            for idx in out.index[missing_ev_24h]:
                ev_24h = self._derive_ev_24h(cast(dict[str, Any], out.loc[idx].to_dict()))
                if ev_24h is not None:
                    out.at[idx, "ev_24h"] = ev_24h

        if "regime_conf" not in out.columns:
            out["regime_conf"] = np.nan
        missing_regime_conf = cast(pd.Series, out["regime_conf"].isna())
        for idx in out.index[missing_regime_conf]:
            regime_conf = self._derive_regime_conf(cast(dict[str, Any], out.loc[idx].to_dict()))
            if regime_conf is not None:
                out.at[idx, "regime_conf"] = regime_conf

        # Derive funding_rate_zscore: cross-sectional z-score of funding_rate
        if "funding_rate_zscore" in out.columns and "funding_rate" in out.columns:
            missing_frz = cast(pd.Series, out["funding_rate_zscore"].isna())
            fr_series = cast(pd.Series, pd.to_numeric(out["funding_rate"], errors="coerce"))
            fr_valid = fr_series.dropna()
            if missing_frz.any() and len(fr_valid) >= 10:
                _arr = np.asarray(fr_valid, dtype=np.float64)
                mu = float(np.mean(_arr))
                sigma = float(np.std(_arr))
                if sigma > 1e-10:
                    derived_count = 0
                    for idx in out.index[missing_frz]:
                        fr_val = _as_float(out.at[idx, "funding_rate"])
                        if fr_val is not None:
                            out.at[idx, "funding_rate_zscore"] = (fr_val - mu) / sigma
                            derived_count += 1
                    if derived_count:
                        logger.info(
                            "Derived funding_rate_zscore for %d rows (cross-sectional z-score)",
                            derived_count,
                        )

        # Derive bb_width_ratio_1h_15m: sqrt(4) ≈ 2.0 scaling approximation
        # BB width scales with sqrt(period_ratio) for stationary processes.
        # 1h = 4 × 15m bars → sqrt(4) = 2.0 is the neutral ratio.
        # Mean-reverting regimes compress this (<2), trending expand it (>2).
        if "bb_width_ratio_1h_15m" in out.columns:
            missing_bwr = cast(pd.Series, out["bb_width_ratio_1h_15m"].isna())
            if missing_bwr.any():
                derived_count = 0
                for idx in out.index[missing_bwr]:
                    rp = _as_float(out.at[idx, "range_prob"]) if "range_prob" in out.columns else None
                    # Base ratio = sqrt(4) = 2.0 for random walk
                    base_ratio = 2.0
                    if rp is not None:
                        # Regime adjustment: range-bound compresses (<2), trending expands (>2)
                        regime_adj = 1.0 - 0.3 * (rp - 0.5)
                        base_ratio *= regime_adj
                    out.at[idx, "bb_width_ratio_1h_15m"] = round(base_ratio, 4)
                    derived_count += 1
                if derived_count:
                    logger.info(
                        "Derived bb_width_ratio_1h_15m for %d rows (sqrt(4) scaling with regime adjustment)",
                        derived_count,
                    )

        # Derive long_short_ratio: default to neutral (1.0) when API data unavailable
        if "long_short_ratio" in out.columns:
            missing_lsr = cast(pd.Series, out["long_short_ratio"].isna())
            if missing_lsr.any():
                out.loc[missing_lsr, "long_short_ratio"] = 1.0
                logger.info(
                    "Set long_short_ratio to neutral (1.0) for %d rows (API data unavailable)",
                    int(missing_lsr.sum()),
                )

        # Derive open_interest_change_pct: default to 0.0 when OI history unavailable
        if "open_interest_change_pct" in out.columns:
            missing_oic = cast(pd.Series, out["open_interest_change_pct"].isna())
            if missing_oic.any():
                out.loc[missing_oic, "open_interest_change_pct"] = 0.0
                logger.info(
                    "Set open_interest_change_pct to neutral (0.0) for %d rows (OI history unavailable)",
                    int(missing_oic.sum()),
                )

        if "source" not in out.columns:
            out["source"] = "backtest"
        else:
            out["source"] = cast(pd.Series, out["source"]).fillna("backtest")
        if "sample_weight_override" not in out.columns:
            out["sample_weight_override"] = 0.5
        else:
            out["sample_weight_override"] = cast(
                pd.Series,
                pd.to_numeric(out["sample_weight_override"], errors="coerce"),
            ).fillna(0.5)
        out["feature_count"] = cast(pd.DataFrame, out[ALL_META_FEATURES]).notna().sum(axis=1)
        return out

    def _apply_ingestion_gate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 6 (Plan v6.0): Ingestion gate — validate label_contract_version.

        label_contract_version uses a date-based format (YYYY-MM-DD).
        Rows whose version is older than LABEL_CONTRACT_VERSION are marked
        ``version_gated=True``.  Rows with the current version or newer pass.

        Rows without a label_contract_version (pre-20260226 legacy) are
        marked as ``version_gated=True`` and ``source_class='legacy'``.
        """
        if df.empty:
            return df

        out = df.copy()
        if "version_gated" not in out.columns:
            out["version_gated"] = False
        if "version_gate_reason" not in out.columns:
            out["version_gate_reason"] = ""

        def _mark_gated(mask: pd.Series, reason: str) -> None:
            n_bad = int(mask.sum())
            if n_bad <= 0:
                return
            out.loc[mask, "version_gated"] = True
            # E7 (2026-05-22): do not downgrade a more-specific "legacy"
            # classification (missing label_contract_version, assigned below)
            # to the generic "non_authoritative". Legacy is the row's origin
            # class and must survive subsequent gate failures.
            if "source_class" in out.columns:
                downgrade_mask = mask & cast(pd.Series, out["source_class"]).ne("legacy")
            else:
                downgrade_mask = mask
            out.loc[downgrade_mask, "source_class"] = "non_authoritative"
            current_reason = (
                cast(pd.Series, out.loc[mask, "version_gate_reason"])
                .fillna("")
                .astype(str)
            )
            separator = current_reason.ne("")
            out.loc[mask, "version_gate_reason"] = (
                current_reason
                + separator.map({True: ";", False: ""})
                + reason
            )
            logger.warning("Ingestion gate: %d rows failed %s", n_bad, reason)

        current_dt = pd.to_datetime(LABEL_CONTRACT_VERSION, errors="coerce")

        if "label_contract_version" in out.columns:
            has_version = cast(pd.Series, out["label_contract_version"]).notna()

            # Parse row versions as dates; unparseable → NaT → treated as legacy
            row_versions = pd.to_datetime(
                out["label_contract_version"], errors="coerce",
            )
            # Gate rows whose date-based version is older than the current one
            mismatched = has_version & (
                row_versions.isna() | (row_versions < current_dt)
            )
            out.loc[mismatched, "version_gated"] = True

            # Step 7: Legacy rows — no version at all
            no_version = ~has_version
            out.loc[no_version, "version_gated"] = True
            out.loc[no_version, "source_class"] = "legacy"

            n_gated = int(mismatched.sum())
            n_legacy = int(no_version.sum())
            if n_gated > 0:
                logger.warning(
                    "Ingestion gate: %d rows have label_contract_version older "
                    "than %s (marked version_gated=True)",
                    n_gated, LABEL_CONTRACT_VERSION,
                )
                _mark_gated(mismatched, "label_contract_version")
            if n_legacy > 0:
                logger.info(
                    "Ingestion gate: %d rows have no label_contract_version "
                    "(marked source_class=legacy, version_gated=True)",
                    n_legacy,
                )
                _mark_gated(no_version, "missing_label_contract_version")
        else:
            # No version column at all — entire batch is legacy
            out["version_gated"] = True
            _mark_gated(pd.Series(True, index=out.index), "missing_label_contract_version")
            if "source_class" not in out.columns:
                out["source_class"] = "legacy"
            logger.info(
                "Ingestion gate: no label_contract_version column — "
                "all %d rows marked legacy", len(out),
            )

        # Engine-settings authority gate. NOTE (Grid Mode Authority,
        # safety-invariants.md): the backtest engine validates BOTH arithmetic
        # and geometric modes (btk_label_contract.validate_engine_result), but
        # the AUTHORITATIVE TRAINING pool is geometric-only
        # (FORMULA_VERSION alignment-v2-geometric-realism). Arithmetic-mode rows
        # are intentionally gated non_authoritative here: valid for backtest,
        # not authoritative for training. (Confirmed by operator 2026-05-22.)
        for col, expected in (
            ("engine_version", ENGINE_VERSION),
            ("formula_version", FORMULA_VERSION),
            ("mode", "geometric"),
            ("fill_mode", "wick"),
        ):
            if col not in out.columns:
                _mark_gated(pd.Series(True, index=out.index), f"missing_{col}")
                continue
            observed = (
                cast(pd.Series, out[col])
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )
            _mark_gated(~observed.eq(str(expected).lower()), col)

        if "global_cooldown_bars" not in out.columns:
            _mark_gated(pd.Series(True, index=out.index), "missing_global_cooldown_bars")
        else:
            cooldown = cast(
                pd.Series,
                pd.to_numeric(
                    cast(pd.Series, out["global_cooldown_bars"]),
                    errors="coerce",
                ),
            )
            _mark_gated(cooldown.ne(0.0) | cooldown.isna(), "global_cooldown_bars")

        if "cb_enabled" not in out.columns:
            _mark_gated(pd.Series(True, index=out.index), "missing_cb_enabled")
        else:
            cb_ok: list[bool] = []
            for value in cast(pd.Series, out["cb_enabled"]).tolist():
                try:
                    cb_ok.append(_as_bool_strict(value) is False)
                except ValueError:
                    cb_ok.append(False)
            _mark_gated(~pd.Series(cb_ok, index=out.index), "cb_enabled")

        if "is_authoritative" not in out.columns:
            _mark_gated(pd.Series(True, index=out.index), "missing_is_authoritative")
        else:
            auth_ok: list[bool] = []
            for value in cast(pd.Series, out["is_authoritative"]).tolist():
                try:
                    auth_ok.append(_as_bool_strict(value) is True)
                except ValueError:
                    auth_ok.append(False)
            _mark_gated(~pd.Series(auth_ok, index=out.index), "is_authoritative")

        if "realism_profile" in out.columns:
            profile = (
                cast(pd.Series, out["realism_profile"])
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )
            blank_mask = profile.eq("")
            shadow_mask = profile.isin(SHADOW_REALISM_PROFILES)
            unknown_mask = profile.ne("") & ~profile.isin(REALISM_PROFILES)
            _mark_gated(blank_mask, "missing_realism_profile")
            _mark_gated(
                shadow_mask,
                "shadow_realism_profile_not_authoritative",
            )
            _mark_gated(unknown_mask, "unsupported_realism_profile")

            public_mask = profile.eq(CANDIDATE_TIME_PUBLIC_MARKET_PROFILE)
            if bool(public_mask.any()):
                for col in _PUBLIC_MARKET_PROVENANCE_COLUMNS:
                    if col not in out.columns:
                        _mark_gated(public_mask, f"missing_{col}")
                        continue
                    values = (
                        cast(pd.Series, out[col])
                        .fillna("")
                        .astype(str)
                        .str.strip()
                    )
                    _mark_gated(public_mask & values.eq(""), f"missing_{col}")

                if "exchange_filter_validation_status" in out.columns:
                    status = (
                        cast(pd.Series, out["exchange_filter_validation_status"])
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .str.lower()
                    )
                    _mark_gated(public_mask & status.ne("valid"), "exchange_filter_validation_status")

        return out

    def _load_backtest_rows(self) -> pd.DataFrame:
        if not self._bt_dir.exists():
            return pd.DataFrame()

        # Step 2 (Plan v6.0): Add training_data_*.csv glob — this is what
        # backtest_candidates.py actually writes (line 349).  The previous
        # globs only matched legacy filenames, causing authoritative training
        # CSVs with hierarchical labels to be silently skipped.
        training_csvs = sorted(self._bt_dir.glob("training_data_*.csv"))
        training_csvs.extend(sorted(self._bt_dir.glob("training_rows_*.csv")))
        training_csvs.extend(sorted(self._bt_dir.glob("backtest_training_*.csv")))
        frames: list[pd.DataFrame] = []
        for path in training_csvs:
            try:
                frames.append(pd.read_csv(path))
            except Exception as exc:
                logger.warning("Could not read %s: %s", path.name, exc)

        if frames:
            combined = cast(pd.DataFrame, pd.concat(frames, ignore_index=True))
            geometry_frames: list[pd.DataFrame] = []
            for result_path in sorted(self._bt_dir.glob("backtest_results_*.csv")):
                try:
                    result_df = pd.read_csv(
                        result_path,
                        usecols=lambda col: col in {
                            "candidate_id",
                            "grid_spacing_pct",
                            "grid_lower",
                            "grid_upper",
                        },
                    )
                    geometry_frames.append(result_df)
                except Exception as exc:
                    logger.debug(
                        "Could not read geometry columns from %s: %s",
                        result_path.name,
                        exc,
                    )
            if geometry_frames and "candidate_id" in combined.columns:
                geometry_df = cast(
                    pd.DataFrame,
                    pd.concat(geometry_frames, ignore_index=True),
                )
                if "candidate_id" in geometry_df.columns:
                    geometry_df["candidate_id"] = (
                        cast(pd.Series, geometry_df["candidate_id"])
                        .fillna("")
                        .astype(str)
                        .str.strip()
                    )
                    geometry_df = geometry_df.drop_duplicates(
                        subset=["candidate_id"],
                        keep="last",
                    )
                    combined["candidate_id"] = (
                        cast(pd.Series, combined["candidate_id"])
                        .fillna("")
                        .astype(str)
                        .str.strip()
                    )
                    combined = pd.merge(
                        combined,
                        geometry_df,
                        on="candidate_id",
                        how="left",
                        suffixes=("", "_bt_geometry"),
                    )
                    for col in ("grid_spacing_pct", "grid_lower", "grid_upper"):
                        repair_col = f"{col}_bt_geometry"
                        if repair_col in combined.columns:
                            if col not in combined.columns:
                                combined[col] = combined[repair_col]
                            else:
                                missing = cast(pd.Series, combined[col].isna())
                                combined.loc[missing, col] = combined.loc[missing, repair_col]
                            combined = combined.drop(columns=[repair_col])
                    combined["geometry_feature_source"] = "backtest_candidate_config"
            for col in _EXCURSION_OUTCOME_FIELDS + _PNL_CURVE_COLUMNS:
                if col not in combined.columns:
                    combined[col] = np.nan

            # Step 5 (Plan v6.0): Source-class split — training_data CSVs are
            # authoritative (produced by convert_to_training_row with
            # hierarchical labels, provenance columns, rotation metrics).
            # Mark them as source_class="authoritative" for downstream gating.
            if "source_class" not in combined.columns:
                combined["source_class"] = "authoritative"

            # Step 6 (Plan v6.0): Ingestion gate — check label_contract_version
            # and mark rows with mismatched or missing versions.
            combined = self._apply_ingestion_gate(combined)

            return self._enrich_backtest_feature_coverage(combined)

        raw_result_csvs = sorted(self._bt_dir.glob("backtest_results_*.csv"))
        if not raw_result_csvs:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for path in raw_result_csvs:
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                logger.warning("Could not read %s: %s", path.name, exc)
                continue

            for _, raw in df.iterrows():
                symbol = _non_empty_str(raw.get("symbol"))
                if not symbol:
                    continue

                candidate_id = _non_empty_str(raw.get("candidate_id"))
                if not candidate_id:
                    scan_stamp = _non_empty_str(raw.get("scan_timestamp"))
                    candidate_id = f"{symbol}_{scan_stamp}" if scan_stamp else symbol

                start_dt = self._parse_start_time(candidate_id, raw.get("backtest_timestamp"))
                if start_dt is None:
                    continue

                net_pnl_pct = _as_float(raw.get("net_pnl_pct"))
                if net_pnl_pct is None:
                    continue

                # Step 1b: Slow-path target precedence — same fix as fast path.
                # Try hierarchical y first, then y_horizon, then lenient label.
                y = _label_from_any(raw.get("y"))
                if y is None:
                    y = _label_from_any(raw.get("y_horizon"))
                if y is None:
                    y = _label_from_any(raw.get("label_positive_by_horizon"))
                if y is None:
                    y = 1 if net_pnl_pct >= float(self._label_config.meta_hurdle_pct) else 0

                if net_pnl_pct >= float(self._label_config.pt_pct):
                    barrier = "pt"
                elif net_pnl_pct <= float(self._label_config.sl_pct):
                    barrier = "sl"
                else:
                    barrier = "time"

                # Gate 6 (Fix 3): Detect truncated backtests and
                # synthesise t1 from actual end time, not canonical horizon.
                duration_hours = _as_float(raw.get("duration_hours"))
                _horizon_hours = float(self._label_config.horizon_hours)
                _sl_hit = bool(net_pnl_pct <= float(self._label_config.sl_pct))
                _t1_truncated = False
                _t1_is_synthetic = True
                if (
                    duration_hours is not None
                    and duration_hours > 0
                    and (
                        _sl_hit
                        or duration_hours < _horizon_hours * 0.95
                    )
                ):
                    # Truncated backtest — use actual duration
                    _t1 = (start_dt + timedelta(hours=duration_hours)).isoformat()
                    _t1_truncated = True
                else:
                    _t1 = (start_dt + timedelta(hours=_horizon_hours)).isoformat()

                row: dict[str, Any] = {
                    "symbol": symbol,
                    "candidate_id": candidate_id,
                    "source": "backtest",
                    "sample_weight_override": 0.5,
                    "start_time_utc": start_dt.isoformat(),
                    "backtest_timestamp": raw.get("backtest_timestamp"),
                    "t1": _t1,
                    "t1_is_synthetic": _t1_is_synthetic,
                    "t1_truncated": _t1_truncated,
                    "pnl_pct": float(net_pnl_pct),
                    "net_pnl_pct": float(net_pnl_pct),
                    "y": int(y),
                    "y_horizon": int(y),
                    "sl_hit": _sl_hit,
                    "barrier_touched": barrier,
                    "grid_lower": _as_float(raw.get("grid_lower")),
                    "grid_upper": _as_float(raw.get("grid_upper")),
                    "price_start": _as_float(raw.get("price_start")),
                    "price_end": _as_float(raw.get("price_end")),
                }

                if duration_hours is not None:
                    row["duration_hours"] = duration_hours

                for field in _EXCURSION_OUTCOME_FIELDS:
                    value = _as_float(raw.get(field))
                    if value is not None:
                        row[field] = value
                if not all(field in row for field in _EXCURSION_OUTCOME_FIELDS):
                    _derive_excursion_from_engine_fields(raw, row, symbol)

                for feature in ALL_META_FEATURES + ["primary_pipeline_score"]:
                    if feature in raw.index and bool(pd.notna(raw.get(feature))):
                        numeric = _as_float(raw.get(feature))
                        if numeric is not None:
                            row[feature] = numeric

                if "primary_pipeline_score" not in row and bool(pd.notna(raw.get("score"))):
                    score_value = _as_float(raw.get("score"))
                    if score_value is not None:
                        row["primary_pipeline_score"] = score_value

                for audit_col in (
                    "hlabel",
                    "hlabel_meta",
                    "hlabel_L1",
                    "hlabel_L2",
                    "hlabel_L3",
                    "capital_fraction",
                ):
                    if bool(pd.notna(raw.get(audit_col))):
                        row[audit_col] = raw.get(audit_col)

                # H*-censoring audit flag (DURATION_FIX §21.4 slow-path preserve).
                # Parse strictly: object-typed CSVs can carry string "False"
                # which vanilla bool() would silently coerce to True. A
                # non-missing value that cannot be resolved to a bool breaches
                # the §11 "known and current" label-semantics invariant, so the
                # row is gated out of final training rather than admitted with
                # ambiguous provenance.
                raw_censored = raw.get("horizon_censored")
                if bool(pd.notna(raw_censored)):
                    try:
                        row["horizon_censored"] = _as_bool_strict(raw_censored)
                    except ValueError:
                        logger.warning(
                            "Slow-path row for %s has unparseable "
                            "horizon_censored=%r; gating row as "
                            "non_authoritative",
                            symbol, raw_censored,
                        )
                        row["version_gated"] = True
                        row["source_class"] = "non_authoritative"

                for col in _PNL_CURVE_COLUMNS:
                    if bool(pd.notna(raw.get(col))):
                        row[col] = raw.get(col)

                # Preserve label_contract_version for ingestion gate
                if bool(pd.notna(raw.get("label_contract_version"))):
                    row["label_contract_version"] = str(raw["label_contract_version"])
                for contract_col in (
                    "engine_version",
                    "formula_version",
                    "mode",
                    "fill_mode",
                    "fill_price_source",
                    "valuation_price_source",
                    "global_cooldown_bars",
                    "cb_enabled",
                    "is_authoritative",
                    "realism_profile",
                    *_PUBLIC_MARKET_PROVENANCE_COLUMNS,
                ):
                    if bool(pd.notna(raw.get(contract_col))):
                        row[contract_col] = raw.get(contract_col)

                # Step 5 (Plan v6.0): Slow-path = reconstruction (non-authoritative).
                # Missing hierarchical labels, rotation metrics, ~15 features.
                # Do not clobber a more specific classification already set
                # above (e.g. non_authoritative gating from a failed censor
                # parse) — those rows must retain their gating tag.
                if "source_class" not in row:
                    row["source_class"] = "reconstruction"
                rows.append(row)

        if not rows:
            return pd.DataFrame()
        slow_df = pd.DataFrame(rows)
        # Step 7 (Plan v6.0): Mark legacy/reconstructed rows distinctly.
        # If authoritative data exists, log that reconstruction rows are being
        # loaded as a supplement (not as primary training data).
        logger.info(
            "Loaded %d backtest rows via slow-path reconstruction "
            "(source_class=reconstruction, no hierarchical labels)",
            len(slow_df),
        )
        # Apply ingestion gate to slow-path too
        slow_df = self._apply_ingestion_gate(slow_df)
        return self._enrich_backtest_feature_coverage(slow_df)
