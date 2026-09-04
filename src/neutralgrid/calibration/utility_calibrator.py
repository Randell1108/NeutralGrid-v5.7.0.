"""Governed recalibration for the utility score coefficients.

The calibration contract is intentionally narrow:
- outcomes come from ``General``;
- features and ``backfill_status`` come from ``Meta Features``;
- labels are binary fast-horizon ``duration_hours <= 7`` outcomes, with class 1 equal to
  ``pnl_pct > 0``;
- stored workbook ``utility_score`` values are never read.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd

from neutralgrid.models.artifacts import get_active_hmm_version
from neutralgrid.validation.utility import (
    DEFAULT_UTILITY_ARTIFACT_PATH,
    FALLBACK_HORIZON_HOURS,
    FALLBACK_KAPPA_TREND,
    FALLBACK_LAMBDA_RISK,
    FALLBACK_MIN_UTILITY,
    UtilityConfig,
)

OUTCOME_COLS = (
    "strategy_id",
    "symbol",
    "pnl_pct",
    "duration_hours",
    "start_time_utc",
)
STATUS_COLS = ("backfill_status",)
REQUIRED_FEATURE_COLS = (
    "num_grids",
    "range_prob",
    "trend_prob",
    "profit_per_grid_pct",
    "range_size_pct",
)
OPTIONAL_FEATURE_COLS = ("survival_prob", "hurst_exponent")
HMM_LINEAGE_COL = "hmm_artifact_version"
HMM_PROBABILITY_COLS = ("range_prob", "trend_prob", "persistence_prob")
FEATURE_CUTOFF_COLS = ("hmm_feature_cutoff_utc", "feature_cutoff_utc")
TWO_SHEET_INPUT_LAYOUT = "general_meta_features"
FLAT_BACKFILL_INPUT_LAYOUT = "flat_backfill_sheet"
LABEL_COL = "utility_outcome_7h"
MAX_DURATION_HOURS = 7.0
ELIGIBLE_UNIVERSE_RULE = (
    "duration_hours <= 7.0 and backfill_status != 'hmm_failed' "
    "and pnl_pct is observed"
)

LAMBDA_GRID = tuple(np.linspace(0.05, 4.0, 24))
KAPPA_GRID = tuple(np.linspace(0.05, 4.0, 24))
HORIZON_GRID = tuple(np.linspace(3.25, 10.0, 28))


@dataclass(frozen=True)
class ThresholdSelection:
    """Fit-only threshold selection result."""

    threshold: float
    balanced_accuracy: float
    winner_recall: float
    loser_recall: float


@dataclass(frozen=True)
class FitCandidate:
    """Coefficient candidate selected on the fit segment only."""

    config: UtilityConfig
    threshold: ThresholdSelection
    fit_auc: float


@dataclass(frozen=True)
class CalibrationResult:
    """Full utility calibration result and write locations."""

    artifact: dict[str, Any]
    candidate_path: Path | None
    current_path: Path
    current_updated: bool = False

    @property
    def promotable(self) -> bool:
        return bool(self.artifact["promotable"])


def _require_columns(df: pd.DataFrame, columns: Sequence[str], sheet_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {sheet_name}: {missing}")


def _validate_strategy_id(df: pd.DataFrame, sheet_name: str) -> None:
    strategy_id = cast(pd.Series, df["strategy_id"])
    blank = strategy_id.isna() | strategy_id.astype(str).str.strip().eq("")
    blank_mask = blank.to_numpy(dtype=bool)
    if blank_mask.any():
        raise ValueError(f"Blank strategy_id in {sheet_name}: {int(blank_mask.sum())}")
    duplicate_mask = strategy_id.duplicated().to_numpy(dtype=bool)
    if duplicate_mask.any():
        values = strategy_id.loc[duplicate_mask].astype(str).head(5).tolist()
        raise ValueError(f"Duplicate strategy_id in {sheet_name}: {values}")


def _coerce_required_numeric(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    bad: dict[str, int] = {}
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        missing = int(cast(pd.Series, df[column]).isna().sum())
        if missing:
            bad[column] = missing
    if bad:
        raise ValueError(f"Missing or non-numeric {context}: {bad}")


def _dedup_feature_sheet(meta_features: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate Meta Features by strategy_id, preferring timestamped rows.

    The current workbook can contain auxiliary duplicate feature rows from
    partial/manual writes. For calibration we keep one decision-time row per
    strategy_id, preferring rows with a parseable ``start_time_utc`` and then
    the latest timestamp when multiple timestamped rows exist.
    """
    if "strategy_id" not in meta_features.columns:
        return meta_features
    duplicate_mask = cast(pd.Series, meta_features["strategy_id"]).duplicated(keep=False)
    if not bool(duplicate_mask.any()):
        return meta_features

    ranked = cast(pd.DataFrame, meta_features.copy())
    if "start_time_utc" in ranked.columns:
        parsed = cast(pd.Series, pd.to_datetime(ranked["start_time_utc"], errors="coerce", utc=True))
        ranked["_dedup_has_start"] = parsed.notna().astype(int)
        ranked["_dedup_start_time"] = parsed
        ranked = cast(
            pd.DataFrame,
            ranked.sort_values(
                by=["strategy_id", "_dedup_has_start", "_dedup_start_time"],
                ascending=[True, False, False],
                na_position="last",
            ),
        )
    ranked = cast(pd.DataFrame, ranked.drop_duplicates(subset=["strategy_id"], keep="first"))
    return cast(
        pd.DataFrame,
        ranked.drop(columns=["_dedup_has_start", "_dedup_start_time"], errors="ignore"),
    )


def _status_from_hmm_feature_source(frame: pd.DataFrame) -> pd.Series:
    """Derive backfill_status from hmm_feature_source for flat-layout inputs.

    The canonical backfill output stamps hmm_feature_source per row; rows
    successfully inferenced against the pinned artifact map to "complete",
    everything else to "hmm_failed" so the downstream pool filter excludes them.
    """
    if "hmm_feature_source" not in frame.columns:
        raise ValueError(
            "Missing required columns in flat backfill workbook: "
            "['backfill_status'] or ['hmm_feature_source']"
        )
    source = cast(pd.Series, frame["hmm_feature_source"]).astype("string").str.strip()
    return cast(
        pd.Series,
        pd.Series(
            np.where(source.eq("pinned_artifact_replay"), "complete", "hmm_failed"),
            index=frame.index,
            dtype="string",
        ),
    )


def _read_calibration_sources(workbook: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (general, meta, layout) supporting both two-sheet and flat layouts.

    Two-sheet layout: legacy ``General`` + ``Meta Features`` workbook.
    Flat layout: single-sheet output of ``scripts/backfill_training_features.py``.
    For flat layout, ``grids_count`` (preserved from the main extractor sheet) is
    aliased to ``num_grids`` and ``backfill_status`` is derived from
    ``hmm_feature_source`` when absent.
    """
    excel = pd.ExcelFile(workbook)
    sheet_names = [str(name) for name in excel.sheet_names]
    if "General" in sheet_names and "Meta Features" in sheet_names:
        general = cast(pd.DataFrame, pd.read_excel(excel, sheet_name="General")).copy()
        meta = _dedup_feature_sheet(
            cast(pd.DataFrame, pd.read_excel(excel, sheet_name="Meta Features"))
        )
        return general, meta, TWO_SHEET_INPUT_LAYOUT

    if len(sheet_names) == 1:
        flat = cast(pd.DataFrame, pd.read_excel(excel, sheet_name=sheet_names[0])).copy()
        if "num_grids" not in flat.columns and "grids_count" in flat.columns:
            flat["num_grids"] = flat["grids_count"]
        flat_required = [*OUTCOME_COLS, *REQUIRED_FEATURE_COLS]
        _require_columns(flat, flat_required, sheet_names[0])
        if not any(column in flat.columns for column in FEATURE_CUTOFF_COLS):
            raise ValueError(
                f"Missing required columns in {sheet_names[0]}: one of {list(FEATURE_CUTOFF_COLS)}"
            )
        meta = flat.copy()
        if "backfill_status" not in meta.columns:
            meta["backfill_status"] = _status_from_hmm_feature_source(meta)
        return flat, _dedup_feature_sheet(meta), FLAT_BACKFILL_INPUT_LAYOUT

    raise ValueError(
        "Calibration workbook must contain sheets ['General', 'Meta Features'] "
        "or one strict flat backfill sheet with outcome and utility feature columns; "
        f"found {sheet_names}"
    )


def _validate_hmm_lineage(
    pool: pd.DataFrame,
    *,
    expected_hmm_artifact_version: str | None,
) -> None:
    """Fail closed unless eligible calibration rows have one complete HMM lineage."""
    versions = cast(pd.Series, pool[HMM_LINEAGE_COL]).astype("string").str.strip()
    blank_version = versions.isna() | versions.eq("")
    if bool(blank_version.any()):
        raise ValueError(
            f"Missing HMM lineage in calibration pool: {int(blank_version.sum())}"
        )

    unique_versions = sorted(str(value) for value in versions.unique())
    if len(unique_versions) != 1:
        raise ValueError(f"Mixed HMM lineage in calibration pool: {unique_versions}")

    actual_version = unique_versions[0]
    if (
        expected_hmm_artifact_version is not None
        and actual_version != expected_hmm_artifact_version
    ):
        raise ValueError(
            "HMM lineage mismatch in calibration pool: "
            f"expected {expected_hmm_artifact_version}, found {actual_version}"
        )

    nonfinite: dict[str, int] = {}
    for column in HMM_PROBABILITY_COLS:
        numeric = np.asarray(
            pd.to_numeric(cast(pd.Series, pool[column]), errors="coerce"),
            dtype=float,
        )
        bad_count = int((~np.isfinite(numeric)).sum())
        if bad_count:
            nonfinite[column] = bad_count
    if nonfinite:
        raise ValueError(f"Non-finite HMM probabilities in calibration pool: {nonfinite}")


def _load_calibration_pool(
    path: Path,
    *,
    expected_hmm_artifact_version: str | None = None,
) -> pd.DataFrame:
    """Load the utility calibration pool from either layout."""
    s1, mf, _input_layout = _read_calibration_sources(path)

    _require_columns(s1, OUTCOME_COLS, "General")
    _require_columns(
        mf,
        (
            "strategy_id",
            *STATUS_COLS,
            *REQUIRED_FEATURE_COLS,
            HMM_LINEAGE_COL,
            "persistence_prob",
        ),
        "Meta Features",
    )

    out = cast(pd.DataFrame, s1.loc[:, list(OUTCOME_COLS)].copy())
    feature_cols = [
        "strategy_id",
        *STATUS_COLS,
        *REQUIRED_FEATURE_COLS,
        HMM_LINEAGE_COL,
        "persistence_prob",
    ]
    for column in OPTIONAL_FEATURE_COLS:
        if column in mf.columns:
            feature_cols.append(column)
    feat = cast(pd.DataFrame, mf.loc[:, feature_cols].copy())
    for column in OPTIONAL_FEATURE_COLS:
        if column not in feat.columns:
            feat[column] = pd.NA

    _validate_strategy_id(out, "General")
    _validate_strategy_id(feat, "Meta Features")

    status = cast(pd.Series, feat["backfill_status"]).astype("string").str.strip()
    blank_status = status.isna() | status.eq("")
    if bool(blank_status.any()):
        required_missing = cast(
            pd.Series,
            feat.loc[:, list(REQUIRED_FEATURE_COLS)].isna().all(axis=1),
        )
        drop_mask = cast(pd.Series, blank_status & required_missing)
        if bool(drop_mask.any()):
            feat = cast(pd.DataFrame, feat.loc[~drop_mask].copy())
            status = cast(pd.Series, feat["backfill_status"]).astype("string").str.strip()
            blank_status = status.isna() | status.eq("")
    blank_status_mask = blank_status.to_numpy(dtype=bool)
    if blank_status_mask.any():
        raise ValueError(f"Missing backfill_status in Meta Features: {int(blank_status_mask.sum())}")
    feat["backfill_status"] = status

    pool = cast(pd.DataFrame, out.merge(feat, on="strategy_id", how="inner", validate="one_to_one"))

    _coerce_required_numeric(pool, ("duration_hours",), "duration_hours")
    duration_mask = cast(pd.Series, pool["duration_hours"]).le(MAX_DURATION_HOURS)
    pool = cast(pd.DataFrame, pool.loc[duration_mask].copy())
    status_mask = cast(pd.Series, pool["backfill_status"]).ne("hmm_failed")
    pool = cast(pd.DataFrame, pool.loc[status_mask].copy())

    # A missing outcome cannot be classified without inventing a supervised
    # target. Preserve the source row in the workbook, but exclude it from the
    # calibration cohort before numeric validation. Non-blank, non-numeric
    # values still fail closed in _coerce_required_numeric below.
    raw_pnl = cast(pd.Series, pool["pnl_pct"])
    missing_outcome = raw_pnl.isna() | raw_pnl.astype("string").str.strip().eq("")
    pool = cast(pd.DataFrame, pool.loc[~missing_outcome].copy())

    _validate_hmm_lineage(
        pool,
        expected_hmm_artifact_version=expected_hmm_artifact_version,
    )
    _coerce_required_numeric(pool, ("pnl_pct",) + REQUIRED_FEATURE_COLS, "utility inputs")
    for column in OPTIONAL_FEATURE_COLS:
        pool[column] = pd.to_numeric(pool[column], errors="coerce")

    parsed_time = cast(pd.Series, pd.to_datetime(pool["start_time_utc"], errors="coerce", utc=True))
    if parsed_time.isna().any():
        bad_count = int(parsed_time.isna().sum())
        raise ValueError(f"Unparseable start_time_utc rows in calibration pool: {bad_count}")

    pool["_start_time_sort"] = parsed_time
    pool = cast(pd.DataFrame, pool.sort_values(by=["_start_time_sort", "strategy_id"]).reset_index(drop=True))
    pool[LABEL_COL] = cast(pd.Series, pool["pnl_pct"]).gt(0.0).astype(int)

    if cast(pd.Series, pool[LABEL_COL]).nunique() != 2:
        raise ValueError("Calibration pool must contain both winners and losers")

    return pool


def _pool_feature_coverage(pool: pd.DataFrame) -> list[dict[str, int | str]]:
    coverage: list[dict[str, int | str]] = []
    for feature in (*REQUIRED_FEATURE_COLS, *OPTIONAL_FEATURE_COLS):
        if feature in pool.columns:
            non_null = int(cast(pd.Series, pool[feature]).notna().sum())
        else:
            non_null = 0
        coverage.append(
            {
                "feature": feature,
                "non_null_count": non_null,
                "null_count": int(len(pool) - non_null),
            }
        )
    return coverage


def _split_pool(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    k_fit = int(round(0.75 * len(pool)))
    fit_df = cast(pd.DataFrame, pool.iloc[:k_fit].copy())
    holdout_df = cast(pd.DataFrame, pool.iloc[k_fit:].copy())
    _require_two_classes(cast(pd.Series, fit_df[LABEL_COL]).to_numpy(), "fit")
    _require_two_classes(cast(pd.Series, holdout_df[LABEL_COL]).to_numpy(), "holdout")
    return fit_df, holdout_df


def _require_two_classes(y: np.ndarray, segment_name: str) -> None:
    classes = set(np.asarray(y, dtype=int).tolist())
    if classes != {0, 1}:
        raise ValueError(f"{segment_name} segment must contain both utility classes")


def _utility_scores(df: pd.DataFrame, config: UtilityConfig) -> np.ndarray:
    range_prob = df["range_prob"].to_numpy(dtype=float)
    trend_prob = df["trend_prob"].to_numpy(dtype=float)
    profit_per_grid_pct = df["profit_per_grid_pct"].to_numpy(dtype=float)
    num_grids = df["num_grids"].to_numpy(dtype=float)
    range_size_pct = df["range_size_pct"].to_numpy(dtype=float)
    survival_prob = df["survival_prob"].to_numpy(dtype=float)
    hurst_exponent = df["hurst_exponent"].to_numpy(dtype=float)

    survival_prob = np.where(np.isfinite(survival_prob), survival_prob, range_prob)

    adjusted_fills = config.baseline_fills_per_hour * range_prob
    total_fills = adjusted_fills * config.horizon_hours
    active_fraction = np.minimum(0.75, np.sqrt(num_grids / 50.0) * 0.5)
    expected_return = total_fills * (profit_per_grid_pct / 100.0) * active_fraction * 100.0

    trend_factor = trend_prob ** 0.5
    expected_drawdown = (
        (range_size_pct / 2.0) * (1.0 - trend_factor)
        + abs(config.stop_loss_pct) * trend_factor
    )

    breakout_prob = 1.0 - survival_prob
    trend_strength = trend_prob ** 0.7
    hurst_factor = np.where(
        np.isfinite(hurst_exponent) & (hurst_exponent > 0.5),
        1.0 + (hurst_exponent - 0.5) * 2.0,
        1.0,
    )
    base_loss = range_size_pct * 0.5 * trend_strength
    stop_loss_loss = abs(config.stop_loss_pct)
    loss_magnitude = base_loss * (1.0 - trend_prob) + stop_loss_loss * trend_prob
    expected_breakout_loss = breakout_prob * loss_magnitude * hurst_factor

    return (
        expected_return
        - config.lambda_risk * expected_drawdown
        - config.kappa_trend * expected_breakout_loss
    )


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    _require_two_classes(y_true, "AUC")
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=float)
    y = np.asarray(y_true, dtype=int)
    positives = y == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    rank_sum_pos = float(ranks[positives].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    predictions = np.asarray(scores, dtype=float) >= float(threshold)
    positives = y == 1
    negatives = y == 0

    winner_recall = float((predictions & positives).sum() / positives.sum())
    loser_recall = float(((~predictions) & negatives).sum() / negatives.sum())
    false_positive_rate = float((predictions & negatives).sum() / negatives.sum())
    balanced_accuracy = float((winner_recall + loser_recall) / 2.0)
    return {
        "threshold": float(threshold),
        "balanced_accuracy": balanced_accuracy,
        "winner_recall": winner_recall,
        "loser_recall": loser_recall,
        "false_positive_rate": false_positive_rate,
    }


def _select_threshold_fit_only(y_true: np.ndarray, scores: np.ndarray) -> ThresholdSelection:
    finite_scores = np.asarray(scores, dtype=float)
    unique_scores = np.unique(finite_scores)
    if unique_scores.size == 0:
        raise ValueError("Cannot select threshold with no scores")

    eps = max(1e-9, float(np.ptp(unique_scores)) * 1e-9)
    thresholds = np.concatenate(
        [
            [float(unique_scores.min() - eps)],
            unique_scores,
            [float(unique_scores.max() + eps)],
        ]
    )
    best: tuple[tuple[float, float, float, float], ThresholdSelection] | None = None
    for threshold in thresholds:
        metrics = _threshold_metrics(y_true, finite_scores, float(threshold))
        key = (
            metrics["balanced_accuracy"],
            metrics["winner_recall"],
            metrics["loser_recall"],
            -abs(float(threshold)),
        )
        selection = ThresholdSelection(
            threshold=float(threshold),
            balanced_accuracy=metrics["balanced_accuracy"],
            winner_recall=metrics["winner_recall"],
            loser_recall=metrics["loser_recall"],
        )
        if best is None or key > best[0]:
            best = (key, selection)

    assert best is not None
    return best[1]


def _fit_coefficients(fit_df: pd.DataFrame) -> FitCandidate:
    y_fit = fit_df[LABEL_COL].to_numpy(dtype=int)
    best: tuple[tuple[float, float, float, float], FitCandidate] | None = None

    for lambda_risk in LAMBDA_GRID:
        for kappa_trend in KAPPA_GRID:
            for horizon_hours in HORIZON_GRID:
                config = UtilityConfig(
                    lambda_risk=float(lambda_risk),
                    kappa_trend=float(kappa_trend),
                    horizon_hours=float(horizon_hours),
                )
                scores = _utility_scores(fit_df, config)
                fit_auc = _auc(y_fit, scores)
                threshold = _select_threshold_fit_only(y_fit, scores)
                key = (
                    fit_auc,
                    threshold.balanced_accuracy,
                    threshold.winner_recall,
                    -abs(threshold.threshold),
                )
                candidate = FitCandidate(
                    config=config,
                    threshold=threshold,
                    fit_auc=fit_auc,
                )
                if best is None or key > best[0]:
                    best = (key, candidate)

    assert best is not None
    return best[1]


def _segment_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    winner_scores = scores[y == 1]
    loser_scores = scores[y == 0]
    metrics = {
        "auc": _auc(y, scores),
        "mean_winner_utility": float(np.mean(winner_scores)),
        "mean_loser_utility": float(np.mean(loser_scores)),
        "n_winners": int((y == 1).sum()),
        "n_losers": int((y == 0).sum()),
    }
    metrics.update(_threshold_metrics(y, scores, threshold))
    return metrics


def _class_counts(series: pd.Series) -> dict[str, int]:
    counts = series.astype(int).value_counts().sort_index()
    return {
        str(int(cast(Any, label))): int(cast(Any, count))
        for label, count in counts.items()
    }


def _boundary_gate(config: UtilityConfig) -> bool:
    values = (config.lambda_risk, config.kappa_trend, config.horizon_hours)
    if not all(np.isfinite(value) for value in values):
        return False
    if config.lambda_risk < 0 or config.kappa_trend < 0 or config.horizon_hours <= 0:
        return False
    return (
        min(LAMBDA_GRID) < config.lambda_risk < max(LAMBDA_GRID)
        and min(KAPPA_GRID) < config.kappa_trend < max(KAPPA_GRID)
        and min(HORIZON_GRID) < config.horizon_hours < max(HORIZON_GRID)
    )


def _build_artifact(
    *,
    expired_bots_path: Path,
    pool: pd.DataFrame,
    fit_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    candidate: FitCandidate,
) -> dict[str, Any]:
    y_fit = fit_df[LABEL_COL].to_numpy(dtype=int)
    y_holdout = holdout_df[LABEL_COL].to_numpy(dtype=int)

    candidate_fit_scores = _utility_scores(fit_df, candidate.config)
    candidate_holdout_scores = _utility_scores(holdout_df, candidate.config)

    fallback_config = UtilityConfig(
        lambda_risk=FALLBACK_LAMBDA_RISK,
        kappa_trend=FALLBACK_KAPPA_TREND,
        min_utility=FALLBACK_MIN_UTILITY,
        horizon_hours=FALLBACK_HORIZON_HOURS,
    )
    fallback_fit_scores = _utility_scores(fit_df, fallback_config)
    fallback_holdout_scores = _utility_scores(holdout_df, fallback_config)

    candidate_fit = _segment_metrics(
        y_fit,
        candidate_fit_scores,
        candidate.threshold.threshold,
    )
    candidate_holdout = _segment_metrics(
        y_holdout,
        candidate_holdout_scores,
        candidate.threshold.threshold,
    )
    fallback_fit = _segment_metrics(y_fit, fallback_fit_scores, fallback_config.min_utility)
    fallback_holdout = _segment_metrics(
        y_holdout,
        fallback_holdout_scores,
        fallback_config.min_utility,
    )

    gates = {
        "G0_schema": True,
        "G1_dedup": True,
        "G2_class_fit_holdout": set(y_fit.tolist()) == {0, 1}
        and set(y_holdout.tolist()) == {0, 1},
        "G3_holdout_auc_gt_0_5": candidate_holdout["auc"] > 0.5,
        "G4_holdout_mean_winner_gt_loser": (
            candidate_holdout["mean_winner_utility"]
            > candidate_holdout["mean_loser_utility"]
        ),
        "G5_threshold_selected_on_fit": True,
        "G6_winner_recall_not_regressed_vs_v0": (
            candidate_holdout["winner_recall"] >= fallback_holdout["winner_recall"]
        ),
        "G7_finite_nonnegative_not_boundary_pinned": _boundary_gate(candidate.config),
    }
    promotable = all(bool(value) for value in gates.values())

    created_at = datetime.now(timezone.utc)
    return {
        "artifact_type": "neutralgrid.utility_score",
        "schema_version": 2,
        "artifact_version": created_at.strftime("utility_%Y%m%d_%H%M%S_%f"),
        "created_at_utc": created_at.isoformat(),
        "source_workbook": str(expired_bots_path),
        "label_contract": {
            "eligible_universe": ELIGIBLE_UNIVERSE_RULE,
            "positive_class": "pnl_pct > 0.0",
            "negative_class": "pnl_pct <= 0.0",
            "label_column": LABEL_COL,
        },
        "data_contract": {
            "outcome_sheet": "General",
            "feature_sheet": "Meta Features",
            "merge_key": "strategy_id",
            "hmm_artifact_version": str(
                cast(pd.Series, pool[HMM_LINEAGE_COL]).iloc[0]
            ),
            "hmm_probability_columns": list(HMM_PROBABILITY_COLS),
            "outcome_columns": list(OUTCOME_COLS),
            "status_columns": list(STATUS_COLS),
            "required_feature_columns": list(REQUIRED_FEATURE_COLS),
            "optional_feature_columns": list(OPTIONAL_FEATURE_COLS),
            "ignored_columns": ["utility_score", "ou_halflife", "take_profit_pct"],
            "range_size_pct_source": "Meta Features",
            "stored_utility_score_used": False,
            "live_outcome_ingestor_used": False,
        },
        "split": {
            "rule": "sort by start_time_utc then strategy_id; k_fit = int(round(0.75 * n))",
            "n_pool": int(len(pool)),
            "n_fit": int(len(fit_df)),
            "n_holdout": int(len(holdout_df)),
            "pool_class_counts": _class_counts(cast(pd.Series, pool[LABEL_COL])),
            "fit_class_counts": _class_counts(cast(pd.Series, fit_df[LABEL_COL])),
            "holdout_class_counts": _class_counts(cast(pd.Series, holdout_df[LABEL_COL])),
        },
        "pool_feature_coverage": _pool_feature_coverage(pool),
        "coefficients": {
            "lambda_risk": float(candidate.config.lambda_risk),
            "kappa_trend": float(candidate.config.kappa_trend),
            "horizon_hours": float(candidate.config.horizon_hours),
        },
        "threshold": {
            "min_utility": float(candidate.threshold.threshold),
            "criterion": "balanced_accuracy",
            "selected_on": "fit",
        },
        "search_space": {
            "lambda_risk": [float(min(LAMBDA_GRID)), float(max(LAMBDA_GRID)), len(LAMBDA_GRID)],
            "kappa_trend": [float(min(KAPPA_GRID)), float(max(KAPPA_GRID)), len(KAPPA_GRID)],
            "horizon_hours": [float(min(HORIZON_GRID)), float(max(HORIZON_GRID)), len(HORIZON_GRID)],
        },
        "metrics": {
            "candidate": {
                "fit": candidate_fit,
                "holdout": candidate_holdout,
            },
            "v0_fallback": {
                "coefficients": {
                    "lambda_risk": FALLBACK_LAMBDA_RISK,
                    "kappa_trend": FALLBACK_KAPPA_TREND,
                    "horizon_hours": FALLBACK_HORIZON_HOURS,
                },
                "threshold": {"min_utility": FALLBACK_MIN_UTILITY},
                "fit": fallback_fit,
                "holdout": fallback_holdout,
            },
        },
        "gates": gates,
        "promotable": promotable,
        "promotion_policy": "current.json is updated only when all gates are true",
        "false_optionality_excluded": [
            "scripts/check_utility_drift.py",
            "automatic_recalibration_scheduler",
            "LiveOutcomeIngestor_rewrite",
            "ou_halflife_required_feature",
            "take_profit_pct_artifact_field",
            "hard_liquidation_blacklist",
            "new_calibration_package",
            "meta_labeler_rewrite",
            "scanner_profile_edits",
            "winner_only_ingest",
            "dual_provisional_utility_semantics",
        ],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def run_calibration(
    expired_bots_path: Path | str = Path("data/new_expired_bots.xlsx"),
    artifact_dir: Path | str = DEFAULT_UTILITY_ARTIFACT_PATH.parent,
    *,
    write_candidate: bool = True,
    promote: bool = False,
    expected_hmm_artifact_version: str | None = None,
) -> CalibrationResult:
    """Run utility calibration and optionally promote a passing artifact."""
    workbook_path = Path(expired_bots_path)
    utility_artifact_dir = Path(artifact_dir)

    active_hmm_version = expected_hmm_artifact_version or get_active_hmm_version()
    if not active_hmm_version:
        raise ValueError(
            "Cannot calibrate utility without an active HMM version in artifact_manifest.json"
        )
    pool = _load_calibration_pool(
        workbook_path,
        expected_hmm_artifact_version=active_hmm_version,
    )
    fit_df, holdout_df = _split_pool(pool)
    candidate = _fit_coefficients(fit_df)
    artifact = _build_artifact(
        expired_bots_path=workbook_path,
        pool=pool,
        fit_df=fit_df,
        holdout_df=holdout_df,
        candidate=candidate,
    )

    candidate_path: Path | None = None
    if write_candidate:
        candidate_path = utility_artifact_dir / f"{artifact['artifact_version']}.json"
        _write_json_atomic(candidate_path, artifact)

    current_path = utility_artifact_dir / "current.json"
    current_updated = False
    if promote and artifact["promotable"]:
        _write_json_atomic(current_path, artifact)
        current_updated = True

    return CalibrationResult(
        artifact=artifact,
        candidate_path=candidate_path,
        current_path=current_path,
        current_updated=current_updated,
    )


def build_cli_proof_summary(result: CalibrationResult) -> dict[str, Any]:
    artifact = result.artifact
    split = cast(dict[str, Any], artifact["split"])
    pool_class_counts = cast(dict[str, int], split["pool_class_counts"])
    data_contract = cast(dict[str, Any], artifact["data_contract"])
    label_contract = cast(dict[str, Any], artifact["label_contract"])
    holdout_gate_results = {
        str(key): bool(value)
        for key, value in cast(dict[str, Any], artifact["gates"]).items()
    }
    feature_coverage = [
        {
            "feature": str(item["feature"]),
            "non_null_count": int(item["non_null_count"]),
            "null_count": int(item["null_count"]),
        }
        for item in cast(list[dict[str, Any]], artifact["pool_feature_coverage"])
    ]
    return {
        "artifact_version": str(artifact["artifact_version"]),
        "eligible_universe": str(label_contract["eligible_universe"]),
        "positive_class": str(label_contract["positive_class"]),
        "negative_class": str(label_contract["negative_class"]),
        "feature_count": int(
            len(cast(list[str], data_contract["required_feature_columns"]))
            + len(cast(list[str], data_contract["optional_feature_columns"]))
        ),
        "feature_coverage": feature_coverage,
        "pool_rows": int(split["n_pool"]),
        "fit_rows": int(split["n_fit"]),
        "holdout_rows": int(split["n_holdout"]),
        "positive_count": int(pool_class_counts.get("1", 0)),
        "negative_count": int(pool_class_counts.get("0", 0)),
        "fit_class_counts": cast(dict[str, int], split["fit_class_counts"]),
        "holdout_class_counts": cast(dict[str, int], split["holdout_class_counts"]),
        "holdout_gate_results": holdout_gate_results,
        "promotable": bool(artifact["promotable"]),
        "current_json_updated": bool(result.current_updated),
        "current_json_path": str(result.current_path),
        "candidate_path": str(result.candidate_path) if result.candidate_path is not None else None,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run governed utility calibration and print a proof summary",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/new_expired_bots.xlsx",
        help=(
            "Workbook path: canonical General+Meta Features workbook or a "
            "compatible derived flat backfill staging file"
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=str(DEFAULT_UTILITY_ARTIFACT_PATH.parent),
        help="Directory where candidate/current utility artifacts are written",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote the candidate to current.json when all gates pass",
    )
    parser.add_argument(
        "--skip-candidate-write",
        action="store_true",
        help="Skip writing the timestamped candidate artifact",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_calibration(
        expired_bots_path=Path(args.input),
        artifact_dir=Path(args.artifact_dir),
        write_candidate=not bool(args.skip_candidate_write),
        promote=bool(args.promote),
    )
    print(
        json.dumps(
            build_cli_proof_summary(result),
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
