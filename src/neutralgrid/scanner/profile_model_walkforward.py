"""Walk-forward purged K-fold CV and promotion gate for the profile model.

Implements PATTERN_PROFILE_FIX.md Phase 3.1-3.6:
- 3.1 `walkforward_evaluate` — AFML-style purged K-fold over the labeled universe,
      purge window >= `max_duration_hours` so winner/loser windows do not straddle folds.
      Per-fold AUC + KS separation; aggregate `mean_auc`, `mean_pass_rate`.
- 3.2 `promote_profile_version` — refuses to overwrite the active artifact unless
      `mean_pass_rate >= 0.50` (matches HMM convention).
- 3.3 Artifact naming helper `make_profile_model_filename` — `profile_model_YYYYMMDD_HHMMSS.json`.
- 3.4 Trial tracking — every promotion attempt logs a `TrialRecord`.
- 3.5 Fail-closed caller contract verified via `resolve_active_profile_model_path`.
- 3.6 Silent-drop guard — `len(admitted) >= 0.9 * len(requested_features)` at promotion.

Not implemented here: the CLI wiring in `retrain_scanner.py` — that is an
integration step landed separately.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, cast

import numpy as np
import pandas as pd

from neutralgrid.core.config import get_config
from neutralgrid.core.constants import PROFIT_FACTOR_CAP
from neutralgrid.scanner._xlsx_io import (
    detect_format,
    raise_on_duplicate_strategy_id,
    read_sheet,
    read_single_sheet,
    validate_dataframe,
)
from neutralgrid.scanner.pattern_profile import (
    DEFAULT_FEATURES,
    PatternProfile,
    build_profile_from_enhanced_xlsx,
)
from neutralgrid.scanner.profile_model import (
    ProfileModel,
    save_profile_model,
    train_profile_model_from_enhanced_xlsx,
)

logger = logging.getLogger(__name__)


# ── Thresholds (single source of truth) ──────────────────────────────────────
AUC_FOLD_PASS_THRESHOLD: float = 0.55
MEAN_PASS_RATE_FLOOR: float = 0.50
COVERAGE_FLOOR: float = 0.90
MIN_FINITE_FOLDS: int = 3
FINITE_FOLD_COVERAGE_FLOOR: float = 0.60
MEAN_AUC_FLOOR: float = 0.55
POOLED_OOF_AUC_FLOOR: float = 0.55
PAIRED_BOOTSTRAP_REPLICATES: int = 5_000
PAIRED_BOOTSTRAP_SEED: int = 20260801


@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregate and per-fold metrics from purged K-fold CV.

    Fields
    ------
    n_folds : int
        Number of folds actually evaluated.
    fold_auc : list[float]
        Per-fold AUC. NaN for folds whose test set is single-class.
    fold_ks : list[float]
        Per-fold KS separation between winner/loser llr distributions.
    mean_auc : float
        Mean of finite `fold_auc` values.
    mean_ks : float
        Mean of finite `fold_ks` values.
    mean_pass_rate : float
        Fraction of folds with `auc >= AUC_FOLD_PASS_THRESHOLD`.
    purge_hours : float
        Purge window applied between train and test folds.
    requested_features : list[str]
        Features declared by the caller (pre-availability-filter).
    admitted_features : list[str]
        Features that survived the labeled-universe availability filter.
    feature_coverage : float
        `len(admitted_features) / len(requested_features)`.
    """

    n_folds: int
    fold_auc: list[float]
    fold_ks: list[float]
    mean_auc: float
    mean_ks: float
    mean_pass_rate: float
    purge_hours: float
    requested_features: list[str]
    admitted_features: list[str]
    feature_coverage: float
    fold_train_rows: list[int] = field(default_factory=list)
    fold_test_rows: list[int] = field(default_factory=list)
    fold_train_winners: list[int] = field(default_factory=list)
    fold_test_winners: list[int] = field(default_factory=list)
    fold_pnl_thresholds: list[float] = field(default_factory=list)
    fold_test_start_utc: list[str] = field(default_factory=list)
    fold_test_end_utc: list[str] = field(default_factory=list)
    oof_strategy_ids: list[str] = field(default_factory=list)
    oof_labels: list[int] = field(default_factory=list)
    oof_scores: list[float] = field(default_factory=list)
    oof_probabilities: list[float] = field(default_factory=list)
    pooled_oof_auc: float = float("nan")
    pooled_oof_ks: float = float("nan")
    pooled_oof_brier: float = float("nan")
    pooled_oof_ece: float = float("nan")
    source_sha256: Optional[str] = None
    labeled_rows: int = 0
    duplicate_strategy_ids: int = 0
    holdout_start_after_utc: Optional[str] = None


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of a promotion attempt. Frozen for audit-trail parity."""

    promoted: bool
    reason: str
    mean_pass_rate: Optional[float]
    feature_coverage: Optional[float]
    dropped_features: list[str] = field(default_factory=list)
    artifact_filename: Optional[str] = None
    evaluation_filename: Optional[str] = None
    paired_auc_delta: Optional[float] = None
    paired_auc_delta_ci_low: Optional[float] = None
    paired_auc_delta_ci_high: Optional[float] = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_profile_model_filename(ts: Optional[datetime] = None) -> str:
    """Return timestamped filename `profile_model_YYYYMMDD_HHMMSS.json` (UTC).

    Mirrors HMM `rolling_180d_YYYYMMDD_HHMMSS` naming convention.
    """
    if ts is None:
        ts = datetime.now(timezone.utc)
    return f"profile_model_{ts.strftime('%Y%m%d_%H%M%S')}.json"


def make_pattern_profile_filename(ts: Optional[datetime] = None) -> str:
    if ts is None:
        ts = datetime.now(timezone.utc)
    return f"pattern_profile_{ts.strftime('%Y%m%d_%H%M%S')}.json"


def _evaluation_pointer_is_valid(pointer: dict[str, Any], profile_dir: Path) -> bool:
    """Validate the evidence record linked by a governed current pointer.

    Pointers created before evaluation hashing are accepted only when both
    evidence fields are absent.  A partially populated or hash-mismatched
    evidence link fails closed.
    """
    name = pointer.get("evaluation")
    expected_sha256 = pointer.get("evaluation_sha256")
    if name is None and expected_sha256 is None:
        return True
    if not isinstance(name, str) or not isinstance(expected_sha256, str):
        logger.warning("profile current.json has an incomplete evaluation link")
        return False
    relative = Path(name)
    if (
        relative.is_absolute()
        or relative.parts != ("evaluations", relative.name)
        or not relative.name.startswith("profile_evaluation_")
        or relative.suffix != ".json"
    ):
        logger.warning("profile current.json has an invalid evaluation path: %r", name)
        return False
    evaluation_path = profile_dir / relative
    if not evaluation_path.exists() or _hash_file(evaluation_path) != expected_sha256:
        logger.warning(
            "profile evaluation evidence missing or hash-mismatched: %s",
            evaluation_path,
        )
        return False
    try:
        evidence = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("profile evaluation evidence is not valid JSON: %s", exc)
        return False
    candidate = evidence.get("candidate")
    pattern = evidence.get("candidate_pattern_profile")
    if evidence.get("gate_decision") != "pass" or not isinstance(candidate, dict):
        logger.warning("profile evaluation evidence does not record a passing gate")
        return False
    if candidate.get("sha256") != pointer.get("sha256"):
        logger.warning("profile evaluation/model hashes disagree")
        return False
    if not isinstance(pattern, dict) or pattern.get("sha256") != pointer.get(
        "pattern_profile_sha256"
    ):
        logger.warning("profile evaluation/pattern hashes disagree")
        return False
    return True


def resolve_active_profile_model_path(profile_dir: Optional[Path] = None) -> Path:
    """Resolve the active profile_model.json path.

    Resolution order:
      1. If `current.json` exists and carries a valid `active` key, return the
         promoted artifact it points to.
      2. If `current.json` is absent and `profile_model.json` exists, return
         that file as a bootstrap candidate so the pipeline can run before
         walk-forward promotion is statistically meaningful.
      3. If `current.json` is corrupt / missing `active`, or if neither
         promoted nor bootstrap artifacts exist, return a non-existent sentinel
         so the caller's `exists()` check yields `data_missing`.

    Bootstrap fallback is allowed only when the promotion pointer is absent.
    A corrupt pointer remains fail-closed to avoid silently bypassing a broken
    promotion state.
    """
    if profile_dir is None:
        profile_dir = Path(get_config().base_dir) / "data" / "profile"
    current = profile_dir / "current.json"
    bootstrap_candidate = profile_dir / "profile_model.json"
    if current.exists():
        try:
            pointer = json.loads(current.read_text(encoding="utf-8"))
            if not isinstance(pointer, dict) or not _evaluation_pointer_is_valid(
                pointer, profile_dir
            ):
                return profile_dir / "_invalid_profile_evaluation.missing"
            name = pointer.get("active")
            if (
                isinstance(name, str)
                and name
                and Path(name).name == name
                and not Path(name).is_absolute()
            ):
                candidate = profile_dir / name
                if not candidate.exists():
                    logger.warning(
                        "profile current.json points to missing model artifact: %s",
                        candidate,
                    )
                    return profile_dir / "_missing_promoted_model.missing"
                expected_sha256 = pointer.get("sha256")
                if (
                    isinstance(expected_sha256, str)
                    and expected_sha256
                    and _hash_file(candidate) != expected_sha256
                ):
                    logger.warning(
                        "profile model hash mismatch for %s — fail-closed sentinel returned",
                        candidate,
                    )
                    return profile_dir / "_profile_model_hash_mismatch.missing"
                return candidate
            logger.warning(
                "profile current.json missing valid 'active' key: %r — "
                "fail-closed sentinel returned", pointer,
            )
            return profile_dir / "_corrupt_current_json.missing"
        except Exception as exc:
            logger.warning(
                "Failed to parse profile current.json at %s: %s — "
                "fail-closed sentinel returned", current, exc,
            )
            return profile_dir / "_corrupt_current_json.missing"
    if bootstrap_candidate.exists():
        logger.warning(
            "profile current.json absent at %s — using bootstrap candidate %s. "
            "This artifact is unpromoted and intended only to break the cold-"
            "start loop until enough outcome data exists for walk-forward "
            "promotion.",
            current,
            bootstrap_candidate,
        )
        return bootstrap_candidate
    logger.warning(
        "profile current.json absent at %s and no bootstrap candidate exists "
        "at %s — fail-closed sentinel returned. Run retrain_scanner to create "
        "a bootstrap artifact or promote a walk-forward-validated artifact "
        "before production scans.",
        current,
        bootstrap_candidate,
    )
    return profile_dir / "_no_current_json.missing"


def resolve_active_pattern_profile_path(profile_dir: Optional[Path] = None) -> Path:
    """Resolve the pattern artifact paired with the active profile model."""
    if profile_dir is None:
        profile_dir = Path(get_config().base_dir) / "data" / "profile"
    current = profile_dir / "current.json"
    bootstrap_candidate = profile_dir / "pattern_profile.json"
    if current.exists():
        try:
            pointer = json.loads(current.read_text(encoding="utf-8"))
            if not isinstance(pointer, dict) or not _evaluation_pointer_is_valid(
                pointer, profile_dir
            ):
                return profile_dir / "_invalid_pattern_evaluation.missing"
            name = pointer.get("active_pattern_profile")
            if (
                isinstance(name, str)
                and name
                and Path(name).name == name
                and not Path(name).is_absolute()
            ):
                candidate = profile_dir / name
                if candidate.exists():
                    expected_sha256 = pointer.get("pattern_profile_sha256")
                    if (
                        isinstance(expected_sha256, str)
                        and expected_sha256
                        and _hash_file(candidate) != expected_sha256
                    ):
                        logger.warning(
                            "pattern profile hash mismatch for %s — "
                            "fail-closed sentinel returned",
                            candidate,
                        )
                        return profile_dir / "_pattern_profile_hash_mismatch.missing"
                    return candidate
            logger.warning(
                "profile current.json missing a valid existing "
                "active_pattern_profile: %r — fail-closed sentinel returned",
                pointer,
            )
        except Exception as exc:
            logger.warning(
                "Failed to parse profile current.json at %s: %s — "
                "fail-closed sentinel returned",
                current,
                exc,
            )
        return profile_dir / "_corrupt_pattern_current_json.missing"
    if bootstrap_candidate.exists():
        logger.warning(
            "profile current.json absent at %s — using unpromoted bootstrap "
            "pattern candidate %s",
            current,
            bootstrap_candidate,
        )
        return bootstrap_candidate
    return profile_dir / "_no_pattern_current_json.missing"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _read_evaluation_history(
    profile_dir: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    evaluation_dir = profile_dir / "evaluations"
    if not evaluation_dir.exists():
        return []
    history: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(evaluation_dir.glob("profile_evaluation_*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(
                f"Invalid profile evaluation history at {path}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"Profile evaluation history is not an object: {path}")
        history.append((path, value))
    return history


def resolve_latest_evaluated_holdout_end(
    profile_dir: Optional[Path] = None,
) -> Optional[str]:
    """Return the latest disclosed OOF endpoint in governed evaluation history."""
    if profile_dir is None:
        profile_dir = Path(get_config().base_dir) / "data" / "profile"
    latest: Optional[pd.Timestamp] = None
    for path, evidence in _read_evaluation_history(profile_dir):
        walkforward = evidence.get("walkforward")
        if not isinstance(walkforward, dict):
            raise ValueError(f"Missing walkforward evidence in {path}")
        raw_endpoints = walkforward.get("fold_test_end_utc", [])
        if not isinstance(raw_endpoints, list):
            raise ValueError(f"Invalid fold_test_end_utc evidence in {path}")
        for raw_value in raw_endpoints:
            value = pd.to_datetime(raw_value, utc=True, errors="coerce")
            if pd.isna(value):
                raise ValueError(f"Invalid OOF endpoint {raw_value!r} in {path}")
            timestamp = cast(pd.Timestamp, value)
            if latest is None or timestamp > latest:
                latest = timestamp
    return latest.isoformat() if latest is not None else None


def _prior_holdout_overlap(
    profile_dir: Path,
    strategy_ids: list[str],
) -> tuple[int, list[str]]:
    candidate_ids = set(strategy_ids)
    if not candidate_ids:
        return 0, []
    overlapping_ids: set[str] = set()
    evidence_files: list[str] = []
    for path, evidence in _read_evaluation_history(profile_dir):
        walkforward = evidence.get("walkforward")
        if not isinstance(walkforward, dict):
            raise ValueError(f"Missing walkforward evidence in {path}")
        raw_ids = walkforward.get("oof_strategy_ids", [])
        if not isinstance(raw_ids, list):
            raise ValueError(f"Invalid oof_strategy_ids evidence in {path}")
        overlap = candidate_ids.intersection(str(value) for value in raw_ids)
        if overlap:
            overlapping_ids.update(overlap)
            evidence_files.append(path.name)
    return len(overlapping_ids), evidence_files


# ── Purged K-fold CV ─────────────────────────────────────────────────────────


def _load_labeled_frame(
    xlsx_path: Path,
    *,
    max_duration_hours: float,
    min_profit_factor: float,
    min_avg_profit_per_grid: float,
    top_quantile: float,
    features: Iterable[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Reload xlsx → df_labeled with winner/loser labels attached.

    Returns (df_labeled_with_labels, requested_numeric_features, admitted_features).
    """
    fmt = detect_format(xlsx_path)
    if fmt == "multi":
        df_entry = read_sheet(xlsx_path, "Entry Validation Metrics")
        df_perf = read_sheet(xlsx_path, "Performance Risk-Adjusted")
        df_mkt = read_sheet(xlsx_path, "Market and Volatility")
        df_entry, _ = validate_dataframe(df_entry, "Entry Validation Metrics")
        df_perf, _ = validate_dataframe(df_perf, "Performance Risk-Adjusted")
        df_mkt, _ = validate_dataframe(df_mkt, "Market and Volatility")
        raise_on_duplicate_strategy_id(df_entry, context="Entry Validation Metrics")
        raise_on_duplicate_strategy_id(df_perf, context="Performance Risk-Adjusted")
        raise_on_duplicate_strategy_id(df_mkt, context="Market and Volatility")
        perf_cols = [
            "strategy_id", "pnl_pct", "profit_factor", "duration_hours",
            "total_profit_usdt", "avg_profit_per_grid", "status",
            "start_time_utc", "end_time_utc",
        ]
        perf_to_merge = ["strategy_id"] + [c for c in perf_cols[1:] if c not in df_entry.columns]
        # Post-Phase 5.1 no mkt-sheet columns are consumed as features; merge auto-skips.
        mkt_cols = ["strategy_id"]
        mkt_to_merge = ["strategy_id"] + [c for c in mkt_cols[1:] if c not in df_entry.columns]
        df = df_entry.copy()
        if len(perf_to_merge) > 1:
            df = df.merge(df_perf[perf_to_merge], on="strategy_id", how="left")
        if len(mkt_to_merge) > 1:
            df = df.merge(df_mkt[mkt_to_merge], on="strategy_id", how="left")
    else:
        df, sheet_name = read_single_sheet(xlsx_path)
        df, _ = validate_dataframe(df, sheet_name)
        raise_on_duplicate_strategy_id(df, context=sheet_name)

    requested_numeric = list(features)
    numeric_cols = set(requested_numeric) | {
        "pnl_pct", "profit_factor", "duration_hours", "avg_profit_per_grid",
    }
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "profit_factor" in df.columns:
        df["profit_factor"] = df["profit_factor"].replace([np.inf, -np.inf], np.nan)
        df["profit_factor"] = df["profit_factor"].clip(lower=0.0, upper=PROFIT_FACTOR_CAP)

    duration_series = (
        cast(pd.Series, df["duration_hours"])
        if "duration_hours" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    df_train = cast(pd.DataFrame, df.loc[
        (duration_series >= 0.0) & (duration_series < max_duration_hours)
    ].copy())
    pf_train = (
        cast(pd.Series, df_train["profit_factor"])
        if "profit_factor" in df_train.columns
        else pd.Series(np.nan, index=df_train.index, dtype=float)
    )
    pnl_train = (
        cast(pd.Series, df_train["pnl_pct"])
        if "pnl_pct" in df_train.columns
        else pd.Series(np.nan, index=df_train.index, dtype=float)
    )
    mask = pf_train.notna() & pnl_train.notna()
    df_labeled = cast(pd.DataFrame, df_train.loc[mask].copy())

    if df_labeled.empty:
        return df_labeled, requested_numeric, []

    # Attach a global winner label (used only for the *final* full-data fit).
    # Per-fold CV below recomputes pnl_thr on the train fold only to prevent
    # test-set pnl from leaking into the winner-label definition.
    df_labeled = _apply_winner_labels(
        df_labeled,
        pnl_thr=float(cast(pd.Series, df_labeled["pnl_pct"]).quantile(top_quantile)),
        min_profit_factor=min_profit_factor,
        min_avg_profit_per_grid=min_avg_profit_per_grid,
    )

    admitted = [
        f for f in requested_numeric
        if f in df_labeled.columns
        and int(cast(pd.Series, df_labeled[f]).notna().sum()) >= 10
    ]
    return df_labeled, requested_numeric, admitted


def _apply_winner_labels(
    df: pd.DataFrame,
    *,
    pnl_thr: float,
    min_profit_factor: float,
    min_avg_profit_per_grid: float,
) -> pd.DataFrame:
    """Assign `_is_winner` column using the supplied pnl threshold.

    Extracted so fold-local CV can recompute `pnl_thr` on train-only rows,
    keeping test-fold pnl out of the winner-label definition.
    """
    if df.empty:
        return df.assign(_is_winner=pd.Series([], dtype=int))
    pf = cast(pd.Series, df["profit_factor"])
    pnl = cast(pd.Series, df["pnl_pct"])
    winners_mask = (pf >= min_profit_factor) & (pnl >= pnl_thr)
    if "avg_profit_per_grid" in df.columns:
        apg = cast(pd.Series, df["avg_profit_per_grid"])
        winners_mask = winners_mask & ((apg.isna()) | (apg >= min_avg_profit_per_grid))
    return df.assign(_is_winner=winners_mask.astype(int))


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    combined = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), combined, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), combined, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U formulation — no sklearn dependency."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    n_pos = pos.size
    n_neg = neg.size
    ranks = pd.Series(np.concatenate([pos, neg])).rank(method="average").to_numpy()
    rank_sum_pos = float(np.sum(ranks[:n_pos]))
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    if probabilities.size == 0 or labels.size != probabilities.size:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(probabilities.size)
    ece = 0.0
    for index in range(n_bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == n_bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(np.mean(probabilities[mask]))
        accuracy = float(np.mean(labels[mask]))
        ece += (count / total) * abs(confidence - accuracy)
    return float(ece)


def _purged_train_prefix(
    df_labeled: pd.DataFrame,
    *,
    test_start_idx: int,
    purge_delta: pd.Timedelta,
    max_duration_hours: float,
) -> pd.DataFrame:
    test_start_time = cast(
        pd.Timestamp, df_labeled["start_time_utc"].iloc[test_start_idx]
    )
    train_slice = df_labeled.iloc[:test_start_idx].copy()
    dur_raw = (
        cast(pd.Series, train_slice["duration_hours"])
        if "duration_hours" in train_slice.columns
        else pd.Series(np.nan, index=train_slice.index, dtype=float)
    )
    train_dur = cast(pd.Series, pd.to_numeric(dur_raw, errors="coerce")).fillna(
        float(max_duration_hours)
    )
    train_end = train_slice["start_time_utc"] + pd.to_timedelta(train_dur, unit="h")
    train_cutoff = test_start_time - purge_delta
    keep = (train_slice["start_time_utc"] < train_cutoff) & (train_end < test_start_time)
    return cast(pd.DataFrame, train_slice.loc[keep].copy())


def walkforward_evaluate(
    xlsx_path: str | Path,
    *,
    n_folds: int = 5,
    purge_hours: float = 7.0,
    max_duration_hours: float = 7.0,
    min_profit_factor: float = 1.5,
    min_avg_profit_per_grid: Optional[float] = None,
    top_quantile: float = 0.75,
    features: Iterable[str] = DEFAULT_FEATURES,
    shrinkage: float = 0.30,
    holdout_start_after_utc: Optional[str] = None,
) -> WalkForwardResult:
    """Evaluate on purge-safe expanding-window folds after a viable warm-up.

    The first test origin is the earliest chronological row whose purged train
    prefix satisfies the same per-class floor used by the final fit.  This
    avoids manufacturing unusable early folds while retaining every genuinely
    future observation after the first statistically viable training origin.
    Test labels use the train-only PnL threshold for their fold.  When
    `holdout_start_after_utc` is supplied, OOF rows begin strictly after that
    already-disclosed endpoint; earlier rows remain available only to the
    expanding training prefix.
    """
    xlsx_path = Path(xlsx_path)
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if purge_hours < max_duration_hours:
        raise ValueError(
            f"purge_hours ({purge_hours}) must be >= max_duration_hours "
            f"({max_duration_hours}) to prevent window-overlap leakage across folds."
        )
    if min_avg_profit_per_grid is None:
        min_avg_profit_per_grid = get_config().grid.profit_grid_min_pct_static_fallback

    holdout_cutoff: Optional[pd.Timestamp] = None
    normalized_holdout_cutoff: Optional[str] = None
    if holdout_start_after_utc is not None:
        parsed_cutoff = pd.to_datetime(
            holdout_start_after_utc, utc=True, errors="coerce"
        )
        if pd.isna(parsed_cutoff):
            raise ValueError(
                "holdout_start_after_utc must be a valid UTC timestamp, got "
                f"{holdout_start_after_utc!r}"
            )
        holdout_cutoff = cast(pd.Timestamp, parsed_cutoff)
        normalized_holdout_cutoff = holdout_cutoff.isoformat()

    source_sha256 = _hash_file(xlsx_path) if xlsx_path.exists() else None
    df_labeled, requested_numeric, admitted = _load_labeled_frame(
        xlsx_path,
        max_duration_hours=max_duration_hours,
        min_profit_factor=min_profit_factor,
        min_avg_profit_per_grid=min_avg_profit_per_grid,
        top_quantile=top_quantile,
        features=list(features),
    )
    coverage = len(admitted) / len(requested_numeric) if requested_numeric else 0.0

    def _empty_result() -> WalkForwardResult:
        return WalkForwardResult(
            n_folds=0,
            fold_auc=[],
            fold_ks=[],
            mean_auc=float("nan"),
            mean_ks=float("nan"),
            mean_pass_rate=0.0,
            purge_hours=purge_hours,
            requested_features=requested_numeric,
            admitted_features=admitted,
            feature_coverage=coverage,
            source_sha256=source_sha256,
            labeled_rows=len(df_labeled),
            holdout_start_after_utc=normalized_holdout_cutoff,
        )

    if not admitted:
        logger.warning(
            "Walk-forward aborted: no admitted features in labeled universe "
            "(requested=%s, coverage=%.3f)",
            requested_numeric,
            coverage,
        )
        return _empty_result()
    if df_labeled.empty or "start_time_utc" not in df_labeled.columns:
        logger.warning(
            "Walk-forward aborted: empty labeled set or missing start_time_utc "
            "(labeled=%d, cols=%s)",
            len(df_labeled),
            list(df_labeled.columns)[:5],
        )
        return _empty_result()

    df_labeled = df_labeled.copy()
    df_labeled["start_time_utc"] = pd.to_datetime(
        df_labeled["start_time_utc"], utc=True, errors="coerce"
    )
    df_labeled = cast(
        pd.DataFrame,
        df_labeled.dropna(subset=["start_time_utc"])
        .sort_values("start_time_utc")
        .reset_index(drop=True),
    )
    n = len(df_labeled)
    min_samples = max(30, 3 * len(admitted))
    purge_delta = cast(pd.Timedelta, pd.Timedelta(hours=float(purge_hours)))
    first_test_idx: Optional[int] = None
    for test_start_idx in range(min_samples * 2, n):
        train_df = _purged_train_prefix(
            df_labeled,
            test_start_idx=test_start_idx,
            purge_delta=purge_delta,
            max_duration_hours=max_duration_hours,
        )
        if train_df.empty:
            continue
        pnl_threshold = float(
            cast(pd.Series, train_df["pnl_pct"]).quantile(top_quantile)
        )
        train_df = _apply_winner_labels(
            train_df,
            pnl_thr=pnl_threshold,
            min_profit_factor=min_profit_factor,
            min_avg_profit_per_grid=float(min_avg_profit_per_grid),
        )
        winner_count = int(cast(pd.Series, train_df["_is_winner"]).sum())
        if winner_count >= min_samples and len(train_df) - winner_count >= min_samples:
            first_test_idx = test_start_idx
            break

    if first_test_idx is not None and holdout_cutoff is not None:
        fresh_positions = np.flatnonzero(
            np.asarray(df_labeled["start_time_utc"] > holdout_cutoff, dtype=bool)
        )
        if fresh_positions.size == 0:
            logger.warning(
                "Walk-forward aborted: no labeled rows occur after disclosed "
                "holdout endpoint %s",
                normalized_holdout_cutoff,
            )
            return _empty_result()
        first_test_idx = max(first_test_idx, int(fresh_positions[0]))

    if first_test_idx is None or n - first_test_idx < n_folds * 2:
        logger.warning(
            "Walk-forward aborted: no purge-safe origin leaves %d folds with at "
            "least two test rows (labeled=%d, class_floor=%d)",
            n_folds,
            n,
            min_samples,
        )
        return _empty_result()

    fold_indices = [
        np.asarray(indices, dtype=int)
        for indices in np.array_split(np.arange(first_test_idx, n), n_folds)
    ]
    fold_auc: list[float] = []
    fold_ks: list[float] = []
    fold_train_rows: list[int] = []
    fold_test_rows: list[int] = []
    fold_train_winners: list[int] = []
    fold_test_winners: list[int] = []
    fold_pnl_thresholds: list[float] = []
    fold_test_start_utc: list[str] = []
    fold_test_end_utc: list[str] = []
    oof_strategy_ids: list[str] = []
    oof_labels: list[int] = []
    oof_scores: list[float] = []
    oof_probabilities: list[float] = []

    for indices in fold_indices:
        test_start_idx = int(indices[0])
        test_df = df_labeled.iloc[indices].copy()
        train_df = _purged_train_prefix(
            df_labeled,
            test_start_idx=test_start_idx,
            purge_delta=purge_delta,
            max_duration_hours=max_duration_hours,
        )
        pnl_threshold = float(
            cast(pd.Series, train_df["pnl_pct"]).quantile(top_quantile)
        )
        train_df = _apply_winner_labels(
            train_df,
            pnl_thr=pnl_threshold,
            min_profit_factor=min_profit_factor,
            min_avg_profit_per_grid=float(min_avg_profit_per_grid),
        )
        test_df = _apply_winner_labels(
            test_df,
            pnl_thr=pnl_threshold,
            min_profit_factor=min_profit_factor,
            min_avg_profit_per_grid=float(min_avg_profit_per_grid),
        )
        train_winners = int(cast(pd.Series, train_df["_is_winner"]).sum())
        test_winners = int(cast(pd.Series, test_df["_is_winner"]).sum())
        fold_train_rows.append(len(train_df))
        fold_test_rows.append(len(test_df))
        fold_train_winners.append(train_winners)
        fold_test_winners.append(test_winners)
        fold_pnl_thresholds.append(pnl_threshold)
        fold_test_start_utc.append(
            cast(pd.Timestamp, test_df["start_time_utc"].min()).isoformat()
        )
        fold_test_end_utc.append(
            cast(pd.Timestamp, test_df["start_time_utc"].max()).isoformat()
        )

        if train_winners < min_samples or len(train_df) - train_winners < min_samples:
            fold_auc.append(float("nan"))
            fold_ks.append(float("nan"))
            continue
        model = _train_from_frame(
            train_df,
            admitted,
            shrinkage=shrinkage,
            max_duration_hours=max_duration_hours,
        )
        if model is None:
            fold_auc.append(float("nan"))
            fold_ks.append(float("nan"))
            continue

        fold_scores: list[float] = []
        fold_labels: list[int] = []
        for row_index, row in test_df.iterrows():
            raw = {feature: row.get(feature) for feature in admitted}
            llr = model.llr(raw)
            probability = model.proba(raw)
            if (
                llr is None
                or probability is None
                or not np.isfinite(llr)
                or not np.isfinite(probability)
            ):
                continue
            label = int(cast(float, row["_is_winner"]))
            strategy_id = str(row.get("strategy_id", row_index))
            fold_scores.append(llr)
            fold_labels.append(label)
            oof_strategy_ids.append(strategy_id)
            oof_labels.append(label)
            oof_scores.append(float(llr))
            oof_probabilities.append(float(probability))
        scores_array = np.asarray(fold_scores, dtype=float)
        labels_array = np.asarray(fold_labels, dtype=int)
        fold_auc.append(_auc(scores_array, labels_array))
        fold_ks.append(
            _ks_statistic(
                scores_array[labels_array == 1], scores_array[labels_array == 0]
            )
        )

    finite_auc = [value for value in fold_auc if np.isfinite(value)]
    finite_ks = [value for value in fold_ks if np.isfinite(value)]
    mean_auc = float(np.mean(finite_auc)) if finite_auc else float("nan")
    mean_ks = float(np.mean(finite_ks)) if finite_ks else float("nan")
    pass_rate = (
        sum(value >= AUC_FOLD_PASS_THRESHOLD for value in finite_auc) / len(finite_auc)
        if finite_auc
        else 0.0
    )
    scores_array = np.asarray(oof_scores, dtype=float)
    labels_array = np.asarray(oof_labels, dtype=int)
    probabilities_array = np.asarray(oof_probabilities, dtype=float)
    pooled_auc = _auc(scores_array, labels_array)
    pooled_ks = _ks_statistic(
        scores_array[labels_array == 1], scores_array[labels_array == 0]
    )
    pooled_brier = (
        float(np.mean(np.square(probabilities_array - labels_array)))
        if probabilities_array.size
        else float("nan")
    )

    return WalkForwardResult(
        n_folds=len(fold_auc),
        fold_auc=fold_auc,
        fold_ks=fold_ks,
        mean_auc=mean_auc,
        mean_ks=mean_ks,
        mean_pass_rate=float(pass_rate),
        purge_hours=purge_hours,
        requested_features=requested_numeric,
        admitted_features=admitted,
        feature_coverage=coverage,
        fold_train_rows=fold_train_rows,
        fold_test_rows=fold_test_rows,
        fold_train_winners=fold_train_winners,
        fold_test_winners=fold_test_winners,
        fold_pnl_thresholds=fold_pnl_thresholds,
        fold_test_start_utc=fold_test_start_utc,
        fold_test_end_utc=fold_test_end_utc,
        oof_strategy_ids=oof_strategy_ids,
        oof_labels=oof_labels,
        oof_scores=oof_scores,
        oof_probabilities=oof_probabilities,
        pooled_oof_auc=pooled_auc,
        pooled_oof_ks=pooled_ks,
        pooled_oof_brier=pooled_brier,
        pooled_oof_ece=_expected_calibration_error(
            probabilities_array, labels_array
        ),
        source_sha256=source_sha256,
        labeled_rows=len(df_labeled),
        duplicate_strategy_ids=0,
        holdout_start_after_utc=normalized_holdout_cutoff,
    )


def _train_from_frame(
    df_labeled: pd.DataFrame,
    admitted: list[str],
    *,
    shrinkage: float,
    max_duration_hours: float,
) -> Optional[ProfileModel]:
    """Train a ProfileModel directly from a pre-labeled frame (fold-local).

    Mirrors the Phase 2 training path (shared median + z-score + pooled cov)
    without re-reading the xlsx.
    """
    if df_labeled.empty:
        return None
    if not admitted:
        return None
    df_w = df_labeled.loc[cast(pd.Series, df_labeled["_is_winner"]) == 1]
    df_l = df_labeled.loc[cast(pd.Series, df_labeled["_is_winner"]) == 0]
    min_samples = max(30, 3 * len(admitted))
    if len(df_w) < min_samples or len(df_l) < min_samples:
        return None

    Xw = cast(pd.DataFrame, df_w[admitted]).astype(float).to_numpy()
    Xl = cast(pd.DataFrame, df_l[admitted]).astype(float).to_numpy()
    X_all = np.vstack([Xw, Xl])
    col_medians = np.nanmedian(X_all, axis=0)
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)

    def _fill(X: np.ndarray) -> np.ndarray:
        X = X.copy()
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(col_medians, inds[1])
        return X

    Xw = _fill(Xw)
    Xl = _fill(Xl)
    X_all = np.vstack([Xw, Xl])
    feat_mean = np.mean(X_all, axis=0)
    feat_std = np.std(X_all, axis=0, ddof=0)
    feat_std = np.where(feat_std > 1e-12, feat_std, 1.0)
    Xw = (Xw - feat_mean) / feat_std
    Xl = (Xl - feat_mean) / feat_std

    mu_w = np.mean(Xw, axis=0)
    mu_l = np.mean(Xl, axis=0)
    n_w = max(1, Xw.shape[0])
    n_l = max(1, Xl.shape[0])
    Sw = np.cov(Xw, rowvar=False) if Xw.shape[0] > 1 else np.eye(Xw.shape[1])
    Sl = np.cov(Xl, rowvar=False) if Xl.shape[0] > 1 else np.eye(Xl.shape[1])
    S = ((n_w - 1) * Sw + (n_l - 1) * Sl) / max(1, (n_w + n_l - 2))
    shrink = float(shrinkage)
    S = (1.0 - shrink) * S + shrink * np.diag(np.diag(S))
    S = S + 1e-6 * np.eye(S.shape[0])
    inv = np.linalg.inv(S)
    prior = float(n_w / (n_w + n_l))
    return ProfileModel(
        features=list(admitted),
        winner_mu={f: float(mu_w[i]) for i, f in enumerate(admitted)},
        loser_mu={f: float(mu_l[i]) for i, f in enumerate(admitted)},
        inv_cov=inv.tolist(),
        prior_winner=prior,
        duration_band={"min_hours": 0.0, "max_hours": float(max_duration_hours)},
        feature_mean={f: float(feat_mean[i]) for i, f in enumerate(admitted)},
        feature_std={f: float(feat_std[i]) for i, f in enumerate(admitted)},
        feature_impute={f: float(col_medians[i]) for i, f in enumerate(admitted)},
    )


def _paired_auc_delta_confidence_interval(
    candidate: WalkForwardResult,
    incumbent: WalkForwardResult,
    *,
    seed: int = PAIRED_BOOTSTRAP_SEED,
    replicates: int = PAIRED_BOOTSTRAP_REPLICATES,
) -> tuple[float, float, float]:
    """Return paired OOF AUC delta and a stratified bootstrap interval."""
    if candidate.oof_strategy_ids != incumbent.oof_strategy_ids:
        raise ValueError("candidate/incumbent OOF strategy_id order differs")
    if candidate.oof_labels != incumbent.oof_labels:
        raise ValueError("candidate/incumbent OOF labels differ")
    labels = np.asarray(candidate.oof_labels, dtype=int)
    candidate_scores = np.asarray(candidate.oof_scores, dtype=float)
    incumbent_scores = np.asarray(incumbent.oof_scores, dtype=float)
    if (
        labels.size == 0
        or candidate_scores.size != labels.size
        or incumbent_scores.size != labels.size
        or not np.all(np.isfinite(candidate_scores))
        or not np.all(np.isfinite(incumbent_scores))
    ):
        return float("nan"), float("nan"), float("nan")
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if positive.size == 0 or negative.size == 0:
        return float("nan"), float("nan"), float("nan")
    candidate_auc = _auc(candidate_scores, labels)
    incumbent_auc = _auc(incumbent_scores, labels)
    delta = float(candidate_auc - incumbent_auc)
    rng = np.random.default_rng(seed)
    bootstrap_deltas: list[float] = []
    for _ in range(replicates):
        sample_indices = np.concatenate(
            [
                rng.choice(positive, size=positive.size, replace=True),
                rng.choice(negative, size=negative.size, replace=True),
            ]
        )
        sample_labels = labels[sample_indices]
        sample_delta = _auc(
            candidate_scores[sample_indices], sample_labels
        ) - _auc(incumbent_scores[sample_indices], sample_labels)
        if np.isfinite(sample_delta):
            bootstrap_deltas.append(float(sample_delta))
    if not bootstrap_deltas:
        return delta, float("nan"), float("nan")
    return (
        delta,
        float(np.quantile(bootstrap_deltas, 0.025)),
        float(np.quantile(bootstrap_deltas, 0.975)),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _payload_sha256(payload_obj: dict[str, Any]) -> str:
    payload = json.dumps(
        payload_obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _model_sha256(model: ProfileModel) -> str:
    return _payload_sha256(model.to_json())


def _walkforward_integrity_error(result: WalkForwardResult) -> Optional[str]:
    """Return a fail-closed reason when stored metrics lack reproducible evidence."""
    if result.n_folds != len(result.fold_auc) or result.n_folds != len(
        result.fold_ks
    ):
        return "fold_metric_lengths_disagree_with_n_folds"
    for field_name in (
        "fold_train_rows",
        "fold_test_rows",
        "fold_train_winners",
        "fold_test_winners",
        "fold_pnl_thresholds",
        "fold_test_start_utc",
        "fold_test_end_utc",
    ):
        if len(getattr(result, field_name)) != result.n_folds:
            return f"{field_name}_length_disagrees_with_n_folds"
    if not isinstance(result.source_sha256, str) or len(result.source_sha256) != 64:
        return "source_sha256_missing_or_invalid"
    try:
        int(result.source_sha256, 16)
    except ValueError:
        return "source_sha256_missing_or_invalid"
    if result.duplicate_strategy_ids != 0:
        return f"duplicate_strategy_ids={result.duplicate_strategy_ids}"
    if len(set(result.requested_features)) != len(result.requested_features):
        return "requested_features_contain_duplicates"
    if len(set(result.admitted_features)) != len(result.admitted_features):
        return "admitted_features_contain_duplicates"
    expected_coverage = (
        len(result.admitted_features) / len(result.requested_features)
        if result.requested_features
        else 0.0
    )
    if not np.isclose(result.feature_coverage, expected_coverage, atol=1e-12):
        return "feature_coverage_disagrees_with_feature_lists"

    row_count = len(result.oof_strategy_ids)
    if row_count == 0:
        return "oof_evidence_is_empty"
    if len(set(result.oof_strategy_ids)) != row_count:
        return "oof_strategy_ids_contain_duplicates"
    if not (
        len(result.oof_labels)
        == len(result.oof_scores)
        == len(result.oof_probabilities)
        == row_count
    ):
        return "oof_evidence_lengths_disagree"
    labels = np.asarray(result.oof_labels, dtype=int)
    scores = np.asarray(result.oof_scores, dtype=float)
    probabilities = np.asarray(result.oof_probabilities, dtype=float)
    if not set(labels.tolist()).issubset({0, 1}) or len(set(labels.tolist())) != 2:
        return "oof_labels_must_contain_both_binary_classes"
    if not np.all(np.isfinite(scores)):
        return "oof_scores_contain_non_finite_values"
    if not np.all(np.isfinite(probabilities)) or not np.all(
        (probabilities >= 0.0) & (probabilities <= 1.0)
    ):
        return "oof_probabilities_are_not_finite_unit_interval_values"

    finite_auc = np.asarray(
        [value for value in result.fold_auc if np.isfinite(value)], dtype=float
    )
    finite_ks = np.asarray(
        [value for value in result.fold_ks if np.isfinite(value)], dtype=float
    )
    expected_mean_auc = (
        float(np.mean(finite_auc)) if finite_auc.size else float("nan")
    )
    expected_mean_ks = float(np.mean(finite_ks)) if finite_ks.size else float("nan")
    expected_pass_rate = (
        float(np.mean(finite_auc >= AUC_FOLD_PASS_THRESHOLD))
        if finite_auc.size
        else 0.0
    )
    if not np.isclose(result.mean_auc, expected_mean_auc, atol=1e-12):
        return "mean_auc_disagrees_with_fold_auc"
    if not np.isclose(result.mean_ks, expected_mean_ks, atol=1e-12):
        return "mean_ks_disagrees_with_fold_ks"
    if not np.isclose(result.mean_pass_rate, expected_pass_rate, atol=1e-12):
        return "mean_pass_rate_disagrees_with_fold_auc"

    expected_auc = _auc(scores, labels)
    expected_ks = _ks_statistic(scores[labels == 1], scores[labels == 0])
    expected_brier = float(np.mean(np.square(probabilities - labels)))
    expected_ece = _expected_calibration_error(probabilities, labels)
    for metric_name, actual, expected in (
        ("pooled_oof_auc", result.pooled_oof_auc, expected_auc),
        ("pooled_oof_ks", result.pooled_oof_ks, expected_ks),
        ("pooled_oof_brier", result.pooled_oof_brier, expected_brier),
        ("pooled_oof_ece", result.pooled_oof_ece, expected_ece),
    ):
        if not np.isclose(actual, expected, atol=1e-12):
            return f"{metric_name}_disagrees_with_oof_evidence"

    if result.labeled_rows < row_count:
        return "labeled_rows_is_smaller_than_oof_rows"
    cutoff = (
        pd.to_datetime(result.holdout_start_after_utc, utc=True, errors="coerce")
        if result.holdout_start_after_utc is not None
        else None
    )
    if cutoff is not None and pd.isna(cutoff):
        return "holdout_start_after_utc_is_invalid"
    for raw_start, raw_end in zip(
        result.fold_test_start_utc, result.fold_test_end_utc
    ):
        start = pd.to_datetime(raw_start, utc=True, errors="coerce")
        end = pd.to_datetime(raw_end, utc=True, errors="coerce")
        if pd.isna(start) or pd.isna(end) or cast(pd.Timestamp, end) < cast(
            pd.Timestamp, start
        ):
            return "fold_test_time_range_is_invalid"
        if cutoff is not None and cast(pd.Timestamp, start) <= cast(
            pd.Timestamp, cutoff
        ):
            return "fold_test_start_does_not_follow_disclosed_holdout_cutoff"
    return None


# ── Promotion ────────────────────────────────────────────────────────────────


def promote_profile_version(
    candidate_model: ProfileModel,
    *,
    requested_features: list[str],
    wf_result: WalkForwardResult,
    incumbent_wf_result: Optional[WalkForwardResult] = None,
    candidate_pattern_profile: Optional[PatternProfile] = None,
    profile_dir: Optional[Path] = None,
    ts: Optional[datetime] = None,
    trial_logger: Optional[object] = None,
    trial_hyperparameters: Optional[dict] = None,
) -> PromotionDecision:
    """Write timestamped artifact + atomic current.json if gates pass.

    Gates (all must hold):
      - a feature-matched pattern profile is supplied for atomic bundle promotion
      - an incumbent walk-forward result is supplied when an incumbent artifact exists
      - at least `MIN_FINITE_FOLDS` AUC values are finite
      - finite AUC coverage is at least `FINITE_FOLD_COVERAGE_FLOOR`
      - `wf_result.mean_auc >= MEAN_AUC_FLOOR`
      - `wf_result.mean_pass_rate >= MEAN_PASS_RATE_FLOOR`
      - `wf_result.feature_coverage >= COVERAGE_FLOOR`
      - candidate's features are a subset of `requested_features` (no extras)
      - candidate's features equal `wf_result.admitted_features`

    Every attempt writes a hash-linked evaluation artifact.  Model/current
    artifacts are written only when the absolute gates pass and the paired
    95% AUC-delta interval is strictly above zero whenever an incumbent exists.
    """
    if profile_dir is None:
        profile_dir = Path(get_config().base_dir) / "data" / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    attempt_time = (ts or datetime.now(timezone.utc)).astimezone(timezone.utc)
    evaluation_filename = (
        f"profile_evaluation_{attempt_time.strftime('%Y%m%d_%H%M%S_%f')}.json"
    )

    requested_set = set(requested_features)
    candidate_set = set(candidate_model.features)
    dropped = sorted(requested_set - candidate_set)
    extras = sorted(candidate_set - requested_set)
    admitted_set = set(wf_result.admitted_features)
    finite_fold_count = sum(
        1 for value in wf_result.fold_auc if np.isfinite(value)
    )
    fold_slot_count = len(wf_result.fold_auc)
    finite_fold_coverage = (
        finite_fold_count / fold_slot_count if fold_slot_count else 0.0
    )
    paired_delta: Optional[float] = None
    paired_ci_low: Optional[float] = None
    paired_ci_high: Optional[float] = None
    candidate_integrity_error = _walkforward_integrity_error(wf_result)
    incumbent_integrity_error = (
        _walkforward_integrity_error(incumbent_wf_result)
        if incumbent_wf_result is not None
        else None
    )
    paired_alignment_error: Optional[str] = None
    if (
        incumbent_wf_result is not None
        and candidate_integrity_error is None
        and incumbent_integrity_error is None
    ):
        try:
            paired_delta, paired_ci_low, paired_ci_high = (
                _paired_auc_delta_confidence_interval(
                    wf_result,
                    incumbent_wf_result,
                )
            )
        except ValueError as exc:
            paired_alignment_error = str(exc)

    holdout_overlap_count = 0
    holdout_overlap_evidence: list[str] = []
    evaluation_history_error: Optional[str] = None
    if candidate_integrity_error is None:
        try:
            holdout_overlap_count, holdout_overlap_evidence = (
                _prior_holdout_overlap(profile_dir, wf_result.oof_strategy_ids)
            )
        except ValueError as exc:
            evaluation_history_error = str(exc)

    decision: PromotionDecision
    pending_current_obj: Optional[dict[str, Any]] = None
    incumbent_artifact_exists = any(
        (profile_dir / filename).exists()
        for filename in (
            "current.json",
            "profile_model.json",
            "pattern_profile.json",
        )
    )

    if candidate_pattern_profile is None:
        decision = PromotionDecision(
            promoted=False,
            reason="candidate_pattern_profile_required",
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    elif incumbent_artifact_exists and incumbent_wf_result is None:
        decision = PromotionDecision(
            promoted=False,
            reason="incumbent_walkforward_comparison_required",
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    elif (
        candidate_pattern_profile is not None
        and set(candidate_pattern_profile.features) != candidate_set
    ):
        decision = PromotionDecision(
            promoted=False,
            reason=(
                "pattern_model_features_disagree="
                f"{sorted(set(candidate_pattern_profile.features) ^ candidate_set)}"
            ),
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    elif extras:
        logger.warning(
            "Promotion refused: candidate carries features not in requested set: %s",
            extras,
        )
        decision = PromotionDecision(
            promoted=False,
            reason=f"candidate_features_not_in_requested={extras}",
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    elif candidate_set != admitted_set:
        logger.warning(
            "Promotion refused: candidate features %s != wf_result.admitted_features %s",
            sorted(candidate_set), sorted(admitted_set),
        )
        decision = PromotionDecision(
            promoted=False,
            reason=(
                f"candidate_features_disagree_with_walkforward="
                f"{sorted(candidate_set ^ admitted_set)}"
            ),
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    elif finite_fold_count < MIN_FINITE_FOLDS:
        decision = PromotionDecision(
            promoted=False,
            reason=(
                f"finite_fold_count={finite_fold_count} < "
                f"floor={MIN_FINITE_FOLDS}"
            ),
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    elif finite_fold_coverage < FINITE_FOLD_COVERAGE_FLOOR:
        decision = PromotionDecision(
            promoted=False,
            reason=(
                f"finite_fold_coverage={finite_fold_coverage:.3f} < "
                f"floor={FINITE_FOLD_COVERAGE_FLOOR:.2f}"
            ),
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    elif (
        not np.isfinite(wf_result.mean_auc)
        or wf_result.mean_auc < MEAN_AUC_FLOOR
    ):
        decision = PromotionDecision(
            promoted=False,
            reason=(
                f"mean_auc={wf_result.mean_auc:.3f} < "
                f"floor={MEAN_AUC_FLOOR:.2f}"
            ),
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    elif (
        not np.isfinite(wf_result.pooled_oof_auc)
        or wf_result.pooled_oof_auc < POOLED_OOF_AUC_FLOOR
    ):
        decision = PromotionDecision(
            promoted=False,
            reason=(
                f"pooled_oof_auc={wf_result.pooled_oof_auc:.3f} < "
                f"floor={POOLED_OOF_AUC_FLOOR:.2f}"
            ),
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    elif wf_result.mean_pass_rate < MEAN_PASS_RATE_FLOOR:
        decision = PromotionDecision(
            promoted=False,
            reason=(
                f"mean_pass_rate={wf_result.mean_pass_rate:.3f} < "
                f"floor={MEAN_PASS_RATE_FLOOR:.2f}"
            ),
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    elif wf_result.feature_coverage < COVERAGE_FLOOR:
        logger.warning(
            "Promotion refused by silent-drop guard: coverage=%.2f < %.2f. "
            "Dropped features: %s",
            wf_result.feature_coverage, COVERAGE_FLOOR, dropped,
        )
        decision = PromotionDecision(
            promoted=False,
            reason=(
                f"feature_coverage={wf_result.feature_coverage:.3f} < "
                f"floor={COVERAGE_FLOOR:.2f}"
            ),
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    elif candidate_integrity_error is not None:
        decision = PromotionDecision(
            promoted=False,
            reason=f"candidate_walkforward_integrity={candidate_integrity_error}",
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    elif incumbent_integrity_error is not None:
        decision = PromotionDecision(
            promoted=False,
            reason=f"incumbent_walkforward_integrity={incumbent_integrity_error}",
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    elif evaluation_history_error is not None:
        decision = PromotionDecision(
            promoted=False,
            reason=f"evaluation_history_invalid={evaluation_history_error}",
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    elif holdout_overlap_count > 0:
        decision = PromotionDecision(
            promoted=False,
            reason=(
                f"oof_holdout_reused={holdout_overlap_count}; "
                f"prior_evaluations={holdout_overlap_evidence}"
            ),
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    elif paired_alignment_error is not None:
        decision = PromotionDecision(
            promoted=False,
            reason=f"paired_oof_alignment_invalid={paired_alignment_error}",
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    elif (
        incumbent_wf_result is not None
        and (
            paired_ci_low is None
            or not np.isfinite(paired_ci_low)
            or paired_ci_low <= 0.0
        )
    ):
        decision = PromotionDecision(
            promoted=False,
            reason=(
                "paired_auc_delta_ci_low="
                f"{paired_ci_low if paired_ci_low is not None else 'unavailable'} "
                "must be > 0.0"
            ),
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
        )
    else:
        filename = make_profile_model_filename(attempt_time)
        artifact_path = profile_dir / filename
        # Atomic artifact write: stage to .tmp, then rename. Prevents a crash
        # between save and current.json update from leaving a half-written
        # artifact that a later run could observe.
        artifact_tmp = artifact_path.with_suffix(".json.tmp")
        save_profile_model(candidate_model, artifact_tmp)
        artifact_tmp.replace(artifact_path)
        pattern_filename: Optional[str] = None
        pattern_sha256: Optional[str] = None
        if candidate_pattern_profile is not None:
            pattern_filename = make_pattern_profile_filename(attempt_time)
            pattern_path = profile_dir / pattern_filename
            pattern_tmp = pattern_path.with_suffix(".json.tmp")
            candidate_pattern_profile.save_json(pattern_tmp)
            pattern_tmp.replace(pattern_path)
            pattern_sha256 = _hash_file(pattern_path)
        pending_current_obj = {
            "active": filename,
            "mean_pass_rate": wf_result.mean_pass_rate,
            "mean_auc": wf_result.mean_auc,
            "finite_fold_count": finite_fold_count,
            "finite_fold_coverage": finite_fold_coverage,
            "feature_coverage": wf_result.feature_coverage,
            "dropped_features": dropped,
            "sha256": _hash_file(artifact_path),
            "evaluation": f"evaluations/{evaluation_filename}",
            "promoted_utc": attempt_time.isoformat(),
        }
        if pattern_filename is not None and pattern_sha256 is not None:
            pending_current_obj["active_pattern_profile"] = pattern_filename
            pending_current_obj["pattern_profile_sha256"] = pattern_sha256
        decision = PromotionDecision(
            promoted=True,
            reason="gates_passed",
            mean_pass_rate=wf_result.mean_pass_rate,
            feature_coverage=wf_result.feature_coverage,
            dropped_features=dropped,
            artifact_filename=filename,
        )

    decision = replace(
        decision,
        evaluation_filename=evaluation_filename,
        paired_auc_delta=paired_delta,
        paired_auc_delta_ci_low=paired_ci_low,
        paired_auc_delta_ci_high=paired_ci_high,
    )
    evaluation_dir = profile_dir / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = evaluation_dir / evaluation_filename
    evaluation_obj = {
        "schema_version": 2,
        "attempted_utc": attempt_time.isoformat(),
        "gate_decision": "pass" if decision.promoted else "reject",
        "reason": decision.reason,
        "candidate": {
            "sha256": (
                str(pending_current_obj["sha256"])
                if pending_current_obj is not None
                else _model_sha256(candidate_model)
            ),
            "sha256_scope": (
                "artifact_file"
                if pending_current_obj is not None
                else "canonical_model_payload"
            ),
            "features": list(candidate_model.features),
            "selection_summary": candidate_model.selection_summary,
        },
        "candidate_pattern_profile": (
            {
                "sha256": (
                    str(pending_current_obj["pattern_profile_sha256"])
                    if pending_current_obj is not None
                    and "pattern_profile_sha256" in pending_current_obj
                    else _payload_sha256(candidate_pattern_profile.to_json())
                ),
                "sha256_scope": (
                    "artifact_file"
                    if pending_current_obj is not None
                    and "pattern_profile_sha256" in pending_current_obj
                    else "canonical_pattern_payload"
                ),
                "features": list(candidate_pattern_profile.features),
                "selection_summary": candidate_pattern_profile.selection_summary,
            }
            if candidate_pattern_profile is not None
            else None
        ),
        "walkforward": asdict(wf_result),
        "incumbent_walkforward": (
            asdict(incumbent_wf_result)
            if incumbent_wf_result is not None
            else None
        ),
        "paired_comparison": {
            "auc_delta": paired_delta,
            "auc_delta_ci_low": paired_ci_low,
            "auc_delta_ci_high": paired_ci_high,
            "bootstrap_replicates": PAIRED_BOOTSTRAP_REPLICATES,
            "bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
        },
        "evidence_integrity": {
            "candidate_error": candidate_integrity_error,
            "incumbent_error": incumbent_integrity_error,
            "paired_alignment_error": paired_alignment_error,
            "evaluation_history_error": evaluation_history_error,
            "prior_oof_overlap_count": holdout_overlap_count,
            "prior_oof_overlap_evaluations": holdout_overlap_evidence,
        },
        "gates": {
            "finite_fold_count": finite_fold_count,
            "finite_fold_count_floor": MIN_FINITE_FOLDS,
            "finite_fold_coverage": finite_fold_coverage,
            "finite_fold_coverage_floor": FINITE_FOLD_COVERAGE_FLOOR,
            "mean_auc_floor": MEAN_AUC_FLOOR,
            "pooled_oof_auc_floor": POOLED_OOF_AUC_FLOOR,
            "mean_pass_rate_floor": MEAN_PASS_RATE_FLOOR,
            "feature_coverage_floor": COVERAGE_FLOOR,
            "paired_auc_delta_ci_low_floor_exclusive": 0.0,
        },
        "hyperparameters": dict(trial_hyperparameters or {}),
    }
    evaluation_tmp = evaluation_path.with_suffix(".json.tmp")
    evaluation_tmp.write_text(
        json.dumps(_json_safe(evaluation_obj), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    evaluation_tmp.replace(evaluation_path)

    if pending_current_obj is not None:
        pending_current_obj["evaluation_sha256"] = _hash_file(evaluation_path)
        current_path = profile_dir / "current.json"
        current_tmp = current_path.with_suffix(".json.tmp")
        current_tmp.write_text(
            json.dumps(pending_current_obj, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        current_tmp.replace(current_path)
        logger.info(
            "Promoted profile model %s (mean_pass_rate=%.3f, coverage=%.3f)",
            decision.artifact_filename,
            wf_result.mean_pass_rate,
            wf_result.feature_coverage,
        )

    if trial_logger is not None and hasattr(trial_logger, "log_trial"):
        try:
            from neutralgrid.training.trial_tracker import TrialRecord

            record = TrialRecord(
                trial_id=f"profile_model_{(ts or datetime.now(timezone.utc)).strftime('%Y%m%d_%H%M%S')}",
                timestamp=ts or datetime.now(timezone.utc),
                model_type="profile_model",
                hyperparameters=dict(trial_hyperparameters or {}),
                cv_score=(
                    wf_result.mean_auc if np.isfinite(wf_result.mean_auc) else 0.0
                ),
                feature_set=list(candidate_model.features),
                notes=(
                    f"promoted={decision.promoted}; reason={decision.reason}; "
                    f"coverage={wf_result.feature_coverage:.3f}; "
                    f"pass_rate={wf_result.mean_pass_rate:.3f}"
                ),
            )
            trial_logger.log_trial(record)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("Trial logging failed (non-fatal): %s", exc)

    return decision


def train_and_promote(
    xlsx_path: str | Path,
    *,
    n_folds: int = 5,
    purge_hours: float = 7.0,
    max_duration_hours: float = 7.0,
    min_profit_factor: float = 1.5,
    min_avg_profit_per_grid: Optional[float] = None,
    top_quantile: float = 0.75,
    features: Iterable[str] = DEFAULT_FEATURES,
    shrinkage: float = 0.30,
    profile_dir: Optional[Path] = None,
    incumbent_wf_result: Optional[WalkForwardResult] = None,
    trial_logger: Optional[object] = None,
    ts: Optional[datetime] = None,
) -> PromotionDecision:
    """End-to-end: CV → train final → gate → (optionally) write artifact.

    Raises `ValueError` if the full-data training itself fails the Phase 2.4
    sample floor. Otherwise returns a `PromotionDecision` (promoted or not).
    """
    features_list = list(features)
    effective_profile_dir = (
        profile_dir
        if profile_dir is not None
        else Path(get_config().base_dir) / "data" / "profile"
    )
    holdout_start_after_utc = resolve_latest_evaluated_holdout_end(
        effective_profile_dir
    )
    wf = walkforward_evaluate(
        xlsx_path,
        n_folds=n_folds,
        purge_hours=purge_hours,
        max_duration_hours=max_duration_hours,
        min_profit_factor=min_profit_factor,
        min_avg_profit_per_grid=min_avg_profit_per_grid,
        top_quantile=top_quantile,
        features=features_list,
        shrinkage=shrinkage,
        holdout_start_after_utc=holdout_start_after_utc,
    )
    candidate = train_profile_model_from_enhanced_xlsx(
        xlsx_path,
        max_duration_hours=max_duration_hours,
        min_profit_factor=min_profit_factor,
        min_avg_profit_per_grid=min_avg_profit_per_grid,
        top_quantile=top_quantile,
        features=features_list,
        shrinkage=shrinkage,
    )
    candidate_pattern_profile = build_profile_from_enhanced_xlsx(
        xlsx_path,
        max_duration_hours=max_duration_hours,
        min_profit_factor=min_profit_factor,
        min_avg_profit_per_grid=min_avg_profit_per_grid,
        top_quantile=top_quantile,
        features=features_list,
    )
    return promote_profile_version(
        candidate,
        requested_features=features_list,
        wf_result=wf,
        incumbent_wf_result=incumbent_wf_result,
        candidate_pattern_profile=candidate_pattern_profile,
        profile_dir=effective_profile_dir,
        ts=ts,
        trial_logger=trial_logger,
        trial_hyperparameters={
            "n_folds": n_folds,
            "purge_hours": purge_hours,
            "max_duration_hours": max_duration_hours,
            "min_profit_factor": min_profit_factor,
            "top_quantile": top_quantile,
            "shrinkage": shrinkage,
            "holdout_start_after_utc": holdout_start_after_utc,
        },
    )
