"""Offline shadow analysis for live meta-labeler tilt calibration.

This module joins D7 live-decision JSONL records to finalized live outcomes and
computes the D5/D6 evidence needed before enabling ``meta_tilt_enabled``. It is
deliberately non-gating: it reports whether evidence is sufficient, but never
edits runtime config or flips the live tilt.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, cast

import numpy as np
import pandas as pd

from neutralgrid.training.live_outcome_ingestor import LiveOutcomeIngestor


DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(round(i / 100.0, 2) for i in range(5, 96, 5))


@dataclass(frozen=True)
class ShadowAnalysisConfig:
    """Controls fail-closed sample-size checks for D5/D6 shadow analysis."""

    outcome_positive_pnl_pct: float = 0.0
    min_joined_rows: int = 30
    min_fit_adjust_rows: int = 10
    min_fit_adjust_join_keys: int = 5
    min_oos_rows: int = 10
    min_oos_join_keys: int = 5
    min_oos_tilt_rows: int = 3
    min_oos_tilt_join_keys: int = 3
    fit_fraction: float = 0.60
    calibration_bins: int = 5
    thresholds: tuple[float, ...] = field(default_factory=lambda: DEFAULT_THRESHOLDS)


def load_decision_jsonl(paths: Sequence[Path]) -> pd.DataFrame:
    """Load D7 JSONL records into one row per live decision tick."""

    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
            rows.append(_decision_row(record, path=path, line_no=line_no))

    return pd.DataFrame(rows)


def load_live_outcomes(
    *,
    expired_bots_path: Path,
    linkage_dir: Path = Path("data/linkage"),
    scanner_results_dir: Path = Path("results"),
    min_bot_date: str = "2026-02-01",
) -> pd.DataFrame:
    """Load finalized outcomes through the canonical ingestor."""

    ingestor = LiveOutcomeIngestor(
        expired_bots_path=expired_bots_path,
        linkage_dir=linkage_dir,
        scanner_results_dir=scanner_results_dir,
        min_bot_date=min_bot_date,
    )
    return ingestor.ingest()


def join_shadow_to_outcomes(decisions: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Inner-join live shadow decisions to outcomes by exact live-bot identity.

    Preferred identity is ``candidate_id``. Legacy JSONL rows often predate D7
    candidate stamping, so rows without a candidate may fall back to the exact
    live ``strategy_id`` + ``symbol`` pair from the expired-bot outcome row. Rows
    that already claim a candidate never use the fallback; a candidate mismatch
    should stay visible rather than being hidden by strategy-only recovery.
    """

    if decisions.empty or outcomes.empty:
        return pd.DataFrame()

    decisions_norm = _normalise_join_identity(decisions)
    outcomes_norm = _normalise_join_identity(outcomes)
    _assert_unique_outcome_identities(outcomes_norm)
    joined_parts: list[pd.DataFrame] = []

    if "candidate_id" in decisions_norm.columns and "candidate_id" in outcomes_norm.columns:
        decision_candidates = cast(pd.Series, decisions_norm["candidate_id"]).str.strip() != ""
        outcome_candidates = cast(pd.Series, outcomes_norm["candidate_id"]).str.strip() != ""
        by_candidate = cast(
            pd.DataFrame,
            decisions_norm.loc[decision_candidates].merge(
                outcomes_norm.loc[outcome_candidates],
                on="candidate_id",
                how="inner",
                suffixes=("_decision", "_outcome"),
            ),
        )
        if not by_candidate.empty:
            by_candidate["join_method"] = "candidate_id"
            by_candidate["join_key"] = cast(pd.Series, by_candidate["candidate_id"]).astype(str)
            joined_parts.append(by_candidate)

    fallback_required = {"strategy_id", "symbol"}
    if fallback_required.issubset(decisions_norm.columns) and fallback_required.issubset(
        outcomes_norm.columns
    ):
        decision_blank_candidate = cast(pd.Series, decisions_norm.get("candidate_id", "")).astype(str).str.strip() == ""
        decision_strategy = cast(pd.Series, decisions_norm["strategy_id"]).str.strip() != ""
        decision_symbol = cast(pd.Series, decisions_norm["symbol"]).str.strip() != ""
        outcome_strategy = cast(pd.Series, outcomes_norm["strategy_id"]).str.strip() != ""
        outcome_symbol = cast(pd.Series, outcomes_norm["symbol"]).str.strip() != ""
        fallback_decisions = cast(
            pd.DataFrame,
            decisions_norm.loc[decision_blank_candidate & decision_strategy & decision_symbol],
        )
        fallback_outcomes = cast(
            pd.DataFrame,
            outcomes_norm.loc[outcome_strategy & outcome_symbol],
        )
        by_strategy = cast(
            pd.DataFrame,
            fallback_decisions.merge(
                fallback_outcomes,
                on=["strategy_id", "symbol"],
                how="inner",
                suffixes=("_decision", "_outcome"),
            ),
        )
        if not by_strategy.empty:
            by_strategy["join_method"] = "strategy_symbol"
            by_strategy["join_key"] = (
                cast(pd.Series, by_strategy["strategy_id"]).astype(str)
                + "|"
                + cast(pd.Series, by_strategy["symbol"]).astype(str)
            )
            if "candidate_id_outcome" in by_strategy.columns:
                by_strategy["candidate_id"] = cast(
                    pd.Series, by_strategy["candidate_id_outcome"]
                ).astype(str)
            else:
                by_strategy["candidate_id"] = ""
            joined_parts.append(by_strategy)

    if not joined_parts:
        return pd.DataFrame()
    return cast(pd.DataFrame, pd.concat(joined_parts, ignore_index=True, sort=False))


def _assert_unique_outcome_identities(outcomes: pd.DataFrame) -> None:
    """Reject outcome pools that would create an ambiguous many-to-many join."""

    candidate_mask = _nonempty_text_mask(outcomes, "candidate_id")
    candidate_ids = cast(pd.Series, outcomes.loc[candidate_mask, "candidate_id"])
    duplicate_candidates = sorted(
        set(candidate_ids.loc[candidate_ids.duplicated(keep=False)].astype(str))
    )
    if duplicate_candidates:
        raise ValueError(
            "finalized outcomes contain duplicate candidate_id values: "
            f"{duplicate_candidates[:5]}"
        )

    strategy_mask = _nonempty_text_mask(outcomes, "strategy_id") & _nonempty_text_mask(
        outcomes, "symbol"
    )
    strategy_rows = cast(
        pd.DataFrame,
        outcomes.loc[strategy_mask, ["strategy_id", "symbol"]].copy(),
    )
    duplicate_strategy = strategy_rows.duplicated(
        subset=["strategy_id", "symbol"],
        keep=False,
    )
    if bool(duplicate_strategy.any()):
        pairs = sorted(
            {
                f"{row.strategy_id}|{row.symbol}"
                for row in strategy_rows.loc[duplicate_strategy].itertuples(index=False)
            }
        )
        raise ValueError(
            "finalized outcomes contain duplicate strategy_id/symbol values: "
            f"{pairs[:5]}"
        )


def analyze_joined_shadow(
    joined: pd.DataFrame,
    *,
    config: ShadowAnalysisConfig = ShadowAnalysisConfig(),
) -> dict[str, Any]:
    """Compute D5/D6 metrics from joined shadow rows.

    The positive class for calibration drift is ``pnl_pct > outcome_positive_pnl_pct``.
    The positive class for low-confidence tilt selection is the complement: rows
    whose finalized PnL is at or below that threshold.
    """

    eligible = _prepare_eligible(joined, config=config)
    unique_candidates = 0
    if not eligible.empty and "candidate_id" in eligible.columns:
        candidate_ids = (
            cast(pd.Series, eligible["candidate_id"])
            .astype("string")
            .fillna("")
            .str.strip()
        )
        unique_candidates = int(
            candidate_ids.mask(candidate_ids.eq("")).dropna().nunique()
        )
    payload: dict[str, Any] = {
        "status": "insufficient_joined_rows",
        "config": _jsonable(asdict(config)),
        "counts": {
            "joined_rows": int(len(joined)),
            "eligible_rows": int(len(eligible)),
            "unique_join_keys": (
                int(cast(pd.Series, eligible["join_key"]).nunique())
                if not eligible.empty and "join_key" in eligible.columns
                else 0
            ),
            "unique_candidates": unique_candidates,
        },
        "d5_calibrated": False,
        "d6_oos_proven": False,
        "ready_to_enable": False,
        "recommended_meta_tilt_low_threshold": None,
        "eligibility_funnel": _eligibility_funnel(joined, config=config),
    }

    if len(eligible) < config.min_joined_rows:
        payload["reason"] = (
            f"eligible rows {len(eligible)} < min_joined_rows {config.min_joined_rows}"
        )
        return _finalize_payload(payload)

    eligible = cast(pd.DataFrame, eligible.sort_values("ts_utc").reset_index(drop=True))
    fit, oos = _split_fit_oos_by_join_key(eligible, fit_fraction=config.fit_fraction)
    fit_adjust = _adjust_rows(fit)

    payload["counts"].update(
        {
            "fit_rows": int(len(fit)),
            "fit_join_keys": _unique_join_key_count(fit),
            "fit_adjust_rows": int(len(fit_adjust)),
            "fit_adjust_join_keys": _unique_join_key_count(fit_adjust),
            "oos_rows": int(len(oos)),
            "oos_join_keys": _unique_join_key_count(oos),
        }
    )
    payload["split_audit"] = _split_audit(fit, oos)
    payload["calibration"] = _calibration_metrics(eligible, config=config)
    payload["decision_metrics_all_eligible"] = _decision_metrics(eligible)

    if len(fit_adjust) < config.min_fit_adjust_rows:
        payload["status"] = "insufficient_fit_adjust_rows"
        payload["reason"] = (
            f"fit ADJUST rows {len(fit_adjust)} < min_fit_adjust_rows {config.min_fit_adjust_rows}"
        )
        return _finalize_payload(payload)
    fit_adjust_join_keys = _unique_join_key_count(fit_adjust)
    if fit_adjust_join_keys < config.min_fit_adjust_join_keys:
        payload["status"] = "insufficient_fit_adjust_join_keys"
        payload["reason"] = (
            f"fit ADJUST join keys {fit_adjust_join_keys} "
            f"< min_fit_adjust_join_keys {config.min_fit_adjust_join_keys}"
        )
        return _finalize_payload(payload)
    if len(oos) < config.min_oos_rows:
        payload["status"] = "insufficient_oos_rows"
        payload["reason"] = f"OOS rows {len(oos)} < min_oos_rows {config.min_oos_rows}"
        return _finalize_payload(payload)
    oos_join_keys = _unique_join_key_count(oos)
    if oos_join_keys < config.min_oos_join_keys:
        payload["status"] = "insufficient_oos_join_keys"
        payload["reason"] = (
            f"OOS join keys {oos_join_keys} < min_oos_join_keys {config.min_oos_join_keys}"
        )
        return _finalize_payload(payload)
    if not _has_both_classes(fit_adjust):
        payload["status"] = "insufficient_fit_class_variation"
        payload["reason"] = "fit ADJUST rows do not contain both outcome classes"
        return _finalize_payload(payload)

    sweep = [_threshold_metrics(fit_adjust, threshold) for threshold in config.thresholds]
    best = _select_threshold(sweep)
    if best is None:
        payload["status"] = "no_threshold_candidate"
        payload["reason"] = "no threshold predicted any low-confidence ADJUST rows"
        payload["threshold_sweep_fit"] = sweep
        return _finalize_payload(payload)

    threshold = float(best["threshold"])
    payload["status"] = "calibrated_pending_oos"
    payload["d5_calibrated"] = True
    payload["recommended_meta_tilt_low_threshold"] = threshold
    payload["threshold_sweep_fit"] = sweep
    payload["selected_threshold_fit"] = best

    oos_adjust = _adjust_rows(oos)
    oos_baseline = _binary_metrics(
        _bool_series(oos, "is_adjust"),
        _bool_series(oos, "bad_outcome"),
    )
    oos_tilt = _threshold_metrics(oos_adjust, threshold)
    payload["oos_baseline_adjust_metrics"] = oos_baseline
    payload["oos_meta_low_confidence_metrics"] = oos_tilt

    if int(oos_tilt["predicted_positive"]) < config.min_oos_tilt_rows:
        payload["status"] = "insufficient_oos_tilt_support"
        payload["reason"] = (
            f"OOS low-confidence ADJUST support {oos_tilt['predicted_positive']} "
            f"< min_oos_tilt_rows {config.min_oos_tilt_rows}"
        )
        return _finalize_payload(payload)
    if int(oos_tilt.get("unique_predicted_join_keys", 0)) < config.min_oos_tilt_join_keys:
        payload["status"] = "insufficient_oos_tilt_join_keys"
        payload["reason"] = (
            f"OOS low-confidence ADJUST join-key support "
            f"{oos_tilt.get('unique_predicted_join_keys', 0)} "
            f"< min_oos_tilt_join_keys {config.min_oos_tilt_join_keys}"
        )
        return _finalize_payload(payload)

    precision_lift = _safe_float(oos_tilt["precision"]) - _safe_float(oos_baseline["precision"])
    payload["oos_precision_lift_vs_all_adjust"] = precision_lift
    if precision_lift > 0.0:
        payload["status"] = "oos_lift_observed"
        payload["d6_oos_proven"] = True
        payload["ready_to_enable"] = True
    else:
        payload["status"] = "oos_no_lift"
        payload["reason"] = "low-confidence ADJUST subset did not improve precision OOS"
    return _finalize_payload(payload)


def analyze_from_files(
    *,
    decision_paths: Sequence[Path],
    expired_bots_path: Path,
    linkage_dir: Path = Path("data/linkage"),
    scanner_results_dir: Path = Path("results"),
    min_bot_date: str = "2026-02-01",
    config: ShadowAnalysisConfig = ShadowAnalysisConfig(),
) -> dict[str, Any]:
    """Run the full D5/D6 shadow analysis from files."""

    decisions = load_decision_jsonl(decision_paths)
    outcomes = load_live_outcomes(
        expired_bots_path=expired_bots_path,
        linkage_dir=linkage_dir,
        scanner_results_dir=scanner_results_dir,
        min_bot_date=min_bot_date,
    )
    joined = join_shadow_to_outcomes(decisions, outcomes)
    result = analyze_joined_shadow(joined, config=config)
    result["input_counts"] = {
        "decision_rows": int(len(decisions)),
        "outcome_rows": int(len(outcomes)),
        "joined_rows": int(len(joined)),
    }
    result["join_coverage"] = _join_coverage(decisions, outcomes, joined)
    result["pending_authoritative_outcomes"] = _pending_authoritative_outcomes(
        decisions, joined
    )
    return result


def _normalise_join_identity(df: pd.DataFrame) -> pd.DataFrame:
    normalised = df.copy()
    for col in ("candidate_id", "strategy_id", "symbol"):
        if col not in normalised.columns:
            normalised[col] = ""
        normalised[col] = cast(pd.Series, normalised[col]).fillna("").astype(str).str.strip()
    return normalised


def _nonempty_text_mask(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return cast(pd.Series, df[col]).fillna("").astype(str).str.strip() != ""


def _join_coverage(
    decisions: pd.DataFrame,
    outcomes: pd.DataFrame,
    joined: pd.DataFrame,
) -> dict[str, Any]:
    joined_methods: dict[str, int] = {}
    if not joined.empty and "join_method" in joined.columns:
        counts = cast(pd.Series, joined["join_method"]).fillna("").astype(str).value_counts()
        joined_methods = {str(key): int(value) for key, value in counts.items()}
    decision_full_fidelity = (
        int(
            (
                cast(pd.Series, pd.to_numeric(decisions["meta_proba"], errors="coerce")).notna()
                & _bool_series(decisions, "meta_authoritative")
                & _bool_series(decisions, "meta_full_fidelity")
            ).sum()
        )
        if not decisions.empty and "meta_proba" in decisions.columns
        else 0
    )
    return {
        "decision_rows_with_candidate_id": int(_nonempty_text_mask(decisions, "candidate_id").sum()),
        "decision_rows_with_strategy_id": int(_nonempty_text_mask(decisions, "strategy_id").sum()),
        "decision_rows_with_meta_proba": int(
            cast(pd.Series, pd.to_numeric(decisions["meta_proba"], errors="coerce")).notna().sum()
        )
        if not decisions.empty and "meta_proba" in decisions.columns
        else 0,
        "decision_rows_authoritative_full_fidelity": decision_full_fidelity,
        "outcome_rows_with_candidate_id": int(_nonempty_text_mask(outcomes, "candidate_id").sum()),
        "outcome_rows_with_strategy_id": int(_nonempty_text_mask(outcomes, "strategy_id").sum()),
        "joined_rows_by_method": joined_methods,
    }


def _pending_authoritative_outcomes(
    decisions: pd.DataFrame,
    joined: pd.DataFrame,
) -> dict[str, Any]:
    if decisions.empty:
        return {
            "authoritative_full_fidelity_rows": 0,
            "matched_rows": 0,
            "pending_rows": 0,
            "pending_unique_join_keys": 0,
            "pending_by_join_key": [],
        }

    df = _decision_identity_frame(decisions)
    meta_proba = _numeric_series(df, "meta_proba")
    eligible_meta_mask = (
        meta_proba.notna()
        & _bool_series(df, "meta_authoritative")
        & _bool_series(df, "meta_full_fidelity")
    )
    authoritative = cast(pd.DataFrame, df.loc[eligible_meta_mask].copy())
    if authoritative.empty:
        return {
            "authoritative_full_fidelity_rows": 0,
            "matched_rows": 0,
            "pending_rows": 0,
            "pending_unique_join_keys": 0,
            "pending_by_join_key": [],
        }

    matched_refs: set[tuple[str, int]] = set()
    if not joined.empty and {"source_path", "source_line"}.issubset(joined.columns):
        for source_path, source_line in zip(
            cast(pd.Series, joined["source_path"]).astype(str),
            cast(pd.Series, joined["source_line"]).astype(int),
        ):
            matched_refs.add((str(source_path), int(source_line)))

    authoritative["_source_ref"] = list(
        zip(
            cast(pd.Series, authoritative["source_path"]).astype(str),
            cast(pd.Series, authoritative["source_line"]).astype(int),
        )
    )
    matched_mask = cast(pd.Series, authoritative["_source_ref"]).map(
        lambda ref: ref in matched_refs
    )
    pending = cast(pd.DataFrame, authoritative.loc[~matched_mask].copy())
    return {
        "authoritative_full_fidelity_rows": int(len(authoritative)),
        "matched_rows": int(matched_mask.sum()),
        "pending_rows": int(len(pending)),
        "pending_unique_join_keys": _unique_join_key_count(pending),
        "pending_by_join_key": _pending_by_join_key(pending),
    }


def _decision_identity_frame(decisions: pd.DataFrame) -> pd.DataFrame:
    df = _normalise_join_identity(decisions)
    candidate_id = cast(pd.Series, df["candidate_id"]).fillna("").astype(str).str.strip()
    strategy_id = cast(pd.Series, df["strategy_id"]).fillna("").astype(str).str.strip()
    symbol = cast(pd.Series, df["symbol"]).fillna("").astype(str).str.strip()
    strategy_key = strategy_id + "|" + symbol
    has_strategy_key = (strategy_id != "") & (symbol != "")
    df["join_method"] = np.where(candidate_id != "", "candidate_id", "strategy_symbol")
    df["join_key"] = np.where(candidate_id != "", candidate_id, np.where(has_strategy_key, strategy_key, ""))
    return df


def _pending_by_join_key(pending: pd.DataFrame) -> list[dict[str, Any]]:
    if pending.empty or "join_key" not in pending.columns:
        return []
    rows: list[dict[str, Any]] = []
    grouped = pending.groupby("join_key", dropna=False, sort=True)
    for join_key, group in grouped:
        group_df = cast(pd.DataFrame, group)
        if not str(join_key).strip():
            continue
        rows.append(
            {
                "join_key": str(join_key),
                "join_method": _first_text(group_df, "join_method"),
                "symbol": _first_text(group_df, "symbol"),
                "strategy_id": _first_text(group_df, "strategy_id"),
                "candidate_id": _first_text(group_df, "candidate_id"),
                "decision_rows": int(len(group_df)),
                "first_ts_utc": _first_timestamp(group_df),
            }
        )
    return rows


def _first_text(df: pd.DataFrame, col: str) -> str:
    if df.empty or col not in df.columns:
        return ""
    series = cast(pd.Series, df[col]).fillna("").astype(str).str.strip()
    for value in series.tolist():
        if value:
            return str(value)
    return ""


def _first_timestamp(df: pd.DataFrame) -> Optional[str]:
    if df.empty or "ts_utc" not in df.columns:
        return None
    timestamps = pd.to_datetime(cast(pd.Series, df["ts_utc"]), utc=True, errors="coerce")
    timestamps = cast(pd.Series, timestamps.dropna().sort_values())
    if timestamps.empty:
        return None
    first = timestamps.iloc[0]
    if isinstance(first, pd.Timestamp):
        return first.isoformat()
    return str(first)


def _eligibility_funnel(joined: pd.DataFrame, *, config: ShadowAnalysisConfig) -> dict[str, Any]:
    if joined.empty:
        return {
            "joined_rows": 0,
            "with_join_key": 0,
            "with_meta_proba": 0,
            "with_pnl_pct": 0,
            "with_ts_utc": 0,
            "meta_authoritative_true": 0,
            "meta_full_fidelity_true": 0,
            "with_authoritative_full_fidelity_meta": 0,
            "all_eligible": 0,
            "failed_reason_counts": {},
        }

    df = joined.copy()
    if "join_key" not in df.columns:
        if "candidate_id" in df.columns:
            df["join_key"] = cast(pd.Series, df["candidate_id"]).fillna("").astype(str)
        else:
            df["join_key"] = ""

    join_key = _nonempty_text_mask(df, "join_key")
    meta_proba = _numeric_series(df, "meta_proba")
    pnl_pct = _numeric_series(df, "pnl_pct")
    ts_utc = _datetime_series(df, "ts_utc")
    meta_authoritative = _bool_series(df, "meta_authoritative")
    meta_full_fidelity = _bool_series(df, "meta_full_fidelity")
    post_outcome, pre_deploy, window_missing = _life_window_outside_mask(df)
    authoritative_full_fidelity_meta = (
        meta_proba.notna() & meta_authoritative & meta_full_fidelity
    )
    eligible = (
        join_key
        & meta_proba.notna()
        & pnl_pct.notna()
        & ts_utc.notna()
        & meta_authoritative
        & meta_full_fidelity
        & ~post_outcome
        & ~pre_deploy
    )
    return {
        "joined_rows": int(len(df)),
        "with_join_key": int(join_key.sum()),
        "with_meta_proba": int(meta_proba.notna().sum()),
        "with_pnl_pct": int(pnl_pct.notna().sum()),
        "with_ts_utc": int(ts_utc.notna().sum()),
        "meta_authoritative_true": int(meta_authoritative.sum()),
        "meta_full_fidelity_true": int(meta_full_fidelity.sum()),
        "with_authoritative_full_fidelity_meta": int(authoritative_full_fidelity_meta.sum()),
        "within_bot_life_window": int((~post_outcome & ~pre_deploy).sum()),
        "outcome_life_window_missing": int(window_missing.sum()),
        "all_eligible": int(eligible.sum()),
        "failed_reason_counts": {
            "missing_join_key": int((~join_key).sum()),
            "missing_meta_proba": int(meta_proba.isna().sum()),
            "missing_pnl_pct": int(pnl_pct.isna().sum()),
            "missing_ts_utc": int(ts_utc.isna().sum()),
            "not_authoritative": int((~meta_authoritative).sum()),
            "not_full_fidelity": int((~meta_full_fidelity).sum()),
            "post_outcome_tick": int(post_outcome.sum()),
            "pre_deploy_tick": int(pre_deploy.sum()),
        },
        "thresholds": {
            "outcome_positive_pnl_pct": float(config.outcome_positive_pnl_pct),
        },
    }


def _finalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    ready = bool(payload.get("ready_to_enable"))
    threshold = payload.get("recommended_meta_tilt_low_threshold")
    metric_availability = {
        "false_end_missed_end_metrics": "decision_metrics_all_eligible" in payload,
        "adjust_precision_recall": "decision_metrics_all_eligible" in payload,
        "calibration_drift": "calibration" in payload,
        "threshold_sweep": "threshold_sweep_fit" in payload,
        "oos_precision_lift": "oos_precision_lift_vs_all_adjust" in payload,
    }
    missing_before_enable = []
    if not bool(payload.get("d5_calibrated")):
        missing_before_enable.append("D5 calibrated meta_tilt_low_threshold")
    if not bool(payload.get("d6_oos_proven")):
        missing_before_enable.append("D6 OOS accuracy improvement proof")
    payload["metric_availability"] = metric_availability
    payload["gate_audit"] = {
        "d5_threshold_calibrated": bool(payload.get("d5_calibrated")),
        "d6_oos_proven": bool(payload.get("d6_oos_proven")),
        "ready_to_enable": ready,
        "config_action": (
            "operator_may_enable_meta_tilt_after_review"
            if ready
            else "keep_meta_tilt_enabled_false"
        ),
        "missing_before_enable": missing_before_enable,
    }
    payload["recommended_live_config"] = (
        {
            "meta_tilt_enabled": True,
            "meta_tilt_low_threshold": float(threshold),
        }
        if ready and threshold is not None
        else {"meta_tilt_enabled": False}
    )
    return payload


def _split_fit_oos_by_join_key(
    eligible: pd.DataFrame,
    *,
    fit_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if eligible.empty or "join_key" not in eligible.columns:
        return eligible.copy(), pd.DataFrame()

    df = cast(pd.DataFrame, eligible.sort_values(["ts_utc", "join_key"]).reset_index(drop=True))
    group_times = cast(
        pd.DataFrame,
        df.groupby("join_key", dropna=False, as_index=False)["ts_utc"].min(),
    )
    group_times = cast(
        pd.DataFrame,
        group_times.sort_values(["ts_utc", "join_key"]).reset_index(drop=True),
    )
    split_idx = _fit_split_index(len(group_times), fit_fraction)
    fit_keys = set(
        cast(pd.Series, group_times.iloc[:split_idx]["join_key"]).fillna("").astype(str)
    )
    join_keys = cast(pd.Series, df["join_key"]).fillna("").astype(str)
    fit = cast(pd.DataFrame, df.loc[join_keys.isin(fit_keys)].copy())
    oos = cast(pd.DataFrame, df.loc[~join_keys.isin(fit_keys)].copy())
    return fit, oos


def _split_audit(fit: pd.DataFrame, oos: pd.DataFrame) -> dict[str, Any]:
    fit_keys = _join_key_set(fit)
    oos_keys = _join_key_set(oos)
    overlap = sorted(fit_keys & oos_keys)
    return {
        "split_strategy": "temporal_join_key",
        "fit_join_keys": len(fit_keys),
        "oos_join_keys": len(oos_keys),
        "overlap_join_keys": overlap,
        "overlap_join_key_count": len(overlap),
    }


def _join_key_set(df: pd.DataFrame) -> set[str]:
    if df.empty or "join_key" not in df.columns:
        return set()
    series = cast(pd.Series, df["join_key"]).fillna("").astype(str).str.strip()
    return {value for value in series.tolist() if value}


def _unique_join_key_count(df: pd.DataFrame) -> int:
    return len(_join_key_set(df))


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return cast(pd.Series, pd.to_numeric(df[col], errors="coerce"))


def _datetime_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    return cast(pd.Series, pd.to_datetime(cast(pd.Series, df[col]), utc=True, errors="coerce"))


def _decision_row(record: dict[str, Any], *, path: Path, line_no: int) -> dict[str, Any]:
    evaluation_raw = record.get("evaluation")
    evaluation = cast(dict[str, Any], evaluation_raw) if isinstance(evaluation_raw, dict) else {}
    deploy_snapshot_raw = evaluation.get("deploy_snapshot")
    deploy_snapshot = (
        cast(dict[str, Any], deploy_snapshot_raw)
        if isinstance(deploy_snapshot_raw, dict)
        else {}
    )
    execution_risk_raw = evaluation.get("execution_risk")
    execution_risk = (
        cast(dict[str, Any], execution_risk_raw)
        if isinstance(execution_risk_raw, dict)
        else {}
    )
    l2_risk_raw = evaluation.get("l2_risk")
    l2_risk = (
        cast(dict[str, Any], l2_risk_raw)
        if isinstance(l2_risk_raw, dict)
        else {}
    )
    private_raw = evaluation.get("private_event_evidence")
    private_events = (
        cast(dict[str, Any], private_raw)
        if isinstance(private_raw, dict)
        else {}
    )
    profit_raw = record.get("profit_deterioration")
    profit = (
        cast(dict[str, Any], profit_raw)
        if isinstance(profit_raw, dict)
        else {}
    )
    candidate_id = _clean_text(record.get("candidate_id"))
    if not candidate_id:
        candidate_id = _clean_text(deploy_snapshot.get("candidate_id"))

    return {
        "source_path": str(path),
        "source_line": int(line_no),
        "ts_utc": record.get("ts"),
        "symbol": record.get("symbol"),
        "strategy_id": _clean_text(record.get("strategy_id")),
        "candidate_id": candidate_id,
        "verdict": _clean_text(record.get("verdict")).upper(),
        # Contract v1.1: carried through so D5/D6 calibration pools can be
        # segmented by contract era and reason class — post-v1.1 watch-ADJUST
        # ticks (which were ENDs under v1.0) would otherwise silently shift
        # the ADJUST-population calibration.
        "contract_version": record.get("contract_version"),
        "reasons": record.get("reasons"),
        "meta_proba": evaluation.get("meta_proba"),
        "meta_authoritative": evaluation.get("meta_authoritative"),
        "meta_full_fidelity": evaluation.get("meta_full_fidelity"),
        "meta_would_tilt": record.get("meta_would_tilt"),
        "meta_influenced_verdict": record.get("meta_influenced_verdict"),
        # Verdict-inert execution evidence is flattened for exact finalized-
        # outcome linkage. Missing fields remain missing; no value is imputed.
        "execution_risk_source": execution_risk.get("source"),
        "l2_run_id": execution_risk.get("l2_run_id"),
        "l2_segment_id": execution_risk.get("l2_segment_id"),
        "liquidity_state": execution_risk.get("liquidity_state"),
        "sustained_joint_deterioration": execution_risk.get(
            "sustained_joint_deterioration"
        ),
        "sustained_spread_deterioration": execution_risk.get(
            "sustained_spread_deterioration"
        ),
        "sustained_exit_depth_deterioration": execution_risk.get(
            "sustained_exit_depth_deterioration"
        ),
        "temporary_joint_deterioration": execution_risk.get(
            "temporary_joint_deterioration"
        ),
        "joint_deterioration_trailing_duration_seconds": execution_risk.get(
            "joint_deterioration_trailing_duration_seconds"
        ),
        "current_spread_bps": execution_risk.get("current_spread_bps"),
        "baseline_spread_median_bps": execution_risk.get(
            "baseline_spread_median_bps"
        ),
        "exit_depth_current_to_baseline": execution_risk.get(
            "exit_depth_current_to_baseline"
        ),
        "exit_side_imbalance": execution_risk.get("exit_side_imbalance"),
        "exit_side_net_withdrawal_to_position_ratio": execution_risk.get(
            "exit_side_net_withdrawal_to_position_ratio"
        ),
        "unexplained_removal_to_position_ratio": execution_risk.get(
            "unexplained_removal_to_position_ratio"
        ),
        "refill_to_position_ratio": execution_risk.get(
            "refill_to_position_ratio"
        ),
        "private_cancel_update_fraction": execution_risk.get(
            "private_cancel_update_fraction"
        ),
        "public_trade_status": execution_risk.get("public_trade_status"),
        "private_event_status": execution_risk.get("private_event_status"),
        "mean_estimated_slippage_bps": execution_risk.get(
            "mean_estimated_slippage_bps"
        ),
        "mean_adverse_selection_5s_bps": execution_risk.get(
            "mean_adverse_selection_5s_bps"
        ),
        "mean_adverse_selection_30s_bps": execution_risk.get(
            "mean_adverse_selection_30s_bps"
        ),
        "expected_exit_impact_bps": l2_risk.get("expected_exit_impact_bps"),
        "exit_depth_to_position_ratio": l2_risk.get(
            "exit_depth_to_position_ratio"
        ),
        "private_event_completeness": private_events.get("event_completeness"),
        "private_event_run_id": private_events.get("run_id"),
        "private_realized_pnl_usdt": private_events.get("realized_pnl_usdt"),
        "private_commission_usdt": private_events.get("commission_usdt"),
        "private_funding_fee_usdt": private_events.get("funding_fee_usdt"),
        "current_total_profit_usdt": profit.get("current_total_profit_usdt"),
        "peak_total_profit_usdt": profit.get("peak_total_profit_usdt"),
        "gain_giveback_usdt": profit.get("giveback_usdt"),
        "gain_giveback_pct": profit.get("giveback_pct_of_positive_peak"),
    }


def _life_window_outside_mask(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Masks for ticks provably outside the joined outcome's bot life window.

    A decision tick timestamped AFTER the bot's finalized ``end_time_utc`` (or
    before ``start_time_utc``) never had a live bot behind it — e.g. the
    ENAUSDT ghost ticks of 2026-06-21/07-03 recorded days after the bot was
    canceled on 2026-06-18. Joining such ticks to the finalized outcome would
    calibrate D5/D6 on meta probabilities computed against post-mortem market
    data. Rows whose outcome lacks a parseable window are NOT excluded (the
    join itself is still valid) but are surfaced separately for visibility.

    Returns ``(post_outcome, pre_deploy, window_missing)``.
    """
    ts = pd.to_datetime(cast(pd.Series, df.get("ts_utc")), utc=True, errors="coerce")
    idx = df.index

    def _col(*names: str) -> pd.Series:
        for name in names:
            if name in df.columns:
                return pd.to_datetime(
                    cast(pd.Series, df[name]), utc=True, errors="coerce"
                )
        return pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, UTC]")

    end = _col("end_time_utc", "end_time_utc_outcome")
    start = _col("start_time_utc", "start_time_utc_outcome")
    post_outcome = ts.notna() & end.notna() & (ts > end)
    pre_deploy = ts.notna() & start.notna() & (ts < start)
    window_missing = end.isna() & start.isna()
    return post_outcome, pre_deploy, window_missing


def _prepare_eligible(joined: pd.DataFrame, *, config: ShadowAnalysisConfig) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()
    df = joined.copy()
    if "join_key" not in df.columns:
        if "candidate_id" in df.columns:
            df["join_key"] = cast(pd.Series, df["candidate_id"]).fillna("").astype(str)
        else:
            df["join_key"] = ""
    required = {"join_key", "meta_proba", "meta_authoritative", "meta_full_fidelity", "pnl_pct"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df["join_key"] = cast(pd.Series, df["join_key"]).fillna("").astype(str)
    if "candidate_id" not in df.columns:
        df["candidate_id"] = ""
    df["candidate_id"] = cast(pd.Series, df["candidate_id"]).fillna("").astype(str)
    df["meta_proba"] = pd.to_numeric(df["meta_proba"], errors="coerce")
    df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
    df["ts_utc"] = pd.to_datetime(cast(pd.Series, df["ts_utc"]), utc=True, errors="coerce")
    df["meta_authoritative"] = _bool_series(df, "meta_authoritative")
    df["meta_full_fidelity"] = _bool_series(df, "meta_full_fidelity")
    df["outcome_positive"] = cast(pd.Series, df["pnl_pct"]) > config.outcome_positive_pnl_pct
    df["bad_outcome"] = ~cast(pd.Series, df["outcome_positive"])
    df["is_end"] = cast(pd.Series, df["verdict"]).fillna("").astype(str).str.upper() == "END"
    df["is_adjust"] = cast(pd.Series, df["verdict"]).fillna("").astype(str).str.upper() == "ADJUST"
    post_outcome, pre_deploy, _window_missing = _life_window_outside_mask(df)
    mask = (
        (cast(pd.Series, df["join_key"]).str.strip() != "")
        & cast(pd.Series, df["meta_proba"]).notna()
        & cast(pd.Series, df["pnl_pct"]).notna()
        & cast(pd.Series, df["ts_utc"]).notna()
        & _bool_series(df, "meta_authoritative")
        & _bool_series(df, "meta_full_fidelity")
        & ~post_outcome
        & ~pre_deploy
    )
    return cast(pd.DataFrame, df.loc[mask].copy())


def _decision_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    actual_bad = _bool_series(df, "bad_outcome")
    is_end = _bool_series(df, "is_end")
    is_adjust = _bool_series(df, "is_adjust")
    outcome_positive = _bool_series(df, "outcome_positive")
    return {
        "false_end_count": int((is_end & outcome_positive).sum()),
        "missed_end_count": int((~is_end & actual_bad).sum()),
        "adjust_metrics_for_bad_outcome": _binary_metrics(is_adjust, actual_bad),
    }


def _calibration_metrics(df: pd.DataFrame, *, config: ShadowAnalysisConfig) -> dict[str, Any]:
    if df.empty:
        return {}
    proba = cast(pd.Series, df["meta_proba"]).astype(float)
    actual = _bool_series(df, "outcome_positive").astype(float)
    brier = float(((proba - actual) ** 2).mean())
    bins: list[dict[str, Any]] = []
    ece = 0.0
    n = max(len(df), 1)
    for idx in range(config.calibration_bins):
        lower = idx / config.calibration_bins
        upper = (idx + 1) / config.calibration_bins
        if idx == config.calibration_bins - 1:
            mask = (proba >= lower) & (proba <= upper)
        else:
            mask = (proba >= lower) & (proba < upper)
        count = int(mask.sum())
        if count == 0:
            bins.append({"lower": lower, "upper": upper, "count": 0})
            continue
        avg_pred = float(proba.loc[mask].mean())
        observed = float(actual.loc[mask].mean())
        abs_gap = abs(avg_pred - observed)
        ece += (count / n) * abs_gap
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "avg_pred": avg_pred,
                "observed_positive_rate": observed,
                "abs_gap": abs_gap,
            }
        )
    return {"brier_score": brier, "ece": float(ece), "bins": bins}


def _threshold_metrics(df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    if df.empty:
        metrics = _binary_metrics(pd.Series(dtype=bool), pd.Series(dtype=bool))
        metrics["unique_join_keys"] = 0
        metrics["unique_predicted_join_keys"] = 0
    else:
        proba = cast(pd.Series, df["meta_proba"]).astype(float)
        pred = proba < threshold
        actual = _bool_series(df, "bad_outcome")
        metrics = _binary_metrics(pred, actual)
        predicted_rows = cast(pd.DataFrame, df.loc[pred].copy())
        metrics["unique_join_keys"] = _unique_join_key_count(df)
        metrics["unique_predicted_join_keys"] = _unique_join_key_count(predicted_rows)
    metrics["threshold"] = float(threshold)
    return metrics


def _select_threshold(sweep: Sequence[dict[str, Any]]) -> Optional[dict[str, Any]]:
    candidates = [m for m in sweep if int(m.get("predicted_positive", 0)) > 0]
    if not candidates:
        return None
    return dict(
        max(
            candidates,
            key=lambda m: (
                _safe_float(m.get("f1")),
                _safe_float(m.get("precision")),
                -_safe_float(m.get("threshold")),
            ),
        )
    )


def _binary_metrics(pred: pd.Series, actual: pd.Series) -> dict[str, Any]:
    pred = pred.fillna(False).astype(bool)
    actual = actual.fillna(False).astype(bool)
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_positive": int(pred.sum()),
        "actual_positive": int(actual.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _fit_split_index(n_rows: int, fit_fraction: float) -> int:
    if n_rows <= 1:
        return n_rows
    bounded = min(max(fit_fraction, 0.1), 0.9)
    return min(max(int(round(n_rows * bounded)), 1), n_rows - 1)


def _has_both_classes(df: pd.DataFrame) -> bool:
    if df.empty or "bad_outcome" not in df.columns:
        return False
    return int(_bool_series(df, "bad_outcome").nunique()) == 2


def _adjust_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "is_adjust" not in df.columns:
        return pd.DataFrame()
    return cast(pd.DataFrame, df.loc[_bool_series(df, "is_adjust")].copy())


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    series = cast(pd.Series, df[col])
    if series.dtype == bool:
        return cast(pd.Series, series.fillna(False).astype(bool))

    def _coerce_bool(value: Any) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value).strip().lower() in {"true", "1", "yes"}

    return cast(pd.Series, series.map(_coerce_bool).astype(bool))


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _expand_paths(patterns: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if any(ch in pattern for ch in "*?[]"):
            parent = path.parent if str(path.parent) != "." else Path(".")
            out.extend(sorted(parent.glob(path.name)))
        else:
            out.append(path)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline D5/D6 meta tilt shadow analysis (non-gating)."
    )
    parser.add_argument("--decisions", nargs="+", default=["logs/live_decisions_*.jsonl"])
    parser.add_argument("--expired-bots", type=Path, default=Path("data/new_expired_bots.xlsx"))
    parser.add_argument("--linkage-dir", type=Path, default=Path("data/linkage"))
    parser.add_argument("--scanner-results-dir", type=Path, default=Path("results"))
    parser.add_argument("--min-bot-date", default="2026-02-01")
    parser.add_argument("--min-joined-rows", type=int, default=30)
    parser.add_argument("--min-fit-adjust-rows", type=int, default=10)
    parser.add_argument("--min-oos-rows", type=int, default=10)
    parser.add_argument("--min-oos-tilt-rows", type=int, default=3)
    parser.add_argument("--outcome-positive-pnl-pct", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    config = ShadowAnalysisConfig(
        outcome_positive_pnl_pct=float(args.outcome_positive_pnl_pct),
        min_joined_rows=int(args.min_joined_rows),
        min_fit_adjust_rows=int(args.min_fit_adjust_rows),
        min_oos_rows=int(args.min_oos_rows),
        min_oos_tilt_rows=int(args.min_oos_tilt_rows),
    )
    result = analyze_from_files(
        decision_paths=_expand_paths(args.decisions),
        expired_bots_path=args.expired_bots,
        linkage_dir=args.linkage_dir,
        scanner_results_dir=args.scanner_results_dir,
        min_bot_date=str(args.min_bot_date),
        config=config,
    )
    text = json.dumps(_jsonable(result), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
