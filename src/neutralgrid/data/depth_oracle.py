"""Fail-closed depth-aware deployability labels from depth-shadow evidence.

This module converts candidate-keyed, time-varying depth-shadow records plus
explicit forward outcomes into training labels.  It intentionally refuses to
invent missing fills, PnL, or tail-risk evidence: rows with incomplete depth
windows or missing outcome columns remain unlabeled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd


DEPTH_ORACLE_SCHEMA_VERSION = "depth_oracle_v1"

_PNL_COLUMNS = (
    "depth_aware_pnl_pct",
    "net_pnl_pct",
    "pnl_pct",
    "final_pnl_pct",
)
_TIME_TO_TARGET_COLUMNS = (
    "time_to_target_hours",
    "time_to_3pct_hours",
    "fast_winner_time_to_3pct_hours",
    "duration_hours",
)
_TAIL_COLUMNS = (
    "min_pnl_pct",
    "tail_pnl_pct",
    "mae_pct",
    "max_drawdown_pct",
)
_TARGET_HIT_COLUMNS = (
    "target_hit",
    "hit_target",
    "fast_winner_target",
    "label_positive_by_horizon",
)


@dataclass(frozen=True)
class DepthOracleConfig:
    """Policy knobs for the depth-aware deployability label builder."""

    horizon_hours: float = 7.0
    target_pnl_pct: float = 3.0
    max_tail_loss_pct: float = -20.0
    min_fill_ratio: float = 1.0
    min_capacity_ratio: float = 1.0
    min_snapshots: int = 2
    min_window_hours: float | None = None
    pnl_is_net_of_costs: bool = False
    max_spread_pct: float | None = None
    max_impact_bps: float | None = None

    @property
    def required_window_hours(self) -> float:
        return self.horizon_hours if self.min_window_hours is None else self.min_window_hours


@dataclass(frozen=True)
class DepthOracleColumns:
    """Outcome columns used to produce labels."""

    pnl_pct: str | None = None
    time_to_target_hours: str | None = None
    tail_pnl_pct: str | None = None
    target_hit: str | None = None


@dataclass(frozen=True)
class DepthOracleResult:
    """All persisted tables produced by the oracle builder."""

    labels: pd.DataFrame
    replay_diagnostics: pd.DataFrame
    failed_fill_reasons: pd.DataFrame
    manifest: dict[str, Any]


def _float_or_nan(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return float(numeric) if np.isfinite(numeric) else float("nan")


def _series_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return cast(pd.Series, pd.to_numeric(df[column], errors="coerce"))


def _first_present(columns: Sequence[str], available: Sequence[str]) -> str | None:
    available_set = set(available)
    for column in columns:
        if column in available_set:
            return column
    return None


def resolve_depth_oracle_columns(
    outcomes: pd.DataFrame,
    *,
    requested: DepthOracleColumns | None = None,
) -> DepthOracleColumns:
    """Resolve explicit outcome columns, falling back to known names only."""
    requested = requested or DepthOracleColumns()
    return DepthOracleColumns(
        pnl_pct=requested.pnl_pct or _first_present(_PNL_COLUMNS, list(outcomes.columns)),
        time_to_target_hours=(
            requested.time_to_target_hours
            or _first_present(_TIME_TO_TARGET_COLUMNS, list(outcomes.columns))
        ),
        tail_pnl_pct=requested.tail_pnl_pct or _first_present(_TAIL_COLUMNS, list(outcomes.columns)),
        target_hit=requested.target_hit or _first_present(_TARGET_HIT_COLUMNS, list(outcomes.columns)),
    )


def _candidate_key(series: pd.Series) -> pd.Series:
    return cast(pd.Series, series.astype(str).str.strip())


def summarize_depth_windows(records: pd.DataFrame, config: DepthOracleConfig) -> pd.DataFrame:
    """Summarize time-varying L2 depth records into candidate replay diagnostics."""
    required = {"symbol", "candidate_id", "capture_time_utc"}
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"Depth-shadow records missing required columns: {missing}")
    if records.empty:
        return pd.DataFrame()

    df = records.copy()
    df["_candidate_key"] = _candidate_key(cast(pd.Series, df["candidate_id"]))
    df["_capture_ts"] = pd.to_datetime(df["capture_time_utc"], utc=True, errors="coerce")
    df = cast(pd.DataFrame, df.sort_values(["_candidate_key", "_capture_ts"]))

    rows: list[dict[str, Any]] = []
    for candidate_key, group in df.groupby("_candidate_key", sort=False):
        g = cast(pd.DataFrame, group)
        capture_ts = cast(pd.Series, g["_capture_ts"]).dropna()
        first_ts: pd.Timestamp | None = None
        last_ts: pd.Timestamp | None = None
        window_hours = float("nan")
        if not capture_ts.empty:
            first_current = cast(pd.Timestamp, pd.Timestamp(cast(Any, capture_ts.min())))
            last_current = cast(pd.Timestamp, pd.Timestamp(cast(Any, capture_ts.max())))
            window_hours = float((last_current - first_current).total_seconds() / 3600.0)
            first_ts = first_current
            last_ts = last_current
        min_fill = _series_numeric(g, "min_side_fill_ratio")
        capacity = _series_numeric(g, "partial_fill_capacity_ratio")
        spread = _series_numeric(g, "spread_pct")
        impact = _series_numeric(g, "max_side_impact_bps")
        fee = _series_numeric(g, "round_trip_fee_pct")
        funding = _series_numeric(g, "estimated_abs_funding_pct")
        depth = _series_numeric(g, "top_n_depth_min_usdt")
        position = _series_numeric(g, "position_notional_usdt")

        rows.append(
            {
                "schema_version": DEPTH_ORACLE_SCHEMA_VERSION,
                "symbol": g["symbol"].iloc[0],
                "candidate_id": g["candidate_id"].iloc[0],
                "_candidate_key": candidate_key,
                "snapshot_count": int(len(g)),
                "first_capture_time_utc": first_ts.isoformat() if first_ts is not None else None,
                "last_capture_time_utc": last_ts.isoformat() if last_ts is not None else None,
                "window_hours": window_hours,
                "required_window_hours": config.required_window_hours,
                "min_top_n_depth_min_usdt": (
                    float(depth.min()) if bool(depth.notna().any()) else np.nan
                ),
                "median_top_n_depth_min_usdt": (
                    float(depth.median()) if bool(depth.notna().any()) else np.nan
                ),
                "position_notional_usdt": (
                    float(position.dropna().iloc[0]) if bool(position.notna().any()) else np.nan
                ),
                "min_side_fill_ratio": (
                    float(min_fill.min()) if bool(min_fill.notna().any()) else np.nan
                ),
                "min_partial_fill_capacity_ratio": (
                    float(capacity.min()) if bool(capacity.notna().any()) else np.nan
                ),
                "max_spread_pct": float(spread.max()) if bool(spread.notna().any()) else np.nan,
                "median_spread_pct": (
                    float(spread.median()) if bool(spread.notna().any()) else np.nan
                ),
                "max_side_impact_bps": (
                    float(impact.max()) if bool(impact.notna().any()) else np.nan
                ),
                "median_max_side_impact_bps": (
                    float(impact.median()) if bool(impact.notna().any()) else np.nan
                ),
                "round_trip_fee_pct": float(fee.max()) if bool(fee.notna().any()) else np.nan,
                "estimated_abs_funding_pct": (
                    float(funding.max()) if bool(funding.notna().any()) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _boolish(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return None


def _reason_text(reasons: Sequence[str]) -> str:
    return ";".join(sorted(set(reasons)))


def _candidate_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    if "candidate_id" not in outcomes.columns:
        raise ValueError("Outcomes input is missing required column 'candidate_id'")
    out = outcomes.copy()
    out["_candidate_key"] = _candidate_key(cast(pd.Series, out["candidate_id"]))
    return out


def build_depth_oracle_labels(
    records: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    config: DepthOracleConfig | None = None,
    columns: DepthOracleColumns | None = None,
) -> DepthOracleResult:
    """Build fail-closed depth-aware deployability labels.

    Positive labels require complete depth coverage, explicit outcome and tail
    columns, full-fill capacity, and net depth-adjusted PnL above the target.
    Missing evidence yields a null label instead of a negative label.
    """
    cfg = config or DepthOracleConfig()
    diagnostics = summarize_depth_windows(records, cfg)
    outcome_df = _candidate_outcomes(outcomes)
    resolved = resolve_depth_oracle_columns(outcome_df, requested=columns)

    duplicate_keys = set(
        cast(pd.Series, outcome_df.loc[outcome_df["_candidate_key"].duplicated(keep=False), "_candidate_key"])
    )
    deduped_outcomes = cast(pd.DataFrame, outcome_df.drop_duplicates("_candidate_key", keep="first"))
    merged = diagnostics.merge(
        deduped_outcomes,
        on="_candidate_key",
        how="left",
        suffixes=("", "_outcome"),
        indicator=True,
    )

    label_rows: list[dict[str, Any]] = []
    reason_rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        r = cast(pd.Series, row)
        candidate_key = str(r["_candidate_key"])
        reasons: list[str] = []
        execution_reasons: list[str] = []

        if candidate_key in duplicate_keys:
            reasons.append("duplicate_outcome_rows")
        if r["_merge"] != "both":
            reasons.append("missing_outcome_row")
        if resolved.pnl_pct is None:
            reasons.append("missing_outcome_pnl_column")
        if resolved.time_to_target_hours is None and resolved.target_hit is None:
            reasons.append("missing_target_timing_or_hit_column")
        if resolved.tail_pnl_pct is None:
            reasons.append("missing_tail_risk_column")

        snapshot_count = int(_float_or_nan(r.get("snapshot_count")))
        window_hours = _float_or_nan(r.get("window_hours"))
        if snapshot_count < cfg.min_snapshots:
            reasons.append("insufficient_depth_snapshots")
        if not np.isfinite(window_hours) or window_hours < cfg.required_window_hours:
            reasons.append("insufficient_depth_window")

        min_fill = _float_or_nan(r.get("min_side_fill_ratio"))
        min_capacity = _float_or_nan(r.get("min_partial_fill_capacity_ratio"))
        max_impact_bps = _float_or_nan(r.get("max_side_impact_bps"))
        max_spread_pct = _float_or_nan(r.get("max_spread_pct"))
        round_trip_fee_pct = _float_or_nan(r.get("round_trip_fee_pct"))
        funding_pct = _float_or_nan(r.get("estimated_abs_funding_pct"))

        if not np.isfinite(min_fill):
            reasons.append("missing_fill_ratio")
            execution_reasons.append("missing_fill_ratio")
        elif min_fill < cfg.min_fill_ratio:
            execution_reasons.append("partial_fill")
        if not np.isfinite(min_capacity):
            reasons.append("missing_capacity_ratio")
            execution_reasons.append("missing_capacity_ratio")
        elif min_capacity < cfg.min_capacity_ratio:
            execution_reasons.append("insufficient_capacity")
        if not np.isfinite(max_impact_bps):
            reasons.append("missing_impact")
            execution_reasons.append("missing_impact")
        if cfg.max_impact_bps is not None and np.isfinite(max_impact_bps) and max_impact_bps > cfg.max_impact_bps:
            execution_reasons.append("impact_breach")
        if not np.isfinite(max_spread_pct):
            reasons.append("missing_spread")
            execution_reasons.append("missing_spread")
        if cfg.max_spread_pct is not None and np.isfinite(max_spread_pct) and max_spread_pct > cfg.max_spread_pct:
            execution_reasons.append("spread_breach")
        if not cfg.pnl_is_net_of_costs:
            if not np.isfinite(round_trip_fee_pct):
                reasons.append("missing_fee_context")
            if not np.isfinite(funding_pct):
                reasons.append("missing_funding_context")

        target_hit = None
        if resolved.target_hit is not None:
            target_hit = _boolish(r.get(resolved.target_hit))
            if target_hit is None:
                reasons.append("missing_target_hit")

        outcome_pnl = _float_or_nan(r.get(resolved.pnl_pct)) if resolved.pnl_pct is not None else np.nan
        time_to_target = (
            _float_or_nan(r.get(resolved.time_to_target_hours))
            if resolved.time_to_target_hours is not None
            else np.nan
        )
        tail_pnl = _float_or_nan(r.get(resolved.tail_pnl_pct)) if resolved.tail_pnl_pct is not None else np.nan
        if resolved.pnl_pct is not None and not np.isfinite(outcome_pnl):
            reasons.append("missing_outcome_pnl")
        if resolved.time_to_target_hours is not None and not np.isfinite(time_to_target) and target_hit is None:
            reasons.append("missing_time_to_target")
        if resolved.tail_pnl_pct is not None and not np.isfinite(tail_pnl):
            reasons.append("missing_tail_risk")

        if target_hit is None and np.isfinite(time_to_target):
            target_hit = bool(time_to_target <= cfg.horizon_hours)

        impact_pct = max_impact_bps / 100.0 if np.isfinite(max_impact_bps) else np.nan
        explicit_cost_pct = 0.0
        if not cfg.pnl_is_net_of_costs:
            explicit_cost_pct = (
                (round_trip_fee_pct if np.isfinite(round_trip_fee_pct) else np.nan)
                + (funding_pct if np.isfinite(funding_pct) else np.nan)
                + (impact_pct if np.isfinite(impact_pct) else np.nan)
            )
        depth_adjusted_pnl = outcome_pnl - explicit_cost_pct if np.isfinite(outcome_pnl) else np.nan

        tail_ok = np.isfinite(tail_pnl) and tail_pnl >= cfg.max_tail_loss_pct
        target_ok = bool(target_hit) and np.isfinite(time_to_target) and time_to_target <= cfg.horizon_hours
        if resolved.target_hit is not None and target_hit is not None:
            target_ok = bool(target_hit)
        net_ok = np.isfinite(depth_adjusted_pnl) and depth_adjusted_pnl >= cfg.target_pnl_pct
        execution_ok = len(execution_reasons) == 0

        label_value: int | None = None
        status = "unlabeled"
        if not reasons:
            label_value = 1 if target_ok and net_ok and tail_ok and execution_ok else 0
            status = "positive" if label_value == 1 else "negative"
            if not target_ok:
                reasons.append("target_not_met")
            if not net_ok:
                reasons.append("net_pnl_below_target")
            if not tail_ok:
                reasons.append("tail_loss_breach")
            reasons.extend(execution_reasons)
        else:
            reasons.extend(execution_reasons)

        label_rows.append(
            {
                "schema_version": DEPTH_ORACLE_SCHEMA_VERSION,
                "symbol": r.get("symbol"),
                "candidate_id": r.get("candidate_id"),
                "depth_oracle_label": label_value,
                "label_status": status,
                "failed_fill_reason": _reason_text(execution_reasons),
                "label_reason": _reason_text(reasons),
                "outcome_pnl_pct": outcome_pnl,
                "time_to_target_hours": time_to_target,
                "tail_pnl_pct": tail_pnl,
                "depth_adjusted_pnl_pct": depth_adjusted_pnl,
                "snapshot_count": snapshot_count,
                "window_hours": window_hours,
                "min_side_fill_ratio": min_fill,
                "min_partial_fill_capacity_ratio": min_capacity,
                "max_spread_pct": max_spread_pct,
                "max_side_impact_bps": max_impact_bps,
                "round_trip_fee_pct": round_trip_fee_pct,
                "estimated_abs_funding_pct": funding_pct,
            }
        )
        if execution_reasons or label_value != 1:
            reason_rows.append(
                {
                    "symbol": r.get("symbol"),
                    "candidate_id": r.get("candidate_id"),
                    "depth_oracle_label": label_value,
                    "label_status": status,
                    "failed_fill_reason": _reason_text(execution_reasons),
                    "label_reason": _reason_text(reasons),
                }
            )

    labels = pd.DataFrame(label_rows)
    if "depth_oracle_label" in labels.columns:
        labels["depth_oracle_label"] = labels["depth_oracle_label"].astype("Int64")
    failed = pd.DataFrame(reason_rows)
    labelable_rows = int(labels["depth_oracle_label"].notna().sum()) if not labels.empty else 0
    positive_rows = int((labels["depth_oracle_label"] == 1).sum()) if not labels.empty else 0
    negative_rows = int((labels["depth_oracle_label"] == 0).sum()) if not labels.empty else 0
    status = "complete" if labelable_rows > 0 else "blocked_no_labelable_rows"
    manifest = {
        "schema_version": DEPTH_ORACLE_SCHEMA_VERSION,
        "status": status,
        "config": asdict(cfg),
        "resolved_columns": asdict(resolved),
        "depth_input_rows": int(len(records)),
        "depth_candidate_count": int(len(diagnostics)),
        "outcome_rows": int(len(outcomes)),
        "label_rows": int(len(labels)),
        "labelable_rows": labelable_rows,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "unlabeled_rows": int(len(labels) - labelable_rows),
        "note": (
            "Rows are unlabeled, not negative, when depth window, outcome, cost, "
            "fill, or tail-risk evidence is missing."
        ),
    }
    diagnostics = diagnostics.drop(columns=["_candidate_key"], errors="ignore")
    return DepthOracleResult(
        labels=labels,
        replay_diagnostics=diagnostics,
        failed_fill_reasons=failed,
        manifest=cast(dict[str, Any], manifest),
    )


def manifest_with_outputs(
    manifest: Mapping[str, Any],
    *,
    labels_path: str,
    diagnostics_path: str,
    failed_reasons_path: str,
) -> dict[str, Any]:
    """Attach output paths to an oracle manifest for persistence."""
    out = dict(manifest)
    out.update(
        {
            "depth_labels": labels_path,
            "replay_diagnostics": diagnostics_path,
            "failed_fill_reasons": failed_reasons_path,
        }
    )
    return out
