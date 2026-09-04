"""Bot-level observational linkage for execution-risk telemetry and outcomes.

This module deliberately does not select thresholds, create a promotion gate,
or modify live verdicts. It collapses repeated scanner ticks to one row per
exact bot identity and exposes a chronological development/later split so that
future evidence can be audited without treating correlated ticks as independent
bots.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, cast

import numpy as np
import pandas as pd

from neutralgrid.live.decision.meta_shadow_analysis import (
    _expand_paths,
    _life_window_outside_mask,
    join_shadow_to_outcomes,
    load_decision_jsonl,
    load_live_outcomes,
)


@dataclass(frozen=True)
class ExecutionOutcomeAnalysisConfig:
    """Reporting-only temporal split configuration; contains no action rule."""

    development_fraction: float = 0.60


_NUMERIC_AGGREGATES: tuple[tuple[str, str], ...] = (
    ("gain_giveback_pct", "max_gain_giveback_pct"),
    ("gain_giveback_usdt", "max_gain_giveback_usdt"),
    ("exit_depth_current_to_baseline", "min_exit_depth_current_to_baseline"),
    ("mean_estimated_slippage_bps", "max_mean_estimated_slippage_bps"),
    ("mean_adverse_selection_5s_bps", "max_mean_adverse_selection_5s_bps"),
    ("mean_adverse_selection_30s_bps", "max_mean_adverse_selection_30s_bps"),
    ("expected_exit_impact_bps", "max_expected_exit_impact_bps"),
    (
        "joint_deterioration_trailing_duration_seconds",
        "max_joint_deterioration_trailing_duration_seconds",
    ),
    (
        "exit_side_net_withdrawal_to_position_ratio",
        "max_exit_side_net_withdrawal_to_position_ratio",
    ),
    (
        "unexplained_removal_to_position_ratio",
        "max_unexplained_removal_to_position_ratio",
    ),
    ("private_cancel_update_fraction", "max_private_cancel_update_fraction"),
)


def analyze_joined_execution_outcomes(
    joined: pd.DataFrame,
    *,
    config: ExecutionOutcomeAnalysisConfig = ExecutionOutcomeAnalysisConfig(),
) -> dict[str, Any]:
    """Create a verdict-inert, bot-level temporal evidence report."""

    if not 0.0 < config.development_fraction < 1.0:
        raise ValueError("development_fraction must be between zero and one")
    payload: dict[str, Any] = {
        "analysis_contract": "execution_outcome_observational_v1",
        "status": "insufficient_no_exact_joined_outcomes",
        "config": asdict(config),
        "runtime_effect": "none",
        "threshold_selected": False,
        "promotion_gate_created": False,
        "verdict_influence_validated": False,
        "counts": {
            "joined_decision_rows": int(len(joined)),
            "eligible_evidence_rows_before_dedup": 0,
            "duplicate_evidence_rows_dropped": 0,
            "eligible_evidence_rows": 0,
            "bot_rows": 0,
        },
        "limitations": [
            "The report is observational and cannot establish causal gain protection.",
            "No counterfactual END or ADJUST PnL is fabricated.",
            "No execution-risk threshold is selected from the available outcomes.",
            "Only exact candidate_id or exact strategy_id plus symbol joins are accepted.",
        ],
    }
    if joined.empty:
        payload["reason"] = "no decision row has an exact finalized-outcome join"
        return payload

    eligible = _eligible_execution_rows(joined)
    payload["counts"]["eligible_evidence_rows_before_dedup"] = int(len(eligible))
    if eligible.empty:
        payload["status"] = "insufficient_no_execution_evidence"
        payload["reason"] = (
            "exact outcome joins exist, but no in-life scanner row carries execution-risk "
            "or profit-deterioration evidence"
        )
        return payload

    dedup_subset = [
        column
        for column in (
            "join_key",
            "ts_utc",
            "verdict",
            "l2_run_id",
            "l2_segment_id",
            "private_event_run_id",
            "current_total_profit_usdt",
            "gain_giveback_usdt",
        )
        if column in eligible.columns
    ]
    deduped = cast(
        pd.DataFrame,
        eligible.drop_duplicates(subset=dedup_subset, keep="first").copy(),
    )
    payload["counts"]["duplicate_evidence_rows_dropped"] = int(
        len(eligible) - len(deduped)
    )
    payload["counts"]["eligible_evidence_rows"] = int(len(deduped))

    bots = _collapse_to_bot_rows(deduped)
    payload["counts"]["bot_rows"] = int(len(bots))
    payload["field_coverage"] = _field_coverage(deduped)
    if bots.empty:
        payload["status"] = "insufficient_no_bot_rows"
        payload["reason"] = "eligible evidence could not be collapsed to an exact bot identity"
        return payload

    payload["all_bots_observational_summary"] = _cohort_summary(bots)
    payload["bot_rows"] = _json_records(bots)
    if len(bots) < 2:
        payload["status"] = "insufficient_bot_count_for_temporal_split"
        payload["reason"] = "fewer than two exact bots have linked evidence and outcomes"
        return payload

    split_index = min(
        max(int(round(len(bots) * config.development_fraction)), 1),
        len(bots) - 1,
    )
    development = cast(pd.DataFrame, bots.iloc[:split_index].copy())
    later = cast(pd.DataFrame, bots.iloc[split_index:].copy())
    development_keys = set(cast(pd.Series, development["join_key"]).astype(str))
    later_keys = set(cast(pd.Series, later["join_key"]).astype(str))
    overlap = sorted(development_keys & later_keys)
    payload["temporal_split"] = {
        "strategy": "chronological_exact_bot_identity",
        "development": _cohort_summary(development),
        "later_holdout": _cohort_summary(later),
        "development_join_keys": len(development_keys),
        "later_join_keys": len(later_keys),
        "overlap_join_keys": overlap,
        "overlap_join_key_count": len(overlap),
    }
    payload["status"] = "observational_temporal_split_available"
    payload["reason"] = (
        "a bot-disjoint temporal description is available, but no action threshold or "
        "counterfactual policy has been validated"
    )
    return payload


def analyze_execution_outcomes_from_files(
    *,
    decision_paths: Sequence[Path],
    expired_bots_path: Path,
    linkage_dir: Path = Path("data/linkage"),
    scanner_results_dir: Path = Path("results"),
    min_bot_date: str = "2026-02-01",
    config: ExecutionOutcomeAnalysisConfig = ExecutionOutcomeAnalysisConfig(),
) -> dict[str, Any]:
    """Load scanner logs and finalized outcomes through canonical lineage."""

    decisions = load_decision_jsonl(decision_paths)
    outcomes = load_live_outcomes(
        expired_bots_path=expired_bots_path,
        linkage_dir=linkage_dir,
        scanner_results_dir=scanner_results_dir,
        min_bot_date=min_bot_date,
    )
    joined = join_shadow_to_outcomes(decisions, outcomes)
    result = analyze_joined_execution_outcomes(joined, config=config)
    result["input_counts"] = {
        "decision_rows": int(len(decisions)),
        "outcome_rows": int(len(outcomes)),
        "joined_rows": int(len(joined)),
    }
    return result


def _eligible_execution_rows(joined: pd.DataFrame) -> pd.DataFrame:
    df = joined.copy()
    if "join_key" not in df.columns:
        return pd.DataFrame()
    ts_source = (
        cast(pd.Series, df["ts_utc"])
        if "ts_utc" in df.columns
        else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    )
    ts = cast(
        pd.Series,
        pd.to_datetime(ts_source, utc=True, errors="coerce"),
    )
    pnl = _numeric_series(df, "pnl_pct")
    join_key = cast(pd.Series, df["join_key"]).fillna("").astype(str).str.strip()
    execution_source = _text_series(df, "execution_risk_source")
    profit_present = _numeric_series(df, "current_total_profit_usdt").notna()
    evidence_present = execution_source.ne("") | profit_present
    post_outcome, pre_deploy, _missing_window = _life_window_outside_mask(df)
    mask = (
        join_key.ne("")
        & ts.notna()
        & pnl.notna()
        & evidence_present
        & ~post_outcome
        & ~pre_deploy
    )
    eligible = cast(pd.DataFrame, df.loc[mask].copy())
    eligible["ts_utc"] = ts.loc[mask]
    eligible["pnl_pct"] = pnl.loc[mask]
    return cast(
        pd.DataFrame,
        eligible.sort_values(["ts_utc", "join_key"]).reset_index(drop=True),
    )


def _collapse_to_bot_rows(rows: pd.DataFrame) -> pd.DataFrame:
    collapsed: list[dict[str, Any]] = []
    for join_key, group in rows.groupby("join_key", sort=False, dropna=False):
        group_df = cast(pd.DataFrame, group.sort_values("ts_utc"))
        outcome_values = sorted(
            set(_numeric_series(group_df, "pnl_pct").dropna().astype(float))
        )
        if len(outcome_values) != 1:
            raise ValueError(
                f"exact bot {join_key!s} maps to {len(outcome_values)} finalized PnL values"
            )
        row: dict[str, Any] = {
            "join_key": str(join_key),
            "join_method": _first_text(group_df, "join_method"),
            "symbol": _first_text(group_df, "symbol", "symbol_decision"),
            "strategy_id": _first_text(group_df, "strategy_id", "strategy_id_decision"),
            "candidate_id": _first_text(group_df, "candidate_id"),
            "first_decision_at_utc": cast(pd.Timestamp, group_df["ts_utc"].iloc[0]),
            "last_decision_at_utc": cast(pd.Timestamp, group_df["ts_utc"].iloc[-1]),
            "decision_ticks": int(len(group_df)),
            "ticks_with_execution_risk": int(
                _text_series(group_df, "execution_risk_source").ne("").sum()
            ),
            "final_pnl_pct": outcome_values[0],
            "final_outcome_class": (
                "positive" if outcome_values[0] > 0 else "zero_or_negative"
            ),
            "sustained_deterioration_observed": bool(
                _bool_series(group_df, "sustained_joint_deterioration").any()
            ),
            "sustained_spread_deterioration_observed": bool(
                _bool_series(group_df, "sustained_spread_deterioration").any()
            ),
            "sustained_exit_depth_deterioration_observed": bool(
                _bool_series(group_df, "sustained_exit_depth_deterioration").any()
            ),
            "temporary_deterioration_observed": bool(
                _bool_series(group_df, "temporary_joint_deterioration").any()
            ),
            "private_event_completeness": sorted(
                set(_text_series(group_df, "private_event_completeness")) - {""}
            ),
            "public_trade_statuses": sorted(
                set(_text_series(group_df, "public_trade_status")) - {""}
            ),
        }
        for source, target in _NUMERIC_AGGREGATES:
            numeric = _numeric_series(group_df, source).dropna()
            if numeric.empty:
                row[target] = None
            elif source == "exit_depth_current_to_baseline":
                row[target] = float(numeric.min())
            else:
                row[target] = float(numeric.max())
        collapsed.append(row)
    output = pd.DataFrame(collapsed)
    if output.empty:
        return output
    return cast(
        pd.DataFrame,
        output.sort_values(["first_decision_at_utc", "join_key"]).reset_index(drop=True),
    )


def _field_coverage(rows: pd.DataFrame) -> dict[str, int]:
    fields = (
        "execution_risk_source",
        "sustained_joint_deterioration",
        "exit_depth_current_to_baseline",
        "mean_estimated_slippage_bps",
        "mean_adverse_selection_5s_bps",
        "mean_adverse_selection_30s_bps",
        "expected_exit_impact_bps",
        "private_event_completeness",
        "gain_giveback_pct",
        "joint_deterioration_trailing_duration_seconds",
        "exit_side_net_withdrawal_to_position_ratio",
        "unexplained_removal_to_position_ratio",
        "private_cancel_update_fraction",
    )
    coverage: dict[str, int] = {}
    for field in fields:
        if field not in rows.columns:
            coverage[field] = 0
        elif field in {"execution_risk_source", "private_event_completeness"}:
            coverage[field] = int(_text_series(rows, field).ne("").sum())
        else:
            coverage[field] = int(cast(pd.Series, rows[field]).notna().sum())
    return coverage


def _cohort_summary(bots: pd.DataFrame) -> dict[str, Any]:
    if bots.empty:
        return {
            "bots": 0,
            "positive_outcomes": 0,
            "zero_or_negative_outcomes": 0,
            "median_final_pnl_pct": None,
            "sustained_deterioration_bots": 0,
        }
    pnl = _numeric_series(bots, "final_pnl_pct").dropna()
    sustained = _bool_series(bots, "sustained_deterioration_observed")
    summary: dict[str, Any] = {
        "bots": int(len(bots)),
        "positive_outcomes": int((pnl > 0).sum()),
        "zero_or_negative_outcomes": int((pnl <= 0).sum()),
        "median_final_pnl_pct": float(pnl.median()) if not pnl.empty else None,
        "sustained_deterioration_bots": int(sustained.sum()),
    }
    for label, mask in (
        ("sustained", sustained),
        ("not_sustained", ~sustained),
    ):
        cohort_pnl = _numeric_series(cast(pd.DataFrame, bots.loc[mask]), "final_pnl_pct").dropna()
        summary[f"{label}_bots"] = int(mask.sum())
        summary[f"{label}_median_final_pnl_pct"] = (
            float(cohort_pnl.median()) if not cohort_pnl.empty else None
        )
    return summary


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return cast(pd.Series, pd.to_numeric(df[column], errors="coerce"))


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="string")
    return cast(pd.Series, df[column]).fillna("").astype(str).str.strip()


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)

    def _coerce(value: Any) -> bool:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value).strip().lower() in {"true", "1", "yes"}

    return cast(pd.Series, cast(pd.Series, df[column]).map(_coerce).astype(bool))


def _first_text(df: pd.DataFrame, *columns: str) -> str:
    for column in columns:
        if column not in df.columns:
            continue
        values = _text_series(df, column)
        nonempty = values.loc[values.ne("")]
        if not nonempty.empty:
            return str(nonempty.iloc[0])
    return ""


def _json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in df.to_dict(orient="records"):
        record: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, (pd.Timestamp,)):
                record[str(key)] = value.isoformat()
            elif isinstance(value, (np.integer,)):
                record[str(key)] = int(value)
            elif isinstance(value, (np.floating,)):
                record[str(key)] = float(value)
            else:
                record[str(key)] = value
        records.append(record)
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Observational bot-level execution-risk/outcome linkage; no gate."
    )
    parser.add_argument("--decisions", nargs="+", default=["logs/live_decisions_*.jsonl"])
    parser.add_argument("--expired-bots", type=Path, default=Path("data/new_expired_bots.xlsx"))
    parser.add_argument("--linkage-dir", type=Path, default=Path("data/linkage"))
    parser.add_argument("--scanner-results-dir", type=Path, default=Path("results"))
    parser.add_argument("--min-bot-date", default="2026-02-01")
    parser.add_argument("--development-fraction", type=float, default=0.60)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    result = analyze_execution_outcomes_from_files(
        decision_paths=_expand_paths(args.decisions),
        expired_bots_path=args.expired_bots,
        linkage_dir=args.linkage_dir,
        scanner_results_dir=args.scanner_results_dir,
        min_bot_date=str(args.min_bot_date),
        config=ExecutionOutcomeAnalysisConfig(
            development_fraction=float(args.development_fraction)
        ),
    )
    output = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
