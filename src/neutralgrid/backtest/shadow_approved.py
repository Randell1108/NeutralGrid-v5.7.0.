"""Selection and provenance helpers for shadow-approved candidate backtests.

This module never promotes a diagnostic surrogate to runtime authority.  It
replays a narrow counterfactual instead: rows that reached Stage B and were
rejected *only* because authoritative ``meta_prob`` was absent are eligible
when the quarantined shadow probability is finite and clears the recorded
meta threshold.  Outcomes remain unfiltered, diagnostic, and non-authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, cast

import numpy as np
import pandas as pd


SHADOW_BACKTEST_CONTRACT = "shadow_approved_unfiltered_v1"
DEFAULT_DURATION_MINUTES = 362
_DEPLOYMENT_STAMP_RE = re.compile(r"deployment_ready_(\d{8}_\d{6})\.csv$")
_TRUTHY = frozenset({"true", "1", "yes"})


@dataclass(frozen=True)
class ShadowRunSources:
    """Immutable source paths and availability time for one pipeline run."""

    run_key: str
    deployment_stamp: str
    deployment_path: Path
    shadow_path: Path
    shadow_manifest_path: Path
    candidate_available_ts_utc: datetime


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest without loading the whole file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _truthy_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return cast(pd.Series, values.fillna(False).astype(bool))
    return cast(
        pd.Series,
        values.fillna("").astype(str).str.strip().str.lower().isin(_TRUTHY),
    )


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _normalized_ids(frame: pd.DataFrame, label: str) -> pd.Series:
    _require_columns(frame, {"candidate_id"}, label)
    candidate_ids = cast(pd.Series, frame["candidate_id"]).fillna("").astype(str).str.strip()
    if bool(candidate_ids.eq("").any()):
        raise ValueError(f"{label} contains blank candidate_id values")
    if bool(candidate_ids.duplicated().any()):
        duplicates = sorted(candidate_ids.loc[candidate_ids.duplicated(keep=False)].unique().tolist())
        raise ValueError(f"{label} contains duplicate candidate_id values: {duplicates[:5]}")
    return candidate_ids


def discover_shadow_runs(
    pipeline_runs_root: Path,
    shadow_root: Path,
) -> list[ShadowRunSources]:
    """Find pipeline CSVs with their exact companion shadow report/manifest."""
    if not pipeline_runs_root.is_dir():
        raise FileNotFoundError(f"Pipeline-runs directory not found: {pipeline_runs_root}")
    if not shadow_root.is_dir():
        raise FileNotFoundError(f"Shadow-report directory not found: {shadow_root}")

    discovered: list[ShadowRunSources] = []
    seen_stamps: set[str] = set()
    for deployment_path in sorted(pipeline_runs_root.rglob("deployment_ready_*.csv")):
        match = _DEPLOYMENT_STAMP_RE.search(deployment_path.name)
        if match is None:
            continue
        stamp = match.group(1)
        if stamp in seen_stamps:
            raise ValueError(f"Multiple deployment CSVs share timestamp {stamp}")
        shadow_dir = shadow_root / f"pipeline_{stamp}"
        shadow_path = shadow_dir / "meta_prob_shadow.csv"
        manifest_path = shadow_dir / "meta_prob_shadow.manifest.json"
        if not shadow_path.is_file() or not manifest_path.is_file():
            continue
        seen_stamps.add(stamp)
        discovered.append(
            ShadowRunSources(
                run_key=f"pipeline_{stamp}",
                deployment_stamp=stamp,
                deployment_path=deployment_path.resolve(),
                shadow_path=shadow_path.resolve(),
                shadow_manifest_path=manifest_path.resolve(),
                candidate_available_ts_utc=datetime.fromtimestamp(
                    deployment_path.stat().st_mtime,
                    tz=timezone.utc,
                ),
            )
        )
    return discovered


def validate_shadow_sources(sources: ShadowRunSources) -> dict[str, Any]:
    """Verify source hashes and diagnostic-only markers from the shadow manifest."""
    manifest = json.loads(sources.shadow_manifest_path.read_text(encoding="utf-8"))
    required_markers = {
        "diagnostic_only": True,
        "promotion_eligible": False,
        "runtime_meta_labeler_compatible": False,
    }
    for key, expected in required_markers.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Shadow manifest marker mismatch for {sources.run_key}: "
                f"{key}={manifest.get(key)!r}, expected {expected!r}"
            )

    deployment_hash = sha256_file(sources.deployment_path)
    shadow_hash = sha256_file(sources.shadow_path)
    if str(manifest.get("input_sha256", "")).lower() != deployment_hash:
        raise ValueError(f"Deployment hash does not match shadow manifest: {sources.run_key}")
    if str(manifest.get("output_sha256", "")).lower() != shadow_hash:
        raise ValueError(f"Shadow output hash does not match manifest: {sources.run_key}")
    return cast(dict[str, Any], manifest)


def is_run_mature(
    sources: ShadowRunSources,
    *,
    now_utc: datetime,
    duration_minutes: int = DEFAULT_DURATION_MINUTES,
) -> bool:
    """Return True only after the complete fixed outcome window is available."""
    if duration_minutes <= 0:
        raise ValueError("duration_minutes must be positive")
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    maturity = backtest_window_start_utc(
        sources.candidate_available_ts_utc
    ) + timedelta(minutes=duration_minutes)
    return now_utc.astimezone(timezone.utc) >= maturity


def backtest_window_start_utc(candidate_available_ts_utc: datetime) -> datetime:
    """Return the first complete one-minute bar boundary after availability."""
    if candidate_available_ts_utc.tzinfo is None:
        raise ValueError("candidate_available_ts_utc must be timezone-aware")
    available = candidate_available_ts_utc.astimezone(timezone.utc)
    boundary = available.replace(second=0, microsecond=0)
    if available > boundary:
        boundary += timedelta(minutes=1)
    return boundary


def select_shadow_approved_candidates(
    deployment: pd.DataFrame,
    shadow: pd.DataFrame,
    *,
    min_meta_prob: float,
) -> pd.DataFrame:
    """Return the unfiltered Stage-B counterfactual cohort.

    Eligibility is deliberately narrow:

    * the row reached Stage B;
    * its recorded Stage-B rejection was exactly ``data_missing:meta``;
    * production metadata says the original probability was diagnostic-only;
    * the shadow row is feature-complete and its probability clears the
      configured threshold.

    No EV, PnL, symbol, or outcome filter is applied.
    """
    if not 0.0 <= float(min_meta_prob) <= 1.0:
        raise ValueError("min_meta_prob must be in [0, 1]")
    _require_columns(
        deployment,
        {
            "candidate_id",
            "failure_stage",
            "stage_b_reason",
            "meta_prob_authority",
            "grid_lower",
            "grid_upper",
            "num_grids",
            "capital_base_usdt",
            "capital_fraction",
            "deploy_margin_usdt",
            "leverage",
        },
        "deployment frame",
    )
    _require_columns(
        shadow,
        {
            "candidate_id",
            "surrogate_meta_prob_diagnostic",
            "feature_complete",
            "diagnostic_only",
            "promotion_eligible",
            "runtime_meta_labeler_compatible",
            "teacher_hmm_artifact_version",
            "snapshot_hmm_artifact_version",
        },
        "shadow frame",
    )

    deployment_ids = _normalized_ids(deployment, "deployment frame")
    shadow_ids = _normalized_ids(shadow, "shadow frame")
    if set(deployment_ids) != set(shadow_ids):
        missing_shadow = sorted(set(deployment_ids).difference(shadow_ids))
        extra_shadow = sorted(set(shadow_ids).difference(deployment_ids))
        raise ValueError(
            "Shadow/deployment candidate-ID sets differ: "
            f"missing_shadow={missing_shadow[:5]}, extra_shadow={extra_shadow[:5]}"
        )

    left = deployment.copy()
    right = shadow.copy()
    left["candidate_id"] = deployment_ids
    right["candidate_id"] = shadow_ids
    shadow_columns = [
        "candidate_id",
        "surrogate_meta_prob_diagnostic",
        "feature_complete",
        "diagnostic_only",
        "promotion_eligible",
        "runtime_meta_labeler_compatible",
        "teacher_hmm_artifact_version",
        "snapshot_hmm_artifact_version",
        "hmm_lineage_matches_teacher",
    ]
    shadow_columns = [column for column in shadow_columns if column in right.columns]
    joined = left.merge(
        right[shadow_columns],
        on="candidate_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_shadow"),
    )

    shadow_prob = cast(
        pd.Series,
        pd.to_numeric(
            cast(pd.Series, joined["surrogate_meta_prob_diagnostic"]),
            errors="coerce",
        ),
    )
    exact_meta_only = (
        cast(pd.Series, joined["stage_b_reason"])
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("data_missing:meta")
    )
    stage_b = (
        cast(pd.Series, joined["failure_stage"])
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("stage_b")
    )
    diagnostic_authority = (
        cast(pd.Series, joined["meta_prob_authority"])
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("diagnostic_only")
    )
    feature_complete = _truthy_series(cast(pd.Series, joined["feature_complete"]))
    diagnostic_only = _truthy_series(cast(pd.Series, joined["diagnostic_only"]))
    promotion_eligible = _truthy_series(cast(pd.Series, joined["promotion_eligible"]))
    runtime_compatible = _truthy_series(
        cast(pd.Series, joined["runtime_meta_labeler_compatible"])
    )
    selected_mask = (
        stage_b
        & exact_meta_only
        & diagnostic_authority
        & feature_complete
        & diagnostic_only
        & ~promotion_eligible
        & ~runtime_compatible
        & shadow_prob.notna()
        & shadow_prob.ge(float(min_meta_prob))
    )
    selected = cast(pd.DataFrame, joined.loc[selected_mask].copy())

    for column in (
        "grid_lower",
        "grid_upper",
        "num_grids",
        "capital_base_usdt",
        "capital_fraction",
        "deploy_margin_usdt",
        "leverage",
    ):
        values = cast(
            pd.Series,
            pd.to_numeric(cast(pd.Series, selected[column]), errors="coerce"),
        )
        if bool(values.isna().any()):
            raise ValueError(f"Selected cohort contains missing {column}")
    if not selected.empty:
        lower = cast(
            pd.Series,
            pd.to_numeric(cast(pd.Series, selected["grid_lower"]), errors="coerce"),
        )
        upper = cast(
            pd.Series,
            pd.to_numeric(cast(pd.Series, selected["grid_upper"]), errors="coerce"),
        )
        grids = cast(
            pd.Series,
            pd.to_numeric(cast(pd.Series, selected["num_grids"]), errors="coerce"),
        )
        capital_base = cast(
            pd.Series,
            pd.to_numeric(
                cast(pd.Series, selected["capital_base_usdt"]), errors="coerce"
            ),
        )
        capital_fraction = cast(
            pd.Series,
            pd.to_numeric(
                cast(pd.Series, selected["capital_fraction"]), errors="coerce"
            ),
        )
        deploy_margin = cast(
            pd.Series,
            pd.to_numeric(
                cast(pd.Series, selected["deploy_margin_usdt"]), errors="coerce"
            ),
        )
        leverage = cast(
            pd.Series,
            pd.to_numeric(cast(pd.Series, selected["leverage"]), errors="coerce"),
        )
        if bool((lower <= 0).any() or (upper <= lower).any() or (grids < 2).any()):
            raise ValueError("Selected cohort contains invalid grid geometry")
        if bool(
            (capital_base <= 0).any()
            or (capital_fraction <= 0).any()
            or (capital_fraction > 1).any()
            or (deploy_margin <= 0).any()
            or (leverage <= 0).any()
            or (~np.isclose(leverage, np.round(leverage))).any()
        ):
            raise ValueError("Selected cohort contains invalid recorded execution terms")
        expected_margin = capital_base * capital_fraction
        if not bool(
            np.isclose(
                np.asarray(deploy_margin, dtype=float),
                np.asarray(expected_margin, dtype=float),
                rtol=1e-8,
                atol=1e-8,
            ).all()
        ):
            raise ValueError(
                "Selected cohort deploy_margin_usdt does not match "
                "capital_base_usdt * capital_fraction"
            )

    selected["shadow_meta_prob_diagnostic"] = shadow_prob.loc[selected.index]
    selected["counterfactual_meta_threshold"] = float(min_meta_prob)
    selected["counterfactual_meta_gate_passed"] = True
    selected["counterfactual_all_recorded_gates_passed"] = True
    selected["counterfactual_selection_reason"] = (
        "recorded_stage_b_meta_only_and_shadow_prob_at_or_above_threshold"
    )
    selected["shadow_backtest_contract"] = SHADOW_BACKTEST_CONTRACT
    selected["source_class"] = SHADOW_BACKTEST_CONTRACT
    selected["unfiltered_outcome_pool"] = True
    selected["governed_training_eligible"] = False
    selected["promotion_eligible_counterfactual"] = False
    selected["deployment_eligible_counterfactual"] = False
    return cast(pd.DataFrame, selected.sort_values("candidate_id").reset_index(drop=True))


def validate_kline_window(
    klines: pd.DataFrame,
    *,
    start_utc: datetime,
    duration_minutes: int = DEFAULT_DURATION_MINUTES,
) -> pd.DataFrame:
    """Return exactly ``duration_minutes`` contiguous closed 1-minute bars."""
    if duration_minutes <= 0:
        raise ValueError("duration_minutes must be positive")
    _require_columns(klines, {"timestamp", "open", "high", "low", "close", "volume"}, "kline frame")
    if len(klines) < duration_minutes:
        raise ValueError(
            f"insufficient kline bars: {len(klines)} available, {duration_minutes} required"
        )
    window = cast(pd.DataFrame, klines.iloc[:duration_minutes].copy())
    timestamps = cast(
        pd.Series,
        pd.to_datetime(cast(pd.Series, window["timestamp"]), errors="coerce", utc=True),
    )
    if bool(timestamps.isna().any()):
        raise ValueError("kline window contains invalid timestamps")
    if bool(timestamps.duplicated().any()):
        raise ValueError("kline window contains duplicate timestamps")
    deltas = cast(pd.Series, timestamps.diff().dropna())
    if not deltas.empty and bool(deltas.ne(pd.Timedelta(minutes=1)).any()):
        raise ValueError("kline window is not contiguous at one-minute frequency")
    start = start_utc.astimezone(timezone.utc)
    first = cast(pd.Timestamp, timestamps.iloc[0]).to_pydatetime()
    last_close = cast(pd.Timestamp, timestamps.iloc[-1]).to_pydatetime() + timedelta(minutes=1)
    if first != start:
        raise ValueError("first kline is not aligned to the backtest start boundary")
    if last_close < start + timedelta(minutes=duration_minutes):
        raise ValueError("kline window does not cover the complete outcome horizon")
    numeric_columns = ["open", "high", "low", "close", "volume"]
    numeric = window[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not bool(np.isfinite(np.asarray(numeric, dtype=float)).all()):
        raise ValueError("kline window contains non-finite OHLCV values")
    return window


def summarize_final_pnl(outcomes: pd.DataFrame) -> dict[str, Any]:
    """Summarize every finite final net-PnL outcome without selection filters."""
    if outcomes.empty:
        return {
            "rows": 0,
            "positive_rows": 0,
            "zero_rows": 0,
            "negative_rows": 0,
            "net_pnl_usdt_sum": None,
            "net_pnl_usdt_mean": None,
            "net_pnl_usdt_median": None,
            "net_pnl_usdt_min": None,
            "net_pnl_usdt_max": None,
            "net_pnl_pct_mean": None,
            "net_pnl_pct_median": None,
            "net_pnl_pct_min": None,
            "net_pnl_pct_max": None,
        }
    _require_columns(outcomes, {"net_pnl", "net_pnl_pct"}, "outcome frame")
    net_pnl = cast(
        pd.Series,
        pd.to_numeric(cast(pd.Series, outcomes["net_pnl"]), errors="coerce"),
    )
    net_pnl_pct = cast(
        pd.Series,
        pd.to_numeric(cast(pd.Series, outcomes["net_pnl_pct"]), errors="coerce"),
    )
    if not bool(
        np.isfinite(np.asarray(net_pnl, dtype=float)).all()
        and np.isfinite(np.asarray(net_pnl_pct, dtype=float)).all()
    ):
        raise ValueError("Outcome frame contains non-finite final PnL values")
    return {
        "rows": int(len(outcomes)),
        "positive_rows": int(net_pnl.gt(0).sum()),
        "zero_rows": int(net_pnl.eq(0).sum()),
        "negative_rows": int(net_pnl.lt(0).sum()),
        "net_pnl_usdt_sum": float(net_pnl.sum()),
        "net_pnl_usdt_mean": float(net_pnl.mean()),
        "net_pnl_usdt_median": float(net_pnl.median()),
        "net_pnl_usdt_min": float(net_pnl.min()),
        "net_pnl_usdt_max": float(net_pnl.max()),
        "net_pnl_pct_mean": float(net_pnl_pct.mean()),
        "net_pnl_pct_median": float(net_pnl_pct.median()),
        "net_pnl_pct_min": float(net_pnl_pct.min()),
        "net_pnl_pct_max": float(net_pnl_pct.max()),
    }


def assert_diagnostic_output_root(path: Path, project_root: Path) -> Path:
    """Refuse canonical model/training trees for shadow-counterfactual output."""
    resolved = path.resolve()
    project = project_root.resolve()
    forbidden = (
        project / "data" / "backtest_candidates",
        project / "data" / "fastwin_dataset",
        project / "models",
        project / "artifacts" / "hmm",
        project / "artifacts" / "utility",
    )
    if any(resolved == item or item in resolved.parents for item in forbidden):
        raise ValueError(f"Shadow backtest output cannot use a governed path: {resolved}")
    if not any(part.lower() == "diagnostics" for part in resolved.parts):
        raise ValueError("Shadow backtest output path must include a diagnostics directory")
    return resolved
