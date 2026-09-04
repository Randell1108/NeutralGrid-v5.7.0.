#!/usr/bin/env python3
"""Backtest matured shadow-approved candidates into a quarantined raw pool.

The runner discovers full-pipeline deployment CSVs with companion diagnostic
shadow reports, selects rows rejected only by the missing authoritative meta
gate, waits for an exact 362-minute outcome window, and delegates simulation to
the canonical ``run_single_backtest``/``btk_unified_runner.run_backtest`` path.

Outputs are intentionally stored below ``artifacts/diagnostics`` and are not
automatically consumed by any governed retraining or promotion path.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any, cast

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.backtest.candidate_pipeline import (
    convert_to_training_row,
    fetch_historical_klines,
    resolve_backtest_start_timestamp,
    run_single_backtest,
)
from neutralgrid.backtest.shadow_approved import (
    DEFAULT_DURATION_MINUTES,
    SHADOW_BACKTEST_CONTRACT,
    ShadowRunSources,
    assert_diagnostic_output_root,
    backtest_window_start_utc,
    discover_shadow_runs,
    is_run_mature,
    select_shadow_approved_candidates,
    sha256_file,
    summarize_final_pnl,
    validate_kline_window,
    validate_shadow_sources,
)
from neutralgrid.core.config import get_config
from neutralgrid.core.constants import (
    ENGINE_VERSION,
    FORMULA_VERSION,
    LABEL_CONTRACT_VERSION,
)


logger = logging.getLogger("shadow_approved_backtest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest matured candidates that would have passed all recorded "
            "gates if the diagnostic shadow meta probability were authoritative."
        )
    )
    parser.add_argument(
        "--pipeline-runs-root",
        default="artifacts/pipeline_runs",
        help="Root containing immutable full-pipeline run directories.",
    )
    parser.add_argument(
        "--shadow-root",
        default="artifacts/diagnostics/meta_prob_shadow",
        help="Root containing pipeline_<timestamp> shadow reports.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/diagnostics/shadow_approved_backtests",
        help="Diagnostic-only output root; governed training paths are refused.",
    )
    parser.add_argument(
        "--duration-minutes",
        type=int,
        default=DEFAULT_DURATION_MINUTES,
        help="Fixed outcome window in one-minute bars (default: 362 = 6h02m).",
    )
    parser.add_argument(
        "--min-meta-prob",
        type=float,
        default=None,
        help="Counterfactual meta threshold; defaults to configured Stage-B threshold.",
    )
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument(
        "--run-key",
        default="",
        help="Optional pipeline_<timestamp> key to process exclusively.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate sources and report selection/maturity without network calls or writes.",
    )
    args = parser.parse_args()
    if args.duration_minutes <= 0:
        parser.error("--duration-minutes must be positive")
    if args.duration_minutes > 1500:
        parser.error("--duration-minutes cannot exceed the Binance single-request limit of 1500")
    if args.min_meta_prob is not None and not 0.0 <= args.min_meta_prob <= 1.0:
        parser.error("--min-meta-prob must be in [0, 1]")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    return args


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _load_unique_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path, low_memory=False)
    if "candidate_id" not in frame.columns:
        raise ValueError(f"{label} is missing candidate_id: {path}")
    ids = cast(pd.Series, frame["candidate_id"]).fillna("").astype(str).str.strip()
    if bool(ids.eq("").any() or ids.duplicated().any()):
        raise ValueError(f"{label} has blank or duplicate candidate IDs: {path}")
    frame["candidate_id"] = ids
    return frame


def _sanitize_error(exc: Exception) -> str:
    text = str(exc).replace("\r", " ").replace("\n", " ")
    if "?" in text:
        text = text.split("?", 1)[0] + "?[query redacted]"
    return text[:500]


def _prepare_selection(
    sources: ShadowRunSources,
    *,
    min_meta_prob: float,
    duration_minutes: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    shadow_manifest = validate_shadow_sources(sources)
    deployment = pd.read_csv(sources.deployment_path, low_memory=False)
    shadow = pd.read_csv(sources.shadow_path, low_memory=False)
    selected = select_shadow_approved_candidates(
        deployment,
        shadow,
        min_meta_prob=min_meta_prob,
    )
    available = sources.candidate_available_ts_utc.astimezone(timezone.utc)
    backtest_start = backtest_window_start_utc(available)
    selected["candidate_available_ts_utc"] = available.isoformat()
    selected["candidate_available_source"] = "deployment_ready_csv_mtime"
    selected["backtest_start_ts_utc"] = backtest_start.isoformat()
    selected["outcome_window_end_utc"] = (
        backtest_start + timedelta(minutes=duration_minutes)
    ).isoformat()
    selected["bot_duration_minutes"] = int(duration_minutes)
    selected["bot_duration_hours"] = float(duration_minutes / 60.0)
    selected["source_pipeline_run_key"] = sources.run_key
    selected["source_deployment_sha256"] = sha256_file(sources.deployment_path)
    selected["source_shadow_sha256"] = sha256_file(sources.shadow_path)
    selected["source_shadow_manifest_sha256"] = sha256_file(
        sources.shadow_manifest_path
    )
    selected["backtest_realism_profile"] = "legacy"
    return selected, shadow_manifest


def _preserve_selection(path: Path, selected: pd.DataFrame) -> pd.DataFrame:
    if not path.is_file():
        _atomic_write_csv(path, selected)
        return selected
    existing = _load_unique_csv(path, "preserved selection")
    existing_ids = sorted(existing["candidate_id"].astype(str).tolist())
    selected_ids = sorted(selected["candidate_id"].astype(str).tolist())
    if existing_ids != selected_ids:
        raise ValueError(
            f"Preserved selection differs from current verified sources: {path}"
        )
    for column in (
        "source_deployment_sha256",
        "source_shadow_sha256",
        "counterfactual_meta_threshold",
        "bot_duration_minutes",
    ):
        if column not in existing.columns:
            raise ValueError(f"Preserved selection is missing {column}: {path}")
        old_values = sorted(existing[column].astype(str).unique().tolist())
        new_values = sorted(selected[column].astype(str).unique().tolist())
        if old_values != new_values:
            raise ValueError(f"Preserved selection provenance changed for {column}: {path}")
    return existing


async def _backtest_candidate(
    client: BinanceClient,
    row: pd.Series,
    *,
    duration_minutes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = cast(dict[str, Any], row.to_dict())
    candidate_id = str(candidate["candidate_id"])
    symbol = str(candidate["symbol"])
    capital = float(candidate["capital_base_usdt"])
    leverage = int(float(candidate["leverage"]))
    start_utc = resolve_backtest_start_timestamp(candidate)
    fetch_hours = int(math.ceil(duration_minutes / 60.0))
    klines = await fetch_historical_klines(
        client,
        symbol,
        start_utc,
        hours=fetch_hours,
    )
    window = validate_kline_window(
        klines,
        start_utc=start_utc,
        duration_minutes=duration_minutes,
    )
    result = run_single_backtest(
        candidate_row=candidate,
        klines_df=window,
        capital=capital,
        leverage=leverage,
        max_holding_bars=duration_minutes,
        duration_bars=duration_minutes,
        realism_profile="legacy",
    )
    result["candidate_id"] = candidate_id
    result["symbol"] = symbol
    result["candidate_available_ts_utc"] = candidate["candidate_available_ts_utc"]
    result["backtest_start_ts_utc"] = start_utc.isoformat()
    result["outcome_window_end_utc"] = (
        start_utc + timedelta(minutes=duration_minutes)
    ).isoformat()
    result["bot_duration_minutes"] = duration_minutes
    result["bot_duration_hours"] = duration_minutes / 60.0
    result["shadow_meta_prob_diagnostic"] = candidate.get(
        "shadow_meta_prob_diagnostic"
    )
    result["source_pipeline_run_key"] = candidate.get("source_pipeline_run_key")
    result["source_deployment_sha256"] = candidate.get("source_deployment_sha256")
    result["source_shadow_sha256"] = candidate.get("source_shadow_sha256")
    result["shadow_backtest_contract"] = SHADOW_BACKTEST_CONTRACT
    result["source_class"] = SHADOW_BACKTEST_CONTRACT
    result["unfiltered_outcome_pool"] = True
    result["governed_training_eligible"] = False
    result["promotion_eligible"] = False
    result["backtest_realism_profile"] = "legacy"
    result["engine_version"] = ENGINE_VERSION
    result["formula_version"] = FORMULA_VERSION
    result["label_contract_version"] = LABEL_CONTRACT_VERSION
    for column, value in candidate.items():
        if column not in result:
            result[column] = value

    training = convert_to_training_row(
        backtest_result=result,
        candidate_row=candidate,
        horizon_hours=duration_minutes / 60.0,
    )
    training["candidate_id"] = candidate_id
    training["symbol"] = symbol
    training["candidate_available_ts_utc"] = candidate[
        "candidate_available_ts_utc"
    ]
    training["backtest_start_ts_utc"] = start_utc.isoformat()
    training["outcome_window_end_utc"] = result["outcome_window_end_utc"]
    training["bot_duration_minutes"] = duration_minutes
    training["bot_duration_hours"] = duration_minutes / 60.0
    training["shadow_meta_prob_diagnostic"] = result[
        "shadow_meta_prob_diagnostic"
    ]
    training["source_pipeline_run_key"] = result["source_pipeline_run_key"]
    training["source_deployment_sha256"] = result["source_deployment_sha256"]
    training["source_shadow_sha256"] = result["source_shadow_sha256"]
    training["shadow_backtest_contract"] = SHADOW_BACKTEST_CONTRACT
    training["source_class"] = SHADOW_BACKTEST_CONTRACT
    training["unfiltered_outcome_pool"] = True
    training["governed_training_eligible"] = False
    training["promotion_eligible"] = False
    training["backtest_realism_profile"] = "legacy"
    training["engine_version"] = ENGINE_VERSION
    training["formula_version"] = FORMULA_VERSION
    training["label_contract_version"] = LABEL_CONTRACT_VERSION
    result["y"] = training.get("y")
    return result, training


def _refresh_unfiltered_pool(
    output_root: Path,
    *,
    duration_minutes: int,
) -> dict[str, Any]:
    training_frames: list[pd.DataFrame] = []
    outcome_frames: list[pd.DataFrame] = []
    run_keys: list[str] = []
    for path in sorted(output_root.glob("pipeline_*/training_data_unfiltered.csv")):
        training_frame = _load_unique_csv(path, "per-run unfiltered training data")
        outcome_path = path.parent / "unfiltered_outcomes.csv"
        if not outcome_path.is_file():
            raise FileNotFoundError(
                f"Per-run outcome file is missing for aggregate pool: {outcome_path}"
            )
        outcome_frame = _load_unique_csv(outcome_path, "per-run unfiltered outcomes")
        if set(training_frame["candidate_id"].astype(str)) != set(
            outcome_frame["candidate_id"].astype(str)
        ):
            raise ValueError(f"Per-run outcome/training candidate-ID mismatch: {path.parent}")
        if training_frame.empty:
            continue
        training_frames.append(training_frame)
        outcome_frames.append(outcome_frame)
        run_keys.append(path.parent.name)
    if training_frames:
        pool = pd.concat(training_frames, ignore_index=True, sort=False)
        outcome_pool = pd.concat(outcome_frames, ignore_index=True, sort=False)
        ids = cast(pd.Series, pool["candidate_id"]).fillna("").astype(str).str.strip()
        if bool(ids.eq("").any() or ids.duplicated().any()):
            duplicates = sorted(ids.loc[ids.duplicated(keep=False)].unique().tolist())
            raise ValueError(
                "Aggregate unfiltered pool has duplicate candidate IDs: "
                f"{duplicates[:5]}"
            )
        pool["candidate_id"] = ids
        pool = cast(pd.DataFrame, pool.sort_values("candidate_id").reset_index(drop=True))
        outcome_ids = (
            cast(pd.Series, outcome_pool["candidate_id"])
            .fillna("")
            .astype(str)
            .str.strip()
        )
        if bool(outcome_ids.eq("").any() or outcome_ids.duplicated().any()):
            raise ValueError("Aggregate unfiltered outcome pool has invalid candidate IDs")
        outcome_pool["candidate_id"] = outcome_ids
        outcome_pool = cast(
            pd.DataFrame,
            outcome_pool.sort_values("candidate_id").reset_index(drop=True),
        )
        if outcome_pool["candidate_id"].tolist() != pool["candidate_id"].tolist():
            raise ValueError("Aggregate outcome/training candidate-ID mismatch")
    else:
        pool = pd.DataFrame()
        outcome_pool = pd.DataFrame()

    pool_path = output_root / "unfiltered_training_pool.csv"
    outcome_pool_path = output_root / "unfiltered_outcome_pool.csv"
    _atomic_write_csv(pool_path, pool)
    _atomic_write_csv(outcome_pool_path, outcome_pool)
    hmm_versions: list[str] = []
    for column in ("hmm_artifact_version", "snapshot_hmm_artifact_version"):
        if column in pool.columns:
            hmm_versions.extend(
                cast(pd.Series, pool[column])
                .dropna()
                .astype(str)
                .str.strip()
                .loc[lambda values: values.ne("")]
                .unique()
                .tolist()
            )
    manifest = {
        "contract": SHADOW_BACKTEST_CONTRACT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(pool)),
        "candidate_ids_unique": int(pool["candidate_id"].nunique())
        if "candidate_id" in pool.columns
        else 0,
        "source_run_keys": run_keys,
        "hmm_artifact_versions": sorted(set(hmm_versions)),
        "duration_minutes": duration_minutes,
        "filtered_by_outcome": False,
        "diagnostic_shadow_selection": True,
        "governed_training_eligible": False,
        "promotion_eligible": False,
        "automatic_training_ingestion": False,
        "final_pnl_summary": summarize_final_pnl(outcome_pool),
        "pool_file": str(pool_path.resolve()),
        "pool_sha256": sha256_file(pool_path),
        "outcome_pool_file": str(outcome_pool_path.resolve()),
        "outcome_pool_sha256": sha256_file(outcome_pool_path),
    }
    _atomic_write_json(output_root / "unfiltered_training_pool.manifest.json", manifest)
    return manifest


async def _process_run(
    sources: ShadowRunSources,
    *,
    selected: pd.DataFrame,
    shadow_manifest: dict[str, Any],
    output_root: Path,
    duration_minutes: int,
    min_meta_prob: float,
    delay: float,
) -> dict[str, Any]:
    run_dir = output_root / sources.run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    selection_path = run_dir / "selected_candidates.csv"
    results_path = run_dir / "unfiltered_outcomes.csv"
    training_path = run_dir / "training_data_unfiltered.csv"
    failures_path = run_dir / "failures.csv"
    manifest_path = run_dir / "run_manifest.json"

    selected = _preserve_selection(selection_path, selected)
    existing_results = _load_unique_csv(results_path, "existing outcomes")
    existing_training = _load_unique_csv(training_path, "existing training data")
    result_ids = (
        set(existing_results["candidate_id"].astype(str))
        if not existing_results.empty
        else set()
    )
    training_ids = (
        set(existing_training["candidate_id"].astype(str))
        if not existing_training.empty
        else set()
    )
    if result_ids != training_ids:
        raise ValueError(f"Outcome/training candidate-ID mismatch in {run_dir}")

    pending = cast(
        pd.DataFrame,
        selected.loc[
            ~selected["candidate_id"].astype(str).isin(sorted(result_ids))
        ].copy(),
    )
    new_results: list[dict[str, Any]] = []
    new_training: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    client = BinanceClient()
    try:
        for position, (_, row) in enumerate(pending.iterrows(), start=1):
            candidate_id = str(row["candidate_id"])
            symbol = str(row["symbol"])
            logger.info(
                "%s [%d/%d] backtesting %s (%s)",
                sources.run_key,
                position,
                len(pending),
                symbol,
                candidate_id,
            )
            try:
                result, training = await _backtest_candidate(
                    client,
                    cast(pd.Series, row),
                    duration_minutes=duration_minutes,
                )
                new_results.append(result)
                new_training.append(training)
            except Exception as exc:
                logger.warning("%s failed: %s", candidate_id, _sanitize_error(exc))
                failures.append(
                    {
                        "candidate_id": candidate_id,
                        "symbol": symbol,
                        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "error_type": type(exc).__name__,
                        "error": _sanitize_error(exc),
                        "retryable": True,
                    }
                )
            if delay:
                await asyncio.sleep(delay)
    finally:
        await client.close()

    result_frames = [
        frame
        for frame in (existing_results, pd.DataFrame(new_results))
        if not frame.empty
    ]
    training_frames = [
        frame
        for frame in (existing_training, pd.DataFrame(new_training))
        if not frame.empty
    ]
    combined_results = (
        pd.concat(result_frames, ignore_index=True, sort=False)
        if result_frames
        else pd.DataFrame({"candidate_id": pd.Series(dtype="string")})
    )
    combined_training = (
        pd.concat(training_frames, ignore_index=True, sort=False)
        if training_frames
        else pd.DataFrame({"candidate_id": pd.Series(dtype="string")})
    )
    for label, frame in (
        ("outcomes", combined_results),
        ("training data", combined_training),
    ):
        ids = cast(pd.Series, frame["candidate_id"]).fillna("").astype(str).str.strip()
        if bool(ids.eq("").any() or ids.duplicated().any()):
            raise ValueError(f"Combined {label} contains blank or duplicate candidate IDs")
        frame["candidate_id"] = ids
    combined_results = cast(
        pd.DataFrame,
        combined_results.sort_values("candidate_id").reset_index(drop=True),
    )
    combined_training = cast(
        pd.DataFrame,
        combined_training.sort_values("candidate_id").reset_index(drop=True),
    )
    _atomic_write_csv(results_path, combined_results)
    _atomic_write_csv(training_path, combined_training)
    _atomic_write_csv(failures_path, pd.DataFrame(failures))

    selected_count = int(len(selected))
    success_count = int(len(combined_results))
    missing_ids = sorted(
        set(selected["candidate_id"].astype(str)).difference(
            combined_results["candidate_id"].astype(str)
        )
    )
    manifest = {
        "contract": SHADOW_BACKTEST_CONTRACT,
        "status": "complete" if not missing_ids else "partial_retryable",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_pipeline_run_key": sources.run_key,
        "source_deployment_path": str(sources.deployment_path),
        "source_deployment_sha256": sha256_file(sources.deployment_path),
        "source_shadow_path": str(sources.shadow_path),
        "source_shadow_sha256": sha256_file(sources.shadow_path),
        "source_shadow_manifest_path": str(sources.shadow_manifest_path),
        "source_shadow_manifest_sha256": sha256_file(sources.shadow_manifest_path),
        "teacher_hmm_artifact_version": shadow_manifest.get(
            "teacher_hmm_artifact_version"
        ),
        "snapshot_hmm_artifact_version": shadow_manifest.get(
            "snapshot_hmm_artifact_version"
        ),
        "hmm_lineage_matches_teacher": shadow_manifest.get(
            "hmm_lineage_matches_teacher"
        ),
        "candidate_available_ts_utc": sources.candidate_available_ts_utc.isoformat(),
        "backtest_start_ts_utc": backtest_window_start_utc(
            sources.candidate_available_ts_utc
        ).isoformat(),
        "outcome_window_end_utc": (
            backtest_window_start_utc(sources.candidate_available_ts_utc)
            + timedelta(minutes=duration_minutes)
        ).isoformat(),
        "duration_minutes": duration_minutes,
        "duration_hours": duration_minutes / 60.0,
        "min_shadow_meta_prob": min_meta_prob,
        "selected_rows": selected_count,
        "successful_rows": success_count,
        "failed_rows_this_attempt": len(failures),
        "remaining_candidate_ids": missing_ids,
        "final_pnl_summary": summarize_final_pnl(combined_results),
        "filtered_by_outcome": False,
        "diagnostic_shadow_selection": True,
        "governed_training_eligible": False,
        "promotion_eligible": False,
        "automatic_training_ingestion": False,
        "backtest_realism_profile": "legacy",
        "execution_terms_source": "recorded_candidate_fields",
        "engine_version": ENGINE_VERSION,
        "formula_version": FORMULA_VERSION,
        "label_contract_version": LABEL_CONTRACT_VERSION,
        "selection_file": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "outcomes_file": str(results_path.resolve()),
        "outcomes_sha256": sha256_file(results_path),
        "training_file": str(training_path.resolve()),
        "training_sha256": sha256_file(training_path),
        "failures_file": str(failures_path.resolve()),
        "failures_sha256": sha256_file(failures_path),
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


async def main() -> int:
    args = parse_args()
    output_root = assert_diagnostic_output_root(
        Path(args.output_root),
        PROJECT_ROOT,
    )
    min_meta_prob = (
        float(args.min_meta_prob)
        if args.min_meta_prob is not None
        else float(get_config().two_stage.min_meta_prob)
    )
    sources_list = discover_shadow_runs(
        Path(args.pipeline_runs_root),
        Path(args.shadow_root),
    )
    if args.run_key:
        sources_list = [item for item in sources_list if item.run_key == args.run_key]
        if not sources_list:
            raise ValueError(f"Requested run key not found with a shadow report: {args.run_key}")

    now_utc = datetime.now(timezone.utc)
    summaries: list[dict[str, Any]] = []
    for sources in sources_list:
        selected, shadow_manifest = _prepare_selection(
            sources,
            min_meta_prob=min_meta_prob,
            duration_minutes=args.duration_minutes,
        )
        maturity_utc = backtest_window_start_utc(
            sources.candidate_available_ts_utc
        ) + timedelta(minutes=args.duration_minutes)
        mature = is_run_mature(
            sources,
            now_utc=now_utc,
            duration_minutes=args.duration_minutes,
        )
        summary: dict[str, Any] = {
            "run_key": sources.run_key,
            "selected_rows": int(len(selected)),
            "mature": mature,
            "maturity_utc": maturity_utc.isoformat(),
        }
        if args.dry_run:
            summary["status"] = "dry_run_mature" if mature else "dry_run_pending_maturity"
            summaries.append(summary)
            continue
        if not mature:
            summary["status"] = "pending_maturity"
            summaries.append(summary)
            continue

        existing_manifest_path = output_root / sources.run_key / "run_manifest.json"
        if existing_manifest_path.is_file():
            existing_manifest = json.loads(
                existing_manifest_path.read_text(encoding="utf-8")
            )
            if (
                existing_manifest.get("status") == "complete"
                and existing_manifest.get("source_deployment_sha256")
                == sha256_file(sources.deployment_path)
                and existing_manifest.get("source_shadow_sha256")
                == sha256_file(sources.shadow_path)
                and int(existing_manifest.get("duration_minutes", -1))
                == int(args.duration_minutes)
            ):
                summary["status"] = "already_complete"
                summary["successful_rows"] = int(
                    existing_manifest.get("successful_rows", 0)
                )
                summaries.append(summary)
                continue

        manifest = await _process_run(
            sources,
            selected=selected,
            shadow_manifest=shadow_manifest,
            output_root=output_root,
            duration_minutes=args.duration_minutes,
            min_meta_prob=min_meta_prob,
            delay=args.delay,
        )
        summary["status"] = manifest["status"]
        summary["successful_rows"] = manifest["successful_rows"]
        summary["failed_rows_this_attempt"] = manifest["failed_rows_this_attempt"]
        summaries.append(summary)

    pool_manifest: dict[str, Any] | None = None
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        pool_manifest = _refresh_unfiltered_pool(
            output_root,
            duration_minutes=args.duration_minutes,
        )
    report = {
        "contract": SHADOW_BACKTEST_CONTRACT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_minutes": args.duration_minutes,
        "duration_hours": args.duration_minutes / 60.0,
        "min_shadow_meta_prob": min_meta_prob,
        "dry_run": bool(args.dry_run),
        "runs": summaries,
        "pool_manifest": pool_manifest,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _acquire_lock(output_root: Path) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".backtest.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another shadow-approved backtest run holds the lock: {lock_path}"
        ) from exc
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    return descriptor


if __name__ == "__main__":
    parsed_output_root = Path("artifacts/diagnostics/shadow_approved_backtests")
    if "--output-root" in sys.argv:
        index = sys.argv.index("--output-root")
        if index + 1 < len(sys.argv):
            parsed_output_root = Path(sys.argv[index + 1])
    lock_descriptor: int | None = None
    lock_path = parsed_output_root / ".backtest.lock"
    try:
        if "--dry-run" not in sys.argv:
            lock_descriptor = _acquire_lock(parsed_output_root)
        raise SystemExit(asyncio.run(main()))
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
            lock_path.unlink(missing_ok=True)
