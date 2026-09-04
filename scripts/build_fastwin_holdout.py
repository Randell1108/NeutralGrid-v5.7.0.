"""Build and execute an isolated temporal holdout for the FASTWIN target.

The production FASTWIN artifacts were trained from 24-hour candidate-time
geometric backtests and label a row positive when the net MTM PnL curve first
reaches +3% within seven hours.  This tool reproduces that label physics in an
audit directory only.  It never writes model, calibration, or deployment
artifacts.

The workflow is intentionally two phase::

    python scripts/build_fastwin_holdout.py freeze --output-dir outputs/audits/fastwin_holdout_...
    python scripts/build_fastwin_holdout.py run --output-dir outputs/audits/fastwin_holdout_...

``freeze`` fixes candidate identity and source hashes before outcomes are
observed.  ``run`` checkpoints one JSON file per candidate and is resumable.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.backtest.candidate_pipeline import (
    _parse_scan_timestamp,
    convert_to_training_row,
    fetch_historical_klines,
    filter_backtest_candidates,
    load_all_scanner_csvs,
    resolve_backtest_start_timestamp,
    run_single_backtest,
)
from neutralgrid.backtest.realism_governance import (
    CANDIDATE_TIME_GEOMETRIC_PROFILE,
    validate_realism_output_path,
)
from neutralgrid.core.constants import (
    ENGINE_VERSION,
    FORMULA_VERSION,
    LABEL_CONTRACT_VERSION,
)
from neutralgrid.models.meta_labeler import (
    ACTIVE_META_TARGET_CONTRACT,
    ACTIVE_SNAPSHOT_META_FEATURES,
    normalize_inference_feature_frame,
)
from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES

logger = logging.getLogger("build_fastwin_holdout")

HOLDOUT_SCHEMA_VERSION = 3
SUPPORTED_HOLDOUT_SCHEMA_VERSIONS = frozenset({1, 2, HOLDOUT_SCHEMA_VERSION})
FAST_TARGET_HOURS = 7.0
TARGET_PNL_PCT = 3.0
OBSERVATION_HOURS = 24
OBSERVATION_BARS = OBSERVATION_HOURS * 60
REALISM_PROFILE = CANDIDATE_TIME_GEOMETRIC_PROFILE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (str, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return cast(pd.Timestamp, value).isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _read_candidate_ids(paths: Iterable[Path]) -> set[str]:
    candidate_ids: set[str] = set()
    for path in paths:
        try:
            frame = pd.read_csv(
                path,
                usecols=lambda column: column == "candidate_id",
                low_memory=False,
            )
        except Exception as exc:
            raise ValueError(f"cannot read evaluated candidate IDs from {path}: {exc}") from exc
        if "candidate_id" not in frame.columns:
            raise ValueError(f"evaluated source missing candidate_id: {path}")
        values = cast(pd.Series, frame["candidate_id"]).dropna().astype(str).str.strip()
        candidate_ids.update(value for value in values if value and value.lower() != "nan")
    return candidate_ids


def _prior_holdout_sources(
    output_dir: Path,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Read sibling frozen cohorts as single-use evidence, failing closed.

    A completed or merely frozen audit has consumed its candidate identities.
    Those identities must never enter a later holdout, regardless of whether
    its materialized outcomes were copied to the general evaluated directory.
    """
    candidate_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    current_output = output_dir.resolve()
    for path in sorted(output_dir.parent.glob("fastwin_holdout_*/cohort.csv")):
        resolved = path.resolve()
        if current_output in resolved.parents:
            continue
        try:
            frame = pd.read_csv(
                path,
                usecols=lambda column: column == "candidate_id",
                low_memory=False,
            )
        except Exception as exc:
            raise ValueError(f"cannot read prior FASTWIN holdout cohort {path}: {exc}") from exc
        if "candidate_id" not in frame.columns:
            raise ValueError(f"prior FASTWIN holdout missing candidate_id: {path}")
        values = cast(pd.Series, frame["candidate_id"]).dropna().astype(str).str.strip()
        source_ids = {
            value for value in values if value and value.lower() != "nan"
        }
        if not source_ids:
            raise ValueError(f"prior FASTWIN holdout has no candidate IDs: {path}")
        candidate_ids.update(source_ids)
        records.append(
            {
                "path": str(resolved),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "candidate_ids": len(source_ids),
            }
        )
    return candidate_ids, records


def _training_cutoff(training_files: list[Path]) -> pd.Timestamp:
    timestamps: list[pd.Timestamp] = []
    for path in training_files:
        try:
            frame = pd.read_csv(
                path,
                usecols=lambda column: column == "backtest_timestamp",
                low_memory=False,
            )
        except Exception as exc:
            raise ValueError(f"cannot read training cutoff from {path}: {exc}") from exc
        if "backtest_timestamp" not in frame.columns:
            raise ValueError(f"training file missing backtest_timestamp: {path}")
        parsed = pd.to_datetime(frame["backtest_timestamp"], errors="coerce", utc=True)
        if bool(parsed.notna().any()):
            timestamps.append(cast(pd.Timestamp, parsed.max()))
    if not timestamps:
        raise ValueError("no finite backtest_timestamp in FASTWIN training files")
    return max(timestamps)


def _source_records(frame: pd.DataFrame, results_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in sorted(cast(pd.Series, frame["scan_file"]).dropna().astype(str).unique()):
        path = results_dir / name
        if not path.exists():
            raise FileNotFoundError(f"scanner source disappeared during freeze: {path}")
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )
    return records


def _implementation_records() -> list[dict[str, Any]]:
    paths = [
        Path(__file__).resolve(),
        (_REPO_ROOT / "src/neutralgrid/backtest/candidate_pipeline.py").resolve(),
        (_REPO_ROOT / "src/neutralgrid/core/constants.py").resolve(),
        (_REPO_ROOT / "backtest/btk_unified_runner.py").resolve(),
        (_REPO_ROOT / "backtest/backtest_realistic.py").resolve(),
    ]
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"holdout implementation source is missing: {path}")
        records.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def freeze_holdout(
    *,
    output_dir: Path,
    results_dir: Path,
    linkage_path: Path,
    fastwin_dir: Path,
    evaluated_dir: Path,
    as_of: datetime,
    min_rows: int,
    min_scan_groups: int,
) -> dict[str, Any]:
    """Freeze a never-evaluated post-training cohort and its provenance."""
    validate_realism_output_path(REALISM_PROFILE, output_dir)
    if output_dir.exists():
        raise FileExistsError(f"holdout output directory already exists: {output_dir}")
    training_files = sorted(fastwin_dir.glob("training_data_*.csv"))
    if not training_files:
        raise FileNotFoundError(f"no FASTWIN training_data_*.csv files in {fastwin_dir}")

    cutoff = _training_cutoff(training_files)
    evaluated_files = training_files + sorted(evaluated_dir.glob("*.csv"))
    evaluated_ids = _read_candidate_ids(evaluated_files)
    prior_holdout_ids, prior_holdout_records = _prior_holdout_sources(output_dir)
    evaluated_ids.update(prior_holdout_ids)

    loaded = load_all_scanner_csvs(results_dir)
    filtered = filter_backtest_candidates(
        loaded,
        linkage_path=linkage_path,
        min_score=45.0,
        bypass_only=False,
    )
    candidate_ids = cast(pd.Series, filtered["candidate_id"]).astype(str).str.strip()
    scan_times = pd.to_datetime(
        candidate_ids.map(lambda value: _parse_scan_timestamp(value)),
        errors="coerce",
        utc=True,
    )
    availability = pd.to_datetime(
        filtered["candidate_available_ts_utc"], errors="coerce", utc=True
    )
    maturity_cutoff = pd.Timestamp(as_of - timedelta(hours=OBSERVATION_HOURS))
    mask = (
        scan_times.gt(cutoff)
        & availability.gt(cutoff)
        & availability.le(maturity_cutoff)
        & ~candidate_ids.isin(evaluated_ids)
    )
    cohort = cast(pd.DataFrame, filtered.loc[mask].copy())
    cohort["holdout_scan_ts_utc"] = scan_times.loc[mask]
    cohort["holdout_available_ts_utc"] = availability.loc[mask]
    cohort = cast(
        pd.DataFrame,
        cohort.sort_values(
            ["holdout_available_ts_utc", "candidate_id"], kind="stable"
        ).reset_index(drop=True),
    )
    if cohort.empty:
        raise ValueError("frozen FASTWIN holdout cohort is empty")
    if bool(cast(pd.Series, cohort["candidate_id"]).duplicated().any()):
        raise ValueError("frozen FASTWIN holdout contains duplicate candidate_id values")

    if min_rows < 1 or min_scan_groups < 1:
        raise ValueError("minimum cohort rows and scan groups must both be positive")
    scan_group_count = int(cast(pd.Series, cohort["scan_file"]).nunique())
    if len(cohort) < min_rows or scan_group_count < min_scan_groups:
        raise ValueError(
            "fresh FASTWIN cohort is below the pre-registered evidence floor: "
            f"rows={len(cohort)}<{min_rows} or "
            f"scan_groups={scan_group_count}<{min_scan_groups}"
        )

    normalized = normalize_inference_feature_frame(cohort)
    missing_counts: dict[str, int] = {}
    required_features = list(
        dict.fromkeys([*ACTIVE_SNAPSHOT_META_FEATURES, *DEFAULT_FEATURES])
    )
    for feature in required_features:
        if feature not in normalized.columns:
            missing_counts[feature] = len(cohort)
            continue
        numeric = pd.to_numeric(normalized[feature], errors="coerce")
        missing = int((~np.isfinite(np.asarray(numeric, dtype=float))).sum())
        if missing:
            missing_counts[feature] = missing
    if missing_counts:
        raise ValueError(f"frozen cohort is not feature-complete: {missing_counts}")

    output_dir.mkdir(parents=True, exist_ok=False)
    cohort_path = output_dir / "cohort.csv"
    cohort.to_csv(cohort_path, index=False)
    source_records = _source_records(cohort, results_dir)
    model_paths = [
        Path("models/meta_labeler.pkl"),
        Path("models/meta_labeler/metadata.json"),
    ]
    model_records = [
        {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in model_paths
        if path.exists()
    ]
    manifest: dict[str, Any] = {
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "created_at_utc": as_of.isoformat(),
        "training_cutoff_utc": cutoff.isoformat(),
        "maturity_cutoff_utc": cast(pd.Timestamp, maturity_cutoff).isoformat(),
        "selection": {
            "loaded_rows": len(loaded),
            "canonically_eligible_rows": len(filtered),
            "known_evaluated_candidate_ids": len(evaluated_ids),
            "known_prior_holdout_candidate_ids": len(prior_holdout_ids),
            "frozen_rows": len(cohort),
            "unique_candidate_ids": int(cast(pd.Series, cohort["candidate_id"]).nunique()),
            "unique_symbols": int(cast(pd.Series, cohort["symbol"]).nunique()),
            "unique_scan_groups": scan_group_count,
            "minimum_rows_required": min_rows,
            "minimum_scan_groups_required": min_scan_groups,
        },
        "feature_contract": {
            "meta_features": list(ACTIVE_SNAPSHOT_META_FEATURES),
            "profile_features": list(DEFAULT_FEATURES),
            "finite_required": True,
            "imputation_allowed": False,
        },
        "target_contract": {
            "name": ACTIVE_META_TARGET_CONTRACT,
            "positive_rule": "time_to_target_hours <= 7.0",
            "threshold_pct_of_capital": TARGET_PNL_PCT,
            "target_hours": FAST_TARGET_HOURS,
            "observation_hours": OBSERVATION_HOURS,
            "observation_bars": OBSERVATION_BARS,
        },
        "engine_contract": {
            "engine_version": ENGINE_VERSION,
            "label_contract_version": LABEL_CONTRACT_VERSION,
            "formula_version": FORMULA_VERSION,
            "realism_profile": REALISM_PROFILE,
            "capital": 400.0,
            "leverage": 10,
        },
        "cohort": {
            "path": str(cohort_path.resolve()),
            "sha256": _sha256(cohort_path),
            "size_bytes": cohort_path.stat().st_size,
        },
        "scanner_sources": source_records,
        "implementation_sources": _implementation_records(),
        "prior_holdout_sources": prior_holdout_records,
        "incumbent_artifacts": model_records,
        "fastwin_training_sources": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in training_files
        ],
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def _validate_frozen_contract(output_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing frozen manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in SUPPORTED_HOLDOUT_SCHEMA_VERSIONS:
        raise ValueError("unsupported holdout manifest schema")
    expected_engine = {
        "engine_version": ENGINE_VERSION,
        "label_contract_version": LABEL_CONTRACT_VERSION,
        "formula_version": FORMULA_VERSION,
        "realism_profile": REALISM_PROFILE,
        "capital": 400.0,
        "leverage": 10,
    }
    if manifest.get("engine_contract") != expected_engine:
        raise ValueError(
            "frozen engine contract differs from current code; refusing mixed-version resume"
        )
    cohort_info = manifest["cohort"]
    cohort_path = Path(cohort_info["path"])
    if not cohort_path.exists() or _sha256(cohort_path) != cohort_info["sha256"]:
        raise ValueError("frozen cohort hash mismatch")
    for source in manifest.get("scanner_sources", []):
        path = Path(source["path"])
        if not path.exists() or _sha256(path) != source["sha256"]:
            raise ValueError(f"frozen scanner source hash mismatch: {path}")
    if int(manifest.get("schema_version", 0)) >= 3:
        implementation_sources = manifest.get("implementation_sources", [])
        if not implementation_sources:
            raise ValueError("frozen manifest lacks implementation source hashes")
        for source in implementation_sources:
            path = Path(source["path"])
            if not path.exists() or _sha256(path) != source["sha256"]:
                raise ValueError(f"frozen implementation hash mismatch: {path}")
    cohort = pd.read_csv(cohort_path, low_memory=False)
    if len(cohort) != int(manifest["selection"]["frozen_rows"]):
        raise ValueError("frozen cohort row count mismatch")
    return manifest, cohort


def _checkpoint_path(output_dir: Path, candidate_id: str) -> Path:
    safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in candidate_id)
    return output_dir / "rows" / f"{safe_id}.json"


def _load_checkpoints(output_dir: Path) -> dict[str, dict[str, Any]]:
    checkpoints: dict[str, dict[str, Any]] = {}
    for path in sorted((output_dir / "rows").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate_id = str(payload.get("candidate_id", ""))
        if candidate_id:
            checkpoints[candidate_id] = payload
    return checkpoints


def _materialize(output_dir: Path) -> dict[str, int]:
    checkpoints = _load_checkpoints(output_dir)
    ok = [payload for payload in checkpoints.values() if payload.get("status") == "ok"]
    errors = [payload for payload in checkpoints.values() if payload.get("status") != "ok"]
    pd.DataFrame([payload["backtest_result"] for payload in ok]).to_csv(
        output_dir / "backtest_results.csv", index=False
    )
    pd.DataFrame([payload["training_row"] for payload in ok]).to_csv(
        output_dir / "training_data.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "candidate_id": payload.get("candidate_id"),
                "symbol": payload.get("symbol"),
                "stage": payload.get("stage"),
                "error": payload.get("error"),
                "attempted_at_utc": payload.get("attempted_at_utc"),
            }
            for payload in errors
        ]
    ).to_csv(output_dir / "errors.csv", index=False)
    return {"checkpointed": len(checkpoints), "ok": len(ok), "errors": len(errors)}


async def run_holdout(
    *,
    output_dir: Path,
    delay_seconds: float,
    max_candidates: int,
) -> dict[str, int]:
    """Execute or resume the frozen cohort without production side effects."""
    validate_realism_output_path(REALISM_PROFILE, output_dir)
    _, cohort = _validate_frozen_contract(output_dir)
    checkpoints = _load_checkpoints(output_dir)
    pending = [
        row
        for _, row in cohort.iterrows()
        if checkpoints.get(str(row["candidate_id"]), {}).get("status") != "ok"
    ]
    if max_candidates > 0:
        pending = pending[:max_candidates]
    logger.info(
        "FASTWIN holdout: frozen=%d successful=%d pending_this_run=%d",
        len(cohort),
        sum(payload.get("status") == "ok" for payload in checkpoints.values()),
        len(pending),
    )
    client = BinanceClient()
    try:
        for position, row in enumerate(pending, start=1):
            candidate = cast(dict[str, Any], row.to_dict())
            candidate_id = str(candidate["candidate_id"])
            symbol = str(candidate["symbol"])
            checkpoint = _checkpoint_path(output_dir, candidate_id)
            attempted_at = datetime.now(timezone.utc).isoformat()
            logger.info("[%d/%d] %s %s", position, len(pending), symbol, candidate_id)
            try:
                start = resolve_backtest_start_timestamp(candidate)
                klines = await fetch_historical_klines(
                    client,
                    symbol,
                    start,
                    hours=OBSERVATION_HOURS,
                )
                if len(klines) < OBSERVATION_BARS:
                    raise ValueError(
                        f"insufficient_1m_bars:{len(klines)}<{OBSERVATION_BARS}"
                    )
                result = run_single_backtest(
                    candidate_row=candidate,
                    klines_df=klines,
                    capital=400.0,
                    leverage=10,
                    max_holding_bars=OBSERVATION_BARS,
                    realism_profile=REALISM_PROFILE,
                )
                result["backtest_start_ts_utc"] = start.isoformat()
                result["backtest_time_anchor"] = str(
                    candidate.get("candidate_available_source", "deployment_ready_csv_mtime")
                )
                training_row = convert_to_training_row(
                    backtest_result=result,
                    candidate_row=candidate,
                )
                payload = {
                    "status": "ok",
                    "candidate_id": candidate_id,
                    "symbol": symbol,
                    "attempted_at_utc": attempted_at,
                    "backtest_result": result,
                    "training_row": training_row,
                }
            except Exception as exc:
                payload = {
                    "status": "error",
                    "candidate_id": candidate_id,
                    "symbol": symbol,
                    "stage": "fetch_or_backtest",
                    "error": f"{type(exc).__name__}: {exc}",
                    "attempted_at_utc": attempted_at,
                }
                logger.warning("Holdout row failed for %s: %s", candidate_id, exc)
            _atomic_write_json(checkpoint, payload)
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
    finally:
        await client.close()
    summary = _materialize(output_dir)
    _atomic_write_json(
        output_dir / "run_summary.json",
        {
            **summary,
            "frozen_rows": len(cohort),
            "complete": summary["ok"] == len(cohort),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--output-dir", type=Path, required=True)
    freeze_parser.add_argument("--results-dir", type=Path, default=Path("results"))
    freeze_parser.add_argument(
        "--linkage-path",
        type=Path,
        default=Path("data/linkage/deploy_linkage_log.csv"),
    )
    freeze_parser.add_argument(
        "--fastwin-dir", type=Path, default=Path("data/fastwin_dataset")
    )
    freeze_parser.add_argument(
        "--evaluated-dir", type=Path, default=Path("data/backtest_candidates")
    )
    freeze_parser.add_argument(
        "--min-rows",
        type=int,
        default=150,
        help="Fail before freezing when fewer genuinely new rows are available.",
    )
    freeze_parser.add_argument(
        "--min-scan-groups",
        type=int,
        default=10,
        help="Fail before freezing when temporal scan-group diversity is too small.",
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--delay", type=float, default=0.2)
    run_parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Bound this invocation for diagnostics; zero runs every pending row.",
    )
    args = parser.parse_args()
    try:
        validate_realism_output_path(REALISM_PROFILE, args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    if args.command == "freeze":
        manifest = freeze_holdout(
            output_dir=args.output_dir,
            results_dir=args.results_dir,
            linkage_path=args.linkage_path,
            fastwin_dir=args.fastwin_dir,
            evaluated_dir=args.evaluated_dir,
            as_of=datetime.now(timezone.utc),
            min_rows=args.min_rows,
            min_scan_groups=args.min_scan_groups,
        )
        logger.info("Frozen %d rows", manifest["selection"]["frozen_rows"])
    else:
        summary = asyncio.run(
            run_holdout(
                output_dir=args.output_dir,
                delay_seconds=args.delay,
                max_candidates=args.max_candidates,
            )
        )
        logger.info("Run summary: %s", summary)


if __name__ == "__main__":
    main()
