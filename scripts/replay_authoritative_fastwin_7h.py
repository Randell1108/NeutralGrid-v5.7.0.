#!/usr/bin/env python3
"""Rebuild an exact historical candidate cohort with seven-hour observability.

This is deliberately distinct from ``backtest_candidates.py``: it never
rescans today's universe.  For every stored candidate it fetches a 420-bar
window from the stored ``backtest_start_ts_utc``, first proves that the first
360 bars reproduce the stored result, then emits the 420-bar outcome.  A new
source directory is atomically published only when *every* stored candidate is
reproduced; otherwise durable checkpoint files name the failing candidates.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any, Awaitable, Callable, cast

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.backtest.candidate_pipeline import (
    convert_to_training_row,
    fetch_historical_klines,
    run_single_backtest,
)


logger = logging.getLogger(__name__)

BASELINE_BARS = 360
EXTENDED_BARS = 420
EXTENDED_HOURS = 7.0
_RTOL = 1e-8
_ATOL = 1e-8
_STALE_HMM_COLUMNS = {
    "hmm_artifact_version",
    "hmm_feature_source",
    "range_prob",
    "trend_prob",
    "persistence_prob",
    "regime_conf",
    "hmm_tail_cvar_95",
    "ev_score",
    "utility_score",
    "hmm_range_prob",
    "hmm_trend_prob",
    "meta_labeler_hmm_artifact_version",
}
# These are recorded ex-ante snapshot fields not present in historical raw
# backtest-result exports.  They are not labels or outcomes and are copied only
# when the canonical conversion cannot regenerate them from the raw candidate.
_PRESERVED_TRAINING_SNAPSHOT_COLUMNS = {"valid_indicator"}


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def _as_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _parse_start(value: Any) -> datetime | None:
    parsed = pd.to_datetime(pd.Series([value]), utc=True, errors="coerce", format="mixed").iloc[0]
    return parsed.to_pydatetime() if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed) else None


def _same_optional_float(left: Any, right: Any) -> bool:
    left_num = _as_float(left)
    right_num = _as_float(right)
    if left_num is None or right_num is None:
        return left_num is None and right_num is None
    return bool(np.isclose(left_num, right_num, rtol=_RTOL, atol=_ATOL))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(backtest: pd.DataFrame, training: pd.DataFrame) -> None:
    for name, frame in (("backtest", backtest), ("training", training)):
        if "candidate_id" not in frame.columns:
            raise ValueError(f"{name} source lacks candidate_id")
        ids = cast(pd.Series, frame["candidate_id"]).fillna("").astype(str).str.strip()
        if bool(ids.eq("").any()) or bool(ids.duplicated().any()):
            raise ValueError(f"{name} source has blank or duplicate candidate_id values")
    source_ids = set(cast(pd.Series, backtest["candidate_id"]).astype(str))
    training_ids = set(cast(pd.Series, training["candidate_id"]).astype(str))
    if source_ids != training_ids:
        raise ValueError(
            "backtest/training candidate_id sets differ: "
            f"backtest_only={len(source_ids-training_ids)} training_only={len(training_ids-source_ids)}"
        )
    required = {"symbol", "backtest_start_ts_utc", "grid_lower", "grid_upper", "num_grids", "net_pnl_pct", "target_reached", "time_to_target_hours"}
    missing = sorted(required - set(backtest.columns))
    if missing:
        raise ValueError(f"backtest source lacks exact-replay columns: {missing}")


def baseline_matches(source: dict[str, Any], replay: dict[str, Any]) -> tuple[bool, str]:
    if not _same_optional_float(source.get("net_pnl_pct"), replay.get("net_pnl_pct")):
        return False, "net_pnl_pct_mismatch"
    if _as_bool(source.get("target_reached")) is not _as_bool(replay.get("target_reached")):
        return False, "target_reached_mismatch"
    if not _same_optional_float(source.get("time_to_target_hours"), replay.get("time_to_target_hours")):
        return False, "time_to_target_hours_mismatch"
    return True, ""


def _clear_stale_hmm(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(row)
    for column in _STALE_HMM_COLUMNS:
        if column in cleaned:
            cleaned[column] = None
    return cleaned


def replay_one(
    source: dict[str, Any],
    bars: pd.DataFrame,
    *,
    leverage: int,
    training_snapshot: dict[str, Any] | None = None,
    runner: Callable[..., dict[str, Any]] = run_single_backtest,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    """Return exact 420-bar rows, or an explicit non-publication reason."""
    if len(bars) < EXTENDED_BARS:
        return None, None, f"insufficient_klines:{len(bars)}"
    capital = _as_float(source.get("capital_base"))
    if capital is None or capital <= 0:
        return None, None, "invalid_capital_base"
    funding_mode = str(source.get("funding_mode", "continuous"))
    realism_profile = str(source.get("realism_profile", "legacy"))
    baseline = runner(
        source, bars.iloc[:BASELINE_BARS].copy(), capital=capital, leverage=leverage,
        max_holding_bars=BASELINE_BARS, funding_mode=funding_mode, realism_profile=realism_profile,
    )
    matches, reason = baseline_matches(source, baseline)
    if not matches:
        return None, None, reason
    extended = runner(
        source, bars.iloc[:EXTENDED_BARS].copy(), capital=capital, leverage=leverage,
        max_holding_bars=EXTENDED_BARS, funding_mode=funding_mode, realism_profile=realism_profile,
    )
    raw = _clear_stale_hmm({**source, **extended})
    candidate_snapshot = dict(training_snapshot or {})
    candidate_snapshot.update(source)
    training = _clear_stale_hmm(
        convert_to_training_row(extended, candidate_snapshot, horizon_hours=EXTENDED_HOURS)
    )
    for column in _PRESERVED_TRAINING_SNAPSHOT_COLUMNS:
        if column not in training and training_snapshot is not None and column in training_snapshot:
            training[column] = training_snapshot[column]
    return raw, training, ""


def _checkpoint_manifest(path: Path, *, backtest_path: Path, training_path: Path, leverage: int) -> None:
    payload = {
        "contract": "exact_candidate_420bar_replay_v1",
        "backtest_sha256": _file_sha256(backtest_path),
        "training_sha256": _file_sha256(training_path),
        "baseline_bars": BASELINE_BARS,
        "extended_bars": EXTENDED_BARS,
        "leverage": leverage,
    }
    manifest = path / "checkpoint_manifest.json"
    if manifest.exists():
        observed = json.loads(manifest.read_text(encoding="utf-8"))
        if observed != payload:
            raise ValueError("checkpoint manifest differs from requested source/configuration")
        return
    path.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


async def replay_all(
    backtest: pd.DataFrame,
    training: pd.DataFrame,
    *,
    checkpoint_dir: Path,
    leverage: int,
    fetcher: Callable[[Any, str, datetime, int, str], Awaitable[pd.DataFrame]] = fetch_historical_klines,
    runner: Callable[..., dict[str, Any]] = run_single_backtest,
    client_factory: Callable[[], Any] = BinanceClient,
) -> tuple[int, int]:
    """Run/resume sequential exact replay; malformed rows are durable failures."""
    raw_path = checkpoint_dir / "replayed_backtest.csv"
    train_path = checkpoint_dir / "replayed_training.csv"
    failure_path = checkpoint_dir / "failures.csv"
    completed: set[str] = set()
    if raw_path.exists() != train_path.exists():
        raise ValueError("checkpoint has an unpaired raw/training replay file")
    if raw_path.exists():
        completed_frame = pd.read_csv(raw_path, low_memory=False)
        completed_training_frame = pd.read_csv(train_path, low_memory=False)
        completed = set(cast(pd.Series, completed_frame["candidate_id"]).astype(str))
        completed_training = set(
            cast(pd.Series, completed_training_frame["candidate_id"]).astype(str)
        )
        if len(completed_frame) != len(completed) or len(completed_training_frame) != len(completed_training):
            raise ValueError("checkpoint contains duplicate candidate_id values")
        if completed != completed_training:
            raise ValueError("checkpoint raw/training candidate_id sets differ")
    failures: set[str] = set()
    if failure_path.exists():
        failure_frame = pd.read_csv(failure_path, low_memory=False)
        failures = set(cast(pd.Series, failure_frame["candidate_id"]).astype(str))
    pending = cast(pd.DataFrame, backtest.loc[~cast(pd.Series, backtest["candidate_id"]).astype(str).isin(completed | failures)])
    raw_fields = list(backtest.columns)
    train_fields = list(training.columns)
    training_by_id = {
        str(row["candidate_id"]): row.to_dict()
        for _, row in training.iterrows()
    }
    client = client_factory()
    success_count = 0
    failure_count = 0
    try:
        for position, (_, row) in enumerate(pending.iterrows(), start=1):
            source = row.to_dict()
            candidate_id = str(source["candidate_id"])
            try:
                start = _parse_start(source.get("backtest_start_ts_utc"))
                if start is None:
                    raise ValueError("invalid_backtest_start_ts_utc")
                bars = await fetcher(client, str(source["symbol"]), start, int(EXTENDED_HOURS), "1m")
                raw, train, reason = replay_one(
                    source,
                    bars,
                    leverage=leverage,
                    training_snapshot=training_by_id[candidate_id],
                    runner=runner,
                )
                if reason:
                    raise ValueError(reason)
                assert raw is not None and train is not None
                if set(raw) != set(raw_fields):
                    raise ValueError("replayed_backtest_schema_mismatch")
                if set(train) != set(train_fields):
                    raise ValueError("replayed_training_schema_mismatch")
                _append_rows(raw_path, [raw], raw_fields)
                _append_rows(train_path, [train], train_fields)
                success_count += 1
            except Exception as exc:  # Keep a single source row from corrupting the batch.
                _append_rows(failure_path, [{"candidate_id": candidate_id, "error": f"{type(exc).__name__}: {exc}"}], ["candidate_id", "error"])
                failure_count += 1
            if position % 25 == 0:
                logger.info("replay progress pending=%d completed_now=%d failed_now=%d", position, success_count, failure_count)
    finally:
        await client.close()
    return success_count, failure_count


def publish_if_complete(*, checkpoint_dir: Path, output_dir: Path, expected_rows: int) -> None:
    failures = checkpoint_dir / "failures.csv"
    raw_path = checkpoint_dir / "replayed_backtest.csv"
    train_path = checkpoint_dir / "replayed_training.csv"
    if failures.exists() or not raw_path.exists() or not train_path.exists():
        raise ValueError("checkpoint has failures or lacks complete replay files; refusing publication")
    raw = pd.read_csv(raw_path, low_memory=False)
    training = pd.read_csv(train_path, low_memory=False)
    validate_inputs(raw, training)
    if len(raw) != expected_rows or len(training) != expected_rows:
        raise ValueError("checkpoint row count is incomplete; refusing publication")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if temp_dir.exists():
        raise FileExistsError(f"temporary publication directory already exists: {temp_dir}")
    temp_dir.mkdir(parents=True)
    try:
        raw.to_csv(temp_dir / "backtest_results_20260820_7h.csv", index=False)
        training.to_csv(temp_dir / "training_data_20260820_7h.csv", index=False)
        (temp_dir / "replay_manifest.json").write_text(json.dumps({
            "contract": "exact_candidate_420bar_replay_v1",
            "generation_mode": "historical_exact_replay",
            "full_pool": False,
            "source_rows": expected_rows,
            "baseline_bars": BASELINE_BARS,
            "extended_bars": EXTENDED_BARS,
            "published_only_after_all_baselines_reproduced": True,
        }, indent=2) + "\n", encoding="utf-8")
        temp_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


async def _main_async(args: argparse.Namespace) -> int:
    backtest = pd.read_csv(args.backtest_source, low_memory=False)
    training = pd.read_csv(args.training_source, low_memory=False)
    validate_inputs(backtest, training)
    _checkpoint_manifest(args.checkpoint_dir, backtest_path=args.backtest_source, training_path=args.training_source, leverage=args.leverage)
    completed, failed = await replay_all(backtest, training, checkpoint_dir=args.checkpoint_dir, leverage=args.leverage)
    publish_if_complete(checkpoint_dir=args.checkpoint_dir, output_dir=args.output_dir, expected_rows=len(backtest))
    print(json.dumps({"completed_now": completed, "failed_now": failed, "output_dir": str(args.output_dir)}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-source", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--leverage", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if args.leverage <= 0:
        raise ValueError("leverage must be positive")
    raise SystemExit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
