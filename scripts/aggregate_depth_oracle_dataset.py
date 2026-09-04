"""Aggregate depth-oracle bundles into a leakage-safe training frame.

This script does not train or promote a model. It joins oracle labels to
ex-ante depth features, persists the combined audit frame, and reports whether
the accumulated evidence is sufficient to run an OOS challenger comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutralgrid.data.depth_shadow import parse_candidate_scan_time_utc  # noqa: E402


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _oracle_manifest_path(path: Path) -> Path:
    if path.is_dir():
        return path / "depth_oracle_manifest.json"
    return path


def _feature_path_from_manifest(manifest: dict[str, Any]) -> Path | None:
    depth_input = manifest.get("depth_input")
    if depth_input is None:
        return None
    candidate = Path(str(depth_input)) / "features" / "depth_exante_features.csv"
    return candidate if candidate.exists() else None


def _candidate_scan_time(candidate_id: Any) -> str | None:
    return parse_candidate_scan_time_utc(str(candidate_id))


def _candidate_key(series: pd.Series) -> pd.Series:
    return cast(pd.Series, series.astype(str).str.strip())


def _label_summary(labels: pd.DataFrame) -> dict[str, int]:
    if labels.empty or "depth_oracle_label" not in labels.columns:
        return {"label_rows": int(len(labels)), "labelable_rows": 0, "positive_rows": 0, "negative_rows": 0}
    numeric = cast(pd.Series, pd.to_numeric(labels["depth_oracle_label"], errors="coerce"))
    return {
        "label_rows": int(len(labels)),
        "labelable_rows": int(numeric.notna().sum()),
        "positive_rows": int((numeric == 1).sum()),
        "negative_rows": int((numeric == 0).sum()),
    }


def _oos_split_status(training: pd.DataFrame, *, eval_fraction: float) -> dict[str, Any]:
    if training.empty:
        return {
            "ready": False,
            "reason": "no_labelable_rows",
            "train_rows": 0,
            "eval_rows": 0,
        }
    df = training.copy()
    df["_scan_ts"] = pd.to_datetime(df["scan_time_utc"], utc=True, errors="coerce")
    df = cast(pd.DataFrame, df.sort_values(["_scan_ts", "candidate_id"]))
    n = len(df)
    eval_rows = max(1, int(np.ceil(n * eval_fraction)))
    train_rows = n - eval_rows
    if train_rows <= 0:
        return {
            "ready": False,
            "reason": "not_enough_rows_for_train_eval_split",
            "train_rows": int(train_rows),
            "eval_rows": int(eval_rows),
        }
    train = cast(pd.DataFrame, df.iloc[:train_rows])
    eval_df = cast(pd.DataFrame, df.iloc[train_rows:])
    train_label = cast(pd.Series, pd.to_numeric(train["depth_oracle_label"], errors="coerce"))
    eval_label = cast(pd.Series, pd.to_numeric(eval_df["depth_oracle_label"], errors="coerce"))
    train_classes = sorted(train_label.dropna().unique().tolist())
    eval_classes = sorted(eval_label.dropna().unique().tolist())
    if len(train_classes) < 2:
        reason = "train_split_missing_class"
    elif len(eval_classes) < 2:
        reason = "eval_split_missing_class"
    else:
        reason = "ready"
    return {
        "ready": reason == "ready",
        "reason": reason,
        "train_rows": int(len(train)),
        "eval_rows": int(len(eval_df)),
        "train_classes": [int(v) for v in train_classes],
        "eval_classes": [int(v) for v in eval_classes],
        "eval_fraction": float(eval_fraction),
    }


def _load_bundle(path: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    manifest_path = _oracle_manifest_path(path)
    manifest = _read_json(manifest_path)
    labels_path = Path(str(manifest.get("depth_labels", manifest_path.parent / "depth_labels.csv")))
    labels = _read_csv(labels_path)
    labels["oracle_bundle"] = str(manifest_path.parent)
    feature_path = _feature_path_from_manifest(manifest)
    features = _read_csv(feature_path) if feature_path is not None else None
    if features is not None:
        features["oracle_bundle"] = str(manifest_path.parent)
    bundle = {
        "oracle_dir": str(manifest_path.parent),
        "manifest_path": str(manifest_path),
        "labels_path": str(labels_path),
        "features_path": str(feature_path) if feature_path is not None else None,
        "manifest": manifest,
        "label_summary": _label_summary(labels),
    }
    return labels, features, bundle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", action="append", required=True, help="Oracle output dir or manifest path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.20)
    args = parser.parse_args(argv)
    if args.eval_fraction <= 0 or args.eval_fraction >= 1:
        parser.error("--eval-fraction must be between 0 and 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    bundles: list[dict[str, Any]] = []
    for raw_path in args.oracle_dir:
        labels, features, bundle = _load_bundle(Path(raw_path))
        label_frames.append(labels)
        if features is not None:
            feature_frames.append(features)
        bundles.append(bundle)

    labels_all = pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame()
    if "candidate_id" not in labels_all.columns:
        raise ValueError("Aggregated labels are missing candidate_id")
    labels_all["_candidate_key"] = _candidate_key(cast(pd.Series, labels_all["candidate_id"]))
    duplicates = sorted(
        set(cast(pd.Series, labels_all.loc[labels_all["_candidate_key"].duplicated(keep=False), "_candidate_key"]))
    )
    if duplicates:
        raise ValueError(f"Duplicate candidate_id across oracle bundles: {duplicates[:10]}")

    labels_all["scan_time_utc"] = labels_all["candidate_id"].map(_candidate_scan_time)
    features_all = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    if not features_all.empty:
        features_all["_candidate_key"] = _candidate_key(cast(pd.Series, features_all["candidate_id"]))
        blocked = {"symbol", "candidate_id", "oracle_bundle", "_candidate_key", "scan_time_utc"}
        feature_cols = [col for col in features_all.columns if col not in blocked]
        candidate_frame = labels_all.merge(
            features_all[["_candidate_key", *feature_cols]],
            on="_candidate_key",
            how="left",
            validate="one_to_one",
        )
    else:
        feature_cols = []
        candidate_frame = labels_all.copy()

    label_numeric = cast(pd.Series, pd.to_numeric(candidate_frame["depth_oracle_label"], errors="coerce"))
    training_frame = candidate_frame.loc[label_numeric.notna()].copy()
    if not training_frame.empty:
        training_frame["depth_oracle_label"] = cast(
            pd.Series,
            pd.to_numeric(training_frame["depth_oracle_label"], errors="coerce"),
        ).astype(int)

    candidate_path = output_dir / "depth_candidate_frame.csv"
    training_path = output_dir / "depth_training_frame.csv"
    labels_all.drop(columns=["_candidate_key"], errors="ignore").to_csv(output_dir / "depth_labels_all.csv", index=False)
    candidate_frame.drop(columns=["_candidate_key"], errors="ignore").to_csv(candidate_path, index=False)
    training_frame.drop(columns=["_candidate_key"], errors="ignore").to_csv(training_path, index=False)

    summary = _label_summary(labels_all)
    oos = _oos_split_status(training_frame, eval_fraction=float(args.eval_fraction))
    if feature_cols:
        feature_na = cast(pd.Series, cast(pd.DataFrame, candidate_frame[feature_cols]).isna().all(axis=1))
        missing_feature_rows = int(feature_na.sum())
    else:
        missing_feature_rows = int(len(candidate_frame))
    status = "ready_for_isolated_oos_challenger" if oos["ready"] else "blocked_insufficient_oos_depth_labels"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
        "output_dir": str(output_dir),
        "oracle_dirs": [str(path) for path in args.oracle_dir],
        "bundles": bundles,
        "candidate_frame": str(candidate_path),
        "training_frame": str(training_path),
        "feature_columns": feature_cols,
        "feature_column_count": int(len(feature_cols)),
        "missing_exante_feature_rows": missing_feature_rows,
        "label_summary": summary,
        "oos_split": oos,
        "leakage_note": (
            "depth_training_frame joins oracle labels only to depth_exante_features. "
            "Forward replay diagnostics and window diagnostics are excluded from model features."
        ),
        "git_head": _git_output(["rev-parse", "--short", "HEAD"]),
        "git_status_short": _git_output(["status", "--short"]),
        "command": " ".join(sys.argv),
    }
    (output_dir / "depth_oracle_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if oos["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
