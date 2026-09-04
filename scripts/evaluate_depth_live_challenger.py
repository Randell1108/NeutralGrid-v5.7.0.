"""Evaluate a tiny isolated depth-aware challenger on live-shadow labels.

This diagnostic consumes the accumulated depth-oracle dataset, joins the
original scan-time feature rows, scores active/zero/cost controls, and trains a
throwaway active+depth challenger on the time-split train side. It never writes
to production model paths and is fail-closed for promotion when support is too
small.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutralgrid.models.meta_labeler import MetaLabeler  # noqa: E402


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


def _read_training_frame(dataset_dir: Path) -> pd.DataFrame:
    path = dataset_dir / "depth_training_frame.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _read_scans(paths: Sequence[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("At least one --scan-csv is required")
    scans = pd.concat(frames, ignore_index=True)
    if "candidate_id" not in scans.columns:
        raise ValueError("Scan CSV is missing candidate_id")
    duplicated = cast(pd.Series, scans["candidate_id"].astype(str)).duplicated(keep=False)
    if bool(duplicated.any()):
        duplicates = sorted(cast(pd.Series, scans.loc[duplicated, "candidate_id"].astype(str)).unique().tolist())
        raise ValueError(f"Duplicate candidate_id across scan CSVs: {duplicates[:10]}")
    return scans


def _load_labeler(path: str) -> MetaLabeler:
    return MetaLabeler.load(Path(path))


def _try_load_control(path: str, *, name: str, blockers: list[str]) -> MetaLabeler | None:
    try:
        return _load_labeler(path)
    except Exception as exc:
        blockers.append(f"{name}_load_failed:{exc}")
        return None


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, score))


def _safe_ap(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, score))


def _metrics_at_top_fraction(frame: pd.DataFrame, score_col: str, *, fraction: float = 0.25) -> dict[str, Any]:
    if frame.empty:
        return {
            "k": 0,
            "precision": None,
            "median_pnl": None,
            "tail_lt_-20_pct": None,
            "false_positive_rate": None,
        }
    k = max(1, int(np.ceil(len(frame) * fraction)))
    ranked = cast(pd.DataFrame, frame.sort_values(score_col, ascending=False).head(k))
    y = cast(pd.Series, pd.to_numeric(ranked["depth_oracle_label"], errors="coerce"))
    pnl = cast(pd.Series, pd.to_numeric(ranked["depth_adjusted_pnl_pct"], errors="coerce"))
    tail = cast(pd.Series, pd.to_numeric(ranked["tail_pnl_pct"], errors="coerce"))
    frame_label = cast(pd.Series, pd.to_numeric(frame["depth_oracle_label"], errors="coerce"))
    negatives = int(cast(pd.Series, frame_label == 0).sum())
    false_positives = int(cast(pd.Series, y == 0).sum())
    return {
        "k": int(k),
        "precision": float((y == 1).mean()) if len(y) else None,
        "median_pnl": float(pnl.median()) if bool(pnl.notna().any()) else None,
        "tail_lt_-20_pct": float((tail < -20.0).mean()) if bool(tail.notna().any()) else None,
        "false_positive_rate": float(false_positives / negatives) if negatives > 0 else None,
        "selected_candidate_ids": ranked["candidate_id"].astype(str).tolist(),
    }


def _model_metrics(frame: pd.DataFrame, score_col: str) -> dict[str, Any]:
    y = np.asarray(pd.to_numeric(frame["depth_oracle_label"], errors="coerce"), dtype=int)
    score = np.asarray(pd.to_numeric(frame[score_col], errors="coerce"), dtype=float)
    finite = np.isfinite(score)
    y = y[finite]
    score = score[finite]
    scored_frame = frame.loc[finite].copy()
    return {
        "eval_rows": int(len(y)),
        "positive_rows": int((y == 1).sum()) if len(y) else 0,
        "negative_rows": int((y == 0).sum()) if len(y) else 0,
        "roc_auc": _safe_auc(y, score) if len(y) else None,
        "average_precision": _safe_ap(y, score) if len(y) else None,
        "top_25pct": _metrics_at_top_fraction(scored_frame, score_col, fraction=0.25),
    }


def _time_split(frame: pd.DataFrame, *, eval_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = frame.copy()
    df["_scan_ts"] = pd.to_datetime(df["scan_time_utc"], utc=True, errors="coerce")
    df = cast(pd.DataFrame, df.sort_values(["_scan_ts", "candidate_id"]))
    eval_rows = max(1, int(np.ceil(len(df) * eval_fraction)))
    train_rows = len(df) - eval_rows
    return cast(pd.DataFrame, df.iloc[:train_rows].copy()), cast(pd.DataFrame, df.iloc[train_rows:].copy())


def _feature_columns(frame: pd.DataFrame, active_features: Sequence[str]) -> list[str]:
    depth_features = [
        col
        for col in frame.columns
        if col.startswith("depth_scan_") and col not in {"depth_scan_funding_rate"}
    ]
    return [col for col in [*active_features, *depth_features] if col in frame.columns]


def _train_depth_challenger(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    features: Sequence[str],
) -> tuple[Pipeline, np.ndarray, list[dict[str, Any]]]:
    y_train = np.asarray(pd.to_numeric(train["depth_oracle_label"], errors="coerce"), dtype=int)
    if len(np.unique(y_train)) < 2:
        raise ValueError("Cannot train challenger: train split has one class")
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    class_weight="balanced",
                    C=0.25,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )
    x_train = train[list(features)]
    model.fit(x_train, y_train)
    score = np.asarray(model.predict_proba(eval_df[list(features)])[:, 1], dtype=float)
    logistic = cast(LogisticRegression, model.named_steps["logistic"])
    coefs = np.asarray(logistic.coef_[0], dtype=float)
    top = sorted(
        (
            {
                "feature": feature,
                "coef": float(coef),
                "abs_coef": float(abs(coef)),
            }
            for feature, coef in zip(features, coefs)
        ),
        key=lambda item: item["abs_coef"],
        reverse=True,
    )
    return model, score, top[:20]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--scan-csv", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--active-model", default="models/meta_labeler.pkl")
    parser.add_argument("--zero-control-model", default="outputs/audits/challenger_20260625_231555/control/model.pkl")
    parser.add_argument(
        "--cost-control-model",
        default="models/meta_labeler_challenger/20260625_231555/model.pkl",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.20)
    args = parser.parse_args(argv)
    if args.eval_fraction <= 0 or args.eval_fraction >= 1:
        parser.error("--eval-fraction must be between 0 and 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)
    dataset_manifest_path = dataset_dir / "depth_oracle_dataset_manifest.json"
    dataset_manifest = _read_json(dataset_manifest_path) if dataset_manifest_path.exists() else {}

    depth_frame = _read_training_frame(dataset_dir)
    scans = _read_scans(args.scan_csv)
    merged = depth_frame.merge(
        scans,
        on=["candidate_id", "symbol"],
        how="left",
        suffixes=("", "_scan"),
        validate="one_to_one",
    )
    if bool(cast(pd.Series, merged["grid_is_valid"].isna()).all()) if "grid_is_valid" in merged.columns else True:
        raise ValueError("No scan rows joined to depth training frame")

    promotion_blockers: list[str] = []
    active = _load_labeler(str(args.active_model))
    zero = _try_load_control(str(args.zero_control_model), name="zero_control", blockers=promotion_blockers)
    cost = _try_load_control(str(args.cost_control_model), name="cost_control", blockers=promotion_blockers)
    active_features = list(getattr(active, "_feature_names", []) or [])

    merged["score_active"] = active.predict_proba_batch(merged)
    merged["score_zero_control"] = zero.predict_proba_batch(merged) if zero is not None else np.nan
    merged["score_cost_control"] = cost.predict_proba_batch(merged) if cost is not None else np.nan

    train, eval_df = _time_split(merged, eval_fraction=float(args.eval_fraction))
    features = _feature_columns(merged, active_features)
    if len(merged) < 100:
        promotion_blockers.append("insufficient_labelable_rows_lt_100")
    if len(eval_df) < 30:
        promotion_blockers.append("insufficient_eval_rows_lt_30")
    eval_label = cast(pd.Series, pd.to_numeric(eval_df["depth_oracle_label"], errors="coerce"))
    eval_positive_count = int(cast(pd.Series, eval_label == 1).sum())
    eval_negative_count = int(cast(pd.Series, eval_label == 0).sum())
    if eval_positive_count < 10:
        promotion_blockers.append("insufficient_eval_positives_lt_10")
    if eval_negative_count < 10:
        promotion_blockers.append("insufficient_eval_negatives_lt_10")

    challenger_saved = None
    challenger_top_features: list[dict[str, Any]] = []
    try:
        challenger, challenger_score, challenger_top_features = _train_depth_challenger(
            train,
            eval_df,
            features=features,
        )
        eval_df = eval_df.copy()
        eval_df["score_depth_challenger"] = challenger_score
        challenger_saved = output_dir / "depth_live_challenger.joblib"
        joblib.dump(challenger, challenger_saved)
    except ValueError as exc:
        eval_df = eval_df.copy()
        eval_df["score_depth_challenger"] = np.nan
        promotion_blockers.append(str(exc))

    metrics = {
        "active": _model_metrics(eval_df, "score_active"),
        "zero_control": _model_metrics(eval_df, "score_zero_control"),
        "cost_control": _model_metrics(eval_df, "score_cost_control"),
        "depth_challenger": _model_metrics(eval_df, "score_depth_challenger"),
    }
    challenger_precision = metrics["depth_challenger"]["top_25pct"]["precision"]
    active_precision = metrics["active"]["top_25pct"]["precision"]
    zero_precision = metrics["zero_control"]["top_25pct"]["precision"]
    cost_precision = metrics["cost_control"]["top_25pct"]["precision"]
    precision_beats_controls = (
        challenger_precision is not None
        and active_precision is not None
        and zero_precision is not None
        and cost_precision is not None
        and challenger_precision > active_precision
        and challenger_precision > zero_precision
        and challenger_precision > cost_precision
    )
    if not precision_beats_controls:
        promotion_blockers.append("precision_does_not_beat_all_controls")

    predictions_path = output_dir / "depth_live_challenger_predictions.csv"
    eval_df.to_csv(predictions_path, index=False)
    train.to_csv(output_dir / "depth_live_challenger_train_rows.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "diagnostic_complete",
        "PROMOTE": False,
        "promotion_blockers": sorted(set(promotion_blockers)),
        "dataset_dir": str(dataset_dir),
        "dataset_manifest": dataset_manifest,
        "scan_csv": [str(path) for path in args.scan_csv],
        "output_dir": str(output_dir),
        "predictions": str(predictions_path),
        "challenger_model": str(challenger_saved) if challenger_saved is not None else None,
        "rows": {
            "labelable_rows": int(len(merged)),
            "train_rows": int(len(train)),
            "eval_rows": int(len(eval_df)),
            "eval_positive_rows": eval_positive_count,
            "eval_negative_rows": eval_negative_count,
        },
        "feature_columns": features,
        "feature_column_count": int(len(features)),
        "metrics": metrics,
        "depth_challenger_top_features": challenger_top_features,
        "note": (
            "Diagnostic only. Promotion is forced false until labelable/eval support is large enough "
            "for a robust depth-aware OOS comparison."
        ),
        "git_head": _git_output(["rev-parse", "--short", "HEAD"]),
        "git_status_short": _git_output(["status", "--short"]),
        "command": " ".join(sys.argv),
    }
    (output_dir / "depth_live_challenger_report.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
