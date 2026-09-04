"""Train an isolated bookDepth-archive challenger against depth-aware labels."""

from __future__ import annotations

import argparse
import json
import pickle
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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _active_features(metadata_path: Path) -> list[str]:
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = data.get("features")
    if not isinstance(features, list) or not all(isinstance(item, str) for item in features):
        raise ValueError(f"{metadata_path} does not contain a string list 'features'")
    return list(features)


def _bookdepth_feature_columns(df: pd.DataFrame) -> list[str]:
    include_exact = {
        "bookdepth_archive_available",
        "bookdepth_candidate_zip_count",
        "bookdepth_snapshot_lag_seconds",
        "bookdepth_bucket_count",
    }
    include_suffixes = ("_notional", "_depth", "_imbalance", "_to_position")
    blocked = {
        "bookdepth_archive_paths",
        "bookdepth_available_percentages",
        "bookdepth_missing_reason",
        "bookdepth_snapshot_time_utc",
    }
    cols: list[str] = []
    for col in df.columns:
        if not str(col).startswith("bookdepth_") or col in blocked:
            continue
        if col in include_exact or str(col).endswith(include_suffixes):
            cols.append(str(col))
    return cols


def _coerce_feature_matrix(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for feature in features:
        if feature in df.columns:
            out[feature] = pd.to_numeric(cast(pd.Series, df[feature]), errors="coerce")
        else:
            out[feature] = np.nan
    return out


def _model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def _as_int_label(series: pd.Series) -> pd.Series:
    numeric = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    return numeric.astype("Int64")


def _safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, prob))


def _topk_metrics(y_true: np.ndarray, prob: np.ndarray, pnl: np.ndarray, frac: float) -> dict[str, Any]:
    k = max(1, int(np.ceil(len(prob) * frac)))
    order = np.argsort(-prob)[:k]
    selected_y = y_true[order]
    selected_pnl = pnl[order]
    positives = int(selected_y.sum())
    false_positives = int(k - positives)
    negatives = int((y_true == 0).sum())
    return {
        "k": int(k),
        "precision": float(positives / k),
        "positives": positives,
        "false_positives": false_positives,
        "false_positive_rate": float(false_positives / negatives) if negatives else None,
        "median_pnl": float(np.nanmedian(selected_pnl)) if len(selected_pnl) else None,
        "tail_lt_-20_pct": float(np.nanmean(selected_pnl < -20.0)) if len(selected_pnl) else None,
    }


def _threshold_metrics(y_true: np.ndarray, prob: np.ndarray, pnl: np.ndarray, threshold: float) -> dict[str, Any]:
    selected = prob >= threshold
    if not bool(selected.any()):
        return {
            "threshold": threshold,
            "selected_count": 0,
            "precision": None,
            "median_pnl": None,
            "tail_lt_-20_pct": None,
        }
    selected_y = y_true[selected]
    selected_pnl = pnl[selected]
    return {
        "threshold": threshold,
        "selected_count": int(selected.sum()),
        "precision": float(selected_y.mean()),
        "median_pnl": float(np.nanmedian(selected_pnl)),
        "tail_lt_-20_pct": float(np.nanmean(selected_pnl < -20.0)),
    }


def _evaluate_model(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    train_label_col: str,
    eval_label_col: str,
    train_mask: pd.Series,
    eval_mask: pd.Series,
) -> tuple[Pipeline, pd.Series, dict[str, Any]]:
    train_label = _as_int_label(cast(pd.Series, df[train_label_col]))
    eval_label = _as_int_label(cast(pd.Series, df[eval_label_col]))
    usable_train = train_mask & train_label.notna()
    usable_eval = eval_mask & eval_label.notna()
    y_train = train_label.loc[usable_train].astype(int).to_numpy()
    y_eval = eval_label.loc[usable_eval].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"{train_label_col}: training split has fewer than two classes")
    if len(y_eval) == 0:
        raise ValueError(f"{eval_label_col}: eval split has no rows")

    model = _model()
    x_train = _coerce_feature_matrix(df.loc[usable_train], features)
    x_eval = _coerce_feature_matrix(df.loc[usable_eval], features)
    model.fit(x_train, y_train)
    prob = cast(np.ndarray, model.predict_proba(x_eval))[:, 1]

    all_probs = pd.Series(np.nan, index=df.index, dtype=float)
    all_probs.loc[usable_eval] = prob
    raw_pnl = cast(pd.Series, df.get("db_pnl", pd.Series(np.nan, index=df.index)))
    pnl_series = cast(pd.Series, pd.to_numeric(raw_pnl, errors="coerce"))
    pnl_eval = cast(pd.Series, pnl_series.loc[usable_eval]).to_numpy(dtype=float)
    metrics = {
        "train_label_col": train_label_col,
        "eval_label_col": eval_label_col,
        "feature_count": len(features),
        "train_rows": int(usable_train.sum()),
        "eval_rows": int(usable_eval.sum()),
        "train_positive_rate": float(y_train.mean()),
        "eval_positive_rate": float(y_eval.mean()),
        "roc_auc": _safe_auc(y_eval, prob),
        "average_precision": float(average_precision_score(y_eval, prob)) if len(np.unique(y_eval)) > 1 else None,
        "brier": float(brier_score_loss(y_eval, prob)),
        "top_10pct": _topk_metrics(y_eval, prob, pnl_eval, 0.10),
        "top_25pct": _topk_metrics(y_eval, prob, pnl_eval, 0.25),
        "threshold_0p5": _threshold_metrics(y_eval, prob, pnl_eval, 0.50),
    }
    return model, all_probs, metrics


def _coef_report(model: Pipeline, features: Sequence[str], *, top_n: int = 30) -> list[dict[str, Any]]:
    estimator = cast(LogisticRegression, model.named_steps["model"])
    coef = estimator.coef_[0]
    rows = [
        {"feature": feature, "coef": float(value), "abs_coef": float(abs(value))}
        for feature, value in zip(features, coef, strict=False)
    ]
    return sorted(rows, key=lambda row: cast(float, row["abs_coef"]), reverse=True)[:top_n]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-frame", required=True)
    parser.add_argument("--bookdepth-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--active-metadata", default=str(ROOT / "models" / "meta_labeler" / "metadata.json"))
    parser.add_argument("--depth-label", default="gt_depth")
    args = parser.parse_args(argv)

    training = pd.read_csv(args.training_frame)
    bookdepth = pd.read_csv(args.bookdepth_features)
    if "candidate_id" not in training.columns or "candidate_id" not in bookdepth.columns:
        raise ValueError("Both training frame and bookDepth features need candidate_id")
    merged = training.merge(
        bookdepth,
        on=["candidate_id", "symbol"],
        how="left",
        suffixes=("", "_bookdepth"),
        validate="one_to_one",
    )
    active_features = _active_features(Path(args.active_metadata))
    bookdepth_features = _bookdepth_feature_columns(bookdepth)
    enriched_features = [*active_features, *bookdepth_features]

    split = cast(pd.Series, merged["_split"]).astype(str)
    train_mask = split == "train"
    eval_mask = split == "eval"

    specs = {
        "control_zero_active_features": (active_features, "y_zero"),
        "control_cost_active_features": (active_features, "y_cost"),
        "depth_label_active_features": (active_features, args.depth_label),
        "depth_archive_challenger": (enriched_features, args.depth_label),
    }
    models: dict[str, Pipeline] = {}
    predictions = merged[["candidate_id", "symbol", "start_time_utc", "_split", args.depth_label, "db_pnl", "dh_pnl"]].copy()
    metrics: dict[str, Any] = {}
    coef_reports: dict[str, Any] = {}
    for name, (features, label_col) in specs.items():
        model, probs, model_metrics = _evaluate_model(
            merged,
            features=features,
            train_label_col=label_col,
            eval_label_col=args.depth_label,
            train_mask=train_mask,
            eval_mask=eval_mask,
        )
        models[name] = model
        predictions[f"{name}_prob"] = probs
        metrics[name] = model_metrics
        coef_reports[name] = _coef_report(model, features)

    challenger = metrics["depth_archive_challenger"]
    depth_baseline = metrics["depth_label_active_features"]
    zero_control = metrics["control_zero_active_features"]
    cost_control = metrics["control_cost_active_features"]
    promotion_screen = {
        "PROMOTE_OR_MODIFY_PRODUCTION": False,
        "reason": (
            "Isolated archive challenger only. Binance bookDepth is percent-bucket L2-style evidence, "
            "not raw queue-level replay; production feature schema and active artifacts are untouched."
        ),
        "beats_depth_label_active_auc": (
            challenger["roc_auc"] is not None
            and depth_baseline["roc_auc"] is not None
            and challenger["roc_auc"] > depth_baseline["roc_auc"]
        ),
        "beats_zero_control_precision_top25": (
            challenger["top_25pct"]["precision"] > zero_control["top_25pct"]["precision"]
        ),
        "beats_cost_control_precision_top25": (
            challenger["top_25pct"]["precision"] > cost_control["top_25pct"]["precision"]
        ),
        "auc_above_chance": challenger["roc_auc"] is not None and challenger["roc_auc"] > 0.5,
        "median_pnl_not_worse_than_zero_control": (
            challenger["top_25pct"]["median_pnl"] >= zero_control["top_25pct"]["median_pnl"]
        ),
        "tail_not_worse_than_zero_control": (
            challenger["top_25pct"]["tail_lt_-20_pct"] <= zero_control["top_25pct"]["tail_lt_-20_pct"]
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "bookdepth_archive_challenger_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    for name, model in models.items():
        joblib.dump(model, output_dir / f"{name}.joblib")

    report = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "training_frame": str(args.training_frame),
        "bookdepth_features": str(args.bookdepth_features),
        "active_metadata": str(args.active_metadata),
        "rows": int(len(merged)),
        "train_rows": int(train_mask.sum()),
        "eval_rows": int(eval_mask.sum()),
        "active_feature_count": len(active_features),
        "bookdepth_feature_count": len(bookdepth_features),
        "bookdepth_features": bookdepth_features,
        "metrics": metrics,
        "top_coefficients": coef_reports,
        "promotion_screen": promotion_screen,
        "predictions": str(predictions_path),
        "git_head": _git_output(["rev-parse", "--short", "HEAD"]),
        "git_status_short": _git_output(["status", "--short"]),
    }
    _write_json(output_dir / "bookdepth_archive_challenger_report.json", report)
    _write_json(output_dir / "promotion_screen.json", promotion_screen)
    with (output_dir / "model_specs.pkl").open("wb") as handle:
        pickle.dump({"specs": specs}, handle)

    print(json.dumps({"output_dir": str(output_dir), "promotion_screen": promotion_screen}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
