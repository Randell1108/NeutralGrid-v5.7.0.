"""Development-only FASTWIN logistic profile challenger.

The module consumes only hash-pinned, already-disclosed canonical rows. It
performs nested, expanding, scan-group-preserving temporal validation and
writes a shadow candidate bundle. It deliberately has no production promotion
entry point: a separate, newer canonical cohort must evaluate the frozen
candidate before the governed profile promoter may be called.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from neutralgrid.backtest.realism_governance import (
    CANDIDATE_TIME_GEOMETRIC_PROFILE,
    validate_realism_output_path,
)
from neutralgrid.core.constants import (
    ENGINE_VERSION,
    FORMULA_VERSION,
    LABEL_CONTRACT_VERSION,
)
from neutralgrid.scanner.canonical_fastwin_profile import (
    OBSERVATION_HOURS,
    TARGET_HOURS,
    TARGET_NAME,
    _atomic_json,
    _build_pattern,
    _expected_calibration_error,
    _ks_statistic,
    _metrics,
    _paired_cluster_auc_interval,
    sha256_file,
)
from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES
from neutralgrid.scanner.profile_model import ProfileModel, load_profile_model, save_profile_model
from neutralgrid.scanner.profile_model_walkforward import (
    AUC_FOLD_PASS_THRESHOLD,
    COVERAGE_FLOOR,
    FINITE_FOLD_COVERAGE_FLOOR,
    MEAN_AUC_FLOOR,
    MEAN_PASS_RATE_FLOOR,
    MIN_FINITE_FOLDS,
    POOLED_OOF_AUC_FLOOR,
    WalkForwardResult,
)


MODEL_FAMILY = "robust_logistic_v1"
REALISM_PROFILE = CANDIDATE_TIME_GEOMETRIC_PROFILE
MIN_CLASS_ROWS = 30
OUTER_FOLDS = 5
INNER_FOLDS = 4
INITIAL_TRAIN_FRACTION = 0.50
C_VALUES = (0.01, 0.1, 1.0, 10.0)
BRIER_REGRESSION_TOLERANCE = 0.005
ECE_REGRESSION_TOLERANCE = 0.01


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    greater = (positives[:, None] > negatives[None, :]).mean()
    equal = (positives[:, None] == negatives[None, :]).mean()
    return float(greater + 0.5 * equal)


def _verify_hash(path: Path, expected: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"pre-registered source disappeared: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"pre-registered source hash mismatch: {path}; "
            f"expected={expected}, actual={actual}"
        )


def _load_preregistration(path: Path) -> dict[str, Any]:
    import json

    preregistration = json.loads(path.read_text(encoding="utf-8"))
    if preregistration.get("schema_version") != 1:
        raise ValueError("unsupported logistic profile preregistration schema")
    challenger = preregistration.get("challenger", {})
    if challenger.get("model_family") != MODEL_FAMILY:
        raise ValueError("pre-registration does not authorize robust_logistic_v1")
    if tuple(float(value) for value in challenger.get("candidate_c_values", [])) != C_VALUES:
        raise ValueError("pre-registered C grid differs from implementation")
    feature_contract = preregistration.get("feature_contract", {})
    if list(feature_contract.get("features", [])) != list(DEFAULT_FEATURES):
        raise ValueError("pre-registered feature order differs from active profile contract")
    protocol = preregistration.get("development_protocol", {})
    if int(protocol.get("outer_folds", -1)) != OUTER_FOLDS:
        raise ValueError("pre-registered outer fold count differs from implementation")
    if int(protocol.get("inner_folds", -1)) != INNER_FOLDS:
        raise ValueError("pre-registered inner fold count differs from implementation")
    if float(protocol.get("purge_hours", -1.0)) != OBSERVATION_HOURS:
        raise ValueError("pre-registered purge differs from outcome horizon")
    for record in preregistration.get("development_sources", []):
        _verify_hash(Path(str(record["path"])), str(record["sha256"]))
    implementation_sources = preregistration.get("implementation_sources", [])
    if not implementation_sources:
        raise ValueError("pre-registration lacks implementation source hashes")
    for record in implementation_sources:
        _verify_hash(Path(str(record["path"])), str(record["sha256"]))
    incumbent = preregistration.get("incumbent", {})
    _verify_hash(Path(str(incumbent["model_path"])), str(incumbent["model_sha256"]))
    _verify_hash(Path(str(incumbent["pattern_path"])), str(incumbent["pattern_sha256"]))
    return preregistration


def _validate_exact_frame(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    required = {
        "candidate_id",
        "symbol",
        "start_time_utc",
        "fastwin_label",
        *DEFAULT_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")
    data = frame.copy()
    candidate_ids = cast(pd.Series, data["candidate_id"]).astype(str).str.strip()
    if bool(candidate_ids.eq("").any()) or bool(candidate_ids.duplicated().any()):
        raise ValueError(f"{context} has missing or duplicate candidate_id")
    data["candidate_id"] = candidate_ids
    data["start_time_utc"] = pd.to_datetime(
        data["start_time_utc"], utc=True, errors="coerce"
    )
    if bool(cast(pd.Series, data["start_time_utc"]).isna().any()):
        raise ValueError(f"{context} has invalid start_time_utc")
    labels = cast(pd.Series, pd.to_numeric(data["fastwin_label"], errors="coerce"))
    if bool(labels.isna().any()) or not set(labels.astype(int).unique()).issubset({0, 1}):
        raise ValueError(f"{context} labels are not complete binary values")
    data["fastwin_label"] = labels.astype(int)
    for feature in DEFAULT_FEATURES:
        numeric = cast(pd.Series, pd.to_numeric(data[feature], errors="coerce"))
        if not bool(np.isfinite(np.asarray(numeric, dtype=float)).all()):
            raise ValueError(f"{context} feature is non-finite: {feature}")
        data[feature] = numeric
    return cast(pd.DataFrame, data)


def _newer_disclosed_rows(cohort_path: Path, training_path: Path) -> pd.DataFrame:
    cohort = pd.read_csv(cohort_path, low_memory=False)
    training = pd.read_csv(training_path, low_memory=False)
    for context, frame in (("cohort", cohort), ("training", training)):
        if "candidate_id" not in frame.columns:
            raise ValueError(f"{context} lacks candidate_id")
        ids = cast(pd.Series, frame["candidate_id"]).astype(str).str.strip()
        if bool(ids.eq("").any()) or bool(ids.duplicated().any()):
            raise ValueError(f"{context} has missing or duplicate candidate_id")
        frame["candidate_id"] = ids
    cohort_ids = set(cast(pd.Series, cohort["candidate_id"]))
    training_ids = set(cast(pd.Series, training["candidate_id"]))
    if cohort_ids != training_ids:
        raise ValueError("disclosed cohort and canonical outcomes have different candidate IDs")
    required_contract = {
        "source": "backtest",
        "engine_version": ENGINE_VERSION,
        "label_contract_version": LABEL_CONTRACT_VERSION,
        "formula_version": FORMULA_VERSION,
        "realism_profile": REALISM_PROFILE,
    }
    for column, expected in required_contract.items():
        if column not in training.columns:
            raise ValueError(f"canonical outcomes lack {column}")
        values = cast(pd.Series, training[column]).astype(str).str.strip()
        if bool(values.ne(expected).any()):
            raise ValueError(f"canonical outcomes violate {column}={expected}")
    if "is_authoritative" not in training.columns:
        raise ValueError("canonical outcomes lack is_authoritative")
    authoritative = (
        cast(pd.Series, training["is_authoritative"])
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    if not bool(authoritative.all()):
        raise ValueError("canonical outcomes contain non-authoritative rows")
    time_to_target = cast(
        pd.Series,
        pd.to_numeric(training["time_to_target_hours"], errors="coerce"),
    )
    labels = time_to_target.le(TARGET_HOURS).fillna(False).astype(int)
    outcome = training[["candidate_id", "symbol", "start_time_utc"]].copy()
    outcome["fastwin_label"] = labels
    joined = outcome.merge(
        cohort[["candidate_id", *DEFAULT_FEATURES]],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(cohort):
        raise ValueError("canonical outcome/profile join is not complete")
    return _validate_exact_frame(joined, context="newer disclosed rows")


def assemble_development_pool(preregistration_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    preregistration = _load_preregistration(preregistration_path)
    sources = preregistration["development_sources"]
    if len(sources) != 4:
        raise ValueError("logistic v1 requires exactly four pre-registered development sources")
    older_frames = [
        _validate_exact_frame(
            pd.read_csv(Path(str(record["path"])), low_memory=False),
            context=f"disclosed canonical source {record['path']}",
        )
        for record in sources[:2]
    ]
    newer = _newer_disclosed_rows(
        Path(str(sources[2]["path"])), Path(str(sources[3]["path"]))
    )
    combined = pd.concat([*older_frames, newer], ignore_index=True, sort=False)
    combined = _validate_exact_frame(combined, context="combined development pool")
    if bool(cast(pd.Series, combined["candidate_id"]).duplicated().any()):
        raise ValueError("development sources overlap by candidate_id")
    combined = cast(
        pd.DataFrame,
        combined.sort_values(["start_time_utc", "candidate_id"], kind="stable")
        .reset_index(drop=True),
    )
    return combined, preregistration


def _temporal_splits(
    frame: pd.DataFrame,
    *,
    n_folds: int,
    initial_fraction: float = INITIAL_TRAIN_FRACTION,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    groups = sorted(cast(pd.Series, frame["start_time_utc"]).drop_duplicates().tolist())
    if len(groups) < n_folds + 2:
        raise ValueError("insufficient scan groups for temporal folds")
    initial_group_count = max(1, int(np.ceil(len(groups) * initial_fraction)))
    remaining = groups[initial_group_count:]
    if len(remaining) < n_folds:
        raise ValueError("insufficient post-prefix scan groups for temporal folds")
    chunks = [list(chunk) for chunk in np.array_split(remaining, n_folds)]
    purge = pd.Timedelta(hours=OBSERVATION_HOURS)
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for test_groups in chunks:
        test_start = cast(pd.Timestamp, min(test_groups))
        train = cast(
            pd.DataFrame,
            frame.loc[
                cast(pd.Series, frame["start_time_utc"]) + purge < test_start
            ].copy(),
        )
        test = cast(
            pd.DataFrame,
            frame.loc[
                cast(pd.Series, frame["start_time_utc"]).isin(test_groups)
            ].copy(),
        )
        splits.append((train, test))
    return splits


def _fit_model(
    frame: pd.DataFrame,
    *,
    c_value: float,
    selection_summary: dict[str, Any] | None = None,
) -> ProfileModel:
    labels = np.asarray(frame["fastwin_label"], dtype=int)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives < MIN_CLASS_ROWS or negatives < MIN_CLASS_ROWS:
        raise ValueError(
            f"logistic fit class floor failed: positives={positives}, negatives={negatives}"
        )
    raw = np.asarray(frame[list(DEFAULT_FEATURES)], dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("logistic fit received non-finite feature values")
    medians = np.median(raw, axis=0)
    means = np.mean(raw, axis=0)
    stds = np.std(raw, axis=0, ddof=0)
    stds = np.where(stds > 1e-12, stds, 1.0)
    standardized = (raw - means) / stds
    estimator = LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        class_weight=None,
        fit_intercept=True,
        max_iter=2000,
    )
    estimator.fit(standardized, labels)
    if int(np.asarray(estimator.n_iter_).max()) >= estimator.max_iter:
        raise ValueError("logistic optimizer reached max_iter without convergence")
    coefficients = np.asarray(estimator.coef_, dtype=float)[0]
    intercept = float(np.asarray(estimator.intercept_, dtype=float)[0])
    if not np.isfinite(coefficients).all() or not np.isfinite(intercept):
        raise ValueError("logistic fit produced non-finite parameters")
    winner_mean = np.mean(standardized[labels == 1], axis=0)
    loser_mean = np.mean(standardized[labels == 0], axis=0)
    return ProfileModel(
        features=list(DEFAULT_FEATURES),
        winner_mu={
            feature: float(winner_mean[index])
            for index, feature in enumerate(DEFAULT_FEATURES)
        },
        loser_mu={
            feature: float(loser_mean[index])
            for index, feature in enumerate(DEFAULT_FEATURES)
        },
        inv_cov=np.eye(len(DEFAULT_FEATURES)).tolist(),
        prior_winner=float(labels.mean()),
        duration_band={"min_hours": 0.0, "max_hours": TARGET_HOURS},
        feature_mean={
            feature: float(means[index])
            for index, feature in enumerate(DEFAULT_FEATURES)
        },
        feature_std={
            feature: float(stds[index])
            for index, feature in enumerate(DEFAULT_FEATURES)
        },
        feature_impute={
            feature: float(medians[index])
            for index, feature in enumerate(DEFAULT_FEATURES)
        },
        selection_summary=selection_summary,
        model_family=MODEL_FAMILY,
        linear_coef={
            feature: float(coefficients[index])
            for index, feature in enumerate(DEFAULT_FEATURES)
        },
        linear_intercept=intercept,
    )


def _score(model: ProfileModel, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    probabilities: list[float] = []
    for _, row in frame.iterrows():
        raw = {feature: row[feature] for feature in DEFAULT_FEATURES}
        score = model.llr(raw)
        probability = model.proba(raw)
        if score is None or probability is None:
            raise ValueError("profile model could not score an exact feature row")
        scores.append(float(score))
        probabilities.append(float(probability))
    return np.asarray(scores, dtype=float), np.asarray(probabilities, dtype=float)


def _select_c(frame: pd.DataFrame) -> tuple[float, dict[str, list[float]]]:
    splits = _temporal_splits(frame, n_folds=INNER_FOLDS)
    auc_by_c: dict[str, list[float]] = {}
    ranked: list[tuple[float, float]] = []
    for c_value in C_VALUES:
        fold_auc: list[float] = []
        for train, test in splits:
            model = _fit_model(train, c_value=c_value)
            scores, _ = _score(model, test)
            labels = np.asarray(test["fastwin_label"], dtype=int)
            fold_auc.append(_auc(scores, labels))
        finite = np.asarray([value for value in fold_auc if np.isfinite(value)])
        if finite.size < MIN_FINITE_FOLDS:
            raise ValueError(
                f"C={c_value} has only {finite.size} finite inner folds"
            )
        mean_auc = float(np.mean(finite))
        auc_by_c[str(c_value)] = fold_auc
        ranked.append((mean_auc, -float(c_value)))
    best_mean, negative_c = max(ranked)
    if not np.isfinite(best_mean):
        raise ValueError("all pre-registered C values have non-finite inner AUC")
    return -negative_c, auc_by_c


def walkforward_logistic(frame: pd.DataFrame, *, source_sha256: str) -> tuple[WalkForwardResult, list[dict[str, Any]]]:
    splits = _temporal_splits(frame, n_folds=OUTER_FOLDS)
    fold_auc: list[float] = []
    fold_ks: list[float] = []
    fold_train_rows: list[int] = []
    fold_test_rows: list[int] = []
    fold_train_winners: list[int] = []
    fold_test_winners: list[int] = []
    fold_test_start_utc: list[str] = []
    fold_test_end_utc: list[str] = []
    oof_ids: list[str] = []
    oof_labels: list[int] = []
    oof_scores: list[float] = []
    oof_probabilities: list[float] = []
    selections: list[dict[str, Any]] = []
    for train, test in splits:
        selected_c, inner_auc = _select_c(train)
        model = _fit_model(train, c_value=selected_c)
        scores, probabilities = _score(model, test)
        labels = np.asarray(test["fastwin_label"], dtype=int)
        fold_auc.append(_auc(scores, labels))
        fold_ks.append(
            _ks_statistic(scores[labels == 1], scores[labels == 0])
        )
        fold_train_rows.append(len(train))
        fold_test_rows.append(len(test))
        fold_train_winners.append(int(np.asarray(train["fastwin_label"], dtype=int).sum()))
        fold_test_winners.append(int(labels.sum()))
        fold_test_start_utc.append(
            cast(pd.Timestamp, test["start_time_utc"].min()).isoformat()
        )
        fold_test_end_utc.append(
            cast(pd.Timestamp, test["start_time_utc"].max()).isoformat()
        )
        oof_ids.extend(cast(pd.Series, test["candidate_id"]).astype(str).tolist())
        oof_labels.extend(labels.tolist())
        oof_scores.extend(scores.tolist())
        oof_probabilities.extend(probabilities.tolist())
        selections.append({"selected_c": selected_c, "inner_auc_by_c": inner_auc})
    finite_auc = np.asarray([value for value in fold_auc if np.isfinite(value)])
    finite_ks = np.asarray([value for value in fold_ks if np.isfinite(value)])
    labels = np.asarray(oof_labels, dtype=int)
    scores = np.asarray(oof_scores, dtype=float)
    probabilities = np.asarray(oof_probabilities, dtype=float)
    result = WalkForwardResult(
        n_folds=OUTER_FOLDS,
        fold_auc=fold_auc,
        fold_ks=fold_ks,
        mean_auc=float(np.mean(finite_auc)) if finite_auc.size else float("nan"),
        mean_ks=float(np.mean(finite_ks)) if finite_ks.size else float("nan"),
        mean_pass_rate=float(np.mean(finite_auc >= AUC_FOLD_PASS_THRESHOLD)),
        purge_hours=OBSERVATION_HOURS,
        requested_features=list(DEFAULT_FEATURES),
        admitted_features=list(DEFAULT_FEATURES),
        feature_coverage=1.0,
        fold_train_rows=fold_train_rows,
        fold_test_rows=fold_test_rows,
        fold_train_winners=fold_train_winners,
        fold_test_winners=fold_test_winners,
        fold_pnl_thresholds=[float("nan")] * OUTER_FOLDS,
        fold_test_start_utc=fold_test_start_utc,
        fold_test_end_utc=fold_test_end_utc,
        oof_strategy_ids=oof_ids,
        oof_labels=oof_labels,
        oof_scores=oof_scores,
        oof_probabilities=oof_probabilities,
        pooled_oof_auc=_auc(scores, labels),
        pooled_oof_ks=_ks_statistic(scores[labels == 1], scores[labels == 0]),
        pooled_oof_brier=float(np.mean(np.square(probabilities - labels))),
        pooled_oof_ece=_expected_calibration_error(probabilities, labels),
        source_sha256=source_sha256,
        labeled_rows=len(frame),
        duplicate_strategy_ids=0,
        holdout_start_after_utc=None,
    )
    return result, selections


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def evaluate_development(
    *,
    preregistration_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Fit and evaluate the pre-registered challenger without fresh evidence."""
    validate_realism_output_path(REALISM_PROFILE, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / "development_pool.csv",
        output_dir / "development_evaluation.json",
        output_dir / "candidate_profile_model.json",
        output_dir / "candidate_pattern_profile.json",
        output_dir / "candidate_manifest.json",
    ]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("logistic profile development was already materialized")
    pool, preregistration = assemble_development_pool(preregistration_path)
    pool_path = output_dir / "development_pool.csv"
    serializable_pool = pool.copy()
    serializable_pool["start_time_utc"] = cast(
        pd.Series, serializable_pool["start_time_utc"]
    ).astype(str)
    _atomic_csv(pool_path, serializable_pool)
    pool_sha256 = sha256_file(pool_path)
    candidate_result, c_selections = walkforward_logistic(
        pool, source_sha256=pool_sha256
    )
    indexed = pool.set_index("candidate_id", drop=False)
    oof = cast(pd.DataFrame, indexed.loc[candidate_result.oof_strategy_ids].copy())
    incumbent_path = Path(str(preregistration["incumbent"]["model_path"]))
    incumbent = load_profile_model(incumbent_path)
    incumbent_scores, incumbent_probabilities = _score(incumbent, oof)
    labels = np.asarray(candidate_result.oof_labels, dtype=int)
    candidate_scores = np.asarray(candidate_result.oof_scores, dtype=float)
    candidate_probabilities = np.asarray(candidate_result.oof_probabilities, dtype=float)
    clusters = np.asarray(cast(pd.Series, oof["start_time_utc"]).astype(str))
    candidate_metrics = _metrics(labels, candidate_scores, candidate_probabilities)
    incumbent_metrics = _metrics(labels, incumbent_scores, incumbent_probabilities)
    paired = _paired_cluster_auc_interval(
        labels=labels,
        candidate_scores=candidate_scores,
        incumbent_scores=incumbent_scores,
        clusters=clusters,
    )
    finite_fold_count = int(
        sum(np.isfinite(value) for value in candidate_result.fold_auc)
    )
    finite_fold_coverage = finite_fold_count / len(candidate_result.fold_auc)
    reasons: list[str] = []
    if finite_fold_count < MIN_FINITE_FOLDS:
        reasons.append("finite_fold_count_below_floor")
    if finite_fold_coverage < FINITE_FOLD_COVERAGE_FLOOR:
        reasons.append("finite_fold_coverage_below_floor")
    if candidate_result.mean_auc < MEAN_AUC_FLOOR:
        reasons.append("mean_auc_below_floor")
    if candidate_result.pooled_oof_auc < POOLED_OOF_AUC_FLOOR:
        reasons.append("pooled_oof_auc_below_floor")
    if candidate_result.mean_pass_rate < MEAN_PASS_RATE_FLOOR:
        reasons.append("mean_pass_rate_below_floor")
    if candidate_result.feature_coverage < COVERAGE_FLOOR:
        reasons.append("feature_coverage_below_floor")
    paired_interval = paired["delta_auc_ci_95"]
    if paired_interval is None or float(paired_interval[0]) <= 0.0:
        reasons.append("clustered_paired_auc_delta_ci_includes_zero")
    if candidate_metrics["brier"] > incumbent_metrics["brier"] + BRIER_REGRESSION_TOLERANCE:
        reasons.append("brier_regression_gt_0_005")
    if candidate_metrics["ece_10_equal_width"] > incumbent_metrics["ece_10_equal_width"] + ECE_REGRESSION_TOLERANCE:
        reasons.append("ece_regression_gt_0_01")

    final_c, final_inner_auc = _select_c(pool)
    selection_summary: dict[str, Any] = {
        "label_name": TARGET_NAME,
        "label_definition": {
            "time_to_target_rule": "finite time_to_target_hours <= 7.0",
            "time_to_target_claimed": True,
            "source": "authoritative canonical realistic backtest",
        },
        "model_family": MODEL_FAMILY,
        "features": list(DEFAULT_FEATURES),
        "fit_rows": len(pool),
        "winners_count": int(cast(pd.Series, pool["fastwin_label"]).sum()),
        "losers_count": int(len(pool) - cast(pd.Series, pool["fastwin_label"]).sum()),
        "purge_hours": OBSERVATION_HOURS,
        "selected_c": final_c,
        "candidate_c_values": list(C_VALUES),
        "development_sha256": pool_sha256,
        "development_outcomes_disclosed": True,
        "fresh_holdout_evaluated": False,
    }
    candidate_model = _fit_model(
        pool, c_value=final_c, selection_summary=selection_summary
    )
    pattern_source = pool.copy()
    pattern_source["_is_winner"] = pattern_source["fastwin_label"]
    candidate_pattern = _build_pattern(pattern_source, selection_summary)
    candidate_model_path = output_dir / "candidate_profile_model.json"
    candidate_model_tmp = candidate_model_path.with_suffix(".json.tmp")
    save_profile_model(candidate_model, candidate_model_tmp)
    candidate_model_tmp.replace(candidate_model_path)
    candidate_pattern_path = output_dir / "candidate_pattern_profile.json"
    candidate_pattern_tmp = candidate_pattern_path.with_suffix(".json.tmp")
    candidate_pattern.save_json(candidate_pattern_tmp)
    candidate_pattern_tmp.replace(candidate_pattern_path)

    report: dict[str, Any] = {
        "schema_version": 1,
        "preregistration_path": str(preregistration_path.resolve()),
        "preregistration_sha256": sha256_file(preregistration_path),
        "development_pool": {
            "path": str(pool_path.resolve()),
            "sha256": pool_sha256,
            "rows": len(pool),
            "candidate_ids_unique": int(cast(pd.Series, pool["candidate_id"]).nunique()),
            "scan_groups": int(cast(pd.Series, pool["start_time_utc"]).nunique()),
            "positives": int(cast(pd.Series, pool["fastwin_label"]).sum()),
            "negatives": int(len(pool) - cast(pd.Series, pool["fastwin_label"]).sum()),
            "start_utc": cast(pd.Timestamp, pool["start_time_utc"].min()).isoformat(),
            "end_utc": cast(pd.Timestamp, pool["start_time_utc"].max()).isoformat(),
        },
        "candidate_walkforward": asdict(candidate_result),
        "candidate_metrics": candidate_metrics,
        "incumbent_metrics": incumbent_metrics,
        "paired_cluster_bootstrap": paired,
        "nested_c_selections": c_selections,
        "final_fit": {
            "selected_c": final_c,
            "inner_auc_by_c": final_inner_auc,
        },
        "decision": {
            "fresh_holdout_eligible": not reasons,
            "promotion_eligible": False,
            "promotion_attempted": False,
            "production_artifacts_modified": False,
            "reasons": reasons,
            "promotion_blocker": "genuinely_new_canonical_holdout_required",
        },
    }
    _atomic_json(output_dir / "development_evaluation.json", report)
    manifest = {
        "schema_version": 1,
        "model_family": MODEL_FAMILY,
        "preregistration": {
            "path": str(preregistration_path.resolve()),
            "sha256": sha256_file(preregistration_path),
        },
        "development_evaluation": {
            "path": str((output_dir / "development_evaluation.json").resolve()),
            "sha256": sha256_file(output_dir / "development_evaluation.json"),
        },
        "candidate_model": {
            "path": str(candidate_model_path.resolve()),
            "sha256": sha256_file(candidate_model_path),
        },
        "candidate_pattern": {
            "path": str(candidate_pattern_path.resolve()),
            "sha256": sha256_file(candidate_pattern_path),
        },
        "activation_status": "shadow_not_promotable_without_fresh_holdout",
        "production_artifacts_modified": False,
    }
    _atomic_json(output_dir / "candidate_manifest.json", manifest)
    return report
