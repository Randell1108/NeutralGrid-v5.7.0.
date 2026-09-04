"""Quarantined scoring for the diagnostic meta-probability surrogate.

This module deliberately exposes a report writer, rather than a runtime
``MetaLabeler`` interface.  The returned values are for offline comparison
only and carry non-promotion markers in both the CSV and its manifest.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd


_REQUIRED_MARKERS: dict[str, Any] = {
    "artifact_kind": "diagnostic_meta_prob_distillation_surrogate",
    "diagnostic_only": True,
    "promotion_eligible": False,
    "runtime_meta_labeler_compatible": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_diagnostic_output(path: Path) -> Path:
    """Reject any output path that could be mistaken for a production artifact."""

    resolved = path.resolve()
    project = Path.cwd().resolve()
    forbidden = (
        project / "models",
        project / "artifacts" / "hmm",
        project / "artifacts" / "utility",
    )
    if any(resolved == item or item in resolved.parents for item in forbidden):
        raise ValueError(f"Diagnostic output must not be placed in a production path: {resolved}")
    if not any("diagnostic" in part.lower() for part in resolved.parts):
        raise ValueError("Diagnostic output path must contain a 'diagnostic' path component")
    return resolved


def load_diagnostic_surrogate(artifact_path: Path) -> dict[str, Any]:
    """Load an explicitly non-runtime surrogate, rejecting lookalike artifacts."""

    resolved = artifact_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Diagnostic surrogate does not exist: {resolved}")
    artifact = cast(dict[str, Any], joblib.load(resolved))
    for key, expected in _REQUIRED_MARKERS.items():
        if artifact.get(key) != expected:
            raise ValueError(f"Artifact is not an approved diagnostic surrogate: {key}")
    features = artifact.get("features")
    if (
        not isinstance(features, list)
        or not features
        or artifact.get("estimator") is None
        or not str(artifact.get("hmm_artifact_version", "")).strip()
    ):
        raise ValueError("Diagnostic artifact is missing features, estimator, or teacher HMM lineage")
    return artifact


def score_frame(
    frame: pd.DataFrame,
    *,
    artifact: dict[str, Any],
    allow_hmm_lineage_mismatch: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score a snapshot frame without exposing a production prediction API."""

    features = cast(list[str], artifact["features"])
    required_columns = {"candidate_id", "hmm_artifact_version", *features}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    hmm_series = cast(pd.Series, frame["hmm_artifact_version"])
    student_hmms = sorted(
        hmm_series.dropna().astype(str).str.strip().loc[lambda series: series.ne("")].unique().tolist()
    )
    if len(student_hmms) != 1:
        raise ValueError(f"Input must have one HMM lineage, found: {student_hmms}")
    student_hmm = student_hmms[0]
    teacher_hmm = str(artifact["hmm_artifact_version"]).strip()
    lineage_matches = student_hmm == teacher_hmm
    if not lineage_matches and not allow_hmm_lineage_mismatch:
        raise ValueError(
            "Input HMM lineage differs from the teacher artifact. Re-run only with "
            "--allow-hmm-lineage-mismatch to produce an explicitly extrapolative diagnostic."
        )

    feature_frame = frame[features].copy()
    for feature in features:
        feature_frame[feature] = pd.to_numeric(cast(pd.Series, feature_frame[feature]), errors="coerce")
    complete_mask = cast(pd.Series, feature_frame.notna().all(axis=1))
    predicted = np.full(len(frame), np.nan, dtype=float)
    if bool(complete_mask.any()):
        estimator = artifact["estimator"]
        predicted[complete_mask.to_numpy()] = np.clip(
            np.asarray(estimator.predict(feature_frame.loc[complete_mask]), dtype=float), 0.0, 1.0
        )

    report = pd.DataFrame(
        {
            "candidate_id": frame["candidate_id"],
            "teacher_hmm_artifact_version": teacher_hmm,
            "snapshot_hmm_artifact_version": student_hmm,
            "hmm_lineage_matches_teacher": lineage_matches,
            "surrogate_meta_prob_diagnostic": predicted,
            "feature_complete": complete_mask,
            "diagnostic_only": True,
            "promotion_eligible": False,
            "runtime_meta_labeler_compatible": False,
        }
    )
    if "meta_prob" in frame.columns:
        report["observed_snapshot_meta_prob"] = pd.to_numeric(
            cast(pd.Series, frame["meta_prob"]), errors="coerce"
        )
    summary: dict[str, Any] = {
        "rows": len(report),
        "scored_rows": int(complete_mask.sum()),
        "teacher_hmm_artifact_version": teacher_hmm,
        "snapshot_hmm_artifact_version": student_hmm,
        "hmm_lineage_matches_teacher": lineage_matches,
        "cross_lineage_extrapolation": not lineage_matches,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "runtime_meta_labeler_compatible": False,
    }
    return report, summary


def score_csv_to_shadow_report(
    *,
    artifact_path: Path,
    input_path: Path,
    output_path: Path,
    allow_hmm_lineage_mismatch: bool = False,
    input_context: str = "snapshot_csv",
) -> dict[str, Any]:
    """Write an auditable, quarantined CSV report and adjacent manifest."""

    resolved_input = input_path.resolve()
    resolved_output = assert_diagnostic_output(output_path)
    if resolved_input.suffix.lower() != ".csv":
        raise ValueError("Diagnostic scoring accepts CSV snapshot inputs only")
    if not resolved_input.is_file():
        raise FileNotFoundError(f"Diagnostic input does not exist: {resolved_input}")
    artifact = load_diagnostic_surrogate(artifact_path)
    report, summary = score_frame(
        pd.read_csv(resolved_input, low_memory=False),
        artifact=artifact,
        allow_hmm_lineage_mismatch=allow_hmm_lineage_mismatch,
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(resolved_output, index=False)
    manifest_path = resolved_output.with_suffix(".manifest.json")
    summary.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact": str(artifact_path.resolve()),
            "artifact_sha256": _sha256(artifact_path.resolve()),
            "input": str(resolved_input),
            "input_sha256": _sha256(resolved_input),
            "input_context": input_context,
            "output": str(resolved_output),
            "output_sha256": _sha256(resolved_output),
            "manifest": str(manifest_path),
            "explicit_exclusions": [
                "Not a MetaLabeler and cannot supply production meta_prob.",
                "Cannot affect gates, sizing, ranking, calibration, or deployment.",
                "Cannot be used for retraining or promotion evidence.",
            ],
        }
    )
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
