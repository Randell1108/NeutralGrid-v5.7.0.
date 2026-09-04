from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from neutralgrid.diagnostics.meta_prob_shadow import score_csv_to_shadow_report
from run_full_pipeline import _write_diagnostic_surrogate_shadow


def _write_artifact(path: Path) -> None:
    estimator = DummyRegressor(strategy="constant", constant=0.42)
    estimator.fit([[0.0], [1.0]], [0.0, 1.0])
    joblib.dump(
        {
            "artifact_kind": "diagnostic_meta_prob_distillation_surrogate",
            "diagnostic_only": True,
            "promotion_eligible": False,
            "runtime_meta_labeler_compatible": False,
            "hmm_artifact_version": "teacher_hmm",
            "features": ["feature_one"],
            "estimator": estimator,
        },
        path,
    )


def _write_snapshot(path: Path, hmm: str = "teacher_hmm") -> None:
    pd.DataFrame(
        {
            "candidate_id": ["BTCUSDT_20260827_000000_1", "ETHUSDT_20260827_000000_2"],
            "hmm_artifact_version": [hmm, hmm],
            "feature_one": [1.0, None],
        }
    ).to_csv(path, index=False)


def test_shadow_report_is_quarantined_and_auditable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "surrogate.joblib"
    input_path = tmp_path / "snapshot.csv"
    output = tmp_path / "artifacts" / "diagnostics" / "shadow" / "report.csv"
    _write_artifact(artifact)
    _write_snapshot(input_path)

    summary = score_csv_to_shadow_report(
        artifact_path=artifact,
        input_path=input_path,
        output_path=output,
        input_context="test",
    )

    report = pd.read_csv(output)
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert summary["scored_rows"] == 1
    assert report["diagnostic_only"].tolist() == [True, True]
    assert report["promotion_eligible"].tolist() == [False, False]
    assert report["runtime_meta_labeler_compatible"].tolist() == [False, False]
    assert report.loc[0, "surrogate_meta_prob_diagnostic"] == pytest.approx(0.42)
    assert pd.isna(report.loc[1, "surrogate_meta_prob_diagnostic"])
    assert manifest["input_context"] == "test"
    assert manifest["cross_lineage_extrapolation"] is False


def test_shadow_refuses_cross_lineage_without_explicit_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "surrogate.joblib"
    input_path = tmp_path / "snapshot.csv"
    _write_artifact(artifact)
    _write_snapshot(input_path, hmm="other_hmm")

    with pytest.raises(ValueError, match="differs from the teacher artifact"):
        score_csv_to_shadow_report(
            artifact_path=artifact,
            input_path=input_path,
            output_path=tmp_path / "artifacts" / "diagnostics" / "shadow" / "report.csv",
        )


def test_shadow_refuses_production_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "surrogate.joblib"
    input_path = tmp_path / "snapshot.csv"
    _write_artifact(artifact)
    _write_snapshot(input_path)

    with pytest.raises(ValueError, match="production path"):
        score_csv_to_shadow_report(
            artifact_path=artifact,
            input_path=input_path,
            output_path=tmp_path / "models" / "meta_prob_shadow.csv",
        )


def test_pipeline_shadow_export_does_not_mutate_deployment_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "surrogate.joblib"
    deployment_csv = tmp_path / "deployment_ready.csv"
    output_dir = tmp_path / "artifacts" / "diagnostics" / "shadow"
    _write_artifact(artifact)
    _write_snapshot(deployment_csv, hmm="other_hmm")
    original = deployment_csv.read_bytes()
    args = SimpleNamespace(
        diagnostic_surrogate_artifact=artifact,
        diagnostic_surrogate_output_dir=output_dir,
        diagnostic_surrogate_allow_hmm_lineage_mismatch=True,
    )

    _write_diagnostic_surrogate_shadow(
        args=args,
        pipeline_output=deployment_csv,
        timestamp="20260827_000000",
    )

    report = output_dir / "pipeline_20260827_000000" / "meta_prob_shadow.csv"
    assert deployment_csv.read_bytes() == original
    assert report.is_file()
    assert pd.read_csv(report)["promotion_eligible"].tolist() == [False, False]
