"""Policy tests for HMM artifact resolution and promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neutralgrid.core.exceptions import NeutralGridError
from neutralgrid.models import artifacts


def _write_metadata(path: Path, version: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"artifact_version": version}, f)


def test_resolve_hmm_artifact_dir_rejects_legacy_global_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    global_dir = tmp_path / "artifacts" / "hmm" / "global"
    _write_metadata(global_dir, "rolling_180d_20260228_005417")

    monkeypatch.delenv("HMM_ARTIFACT_DIR", raising=False)
    monkeypatch.setattr(artifacts, "get_active_hmm_dir", lambda: global_dir)
    monkeypatch.setattr(
        artifacts,
        "get_active_hmm_version",
        lambda: "rolling_180d_20260228_005417",
    )

    with pytest.raises(NeutralGridError, match="legacy HMM global artifact"):
        artifacts.resolve_hmm_artifact_dir()


def test_resolve_hmm_artifact_dir_rejects_metadata_dir_version_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    version = "rolling_180d_20260228_005417"
    artifact_dir = tmp_path / "artifacts" / "hmm" / version
    _write_metadata(artifact_dir, "rolling_180d_20260227_123000")

    monkeypatch.delenv("HMM_ARTIFACT_DIR", raising=False)
    monkeypatch.setattr(artifacts, "get_active_hmm_dir", lambda: artifact_dir)
    monkeypatch.setattr(artifacts, "get_active_hmm_version", lambda: version)

    with pytest.raises(NeutralGridError, match="metadata mismatch"):
        artifacts.resolve_hmm_artifact_dir()


def test_promote_hmm_version_requires_rolling_180d_and_metadata_match(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts" / "hmm" / "rolling_180d_20260228_005417"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "model.joblib").write_text("dummy", encoding="utf-8")
    (artifact_dir / "metadata.json").write_text(
        json.dumps({"artifact_version": "rolling_180d_20260227_123000"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inconsistent metadata"):
        artifacts.promote_hmm_version("rolling_180d_20260228_005417", artifact_dir)

    with pytest.raises(ValueError, match="must match rolling_180d_YYYYMMDD_HHMMSS"):
        artifacts.promote_hmm_version("rolling_90d_20260228_005417", artifact_dir)
