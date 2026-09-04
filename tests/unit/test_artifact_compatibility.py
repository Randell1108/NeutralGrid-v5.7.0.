from __future__ import annotations

import pytest

from neutralgrid.models.artifact_compat import (
    evaluate_hmm_meta_labeler_compatibility,
    require_hmm_meta_labeler_compatibility,
)


def test_lineage_match_is_compatible_even_with_different_artifact_timestamps() -> None:
    hmm_meta = {
        "artifact_version": "rolling_180d_20260301_010101",
        "pipeline_version": "6.5.6",
    }
    meta_meta = {
        "artifact_version": "20260301_020202",  # independent timestamp
        "pipeline_version": "6.5.6",
        "lineage": {
            "hmm_artifact_version": "rolling_180d_20260301_010101",
            "hmm_pipeline_version": "6.5.6",
        },
    }

    issues = evaluate_hmm_meta_labeler_compatibility(
        hmm_metadata=hmm_meta,
        meta_metadata=meta_meta,
    )
    assert issues == []


def test_lineage_hmm_version_mismatch_is_reported() -> None:
    hmm_meta = {
        "artifact_version": "rolling_180d_20260301_010101",
        "pipeline_version": "6.5.6",
    }
    meta_meta = {
        "artifact_version": "20260301_020202",
        "pipeline_version": "6.5.6",
        "lineage": {"hmm_artifact_version": "rolling_180d_20260228_235959"},
    }

    issues = evaluate_hmm_meta_labeler_compatibility(
        hmm_metadata=hmm_meta,
        meta_metadata=meta_meta,
    )
    assert len(issues) == 1
    assert "lineage mismatch" in issues[0]


def test_pipeline_version_fallback_detects_mismatch_when_lineage_missing() -> None:
    hmm_meta = {
        "artifact_version": "rolling_180d_20260301_010101",
        "pipeline_version": "6.5.6",
    }
    meta_meta = {
        "artifact_version": "20260301_020202",
        "pipeline_version": "6.4.9",
    }

    issues = evaluate_hmm_meta_labeler_compatibility(
        hmm_metadata=hmm_meta,
        meta_metadata=meta_meta,
    )
    assert len(issues) == 1
    assert "pipeline_version mismatch" in issues[0]


def test_require_hmm_meta_labeler_compatibility_raises_on_mismatch() -> None:
    hmm_meta = {
        "artifact_version": "rolling_180d_20260301_010101",
        "pipeline_version": "6.5.6",
    }
    meta_meta = {
        "artifact_version": "20260301_020202",
        "pipeline_version": "6.5.6",
        "lineage": {"hmm_artifact_version": "rolling_180d_20260228_235959"},
    }

    with pytest.raises(ValueError, match="lineage mismatch"):
        require_hmm_meta_labeler_compatibility(
            hmm_metadata=hmm_meta,
            meta_metadata=meta_meta,
        )
