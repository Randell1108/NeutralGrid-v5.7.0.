"""
Cross-artifact compatibility checks.

This module centralizes compatibility logic between independently trained
artifacts (for example HMM and meta-labeler) so pipeline-level warnings are
specific and lineage-aware instead of comparing unrelated timestamp strings.
"""

from __future__ import annotations

from typing import Any, Mapping


def evaluate_hmm_meta_labeler_compatibility(
    *,
    hmm_metadata: Mapping[str, Any] | None,
    meta_metadata: Mapping[str, Any] | None,
) -> list[str]:
    """Return compatibility issues between HMM and meta-labeler metadata.

    Rules:
    - If meta-labeler declares explicit HMM lineage, lineage must match active HMM.
    - If lineage is absent, fall back to pipeline_version comparison.
    - Never compare artifact_version timestamps directly across different model types.
    """
    issues: list[str] = []
    if not hmm_metadata or not meta_metadata:
        return issues

    hmm_version = hmm_metadata.get("artifact_version")
    hmm_pipeline = hmm_metadata.get("pipeline_version")

    lineage = meta_metadata.get("lineage")
    lineage = lineage if isinstance(lineage, Mapping) else {}

    linked_hmm = lineage.get("hmm_artifact_version")
    linked_hmm_pipeline = lineage.get("hmm_pipeline_version")
    meta_pipeline = meta_metadata.get("pipeline_version")

    if linked_hmm and hmm_version and str(linked_hmm) != str(hmm_version):
        issues.append(
            "meta_labeler lineage mismatch: "
            f"linked hmm_artifact_version={linked_hmm} "
            f"but active hmm artifact_version={hmm_version}"
        )

    if linked_hmm_pipeline and hmm_pipeline and str(linked_hmm_pipeline) != str(hmm_pipeline):
        issues.append(
            "meta_labeler lineage mismatch: "
            f"linked hmm_pipeline_version={linked_hmm_pipeline} "
            f"but active hmm pipeline_version={hmm_pipeline}"
        )

    # Backward-compatible fallback for older metadata without lineage.
    if not linked_hmm and not linked_hmm_pipeline:
        if meta_pipeline and hmm_pipeline and str(meta_pipeline) != str(hmm_pipeline):
            issues.append(
                "pipeline_version mismatch without lineage metadata: "
                f"hmm={hmm_pipeline}, meta_labeler={meta_pipeline}"
            )

    return issues


def require_hmm_meta_labeler_compatibility(
    *,
    hmm_metadata: Mapping[str, Any] | None,
    meta_metadata: Mapping[str, Any] | None,
) -> None:
    """Raise when HMM/meta-labeler artifact lineage is incompatible."""
    issues = evaluate_hmm_meta_labeler_compatibility(
        hmm_metadata=hmm_metadata,
        meta_metadata=meta_metadata,
    )
    if issues:
        raise ValueError("; ".join(str(issue) for issue in issues))
