"""Authority boundaries for backtest realism profiles.

The non-legacy profiles are diagnostic challengers.  This module centralizes
their names and prevents any label-bearing shadow output from being written
into an authoritative training directory.
"""

from __future__ import annotations

from pathlib import Path

from neutralgrid.core.config import get_config


LEGACY_REALISM_PROFILE = "legacy"
CANDIDATE_TIME_GEOMETRIC_PROFILE = "candidate_time_geometric_v1"
CANDIDATE_TIME_PUBLIC_MARKET_PROFILE = "candidate_time_public_market_v1"
REALISM_PROFILES = (
    LEGACY_REALISM_PROFILE,
    CANDIDATE_TIME_GEOMETRIC_PROFILE,
    CANDIDATE_TIME_PUBLIC_MARKET_PROFILE,
)
SHADOW_REALISM_PROFILES = frozenset(
    {
        CANDIDATE_TIME_GEOMETRIC_PROFILE,
        CANDIDATE_TIME_PUBLIC_MARKET_PROFILE,
    }
)
CANONICAL_REALISM_TRAINING_DIRS = (
    Path("data/backtest_candidates"),
    Path("data/fastwin_dataset"),
)


def validate_realism_profile(profile: str) -> str:
    """Return a supported normalized profile or fail closed."""
    normalized = str(profile).strip().lower()
    if normalized not in REALISM_PROFILES:
        raise ValueError(
            f"Unsupported realism_profile={profile!r}; "
            f"expected one of {', '.join(REALISM_PROFILES)}"
        )
    return normalized


def validate_realism_output_path(
    realism_profile: str,
    output: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> Path:
    """Resolve an output path and reject shadow writes into canonical pools.

    Relative output paths resolve exactly as ``Path(output)`` resolves at the
    writer: against the process working directory.  Canonical roots resolve
    against the configured project base directory.  ``Path.resolve`` also
    closes ``..`` and symlink aliases before the containment comparison.
    """
    profile = validate_realism_profile(realism_profile)
    resolved_output = Path(output).resolve()
    if profile == LEGACY_REALISM_PROFILE:
        return resolved_output

    project_root = Path(base_dir or get_config().base_dir).resolve()
    for relative_root in CANONICAL_REALISM_TRAINING_DIRS:
        canonical_root = (project_root / relative_root).resolve()
        if resolved_output == canonical_root or canonical_root in resolved_output.parents:
            raise ValueError(
                f"realism_profile={profile!r} is shadow-only and cannot write "
                f"into the canonical training tree {canonical_root}; pass an "
                "explicit isolated output path"
            )
    return resolved_output
