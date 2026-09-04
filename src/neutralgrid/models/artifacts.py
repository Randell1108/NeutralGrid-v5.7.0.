"""
Model artifact management with metadata and versioning.

Every deployed model must be traceable with:
- Version information (semantic or timestamp)
- Training metadata (when, what data, what features)
- Feature schema (ordered list, transformation details)
- Evaluation metrics (offline performance summary)
- Code provenance (git commit hash if available)

This ensures production stability and prevents silent breakage from
training/inference mismatch.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List, Dict

from neutralgrid.core.exceptions import ArtifactVersionError, FeatureSchemaError

logger = logging.getLogger(__name__)


_ROLLING_180D_VERSION_RE = re.compile(r"^rolling_180d_\d{8}_\d{6}$")


def _resolve_path(path_like: str | Path) -> Path:
    """Resolve an absolute/project-relative path consistently."""
    from neutralgrid.core.config import get_config

    cfg = get_config()
    p = Path(path_like)
    return p if p.is_absolute() else cfg.resolve_path(str(p))


def _is_legacy_global_hmm_dir(path: Path) -> bool:
    """Return True when path points to legacy artifacts/hmm/global."""
    try:
        parts = [p.lower() for p in path.parts]
        return len(parts) >= 2 and parts[-2:] == ["hmm", "global"]
    except Exception:
        return False


def _read_metadata_artifact_version(artifact_dir: Path) -> Optional[str]:
    """Read metadata.artifact_version from an artifact directory."""
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    version = metadata.get("artifact_version")
    return str(version) if version is not None else None


def _validate_hmm_artifact_dir(
    artifact_dir: Path,
    *,
    source: str,
    expected_version: Optional[str] = None,
) -> None:
    """Fail-loud validation for production HMM artifact directories."""
    from neutralgrid.core.exceptions import NeutralGridError

    resolved = _resolve_path(artifact_dir)

    if _is_legacy_global_hmm_dir(resolved):
        raise NeutralGridError(
            f"{source} points to legacy HMM global artifact dir ({resolved}). "
            "Use manifest-promoted rolling_180d artifact directories only."
        )

    dir_version = resolved.name
    if not _ROLLING_180D_VERSION_RE.match(dir_version):
        raise NeutralGridError(
            f"{source} resolved to unsupported HMM artifact dir ({resolved}). "
            "Expected directory name: rolling_180d_YYYYMMDD_HHMMSS"
        )

    if expected_version is not None and dir_version != expected_version:
        raise NeutralGridError(
            f"{source} version mismatch: expected {expected_version}, "
            f"resolved directory {resolved}"
        )

    if not resolved.exists():
        raise NeutralGridError(
            f"{source} resolved to missing HMM artifact dir: {resolved}"
        )

    metadata_version = _read_metadata_artifact_version(resolved)
    if metadata_version is None:
        raise NeutralGridError(
            f"{source} missing metadata.artifact_version in {resolved / 'metadata.json'}"
        )

    if metadata_version != dir_version:
        raise NeutralGridError(
            f"{source} metadata mismatch: dir={dir_version} "
            f"but metadata.artifact_version={metadata_version} in {resolved}"
        )

    if expected_version is not None and metadata_version != expected_version:
        raise NeutralGridError(
            f"{source} metadata version mismatch: expected {expected_version}, "
            f"found {metadata_version} in {resolved}"
        )


def resolve_hmm_artifact_dir() -> Path:
    """Return the canonical HMM artifact directory from config.

    Resolution order:
    1. ``HMM_ARTIFACT_DIR`` environment variable (explicit override).
    2. Active version from the artifact manifest.

    Raises ``NeutralGridError`` if neither tier resolves to a valid directory.
    This fail-loud behavior ensures broken configurations are caught immediately
    instead of silently falling back to a stale legacy directory.

    This is the single source of truth for locating HMM model artifacts.
    All modules that need to load or save HMM artifacts should call this
    instead of constructing paths themselves.
    """
    from neutralgrid.core.exceptions import NeutralGridError

    active_version = get_active_hmm_version()
    active_dir = get_active_hmm_dir()

    # 1. Explicit env-var override (allowed only when consistent with manifest)
    env_dir = os.getenv("HMM_ARTIFACT_DIR")
    if env_dir:
        resolved_env_dir = _resolve_path(env_dir)
        if active_dir is not None and resolved_env_dir != active_dir:
            raise NeutralGridError(
                "Conflicting HMM artifact configuration: HMM_ARTIFACT_DIR "
                f"({resolved_env_dir}) != manifest active dir ({active_dir}). "
                "Use one canonical artifact source (artifact_manifest.json)."
            )
        _validate_hmm_artifact_dir(
            resolved_env_dir,
            source="HMM_ARTIFACT_DIR",
            expected_version=active_version if active_dir is not None else None,
        )
        return resolved_env_dir

    # 2. Manifest active version
    if active_dir is not None:
        _validate_hmm_artifact_dir(
            active_dir,
            source="artifact_manifest.json",
            expected_version=active_version,
        )
        return active_dir

    raise NeutralGridError(
        "No HMM artifact directory found. Neither HMM_ARTIFACT_DIR env var "
        "nor artifact_manifest.json resolve to a valid directory. "
        "Run: neutralgrid-retrain --window 180d --top-n 60"
    )


@dataclass
class ArtifactMetadata:
    """
    Metadata for ML model artifacts.

    All deployed models MUST have this metadata to ensure traceability
    and prevent silent training/inference mismatch.

    AFML Compliance (Station 4):
    - symbol_selection_method documents how symbols were chosen (survivorship bias mitigation)
    - holdout_info documents true holdout period used
    """

    # Version and provenance
    artifact_version: str                    # Semantic version or timestamp
    trained_at_utc: str                      # ISO 8601 timestamp
    pipeline_version: Optional[str] = None   # neutralgrid package version used to train/save
    code_commit: Optional[str] = None        # Git commit hash (if available)
    lineage: Optional[Dict[str, Any]] = None # Parent artifact references (e.g., active HMM for meta-labeler)

    # Training data provenance
    training_start_utc: Optional[str] = None # Data window start
    training_end_utc: Optional[str] = None   # Data window end
    symbols_used: Optional[List[str]] = None # Symbols in training set
    num_sequences: Optional[int] = None      # Number of training sequences
    total_samples: Optional[int] = None      # Total training samples

    # AFML Symbol Selection Documentation (Station 4 Compliance)
    # Documents how symbols were selected to address survivorship bias
    symbol_selection_method: Optional[str] = None  # e.g., "top_100_by_volume", "all_perpetual"
    symbol_selection_date: Optional[str] = None    # When symbol list was generated
    symbol_selection_criteria: Optional[Dict[str, Any]] = None  # e.g., {"min_volume_24h": 1000000}
    includes_delisted: bool = False                # Whether delisted symbols included

    # AFML True Holdout Documentation (Station 4 Compliance)
    holdout_info: Optional[Dict[str, Any]] = None  # From CPCV.get_holdout_metrics()

    # Feature schema
    features: Optional[List[str]] = None      # Ordered feature names
    timeframes_used: Optional[List[str]] = None  # e.g., ["1h", "15m"]
    transform_type: Optional[str] = None     # e.g., "RobustScaler"
    transform_params: Optional[Dict[str, Any]] = None  # e.g., {"quantile_range": [25, 75]}

    # Feature transform schema version (v1=raw, v2=stationary transforms)
    feature_schema: Optional[str] = None     # e.g., "v1" or "v2"

    # Model hyperparameters
    model_type: str = "GaussianHMM"          # Model class name
    model_params: Optional[Dict[str, Any]] = None  # Hyperparameters

    # Evaluation metrics (offline)
    eval_metrics: Optional[Dict[str, Any]] = None  # Performance summary

    # State mapping (for HMM)
    state_mapping: Optional[Dict[str, int]] = None  # e.g., {"range": 0, "trend": 1}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ArtifactMetadata:
        """Load from dictionary (ignores unknown keys for forward compatibility)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def get_git_commit() -> Optional[str]:
    """
    Get current git commit hash.

    Returns:
        Git commit hash (short form) or None if not in git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("git rev-parse failed: %s", e)
    return None


def save_artifact(
    artifact_dir: Path,
    model: Any,
    scaler: Any,
    metadata: ArtifactMetadata,
    feature_schema: Optional[Dict[str, Any]] = None,
    eval_metrics: Optional[Dict[str, Any]] = None,
    state_means_unscaled: Optional[Any] = None,
) -> Path:
    """
    Save model artifact with metadata and versioning.

    Artifact directory structure:
        artifacts/
          hmm/
            rolling_180d_YYYYMMDD_HHMMSS/
              model.joblib           # HMM model
              scaler.joblib          # Feature scaler (if used)
              metadata.json          # Training metadata
              feature_schema.json    # Feature definitions
              eval.json              # Offline metrics (optional)
              state_means.npy        # State means (for HMM)

    Args:
        artifact_dir: Directory to save artifacts (e.g., "artifacts/hmm/rolling_180d_...")
        model: Trained model object (e.g., GaussianHMM)
        scaler: Feature scaler object (e.g., RobustScaler)
        metadata: ArtifactMetadata with training provenance
        feature_schema: Optional feature schema dict
        eval_metrics: Optional evaluation metrics dict
        state_means_unscaled: Optional state means (for HMM interpretability)

    Returns:
        Path to artifact directory

    Raises:
        ValueError: If metadata is incomplete or invalid
    """
    from joblib import dump
    import numpy as np

    # Validate metadata
    if not metadata.artifact_version:
        raise ValueError("artifact_version is required")
    if not metadata.trained_at_utc:
        raise ValueError("trained_at_utc is required")
    if not metadata.features:
        raise ValueError("features list is required")

    artifact_dir = Path(artifact_dir)
    staging_dir = artifact_dir.parent / f".staging_{artifact_dir.name}"
    backup_dir = artifact_dir.parent / f".backup_{artifact_dir.name}"

    try:
        # Write to staging directory first
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = staging_dir / "model.joblib"
        dump(model, model_path, compress=3)

        # Save scaler
        if scaler is not None:
            scaler_path = staging_dir / "scaler.joblib"
            dump(scaler, scaler_path, compress=3)

        # Save state means (for HMM)
        if state_means_unscaled is not None:
            state_means_path = staging_dir / "state_means.npy"
            np.save(state_means_path, state_means_unscaled)

        # Update metadata with eval metrics if provided
        if eval_metrics is not None:
            metadata.eval_metrics = eval_metrics

        # Save metadata
        metadata_path = staging_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

        # Save feature schema
        if feature_schema is not None:
            schema_path = staging_dir / "feature_schema.json"
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(feature_schema, f, indent=2, ensure_ascii=False)

        # Save evaluation metrics separately (optional)
        if eval_metrics is not None:
            eval_path = staging_dir / "eval.json"
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(eval_metrics, f, indent=2, ensure_ascii=False)

        # Atomic swap: existing → backup, staging → final
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if artifact_dir.exists():
            artifact_dir.rename(backup_dir)
        staging_dir.rename(artifact_dir)

        # Clean backup on success
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

    except Exception:
        # Clean staging on failure, leave existing dir intact
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        # Restore from backup if the original was moved
        if not artifact_dir.exists() and backup_dir.exists():
            backup_dir.rename(artifact_dir)
        raise

    return artifact_dir


def load_artifact(
    artifact_dir: Path,
    expected_version: Optional[str] = None,
    validate_schema: bool = True,
) -> Dict[str, Any]:
    """
    Load model artifact with metadata validation.

    Args:
        artifact_dir: Directory containing artifacts (e.g., "artifacts/hmm/rolling_180d_...")
        expected_version: Optional version to validate against
        validate_schema: Whether to validate feature schema

    Returns:
        Dictionary with keys:
        - model: Loaded model object
        - scaler: Loaded scaler object (or None)
        - metadata: ArtifactMetadata object
        - feature_schema: Feature schema dict (or None)
        - state_means_unscaled: State means array (or None)

    Raises:
        FileNotFoundError: If artifact directory or required files missing
        ValueError: If version mismatch or schema validation fails
    """
    from joblib import load
    import numpy as np

    artifact_dir = Path(artifact_dir)

    # Check directory exists
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")

    # Load metadata
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_dict = json.load(f)
    metadata = ArtifactMetadata.from_dict(metadata_dict)

    # Validate version if specified
    if expected_version is not None and metadata.artifact_version != expected_version:
        # AFML: Alert on version mismatch (Station 6 compliance)
        try:
            from neutralgrid.models.alerts import alert_version_mismatch
            alert_version_mismatch(
                expected_version=expected_version,
                found_version=metadata.artifact_version,
                artifact_dir=str(artifact_dir),
            )
        except ImportError:
            pass  # Alerts module not available
        raise ArtifactVersionError(
            expected=expected_version,
            found=metadata.artifact_version,
            artifact_dir=str(artifact_dir),
        )

    # Validate feature schema against repository canonical schema (default HMM features).
    # This prevents silent training/inference mismatch even when no explicit
    # feature_schema.json file is present.
    if validate_schema:
        try:
            from neutralgrid.models.hmm.schema import FEATURE_NAMES as _CANON_FEATURES
            canon = list(_CANON_FEATURES)
            if metadata.features != canon:
                raise FeatureSchemaError(
                    expected=canon,
                    found=metadata.features,
                )
        except ImportError:
            # If canonical schema is unavailable, skip this validation.
            pass

    # Load model
    model_path = artifact_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = load(model_path)

    # Load scaler (optional)
    scaler = None
    scaler_path = artifact_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = load(scaler_path)

    # Load feature schema (optional)
    feature_schema = None
    schema_path = artifact_dir / "feature_schema.json"
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            feature_schema = json.load(f)

        # Validate schema if requested
        if validate_schema and feature_schema is not None:
            schema_features = feature_schema.get("features", [])
            if schema_features != metadata.features:
                # AFML: Alert on schema mismatch (Station 6 compliance)
                try:
                    from neutralgrid.models.alerts import alert_schema_mismatch
                    alert_schema_mismatch(
                        expected_features=metadata.features or [],
                        found_features=schema_features,
                    )
                except ImportError:
                    pass  # Alerts module not available
                raise FeatureSchemaError(
                    expected=metadata.features,
                    found=schema_features,
                    detail=f"Feature schema file mismatch: metadata features={metadata.features}, schema features={schema_features}",
                )

    # Load state means (optional, for HMM)
    state_means_unscaled = None
    state_means_path = artifact_dir / "state_means.npy"
    if state_means_path.exists():
        state_means_unscaled = np.load(state_means_path, allow_pickle=False)

    return {
        "model": model,
        "scaler": scaler,
        "metadata": metadata,
        "feature_schema": feature_schema,
        "state_means_unscaled": state_means_unscaled,
    }


# =========================================================================
# ARTIFACT MANIFEST
# =========================================================================


def _manifest_path() -> Path:
    """Return path to the artifact manifest (project root, git-trackable)."""
    from neutralgrid.core.config import get_config

    cfg = get_config()
    return cfg.base_dir / "artifact_manifest.json"


def load_manifest() -> Optional[Dict[str, Any]]:
    """Load the artifact manifest. Returns ``None`` if absent or corrupt."""
    mp = _manifest_path()
    if not mp.exists():
        return None
    try:
        with open(mp, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read artifact manifest %s: %s", mp, e)
        return None


def save_manifest(manifest: Dict[str, Any]) -> None:
    """Atomically write the artifact manifest (tmp + os.replace)."""
    mp = _manifest_path()
    mp.parent.mkdir(parents=True, exist_ok=True)

    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()

    fd, tmp = tempfile.mkstemp(
        dir=str(mp.parent), prefix=".manifest_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        os.replace(tmp, str(mp))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def get_active_hmm_version() -> Optional[str]:
    """Read ``manifest["hmm"]["active_version"]``, or ``None``."""
    manifest = load_manifest()
    if manifest is None:
        return None
    hmm = manifest.get("hmm")
    if not isinstance(hmm, dict):
        return None
    return hmm.get("active_version")


def get_active_hmm_dir() -> Optional[Path]:
    """Resolve the versioned artifact dir from the manifest.

    Returns ``None`` if no manifest or missing entry.
    """
    manifest = load_manifest()
    if manifest is None:
        return None
    hmm = manifest.get("hmm")
    if not isinstance(hmm, dict):
        return None
    rel = hmm.get("artifact_dir")
    if not rel:
        return None

    from neutralgrid.core.config import get_config

    cfg = get_config()
    resolved = cfg.resolve_path(rel)
    if resolved.exists():
        return resolved
    # Manifest points to a dir that no longer exists — fall through
    logger.warning("Manifest HMM artifact_dir does not exist: %s", resolved)
    return None


def promote_hmm_version(version: str, artifact_dir: Path) -> None:
    """Validate required files and atomically update the manifest.

    Args:
        version: The version string (e.g. ``rolling_180d_20260215_120000``).
        artifact_dir: Absolute or project-relative path to the versioned dir.

    Raises:
        FileNotFoundError: If required artifact files are missing.
    """
    artifact_dir = Path(artifact_dir)

    if not _ROLLING_180D_VERSION_RE.match(version):
        raise ValueError(
            "HMM production version must match rolling_180d_YYYYMMDD_HHMMSS; "
            f"got {version!r}"
        )
    if artifact_dir.name != version:
        raise ValueError(
            f"Artifact dir/version mismatch: dir={artifact_dir.name!r}, version={version!r}"
        )
    if _is_legacy_global_hmm_dir(artifact_dir):
        raise ValueError(
            f"Cannot promote legacy global HMM artifact dir: {artifact_dir}"
        )

    required = ["model.joblib", "metadata.json"]
    for fname in required:
        if not (artifact_dir / fname).exists():
            raise FileNotFoundError(
                f"Cannot promote version {version}: "
                f"missing {fname} in {artifact_dir}"
            )

    metadata_version = _read_metadata_artifact_version(artifact_dir)
    if metadata_version != version:
        raise ValueError(
            "Cannot promote HMM artifact with inconsistent metadata: "
            f"metadata.artifact_version={metadata_version!r}, expected={version!r}"
        )

    # Build relative path for manifest (relative to project root)
    from neutralgrid.core.config import get_config

    cfg = get_config()
    try:
        rel_dir = str(artifact_dir.relative_to(cfg.base_dir))
    except ValueError:
        rel_dir = str(artifact_dir)

    manifest = load_manifest() or {"schema_version": "1.0.0"}
    manifest["hmm"] = {
        "active_version": version,
        "artifact_dir": rel_dir,
        "promoted_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_manifest(manifest)
    logger.info("Promoted HMM version %s (dir: %s)", version, rel_dir)
