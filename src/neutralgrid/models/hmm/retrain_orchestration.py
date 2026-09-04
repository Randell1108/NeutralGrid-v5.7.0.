"""Shared HMM retrain orchestration helpers.

This module keeps entrypoint behavior aligned without changing the core
trainer. Both retrain entrypoints should source promotion, eval artifact
materialization, scaler policy, and trial logging from here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from neutralgrid.models.artifacts import promote_hmm_version
from neutralgrid.models.hmm.schema import FEATURE_NAMES

logger = logging.getLogger(__name__)

MIN_PASS_RATE = 0.50


def load_hmm_metadata(artifact_path: Path) -> dict[str, Any]:
    """Load HMM artifact metadata from disk."""
    metadata_path = Path(artifact_path) / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid metadata payload at {metadata_path}")
    return data


def sync_eval_json_from_metadata(artifact_path: Path) -> dict[str, Any]:
    """Materialize eval.json from the authoritative metadata eval_metrics."""
    artifact_path = Path(artifact_path)
    metadata = load_hmm_metadata(artifact_path)
    eval_metrics = metadata.get("eval_metrics")
    if not isinstance(eval_metrics, dict) or not eval_metrics:
        raise ValueError(f"No eval_metrics in {artifact_path / 'metadata.json'}")
    eval_path = artifact_path / "eval.json"
    eval_path.write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
    return eval_metrics


def disable_temperature_scaling_artifact(
    artifact_path: Path,
    *,
    status: str = "disabled_self_supervised",
    label_source: str = "self_argmax",
) -> Path:
    """Write an identity temperature scaler with explicit non-runtime provenance."""
    from neutralgrid.calibration.temperature_scaling_v20260311 import TemperatureScaler

    artifact_path = Path(artifact_path)
    scaler = TemperatureScaler()
    scaler.set_metadata(
        runtime_validated=False,
        status=status,
        label_source=label_source,
        calibration_target="disabled",
        notes="Self-labeled temperature scaling disabled; identity scaler retained for provenance.",
    )
    ts_path = artifact_path / "temperature_scaler.json"
    scaler.save(ts_path)
    return ts_path


def evaluate_and_maybe_promote(
    *,
    version: str,
    artifact_path: Path,
    requested_symbols: int,
    fetched_datasets: Mapping[str, Any],
    window_days: int,
    actual_window_days: float | None = None,
    canonical_mode: bool = False,
    accepted_count: int | None = None,
    trained_count: int | None = None,
    probe_count: int | None = None,
) -> tuple[dict[str, Any], bool, list[str]]:
    """Apply shared promotion policy using the trainer's authoritative evaluation.

    When *canonical_mode* is True the symbol-coverage gate requires exact
    counts (``accepted_count == 50 AND trained_count == 50``) rather than
    the looser ``len(fetched_datasets) >= max(3, requested // 2)`` rule.
    The additional counters (``probe_count``, ``accepted_count``,
    ``trained_count``) are recorded in *eval_metrics* for audit.
    """
    eval_metrics = sync_eval_json_from_metadata(artifact_path)

    reasons: list[str] = []
    mean_pass_rate = float(eval_metrics.get("mean_pass_rate", 0.0))

    if canonical_mode:
        # ── Strict canonical gates ────────────────────────────────
        if accepted_count != 50 or trained_count != 50:
            reasons.append(
                f"canonical_symbol_coverage: accepted_count={accepted_count}, "
                f"trained_count={trained_count} (both must be 50)"
            )
        logger.info(
            "Canonical promotion check: probe_count=%s, accepted_count=%s, "
            "trained_count=%s",
            probe_count, accepted_count, trained_count,
        )
    else:
        # ── Legacy / backward-compatible gate ─────────────────────
        min_symbols = max(3, requested_symbols // 2)
        if len(fetched_datasets) < min_symbols:
            reasons.append(
                f"symbol_coverage={len(fetched_datasets)}/{requested_symbols} "
                f"< {min_symbols}"
            )

    if window_days != 180:
        reasons.append(f"window_days={window_days} != 180")
    if actual_window_days is not None and actual_window_days < (window_days * 0.90):
        reasons.append(
            f"actual_window_days={actual_window_days:.1f} < required_min={window_days * 0.90:.1f}"
        )
    if mean_pass_rate < MIN_PASS_RATE:
        reasons.append(
            f"mean_pass_rate={mean_pass_rate:.4f} < threshold={MIN_PASS_RATE:.2f}"
        )

    # ── Record canonical counters in eval_metrics for audit ───────
    eval_metrics["canonical_mode"] = canonical_mode
    eval_metrics["probe_count"] = probe_count
    eval_metrics["accepted_count"] = accepted_count
    eval_metrics["trained_count"] = trained_count

    promoted = not reasons
    if promoted:
        promote_hmm_version(version, artifact_path)
    return eval_metrics, promoted, reasons


def log_hmm_training_trial(
    *,
    artifact_path: Path,
    version: str,
    requested_symbols: int,
    fetched_datasets: Mapping[str, Any],
    bars_per_symbol: int,
    n_components: int,
    n_iter: int,
    window_days: int,
    promoted: bool,
    promotion_reasons: list[str],
) -> None:
    """Write a consistent trial log record for HMM retrains."""
    from neutralgrid.training.trial_tracker import TrialRecord, TrialTracker

    metadata = load_hmm_metadata(artifact_path)
    eval_metrics = metadata.get("eval_metrics") or {}
    model_params = metadata.get("model_params") or {}

    ts = datetime.now(timezone.utc)
    notes_parts = [
        f"artifact_version={version}",
        f"artifact_path={Path(artifact_path)}",
        f"promoted={promoted}",
        f"evaluation_source=metadata_eval_metrics",
        f"window_days={window_days}",
    ]
    if promotion_reasons:
        notes_parts.append("promotion_reasons=" + ",".join(promotion_reasons))

    trial = TrialRecord(
        trial_id=f"hmm_{ts.strftime('%Y%m%d_%H%M%S')}",
        timestamp=ts,
        model_type="hmm",
        hyperparameters={
            "n_components": n_components,
            "n_iter": n_iter,
            "n_symbols_requested": requested_symbols,
            "n_symbols_fetched": len(fetched_datasets),
            "bars_per_symbol": bars_per_symbol,
            "window_days": window_days,
            "covariance_type": model_params.get("covariance_type", "diag"),
            "artifact_version": version,
            "promoted": promoted,
        },
        cv_score=float(eval_metrics.get("mean_pass_rate", 0.0)),
        feature_set=list(FEATURE_NAMES),
        notes="; ".join(notes_parts),
    )
    TrialTracker().log_trial(trial)
    logger.info("Trial logged: %s", trial.trial_id)


def check_threshold_artifact_compatibility(
    artifact_dir: Path,
    hmm_version: str,
) -> tuple[bool, str]:
    """Verify that a CPCV threshold artifact is compatible with the HMM version.

    Checks:
    1. ``cpcv_selected_threshold.json`` exists in *artifact_dir*.
    2. If present, ``hmm_version`` recorded inside matches *hmm_version*.
    3. ``pass_mode`` matches the current runtime config (already enforced by
       ``regime_validator._load_cpcv_auto_threshold``, but verified here for
       promotion-time audit).

    Returns:
        ``(True, "ok")`` if compatible or absent (no threshold artifact is
        acceptable — runtime falls back to config defaults).
        ``(False, reason)`` if a threshold artifact exists but is
        incompatible with the specified HMM version.
    """
    from neutralgrid.core.config import get_config

    artifact_dir = Path(artifact_dir)
    threshold_path = artifact_dir / "cpcv_selected_threshold.json"

    if not threshold_path.exists():
        logger.info(
            "No cpcv_selected_threshold.json in %s — compatible (absent).",
            artifact_dir,
        )
        return (True, "ok")

    try:
        with open(threshold_path, "r", encoding="utf-8") as f:
            artifact = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        reason = f"Failed to read {threshold_path}: {exc}"
        logger.warning(reason)
        return (False, reason)

    # ── HMM version lineage check ────────────────────────────────
    artifact_hmm_version = artifact.get("hmm_version")
    if artifact_hmm_version is not None and str(artifact_hmm_version) != hmm_version:
        reason = (
            f"Threshold artifact hmm_version={artifact_hmm_version!r} does not "
            f"match promoted hmm_version={hmm_version!r}"
        )
        logger.warning(reason)
        return (False, reason)

    # ── pass_mode compatibility check ────────────────────────────
    _cfg = get_config()
    runtime_pass_mode = str(_cfg.hmm.pass_mode).lower()
    artifact_pass_mode = artifact.get("pass_mode")

    if (
        artifact_pass_mode is not None
        and str(artifact_pass_mode).lower() != runtime_pass_mode
    ):
        reason = (
            f"Threshold artifact pass_mode={artifact_pass_mode!r} does not "
            f"match runtime HMM_PASS_MODE={runtime_pass_mode!r}"
        )
        logger.warning(reason)
        return (False, reason)

    logger.info(
        "Threshold artifact in %s is compatible with HMM version %s.",
        artifact_dir,
        hmm_version,
    )
    return (True, "ok")
