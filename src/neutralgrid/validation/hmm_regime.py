"""validation.hmm_regime

Hidden Markov Model (HMM) regime detection.

Design goals (aligned with the project requirements):
 - Cross-sectional regime: one model fit across many symbols (stacking).
 - Calibrated with historical data (training uses multiple sequences; no look-ahead).
 - Non-stationary: the model is trained on a rolling window (recent history), so
   regimes can drift as market microstructure and volatility regimes evolve.
 - Multicollinearity robustness: diagonal covariance + robust scaling.
 - Interpretability: states are mapped to {RANGE, TREND} via their learned
   (inverse-scaled) feature means, and we expose per-state statistics.

This module intentionally keeps a narrow surface area:
 - ensure_hmm_model(...) validates that a trained model exists (fails loudly if not).
 - load_hmm_model(...) loads the persisted model.
 - infer_regime(...) runs inference on a single symbol.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, List, Optional, Mapping

from neutralgrid.core.protocols import RegimePredictor

import pandas as pd

from neutralgrid.models.artifacts import resolve_hmm_artifact_dir

logger = logging.getLogger(__name__)


_PREDICTOR = None
_PREDICTOR_META_MTIME = None
_PREDICTOR_ARTIFACT_DIR = None


def _default_artifact_dir() -> Path:
    """Return the AFML-aligned artifact directory used by the retrain CLI.

    Env var override and manifest lookup are now handled inside
    ``resolve_hmm_artifact_dir()``, so this is a thin pass-through.
    """
    return resolve_hmm_artifact_dir()



def _artifact_meta_mtime(artifact_dir: Path) -> Optional[float]:
    meta = Path(artifact_dir) / "metadata.json"
    if meta.exists():
        try:
            return float(meta.stat().st_mtime)
        except OSError as e:
            logger.debug("Cannot stat %s: %s", meta, e)
            return None
    return None


_STALE_CHECK_DONE = False


def _check_model_staleness(predictor, threshold_days: float = 7.0) -> None:
    """Emit a WARNING alert if the loaded model is older than *threshold_days*."""
    global _STALE_CHECK_DONE
    if _STALE_CHECK_DONE:
        return
    _STALE_CHECK_DONE = True

    trained_at_str = getattr(getattr(predictor, "metadata", None), "trained_at_utc", None)
    if not trained_at_str:
        return
    try:
        trained_at = datetime.fromisoformat(trained_at_str)
        if trained_at.tzinfo is None:
            trained_at = trained_at.replace(tzinfo=timezone.utc)
        staleness_days = (datetime.now(timezone.utc) - trained_at).total_seconds() / 86400
        if staleness_days > threshold_days:
            from neutralgrid.models.alerts import alert_stale_model
            alert_stale_model(
                trained_at_utc=trained_at_str,
                staleness_days=staleness_days,
                threshold_days=threshold_days,
            )
    except Exception as e:
        logger.debug("Staleness check failed (non-fatal): %s", e)


def load_hmm_model() -> Optional[RegimePredictor]:
    """Load the HMM inference object from the manifest-resolved artifact directory.

    Resolution is handled by ``resolve_hmm_artifact_dir()`` (env var → manifest).

    Returns:
        HMMRegimePredictor instance or None if artifact not found or not configured.
    """
    global _PREDICTOR, _PREDICTOR_META_MTIME, _PREDICTOR_ARTIFACT_DIR

    try:
        artifact_dir = _default_artifact_dir()
    except Exception:
        # No artifact directory configured — return None so callers can
        # distinguish "not configured" from "configured but broken".
        return None

    meta_mtime = _artifact_meta_mtime(artifact_dir)

    if meta_mtime is not None:
        # Reload if first time or artifact was updated on disk.
        resolved_dir = str(Path(artifact_dir).resolve())
        if (
            _PREDICTOR is None
            or _PREDICTOR_META_MTIME != meta_mtime
            or _PREDICTOR_ARTIFACT_DIR != resolved_dir
        ):
            try:
                from neutralgrid.models.hmm.inference import HMMRegimePredictor

                _PREDICTOR = HMMRegimePredictor(artifact_dir=artifact_dir)
                _PREDICTOR_META_MTIME = meta_mtime
                _PREDICTOR_ARTIFACT_DIR = resolved_dir
            except (FileNotFoundError, OSError, ValueError) as e:
                logger.debug("Failed to load HMMRegimePredictor from %s: %s", artifact_dir, e)
                _PREDICTOR = None
                _PREDICTOR_META_MTIME = None
                _PREDICTOR_ARTIFACT_DIR = None
            except Exception as e:
                logger.warning("Unexpected error loading HMMRegimePredictor from %s: %s", artifact_dir, e)
                _PREDICTOR = None
                _PREDICTOR_META_MTIME = None
                _PREDICTOR_ARTIFACT_DIR = None

        if _PREDICTOR is not None:
            _check_model_staleness(_PREDICTOR)
            return _PREDICTOR

    return None


async def ensure_hmm_model(
    *,
    client: Any = None,
    symbols: List[str] | None = None,
    force: bool = False,
) -> Path:
    """Ensure a trained HMM artifact exists at the manifest-resolved directory.

    Validates that ``model.joblib`` is present in the resolved artifact dir.
    Raises ``NeutralGridError`` if the model is missing, directing the operator
    to run the retrain CLI.

    The ``client`` and ``symbols`` parameters are accepted for backward
    compatibility but are no longer used (bootstrap training has been removed).

    Returns:
        Path to the artifact directory.

    Raises:
        NeutralGridError: If no trained HMM model is found.
    """
    from neutralgrid.core.exceptions import NeutralGridError

    artifact_dir = _default_artifact_dir()
    model_path = artifact_dir / "model.joblib"

    if model_path.exists() and not force:
        return artifact_dir

    raise NeutralGridError(
        f"No HMM model found at {artifact_dir}/model.joblib. "
        "Run: neutralgrid-retrain --window 180d --top-n 60"
    )


def infer_regime(
    artifact: RegimePredictor,
    df: pd.DataFrame,
    *,
    infer_limit: int,
) -> dict[str, Any]:
    """Infer the current regime probabilities for a single symbol.

    Args:
        artifact: HMMRegimePredictor instance (from load_hmm_model).
        df: OHLCV DataFrame for the symbol.
        infer_limit: Max bars to use for inference.

    Returns:
        Dict with regime probabilities and model metadata.
    """
    if not isinstance(artifact, RegimePredictor):
        raise TypeError("Unsupported HMM artifact type — expected RegimePredictor")

    res = artifact.predict(df, infer_limit=infer_limit)
    payload: Any
    if hasattr(res, "to_dict"):
        payload = getattr(res, "to_dict")()
    else:
        payload = res

    if not isinstance(payload, Mapping):
        raise TypeError("Unsupported HMM predictor output type - expected mapping payload")
    d = dict(payload)
    return {
        "trend_state": int(d.get("trend_state", 0)),
        "range_state": int(d.get("range_state", 0)),
        "trend_prob": float(d.get("trend_prob", 0.0)),
        "range_prob": float(d.get("range_prob", 0.0)),
        "range_prob_agg": float(d.get("range_prob_agg", d.get("range_prob", 0.0))),
        "trend_prob_agg": float(d.get("trend_prob_agg", d.get("trend_prob", 0.0))),
        "posteriors": d.get("posteriors"),
        "state_means": d.get("state_means"),
        "trained_at_utc": d.get("trained_at_utc"),
        "artifact_version": d.get("artifact_version"),
        "pipeline_version": d.get("pipeline_version"),
        "calibration_provenance": d.get("calibration_provenance", {}),
        "posterior_mode": d.get("posterior_mode"),
        "volatility_tier": d.get("dominant_volatility_tier"),
        "volatility_tier_probs": d.get("volatility_tier_probs"),
        "conditional_tail_risk": d.get("conditional_tail_risk"),
        "tail_correction_applied": d.get("tail_correction_applied", False),
        "conditional_tail_risk_enhanced": d.get("conditional_tail_risk_enhanced", {}),
        "volatility_tier_kurtosis_aware": d.get("dominant_volatility_tier_kurtosis_aware"),
        "volatility_tier_probs_kurtosis_aware": d.get("volatility_tier_probs_kurtosis_aware", {}),
        # afml_enriched_v20260318: HMM dominant-state self-transition probability
        "persistence_prob": d.get("persistence_prob"),
    }
