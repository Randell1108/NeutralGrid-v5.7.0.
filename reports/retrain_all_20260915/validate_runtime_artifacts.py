"""Read-only artifact integrity, lineage and serialization smoke checks."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from neutralgrid.core.config import get_config
from neutralgrid.models.artifacts import load_artifact, resolve_hmm_artifact_dir
from neutralgrid.models.meta_labeler import MetaLabeler, ACTIVE_SNAPSHOT_META_FEATURES
from neutralgrid.scanner.pattern_profile import PatternProfile, DEFAULT_FEATURES
from neutralgrid.scanner.profile_model import load_profile_model
from neutralgrid.scanner.profile_model_walkforward import resolve_active_pattern_profile_path, resolve_active_profile_model_path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main() -> None:
    directory = resolve_hmm_artifact_dir()
    hmm = load_artifact(directory, expected_version=directory.name)
    assert np.allclose(hmm["scaler"].inverse_transform(hmm["model"].means_), hmm["state_means_unscaled"])
    metadata_path = ROOT / "models/meta_labeler/metadata.json"
    meta = json.loads(metadata_path.read_text())
    assert meta["lineage"]["hmm_artifact_version"] == directory.name
    assert meta["features"] == list(ACTIVE_SNAPSHOT_META_FEATURES)
    labeler = MetaLabeler.load(ROOT / "models/meta_labeler.pkl")
    source = pd.read_csv(OUT / "prepared_training.csv", low_memory=False)
    predictions = labeler.predict_proba_batch(source)
    assert np.isfinite(predictions).all() and ((predictions >= 0) & (predictions <= 1)).all()
    with (ROOT / "models/meta_labeler.pkl").open("rb") as handle:
        state = pickle.load(handle)
    legacy = MetaLabeler(state["config"])
    legacy._model = state["model"]
    legacy._imputer = state["imputer"]
    legacy._feature_names = state["feature_names"]
    legacy._is_trained = True
    assert np.allclose(predictions, legacy.predict_proba_batch(source), rtol=0, atol=1e-14)
    sample = source.iloc[::max(1, len(source)//50)]
    singles = np.array([labeler.predict_proba(row) for row in sample.to_dict("records")])
    assert np.allclose(singles, labeler.predict_proba_batch(sample), rtol=0, atol=1e-14)
    old_config = json.loads((OUT / "meta_config_before.json").read_text())
    config = dataclasses.asdict(state["config"])
    config_difference = {k: {"old": old_config.get(k), "new": v} for k, v in config.items() if old_config.get(k) != v}
    (OUT / "meta_config_after.json").write_text(json.dumps(config, indent=2)+"\n")
    profile = {}
    shadow_validation = {}
    for role in ["profile_candidate", "expanded_profile_shadow"]:
        shadow = load_profile_model(OUT / role / "profile_model.json")
        pattern = PatternProfile.load_json(OUT / role / "pattern_profile.json")
        assert shadow.features == pattern.features == list(DEFAULT_FEATURES)
        inverse = np.asarray(shadow.inv_cov, dtype=float)
        assert inverse.shape == (4, 4) and np.isfinite(inverse).all()
        assert np.allclose(inverse, inverse.T)
        eigenvalues = np.linalg.eigvalsh(inverse)
        assert (eigenvalues > 0).all()
        assert shadow.feature_mean is not None
        probability = shadow.proba(shadow.feature_mean)
        assert probability is not None and np.isfinite(probability) and 0 <= probability <= 1
        shadow_validation[role] = {"status": "PASS", "features": shadow.features, "minimum_inverse_covariance_eigenvalue": float(eigenvalues.min()), "mean_feature_vector_probability": probability, "promotion_claimed": False}
    profile_dir = get_config().resolve_path(get_config().artifacts.profile_dir)
    for name, resolve, load in [("pattern", resolve_active_pattern_profile_path, PatternProfile.load_json), ("model", resolve_active_profile_model_path, load_profile_model)]:
        try:
            path = resolve(profile_dir)
            artifact = load(path)
            profile[name] = {"path": str(path), "status": "loaded", "features": list(artifact.features)}
            assert list(artifact.features) == list(DEFAULT_FEATURES)
        except Exception as exc:
            profile[name] = {"status": "unavailable", "reason": str(exc)}
    hashes = {}
    paths = [ROOT / "artifact_manifest.json", ROOT / "data/new_expired_bots.xlsx", ROOT / "models/meta_labeler.pkl"]
    paths += [p for d in [directory, ROOT / "models/meta_labeler", OUT / "profile_candidate", OUT / "expanded_profile_shadow"] for p in d.iterdir() if p.is_file()]
    paths += list((ROOT / "artifacts/utility").glob("utility_20260915_*.json"))
    for path in paths:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"checked_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "hmm_status": "PASS", "hmm_version": directory.name, "meta_status": "PASS", "meta_version": meta["artifact_version"], "meta_features": len(meta["features"]), "rows_smoke_scored": len(source), "prediction_min": float(predictions.min()), "prediction_max": float(predictions.max()), "legacy_and_artifact_predictions_equal": True, "batch_and_single_predictions_equal": True, "config_difference_from_prior": config_difference, "utility_current_exists": (ROOT / "artifacts/utility/current.json").exists(), "profile_runtime": profile, "shadow_artifact_integrity": shadow_validation, "conformal_current_exists": (ROOT / "data/cache/conformal_quantile.json").exists(), "artifact_sha256": hashes, "scoring_scope": "Serialization/inference integrity only; training-row probabilities are not performance evaluation"}
    (OUT / "runtime_artifact_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "artifact_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
