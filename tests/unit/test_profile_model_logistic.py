from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES
from neutralgrid.scanner.profile_model import (
    ProfileModel,
    load_profile_model,
    save_profile_model,
)


def _logistic_model() -> ProfileModel:
    features = list(DEFAULT_FEATURES[:2])
    return ProfileModel(
        features=features,
        winner_mu={feature: 0.25 for feature in features},
        loser_mu={feature: -0.25 for feature in features},
        inv_cov=np.eye(len(features)).tolist(),
        prior_winner=0.4,
        feature_mean={features[0]: 10.0, features[1]: 20.0},
        feature_std={features[0]: 2.0, features[1]: 4.0},
        feature_impute={features[0]: 10.0, features[1]: 20.0},
        model_family="robust_logistic_v1",
        linear_coef={features[0]: 0.5, features[1]: -1.0},
        linear_intercept=0.25,
    )


def test_logistic_profile_scores_standardized_features_and_round_trips(
    tmp_path: Path,
) -> None:
    model = _logistic_model()
    features = model.features
    row = {features[0]: 12.0, features[1]: 16.0}

    # standardized x=(1,-1), so logit = 0.5*1 + -1*-1 + 0.25
    assert model.llr(row) == pytest.approx(1.75)
    assert model.proba(row) == pytest.approx(1.0 / (1.0 + np.exp(-1.75)))

    path = tmp_path / "profile.json"
    save_profile_model(model, path)
    loaded = load_profile_model(path)
    assert loaded.model_family == "robust_logistic_v1"
    assert loaded.linear_coef == model.linear_coef
    assert loaded.linear_intercept == model.linear_intercept
    assert loaded.llr(row) == pytest.approx(1.75)


def test_legacy_profile_without_family_retains_gaussian_score(tmp_path: Path) -> None:
    feature = DEFAULT_FEATURES[0]
    payload = {
        "features": [feature],
        "winner_mu": {feature: 1.0},
        "loser_mu": {feature: -1.0},
        "inv_cov": [[2.0]],
        "prior_winner": 0.25,
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_profile_model(path)

    assert loaded.model_family == "gaussian_lda_v1"
    assert loaded.llr({feature: 0.5}) == pytest.approx(
        2.0 - np.log(3.0)
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"model_family": "unknown"}, "unsupported model_family"),
        ({"linear_coef": None}, "missing linear_coef"),
        (
            {"linear_coef": {DEFAULT_FEATURES[0]: 0.5}},
            "coefficient schema differs",
        ),
    ],
)
def test_malformed_logistic_artifact_fails_closed(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    payload = _logistic_model().to_json()
    payload.update(mutation)
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_profile_model(path)
