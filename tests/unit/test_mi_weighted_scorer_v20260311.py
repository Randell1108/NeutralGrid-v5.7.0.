from __future__ import annotations

import pytest

from neutralgrid.scanner.mi_weighted_scorer_v20260311 import (
    compute_mi_weighted_score,
    validate_mi_weights,
)


def test_missing_feature_is_renormalized_over_available_weights():
    score = compute_mi_weighted_score(
        features={"similarity_score": 80.0, "meta_prob": None},
        weights={"similarity_score": 0.5, "meta_prob": 0.5},
    )

    assert score == pytest.approx(80.0)


def test_all_present_features_keep_plain_weighted_sum():
    score = compute_mi_weighted_score(
        features={"similarity_score": 80.0, "meta_prob": 60.0},
        weights={"similarity_score": 0.5, "meta_prob": 0.5},
    )

    assert score == pytest.approx(70.0)


def test_all_missing_features_return_zero():
    score = compute_mi_weighted_score(
        features={"similarity_score": None, "meta_prob": None},
        weights={"similarity_score": 0.5, "meta_prob": 0.5},
    )

    assert score == pytest.approx(0.0)


def test_non_finite_weights_are_rejected_by_validation():
    weights = validate_mi_weights(
        {"similarity_score": float("nan"), "meta_prob": 0.5}
    )

    assert weights is None
