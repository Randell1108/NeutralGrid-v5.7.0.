from __future__ import annotations

import logging

import numpy as np

from neutralgrid.models.meta_labeler import MetaLabeler, MetaLabelerConfig


class _IdentityImputer:
    def __init__(self, n_features: int) -> None:
        self.n_features_in_ = n_features

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)


class _ConstantModel:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        rows = len(X)
        return np.tile(np.array([[0.3, 0.7]], dtype=float), (rows, 1))


def _feature_row() -> dict[str, float]:
    """Supply all active (FASTWIN-01) snapshot meta-labeler features.

    adx_15m is supplied under its runtime alias (``scan_adx_15m``) to exercise
    inference-time alias resolution; every other active feature is canonical.
    """
    return {
        "ev_score": 0.5,
        "adx_1h": 22.0,
        "scan_adx_15m": 20.0,  # alias -> adx_15m
        "adx_5m": 24.0,
        "atr_pct_15m": 0.01,
        "ou_halflife": 18.0,
        "rsi_15m": 50.0,
        "ema_slope_1h": 0.0,
        "ema_crosses_5m": 1.0,
        "vwap_crosses_5m": 1.0,
        "hurst_exponent": 0.5,
        "range_size_pct": 5.0,
        "bb_width": 0.03,
        "grid_spacing_pct": 0.45,
        "profit_per_grid_pct": 0.42,
        "num_grids": 28.0,
        "quote_volume_24h": 5.0e7,
        "open_interest": 5.0e7,
        "micro_round_trip_cost_pct": 0.15,
        "funding_rate": 0.0001,
    }


def _trained_labeler() -> MetaLabeler:
    labeler = MetaLabeler()
    labeler._feature_names = list(MetaLabelerConfig().features)
    labeler._imputer = _IdentityImputer(len(labeler._feature_names))
    labeler._model = _ConstantModel()
    labeler._is_trained = True
    return labeler


def test_predict_proba_resolves_scan_alias_without_imputation_warning(caplog) -> None:
    labeler = _trained_labeler()

    caplog.set_level(logging.WARNING)
    proba = labeler.predict_proba(_feature_row())

    assert proba == 0.7
    # adx_15m is supplied via its scan_ alias; with the full active feature set
    # resolving, no train/serve-skew imputation warning should fire.
    assert not any(
        "features being imputed" in record.message for record in caplog.records
    )


def test_get_missing_feature_names_reports_true_scan_time_gaps() -> None:
    labeler = _trained_labeler()
    feature_row = _feature_row()
    feature_row.pop("grid_spacing_pct")
    feature_row.pop("adx_1h")

    missing = set(labeler.get_missing_feature_names(feature_row))

    assert "grid_spacing_pct" in missing
    assert "adx_1h" in missing
    # Features still present should not be reported missing, including the
    # alias-resolved adx_15m (supplied as scan_adx_15m).
    assert "ou_halflife" not in missing
    assert "adx_15m" not in missing


def test_get_missing_feature_names_treats_arraylike_as_missing_err054() -> None:
    """ERR-054: a raw Binance funding-rate LIST and open-interest DICT (what
    ``get_all_market_data`` returns) must be treated as MISSING (fail-closed),
    not raise "truth value of an array ... is ambiguous" and abort the probe.

    Pre-fix, ``if value is None or pd.isna(value)`` evaluated ``x or <ndarray>``
    on the list/dict and raised ValueError, killing the entire overlay.
    """
    labeler = _trained_labeler()
    feature_row = _feature_row()
    # Raw payloads, unconverted (the ERR-054 failure mode):
    feature_row["funding_rate"] = [
        {"fundingRate": "0.0001"},
        {"fundingRate": "0.0002"},
    ]  # 90-element history list in production
    feature_row["open_interest"] = {"symbol": "X", "openInterest": "1.0", "time": 0}

    # Must NOT raise:
    missing = set(labeler.get_missing_feature_names(feature_row))

    assert "funding_rate" in missing
    assert "open_interest" in missing
    # Scalar features remain unaffected:
    assert "ev_score" not in missing
    assert "adx_1h" not in missing


def test_get_missing_feature_names_ev_score_gates_prediction_err058() -> None:
    """ERR-058: `ev_score` (the meta-labeler's first feature, a post-scoring
    artifact) gates prediction. When it is absent — the enrich-time condition
    before the post-scoring backfill — it is flagged missing, so a strict caller
    skips the probe (the fleet-wide `meta_prob_source="missing"` symptom). When
    present (after `run_full_pipeline.py:303-323` attaches it), it is not flagged.
    """
    labeler = _trained_labeler()
    row_without_ev = _feature_row()
    row_without_ev.pop("ev_score")
    assert "ev_score" in labeler.get_missing_feature_names(row_without_ev)
    # Once ev_score is attached, the gate clears:
    assert "ev_score" not in labeler.get_missing_feature_names(_feature_row())
