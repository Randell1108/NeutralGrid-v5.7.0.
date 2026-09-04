"""Unit tests for PnL curve feature extraction (pnl_curve_features_v20260310)."""
from __future__ import annotations

import json
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from neutralgrid.metrics.pnl_curve_features_v20260310 import (
    PnlCurveFeatures,
    extract_pnl_curve_features,
    _compute_return_entropy,
    _classify_shape,
    _MIN_CURVE_LEN,
)


# ═══════════════════════════════════════════════════════════════════════
# BASIC EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

class TestExtractBasic:
    """Basic extraction: returns PnlCurveFeatures or None."""

    def test_too_short_returns_none(self):
        assert extract_pnl_curve_features([1.0, 2.0]) is None
        assert extract_pnl_curve_features([]) is None
        assert extract_pnl_curve_features([5.0]) is None

    def test_min_length_returns_features(self):
        result = extract_pnl_curve_features([0.0, 1.0, 2.0])
        assert result is not None
        assert isinstance(result, PnlCurveFeatures)

    def test_returns_all_fields(self):
        result = extract_pnl_curve_features([0.0, 1.0, 2.0, 3.0])
        assert result is not None
        d = result.to_dict()
        expected_keys = {
            "pnl_curve_fill_concentration",
            "pnl_curve_active_ratio",
            "pnl_curve_time_to_peak",
            "pnl_curve_flatline_pct",
            "pnl_curve_entropy",
            "pnl_curve_shape_class",
            "pnl_curve_points",
            "pnl_curve_raw_json",
        }
        assert set(d.keys()) == expected_keys

    def test_points_matches_input_length(self):
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        result = extract_pnl_curve_features(values)
        assert result.points == 5


# ═══════════════════════════════════════════════════════════════════════
# FILL CONCENTRATION
# ═══════════════════════════════════════════════════════════════════════

class TestFillConcentration:
    """fill_concentration = max(|change|) / sum(|changes|)."""

    def test_single_step(self):
        """One big jump then flat → concentration = 1.0."""
        # [0, 0, 0, 5, 5, 5, 5]
        values = [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0]
        result = extract_pnl_curve_features(values)
        assert result.fill_concentration == pytest.approx(1.0)

    def test_uniform_steps(self):
        """Equal increments → concentration = 1/N changes."""
        # [0, 1, 2, 3, 4] → 4 changes of 1.0 each
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        result = extract_pnl_curve_features(values)
        assert result.fill_concentration == pytest.approx(0.25, abs=1e-6)

    def test_all_zeros(self):
        """Flat curve → concentration = 0.0."""
        values = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = extract_pnl_curve_features(values)
        assert result.fill_concentration == 0.0


# ═══════════════════════════════════════════════════════════════════════
# ACTIVE RATIO
# ═══════════════════════════════════════════════════════════════════════

class TestActiveRatio:
    """active_ratio = non-zero-change bars / total bars."""

    def test_all_active(self):
        """Every bar has a change."""
        values = [0.0, 1.0, 3.0, 2.0, 5.0]
        result = extract_pnl_curve_features(values)
        assert result.active_ratio == pytest.approx(1.0)

    def test_half_active(self):
        """Half the bars are flat."""
        # changes: [1, 0, 1, 0] → 2/4 = 0.5
        values = [0.0, 1.0, 1.0, 2.0, 2.0]
        result = extract_pnl_curve_features(values)
        assert result.active_ratio == pytest.approx(0.5)

    def test_all_flat(self):
        """No activity at all."""
        values = [5.0, 5.0, 5.0, 5.0]
        result = extract_pnl_curve_features(values)
        assert result.active_ratio == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════
# TIME TO PEAK
# ═══════════════════════════════════════════════════════════════════════

class TestTimeToPeak:
    """time_to_peak = first index of max / (n-1)."""

    def test_peak_at_end(self):
        """Monotonically increasing → peak at last bar."""
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        result = extract_pnl_curve_features(values)
        assert result.time_to_peak == pytest.approx(1.0)

    def test_peak_at_start(self):
        """Peak at first bar, then declining."""
        values = [10.0, 5.0, 3.0, 1.0]
        result = extract_pnl_curve_features(values)
        assert result.time_to_peak == pytest.approx(0.0)

    def test_peak_in_middle(self):
        """Peak reached mid-curve."""
        # [0, 5, 5, 5, 2] → peak=5 first at index 1, n=5
        values = [0.0, 5.0, 5.0, 5.0, 2.0]
        result = extract_pnl_curve_features(values)
        assert result.time_to_peak == pytest.approx(1 / 4)  # 0.25

    def test_steem_curve(self):
        """STEEMUSDT actual curve: peak at bar 6 (index 6), n=17."""
        values = [
            0.0, 0.0, 0.0, 0.0, 0.0,
            1.869701, 4.990441, 4.990441, 4.990441, 4.990441,
            4.990441, 4.990441, 4.990441, 4.990441, 4.990441,
            4.990441, 4.990441,
        ]
        result = extract_pnl_curve_features(values)
        assert result.time_to_peak == pytest.approx(6 / 16, abs=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# FLATLINE PERCENTAGE
# ═══════════════════════════════════════════════════════════════════════

class TestFlatlinePct:
    """flatline_pct = consecutive identical values / total changes."""

    def test_no_flatline(self):
        """Every bar changes."""
        values = [0.0, 1.0, 3.0, 2.0, 5.0]
        result = extract_pnl_curve_features(values)
        assert result.flatline_pct == pytest.approx(0.0)

    def test_all_flatline(self):
        """Completely flat."""
        values = [3.0, 3.0, 3.0, 3.0, 3.0]
        result = extract_pnl_curve_features(values)
        assert result.flatline_pct == pytest.approx(1.0)

    def test_steem_flatline(self):
        """STEEMUSDT: bars 0-4 flat (4), bars 6-16 flat (10), bars 5 and 6 active (2).
        Total changes = 16. Flat = 14. Flatline_pct = 14/16 = 0.875."""
        values = [
            0.0, 0.0, 0.0, 0.0, 0.0,
            1.869701, 4.990441, 4.990441, 4.990441, 4.990441,
            4.990441, 4.990441, 4.990441, 4.990441, 4.990441,
            4.990441, 4.990441,
        ]
        result = extract_pnl_curve_features(values)
        assert result.flatline_pct == pytest.approx(14 / 16, abs=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# ENTROPY
# ═══════════════════════════════════════════════════════════════════════

class TestEntropy:
    """Shannon entropy of discretized returns."""

    def test_zero_changes_zero_entropy(self):
        """All zeros → entropy = 0."""
        assert _compute_return_entropy([0.0, 0.0, 0.0]) == 0.0

    def test_single_value_low_entropy(self):
        """Single non-zero change → entropy = 0 (all in one bin)."""
        assert _compute_return_entropy([5.0]) == 0.0

    def test_uniform_high_entropy(self):
        """Many diverse changes → higher entropy."""
        # Create changes spread across many bins
        changes = [float(i) for i in range(-5, 6)]  # 11 values
        ent = _compute_return_entropy(changes)
        assert ent > 2.0  # High entropy

    def test_concentrated_low_entropy(self):
        """Many identical changes → low entropy."""
        changes = [1.0] * 20
        ent = _compute_return_entropy(changes)
        assert ent == pytest.approx(0.0)

    def test_steem_entropy(self):
        """STEEMUSDT curve has very low entropy (2 active bars out of 16)."""
        values = [
            0.0, 0.0, 0.0, 0.0, 0.0,
            1.869701, 4.990441, 4.990441, 4.990441, 4.990441,
            4.990441, 4.990441, 4.990441, 4.990441, 4.990441,
            4.990441, 4.990441,
        ]
        changes = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        ent = _compute_return_entropy(changes)
        assert ent < 1.5  # Low entropy for a step function


# ═══════════════════════════════════════════════════════════════════════
# SHAPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════

class TestShapeClassification:
    """Shape class based on feature thresholds."""

    def test_flat_shape(self):
        assert _classify_shape(active_ratio=0.05, fill_concentration=0.5,
                               flatline_pct=0.95, entropy=0.0) == "flat"

    def test_step_shape(self):
        assert _classify_shape(active_ratio=0.3, fill_concentration=0.8,
                               flatline_pct=0.7, entropy=0.5) == "step"

    def test_volatile_shape(self):
        assert _classify_shape(active_ratio=0.8, fill_concentration=0.3,
                               flatline_pct=0.2, entropy=2.5) == "volatile"

    def test_gradual_shape(self):
        assert _classify_shape(active_ratio=0.6, fill_concentration=0.4,
                               flatline_pct=0.4, entropy=1.5) == "gradual"

    def test_steem_is_gradual(self):
        """STEEMUSDT: active_ratio=2/16=0.125 (>0.1, not flat),
        fill_conc=3.12074/4.990441≈0.625 (<0.7, not step),
        entropy low + active<0.5 (not volatile) → "gradual"."""
        values = [
            0.0, 0.0, 0.0, 0.0, 0.0,
            1.869701, 4.990441, 4.990441, 4.990441, 4.990441,
            4.990441, 4.990441, 4.990441, 4.990441, 4.990441,
            4.990441, 4.990441,
        ]
        result = extract_pnl_curve_features(values)
        assert result.shape_class == "gradual"


# ═══════════════════════════════════════════════════════════════════════
# RAW JSON PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════

class TestRawJsonPersistence:
    """Raw PnL curve is stored as JSON for post-hoc analysis."""

    def test_json_roundtrip(self):
        values = [0.0, 1.5, 3.0, 2.5, 4.0]
        result = extract_pnl_curve_features(values)
        reconstructed = json.loads(result.raw_json)
        assert len(reconstructed) == len(values)
        for orig, restored in zip(values, reconstructed):
            assert orig == pytest.approx(restored)

    def test_steem_json_roundtrip(self):
        values = [
            0.0, 0.0, 0.0, 0.0, 0.0,
            1.869701, 4.990441, 4.990441, 4.990441, 4.990441,
            4.990441, 4.990441, 4.990441, 4.990441, 4.990441,
            4.990441, 4.990441,
        ]
        result = extract_pnl_curve_features(values)
        reconstructed = json.loads(result.raw_json)
        assert len(reconstructed) == 17
        assert reconstructed[5] == pytest.approx(1.869701)
        assert reconstructed[6] == pytest.approx(4.990441)


# ═══════════════════════════════════════════════════════════════════════
# TO_DICT PREFIX
# ═══════════════════════════════════════════════════════════════════════

class TestToDict:
    """to_dict() uses configurable prefix."""

    def test_default_prefix(self):
        result = extract_pnl_curve_features([0.0, 1.0, 2.0])
        d = result.to_dict()
        assert all(k.startswith("pnl_curve_") for k in d.keys())

    def test_custom_prefix(self):
        result = extract_pnl_curve_features([0.0, 1.0, 2.0])
        d = result.to_dict(prefix="curve_")
        assert all(k.startswith("curve_") for k in d.keys())
        assert "curve_fill_concentration" in d


# ═══════════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_negative_pnl(self):
        """Declining curve should still extract features."""
        values = [0.0, -1.0, -3.0, -2.0, -5.0]
        result = extract_pnl_curve_features(values)
        assert result is not None
        assert result.active_ratio == pytest.approx(1.0)
        assert result.points == 5

    def test_single_large_drop(self):
        """One large drop: fill_concentration should be high."""
        values = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0]
        result = extract_pnl_curve_features(values)
        assert result.fill_concentration == pytest.approx(1.0)

    def test_very_small_values(self):
        """Very small changes should still be detected."""
        values = [0.0, 0.00001, 0.00002, 0.00003]
        result = extract_pnl_curve_features(values)
        assert result is not None
        assert result.active_ratio == pytest.approx(1.0)

    def test_constant_nonzero(self):
        """Constant non-zero curve → all flat."""
        values = [42.0, 42.0, 42.0, 42.0]
        result = extract_pnl_curve_features(values)
        assert result.active_ratio == 0.0
        assert result.flatline_pct == 1.0
        assert result.fill_concentration == 0.0


# ═══════════════════════════════════════════════════════════════════════
# NaN / Inf SANITIZATION
# ═══════════════════════════════════════════════════════════════════════

class TestNanInfSanitization:
    """NaN and Inf values are replaced with 0.0 before computation."""

    def test_nan_replaced(self):
        """NaN values should be replaced with 0.0."""
        values = [0.0, float("nan"), 2.0, 3.0]
        result = extract_pnl_curve_features(values)
        assert result is not None
        reconstructed = json.loads(result.raw_json)
        assert reconstructed[1] == 0.0  # NaN → 0.0

    def test_inf_replaced(self):
        """Inf values should be replaced with 0.0."""
        values = [0.0, float("inf"), 2.0, 3.0]
        result = extract_pnl_curve_features(values)
        assert result is not None
        reconstructed = json.loads(result.raw_json)
        assert reconstructed[1] == 0.0

    def test_neg_inf_replaced(self):
        """Negative Inf should also be replaced."""
        values = [0.0, float("-inf"), 2.0, 3.0]
        result = extract_pnl_curve_features(values)
        assert result is not None

    def test_all_nan_becomes_flat(self):
        """All NaN → all zeros → flat curve."""
        values = [float("nan")] * 5
        result = extract_pnl_curve_features(values)
        assert result is not None
        assert result.active_ratio == 0.0
        assert result.fill_concentration == 0.0

    def test_mixed_nan_inf(self):
        """Mixed NaN/Inf doesn't crash json.dumps."""
        values = [0.0, float("nan"), float("inf"), 1.0, float("-inf")]
        result = extract_pnl_curve_features(values)
        assert result is not None
        # Verify JSON roundtrip doesn't fail
        reconstructed = json.loads(result.raw_json)
        assert len(reconstructed) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
