"""
Unit tests for artifact management and schema validation.

Tests that schema mismatches raise clear exceptions.
"""

import pytest
import tempfile
from pathlib import Path
from neutralgrid.models.hmm.schema import validate_artifact_features, FEATURE_NAMES


class TestArtifactSchemaValidation:
    """Test artifact schema validation."""

    def test_matching_schema_passes(self):
        """Test that matching schema passes validation."""
        # Should not raise
        validate_artifact_features(list(FEATURE_NAMES))

    def test_different_order_raises(self):
        """Test that different feature order raises clear exception."""
        # Swap two features
        wrong_order = list(FEATURE_NAMES)
        wrong_order[1], wrong_order[2] = wrong_order[2], wrong_order[1]

        with pytest.raises(ValueError, match="FATAL.*feature schema mismatch"):
            validate_artifact_features(wrong_order)

    def test_missing_feature_raises(self):
        """Test that missing feature raises exception."""
        incomplete = list(FEATURE_NAMES)[:-1]  # Drop last feature

        with pytest.raises(ValueError, match="FATAL.*feature schema mismatch"):
            validate_artifact_features(incomplete)

    def test_extra_feature_raises(self):
        """Test that extra feature raises exception."""
        extra = list(FEATURE_NAMES) + ["extra_feature"]

        with pytest.raises(ValueError, match="FATAL.*feature schema mismatch"):
            validate_artifact_features(extra)

    def test_wrong_feature_names_raises(self):
        """Test that wrong feature names raise exception."""
        wrong_names = ["wrong_1", "wrong_2", "wrong_3", "wrong_4", "wrong_5"]

        with pytest.raises(ValueError, match="FATAL.*feature schema mismatch"):
            validate_artifact_features(wrong_names)

    def test_error_message_shows_both_schemas(self):
        """Test that error message shows both schemas for debugging."""
        wrong = ["r_t", "vol_t", "adx_t", "trend_t", "bbwidth_t"]  # Wrong order

        try:
            validate_artifact_features(wrong)
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            # Should show what was expected
            assert "Current schema" in error_msg or "Expected" in error_msg
            # Should show what was found
            assert "Artifact features" in error_msg or "Got" in error_msg
            # Should mention it's a FATAL error
            assert "FATAL" in error_msg

    def test_error_message_suggests_solutions(self):
        """Test that error message suggests how to fix the problem."""
        wrong = list(FEATURE_NAMES)[::-1]  # Reversed

        try:
            validate_artifact_features(wrong)
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            # Should suggest retrain or update code
            assert "retrain" in error_msg.lower() or "update" in error_msg.lower()


class TestArtifactIO:
    """Test artifact save/load functionality."""

    def test_save_and_load_artifact(self):
        """Test basic save and load of artifact."""
        from neutralgrid.models.artifacts import (
            save_artifact,
            load_artifact,
            ArtifactMetadata,
        )
        from datetime import datetime, timezone
        import numpy as np

        # Create a dummy model (just a dict for testing)
        model = {"weights": [1, 2, 3]}
        scaler = {"mean": 0, "std": 1}
        state_means = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]])

        metadata = ArtifactMetadata(
            artifact_version="test_1.0.0",
            trained_at_utc=datetime.now(timezone.utc).isoformat(),
            features=list(FEATURE_NAMES),
            model_type="TestModel",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "test_artifact"

            # Save
            save_artifact(
                artifact_dir=artifact_dir,
                model=model,
                scaler=scaler,
                metadata=metadata,
                state_means_unscaled=state_means,
            )

            # Load
            loaded = load_artifact(artifact_dir)

            # Verify
            assert loaded["model"] == model
            assert loaded["scaler"] == scaler
            assert loaded["metadata"].artifact_version == "test_1.0.0"
            assert loaded["metadata"].features == list(FEATURE_NAMES)
            np.testing.assert_array_equal(loaded["state_means_unscaled"], state_means)

    def test_load_with_schema_mismatch_raises(self):
        """Test that loading artifact with schema mismatch raises."""
        from neutralgrid.models.artifacts import (
            save_artifact,
            load_artifact,
            ArtifactMetadata,
        )
        from neutralgrid.core.exceptions import FeatureSchemaError
        from datetime import datetime, timezone

        # Create artifact with wrong schema
        wrong_features = ["wrong_1", "wrong_2", "wrong_3", "wrong_4", "wrong_5"]
        metadata = ArtifactMetadata(
            artifact_version="wrong_schema",
            trained_at_utc=datetime.now(timezone.utc).isoformat(),
            features=wrong_features,
            model_type="TestModel",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "wrong_artifact"

            # Save with wrong schema
            save_artifact(
                artifact_dir=artifact_dir,
                model={},
                scaler={},
                metadata=metadata,
            )

            # Load should fail validation with typed exception
            with pytest.raises(FeatureSchemaError):
                load_artifact(artifact_dir, validate_schema=True)

    def test_load_without_validation_succeeds(self):
        """Test that loading without validation succeeds even with mismatch."""
        from neutralgrid.models.artifacts import (
            save_artifact,
            load_artifact,
            ArtifactMetadata,
        )
        from datetime import datetime, timezone

        wrong_features = ["wrong_1", "wrong_2"]
        metadata = ArtifactMetadata(
            artifact_version="no_validation",
            trained_at_utc=datetime.now(timezone.utc).isoformat(),
            features=wrong_features,
            model_type="TestModel",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "no_val_artifact"

            save_artifact(
                artifact_dir=artifact_dir,
                model={},
                scaler={},
                metadata=metadata,
            )

            # Load without validation should succeed
            loaded = load_artifact(artifact_dir, validate_schema=False)
            assert loaded["metadata"].features == wrong_features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
