"""
Training utilities for meta-labeler.

This module provides tools for building training datasets
with proper AFML-compliant labeling and feature extraction.
"""

from neutralgrid.training.data_generator import (
    # Schema
    TrainingTableSchema,
    DEFAULT_SCHEMA,
    FeatureSnapshot,
    OutcomeLabel,
    # Label generation
    LabelConfig,
    BarrierLabelGenerator,
    # Data mapping
    ExistingDataMapper,
    # Logging
    FeatureSnapshotLogger,
    # Builder
    TrainingDataBuilder,
    # Convenience
    load_and_prepare_training_data,
    get_missing_features_report,
)

from neutralgrid.training.scanner_integration import (
    build_feature_snapshot,
    FeatureCollector,
    create_collector,
)

__all__ = [
    # Data generator
    "TrainingTableSchema",
    "DEFAULT_SCHEMA",
    "FeatureSnapshot",
    "OutcomeLabel",
    "LabelConfig",
    "BarrierLabelGenerator",
    "ExistingDataMapper",
    "FeatureSnapshotLogger",
    "TrainingDataBuilder",
    "load_and_prepare_training_data",
    "get_missing_features_report",
    # Scanner integration
    "build_feature_snapshot",
    "FeatureCollector",
    "create_collector",
]
