"""
Custom exception hierarchy for the core trading path.

These exceptions are raised in the live path (model loading, inference,
validation pipeline) and are caught by the API exception handlers to return
structured error responses.  The ~60 resilience/boundary except blocks
elsewhere in the codebase are intentionally untouched.
"""


class NeutralGridError(Exception):
    """Base for all neutralgrid domain errors."""
    error_code: str = "NEUTRALGRID_ERROR"


class ModelError(NeutralGridError):
    """Model loading or inference failed."""
    error_code = "MODEL_ERROR"


class ModelLoadError(ModelError):
    """Model artifact missing or corrupt."""
    error_code = "MODEL_LOAD_FAILED"

    def __init__(self, artifact_path: str, message: str):
        self.artifact_path = artifact_path
        super().__init__(f"Failed to load model from {artifact_path}: {message}")


class InferenceError(ModelError):
    """Model inference (prediction) failed at runtime."""
    error_code = "INFERENCE_FAILED"

    def __init__(self, symbol: str, model_type: str, message: str):
        self.symbol = symbol
        self.model_type = model_type
        super().__init__(f"[{model_type}] {symbol}: {message}")


class ValidationPipelineError(NeutralGridError):
    """Core validation pipeline failed unexpectedly."""
    error_code = "VALIDATION_PIPELINE_ERROR"

    def __init__(self, symbol: str, stage: str, message: str):
        self.symbol = symbol
        self.stage = stage
        super().__init__(f"[{stage}] {symbol}: {message}")


class ArtifactError(ModelError):
    """Artifact storage or integrity failure."""
    error_code = "ARTIFACT_ERROR"


class ArtifactVersionError(ArtifactError):
    """Artifact version mismatch between expected and found."""
    error_code = "ARTIFACT_VERSION_ERROR"

    def __init__(self, expected: str, found: str, artifact_dir: str = ""):
        self.expected = expected
        self.found = found
        self.artifact_dir = artifact_dir
        super().__init__(
            f"Version mismatch: expected {expected}, found {found}"
            + (f" in {artifact_dir}" if artifact_dir else "")
        )


class FeatureSchemaError(ModelError):
    """Feature schema mismatch between training and inference."""
    error_code = "FEATURE_SCHEMA_ERROR"

    def __init__(self, expected: list | None = None, found: list | None = None, detail: str = ""):
        self.expected = expected
        self.found = found
        msg = detail or f"Feature schema mismatch: expected={expected}, found={found}"
        super().__init__(msg)


class PriceSeriesError(NeutralGridError):
    """Price series streaming or storage error."""
    error_code = "PRICE_SERIES_ERROR"


class ReplayError(NeutralGridError):
    """Offline order-book replay pipeline error."""
    error_code = "REPLAY_ERROR"

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"[replay/{stage}] {message}")
