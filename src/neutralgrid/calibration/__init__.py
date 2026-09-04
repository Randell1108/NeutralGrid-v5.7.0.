"""
Calibration package for NEUTRAL Grid Bot.

Phase 1: Reliability diagnostics, effective sample size, meta-labeler config overrides.
Phase 2: Temperature scaling (HMM), beta calibration (meta-labeler).
Phase 3: Conformal risk control.
"""
from __future__ import annotations

__all__ = [
    # Phase 1
    "ReliabilityDiagnostic",
    "ReliabilityReport",
    "compute_kish_effective_n",
    "should_disable_time_decay",
    "compute_effective_sample_report",
    "EffectiveSampleReport",
    "Phase1MetaLabelerOverrides",
    "build_phase1_meta_config",
    "validate_phase1_auc_floor",
    # Phase 2
    "TemperatureScaler",
    "TemperatureScalingResult",
    "BetaCalibrator",
    "BetaCalibrationResult",
    # Phase 3
    "ConformalRiskController",
    "ConformalConfig",
    "ConformalQuantile",
    # Utility score
    "run_calibration",
    "CalibrationResult",
]

# Phase 1
try:
    from neutralgrid.calibration.reliability_diagnostic_v20260311 import (  # noqa: F401
        ReliabilityDiagnostic,
        ReliabilityReport,
    )
except ImportError:
    pass

try:
    from neutralgrid.calibration.effective_sample_size_v20260311 import (  # noqa: F401
        compute_kish_effective_n,
        should_disable_time_decay,
        compute_effective_sample_report,
        EffectiveSampleReport,
    )
except ImportError:
    pass

try:
    from neutralgrid.calibration.phase1_config_v20260311 import (  # noqa: F401
        Phase1MetaLabelerOverrides,
        build_phase1_meta_config,
        validate_phase1_auc_floor,
    )
except ImportError:
    pass

# Phase 2
try:
    from neutralgrid.calibration.temperature_scaling_v20260311 import (  # noqa: F401
        TemperatureScaler,
        TemperatureScalingResult,
    )
except ImportError:
    pass

try:
    from neutralgrid.calibration.beta_calibrator_v20260311 import (  # noqa: F401
        BetaCalibrator,
        BetaCalibrationResult,
    )
except ImportError:
    pass

# Phase 3
try:
    from neutralgrid.calibration.conformal_risk_control_v20260311 import (  # noqa: F401
        ConformalRiskController,
        ConformalConfig,
        ConformalQuantile,
    )
except ImportError:
    pass

try:
    from neutralgrid.calibration.utility_calibrator import (  # noqa: F401
        CalibrationResult,
        run_calibration,
    )
except ImportError:
    pass
