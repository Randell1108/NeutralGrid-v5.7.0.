"""
Optimization package for NEUTRAL Grid Bot.

Phase 3: Bayesian threshold optimization via GP-UCB.
"""
from __future__ import annotations

__all__ = [
    "BayesianThresholdOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "ThresholdSearchSpace",
]

try:
    from neutralgrid.optimization.bayesian_threshold_optimizer_v20260311 import (  # noqa: F401
        BayesianThresholdOptimizer,
        OptimizationConfig,
        OptimizationResult,
        ThresholdSearchSpace,
    )
except ImportError:
    pass
