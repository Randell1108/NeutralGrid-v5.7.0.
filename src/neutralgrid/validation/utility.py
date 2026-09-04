"""
Utility-Based Scoring for AFML-Compliant Validation.

Replaces binary pass/fail thresholds with continuous utility scores.

Formula: U = E[grid_return] - lambda * E[drawdown]

Design principles:
- Calibrated probabilities from HMM/stochastic models
- Risk-adjusted scoring (drawdown penalty)
- Transparent utility decomposition
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

from neutralgrid.core.constants import BOT_HORIZON_HOURS
from neutralgrid.core.exceptions import NeutralGridError

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UTILITY_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "utility" / "current.json"

# Fallback constants are retained for two legitimate uses ONLY:
#   1. Default field values on the UtilityConfig dataclass (so direct
#      `UtilityConfig(...)` instantiation works without arguments).
#   2. The calibrator's internal G6 baseline at
#      `neutralgrid.calibration.utility_calibrator._build_artifact()`, which
#      compares candidate metrics against a v0 reference.
# They are NOT a runtime fallback. `UtilityConfig.from_artifact()` raises
# `UtilityCalibratorUnavailable` when `current.json` is missing or malformed
# rather than silently substituting these values for a calibrated artifact.
FALLBACK_LAMBDA_RISK = 2.0
FALLBACK_KAPPA_TREND = 1.5
FALLBACK_MIN_UTILITY = 0.0
FALLBACK_BASELINE_FILLS_PER_HOUR = 2.0
FALLBACK_HORIZON_HOURS = BOT_HORIZON_HOURS
FALLBACK_STOP_LOSS_PCT = -12.0
FALLBACK_TAKE_PROFIT_PCT = 15.0


class UtilityCalibratorUnavailable(NeutralGridError):
    """Raised when the runtime utility calibrator artifact cannot be loaded.

    Per `safety-invariants.md` §Fail-Closed Behavior, callers in the
    decision-time path must propagate this exception. Offline callers
    (training, backfill, scan-time feature snapshot) may catch and emit
    a `utility_score=None`/NaN, logging once.

    The Stage B rejection-reason convention for this condition is
    `data_missing:utility`, parallel to `data_missing:tos` and
    `data_missing:range_prob` in `two_stage_selector.py`.
    """

    error_code = "UTILITY_CALIBRATOR_UNAVAILABLE"

    def __init__(self, artifact_path: str, message: str = ""):
        self.artifact_path = artifact_path
        detail = f": {message}" if message else ""
        super().__init__(f"Utility calibrator artifact unavailable at {artifact_path}{detail}")


@dataclass(frozen=True)
class UtilityConfig:
    """Configuration for utility-based scoring.

    Implements AFML-aligned utility function:
    U = E[net PnL over 6h] - λ * E[drawdown] - κ * E[trend_breakout_loss]
    """

    # Risk aversion coefficient for drawdown
    # Higher lambda = more penalty for drawdown risk
    lambda_risk: float = FALLBACK_LAMBDA_RISK

    # Trend breakout loss aversion coefficient
    # Higher kappa = more penalty for trend breakout scenarios
    kappa_trend: float = FALLBACK_KAPPA_TREND

    # Minimum utility score to pass validation
    min_utility: float = FALLBACK_MIN_UTILITY

    # Expected fills per hour (baseline assumption)
    baseline_fills_per_hour: float = FALLBACK_BASELINE_FILLS_PER_HOUR

    # Horizon in hours for return/drawdown estimation
    horizon_hours: float = FALLBACK_HORIZON_HOURS

    # Stop loss and take profit for breakout loss estimation
    stop_loss_pct: float = FALLBACK_STOP_LOSS_PCT  # -12%
    take_profit_pct: float = FALLBACK_TAKE_PROFIT_PCT  # +15%

    @classmethod
    def from_artifact(
        cls,
        artifact_path: Optional[Path | str] = None,
        *,
        expected_hmm_artifact_version: Optional[str] = None,
        **overrides: Any,
    ) -> "UtilityConfig":
        """Load governed utility parameters from a calibrated artifact.

        Artifact fields are narrow: calibrated score coefficients live under
        ``coefficients`` and the classification threshold lives under
        ``threshold``. Call-site keyword overrides are explicit operator
        overrides and win over the artifact when they are not ``None``.

        Fail-closed: raises :class:`UtilityCalibratorUnavailable` when the
        artifact is absent, unreadable, or fails schema validation.  Callers
        in the decision-time path must propagate; offline callers (scan-time
        feature snapshot, training, backfill) may catch and emit
        ``utility_score=None``/NaN.
        """
        path = Path(artifact_path) if artifact_path is not None else DEFAULT_UTILITY_ARTIFACT_PATH

        if not path.exists():
            raise UtilityCalibratorUnavailable(str(path), "current.json not found")

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise UtilityCalibratorUnavailable(str(path), f"unreadable JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise UtilityCalibratorUnavailable(str(path), "artifact root must be an object")

        schema_version = payload.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, (int, float))
            or int(schema_version) < 2
        ):
            raise UtilityCalibratorUnavailable(
                str(path),
                "schema_version must be >= 2 with governed HMM lineage",
            )
        if payload.get("promotable") is not True:
            raise UtilityCalibratorUnavailable(
                str(path),
                "artifact is not promotable",
            )

        from neutralgrid.models.artifacts import get_active_hmm_version

        expected_lineage = (
            expected_hmm_artifact_version or get_active_hmm_version()
        )
        if not expected_lineage:
            raise UtilityCalibratorUnavailable(
                str(path),
                "active HMM lineage is unavailable",
            )
        data_contract = payload.get("data_contract")
        artifact_lineage = (
            data_contract.get("hmm_artifact_version")
            if isinstance(data_contract, dict)
            else None
        )
        if not isinstance(artifact_lineage, str) or not artifact_lineage.strip():
            raise UtilityCalibratorUnavailable(
                str(path),
                "missing data_contract.hmm_artifact_version",
            )
        if artifact_lineage.strip() != expected_lineage:
            raise UtilityCalibratorUnavailable(
                str(path),
                "HMM lineage mismatch: "
                f"expected {expected_lineage}, found {artifact_lineage.strip()}",
            )

        coefficients = payload.get("coefficients") if isinstance(payload, dict) else None
        threshold = payload.get("threshold") if isinstance(payload, dict) else None
        if not isinstance(coefficients, dict) or not isinstance(threshold, dict):
            raise UtilityCalibratorUnavailable(
                str(path),
                "missing required 'coefficients' / 'threshold' sections",
            )

        values: Dict[str, float] = {
            "lambda_risk": FALLBACK_LAMBDA_RISK,
            "kappa_trend": FALLBACK_KAPPA_TREND,
            "min_utility": FALLBACK_MIN_UTILITY,
            "baseline_fills_per_hour": FALLBACK_BASELINE_FILLS_PER_HOUR,
            "horizon_hours": FALLBACK_HORIZON_HOURS,
            "stop_loss_pct": FALLBACK_STOP_LOSS_PCT,
            "take_profit_pct": FALLBACK_TAKE_PROFIT_PCT,
        }

        try:
            for key in ("lambda_risk", "kappa_trend", "horizon_hours"):
                if key in coefficients and coefficients[key] is not None:
                    values[key] = float(coefficients[key])
            if "min_utility" in threshold and threshold["min_utility"] is not None:
                values["min_utility"] = float(threshold["min_utility"])
        except (TypeError, ValueError) as exc:
            raise UtilityCalibratorUnavailable(
                str(path),
                f"non-numeric coefficient/threshold value: {exc}",
            ) from exc

        allowed_overrides = set(values)
        unknown = sorted(set(overrides) - allowed_overrides)
        if unknown:
            raise TypeError(f"Unknown UtilityConfig override(s): {unknown}")
        for key, value in overrides.items():
            if value is not None:
                values[key] = float(value)

        return cls(**values)

    def __post_init__(self):
        if self.lambda_risk < 0:
            raise ValueError("lambda_risk must be >= 0")
        if self.kappa_trend < 0:
            raise ValueError("kappa_trend must be >= 0")
        if self.horizon_hours <= 0:
            raise ValueError("horizon_hours must be > 0")


@dataclass
class UtilityComponents:
    """Decomposed utility score components for transparency.

    U = E[grid_return] - λ*E[drawdown] - κ*E[trend_breakout_loss]
    """

    expected_grid_return: float
    expected_drawdown: float
    expected_trend_breakout_loss: float  # New: κ term
    utility_score: float
    range_prob_factor: float
    trend_prob_factor: float
    survival_prob_factor: float = 1.0  # New: survival probability input
    config: Dict[str, Any] = field(default_factory=dict)


class UtilityScorer:
    """
    Compute continuous utility scores for grid bot candidates.

    Replaces hard thresholds with a calibrated utility function that
    balances expected return against drawdown risk.

    Usage:
        scorer = UtilityScorer(UtilityConfig.from_artifact(lambda_risk=2.0))
        result = scorer.compute_utility(
            range_prob=0.7,
            trend_prob=0.2,
            profit_per_grid_pct=0.8,
            num_grids=20,
            range_size_pct=3.0,
        )
        print(f"Utility: {result.utility_score:.4f}")
    """

    def __init__(self, config: Optional[UtilityConfig] = None):
        """
        Initialize UtilityScorer.

        Args:
            config: UtilityConfig with lambda_risk, min_utility, etc.
        """
        self.config = config or UtilityConfig.from_artifact()

    def expected_grid_return(
        self,
        profit_per_grid_pct: float,
        num_grids: int,
        range_prob: float,
        fills_per_hour: Optional[float] = None,
    ) -> float:
        """
        Compute expected grid return adjusted by range probability.

        When range_prob is high, we expect more fills (price stays in range).
        When range_prob is low, fill rate decreases.

        Args:
            profit_per_grid_pct: Net profit per grid in percent
            num_grids: Number of grid levels
            range_prob: HMM posterior probability of range regime
            fills_per_hour: Override baseline fills per hour

        Returns:
            Expected return as percentage over horizon
        """
        if fills_per_hour is None:
            fills_per_hour = self.config.baseline_fills_per_hour

        # Adjust fill rate by range probability
        # Higher range_prob -> more likely to fill grids
        adjusted_fills = fills_per_hour * range_prob

        # Total fills over horizon
        total_fills = adjusted_fills * self.config.horizon_hours

        # Active grid fraction — sqrt scaling for smooth saturation
        # (consistent with pnl_ranker.py active_fraction formula)
        active_fraction = min(0.75, (num_grids / 50.0) ** 0.5 * 0.5)

        # Expected return
        expected_return = total_fills * (profit_per_grid_pct / 100.0) * active_fraction

        return expected_return * 100.0  # Return as percentage

    def expected_drawdown(
        self,
        trend_prob: float,
        range_size_pct: float,
        stop_loss_pct: float = -12.0,
    ) -> float:
        """
        Compute expected drawdown scaled by trend probability.

        When trend_prob is high, we expect larger adverse moves.
        Drawdown is bounded by stop loss but weighted by probability.

        Args:
            trend_prob: HMM posterior probability of trend regime
            range_size_pct: Range size as percentage of price
            stop_loss_pct: Stop loss percentage (negative)

        Returns:
            Expected drawdown as positive percentage
        """
        # Base drawdown from range size (if price moves to boundary)
        base_drawdown = range_size_pct / 2.0

        # Trend probability increases expected drawdown
        # At trend_prob=1, expect full stop loss hit
        # At trend_prob=0, expect minimal drawdown
        trend_factor = trend_prob ** 0.5  # Non-linear scaling

        # Expected drawdown is weighted combination
        # - Base case: half the range (moderate drawdown)
        # - Trend case: closer to stop loss
        expected_dd = base_drawdown * (1 - trend_factor) + abs(stop_loss_pct) * trend_factor

        return expected_dd

    def expected_trend_breakout_loss(
        self,
        trend_prob: float,
        survival_prob: float,
        range_size_pct: float,
        hurst_exponent: Optional[float] = None,
    ) -> float:
        """
        Compute expected loss from trend breakout scenarios.

        This is the κ term in U = E[PnL] - λ*E[DD] - κ*E[trend_breakout_loss]

        When price breaks out of the range due to trending behavior:
        - Grid positions become adversely positioned
        - Stop loss may be hit
        - Unrealized losses accumulate

        The expected loss combines:
        - Probability of breakout (1 - survival_prob)
        - Magnitude of loss (scaled by trend strength)
        - Hurst exponent adjustment (more persistent trends = larger losses)

        Args:
            trend_prob: HMM posterior probability of trend regime
            survival_prob: Probability of staying in range over horizon
            range_size_pct: Range size as percentage of price
            hurst_exponent: Hurst exponent (>0.5 = trending), None uses trend_prob proxy

        Returns:
            Expected trend breakout loss as positive percentage
        """
        # Breakout probability
        breakout_prob = 1.0 - survival_prob

        # Trend strength factor from HMM
        # Higher trend_prob = more likely to have directional breakout
        trend_strength = trend_prob ** 0.7  # Slight dampening

        # Hurst adjustment: if provided and > 0.5, increases expected loss
        # Higher Hurst = more persistent trends = larger expected losses
        if hurst_exponent is not None and hurst_exponent > 0.5:
            hurst_factor = 1.0 + (hurst_exponent - 0.5) * 2.0  # [1.0, 2.0] for H in [0.5, 1.0]
        else:
            hurst_factor = 1.0

        # Base loss magnitude: if breakout occurs, expect to lose ~range/2 to stop_loss
        # Weighted by trend probability
        base_loss = range_size_pct * 0.5 * trend_strength

        # Worst case: full stop loss hit
        stop_loss_loss = abs(self.config.stop_loss_pct)

        # Expected breakout loss = P(breakout) * (weighted loss * hurst_factor)
        # Blend between base loss and stop loss based on trend probability
        loss_magnitude = base_loss * (1 - trend_prob) + stop_loss_loss * trend_prob
        expected_loss = breakout_prob * loss_magnitude * hurst_factor

        return expected_loss

    def compute_utility(
        self,
        range_prob: float,
        trend_prob: float,
        profit_per_grid_pct: float,
        num_grids: int,
        range_size_pct: float,
        stop_loss_pct: float = -12.0,
        fills_per_hour: Optional[float] = None,
        survival_prob: Optional[float] = None,
        hurst_exponent: Optional[float] = None,
    ) -> UtilityComponents:
        """
        Compute full utility score with decomposition.

        AFML-aligned formula:
        U = E[grid_return] - λ * E[drawdown] - κ * E[trend_breakout_loss]

        Args:
            range_prob: HMM posterior probability of range regime
            trend_prob: HMM posterior probability of trend regime
            profit_per_grid_pct: Net profit per grid in percent
            num_grids: Number of grid levels
            range_size_pct: Range size as percentage of price
            stop_loss_pct: Stop loss percentage (negative)
            fills_per_hour: Override baseline fills per hour
            survival_prob: Probability of staying in range (from stochastic checker)
            hurst_exponent: Hurst exponent for trend persistence

        Returns:
            UtilityComponents with score and decomposition
        """
        # Use range_prob as survival_prob proxy if not provided
        if survival_prob is None:
            survival_prob = range_prob

        # Compute components
        exp_return = self.expected_grid_return(
            profit_per_grid_pct=profit_per_grid_pct,
            num_grids=num_grids,
            range_prob=range_prob,
            fills_per_hour=fills_per_hour,
        )

        exp_drawdown = self.expected_drawdown(
            trend_prob=trend_prob,
            range_size_pct=range_size_pct,
            stop_loss_pct=stop_loss_pct,
        )

        # NEW: Compute trend breakout loss (κ term)
        exp_breakout_loss = self.expected_trend_breakout_loss(
            trend_prob=trend_prob,
            survival_prob=survival_prob,
            range_size_pct=range_size_pct,
            hurst_exponent=hurst_exponent,
        )

        # AFML Utility = E[Return] - λ*E[Drawdown] - κ*E[Trend_Breakout_Loss]
        utility = (
            exp_return
            - self.config.lambda_risk * exp_drawdown
            - self.config.kappa_trend * exp_breakout_loss
        )

        return UtilityComponents(
            expected_grid_return=exp_return,
            expected_drawdown=exp_drawdown,
            expected_trend_breakout_loss=exp_breakout_loss,
            utility_score=utility,
            range_prob_factor=range_prob,
            trend_prob_factor=trend_prob,
            survival_prob_factor=survival_prob,
            config={
                "lambda_risk": self.config.lambda_risk,
                "kappa_trend": self.config.kappa_trend,
                "horizon_hours": self.config.horizon_hours,
                "min_utility": self.config.min_utility,
            },
        )

    def passes_threshold(self, utility_components: UtilityComponents) -> bool:
        """
        Check if utility score passes minimum threshold.

        Args:
            utility_components: Result from compute_utility()

        Returns:
            True if utility >= min_utility
        """
        return utility_components.utility_score >= self.config.min_utility


def compute_governed_provisional_utility(
    *,
    range_prob: float,
    trend_prob: float,
    artifact_path: Optional[Path | str] = None,
    lambda_risk: Optional[float] = None,
    min_utility: Optional[float] = None,
) -> UtilityComponents:
    """Compute the single governed provisional utility used by the bootstrap path.

    This intentionally ignores candidate-specific geometry, survival, and Hurst
    so training, scan-time inference, and sidecar backfills all share the same
    provisional utility meaning.
    """
    from neutralgrid.core.config import get_config

    provisional_cfg = get_config().utility
    overrides: Dict[str, Any] = {}
    if lambda_risk is not None:
        overrides["lambda_risk"] = float(lambda_risk)
    if min_utility is not None:
        overrides["min_utility"] = float(min_utility)

    scorer = UtilityScorer(UtilityConfig.from_artifact(artifact_path=artifact_path, **overrides))
    return scorer.compute_utility(
        range_prob=float(range_prob),
        trend_prob=float(trend_prob),
        profit_per_grid_pct=float(provisional_cfg.provisional_profit_per_grid_pct),
        num_grids=int(provisional_cfg.provisional_num_grids),
        range_size_pct=float(provisional_cfg.provisional_range_size_pct),
    )
