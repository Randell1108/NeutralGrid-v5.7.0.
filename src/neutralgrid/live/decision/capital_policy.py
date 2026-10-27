"""Dependency-light statement of the operator's capital-preservation objective."""
from __future__ import annotations

from typing import Any


def policy_summary() -> dict[str, Any]:
    """Return fresh metadata without granting execution or inferring risk limits."""
    return {
        "primary_objective": "preserve_invested_capital",
        "reference": "invested_capital",
        "maximum_capital_loss_pct": 0,
        "applicability": "bot_lifetime",
        "minimum_liquidation_buffer_pct": None,
        "independent_risk_limits_required": True,
        "future_preservation_guaranteed": False,
        "execution_authorized": False,
        "training_eligible": False,
    }
