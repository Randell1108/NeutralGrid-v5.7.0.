"""Deployment payload helpers with enforced capital_fraction sizing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from neutralgrid.core.config import get_config


@dataclass(frozen=True)
class DeploymentSizing:
    """Resolved deployment sizing values."""

    capital_base_usdt: float
    capital_fraction: float
    effective_margin_usdt: float


def _to_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        if out != out:  # NaN
            return float(default)
        return out
    except Exception:
        return float(default)


def _bounded_fraction(value: Any) -> float:
    frac = _to_float(value, 1.0)
    return float(min(1.0, max(0.0, frac)))


def resolve_deployment_sizing(
    candidate_row: Dict[str, Any],
    *,
    capital_base_usdt: float | None = None,
) -> DeploymentSizing:
    """
    Resolve enforced deployment margin from candidate sizing fields.

    Priority for base capital:
      1) explicit capital_base_usdt argument
      2) candidate_row["capital_base_usdt"]
      3) candidate_row["capital"]
      4) config.grid.capital
    """
    cfg_base = float(get_config().grid.capital)
    base = cfg_base
    if capital_base_usdt is not None:
        base = _to_float(capital_base_usdt, cfg_base)
    elif "capital_base_usdt" in candidate_row:
        base = _to_float(candidate_row.get("capital_base_usdt"), cfg_base)
    elif "capital" in candidate_row:
        base = _to_float(candidate_row.get("capital"), cfg_base)

    frac = _bounded_fraction(candidate_row.get("capital_fraction"))
    effective = float(base) * float(frac)

    return DeploymentSizing(
        capital_base_usdt=float(base),
        capital_fraction=float(frac),
        effective_margin_usdt=float(effective),
    )


def build_binance_grid_payload(
    candidate_row: Dict[str, Any],
    *,
    capital_base_usdt: float | None = None,
) -> Dict[str, Any]:
    """
    Build a deployment payload with enforced margin sizing.

    This does not place orders; it standardizes the execution contract.
    """
    sizing = resolve_deployment_sizing(
        candidate_row,
        capital_base_usdt=capital_base_usdt,
    )
    return {
        "candidate_id": candidate_row.get("candidate_id"),
        "symbol": candidate_row.get("symbol"),
        "grid_lower": candidate_row.get("grid_lower"),
        "grid_upper": candidate_row.get("grid_upper"),
        "num_grids": candidate_row.get("num_grids"),
        "leverage": candidate_row.get("leverage"),
        "investment_margin_usdt": sizing.effective_margin_usdt,
        "capital_fraction": sizing.capital_fraction,
        "capital_base_usdt": sizing.capital_base_usdt,
    }

