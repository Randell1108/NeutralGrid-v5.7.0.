"""FASTWIN-01: Stage B soft meta gate (TwoStageConfig.meta_gate_enabled).

The soft meta gate is default-ON as of ERR-059 (2026-06-22); these tests toggle
``meta_gate_enabled`` explicitly via the helper. When enabled it must:
  * be a no-op when disabled (no ``meta_gate`` key contributed),
  * fail-closed on a missing/NaN meta_prob with ``data_missing:meta``
    (parallel to ``data_missing:tos`` — never a silent skip),
  * reject when ``meta_prob < min_meta_prob``,
  * pass when ``meta_prob >= min_meta_prob``.

Gates 1-4 are forced to pass so the meta gate's effect is isolated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

from neutralgrid.core.config import MicroOscConfig, TwoStageConfig
from neutralgrid.scanner.two_stage_selector import StageBResult, TwoStageSelector


@dataclass
class _FakeGlobalConfig:
    micro_osc: MicroOscConfig


def _approve(
    *,
    meta_gate_enabled: bool = False,
    min_meta_prob: float = 0.50,
    meta_prob: Optional[float] = None,
    tos: float = 60.0,
) -> StageBResult:
    """Run Stage B with gates 1-4 forced to pass, isolating the meta gate."""
    cfg = TwoStageConfig(meta_gate_enabled=meta_gate_enabled, min_meta_prob=min_meta_prob)
    sel = TwoStageSelector(config=cfg)
    fake_global = _FakeGlobalConfig(micro_osc=MicroOscConfig(enabled=False))
    with patch(
        "neutralgrid.scanner.two_stage_selector.get_config",
        return_value=fake_global,
    ):
        return sel.approve(
            hard_gate_passed=True,
            tos=tos,
            position_size_fraction=0.50,
            range_prob=0.50,  # standard-mode Gate 4 passes
            symbol="BTCUSDT",
            meta_prob=meta_prob,
        )


def test_meta_gate_disabled_by_default_is_noop_even_with_missing_meta_prob():
    result = _approve(meta_prob=None)
    assert result.approved is True
    assert "meta_gate" not in result.gate_results
    assert not any("data_missing:meta" in c for c in result.rejection_codes)


def test_meta_gate_disabled_ignores_low_meta_prob():
    result = _approve(min_meta_prob=0.9, meta_prob=0.01)
    assert result.approved is True
    assert "meta_gate" not in result.gate_results


def test_meta_gate_enabled_fails_closed_on_none():
    result = _approve(meta_gate_enabled=True, min_meta_prob=0.5, meta_prob=None)
    assert result.approved is False
    assert result.gate_results["meta_gate"] is False
    assert "data_missing:meta" in result.rejection_codes


def test_meta_gate_enabled_fails_closed_on_nan():
    result = _approve(meta_gate_enabled=True, min_meta_prob=0.5, meta_prob=float("nan"))
    assert result.approved is False
    assert result.gate_results["meta_gate"] is False
    assert "data_missing:meta" in result.rejection_codes


def test_meta_gate_enabled_rejects_below_min():
    result = _approve(meta_gate_enabled=True, min_meta_prob=0.6, meta_prob=0.40)
    assert result.approved is False
    assert result.gate_results["meta_gate"] is False
    assert any("meta_prob_below_min" in c for c in result.rejection_codes)
    assert result.details["meta_prob"] == 0.40
    assert result.details["min_meta_prob"] == 0.6


def test_meta_gate_enabled_passes_at_or_above_min():
    result = _approve(meta_gate_enabled=True, min_meta_prob=0.6, meta_prob=0.60)
    assert result.approved is True
    assert result.gate_results["meta_gate"] is True
    assert not any("data_missing:meta" in c for c in result.rejection_codes)


def test_meta_gate_does_not_mask_other_gate_failures():
    # Gate 2 (tos) fails AND meta gate passes -> still rejected, both surfaced.
    result = _approve(meta_gate_enabled=True, min_meta_prob=0.5, meta_prob=0.9, tos=0.0)
    assert result.approved is False
    assert result.gate_results["meta_gate"] is True
    assert result.gate_results["tos"] is False


def test_meta_gate_nan_helper_sanity():
    # math.isnan guard used by the gate behaves as expected for a Python float.
    assert math.isnan(float("nan")) is True
