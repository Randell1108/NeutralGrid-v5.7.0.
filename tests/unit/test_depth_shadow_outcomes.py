from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_depth_shadow_outcomes import _capital_fraction


def test_capital_fraction_preserves_explicit_zero_over_ps_fraction() -> None:
    row = pd.Series(
        {
            "capital_fraction": 0.0,
            "ps_fraction": 0.5,
            "deploy_margin_usdt": 400.0,
            "capital_base_usdt": 400.0,
        }
    )

    assert _capital_fraction(row) == pytest.approx(0.0)


def test_capital_fraction_derives_from_deploy_margin_when_no_fraction_exists() -> None:
    row = pd.Series(
        {
            "capital_fraction": None,
            "ps_fraction": None,
            "deploy_margin_usdt": 20.0,
            "capital_base_usdt": 400.0,
        }
    )

    assert _capital_fraction(row) == pytest.approx(0.05)
