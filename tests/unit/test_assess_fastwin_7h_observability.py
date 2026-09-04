from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys

import pandas as pd


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "assess_fastwin_7h_observability.py"
)
_SPEC = importlib.util.spec_from_file_location("assess_fastwin_7h_observability", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _row(candidate_id: str, *, target_reached: bool = False) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "symbol": "BTCUSDT",
        "backtest_start_ts_utc": "2026-06-08T12:00:00+00:00",
        "duration_hours": 6.0,
        "target_reached": target_reached,
        "time_to_target_hours": None,
        "net_pnl_pct": 1.0,
        "grid_lower": 100.0,
        "grid_upper": 110.0,
        "num_grids": 5,
        "capital_base": 400.0,
        "capital_fraction": 1.0,
        "funding_mode": "continuous",
        "mode": "geometric",
        "realism_profile": "legacy",
        "is_authoritative": True,
        "max_holding_bars": 360,
        "price_source": "last",
    }


def test_selects_only_six_hour_negative_observations_deterministically() -> None:
    accepted = _row("BTCUSDT_20260608_120000_deadbeef")
    rejected = _row("ETHUSDT_20260608_120000_cafebabe", target_reached=True)
    frame = pd.DataFrame([accepted, rejected])

    cohort = _MODULE.select_isolated_cohort(frame, sample_size=10)

    assert cohort["candidate_id"].tolist() == [accepted["candidate_id"]]


def test_records_a_seven_hour_label_flip_only_after_baseline_reproduction() -> None:
    cohort = pd.DataFrame([_row("BTCUSDT_20260608_120000_deadbeef")])

    class _Client:
        async def close(self) -> None:
            return None

    async def _fetch(*_args, **_kwargs) -> pd.DataFrame:
        return pd.DataFrame({"close": range(420)})

    calls: list[int] = []

    def _runner(*, max_holding_bars: int, **_kwargs):
        calls.append(max_holding_bars)
        if max_holding_bars == 360:
            return {"net_pnl_pct": 1.0, "target_reached": False, "time_to_target_hours": None}
        return {"net_pnl_pct": 3.1, "target_reached": True, "time_to_target_hours": 6.5}

    detail, summary = asyncio.run(
        _MODULE.assess_cohort(
            cohort,
            config=_MODULE.AssessmentConfig(sample_size=1, leverage=10),
            fetcher=_fetch,
            runner=_runner,
            client_factory=_Client,
        )
    )

    assert calls == [360, 420]
    assert detail.loc[0, "status"] == "comparable"
    assert bool(detail.loc[0, "label_flip_at_hour_7"])
    assert summary["label_flips"] == 1
    assert summary["calibration_input_impact"] == "affected_in_isolated_cohort"


def test_excludes_nonreproducible_baselines_from_label_impact() -> None:
    cohort = pd.DataFrame([_row("BTCUSDT_20260608_120000_deadbeef")])

    class _Client:
        async def close(self) -> None:
            return None

    async def _fetch(*_args, **_kwargs) -> pd.DataFrame:
        return pd.DataFrame({"close": range(420)})

    def _runner(*, max_holding_bars: int, **_kwargs):
        if max_holding_bars == 360:
            return {"net_pnl_pct": 2.0, "target_reached": False, "time_to_target_hours": None}
        raise AssertionError("seven-hour run must not follow a nonreproducible baseline")

    detail, summary = asyncio.run(
        _MODULE.assess_cohort(
            cohort,
            config=_MODULE.AssessmentConfig(sample_size=1, leverage=10),
            fetcher=_fetch,
            runner=_runner,
            client_factory=_Client,
        )
    )

    assert detail.loc[0, "status"] == "baseline_not_reproducible"
    assert summary["comparable_rows"] == 0
    assert summary["label_flips"] == 0
