from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
from typing import cast

import pandas as pd
import pytest


_PATH = Path(__file__).resolve().parents[2] / "scripts" / "replay_authoritative_fastwin_7h.py"
_SPEC = importlib.util.spec_from_file_location("replay_authoritative_fastwin_7h", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _source() -> dict[str, object]:
    return {
        "candidate_id": "BTCUSDT_20260608_120000_deadbeef",
        "symbol": "BTCUSDT",
        "backtest_start_ts_utc": "2026-06-08T12:00:00+00:00",
        "grid_lower": 100.0,
        "grid_upper": 110.0,
        "num_grids": 5,
        "capital_base": 400.0,
        "funding_mode": "continuous",
        "realism_profile": "legacy",
        "net_pnl_pct": 1.0,
        "target_reached": False,
        "time_to_target_hours": None,
        "hmm_artifact_version": "stale",
        "range_prob": 0.1,
    }


def test_input_contract_rejects_candidate_id_set_difference() -> None:
    backtest = pd.DataFrame([_source()])
    training = pd.DataFrame([{"candidate_id": "ETHUSDT_20260608_120000_deadbeef"}])
    with pytest.raises(ValueError, match="candidate_id sets differ"):
        _MODULE.validate_inputs(backtest, training)


def test_baseline_comparison_rejects_changed_target_time() -> None:
    source = _source()
    replay = {"net_pnl_pct": 1.0, "target_reached": False, "time_to_target_hours": 5.9}
    assert _MODULE.baseline_matches(source, replay) == (False, "time_to_target_hours_mismatch")


def test_replay_only_emits_rows_after_exact_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source()
    bars = pd.DataFrame({"close": range(420)})

    def fake_runner(_source: dict[str, object], _bars: pd.DataFrame, **kwargs: object) -> dict[str, object]:
        if kwargs["max_holding_bars"] == 360:
            return {"net_pnl_pct": 1.0, "target_reached": False, "time_to_target_hours": None}
        return {"net_pnl_pct": 4.0, "target_reached": True, "time_to_target_hours": 6.5}

    monkeypatch.setattr(_MODULE, "convert_to_training_row", lambda result, candidate, horizon_hours: dict(candidate, **result))
    raw, training, reason = _MODULE.replay_one(source, bars, leverage=10, runner=fake_runner)

    assert reason == ""
    assert raw is not None and training is not None
    assert raw["target_reached"] is True
    assert raw["hmm_artifact_version"] is None
    assert training["range_prob"] is None


def test_replay_rejects_baseline_before_extended_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_runner(_source: dict[str, object], _bars: pd.DataFrame, **kwargs: object) -> dict[str, object]:
        calls.append(int(cast(int, kwargs["max_holding_bars"])))
        return {"net_pnl_pct": 9.0, "target_reached": False, "time_to_target_hours": None}

    monkeypatch.setattr(_MODULE, "convert_to_training_row", lambda result, candidate, horizon_hours: dict(candidate, **result))
    raw, training, reason = _MODULE.replay_one(_source(), pd.DataFrame({"close": range(420)}), leverage=10, runner=fake_runner)

    assert raw is None and training is None
    assert reason == "net_pnl_pct_mismatch"
    assert calls == [360]


def test_resume_rejects_unpaired_candidate_checkpoints(tmp_path: Path) -> None:
    pd.DataFrame([{"candidate_id": "BTCUSDT_20260608_120000_deadbeef"}]).to_csv(
        tmp_path / "replayed_backtest.csv", index=False
    )
    pd.DataFrame([{"candidate_id": "ETHUSDT_20260608_120000_deadbeef"}]).to_csv(
        tmp_path / "replayed_training.csv", index=False
    )
    source = pd.DataFrame([_source()])
    training = pd.DataFrame([{"candidate_id": _source()["candidate_id"]}])

    with pytest.raises(ValueError, match="candidate_id sets differ"):
        asyncio.run(
            _MODULE.replay_all(
                source,
                training,
                checkpoint_dir=tmp_path,
                leverage=10,
            )
        )
