"""CLI regression tests for retrain_scanner.py + scan.py degraded-mode contract."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from neutralgrid.scanner.pattern_profile import PatternProfile
from neutralgrid.scanner.profile_model import ProfileModel
from neutralgrid.scanner.profile_model_walkforward import (
    COVERAGE_FLOOR,
    WalkForwardResult,
    promote_profile_version,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_FEATURES = [
    "parkinson_vol_ratio_4h_24h_pre",
    "variance_ratio_1m_15m_pre_2h",
    "funding_carry_expected_next_7h",
    "liquidity_stability_z_1h",
]


def _load_retrain_scanner() -> ModuleType:
    """Import retrain_scanner.py from repo root without polluting sys.path globally."""
    path = _REPO_ROOT / "retrain_scanner.py"
    spec = importlib.util.spec_from_file_location("retrain_scanner_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workspace_tmp() -> Path:
    p = Path.cwd() / ".pytest_tmp" / f"cli_{uuid.uuid4().hex}"
    p.mkdir(parents=True, exist_ok=False)
    return p


def _dummy_model(feats: list[str]) -> ProfileModel:
    n = len(feats)
    return ProfileModel(
        features=list(feats),
        winner_mu={f: 0.5 for f in feats},
        loser_mu={f: -0.5 for f in feats},
        inv_cov=np.eye(n).tolist(),
        prior_winner=0.5,
        duration_band={"min_hours": 0.0, "max_hours": 7.0},
        feature_mean={f: 0.0 for f in feats},
        feature_std={f: 1.0 for f in feats},
    )


def _dummy_pattern(feats: list[str]) -> PatternProfile:
    return PatternProfile(
        features=list(feats),
        means={f: 0.0 for f in feats},
        stds={f: 1.0 for f in feats},
        q10={f: -1.0 for f in feats},
        q90={f: 1.0 for f in feats},
        selection_summary={"winners_count": 40, "pnl_threshold": 5.0},
    )


def _wf(pass_rate: float, coverage: float, requested: list[str], admitted: list[str]) -> WalkForwardResult:
    passing_fixture = np.isclose(pass_rate, 0.60)
    fold_auc = (
        [0.6, 0.6, 0.6, 0.5, 0.5]
        if passing_fixture
        else [0.6, 0.6, 0.6, 0.6]
    )
    fold_count = len(fold_auc)
    labels = [0] * 10 + [1] * 10
    scores = [float(value) for value in range(10)] + [
        -4.0,
        -3.0,
        -2.0,
        -1.0,
        20.0,
        21.0,
        22.0,
        23.0,
        24.0,
        25.0,
    ]
    return WalkForwardResult(
        n_folds=fold_count,
        fold_auc=fold_auc,
        fold_ks=[0.3] * fold_count,
        mean_auc=float(np.mean(fold_auc)),
        mean_ks=0.3,
        mean_pass_rate=pass_rate,
        purge_hours=7.0,
        requested_features=requested,
        admitted_features=admitted,
        feature_coverage=coverage,
        fold_train_rows=[100] * fold_count,
        fold_test_rows=[4] * fold_count,
        fold_train_winners=[40] * fold_count,
        fold_test_winners=[2] * fold_count,
        fold_pnl_thresholds=[5.0] * fold_count,
        fold_test_start_utc=[
            f"2026-01-{index + 1:02d}T00:00:00+00:00"
            for index in range(fold_count)
        ],
        fold_test_end_utc=[
            f"2026-01-{index + 1:02d}T03:00:00+00:00"
            for index in range(fold_count)
        ],
        oof_strategy_ids=[f"bot_{index}" for index in range(20)],
        oof_labels=labels,
        oof_scores=scores,
        oof_probabilities=[0.4] * 20,
        pooled_oof_auc=0.60,
        pooled_oof_ks=0.60,
        pooled_oof_brier=0.26,
        pooled_oof_ece=0.10,
        source_sha256="a" * 64,
        labeled_rows=100,
    )


def test_skip_model_flag_rejected_by_argparse(monkeypatch):
    """`--skip-model` flag is not recognized; argparse exits non-zero."""
    rs = _load_retrain_scanner()
    monkeypatch.setattr(sys, "argv", ["retrain_scanner.py", "--skip-model"])
    with pytest.raises(SystemExit) as exc:
        rs.parse_args()
    assert exc.value.code != 0


def test_profile_model_training_failure_exits_nonzero_without_completion_banner(
    monkeypatch,
    caplog,
):
    """Training exception triggers sys.exit(1); no completion banner."""
    rs = _load_retrain_scanner()
    import pandas as pd

    tmp = _workspace_tmp()
    try:
        xlsx = tmp / "stub.xlsx"
        pd.DataFrame([{"strategy_id": "x", "pnl_pct": 1.0}]).to_excel(
            str(xlsx), index=False
        )
        out_dir = tmp / "out"

        class _StubProfile:
            features = []
            selection_summary = {"winners_count": 0, "pnl_threshold": 0.0}

            def save_json(self, _p):
                pass

        def _fake_build(*_a, **_k):
            return _StubProfile()

        def _fake_train(*_a, **_k):
            raise RuntimeError("synthetic training failure")

        monkeypatch.setattr(rs, "build_profile_from_enhanced_xlsx", _fake_build)
        monkeypatch.setattr(rs, "train_profile_model_from_enhanced_xlsx", _fake_train)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "retrain_scanner.py",
                "--input",
                str(xlsx),
                "--output-dir",
                str(out_dir),
                "--skip-gate",
            ],
        )
        caplog.set_level("INFO", logger="retrain_scanner")
        with pytest.raises(SystemExit) as exc:
            rs.main()
        assert exc.value.code == 1
        assert not any("Training complete!" in rec.getMessage() for rec in caplog.records)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_scan_py_profile_model_absent_emits_warn_and_flag():
    """scan.py source must carry both the boot WARN and the per-row flag."""
    scan_src = (
        _REPO_ROOT / "src" / "neutralgrid" / "scanner" / "scan.py"
    ).read_text(encoding="utf-8")
    assert "profile_model is None" in scan_src
    assert "profile_model is absent" in scan_src
    assert 'scoring_flags.append("profile_model_absent")' in scan_src


def test_requested_features_shrinkage_does_not_lower_coverage_floor():
    """COVERAGE_FLOOR is applied independently to each request."""
    tmp1 = _workspace_tmp()
    tmp2 = _workspace_tmp()
    try:
        req1 = [f"f{i}" for i in range(10)]
        adm1 = req1[:8]
        model1 = _dummy_model(adm1)
        wf1 = _wf(pass_rate=0.60, coverage=0.80, requested=req1, admitted=adm1)
        d1 = promote_profile_version(
            model1,
            requested_features=req1,
            wf_result=wf1,
            candidate_pattern_profile=_dummy_pattern(adm1),
            profile_dir=tmp1,
        )
        assert d1.promoted is False
        assert "feature_coverage" in d1.reason
        assert not (tmp1 / "current.json").exists()

        req2 = [f"f{i}" for i in range(5)]
        adm2 = req2[:4]
        model2 = _dummy_model(adm2)
        wf2 = _wf(pass_rate=0.60, coverage=0.80, requested=req2, admitted=adm2)
        d2 = promote_profile_version(
            model2,
            requested_features=req2,
            wf_result=wf2,
            candidate_pattern_profile=_dummy_pattern(adm2),
            profile_dir=tmp2,
        )
        assert d2.promoted is False
        assert "feature_coverage" in d2.reason
        assert not (tmp2 / "current.json").exists()
        assert COVERAGE_FLOOR == pytest.approx(0.90)
    finally:
        shutil.rmtree(tmp1, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)


def test_retrain_skips_legacy_profile_gate_for_four_feature_contract(monkeypatch, caplog):
    rs = _load_retrain_scanner()
    tmp = _workspace_tmp()
    try:
        xlsx = tmp / "stub.xlsx"
        pd.DataFrame([{"strategy_id": "x", "pnl_pct": 1.0}]).to_excel(
            str(xlsx), index=False, sheet_name="General"
        )
        out_dir = tmp / "out"

        class _StubProfile:
            features = list(_PROFILE_FEATURES)
            q10 = {}
            q90 = {}
            selection_summary = {"winners_count": 30, "pnl_threshold": 3.43}

            def save_json(self, path):
                Path(path).write_text("{}", encoding="utf-8")

            def to_json(self):
                return {
                    "features": self.features,
                    "means": {},
                    "stds": {},
                    "q10": {},
                    "q90": {},
                    "selection_summary": self.selection_summary,
                }

        def _fake_backfill(*_a, **_k):
            return None

        def _fake_build(*_a, **_k):
            return _StubProfile()

        def _fake_train(*_a, **_k):
            return _dummy_model(_PROFILE_FEATURES)

        monkeypatch.setattr(rs, "backfill_profile_features", _fake_backfill)
        monkeypatch.setattr(rs, "build_profile_from_enhanced_xlsx", _fake_build)
        monkeypatch.setattr(rs, "train_profile_model_from_enhanced_xlsx", _fake_train)
        monkeypatch.setattr(
            rs,
            "walkforward_evaluate",
            lambda *_a, **_k: _wf(
                pass_rate=0.60,
                coverage=1.0,
                requested=_PROFILE_FEATURES,
                admitted=_PROFILE_FEATURES,
            ),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "retrain_scanner.py",
                "--input",
                str(xlsx),
                "--output-dir",
                str(out_dir),
            ],
        )
        caplog.set_level("INFO", logger="retrain_scanner")
        rs.main()
        assert (out_dir / "current.json").exists()
        current = json.loads((out_dir / "current.json").read_text(encoding="utf-8"))
        assert current["active"].startswith("profile_model_")
        assert current["active_pattern_profile"].startswith("pattern_profile_")
        assert (out_dir / current["active"]).exists()
        assert (out_dir / current["active_pattern_profile"]).exists()
        assert (out_dir / current["evaluation"]).exists()
        assert not (out_dir / "profile_model.json").exists()
        assert not (out_dir / "pattern_profile.json").exists()
        assert not (out_dir / "profile_gate.json").exists()
        assert any(
            "Skipping profile_gate generation" in rec.getMessage()
            for rec in caplog.records
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rejected_retrain_retains_existing_bootstrap_pair(monkeypatch):
    """A failed gate may write evidence but cannot mutate effective bootstrap files."""
    rs = _load_retrain_scanner()
    tmp = _workspace_tmp()
    try:
        xlsx = tmp / "stub.xlsx"
        pd.DataFrame([{"strategy_id": "x", "pnl_pct": 1.0}]).to_excel(
            str(xlsx), index=False, sheet_name="General"
        )
        out_dir = tmp / "out"
        out_dir.mkdir()
        incumbent = _dummy_model(_PROFILE_FEATURES)
        incumbent_model_text = json.dumps(incumbent.to_json(), sort_keys=True)
        incumbent_pattern_text = '{"incumbent": true}'
        (out_dir / "profile_model.json").write_text(
            incumbent_model_text, encoding="utf-8"
        )
        (out_dir / "pattern_profile.json").write_text(
            incumbent_pattern_text, encoding="utf-8"
        )

        class _StubProfile:
            features = list(_PROFILE_FEATURES)
            q10 = {}
            q90 = {}
            selection_summary = {"winners_count": 30, "pnl_threshold": 3.43}

            def save_json(self, path):
                Path(path).write_text(json.dumps(self.to_json()), encoding="utf-8")

            def to_json(self):
                return {
                    "features": self.features,
                    "means": {},
                    "stds": {},
                    "q10": {},
                    "q90": {},
                    "selection_summary": self.selection_summary,
                }

        monkeypatch.setattr(rs, "backfill_profile_features", lambda *_a, **_k: None)
        monkeypatch.setattr(
            rs, "build_profile_from_enhanced_xlsx", lambda *_a, **_k: _StubProfile()
        )
        monkeypatch.setattr(
            rs,
            "train_profile_model_from_enhanced_xlsx",
            lambda *_a, **_k: _dummy_model(_PROFILE_FEATURES),
        )
        monkeypatch.setattr(
            rs,
            "walkforward_evaluate",
            lambda *_a, **_k: _wf(
                pass_rate=0.0,
                coverage=1.0,
                requested=_PROFILE_FEATURES,
                admitted=_PROFILE_FEATURES,
            ),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "retrain_scanner.py",
                "--input",
                str(xlsx),
                "--output-dir",
                str(out_dir),
                "--incumbent-shrinkage",
                "0.30",
                "--skip-gate",
            ],
        )

        rs.main()

        assert (out_dir / "profile_model.json").read_text(
            encoding="utf-8"
        ) == incumbent_model_text
        assert (out_dir / "pattern_profile.json").read_text(
            encoding="utf-8"
        ) == incumbent_pattern_text
        assert not (out_dir / "current.json").exists()
        evaluations = list((out_dir / "evaluations").glob("profile_evaluation_*.json"))
        assert len(evaluations) == 1
        assert json.loads(evaluations[0].read_text(encoding="utf-8"))[
            "gate_decision"
        ] == "reject"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _history_df(
    periods: int,
    freq: str,
    *,
    base_price: float,
    step: float,
    start: str | pd.Timestamp = "2026-01-01T00:00:00Z",
) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    close = base_price + np.arange(periods, dtype=float) * step
    return pd.DataFrame(
        {
            "open_time": idx,
            "close_time": idx + pd.to_timedelta(pd.Timedelta(freq)),
            "open": close - 0.05,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(periods, 10.0),
            "quote_volume": 1000.0 + np.arange(periods, dtype=float) * 25.0,
        }
    )


def test_backfill_profile_features_updates_general_sheet_only(monkeypatch):
    rs = _load_retrain_scanner()
    tmp = _workspace_tmp()
    try:
        entry_time = pd.Timestamp("2026-01-02T12:00:00Z")
        xlsx = tmp / "backfill.xlsx"
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            pd.DataFrame(
                [
                    {
                        "strategy_id": "bot_1",
                        "symbol": "BTCUSDT",
                        "start_time_utc": entry_time.isoformat(),
                    }
                ]
            ).to_excel(writer, index=False, sheet_name="General")
            pd.DataFrame([{"sentinel": 1}]).to_excel(
                writer, index=False, sheet_name="Entry Validation Metrics"
            )

        history_15m = _history_df(
            200,
            "15min",
            base_price=100.0,
            step=0.05,
            start=entry_time - pd.Timedelta(hours=50),
        )
        history_1m = _history_df(
            220,
            "1min",
            base_price=100.0,
            step=0.005,
            start=entry_time - pd.Timedelta(minutes=220),
        )

        async def _fake_ensure_and_load_history(symbol, *, interval, start_ms, end_ms, min_years):
            assert symbol == "BTCUSDT"
            assert end_ms >= int(entry_time.timestamp() * 1000)
            return history_15m.copy() if interval == "15m" else history_1m.copy()

        class _FakeFundingLoader:
            def __init__(self, client):
                self.client = client

            async def fetch_raw(self, symbol, start_time_ms, end_time_ms):
                assert symbol == "BTCUSDT"
                return [
                    {
                        "fundingTime": int(entry_time.timestamp() * 1000) + 2 * 3600 * 1000,
                        "fundingRate": "0.0002",
                    }
                ]

        class _FakeClient:
            async def close(self):
                return None

        monkeypatch.setattr(rs, "_ensure_and_load_history", _fake_ensure_and_load_history)
        monkeypatch.setattr(rs, "HistoricalFundingLoader", _FakeFundingLoader)
        monkeypatch.setattr(rs, "BinanceClient", _FakeClient)

        rs.backfill_profile_features(xlsx)

        with pd.ExcelFile(xlsx) as xl:
            assert "General" in xl.sheet_names
            assert "Entry Validation Metrics" in xl.sheet_names
            general = pd.read_excel(xl, sheet_name="General")
            legacy = pd.read_excel(xl, sheet_name="Entry Validation Metrics")

        for feature in (
            "parkinson_vol_ratio_4h_24h_pre",
            "variance_ratio_1m_15m_pre_2h",
            "funding_carry_expected_next_7h",
            "liquidity_stability_z_1h",
        ):
            assert feature in general.columns
            assert pd.notna(general.loc[0, feature])

        assert list(legacy.columns) == ["sentinel"]
        assert legacy.loc[0, "sentinel"] == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ensure_and_load_history_rebuilds_unreadable_cache(monkeypatch):
    rs = _load_retrain_scanner()
    tmp = _workspace_tmp()
    try:
        cache_root = tmp / "data" / "cache" / "klines"
        pq_root = cache_root / "futures_um" / "BTCUSDT" / "15m" / "parquet"
        pq_root.mkdir(parents=True, exist_ok=False)
        (pq_root / "broken.parquet").write_text("", encoding="utf-8")

        load_calls = {"count": 0}
        ensure_args = {}

        def _fake_load_parquet_range(root, start_ms=None, end_ms=None):
            load_calls["count"] += 1
            if load_calls["count"] == 1:
                raise RuntimeError("parquet file size is 0 bytes")
            return _history_df(
                120,
                "15min",
                base_price=100.0,
                step=0.05,
                start="2026-01-01T00:00:00Z",
            )

        async def _fake_ensure_kline_store(**kwargs):
            ensure_args.update(kwargs)
            return {}

        monkeypatch.setattr(
            rs,
            "get_config",
            lambda: SimpleNamespace(base_dir=str(tmp)),
        )
        monkeypatch.setattr(rs, "load_parquet_range", _fake_load_parquet_range)
        monkeypatch.setattr(rs, "ensure_kline_store", _fake_ensure_kline_store)

        df = asyncio.run(
            rs._ensure_and_load_history(
                "BTCUSDT",
                interval="15m",
                start_ms=0,
                end_ms=1_000_000,
                min_years=0.25,
            )
        )

        assert not df.empty
        assert ensure_args["mode"] == "backfill"
        assert ensure_args["force"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ensure_and_load_history_degrades_when_ensure_kline_store_fails(monkeypatch):
    """A store-validation failure (e.g. young listing below the row-count
    minimum) must degrade to whatever cached history exists, not abort the
    whole retrain (LABUSDT 2026-07-10 regression)."""
    rs = _load_retrain_scanner()
    tmp = _workspace_tmp()
    try:
        partial = _history_df(
            120,
            "15min",
            base_price=100.0,
            step=0.05,
            start="2026-03-01T00:00:00Z",
        )

        async def _failing_ensure_kline_store(**kwargs):
            raise ValueError(
                "Kline store validation failed for LABUSDT: [FAIL] Issues: "
                "row_count 12576 < minimum 15818"
            )

        monkeypatch.setattr(
            rs,
            "get_config",
            lambda: SimpleNamespace(base_dir=str(tmp)),
        )
        monkeypatch.setattr(
            rs,
            "load_parquet_range",
            lambda root, start_ms=None, end_ms=None: partial,
        )
        monkeypatch.setattr(rs, "ensure_kline_store", _failing_ensure_kline_store)

        df = asyncio.run(
            rs._ensure_and_load_history(
                "LABUSDT",
                interval="15m",
                start_ms=int(pd.Timestamp("2026-02-01T00:00:00Z").timestamp() * 1000),
                end_ms=int(pd.Timestamp("2026-04-01T00:00:00Z").timestamp() * 1000),
                min_years=0.25,
            )
        )

        assert not df.empty
        assert df["open_time"].min() == partial["open_time"].min()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ensure_and_load_history_returns_partial_frame_when_window_is_not_fully_covered(monkeypatch):
    rs = _load_retrain_scanner()
    tmp = _workspace_tmp()
    try:
        partial = _history_df(
            120,
            "15min",
            base_price=100.0,
            step=0.05,
            start="2026-03-01T00:00:00Z",
        )

        async def _fake_ensure_kline_store(**kwargs):
            return {}

        monkeypatch.setattr(
            rs,
            "get_config",
            lambda: SimpleNamespace(base_dir=str(tmp)),
        )
        monkeypatch.setattr(
            rs,
            "load_parquet_range",
            lambda root, start_ms=None, end_ms=None: partial,
        )
        monkeypatch.setattr(rs, "ensure_kline_store", _fake_ensure_kline_store)

        df = asyncio.run(
            rs._ensure_and_load_history(
                "BTCUSDT",
                interval="15m",
                start_ms=int(pd.Timestamp("2026-02-01T00:00:00Z").timestamp() * 1000),
                end_ms=int(pd.Timestamp("2026-04-01T00:00:00Z").timestamp() * 1000),
                min_years=0.25,
            )
        )

        assert not df.empty
        assert df["open_time"].min() == partial["open_time"].min()
        assert df["close_time"].max() == partial["close_time"].max()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
