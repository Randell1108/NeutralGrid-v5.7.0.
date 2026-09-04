from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid
import warnings

import numpy as np
import pandas as pd
import pytest


_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "backfill_training_features.py"
_SPEC = importlib.util.spec_from_file_location("backfill_training_features_v20260312", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
TrainingDataBackfiller = _MODULE.TrainingDataBackfiller


def _workspace_tmp_dir() -> Path:
    path = Path.cwd() / ".pytest_local_tmp" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_backfill_cli_accepts_candidate_scan_feature_cutoff() -> None:
    args = _MODULE._parse_args(
        [
            "--feature-cutoff-source",
            "candidate_id_scan_time",
            "--replay-scope",
            "hmm_lineage_only",
        ]
    )

    assert args.feature_cutoff_source == "candidate_id_scan_time"
    assert args.replay_scope == "hmm_lineage_only"


def test_backfill_require_fresh_output_refuses_existing_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = _workspace_tmp_dir()
    try:
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        pd.DataFrame(
            [{"symbol": "BTCUSDT", "start_time_utc": "2026-07-01T00:00:00Z"}]
        ).to_csv(input_path, index=False)
        output_path.write_text("existing\n", encoding="utf-8")
        backfiller = TrainingDataBackfiller(require_fresh_output=True)

        async def _network_must_not_start() -> None:
            raise AssertionError("network client initialized before output collision check")

        monkeypatch.setattr(backfiller, "init_client", _network_must_not_start)

        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            asyncio.run(backfiller.backfill_all(str(input_path), str(output_path)))

        assert output_path.read_text(encoding="utf-8") == "existing\n"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_backfill_bounded_concurrency_preserves_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = _workspace_tmp_dir()
    try:
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        rows = [
            {
                "candidate_id": f"cid_{idx}",
                "symbol": "BTCUSDT",
                "start_time_utc": f"2026-07-01T0{idx}:00:00Z",
            }
            for idx in range(4)
        ]
        pd.DataFrame(rows).to_csv(input_path, index=False)
        backfiller = TrainingDataBackfiller(max_concurrency=2)
        active = 0
        maximum_active = 0

        async def _noop() -> None:
            return None

        async def _fake_backfill_single_bot(row: pd.Series) -> dict[str, float]:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return {"range_prob": float(str(row["candidate_id"]).split("_")[-1])}

        monkeypatch.setattr(backfiller, "init_client", _noop)
        monkeypatch.setattr(backfiller, "close_client", _noop)
        monkeypatch.setattr(
            backfiller, "backfill_single_bot", _fake_backfill_single_bot
        )

        asyncio.run(backfiller.backfill_all(str(input_path), str(output_path)))

        result = pd.read_csv(output_path)
        assert maximum_active == 2
        assert result["candidate_id"].tolist() == [f"cid_{idx}" for idx in range(4)]
        assert result["range_prob"].tolist() == [0.0, 1.0, 2.0, 3.0]
        assert not list(tmp_path.glob(".*.tmp*"))
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_backfiller_derives_profit_from_grid_geometry_not_spacing_proxy() -> None:
    backfiller = TrainingDataBackfiller()
    row = pd.Series(
        {
            "price_range_low": 100.0,
            "price_range_high": 110.0,
            "grids_count": 5,
            "grid_spacing_pct": 0.40,
            # GRIDFIX-001 / GRID_SYNCH §2.1 — backfiller wrapper now requires
            # explicit `mode` (no default). Geometric is the live-deployment
            # default per Step 4.
            "mode": "geometric",
        }
    )

    derived = backfiller._derive_profit_per_grid_pct(row)

    assert derived is not None
    assert derived > 0
    assert derived != pytest.approx(0.40)


def test_backfiller_preserves_existing_values_by_symbol_and_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = _workspace_tmp_dir()
    try:
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"

        pd.DataFrame(
            [
                {"symbol": "ETHUSDT", "start_time_utc": "2026-03-02T00:00:00+00:00"},
                {"symbol": "BTCUSDT", "start_time_utc": "2026-03-01T00:00:00+00:00"},
            ]
        ).to_csv(input_path, index=False)

        pd.DataFrame(
            [
                {
                    "symbol": "BTCUSDT",
                    "start_time_utc": "2026-03-01T00:00:00+00:00",
                    "range_prob": 0.81,
                    "hmm_artifact_version": "rolling_180d_20260312_192602",
                },
                {
                    "symbol": "ETHUSDT",
                    "start_time_utc": "2026-03-02T00:00:00+00:00",
                    "range_prob": 0.22,
                    "hmm_artifact_version": "rolling_180d_20260312_192602",
                },
            ]
        ).to_csv(output_path, index=False)

        backfiller = TrainingDataBackfiller()

        async def _noop(*args, **kwargs):
            return None

        async def _fake_backfill_single_bot(row):
            return {}

        monkeypatch.setattr(backfiller, "init_client", _noop)
        monkeypatch.setattr(backfiller, "close_client", _noop)
        monkeypatch.setattr(backfiller, "backfill_single_bot", _fake_backfill_single_bot)

        asyncio.run(backfiller.backfill_all(str(input_path), str(output_path)))

        result = pd.read_csv(output_path)
        btc = result.loc[result["symbol"] == "BTCUSDT"].iloc[0]
        eth = result.loc[result["symbol"] == "ETHUSDT"].iloc[0]

        assert btc["range_prob"] == pytest.approx(0.81)
        assert eth["range_prob"] == pytest.approx(0.22)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_backfiller_invalidates_stale_lineage_when_default_artifact_version_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UTILFIX-01: --default-artifact-version is authoritative.

    When the existing-output preserved hmm_artifact_version differs from the
    explicit default, HMM_DERIVED_COLUMNS and HMM_LINEAGE_COLUMNS must NOT be
    preserved for that row, so backfill_single_bot can re-attempt inference
    against the explicit default. Pre-fix, the merge silently propagated the
    stale version, causing per-row inference to use the OLD HMM regardless of
    the CLI flag.
    """
    tmp_path = _workspace_tmp_dir()
    try:
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"

        pd.DataFrame(
            [
                {"symbol": "BTCUSDT", "start_time_utc": "2026-03-01T00:00:00+00:00"},
                {"symbol": "ETHUSDT", "start_time_utc": "2026-03-02T00:00:00+00:00"},
            ]
        ).to_csv(input_path, index=False)

        # Existing output: BTC was previously backfilled with the OLD HMM,
        # ETH with the matching (current) HMM.
        old_hmm = "rolling_180d_20260312_192602"
        new_hmm = "rolling_180d_20260503_171910"
        pd.DataFrame(
            [
                {
                    "symbol": "BTCUSDT",
                    "start_time_utc": "2026-03-01T00:00:00+00:00",
                    "range_prob": 0.81,
                    "trend_prob": 0.10,
                    "hmm_artifact_version": old_hmm,
                    "hmm_feature_source": "pinned_artifact_replay",
                },
                {
                    "symbol": "ETHUSDT",
                    "start_time_utc": "2026-03-02T00:00:00+00:00",
                    "range_prob": 0.22,
                    "trend_prob": 0.55,
                    "hmm_artifact_version": new_hmm,
                    "hmm_feature_source": "pinned_artifact_replay",
                },
            ]
        ).to_csv(output_path, index=False)

        backfiller = TrainingDataBackfiller(default_artifact_version=new_hmm)

        rows_passed_to_inference: list[dict] = []

        async def _noop(*_args, **_kwargs):
            return None

        async def _capture_backfill_single_bot(row):
            # Capture the row state at the moment per-row inference would run.
            # The fix guarantees that for stale rows, hmm_artifact_version /
            # range_prob / trend_prob are NOT preserved from the existing
            # output (so they show as NaN here), forcing re-inference.
            rows_passed_to_inference.append(
                {
                    "symbol": row.get("symbol"),
                    "range_prob": row.get("range_prob"),
                    "trend_prob": row.get("trend_prob"),
                    "hmm_artifact_version": row.get("hmm_artifact_version"),
                }
            )
            return {}

        monkeypatch.setattr(backfiller, "init_client", _noop)
        monkeypatch.setattr(backfiller, "close_client", _noop)
        monkeypatch.setattr(
            backfiller, "backfill_single_bot", _capture_backfill_single_bot
        )

        asyncio.run(backfiller.backfill_all(str(input_path), str(output_path)))

        captured = {row["symbol"]: row for row in rows_passed_to_inference}
        # BTC is stale: must NOT carry the preserved old HMM into inference.
        assert pd.isna(captured["BTCUSDT"]["range_prob"]), (
            "stale-lineage row should have range_prob=NaN at inference time"
        )
        assert pd.isna(captured["BTCUSDT"]["trend_prob"]), (
            "stale-lineage row should have trend_prob=NaN at inference time"
        )
        old_label = captured["BTCUSDT"]["hmm_artifact_version"]
        assert old_label is None or pd.isna(old_label) or old_label == "", (
            f"stale-lineage row leaked old HMM version into row.hmm_artifact_version: "
            f"{old_label!r}"
        )
        # ETH matches the explicit default: preservation is allowed.
        assert captured["ETHUSDT"]["range_prob"] == pytest.approx(0.22), (
            "matching-lineage row should preserve range_prob"
        )
        assert captured["ETHUSDT"]["hmm_artifact_version"] == new_hmm, (
            "matching-lineage row should preserve hmm_artifact_version label"
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_hmm_only_backfill_fetches_only_the_causal_15m_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backfiller = TrainingDataBackfiller(
        default_artifact_version="rolling_180d_test",
        hmm_only=True,
    )
    backfiller.client = object()
    requested: list[tuple[str, int, datetime]] = []

    async def _fake_klines(
        symbol: str,
        interval: str,
        start_time: datetime,
        lookback_bars: int,
    ) -> pd.DataFrame:
        requested.append((interval, lookback_bars, start_time))
        return pd.DataFrame({"close": np.linspace(1.0, 2.0, lookback_bars)})

    monkeypatch.setattr(backfiller, "fetch_historical_klines", _fake_klines)
    monkeypatch.setattr(
        backfiller,
        "_get_hmm_predictor",
        lambda _version: SimpleNamespace(
            predict=lambda _bars: SimpleNamespace(
                range_prob_agg=0.70,
                trend_prob_agg=0.20,
                persistence_prob=0.80,
                trained_at_utc="2026-08-19T01:44:24+00:00",
                artifact_version="rolling_180d_test",
                pipeline_version="hmm_pipeline_test",
                calibration_provenance={"status": "identity"},
            )
        ),
    )

    result = asyncio.run(
        backfiller.backfill_single_bot(
            pd.Series(
                {
                    "symbol": "BTCUSDT",
                    "candidate_id": "BTCUSDT_20260820_000000_deadbeef",
                    "start_time_utc": "2026-08-20T01:00:00+00:00",
                    "funding_rate": 0.0001,
                    "profit_per_grid_pct": 0.5,
                    "num_grids": 10,
                    "survival_prob": 0.8,
                    "range_size_pct": 8.0,
                }
            )
        )
    )

    assert requested == [("15m", 800, datetime(2026, 8, 20, tzinfo=timezone.utc))]
    assert result["hmm_artifact_version"] == "rolling_180d_test"
    assert result["hmm_feature_source"] == "pinned_artifact_replay"
    assert result["hmm_replay_scope"] == "hmm_lineage_only"
    assert result["feature_cutoff_utc"] == "2026-08-20T00:00:00+00:00"
    assert np.isfinite(result["ev_score"])


def test_explicit_default_hmm_overrides_stale_input_row_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UTILFIX-01 also applies when the stale version is in the input CSV."""
    old_hmm = "rolling_180d_20260312_192602"
    new_hmm = "rolling_180d_20260503_171910"
    backfiller = TrainingDataBackfiller(default_artifact_version=new_hmm)
    backfiller.client = object()  # The network methods below are fully stubbed.

    async def _empty_klines(*_args, **_kwargs) -> pd.DataFrame:
        return pd.DataFrame()

    async def _no_funding(*_args, **_kwargs):
        return None

    predictor_versions: list[str | None] = []

    def _capture_predictor(version: str | None):
        predictor_versions.append(version)
        return None

    monkeypatch.setattr(backfiller, "fetch_historical_klines", _empty_klines)
    monkeypatch.setattr(backfiller, "fetch_historical_funding_rate", _no_funding)
    monkeypatch.setattr(backfiller, "_get_hmm_predictor", _capture_predictor)
    monkeypatch.setattr(
        _MODULE,
        "compute_features",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    result = asyncio.run(
        backfiller.backfill_single_bot(
            pd.Series(
                {
                    "symbol": "BTCUSDT",
                    "start_time_utc": "2026-03-01T00:00:00+00:00",
                    "hmm_artifact_version": old_hmm,
                }
            )
        )
    )

    assert predictor_versions == [new_hmm]
    assert result["hmm_artifact_version"] == new_hmm
    assert result["hmm_feature_source"] == "artifact_unavailable"


def test_stale_input_hmm_dependent_values_are_invalidated_before_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed replay cannot retain stale probabilities or HMM-derived EV."""
    tmp_path = _workspace_tmp_dir()
    try:
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        old_hmm = "rolling_180d_20260312_192602"
        new_hmm = "rolling_180d_20260503_171910"
        pd.DataFrame(
            [
                {
                    "symbol": "BTCUSDT",
                    "start_time_utc": "2026-03-01T00:00:00+00:00",
                    "hmm_artifact_version": old_hmm,
                    "range_prob": 0.8,
                    "trend_prob": 0.1,
                    "persistence_prob": 0.7,
                    "ev_score": 1.2,
                    "regime_conf": 0.7,
                }
            ]
        ).to_csv(input_path, index=False)
        backfiller = TrainingDataBackfiller(default_artifact_version=new_hmm)
        captured: list[pd.Series] = []

        async def _noop(*_args, **_kwargs):
            return None

        async def _capture(row: pd.Series):
            captured.append(row.copy())
            return {}

        monkeypatch.setattr(backfiller, "init_client", _noop)
        monkeypatch.setattr(backfiller, "close_client", _noop)
        monkeypatch.setattr(backfiller, "backfill_single_bot", _capture)

        asyncio.run(backfiller.backfill_all(str(input_path), str(output_path)))

        assert len(captured) == 1
        for column in (
            "hmm_artifact_version",
            "range_prob",
            "trend_prob",
            "persistence_prob",
            "ev_score",
            "regime_conf",
        ):
            value = captured[0].at[column]
            assert bool(pd.isna(value)), column
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_skip_if_fresh_skips_matching_lineage_with_finite_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--skip-if-fresh: post-merge predicate.

    Predicate skips backfill_single_bot for any row whose preserved
    hmm_artifact_version matches --default-artifact-version AND whose
    range_prob / trend_prob / persistence_prob are all finite. Stale-lineage
    rows still re-infer (the merge invalidated them); NaN-feature rows still
    re-infer (the finiteness check fails).
    """
    tmp_path = _workspace_tmp_dir()
    try:
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"

        pd.DataFrame(
            [
                {"symbol": "AAAUSDT", "start_time_utc": "2026-03-01T00:00:00+00:00"},
                {"symbol": "BBBUSDT", "start_time_utc": "2026-03-02T00:00:00+00:00"},
                {"symbol": "CCCUSDT", "start_time_utc": "2026-03-03T00:00:00+00:00"},
            ]
        ).to_csv(input_path, index=False)

        old_hmm = "rolling_180d_20260312_192602"
        new_hmm = "rolling_180d_20260503_171910"
        pd.DataFrame(
            [
                # AAA: fresh lineage + finite HMM features  -> SKIP
                {
                    "symbol": "AAAUSDT",
                    "start_time_utc": "2026-03-01T00:00:00+00:00",
                    "range_prob": 0.40,
                    "trend_prob": 0.30,
                    "persistence_prob": 0.70,
                    "hmm_artifact_version": new_hmm,
                    "hmm_feature_source": "pinned_artifact_replay",
                },
                # BBB: stale lineage -> merge invalidates -> RE-INFER
                {
                    "symbol": "BBBUSDT",
                    "start_time_utc": "2026-03-02T00:00:00+00:00",
                    "range_prob": 0.55,
                    "trend_prob": 0.20,
                    "persistence_prob": 0.65,
                    "hmm_artifact_version": old_hmm,
                    "hmm_feature_source": "pinned_artifact_replay",
                },
                # CCC: fresh lineage but range_prob NaN -> finiteness fails -> RE-INFER
                {
                    "symbol": "CCCUSDT",
                    "start_time_utc": "2026-03-03T00:00:00+00:00",
                    "range_prob": float("nan"),
                    "trend_prob": 0.40,
                    "persistence_prob": 0.60,
                    "hmm_artifact_version": new_hmm,
                    "hmm_feature_source": "pinned_artifact_replay",
                },
            ]
        ).to_csv(output_path, index=False)

        backfiller = TrainingDataBackfiller(
            default_artifact_version=new_hmm,
            skip_if_fresh=True,
        )

        symbols_inferred: list[str] = []

        async def _noop(*_args, **_kwargs):
            return None

        async def _capture_backfill_single_bot(row):
            symbols_inferred.append(row.get("symbol"))
            return {}

        monkeypatch.setattr(backfiller, "init_client", _noop)
        monkeypatch.setattr(backfiller, "close_client", _noop)
        monkeypatch.setattr(
            backfiller, "backfill_single_bot", _capture_backfill_single_bot
        )

        asyncio.run(backfiller.backfill_all(str(input_path), str(output_path)))

        # AAA must NOT have been inferred; BBB and CCC must have been.
        assert "AAAUSDT" not in symbols_inferred, (
            f"fresh-lineage row was re-inferenced unnecessarily: {symbols_inferred}"
        )
        assert "BBBUSDT" in symbols_inferred, (
            "stale-lineage row should be re-inferenced"
        )
        assert "CCCUSDT" in symbols_inferred, (
            "row with NaN range_prob should be re-inferenced"
        )

        # Output row AAA retains its preserved HMM features.
        result = pd.read_csv(output_path)
        aaa = result.loc[result["symbol"] == "AAAUSDT"].iloc[0]
        assert aaa["range_prob"] == pytest.approx(0.40)
        assert aaa["trend_prob"] == pytest.approx(0.30)
        assert aaa["persistence_prob"] == pytest.approx(0.70)
        assert aaa["hmm_artifact_version"] == new_hmm
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_backfiller_writes_hmm_lineage_columns_as_text(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = _workspace_tmp_dir()
    try:
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"

        pd.DataFrame(
            [
                {"symbol": "BTCUSDT", "start_time_utc": "2026-03-01T00:00:00+00:00"},
            ]
        ).to_csv(input_path, index=False)

        backfiller = TrainingDataBackfiller()

        async def _noop(*args, **kwargs):
            return None

        async def _fake_backfill_single_bot(row):
            return {
                "range_prob": 0.81,
                "ev_contract_fingerprint": "f" * 64,
                "hmm_artifact_version": "rolling_180d_20260401_025936",
                "hmm_feature_semantics_version": "hmm_features_v1",
                "hmm_feature_source": "pinned_artifact_replay",
            }

        monkeypatch.setattr(backfiller, "init_client", _noop)
        monkeypatch.setattr(backfiller, "close_client", _noop)
        monkeypatch.setattr(backfiller, "backfill_single_bot", _fake_backfill_single_bot)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message="Setting an item of incompatible dtype.*",
                category=FutureWarning,
            )
            asyncio.run(backfiller.backfill_all(str(input_path), str(output_path)))

        result = pd.read_csv(output_path)
        row = result.iloc[0]

        assert row["range_prob"] == pytest.approx(0.81)
        assert row["ev_contract_fingerprint"] == "f" * 64
        assert row["hmm_artifact_version"] == "rolling_180d_20260401_025936"
        assert row["hmm_feature_semantics_version"] == "hmm_features_v1"
        assert row["hmm_feature_source"] == "pinned_artifact_replay"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_backfiller_uses_candidate_id_when_available_to_avoid_cross_row_preserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = _workspace_tmp_dir()
    try:
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"

        pd.DataFrame(
            [
                {
                    "symbol": "BTCUSDT",
                    "candidate_id": "cid_a",
                    "start_time_utc": "2026-03-01T00:00:00+00:00",
                },
                {
                    "symbol": "BTCUSDT",
                    "candidate_id": "cid_b",
                    "start_time_utc": "2026-03-01T00:00:00+00:00",
                },
            ]
        ).to_csv(input_path, index=False)

        pd.DataFrame(
            [
                {
                    "symbol": "BTCUSDT",
                    "candidate_id": "cid_a",
                    "start_time_utc": "2026-03-01T00:00:00+00:00",
                    "range_prob": 0.81,
                    "hmm_artifact_version": "rolling_180d_20260312_192602",
                },
                {
                    "symbol": "BTCUSDT",
                    "candidate_id": "cid_b",
                    "start_time_utc": "2026-03-01T00:00:00+00:00",
                    "range_prob": 0.22,
                    "hmm_artifact_version": "rolling_180d_20260312_192602",
                },
            ]
        ).to_csv(output_path, index=False)

        backfiller = TrainingDataBackfiller()

        async def _noop(*args, **kwargs):
            return None

        async def _fake_backfill_single_bot(row):
            return {}

        monkeypatch.setattr(backfiller, "init_client", _noop)
        monkeypatch.setattr(backfiller, "close_client", _noop)
        monkeypatch.setattr(backfiller, "backfill_single_bot", _fake_backfill_single_bot)

        asyncio.run(backfiller.backfill_all(str(input_path), str(output_path)))

        result = pd.read_csv(output_path).sort_values("candidate_id").reset_index(drop=True)

        assert result.loc[0, "range_prob"] == pytest.approx(0.81)
        assert result.loc[1, "range_prob"] == pytest.approx(0.22)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_backfiller_treats_nan_artifact_version_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    backfiller = TrainingDataBackfiller()
    backfiller.client = object()

    async def _fake_fetch_klines(symbol: str, interval: str, start_time, lookback_bars: int):
        return pd.DataFrame({"close": [1.0, 1.1, 1.2]})

    async def _fake_fetch_funding(symbol: str, start_time):
        return 0.0

    seen_versions: list[str | None] = []

    def _fake_get_hmm_predictor(version: str | None):
        seen_versions.append(version)
        return None

    monkeypatch.setattr(backfiller, "fetch_historical_klines", _fake_fetch_klines)
    monkeypatch.setattr(backfiller, "fetch_historical_funding_rate", _fake_fetch_funding)
    monkeypatch.setattr(backfiller, "_get_hmm_predictor", _fake_get_hmm_predictor)
    monkeypatch.setattr(
        _MODULE,
        "compute_features",
        lambda *args, **kwargs: SimpleNamespace(
            persistence_prob=None,
            adx_1h=None,
            adx_15m=None,
            adx_5m=None,
            rsi_15m=None,
            ema_slope_1h=None,
            ema_crosses_5m=None,
            vwap_crosses_5m=None,
            bb_width=None,
            atr_pct_15m=None,
            quote_volume_24h=None,
            regime_conf=None,
        ),
    )

    row = pd.Series(
        {
            "symbol": "BTCUSDT",
            "start_time_utc": "2026-03-01T00:00:00+00:00",
            "hmm_artifact_version": float("nan"),
        }
    )

    result = asyncio.run(backfiller.backfill_single_bot(row))

    assert seen_versions == [""]
    assert pd.isna(result["hmm_artifact_version"])
    assert result["hmm_feature_source"] == "missing_artifact_version"


def test_backfiller_attempts_hmm_inference_when_exchange_returns_fewer_than_requested_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay follows predictor validity, not the requested 800-bar fetch size."""
    backfiller = TrainingDataBackfiller(default_artifact_version="hmm_v1")
    backfiller.client = object()
    observed_lengths: list[int] = []

    async def _fake_fetch_klines(
        symbol: str, interval: str, start_time, lookback_bars: int
    ) -> pd.DataFrame:
        size = 314 if interval == "15m" else max(lookback_bars, 100)
        return pd.DataFrame(
            {
                "close": np.linspace(100.0, 101.0, size),
                "high": np.linspace(100.5, 101.5, size),
                "low": np.linspace(99.5, 100.5, size),
                "volume": np.full(size, 10.0),
                "quote_volume": np.full(size, 1000.0),
            }
        )

    async def _fake_fetch_funding(symbol: str, start_time) -> float:
        return 0.0

    class _Predictor:
        def predict(self, frame: pd.DataFrame) -> SimpleNamespace:
            observed_lengths.append(len(frame))
            return SimpleNamespace(
                range_prob_agg=0.75,
                trend_prob_agg=0.25,
                persistence_prob=0.60,
                trained_at_utc="2026-08-15T23:14:45.834213+00:00",
                artifact_version="hmm_v1",
                pipeline_version="6.5.8",
                calibration_provenance={"status": "ok"},
            )

    monkeypatch.setattr(backfiller, "fetch_historical_klines", _fake_fetch_klines)
    monkeypatch.setattr(backfiller, "fetch_historical_funding_rate", _fake_fetch_funding)
    monkeypatch.setattr(backfiller, "_get_hmm_predictor", lambda _version: _Predictor())
    monkeypatch.setattr(
        _MODULE,
        "compute_features",
        lambda *args, **kwargs: SimpleNamespace(
            persistence_prob=None,
            adx_1h=None,
            adx_15m=None,
            adx_5m=None,
            rsi_15m=None,
            ema_slope_1h=None,
            ema_crosses_5m=None,
            vwap_crosses_5m=None,
            bb_width=None,
            atr_pct_15m=None,
            quote_volume_24h=None,
            regime_conf=None,
        ),
    )

    result = asyncio.run(
        backfiller.backfill_single_bot(
            pd.Series(
                {
                    "symbol": "DATAIPUSDT",
                    "start_time_utc": "2026-07-06T00:00:00+00:00",
                }
            )
        )
    )

    assert observed_lengths == [314]
    assert result["hmm_artifact_version"] == "hmm_v1"
    assert result["hmm_feature_source"] == "pinned_artifact_replay"


def test_backfiller_uses_candidate_scan_time_for_scanner_origin_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backfiller = TrainingDataBackfiller(
        default_artifact_version="hmm_v1",
        feature_cutoff_source="candidate_id_scan_time",
    )
    backfiller.client = object()
    observed_cutoffs: list[datetime] = []

    async def _fake_fetch_klines(
        symbol: str, interval: str, cutoff: datetime, lookback_bars: int
    ) -> pd.DataFrame:
        observed_cutoffs.append(cutoff)
        size = max(lookback_bars, 100)
        return pd.DataFrame(
            {
                "close": np.linspace(100.0, 101.0, size),
                "high": np.linspace(100.5, 101.5, size),
                "low": np.linspace(99.5, 100.5, size),
                "volume": np.full(size, 10.0),
                "quote_volume": np.full(size, 1000.0),
            }
        )

    async def _fake_fetch_funding(symbol: str, cutoff: datetime) -> float:
        observed_cutoffs.append(cutoff)
        return 0.0

    class _Predictor:
        def predict(self, frame: pd.DataFrame) -> SimpleNamespace:
            return SimpleNamespace(
                range_prob_agg=0.75,
                trend_prob_agg=0.25,
                persistence_prob=0.60,
                trained_at_utc="2026-08-15T23:14:45.834213+00:00",
                artifact_version="hmm_v1",
                pipeline_version="6.5.8",
                calibration_provenance={"status": "ok"},
            )

    monkeypatch.setattr(backfiller, "fetch_historical_klines", _fake_fetch_klines)
    monkeypatch.setattr(backfiller, "fetch_historical_funding_rate", _fake_fetch_funding)
    monkeypatch.setattr(backfiller, "_get_hmm_predictor", lambda _version: _Predictor())
    monkeypatch.setattr(
        _MODULE,
        "compute_features",
        lambda *args, **kwargs: SimpleNamespace(
            persistence_prob=None,
            adx_1h=None,
            adx_15m=None,
            adx_5m=None,
            rsi_15m=None,
            ema_slope_1h=None,
            ema_crosses_5m=None,
            vwap_crosses_5m=None,
            bb_width=None,
            atr_pct_15m=None,
            quote_volume_24h=None,
            regime_conf=None,
        ),
    )

    result = asyncio.run(
        backfiller.backfill_single_bot(
            pd.Series(
                {
                    "candidate_id": "BTCUSDT_20260608_120000_deadbeef",
                    "symbol": "BTCUSDT",
                    "start_time_utc": "2026-06-08T12:45:00+00:00",
                }
            )
        )
    )

    expected = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    assert observed_cutoffs == [expected] * 5
    assert result["feature_cutoff_utc"] == expected.isoformat()


def test_backfiller_rejects_invalid_candidate_scan_cutoff_before_network() -> None:
    backfiller = TrainingDataBackfiller(
        feature_cutoff_source="candidate_id_scan_time"
    )

    result = asyncio.run(
        backfiller.backfill_single_bot(
            pd.Series(
                {
                    "candidate_id": "not-a-canonical-id",
                    "symbol": "BTCUSDT",
                    "start_time_utc": "2026-06-08T12:45:00+00:00",
                }
            )
        )
    )

    assert result["hmm_feature_source"] == "invalid_candidate_id_scan_time"
    assert pd.isna(result["hmm_artifact_version"])


def test_hmm_lineage_only_accepts_snapshot_candidate_id_without_start_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deploy snapshot's candidate scan time is its causal HMM boundary."""
    backfiller = TrainingDataBackfiller(
        default_artifact_version="hmm_v1",
        feature_cutoff_source="candidate_id_scan_time",
        replay_scope="hmm_lineage_only",
    )
    backfiller.client = object()
    observed_cutoffs: list[datetime] = []

    async def _fake_fetch_klines(
        symbol: str, interval: str, cutoff: datetime, lookback_bars: int
    ) -> pd.DataFrame:
        observed_cutoffs.append(cutoff)
        return pd.DataFrame({"close": np.linspace(100.0, 101.0, lookback_bars)})

    class _Predictor:
        def predict(self, frame: pd.DataFrame) -> SimpleNamespace:
            return SimpleNamespace(
                range_prob_agg=0.75,
                trend_prob_agg=0.25,
                persistence_prob=0.60,
                trained_at_utc="2026-08-15T23:14:45.834213+00:00",
                artifact_version="hmm_v1",
                pipeline_version="6.5.8",
                calibration_provenance={"status": "ok"},
            )

    monkeypatch.setattr(backfiller, "fetch_historical_klines", _fake_fetch_klines)
    monkeypatch.setattr(backfiller, "_get_hmm_predictor", lambda _version: _Predictor())

    result = asyncio.run(
        backfiller.backfill_single_bot(
            pd.Series(
                {
                    "candidate_id": "BTCUSDT_20260825_114628_deadbeef",
                    "symbol": "BTCUSDT",
                    "funding_rate": 0.0001,
                }
            )
        )
    )

    expected = datetime(2026, 8, 25, 11, 46, 28, tzinfo=timezone.utc)
    assert observed_cutoffs == [expected]
    assert result["hmm_feature_source"] == "pinned_artifact_replay"
    assert result["hmm_feature_cutoff_utc"] == expected.isoformat()


def test_candidate_scan_mode_uses_event_start_for_legacy_row_without_candidate_id() -> None:
    backfiller = TrainingDataBackfiller(
        feature_cutoff_source="candidate_id_scan_time"
    )
    event_start = datetime(2026, 6, 8, 12, 45, tzinfo=timezone.utc)

    cutoff = backfiller._resolve_feature_cutoff(
        pd.Series({"candidate_id": None}),
        event_start,
    )

    assert cutoff == event_start


def test_hmm_lineage_only_scope_fetches_only_15m_and_preserves_snapshot_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backfiller = TrainingDataBackfiller(
        default_artifact_version="hmm_v1",
        feature_cutoff_source="candidate_id_scan_time",
        replay_scope="hmm_lineage_only",
    )
    backfiller.client = object()
    observed_intervals: list[str] = []

    async def _fake_fetch_klines(
        symbol: str, interval: str, cutoff: datetime, lookback_bars: int
    ) -> pd.DataFrame:
        observed_intervals.append(interval)
        size = 314
        return pd.DataFrame(
            {
                "close": np.linspace(100.0, 101.0, size),
                "high": np.linspace(100.5, 101.5, size),
                "low": np.linspace(99.5, 100.5, size),
                "volume": np.full(size, 10.0),
                "quote_volume": np.full(size, 1000.0),
            }
        )

    async def _funding_must_not_be_refetched(*args, **kwargs) -> float:
        raise AssertionError("hmm_lineage_only refetched funding")

    class _Predictor:
        def predict(self, frame: pd.DataFrame) -> SimpleNamespace:
            return SimpleNamespace(
                range_prob_agg=0.75,
                trend_prob_agg=0.25,
                persistence_prob=0.60,
                trained_at_utc="2026-08-15T23:14:45.834213+00:00",
                artifact_version="hmm_v1",
                pipeline_version="6.5.8",
                calibration_provenance={"status": "ok"},
            )

    monkeypatch.setattr(backfiller, "fetch_historical_klines", _fake_fetch_klines)
    monkeypatch.setattr(
        backfiller, "fetch_historical_funding_rate", _funding_must_not_be_refetched
    )
    monkeypatch.setattr(backfiller, "_get_hmm_predictor", lambda _version: _Predictor())
    monkeypatch.setattr(
        _MODULE,
        "compute_features",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("hmm_lineage_only recomputed independent features")
        ),
    )

    result = asyncio.run(
        backfiller.backfill_single_bot(
            pd.Series(
                {
                    "candidate_id": "DATAIPUSDT_20260706_122139_878af32f",
                    "symbol": "DATAIPUSDT",
                    "start_time_utc": "2026-07-06T13:15:09+00:00",
                    "funding_rate": 0.0001,
                }
            )
        )
    )

    assert observed_intervals == ["15m"]
    assert result["hmm_artifact_version"] == "hmm_v1"
    assert result["hmm_replay_scope"] == "hmm_lineage_only"
    assert result["funding_rate"] == pytest.approx(0.0001)
    assert pd.isna(result["adx_1h"])


def test_historical_fetch_excludes_interval_not_closed_at_decision_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backfiller = TrainingDataBackfiller()
    decision = datetime(2026, 3, 1, 1, 2, tzinfo=timezone.utc)
    starts = pd.date_range("2026-03-01T00:00:00Z", periods=5, freq="15min")

    def _raw_kline(open_time: pd.Timestamp) -> list[object]:
        open_ms = int(open_time.timestamp() * 1000)
        return [
            open_ms,
            "100",
            "101",
            "99",
            "100.5",
            "10",
            open_ms + 15 * 60 * 1000 - 1,
            "1000",
            1,
            "5",
            "500",
            "0",
        ]

    class _Client:
        async def get_klines(self, **kwargs):
            return [_raw_kline(ts) for ts in starts]

    backfiller.client = _Client()

    async def _no_vision(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(backfiller, "_fetch_binance_vision_klines", _no_vision)

    result = asyncio.run(
        backfiller.fetch_historical_klines(
            "BTCUSDT", "15m", decision, lookback_bars=4
        )
    )

    assert len(result) == 4
    assert result["open_time"].max() == pd.Timestamp("2026-03-01T00:45:00Z")
    assert pd.Timestamp("2026-03-01T01:00:00Z") not in set(result["open_time"])


def test_historical_fetch_uses_vision_fallback_for_short_15m_rest_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backfiller = TrainingDataBackfiller()
    decision = datetime(2026, 3, 10, tzinfo=timezone.utc)

    class _Client:
        async def get_klines(self, **kwargs):
            return []

    backfiller.client = _Client()
    vision_calls: list[tuple[str, str, int]] = []

    async def _vision(symbol: str, interval: str, start_time, lookback_bars: int):
        vision_calls.append((symbol, interval, lookback_bars))
        opens = pd.date_range(end=pd.Timestamp(decision) - pd.Timedelta(minutes=15), periods=800, freq="15min")
        return pd.DataFrame(
            {
                "open_time": opens,
                "close_time": opens + pd.Timedelta(minutes=15) - pd.Timedelta(milliseconds=1),
                "open": np.full(800, 100.0),
                "high": np.full(800, 101.0),
                "low": np.full(800, 99.0),
                "close": np.full(800, 100.5),
                "volume": np.full(800, 10.0),
                "quote_volume": np.full(800, 1000.0),
            }
        )

    monkeypatch.setattr(backfiller, "_fetch_binance_vision_klines", _vision)

    result = asyncio.run(
        backfiller.fetch_historical_klines(
            "NIGHTUSDT", "15m", decision, lookback_bars=800
        )
    )

    assert vision_calls == [("NIGHTUSDT", "15m", 800)]
    assert len(result) == 800
    assert result["open_time"].max() < pd.Timestamp(decision)


def test_historical_fetch_contains_rest_failure_and_uses_vision_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backfiller = TrainingDataBackfiller()
    decision = datetime(2026, 3, 10, tzinfo=timezone.utc)

    class _Client:
        async def get_klines(self, **kwargs):
            raise TimeoutError("simulated REST timeout")

    backfiller.client = _Client()
    opens = pd.date_range(
        end=pd.Timestamp(decision) - pd.Timedelta(minutes=15),
        periods=800,
        freq="15min",
    )

    async def _vision(*args, **kwargs):
        return pd.DataFrame(
            {
                "open_time": opens,
                "close_time": opens
                + pd.Timedelta(minutes=15)
                - pd.Timedelta(milliseconds=1),
                "open": np.full(800, 100.0),
                "high": np.full(800, 101.0),
                "low": np.full(800, 99.0),
                "close": np.full(800, 100.5),
                "volume": np.full(800, 10.0),
                "quote_volume": np.full(800, 1000.0),
            }
        )

    monkeypatch.setattr(backfiller, "_fetch_binance_vision_klines", _vision)

    result = asyncio.run(
        backfiller.fetch_historical_klines(
            "NIGHTUSDT", "15m", decision, lookback_bars=800
        )
    )

    assert len(result) == 800
    assert result["open_time"].max() < pd.Timestamp(decision)


def test_backfiller_derives_range_size_from_grid_midpoint() -> None:
    backfiller = TrainingDataBackfiller()
    row = pd.Series(
        {
            "price_range_low": 100.0,
            "price_range_high": 120.0,
        }
    )

    derived = backfiller._derive_range_size_pct(row, market_range_size_pct=None)

    assert derived is not None
    assert derived == pytest.approx((20.0 / 110.0) * 100.0)


def test_backfiller_derives_range_size_from_grid_alias_midpoint() -> None:
    backfiller = TrainingDataBackfiller()
    row = pd.Series(
        {
            "grid_lower": 95.0,
            "grid_upper": 105.0,
        }
    )

    derived = backfiller._derive_range_size_pct(row, market_range_size_pct=None)

    assert derived is not None
    assert derived == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("row_leverage", "expected_leverage"),
    [(3, 3), (None, 10)],
)
def test_backfiller_uses_governed_provisional_utility_and_ev_semantics(
    monkeypatch: pytest.MonkeyPatch,
    row_leverage: int | None,
    expected_leverage: int,
) -> None:
    backfiller = TrainingDataBackfiller()
    backfiller.client = object()

    async def _fake_fetch_klines(symbol: str, interval: str, start_time, lookback_bars: int):
        size = max(lookback_bars, 10)
        base = pd.DataFrame(
            {
                "close": np.linspace(100.0, 101.0, size),
                "high": np.linspace(100.5, 101.5, size),
                "low": np.linspace(99.5, 100.5, size),
                "volume": np.full(size, 10.0),
                "quote_volume": np.full(size, 1000.0),
            }
        )
        return base

    async def _fake_fetch_funding(symbol: str, start_time):
        return 0.001

    def _fake_get_hmm_predictor(version: str | None):
        class _Predictor:
            def predict(self, df):
                return SimpleNamespace(
                    range_prob_agg=0.81,
                    trend_prob_agg=0.19,
                    persistence_prob=0.55,
                    trained_at_utc="2026-04-21T00:00:00+00:00",
                    artifact_version="hmm_v1",
                    pipeline_version="pipeline_v1",
                    calibration_provenance={"status": "ok"},
                )

        return _Predictor()

    seen_utility_kwargs: list[dict[str, float]] = []
    seen_ranker_kwargs: list[dict[str, object]] = []

    def _fake_governed_provisional_utility(**kwargs):
        seen_utility_kwargs.append(kwargs)
        return SimpleNamespace(utility_score=1.23)

    def _fake_compute_score(**kwargs):
        seen_ranker_kwargs.append(kwargs)
        return SimpleNamespace(rank_score=12.34, ev_24h=56.78)

    monkeypatch.setattr(backfiller, "fetch_historical_klines", _fake_fetch_klines)
    monkeypatch.setattr(backfiller, "fetch_historical_funding_rate", _fake_fetch_funding)
    monkeypatch.setattr(backfiller, "_get_hmm_predictor", _fake_get_hmm_predictor)
    monkeypatch.setattr(_MODULE, "compute_governed_provisional_utility", _fake_governed_provisional_utility)
    monkeypatch.setattr(backfiller.pnl_ranker, "compute_score", _fake_compute_score)
    monkeypatch.setattr(
        _MODULE,
        "compute_features",
        lambda *args, **kwargs: SimpleNamespace(
            persistence_prob=None,
            adx_1h=1.1,
            adx_15m=2.2,
            adx_5m=3.3,
            rsi_15m=4.4,
            ema_slope_1h=5.5,
            ema_crosses_5m=6,
            vwap_crosses_5m=7,
            bb_width=0.8,
            atr_pct_15m=0.9,
            quote_volume_24h=10.0,
            regime_conf=0.7,
        ),
    )
    monkeypatch.setattr(
        backfiller.stochastic_checker,
        "analyze",
        lambda **kwargs: SimpleNamespace(
            survival_prob=0.66,
            hurst_exponent=0.58,
            ou_halflife=9.0,
        ),
    )

    row = pd.Series(
        {
            "symbol": "BTCUSDT",
            "start_time_utc": "2026-03-01T00:00:00+00:00",
            "hmm_artifact_version": "hmm_v1",
            "price_range_low": 100.0,
            "price_range_high": 120.0,
            "grids_count": 10,
            "leverage": row_leverage,
            "mode": "geometric",
        }
    )

    result = asyncio.run(backfiller.backfill_single_bot(row))

    assert result["utility_score"] == pytest.approx(1.23)
    assert seen_utility_kwargs == [{"range_prob": 0.81, "trend_prob": 0.19}]
    assert result["ev_score"] == pytest.approx(12.34)
    assert len(seen_ranker_kwargs) == 1
    assert seen_ranker_kwargs[0]["leverage"] == expected_leverage


def test_backfiller_preserves_existing_range_size_pct() -> None:
    backfiller = TrainingDataBackfiller()
    row = pd.Series(
        {
            "price_range_low": 100.0,
            "price_range_high": 120.0,
            "range_size_pct": 7.5,
        }
    )

    derived = backfiller._derive_range_size_pct(row, market_range_size_pct=3.2)

    assert derived == pytest.approx(7.5)


def test_backfill_writes_separate_flat_workbook_and_leaves_raw_input_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = _workspace_tmp_dir()
    try:
        input_path = tmp_path / "raw_input.xlsx"
        output_path = tmp_path / "flat_backfilled.xlsx"

        raw_general = pd.DataFrame(
            [
                {
                    "strategy_id": "1001",
                    "symbol": "DOGEUSDT",
                    "start_time_utc": "2026-05-14 13:00:00",
                    "end_time_utc": "2026-05-14 15:00:00",
                    "duration_hours": 2.0,
                    "mode": "geometric",
                }
            ]
        )
        raw_pnl = pd.DataFrame(
            [{"strategy_id": "1001", "symbol": "DOGEUSDT", "pnl_curve_points": 3}]
        )
        with pd.ExcelWriter(input_path, engine="openpyxl") as writer:
            raw_general.to_excel(writer, sheet_name="General", index=False)
            raw_pnl.to_excel(writer, sheet_name="PnL Curve Features", index=False)

        backfiller = TrainingDataBackfiller()

        async def _noop(*args, **kwargs):
            return None

        async def _fake_backfill_single_bot(row):
            return {
                "range_prob": 0.77,
                "trend_prob": 0.23,
                "hmm_artifact_version": "hmm_test",
            }

        monkeypatch.setattr(backfiller, "init_client", _noop)
        monkeypatch.setattr(backfiller, "close_client", _noop)
        monkeypatch.setattr(backfiller, "backfill_single_bot", _fake_backfill_single_bot)

        asyncio.run(backfiller.backfill_all(str(input_path), str(output_path)))

        raw_after = pd.read_excel(input_path, sheet_name="General")
        assert list(raw_after.columns) == list(raw_general.columns)
        assert "range_prob" not in raw_after.columns
        assert "hmm_artifact_version" not in raw_after.columns

        with pd.ExcelFile(output_path) as workbook:
            assert workbook.sheet_names == ["Sheet1"]
            derived = pd.read_excel(workbook, sheet_name="Sheet1")
        assert "range_prob" in derived.columns
        assert "hmm_artifact_version" in derived.columns
        assert derived.loc[0, "range_prob"] == pytest.approx(0.77)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
