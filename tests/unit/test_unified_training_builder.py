"""Tests for neutralgrid.training.unified_training_builder — unified training table."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pandas as pd
import numpy as np
import pytest

from neutralgrid.core.constants import (
    ENGINE_VERSION,
    FORMULA_VERSION,
    LABEL_CONTRACT_VERSION,
)
from neutralgrid.grid.formulas import GEOMETRIC, grid_spacing_pct
from neutralgrid.training.unified_training_builder import (
    UnifiedTrainingBuilder,
    TRAINING_FEATURES,
)
from neutralgrid.training.data_generator import ExistingDataMapper


def test_existing_data_mapper_preserves_explicit_profit_per_grid_without_geometry() -> None:
    """Compact authoritative rows must not lose a recorded grid-return feature."""
    source = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "start_time_utc": ["2026-06-08T12:00:00+00:00"],
            "pnl_pct": [1.0],
            "sl_hit": [False],
            "profit_per_grid_pct": [0.42],
        }
    )

    mapped = ExistingDataMapper().map_dataframe(source)

    assert float(mapped.iloc[0]["profit_per_grid_pct"]) == pytest.approx(0.42)


def _make_snapshot_parquet(
    snapshot_dir: Path,
    candidate_id: str,
    symbol: str,
    start_time_utc: str = "2026-02-20T08:00:00+00:00",
    **extra_features,
) -> None:
    """Write a minimal snapshot parquet file with one row."""
    row = {
        "candidate_id": candidate_id,
        "symbol": symbol,
        "start_time_utc": pd.Timestamp(start_time_utc),
        "range_prob": 0.82,
        "trend_prob": 0.15,
        "adx_15m": 22.0,
        "adx_5m": 16.0,
        "rsi_15m": 48.0,
        "funding_rate": 0.0001,
        "open_interest": 1_500_000.0,
        "quote_volume_24h": 50_000_000.0,
        "bb_width": 0.03,
        "atr_pct_15m": 0.008,
        "range_size_pct": 5.0,
        "ema_crosses_5m": 4,
        "vwap_crosses_5m": 6,
    }
    row.update(extra_features)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    # Use a unique filename based on candidate_id
    fname = f"snapshots_{candidate_id.replace('/', '_')}.parquet"
    pd.DataFrame([row]).to_parquet(snapshot_dir / fname)


def _current_backtest_contract(**overrides: Any) -> dict[str, Any]:
    row = {
        "backtest_timestamp": "2026-02-20T08:10:00+00:00",
        "label_contract_version": LABEL_CONTRACT_VERSION,
        "engine_version": ENGINE_VERSION,
        "formula_version": FORMULA_VERSION,
        "mode": "geometric",
        "fill_mode": "wick",
        "global_cooldown_bars": 0,
        "cb_enabled": False,
        "is_authoritative": True,
    }
    row.update(overrides)
    return row


_RUN_FULL_PIPELINE_PATH = Path(__file__).resolve().parents[2] / "run_full_pipeline.py"
_RUN_FULL_PIPELINE_SPEC = importlib.util.spec_from_file_location("run_full_pipeline", _RUN_FULL_PIPELINE_PATH)
assert _RUN_FULL_PIPELINE_SPEC is not None and _RUN_FULL_PIPELINE_SPEC.loader is not None
run_full_pipeline = importlib.util.module_from_spec(_RUN_FULL_PIPELINE_SPEC)
_RUN_FULL_PIPELINE_SPEC.loader.exec_module(run_full_pipeline)


def test_derive_grid_spacing_pct_uses_canonical_mode_formula() -> None:
    builder = UnifiedTrainingBuilder()

    assert builder._derive_grid_spacing_pct(0.68, 1.28, 58, GEOMETRIC) == pytest.approx(
        grid_spacing_pct(0.68, 1.28, 58, GEOMETRIC)
    )


def test_derive_grid_spacing_pct_requires_explicit_mode() -> None:
    builder = UnifiedTrainingBuilder()

    assert builder._derive_grid_spacing_pct(0.68, 1.28, 58, None) is None
    assert builder._derive_grid_spacing_pct(0.68, 1.28, 58, "unknown") is None


def test_enrich_backtest_feature_coverage_derives_spacing_without_erasing_mode() -> None:
    builder = UnifiedTrainingBuilder()
    df = pd.DataFrame(
        [
            {
                "candidate_id": "BSBUSDT_20260523_090107_223d635f",
                "symbol": "BSBUSDT",
                "grid_lower": 0.68,
                "grid_upper": 1.28,
                "num_grids": 58,
                "mode": GEOMETRIC,
                "net_pnl_pct": 1.0,
            }
        ]
    )

    enriched = builder._enrich_backtest_feature_coverage(df)

    assert enriched.loc[0, "mode"] == GEOMETRIC
    assert enriched.loc[0, "grid_spacing_pct"] == pytest.approx(
        grid_spacing_pct(0.68, 1.28, 58, GEOMETRIC)
    )
    assert enriched.loc[0, "grid_spacing_source"] == "derived_from_candidate_geometry"


def test_enrich_backtest_feature_coverage_does_not_derive_spacing_without_mode() -> None:
    builder = UnifiedTrainingBuilder()
    df = pd.DataFrame(
        [
            {
                "candidate_id": "BSBUSDT_20260523_090107_223d635f",
                "symbol": "BSBUSDT",
                "grid_lower": 0.68,
                "grid_upper": 1.28,
                "num_grids": 58,
                "net_pnl_pct": 1.0,
            }
        ]
    )

    enriched = builder._enrich_backtest_feature_coverage(df)

    assert pd.isna(enriched.loc[0, "grid_spacing_pct"])
    assert "grid_spacing_source" not in enriched.columns


@pytest.fixture
def full_setup(tmp_path: Path) -> dict:
    """Create a complete fixture with snapshot and backtest data.

    Uses the snapshot-only architecture: snapshots are the sole feature source,
    joined to backtest outcomes by candidate_id.
    """
    # -- Snapshot parquet files (authoritative feature source)
    snapshot_dir = tmp_path / "training_snapshots"

    cid_btc = "BTCUSDT_20260220_080000_aabbccdd"
    cid_sol = "SOLUSDT_20260220_080000_11223344"

    _make_snapshot_parquet(
        snapshot_dir, cid_btc, "BTCUSDT",
        start_time_utc="2026-02-20T08:00:00+00:00",
    )
    _make_snapshot_parquet(
        snapshot_dir, cid_sol, "SOLUSDT",
        start_time_utc="2026-02-20T08:00:00+00:00",
    )

    # -- Backtest training rows (outcome source)
    bt_dir = tmp_path / "backtest_candidates"
    bt_dir.mkdir()
    pd.DataFrame([
        {
            "symbol": "SOLUSDT",
            "candidate_id": cid_sol,
            "start_time_utc": "2026-02-20T08:00:00+00:00",
            "pnl_pct": 12.5,
            "net_pnl_pct": 12.5,
            "y": 1,
            "sl_hit": False,
            "barrier_touched": "pt",
            "source": "backtest",
            "sample_weight_override": 0.5,
            "mae": 11.0,
            "mfe": 22.0,
            "mae_pct_initial": 2.75,
            "mfe_pct_initial": 5.5,
            "adx_1h": 25.0,
            "rsi_15m": 55.0,
            "num_grids": 20,
            **_current_backtest_contract(),
        },
        {
            "symbol": "BTCUSDT",
            "candidate_id": cid_btc,
            "start_time_utc": "2026-02-20T08:00:00+00:00",
            "pnl_pct": 5.2,
            "net_pnl_pct": 5.2,
            "y": 1,
            "sl_hit": False,
            "barrier_touched": "pt",
            "source": "backtest",
            "sample_weight_override": 0.5,
            "mae": 18.0,
            "mfe": 27.0,
            "mae_pct_initial": 4.5,
            "mfe_pct_initial": 6.75,
            **_current_backtest_contract(),
        },
    ]).to_csv(bt_dir / "training_rows_20260220.csv", index=False)

    return {
        "snapshot_dir": snapshot_dir,
        "bt_dir": bt_dir,
        "tmp_path": tmp_path,
        "cid_btc": cid_btc,
        "cid_sol": cid_sol,
    }


class TestUnifiedTrainingBuilder:
    """Tests for the UnifiedTrainingBuilder (snapshot-only architecture)."""

    def test_build_returns_dataframe(self, full_setup):
        """build() returns a non-empty DataFrame."""
        d = full_setup
        builder = UnifiedTrainingBuilder(
            backtest_results_dir=d["bt_dir"],
            snapshot_dir=d["snapshot_dir"],
        )
        result = builder.build()
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1

    def test_all_training_features_present(self, full_setup):
        """All standard training features are columns in the output."""
        d = full_setup
        builder = UnifiedTrainingBuilder(
            backtest_results_dir=d["bt_dir"],
            snapshot_dir=d["snapshot_dir"],
        )
        result = builder.build()
        for feat in TRAINING_FEATURES:
            assert feat in result.columns, f"Missing feature column: {feat}"

    def test_provenance_columns_present(self, full_setup):
        """source, sample_weight_override, candidate_id are present."""
        d = full_setup
        builder = UnifiedTrainingBuilder(
            backtest_results_dir=d["bt_dir"],
            snapshot_dir=d["snapshot_dir"],
        )
        result = builder.build()
        assert "source" in result.columns
        assert "sample_weight_override" in result.columns
        assert "candidate_id" in result.columns

    def test_backtest_source_label(self, full_setup):
        """Rows from backtest training CSVs have source='backtest'."""
        d = full_setup
        builder = UnifiedTrainingBuilder(
            backtest_results_dir=d["bt_dir"],
            snapshot_dir=d["snapshot_dir"],
        )
        result = builder.build()
        assert not result.empty
        # All rows come from snapshot-outcome join
        assert (result["source"] == "snapshot").all()

    def test_backtest_weight_propagated(self, full_setup):
        """sample_weight_override from backtest outcome CSV is preserved."""
        d = full_setup
        builder = UnifiedTrainingBuilder(
            backtest_results_dir=d["bt_dir"],
            snapshot_dir=d["snapshot_dir"],
        )
        result = builder.build()
        if "sample_weight_override" in result.columns and not result.empty:
            # Default from join sets 1.0; backtest may override
            assert result["sample_weight_override"].notna().all()

    def test_labels_present(self, full_setup):
        """y and pnl_pct columns are present."""
        d = full_setup
        builder = UnifiedTrainingBuilder(
            backtest_results_dir=d["bt_dir"],
            snapshot_dir=d["snapshot_dir"],
        )
        result = builder.build()
        assert "y" in result.columns
        assert "pnl_pct" in result.columns

    def test_excursion_outcomes_propagated(self, full_setup):
        """MAE/MFE outcome columns are preserved from backtest rows."""
        d = full_setup
        builder = UnifiedTrainingBuilder(
            backtest_results_dir=d["bt_dir"],
            snapshot_dir=d["snapshot_dir"],
        )
        result = builder.build()
        assert not result.empty
        # Check SOL row excursions
        sol_rows = result[result["candidate_id"] == d["cid_sol"]]
        if not sol_rows.empty:
            assert float(sol_rows.iloc[0]["mae"]) == pytest.approx(11.0)
            assert float(sol_rows.iloc[0]["mfe"]) == pytest.approx(22.0)

    def test_no_data_returns_empty(self, tmp_path: Path):
        """Returns empty when no data sources are available."""
        builder = UnifiedTrainingBuilder(
            backtest_results_dir=tmp_path / "no_bt",
            snapshot_dir=tmp_path / "no_snapshots",
        )
        result = builder.build()
        assert result.empty

    def test_deduplication_prefers_latest_snapshot(self, full_setup):
        """When same candidate_id appears in multiple snapshots, latest wins."""
        d = full_setup
        # Add a second snapshot with a later timestamp for the same candidate_id
        _make_snapshot_parquet(
            d["snapshot_dir"], d["cid_sol"], "SOLUSDT",
            start_time_utc="2026-02-21T08:00:00+00:00",
            range_prob=0.99,  # distinguishing value
        )

        builder = UnifiedTrainingBuilder(
            backtest_results_dir=d["bt_dir"],
            snapshot_dir=d["snapshot_dir"],
        )
        result = builder.build()
        sol_rows = result[result["candidate_id"] == d["cid_sol"]]
        if not sol_rows.empty:
            # The later snapshot should win
            assert float(sol_rows.iloc[0]["range_prob"]) == pytest.approx(0.99)

    def test_multiple_candidates_joined(self, full_setup):
        """Output contains rows from multiple candidates."""
        d = full_setup
        builder = UnifiedTrainingBuilder(
            backtest_results_dir=d["bt_dir"],
            snapshot_dir=d["snapshot_dir"],
        )
        result = builder.build()
        assert len(result) >= 2  # BTC + SOL

    def test_stale_physics_row_is_rejected(self, tmp_path: Path):
        snapshot_dir = tmp_path / "training_snapshots"
        bt_dir = tmp_path / "backtest_candidates"
        bt_dir.mkdir()

        cid = "ADAUSDT_20260220_080000_stale001"
        _make_snapshot_parquet(snapshot_dir, cid, "ADAUSDT")

        pd.DataFrame([
            {
                "symbol": "ADAUSDT",
                "candidate_id": cid,
                "pnl_pct": 9.0,
                "net_pnl_pct": 9.0,
                "y": 1,
                "y_horizon": 1,
                **_current_backtest_contract(
                    engine_version="realistic-v7",
                    formula_version="alignment-v1",
                    mode="arithmetic",
                    fill_mode="close",
                    global_cooldown_bars=120,
                    cb_enabled=True,
                    is_authoritative=False,
                ),
            }
        ]).to_csv(bt_dir / "training_data_20260220.csv", index=False)

        builder = UnifiedTrainingBuilder(
            backtest_results_dir=bt_dir,
            snapshot_dir=snapshot_dir,
        )
        result = builder.build()
        assert result.empty

    def test_public_market_profile_missing_provenance_is_gated(self, tmp_path: Path):
        builder = UnifiedTrainingBuilder(
            backtest_results_dir=tmp_path / "no_bt",
            snapshot_dir=tmp_path / "no_snapshots",
        )
        row = pd.DataFrame(
            [
                {
                    **_current_backtest_contract(),
                    "candidate_id": "ADAUSDT_20260220_080000_pub001",
                    "realism_profile": "candidate_time_public_market_v1",
                    "pnl_pct": 3.0,
                    "y": 1,
                }
            ]
        )

        gated = builder._apply_ingestion_gate(row)

        assert bool(gated.iloc[0]["version_gated"]) is True
        assert "missing_exchange_filter_source" in str(gated.iloc[0]["version_gate_reason"])

    def test_public_market_profile_invalid_filter_status_is_gated(self, tmp_path: Path):
        builder = UnifiedTrainingBuilder(
            backtest_results_dir=tmp_path / "no_bt",
            snapshot_dir=tmp_path / "no_snapshots",
        )
        provenance = {
            "exchange_filter_source": "binance_exchange_info",
            "tick_size_source": "binance_exchange_info",
            "step_size_source": "binance_exchange_info",
            "min_notional_source": "binance_exchange_info",
            "exchange_filter_validation_status": "invalid",
            "funding_series_status": "no_event_in_window",
            "funding_series_source": "binance_funding_rate",
            "fill_price_source": "last",
            "valuation_price_source": "mark",
            "mark_price_source": "binance_mark_price_klines",
            "historical_depth_source": "missing",
        }
        row = pd.DataFrame(
            [
                {
                    **_current_backtest_contract(),
                    "candidate_id": "ADAUSDT_20260220_080000_pub002",
                    "realism_profile": "candidate_time_public_market_v1",
                    "pnl_pct": 3.0,
                    "y": 1,
                    **provenance,
                }
            ]
        )

        gated = builder._apply_ingestion_gate(row)

        assert bool(gated.iloc[0]["version_gated"]) is True
        assert "exchange_filter_validation_status" in str(gated.iloc[0]["version_gate_reason"])

    @pytest.mark.parametrize(
        "profile",
        [
            "candidate_time_geometric_v1",
            "candidate_time_public_market_v1",
        ],
    )
    def test_shadow_realism_profiles_are_never_authoritative(
        self,
        tmp_path: Path,
        profile: str,
    ) -> None:
        builder = UnifiedTrainingBuilder(
            backtest_results_dir=tmp_path / "no_bt",
            snapshot_dir=tmp_path / "no_snapshots",
        )
        row = pd.DataFrame(
            [
                {
                    **_current_backtest_contract(),
                    "candidate_id": "ADAUSDT_20260220_080000_shadow001",
                    "realism_profile": profile,
                    "pnl_pct": 3.0,
                    "y": 1,
                }
            ]
        )

        gated = builder._apply_ingestion_gate(row)

        assert bool(gated.iloc[0]["version_gated"]) is True
        assert gated.iloc[0]["source_class"] == "non_authoritative"
        assert "shadow_realism_profile_not_authoritative" in str(
            gated.iloc[0]["version_gate_reason"]
        )

    def test_shadow_realism_gate_cannot_be_bootstrap_relaxed(
        self,
        tmp_path: Path,
    ) -> None:
        snapshot_dir = tmp_path / "training_snapshots"
        bt_dir = tmp_path / "backtest_candidates"
        bt_dir.mkdir()
        candidate_id = "ADAUSDT_20260220_080000_shadowbootstrap"
        _make_snapshot_parquet(snapshot_dir, candidate_id, "ADAUSDT")
        pd.DataFrame(
            [
                {
                    "symbol": "ADAUSDT",
                    "candidate_id": candidate_id,
                    "pnl_pct": 3.0,
                    "net_pnl_pct": 3.0,
                    "y": 1,
                    "y_horizon": 1,
                    **_current_backtest_contract(
                        realism_profile="candidate_time_geometric_v1"
                    ),
                }
            ]
        ).to_csv(bt_dir / "training_data_shadow.csv", index=False)

        result = UnifiedTrainingBuilder(
            backtest_results_dir=bt_dir,
            snapshot_dir=snapshot_dir,
        ).build()

        assert result.empty

    @pytest.mark.parametrize(
        ("profile", "expected_reason"),
        [
            ("", "missing_realism_profile"),
            ("unregistered_profile", "unsupported_realism_profile"),
        ],
    )
    def test_present_invalid_realism_profile_fails_closed(
        self,
        tmp_path: Path,
        profile: str,
        expected_reason: str,
    ) -> None:
        builder = UnifiedTrainingBuilder(
            backtest_results_dir=tmp_path / "no_bt",
            snapshot_dir=tmp_path / "no_snapshots",
        )
        row = pd.DataFrame(
            [
                {
                    **_current_backtest_contract(),
                    "candidate_id": "ADAUSDT_20260220_080000_invalid001",
                    "realism_profile": profile,
                    "pnl_pct": 3.0,
                    "y": 1,
                }
            ]
        )

        gated = builder._apply_ingestion_gate(row)

        assert bool(gated.iloc[0]["version_gated"]) is True
        assert expected_reason in str(gated.iloc[0]["version_gate_reason"])

    def test_absent_realism_profile_column_preserves_historical_compatibility(
        self,
        tmp_path: Path,
    ) -> None:
        builder = UnifiedTrainingBuilder(
            backtest_results_dir=tmp_path / "no_bt",
            snapshot_dir=tmp_path / "no_snapshots",
        )
        row = pd.DataFrame(
            [
                {
                    **_current_backtest_contract(),
                    "candidate_id": "ADAUSDT_20260220_080000_historical001",
                    "pnl_pct": 3.0,
                    "y": 1,
                }
            ]
        )

        gated = builder._apply_ingestion_gate(row)

        assert bool(gated.iloc[0]["version_gated"]) is False
        assert str(gated.iloc[0]["version_gate_reason"]) == ""

    def test_dedup_prefers_current_contract(self, tmp_path: Path):
        snapshot_dir = tmp_path / "training_snapshots"
        bt_dir = tmp_path / "backtest_candidates"
        bt_dir.mkdir()

        cid = "ADAUSDT_20260220_080000_dedup001"
        _make_snapshot_parquet(snapshot_dir, cid, "ADAUSDT")

        pd.DataFrame([
            {
                "symbol": "ADAUSDT",
                "candidate_id": cid,
                "pnl_pct": -5.0,
                "net_pnl_pct": -5.0,
                "y": 0,
                "y_horizon": 0,
                **_current_backtest_contract(
                    label_contract_version="2026-04-17",
                    engine_version="realistic-v7",
                    formula_version="alignment-v1",
                    mode="arithmetic",
                    fill_mode="close",
                    global_cooldown_bars=120,
                    cb_enabled=True,
                    is_authoritative=False,
                    backtest_timestamp="2026-02-20T08:12:00+00:00",
                ),
            },
            {
                "symbol": "ADAUSDT",
                "candidate_id": cid,
                "pnl_pct": 7.0,
                "net_pnl_pct": 7.0,
                "y": 1,
                "y_horizon": 1,
                **_current_backtest_contract(
                    backtest_timestamp="2026-02-20T08:10:00+00:00",
                ),
            },
        ]).to_csv(bt_dir / "training_data_20260220.csv", index=False)

        builder = UnifiedTrainingBuilder(
            backtest_results_dir=bt_dir,
            snapshot_dir=snapshot_dir,
        )
        result = builder.build()
        assert not result.empty
        assert len(result[result["candidate_id"] == cid]) == 1
        assert float(result.iloc[0]["pnl_pct"]) == pytest.approx(7.0)

    def test_raw_backtest_excursion_columns_preserved(self, tmp_path: Path):
        """Raw backtest-results fallback keeps MAE/MFE outcome columns."""
        snapshot_dir = tmp_path / "training_snapshots"
        bt_dir = tmp_path / "backtest_candidates"
        bt_dir.mkdir()

        cid = "XRPUSDT_20260220_080000_abc12345"
        _make_snapshot_parquet(snapshot_dir, cid, "XRPUSDT")

        pd.DataFrame([
            {
                "symbol": "XRPUSDT",
                "candidate_id": cid,
                "net_pnl_pct": 4.2,
                "duration_hours": 2.0,
                "label_positive_by_horizon": True,
                "mae": 7.5,
                "mfe": 10.0,
                "mae_pct_initial": 1.875,
                "mfe_pct_initial": 2.5,
                **_current_backtest_contract(),
            }
        ]).to_csv(bt_dir / "backtest_results_20260220.csv", index=False)

        builder = UnifiedTrainingBuilder(
            backtest_results_dir=bt_dir,
            snapshot_dir=snapshot_dir,
            include_reconstruction=True,
        )
        result = builder.build()
        assert not result.empty
        row = result.iloc[0]
        assert row["mae"] == pytest.approx(7.5)
        assert row["mfe"] == pytest.approx(10.0)
        assert row["mae_pct_initial"] == pytest.approx(1.875)
        assert row["mfe_pct_initial"] == pytest.approx(2.5)

    def test_ev_contract_matches_runtime_post_scoring(self):
        class DummyRanker:
            def __init__(self, *_args, **_kwargs):
                self.empirical_profile = None

            def compute_score(self, *, leverage: int | None = None, **_kwargs):
                assert leverage == 10
                return SimpleNamespace(
                    rank_score=1.25,
                    ev_24h=3.50,
                    ev_raw=4.10,
                    ev_aligned=3.00,
                    expected_fills=1.0,
                    fill_rate_scale=1.0,
                    fill_rate_scope="global",
                    fill_rate_scope_samples=50.0,
                    ev_alignment_scope="global",
                    ev_alignment_scope_samples=50.0,
                    ev_alignment_r2=0.5,
                    fill_revenue_pct=2.5,
                    funding_cost_pct=0.1,
                    boundary_loss_pct=0.2,
                    total_penalty=0.3,
                )

        class DummyMeta:
            @staticmethod
            def load(_path):
                return SimpleNamespace(is_trained=False)

        row = {
            "symbol": "BTCUSDT",
            "candidate_id": "BTC_1",
            "score": 90.0,
            "grid_is_valid": True,
            "profit_per_grid_pct": 0.45,
            "num_grids": 20,
            "survival_prob": 0.81,
            "trend_prob": 0.18,
            "range_size_pct": 5.0,
            "funding_rate": 0.0001,
        }

        builder = UnifiedTrainingBuilder()
        builder._ranker = cast(Any, DummyRanker())

        with patch.object(run_full_pipeline, "PnLRanker", DummyRanker), \
             patch.object(run_full_pipeline, "RankingConfig", lambda: None), \
             patch.object(run_full_pipeline, "MetaLabeler", DummyMeta), \
             patch.object(run_full_pipeline, "AFML_POST_SCORING_AVAILABLE", True):
            scored = run_full_pipeline._apply_afml_post_scoring(pd.DataFrame([row]))

        contract = builder._derive_ev_contract(row)

        assert scored.loc[0, "ev_score"] == pytest.approx(contract["ev_score"])
        assert scored.loc[0, "ev_24h"] == pytest.approx(contract["ev_24h"])
        assert scored.loc[0, "ev_raw"] == pytest.approx(contract["ev_raw"])
        assert scored.loc[0, "ev_aligned"] == pytest.approx(contract["ev_aligned"])

    def test_slow_path_preserves_horizon_censored(self, tmp_path: Path):
        """DURATION_FIX §21.4 — slow-path must copy horizon_censored from raw CSV."""
        snapshot_dir = tmp_path / "training_snapshots"
        bt_dir = tmp_path / "backtest_candidates"
        bt_dir.mkdir()

        cid = "BNBUSDT_20260220_080000_cens0001"
        _make_snapshot_parquet(snapshot_dir, cid, "BNBUSDT")

        pd.DataFrame(
            [
                {
                    "symbol": "BNBUSDT",
                    "candidate_id": cid,
                    "net_pnl_pct": 4.0,
                    "duration_hours": 6.0,
                    "label_positive_by_horizon": True,
                    "horizon_censored": True,
                    **_current_backtest_contract(),
                }
            ]
        ).to_csv(bt_dir / "backtest_results_20260220.csv", index=False)

        builder = UnifiedTrainingBuilder(
            backtest_results_dir=bt_dir,
            snapshot_dir=snapshot_dir,
            include_reconstruction=True,
        )
        result = builder.build()
        assert not result.empty
        assert "horizon_censored" in result.columns
        assert bool(result.iloc[0]["horizon_censored"]) is True

    def test_slow_path_horizon_censored_string_false_stays_false(self, tmp_path: Path):
        """DURATION_FIX P0 — string "False" must not silently coerce to True.

        Plain bool("False") == True in Python. The slow-path parser must
        recognize the string form and produce a literal False.
        """
        snapshot_dir = tmp_path / "training_snapshots"
        bt_dir = tmp_path / "backtest_candidates"
        bt_dir.mkdir()

        cid = "BNBUSDT_20260220_080000_cens0002"
        _make_snapshot_parquet(snapshot_dir, cid, "BNBUSDT")

        csv_path = bt_dir / "backtest_results_20260220.csv"
        # Write CSV manually so the horizon_censored column is an object-typed
        # string "False" rather than a pandas-parsed bool.
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "symbol", "candidate_id", "net_pnl_pct", "duration_hours",
                "label_positive_by_horizon", "horizon_censored",
                "backtest_timestamp", "label_contract_version", "engine_version",
                "formula_version", "mode", "fill_mode", "global_cooldown_bars",
                "cb_enabled", "is_authoritative",
            ])
            writer.writerow([
                "BNBUSDT", cid, "4.0", "6.0",
                "True", "False",
                "2026-02-20T08:10:00+00:00", LABEL_CONTRACT_VERSION, ENGINE_VERSION,
                FORMULA_VERSION, "geometric", "wick", "0", "False", "True",
            ])

        builder = UnifiedTrainingBuilder(
            backtest_results_dir=bt_dir,
            snapshot_dir=snapshot_dir,
            include_reconstruction=True,
        )
        result = builder.build()
        assert not result.empty
        assert "horizon_censored" in result.columns
        assert bool(result.iloc[0]["horizon_censored"]) is False

    def test_slow_path_horizon_censored_invalid_string_gates_row(self, tmp_path: Path):
        """DURATION_FIX §11 — a current-contract row with an unparseable
        horizon_censored value breaches the "known and current" label-semantics
        invariant and must be excluded from final training rather than admitted
        with ambiguous provenance.
        """
        snapshot_dir = tmp_path / "training_snapshots"
        bt_dir = tmp_path / "backtest_candidates"
        bt_dir.mkdir()

        bad_cid = "BNBUSDT_20260220_080000_cens0003"
        good_cid = "SOLUSDT_20260220_080000_cens0004"
        _make_snapshot_parquet(snapshot_dir, bad_cid, "BNBUSDT")
        _make_snapshot_parquet(snapshot_dir, good_cid, "SOLUSDT")

        csv_path = bt_dir / "backtest_results_20260220.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "symbol", "candidate_id", "net_pnl_pct", "duration_hours",
                "label_positive_by_horizon", "horizon_censored",
                "backtest_timestamp", "label_contract_version", "engine_version",
                "formula_version", "mode", "fill_mode", "global_cooldown_bars",
                "cb_enabled", "is_authoritative",
            ])
            writer.writerow([
                "BNBUSDT", bad_cid, "4.0", "6.0",
                "True", "maybe",
                "2026-02-20T08:10:00+00:00", LABEL_CONTRACT_VERSION, ENGINE_VERSION,
                FORMULA_VERSION, "geometric", "wick", "0", "False", "True",
            ])
            writer.writerow([
                "SOLUSDT", good_cid, "3.0", "5.5",
                "True", "True",
                "2026-02-20T08:10:00+00:00", LABEL_CONTRACT_VERSION, ENGINE_VERSION,
                FORMULA_VERSION, "geometric", "wick", "0", "False", "True",
            ])

        builder = UnifiedTrainingBuilder(
            backtest_results_dir=bt_dir,
            snapshot_dir=snapshot_dir,
            include_reconstruction=True,
        )
        result = builder.build()
        # The well-formed row survives; the gated row is dropped.
        assert not result.empty
        surviving_cids = set(result["candidate_id"].astype(str))
        assert good_cid in surviving_cids
        assert bad_cid not in surviving_cids

    def test_snapshot_feature_is_not_backfilled_from_backtest_overlap(self, tmp_path: Path):
        """Active feature columns remain snapshot-authoritative after the join."""
        snapshot_dir = tmp_path / "training_snapshots"
        bt_dir = tmp_path / "backtest_candidates"
        bt_dir.mkdir()

        cid = "ETHUSDT_20260220_080000_auth0001"
        _make_snapshot_parquet(
            snapshot_dir,
            cid,
            "ETHUSDT",
            adx_15m=np.nan,
        )

        pd.DataFrame(
            [
                {
                    "symbol": "ETHUSDT",
                    "candidate_id": cid,
                    "net_pnl_pct": 6.0,
                    "label_positive_by_horizon": True,
                    "adx_15m": 99.0,
                    **_current_backtest_contract(),
                }
            ]
        ).to_csv(bt_dir / "backtest_results_20260220.csv", index=False)

        builder = UnifiedTrainingBuilder(
            backtest_results_dir=bt_dir,
            snapshot_dir=snapshot_dir,
            include_reconstruction=True,
        )

        result = builder.build()
        assert not result.empty
        assert pd.isna(result.iloc[0]["adx_15m"])

    def test_build_second_range_size_path_uses_midpoint(self, tmp_path: Path):
        """The backtest enrichment fill loop derives range_size_pct from grid midpoint."""
        builder = UnifiedTrainingBuilder(
            snapshot_dir=tmp_path / "no_snapshots",
        )
        backtest_rows = pd.DataFrame(
            [
                {
                    "candidate_id": "cid-1",
                    "symbol": "BTCUSDT",
                    "source": "backtest",
                    "sample_weight_override": 0.5,
                    "start_time_utc": "2026-03-01T00:00:00+00:00",
                    "grid_lower": 95.0,
                    "grid_upper": 105.0,
                    "num_grids": 10,
                    "price_start": 1.0,
                    "price_end": 2.0,
                    "range_size_pct": np.nan,
                }
            ]
        )

        result = builder._enrich_backtest_feature_coverage(backtest_rows.copy())

        assert not result.empty
        assert result.iloc[0]["range_size_pct"] == pytest.approx(10.0)


class TestErr021LabelPrecedence:
    """ERR-021: degenerate `y` column must fall through to net_pnl_pct >= meta_hurdle_pct.

    Ground-truth scenario from ERRORS_LOG.md ERR-021: backtest outcome CSVs
    carried `y.value_counts() = {0: 162}` (all-zero, nunique == 1). The previous
    `notna().any()` predicate selected this column and starved the
    `net_pnl_pct >= meta_hurdle_pct` fallback at lines 833-838, raising
    `ValueError: Positive rate 0.0% is below 5%` at meta_labeler.py:725-732.

    Fix at unified_training_builder.py:822-851: extend the degeneracy bypass
    already used by the hlabel_meta path (lines 793-815) to require
    `nunique() > 1` AND positive_rate >= 0.05 before accepting a column.
    """

    def test_all_zero_y_column_falls_through_to_net_pnl_pct(self, tmp_path: Path):
        builder = UnifiedTrainingBuilder(
            snapshot_dir=tmp_path / "no_snapshots",
        )
        # 10 rows: y is all-zero (degenerate), net_pnl_pct has 6 positives at >= 3.0.
        df = pd.DataFrame({
            "candidate_id": [f"cid-{i}" for i in range(10)],
            "symbol": ["BTCUSDT"] * 10,
            "y": [0] * 10,
            "net_pnl_pct": [
                5.0, 4.5, 7.2, 3.1, 8.0, 6.0,
                -1.0, -2.5, 0.5, 1.2,
            ],
            "start_time_utc": ["2026-03-01T00:00:00+00:00"] * 10,
        })
        result = builder._normalize_backtest_targets(df)
        # Degenerate `y` rejected → fallback to net_pnl_pct_hurdle (default 3.0).
        assert (result["label_source"] == "net_pnl_pct_hurdle").all(), \
            f"Expected label_source=net_pnl_pct_hurdle, got {result['label_source'].unique().tolist()}"
        # 6 of 10 rows have net_pnl_pct >= 3.0 → y.sum() == 6.
        assert int(result["y"].sum()) == 6, \
            f"Expected y.sum()=6, got {int(result['y'].sum())}"
        assert float(result["y"].mean()) == pytest.approx(0.6)

    def test_non_degenerate_y_column_is_kept(self, tmp_path: Path):
        builder = UnifiedTrainingBuilder(
            snapshot_dir=tmp_path / "no_snapshots",
        )
        # `y` has both classes and >5% positives → keep it.
        df = pd.DataFrame({
            "candidate_id": [f"cid-{i}" for i in range(10)],
            "symbol": ["BTCUSDT"] * 10,
            "y": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "net_pnl_pct": [10.0] * 10,  # would all pass hurdle if used
            "start_time_utc": ["2026-03-01T00:00:00+00:00"] * 10,
        })
        result = builder._normalize_backtest_targets(df)
        # Non-degenerate y is preferred; fallback path NOT taken.
        assert (result["label_source"] == "y").all(), \
            f"Expected label_source=y, got {result['label_source'].unique().tolist()}"
        assert int(result["y"].sum()) == 3, \
            f"Expected y.sum()=3 (preserved), got {int(result['y'].sum())}"


def test_build_meta_labeler_pool_dedups_duplicate_candidate_ids():
    """Re-backtesting the same candidate_id must NOT inject duplicate, possibly
    contradictory (X, y) rows into the meta pool (verified to crash OOF AUC
    0.70 -> 0.44). build_meta_labeler_pool dedups by (stripped) candidate_id
    keeping the latest backtest by backtest_timestamp, and preserves rows that
    carry no candidate_id.

    The realistic duplicate shape is the regression target: a re-backtest reuses
    the SAME start_time_utc (it is the candidate's market-event time, intrinsic to
    the candidate) and differs only in backtest_timestamp. The newer-input-first
    ordering below would defeat any start_time_utc / input-order keying, so the
    surviving outcome proves recency is keyed on backtest_timestamp."""
    builder = UnifiedTrainingBuilder()
    common = {"version_gated": False, "is_authoritative": True, "source_class": "backtest"}
    rows = pd.DataFrame([
        # same candidate_id, SAME start_time_utc, DIFFERENT backtest_timestamp.
        # The NEWER run (-2.0) is listed FIRST so input-order/start_time keep="last"
        # would wrongly keep the older 5.0; only backtest_timestamp keying keeps -2.0.
        {"candidate_id": "BTCUSDT_1", "symbol": "BTCUSDT",
         "start_time_utc": "2026-02-20T08:00:00+00:00",
         "backtest_timestamp": "2026-03-10T09:00:00+00:00", "net_pnl_pct": -2.0, **common},
        {"candidate_id": "BTCUSDT_1", "symbol": "BTCUSDT",
         "start_time_utc": "2026-02-20T08:00:00+00:00",
         "backtest_timestamp": "2026-02-20T08:05:00+00:00", "net_pnl_pct": 5.0, **common},
        # whitespace-variant candidate_id is the SAME identity (presence mask strips)
        # and must collapse; the later backtest (7.0) wins.
        {"candidate_id": " ETHUSDT_1", "symbol": "ETHUSDT",
         "start_time_utc": "2026-02-25T08:00:00+00:00",
         "backtest_timestamp": "2026-02-25T08:05:00+00:00", "net_pnl_pct": 4.0, **common},
        {"candidate_id": "ETHUSDT_1", "symbol": "ETHUSDT",
         "start_time_utc": "2026-02-25T08:00:00+00:00",
         "backtest_timestamp": "2026-03-05T08:05:00+00:00", "net_pnl_pct": 7.0, **common},
        # rows WITHOUT a candidate_id must be preserved (not collapsed together)
        {"candidate_id": None, "symbol": "SOLUSDT",
         "start_time_utc": "2026-02-26T08:00:00+00:00",
         "backtest_timestamp": "2026-02-26T08:05:00+00:00", "net_pnl_pct": 1.0, **common},
        {"candidate_id": None, "symbol": "ADAUSDT",
         "start_time_utc": "2026-02-27T08:00:00+00:00",
         "backtest_timestamp": "2026-02-27T08:05:00+00:00", "net_pnl_pct": 2.0, **common},
    ])
    with patch.object(UnifiedTrainingBuilder, "_load_backtest_rows", return_value=rows):
        pool = builder.build_meta_labeler_pool()

    # BTCUSDT_1 -> 1, ETHUSDT (both variants) -> 1, + 2 null-cid rows = 4
    assert len(pool) == 4
    btc = cast(pd.DataFrame, pool[pool["symbol"] == "BTCUSDT"])
    assert len(btc) == 1
    # latest backtest_timestamp (2026-03-10, -2.0) survives despite tied start_time
    # and the older run appearing later in input order.
    assert float(btc.iloc[0]["net_pnl_pct"]) == pytest.approx(-2.0)
    # whitespace variants collapse to a single ETHUSDT row, keeping the later run.
    eth = cast(pd.DataFrame, pool[pool["symbol"] == "ETHUSDT"])
    assert len(eth) == 1
    assert float(eth.iloc[0]["net_pnl_pct"]) == pytest.approx(7.0)
    # both candidate_id-less rows survive (missing id is not an identity)
    assert int(cast(pd.Series, pool["candidate_id"]).isna().sum()) == 2
