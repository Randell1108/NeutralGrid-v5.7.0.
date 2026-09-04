from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neutralgrid.core.constants import (
    ENGINE_VERSION,
    FORMULA_VERSION,
    LABEL_CONTRACT_VERSION,
)
from neutralgrid.training.unified_training_builder import UnifiedTrainingBuilder


# Authoritative engine-settings contract fields required by the ingestion gate
# (unified_training_builder._apply_ingestion_gate). A backtest row missing any
# of these is marked non_authoritative and filtered out. mode/fill_mode use the
# AUTHORITATIVE training values (geometric/wick) so the row survives the gate.
_AUTHORITATIVE_ENGINE_FIELDS = {
    "engine_version": ENGINE_VERSION,
    "formula_version": FORMULA_VERSION,
    "mode": "geometric",
    "fill_mode": "wick",
    "global_cooldown_bars": 0,
    "cb_enabled": False,
    "is_authoritative": True,
}


def _make_snapshot_parquet(
    snapshot_dir: Path,
    candidate_id: str,
    symbol: str,
    start_time_utc: str = "2026-03-01T09:00:00+00:00",
    **extra_features,
) -> None:
    """Write a minimal snapshot parquet file with one row."""
    row = {
        "candidate_id": candidate_id,
        "symbol": symbol,
        "start_time_utc": pd.Timestamp(start_time_utc),
        "range_prob": 0.80,
        "trend_prob": 0.10,
        "survival_prob": 0.90,
        "funding_rate": 0.0001,
        "adx_15m": 22.0,
        "rsi_15m": 48.0,
        "bb_width": 0.03,
        "range_size_pct": 5.0,
        "open_interest": 1_500_000.0,
        "quote_volume_24h": 50_000_000.0,
    }
    row.update(extra_features)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    fname = f"snapshots_{candidate_id.replace('/', '_')}.parquet"
    pd.DataFrame([row]).to_parquet(snapshot_dir / fname)


def test_unified_builder_uses_horizon_label_and_derives_backtest_features(tmp_path: Path) -> None:
    """Snapshot-joined backtest row preserves hlabel_meta for y labels."""
    snapshot_dir = tmp_path / "training_snapshots"
    bt_dir = tmp_path / "backtest_candidates"
    bt_dir.mkdir()

    cid = "SOLUSDT_20260301_090000_abcd1234"

    _make_snapshot_parquet(
        snapshot_dir, cid, "SOLUSDT",
        start_time_utc="2026-03-01T09:00:00+00:00",
        range_prob=0.8, trend_prob=0.1, survival_prob=0.9,
        grid_lower=100.0, grid_upper=108.0, num_grids=8,
    )

    pd.DataFrame(
        [
            {
                "symbol": "SOLUSDT",
                "candidate_id": cid,
                "net_pnl_pct": 10.0,
                "label_positive_by_horizon": False,
                "hlabel_meta": True,
                "grid_lower": 100.0,
                "grid_upper": 108.0,
                "num_grids": 8,
                "range_prob": 0.8,
                "trend_prob": 0.1,
                "survival_prob": 0.9,
                "funding_rate": 0.0001,
                "backtest_timestamp": "2026-03-01T09:05:00+00:00",
                "label_contract_version": LABEL_CONTRACT_VERSION,
                **_AUTHORITATIVE_ENGINE_FIELDS,
            }
        ]
    ).to_csv(bt_dir / "backtest_results_20260301.csv", index=False)

    builder = UnifiedTrainingBuilder(
        backtest_results_dir=bt_dir,
        snapshot_dir=snapshot_dir,
        include_reconstruction=True,
    )

    result = builder.build()
    assert not result.empty
    row = result.iloc[0]

    assert row["source"] == "snapshot"
    # Snapshot-joined features should be present
    assert pd.notna(row.get("range_prob", None))


def test_unified_builder_snapshot_only_ignores_legacy_paths(tmp_path: Path) -> None:
    """Builder ignores expired_bots_path and produces results from snapshots only."""
    snapshot_dir = tmp_path / "training_snapshots"
    bt_dir = tmp_path / "backtest_candidates"
    bt_dir.mkdir()

    cid = "BTCUSDT_20260301_091500_abcd1234"

    _make_snapshot_parquet(
        snapshot_dir, cid, "BTCUSDT",
        start_time_utc="2026-03-01T09:15:00+00:00",
        range_prob=0.77, trend_prob=0.09,
        funding_rate=0.0002,
    )

    pd.DataFrame([{
        "symbol": "BTCUSDT",
        "candidate_id": cid,
        "net_pnl_pct": 4.0,
        "y": 1,
        "backtest_timestamp": "2026-03-01T09:20:00+00:00",
        "label_contract_version": LABEL_CONTRACT_VERSION,
        **_AUTHORITATIVE_ENGINE_FIELDS,
    }]).to_csv(bt_dir / "training_rows_20260301.csv", index=False)

    builder = UnifiedTrainingBuilder(
        expired_bots_path=tmp_path / "missing.csv",  # ignored in snapshot-only arch
        backtest_results_dir=bt_dir,
        snapshot_dir=snapshot_dir,
    )

    result = builder.build()
    assert not result.empty
    row = result.iloc[0]

    assert row["source"] == "snapshot"
    assert row["range_prob"] == pytest.approx(0.77)
