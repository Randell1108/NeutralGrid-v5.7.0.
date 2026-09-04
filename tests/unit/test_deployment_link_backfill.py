from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neutralgrid.live.deployment_link_backfill import (
    LinkBackfillError,
    backfill_link_from_live_yaml,
)


def _write_live_yaml(path: Path, *, candidate_id: str = "PIPPINUSDT_20260516_134133_20cfa655") -> None:
    path.write_text(
        f"""
schema_version: 1
data_class: live_bot_telemetry
status: active_live_snapshot
candidate_id: {candidate_id}
ingestion:
  ingestion_date: "2026-05-16"
  latest_part_ingested_at: "2026-05-16T12:24:17-05:00"
  symbol: PIPPINUSDT
bot:
  symbol: PIPPINUSDT
  status: Working
  contract: Perp
  product: Futures Grid
  margin_mode: Isolated
  strategy_mode: Neutral
  leverage: 10
  strategy_id: "412048516"
  deploy_ts: "2026-05-16T15:01:18+00:00"
grid:
  price_range_lower: 0.02274
  price_range_upper: 0.02500
  num_grids: 10
  invested_margin_usdt: 7.50
  current_leverage: 10
  realized_profit_usdt: -0.01
pnl:
  invested_margin_usdt: 7.50
  total_profit_usdt: 0.00
position:
  symbol: PIPPINUSDT
risk:
  risk_ratio: 2.3
open_order_ladder:
  qty_per_order_base: 228
  last_price: 0.02439
  buy: []
  sell: []
advanced:
  stop_loss:
    pnl_usdt: -0.75
  take_profit:
    pnl_usdt: 1.12
""".lstrip(),
        encoding="utf-8",
    )


def _write_candidate_csv(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "symbol": "PIPPINUSDT",
                "candidate_id": "PIPPINUSDT_20260516_134133_20cfa655",
                "grid_lower": 0.022742,
                "grid_upper": 0.025007,
                "num_grids": 10,
                "leverage": 10,
                "grid_spacing_pct": 1.020651,
                "profit_per_grid_pct": 1.020651,
                "score": 88.0,
                "scan_score": 88.0,
                "ev_score": 1.2,
                "ev_24h": 0.4,
                "meta_prob": 0.55,
                "meta_prob_source": "scan",
                "deployment_score": 91.0,
                "pipeline_version": "test",
                "capital_fraction": 0.01,
                "capital_base_usdt": 750.0,
                "kelly_volatility_scale": 1.0,
                "grid_is_valid": True,
                "hard_gate_passed": True,
                "stage_b_approved": True,
            }
        ]
    ).to_csv(path, index=False)


def _write_geometry_candidate_csv(path: Path, *, candidate_id: str) -> None:
    pd.DataFrame(
        [
            {
                "symbol": "PIPPINUSDT",
                "candidate_id": candidate_id,
                "grid_lower": 0.022741,
                "grid_upper": 0.025001,
                "num_grids": 10,
                "leverage": 10,
                "grid_spacing_pct": 1.0607,
                "profit_per_grid_pct": 1.020651,
                "score": 88.0,
                "scan_score": 88.0,
                "ev_score": 1.2,
                "ev_24h": 0.4,
                "meta_prob": 0.55,
                "meta_prob_source": "scan",
                "deployment_score": 91.0,
                "pipeline_version": "test",
                "capital_fraction": 0.01,
                "capital_base_usdt": 750.0,
                "kelly_volatility_scale": 1.0,
                "grid_is_valid": True,
                "hard_gate_passed": True,
                "stage_b_approved": True,
            }
        ]
    ).to_csv(path, index=False)


def test_backfills_linkage_from_live_yaml(tmp_path: Path) -> None:
    live_yaml = tmp_path / "live_bot_data_scanner.yaml"
    results_dir = tmp_path / "results"
    linkage_dir = tmp_path / "linkage"
    results_dir.mkdir()
    _write_live_yaml(live_yaml)
    _write_candidate_csv(results_dir / "deployment_ready_20260516_134133.csv")

    result = backfill_link_from_live_yaml(
        live_yaml,
        results_dir=results_dir,
        linkage_dir=linkage_dir,
    )

    assert result.wrote_row is True
    link_df = pd.read_csv(linkage_dir / "deploy_linkage_log.csv")
    assert len(link_df) == 1
    row = link_df.iloc[0]
    assert row["candidate_id"] == "PIPPINUSDT_20260516_134133_20cfa655"
    assert str(row["strategy_id"]) == "412048516"
    assert row["deploy_time_utc"] == "2026-05-16T15:01:18+00:00"
    assert row["notes"].startswith("backfilled_from_live_yaml=")


def test_backfill_is_idempotent_for_same_strategy_and_candidate(tmp_path: Path) -> None:
    live_yaml = tmp_path / "live_bot_data_scanner.yaml"
    results_dir = tmp_path / "results"
    linkage_dir = tmp_path / "linkage"
    results_dir.mkdir()
    _write_live_yaml(live_yaml)
    _write_candidate_csv(results_dir / "deployment_ready_20260516_134133.csv")

    first = backfill_link_from_live_yaml(
        live_yaml,
        results_dir=results_dir,
        linkage_dir=linkage_dir,
    )
    second = backfill_link_from_live_yaml(
        live_yaml,
        results_dir=results_dir,
        linkage_dir=linkage_dir,
    )

    assert first.wrote_row is True
    assert second.wrote_row is False
    assert second.reason == "link already exists"
    assert len(pd.read_csv(linkage_dir / "deploy_linkage_log.csv")) == 1


def test_backfill_rejects_missing_candidate_row(tmp_path: Path) -> None:
    live_yaml = tmp_path / "live_bot_data_scanner.yaml"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_live_yaml(live_yaml)

    with pytest.raises(LinkBackfillError, match="candidate_id not found"):
        backfill_link_from_live_yaml(live_yaml, results_dir=results_dir)


def test_backfill_rejects_conflicting_strategy_link(tmp_path: Path) -> None:
    live_yaml = tmp_path / "live_bot_data_scanner.yaml"
    results_dir = tmp_path / "results"
    linkage_dir = tmp_path / "linkage"
    results_dir.mkdir()
    _write_live_yaml(live_yaml)
    _write_candidate_csv(results_dir / "deployment_ready_20260516_134133.csv")
    first = backfill_link_from_live_yaml(
        live_yaml,
        results_dir=results_dir,
        linkage_dir=linkage_dir,
    )
    assert first.wrote_row is True

    _write_live_yaml(live_yaml, candidate_id="PIPPINUSDT_20260516_134133_different")
    pd.DataFrame(
        [
                        {
                            "symbol": "PIPPINUSDT",
                            "candidate_id": "PIPPINUSDT_20260516_134133_different",
                            "grid_is_valid": True,
                            "hard_gate_passed": True,
                            "stage_b_approved": True,
            }
        ]
    ).to_csv(results_dir / "deployment_ready_20260516_134134.csv", index=False)

    with pytest.raises(LinkBackfillError, match="already linked"):
        backfill_link_from_live_yaml(
            live_yaml,
            results_dir=results_dir,
            linkage_dir=linkage_dir,
        )


def test_missing_candidate_id_still_requires_explicit_geometry_opt_in(tmp_path: Path) -> None:
    live_yaml = tmp_path / "live_bot_data_scanner.yaml"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_live_yaml(live_yaml, candidate_id="null")
    _write_geometry_candidate_csv(
        results_dir / "deployment_ready_20260516_134133.csv",
        candidate_id="PIPPINUSDT_20260516_134133_20cfa655",
    )

    with pytest.raises(LinkBackfillError, match="missing candidate_id"):
        backfill_link_from_live_yaml(live_yaml, results_dir=results_dir)


def test_geometry_backfill_for_missing_candidate_id_when_unique_and_causal(
    tmp_path: Path,
) -> None:
    live_yaml = tmp_path / "live_bot_data_scanner.yaml"
    results_dir = tmp_path / "results"
    linkage_dir = tmp_path / "linkage"
    results_dir.mkdir()
    _write_live_yaml(live_yaml, candidate_id="null")
    geometry_candidate_id = "PIPPINUSDT_20260516_134133_cf53cdb9"
    _write_geometry_candidate_csv(
        results_dir / "deployment_ready_20260516_134133.csv",
        candidate_id=geometry_candidate_id,
    )

    result = backfill_link_from_live_yaml(
        live_yaml,
        results_dir=results_dir,
        linkage_dir=linkage_dir,
        allow_geometry_match=True,
    )

    assert result.wrote_row is True
    assert result.candidate_id == geometry_candidate_id
    link_df = pd.read_csv(linkage_dir / "deploy_linkage_log.csv")
    assert link_df.iloc[0]["candidate_id"] == geometry_candidate_id
    assert "match_method=geometry" in link_df.iloc[0]["notes"]


def test_geometry_backfill_rejects_future_candidate_scan(tmp_path: Path) -> None:
    live_yaml = tmp_path / "live_bot_data_scanner.yaml"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_live_yaml(live_yaml, candidate_id="null")
    _write_geometry_candidate_csv(
        results_dir / "deployment_ready_20260516_160000.csv",
        candidate_id="PIPPINUSDT_20260516_160000_20cfa655",
    )

    with pytest.raises(LinkBackfillError, match="not found by geometry"):
        backfill_link_from_live_yaml(
            live_yaml,
            results_dir=results_dir,
            allow_geometry_match=True,
        )


def test_geometry_backfill_rejects_ambiguous_matches(tmp_path: Path) -> None:
    live_yaml = tmp_path / "live_bot_data_scanner.yaml"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_live_yaml(live_yaml, candidate_id="null")
    _write_geometry_candidate_csv(
        results_dir / "deployment_ready_20260516_134133.csv",
        candidate_id="PIPPINUSDT_20260516_134133_20cfa655",
    )
    _write_geometry_candidate_csv(
        results_dir / "deployment_ready_20260516_134134.csv",
        candidate_id="PIPPINUSDT_20260516_134134_30cfa655",
    )

    with pytest.raises(LinkBackfillError, match="geometry matched 2"):
        backfill_link_from_live_yaml(
            live_yaml,
            results_dir=results_dir,
            allow_geometry_match=True,
        )


def test_exact_backfill_finds_nested_pipeline_artifact(tmp_path: Path) -> None:
    live_yaml = tmp_path / "live_bot_data_scanner.yaml"
    results_dir = tmp_path / "results"
    nested_run = results_dir / "runs" / "pipeline_20260516_001"
    linkage_dir = tmp_path / "linkage"
    nested_run.mkdir(parents=True)
    _write_live_yaml(live_yaml)
    _write_candidate_csv(nested_run / "deployment_ready_20260516_134133.csv")

    result = backfill_link_from_live_yaml(
        live_yaml,
        results_dir=results_dir,
        linkage_dir=linkage_dir,
    )

    assert result.wrote_row is True
    assert result.deployment_ready_path.parent == nested_run


def test_exact_backfill_rejects_future_candidate_scan(tmp_path: Path) -> None:
    live_yaml = tmp_path / "live_bot_data_scanner.yaml"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    future_id = "PIPPINUSDT_20260516_160000_20cfa655"
    _write_live_yaml(live_yaml, candidate_id=future_id)
    _write_geometry_candidate_csv(
        results_dir / "deployment_ready_20260516_160000.csv",
        candidate_id=future_id,
    )

    with pytest.raises(LinkBackfillError, match="precedes candidate scan"):
        backfill_link_from_live_yaml(
            live_yaml,
            results_dir=results_dir,
            linkage_dir=tmp_path / "linkage",
        )


def test_exact_backfill_rejects_candidate_that_failed_terminal_gates(tmp_path: Path) -> None:
    live_yaml = tmp_path / "live_bot_data_scanner.yaml"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_live_yaml(live_yaml)
    candidate_path = results_dir / "deployment_ready_20260516_134133.csv"
    _write_candidate_csv(candidate_path)
    candidate_df = pd.read_csv(candidate_path)
    candidate_df["stage_b_approved"] = False
    candidate_df.to_csv(candidate_path, index=False)

    with pytest.raises(LinkBackfillError, match="terminal admission gates"):
        backfill_link_from_live_yaml(
            live_yaml,
            results_dir=results_dir,
            linkage_dir=tmp_path / "linkage",
        )
