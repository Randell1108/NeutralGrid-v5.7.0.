"""Tests for neutralgrid.training.live_outcome_ingestor — live outcome ingestion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neutralgrid.training.live_outcome_ingestor import LiveOutcomeIngestor


@pytest.fixture
def setup_data(tmp_path: Path) -> dict:
    """Create minimal expired_bots, linkage, and scanner CSV fixtures."""
    # -- Expired bots CSV
    bots_path = tmp_path / "expired_bots.csv"
    bots_df = pd.DataFrame([
        {
            "strategy_id": "ng_001",
            "symbol": "BTCUSDT",
            "pnl_pct": 5.2,
            "total_profit_usdt": 20.8,
            "duration_hours": 8.5,
            "total_trades": 42,
            "grids_count": 30,
            "invested_margin_usdt": 400,
            "leverage": 10,
            "price_range_low": 40000.0,
            "price_range_high": 42000.0,
            "grid_spacing_pct": 0.67,
            "start_time_utc": "2026-02-20 10:00:00+00:00",
            "end_time_utc": "2026-02-20 18:30:00+00:00",
            "realized_pnl_usdt": 22.0,
            "commission_usdt": 1.2,
            "funding_fee_usdt": 0.0,
            "mae": 18.0,
            "mfe": 27.0,
            "mae_pct_initial": 4.5,
            "mfe_pct_initial": 6.75,
            "profit_factor": 1.5,
            "maker_count": 30,
            "taker_count": 12,
            "unrealized_pnl_usdt": -1.2,
            "trend_structure": "range",
        },
        {
            "strategy_id": "ng_002",
            "symbol": "ETHUSDT",
            "pnl_pct": -2.1,
            "total_profit_usdt": -8.4,
            "duration_hours": 12.0,
            "total_trades": 15,
            "grids_count": 20,
            "invested_margin_usdt": 400,
            "leverage": 10,
            "price_range_low": 3000.0,
            "price_range_high": 3500.0,
            "grid_spacing_pct": 2.5,
            "start_time_utc": "2026-02-21 08:00:00+00:00",
            "end_time_utc": "2026-02-21 20:00:00+00:00",
            "realized_pnl_usdt": -6.0,
            "commission_usdt": 2.4,
            "funding_fee_usdt": 0.0,
            "mae": 26.0,
            "mfe": 9.0,
            "mae_pct_initial": 6.5,
            "mfe_pct_initial": 2.25,
            "profit_factor": 0.8,
            "maker_count": 10,
            "taker_count": 5,
            "unrealized_pnl_usdt": -2.4,
            "trend_structure": "trending",
        },
    ])
    bots_df.to_csv(bots_path, index=False)

    # -- Linkage CSV
    linkage_dir = tmp_path / "linkage"
    linkage_dir.mkdir()
    linkage_path = linkage_dir / "deploy_linkage_log.csv"
    linkage_df = pd.DataFrame([
        {
            "candidate_id": "BTCUSDT_20260220_080000_aabbccdd",
            "strategy_id": "ng_001",
            "deploy_time_utc": "2026-02-20T09:00:00+00:00",
            "symbol": "BTCUSDT",
            "scan_ts": "20260220_080000",
            "config_hash": "aabbccdd",
            "grid_lower": 40000.0,
            "grid_upper": 42000.0,
            "num_grids": 30,
            "leverage": 10,
        },
    ])
    linkage_df.to_csv(linkage_path, index=False)

    # -- Scanner CSV
    scanner_dir = tmp_path / "results"
    scanner_dir.mkdir()
    scanner_csv = scanner_dir / "deployment_ready_20260220_080000.csv"
    scanner_df = pd.DataFrame([
        {
            "symbol": "BTCUSDT",
            "score": 85.0,
            "grid_is_valid": True,
            "grid_lower": 40000.0,
            "grid_upper": 42000.0,
            "num_grids": 30,
            "leverage": 10,
            "range_prob": 0.82,
            "trend_prob": 0.15,
            "adx_1h": 18.5,
            "adx_15m": 22.0,
            "rsi_15m": 48.0,
            "ev_score": 3.2,
            "ev_24h": 4.1,
            "meta_prob": 0.72,
            "meta_prob_source": "enrich",
            "deployment_score": 98.0,
            "funding_rate": 0.0001,
            "survival_prob": 0.9,
            "hurst_exponent": 0.45,
            "ou_halflife": 6.0,
            "profit_per_grid_pct": 0.45,
            "range_size_pct": 5.0,
        },
    ])
    scanner_df.to_csv(scanner_csv, index=False)

    return {
        "bots_path": bots_path,
        "linkage_dir": linkage_dir,
        "scanner_dir": scanner_dir,
        "tmp_path": tmp_path,
    }


class TestLiveOutcomeIngestor:
    """Tests for the LiveOutcomeIngestor class."""

    def test_ingest_returns_dataframe(self, setup_data):
        """ingest() returns a non-empty DataFrame."""
        d = setup_data
        ingestor = LiveOutcomeIngestor(
            expired_bots_path=d["bots_path"],
            linkage_dir=d["linkage_dir"],
            scanner_results_dir=d["scanner_dir"],
            min_bot_date="2026-02-01",
        )
        result = ingestor.ingest()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_linkage_match_has_candidate_id(self, setup_data):
        """Bot matched via linkage has non-empty candidate_id."""
        d = setup_data
        ingestor = LiveOutcomeIngestor(
            expired_bots_path=d["bots_path"],
            linkage_dir=d["linkage_dir"],
            scanner_results_dir=d["scanner_dir"],
            min_bot_date="2026-02-01",
        )
        result = ingestor.ingest()
        btc_row = result[result["symbol"] == "BTCUSDT"].iloc[0]
        assert btc_row["candidate_id"] == "BTCUSDT_20260220_080000_aabbccdd"
        assert btc_row["match_method"] == "linkage"

    def test_unlinked_bot_uses_forensic_or_unmatched(self, setup_data):
        """Bot without linkage is either forensically matched or unmatched."""
        d = setup_data
        ingestor = LiveOutcomeIngestor(
            expired_bots_path=d["bots_path"],
            linkage_dir=d["linkage_dir"],
            scanner_results_dir=d["scanner_dir"],
            min_bot_date="2026-02-01",
        )
        result = ingestor.ingest()
        eth_row = result[result["symbol"] == "ETHUSDT"].iloc[0]
        # ETHUSDT has no linkage record — should be forensic or unmatched
        assert eth_row["match_method"] in ("forensic", "unmatched")

    def test_scan_features_populated_for_linked_bot(self, setup_data):
        """Linked bot has scan-time features prefixed with scan_."""
        d = setup_data
        ingestor = LiveOutcomeIngestor(
            expired_bots_path=d["bots_path"],
            linkage_dir=d["linkage_dir"],
            scanner_results_dir=d["scanner_dir"],
            min_bot_date="2026-02-01",
        )
        result = ingestor.ingest()
        btc_row = result[result["symbol"] == "BTCUSDT"].iloc[0]
        # At least score should be present
        assert "scan_score" in btc_row.index or "scan_range_prob" in btc_row.index
        assert btc_row["scan_ev_24h"] == pytest.approx(4.1)
        assert btc_row["scan_deployment_score"] == pytest.approx(98.0)
        assert btc_row["scan_meta_prob_source"] == "enrich"

    def test_conflicting_strategy_links_fail_closed(self, setup_data):
        """One strategy linked to two candidates must not become training identity."""
        d = setup_data
        linkage_path = d["linkage_dir"] / "deploy_linkage_log.csv"
        linkage_df = pd.read_csv(linkage_path)
        linkage_df = pd.concat(
            [
                linkage_df,
                pd.DataFrame(
                    [
                        {
                            "candidate_id": "BTCUSDT_20260220_090000_ddddeeee",
                            "strategy_id": "ng_001",
                            "deploy_time_utc": "2026-02-20T09:30:00+00:00",
                            "symbol": "BTCUSDT",
                            "scan_ts": "20260220_090000",
                            "config_hash": "ddddeeee",
                            "grid_lower": 40100.0,
                            "grid_upper": 42100.0,
                            "num_grids": 30,
                            "leverage": 10,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        linkage_df.to_csv(linkage_path, index=False)

        scanner_csv = d["scanner_dir"] / "deployment_ready_20260220_090000.csv"
        pd.DataFrame(
            [
                {
                    "symbol": "BTCUSDT",
                    "candidate_id": "BTCUSDT_20260220_090000_ddddeeee",
                    "score": 91.0,
                    "grid_is_valid": True,
                    "grid_lower": 40100.0,
                    "grid_upper": 42100.0,
                    "num_grids": 30,
                    "leverage": 10,
                    "ev_24h": 9.9,
                    "meta_prob": 0.81,
                    "meta_prob_source": "latest",
                    "deployment_score": 99.0,
                }
            ]
        ).to_csv(scanner_csv, index=False)

        ingestor = LiveOutcomeIngestor(
            expired_bots_path=d["bots_path"],
            linkage_dir=d["linkage_dir"],
            scanner_results_dir=d["scanner_dir"],
        )
        result = ingestor.ingest()
        btc_row = result[result["symbol"] == "BTCUSDT"].iloc[0]
        assert btc_row["candidate_id"] == ""
        assert btc_row["match_method"] == "linkage_conflict"

    def test_linkage_scan_features_use_exact_candidate_id(self, setup_data):
        """Same-symbol rows in one scan must not leak features from the wrong candidate."""
        d = setup_data
        scanner_csv = d["scanner_dir"] / "deployment_ready_20260220_080000.csv"
        original = pd.read_csv(scanner_csv).iloc[0].to_dict()
        wrong = dict(original)
        wrong["candidate_id"] = "BTCUSDT_20260220_080000_wrong000"
        wrong["ev_24h"] = -99.0
        wrong["meta_prob"] = 0.01
        wrong["meta_prob_source"] = "wrong"
        correct = dict(original)
        correct["candidate_id"] = "BTCUSDT_20260220_080000_aabbccdd"
        correct["ev_24h"] = 4.1
        correct["meta_prob"] = 0.72
        correct["meta_prob_source"] = "enrich"
        pd.DataFrame([wrong, correct]).to_csv(scanner_csv, index=False)

        ingestor = LiveOutcomeIngestor(
            expired_bots_path=d["bots_path"],
            linkage_dir=d["linkage_dir"],
            scanner_results_dir=d["scanner_dir"],
        )
        result = ingestor.ingest()
        btc_row = result[result["symbol"] == "BTCUSDT"].iloc[0]
        assert btc_row["candidate_id"] == "BTCUSDT_20260220_080000_aabbccdd"
        assert btc_row["scan_ev_24h"] == pytest.approx(4.1)
        assert btc_row["scan_meta_prob"] == pytest.approx(0.72)
        assert btc_row["scan_meta_prob_source"] == "enrich"

    def test_source_is_live(self, setup_data):
        """All rows have source='live'."""
        d = setup_data
        ingestor = LiveOutcomeIngestor(
            expired_bots_path=d["bots_path"],
            linkage_dir=d["linkage_dir"],
            scanner_results_dir=d["scanner_dir"],
        )
        result = ingestor.ingest()
        assert (result["source"] == "live").all()

    def test_outcome_fields_preserved(self, setup_data):
        """Live outcome fields (pnl, trades, etc.) are in the output."""
        d = setup_data
        ingestor = LiveOutcomeIngestor(
            expired_bots_path=d["bots_path"],
            linkage_dir=d["linkage_dir"],
            scanner_results_dir=d["scanner_dir"],
        )
        result = ingestor.ingest()
        btc_row = result[result["symbol"] == "BTCUSDT"].iloc[0]
        assert btc_row["pnl_pct"] == pytest.approx(5.2)
        assert btc_row["total_trades"] == 42
        assert btc_row["mae"] == pytest.approx(18.0)
        assert btc_row["mfe"] == pytest.approx(27.0)

    def test_empty_bots_returns_empty(self, tmp_path: Path):
        """Returns empty DataFrame when no bots file exists."""
        ingestor = LiveOutcomeIngestor(
            expired_bots_path=tmp_path / "nonexistent.csv",
        )
        result = ingestor.ingest()
        assert result.empty

    def test_no_linkage_file_still_works(self, setup_data):
        """Ingestion works without a linkage file (forensic matching only)."""
        d = setup_data
        # Delete linkage file
        (d["linkage_dir"] / "deploy_linkage_log.csv").unlink()

        ingestor = LiveOutcomeIngestor(
            expired_bots_path=d["bots_path"],
            linkage_dir=d["linkage_dir"],
            scanner_results_dir=d["scanner_dir"],
        )
        result = ingestor.ingest()
        assert len(result) == 2
        # All should be forensic or unmatched
        assert all(m in ("forensic", "unmatched") for m in result["match_method"])

    def test_nested_deployment_artifact_is_discovered(self, setup_data):
        d = setup_data
        original = d["scanner_dir"] / "deployment_ready_20260220_080000.csv"
        nested = d["scanner_dir"] / "runs" / "pipeline_001"
        nested.mkdir(parents=True)
        original.replace(nested / original.name)

        ingestor = LiveOutcomeIngestor(
            expired_bots_path=d["bots_path"],
            linkage_dir=d["linkage_dir"],
            scanner_results_dir=d["scanner_dir"],
        )
        result = ingestor.ingest()
        btc_row = result[result["symbol"] == "BTCUSDT"].iloc[0]

        assert btc_row["match_method"] == "linkage"
        assert btc_row["scan_ev_24h"] == pytest.approx(4.1)

    def test_geometry_backfill_note_is_not_treated_as_direct_linkage(self, setup_data):
        d = setup_data
        linkage_path = d["linkage_dir"] / "deploy_linkage_log.csv"
        linkage_df = pd.read_csv(linkage_path)
        linkage_df["notes"] = "match_method=geometry"
        linkage_df.to_csv(linkage_path, index=False)

        ingestor = LiveOutcomeIngestor(
            expired_bots_path=d["bots_path"],
            linkage_dir=d["linkage_dir"],
            scanner_results_dir=d["scanner_dir"],
        )
        result = ingestor.ingest()
        btc_row = result[result["symbol"] == "BTCUSDT"].iloc[0]

        assert btc_row["match_method"] != "linkage"
