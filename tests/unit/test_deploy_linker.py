"""Tests for neutralgrid.live.candidate_deploy_linker — deploy-time linkage logging."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from neutralgrid.core.candidate_id import make_candidate_id
from neutralgrid.live.candidate_deploy_linker import DeployLinker


@pytest.fixture
def tmp_linkage_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for linkage files."""
    d = tmp_path / "linkage"
    d.mkdir()
    return d


class TestDeployLinker:
    """Tests for the DeployLinker class."""

    def test_log_creates_csv(self, tmp_linkage_dir: Path):
        """First log_deployment creates the CSV with headers."""
        linker = DeployLinker(linkage_dir=tmp_linkage_dir)
        candidate_id = self._candidate_id()
        linker.log_deployment(
            candidate_id=candidate_id,
            strategy_id="ng_test_001",
            grid_lower=40000.0,
            grid_upper=42000.0,
            num_grids=30,
            leverage=10,
        )
        assert linker.path.exists()

        # Verify CSV structure
        with open(linker.path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["candidate_id"] == candidate_id
        assert rows[0]["strategy_id"] == "ng_test_001"

    @staticmethod
    def _candidate_id(
        *,
        symbol: str = "BTCUSDT",
        scan_ts: str = "20260227_143000",
        grid_lower: float = 40000.0,
        grid_upper: float = 42000.0,
        num_grids: int = 30,
        leverage: int = 10,
    ) -> str:
        return make_candidate_id(
            symbol,
            scan_ts,
            grid_lower=grid_lower,
            grid_upper=grid_upper,
            num_grids=num_grids,
            leverage=leverage,
        )

    def _log_valid(
        self,
        linker: DeployLinker,
        *,
        strategy_id: str,
        scan_ts: str = "20260227_143000",
    ) -> dict:
        return linker.log_deployment(
            self._candidate_id(scan_ts=scan_ts),
            strategy_id,
            deploy_time_utc="2026-02-27T15:00:00+00:00",
            grid_lower=40000.0,
            grid_upper=42000.0,
            num_grids=30,
            leverage=10,
        )

    def test_append_only(self, tmp_linkage_dir: Path):
        """Multiple log_deployment calls append rows, not overwrite."""
        linker = DeployLinker(linkage_dir=tmp_linkage_dir)
        self._log_valid(linker, strategy_id="SID1", scan_ts="20260227_143000")
        self._log_valid(linker, strategy_id="SID2", scan_ts="20260227_143100")
        self._log_valid(linker, strategy_id="SID3", scan_ts="20260227_143200")

        with open(linker.path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 3

    def test_deploy_time_utc_populated(self, tmp_linkage_dir: Path):
        """deploy_time_utc is auto-populated with current UTC time."""
        linker = DeployLinker(linkage_dir=tmp_linkage_dir)
        row = self._log_valid(linker, strategy_id="SID")
        assert row["deploy_time_utc"]
        assert "T" in row["deploy_time_utc"]  # ISO format

    def test_symbol_extracted_from_candidate_id(self, tmp_linkage_dir: Path):
        """Symbol is extracted from the candidate_id."""
        linker = DeployLinker(linkage_dir=tmp_linkage_dir)
        candidate_id = self._candidate_id(
            symbol="ETHUSDT",
            scan_ts="20260227_120000",
            grid_lower=2000.0,
            grid_upper=2200.0,
            num_grids=20,
            leverage=5,
        )
        row = linker.log_deployment(
            candidate_id,
            "SID",
            grid_lower=2000.0,
            grid_upper=2200.0,
            num_grids=20,
            leverage=5,
        )
        assert row["symbol"] == "ETHUSDT"
        assert row["scan_ts"] == "20260227_120000"
        assert row["config_hash"] == candidate_id.rsplit("_", 1)[-1]

    def test_unicode_symbol_generated_by_candidate_id_contract_is_accepted(
        self, tmp_linkage_dir: Path
    ) -> None:
        linker = DeployLinker(linkage_dir=tmp_linkage_dir)
        candidate_id = self._candidate_id(
            symbol="币安人生USDT",
            scan_ts="20260626_140532",
            grid_lower=0.1,
            grid_upper=0.2,
            num_grids=20,
            leverage=5,
        )

        row = linker.log_deployment(
            candidate_id,
            "SID_UNICODE",
            deploy_time_utc="2026-06-26T15:00:00+00:00",
            grid_lower=0.1,
            grid_upper=0.2,
            num_grids=20,
            leverage=5,
        )

        assert row["symbol"] == "币安人生USDT"
        assert row["candidate_id"] == candidate_id

    def test_load_all_returns_records(self, tmp_linkage_dir: Path):
        """load_all returns list of all logged records."""
        linker = DeployLinker(linkage_dir=tmp_linkage_dir)
        self._log_valid(linker, strategy_id="SID1", scan_ts="20260227_143000")
        self._log_valid(linker, strategy_id="SID2", scan_ts="20260227_143100")

        records = linker.load_all()
        assert len(records) == 2

    def test_load_all_empty_when_no_file(self, tmp_linkage_dir: Path):
        """load_all returns empty list when no CSV exists."""
        linker = DeployLinker(linkage_dir=tmp_linkage_dir)
        assert linker.load_all() == []

    def test_get_strategy_ids_for_candidate(self, tmp_linkage_dir: Path):
        """get_strategy_ids_for_candidate returns correct strategy_ids."""
        linker = DeployLinker(linkage_dir=tmp_linkage_dir)
        cid_a = self._candidate_id(scan_ts="20260227_143000")
        self._log_valid(linker, strategy_id="SID_1", scan_ts="20260227_143000")
        self._log_valid(linker, strategy_id="SID_2", scan_ts="20260227_143000")
        self._log_valid(linker, strategy_id="SID_3", scan_ts="20260227_143100")

        sids = linker.get_strategy_ids_for_candidate(cid_a)
        assert "SID_1" in sids
        assert "SID_2" in sids
        assert "SID_3" not in sids

    def test_get_candidate_id_for_strategy(self, tmp_linkage_dir: Path):
        """get_candidate_id_for_strategy returns correct candidate_id."""
        linker = DeployLinker(linkage_dir=tmp_linkage_dir)
        cid = self._candidate_id()
        self._log_valid(linker, strategy_id="SID_Y")

        assert linker.get_candidate_id_for_strategy("SID_Y") == cid
        assert linker.get_candidate_id_for_strategy("NONEXISTENT") is None

    def test_log_deployment_from_row(self, tmp_linkage_dir: Path):
        """log_deployment_from_row extracts fields from a dict-like row."""
        linker = DeployLinker(linkage_dir=tmp_linkage_dir)
        candidate_id = self._candidate_id(
            symbol="SOLUSDT",
            scan_ts="20260227_100000",
            grid_lower=80.0,
            grid_upper=120.0,
            num_grids=25,
            leverage=5,
        )
        candidate_row = {
            "candidate_id": candidate_id,
            "grid_lower": 80.0,
            "grid_upper": 120.0,
            "num_grids": 25,
            "leverage": 5,
            "grid_spacing_pct": 1.6,
            "profit_per_grid_pct": 0.45,
            "score": 85.0,
            "ev_score": 2.5,
            "ev_24h": 3.4,
            "meta_prob": 0.72,
            "meta_prob_source": "enrich",
            "deployment_score": 97.0,
        }
        row = linker.log_deployment_from_row(candidate_row, "ng_sol_001", pipeline_version="6.5.6")
        assert row["candidate_id"] == candidate_id
        assert row["pipeline_version"] == "6.5.6"
        assert row["scan_score"] == pytest.approx(85.0)
        assert row["ev_24h"] == pytest.approx(3.4)
        assert row["meta_prob_source"] == "enrich"
        assert row["deployment_score"] == pytest.approx(97.0)

    def test_config_fields_preserved(self, tmp_linkage_dir: Path):
        """Grid config fields are stored in the CSV."""
        linker = DeployLinker(linkage_dir=tmp_linkage_dir)
        candidate_id = self._candidate_id(
            grid_lower=1000.0,
            grid_upper=2000.0,
            num_grids=50,
            leverage=20,
        )
        linker.log_deployment(
            candidate_id,
            "SID",
            grid_lower=1000.0,
            grid_upper=2000.0,
            num_grids=50,
            leverage=20,
        )
        records = linker.load_all()
        assert float(records[0]["grid_lower"]) == 1000.0
        assert int(float(records[0]["num_grids"])) == 50

    @pytest.mark.parametrize(
        ("candidate_id", "strategy_id", "message"),
        [
            ("", "SID", "candidate_id"),
            ("not-a-candidate", "SID", "candidate_id"),
            ("BTCUSDT_20260227_143000_a1b2c3d4", "", "strategy_id"),
            ("BTCUSDT_20260227_143000_a1b2c3d4", "nan", "strategy_id"),
        ],
    )
    def test_rejects_missing_or_malformed_identity(
        self,
        tmp_linkage_dir: Path,
        candidate_id: str,
        strategy_id: str,
        message: str,
    ) -> None:
        linker = DeployLinker(linkage_dir=tmp_linkage_dir)
        with pytest.raises(ValueError, match=message):
            linker.log_deployment(
                candidate_id,
                strategy_id,
                grid_lower=40000.0,
                grid_upper=42000.0,
                num_grids=30,
                leverage=10,
            )

    def test_rejects_candidate_config_hash_mismatch(self, tmp_linkage_dir: Path) -> None:
        linker = DeployLinker(linkage_dir=tmp_linkage_dir)
        candidate_id = self._candidate_id()

        with pytest.raises(ValueError, match="config hash"):
            linker.log_deployment(
                candidate_id,
                "SID",
                grid_lower=39000.0,
                grid_upper=42000.0,
                num_grids=30,
                leverage=10,
            )

    def test_rejects_deployment_before_candidate_scan(self, tmp_linkage_dir: Path) -> None:
        linker = DeployLinker(linkage_dir=tmp_linkage_dir)
        with pytest.raises(ValueError, match="precedes candidate scan"):
            linker.log_deployment(
                self._candidate_id(),
                "SID",
                deploy_time_utc="2026-02-27T14:00:00+00:00",
                grid_lower=40000.0,
                grid_upper=42000.0,
                num_grids=30,
                leverage=10,
            )

    def test_same_strategy_candidate_pair_is_idempotent(self, tmp_linkage_dir: Path) -> None:
        linker = DeployLinker(linkage_dir=tmp_linkage_dir)
        self._log_valid(linker, strategy_id="SID")
        self._log_valid(linker, strategy_id="SID")

        assert len(linker.load_all()) == 1

    def test_strategy_cannot_link_to_different_candidates(self, tmp_linkage_dir: Path) -> None:
        linker = DeployLinker(linkage_dir=tmp_linkage_dir)
        self._log_valid(linker, strategy_id="SID", scan_ts="20260227_143000")

        with pytest.raises(ValueError, match="already linked"):
            self._log_valid(linker, strategy_id="SID", scan_ts="20260227_143100")
