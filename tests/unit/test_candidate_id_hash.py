"""Tests for neutralgrid.core.candidate_id — stable candidate_id generation."""

from __future__ import annotations

import pytest
from neutralgrid.core.candidate_id import (
    make_candidate_id,
    make_candidate_id_from_row,
    extract_parts,
    _config_hash,
)


class TestConfigHash:
    """Tests for the deterministic config hash function."""

    def test_deterministic(self):
        """Same inputs always produce the same hash."""
        fields = {"grid_lower": 40000.0, "grid_upper": 42000.0, "num_grids": 30, "leverage": 10}
        h1 = _config_hash(fields)
        h2 = _config_hash(fields)
        assert h1 == h2

    def test_length_is_8(self):
        """Hash is exactly 8 hex characters by default."""
        fields = {"grid_lower": 1.0, "grid_upper": 2.0, "num_grids": 10, "leverage": 5}
        h = _config_hash(fields)
        assert len(h) == 8
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_configs_different_hash(self):
        """Different grid configs produce different hashes."""
        fields_a = {"grid_lower": 40000.0, "grid_upper": 42000.0, "num_grids": 30, "leverage": 10}
        fields_b = {"grid_lower": 40000.0, "grid_upper": 43000.0, "num_grids": 30, "leverage": 10}
        assert _config_hash(fields_a) != _config_hash(fields_b)

    def test_missing_fields_handled(self):
        """Missing or None fields produce a valid hash."""
        fields = {"grid_lower": None, "grid_upper": None}
        h = _config_hash(fields)
        assert len(h) == 8

    def test_nan_treated_as_missing(self):
        """NaN values are treated as missing (same hash as None)."""
        fields_nan = {"grid_lower": float("nan"), "grid_upper": float("nan")}
        fields_none = {"grid_lower": None, "grid_upper": None}
        assert _config_hash(fields_nan) == _config_hash(fields_none)


class TestMakeCandidateId:
    """Tests for the main candidate_id builder."""

    def test_format_with_hash(self):
        """candidate_id has format SYMBOL_YYYYMMDD_HHMMSS_hash8."""
        cid = make_candidate_id(
            "BTCUSDT", "20260227_143000",
            grid_lower=40000.0, grid_upper=42000.0, num_grids=30, leverage=10,
        )
        parts = cid.split("_")
        assert len(parts) == 4
        assert parts[0] == "BTCUSDT"
        assert parts[1] == "20260227"
        assert parts[2] == "143000"
        assert len(parts[3]) == 8

    def test_same_symbol_different_config_different_id(self):
        """Same symbol + time but different config → different candidate_id."""
        cid_a = make_candidate_id(
            "ETHUSDT", "20260227_120000",
            grid_lower=3000.0, grid_upper=3500.0, num_grids=20, leverage=10,
        )
        cid_b = make_candidate_id(
            "ETHUSDT", "20260227_120000",
            grid_lower=3000.0, grid_upper=3500.0, num_grids=25, leverage=10,
        )
        assert cid_a != cid_b

    def test_same_config_same_id(self):
        """Same symbol + time + config → same candidate_id."""
        kwargs = dict(
            symbol="SOLUSDT", scan_ts="20260227_100000",
            grid_lower=100.0, grid_upper=120.0, num_grids=15, leverage=5,
        )
        assert make_candidate_id(**kwargs) == make_candidate_id(**kwargs)

    def test_config_fields_dict(self):
        """config_fields dict overrides individual params."""
        cid = make_candidate_id(
            "BTCUSDT", "20260227_143000",
            config_fields={"grid_lower": 40000.0, "grid_upper": 42000.0, "num_grids": 30, "leverage": 10},
        )
        assert "BTCUSDT" in cid
        assert "20260227_143000" in cid


class TestMakeCandidateIdFromRow:
    """Tests for the DataFrame-row convenience wrapper."""

    def test_from_row(self):
        """Builds correct id from a dict-like row."""
        row = {
            "symbol": "LINKUSDT",
            "grid_lower": 10.5,
            "grid_upper": 15.0,
            "num_grids": 20,
            "leverage": 10,
        }
        cid = make_candidate_id_from_row(row, "20260227_150000")
        assert cid.startswith("LINKUSDT_20260227_150000_")
        assert len(cid.split("_")) == 4


class TestExtractParts:
    """Tests for parsing candidate_id back into components."""

    def test_extract_new_format(self):
        """Parse SYMBOL_TS_HASH into parts."""
        parts = extract_parts("BTCUSDT_20260227_143000_a1b2c3d4")
        assert parts["symbol"] == "BTCUSDT"
        assert parts["scan_ts"] == "20260227_143000"
        assert parts["config_hash"] == "a1b2c3d4"

    def test_extract_legacy_format(self):
        """Parse SYMBOL_TS (no hash) gracefully."""
        parts = extract_parts("ETHUSDT_20260215_170412")
        assert parts["symbol"] == "ETHUSDT"
        assert parts["scan_ts"] == "20260215_170412"
        assert parts["config_hash"] == ""

    def test_extract_minimal(self):
        """Handle bare symbol gracefully."""
        parts = extract_parts("BTCUSDT")
        assert parts["symbol"] == "BTCUSDT"
        assert parts["scan_ts"] == ""
        assert parts["config_hash"] == ""
