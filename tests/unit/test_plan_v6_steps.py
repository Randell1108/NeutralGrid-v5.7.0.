"""
Tests for Plan v6.0 implementation steps.

Covers:
- Step 1: Label target precedence (y > y_horizon > label_positive_by_horizon)
- Step 2: Glob pattern for training_data_*.csv
- Step 3: Provenance columns in convert_to_training_row
- Step 4: Version constants from single source of truth
- Step 6: Ingestion gate (version_gated flag)
- Step 9: Physics lock warnings
- Step 10: Runner-derived is_authoritative
- Step 13: Funding rate sign preservation
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

# Ensure backtest/ is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from neutralgrid.core.constants import (
    ENGINE_VERSION,
    FORMULA_VERSION,
    LABEL_CONTRACT_VERSION,
)


class TestStep1LabelPrecedence:
    """Step 1: Target precedence — y (hierarchical) takes priority."""

    def test_y_preferred_over_y_horizon(self) -> None:
        """When both y and y_horizon exist and y is NON-degenerate, y wins.

        PIPELINE_FIX Step 1 (ERR-021): the precedence loop now requires
        `nunique() > 1 AND positive_rate >= 0.05` before accepting a
        candidate column (mirroring the existing hlabel_meta path at
        unified_training_builder.py:793-815). A single-row fixture is
        structurally degenerate for ANY classifier and would correctly
        fall through to `net_pnl_pct >= meta_hurdle_pct` regardless of
        which column was "intended" — so the fixture is now multi-row
        with a real positive-class signal in `y`.
        """
        from neutralgrid.training.unified_training_builder import (
            UnifiedTrainingBuilder,
        )
        from neutralgrid.training.data_generator import LabelConfig

        builder = UnifiedTrainingBuilder(
            expired_bots_path=Path("data/new_expired_bots.xlsx"),
            backtest_results_dir=Path("data/backtest_candidates"),
            label_config=LabelConfig(),
        )

        # 10 rows: y is non-degenerate (3 ones + 7 zeros = 30% positive),
        # y_horizon disagrees (would lift the same rows AND extras to 1).
        # Plan v6 Step 1 says y wins; verify y values are preserved.
        df = pd.DataFrame({
            "symbol": ["TESTUSDT"] * 10,
            "net_pnl_pct": [4.0] * 10,
            # Hierarchical strict: only 3 cleared all hierarchical levels.
            "y":         [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            # Lenient horizon: 7 cleared the simpler horizon hurdle.
            "y_horizon": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        })

        result = builder._normalize_backtest_targets(df)
        # y (strict, 3 positives) preserved; NOT overwritten by y_horizon (7 positives).
        assert (result["label_source"] == "y").all(), (
            f"Expected label_source=y, got {result['label_source'].unique().tolist()}"
        )
        assert int(result["y"].sum()) == 3, (
            f"Expected y.sum()=3 (hierarchical strict count preserved), "
            f"got {int(result['y'].sum())}"
        )

    def test_y_horizon_used_when_y_absent(self) -> None:
        """When y is missing, y_horizon is used."""
        from neutralgrid.training.unified_training_builder import (
            UnifiedTrainingBuilder,
        )
        from neutralgrid.training.data_generator import LabelConfig

        builder = UnifiedTrainingBuilder(
            expired_bots_path=Path("data/new_expired_bots.xlsx"),
            backtest_results_dir=Path("data/backtest_candidates"),
            label_config=LabelConfig(),
        )

        df = pd.DataFrame({
            "symbol": ["TESTUSDT"],
            "net_pnl_pct": [4.0],
            "y_horizon": [1],
        })

        result = builder._normalize_backtest_targets(df)
        assert int(result["y"].iloc[0]) == 1


class TestStep3Provenance:
    """Step 3: Provenance columns in convert_to_training_row."""

    def test_provenance_columns_propagated(self) -> None:
        """engine_version, label_contract_version, backtest_run_id appear in output."""
        from neutralgrid.backtest.candidate_pipeline import convert_to_training_row

        backtest_result: Dict[str, Any] = {
            "symbol": "TESTUSDT",
            "net_pnl_pct": 5.0,
            "label_positive_by_horizon": True,
            "max_drawdown_pct": 2.0,
            "total_trades": 10,
            "duration_hours": 6.0,
            "time_in_range_pct": 80.0,
            "liquidated": False,
            "fees_paid": 1.5,
            "funding_fees": 0.3,
            "price_end": 100.0,
            "engine_version": "realistic-v7",
            "label_contract_version": LABEL_CONTRACT_VERSION,
            "backtest_run_id": "test-uuid-123",
            "formula_version": "alignment-v1",
            "is_authoritative": True,
        }
        candidate_row = {
            "symbol": "TESTUSDT",
            "candidate_id": "TESTUSDT_20260315_120000",
        }

        row = convert_to_training_row(backtest_result, candidate_row)
        assert row["engine_version"] == "realistic-v7"
        assert row["label_contract_version"] == LABEL_CONTRACT_VERSION
        assert row["backtest_run_id"] == "test-uuid-123"
        assert row["formula_version"] == "alignment-v1"
        assert row["is_authoritative"] is True


class TestStep4Constants:
    """Step 4: Version constants from single source of truth."""

    def test_constants_module_exists(self) -> None:
        from neutralgrid.core.constants import (
            ENGINE_VERSION,
            FORMULA_VERSION,
            LABEL_CONTRACT_VERSION,
        )
        # Version constants bump over time; assert they are exported as
        # non-empty strings rather than freezing brittle literals. Cross-module
        # consistency is enforced by test_btk_label_contract_uses_same_versions
        # and test_btk_unified_runner_uses_same_engine_version below.
        assert isinstance(LABEL_CONTRACT_VERSION, str) and LABEL_CONTRACT_VERSION
        assert isinstance(FORMULA_VERSION, str) and FORMULA_VERSION
        assert isinstance(ENGINE_VERSION, str) and ENGINE_VERSION

    def test_btk_label_contract_uses_same_versions(self) -> None:
        from backtest.btk_label_contract import (
            FORMULA_VERSION,
            LABEL_CONTRACT_VERSION,
        )
        from neutralgrid.core.constants import (
            FORMULA_VERSION as CONST_FORMULA,
            LABEL_CONTRACT_VERSION as CONST_CONTRACT,
        )
        assert LABEL_CONTRACT_VERSION == CONST_CONTRACT
        assert FORMULA_VERSION == CONST_FORMULA

    def test_btk_unified_runner_uses_same_engine_version(self) -> None:
        from backtest.btk_unified_runner import ENGINE_VERSION
        from neutralgrid.core.constants import ENGINE_VERSION as CONST_ENGINE
        assert ENGINE_VERSION == CONST_ENGINE


class TestStep6IngestionGate:
    """Step 6: Ingestion gate marks version-mismatched rows."""

    def test_matching_version_not_gated(self) -> None:
        from neutralgrid.training.unified_training_builder import (
            UnifiedTrainingBuilder,
        )
        from neutralgrid.training.data_generator import LabelConfig

        builder = UnifiedTrainingBuilder(
            expired_bots_path=Path("data/new_expired_bots.xlsx"),
            backtest_results_dir=Path("data/backtest_candidates"),
            label_config=LabelConfig(),
        )

        df = pd.DataFrame({
            "symbol": ["TESTUSDT"],
            "label_contract_version": [LABEL_CONTRACT_VERSION],
            # The ingestion gate now validates the full engine-settings
            # contract; supply passing values so the only variable under test
            # is the (matching) label_contract_version.
            "engine_version": [ENGINE_VERSION],
            "formula_version": [FORMULA_VERSION],
            "mode": ["geometric"],
            "fill_mode": ["wick"],
            "global_cooldown_bars": [0.0],
            "cb_enabled": [False],
            "is_authoritative": [True],
        })

        result = builder._apply_ingestion_gate(df)
        assert not result["version_gated"].iloc[0]

    def test_mismatched_version_gated(self) -> None:
        from neutralgrid.training.unified_training_builder import (
            UnifiedTrainingBuilder,
        )
        from neutralgrid.training.data_generator import LabelConfig

        builder = UnifiedTrainingBuilder(
            expired_bots_path=Path("data/new_expired_bots.xlsx"),
            backtest_results_dir=Path("data/backtest_candidates"),
            label_config=LabelConfig(),
        )

        df = pd.DataFrame({
            "symbol": ["TESTUSDT"],
            "label_contract_version": ["2025-01-01"],
        })

        result = builder._apply_ingestion_gate(df)
        assert result["version_gated"].iloc[0]

    def test_arithmetic_mode_gated(self) -> None:
        """Backtest validates arithmetic mode, but the authoritative training
        pool is geometric-only. An arithmetic-mode row (all other gates
        passing) must be gated as non_authoritative: valid for backtest, not
        authoritative for training (Grid Mode Authority, 2026-05-22).
        """
        from neutralgrid.training.unified_training_builder import (
            UnifiedTrainingBuilder,
        )
        from neutralgrid.training.data_generator import LabelConfig

        builder = UnifiedTrainingBuilder(
            expired_bots_path=Path("data/new_expired_bots.xlsx"),
            backtest_results_dir=Path("data/backtest_candidates"),
            label_config=LabelConfig(),
        )

        df = pd.DataFrame({
            "symbol": ["TESTUSDT"],
            "label_contract_version": [LABEL_CONTRACT_VERSION],
            "engine_version": [ENGINE_VERSION],
            "formula_version": [FORMULA_VERSION],
            "mode": ["arithmetic"],  # valid backtest mode, but not geometric
            "fill_mode": ["wick"],
            "global_cooldown_bars": [0.0],
            "cb_enabled": [False],
            "is_authoritative": [True],
        })

        result = builder._apply_ingestion_gate(df)
        assert result["version_gated"].iloc[0]
        assert result["source_class"].iloc[0] == "non_authoritative"
        assert "mode" in str(result["version_gate_reason"].iloc[0])

    def test_missing_version_marked_legacy(self) -> None:
        from neutralgrid.training.unified_training_builder import (
            UnifiedTrainingBuilder,
        )
        from neutralgrid.training.data_generator import LabelConfig

        builder = UnifiedTrainingBuilder(
            expired_bots_path=Path("data/new_expired_bots.xlsx"),
            backtest_results_dir=Path("data/backtest_candidates"),
            label_config=LabelConfig(),
        )

        df = pd.DataFrame({
            "symbol": ["TESTUSDT"],
            "label_contract_version": [None],
            # Pass every other gate so the only failure under test is the
            # missing label_contract_version (legacy classification).
            "engine_version": [ENGINE_VERSION],
            "formula_version": [FORMULA_VERSION],
            "mode": ["geometric"],
            "fill_mode": ["wick"],
            "global_cooldown_bars": [0.0],
            "cb_enabled": [False],
            "is_authoritative": [True],
        })

        result = builder._apply_ingestion_gate(df)
        assert result["version_gated"].iloc[0]
        assert result["source_class"].iloc[0] == "legacy"


class TestStep15AlignmentAuditor:
    """Step 15: AlignmentAuditor is importable and instantiable."""

    def test_auditor_importable(self) -> None:
        from neutralgrid.training.btk_alignment_audit_v20260316 import AlignmentAuditor
        assert AlignmentAuditor is not None

    def test_auditor_instantiable_with_mock_ingestor(self) -> None:
        from neutralgrid.training.btk_alignment_audit_v20260316 import AlignmentAuditor

        class MockIngestor:
            def ingest(self) -> pd.DataFrame:
                return pd.DataFrame()

        auditor = AlignmentAuditor(ingestor=MockIngestor())
        assert auditor is not None

    def test_auditor_empty_data_returns_empty(self, tmp_path: Path) -> None:
        from neutralgrid.training.btk_alignment_audit_v20260316 import AlignmentAuditor

        class MockIngestor:
            def ingest(self) -> pd.DataFrame:
                return pd.DataFrame()

        auditor = AlignmentAuditor(ingestor=MockIngestor())
        report = auditor.compute_alignment_report(tmp_path)
        assert report.empty


class TestStep10RunnerAuthority:
    """Step 10: Runner-derived is_authoritative stamping."""

    def _make_klines(self, n: int = 100) -> pd.DataFrame:
        """Create synthetic 1m klines for testing."""
        prices = np.linspace(100.0, 102.0, n)
        return pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": np.full(n, 1000.0),
        })

    def test_training_config_is_authoritative(self) -> None:
        from backtest.btk_unified_runner import build_training_config, run_backtest

        cfg = build_training_config("TESTUSDT", 99.0, 103.0, 5, capital=100.0)
        result = run_backtest(cfg, self._make_klines())
        assert result["is_authoritative"] is True

    def test_raw_gridconfig_not_authoritative(self) -> None:
        """Raw GridConfig uses different defaults → not authoritative."""
        from backtest.backtest_realistic import GridConfig
        from backtest.btk_unified_runner import run_backtest

        cfg = GridConfig(
            symbol="TESTUSDT", lower=99.0, upper=103.0,
            num_grids=5, capital=100.0,
        )
        result = run_backtest(cfg, self._make_klines())
        assert result["is_authoritative"] is False

    def test_physics_override_not_authoritative(self) -> None:
        from backtest.btk_unified_runner import build_training_config, run_backtest

        cfg = build_training_config(
            "TESTUSDT", 99.0, 103.0, 5,
            capital=100.0,
            funding_mode="snapshot",  # Override physics → not authoritative
        )
        result = run_backtest(cfg, self._make_klines())
        assert result["is_authoritative"] is False
