"""ERR-084: the strict canonical promotion gate must bind on the
retrain_hmm.py entry point.

Before the fix, retrain_hmm.py never passed canonical_mode /
accepted_count / trained_count into evaluate_and_maybe_promote, so the
strict canonical gate (accepted==50 AND trained==50) was dead code on the
canonical path and only the legacy coverage gates ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import neutralgrid.models.hmm.retrain_orchestration as ro


def _make_artifact(tmp_path: Path, mean_pass_rate: float = 1.0) -> Path:
    artifact = tmp_path / "rolling_180d_20990101_000000"
    artifact.mkdir()
    metadata = {"eval_metrics": {"mean_pass_rate": mean_pass_rate}}
    (artifact / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return artifact


class TestCanonicalGateEnforcement:
    """evaluate_and_maybe_promote strict canonical gates."""

    def test_partial_coverage_blocks_promotion(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ro, "promote_hmm_version",
            lambda *a, **k: pytest.fail("must not promote on partial coverage"),
        )
        artifact = _make_artifact(tmp_path)
        _metrics, promoted, reasons = ro.evaluate_and_maybe_promote(
            version=artifact.name,
            artifact_path=artifact,
            requested_symbols=50,
            fetched_datasets={f"S{i}USDT": object() for i in range(49)},
            window_days=180,
            actual_window_days=180.0,
            canonical_mode=True,
            accepted_count=49,
            trained_count=50,
            probe_count=60,
        )
        assert promoted is False
        assert any("canonical_symbol_coverage" in r for r in reasons)

    def test_full_coverage_promotes(self, tmp_path, monkeypatch):
        promoted_calls: list[str] = []
        monkeypatch.setattr(
            ro, "promote_hmm_version",
            lambda version, path: promoted_calls.append(version),
        )
        artifact = _make_artifact(tmp_path)
        metrics, promoted, reasons = ro.evaluate_and_maybe_promote(
            version=artifact.name,
            artifact_path=artifact,
            requested_symbols=50,
            fetched_datasets={f"S{i}USDT": object() for i in range(50)},
            window_days=180,
            actual_window_days=180.0,
            canonical_mode=True,
            accepted_count=50,
            trained_count=50,
            probe_count=60,
        )
        assert promoted is True
        assert reasons == []
        assert promoted_calls == [artifact.name]
        # Canonical counters recorded for audit
        assert metrics["canonical_mode"] is True
        assert metrics["accepted_count"] == 50
        assert metrics["trained_count"] == 50


class TestRetrainHmmWiresCanonicalGate:
    """ERR-084 wiring proof: the entry point passes the canonical params."""

    def test_entry_point_passes_canonical_params(self):
        source_path = Path(__file__).resolve().parents[2] / "retrain_hmm.py"
        source = source_path.read_text(encoding="utf-8")
        assert "canonical_mode=args.canonical" in source, (
            "retrain_hmm.py must pass canonical_mode into "
            "evaluate_and_maybe_promote (ERR-084)"
        )
        assert "accepted_count=len(symbols) if args.canonical else None" in source
        assert (
            "trained_count=len(training_datasets) if args.canonical else None"
            in source
        )
        assert "probe_count=canonical_probe_count" in source
