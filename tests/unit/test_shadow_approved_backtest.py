"""Tests for the quarantined shadow-approved outcome backtest contract."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from neutralgrid.backtest.shadow_approved import (
    DEFAULT_DURATION_MINUTES,
    ShadowRunSources,
    assert_diagnostic_output_root,
    backtest_window_start_utc,
    is_run_mature,
    select_shadow_approved_candidates,
    sha256_file,
    summarize_final_pnl,
    validate_kline_window,
    validate_shadow_sources,
)


def test_runner_bootstraps_repository_level_backtest_package(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    script = project_root / "scripts" / "backtest_shadow_approved_candidates.py"
    probe = (
        "import runpy, sys; "
        "runpy.run_path(sys.argv[1], run_name='shadow_backtest_bootstrap_probe'); "
        "from backtest.btk_unified_runner import run_backtest; "
        "assert callable(run_backtest)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe, str(script)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _deployment_rows() -> pd.DataFrame:
    common = {
        "symbol": "TESTUSDT",
        "meta_prob_authority": "diagnostic_only",
        "grid_lower": 90.0,
        "grid_upper": 110.0,
        "num_grids": 20,
        "capital_base_usdt": 400.0,
        "capital_fraction": 0.25,
        "deploy_margin_usdt": 100.0,
        "leverage": 10,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "candidate_id": "selected_negative_ev",
                "failure_stage": "stage_b",
                "stage_b_reason": "data_missing:meta",
                "ev_score": -3.0,
            },
            {
                **common,
                "candidate_id": "below_shadow_threshold",
                "failure_stage": "stage_b",
                "stage_b_reason": "data_missing:meta",
                "ev_score": 5.0,
            },
            {
                **common,
                "candidate_id": "other_recorded_failure",
                "failure_stage": "stage_b",
                "stage_b_reason": "data_missing:meta|micro_not_viable",
                "ev_score": 5.0,
            },
            {
                **common,
                "candidate_id": "did_not_reach_stage_b",
                "failure_stage": "pre_reject",
                "stage_b_reason": "data_missing:meta",
                "ev_score": 5.0,
            },
        ]
    )


def _shadow_rows() -> pd.DataFrame:
    probabilities = {
        "selected_negative_ev": 0.80,
        "below_shadow_threshold": 0.36,
        "other_recorded_failure": 0.90,
        "did_not_reach_stage_b": 0.90,
    }
    return pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "surrogate_meta_prob_diagnostic": probability,
                "feature_complete": True,
                "diagnostic_only": True,
                "promotion_eligible": False,
                "runtime_meta_labeler_compatible": False,
                "teacher_hmm_artifact_version": "hmm_teacher",
                "snapshot_hmm_artifact_version": "hmm_snapshot",
                "hmm_lineage_matches_teacher": True,
            }
            for candidate_id, probability in probabilities.items()
        ]
    )


def _sources(tmp_path: Path, available: datetime) -> ShadowRunSources:
    deployment_path = tmp_path / "deployment_ready_20260827_120000.csv"
    shadow_path = tmp_path / "meta_prob_shadow.csv"
    manifest_path = tmp_path / "meta_prob_shadow.manifest.json"
    deployment_path.write_text("candidate_id\na\n", encoding="utf-8")
    shadow_path.write_text("candidate_id\na\n", encoding="utf-8")
    manifest_path.write_text("{}", encoding="utf-8")
    return ShadowRunSources(
        run_key="pipeline_20260827_120000",
        deployment_stamp="20260827_120000",
        deployment_path=deployment_path,
        shadow_path=shadow_path,
        shadow_manifest_path=manifest_path,
        candidate_available_ts_utc=available,
    )


def test_selection_is_exact_and_does_not_filter_negative_ev() -> None:
    selected = select_shadow_approved_candidates(
        _deployment_rows(),
        _shadow_rows(),
        min_meta_prob=0.37,
    )

    assert selected["candidate_id"].tolist() == ["selected_negative_ev"]
    assert selected["ev_score"].tolist() == [-3.0]
    assert bool(selected["unfiltered_outcome_pool"].all())
    assert not bool(selected["governed_training_eligible"].any())


def test_selection_fails_closed_on_candidate_id_mismatch() -> None:
    shadow = _shadow_rows().iloc[:-1].copy()

    with pytest.raises(ValueError, match="candidate-ID sets differ"):
        select_shadow_approved_candidates(
            _deployment_rows(),
            shadow,
            min_meta_prob=0.37,
        )


def test_selection_fails_closed_on_inconsistent_recorded_margin() -> None:
    deployment = _deployment_rows()
    deployment.loc[
        deployment["candidate_id"].eq("selected_negative_ev"),
        "deploy_margin_usdt",
    ] = 99.0

    with pytest.raises(ValueError, match="does not match"):
        select_shadow_approved_candidates(
            deployment,
            _shadow_rows(),
            min_meta_prob=0.37,
        )


def test_maturity_uses_first_complete_minute_and_exact_362_minutes(
    tmp_path: Path,
) -> None:
    available = datetime(2026, 8, 27, 12, 0, 30, tzinfo=timezone.utc)
    sources = _sources(tmp_path, available)
    start = backtest_window_start_utc(available)
    maturity = start + timedelta(minutes=DEFAULT_DURATION_MINUTES)

    assert start == datetime(2026, 8, 27, 12, 1, tzinfo=timezone.utc)
    assert not is_run_mature(
        sources,
        now_utc=maturity - timedelta(microseconds=1),
    )
    assert is_run_mature(sources, now_utc=maturity)


def test_kline_window_requires_exact_contiguous_complete_horizon() -> None:
    start = datetime(2026, 8, 27, 12, 1, tzinfo=timezone.utc)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start,
                periods=DEFAULT_DURATION_MINUTES,
                freq="min",
            ),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1.0,
        }
    )

    validated = validate_kline_window(frame, start_utc=start)
    assert len(validated) == DEFAULT_DURATION_MINUTES

    misaligned = frame.copy()
    misaligned["timestamp"] = pd.Series(
        pd.date_range(start + timedelta(minutes=1), periods=len(frame), freq="min")
    )
    with pytest.raises(ValueError, match="start boundary"):
        validate_kline_window(misaligned, start_utc=start)

    gapped = frame.copy()
    gapped.loc[100, "timestamp"] = start + timedelta(minutes=102)
    with pytest.raises(ValueError, match="duplicate|contiguous"):
        validate_kline_window(gapped, start_utc=start)


def test_shadow_manifest_hashes_and_markers_are_enforced(tmp_path: Path) -> None:
    sources = _sources(
        tmp_path,
        datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc),
    )
    manifest = {
        "diagnostic_only": True,
        "promotion_eligible": False,
        "runtime_meta_labeler_compatible": False,
        "input_sha256": sha256_file(sources.deployment_path),
        "output_sha256": sha256_file(sources.shadow_path),
    }
    sources.shadow_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_shadow_sources(sources)["diagnostic_only"] is True

    manifest["promotion_eligible"] = True
    sources.shadow_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="marker mismatch"):
        validate_shadow_sources(sources)


def test_final_pnl_summary_preserves_positive_zero_and_negative_rows() -> None:
    summary = summarize_final_pnl(
        pd.DataFrame(
            {
                "net_pnl": [2.0, 0.0, -1.0],
                "net_pnl_pct": [2.0, 0.0, -1.0],
            }
        )
    )

    assert summary["rows"] == 3
    assert summary["positive_rows"] == 1
    assert summary["zero_rows"] == 1
    assert summary["negative_rows"] == 1
    assert summary["net_pnl_usdt_sum"] == pytest.approx(1.0)


def test_output_root_must_be_diagnostic_and_not_governed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    diagnostic = project / "artifacts" / "diagnostics" / "shadow_pool"

    assert assert_diagnostic_output_root(diagnostic, project) == diagnostic.resolve()
    with pytest.raises(ValueError, match="governed path"):
        assert_diagnostic_output_root(
            project / "data" / "backtest_candidates",
            project,
        )
    with pytest.raises(ValueError, match="diagnostics"):
        assert_diagnostic_output_root(project / "artifacts" / "shadow_pool", project)
