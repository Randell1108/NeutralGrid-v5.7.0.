"""Phase 3 tests for walk-forward CV + profile_model promotion gate.

Covers PATTERN_PROFILE_FIX.md Phase 3.1-3.6:
  - 3.1: walk-forward CV produces per-fold AUC/KS.
  - 3.2: promote_profile_version refuses mean_pass_rate < 0.50; accepts >= 0.50.
  - 3.3: artifact naming `profile_model_YYYYMMDD_HHMMSS.json`.
  - 3.4: trial logging fires on both pass and fail.
  - 3.5: resolve_active_profile_model_path returns pointer target.
  - 3.6: silent-drop guard refuses when coverage < 0.90, allows at exactly 0.90.
"""
from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES, PatternProfile
from neutralgrid.scanner.profile_model import ProfileModel
from neutralgrid.scanner.profile_model_walkforward import (
    COVERAGE_FLOOR,
    MEAN_PASS_RATE_FLOOR,
    PromotionDecision,
    WalkForwardResult,
    make_profile_model_filename,
    promote_profile_version,
    resolve_active_pattern_profile_path,
    resolve_active_profile_model_path,
    resolve_latest_evaluated_holdout_end,
)


def _workspace_tmp() -> Path:
    p = Path.cwd() / ".pytest_tmp" / f"phase3_{uuid.uuid4().hex}"
    p.mkdir(parents=True, exist_ok=False)
    return p


def _dummy_model(feats: list[str]) -> ProfileModel:
    n = len(feats)
    return ProfileModel(
        features=list(feats),
        winner_mu={f: 0.5 for f in feats},
        loser_mu={f: -0.5 for f in feats},
        inv_cov=np.eye(n).tolist(),
        prior_winner=0.5,
        duration_band={"min_hours": 0.0, "max_hours": 7.0},
        feature_mean={f: 0.0 for f in feats},
        feature_std={f: 1.0 for f in feats},
    )


def _dummy_pattern(feats: list[str]) -> PatternProfile:
    return PatternProfile(
        features=list(feats),
        means={f: 0.0 for f in feats},
        stds={f: 1.0 for f in feats},
        q10={f: -1.0 for f in feats},
        q90={f: 1.0 for f in feats},
        selection_summary={"winners_count": 40, "pnl_threshold": 5.0},
    )


def _wf(pass_rate: float, coverage: float, requested: list[str], admitted: list[str]) -> WalkForwardResult:
    passing_fixture = np.isclose(pass_rate, 0.60)
    fold_auc = (
        [0.6, 0.6, 0.6, 0.5, 0.5]
        if passing_fixture
        else [0.6, 0.6, 0.6, 0.6]
    )
    fold_count = len(fold_auc)
    labels = [0] * 10 + [1] * 10
    scores = [float(value) for value in range(10)] + [
        -4.0,
        -3.0,
        -2.0,
        -1.0,
        20.0,
        21.0,
        22.0,
        23.0,
        24.0,
        25.0,
    ]
    return WalkForwardResult(
        n_folds=fold_count,
        fold_auc=fold_auc,
        fold_ks=[0.3] * fold_count,
        mean_auc=float(np.mean(fold_auc)),
        mean_ks=0.3,
        mean_pass_rate=pass_rate,
        purge_hours=7.0,
        requested_features=requested,
        admitted_features=admitted,
        feature_coverage=coverage,
        fold_train_rows=[100] * fold_count,
        fold_test_rows=[4] * fold_count,
        fold_train_winners=[40] * fold_count,
        fold_test_winners=[2] * fold_count,
        fold_pnl_thresholds=[5.0] * fold_count,
        fold_test_start_utc=[
            f"2026-01-{index + 1:02d}T00:00:00+00:00"
            for index in range(fold_count)
        ],
        fold_test_end_utc=[
            f"2026-01-{index + 1:02d}T03:00:00+00:00"
            for index in range(fold_count)
        ],
        oof_strategy_ids=[f"bot_{index}" for index in range(20)],
        oof_labels=labels,
        oof_scores=scores,
        oof_probabilities=[0.4] * 20,
        pooled_oof_auc=0.60,
        pooled_oof_ks=0.60,
        pooled_oof_brier=0.26,
        pooled_oof_ece=0.10,
        source_sha256="a" * 64,
        labeled_rows=100,
    )


def test_make_filename_matches_convention():
    """3.3 — artifact naming `profile_model_YYYYMMDD_HHMMSS.json`."""
    ts = datetime(2026, 4, 20, 13, 45, 7, tzinfo=timezone.utc)
    name = make_profile_model_filename(ts)
    assert name == "profile_model_20260420_134507.json"


def test_promotion_refused_below_pass_rate_floor():
    """3.2 — mean_pass_rate=0.49 is refused; no files written."""
    tmp = _workspace_tmp()
    try:
        feats = ["adx_1h", "rsi_15m"]
        model = _dummy_model(feats)
        wf = _wf(pass_rate=0.49, coverage=1.0, requested=feats, admitted=feats)
        decision = promote_profile_version(
            model,
            requested_features=feats,
            wf_result=wf,
            candidate_pattern_profile=_dummy_pattern(feats),
            profile_dir=tmp,
            ts=datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc),
        )
        assert decision.promoted is False
        assert "mean_pass_rate" in decision.reason
        assert decision.evaluation_filename is not None
        evaluation = tmp / "evaluations" / decision.evaluation_filename
        assert evaluation.exists()
        evidence = json.loads(evaluation.read_text(encoding="utf-8"))
        assert evidence["gate_decision"] == "reject"
        assert evidence["reason"] == decision.reason
        assert not (tmp / "current.json").exists()
        assert not any(tmp.glob("profile_model_*.json"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_refused_when_fewer_than_three_folds_are_finite():
    tmp = _workspace_tmp()
    try:
        feats = ["adx_1h", "rsi_15m"]
        model = _dummy_model(feats)
        wf = WalkForwardResult(
            n_folds=4,
            fold_auc=[float("nan"), float("nan"), 0.60, 0.60],
            fold_ks=[float("nan"), float("nan"), 0.30, 0.30],
            mean_auc=0.60,
            mean_ks=0.30,
            mean_pass_rate=1.0,
            purge_hours=7.0,
            requested_features=feats,
            admitted_features=feats,
            feature_coverage=1.0,
        )

        decision = promote_profile_version(
            model,
            requested_features=feats,
            wf_result=wf,
            candidate_pattern_profile=_dummy_pattern(feats),
            profile_dir=tmp,
        )

        assert decision.promoted is False
        assert "finite_fold_count=2" in decision.reason
        assert not (tmp / "current.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_refused_when_finite_fold_coverage_is_too_low():
    tmp = _workspace_tmp()
    try:
        feats = ["adx_1h", "rsi_15m"]
        model = _dummy_model(feats)
        wf = WalkForwardResult(
            n_folds=7,
            fold_auc=[float("nan")] * 4 + [0.60, 0.60, 0.60],
            fold_ks=[float("nan")] * 4 + [0.30, 0.30, 0.30],
            mean_auc=0.60,
            mean_ks=0.30,
            mean_pass_rate=1.0,
            purge_hours=7.0,
            requested_features=feats,
            admitted_features=feats,
            feature_coverage=1.0,
        )

        decision = promote_profile_version(
            model,
            requested_features=feats,
            wf_result=wf,
            candidate_pattern_profile=_dummy_pattern(feats),
            profile_dir=tmp,
        )

        assert decision.promoted is False
        assert "finite_fold_coverage" in decision.reason
        assert not (tmp / "current.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_refused_when_mean_auc_is_below_floor():
    tmp = _workspace_tmp()
    try:
        feats = ["adx_1h", "rsi_15m"]
        model = _dummy_model(feats)
        wf = WalkForwardResult(
            n_folds=4,
            fold_auc=[0.56, 0.56, 0.50, 0.50],
            fold_ks=[0.30, 0.30, 0.20, 0.20],
            mean_auc=0.53,
            mean_ks=0.25,
            mean_pass_rate=0.50,
            purge_hours=7.0,
            requested_features=feats,
            admitted_features=feats,
            feature_coverage=1.0,
        )

        decision = promote_profile_version(
            model,
            requested_features=feats,
            wf_result=wf,
            candidate_pattern_profile=_dummy_pattern(feats),
            profile_dir=tmp,
        )

        assert decision.promoted is False
        assert "mean_auc=0.530" in decision.reason
        assert not (tmp / "current.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_refused_when_pooled_oof_auc_is_below_floor():
    tmp = _workspace_tmp()
    try:
        feats = ["adx_1h", "rsi_15m"]
        wf = _wf(pass_rate=0.75, coverage=1.0, requested=feats, admitted=feats)
        wf = WalkForwardResult(
            **{**wf.__dict__, "pooled_oof_auc": 0.49}
        )
        decision = promote_profile_version(
            _dummy_model(feats),
            requested_features=feats,
            wf_result=wf,
            candidate_pattern_profile=_dummy_pattern(feats),
            profile_dir=tmp,
        )
        assert decision.promoted is False
        assert "pooled_oof_auc=0.490" in decision.reason
        assert not (tmp / "current.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_accepted_updates_current_atomically():
    """3.2 + 3.3 — passing gates write artifact + atomic current.json."""
    tmp = _workspace_tmp()
    try:
        feats = ["adx_1h", "rsi_15m"]
        model = _dummy_model(feats)
        wf = _wf(pass_rate=0.60, coverage=1.0, requested=feats, admitted=feats)
        ts = datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc)
        decision = promote_profile_version(
            model,
            requested_features=feats,
            wf_result=wf,
            candidate_pattern_profile=_dummy_pattern(feats),
            profile_dir=tmp,
            ts=ts,
        )
        assert decision.promoted is True
        assert decision.artifact_filename == "profile_model_20260420_100000.json"
        current = tmp / "current.json"
        assert current.exists()
        obj = json.loads(current.read_text())
        assert obj["active"] == decision.artifact_filename
        assert obj["active_pattern_profile"] == "pattern_profile_20260420_100000.json"
        assert (tmp / obj["active_pattern_profile"]).exists()
        assert len(obj["pattern_profile_sha256"]) == 64
        assert "sha256" in obj and len(obj["sha256"]) == 64
        assert obj["mean_pass_rate"] == pytest.approx(0.60)
        assert decision.evaluation_filename is not None
        assert obj["evaluation"] == f"evaluations/{decision.evaluation_filename}"
        evidence = json.loads(
            (tmp / "evaluations" / decision.evaluation_filename).read_text(
                encoding="utf-8"
            )
        )
        assert evidence["gate_decision"] == "pass"
        assert evidence["candidate"]["sha256"] == obj["sha256"]
        assert len(obj["evaluation_sha256"]) == 64
        assert resolve_active_profile_model_path(tmp) == tmp / obj["active"]
        assert resolve_active_pattern_profile_path(tmp) == tmp / obj[
            "active_pattern_profile"
        ]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_active_path_respects_pointer():
    """3.5 — resolve_active_profile_model_path returns current.json['active']."""
    tmp = _workspace_tmp()
    try:
        (tmp / "profile_model_X.json").write_text("{}", encoding="utf-8")
        (tmp / "current.json").write_text(json.dumps({"active": "profile_model_X.json"}))
        path = resolve_active_profile_model_path(tmp)
        assert path.name == "profile_model_X.json"
        assert path.parent == tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_active_path_bootstraps_when_current_missing():
    """3.5 bootstrap exception — absent current.json may use profile_model.json.

    This breaks the cold-start loop: before the first walk-forward promotion is
    statistically meaningful, the freshly retrained profile_model.json is the
    default bootstrap candidate.
    """
    tmp = _workspace_tmp()
    try:
        (tmp / "profile_model.json").write_text("{}")
        path = resolve_active_profile_model_path(tmp)
        assert path.name == "profile_model.json"
        assert path.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_active_model_fails_closed_on_hash_mismatch():
    tmp = _workspace_tmp()
    try:
        (tmp / "profile_model_X.json").write_text("{}", encoding="utf-8")
        (tmp / "current.json").write_text(
            json.dumps({"active": "profile_model_X.json", "sha256": "0" * 64}),
            encoding="utf-8",
        )
        resolved = resolve_active_profile_model_path(tmp)
        assert not resolved.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_active_model_rejects_path_traversal():
    tmp = _workspace_tmp()
    try:
        (tmp / "current.json").write_text(
            json.dumps({"active": "../outside.json"}), encoding="utf-8"
        )
        resolved = resolve_active_profile_model_path(tmp)
        assert resolved.parent == tmp
        assert not resolved.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_active_pattern_path_respects_bundle_pointer():
    tmp = _workspace_tmp()
    try:
        pattern_name = "pattern_profile_20260420_100000.json"
        (tmp / pattern_name).write_text("{}", encoding="utf-8")
        (tmp / "current.json").write_text(
            json.dumps({"active_pattern_profile": pattern_name}), encoding="utf-8"
        )
        assert resolve_active_pattern_profile_path(tmp) == tmp / pattern_name
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_active_pattern_path_bootstraps_when_current_missing():
    tmp = _workspace_tmp()
    try:
        bootstrap = tmp / "pattern_profile.json"
        bootstrap.write_text("{}", encoding="utf-8")
        assert resolve_active_pattern_profile_path(tmp) == bootstrap
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_refuses_pattern_model_feature_mismatch():
    tmp = _workspace_tmp()
    try:
        feats = ["adx_1h", "rsi_15m"]
        decision = promote_profile_version(
            _dummy_model(feats),
            requested_features=feats,
            wf_result=_wf(0.80, 1.0, feats, feats),
            candidate_pattern_profile=_dummy_pattern(["adx_1h"]),
            profile_dir=tmp,
        )
        assert decision.promoted is False
        assert "pattern_model_features_disagree" in decision.reason
        assert not (tmp / "current.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_active_path_fail_closed_when_current_and_bootstrap_missing():
    """3.5 bootstrap exception still fails closed when no artifact exists at all."""
    tmp = _workspace_tmp()
    try:
        path = resolve_active_profile_model_path(tmp)
        assert path.name != "profile_model.json"
        assert not path.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_active_path_fail_closed_on_corrupt_current():
    """3.5 — corrupt current.json returns a non-existent sentinel, not a stale fallback."""
    tmp = _workspace_tmp()
    try:
        (tmp / "profile_model.json").write_text("{}")
        (tmp / "current.json").write_text("not-valid-json{{{")
        path = resolve_active_profile_model_path(tmp)
        assert path.name != "profile_model.json"
        assert not path.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolve_active_path_fail_closed_on_missing_active_key():
    """3.5 — current.json without 'active' key returns fail-closed sentinel."""
    tmp = _workspace_tmp()
    try:
        (tmp / "profile_model.json").write_text("{}")
        (tmp / "current.json").write_text(json.dumps({"mean_pass_rate": 0.6}))
        path = resolve_active_profile_model_path(tmp)
        assert path.name != "profile_model.json"
        assert not path.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_walkforward_rejects_purge_hours_below_max_duration():
    """3.1 — purge < max_duration raises (prevents window-overlap leakage)."""
    from neutralgrid.scanner.profile_model_walkforward import walkforward_evaluate
    tmp = _workspace_tmp()
    try:
        (tmp / "dummy.xlsx").write_text("")  # file won't be read before validation
        with pytest.raises(ValueError, match="purge_hours.*>=.*max_duration_hours"):
            walkforward_evaluate(
                tmp / "dummy.xlsx",
                purge_hours=3.0,
                max_duration_hours=7.0,
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_silent_drop_guard_refuses_below_coverage_floor():
    """3.6 — coverage=0.80 (<0.90) refused; dropped features listed."""
    tmp = _workspace_tmp()
    try:
        requested = ["a", "b", "c", "d", "e"]  # 5 requested
        admitted = ["a", "b", "c", "d"]  # 4/5 = 0.80
        model = _dummy_model(admitted)
        wf = _wf(pass_rate=0.60, coverage=0.80, requested=requested, admitted=admitted)
        decision = promote_profile_version(
            model,
            requested_features=requested,
            wf_result=wf,
            candidate_pattern_profile=_dummy_pattern(admitted),
            profile_dir=tmp,
        )
        assert decision.promoted is False
        assert "feature_coverage" in decision.reason
        assert "e" in decision.dropped_features
        assert not (tmp / "current.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_silent_drop_guard_allows_exactly_at_floor():
    """3.6 boundary — coverage exactly 0.90 is allowed."""
    tmp = _workspace_tmp()
    try:
        requested = [f"f{i}" for i in range(10)]  # 10 features
        admitted = requested[:9]  # 9/10 = 0.90
        model = _dummy_model(admitted)
        wf = _wf(pass_rate=0.60, coverage=0.90, requested=requested, admitted=admitted)
        decision = promote_profile_version(
            model,
            requested_features=requested,
            wf_result=wf,
            candidate_pattern_profile=_dummy_pattern(admitted),
            profile_dir=tmp,
        )
        assert decision.promoted is True
        assert decision.feature_coverage == pytest.approx(0.90)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_refused_when_candidate_has_extra_feature():
    """Subset guard — candidate.features must be a subset of requested_features."""
    tmp = _workspace_tmp()
    try:
        requested = ["adx_1h", "rsi_15m"]
        # Candidate carries an extra feature not in the batch contract.
        extra_feats = ["adx_1h", "rsi_15m", "not_in_contract"]
        model = _dummy_model(extra_feats)
        wf = _wf(pass_rate=0.60, coverage=1.0, requested=requested, admitted=requested)
        decision = promote_profile_version(
            model,
            requested_features=requested,
            wf_result=wf,
            candidate_pattern_profile=_dummy_pattern(extra_feats),
            profile_dir=tmp,
        )
        assert decision.promoted is False
        assert "not_in_requested" in decision.reason
        assert not (tmp / "current.json").exists()
        assert not any(tmp.glob("profile_model_*.json"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_refused_when_candidate_features_disagree_with_walkforward():
    """Subset guard — candidate.features must equal wf_result.admitted_features."""
    tmp = _workspace_tmp()
    try:
        requested = ["adx_1h", "rsi_15m", "bb_width"]
        # Walk-forward admitted 3, but candidate was fit on only 2 (drift).
        model = _dummy_model(["adx_1h", "rsi_15m"])
        wf = _wf(pass_rate=0.60, coverage=1.0, requested=requested, admitted=requested)
        decision = promote_profile_version(
            model,
            requested_features=requested,
            wf_result=wf,
            candidate_pattern_profile=_dummy_pattern(model.features),
            profile_dir=tmp,
        )
        assert decision.promoted is False
        assert "disagree_with_walkforward" in decision.reason
        assert not (tmp / "current.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_trial_logger_fires_on_refusal_and_pass():
    """3.4 — trial_logger receives a TrialRecord on both outcomes."""
    tmp = _workspace_tmp()

    class _Capture:
        def __init__(self):
            self.records: list[object] = []

        def log_trial(self, record):  # noqa: ANN001
            self.records.append(record)

    try:
        feats = ["adx_1h", "rsi_15m"]
        model = _dummy_model(feats)
        logger_fail = _Capture()
        promote_profile_version(
            model,
            requested_features=feats,
            wf_result=_wf(0.40, 1.0, feats, feats),
            candidate_pattern_profile=_dummy_pattern(feats),
            profile_dir=tmp,
            trial_logger=logger_fail,
            trial_hyperparameters={"shrinkage": 0.30},
        )
        assert len(logger_fail.records) == 1

        logger_pass = _Capture()
        promote_profile_version(
            model,
            requested_features=feats,
            wf_result=_wf(0.60, 1.0, feats, feats),
            candidate_pattern_profile=_dummy_pattern(feats),
            profile_dir=tmp,
            trial_logger=logger_pass,
            trial_hyperparameters={"shrinkage": 0.30},
        )
        assert len(logger_pass.records) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_walkforward_on_chronological_xlsx_produces_fold_metrics():
    """3.1 — walk-forward over balanced chronological data yields finite metrics."""
    tmp = _workspace_tmp()
    try:
        from neutralgrid.scanner.profile_model_walkforward import walkforward_evaluate

        # Build 300 rows with chronological start_time_utc.
        base_ts = pd.Timestamp("2026-01-01T00:00:00Z")
        rows = []
        rng = np.random.default_rng(seed=42)
        for i in range(300):
            is_winner = i % 2 == 0
            rows.append({
                "strategy_id": f"bot_{i}",
                "start_time_utc": (base_ts + pd.Timedelta(hours=i)).isoformat(),
                "end_time_utc": (base_ts + pd.Timedelta(hours=i + 3)).isoformat(),
                "pnl_pct": 30.0 + rng.normal(0, 2) if is_winner else 5.0 + rng.normal(0, 2),
                "profit_factor": 2.5 + rng.normal(0, 0.2) if is_winner else 1.0 + rng.normal(0, 0.1),
                "duration_hours": 3.0,
                "adx_1h": 40.0 + rng.normal(0, 2) if is_winner else 20.0 + rng.normal(0, 2),
                "adx_15m": 45.0 + rng.normal(0, 2) if is_winner else 22.0 + rng.normal(0, 2),
                "adx_5m": 42.0 + rng.normal(0, 2) if is_winner else 21.0 + rng.normal(0, 2),
                "rsi_15m": 55.0 + rng.normal(0, 2) if is_winner else 40.0 + rng.normal(0, 2),
                "ema_slope_1h": 0.05 + rng.normal(0, 0.01) if is_winner else -0.02 + rng.normal(0, 0.01),
                "ema_crosses_5m": i % 3,
                "vwap_crosses_5m": i % 2,
                "range_size_pct": 3.0 + rng.normal(0, 0.2),
                "bb_width": 0.03 + rng.normal(0, 0.003),
            })
        df = pd.DataFrame(rows)
        path = tmp / "chronological.xlsx"
        df.to_excel(str(path), index=False, sheet_name="Sheet1")

        feats = [
            "adx_1h", "adx_15m", "adx_5m", "rsi_15m", "ema_slope_1h",
            "ema_crosses_5m", "vwap_crosses_5m", "range_size_pct", "bb_width",
        ]
        wf = walkforward_evaluate(
            path,
            n_folds=5,
            purge_hours=7.0,
            max_duration_hours=7.0,
            min_profit_factor=1.0,
            top_quantile=0.5,
            features=feats,
        )
        assert wf.n_folds > 0
        finite = [a for a in wf.fold_auc if np.isfinite(a)]
        assert wf.n_folds == 5
        assert len(finite) >= 3
        assert len(wf.fold_train_rows) == wf.n_folds
        assert len(wf.fold_test_rows) == wf.n_folds
        assert all(n >= 30 for n in wf.fold_train_winners)
        assert len(wf.oof_strategy_ids) == len(wf.oof_labels) == len(wf.oof_scores)
        assert np.isfinite(wf.pooled_oof_auc)
        # Separable synthetic data -> AUC should exceed chance by a healthy margin.
        assert wf.mean_auc > 0.60
        assert wf.feature_coverage == pytest.approx(1.0)

        disclosed_endpoint = wf.fold_test_end_utc[0]
        fresh_wf = walkforward_evaluate(
            path,
            n_folds=2,
            purge_hours=7.0,
            max_duration_hours=7.0,
            min_profit_factor=1.0,
            top_quantile=0.5,
            features=feats,
            holdout_start_after_utc=disclosed_endpoint,
        )
        cutoff = pd.Timestamp(disclosed_endpoint)
        assert fresh_wf.n_folds == 2
        assert fresh_wf.holdout_start_after_utc == cutoff.isoformat()
        assert all(
            pd.Timestamp(start) > cutoff
            for start in fresh_wf.fold_test_start_utc
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_refuses_challenger_without_paired_oos_improvement():
    """A tied challenger cannot be promoted over an evaluated incumbent."""
    tmp = _workspace_tmp()
    try:
        feats = ["adx_1h", "rsi_15m"]
        model = _dummy_model(feats)
        common = dict(
            n_folds=4,
            fold_auc=[0.70, 0.70, 0.70, 0.70],
            fold_ks=[0.40, 0.40, 0.40, 0.40],
            mean_auc=0.70,
            mean_ks=0.40,
            mean_pass_rate=1.0,
            purge_hours=7.0,
            requested_features=feats,
            admitted_features=feats,
            feature_coverage=1.0,
            oof_strategy_ids=[f"bot_{i}" for i in range(20)],
            oof_labels=[i % 2 for i in range(20)],
            oof_scores=[float(i % 2) for i in range(20)],
            oof_probabilities=[float(i % 2) for i in range(20)],
            pooled_oof_auc=1.0,
            pooled_oof_ks=1.0,
            pooled_oof_brier=0.0,
            pooled_oof_ece=0.0,
            fold_train_rows=[100] * 4,
            fold_test_rows=[5] * 4,
            fold_train_winners=[40] * 4,
            fold_test_winners=[2] * 4,
            fold_pnl_thresholds=[5.0] * 4,
            fold_test_start_utc=[
                f"2026-02-{i + 1:02d}T00:00:00+00:00" for i in range(4)
            ],
            fold_test_end_utc=[
                f"2026-02-{i + 1:02d}T03:00:00+00:00" for i in range(4)
            ],
            source_sha256="b" * 64,
            labeled_rows=100,
        )
        candidate = WalkForwardResult(**common)
        incumbent = WalkForwardResult(**common)

        decision = promote_profile_version(
            model,
            requested_features=feats,
            wf_result=candidate,
            incumbent_wf_result=incumbent,
            candidate_pattern_profile=_dummy_pattern(feats),
            profile_dir=tmp,
        )

        assert decision.promoted is False
        assert "paired_auc_delta_ci_low" in decision.reason
        assert decision.paired_auc_delta == pytest.approx(0.0)
        assert not (tmp / "current.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_refuses_model_only_candidate():
    """A current pointer must never be created without its pattern artifact."""
    tmp = _workspace_tmp()
    try:
        feats = ["adx_1h", "rsi_15m"]
        decision = promote_profile_version(
            _dummy_model(feats),
            requested_features=feats,
            wf_result=_wf(0.60, 1.0, feats, feats),
            profile_dir=tmp,
        )
        assert decision.promoted is False
        assert decision.reason == "candidate_pattern_profile_required"
        assert not (tmp / "current.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_requires_incumbent_comparison_when_bootstrap_exists():
    """Existing runtime artifacts make the paired OOS comparison mandatory."""
    tmp = _workspace_tmp()
    try:
        feats = ["adx_1h", "rsi_15m"]
        (tmp / "profile_model.json").write_text("{}", encoding="utf-8")
        (tmp / "pattern_profile.json").write_text("{}", encoding="utf-8")
        decision = promote_profile_version(
            _dummy_model(feats),
            requested_features=feats,
            wf_result=_wf(0.60, 1.0, feats, feats),
            candidate_pattern_profile=_dummy_pattern(feats),
            profile_dir=tmp,
        )
        assert decision.promoted is False
        assert decision.reason == "incumbent_walkforward_comparison_required"
        assert not (tmp / "current.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_rejects_duplicate_oof_strategy_ids():
    tmp = _workspace_tmp()
    try:
        feats = ["adx_1h", "rsi_15m"]
        wf = _wf(0.60, 1.0, feats, feats)
        duplicate_ids = list(wf.oof_strategy_ids)
        duplicate_ids[-1] = duplicate_ids[0]
        wf = replace(wf, oof_strategy_ids=duplicate_ids)
        decision = promote_profile_version(
            _dummy_model(feats),
            requested_features=feats,
            wf_result=wf,
            candidate_pattern_profile=_dummy_pattern(feats),
            profile_dir=tmp,
        )
        assert decision.promoted is False
        assert "oof_strategy_ids_contain_duplicates" in decision.reason
        assert not (tmp / "current.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_rejects_forged_pooled_auc():
    tmp = _workspace_tmp()
    try:
        feats = ["adx_1h", "rsi_15m"]
        wf = replace(_wf(0.60, 1.0, feats, feats), pooled_oof_auc=0.99)
        decision = promote_profile_version(
            _dummy_model(feats),
            requested_features=feats,
            wf_result=wf,
            candidate_pattern_profile=_dummy_pattern(feats),
            profile_dir=tmp,
        )
        assert decision.promoted is False
        assert "pooled_oof_auc_disagrees_with_oof_evidence" in decision.reason
        assert not (tmp / "current.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promotion_rejects_reused_disclosed_oof_holdout():
    tmp = _workspace_tmp()
    try:
        feats = ["adx_1h", "rsi_15m"]
        wf = _wf(0.60, 1.0, feats, feats)
        first = promote_profile_version(
            _dummy_model(feats),
            requested_features=feats,
            wf_result=wf,
            candidate_pattern_profile=_dummy_pattern(feats),
            profile_dir=tmp,
            ts=datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc),
        )
        assert first.promoted is True
        current_before = (tmp / "current.json").read_bytes()

        second = promote_profile_version(
            _dummy_model(feats),
            requested_features=feats,
            wf_result=wf,
            incumbent_wf_result=wf,
            candidate_pattern_profile=_dummy_pattern(feats),
            profile_dir=tmp,
            ts=datetime(2026, 4, 20, 10, 0, 1, tzinfo=timezone.utc),
        )
        assert second.promoted is False
        assert "oof_holdout_reused=20" in second.reason
        assert (tmp / "current.json").read_bytes() == current_before
        assert resolve_latest_evaluated_holdout_end(tmp) == (
            "2026-01-05T03:00:00+00:00"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_paired_oof_misalignment_rejects_without_raising():
    tmp = _workspace_tmp()
    try:
        feats = ["adx_1h", "rsi_15m"]
        candidate = _wf(0.60, 1.0, feats, feats)
        incumbent = replace(
            candidate,
            oof_strategy_ids=list(reversed(candidate.oof_strategy_ids)),
        )
        decision = promote_profile_version(
            _dummy_model(feats),
            requested_features=feats,
            wf_result=candidate,
            incumbent_wf_result=incumbent,
            candidate_pattern_profile=_dummy_pattern(feats),
            profile_dir=tmp,
        )
        assert decision.promoted is False
        assert "paired_oof_alignment_invalid" in decision.reason
        assert decision.evaluation_filename is not None
        assert (tmp / "evaluations" / decision.evaluation_filename).exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolvers_fail_closed_when_evaluation_evidence_is_tampered():
    tmp = _workspace_tmp()
    try:
        feats = ["adx_1h", "rsi_15m"]
        decision = promote_profile_version(
            _dummy_model(feats),
            requested_features=feats,
            wf_result=_wf(0.60, 1.0, feats, feats),
            candidate_pattern_profile=_dummy_pattern(feats),
            profile_dir=tmp,
        )
        assert decision.promoted is True
        assert decision.evaluation_filename is not None
        evaluation = tmp / "evaluations" / decision.evaluation_filename
        evaluation.write_text("{}", encoding="utf-8")

        assert not resolve_active_profile_model_path(tmp).exists()
        assert not resolve_active_pattern_profile_path(tmp).exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
