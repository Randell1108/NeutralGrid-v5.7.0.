from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import shutil
from typing import cast

import numpy as np
import pandas as pd
import pytest

from neutralgrid.scanner import canonical_fastwin_profile as experiment
from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES, PatternProfile
from neutralgrid.scanner.profile_model import ProfileModel, save_profile_model


def _experiment_rows(
    groups: int = 30,
    rows_per_group: int = 10,
    *,
    signal: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in range(groups):
        start = cast(
            pd.Timestamp,
            pd.Timestamp("2026-01-01T00:00:00Z") + timedelta(days=group),
        )
        for within_group in range(rows_per_group):
            positive = within_group % 2 == 0
            symbol = f"SYM{group:02d}{within_group:02d}USDT"
            candidate_id = (
                f"{symbol}_{start.strftime('%Y%m%d_%H%M%S')}_{group:04x}{within_group:04x}"
            )
            row: dict[str, object] = {
                "candidate_id": candidate_id,
                "symbol": symbol,
                "start_time_utc": start.isoformat(),
                "backtest_timestamp": (start + timedelta(hours=48)).isoformat(),
                "time_to_target_hours": 2.0 if positive else np.nan,
                "target_reached": positive,
                "source": "backtest",
                "is_authoritative": True,
                "engine_version": experiment.ENGINE_VERSION,
                "label_contract_version": experiment.LABEL_CONTRACT_VERSION,
                "formula_version": experiment.FORMULA_VERSION,
                "realism_profile": experiment.REALISM_PROFILE,
            }
            class_shift = 1.0 if positive and signal else (-1.0 if signal else 0.0)
            row.update(
                {
                    DEFAULT_FEATURES[0]: class_shift + group * 0.001,
                    DEFAULT_FEATURES[1]: class_shift * 0.5 + group * 0.001,
                    DEFAULT_FEATURES[2]: class_shift * 0.01,
                    DEFAULT_FEATURES[3]: class_shift + group * 0.01,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _write_sources(
    tmp_path: Path,
    *,
    signal: bool = False,
) -> tuple[Path, Path, pd.DataFrame]:
    rows = _experiment_rows(signal=signal)
    fastwin = tmp_path / "fastwin"
    results = tmp_path / "results"
    fastwin.mkdir()
    results.mkdir()
    canonical_columns = [
        column for column in rows.columns if column not in DEFAULT_FEATURES
    ]
    rows[canonical_columns].to_csv(
        fastwin / "training_data_fastwin_fixture.csv", index=False
    )
    scanner_columns = ["candidate_id", "symbol", *DEFAULT_FEATURES]
    rows[scanner_columns].to_csv(
        results / "deployment_ready_20260130_000000.csv", index=False
    )
    rows[scanner_columns].to_csv(
        results / "potential_candidates_20260130_000000.csv", index=False
    )
    return fastwin, results, rows


def _write_incumbent(tmp_path: Path) -> tuple[Path, Path, Path]:
    model_path = tmp_path / "profile_model.json"
    pattern_path = tmp_path / "pattern_profile.json"
    workbook_path = tmp_path / "new_expired_bots.xlsx"
    features = list(DEFAULT_FEATURES)
    model = ProfileModel(
        features=features,
        winner_mu={feature: 0.0 for feature in features},
        loser_mu={feature: 0.0 for feature in features},
        inv_cov=np.eye(len(features)).tolist(),
        prior_winner=0.5,
        feature_mean={feature: 0.0 for feature in features},
        feature_std={feature: 1.0 for feature in features},
        feature_impute={feature: 0.0 for feature in features},
    )
    save_profile_model(model, model_path)
    PatternProfile(
        features=features,
        means={feature: 0.0 for feature in features},
        stds={feature: 1.0 for feature in features},
        q10={feature: -1.0 for feature in features},
        q90={feature: 1.0 for feature in features},
        selection_summary={},
    ).save_json(pattern_path)
    workbook_path.write_bytes(b"fixture workbook identity only")
    return model_path, pattern_path, workbook_path


def test_canonical_loader_rejects_non_backtest_row(tmp_path: Path) -> None:
    fastwin, _, rows = _write_sources(tmp_path)
    source = fastwin / "training_data_fastwin_fixture.csv"
    frame = rows[
        [column for column in rows.columns if column not in DEFAULT_FEATURES]
    ].copy()
    frame.loc[0, "source"] = "synthetic"
    frame.to_csv(source, index=False)

    with pytest.raises(ValueError, match="source mismatch rows=1"):
        experiment._load_canonical_outcomes(fastwin)


def test_scanner_loader_rejects_duplicate_exact_candidate_match(tmp_path: Path) -> None:
    _, results, rows = _write_sources(tmp_path)
    duplicate = rows.loc[[0], ["candidate_id", "symbol", *DEFAULT_FEATURES]]
    duplicate.to_csv(
        results / "deployment_ready_20260131_000000.csv", index=False
    )

    with pytest.raises(ValueError, match="duplicate canonical candidate IDs"):
        experiment._load_matching_scanner_rows(
            results,
            set(rows["candidate_id"].astype(str)),
            pattern="deployment_ready_*.csv",
        )


def test_freeze_uses_exact_join_and_excludes_incumbent_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fastwin, results, rows = _write_sources(tmp_path)
    model, pattern, workbook = _write_incumbent(tmp_path)
    overlap_id = str(rows.loc[0, "candidate_id"])
    monkeypatch.setattr(
        experiment,
        "_incumbent_training_ids",
        lambda _path: {overlap_id},
    )
    output = tmp_path / "experiment"

    manifest = experiment.freeze_experiment(
        output_dir=output,
        fastwin_dir=fastwin,
        results_dir=results,
        incumbent_model_path=model,
        incumbent_pattern_path=pattern,
        incumbent_workbook_path=workbook,
    )

    assert manifest["identity_contract"]["canonical_rows"] == 300
    assert manifest["identity_contract"]["deployment_exact_matches"] == 300
    assert manifest["identity_contract"]["excluded_incumbent_overlap_ids"] == 1
    assert manifest["feature_contract"]["recovered_complete_rows"] == 300
    assert manifest["feature_contract"]["potential_deployment_crosscheck"][
        "matched_candidate_ids"
    ] == 300
    assert manifest["split"]["development_scan_groups"] == 20
    assert manifest["split"]["holdout_scan_groups"] == 10
    development = pd.read_csv(output / "development.csv")
    holdout = pd.read_csv(output / "holdout.csv")
    assert overlap_id not in set(development["candidate_id"])
    assert overlap_id not in set(holdout["candidate_id"])
    assert len(development) + len(holdout) == 299


def test_walkforward_keeps_scan_groups_intact_and_uses_24h_purge() -> None:
    rows = _experiment_rows()
    rows["fastwin_label"] = rows["time_to_target_hours"].le(7.0).astype(int)

    result = experiment.walkforward_exact_fastwin(
        rows,
        source_sha256="a" * 64,
    )

    assert result.n_folds == 5
    assert result.purge_hours == 24.0
    assert len(result.oof_strategy_ids) == len(set(result.oof_strategy_ids))
    by_id = rows.set_index("candidate_id")
    for fold_start, fold_end in zip(
        result.fold_test_start_utc, result.fold_test_end_utc
    ):
        selected = by_id.loc[
            [
                candidate_id
                for candidate_id in result.oof_strategy_ids
                if fold_start
                <= cast(
                    pd.Timestamp,
                    pd.Timestamp(by_id.loc[candidate_id, "start_time_utc"]),
                ).isoformat()
                <= fold_end
            ]
        ]
        assert set(selected["start_time_utc"]) == {
            value
            for value in rows["start_time_utc"].unique()
            if fold_start
            <= cast(pd.Timestamp, pd.Timestamp(value)).isoformat()
            <= fold_end
        }


def test_failed_development_gate_never_reads_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fastwin, results, _ = _write_sources(tmp_path)
    model, pattern, workbook = _write_incumbent(tmp_path)
    monkeypatch.setattr(experiment, "_incumbent_training_ids", lambda _path: set())
    output = tmp_path / "experiment"
    experiment.freeze_experiment(
        output_dir=output,
        fastwin_dir=fastwin,
        results_dir=results,
        incumbent_model_path=model,
        incumbent_pattern_path=pattern,
        incumbent_workbook_path=workbook,
    )
    original_read_csv = experiment.pd.read_csv

    def guarded_read_csv(path, *args, **kwargs):
        if Path(path).name == "holdout.csv":
            raise AssertionError("development evaluator opened frozen holdout")
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(experiment.pd, "read_csv", guarded_read_csv)

    report = experiment.evaluate_development(output)

    assert report["decision"]["holdout_eligible"] is False
    assert report["holdout_status"] == "unopened_development_gate_failed"
    assert "mean_auc_below_floor" in report["decision"]["reasons"]
    assert (output / "candidate_profile_model.json").exists()
    assert (output / "candidate_pattern_profile.json").exists()


def test_passing_statistical_holdout_remains_shadow_without_execution_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fastwin, results, _ = _write_sources(tmp_path, signal=True)
    model, pattern, workbook = _write_incumbent(tmp_path)
    monkeypatch.setattr(experiment, "_incumbent_training_ids", lambda _path: set())
    output = tmp_path / "experiment"
    experiment.freeze_experiment(
        output_dir=output,
        fastwin_dir=fastwin,
        results_dir=results,
        incumbent_model_path=model,
        incumbent_pattern_path=pattern,
        incumbent_workbook_path=workbook,
    )
    development = experiment.evaluate_development(output)
    assert development["decision"]["holdout_eligible"] is True

    holdout = experiment.evaluate_holdout(output)

    assert holdout["decision"] == {
        "promotion_eligible": False,
        "promotion_attempted": False,
        "production_artifacts_modified": False,
        "reasons": [experiment.REALISM_SHADOW_PROMOTION_BLOCKER],
    }
    assert holdout["paired_cluster_bootstrap"]["delta_auc_ci_95"][0] > 0.0
    assert len(pd.read_csv(output / "holdout_predictions.csv")) == 100

    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    shutil.copy2(model, profile_dir / "profile_model.json")
    shutil.copy2(pattern, profile_dir / "pattern_profile.json")
    with pytest.raises(ValueError, match="holdout gate did not authorize promotion"):
        experiment.promote_from_holdout(
            output,
            profile_dir=profile_dir,
        )

    assert not (profile_dir / "current.json").exists()
    assert not (output / "promotion_result.json").exists()
