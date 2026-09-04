from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = _ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_holdout = _load_script("build_fastwin_holdout.py")
evaluate_holdout = _load_script("evaluate_fastwin_holdout.py")


def test_training_cutoff_uses_latest_recorded_backtest_timestamp(tmp_path: Path) -> None:
    first = tmp_path / "training_data_first.csv"
    second = tmp_path / "training_data_second.csv"
    pd.DataFrame(
        {"backtest_timestamp": ["2026-06-01T00:00:00Z", "2026-06-02T00:00:00Z"]}
    ).to_csv(first, index=False)
    pd.DataFrame(
        {"backtest_timestamp": ["2026-06-03T12:34:56Z"]}
    ).to_csv(second, index=False)

    cutoff = build_holdout._training_cutoff([first, second])

    assert cutoff == pd.Timestamp("2026-06-03T12:34:56Z")


def test_prior_holdout_sources_are_hashed_and_current_output_is_excluded(
    tmp_path: Path,
) -> None:
    prior = tmp_path / "fastwin_holdout_prior"
    prior.mkdir()
    prior_cohort = prior / "cohort.csv"
    pd.DataFrame({"candidate_id": ["PRIOR_1", "PRIOR_2", "PRIOR_2"]}).to_csv(
        prior_cohort, index=False
    )
    current = tmp_path / "fastwin_holdout_current"
    current.mkdir()
    pd.DataFrame({"candidate_id": ["CURRENT_1"]}).to_csv(
        current / "cohort.csv", index=False
    )

    candidate_ids, records = build_holdout._prior_holdout_sources(current)

    assert candidate_ids == {"PRIOR_1", "PRIOR_2"}
    assert records == [
        {
            "path": str(prior_cohort.resolve()),
            "sha256": build_holdout._sha256(prior_cohort),
            "size_bytes": prior_cohort.stat().st_size,
            "candidate_ids": 2,
        }
    ]


def test_evaluated_candidate_source_parse_failure_is_fail_closed(tmp_path: Path) -> None:
    malformed = tmp_path / "evaluated.csv"
    malformed.write_text("symbol\nBTCUSDT\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing candidate_id"):
        build_holdout._read_candidate_ids([malformed])


def test_freeze_excludes_candidate_consumed_by_prior_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fastwin_dir = tmp_path / "fastwin"
    fastwin_dir.mkdir()
    pd.DataFrame(
        {
            "candidate_id": ["TRAIN_1"],
            "backtest_timestamp": ["2026-06-30T00:00:00Z"],
        }
    ).to_csv(fastwin_dir / "training_data_1.csv", index=False)
    evaluated_dir = tmp_path / "evaluated"
    evaluated_dir.mkdir()
    prior = tmp_path / "fastwin_holdout_prior"
    prior.mkdir()
    prior_id = "BTCUSDT_20260701_120000_a1b2c3d4"
    pd.DataFrame({"candidate_id": [prior_id]}).to_csv(
        prior / "cohort.csv", index=False
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    scan_file = results_dir / "deployment_ready_20260701_130000.csv"
    scan_file.write_text("source\nscanner\n", encoding="utf-8")
    rows: list[dict[str, object]] = []
    for symbol, candidate_id in (
        ("BTCUSDT", prior_id),
        ("ETHUSDT", "ETHUSDT_20260701_120100_b1c2d3e4"),
    ):
        row: dict[str, object] = {
            "symbol": symbol,
            "candidate_id": candidate_id,
            "candidate_available_ts_utc": "2026-07-01T13:00:00Z",
            "scan_file": scan_file.name,
        }
        row.update({feature: 0.5 for feature in build_holdout.ACTIVE_SNAPSHOT_META_FEATURES})
        row.update({feature: 0.5 for feature in build_holdout.DEFAULT_FEATURES})
        rows.append(row)
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(build_holdout, "load_all_scanner_csvs", lambda _path: frame)
    monkeypatch.setattr(
        build_holdout,
        "filter_backtest_candidates",
        lambda loaded, **_kwargs: loaded,
    )

    output_dir = tmp_path / "fastwin_holdout_current"
    manifest = build_holdout.freeze_holdout(
        output_dir=output_dir,
        results_dir=results_dir,
        linkage_path=tmp_path / "linkage.csv",
        fastwin_dir=fastwin_dir,
        evaluated_dir=evaluated_dir,
        as_of=pd.Timestamp("2026-07-03T13:00:00Z").to_pydatetime(),
        min_rows=1,
        min_scan_groups=1,
    )

    cohort = pd.read_csv(output_dir / "cohort.csv")
    assert cohort["candidate_id"].tolist() == ["ETHUSDT_20260701_120100_b1c2d3e4"]
    assert manifest["selection"]["known_prior_holdout_candidate_ids"] == 1
    assert manifest["prior_holdout_sources"][0]["sha256"] == build_holdout._sha256(
        prior / "cohort.csv"
    )
    assert manifest["selection"]["unique_scan_groups"] == 1
    assert manifest["feature_contract"]["profile_features"] == list(
        build_holdout.DEFAULT_FEATURES
    )
    assert manifest["implementation_sources"]


def test_freeze_rejects_subthreshold_cohort_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fastwin_dir = tmp_path / "fastwin"
    fastwin_dir.mkdir()
    pd.DataFrame(
        {
            "candidate_id": ["TRAIN_1"],
            "backtest_timestamp": ["2026-06-30T00:00:00Z"],
        }
    ).to_csv(fastwin_dir / "training_data_1.csv", index=False)
    evaluated_dir = tmp_path / "evaluated"
    evaluated_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    scan_file = results_dir / "deployment_ready_20260701_130000.csv"
    scan_file.write_text("source\nscanner\n", encoding="utf-8")
    row: dict[str, object] = {
        "symbol": "ETHUSDT",
        "candidate_id": "ETHUSDT_20260701_120100_b1c2d3e4",
        "candidate_available_ts_utc": "2026-07-01T13:00:00Z",
        "scan_file": scan_file.name,
    }
    row.update({feature: 0.5 for feature in build_holdout.ACTIVE_SNAPSHOT_META_FEATURES})
    row.update({feature: 0.5 for feature in build_holdout.DEFAULT_FEATURES})
    frame = pd.DataFrame([row])
    monkeypatch.setattr(build_holdout, "load_all_scanner_csvs", lambda _path: frame)
    monkeypatch.setattr(
        build_holdout,
        "filter_backtest_candidates",
        lambda loaded, **_kwargs: loaded,
    )
    output_dir = tmp_path / "fastwin_holdout_current"

    with pytest.raises(ValueError, match="below the pre-registered evidence floor"):
        build_holdout.freeze_holdout(
            output_dir=output_dir,
            results_dir=results_dir,
            linkage_path=tmp_path / "linkage.csv",
            fastwin_dir=fastwin_dir,
            evaluated_dir=evaluated_dir,
            as_of=pd.Timestamp("2026-07-03T13:00:00Z").to_pydatetime(),
            min_rows=2,
            min_scan_groups=1,
        )

    assert not output_dir.exists()


def test_schema_three_resume_rejects_implementation_code_drift(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.csv"
    pd.DataFrame({"candidate_id": ["BTCUSDT_1"]}).to_csv(cohort, index=False)
    implementation = tmp_path / "runner.py"
    implementation.write_text("version = 1\n", encoding="utf-8")
    manifest = {
        "schema_version": 3,
        "engine_contract": {
            "engine_version": build_holdout.ENGINE_VERSION,
            "label_contract_version": build_holdout.LABEL_CONTRACT_VERSION,
            "formula_version": build_holdout.FORMULA_VERSION,
            "realism_profile": build_holdout.REALISM_PROFILE,
            "capital": 400.0,
            "leverage": 10,
        },
        "cohort": {
            "path": str(cohort),
            "sha256": build_holdout._sha256(cohort),
        },
        "selection": {"frozen_rows": 1},
        "scanner_sources": [],
        "implementation_sources": [
            {
                "path": str(implementation),
                "sha256": build_holdout._sha256(implementation),
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    implementation.write_text("version = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="implementation hash mismatch"):
        build_holdout._validate_frozen_contract(tmp_path)


def test_materialize_preserves_successes_and_surfaces_errors(tmp_path: Path) -> None:
    rows = tmp_path / "rows"
    rows.mkdir()
    build_holdout._atomic_write_json(
        rows / "ok.json",
        {
            "status": "ok",
            "candidate_id": "BTCUSDT_1",
            "backtest_result": {"candidate_id": "BTCUSDT_1", "net_pnl_pct": 1.0},
            "training_row": {"candidate_id": "BTCUSDT_1", "time_to_target_hours": 2.0},
        },
    )
    build_holdout._atomic_write_json(
        rows / "error.json",
        {
            "status": "error",
            "candidate_id": "ETHUSDT_1",
            "symbol": "ETHUSDT",
            "stage": "fetch_or_backtest",
            "error": "ValueError: insufficient_1m_bars",
        },
    )

    summary = build_holdout._materialize(tmp_path)

    assert summary == {"checkpointed": 2, "ok": 1, "errors": 1}
    assert pd.read_csv(tmp_path / "backtest_results.csv")["candidate_id"].tolist() == [
        "BTCUSDT_1"
    ]
    errors = pd.read_csv(tmp_path / "errors.csv")
    assert errors.loc[0, "candidate_id"] == "ETHUSDT_1"
    assert "insufficient_1m_bars" in errors.loc[0, "error"]


def test_cluster_bootstrap_detects_clear_paired_auc_improvement() -> None:
    y = np.asarray([0, 1] * 20, dtype=int)
    incumbent = np.full(len(y), 0.5, dtype=float)
    challenger = np.where(y == 1, 0.9, 0.1).astype(float)
    clusters = np.asarray([f"scan_{index // 4}" for index in range(len(y))])

    result = evaluate_holdout._paired_cluster_bootstrap(
        y,
        incumbent,
        challenger,
        clusters,
        iterations=200,
        rng=np.random.default_rng(7),
    )

    assert result["valid_iterations"] == 200
    assert result["delta_auc_ci_95"] is not None
    assert result["delta_auc_ci_95"][0] > 0.0


def test_evaluate_writes_fail_closed_promotion_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_ids = [f"CANDIDATE_{index}" for index in range(20)]
    labels = np.asarray([0, 1] * 10, dtype=int)
    time_to_target = [np.nan if label == 0 else 2.0 for label in labels]
    cohort = pd.DataFrame(
        {
            "candidate_id": candidate_ids,
            "symbol": [f"SYM{index % 5}" for index in range(20)],
            "scan_file": [f"scan_{index // 4}.csv" for index in range(20)],
        }
    )
    training = cohort[["candidate_id", "symbol"]].copy()
    training["time_to_target_hours"] = time_to_target
    cohort.to_csv(tmp_path / "cohort.csv", index=False)
    training.to_csv(tmp_path / "training_data.csv", index=False)
    (tmp_path / "run_summary.json").write_text(
        json.dumps({"complete": True, "errors": 0}), encoding="utf-8"
    )
    incumbent = tmp_path / "incumbent.pkl"
    challenger = tmp_path / "challenger.pkl"
    incumbent.write_bytes(b"incumbent")
    challenger.write_bytes(b"challenger")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "target_contract": {"name": "fast_winner_time_to_3pct_le_7h"},
                "incumbent_artifacts": [
                    {
                        "path": str(incumbent),
                        "sha256": evaluate_holdout._sha256(incumbent),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    incumbent_probability = np.full(len(labels), 0.5, dtype=float)
    challenger_probability = labels.astype(float)

    def _load(path: Path):
        probability = (
            incumbent_probability if path.name == incumbent.name else challenger_probability
        )
        return SimpleNamespace(predict_proba_batch=lambda frame: probability)

    monkeypatch.setattr(evaluate_holdout.MetaLabeler, "load", _load)

    report = evaluate_holdout.evaluate(
        holdout_dir=tmp_path,
        incumbent_path=incumbent,
        challenger_path=challenger,
        bootstrap_iterations=200,
        random_state=11,
    )

    assert report["promotion"]["promote"] is True
    assert report["paired_delta_auc"] == pytest.approx(0.5)
    assert (tmp_path / "evaluation.json").exists()
    assert (tmp_path / "paired_predictions.csv").exists()
