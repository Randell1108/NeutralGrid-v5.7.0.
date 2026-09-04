from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "audit_canonical_fastwin_profile.py"
_SPEC = importlib.util.spec_from_file_location(_SCRIPT.stem, _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit_script)


def _canonical_rows(count: int = 60) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(count):
        positive = index % 2 == 0
        row: dict[str, object] = {
            "candidate_id": f"CANDIDATE_{index:04d}",
            "symbol": f"SYM{index % 7}USDT",
            "start_time_utc": f"2026-01-{(index % 28) + 1:02d}T00:00:00Z",
            "backtest_timestamp": f"2026-02-{(index % 28) + 1:02d}T00:00:00Z",
            "time_to_target_hours": 2.0 if positive else float("nan"),
            "target_reached": positive,
            "source": "backtest",
            "is_authoritative": True,
            "engine_version": audit_script.ENGINE_VERSION,
            "label_contract_version": audit_script.LABEL_CONTRACT_VERSION,
            "formula_version": audit_script.FORMULA_VERSION,
            "realism_profile": audit_script.REALISM_PROFILE,
            # This flag describes the barrier horizon, not row origin.
            "t1_is_synthetic": True,
        }
        row.update(
            {
                feature: float(index + feature_index) / 100.0
                for feature_index, feature in enumerate(audit_script.DEFAULT_FEATURES)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _write(frame: pd.DataFrame, directory: Path) -> None:
    directory.mkdir()
    frame.to_csv(directory / "training_data_fastwin_fixture.csv", index=False)


def test_authoritative_backtest_rows_are_fit_ready_but_not_promotion_evidence(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "canonical"
    _write(_canonical_rows(), data_dir)

    report = audit_script.audit_canonical_fastwin_profile(data_dir)

    assert report["fit_assessment"] == {"viable": True, "blockers": []}
    assert report["promotion_assessment"]["promote"] is False
    assert report["promotion_assessment"]["blockers"] == [
        "fresh_unused_temporal_holdout_required"
    ]
    assert report["target_contract"]["positive_rows"] == 30
    assert report["target_contract"]["negative_rows"] == 30
    assert report["row_origin_contract"]["t1_is_synthetic_value_counts"] == {
        "true": 60
    }


def test_non_backtest_rows_fail_the_origin_contract(tmp_path: Path) -> None:
    data_dir = tmp_path / "canonical"
    frame = _canonical_rows()
    frame.loc[0, "source"] = "synthetic"
    _write(frame, data_dir)

    report = audit_script.audit_canonical_fastwin_profile(data_dir)

    assert report["fit_assessment"]["viable"] is False
    assert "source_mismatch_rows=1" in report["fit_assessment"]["blockers"]
    assert report["row_origin_contract"]["synthetic_training_rows_allowed"] is False


def test_duplicate_candidate_ids_fail_closed(tmp_path: Path) -> None:
    data_dir = tmp_path / "canonical"
    frame = _canonical_rows()
    frame.loc[1, "candidate_id"] = frame.loc[0, "candidate_id"]
    _write(frame, data_dir)

    report = audit_script.audit_canonical_fastwin_profile(data_dir)

    assert report["identity"]["duplicate_candidate_id_rows"] == 2
    assert "duplicate_candidate_id_rows=2" in report["fit_assessment"]["blockers"]


def test_missing_profile_features_make_accuracy_not_measurable(tmp_path: Path) -> None:
    data_dir = tmp_path / "canonical"
    frame = _canonical_rows().drop(columns=audit_script.DEFAULT_FEATURES)
    _write(frame, data_dir)

    report = audit_script.audit_canonical_fastwin_profile(data_dir)

    assert report["feature_contract"]["feature_complete_rows"] == 0
    assert report["accuracy_assessment"]["status"] == "not_measurable"
    assert "required_columns_missing" in report["fit_assessment"]["blockers"]
    assert "profile_feature_incomplete_rows=60" in report["fit_assessment"]["blockers"]


def test_exact_target_includes_seven_hours_and_excludes_later_hits(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "canonical"
    frame = _canonical_rows()
    frame.loc[0, "time_to_target_hours"] = 7.0
    frame.loc[0, "target_reached"] = True
    frame.loc[1, "time_to_target_hours"] = 7.000001
    frame.loc[1, "target_reached"] = True
    _write(frame, data_dir)

    report = audit_script.audit_canonical_fastwin_profile(data_dir)

    assert report["target_contract"]["positive_rows"] == 30
    assert report["target_contract"]["negative_rows"] == 30
