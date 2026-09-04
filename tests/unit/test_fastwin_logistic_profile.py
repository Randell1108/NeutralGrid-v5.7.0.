from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neutralgrid.core.constants import (
    ENGINE_VERSION,
    FORMULA_VERSION,
    LABEL_CONTRACT_VERSION,
)
from neutralgrid.scanner.fastwin_logistic_profile import (
    C_VALUES,
    INNER_FOLDS,
    MODEL_FAMILY,
    OUTER_FOLDS,
    assemble_development_pool,
    evaluate_development,
    sha256_file,
)
from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES
from neutralgrid.scanner.profile_model import (
    ProfileModel,
    load_profile_model,
    save_profile_model,
)


def _rows(*, start_group: int, groups: int, rows_per_group: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    for group in range(start_group, start_group + groups):
        timestamp = base + pd.Timedelta(days=2 * group)
        for index in range(rows_per_group):
            label = (group + index) % 2
            signed = 1.0 if label else -1.0
            row: dict[str, object] = {
                "candidate_id": f"CANDIDATE_{group:03d}_{index:02d}",
                "symbol": f"SYM{index:02d}USDT",
                "start_time_utc": timestamp.isoformat(),
                "fastwin_label": label,
            }
            for feature_index, feature in enumerate(DEFAULT_FEATURES):
                row[feature] = signed * (1.0 + 0.1 * feature_index) + index * 0.001
            rows.append(row)
    return pd.DataFrame(rows)


def _neutral_incumbent(path: Path) -> None:
    save_profile_model(
        ProfileModel(
            features=list(DEFAULT_FEATURES),
            winner_mu={feature: 0.0 for feature in DEFAULT_FEATURES},
            loser_mu={feature: 0.0 for feature in DEFAULT_FEATURES},
            inv_cov=np.eye(len(DEFAULT_FEATURES)).tolist(),
            prior_winner=0.25,
            feature_mean={feature: 0.0 for feature in DEFAULT_FEATURES},
            feature_std={feature: 1.0 for feature in DEFAULT_FEATURES},
            feature_impute={feature: 0.0 for feature in DEFAULT_FEATURES},
        ),
        path,
    )


def _write_preregistered_sources(tmp_path: Path) -> Path:
    older_one = tmp_path / "older_one.csv"
    older_two = tmp_path / "older_two.csv"
    newer_cohort = tmp_path / "newer_cohort.csv"
    newer_training = tmp_path / "newer_training.csv"
    _rows(start_group=0, groups=30).to_csv(older_one, index=False)
    _rows(start_group=30, groups=20).to_csv(older_two, index=False)
    newer = _rows(start_group=50, groups=30)
    newer[["candidate_id", *DEFAULT_FEATURES]].to_csv(newer_cohort, index=False)
    outcomes = newer[["candidate_id", "symbol", "start_time_utc"]].copy()
    labels = np.asarray(newer["fastwin_label"], dtype=int)
    outcomes["time_to_target_hours"] = np.where(labels == 1, 2.0, np.nan)
    outcomes["source"] = "backtest"
    outcomes["is_authoritative"] = True
    outcomes["engine_version"] = ENGINE_VERSION
    outcomes["label_contract_version"] = LABEL_CONTRACT_VERSION
    outcomes["formula_version"] = FORMULA_VERSION
    outcomes["realism_profile"] = "candidate_time_geometric_v1"
    outcomes.to_csv(newer_training, index=False)
    incumbent = tmp_path / "incumbent.json"
    pattern = tmp_path / "pattern.json"
    _neutral_incumbent(incumbent)
    pattern.write_text(json.dumps({"features": list(DEFAULT_FEATURES)}), encoding="utf-8")
    source_records = []
    for path in (older_one, older_two, newer_cohort, newer_training):
        source_records.append({"path": str(path), "sha256": sha256_file(path)})
    preregistration = {
        "schema_version": 1,
        "challenger": {
            "model_family": MODEL_FAMILY,
            "candidate_c_values": list(C_VALUES),
        },
        "feature_contract": {"features": list(DEFAULT_FEATURES)},
        "development_protocol": {
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "purge_hours": 24.0,
        },
        "development_sources": source_records,
        "implementation_sources": [
            {
                "path": str(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
            }
        ],
        "incumbent": {
            "model_path": str(incumbent),
            "model_sha256": sha256_file(incumbent),
            "pattern_path": str(pattern),
            "pattern_sha256": sha256_file(pattern),
        },
    }
    preregistration_path = tmp_path / "preregistration.json"
    preregistration_path.write_text(
        json.dumps(preregistration, indent=2), encoding="utf-8"
    )
    return preregistration_path


def test_hash_mismatch_fails_before_development_rows_are_assembled(
    tmp_path: Path,
) -> None:
    preregistration_path = _write_preregistered_sources(tmp_path)
    (tmp_path / "older_one.csv").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source hash mismatch"):
        assemble_development_pool(preregistration_path)


def test_nested_logistic_development_writes_shadow_bundle_only(tmp_path: Path) -> None:
    preregistration_path = _write_preregistered_sources(tmp_path)
    output_dir = tmp_path / "shadow"

    report = evaluate_development(
        preregistration_path=preregistration_path,
        output_dir=output_dir,
    )

    pool = report["development_pool"]
    assert pool["rows"] == 800
    assert pool["candidate_ids_unique"] == 800
    assert pool["scan_groups"] == 80
    assert pool["positives"] == 400
    assert pool["negatives"] == 400
    assert report["candidate_walkforward"]["mean_auc"] == pytest.approx(1.0)
    assert report["candidate_walkforward"]["pooled_oof_auc"] == pytest.approx(1.0)
    assert report["paired_cluster_bootstrap"]["delta_auc_ci_95"][0] > 0.0
    assert report["decision"] == {
        "fresh_holdout_eligible": True,
        "promotion_eligible": False,
        "promotion_attempted": False,
        "production_artifacts_modified": False,
        "reasons": [],
        "promotion_blocker": "genuinely_new_canonical_holdout_required",
    }
    model = load_profile_model(output_dir / "candidate_profile_model.json")
    assert model.model_family == MODEL_FAMILY
    assert model.selection_summary is not None
    assert model.selection_summary["selected_c"] == min(C_VALUES)
    manifest = json.loads((output_dir / "candidate_manifest.json").read_text())
    assert manifest["activation_status"] == "shadow_not_promotable_without_fresh_holdout"
    assert manifest["production_artifacts_modified"] is False
