from __future__ import annotations

import importlib.util
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


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "finalize_authoritative_meta_pool.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "finalize_authoritative_meta_pool", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


_ACTIVE_VERSION = "rolling_180d_20260815_214954"
_TRAINED_AT = "2026-08-15T23:14:45.834213+00:00"


def _active_contract() -> dict[str, str]:
    return {
        "artifact_version": _ACTIVE_VERSION,
        "trained_at_utc": _TRAINED_AT,
        "pipeline_version": "6.5.8",
        "feature_semantics_version": _MODULE.HMM_FEATURE_SEMANTICS_VERSION,
        "metadata_path": "artifacts/hmm/test/metadata.json",
        "metadata_sha256": "a" * 64,
    }


def _row(candidate_id: str, symbol: str, start_time: str) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "symbol": symbol,
        "start_time_utc": start_time,
        "feature_cutoff_utc": start_time,
        "label_contract_version": LABEL_CONTRACT_VERSION,
        "engine_version": ENGINE_VERSION,
        "formula_version": FORMULA_VERSION,
        "mode": "geometric",
        "fill_mode": "wick",
        "global_cooldown_bars": 0,
        "cb_enabled": False,
        "is_authoritative": True,
        "realism_profile": "legacy",
        "hmm_artifact_version": _ACTIVE_VERSION,
        "hmm_trained_at_utc": _TRAINED_AT,
        "hmm_pipeline_version": "6.5.8",
        "hmm_feature_semantics_version": _MODULE.HMM_FEATURE_SEMANTICS_VERSION,
        "hmm_feature_source": "pinned_artifact_replay",
        "hmm_replay_scope": "hmm_lineage_only",
        "range_prob": 0.70,
        "trend_prob": 0.20,
        "persistence_prob": 0.80,
        "pnl_pct": 1.25,
        "y": 1,
        "sl_hit": False,
        "duration_hours": 4.0,
        "time_to_target_hours": 3.5,
    }
    for feature in _MODULE.ACTIVE_SNAPSHOT_META_FEATURES:
        row.setdefault(feature, 0.5)
    return row


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    rows = [
        _row(
            "BTCUSDT_20260608_120000_deadbeef",
            "BTCUSDT",
            "2026-06-08T12:00:00+00:00",
        ),
        _row(
            "ETHUSDT_20260817_010203_cafebabe",
            "ETHUSDT",
            "2026-08-17T01:02:03+00:00",
        ),
    ]
    source = tmp_path / "source.csv"
    replay = tmp_path / "replay.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    pd.DataFrame(rows).to_csv(replay, index=False)
    return source, replay


def test_finalizer_publishes_only_strictly_valid_rows_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, replay = _write_pair(tmp_path)
    output_dir = tmp_path / "authoritative"
    monkeypatch.setattr(_MODULE, "_load_active_contract", lambda _version: _active_contract())

    manifest = _MODULE.finalize_pool(
        source_path=source,
        replay_path=replay,
        output_dir=output_dir,
        start_date="2026-06-08",
        end_date="2026-08-17",
        active_hmm_artifact_version=_ACTIVE_VERSION,
    )

    assert manifest["input_rows"] == 2
    assert manifest["eligible_rows"] == 2
    assert manifest["excluded_rows"] == 0
    pool = pd.read_csv(output_dir / "training_data_20260817.csv")
    assert len(pool) == 2
    assert bool(
        pool["pool_contract_version"].eq(_MODULE.POOL_CONTRACT_VERSION).all()
    )
    stored_manifest = json.loads(
        (output_dir / "authoritative_pool_manifest_20260817.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored_manifest["pool_sha256"] == manifest["pool_sha256"]
    assert not list(tmp_path.glob(".authoritative.*.tmp"))


def test_finalizer_excludes_failed_lineage_without_repairing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, replay = _write_pair(tmp_path)
    replay_frame = pd.read_csv(replay)
    replay_frame.loc[0, "hmm_artifact_version"] = "stale_hmm"
    replay_frame.loc[0, "hmm_feature_source"] = "artifact_inference_failed"
    replay_frame.to_csv(replay, index=False)
    output_dir = tmp_path / "authoritative"
    monkeypatch.setattr(_MODULE, "_load_active_contract", lambda _version: _active_contract())

    manifest = _MODULE.finalize_pool(
        source_path=source,
        replay_path=replay,
        output_dir=output_dir,
        start_date="2026-06-08",
        end_date="2026-08-17",
        active_hmm_artifact_version=_ACTIVE_VERSION,
    )

    assert manifest["eligible_rows"] == 1
    assert manifest["excluded_rows"] == 1
    excluded = pd.read_csv(output_dir / "excluded_rows_20260817.csv")
    assert excluded["candidate_id"].tolist() == [
        "BTCUSDT_20260608_120000_deadbeef"
    ]
    assert "hmm_artifact_version_mismatch" in excluded.loc[0, "exclusion_reasons"]
    assert "artifact_inference_failed" not in set(
        pd.read_csv(output_dir / "training_data_20260817.csv")[
            "hmm_feature_source"
        ]
    )


def test_finalizer_rejects_outcome_mutation_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, replay = _write_pair(tmp_path)
    replay_frame = pd.read_csv(replay)
    replay_frame.loc[0, "pnl_pct"] = 99.0
    replay_frame.to_csv(replay, index=False)
    output_dir = tmp_path / "authoritative"
    monkeypatch.setattr(_MODULE, "_load_active_contract", lambda _version: _active_contract())

    with pytest.raises(ValueError, match="modified outcome column 'pnl_pct'"):
        _MODULE.finalize_pool(
            source_path=source,
            replay_path=replay,
            output_dir=output_dir,
            start_date="2026-06-08",
            end_date="2026-08-17",
            active_hmm_artifact_version=_ACTIVE_VERSION,
        )

    assert not output_dir.exists()

def test_outcome_comparator_accepts_only_machine_precision_csv_float_noise() -> None:
    source = pd.DataFrame(
        [_row("BTCUSDT_20260608_120000_deadbeef", "BTCUSDT", "2026-06-08T12:00:00Z")]
    )
    replay = source.copy()
    replay.loc[0, "pnl_pct"] = np.nextafter(
        float(source.loc[0, "pnl_pct"]), np.inf
    )

    _MODULE._assert_outcomes_unchanged(source, replay)

    replay.loc[0, "pnl_pct"] = float(source.loc[0, "pnl_pct"]) + 1e-10
    with pytest.raises(ValueError, match="modified outcome column 'pnl_pct'"):
        _MODULE._assert_outcomes_unchanged(source, replay)


def test_finalizer_accepts_canonical_unicode_symbol_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _row(
        "币安人生USDT_20260626_140532_e6e12c06",
        "币安人生USDT",
        "2026-06-26T14:05:32+00:00",
    )
    source = tmp_path / "source.csv"
    replay = tmp_path / "replay.csv"
    pd.DataFrame([row]).to_csv(source, index=False)
    pd.DataFrame([row]).to_csv(replay, index=False)
    output_dir = tmp_path / "authoritative"
    monkeypatch.setattr(_MODULE, "_load_active_contract", lambda _version: _active_contract())

    manifest = _MODULE.finalize_pool(
        source_path=source,
        replay_path=replay,
        output_dir=output_dir,
        start_date="2026-06-08",
        end_date="2026-08-17",
        active_hmm_artifact_version=_ACTIVE_VERSION,
    )

    assert manifest["eligible_rows"] == 1
    assert pd.read_csv(output_dir / "training_data_20260817.csv").loc[
        0, "candidate_id"
    ] == row["candidate_id"]


def test_finalizer_requires_candidate_scan_feature_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _row(
        "BTCUSDT_20260608_120000_deadbeef",
        "BTCUSDT",
        "2026-06-08T12:45:00+00:00",
    )
    # A cutoff at outcome start is causal in the weak sense, but it does not
    # match the scanner-origin feature timestamp embedded in candidate_id.
    source = tmp_path / "source.csv"
    replay = tmp_path / "replay.csv"
    pd.DataFrame([row]).to_csv(source, index=False)
    pd.DataFrame([row]).to_csv(replay, index=False)
    output_dir = tmp_path / "authoritative"
    monkeypatch.setattr(_MODULE, "_load_active_contract", lambda _version: _active_contract())

    with pytest.raises(ValueError, match="zero eligible rows"):
        _MODULE.finalize_pool(
            source_path=source,
            replay_path=replay,
            output_dir=output_dir,
            start_date="2026-06-08",
            end_date="2026-08-17",
            active_hmm_artifact_version=_ACTIVE_VERSION,
        )

    assert not output_dir.exists()
