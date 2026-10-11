from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from neutralgrid.training.meta_pool_contract import (
    FRESH_FULL_POOL_GENERATION_MODE,
    HISTORICAL_EXACT_REPLAY_GENERATION_MODE,
    POOL_CONTRACT_VERSION,
    validate_retrain_pool_source,
)


def _write_manifest(path: Path, mode: str) -> None:
    path.mkdir()
    frame = pd.DataFrame([{
        'candidate_id': 'BTCUSDT_20200101_000000_12345678', 'symbol': 'BTCUSDT'
    }])
    frame['start_time_utc'] = datetime.now(timezone.utc).isoformat()
    pool = path / 'training_data_fixture.csv'
    frame.to_csv(pool, index=False)
    (path / "authoritative_pool_manifest.json").write_text(
        json.dumps(
            {
                "generation_mode": mode,
                "pool_contract_version": POOL_CONTRACT_VERSION,
                "active_hmm": {"artifact_version": "rolling_active"},
                "pool_file": pool.name,
                "pool_sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
                "eligible_rows": 1,
            }
        ),
        encoding="utf-8",
    )


def test_fresh_full_pool_is_admitted(tmp_path: Path) -> None:
    source = tmp_path / "fresh"
    _write_manifest(source, FRESH_FULL_POOL_GENERATION_MODE)

    manifest = validate_retrain_pool_source(source)

    assert manifest["generation_mode"] == FRESH_FULL_POOL_GENERATION_MODE


def test_historical_replay_requires_explicit_opt_in(tmp_path: Path) -> None:
    source = tmp_path / "replay"
    _write_manifest(source, HISTORICAL_EXACT_REPLAY_GENERATION_MODE)

    with pytest.raises(ValueError, match="Historical exact-replay"):
        validate_retrain_pool_source(source)

    assert validate_retrain_pool_source(source, allow_historical_replay=True)[
        "generation_mode"
    ] == HISTORICAL_EXACT_REPLAY_GENERATION_MODE


def test_unmanifested_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No authoritative_pool_manifest"):
        validate_retrain_pool_source(tmp_path / "unmanifested")


def test_fresh_pool_does_not_require_retired_capture_fields(tmp_path: Path) -> None:
    source = tmp_path / 'legacy_fresh'
    _write_manifest(source, FRESH_FULL_POOL_GENERATION_MODE)
    path = source / 'authoritative_pool_manifest.json'
    payload = json.loads(path.read_text())
    assert 'availability_contract' not in payload
    assert not any(c.startswith('candidate_capture_') for c in pd.read_csv(source / payload['pool_file']).columns)
    assert validate_retrain_pool_source(source)['eligible_rows'] == 1


@pytest.mark.parametrize('failure', ['changed_bytes', 'extra_input', 'changed_count', 'duplicate_ids'])
def test_finalized_fresh_pool_cannot_drift_before_retraining(tmp_path, failure):
    source = tmp_path / 'fresh'
    _write_manifest(source, FRESH_FULL_POOL_GENERATION_MODE)
    pool = source / 'training_data_fixture.csv'
    if failure == 'changed_bytes':
        pool.write_bytes(pool.read_bytes() + b'\n')
    elif failure == 'extra_input':
        (source / 'backtest_results_extra.csv').write_text('candidate_id\nBAD\n')
    elif failure == 'duplicate_ids':
        frame = pd.read_csv(pool)
        pd.concat([frame, frame], ignore_index=True).to_csv(pool, index=False)
        marker = source / 'authoritative_pool_manifest.json'
        payload = json.loads(marker.read_text())
        payload['eligible_rows'] = 2
        payload['pool_sha256'] = hashlib.sha256(pool.read_bytes()).hexdigest()
        marker.write_text(json.dumps(payload))
    else:
        marker = source / 'authoritative_pool_manifest.json'
        payload = json.loads(marker.read_text())
        payload['eligible_rows'] = 2
        marker.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validate_retrain_pool_source(source)
