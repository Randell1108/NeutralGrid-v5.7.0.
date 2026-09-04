from __future__ import annotations

import json
from pathlib import Path

import pytest

from neutralgrid.training.meta_pool_contract import (
    FRESH_FULL_POOL_GENERATION_MODE,
    HISTORICAL_EXACT_REPLAY_GENERATION_MODE,
    POOL_CONTRACT_VERSION,
    validate_retrain_pool_source,
)


def _write_manifest(path: Path, mode: str) -> None:
    path.mkdir()
    (path / "authoritative_pool_manifest.json").write_text(
        json.dumps(
            {
                "generation_mode": mode,
                "pool_contract_version": POOL_CONTRACT_VERSION,
                "active_hmm": {"artifact_version": "rolling_active"},
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
