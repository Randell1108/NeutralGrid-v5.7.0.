"""Contracts separating fresh backtest pools from historical replays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


POOL_CONTRACT_VERSION = "active_hmm_authoritative_meta_pool_v1"
FRESH_FULL_POOL_GENERATION_MODE = "fresh_full_pool"
HISTORICAL_EXACT_REPLAY_GENERATION_MODE = "historical_exact_replay"
AUTHORITATIVE_POOL_MANIFEST = "authoritative_pool_manifest.json"
BACKTEST_RUN_MANIFEST = "backtest_run_manifest.json"


def load_pool_manifest(source_dir: Path) -> dict[str, Any]:
    """Load the canonical pool manifest, rejecting unmanifested directories."""
    candidates = [source_dir / AUTHORITATIVE_POOL_MANIFEST]
    candidates.extend(sorted(source_dir.glob("authoritative_pool_manifest_*.json")))
    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Pool manifest is not an object: {path}")
            payload["_manifest_path"] = str(path)
            return payload
    raise ValueError(
        f"No {AUTHORITATIVE_POOL_MANIFEST} found in {source_dir}; "
        "raw backtest directories and static legacy pools are not retrain sources"
    )


def validate_retrain_pool_source(
    source_dir: Path,
    *,
    allow_historical_replay: bool = False,
) -> dict[str, Any]:
    """Require a finalized fresh full-pool source for active FASTWIN retraining."""
    manifest = load_pool_manifest(source_dir)
    generation_mode = str(manifest.get("generation_mode", "")).strip()
    if generation_mode == HISTORICAL_EXACT_REPLAY_GENERATION_MODE:
        if not allow_historical_replay:
            raise ValueError(
                "Historical exact-replay pool supplied to active FASTWIN retraining; "
                "generate a fresh full pool or pass --allow-historical-replay for "
                "an explicit diagnostic-only comparison"
            )
    elif generation_mode != FRESH_FULL_POOL_GENERATION_MODE:
        raise ValueError(
            "Pool manifest generation_mode must be 'fresh_full_pool'; "
            f"observed {generation_mode!r}"
        )

    if manifest.get("pool_contract_version") != POOL_CONTRACT_VERSION:
        raise ValueError(
            "Pool manifest is not a finalized active-HMM pool: "
            f"observed contract={manifest.get('pool_contract_version')!r}"
        )
    active_hmm = manifest.get("active_hmm")
    if not isinstance(active_hmm, dict) or not str(
        active_hmm.get("artifact_version", "")
    ).strip():
        raise ValueError("Pool manifest lacks the active_hmm artifact version")
    return manifest
