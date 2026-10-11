"""CSV admission after operator-requested capture-reference removal."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import sys

import pandas as pd
import pytest

from neutralgrid.backtest.candidate_pipeline import load_all_scanner_csvs


def frame(score=75):
    return pd.DataFrame([{'candidate_id': 'BTCUSDT_20260901_110000_12345678',
                         'symbol': 'BTCUSDT', 'score': score, 'grid_lower': 90,
                         'grid_upper': 110, 'num_grids': 20}])


def test_plain_csv_admitted_without_capture_and_old_reference_columns_removed(tmp_path):
    p = tmp_path / 'deployment_ready_20260901_110000.csv'
    f = frame()
    for name in ['candidate_capture_path', 'candidate_capture_sha256', 'candidate_capture_manifest_sha256']:
        f[name] = 'retired'
    f.to_csv(p, index=False)
    p.with_suffix('.csv.capture.json').write_text('{malformed retired sidecar')
    loaded = load_all_scanner_csvs(tmp_path, require_meta_features=False)
    assert len(loaded) == 1
    assert not any(c.startswith('candidate_capture_') for c in loaded.columns)
    assert loaded.iloc[0]['candidate_available_source'] == 'legacy_mtime_unverified'


def test_identical_csv_copies_deduplicate_and_conflicting_identity_is_quarantined(tmp_path):
    a = tmp_path / 'deployment_ready_a.csv'
    b = tmp_path / 'deployment_ready_b.csv'
    frame().to_csv(a, index=False)
    b.write_bytes(a.read_bytes())
    assert len(load_all_scanner_csvs(tmp_path, require_meta_features=False)) == 1
    frame(99).to_csv(tmp_path / 'deployment_ready_conflict.csv', index=False)
    with pytest.raises(ValueError, match='source-conflict'):
        load_all_scanner_csvs(tmp_path, require_meta_features=False)


def test_plain_csv_real_ingestion_finalization_and_admission(tmp_path, monkeypatch):
    from scripts import finalize_fresh_authoritative_meta_pool as finalizer
    from neutralgrid.core.constants import LABEL_CONTRACT_VERSION, ENGINE_VERSION, FORMULA_VERSION
    from neutralgrid.training.data_generator import HMM_FEATURE_SEMANTICS_VERSION
    from neutralgrid.training.meta_pool_contract import validate_retrain_pool_source

    frame().to_csv(tmp_path / 'deployment_ready_20260901_110000.csv', index=False)
    training = load_all_scanner_csvs(tmp_path, require_meta_features=False)
    training['start_time_utc'] = '2026-09-01T12:00:00+00:00'
    for key, value in {
        'label_contract_version': LABEL_CONTRACT_VERSION, 'engine_version': ENGINE_VERSION,
        'formula_version': FORMULA_VERSION, 'mode': 'geometric', 'fill_mode': 'wick',
        'global_cooldown_bars': 0, 'cb_enabled': False, 'is_authoritative': True,
        'realism_profile': 'legacy', 'hmm_artifact_version': 'fixture_hmm',
        'hmm_feature_source': 'pinned_artifact_replay', 'hmm_replay_scope': 'hmm_lineage_only',
        'hmm_feature_semantics_version': HMM_FEATURE_SEMANTICS_VERSION,
    }.items():
        training[key] = value
    for feature in finalizer.ACTIVE_SNAPSHOT_META_FEATURES:
        training[feature] = 0.1
    source = tmp_path / 'source.csv'
    training.to_csv(source, index=False)
    run = tmp_path / 'run.json'
    run.write_text(json.dumps({'generation_mode': 'fresh_full_pool', 'full_pool': True,
        'realism_profile': 'legacy', 'max_candidates': None, 'hours': 7,
        'min_bars': 420, 'successful_rows': 1}))
    monkeypatch.setattr(finalizer, '_load_active_contract', lambda version: {'artifact_version': version})
    output = tmp_path / 'finalized'
    result = finalizer.finalize_fresh_pool(source_path=source, run_manifest_path=run,
        output_dir=output, start_date='2026-09-01', end_date='2026-09-01',
        active_hmm_artifact_version='fixture_hmm')
    assert 'availability_contract' not in result
    assert validate_retrain_pool_source(output)['pool_sha256'] == result['pool_sha256']


@pytest.mark.asyncio
async def test_latest_only_uses_legacy_file_time_without_capture_references(tmp_path, monkeypatch):
    import backtest_candidates as cli
    from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES

    old = tmp_path / 'deployment_ready_old.csv'
    new = tmp_path / 'deployment_ready_new.csv'
    f = frame()
    for feature in ACTIVE_SNAPSHOT_META_FEATURES:
        f[feature] = 0.1
    f.to_csv(old, index=False)
    newer = f.copy()
    newer['candidate_id'] = 'BTCUSDT_20260901_120000_abcdef12'
    newer.to_csv(new, index=False)
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    monkeypatch.setattr(sys, 'argv', ['backtest_candidates.py', '--results-dir', str(tmp_path),
        '--latest-only', '--pool-start-date', '2026-09-01'])
    monkeypatch.setattr(cli, '_load_authoritative_candidate_ids', lambda *args: set())
    class SelectionObserved(Exception):
        pass
    def observe(selected, **kwargs):
        assert selected['candidate_id'].tolist() == newer['candidate_id'].tolist()
        raise SelectionObserved()
    monkeypatch.setattr(cli, 'filter_backtest_candidates', observe)
    with pytest.raises(SelectionObserved):
        await cli.main()
