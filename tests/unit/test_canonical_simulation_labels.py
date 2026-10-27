from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neutralgrid.models.meta_labeler import ACTIVE_META_TARGET_CONTRACT, ACTIVE_SNAPSHOT_META_FEATURES
from neutralgrid.training.canonical_simulation_labels import (
    EVIDENCE_SCHEMA, LABEL_FIELDS, LABEL_SOURCE, SHEET_NAME, apply_canonical_simulation_labels,
)
from neutralgrid.training.unified_training_builder import UnifiedTrainingBuilder
from retrain_meta_labeler import prepare_fast_target_training_frame


@pytest.fixture
def cohort(tmp_path: Path, monkeypatch):
    workbook=tmp_path/'bots.xlsx'
    workbook.write_bytes(b'workbook fixture: spreadsheet reads mocked')
    source=tmp_path/'engine.json'
    source.write_text('{"engine":"test"}')
    evidence_path=tmp_path/'evidence.json'
    config={'price_range_low':100.,'price_range_high':110.,'grids_count':20.,
            'invested_margin_usdt':100.,'leverage':10.,'mode':'geometric'}
    features={name:0.2 for name in ACTIVE_SNAPSHOT_META_FEATURES}
    features['num_grids']=20.
    live=[]; records=[]; general=[]
    for sid,ttt in [('123',2.0),('124',None)]:
        identity={'strategy_id':sid,'candidate_id':'BTCUSDT_20261008_120000_12345678','symbol':'BTCUSDT'}
        base={**identity,'start_time_utc':'2026-10-08T13:00:30Z','t1':'2026-10-09T13:00:30Z',
              'net_pnl_pct':-2.,'pnl_pct':-2.,'duration_hours':24.,'source':'live',
              'sample_weight_override':1.,'hmm_artifact_version':'hmm_test',
              'feature_cutoff_utc':'2026-10-08T12:00:00Z','hmm_feature_cutoff_utc':'2026-10-08T12:00:00Z',**features}
        live.append(base)
        general.append({**identity,**config})
        records.append({**identity,'start_time_utc':base['start_time_utc'],'t1':'2026-10-08T20:01:00Z',
                        'net_pnl_pct':4. if ttt is not None else -1.,'time_to_target_hours':ttt,
                        'target_reached':ttt is not None,'duration_hours':7.,'features':features.copy(),
                        'canonical_configuration':config.copy(),'hmm_artifact_version':'hmm_test',
                        'capital_fraction':1.,'capital_used':100.,
                        'feature_cutoff_utc':base['feature_cutoff_utc'],'hmm_feature_cutoff_utc':base['hmm_feature_cutoff_utc']})
    evidence={'schema':EVIDENCE_SCHEMA,'target_contract':ACTIVE_META_TARGET_CONTRACT,'horizon_hours':7.,
              'complete_market_windows':True,'deterministic_engine_repeat':True,'configuration_checks_passed':True,
              'source_hashes':{'engine.json':hashlib.sha256(source.read_bytes()).hexdigest()},'records':records}
    evidence_path.write_text(json.dumps(evidence))
    labels=pd.DataFrame([{**{k:r[k] for k in LABEL_FIELDS},'label_source':LABEL_SOURCE,
                          'evidence_path':str(evidence_path),'evidence_sha256':hashlib.sha256(evidence_path.read_bytes()).hexdigest()} for r in records])
    sheets={SHEET_NAME:labels,'General':pd.DataFrame(general)}
    monkeypatch.setattr(pd,'read_excel',lambda path,sheet_name,**kwargs:sheets[sheet_name].copy())
    return workbook,pd.DataFrame(live),sheets,evidence_path,source


def test_positive_and_verified_negative_preserve_observed_outcomes(cohort):
    path,live,*_=cohort
    result=apply_canonical_simulation_labels(live,path)
    assert result.source.eq('canonical_simulation').all()
    assert result.sample_weight_override.eq(1.).all()
    assert result.observed_live_outcome.eq(False).all()
    assert result.capital_fraction.eq(1.).all() and result.capital_used.eq(100.).all()
    assert result.hlabel_detail_observed_net_pnl_pct.eq(-2.).all()
    assert result.t1.eq('2026-10-08T20:01:00+00:00').all()
    assert result.hlabel_detail_observed_t1.eq('2026-10-09T13:00:30Z').all()
    trained,summary=prepare_fast_target_training_frame(result,pnl_col='net_pnl_pct')
    assert summary['positive_count']==1 and summary['negative_count']==1
    assert len(trained)==2 and trained.training_event_id.nunique()==2
    pd.testing.assert_frame_equal(live[list(ACTIVE_SNAPSHOT_META_FEATURES)],result[list(ACTIVE_SNAPSHOT_META_FEATURES)])


@pytest.mark.parametrize('fault',[
    'evidence_hash','source_hash','missing_record','duplicate_record','missing_live',
    'duplicate_live','candidate','target','ttt','event_end','pnl','duration','feature',
    'configuration','hmm','cutoff','incomplete_market','wrong_target','missing_evidence_file',
])
def test_corrupt_or_incomplete_simulations_fail_closed(cohort,fault):
    path,live,sheets,evidence_path,source=cohort
    labels=sheets[SHEET_NAME]
    if fault=='evidence_hash':labels.loc[0,'evidence_sha256']='0'*64
    elif fault=='source_hash':source.write_text('changed')
    elif fault=='missing_record':sheets[SHEET_NAME]=labels.iloc[:1]
    elif fault=='duplicate_record':sheets[SHEET_NAME]=pd.concat([labels,labels.iloc[:1]])
    elif fault=='missing_live':live=live.iloc[:1]
    elif fault=='duplicate_live':live=pd.concat([live,live.iloc[:1]])
    elif fault=='candidate':labels.loc[0,'candidate_id']='wrong'
    elif fault=='target':labels.loc[1,'target_reached']=True
    elif fault=='ttt':labels.loc[1,'time_to_target_hours']=8.
    elif fault=='event_end':labels.loc[0,'t1']='2026-10-08T19:00:00Z'
    elif fault=='pnl':labels.loc[0,'net_pnl_pct']=np.inf
    elif fault=='duration':labels.loc[0,'duration_hours']=8.
    elif fault=='feature':live.loc[0,'ev_score']=999.
    elif fault=='configuration':sheets['General'].loc[0,'leverage']=20.
    elif fault=='hmm':live.loc[0,'hmm_artifact_version']='other'
    elif fault=='cutoff':live.loc[0,'feature_cutoff_utc']='2026-10-08T11:00:00Z'
    elif fault=='missing_evidence_file':evidence_path.unlink()
    else:
        evidence=json.loads(evidence_path.read_text())
        evidence['complete_market_windows' if fault=='incomplete_market' else 'target_contract']=False
        evidence_path.write_text(json.dumps(evidence))
        labels['evidence_sha256']=hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    with pytest.raises((ValueError,OSError)):
        apply_canonical_simulation_labels(live,path)


def test_canonical_executions_replace_lower_weight_forecast_only(cohort,monkeypatch):
    path,live,*_=cohort
    canonical=apply_canonical_simulation_labels(live,path)
    forecast=pd.DataFrame([{'candidate_id':cid,'symbol':'BTCUSDT','source':'backtest',
                           'source_class':'backtest','version_gated':False,'is_authoritative':True,
                           'sample_weight_override':.5} for cid in [live.iloc[0].candidate_id,'other_candidate']])
    builder=UnifiedTrainingBuilder()
    monkeypatch.setattr(builder,'_load_backtest_rows',lambda:forecast.copy())
    monkeypatch.setattr(builder,'_load_live_outcome_rows',lambda **kw:canonical.copy())
    result=builder.build_meta_labeler_pool(include_live_outcomes=True,use_canonical_simulation_labels=True,max_rows_per_symbol=0)
    assert len(result)==3
    assert sorted(result.sample_weight_override)==[.5,1.,1.]
    assert result.loc[result.source.eq('canonical_simulation'),'strategy_id'].nunique()==2


def test_simulation_option_requires_explicit_hybrid_mode():
    with pytest.raises(ValueError,match='include_live_outcomes'):
        UnifiedTrainingBuilder().build_meta_labeler_pool(use_canonical_simulation_labels=True)
