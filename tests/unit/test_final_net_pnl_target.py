from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from neutralgrid.models.meta_labeler import (
    ACTIVE_META_TARGET_CONTRACT, ACTIVE_SNAPSHOT_META_FEATURES,
    FINAL_NET_PNL_TARGET_CONTRACT, FINAL_NET_PNL_TARGET_LABEL_COLUMN,
    MetaLabeler, _is_label_column, _normalize_training_event_times,
)

_path = Path(__file__).resolve().parents[2] / 'retrain_meta_labeler.py'
_spec = importlib.util.spec_from_file_location('final_target_cli', _path)
assert _spec is not None and _spec.loader is not None
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def test_final_label_ignores_duration_and_first_passage_and_excludes_unknowns():
    frame = pd.DataFrame({'net_pnl_pct':[3, np.nextafter(3.0, 0), 4, -1, np.nan, np.inf, -np.inf],
        'source':['live']*7, 'duration_hours':[9, 1, np.nan, 100, 1, 1, 1],
        'time_to_target_hours':[np.nan, 1, 99, 0.1, np.nan, 1, 1]})
    selected, summary = cli.prepare_fast_target_training_frame(frame, pnl_col='net_pnl_pct',
        target_contract=FINAL_NET_PNL_TARGET_CONTRACT)
    assert selected.index.tolist() == [0,1,2,3]
    assert selected[FINAL_NET_PNL_TARGET_LABEL_COLUMN].tolist() == [1,0,1,0]
    assert summary['duration_column'] is None and summary['duration_hours_max'] is None
    assert summary['excluded_rows_missing_or_nonfinite_pnl_count'] == 3
    assert _is_label_column(FINAL_NET_PNL_TARGET_LABEL_COLUMN)


def test_final_label_requires_no_duration_column():
    result, _ = cli.prepare_fast_target_training_frame(pd.DataFrame({'net_pnl_pct':[3,-1]}),
        pnl_col='net_pnl_pct', target_contract=FINAL_NET_PNL_TARGET_CONTRACT)
    assert result[FINAL_NET_PNL_TARGET_LABEL_COLUMN].tolist() == [1,0]


@pytest.mark.parametrize('values', [[3,4],[-1,2],[np.nan,np.inf]])
def test_final_label_rejects_one_class_or_unobservable_pool(values):
    with pytest.raises(ValueError, match='both positive and negative'):
        cli.prepare_fast_target_training_frame(pd.DataFrame({'net_pnl_pct':values}),
            pnl_col='net_pnl_pct',target_contract=FINAL_NET_PNL_TARGET_CONTRACT)


@pytest.mark.parametrize('labels,pnl', [([0,1],[3,-1]),([1,np.nan],[3,-1]),([1,0],[np.inf,-1])])
def test_training_rejects_mislabeled_final_target_before_fit(labels,pnl):
    with pytest.raises(ValueError, match='exactly match finite'):
        MetaLabeler().train(pd.DataFrame({'net_pnl_pct':pnl,FINAL_NET_PNL_TARGET_LABEL_COLUMN:labels}),
            pnl_col='net_pnl_pct',y_col=FINAL_NET_PNL_TARGET_LABEL_COLUMN,
            target_contract=FINAL_NET_PNL_TARGET_CONTRACT)


@pytest.mark.parametrize('different_hmm', [False,True])
def test_changed_target_cannot_replace_old_champion_even_after_hmm_rotation(tmp_path,monkeypatch,different_hmm):
    monkeypatch.setattr(cli,'_get_active_hmm_artifact_version',lambda:'current')
    directory=tmp_path/'meta_labeler'; directory.mkdir()
    payload={'features':list(ACTIVE_SNAPSHOT_META_FEATURES),
        'lineage':{'hmm_artifact_version':'old' if different_hmm else 'current'},
        'eval_metrics':{'target_contract':ACTIVE_META_TARGET_CONTRACT,
            'promotion_status':'pass','promotion_evaluation_contract':cli.PROMOTION_EVALUATION_CONTRACT}}
    path=directory/'metadata.json'; path.write_text(json.dumps(payload))
    before=path.read_bytes()
    candidate=SimpleNamespace(promotion_status='pass',target_contract=FINAL_NET_PNL_TARGET_CONTRACT,
        oof_auc=.9,oof_auc_ci_low=.85,oof_auc_ci_high=.95,oof_ece=.01)
    decision=cli.evaluate_champion_challenger_gate(tmp_path/'meta_labeler.pkl',list(ACTIVE_SNAPSHOT_META_FEATURES),candidate)
    assert decision['status']=='fail'
    assert any('target_contract_mismatch' in x for x in decision['reasons'])
    assert path.read_bytes()==before


def test_final_target_saved_in_metadata_pickle_and_legacy_reload(tmp_path,monkeypatch):
    import neutralgrid.models.artifacts as artifacts
    captured={}
    monkeypatch.setattr(artifacts,'save_artifact',lambda **kwargs:captured.update(metadata=kwargs['metadata'].to_dict()))
    monkeypatch.setattr(artifacts,'get_git_commit',lambda:'isolated-test')
    monkeypatch.setattr(artifacts,'resolve_hmm_artifact_dir',lambda:tmp_path/'missing_hmm')
    labeler=MetaLabeler()
    labeler._target_contract=FINAL_NET_PNL_TARGET_CONTRACT
    labeler._is_trained=True
    labeler._feature_names=list(ACTIVE_SNAPSHOT_META_FEATURES)
    cast(Any,labeler)._model={'fixture':True}
    output=tmp_path/'model.pkl'
    labeler.save(output)
    metadata=captured['metadata']['eval_metrics']
    assert metadata['target_contract']==FINAL_NET_PNL_TARGET_CONTRACT
    assert metadata['target_label_column']==FINAL_NET_PNL_TARGET_LABEL_COLUMN
    assert metadata['target_duration_hours_max'] is None
    assert metadata['target_duration_column'] is None
    assert metadata['promotion_evaluation_contract']==cli.PROMOTION_EVALUATION_CONTRACT
    state=pickle.loads(output.read_bytes())
    assert state['target_contract']==FINAL_NET_PNL_TARGET_CONTRACT
    with pytest.warns(DeprecationWarning):
        loaded=MetaLabeler.load(output)
    assert loaded._target_contract==FINAL_NET_PNL_TARGET_CONTRACT
    assert loaded._promotion_evaluation_contract==cli.PROMOTION_EVALUATION_CONTRACT
    state.pop('target_contract')
    state.pop('promotion_evaluation_contract')
    output.write_bytes(pickle.dumps(state))
    with pytest.warns(DeprecationWarning):
        old=MetaLabeler.load(output)
    assert old._target_contract==ACTIVE_META_TARGET_CONTRACT
    assert old._promotion_evaluation_contract is None


def test_cli_requires_explicit_target_selection(monkeypatch):
    monkeypatch.setattr('sys.argv',['retrain_meta_labeler.py'])
    assert cli.parse_args().target_contract==ACTIVE_META_TARGET_CONTRACT
    monkeypatch.setattr('sys.argv',['retrain_meta_labeler.py','--target-contract',FINAL_NET_PNL_TARGET_CONTRACT])
    assert cli.parse_args().target_contract==FINAL_NET_PNL_TARGET_CONTRACT


def test_mixed_iso_times_preserve_every_event_and_label():
    frame=pd.DataFrame({'start_time_utc':['2026-10-02T01:00:00+00:00','2026-10-01 01:00:00+00:00'],
        't1':['2026-10-02T02:00:00Z','2026-10-01 02:00:00+00:00'],
        'event':['a','b'],'y':[0,1]})
    normalized=_normalize_training_event_times(frame,'start_time_utc','t1',require_complete=True)
    order=np.argsort(normalized.start_time_utc.array.asi8,kind='stable')
    assert sorted(order.tolist())==[0,1]
    assert normalized.iloc[order].event.tolist()==['b','a']
    assert normalized.iloc[order].y.tolist()==[1,0]
    assert frame.start_time_utc.dtype==object


@pytest.mark.parametrize('start,end', [('bad','2026-10-01T02:00:00Z'),
    ('2026-10-01T01:00:00Z',None),('2026-10-01T02:00:00Z','2026-10-01T01:00:00Z')])
def test_unknown_or_reversed_final_event_times_fail_closed(start,end):
    with pytest.raises(ValueError):
        _normalize_training_event_times(pd.DataFrame({'start_time_utc':[start],'t1':[end]}),
            'start_time_utc','t1',require_complete=True)


def test_promotion_missing_times_returns_unavailable_without_fitting():
    labeler=MetaLabeler()
    starts=np.array([pd.Timestamp('2026-10-01',tz='UTC')]*40,dtype=object)
    starts[1]=np.nan
    def must_not_fit():
        raise AssertionError('invalid intervals must be rejected before estimation')
    assert labeler._evaluate_promotion_oof(np.ones((40,3)),np.array([0,1]*20),None,
        starts,starts,must_not_fit)==(None,None,None,None)


@pytest.mark.parametrize('failed_fold', [None, 0])
def test_historical_promotion_fallback_and_partial_coverage(monkeypatch, failed_fold):
    import sklearn.calibration

    n=240
    starts=np.arange(n).astype('datetime64[h]')
    ends=starts+np.timedelta64(6,'h')
    symbols=np.array([f'S{i%40}' for i in range(n)])
    calls=[]
    sizes=[]

    class Calibration:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y):
            fold=len(calls)
            calls.append(fold)
            sizes.append(len(y))
            if fold == failed_fold:
                raise ValueError('deliberate fold failure')
            return self

        def predict_proba(self, X):
            return np.tile([.5,.5], (len(X),1))

    monkeypatch.setattr(sklearn.calibration, 'CalibratedClassifierCV', Calibration)
    labeler=MetaLabeler()
    observed=[]

    def ci(y, p, **kwargs):
        observed.append(len(y))
        return .45,.55

    monkeypatch.setattr(labeler, '_bootstrap_auc_ci', ci)
    result=labeler._evaluate_promotion_oof(np.ones((n,3)),np.array([0,1]*(n//2)),
        symbols,starts,ends,lambda: object())
    assert len(calls) == 5 and min(sizes) > 15
    assert result[0] == .5 and result[1:3] == (.45,.55)
    if failed_fold is None:
        assert observed[0] == n
    else:
        assert 30 <= observed[0] < n
