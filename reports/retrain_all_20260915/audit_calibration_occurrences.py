"""Count distinct validation candidates without refitting or changing models."""
from __future__ import annotations
import collections
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from neutralgrid.backtest.cpcv import CPCV, CPCVConfig

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]

def main():
    with (ROOT/'models/meta_labeler.pkl').open('rb') as handle:
        state = pickle.load(handle)
    config = state['config']
    df = pd.read_csv(OUT/'prepared_training.csv', low_memory=False)
    for column in ['start_time_utc', 't1']:
        df[column] = pd.to_datetime(df[column],utc=True,format='mixed')
    df = df.iloc[df.start_time_utc.argsort()].reset_index(drop=True)
    assert df.candidate_id.is_unique and df.fast_winner_target.isin([0,1]).all()
    cv = CPCV(CPCVConfig(n_groups=config.cv_folds,n_test_groups=2,purge_hours=config.purge_hours if config.purge_hours is not None else config.horizon_hours,embargo_hours=config.embargo_hours,horizon_hours=config.horizon_hours))
    counts = collections.Counter()
    folds = []
    for train,test in cv.split(df,timestamp_col='start_time_utc',t1_col='t1',group_col='symbol'):
        train_df = df.loc[df.index.isin(train)]
        test_df = df.loc[df.index.isin(test)]
        if train_df.fast_winner_target.nunique() < 2:
            continue
        counts.update(test_df.candidate_id.tolist())
        folds.append({'train_rows':len(train_df),'test_rows':len(test_df)})
    report = {'unique_training_rows':len(df),'valid_auxiliary_cpcv_folds':len(folds),'validation_prediction_occurrences':sum(counts.values()),'unique_validation_candidates':len(counts),'occurrences_per_candidate_distribution':dict(collections.Counter(counts.values())),'folds':folds,'scope':'Auxiliary CPCV/calibrator fitting observations, not independent repeated samples or the separate promotion OOF predictions'}
    assert report['validation_prediction_occurrences'] == 2523
    (OUT/'calibration_occurrence_audit.json').write_text(json.dumps(report,indent=2))
    utility = json.loads((OUT/'utility_retrain.log').read_text(encoding='utf-8'))
    (OUT/'utility_validation.json').write_text(json.dumps(utility,indent=2))
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
