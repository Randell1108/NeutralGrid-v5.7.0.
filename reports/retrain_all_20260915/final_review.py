"""Cross-check final state, required ordering and unrelated-file preservation."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]

def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))

def main():
    publication = read('full_57_publication_audit.json')
    metadata = read('meta_metadata_final.json')
    finalization = json.loads((OUT/'finalized_fresh_pool/authoritative_pool_manifest.json').read_text())
    trained = pd.Timestamp(metadata['trained_at_utc'])
    published = pd.Timestamp(publication['published_at_utc'])
    assert published < pd.Timestamp('2026-09-15T22:07:24Z') < trained
    assert publication['after_sha256'] == hashlib.sha256((ROOT/'data/new_expired_bots.xlsx').read_bytes()).hexdigest()
    baseline = read('baseline_hashes.json')
    allowed = {'CHANGELOG.md','artifact_manifest.json','data/new_expired_bots.xlsx','models/meta_labeler.pkl','models/meta_labeler/metadata.json','models/meta_labeler_promotion_decision.json','models/meta_labeler_verification.json'}
    protected = []
    for entry in baseline:
        if entry['path'] in allowed:
            continue
        actual = hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()
        assert actual == entry['sha256'], entry['path']
        protected.append(entry['path'])
    before = pd.read_excel(OUT/'backups/new_expired_bots_before.xlsx',sheet_name='General')
    after = pd.read_excel(ROOT/'data/new_expired_bots.xlsx',sheet_name='General')
    columns = [c for c in before if c not in read('utility_lineage_audit.json')['changed_columns']['General']]
    assert_frame_equal(before[columns],after[columns],check_dtype=False,check_exact=True)
    assert set(DEFAULT_FEATURES) <= set(columns)
    sheets = pd.read_excel(ROOT/'data/new_expired_bots.xlsx',sheet_name=['General','Meta Features'])
    lineage = {}
    for name, frame in sheets.items():
        assert len(frame) == 423 and frame.strategy_id.is_unique
        assert frame.hmm_artifact_version.eq(metadata['lineage']['hmm_artifact_version']).all()
        assert np.isfinite(frame[['range_prob','trend_prob','persistence_prob']].to_numpy(dtype=float)).all()
        lineage[name] = len(frame)
    for script in OUT.glob('*.py'):
        ast.parse(script.read_text(encoding='utf-8'))
    utility = read('utility_validation.json')
    assert utility['artifact']['split']['n_pool'] == 273
    assert all(x['null_count'] == 0 for x in utility['artifact']['pool_feature_coverage'])
    assert utility['artifact']['gates']['G7_finite_nonnegative_not_boundary_pinned'] is False
    assert not utility['promoted'] and not (ROOT/'artifacts/utility/current.json').exists()
    result = {'status':'PASS','checked_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),'workbook_publication_before_meta_training':True,'workbook_published_at_utc':str(published),'meta_trained_at_utc':str(trained),'canonical_workbook_unchanged_after_review':True,'all_row_hmm_lineage':lineage,'original_general_outcomes_and_profile_inputs_preserved':True,'protected_files_unchanged':protected,'report_scripts_parse':True,'utility_all_required_and_optional_features_complete':True,'remaining_runtime_blockers':['No promoted profile model','No promoted utility calibrator','No independent new-model conformal calibration cohort'],'production_source_or_gate_changes':False}
    (OUT/'final_review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
