# Backfill exports and canonical workbooks

`backfill_training_features.py` reads one tabular input and writes one flat
feature export. It does not preserve auxiliary sheets, cell formatting, formulas
or native table objects. Its output must not replace a canonical multi-sheet
workbook.

For a multi-sheet source, explicitly acknowledge the flat export and choose a
destination that does not exist:

```powershell
python scripts/backfill_training_features.py `
  --input data/new_expired_bots.xlsx `
  --output <fresh_feature_export.xlsx> `
  --default-artifact-version <active_hmm> `
  --feature-cutoff-source start_time_utc `
  --replay-scope full_feature_refresh `
  --require-fresh-output `
  --allow-flat-workbook-export
```

The first sheet remains the source table, as in prior versions. Confirm it is
the intended `General` sheet before the run. CSV and single-sheet inputs do not
need the new acknowledgement. A multi-sheet input always requires a fresh
destination, whether the chosen export is CSV or XLSX. Existing multi-sheet
destinations are refused for every input type.

The first refusal occurs before initializing the network client. A final
destination check also runs before publication. A fresh export is published with
an atomic same-filesystem hard link, which refuses a destination that appeared
after the checks. If the filesystem does not support this operation or access is
denied, the operation fails without replacing the existing destination. It does
not fall back to an overwrite.

To incorporate an export into canonical storage, use the governed ingestion
workflow to merge only the explicitly refreshed feature columns into a preserved
copy. Validate that row identities are complete, unique and match the intended
canonical records before merging. Reject ambiguous or duplicated keys instead
of selecting a row arbitrarily. Preserve outcome labels, row identity columns,
auxiliary sheets and workbook objects. Compare the resulting workbook with the
source and independently verify uniform active-HMM lineage before publication.
This document is a required integration contract; the backfill utility does not
implement that canonical merge or claim it has been verified for a particular
export.

The acknowledgement does not change HMM authority: an explicit
`--default-artifact-version` still invalidates stale HMM-derived features and
lineage. Failed inference leaves missing values, and the export must still pass
the downstream lineage validation before training or calibration.
