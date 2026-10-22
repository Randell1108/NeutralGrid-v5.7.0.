# Funding observations and expired-bot ingestion

Implemented 2026-10-21. Related audit:
`reports/history_import_20261020/funding_architecture_review/REPORT.txt`.

## Capture before manual creation

The scanner now saves each valid premium-index response before using it. New
receipts are created under `Live/YYYY-MM-DD/SYMBOL/funding_observations/`.
The folder date is the receipt's America/Lima date; all stored observation and
receipt timestamps are UTC. Each file contains the original exchange payload,
source endpoint, symbol, receipt time and payload SHA-256. Exchange `time`,
`nextFundingTime` and the original decimal `lastFundingRate` remain intact.

For a manual bot, capture immediately before creation with the project runtime:

```powershell
.venv/Scripts/python.exe scripts/capture_funding_observation.py --symbol BTCUSDT
```

Replace the example symbol with the actual symbol. For a bounded sequence use
`--count <count> --interval-seconds <interval>` with explicitly chosen positive
values. No collector was started by this implementation. A later capture cannot
recover an earlier observation. The manual capture command does not place orders.
Keep the receipt path printed by the command with the bot's creation evidence.

Scanner rows retain the receipt path and its observation, receipt and next
settlement times. Final snapshot directories additionally contain
`funding_links_<runstamp>.json`, linking candidate ID, symbol, original pipeline
start, snapshot availability and receipt path. Snapshot timestamps now represent
actual snapshot construction, rather than the earlier pipeline start. This does
not alter old snapshots or prove when missing historical observations existed.

## Extract into staging

Use the existing bot extractor with an original receipt and an explicitly chosen
maximum observation age at the bot's UTC entry cutoff:

```powershell
.venv/Scripts/python.exe new_bot_data_extractor.py --text-file <bot-text-path> --funding-observation <receipt-path> --funding-max-age-seconds <approved-seconds> --output <fresh-staging-xlsx>
```

The angle-bracket arguments are placeholders, not defaults. Use `--dry-run` to
validate and calculate without writing. Existing trade/transaction CSV options
remain available. The extractor appends rows: for a strategy already in the
canonical workbook, use staging and merge only verified empty cells by exact
strategy ID. Do not append a duplicate canonical strategy or replace its five
sheets with a flat staging export.

Replay requires matching source/schema, symbol, payload hash and, when present,
strategy ID. Both exchange observation and receipt must be at or before entry;
the exchange observation must satisfy the explicit age limit. The settlement
must not already be past at that cutoff. Missing, invalid, stale or post-entry
evidence is rejected. There is no default maximum age. Without both funding
options, the extractor retains the existing missing-value behavior.

The existing `funding_carry_expected_next_7h` calculation is unchanged: it uses
the signed observed rate in percentage points when the next settlement is within
seven hours, otherwise zero. It is a market-rate proxy, not a seven-hour sum or
a bot's USDT funding payment. Actual `funding_fee_usdt` remains separate.

Validated replay provenance is saved beside the staging workbook under
`funding_ingest_receipts/`. It records source path/hash, timestamps, cutoff,
maximum age, units and calculated value. These are audit metadata; no new model
or workbook feature columns were added. The receipt records validated ingestion
input, not a guarantee that a later workbook append completed.

## Four audit gaps: fixes and operating recommendations

1. **Lost scanner provenance:** original REST receipts and candidate links are
   persisted. Retain the receipt together with deployment and creation evidence;
   a candidate ID or file modification time alone does not prove entry-time
   availability. Capture near manual creation and select a demonstrably eligible
   receipt. Existing raw-wire stream evidence remains a separate supported
   historical source; this new extractor input accepts the REST receipt schema.
2. **Stale-time clamp:** scanner and provider share validation that rejects past
   settlement times rather than clamping them to zero. Keep missing values
   missing when validation fails; inspect the logged failure and obtain a fresh
   prospective observation.
3. **Mixed funding sources:** enrichment reads only validated premium-index
   observations for the current decimal rate. Preserve settlement history for
   realized accounting; do not substitute it for an unavailable forecast.
4. **Missing extractor input:** the new paired options replay original evidence
   at entry with a chosen age limit. Stage and review the resulting value and
   provenance before a keyed canonical merge. Leave historical gaps empty when
   no eligible original receipt exists.

These changes require no utility calibration, HMM rotation, model feature-list
change or trading threshold change. They do not populate historical gaps by
fetching today's rate.
