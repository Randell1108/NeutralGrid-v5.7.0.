# Terminal reported bot outcomes

`terminal_outcomes.py` records an actual reported lifetime outcome from an original terminal Binance grid drawer JSON. It does not assign that outcome to every earlier verdict, establish executed behavior, reconcile exchange accounting, or admit a training label automatically.

The original must remain at:

```
Live/YYYY-MM-DD/SYMBOL/verdict_audit_STRATEGY_ID/strategy_detail.json
```

The JSON must contain original `text`, an explicit `display_timezone: "UTC-5"`, and an aware ISO timestamp in `captured_at_utc` or `captured_at`. If both timestamp fields exist, they must identify the same instant. Optional declared `symbol` and `strategy_id` must match the native text exactly. A page URL is not bot identity. Metadata explicitly identifying a manual transcription is rejected. A hash binds these bytes to the stored record; it does not independently authenticate browser origin or exchange completeness.

The drawer supplies the symbol, terminal status, Strategy Number, Time Created, Time Ended, Total Profit (USDT), and Invested Margin (USDT). Mandatory fields must occur exactly once. Canceled/Cancelled normalize to `canceled`; Expired becomes `expired`. Working drawers, ambiguous fields, malformed amounts, nonpositive capital, nonfinite JSON numbers, duplicate JSON keys, naive timestamps, conflicting identity, and invalid deployment/end/capture order fail closed. Native event times use the explicit UTC-5 display timezone and are stored in UTC.

## Storage and replay

`parse_terminal_capture(NativeRecord, live_root=...)` strictly reads the original and returns a frozen `TerminalObservation`. `capture_terminal_outcome(...)` publishes one immutable JSON per exact symbol, strategy ID, and deployment time:

```
Live/INGESTION-DATE/SYMBOL/terminal_outcomes/BOT-IDENTITY-HASH.json
```

The ingestion date uses Lima time. A per-deployment lock and atomic no-overwrite publication protect concurrent writers. Lookup spans all ingestion dates. An identical later recapture reuses the existing record and its first capture provenance. Differences in terminal economics, displayed return, normalized terminal status or end time reject the recapture. Multiple stored paths for the same deployment also reject it. Originals are never rewritten or deleted.

`load_terminal_outcome(path, live_root=...)` validates the record route and requires its ingestion folder date to be no earlier than the original capture date in Lima. It rereads the hash-bound original, reparses native fields, and requires exact agreement with the stored observation. Backdating an otherwise unchanged stored record therefore prevents replay admission. `score_terminal_outcome(...)` repeats source validation. Altering either original bytes or stored economics prevents scoring.

`audit_terminal_outcomes(live_root)` enumerates terminal JSON files including malformed date directories, independently revalidates each record and original, and returns `scores`, `rejected`, source hashes, outcome counts and the module code hash. A duplicated exact deployment quarantines every copy, including byte-identical copies. An unparseable record with a valid bot-hash filename quarantines other records with that hash. A parsable bot identity in a misnamed record also quarantines the corresponding deployment. Unidentifiable malformed records remain listed as rejected. No duplicate outcome is scored.

## Meaning of the score

The score includes reported terminal PnL in USDT, reported invested margin, normalized return (`PnL / margin`), normalized return percentage, displayed percentage, and positive/negative/flat outcome class. Displayed rounded percentages remain separate from calculated values. Reported capital deficit is `max(0, -PnL)`, also divided by reported margin for the deficit percentage. This is a deficit on the reported PnL basis, not verified equity or a full capital-flow reconciliation. It does not deduct fees a second time from a reported total whose accounting inclusion semantics have not been independently established.

A terminal return at or above 3% within seven hours of deployment sets the descriptive `terminal_return_meets_3pct_before_deployment_deadline` flag. Its denominator is the margin reported at termination. Without independently verified initial capital and capital-flow history, it does not establish FASTWIN eligibility: `capital_basis_at_deployment_verified` is false and `first_passage_status` always remains `path_unobserved`. A negative terminal value does not establish that the bot never reached +3% earlier. Closure before seven hours does not manufacture a negative first-passage label.

`observed_action` and `causal_reward` stay null. `verified_equity`, `training_eligible`, `promotion_eligible`, and `influence_enabled` remain false. Operator action testimony and receipts, where available, belong to separately linked evidence; this module neither deletes that evidence nor invents it.

## Command line

The command defaults to report-only. It scores originals into a fresh directory strictly beneath repository `reports`; it does not publish terminal observations unless `--capture` is supplied. It has no configurable production Live root and refuses redirected canonical roots.

```powershell
python scripts/capture_terminal_bot_outcomes.py --source Live/2026-10-25/FARTCOINUSDT/verdict_audit_415222014/strategy_detail.json --output-dir reports/terminal_review_unique
```

To publish after reviewing the report, repeat with a different fresh output directory and `--capture`. Multiple `--source` arguments are supported. Exit codes are 0 for all requested sources completed, 1 for source-level errors, and 2 for run/report-storage failures.

Each accepted deployment group produces a progress checkpoint; rejected inputs remain in the report. Successful writes remain visible if a later source fails. Errors after publication expose `existing_or_published_path` and `publication_requires_revalidation`; the file must be revalidated, not treated as successful because it exists. A failed checkpoint logs the potentially published path. A final report-write failure leaves the preceding progress checkpoints. Retrying with a fresh report directory is idempotent for unchanged valid terminal observations. Lock files are durable coordination artifacts, not temporary evidence.

Batch validation precedes every terminal write. Conflicting same-deployment economics quarantine all related scores. Identical recaptures and repeated source paths produce one score, with the earliest available original selected; duplicate sources are listed without adding statistical weight. Capture availability cannot follow the ingestion cutoff. Multiple drawer headers are rejected even if only one contains economic fields. Financial tokens are limited to 128 characters as an input-resource bound; scores use a deterministic 50-digit decimal context and controlled arithmetic errors. These are technical bounds, not risk limits.

The isolated unit tests cover round trips, both timestamp metadata formats, malformed/ambiguous originals, source mutation, economic conflicts, cross-date duplicate recapture, duplicate stored paths, report-only operation, partial source failure, and failure after publication. Production captures are not needed for these tests.
