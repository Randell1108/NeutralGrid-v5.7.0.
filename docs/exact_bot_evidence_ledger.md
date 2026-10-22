# Exact-bot private evidence ledger

The ledger has been initialized for the 203 deployments verified in the
2026-10-22 target audit. Registration records original deployment provenance;
it does not create exchange ownership, trades, equity or completed coverage.
No account alias, initial capital amount or liquidation limit was guessed.

## Storage and commands

Registrations and facts are stored under the ingestion date in Lima:

```text
Live/YYYY-MM-DD/SYMBOL/exact_bot_ledger/<exact-bot-key>/registration.json
Live/YYYY-MM-DD/SYMBOL/exact_bot_ledger/<exact-bot-key>/facts/<fact-identity>.json
```

Persistent operating-system lock files live in `Live/.exact_bot_ledger_locks/`.
They are synchronization objects, not financial records. A nonempty lock file
is published atomically before byte locking, avoiding the reproduced Windows
race in concurrent initialization. Temporary publication files are removed.

```powershell
# Initialize from verified evidence indexes. Replays are idempotent.
.\.venv\Scripts\python.exe -B scripts/capture_exact_bot_ledger.py --bootstrap-capture-report reports/live_telemetry_evidence_20261022/capture_final/report.json --output-dir reports/my_fresh_ledger_bootstrap

# Import a source-bound fact, then audit the ledger. The path must contain an
# actual producer record, not this documentation's placeholders or a verdict.
.\.venv\Scripts\python.exe -B scripts/capture_exact_bot_ledger.py --ingest <fact-envelope.json> --output-dir reports/my_fresh_ledger_import

# Audit and export the current fact schema without importing anything.
.\.venv\Scripts\python.exe -B scripts/capture_exact_bot_ledger.py --output-dir reports/my_fresh_ledger_audit
```

Fresh output directories are required under `reports/`. The schema is exported
as `fact_schema.json`. Each fact requires exact bot identity (symbol, strategy,
deployment), an explicit non-secret account scope, a native identifier, event
and availability times, typed fields, an absolute original-source path and
SHA-256, and an optional JSON pointer into that source.

The referenced original producer record must contain matching `bot`,
`account_scope`, `kind`, `native_id`, `occurred_at`, `available_at` and `fields`.
Timestamps are compared as instants. An unrelated hashed file is rejected.
This is a reviewed producer-record interface; it is not a parser for an
unmodified Binance account-wide REST response. An authenticated native-source
adapter still must establish the source's authenticity and exact bot ownership.

## Fact classes

| Kind | Required fields | Additional admission checks |
| --- | --- | --- |
| order_ownership | order_id, ownership_basis | Integer-string ID; native bot/order receipt or reviewed original mapping; no symbol/time inference. |
| order_update | order_id, status, executed_quantity | Previously committed ownership fact for the same account and deployment; supported exchange status. |
| trade_fill | order_id, trade_id, side, price, quantity, realized_pnl_usdt, commission, commission_asset | Exact committed ownership; finite quantities; no missing fee or realized-profit defaults. |
| income | transaction_id, income_type, amount, asset | Funding, realized PnL, commission and other trading income remain separate; transaction identity includes income type. |
| equity | equity_usdt, invested_capital_usdt, unrealized_usdt, peak_equity_usdt | Positive invested capital; peak may explicitly be null; a supplied peak cannot be below current equity. |
| capital_flow | amount, asset, direction | Positive amount and explicit deposit/withdrawal; no currency conversion inferred. |
| intervention | action, receipt | ADJUST, END, ADD_CAPITAL or WITHDRAW_CAPITAL with a hash-bound receipt; CONTINUE is not an intervention. |
| coverage | start_at, end_at, scopes, gaps, producer_claims_complete | Explicit scopes and valid intervals; gaps cannot claim completeness. A producer claim stays unproven. |

Facts are not aggregated into a reward here. Realized income can overlap fill
realized PnL; summing both would double-count. Non-USDT charges require original
conversion evidence before a later accounting adapter can use them. A coverage
declaration is not promoted to `event_complete` by static ingestion.

Fact identity is account + symbol + kind + native ID, with income type for income
records. One order cannot be assigned to different deployments within the same
account/symbol. Exact repeats are idempotent across dates; conflicting content
is rejected. Fill/order-update references revalidate the original ownership
source. Source mutation or malformed ledger records quarantine the deployment
from the audit's accepted facts.

## Capital protection and integration boundary

The operator-approved intent is stored as `reference=invested_capital`,
`maximum_capital_loss_pct=0`, a 420-minute deadline from deployment, and
`minimum_liquidation_buffer_pct=null`. Influence remains false. Registration
does not invent a numerical protected-capital balance from incomplete flows.

For admitted producer equity records, the audit checks
`equity_usdt >= invested_capital_usdt` at that observation. This differs from
requiring equity to remain at its peak. It does not guarantee future preservation
and does not replace the current action shield's peak-drawdown or liquidation
gates. The liquidation-distance limit remains an explicit unresolved input.

The live evidence cohort command reads this ledger into
`private_ledger_cohort.json`. The existing SAR decision/action/reward ledger
remains separate. No automatic SAR conversion is attempted: native approvals,
execution semantics, full interval accounting, intervention coverage and equity
reconciliation must first be proven. Every new private-ledger record remains
`training_eligible=false` and `event_completeness=unproven`.

The remaining acquisition boundary is concrete: no reviewed order-to-strategy
mapping or exact-bot accounting producer existed when this ledger was created.
No private exchange stream was started. The ledger and its tested import/audit
path now exist; their first real ownership/accounting facts still require those
originals. Bootstrap registrations must not be counted as model observations.
