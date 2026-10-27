# Authenticated telemetry evidence and native SAR conversion

## Account and product scope validation (2026-10-22 Lima)

`scripts/probe_telemetry_account_scope.py` compares a captured dashboard UID
with the configured credential's signed Spot `/api/v3/account` UID. It uses
the documented Spot time route, stores only the UID projection of that API
response, validates the UI original's hash and timestamp, and records errors
without exception messages, credentials or signed URLs. A missing or failed
UID response is `unverified`, not a mismatch. Matching identifiers do not
establish native-order ownership or historical account continuity.

The 20:05 Lima check matched the UI/API UIDs. At 20:10 Lima, a visible Filled
WLDUSDT order for native strategy 415238600 returned -2013 from the ordinary
Futures order query. Ordinary Futures open orders/trades were empty. The
unfiltered account income request returned 52 blank-symbol transfer records,
including STRATEGY_UMFUTURES_TRANSFER; per-symbol income returned zero. These
are saved observations, not a claim that every API/product behaves identically.

The supported native ownership source is the Trading Bots USD-M Order History
UI: explicit strategy ID rows and expanded Order No. details. Retain original
UI evidence and independently verified deployment/account context. An order's
Filled summary does not provide every fill ID, fee, funding event or equity
measurement. Transaction rows without strategy/order linkage remain account
evidence. No symbol/time-only ownership mapping is permitted.

Binance documents Trading Bots funds/orders separately from ordinary Futures
accounts: https://www.binance.com/en/support/faq/detail/408ab63b852e47748a9554501669bce5 .
The UID endpoint is documented at
https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/account .

`scripts/capture_account_income_evidence.py` captures income without a symbol
filter into `Live/YYYY-MM-DD/SYMBOL/account_income/`. `SYMBOL` is a storage
context, never ownership or a query filter. Explicit start/end bounds and
request/record budgets are required or recorded; blank-symbol transfers are
retained, identities are deduplicated by account scope, income type and
transaction ID, and conflicts fail closed. This command does not attach income
to scanner actions. All eligibility/influence flags remain false, and retrieval
exhaustion does not establish complete bot accounting.

Full evidence and limitations: `reports/live_telemetry_account_scope_20261022/`.

The bounded September 9 through October 23 01:10 UTC account capture retained
1,083 unique blank-symbol transfers. A repeat 24-hour capture returned 52 rows
and zero new canonical identities. Canonical income identities span ingestion
dates and storage contexts; refetching preserves their first admitted source and
availability. These are account events, not 1,083 additional bot/action labels.

The concurrent SAR lock-initialization correction changes the source-derived
`feature_definition_hash`, even though features, schemas and reward/risk rules
are unchanged. Preserve old event/artifact hashes; existing lineage checks may
reject them as stale. No historical data rewrite, refit or promotion is implied.

The scanner manages running bots throughout their observed operation; its
lifetime is not limited to seven hours. The operator's FASTWIN target is reported
total bot PnL of at least 3% of invested margin by deployment plus seven hours. Capital
protection means preserving invested capital. A numerical minimum liquidation
distance has not been supplied. These definitions do not establish a validated
predictive model or guarantee future capital preservation.

## Acquisition and storage

`scripts/capture_authenticated_bot_evidence.py` performs read-only acquisition.
It has no order submission, model fitting, promotion or activation path.

```powershell
.\.venv\Scripts\python.exe -B scripts/capture_authenticated_bot_evidence.py --all-registered --capture-pnl --fetch --output-dir reports/my_fresh_authenticated_capture
```

The output directory must be new and under `reports/`. New evidence is stored at
`Live/YYYY-MM-DD/SYMBOL/authenticated_evidence/<exact-deployment-key>/` using the
Lima ingestion date. Subdirectories distinguish `responses`, `manifests`,
`producer_records`, `pnl_paths` and `forecast_contracts`. Identical content reuses
its existing file across ingestion dates; conflicting content is rejected.

The collector uses configured Binance credentials and the fixed production
USD-M HTTPS host. It records sanitized endpoint parameters, request/reception
times and response content. It never persists request signatures, API keys or
listen keys. The account namespace is a credential fingerprint, explicitly not
a verified account UID. Different credentials cannot be merged as one account
without independent identity evidence. Responses are locally captured JSON,
not exchange-signed historical attestations.

Each selected deployment is queried from its deployment through one UTC cutoff
fixed when the collection run starts. This API evidence interval is independent
of the seven-hour FASTWIN prediction target. The cutoff does not assert that a
bot remained active until that time, and no terminal date is inferred.

Optional `--start-at` and `--end-at` arguments define a bounded API history
investigation. Both require ISO-8601 timestamps with an explicit UTC offset or
`Z`; timezone-naive, malformed, reversed or future bounds are rejected before a
client is created. The effective start for each bot is the later of its deployment
and the requested start. A bot deployed after the requested end is explicitly
skipped. The end defaults to the immutable run-start cutoff. Inclusive equal
start/end bounds are supported for a single-instant investigation.

```powershell
.\.venv\Scripts\python.exe -B scripts/capture_authenticated_bot_evidence.py --all-registered --fetch --start-at 2026-10-01T00:00:00Z --end-at 2026-10-02T00:00:00Z --output-dir reports/my_fresh_bounded_capture
```

The report records the cutoff, requested bounds and per-deployment effective
intervals. These flags affect authenticated API histories only. `--capture-pnl`
continues to index the full source-verified stored PnL population, preserving
pre-target history for the existing forecast exporter. Neither forecast
contracts nor model/SAR eligibility gates are changed by these collection flags.

Signed `userTrades` and `income` requests are
partitioned by time. Saturated windows are bisected down to one millisecond;
irreducible saturation and exhausted request budgets are recorded as gaps.
The default cap is 32 requests per endpoint per deployment. A transport failure
stops the batch. The client rate limiter remains in force. The new path does not
use the older client's unproven complete-history helpers or income page parameter.

`retrieval_exhausted=true` means the finite query was not truncated by these
mechanisms. It does not prove exchange retention, correct account selection,
native-bot ownership, all interventions or complete bot accounting. Empty
responses cannot be converted into zero trading activity or CONTINUE behavior.

## Ownership and accounting admission

Unlinked account-symbol responses are retained as unproven bot evidence. Neither
symbol/time proximity nor `ORDER_TRADE_UPDATE.o.si` supplies ownership.

An optional `--ownership-mapping` accepts an existing reviewed
`neutralgrid_reviewed_native_ownership_v1` record with exact bot identity,
credential scope, reviewer, review time and original source references. Every
referenced original must explicitly bind `strategyId`, `orderId`, `symbol`,
account scope, deployment identity, event time and availability. This is a
strict reviewed-original interface, not a claim that standard Binance account
responses contain those fields. The collector never manufactures this mapping.

Mapped orders become source-bound ledger ownership facts. Fills require a
committed ownership fact, exact trade/order IDs, explicit realized PnL,
commission and asset, and explicit `marginAsset=USDT` before realized PnL is
named USDT. Missing values are not zero-imputed. Non-USDT commissions retain
their asset and are blocked from dollar accounting without conversion proof.

Admission validates the whole candidate batch before fact publication. Identical
native trades fetched later reuse the first committed fact, source and
availability time. Financial/identity conflicts reject the batch. Original
mapping and response hashes are replayed during ledger verification. Raw income
is not automatically attributed to a bot, and overlapping fill/income accounting
cannot be summed without explicit disambiguation.

The collector does not allocate account equity, funding or capital flows by
symbol, account balance proportions or time proximity. An exact-bot producer
must prove those allocations independently.

## Timestamped PnL and common forecast contracts

Stored canonical observations and their original drawer/envelope sources are
reparsed and checked before a PnL path is published. Points preserve observation
identity, event time, evaluation/availability time, reported total PnL, invested
margin and source hashes. Evaluation before capture or after the collection
time is rejected. These remain UI-reported PnL points, not verified equity.

The path exporter emits a common inert forecast envelope when a pre-deadline,
pre-witnessed-hit origin and consistent denominator exist. Repeated points
cannot increase sample support. Sparse prior history remains
`prior_target_state=unknown`; its conditional label remains `path_unobserved`.
A later observed hit alone does not prove the bot had never hit the target
before that origin. No negative labels or continuous coverage are invented.

The common envelope also defines fixed-horizon dollar PnL, fractional
log-return variance, and observed-action reward contracts. Their horizons,
units, evidence and evaluation meanings remain different. Existing training
readers and live models have not been migrated or enabled. Only the verified
reported-PnL export is wired into this capture command.

## Native-to-SAR integration

```powershell
.\.venv\Scripts\python.exe -B scripts/capture_native_sar.py --decision-source <original-jsonl> --decision-line <one-based-line> --capture --output-dir reports/my_fresh_sar_capture
```

Without `--capture`, the command writes an isolated report. With it, admitted
events use the existing immutable SAR store under `Live/`. Original observations
are reloaded and checked; future observations cannot enter the decision state.
A verdict remains a proposal until independent behavior evidence exists.

Optional original intent/approval/receipt/pre-state/post-state/tick inputs allow
ADJUST execution conversion. Original timestamps, exact identities, bounds,
idempotency keys, approval expiry and position/capital constraints are checked.
Prospective execution receipts now record decision, pre-state availability,
submission and verification timestamps. Old missing timestamps are not inferred
from elapsed duration. No execution is initiated by the conversion command.

Current exchange filters are saved in the adapter's canonical capture format.
They cannot prove an earlier tick size. The existing native executor represents
precision as decimal places; a non-power-of-ten tick such as 0.05 is rejected by
the bridge rather than silently treated as 0.01. This boundary remains explicit.

Optional ledger accounting converts to SAR costs and an outcome with unknown
interval completeness. Fill/income overlap and unsupported currency conversion
are rejected. All transitive original source references are retained so later
mutation invalidates SAR verification. Only exact-time, exact-bot equity evidence
can populate equity/peak fields. Snapshot PnL and account totals cannot do so.

No CONTINUE maintenance action is fabricated. Full accounting/intervention
coverage, verified equity and explicit risk limits remain necessary for action
reward eligibility. The existing SAR 60-minute reward contract and its gates are
unchanged; it is not relabeled as the seven-hour first-passage target. The
operator's invested-capital policy is not substituted for the existing peak
drawdown gate. Missing liquidation limits remain blocked.

## Actual acquisition result and reproducibility

The 2026-10-22 audited run queried 203 registered deployments. All 203 trade
responses and all 203 income responses were empty; no private fact or exact-bot
ownership was established. This observation does not establish why the account
API has no matching rows. Account/UI identity and native-grid account visibility
remain unverified. See `reports/live_telemetry_native_20261022/acquisition/report.json`.

Those archived acquisitions used the collector's former deployment-plus-seven-
hours cap. They are evidence for the intervals recorded in their original
manifests, not for later bot activity. The corrected default window does not
rewrite those manifests or imply that a new lifetime acquisition has occurred.

Two additional deployments from the frozen 205-path cohort were then registered
against original evidence and queried. Their four responses were also empty.
The complete cohort therefore has 205 deployments and 410 primary responses;
one final producer smoke capture adds two repeated-account responses. All 412
saved history responses contain zero native rows. Registration still proves
deployment provenance only, not order ownership.

The subsequent source pass captured 205 PnL paths and 194 inert forecast
envelopes. Nine deployments lacked an eligible pre-deadline pre-hit origin; two
had changing capital denominators. The growing live observation corpus and the
older registered cohort are distinct populations, not identical counts.

One actual SENTUSDT decision was committed as an unexecuted SAR proposal. Its
observed action and reward remain null, and training eligibility remains false.
The detailed audit, source hashes, primary-source research and isolated/full
test results are under `reports/live_telemetry_native_20261022/`.
The final complete suite passed 2,626 tests with two skips.
