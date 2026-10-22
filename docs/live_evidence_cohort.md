# Live telemetry evidence cohort

The scanner supports decisions about already-running bots. The requested target
is **reported total bot PnL >= 3% of invested margin by deployment + 7 hours**.
The remaining forecast time is deployment + 7 hours minus the observation time.
This is different from predicting the dollar change during seven hours after
each observation. The existing PnL-delta trainer has not been repurposed.

## Supported commands

Run from the repository root with its Python environment:

```powershell
.\.venv\Scripts\python.exe -B scripts/capture_live_evidence_cohort.py --output-dir reports/my_fresh_evidence_audit
.\.venv\Scripts\python.exe -B scripts/capture_live_evidence_cohort.py --output-dir reports/my_fresh_evidence_capture --capture --fetch-exchange-info
```

Output directories must be new descendants of `reports/`. The first command is
read-only outside its report directory. The second publishes immutable evidence
indexes under `Live/YYYY-MM-DD/SYMBOL/evidence_cohort/<bot_identity>/`. Indexes
reference original observations and sources; they are not new PnL snapshots or
training examples. Existing training readers do not scan this directory.
No recurring job, execution, promotion or model influence is enabled.

`--fetch-exchange-info` makes one public read of Binance USDT-M exchangeInfo,
with a 15-second timeout and 4 MiB response bound. It records request/receipt
times and preserves the response in the report. Each bot index records the
symbol's PRICE_FILTER tickSize and source hash. Display pricePrecision is not a
substitute. A current fetch does not prove a tick size at an earlier decision.
A failed fetch is recorded explicitly and does not invent precision.

Existing canonical SAR envelopes can be supplied with repeated
`--ingest-event <absolute-file-path>`. The entire supplied batch is schema/source
checked before its first write. Individual events use the existing immutable,
locked, deduplicated SAR publisher. A crash can leave a partial batch; replaying
the same originals is idempotent. Conflicts fail closed. This interface does not
generate approvals, receipts, executions or maintenance assertions.

Exit 0 means a completed audit with accepted observations, not model eligibility.
Exit 2 means failure or no accepted observations. Inspect `report.json` and the
per-record rejection reasons. Fresh report directories preserve failed attempts.

## What is verified

`evidence_cohort.py` checks the canonical observation hash, exact deployment,
capture time, canonical path, native drawer SHA-256, native strategy/symbol and
creation time, companion capture-envelope identity/time, and reparsed PnL and
invested margin. The browser page URL is not used as bot identity: an open drawer
can differ from the page URL. Duplicate observations never increase support;
conflicting full records and malformed records quarantine their deployment.

One bot cohort and every accepted observation are emitted in `target_cohort.json`.
CONTINUE, ADJUST, END and missing verdicts are retained as distinct observations.
A verdict is never substituted for an executed action. First *observed* hits,
pre-hit origins, post-hit observations and censored paths are separate classes.
Reported UI amounts are the measurement basis, with their displayed precision;
this does not establish exact exchange equity or continuous extrema.

Sparse non-hits do not become negative labels. A negative endpoint or negative
final lifetime profit cannot exclude an earlier +3% hit. Observed capital changes
block the target denominator. Constant sampled capital also does not prove the
absence of intervening capital flows. The target audit therefore does not fit a
binary classifier from positive-only/censored evidence. It is an eligibility
experiment, not a trained probability model.

`action_cohort.json` uses the existing SAR projection and association evaluator.
Its eligibility is conditional on that contract's evidence checks. It preserves
the distinctions between proposed verdict, observed behavior and reconciled
reward. Native ADJUST approvals/receipts receive existing semantic checks;
generic state/accounting sources are hash-bound producer assertions, not an
independent proof that arbitrary asserted equity or completeness is true. The
collector does not strengthen those assertions by copying or indexing them.
No fitted action artifact is published by this collector.

## Evidence still required for action comparison

`required_event_schemas.json` exports the exact existing schemas for decisions,
approvals, executions, maintenance, outcomes and finalization. A usable producer
must prove all of the following from originals, rather than filling their fields
from assumptions:

1. Exact strategy, symbol and deployment; causal decision features, HMM lineage,
   contract version and risk-profile hash.
2. An executed action linked to its decision, approval and native receipt where
   applicable. CONTINUE requires complete evidence of unchanged behavior during
   the interval; absence of a receipt alone is insufficient.
3. Exact-bot order/trade ownership, complete private-event and intervention
   coverage, capital movements, realized/unrealized PnL and fee/funding inclusion.
4. Verified equity at both endpoints and verified peak equity. Reconcile equity
   change minus external flows with the component reward. A sampled PnL peak or
   isolated margin balance is not automatically full bot peak equity.
5. Explicit per-bot liquidation-distance and drawdown limits, plus exchange tick
   size available when a grid adjustment was decided and executed.
6. Independent held-out bots, chronological purging, both supported behavior
   classes, reward variation, consistent lineage/config/features and benchmark
   improvements. Association is not proof of causal policy improvement.

No exact-bot private exchange collector is started by this command. The existing
private-event import requires a reviewed strategy/order linkage. Account-wide
trades and repeated UI snapshots cannot supply that missing ownership or prove
full coverage. Automated source adapters for verified equity, peak equity and
complete per-bot accounting remain prerequisites; this script exposes those
gaps rather than claiming they have been acquired.

## Boundaries that remain unchanged

The SAR reward window remains 60 minutes plus up to 5 minutes of endpoint
tolerance. It has not silently become the variable remaining-time target. A
separate validated post-target reward/horizon contract is required before using
it for the requested remaining lifetime. END remains a preserved safety verdict;
the current association learner supports CONTINUE versus ADJUST only.

The operator specified invested capital as the zero-loss reference. This is
recorded in the new exact-bot ledger's inert capital policy. Neither a forecast
nor an observed drawdown threshold guarantees zero future loss. The current
RiskLimits schema requires a positive
maximum drawdown and a positive minimum liquidation-distance percentage. A zero
capital-loss limit must not be replaced by a small positive default, and an unspecified
liquidation distance must not be inferred. No risk schema or limit was changed.

See `docs/exact_bot_evidence_ledger.md` for registration and source-bound private
fact ingestion. The cohort script also audits this ledger and emits
`private_ledger_cohort.json`. Registration is not ownership or a training sample.

The meta-labeler's deployment FASTWIN probability is not automatically a
conditional probability after several hours of realized bot history. Transport
to live observations requires the matching features, remaining-time target and
held-out calibration. Its existing disabled caution-only tilt remains disabled.
The volatility contract remains shadow-only; its variance forecasts currently
do not alter live verdicts or supply action rewards.

Validation and exact captured counts are stored in
`reports/live_telemetry_evidence_20261022/`.
