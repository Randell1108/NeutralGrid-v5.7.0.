# State/action/reward contract

Status: Contract 1.0 frozen for an isolated offline/shadow challenger. Live influence is disabled.
Date: 2026-10-01

### Invested-capital safety clarification (2026-10-25)

The operator's primary objective for ADJUST and CONTINUE is preservation of
invested principal. The pre-state shield now also requires known equity and
invested capital with equity >= invested capital, independently of the existing
peak-drawdown limit. Missing liquidation limits still block. This narrows
admissible initial states; it does not filter negative subsequent reward labels.
Associations from this cohort do not describe recovery actions starting below
the invested-capital floor. A passing current observation cannot certify future
capital preservation or authorize execution.

Ledger wire fields, action semantics and the 60-65-minute reward endpoint are
unchanged. The source-derived feature-definition hash changes with this shield
implementation, so prior artifacts/feature cohorts cannot silently acquire the
new safety behavior. Existing immutable source records remain readable and are
not rewritten. Native terminal lifetime PnL has its own observational schema in
`docs/terminal_bot_outcomes.md`; it never replaces a missing SAR endpoint.

## Version authority

| Constant | Value | Meaning |
|---|---|---|
| DECISION_CONTRACT_VERSION | 1.1 | Existing scanner decision wire contract |
| ACTION_INTENT_SCHEMA | neutralgrid_action_intent_v1 | Existing intent wire identity |
| ACTION_APPROVAL_SCHEMA | neutralgrid_action_approval_v1 | Existing separate approval wire identity |
| ACTION_EXECUTION_SCHEMA | neutralgrid_action_execution_v1 | Existing execution response identity |
| STATE_ACTION_REWARD_CONTRACT_VERSION | 1.0 | Complete lifecycle and reward contract |
| STATE_ACTION_REWARD_SCHEMA | neutralgrid_state_action_reward_v1 | Ledger and sidecar identity |

Runtime values are imported from `src/neutralgrid/core/constants.py`. Tests independently
pin the expected values and check this table. Correcting LIVE_DECISION.md does not
bump the scanner. The original 1.0 proposal is historical; there is no implemented
scanner 1.2 contract. Verdict fields exclusively use the existing Verdict enum:
CONTINUE, ADJUST, END. Missing evidence preserves the incumbent verdict with boolean
availability and diagnostic reasons; it never becomes a fourth verdict.

Required-field, identity, lifecycle, reward, horizon and action-semantic changes
require an explicit contract revision. Old records are immutable and unsupported
versions are rejected. Model weights, feature definitions, safety configuration
and evidence manifests carry separate content hashes. Configuration cohorts are
never silently mixed. Fixed v1 choices are 60-minute outcomes, at most 5 minutes
late, and a recenter displacement of at most 25% of the current grid width.

## Authority and lifecycle

Immutable per-event JSON is authoritative. Finalized transitions and reports are
projections, never a second journal. Event kinds are decision, approval, execution,
maintenance, outcome and finalized. These are lifecycle facts, not shadow statuses.
One transition has one decision; repeats of an identical event are idempotent;
conflicting event identities quarantine that transition. All events identify the
symbol, strategy ID and deployment time. Symbol-only matching is prohibited.

New live evidence is stored under `Live/<ingestion-date>/<SYMBOL>/sar/`. The date
uses America/Lima ingestion time, not the historical event date. Single-writer
publication uses an OS lock, a flushed staging file and atomic no-overwrite
publication. Interrupted staging files are not committed events. Cross-date
duplicates are detected. Source bytes and event payloads are SHA-256 checked.

The public types define the exact field names. The generated
`docs/state_action_reward_v1.schema.catalog.json` supplies standalone event and
per-kind payload schemas. Select `event_schema` and the applicable `payload_schemas`
member separately; the catalog is not itself an event-validation schema. A test
requires exact equality with generated runtime schemas, including defaults and
required fields. Cross-event timing, source bytes and accounting still require the
Python validators; structural JSON Schema alone cannot establish execution.

States retain capture/availability
times, grid bounds/count, invested capital, equity, unrealized PnL, position
quantity/entry price, orders and raw evidence references. Unknown values are null.
Every decision also retains an explicit feature window, feature lineage and numeric
feature allowlist. Exact means exact recorded source, not continuous exchange state
reconstructed from snapshots. Financial numbers are serialized as decimal strings;
raw references preserve any already-rounded legacy input.

An approval contains its exact action, approved time, expiry and source reference.
Execution contains requested and applied actions, submission/verification times,
separate immediate pre/post states, executor receipt, precision, position-preservation
and additional-investment evidence. A valid approval is not execution evidence.
Outcome states are distinct from immediate post-submit states. A finalized event
contains terminal source evidence and lifetime accounting, separate from one-hour
reward accounting. Late terminal evidence never retroactively supplies a missing
one-hour endpoint.

CONTINUE can label observed maintenance only with unchanged grid geometry and
complete intervention coverage. Missing ADJUST logs alone do not establish it.
ADJUST requires exact approval and verified applied geometry. END is recorded but
is not learned by the first scorer; incumbent END cannot be downgraded. An
unexecuted proposal has a null observed reward, even if a later PnL observation
exists. Simulations and uncontrolled observations cannot enter executed training.

## Accounting

The reward window starts at the decision pre-state capture time. The first valid
endpoint must be between capture + 60 minutes and capture + 65 minutes, inclusive.
Elapsed duration is derived from the retained capture timestamps. Only the earliest
eligible supplied endpoint is selected.
An intervention after the initial verified action censors a single-action example.
Coverage must include the entire accounting/intervention interval and exact bot.

`net = gross_realized - commission + signed_funding + delta_unrealized + other_trading`

`equity_net = endpoint_equity - starting_equity - net_external_cash_flow`

The two must reconcile within 0.000001 USDT. This tolerance is an explicit v1
accounting choice, not a measured claim about exchange precision. A source that
reports realized PnL net of commission/funding must explicitly undo those inclusions
before using the gross formula. Unknown inclusion semantics, missing costs, unknown
conversion to USDT or incomplete coverage yield null reward. Known zero is allowed
only with complete event coverage. Normalize by verified positive starting invested
capital; not notional exposure or guessed capital.

Cost records retain exchange identifiers, source asset/amount, USDT amount, event
time, conversion reference and raw source reference. Deduplicate exchange event IDs;
conflicting values invalidate the transition. Funding credits are positive and
debits negative. Slippage retains fill price/quantity/side and a preceding benchmark
with its source and age. It is informational: fill prices already embed slippage,
so subtracting it again would double count. Estimates are marked estimates.

## Shields and action bounds

Liquidation buffer and maximum peak-equity drawdown are required explicit inputs;
there are no invented defaults. Stale private/public data, unavailable symbols,
unknown lineage, invalid bounds and breached/missing limits block candidate eligibility.
The policy cannot waive these shields. Approval gates execution after proposal;
it is not a reason to relabel a proposed ADJUST as executed CONTINUE.

The recenter keeps pre-rounding width and grid count, current position and capital.
Center shift is clamped to 25% of current width. Exchange rounding is outward by
less than one quantum per bound; recheck positive bounds and the center cap after
rounding. Applied width may exceed requested width only by that recorded rounding.
Different parameters require a different exact approval. No shadow API routes orders.

## Learning, support and isolation

Separate regularized Ridge and logistic models learn observed net outcome and its
positive sign. They are associations, not best-action or causal labels. Reuse the
existing shadow PnL numeric feature allowlist, plus explicit action descriptors.
Future data and outcome fields cannot be features. Fit preprocessing on training
only. Use 60/20/20 chronological cutoffs; related symbol/strategy identities remain
together, crossing groups are excluded, feature/reward intervals are purged and
65-minute gaps are enforced. Diagnostic fit minima do not establish efficacy.

Require both actions, uniform HMM lineage, action geometry within observed support
and state support from training. Unknown human propensities remain unknown; no
importance-weighted or doubly robust policy-return claim is produced. Report MAE,
Brier score, the mean of the worst 5% observed outcomes and baseline comparisons separately from
policy benefit. Model files use JSON numeric parameters, not executable pickle.
Live influence and promotion remain false even when forecast diagnostics improve.
Identical transition projections are deduplicated before fitting; conflicts block
the fit. Validation and test behavior support fractions are reported. An eligible
shadow model requires strictly better holdout MAE than both zero and persistence,
no worse Brier score than the fitted class prior, and complete observed-action
support in both holdouts. These are diagnostic gates, not a proven policy benefit.

A separate, explicitly enabled offline ablation uses the same past 60-minute price
window for ordinary summaries and two-channel time/log-price signatures, truncated
at level two. Retain signed components; never use norms as lossless representations.
Require window coverage and availability; train normalization on training only.
Freeze the window, depth and one additional trial before holdout evaluation.

The unified backtest runner remains authoritative. Its current realistic seed path
clears open inventory without restoring it; existing-live-bot action comparisons
are blocked until independently proven inventory/equity/orders/cost restoration.
Fresh-bot simulations are not evidence for an executed recenter outcome.

Offline output must be fresh and outside Live, artifacts and canonical training
trees. Frozen reports include code/input hashes and rejected reasons. A failed
shadow operation preserves the incumbent result, never fabricates successful evidence,
and cannot invoke transport. No HMM/meta-labeler retrain or live promotion is enabled.

## Video analysis and references

The video's numerical demonstrations were not reproduced as bot experiments.
The adopted ideas are checked against the primary references below. Filtering
versus smoothing motivates availability-cutoff tests. Stationarity tests address
defined hypotheses, not guaranteed future stability. Path signatures motivate a
bounded path-history ablation. Tests demonstrate loss of signed information under
norms and straight-segment subdivision invariance. Deep hedging motivates explicit
costs and inventory accounting; its results do not establish scanner gains.

- Original deep-hedging video: https://www.youtube.com/watch?v=_U1fj5LF-D8
- Time-series lecture: https://www.youtube.com/watch?v=JwqjuUnR8OY
- Signature lecture: https://www.youtube.com/watch?v=kv5cJDP-GGo
- Deep Hedging: https://arxiv.org/abs/1802.03042
- Signature method, invariance and time augmentation: https://arxiv.org/html/1603.03788v2
- Filtered versus future-conditioned states: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.mlemodel.MLEResults.get_prediction.html
- Defined stationarity hypotheses: https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
- Doubly robust evaluation assumptions: https://proceedings.mlr.press/v48/jiang16.html

## Acceptance

Tests must cover version drift, three verdicts, unavailable-model fallback, END
preservation, exact approvals, unexecuted null rewards, CONTINUE evidence, accounting
signs/inclusions/conversions/double counting, endpoints, intervention censoring,
deterministic shields, rounding, duplicate conflicts, crash publication recovery,
future perturbations, split separation, signature identities and transport isolation.
Report actual test/type results separately from predictive gains. Existing PnL pairs
are uncontrolled until linked to verified action/intervention evidence. Preserve
raw data and negative reports; remove only identified experiment scratch if it fails.

## Operation

Read-only offline audit, with fresh output and the prespecified common-cohort ablation:

```powershell
.venv/Scripts/python.exe scripts/evaluate_shadow_actions.py --output-dir outputs/sar_shadow_NEW_RUN --path-ablation
```

Import an explicitly supplied lifecycle envelope. It must either already be sealed,
or contain `bot`, `decision_id`, `kind`, `occurred_at`, and the corresponding typed
`payload`. The CLI hashes/seals it and verifies sources; it does not fabricate fields:

```powershell
.venv/Scripts/python.exe scripts/evaluate_shadow_actions.py --ingest-event Live/INPUT_DATE/SYMBOL/event_to_import.json
```

Production ingestion requires this repository's Live root. Library root injection
is reserved for isolated tests. The first optional controller integration requires
`--observational-only`; flags are `--record-shadow-decisions`,
`--shadow-action-limits PATH`, and `--shadow-action-artifact-dir DIR`. None is enabled
by default. Limits JSON maps `BotIdentity.key` to explicit RiskLimits objects. It
does not invent liquidation/drawdown limits from existing scanner thresholds.
Only `snapshot` artifacts may enter controller reports; signatures remain offline.

Hash checking establishes integrity, not exchange authenticity or completeness.
The producer must supply genuine per-bot event coverage and equity semantics.
Current snapshot observations lack that complete contract. The implementation
therefore supplies ingestion/validation and diagnostics, not a new private-event
collector or retrospective labels. Consult `docs/SAR_SHADOW_IMPLEMENTATION_REVIEW.md`
for each modification, its tests, actual audit results and unresolved proof gates.
