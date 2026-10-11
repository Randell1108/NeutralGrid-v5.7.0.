# SAR 1.0 implementation review

Date: 2026-10-01. Scope: isolated action evidence and shadow diagnostics.

The implemented result is an auditable challenger framework. No improvement in
live verdict accuracy or realized policy return has been established. The audit
has no verified training examples, so no fitted model is published. Live influence,
execution eligibility and promotion are false. CONTINUE, ADJUST and END remain
the complete verdict vocabulary.

## Authority and contract choices

The normative document is `D:/Neutral Grid v5.7.0/STATE_ACTION_REWARD_CONTRACT.md`.
The runtime registry is `D:/Neutral Grid v5.7.0/src/neutralgrid/core/constants.py`.
The generated field specification is
`D:/Neutral Grid v5.7.0/docs/state_action_reward_v1.schema.catalog.json`.
`test_versions_and_documentation_are_pinned` checks independent expected version
values, the Markdown registry, the existing scanner header, the enum, and exact
equality of the generated schema catalog with runtime definitions.

| Option | Assessment | Implemented treatment |
|---|---|---|
| Immutable lifecycle journal plus finalized projection | Keeps asynchronous approval, execution, accounting and terminal evidence independently auditable. More validation than a flat row is necessary. | Selected. Six event kinds and one derived transition projection. |
| Flat state/action/reward rows | Convenient for fitting, but an overwritten row cannot explain late evidence or distinguish requested from executed actions. | Export only. `transitions.json` never replaces the journal. |
| Separate version for each event type | Permits independent evolution, but compatible combinations become another contract to validate. | Existing intent/approval/execution wire identities remain separate; the whole SAR interpretation is version 1.0. |
| Sequential RL trajectories | Could eventually model repeated adjustments and terminal decisions. It requires support, action timing, complete account attribution and credible multi-step replay. | Deferred. The first scorer handles one verified behavior and one bounded outcome window. |
| Reuse existing PnL pairs as action returns | Existing snapshots do not establish executed action, interventions or complete costs. | Rejected for executed training. The audit counts them as uncontrolled diagnostics. |

Scanner version 1.1 remains authoritative. The erroneous 1.2 documentation header
was corrected without changing the scanner wire schema. SAR 1.0 is a different
identity. Model weights, configuration, feature definition, code and data have
separate hashes. Unsupported versions are rejected; existing files are not migrated
in place. Required-field or economic-semantic changes require an explicit contract
revision. This is an initial contract release, not a retuning of the existing HMM
or meta-labeler.

## Exact fields and interpretation

All timestamps must be aware and are normalized to UTC. Unknown financial state
remains null. Decimal accounting values serialize as strings. Numeric feature
values remain floats; grid count is a strict positive integer. The schema catalog
provides every required/default/nullable distinction. Its individual schemas are
structural specifications; runtime checks also enforce source and cross-event
relationships.

| Type | Fields | Interpretation and check |
|---|---|---|
| BotIdentity | `symbol`, `strategy_id`, `deployed_at` | Exact deployment identity; `key` hashes canonical identity. Never join by symbol alone. |
| SourceRef | `path`, `sha256` | Absolute source path and SHA-256. A changed or missing source invalidates evidence. |
| LedgerEvent | `schema_version`, `contract_version`, `bot`, `decision_id`, `transition_id`, `kind`, `occurred_at`, `event_id`, `payload`, `record_sha256` | Transition identity derives from bot and decision; event identity additionally includes kind/time. The checksum includes the payload. |
| State: timing/provenance | `bot`, `captured_at`, `available_at`, `feature_window_start`, `sources`, `hmm_lineage`, `features`, `price_path` | Capture is different from availability. A decision cannot use a state available later. Features accept only the existing PnL numeric allowlist. |
| State: grid/account | `grid_lower`, `grid_upper`, `num_grids`, `capital_usdt`, `equity_usdt`, `unrealized_usdt`, `peak_equity_usdt` | Bot-scoped equity is required for reward reconciliation. Invested capital plus displayed profit is not substituted for it. |
| State: inventory/market | `position_quantity`, `entry_price`, `orders`, `market_price`, `market_captured_at`, `symbol_available`, `liquidation_buffer_pct` | Preserve recorded quantities and observations. Null orders do not mean there were no orders. Symbol availability must be explicitly true for eligibility. |
| Action | `verdict`, `lower`, `upper` | Bounds are required only for ADJUST. Other verdicts carry no bounds. |
| RiskLimits | `min_liquidation_buffer_pct`, `max_drawdown_pct`, `max_private_age_seconds`, `max_market_age_seconds` | First two have no defaults. Age defaults are explicitly 900 and 15 seconds; they are diagnostic contract choices. |
| Decision | `decided_at`, `pre_state`, `incumbent`, `shadow`, `scanner_contract_version`, `limits`, `origin`, `config_hash`, `feature_hash`, `diagnostics` | Retains both proposals separately. Origins are live, simulation or uncontrolled; origin is evidence classification, not a verdict. |
| Approval | `action`, `approved_at`, `expires_at`, `source` | Native approval schema, exact bot/bounds, preservation flag and times must match. Approval does not prove submission. |
| Execution | `requested`, `applied`, `approval_event_id`, `pre_state`, `post_state`, `submitted_at`, `verified_at`, `receipt`, `quantum`, `preserve_position`, `additional_investment_usdt` | Native receipt must say executed; approval and receipt need matching nonempty 64-hex action identities. Native pre/post bot, bounds and grid count must match recorded states. |
| Maintenance | `state`, `coverage_start`, `coverage_end`, `intervention_coverage_complete`, `source` | Explicit evidence of CONTINUE. Geometry/count/capital must agree with decision and endpoint; absence of an ADJUST receipt is insufficient. |
| CostEvent | `exchange_id`, `kind`, `occurred_at`, `source_asset`, `source_amount`, `amount_usdt`, `conversion_source`, `source` | Kinds: realized, commission, funding, other_trading, external_flow. Deduplicate by kind/exchange ID; conflicts invalidate accounting. Non-USDT costs need a conversion reference. |
| Slippage | `fill_id`, `fill_at`, `side`, `quantity`, `fill_price`, `benchmark_price`, `benchmark_at`, `estimated`, `source`, `benchmark_source` | Separate fill and benchmark references. Benchmark precedes fill by at most 15 seconds. Duplicate conflicting fills invalidate the outcome. It is not subtracted from PnL again. |
| Outcome | `state`, `coverage_start`, `coverage_end`, `event_completeness`, `intervention_coverage_complete`, `interventions`, `realized_includes_commission`, `realized_includes_funding`, `costs`, `slippage` | Requires event_complete and full-window intervention/accounting coverage to earn a label. Unknown inclusion flags yield null reward. Empty slippage means no recorded measurement, not zero execution slippage. |
| Finalized | `terminal_at`, `terminal_status`, `source`, `lifetime` | Terminal statuses: ended, cancelled, expired. Lifetime accounting is retained separately; it cannot replace a missing one-hour outcome. |
| PricePoint | `captured_at`, `available_at`, `price`, `source_path`, `source_sha256` | Positive finite price and explicit capture/availability/source evidence for offline path analysis. |

The exact pre-state is the recorded source state at its capture time. Snapshot
polling cannot reconstruct continuous exchange state between captures. The legacy
executor already emits floats; converting those to Decimal preserves their recorded
value, not precision that the source never supplied. Hashes prove consistency of
the referenced bytes, not authenticity of a producer's economic claims.

## Modification-by-modification review

Source paths below are relative to `D:/Neutral Grid v5.7.0/`. Function and test names
identify reviewable points without relying on line numbers that change during edits.

| Point / file | Specific change and reason | Objection tested / resolution |
|---|---|---|
| 01. `src/neutralgrid/core/constants.py` | Centralized the existing three action wire strings and added the separate SAR version/schema. | New strings must not drift from old wire values. Independent pins and existing scanner/executor tests check identity. |
| 02. `scripts/execute_live_telemetry_modifications.py` | Existing public aliases import the central strings. | Avoid changing submission/approval semantics. No executor control flow, deadlines, positions or capital handling changed. |
| 03. `LIVE_DECISION.md` | Corrected scanner status/table to implemented 1.1. | Documentation must not imply a new scanner contract. Version test checks the corrected header. |
| 04. `state_action_reward.py`: typed models | Reject unsupported fields, nonfinite numbers, financial booleans, naive timestamps, invalid geometry and future availability. | Defaults must not become invented account state. Null fields stay null; forbidden `hlabel` and unknown verdicts fail validation. |
| 05. `make_event`, `contract_schema_catalog` | Seal canonical JSON and expose generated event/payload schemas. | A schema mirror can become stale. Exact generated-catalog equality is tested. |
| 06. `verify_sources` | Hash every raw reference, compare native approvals/receipts, bind exact action keys and immediate pre/post grid state. | A failed, unrelated or mutated receipt cannot be relabeled executed. Parameterized native-receipt tests reject status/schema/symbol/bounds/precision/state/key defects. |
| 07. `append_event`, `atomic_create_json`, `_writer_lock` | Publish immutable events under Lima ingestion date; flush, fsync and atomically link without overwrite. OS locks protect duplicate checks across writers/dates. | Identical retries return the original path. Conflicts fail. Interrupted publication leaves no committed file. Parallel writers publish one event or report busy. This is process-crash evidence, not a power-loss durability certification. |
| 08. `load_events` | Validate size, real date, symbol/transition/filename identities and sources. Quarantine the entire transition after bad evidence. | Keeping the other records from a conflicted transition could create a false example. Tampered-source and misrouted-symbol tests exclude it. |
| 09. `project_transition` | Separate proposed, approved, applied and observed behavior; require one decision and at most one execution. | An unexecuted ADJUST cannot inherit later PnL. Verified CONTINUE remains its own behavior even if the unexecuted shadow proposed ADJUST. |
| 10. Approval/execution timing | Approval lifetime/age at most 300 seconds, decision-to-submit at most 180 seconds, ordered immediate pre/post evidence, event cannot precede verification. | Tests reject expiry, key mismatch and event-before-verification. These limits follow the bounded existing executor contract. |
| 11. Endpoint and intervention selection | First supplied endpoint at 60-65 minutes; censor later interventions and geometry/count/capital changes. | A later attractive or more complete endpoint cannot replace the first supplied endpoint. Both action classes lose eligibility when capital changes. |
| 12. `_reward` | Event-level decimal accounting, explicit PnL inclusions, funding signs, conversions, external-flow removal and equity reconciliation. | Tests cover net/gross inclusions, duplicate/conflicting exchange events, missing conversion, cash transfer, reconciliation failure, and no double subtraction of slippage. |
| 13. `shield_reasons`, `recenter`, `geometry_reasons` | Deterministic limits; width-preserving requested recenter capped at 25% of observed width; outward rounding less than one quantum per bound. | Rounded applied center must remain within cap. Unknown/breached safety data prevents challenger availability; learner cannot override. |
| 14. `action_challenger.py`: `evaluate` | Deduplicate projections, require actual live behavior, uniform lineage/profile/features, fit Ridge/logistic associations and calibrated direction estimates. | Simulation, conflict and unverified rows cannot fit. Synthetic fit fixtures exercise software behavior; they are not included in the live audit. |
| 15. `temporal_split` | 60/20/20 chronological cuts; group by symbol/strategy, exclude crossing groups, purge intervals and enforce 65-minute gaps. | Test proves no related strategy appears in multiple cohorts and no feature/reward overlap remains. Redeployments of the same strategy stay related. |
| 16. Preprocessing and support | Train-only medians/means/scales; per-action leave-one-out nearest-neighbor support radii and observed shift ranges; report holdout support and both action counts. | Missing action in either holdout blocks fitting. Unsupported candidate states preserve the incumbent. A learned propensity is not invented. |
| 17. `ActionArtifact`, `predict_shadow` | JSON numeric model, checksum, schema/feature/code/config/HMM checks, uncertainty comparison, END preservation and exception fallback. | Corruption, wrong lineage, insufficient features, unknown quantum, missing support or overlapping intervals cannot produce an eligible proposal. Execution and influence remain false even for supported synthetic predictions. |
| 18. `action_path_features.py` | Two-channel normalized time/log-price path; five ordinary summaries and six signed level-two signature components. Log differences avoid price-ratio underflow. | Signed paths can share signature norms; norms are not lossless. Tests prove signed distinction, straight-segment subdivision invariance, future perturbation invariance, and gap/endpoint rejection. |
| 19. `action_shadow_io.py` | Read exact-bot persisted observations; hash all feature-history records and raw snapshots, retain complete feature window, reject mixed deployments/changed persisted content. | Latest-only source references could hide changed inputs. Tests bind all historical sources, ignore future observations, and leave missing equity/peak/symbol fields null. |
| 20. `scripts/run_live_telemetry_controller.py` | Optional separate report section and decision recording; new flags require observational-only mode; no challenger value enters incumbent action intents. | Default integration does not load or route. Per-row faults preserve all three verdicts and identical intents. Optional import failure cannot abort the controller iteration. END records without requiring a model. |
| 21. `scripts/evaluate_shadow_actions.py` | Standalone audit/explicit ingestion, fresh output, protected Live/artifacts/data/config/source trees, immutable manifests, rejected diagnostics and common-cohort ablation. | No canonical training/model output path, no executable pickle, no transport or deployment call. Existing output is never overwritten. Ingestion must use repository Live. |

The new source modules are under `src/neutralgrid/live/decision/`. The modified
controller still builds action intents from the unchanged incumbent rows. The
new scorer has no order transport argument or approval bypass. Existing HMM,
meta-labeler, utility gates and the three canonical feature-pipeline files are
outside the implementation's write paths.

Isolation here consists of separate offline subprocesses, fixture-specific Live
roots, fresh research directories, protected canonical output paths and absent
action transport. Tests use the existing virtual environment; no dependencies
or OS container were installed. The production controller was not restarted or
opted into these new flags by this implementation.

## Accounting proof and limits

The deterministic calculation is:

```
gross = realized events
        + commissions if realized already includes commissions
        - signed funding if realized already includes funding
net = gross - commissions + signed funding + other trading + change in unrealized
equity_net = final equity - starting equity - signed external cash flow
normalized_return = reconciled net / positive initial invested capital
```

The allowed reconciliation difference is 0.000001 USDT. Paid commissions are
positive costs; rebates may be negative. Funding credits are positive and debits
negative. Other trading amounts are signed. This arithmetic is validated on explicit
synthetic fixtures. The tolerance and 60+5-minute horizon are contract choices, not
empirically optimal estimates. Exact cost scope, fee/funding inclusion semantics
and bot-equity ownership still have to come from a genuine producer. Shared-account
balances cannot be apportioned by guesswork.

The ledger can ingest and validate complete producer-supplied facts. It does not
install a new authenticated event collector, retrospectively recover missing
interventions, or certify a producer's completeness assertion. A configured model
or source failure may prevent an attempted decision record; its fallback is
reported and no successful evidence is fabricated. Recording approval, execution,
maintenance, outcome and terminal facts requires explicit envelopes through the
same validated ingestion path.

## Bounded learning and evaluation

The first fit needs at least 30/10/10 fit/validation/test samples and 4/2/2 bot
identities, with at least five fit observations per action and both actions in
each holdout. The 10,000-example maximum bounds training. These minima are explicit
diagnostic choices, not statistical power calculations. Outcomes must include
both positive and nonpositive classes in fit and calibration sets.

Ridge uses alpha=1. Logistic models use C=1, maximum 2,000 iterations and seed 0.
No search chooses these on the holdout. Features that are wholly missing or constant
in training are removed. Imputation/scaling is fitted only on training. Validation
fits the direction calibration and 80% absolute-error interval radius. The final
holdout compares MAE against zero and the prior PnL velocity times 60, and Brier
score against the fitted direction prior. Strict MAE improvement and no worse
Brier score, plus complete observed-action support in both holdouts, are required
for an eligible shadow artifact. Interval coverage is measured, not guaranteed.

Nearest-neighbor support uses standardized training rows, a per-action 95th
percentile leave-one-out radius and observed center-shift range. Both candidate
actions must pass at inference. This detects extrapolation; it is not causal
overlap certification. Human selection propensities are not logged, and action
choice can be confounded with market state. Importance sampling, doubly robust
policy evaluation, claimed counterfactual returns and promotion are therefore
unavailable. The worst-5%-outcome mean is an observed sample statistic; it is not
a demonstrated change in policy tail risk.

The three offline variants are snapshot, summaries, and summaries plus signature.
When ablation is requested, all use the same path-eligible cohort. The path needs
exact past-hour endpoints, at least 13 points and no gap over five minutes.
Future capture/availability is excluded. The path transformation/window/depth
are fixed before holdout use. Both richer variants remain offline; the controller
accepts only snapshot artifacts. No useful ablation result exists for the audited
zero-example cohort.

## Replay objection and resolution

`backtest/backtest_realistic.py::seed_from_state` explicitly clears positions and
restores the resting ladder rather than pre-existing inventory. The existing
`tests/unit/test_btk_seed_state.py` exercises that behavior. It is appropriate to
its existing ladder-seed purpose, but insufficient for an already-running bot's
CONTINUE-versus-ADJUST comparison. Reset equity, missing cost history or omitted
inventory would change the comparison's starting economics.

The canonical entry point remains `backtest.btk_unified_runner.run_backtest`.
No alternate runner or backtest repair was added. `replay_eligible` stays false
until a separate implementation proves identical initial inventory, equity,
orders, accrued fees/funding and action timing, then passes conservation and
scenario checks through that canonical runner. Fresh-bot simulations cannot
replace this proof. This is an explicit unresolved gate, not a replay success.

## Video-derived ideas and rejected extrapolations

The supplied links are preserved in the normative contract. Their demonstrations
are not reproduced as trading experiments. During implementation, web access
did not expose full captions for the first two links; the signature video page
provided its title. The implementation's methodological claims rely on the primary
references below and local tests, not inferred unseen lecture details.

| Candidate idea | Useful adaptation | Objection / bound |
|---|---|---|
| Deep hedging under costs and constraints | Retain inventory, actual action and net costs in the contract. | Derivative-hedging experiments do not establish an edge for this grid scanner. A supervised association scorer precedes any learned sequential policy. |
| Time-series state estimation | Distinguish capture from availability; reject future-conditioned evidence. | A smoothed state can use future observations. No smoother or new HMM feature was added. |
| Stationarity diagnostics | Treat tests as stated statistical hypotheses. | A stationarity p-value cannot authorize live influence or guarantee future stability. No ADF/KPSS runtime gate was added. |
| Path signatures | Evaluate signed, time-augmented history against ordinary summaries on identical cohorts. | Truncation and norms lose information; path transformations do not supply causal labels or justify larger production models. |

Primary references, checked during implementation:

- [Deep Hedging](https://arxiv.org/abs/1802.03042): costs, market impact and constraints in a derivative-hedging framework.
- [Signature-method primer](https://arxiv.org/html/1603.03788v2): Chen's identity, path transformations and signature properties.
- [Statsmodels prediction information sets](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.mlemodel.MLEResults.get_prediction.html): filtered versus smoothed conditioning.
- [Statsmodels stationarity hypotheses](https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html): ADF/KPSS interpretation.
- [Doubly robust off-policy evaluation](https://proceedings.mlr.press/v48/jiang16.html): assumptions required for policy-value estimation.

## Frozen audit evidence

The final validated audit is saved under
`D:/Neutral Grid v5.7.0/outputs/sar_shadow_20261001_validated/`.
`input_manifest.json` contains the exact included input and code hashes;
`transitions.json` is the derived transition table; `report.json` contains blockers
and evidence counts. Earlier initial/final audits are retained separately, with
their own code snapshots. The final validated manifest is the review authority for
the released code. Counts refer to a frozen included set, not to every observation
that a running collector may append later.

| Metric | Frozen result |
|---|---:|
| Valid existing PnL observation records | 551 |
| Uncontrolled one-hour PnL pairs | 69 |
| Records without declared private-event completeness | 551 |
| SAR lifecycle events / transitions | 0 / 0 |
| Verified action/reward training examples | 0 |
| Fitted challenger artifacts | 0 |
| Observed policy gain | null |
| Influence / promotion / existing-bot replay eligibility | false / false / false |

No observed ADJUST return was manufactured. No uncontrolled CONTINUE label was
inferred. Zero examples are an evidence-readiness result, not evidence that the
challenger improves or harms trading performance. Future readiness needs actual
scoped equity, coverage-complete costs/interventions, exact native action receipts,
terminal records, explicit risk profiles, logged behavior support and the replay
proof above. These are data/proof requirements, not parameters tuned to make this
audit pass.

## Validation evidence

The complete repository run is recorded in
`D:/Neutral Grid v5.7.0/outputs/sar_pytest_20261001.txt`: 2,299 passed, 2 skipped,
3 failed in 774.15 seconds. The three failures were existing short-duration Windows
loopback tests for depth, market and private collectors. Their manifests recorded
incomplete subscriptions or labelled gaps. Their source files were not changed.
An isolated rerun of exactly those three passed in 8.69 seconds; both results are
retained in `outputs/sar_loopback_recheck_20261001.txt`. The full run was not uniformly
green and is not presented as such. Final focused checks cover the final subsequent
reference/endpoint/schema changes.

The final source/controller/CLI type check is saved in
`outputs/sar_release_pyright_20261001.txt`: zero errors and zero warnings. An explicit executor check reports two
return-type errors in its existing protocol declarations. Checking an isolated
copy of the committed executor reproduces the same two errors at its original
lines 163 and 171 (`outputs/sar_executor_baseline_pyright_20261001.txt`). The central
schema-alias change did not introduce them. No ignore or unrelated protocol edit
was added to conceal the baseline result.

The final focused pytest output is saved in
`outputs/sar_verified_focused_pytest_20261001.txt`; it includes the new ledger,
challenger and transport-isolation tests plus existing controller, executor,
scanner-1.1 and replay-seed tests: 147 passed in 53.01 seconds. Exact final counts,
code/input hash rechecks and type results are recorded in
`outputs/sar_shadow_20261001_validated/verification.json` alongside these logs. Synthetic training tests verify
serialization, leakage/support handling and fallback behavior only. They do not
establish financial performance or authorize deployment.

Negative reports and raw sources are retained. Test scratch is separate from
production Live data. Atomic-publication staging files are cleaned as part of
normal commit handling. The pre-existing `.tsar_555d2c` test directory is outside
this change and was not removed.
