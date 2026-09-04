# LIVE_DECISION

Status: Contract v1.2 implemented; PnL forecaster remains shadow-only and unpromoted
Date: 2026-08-10
Scope: Recurring tactical scanner that evaluates user-declared live grid bots every X minutes against current Binance market state and emits per-bot CONTINUE / ADJUST / END recommendations with reasons. Advisory only.

Sibling document to `LIVE_BOT_DECISION.md` v2 (single-shot operator-side evaluator). The two share a vocabulary and several contracts but solve different problems; see Reconciliation below.

## Context

The current pipeline (`run_full_pipeline.py`) is one-shot: scan -> enrich -> deploy, then exit. Once a grid bot is deployed, the codebase has:

- No recurring monitor loop.
- No central "currently-live" registry (active = deploy_linkage_log minus new_expired_bots, stale until expired-bot ingestion).
- No notification surface (no Telegram / Discord / email / webhook integrations exist; `models/alerts.py` ships only `LoggingAlertHandler` and `CallbackAlertHandler`).
- No link from a deployed bot back to its current order-book / regime state.

This document specifies a tactical live decision tool that fills those gaps using only existing scanner, HMM, meta-labeler, microstructure, and Binance-client components. It does not introduce new market models. It adds plumbing.

## User-locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Output channels | Console + `logs/live_decisions_YYYYMMDD.jsonl` AND Discord webhook | User runs scanner locally; wants away-from-desk push too. |
| Autonomy | Advisory only. Read-only Binance endpoints. | Zero risk of unintended order placement. Signed/trading endpoints are explicitly forbidden. |
| Live registry | User-maintained active YAML files under `active bots/`, preferably one file per active symbol such as `RENDERUSDT.yaml`. Legacy `DD-MM-YY.yaml` registries are still supported when no symbol files exist. | Explicit active/inactive boundary, no state drift, edits during a session take effect next tick. |
| Default interval | 5 minutes (`--interval 5m`) | Aligns with 5m kline boundary. Plenty of weight headroom for 30-50 bots. Good signal-to-noise. |
| External signals | Binance-only for v1 | Defers per the pending-external-APIs constraint (training pool < 500 rows). |

Inactive symbols are not scanned. Move canceled or expired symbol YAMLs out of `active bots/` (for example into `inactive bots/YYYY-MM-DD/`) while preserving raw telemetry under `Live/YYYY-MM-DD/SYMBOL/`.

## Reconciliation with prior art (`LIVE_BOT_DECISION.md` v2)

`LIVE_BOT_DECISION.md` v2 is an unapproved draft for an operator-side **single-shot** evaluator that reads bot-manager telemetry JSONs and emits `continue` / `end_bot` / `no_decision`. The new tactical scanner is a **sibling, not a replacement**:

| | LIVE_BOT_DECISION.md v2 (existing draft) | Tactical Scanner (this plan) |
|---|---|---|
| Trigger | Manual single-shot per bot | Recurring loop over fleet |
| Input | Bot-manager telemetry JSON | User-curated active YAML files under `active bots/` |
| Verdicts | `continue` / `end_bot` / `no_decision` | `CONTINUE` / `ADJUST` / `END` |
| Output | Stdout + JSON | Console + JSONL + Discord digest |

Shared contracts adopted from v2:

- `DECISION_CONTRACT_VERSION` constant (must be added to `core/constants.py`).
- Fail-closed semantics: missing inputs surface `data_missing:*` reasons, never silently coerce to `CONTINUE`.
- `meta_prob` is secondary, not authoritative (current MetaLabeler trains on scan/enrichment-time features, not live-state telemetry).
- Use the conservative static `min_range_prob = 0.45` rather than the entropy-adaptive scan-time threshold.
- `MetaLabeler` overlay disabled with a `meta_overlay_inactive` reason when more than half of required features are NaN.

Where they diverge: the tactical scanner's `ADJUST` absorbs v2's `no_decision` for *recoverable* missing inputs (e.g., transient fetch failure, bot near boundary) and surfaces persistent ones (e.g., HMM artifact missing, `data_missing:*`) as `ADJUST` with the reason code intact. Persistent operational failures over multiple ticks escalate.

If/when LIVE_BOT_DECISION v2 is approved and lands, shared logic should be factored into `live/decision/` rather than duplicated.

## Architecture

### File layout

New, unless flagged.

| Path | Purpose |
|---|---|
| `live_decision_scanner.py` | Root CLI, mirrors `run_full_pipeline.py` style. Owns argparse, asyncio loop, lifecycle. |
| `src/neutralgrid/live/decision/__init__.py` | Subpackage marker. |
| `src/neutralgrid/live/decision/loader.py` | Find active per-symbol YAML files, fall back to latest legacy `active bots/DD-MM-YY.yaml`, parse + validate `LiveBotSpec`s. |
| `src/neutralgrid/live/decision/monitor.py` | Per-bot evaluator orchestration. |
| `src/neutralgrid/live/decision/recommender.py` | Pure verdict logic. |
| `src/neutralgrid/live/decision/state_store.py` | Per-bot tick history at `data/live_decisions/state/<key>.json`, atomic write. |
| `src/neutralgrid/live/decision/renderer.py` | Console table + JSONL writer. |
| `src/neutralgrid/live/decision/discord_sink.py` | Webhook digest builder + token-bucket rate limiter. |
| `src/neutralgrid/live/decision/private_telemetry.py` | Strict Binance drawer parser for PnL, position, risk, ladder, grid, TP/SL, and UI order-history snapshots. |
| `src/neutralgrid/live/decision/l2_risk.py` | Fresh sequence-linked L2 derivatives, persistent spread/depth summaries, and position-normalized exit simulation. |
| `src/neutralgrid/live/decision/pnl_history.py` | Immutable, exact-bot PnL observations under `Live/YYYY-MM-DD/SYMBOL/pnl_history/`, with collision/tamper detection and source hashes. |
| `src/neutralgrid/live/decision/pnl_forecast.py` | Explicit-horizon, bot-disjoint temporal OOS training/evaluation for a shadow PnL-change candidate. |
| `scripts/collect_diff_depth.py` | Event-complete public diff-depth capture plus bounded scanner derivatives. |
| `scripts/ingest_private_execution_events.py` | Static Binance `userTrades` importer requiring a reviewed exact order-ID linkage artifact; cannot claim `event_complete`. |
| `scripts/train_live_pnl_forecaster.py` | Offline shadow trainer with required horizon/tolerance and baseline/promotion gates. |
| `scripts/run_live_telemetry_controller.py` | Joins exact private-cycle and diff-depth run identities into scanner input. |
| **edit** `src/neutralgrid/models/alerts.py` | Add `DiscordWebhookHandler(AlertHandler)`. |
| **edit** `src/neutralgrid/core/constants.py` | Add `DECISION_CONTRACT_VERSION = "1.0"`. |

### Reused, read-only

| Purpose | File:Line |
|---|---|
| Async Binance REST + 1200 weight/min budget | `src/neutralgrid/api/binance_client.py:44` |
| Order book depth (weight 5) | `binance_client.py:570` `get_order_book` |
| Concurrent multi-fetch (~34 weight/symbol) | `binance_client.py:1039` `get_all_market_data` |
| Per-symbol features | `src/neutralgrid/scanner/feature_extractor.py:92` |
| HMM regime probabilities | `src/neutralgrid/validation/hmm_regime.py:91` + `models/hmm/inference.py:452` |
| Meta-labeler | `src/neutralgrid/models/meta_labeler.py:1568` `predict_proba` |
| Utility calibrator (offline-caller pattern) | `src/neutralgrid/calibration/utility_calibrator.py` |
| Stage-B gate set | `src/neutralgrid/scanner/two_stage_selector.py` |
| Microstructure adaptive gate | `src/neutralgrid/scanner/adaptive_microstructure_gate_v20260311.py:45` |
| Deploy <-> candidate linkage | `src/neutralgrid/live/candidate_deploy_linker.py:51` |
| Output formatter precedent | `run_full_pipeline.py:485-606` |
| Alert dispatcher | `src/neutralgrid/models/alerts.py` |

## YAML schema

Preferred path: `active bots/SYMBOL.yaml`, one active bot per file.

Legacy path: `active bots/DD-MM-YY.yaml`, one or more active bots in the file.

The scanner also accepts a canonical live telemetry file directly via `--bots`,
for example `Live/YYYY-MM-DD/DOGEUSDT/live_bot_data_scanner.yaml`. That format
must contain `data_class: live_bot_telemetry`, `bot.deploy_ts`, `grid`,
`pnl`, `position`, `risk`, `open_order_ladder`, and `advanced` sections. It is
converted in-memory to the same `LiveBotSpec` contract as the active registry.

> **`deploy_ts` timezone rule (ERR-046):** a `deploy_ts` without a timezone
> suffix is silently interpreted as UTC (`loader.py`). Always include `Z` or an
> explicit offset. Operator-pasted Binance UI times are local UTC-5 — add +5h
> to get the UTC value before writing it here (see the AVAXUSDT 2026-06-21
> case, where a local time mislabeled `+00:00` caused the causal
> geometry-backfill to wrongly reject the true candidate).

```yaml
bots:
  - symbol: BTCUSDT
    strategy_id: "410472444"     # optional; falls back to symbol+deploy_ts as state key
    deploy_ts: 2026-05-06T14:30:00Z
    grid_lower: 60000.0
    grid_upper: 70000.0
    num_grids: 50
    leverage: 5
    capital_usdt: 200.0
    candidate_id: "..."          # optional; if present, linker provides deploy-time deltas
    execution_telemetry:         # optional; user-supplied Binance UI/account telemetry
      source: user_provided_binance_ui
      captured_at: null          # ISO-8601 UTC if known
      pnl:
        total_profit_usdt: 23.78
        total_profit_pct: 5.94
        matched_profit_usdt: 24.05
        matched_profit_pct: 6.01
        realized_profit_usdt: 24.03
        unmatched_pnl_usdt: -0.28
        unmatched_pnl_pct: -0.06
        funding_fee_usdt: 0.00
        funding_fee_pct: 0.00
      open_order_ladder:
        qty_per_order_base: 63284
        last_price: 0.00173
        buy:
          - level: 1
            price: 0.001693
            pct_to_fill: -2.13
        sell:
          - level: 1
            price: 0.001748
            pct_to_fill: 1.04
      position_inventory:
        symbol: COSUSDT
        contract: perp
        margin_mode: isolated
        size_usdt: -109.2914680
        size_base: -63284
        margin_usdt: 10.94
        entry_price: 0.0017200
        position_pnl_usdt: -0.45
        position_roe_pct: -4.05
        liquidation_price: 0.0080195
        mark_price: 0.0017282
      risk:
        risk_ratio: 2.5
        risk_label: Low Risk
        margin_ratio_pct: 1.29
        maintenance_margin_usdt: 5.4684
        isolated_margin_balance_usdt: 423.5148
        liquidation_price: 0.0080195
        mark_price: 0.0017282
        liquidation_distance_to_mark_pct: 364.0377
      tp_sl:
        stop_loss:
          pnl_usdt: -40.00
          roi_pct: -10.0
          price_type: mark
        take_profit:
          pnl_usdt: 60.00
          roi_pct: 15.0
          price_type: mark
        close_all_positions_on_stop: true
        close_all_positions_on_tp_sl_stop: true
    l2_stream:                    # optional; exact collector/run reference
      feature_path: C:/.../l2_risk_snapshots.jsonl
      manifest_path: C:/.../manifest.json
      symbol: BTCUSDT
      strategy_id: "410472444"
      run_id: diff_depth_20260801_185109
      max_age_seconds: 15
      history_window_seconds: 300
      deterioration_min_duration_seconds: 60
      deterioration_min_observations: 3
      deterioration_fraction: 0.80
```

Active/inactive rules:

- Present under `active bots/` = scanned.
- Moved out of `active bots/` = inactive and ignored.
- Per-symbol active files must contain exactly one bot.
- Per-symbol active filenames must match the YAML symbol, e.g. `RENDERUSDT.yaml` must contain `symbol: RENDERUSDT`.
- If one or more per-symbol files exist in `active bots/`, legacy dated files in that folder are ignored by `--bots-dir`.

Loader rules:

- Pick all active per-symbol YAML files from `active bots/`. If none exist, pick the legacy file whose **parsed filename date** is the latest. Do not use `mtime` (protects against "save old file" mistakes).
- Empty directory: no-op tick, no alert.
- Whole-file parse error: emit one `yaml_parse_failed` alert, sleep to next tick.
- Per-bot validation error: skip that bot with a `loader_error` JSONL line; keep evaluating the rest.
- `deploy_ts > now()`: skip with a `bot_in_future` warning until the timestamp passes.

## Decision logic

For each bot in the latest YAML, per tick:

1. `BinanceClient.get_all_market_data(symbol)` -> klines (1h/15m/5m/1m), depth, ticker, funding, OI, LSR, premium index. ~34 weight per symbol.
2. Compute features via existing `feature_extractor`.
3. HMM on the 15m kline frame -> `(range_prob, trend_prob, persistence_prob)`.
4. Meta-labeler -> `meta_proba`. Skip with `meta_overlay_inactive` if >50% of required features are NaN.
5. Execution telemetry overlay, when present:
   - TP/SL PNL thresholds are hard END signals when explicitly crossed.
   - A regime-only END is softened to ADJUST when telemetry shows positive
     matched/realized harvest, an open ladder, and an explicit Low Risk label.
   - A regime-only END remains END and gains `telemetry_no_harvest` when the
     ladder exists but matched profit evidence is non-positive.
   - Complete Chrome drawer fields are preserved in JSONL: transaction fees,
     signed position/inventory, risk/liquidation data, pending-order ladder,
     grid margins/geometry, TP/SL, and UI order-history rows.
   - State records a deployment-scoped, timestamp-deduplicated peak PnL and
     reports current gain giveback. This is observational evidence only; no
     giveback threshold changes CONTINUE/ADJUST/END in this phase.
   - Before any controller action routing, the complete scanner observation is
     also committed as one immutable exact-bot record below
     `Live/YYYY-MM-DD/SYMBOL/pnl_history/<bot_identity>/observations/`. The bot
     identity hashes symbol, exact strategy ID, and deployment timestamp. An
     exact repeat is a no-op; a conflicting payload at the same identity/time,
     a changed source snapshot hash, or a corrupt prior record blocks the
     controller iteration.
5. Utility calibrator -> `utility_score`. Catch `UtilityCalibratorUnavailable`, set `NaN`, log warning. Offline-caller pattern from `safety-invariants.md`.
6. Microstructure adaptive gate.
7. Compute price-vs-grid: `pct_inside_grid`, `dist_to_lower_pct`, `dist_to_upper_pct`. Computed locally in `monitor.py`. Does NOT touch the meta-labeler feature schema, so the Feature Pipeline Update Rule is not engaged.
8. If `candidate_id` provided, join `deploy_linkage_log.csv` for deploy-time deltas (`d_range_prob`, `d_funding`, etc.).
   **Linkage staleness rule (ERR-045):** the linkage log is loaded ONCE per
   process at `MonitorContext.create()` — a long-running loop-mode scanner
   never sees deploys linked after startup (`delta_meta_prob` silently stays
   `None`). Restart the scanner (or run scheduled `--once` ticks, which reload
   the linkage every invocation) after stamping new deploy links.
9. If `l2_stream` is configured, load only fresh records from the same symbol,
   exact strategy target, run, and contiguous sequence segment. Report current/median/p90 spread,
   imbalance, top-N side depth, expected exit fill/VWAP/impact for the signed
   private position, depth-to-position ratios, and interval book-removal/addition
   proxies. These fields are written to `evaluation.l2_risk` and do not alter
   verdicts in this phase. Raw diff-depth remains event-complete; the default
   five-second derivative is a bounded scanner input, not a replacement for raw
   evidence.
10. If `private_event_stream` is configured, require the exact active
    `symbol`, `strategy_id`, and collector `run_id` in both its manifest and
    every canonical JSONL event. Validate manifest freshness and exchange-level
    deduplication keys, then report bounded order-status counts, maker/taker
    fills and notional, realized PnL, commissions, funding, other income, and
    rejected/duplicate ingestion counts in
    `evaluation.private_event_evidence`. The producer's completeness label is
    preserved verbatim: REST order history must be labelled `snapshot_only`,
    while only a genuinely continuous user-data stream may declare
    `event_complete`. This evidence is observational and is not read by the
    verdict mapper.
11. When the L2 reference also contains the collector's same-connection
    `public_agg_trades.jsonl`, join trades only to their exact run and contiguous
    L2 segment after both the run and symbol manifests prove that the collector
    targeted the active strategy ID. Normalize the signed private position to
    its exit side, then report duration/observation/fraction-qualified
    spread/depth persistence, exit-side
    imbalance, trade-aligned removal, unexplained removal, refill and sweep
    proxies, private cancellation updates, fill slippage, and 5s/30s adverse
    selection under `evaluation.execution_risk`. Exchange event timestamps are
    used for the join; local capture timestamps remain transport evidence.
    Aggregate-ID discontinuities and dropped duplicate/out-of-order trades are
    surfaced in `public_trade_status`. This entire record is verdict-inert.
12. If `--pnl-forecast-artifact-dir` names a separately trained artifact, load
    it only when its metadata hash, feature schema, and all bot-disjoint OOS
    gates pass. Emit a fixed-horizon shadow PnL-change estimate, calibrated
    direction probability, and calibrated interval. Missing, rejected, or
    invalid artifacts produce an unavailable shadow field. Forecast output has
    `runtime_effect: none` and cannot alter scanner verdicts or action intents.

The controller accepts repeatable `--private-event-manifest` arguments. Once
any are supplied, every active `symbol/strategy_id` must have exactly one
manifest; partial fleet coverage, symbol-only attribution, duplicate streams,
missing event files, or identity mismatches block that controller iteration.
The age and bounded history window are controlled by
`--max-private-event-age-seconds` and
`--private-event-history-window-seconds`.

Finalized outcomes can later be linked without changing runtime behavior:

```powershell
python -m neutralgrid.live.decision.execution_outcome_analysis `
  --decisions "logs/live_decisions_*.jsonl" `
  --expired-bots data/new_expired_bots.xlsx `
  --output data/live_decisions/execution_outcome_observational.json
```

The report rejects duplicate finalized candidate or strategy identities,
excludes pre-deploy/post-closure ticks, deduplicates repeated evidence rows,
collapses ticks to one row per exact bot, and creates a bot-disjoint
chronological description. It selects no threshold, creates no promotion gate,
does not claim causal gain protection, and never edits scanner configuration.

### Persistent PnL evidence and shadow forecasting

The controller is the sole runtime writer for the persistent PnL dataset. Each
record retains the source drawer path and SHA-256, exact strategy and deployment
identity, observed PnL/position/grid values, live regime probabilities, and
available execution-risk features. Raw public/private evidence stays in its
own immutable collector files; the PnL record stores bounded derived evidence
and the exact run IDs used.

The forecast horizon and permitted label delay are required CLI inputs; there
is deliberately no default. A research run must predeclare both values before
reading outcomes, for example:

```powershell
python scripts/train_live_pnl_forecaster.py `
  --live-root Live `
  --output-dir outputs/audits/live_pnl_forecast_<run_id> `
  --horizon-minutes <predeclared_minutes> `
  --label-tolerance-minutes <predeclared_tolerance_minutes>
```

Forward labels never cross bot identities. Fit, probability/interval
calibration, and final test cohorts are bot-disjoint and chronological; labels
crossing cohort boundaries are purged. The candidate is eligible only when it
meets minimum bot/sample/class counts and does not underperform zero-change,
last-slope, training-prior, or last-direction baselines on the held-out bots.
Serialization writes only a run-specific shadow artifact and never changes the
active HMM, meta-labeler, utility, profile, or scanner manifests.

The 2026-08-10 completion audit predeclared a 30-minute horizon and a 5-minute
label tolerance only to exercise the real-data gate. It found zero persistent
observations and zero bot identities, returned
`insufficient_no_forward_labels`, wrote metadata but no model, and set
`forecast_eligible=false`. Synthetic bot-disjoint tests validate implementation
mechanics only; they are not evidence of live predictive accuracy.

### Strategy-linked private execution import

The static importer accepts an exported Binance `userTrades` JSON payload only
with a separate reviewed `neutralgrid_strategy_order_linkage_v1` artifact. The
artifact must contain the exact symbol, strategy ID, explicit exchange order
IDs, and provenance; symbol/time-only attribution is rejected. Only allowlisted
order fills are mapped, exchange trade IDs are deduplicated, and the output is
validated by the runtime private-event consumer before an atomic directory
rename. Static history may declare only `snapshot_only` or
`history_complete`; it can never declare `event_complete`.

```powershell
python scripts/ingest_private_execution_events.py `
  --source <binance-user-trades.json> `
  --strategy-order-linkage <reviewed-order-linkage.json> `
  --live-root Live `
  --run-id <immutable-run-id> `
  --observed-at-utc <utc-iso-time> `
  --event-completeness snapshot_only
```

### Verdict mapping (contract v1.1, GATEFIX-02 2026-07-13)

**END** when any of:

- Price outside `[grid_lower, grid_upper]` **continuously for >=
  `end_outside_persistence_min` (180) wall-clock minutes, observed on >=
  `end_outside_min_ticks` (3) consecutive ticks** — reason
  `price_outside_persistent:{min}min`. Unobserved time does not count: a gap
  since the last recorded tick larger than `end_state_max_gap_min` (15)
  resets the accumulation, and state predating the bot's `deploy_ts` is
  discarded (stale strategy_id state-key reuse).
- **Displacement disaster stop**: `|price - grid_center| / half_width >=
  end_displacement_multiple` (2.0) — reason `price_displacement:{x}x`,
  single tick, stateless. NOTE: unreachable on the downside when
  `grid_upper > 3 * grid_lower` (price >= 0 caps downside displacement);
  such grids exit via persistence.
- A fired price END **latches** (`price_end_latched`): no CONTINUE
  retraction from one inside tick; cleared after `end_latch_clear_ticks`
  (12) consecutive inside ticks.
- HMM `range_prob < 0.45` AND `trend_prob > 0.40`.
- Microstructure gate fails closed for 3 consecutive ticks.
- Symbol unavailable on Binance USDT-M futures.

Legacy v1.0 behavior — END with `price_outside_grid:{pct}pct` on any single
outside tick — is restorable via `end_on_first_outside_tick: true`. Replay
evidence for the change (638 engine-verified pseudo-deployments, 22 symbols,
2026-07-01..13): the single-tick rule's exit precision was 0.38-0.47 across
every cohort split and it destroyed ~3.7x more winner upside than the
hysteresis rule; hysteresis precision replicated 0.61-0.71 out-of-sample.
Dollar aggregates are symbol-concentrated (small cohort) — the precision and
winner-preservation asymmetry, not the dollar total, is the load-bearing
evidence. See CHANGELOG GATEFIX-02.

**ADJUST** when thesis weakened but not broken:

- Price outside the range but below the hysteresis END thresholds — reason
  `price_outside_watch:{pct}pct` (v1.1; deliberately NOT prefixed by
  `price_outside_grid` so v1.0 log greps cannot mis-match it). Suggested
  re-centered bounds accompany it; the first tick of a fresh excursion
  force-emits even through an escalated-ADJUST cool-down.
- Price within 10% of nearest bound (configurable).
- `range_prob` in `[0.30, 0.45]`.
- Transient fetch error (single tick).
- HMM artifact missing (operational).
- Persistent `data_missing:*` reasons.

For ADJUST, the recommender emits **suggested re-centered bounds**:

```
mid = current_price
half_width = (orig_upper - orig_lower) / 2
suggested_lower = mid - half_width
suggested_upper = mid + half_width
```

Always labelled `suggested_*`. The user retains all judgment. If volatility regime has actually shifted, the recommender omits suggested bounds and emits `regime_shift_recompute_recommended` instead, pushing the user toward the existing scan pipeline.

**CONTINUE** otherwise.

Reasons are an ordered, deterministic list of short codes, e.g. `["range_prob_borderline", "price_near_upper:8.2pct"]`. Order is stable across ticks for diff-friendliness in the Discord digest.

## Concurrency, state, cool-down

### Concurrency

`asyncio.gather` with `Semaphore(8)` cap (override via `--concurrency`). Sequential mode is debug only.

For 30 bots at 5-minute cadence: ~ 30 * 34 / 5 = 204 weight/min, well under the 1200 weight/min Binance budget.

### State

Per-bot ring buffer (last 20 ticks) at `data/live_decisions/state/<strategy_id_or_fallback>.json`. Atomic write via `tempfile + os.replace`. Persists across restarts so "third consecutive ADJUST" escalation survives ctrl-C / laptop sleep.

JSONL audit log at `logs/live_decisions_YYYYMMDD.jsonl`. State files are kept separate so they can be wiped without losing audit history.

### Lock file

`data/live_decisions/.scanner.lock` (PID + start time). A second instance refuses to start.

### Cool-down policy

Evaluated in order:

1. Verdict transition: always emit.
2. `END` cool-down: re-emit at most once / 30 min while bot remains in YAML.
3. `ADJUST` escalation: third consecutive `ADJUST` emits with `escalated=True`. Further `ADJUST`s suppressed until the verdict changes.
4. `CONTINUE` heartbeat: emit every 60 min minimum, plus on session start, plus on transition.

All three configurable: `--end-cooldown-min`, `--adjust-escalate-after`, `--continue-heartbeat-min`.

## Discord digest

One embed per scan tick. Fields only for bots whose verdict, reasons, or escalation flag changed since last emission. Footer aggregates: `{n_continue} CONTINUE / {n_adjust} ADJUST / {n_end} END`.

Token bucket: max 1 message / 15 s, queue if exceeded. With 5-min cadence and 30 bots, peak load is ~1 msg/scan = 0.2 msg/min, well under Discord's 30/min webhook limit.

Webhook URL via `--discord-webhook` or `NEUTRALGRID_DISCORD_WEBHOOK` env var. Missing webhook: warn-and-continue (console + JSONL still flow).

## Safety invariants

Verified respected against `.claude/rules/safety-invariants.md`:

- `hlabel` is never assembled into a feature dict. The tool calls existing entry points only.
- Fail-closed: every `data_missing:*` propagates to the recommendation reasoning. No silent `CONTINUE`.
- HMM lineage via existing `load_hmm_model()` (manifest-driven, auto-reload). No hardcoded artifact path.
- Feature Pipeline Update Rule: price-vs-grid math lives in `monitor.py` and never enters the meta-labeler training schema. The three pipeline files (`candidate_pipeline.py`, `data_generator.py`, `unified_training_builder.py`) are not touched.
- Read-only Binance: only the unsigned endpoints in `get_all_market_data`. `get_positions` and any signed endpoint are forbidden.
- Utility calibrator: offline-caller pattern. Catch `UtilityCalibratorUnavailable`, set `utility_score=NaN`, log warning. Never a silent v0 default.

## Phased rollout

| Phase | Deliverable |
|---|---|
| 0 | This document. Approval-style design doc. |
| A | Skeleton, no network: `loader`, `state_store`, `recommender` pure logic + full pytest. CLI dry-run prints synthetic recommendations. |
| B | Live evaluation: `monitor` wired to `get_all_market_data` -> features -> HMM -> meta -> microstructure. Console renderer + JSONL writer. `--once` mode. |
| C | Recurring + Discord: asyncio loop with `--interval`, `DiscordWebhookHandler` in `alerts.py`, digest formatter, cool-downs, lock file, signal handlers (graceful drain -> `await client.close()` -> persist state). |
| D | Polish: deploy-time deltas via `candidate_deploy_linker`, suggested re-centered bounds, escalation, heartbeat, `--config-file` for thresholds, contract test pinning `DECISION_CONTRACT_VERSION`. |
| E | **Agents-team review (mandatory)**: `portfolio-oversight-lifecycle`, `deployment-engineering`, `data-curator`, `backtest-evaluator` in parallel. Synthesise findings, run `verify-feature-pipeline` + `leakage-check`, land a `CHANGELOG.md` entry via the `changelog-entry` skill. |

Each phase reverts cleanly and produces a usable artifact.

### Phase E reviewer briefs

- **portfolio-oversight-lifecycle**: does the CONTINUE / ADJUST / END verdict model + cool-down policy honor the lifecycle stages (Embargo -> Paper -> Graduation -> Re-allocation -> Decommission)? Are escalation thresholds defensible? Should an `END` recommendation auto-flag the bot for the Decommission gate checklist?
- **deployment-engineering**: prototype-to-production readiness: graceful shutdown, lock file correctness, atomic state writes, async lifecycle (`await client.close()`), latency under N=30 bots, rate-limit headroom, reusability of `decision/` subpackage with the existing `LIVE_BOT_DECISION.md` v2 evaluator (no duplication).
- **data-curator**: YAML loader robustness (malformed inputs, latest-by-parsed-date selection, `deploy_ts > now` rejection), JSONL audit-log schema, Binance read-only-only enforcement, no leakage of telemetry into training datasets, date-format ambiguity in `DD-MM-YY.yaml` filenames.
- **backtest-evaluator**: recommendation logic stress-tested under contrived scenarios (regime flip, microstructure deterioration, transient fetch errors, persistent `data_missing:*`). Are the thresholds (range_prob 0.30/0.45, trend_prob 0.40, 10% boundary proximity) defensible or overfit to current memory state?

Synthesis: main thread reads each report, confirms unanimity or surfaces disagreements explicitly. Any blocker becomes an `ERR-###` entry via the `log-err` skill.

## Verification

Pytest under `tests/live/test_decision_*.py`, no live network:

| Test file | Asserts |
|---|---|
| `test_loader.py` | Valid YAML parse; missing required key raises; empty dir -> empty list (no crash); picks latest by parsed-date; rejects YAML with `deploy_ts > now`. |
| `test_recommender.py` | Pure-logic over synthetic `BotEvaluation`s: each verdict path; escalation after 3 ADJUSTs; END cool-down; CONTINUE heartbeat; deterministic reason ordering. |
| `test_monitor.py` | Monkey-patch `BinanceClient.get_all_market_data` with canned payloads; assert `range_prob` / `pct_inside_grid` flow; fail-closed test asserts `data_missing:1h` propagates and verdict != silent CONTINUE. |
| `test_discord_sink.py` | `httpx_mock`-backed webhook target; assert one POST per tick, JSON shape, token-bucket suppresses bursts; missing webhook -> warn-no-crash. |
| `test_jsonl.py` | Append two ticks; daily rollover when UTC date changes mid-run. |
| `test_state_store.py` | Round-trip; simulated mid-write crash leaves target untouched. |
| `test_alerts_discord.py` | `DiscordWebhookHandler.handle()` formats payload from an `AlertEvent`. |

End-to-end smoke (manual, not CI):

```powershell
# One-shot, single bot from a sample YAML, no Discord
python live_decision_scanner.py --once --bots "active bots\sample.yaml" --no-discord
# Expected: console table with one row + appended line in logs/live_decisions_YYYYMMDD.jsonl

# Recurring with Discord
$env:NEUTRALGRID_DISCORD_WEBHOOK = "<url>"
python live_decision_scanner.py --interval 5m --bots-dir "active bots"
# Expected: tick every 5 min; first tick emits all bots to Discord; subsequent ticks emit only deltas + heartbeat
```

Pre-merge gates:

- `python -m pytest tests/` green.
- `pyright` clean (basic mode).
- No orphaned imports.
- `verify-feature-pipeline` skill PASS (the three feature-pipeline files were not touched).
- `leakage-check` skill PASS (`hlabel` guards intact).

## Edge cases (explicit handling)

| Case | Handling |
|---|---|
| Empty `active bots/` dir | Log `no_active_bots`, sleep to next tick, no alert. |
| Malformed YAML (per-bot) | Per-bot warning in JSONL with `loader_error`, skip that bot, continue scan. |
| Malformed YAML (whole-file) | Single `yaml_parse_failed` alert, sleep to next tick. |
| Symbol not on Binance USDT-M futures | `evaluate_bot` catches the failure, returns `BotEvaluation` with `diagnostics=["symbol_unavailable"]`. Recommender emits `END` with `symbol_unavailable` (not recoverable on next tick -> escalate immediately). |
| HMM artifact missing | Per LIVE_BOT_DECISION Contract 4: emit `ADJUST` (not END) with `hmm_artifact_missing`. Operational, not market. User must fix before next tick. |
| Utility calibrator missing | Catch `UtilityCalibratorUnavailable`, set `utility_score=NaN`, add `utility_calibrator_unavailable` to diagnostics, do **not** flip the verdict. |
| Network blip mid-scan | Per-bot try/except around `evaluate_bot`. Failed bot gets `ADJUST` with `transient_fetch_error`, retried next tick. Three consecutive transient errors on the same bot -> escalate to `END(persistent_fetch_failure)`. |
| User edits YAML mid-tick | Latest YAML is reread on each tick start. In-flight tick uses the snapshot read at its own start; next tick picks up edits. Documented behaviour. |
| Bot in YAML with `deploy_ts > now` | Loader warning, skip until `deploy_ts <= now`. |
| `candidate_id` provided but linker has no match | Set `deploy_delta=None`, add `candidate_link_missing` diagnostic, do **not** fail the bot. |
| Concurrent runs of the scanner | Lock file at `data/live_decisions/.scanner.lock`. Second instance refuses to start. |

## Open items deferred to follow-up plans (not in v1)

- **News / sentiment integration** (CoinGecko / Finnhub / CryptoPanic). Gated behind the pending-external-APIs constraint and the >=500-row training-pool precondition. Would require a new `src/neutralgrid/data/news/` subpackage with rate-limit budgeting per provider, and a sentiment feature contribution to the recommender that does not violate the Feature Pipeline Update Rule.
- **Auto-cancel / parameter modification autonomy.** Explicitly out of scope per user lock. Would require signed Binance API plumbing, a separate safety review, a kill-switch, and an explicit user-facing confirmation step. Not added even behind a feature flag in v1.
- **Reconciling with `LIVE_BOT_DECISION.md` v2.** That draft is awaiting approval. If/when it lands, factor shared logic (telemetry parser, identity contract, microstructure invocation) into `live/decision/` rather than duplicating. The verdict vocabularies are intentionally different and should not be unified.
- **L2 verdict thresholds.** Sequence-verified diff-depth and position-normalized
  risk evidence are now integrated, but no L2-derived threshold changes a
  verdict yet. Same-connection public aggregate trades and exact-strategy
  private events are now time-linked to the sequence-derived L2 window.
  Trade-aligned removals and sweep/refill values remain explicitly labelled as
  proxies; only actual private `CANCELED` updates are called cancellations.
  Slippage/adverse-selection estimates and the bot-level finalized-outcome join
  are available for observation, while any action rule remains deferred until
  sufficient finalized bot-level temporal evidence exists.
- **Live SQLite table for decisions.** The JSONL audit log is sufficient for v1. A `live_decisions` table in `src/neutralgrid/storage/database.py` would enable interactive queries and richer dashboarding; follow up if/when a dashboard is built.
- **Local HTML dashboard.** Considered as an output channel; user chose console + JSONL + Discord instead. Could be added later by reading the JSONL audit log directly.

## References

- `.claude/rules/safety-invariants.md` - leakage, fail-closed, HMM lineage, Feature Pipeline Update Rule.
- `LIVE_BOT_DECISION.md` v2 - sibling single-shot evaluator design.
- `CLAUDE.md` - project conventions, Live Bot Data Storage Policy.
- `run_full_pipeline.py:485-606` - existing `_display_results()` formatter precedent.
- `src/neutralgrid/api/binance_client.py` - read-only endpoints + 1200 weight/min budget.

<!-- Implementation tracking: see plan file ultrathink-we-need-to-rustling-glade.md -->
