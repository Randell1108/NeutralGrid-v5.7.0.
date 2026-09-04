# CLAUDE.md

## Project Overview
- NEUTRAL Grid Bot v6.5.8 — Binance USDT-M futures grid bot with AFML-aligned pipeline
- Python 3.11+, package source in 'src/neutralgrid/'
- Pipeline flow: scan → enrich → backtest → deploy

## Commands

### Install
- 'pip install -e .' — editable install from pyproject.toml
- 'pip install -r requirements.txt' — dependencies ('requirements.lock' for CI)

### Tests
- 'python -m pytest tests/' — all tests (~1142 as of 2026-04-29)
- 'python -m pytest tests/unit/' — unit tests only
- 'python -m pytest tests/test_replay.py::test_name -v' — single test

### Type Checking
- 'pyright' — configured in pyproject.toml

### Pipeline
- 'python run_full_pipeline.py' — scan → enrich → deploy 
- 'python retrain_hmm.py' — retrain HMM regime model
- 'python scripts/backfill_training_features.py --input data/new_expired_bots.xlsx --output data/new_expired_bots_backfilled.xlsx --default-artifact-version <active_hmm>' —
  re-backfill expired-bot HMM features against the active HMM. Per UTILFIX-01,
  '--default-artifact-version' is AUTHORITATIVE: rows whose merge-preserved
  hmm_artifact_version differs from this value are invalidated and re-inferenced.
  Add '--skip-if-fresh' for routine ingestions: per-row inference is skipped
  when the post-merge row already has hmm_artifact_version == active_hmm AND
  finite range_prob/trend_prob/persistence_prob. Stale-lineage rows are still
  re-inferenced because the merge invalidates them; after an HMM rotation the
  flag is effectively a no-op for that run. Default off (opt-in).
- 'python retrain_meta_labeler.py --backtest-results-dir <finalized_fresh_pool>' — retrain meta-labeling classifier (auto-pins to active HMM). The source must be a finalized `fresh_full_pool` manifest; historical exact replays require explicit `--allow-historical-replay` and are diagnostic only.
- 'python retrain_scanner.py' — retrain pattern scanner
- 'python scripts/recalibrate_utility.py' — refit utility calibrator (promotion gated by G0-G7 in 'calibration/utility_calibrator.py'; UTILFIX-01 fail-closed runtime)

### Backtesting
- 'python backtest_candidates.py --fastwin-full-pool --output <fresh_run_dir>' — create the unbounded fresh FASTWIN full-pool backtest rows. Backfill those rows against the active HMM, finalize them with `scripts/finalize_fresh_authoritative_meta_pool.py`, and only then pass that finalized directory to `retrain_meta_labeler.py`.
- 'python backtest_candidates_current.py' — out-of-sample validation vs current market

### CLI Entry Points
- 'neutralgrid-retrain' — model retraining ('cli/retrain.py')
- 'neutralgrid-replay' — order-book replay ('replay/cli.py')

### Dependency Check
- 'python scripts/check_deps.py' — validates pyarrow, numpy, pandas, scikit-learn, hmmlearn, joblib

## Coding Conventions

### Imports
- All modules use 'from __future__ import annotations'
- Order: standard lib → third-party → 'neutralgrid.*'
- 'backtest/' is NOT part of the installed package; tests use 'sys.path.insert(0, ...)'

### Config Access
- 'from neutralgrid.core.config import get_config'
- Mutable dataclasses, env vars from '.env' override defaults
- Project root: 'NEUTRALGRID_BASE_DIR' env var or auto-detected

### Type Checking
- Pyright basic mode
- Use 'np.asarray()' instead of 'pd.Series.values' for typed ndarray params
- Wrap numpy scalar returns ('np.mean()' etc.) with 'float()'
- Add 'is not None' guards before accessing Optional attributes

### pandas Pyright Patterns
- 'df[col]' returns union type — use 'cast(pd.Series, df[col])' before '.iloc', '.fillna()', '.notna()', etc.
- For Series-as-bool: 'if bool(df["col"].any()):' or 'if not df.empty:'
- Import 'cast' from 'typing'

### numpy 2.4+ Compatibility
- 'isinstance(np.bool_(True), bool)' returns False on numpy 2.4+
- Use 'isinstance(x, (bool, np.bool_))' for bool checks
- Use 'isinstance(x, (int, float, np.integer, np.floating))' for numeric checks

### pandas Timezone Handling
- Use 'pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")' to avoid mixed tz-aware/naive warnings

### Logging
- Use 'logger = logging.getLogger(__name__)', not 'print()'

## Behavioral Guidelines

### Think Before Acting
- State assumptions explicitly before implementing. If uncertain, ask.
- If multiple valid approaches exist, present them with tradeoffs — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear or ambiguous, stop. Name what's confusing. Ask.
- Before modifying code near safety invariants (leakage guards, fail-closed gates, feature pipeline), state which invariants are in scope and confirm understanding before touching anything.

### Surgical Ownership
- Every changed line must trace directly to the user's request.
- When your changes orphan an import, variable, or function, clean it up — that's your mess.
- Do NOT touch pre-existing dead code, unused imports, or stale comments unless explicitly asked.
- Match the existing style of the file you're editing, even if you'd write it differently.

### Verification-Driven Execution
- Transform tasks into verifiable goals before starting:
  - "Fix the bug" → write a test that reproduces it, then make it pass
  - "Add a feature" → define what 'done' looks like, implement, verify with 'python -m pytest tests/'
  - "Refactor X" → confirm 'pytest' + 'pyright' pass before AND after
- For multi-step tasks, state a brief plan with a check at each step:
  1. [Step] → verify: [how]
  2. [Step] → verify: [how]
- After modifying files covered by the Feature Pipeline Update Rule (safety-invariants.md), verify all three files are consistent before reporting done.

## Live Bot Data Storage Policy

- All newly ingested live bot data must be stored under `C:\Users\cris_\OneDrive\Documents\Christian\Crypto\Neutral Grid Bots\NEUTRAL grid bot v6.5.8 - Clean\Live` (this repository's `Live\` folder; repointed from the retired non-Clean tree 2026-07-10, historical folders copied over)
- For each ingestion event, create a new date folder inside `Live\` using the ingestion date in `YYYY-MM-DD` format
- Inside that date folder, create a symbol-specific folder named with the trading pair/symbol
- Store the user-provided live bot data only inside that `Live\<ingestion-date>\<SYMBOL>\` folder
- Do not place new live bot data anywhere else in the repository

## Package Subpackages
- 'neutralgrid.api' — FastAPI app, Binance client, schemas
- 'neutralgrid.backtest' — candidate pipeline, CPCV, evaluation
- 'neutralgrid.calibration' — temperature scaling, beta calibration, conformal risk control, ESS, utility calibrator
- 'neutralgrid.cli' — retrain CLI entry point
- 'neutralgrid.core' — config, constants, exceptions, logging, candidate IDs, protocols
- 'neutralgrid.data' — market data, funding rates, features, Binance Vision ingest, price series
- 'neutralgrid.grid' — spacing profiles
- 'neutralgrid.indicators' — technical indicators
- 'neutralgrid.live' — deployment payloads, candidate-deploy linkage
- 'neutralgrid.metrics' — PnL calculator, grid bot manager, excursion MtM, PnL curve features
- 'neutralgrid.models' — HMM (train/inference/retrain), meta-labeler, triple barrier, artifacts
- 'neutralgrid.optimization' — Bayesian threshold optimizer
- 'neutralgrid.replay' — order-book replay, normalization, export
- 'neutralgrid.scanner' — two-stage selector, feature extractor, tradable oscillation, PnL ranker, profile model, entropy-adaptive thresholds, microstructure gate
- 'neutralgrid.storage' — database layer
- 'neutralgrid.training' — data generator, unified training builder, hierarchical labels, sample weights, holdout validator, live outcome ingestor, trial tracker
- 'neutralgrid.validation' — HMM regime, stochastic, profile gate, utility

## Custom Agents

### Defined in '.claude/agents/'
- 'backtest-evaluator' — strategy robustness, overfitting probability, scenario analysis
- 'data-curator' — data intake/cleaning/storage, quote context, asset-class nuances
- 'deployment-engineering' — prototype → production, equivalence, latency, component reuse
- 'feature-analyst' — feature extraction, signal cataloguing, quality assessment
- 'market-strategy-architect' — theory development, economic mechanisms, strategy formulation
- 'portfolio-oversight-lifecycle' — lifecycle stages: Embargo → Paper → Graduation → Re-allocation → Decommission
