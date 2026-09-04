"""
Single source of truth for version constants used across the backtest subsystem.

Step 4 (Plan v6.0): Eliminates duplicated version literals across
btk_label_contract.py, btk_unified_runner.py, candidate_pipeline.py, and
unified_training_builder.py.

All version strings MUST be imported from this module.  Never hardcode
version literals elsewhere.
"""
from __future__ import annotations

# ── Label contract version ───────────────────────────────────────────────────
# Date-based format (YYYY-MM-DD).  Bump to today's date when required fields
# change or label semantics are redefined.  The builder compares versions as
# dates so it can deterministically keep the newest backtest per candidate_id
# and gate rows produced under older contracts.
LABEL_CONTRACT_VERSION: str = "2026-05-09"

# ── Formula version ──────────────────────────────────────────────────────────
# Tracks the label-computation semantics (PnL formula, rotation metrics).
# Bump when the formula for net_pnl, labels, or rotation metrics changes.
FORMULA_VERSION: str = "alignment-v2-geometric-realism"

# ── Engine version ───────────────────────────────────────────────────────────
# Bumped when the simulation logic in backtest_realistic.py changes in a
# way that affects label values.
ENGINE_VERSION: str = "realistic-v8"

# ── Bot-lifetime horizon (single source of truth) ────────────────────────────
# Canonical H* for the meta-labeler target, triple-barrier vertical, Monte-Carlo
# survival window, grid runtime, and backtest label horizon.  All bot-lifetime
# defaults MUST import from this module.  See DURATION_FIX.md §5 / §11 Step 1.
BOT_HORIZON_HOURS: float = 6.0
BOT_HORIZON_BARS_15M: int = 24       # = int(BOT_HORIZON_HOURS * 60 / 15)
BOT_HORIZON_SECONDS: int = 21_600    # = int(BOT_HORIZON_HOURS * 3600)

# ── Inclusion tolerance band (PROVISIONAL) ───────────────────────────────────
# Absorbs fill/cancel/metric-flush latency so live bots closing at H*+ε are not
# excluded from training.  PROVISIONAL: no latency distribution has been
# measured from Live/<date>/<SYMBOL>/ ingestion yet.  Do NOT widen without
# evidence.  See DURATION_FIX.md §5 and §8.7.
BOT_INCLUSION_TOLERANCE_HOURS: float = 0.25

# ── Profit-factor sanitization cap (PATTERN_PROFILE_FIX Phase 1.6) ───────────
# Pathological divisions (zero gross loss) yield infinite profit_factor values.
# Both scanner profile builders clip to this cap after replacing ±inf with NaN.
# Moved here from duplicate literals in pattern_profile.py and profile_model.py.
PROFIT_FACTOR_CAP: float = 1000.0

# ── Decision contract version (LIVE_DECISION + LIVE_BOT_DECISION) ────────────
# Bumped when the verdict vocabulary, reason-code set, or evaluation contract
# of the live decision tools changes in a way that would invalidate persisted
# state files or downstream consumers' parsing logic.
# 1.1 (GATEFIX-02, 2026-07-13): hysteresis END trigger. New reason codes
# price_outside_watch: (ADJUST) and price_outside_persistent: /
# price_displacement: / price_end_latched (END); price_outside_grid: is now
# emitted only under the legacy end_on_first_outside_tick=True config.
# BotHistory gains additive state fields; pre-1.1 state files load unchanged.
DECISION_CONTRACT_VERSION: str = "1.1"

# ── Minimum positive-class rate for a trainable meta-labeler target (ERR-035) ─
# Below this the class weight blows up and the model is unstable. Single source
# of truth shared by the trainer's fail-closed gate ('models/meta_labeler.py')
# and the unified training-builder label-precedence fallbacks
# ('training/unified_training_builder.py'). Do not inline the 0.05 literal.
MIN_POSITIVE_RATE: float = 0.05
