# PnL Curve Shape Classification — Review & AFML-Aligned Proposal

## 1. Current Implementation

**File:** `src/neutralgrid/metrics/pnl_curve_features_v20260310.py`
**Function:** `_classify_shape()` (lines 213–239)

**Current categories** (4):
- `flat` — normalized activity < 0.1
- `step` — max_step_ratio > 0.7
- `volatile` — entropy > 2.0 AND active fraction > 0.5
- `gradual` — everything else

**Output column:** `pnl_curve_shape_class`
**Downstream use:** Written to `PnL Curve Features` sheet in `data/new_expired_bots.xlsx`.
**NOT used as a meta-labeler feature** — absent from the active `SNAPSHOT_META_FEATURES_V20260421_BOOTSTRAP` (8 features) in `src/neutralgrid/models/meta_labeler.py:127-138` and from the historical `SNAPSHOT_META_FEATURES_V20260407` / `SNAPSHOT_META_FEATURES_V20260420` profiles.

---

## 2. Why the Current Classifier Is Insufficient

### 2.1 Morphology collapse
The four-bucket scheme collapses **structurally distinct** PnL curve morphologies into the `gradual` bucket:

| Observed morphology (in recent manual bots)  | Current class |
|-----------------------------------------------|---------------|
| Monotone rising (steady grid compounding)     | `gradual`     |
| Peak-then-decay (winner that started reversing) | `gradual`     |
| Dip-then-recovery (survived a drawdown)       | `gradual`     |
| Late-breakout ramp (back-loaded gains)        | `gradual`     |

All four have **very different forward expected value** and **very different regime attribution**, yet the current classifier treats them identically. This is information loss.

### 2.2 Not propagated to training
Even if the classifier were richer, `pnl_curve_shape_class` is not in `SNAPSHOT_META_FEATURES_V20260421_BOOTSTRAP` (the active profile per `meta_labeler.py:127-138`), so the meta-labeler cannot use it. Adding it would require a synchronized update across the three files listed in `.claude/rules/safety-invariants.md` §"Feature Pipeline Update Rule".

### 2.3 Classifier is post-outcome
`_classify_shape()` reads a realized PnL time series. It is a **label-summary** feature, not an ex-ante feature. Per AFML §3.6 it cannot be used directly as a meta-labeler input without violating the observability-at-decision-time invariant.

---

## 3. How a Proper Classification Would Help the Model

The shape class is useful in **three distinct channels**, each with different leakage profiles:

### 3.1 Label enrichment (AFML §3.5 — meta-labeling)
Convert binary `y ∈ {0,1}` to a **hierarchical label** `y ∈ {loss, flat_win, gradual_win, step_win, peak_decay_win, …}`. The meta-labeler becomes a multi-class classifier; downstream aggregation recovers the binary signal plus *shape expectation*. AFML shows hierarchical labels improve calibration.

### 3.2 Sample weighting (AFML §4.5)
Down-weight `flat` wins (low information content) and up-weight `volatile_survived` wins (high uniqueness). Uses the shape signal **through the weighting channel**, zero leakage risk.

### 3.3 Regime stratification in purged CV (AFML §7.4)
Use shape class as a **stratification key** so each fold contains a balanced mix of morphologies. Prevents a fold of all `step` wins from producing a classifier that only recognizes step patterns.

---

## 4. Proposed Classifier (Simple, AFML-Aligned, No Over-Engineering)

**Keep the existing 4 categories as a base, add 3 morphology discriminators.** Final scheme: **7 classes**, all computable from the existing PnL time series — no new data sources.

### 4.1 Categories

| Class              | Criterion (on normalized cumulative PnL `c(t)`, `t ∈ [0,1]`) |
|--------------------|----------------------------------------------------------------|
| `flat`             | `std(c) < 0.1 · |c(1)|` (current definition, unchanged)       |
| `step`             | `max_step_ratio > 0.7` (current, unchanged)                    |
| `monotone_rising`  | `c` non-decreasing on ≥ 90% of bars AND `c(1) > 0`             |
| `peak_decay`       | `argmax(c) < 0.7·T` AND `c(1) < 0.9·max(c)`                    |
| `dip_recovery`     | `argmin(c) < 0.5·T` AND `c(1) > 0` AND `min(c) < −0.3·c(1)`    |
| `volatile`         | `entropy > 2.0` AND `active_frac > 0.5` (current, unchanged)   |
| `gradual`          | None of the above (residual bucket, now much smaller)          |

### 4.2 Classification order
Apply rules **in the order listed** — first match wins. This matches the existing fall-through structure in `_classify_shape()` and avoids ambiguous multi-class membership.

### 4.3 Mathematical precision
All criteria use only:
- `c(t)` — normalized cumulative PnL series (already computed)
- `argmax`, `argmin`, `std`, `entropy`, `active_frac` — already available or trivially derived

**No new parameters, no new dependencies, no new computation beyond `O(T)` per bot.**

---

## 5. Integration Path (AFML-Aligned, Fail-Closed)

### 5.1 Phase 1 — classifier update only
Update `_classify_shape()` in `pnl_curve_features_v20260310.py` to the 7-class scheme. Output continues to flow to the `PnL Curve Features` sheet. **No training pipeline change yet.** Purely diagnostic.

### 5.2 Phase 2 — stratification key
Use the new `pnl_curve_shape_class` as a **stratification key** in purged/embargoed CV fold assignment. Requires no change to the feature set — only to fold construction. Zero leakage risk.

### 5.3 Phase 3 — sample weights
Incorporate shape class into the sample-weight computation in `src/neutralgrid/training/` (currently uses uniqueness weights per AFML §4.5). Down-weight `flat`, up-weight `dip_recovery` and `volatile`. Still no feature-set change.

### 5.4 Phase 4 — hierarchical label (optional, last)
If Phases 1–3 show improved walk-forward metrics, promote to hierarchical meta-labeling (AFML §3.5). Requires label-contract version bump (`LABEL_CONTRACT_VERSION` in `src/neutralgrid/core/constants.py`).

**Each phase is independently reversible.** No phase requires promoting `pnl_curve_shape_class` to an ex-ante feature — which would violate observability and require a forward-looking shape predictor (explicitly out of scope).

---

## 6. What This Proposal Does NOT Do

- ❌ Does **not** add `pnl_curve_shape_class` to `SNAPSHOT_META_FEATURES_V20260421_BOOTSTRAP` or any other profile (would be post-outcome leakage).
- ❌ Does **not** introduce a new file, new dependency, or new config section.
- ❌ Does **not** change the `PnL Curve Features` sheet schema beyond the enum values of one existing column.
- ❌ Does **not** touch `_KNOWN_LABEL_COLUMNS`, `hlabel`, or any leakage guard.
- ❌ Does **not** add a forward-looking shape predictor.

---

## 7. Success Criteria (Walk-Forward)

A phase is promoted only if:
1. **Phase 1:** class distribution on historical data is non-degenerate (no class >60%, no class <2%).
2. **Phase 2:** purged-CV fold variance in AUC/Brier score **decreases** ≥ 10% vs current stratification.
3. **Phase 3:** meta-labeler Brier score on rolling walk-forward **improves** or stays within noise (paired Diebold-Mariano test, p < 0.1).
4. **Phase 4:** hierarchical-label model's marginal binary AUC ≥ current binary-label AUC on ≥ 60% of walk-forward windows.

Criteria mirror the existing `mean_pass_rate >= 0.50` gate in `promote_hmm_version()` per `.claude/rules/safety-invariants.md`.

---

## 8. References

- López de Prado, M. *Advances in Financial Machine Learning* (2018):
  - §3.2 Triple-Barrier Method
  - §3.5 Meta-Labeling
  - §3.6 Feature Observability
  - §4.5 Sample Uniqueness Weights
  - §7.4 Purged/Embargoed Cross-Validation
  - §17 Structural Breaks
- `.claude/rules/safety-invariants.md` (this repo) — Leakage Prevention, Fail-Closed Behavior, Feature Pipeline Update Rule.
- `src/neutralgrid/metrics/pnl_curve_features_v20260310.py:213-239` — current `_classify_shape()`.
- `src/neutralgrid/models/meta_labeler.py:127-138` — active `SNAPSHOT_META_FEATURES_V20260421_BOOTSTRAP` (8 features). Historical: `SNAPSHOT_META_FEATURES_V20260407` (lines 47-80, 29 features), `SNAPSHOT_META_FEATURES_V20260420` (lines 82-125, 38 features). See `CHANGELOG.md:281-292` for downgrade rationale.
