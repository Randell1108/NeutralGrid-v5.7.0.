# AFML Pipeline Evaluation Log

This document records each AFML compliance audit run. Each evaluation follows
the canonical AFML pipeline order:

```
Raw Data (Ch 2) -> Features (Ch 8) -> Labels (Ch 3) -> Weights (Ch 4)
  -> CPCV (Ch 7) -> Meta-Model (Ch 6) -> Evaluation (Ch 11-12) -> Deployment
```

Scores are 0-100 per stage. The overall score is the weighted average
(weights reflect AFML importance to deployment safety).

---

## Evaluation #1 — 2026-03-22

**Codebase version:** v6.5.6-afml-audit-fixes
**Audited by:** 8 parallel AFML sub-agents (one per pipeline stage)
**Remediation:** 3-team parallel (Implementation, Bug Fix, Quality Control)
**Verification:** pyright 0 errors, pytest 996/996 passed

### Stage Scores

| # | AFML Stage | Ch | Score | Weight | Weighted | Key Strengths | Key Gaps |
|---|---|---|---|---|---|---|---|
| 1 | **Raw Data Curation** | Ch 2 | **75** | 10% | 7.5 | Curator gates, dedup policy, manifest provenance, staleness detection | `strict=False` default allows bad data; outlier gate never fails; tick/volume bars absent |
| 2 | **Feature Extraction** | Ch 8 | **65** | 12% | 7.8 | MI-weighted scoring excellent; information-driven features; VIF diagnostics exist | No ADF stationarity tests; no fractional differentiation; no MDI/MDA/SFI importance methods; no substitution analysis |
| 3 | **Label Generation** | Ch 3 | **60** | 15% | 9.0 | Hierarchical L1/L2/L3/Meta system; label contract versioned; leakage guards | No first-touch barrier detection; terminal-PnL barrier inference; no concurrent label handling; fixed horizon not event-driven |
| 4 | **Sample Weights** | Ch 4 | **85** | 10% | 8.5 | Uniqueness weighting correct; time-decay with N_eff guard; sequential bootstrap; per-fold recomputation | Beta calibrator now accepts weights (fixed); OOS metrics now weighted (fixed) |
| 5 | **CPCV** | Ch 7 | **70** | 12% | 8.4 | Core CPCV with purge+embargo; symbol-blocked groups; chronological holdout; within-group temporal purge | Walk-forward not truly combinatorial (sequential folds only); conformal not integrated into meta-labeler training loop |
| 6 | **Meta-Model** | Ch 6 | **80** | 15% | 12.0 | Two-stage calibration (sigmoid+beta); dual leakage guards; 5 versioned feature profiles; CPCV-aware imputation | Silent feature imputation at inference (warning added); sklearn version affects calibration path; permutation importance lacks CI |
| 7 | **Evaluation** | Ch 11-12 | **65** | 14% | 9.1 | Realistic backtest engine (fees, funding, slippage, delays); event-driven; trial tracker exists; DSR/PBO implemented | DSR never applied to results; PBO never computed across paths; Sharpe annualization fixed but short-backtest warning absent |
| 8 | **Deployment** | — | **75** | 12% | 9.0 | Two-stage gate architecture; multiplicative position sizer; Kelly criterion; hard microstructure gate (5 checks) | EV ranking root cause fixed; conformal gate now fail-closed; model staleness only warns, doesn't block |

### Overall Score

| Metric | Value |
|---|---|
| **Weighted Average** | **71.3 / 100** |
| **Minimum Stage** | 60 (Label Generation, Ch 3) |
| **Maximum Stage** | 85 (Sample Weights, Ch 4) |
| **Critical Fixes Applied** | 10 |
| **Remaining Critical Gaps** | 6 |

### Score Interpretation

| Range | Rating | Meaning |
|---|---|---|
| 90-100 | Excellent | Full AFML compliance, production-ready |
| 80-89 | Good | Minor gaps, safe for production with monitoring |
| 70-79 | Adequate | Significant gaps but core framework sound |
| 60-69 | Needs Work | Critical gaps that affect model integrity |
| < 60 | Deficient | Fundamental AFML violations |

**Current Rating: Adequate (71.3)**

The core AFML framework is correctly implemented (CPCV, sample weights,
calibration, fail-closed gates, realistic backtesting). The main gaps are
in statistical validation (stationarity, feature importance methods) and
label methodology (first-touch barriers, concurrent labels). These are
design-level improvements, not bugs.

### Fixes Applied This Run

| Fix | Stage | Impact on Score |
|---|---|---|
| Conformal gate fail-closed | Deployment | +5 (70 -> 75) |
| Beta calibrator sample weights | Ch 4 | +5 (80 -> 85) |
| OOS evaluation weighted | Ch 6 | +3 (77 -> 80) |
| Feature imputation warning | Ch 6 | +2 (included in 80) |
| EV ranking root cause | Deployment | +5 (70 -> 75) |
| Sharpe annualization + moments | Ch 11-12 | +5 (60 -> 65) |
| API kline deduplication | Ch 2 | +2 (73 -> 75) |
| Capital fraction guard | Deployment | +1 (included in 75) |
| AFML Ch3 deviation documented | Ch 3 | +0 (documentation only) |
| Bridge dedup logging | Ch 2 | +1 (included in 75) |

### Remaining Critical Gaps (For Future Runs)

| Gap | Stage | Effort | Impact |
|---|---|---|---|
| ADF stationarity testing on features | Ch 8 | Medium | Would raise Ch 8 to 75+ |
| First-touch barrier detection | Ch 3 | High | Would raise Ch 3 to 75+ |
| Deflated Sharpe integration into backtest results | Ch 11-12 | Medium | Would raise Ch 11-12 to 80+ |
| PBO computation across candidate paths | Ch 11-12 | Medium | Would raise Ch 11-12 to 80+ |
| Combinatorial walk-forward for HMM | Ch 7 | High | Would raise Ch 7 to 80+ |
| Model staleness as hard gate | Deployment | Low | Would raise Deployment to 80+ |

### Audit Methodology

Each AFML stage was audited by a dedicated sub-agent that:

1. Read all relevant source files (exact paths and line numbers)
2. Compared implementation against AFML textbook requirements
3. Identified compliance strengths, violations, and code bugs
4. Produced specific recommendations with file:line references

Three remediation teams then worked in parallel:

- **Team 1 (Implementation):** Fixed conformal gate, calibrator weights, OOS
  metrics, feature imputation warning
- **Team 2 (Bug Fix):** Fixed EV ranking root cause, Sharpe annualization,
  dedup logging, capital fraction guard, AFML documentation
- **Team 3 (Quality Control):** Verified py_compile, pyright, pytest, import
  smoke tests across all modified files

---

*Next evaluation should be run after implementing the remaining critical gaps
(stationarity testing, first-touch barriers, DSR/PBO integration). Expected
score improvement: 71 -> 80+.*
