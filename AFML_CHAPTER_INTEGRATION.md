# AFML Chapter Integration Map (NeutralGrid Scanner)

Scope: This document maps each chapter of *Advances in Financial Machine Learning* (López de Prado) to concrete modules in this repository.

Legend:
- **Implemented**: There is a concrete implementation used by the pipeline.
- **Partial**: Some concepts are present, but the chapter’s methodology is not fully implemented.
- **Not implemented**: No corresponding module exists in this repository.

Note: “Implemented” means implemented in code and reachable from the current training/scoring pipeline; “Partial” means code exists but is not a full reproduction of the chapter’s methodology.

## Chapter 1 Financial Machine Learning as a Distinct Subject

**Status:** Partial

**Code locations:**

- `run_full_pipeline.py`

- `scanner/scan.py`

- `src/neutralgrid/models/artifacts.py`

- `AFML_ALIGNMENT_CHANGES.md`


**How it is used here:**

The repository uses an explicit *research → training → scoring → deployment* pipeline and enforces training/inference consistency via artifact metadata and schema checks. It does **not** implement AFML’s full “meta-strategy” station framework, but it now avoids leakage and fabricated inputs during selection and scoring.



## Chapter 2 Financial Data Structures

**Status:** Partial

**Code locations:**

- `src/neutralgrid/data/curator.py`

- `api/binance_client.py`

- `data/*`


**How it is used here:**

Market data is represented as time bars (OHLCV). AFML’s alternative bar types (e.g., volume/dollar/imbalance bars) are **not** implemented in this codebase.



## Chapter 3 Labeling

**Status:** Implemented

**Code locations:**

- `src/neutralgrid/models/triple_barrier.py`

- `src/neutralgrid/training/data_generator.py`

- `src/neutralgrid/models/barrier_config.py`


**How it is used here:**

Triple-barrier labeling is implemented for training labels, with intrabar PT/SL detection (high/low) and event end-times (`t1`) stored for downstream purging/embargo.



## Chapter 4 Sample Weights

**Status:** Implemented

**Code locations:**

- `src/neutralgrid/training/sample_weights.py`

- `src/neutralgrid/models/meta_labeler.py`


**How it is used here:**

Concurrency/uniqueness-based sample weighting is available and used when `t1` is present, reducing the influence of heavily overlapping events.



## Chapter 5 Fractionally Differentiated Features

**Status:** Not implemented


**How it is used here:**

No fractionally differentiated feature engineering module exists (no fixed-width window fracdiff, no selection of `d` via stationarity tests).



## Chapter 6 Ensemble Methods

**Status:** Implemented

**Code locations:**

- `src/neutralgrid/models/meta_labeler.py`


**How it is used here:**

Meta-labeling uses an ensemble classifier (`GradientBoostingClassifier`) with probability calibration. This is used as a second-stage filter/weighting over candidates produced by the scan/enrichment pipeline.



## Chapter 7 Cross-Validation in Finance

**Status:** Implemented

**Code locations:**

- `src/neutralgrid/backtest/cpcv.py`

- `src/neutralgrid/models/meta_labeler.py`


**How it is used here:**

CPCV with purging/embargo is implemented. When event spans (`t1`) are available, the meta-labeler prefers time-based purging and embargo aligned to event horizons.



## Chapter 8 Feature Importance

**Status:** Partial

**Code locations:**

- `src/neutralgrid/models/meta_labeler.py`

- `src/neutralgrid/training/README.md`


**How it is used here:**

Tree-based feature importance is recorded for the meta-labeler (via `feature_importances_`). AFML’s more comprehensive importance toolset (MDI/MDA/SFI with purged CV) is not fully reproduced as a dedicated module.



## Chapter 9 Hyper-Parameter Tuning with Cross-Validation

**Status:** Not implemented


**How it is used here:**

No systematic hyper-parameter tuning loop (e.g., nested purged CV, Bayesian optimization) is implemented. Model hyperparameters are configured directly in code/config.



## Chapter 10 Bet Sizing

**Status:** Partial

**Code locations:**

- `grid/calculator.py`

- `grid_bot_manager.py`

- `src/neutralgrid/models/meta_labeler.py`


**How it is used here:**

The bot includes regime-aware exposure scaling and a calibrated meta-label probability, but it does **not** implement AFML’s probabilistic bet sizing (e.g., sizing via side/predicted probability and average active signal) as a standalone sizing engine.



## Chapter 11 The Dangers of Backtesting

**Status:** Partial

**Code locations:**

- `backtest/*`

- `src/neutralgrid/backtest/cpcv.py`


**How it is used here:**

The repository includes backtesting utilities and CPCV; it does not include a full AFML backtest error taxonomy and remediation workflow beyond leakage prevention and schema checks.



## Chapter 12 Backtesting through Cross-Validation

**Status:** Implemented

**Code locations:**

- `src/neutralgrid/backtest/cpcv.py`


**How it is used here:**

Backtesting through (purged) cross-validation is implemented via CPCV. This is used for model selection metrics in the meta-labeler training path.



## Chapter 13 Backtesting on Synthetic Data

**Status:** Not implemented


**How it is used here:**

No synthetic data generator/backtesting-on-synthetic framework is included.



## Chapter 14 Backtest Statistics

**Status:** Partial

**Code locations:**

- `src/neutralgrid/backtest/cpcv.py`


**How it is used here:**

The CPCV module includes deflated performance metrics scaffolding; however, a full AFML backtest statistics report (including full set of deflated tests and multiple-testing adjustments for strategy selection) is not integrated end-to-end in the pipeline.



## Chapter 15 Understanding Strategy Risk

**Status:** Partial

**Code locations:**

- `src/neutralgrid/backtest/cpcv.py`

- `metrics/*`


**How it is used here:**

Risk and performance metrics exist, and CPCV includes deflation concepts, but the full “strategy risk” decomposition and reporting framework is not implemented as a dedicated module.



## Chapter 16 Machine Learning Asset Allocation

**Status:** Not implemented


**How it is used here:**

No ML-based asset allocation / portfolio optimization module exists in this repository.



## Chapter 17 Structural Breaks

**Status:** Not implemented


**How it is used here:**

No structural break detection module exists (e.g., CUSUM-based break tests, change-point detection).



## Chapter 18 Entropy Features

**Status:** Not implemented


**How it is used here:**

No entropy feature library exists (e.g., permutation entropy, Lempel–Ziv, etc.).



## Chapter 19 Microstructural Features

**Status:** Partial

**Code locations:**

- `scanner/enrich_grid_params.py`

- `src/neutralgrid/data/curator.py`


**How it is used here:**

The scanner computes regime/volatility-related gates and grid-range-derived microstructure proxies, but AFML’s full microstructural feature set (e.g., order-flow/imbalance measures) is not implemented.



## Chapter 20 Multiprocessing and Vectorization

**Status:** Partial

**Code locations:**

- `retrain_hmm.py`

- `retrain_meta_labeler.py`

- `scanner/*`


**How it is used here:**

Some workflow scripts exist for retraining and scanning, but there is no systematic multiprocessing/vectorization module specifically implementing AFML’s techniques (e.g., multiprocessing patterns for research loops).



## Chapter 21 Brute Force and Quantum Computers

**Status:** Not implemented


**How it is used here:**

No brute-force/quantum computing chapter content is implemented (as expected for a trading bot codebase).



## Chapter 22 High-Performance Computational Intelligence and Forecasting Technologies

**Status:** Not implemented


**How it is used here:**

No computational intelligence / forecasting technologies chapter content is implemented beyond the HMM + supervised meta-labeler used in this repository.


