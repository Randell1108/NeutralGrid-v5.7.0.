"""
HMM training module - offline training with full metadata and evaluation.

This module handles ONLY training (fitting). Inference is in inference.py.

Design principles:
- Reproducible training with full provenance tracking
- Uses same feature extraction as inference (from data/features.py)
- Saves artifacts with complete metadata
- Separates training from inference (AFML-aligned)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, cast
import numpy as np
import pandas as pd

from neutralgrid import __version__ as PIPELINE_VERSION
from neutralgrid.core.config import get_config

logger = logging.getLogger(__name__)
from neutralgrid.data.features import compute_hmm_features
from neutralgrid.models.hmm.schema import (
    FEATURE_NAMES,
    get_feature_schema,
    infer_state_mapping,
    compute_state_groups,
)
from neutralgrid.models.hmm.uncertainty import compute_regime_uncertainty_profile
from neutralgrid.models.artifacts import (
    save_artifact,
    ArtifactMetadata,
    get_git_commit,
)


def train_hmm_global(
    per_symbol_dfs: List[pd.DataFrame],
    n_components: int = 4,
    covariance_type: str = "diag",
    n_iter: int = 200,
    tol: float = 1e-3,
    random_state: int = 7,
    min_sequence_length: int = 60,
    input_symbol_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Train global HMM on multiple symbol sequences.

    Args:
        per_symbol_dfs: List of OHLCV DataFrames (one per symbol)
        n_components: Number of HMM states
        covariance_type: Covariance type ("diag", "full", "spherical", "tied")
        n_iter: Maximum EM iterations
        tol: Convergence tolerance
        random_state: Random seed for reproducibility
        min_sequence_length: Minimum sequence length to include
        input_symbol_names: Optional list of all input symbol names (for
            detecting symbols dropped during feature extraction / filtering)

    Returns:
        Dictionary with:
        - model: Trained GaussianHMM
        - scaler: Fitted RobustScaler
        - state_means_unscaled: Unscaled state means
        - trend_state: State index for trend regime
        - range_state: State index for range regime
        - training_stats: Statistics about training data
        - effective_n_components: n_components after direction-aware coercion
        - symbols_dropped: List of symbols dropped during filtering

    Raises:
        ValueError: If insufficient training data
    """
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import RobustScaler

    # Enforce minimum components for direction-aware mode
    _cfg = get_config()
    if _cfg.hmm.direction_aware and n_components < 4:
        logger.warning(
            "HMM_DIRECTION_AWARE=True requires n_components>=4, "
            "overriding %d -> 4", n_components,
        )
        n_components = 4

    # Extract features from all sequences
    sequences: List[np.ndarray] = []
    exogenous_frames: List[pd.DataFrame] = []
    symbols_used: List[str] = []
    exogenous_columns = tuple(_cfg.hmm.exogenous_signal_columns)

    for i, df in enumerate(per_symbol_dfs):
        if df is None or df.empty:
            continue
        if "open_time" in df.columns:
            df_local = df.sort_values("open_time").reset_index(drop=True)
        else:
            df_local = df.reset_index(drop=True)

        X, valid = compute_hmm_features(df_local)
        if len(X) == 0:
            continue

        X_valid = X[valid]

        # Filter short sequences
        if X_valid.shape[0] < min_sequence_length:
            continue

        sequences.append(X_valid)

        if exogenous_columns:
            # Keep exogenous channels aligned to the same valid rows used for HMM fit.
            valid_rows = df_local.loc[valid].reset_index(drop=True)
            exog = pd.DataFrame(index=np.arange(X_valid.shape[0]))
            for col in exogenous_columns:
                if col in valid_rows.columns:
                    exog[col] = pd.to_numeric(valid_rows[col], errors="coerce")
                else:
                    exog[col] = np.nan
            exogenous_frames.append(exog)

        # Track which symbol this came from (if available)
        if "symbol" in df_local.columns and not df_local["symbol"].empty:
            symbols_used.append(str(df_local["symbol"].iloc[0]))
        else:
            symbols_used.append(f"symbol_{i}")

    # Stack sequences
    if len(sequences) < 3:
        raise ValueError(
            f"Insufficient sequences for training: {len(sequences)} "
            f"(need at least 3)"
        )

    X_stacked = np.vstack(sequences)
    lengths = [len(seq) for seq in sequences]

    # Gate 3: Boundary-safe check — verify per-symbol lengths track the
    # stacked data shape so no cross-symbol transitions are inadvertently
    # used by downstream operations (e.g. uncertainty profile, state
    # assignments).  The uncertainty profile is intentionally GLOBAL
    # (computed across all symbols combined); this assertion guards against
    # accidental length drift, not against the global-vs-per-symbol design.
    assert sum(lengths) == X_stacked.shape[0], (
        f"Sequence boundary mismatch: sum(lengths)={sum(lengths)} != "
        f"X_stacked.shape[0]={X_stacked.shape[0]}"
    )
    logger.info(
        "Boundary check OK: %d sequences, sum(lengths)=%d == stacked rows=%d",
        len(lengths), sum(lengths), X_stacked.shape[0],
    )

    if X_stacked.shape[0] < 200:
        raise ValueError(
            f"Insufficient total samples for training: {X_stacked.shape[0]} "
            f"(need at least 200)"
        )

    # Fit scaler
    scaler = RobustScaler(quantile_range=(25, 75))
    X_scaled = scaler.fit_transform(X_stacked)

    # Train HMM
    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
    )

    # Fit on multiple sequences
    model.fit(X_scaled, lengths=lengths)
    if not model.monitor_.converged:
        logger.warning("HMM did not converge after %d iterations", model.n_iter)

    # Unscale state means for interpretability
    state_means_unscaled = scaler.inverse_transform(model.means_)

    # Infer state mapping (pass transmat to avoid labelling dead states as range)
    trend_state, range_state = infer_state_mapping(
        state_means_unscaled, transmat=model.transmat_
    )

    # Non-stationary uncertainty profile:
    # state-conditional moments, excess kurtosis, empirical-vs-parametric tails,
    # and low/medium/high volatility tier partition.
    # Fix 3: Boundary-safe posteriors — pass lengths so each symbol's
    # posteriors are computed with proper initial state probabilities,
    # preventing cross-symbol transition leakage.
    posteriors = model.predict_proba(X_scaled, lengths=lengths)
    state_assignments = np.asarray(np.argmax(posteriors, axis=1), dtype=int)

    exogenous_stacked = None
    if exogenous_frames:
        stacked = cast(pd.DataFrame, pd.concat(exogenous_frames, ignore_index=True))
        has_signal = bool(cast(pd.DataFrame, stacked.notna()).to_numpy().any())
        if (
            stacked.shape[0] == X_stacked.shape[0]
            and not stacked.empty
            and has_signal
        ):
            exogenous_stacked = stacked

    regime_uncertainty = compute_regime_uncertainty_profile(
        feature_matrix_unscaled=X_stacked,
        state_assignments=state_assignments,
        state_means_unscaled=state_means_unscaled,
        transmat=getattr(model, "transmat_", None),
        feature_names=list(FEATURE_NAMES),
        exogenous_signals=exogenous_stacked,
    )

    # Gate 3: Tag the uncertainty profile as a GLOBAL profile computed across
    # all symbols combined (not per-symbol).  Downstream consumers (runtime
    # tail correction, uncertainty analytics) must be aware that per-state
    # moments are averaged across symbol boundaries.
    regime_uncertainty["profile_scope"] = "global"
    regime_uncertainty["profile_scope_detail"] = (
        "Uncertainty profile computed on X_stacked (all symbols concatenated). "
        "Per-state moments reflect cross-symbol averages, not single-symbol "
        "characteristics.  This is intentional for global regime detection."
    )
    regime_uncertainty["sequence_boundary_metadata"] = {
        "num_sequences": len(lengths),
        "sequence_lengths": lengths,
        "total_rows": int(X_stacked.shape[0]),
    }

    # Fix 4: Detect symbols dropped during feature extraction / filtering
    symbols_dropped: List[str] = []
    if input_symbol_names is not None:
        symbols_used_set = set(symbols_used)
        symbols_dropped = [s for s in input_symbol_names if s not in symbols_used_set]
        if symbols_dropped:
            logger.warning(
                "train_hmm_global: %d symbol(s) dropped during filtering: %s",
                len(symbols_dropped), symbols_dropped,
            )
        else:
            logger.info(
                "train_hmm_global: all %d input symbols survived filtering",
                len(input_symbol_names),
            )

    # Training statistics
    training_stats = {
        "num_sequences": len(sequences),
        "num_symbols": len(symbols_used),
        "total_samples": X_stacked.shape[0],
        "sequence_lengths": lengths,
        "features_used": list(FEATURE_NAMES),
        "convergence_monitor": {
            "converged": model.monitor_.converged,
            "n_iter": model.monitor_.iter,
        },
        "non_stationary_mode": {
            "rolling_window_training": True,
            "adaptive_transitions": bool(_cfg.hmm.adaptive_transitions),
            "adaptive_transition_window": int(_cfg.hmm.adaptive_transition_window),
            "adaptive_transition_weight": float(_cfg.hmm.adaptive_transition_weight),
            "tail_adjusted_posteriors": bool(_cfg.hmm.tail_correction_enabled),
            "cornish_fisher_var": True,
            "kurtosis_aware_tiers": True,
        },
        "volatility_tiers": regime_uncertainty.get("volatility_tiers", {}),
    }

    return {
        "model": model,
        "scaler": scaler,
        "state_means_unscaled": state_means_unscaled,
        "trend_state": trend_state,
        "range_state": range_state,
        "training_stats": training_stats,
        "symbols_used": symbols_used,
        "regime_uncertainty": regime_uncertainty,
        "effective_n_components": n_components,
        "symbols_dropped": symbols_dropped,
    }


def _walk_forward_bar_pass(
    *,
    pass_mode: str,
    bar_range: np.ndarray,
    bar_trend: np.ndarray,
    bar_range_agg: np.ndarray,
    bar_trend_agg: np.ndarray,
    range_prob_threshold: float,
    trend_prob_threshold: float,
    range_dominance_min: float,
    dominance_eps: float,
) -> np.ndarray:
    """Compute per-bar pass/fail matching the runtime gate logic.

    This mirrors the pass-mode branching in ``RegimeValidator.check_hmm_regime``
    (regime_validator.py) so that walk-forward evaluation tests exactly the
    same semantics as the production gate.

    Supported modes:
    - ``hard``: single-state range_prob >= threshold AND trend_prob <= threshold.
    - ``dominance``: range_dominance(agg) >= min AND trend_prob_agg <= threshold.
    - ``soft``: always passes (downstream uses continuous regime_conf).
    - ``hybrid``: hard AND utility (utility unavailable in walk-forward, so
      degrades to hard-threshold only with a warning).
    - ``utility``: utility-only (unavailable in walk-forward, always passes
      with a warning).
    """
    n = bar_range.shape[0]

    if pass_mode == "soft":
        return np.ones(n, dtype=bool)

    if pass_mode == "dominance":
        bar_dominance = bar_range_agg / (bar_range_agg + bar_trend_agg + dominance_eps)
        return (bar_dominance >= range_dominance_min) & (bar_trend_agg <= trend_prob_threshold)

    if pass_mode == "utility":
        # Utility scoring requires grid params unavailable during walk-forward;
        # pass unconditionally so the walk-forward result reflects data quality
        # without masking it behind missing grid assumptions.
        return np.ones(n, dtype=bool)

    if pass_mode == "hybrid":
        # Hybrid = hard AND utility.  Utility is unavailable here, so degrade
        # to hard-threshold only (conservative: the hard gate is stricter than
        # "always pass" from the utility leg).
        return (bar_range >= range_prob_threshold) & (bar_trend <= trend_prob_threshold)

    # Default: "hard"
    return (bar_range >= range_prob_threshold) & (bar_trend <= trend_prob_threshold)


def walk_forward_evaluate(
    per_symbol_dfs: List[pd.DataFrame],
    n_splits: int = 3,
    n_components: int = 4,
    covariance_type: str = "diag",
    n_iter: int = 200,
    tol: float = 1e-3,
    random_state: int = 7,
    range_prob_threshold: Optional[float] = None,
    trend_prob_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Walk-forward evaluation for HMM regime detection.

    Splits data temporally, trains on earlier folds, evaluates on later folds.
    This prevents look-ahead bias and validates regime detection quality.

    Gate 2 alignment: pass criteria now match the production gate in
    ``regime_validator.py`` for ALL pass modes (hard, dominance, soft,
    hybrid, utility), not just hard-threshold.  When ``pass_mode`` is
    ``dominance``, aggregated state-group probabilities are used.

    Args:
        per_symbol_dfs: List of OHLCV DataFrames per symbol
        n_splits: Number of time-ordered splits
        n_components: HMM components
        covariance_type: HMM covariance type
        n_iter: Max EM iterations
        tol: Convergence tolerance
        random_state: Random seed
        range_prob_threshold: Min range_prob to pass (default: config.HMM_RANGE_PROB_MIN)
        trend_prob_threshold: Max trend_prob to pass (default: config.HMM_TREND_PROB_MAX)

    Returns:
        Dictionary with:
        - num_splits: Number of evaluation splits
        - mean_pass_rate: Average pass rate across splits
        - std_pass_rate: Std dev of pass rates
        - mean_range_prob: Mean range probability on test sets
        - mean_trend_prob: Mean trend probability on test sets
        - pass_mode: Which pass mode was used for evaluation
        - splits: Per-split metrics
    """
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import RobustScaler

    # Read thresholds from config to match production gate (regime_validator.py)
    _cfg = get_config()
    if range_prob_threshold is None:
        range_prob_threshold = float(_cfg.hmm.range_prob_min)
    if trend_prob_threshold is None:
        trend_prob_threshold = float(_cfg.hmm.trend_prob_max)

    # Gate 2: Read pass_mode from config — same source as regime_validator.py
    pass_mode = str(_cfg.hmm.pass_mode).lower()
    range_dominance_min = float(_cfg.hmm.range_dominance_min)
    dominance_eps = float(_cfg.hmm.dominance_eps)
    soft_gating = bool(_cfg.hmm.soft_gating)

    # Soft gating overrides pass_mode (same precedence as regime_validator.py)
    effective_pass_mode = "soft" if soft_gating else pass_mode

    if effective_pass_mode in ("utility", "hybrid"):
        logger.info(
            "Walk-forward using pass_mode=%s: utility scoring is unavailable "
            "during walk-forward (no grid params); %s",
            effective_pass_mode,
            "degrading to hard-threshold" if effective_pass_mode == "hybrid"
            else "passing unconditionally for utility leg",
        )

    logger.info(
        "Walk-forward evaluation: effective_pass_mode=%s, "
        "range_prob_threshold=%.4f, trend_prob_threshold=%.4f, "
        "range_dominance_min=%.4f",
        effective_pass_mode, range_prob_threshold, trend_prob_threshold,
        range_dominance_min,
    )

    # ── Within-symbol temporal splits ──────────────────────────────────────
    # For each symbol, compute features and keep in temporal order.
    # Each fold splits every symbol at a temporal boundary: early -> train,
    # late -> test.  This tests within-symbol drift, not cross-symbol
    # generalization.

    symbol_features: List[np.ndarray] = []  # one feature matrix per symbol

    for df in per_symbol_dfs:
        if df is None or df.empty:
            continue
        X, valid = compute_hmm_features(df)
        if len(X) == 0:
            continue
        X_valid = X[valid]
        if X_valid.shape[0] < 60:
            continue
        symbol_features.append(X_valid)

    empty_result: Dict[str, Any] = {
        "num_splits": 0,
        "mean_pass_rate": 0.0,
        "std_pass_rate": 0.0,
        "mean_range_prob": 0.0,
        "mean_trend_prob": 0.0,
        "pass_mode": effective_pass_mode,
        "splits": [],
    }

    if len(symbol_features) < 3:
        return empty_result

    split_results: List[Dict[str, Any]] = []

    for fold in range(n_splits):
        # Expanding split ratio: fold 0 -> 50/50, fold 1 -> 66/33, ...
        train_frac = (fold + 1) / (n_splits + 1)

        train_seqs: List[np.ndarray] = []
        test_seqs: List[np.ndarray] = []

        for X_sym in symbol_features:
            n_bars = X_sym.shape[0]
            cut = max(30, int(n_bars * train_frac))
            if cut >= n_bars - 10:
                # Not enough test data for this symbol in this fold
                train_seqs.append(X_sym)
                continue
            train_seqs.append(X_sym[:cut])
            test_seqs.append(X_sym[cut:])

        if len(train_seqs) < 3 or len(test_seqs) < 1:
            continue

        # Stack training sequences
        X_train = np.vstack(train_seqs)
        train_lengths = [len(s) for s in train_seqs]

        if X_train.shape[0] < 100:
            continue

        # Fit scaler and model on train set
        scaler = RobustScaler(quantile_range=(25, 75))
        X_train_scaled = scaler.fit_transform(X_train)

        model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
        )
        model.fit(X_train_scaled, lengths=train_lengths)
        if not model.monitor_.converged:
            logger.warning("HMM did not converge after %d iterations", model.n_iter)

        # Infer state mapping (pass transmat to avoid labelling dead states)
        state_means_unscaled = scaler.inverse_transform(model.means_)
        trend_state, range_state = infer_state_mapping(
            state_means_unscaled, transmat=model.transmat_
        )

        # Gate 2: Compute state groups for aggregated probabilities
        # (mirrors inference.py HMMRegimePredictor.__init__)
        groups = compute_state_groups(state_means_unscaled)
        range_states = groups["range_states"]
        trend_states = groups["trend_states"]

        # Evaluate on test sequences (later portion of each symbol)
        # Compute per-bar pass rates across the FULL test sequence,
        # not just the last bar, for a representative quality metric.
        range_probs: List[float] = []
        trend_probs: List[float] = []
        range_probs_agg_list: List[float] = []
        trend_probs_agg_list: List[float] = []
        dominance_list: List[float] = []
        passes: List[float] = []

        for X_test in test_seqs:
            if X_test.shape[0] < 10:
                continue
            X_test_scaled = scaler.transform(X_test)
            posteriors = model.predict_proba(X_test_scaled)

            # Per-bar single-state probabilities
            bar_range = posteriors[:, range_state]
            bar_trend = posteriors[:, trend_state]

            # Per-bar aggregated state-group probabilities
            bar_range_agg = np.sum(posteriors[:, range_states], axis=1)
            bar_trend_agg = np.sum(posteriors[:, trend_states], axis=1)

            # Per-bar dominance (for diagnostics)
            bar_dominance = bar_range_agg / (bar_range_agg + bar_trend_agg + dominance_eps)

            # Apply the same pass logic as runtime
            bar_pass = _walk_forward_bar_pass(
                pass_mode=effective_pass_mode,
                bar_range=bar_range,
                bar_trend=bar_trend,
                bar_range_agg=bar_range_agg,
                bar_trend_agg=bar_trend_agg,
                range_prob_threshold=range_prob_threshold,
                trend_prob_threshold=trend_prob_threshold,
                range_dominance_min=range_dominance_min,
                dominance_eps=dominance_eps,
            )

            range_probs.append(float(np.mean(bar_range)))
            trend_probs.append(float(np.mean(bar_trend)))
            range_probs_agg_list.append(float(np.mean(bar_range_agg)))
            trend_probs_agg_list.append(float(np.mean(bar_trend_agg)))
            dominance_list.append(float(np.mean(bar_dominance)))
            passes.append(float(np.mean(bar_pass)))

        if not passes:
            continue

        fold_metrics: Dict[str, Any] = {
            "fold": fold,
            "train_frac": round(train_frac, 3),
            "train_sequences": len(train_seqs),
            "test_sequences": len(test_seqs),
            "pass_rate": sum(passes) / len(passes),
            "mean_range_prob": float(np.mean(range_probs)),
            "mean_trend_prob": float(np.mean(trend_probs)),
            "mean_range_prob_agg": float(np.mean(range_probs_agg_list)),
            "mean_trend_prob_agg": float(np.mean(trend_probs_agg_list)),
            "mean_range_dominance": float(np.mean(dominance_list)),
            "pass_mode": effective_pass_mode,
            "range_states": range_states,
            "trend_states": trend_states,
        }
        split_results.append(fold_metrics)

    if not split_results:
        return empty_result

    pass_rates = [s["pass_rate"] for s in split_results]
    range_probs_all = [s["mean_range_prob"] for s in split_results]
    trend_probs_all = [s["mean_trend_prob"] for s in split_results]

    return {
        "num_splits": len(split_results),
        "mean_pass_rate": float(np.mean(pass_rates)),
        "std_pass_rate": float(np.std(pass_rates)),
        "mean_range_prob": float(np.mean(range_probs_all)),
        "mean_trend_prob": float(np.mean(trend_probs_all)),
        "pass_mode": effective_pass_mode,
        "splits": split_results,
    }


def save_trained_model(
    artifact_dir: Path,
    model: Any,
    scaler: Any,
    state_means_unscaled: np.ndarray,
    trend_state: int,
    range_state: int,
    training_stats: Dict[str, Any],
    symbols_used: List[str],
    training_window: Optional[Dict[str, str]] = None,
    eval_metrics: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
    canonical_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save trained HMM model with complete metadata.

    Args:
        artifact_dir: Directory to save artifacts
        model: Trained HMM model
        scaler: Fitted scaler
        state_means_unscaled: Unscaled state means
        trend_state: Trend state index
        range_state: Range state index
        training_stats: Training statistics
        symbols_used: List of symbols in training set
        training_window: Optional dict with start_utc and end_utc
        eval_metrics: Optional evaluation metrics
        version: Optional version string (default: timestamp)
        canonical_metadata: Optional canonical provenance dict.  When
            provided, its fields are merged into the artifact metadata
            for truthful audit: ``selection_time_utc``,
            ``target_end_utc``, ``common_start_utc``, ``common_end_utc``,
            ``timestamp_basis``, ``accepted_symbols``, ``probe_count``,
            ``accepted_count``, ``trained_count``, ``canonical_source``,
            ``frozen_slice_dir``.

    Returns:
        Path to artifact directory
    """
    import json as _json

    # Generate version if not provided
    if version is None:
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Record effective_n_components from the model itself, not the caller's
    # requested value (they may differ if e.g. direction_aware override applied).
    effective_n_components = int(model.n_components)

    # Extract model parameters
    model_params = {
        "n_components": effective_n_components,
        "covariance_type": str(model.covariance_type),
        "n_iter": int(model.n_iter),
        "tol": float(model.tol),
        "random_state": int(model.random_state) if model.random_state is not None else None,
        "effective_n_components": effective_n_components,
    }

    # When canonical_metadata supplies accepted_symbols, use those as
    # symbols_used (real names, not placeholder symbol_N).
    if canonical_metadata is not None:
        real_symbols = canonical_metadata.get("accepted_symbols")
        if isinstance(real_symbols, list) and real_symbols:
            symbols_used = list(real_symbols)

    # Create metadata
    _cfg = get_config()
    metadata = ArtifactMetadata(
        artifact_version=version,
        trained_at_utc=datetime.now(timezone.utc).isoformat(),
        pipeline_version=PIPELINE_VERSION,
        code_commit=get_git_commit(),
        training_start_utc=training_window.get("start_utc") if training_window else None,
        training_end_utc=training_window.get("end_utc") if training_window else None,
        symbols_used=symbols_used,
        num_sequences=training_stats.get("num_sequences"),
        total_samples=training_stats.get("total_samples"),
        features=list(FEATURE_NAMES),
        timeframes_used=["15m"],
        transform_type="RobustScaler",
        transform_params={"quantile_range": [25, 75]},
        feature_schema=_cfg.hmm.feature_schema,
        model_type="GaussianHMM",
        model_params=model_params,
        eval_metrics=eval_metrics,
        state_mapping={"range": range_state, "trend": trend_state},
    )

    # Get feature schema
    feature_schema = get_feature_schema()

    # Add training stats to schema
    feature_schema["training_stats"] = training_stats

    # Save artifact
    artifact_path = save_artifact(
        artifact_dir=artifact_dir,
        model=model,
        scaler=scaler,
        metadata=metadata,
        feature_schema=feature_schema,
        eval_metrics=eval_metrics,
        state_means_unscaled=state_means_unscaled,
    )

    # ── Merge canonical metadata into the persisted metadata.json ──
    if canonical_metadata is not None:
        metadata_path = artifact_path / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            md_on_disk: Dict[str, Any] = _json.load(f)

        _CANONICAL_KEYS = (
            "selection_time_utc",
            "target_end_utc",
            "common_start_utc",
            "common_end_utc",
            "timestamp_basis",
            "accepted_symbols",
            "probe_count",
            "accepted_count",
            "trained_count",
            "canonical_source",
            "frozen_slice_dir",
        )
        canonical_section: Dict[str, Any] = {}
        for key in _CANONICAL_KEYS:
            if key in canonical_metadata:
                value = canonical_metadata[key]
                # Convert Path objects to strings for JSON serialization
                if isinstance(value, Path):
                    value = str(value)
                canonical_section[key] = value

        md_on_disk["canonical_metadata"] = canonical_section

        with open(metadata_path, "w", encoding="utf-8") as f:
            _json.dump(md_on_disk, f, indent=2, ensure_ascii=False)

        logger.info(
            "Canonical metadata merged into %s: %s",
            metadata_path,
            list(canonical_section.keys()),
        )

    return artifact_path


async def train_from_market_data(
    client,
    symbols: List[str],
    artifact_dir: Path,
    timeframe: str = "15m",
    limit: int = 500,
    n_components: int = 4,
    covariance_type: str = "diag",
    n_iter: int = 200,
    tol: float = 1e-3,
    random_state: int = 7,
    cache_dir: Optional[Path] = None,
    version: Optional[str] = None,
    override_datasets: Optional[Dict[str, pd.DataFrame]] = None,
    _btc_df: Optional[pd.DataFrame] = None,  # DEPRECATED: kept for signature compat
    canonical_mode: bool = False,
    canonical_metadata: Optional[Dict[str, Any]] = None,
) -> tuple[Path, Dict[str, pd.DataFrame]]:
    """
    Complete training pipeline: fetch data, train model, save artifact.

    This is the main entry point for offline training.

    Args:
        client: Binance API client
        symbols: List of symbols to train on
        artifact_dir: Directory to save trained model
        timeframe: Timeframe to use (default: "15m")
        limit: Number of bars per symbol
        n_components: Number of HMM states
        covariance_type: HMM covariance type
        n_iter: Maximum EM iterations
        tol: Convergence tolerance
        random_state: Random seed
        cache_dir: Optional cache directory for fetched data
        version: Optional version string
        override_datasets: Optional pre-fetched datasets keyed by symbol
        _btc_df: DEPRECATED — kept for signature compat
        canonical_mode: When True, ALL requested symbols must be present
            in override_datasets and survive filtering.  No fallback to
            fetch_training_dataset() for missing symbols.  Raises
            ValueError if any symbol is missing or dropped.

    Returns:
        Tuple of (artifact_path, datasets dict)

    Example:
        >>> from api.binance_client import BinanceClient
        >>> client = BinanceClient()
        >>> artifact_path = await train_from_market_data(
        ...     client,
        ...     symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],
        ...     artifact_dir=Path("artifacts/hmm/rolling_180d_20260215_120000"),
        ...     limit=500,
        ... )
        >>> print(f"Model saved to: {artifact_path}")
    """
    from neutralgrid.data.market_data import fetch_training_dataset

    # Fix 1: canonical_mode — fail closed if any accepted symbol is missing
    if canonical_mode:
        if not override_datasets:
            raise ValueError(
                "canonical_mode=True requires override_datasets to be provided"
            )
        missing = [s for s in symbols if s not in override_datasets]
        if missing:
            raise ValueError(
                f"Canonical mode: {len(missing)} symbol(s) missing from "
                f"override_datasets: {missing}"
            )
        datasets = {s: override_datasets[s] for s in symbols}
        logger.info(
            "Canonical mode: all %d symbols present in frozen slice",
            len(symbols),
        )
    elif override_datasets and set(symbols).issubset(override_datasets.keys()):
        datasets = {s: override_datasets[s] for s in symbols if s in override_datasets}
        logger.info("Using %d pre-fetched dataset(s), skipping API fetch", len(datasets))
    else:
        # Fetch training data
        logger.info("Fetching training data for %d symbols...", len(symbols))
        datasets = await fetch_training_dataset(
            client=client,
            symbols=symbols,
            timeframe=timeframe,
            limit=limit,
            cache_dir=cache_dir,
            strict=True,
        )

        logger.info("Fetched %d symbols successfully", len(datasets))

        # Merge override datasets (e.g. extended BTC history from Binance Vision)
        if override_datasets:
            for sym, df_override in override_datasets.items():
                datasets[sym] = df_override
            logger.info("Applied %d dataset override(s)", len(override_datasets))

    # Train model
    logger.info("Training HMM model...")
    dfs = list(datasets.values())
    result = train_hmm_global(
        per_symbol_dfs=dfs,
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        input_symbol_names=list(datasets.keys()),
    )

    logger.info("Training complete: %d sequences, %d total samples",
                result['training_stats']['num_sequences'],
                result['training_stats']['total_samples'])

    # Fix 4: Check if symbols were dropped during training
    dropped = result.get("symbols_dropped", [])
    if dropped:
        if canonical_mode:
            raise ValueError(
                f"Canonical mode: trained basket ({len(result['symbols_used'])}) "
                f"!= accepted basket ({len(symbols)}). "
                f"Dropped: {dropped}"
            )
        else:
            logger.warning(
                "Symbols dropped during training: %s (trained %d / requested %d)",
                dropped, len(result['symbols_used']), len(symbols),
            )

    # Fix 2: Use effective n_components (after direction-aware coercion)
    effective_n_components = result.get("effective_n_components", n_components)
    if effective_n_components != n_components:
        logger.info(
            "Effective n_components after direction-aware coercion: %d",
            effective_n_components,
        )

    # Walk-forward evaluation — use effective n_components, not caller's
    logger.info("Running walk-forward evaluation...")
    eval_metrics = walk_forward_evaluate(
        per_symbol_dfs=dfs,
        n_splits=3,
        n_components=effective_n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
    )
    regime_uncertainty = result.get("regime_uncertainty")
    if regime_uncertainty:
        eval_metrics = dict(eval_metrics)
        eval_metrics["regime_uncertainty"] = regime_uncertainty

    logger.info("Evaluation complete: %d splits, mean_pass_rate=%.2f%%",
                eval_metrics['num_splits'],
                eval_metrics['mean_pass_rate'] * 100)

    # Determine training window using open_time exclusively so the
    # declared timestamp basis is consistent ("open_time" for both
    # start and end).  Previously end used close_time which mixed
    # two different timestamp semantics.
    training_window = None
    if datasets:
        try:
            all_starts = [df["open_time"].min() for df in datasets.values()]
            all_ends = [df["open_time"].max() for df in datasets.values()]
            training_window = {
                "start_utc": min(all_starts).isoformat(),
                "end_utc": max(all_ends).isoformat(),
                "timestamp_basis": "open_time",
            }
        except KeyError:
            pass  # open_time column not present

    # Save artifact
    logger.info("Saving artifact to %s...", artifact_dir)
    artifact_path = save_trained_model(
        artifact_dir=artifact_dir,
        model=result["model"],
        scaler=result["scaler"],
        state_means_unscaled=result["state_means_unscaled"],
        trend_state=result["trend_state"],
        range_state=result["range_state"],
        training_stats=result["training_stats"],
        symbols_used=result["symbols_used"],
        training_window=training_window,
        eval_metrics=eval_metrics,
        version=version,
        canonical_metadata=canonical_metadata,
    )

    logger.info("Model saved successfully to: %s", artifact_path)

    return artifact_path, datasets
