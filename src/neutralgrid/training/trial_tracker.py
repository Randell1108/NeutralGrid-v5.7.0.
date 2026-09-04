"""
Trial Tracking System for AFML-Compliant Deflated Sharpe Calculation.

Tracks all model training trials (hyperparameters, CV scores) to enable
proper deflated Sharpe ratio computation as per AFML Chapter 14.

Without trial tracking, we cannot assess the probability of backtest overfitting.

Design principles:
- Append-only log (never delete trials)
- JSON persistence for portability
- Simple API for integration
- Defensive against corruption

Reference: AFML Chapter 14 (Backtest Statistics)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrialRecord:
    """Record of a single model training trial."""

    trial_id: str  # Unique identifier (e.g., "meta_labeler_20260204_163000")
    timestamp: datetime  # When trial was run
    model_type: str  # Type of model ("meta_labeler", "hmm", etc.)
    hyperparameters: Dict  # All hyperparameters used
    cv_score: float  # Cross-validation score (e.g., AUC)
    test_score: Optional[float] = None  # Optional test set score
    feature_set: List[str] = field(default_factory=list)  # Features used
    notes: str = ""  # Optional notes

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'TrialRecord':
        """Create from dictionary (for JSON deserialization)."""
        d = d.copy()
        d['timestamp'] = datetime.fromisoformat(d['timestamp'])
        return cls(**d)


class TrialTracker:
    """
    Track all model training trials for deflated Sharpe calculation.

    AFML explicitly warns that without reporting trials you cannot assess
    false discovery probability. This class provides the infrastructure
    to log all trials and compute deflated metrics.

    Usage:
        tracker = TrialTracker()

        # Log a trial
        trial = TrialRecord(
            trial_id='meta_labeler_001',
            timestamp=datetime.now(timezone.utc),
            model_type='meta_labeler',
            hyperparameters={'n_estimators': 100, 'max_depth': 5},
            cv_score=0.72,
            feature_set=['range_prob', 'trend_prob', ...]
        )
        tracker.log_trial(trial)

        # Get trial count for deflated Sharpe
        n_trials = tracker.get_trial_count(model_type='meta_labeler')

        # Compute deflated Sharpe
        from neutralgrid.backtest.cpcv import DeflatedSharpeCalculator
        calc = DeflatedSharpeCalculator(n_trials=n_trials)
        result = calc.deflate(observed_sharpe=1.5, n_observations=252)
    """

    def __init__(self, log_file: str = "data/trial_log.json"):
        """
        Initialize TrialTracker.

        Args:
            log_file: Path to JSON log file
        """
        self.log_file = Path(log_file)
        self._trials: List[TrialRecord] = []
        self._load()

    def _load(self):
        """Load existing trials from disk."""
        if not self.log_file.exists():
            logger.info(f"Trial log not found, creating new: {self.log_file}")
            self._trials = []
            return

        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)

            self._trials = [TrialRecord.from_dict(t) for t in data]
            logger.info(f"Loaded {len(self._trials)} trials from {self.log_file}")

        except Exception as e:
            logger.error(f"Failed to load trial log: {e}")
            logger.warning("Starting with empty trial log")
            self._trials = []

    def _save(self):
        """Save trials to disk."""
        try:
            # Ensure directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file first (atomic write)
            temp_file = self.log_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                data = [t.to_dict() for t in self._trials]
                json.dump(data, f, indent=2)

            # Atomic rename
            temp_file.replace(self.log_file)

            logger.debug(f"Saved {len(self._trials)} trials to {self.log_file}")

        except Exception as e:
            logger.error(f"Failed to save trial log: {e}")
            raise

    def log_trial(self, trial: TrialRecord):
        """
        Log a trial (append-only for correct DSR deflation trial counting).

        Args:
            trial: TrialRecord to log
        """
        for existing in self._trials:
            if existing.trial_id == trial.trial_id:
                logger.warning(
                    "Duplicate trial_id '%s' detected. Appending as new entry "
                    "for correct DSR trial counting (not replacing).",
                    trial.trial_id,
                )
                break

        self._trials.append(trial)
        self._save()
        logger.info(
            "Logged trial %s: %s cv_score=%.4f",
            trial.trial_id, trial.model_type, trial.cv_score,
        )

    def get_trial_count(
        self,
        model_type: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> int:
        """
        Get number of trials (for deflated Sharpe N parameter).

        Args:
            model_type: Filter by model type (None = all)
            since: Only count trials since this datetime

        Returns:
            Number of trials matching filters
        """
        trials = self._trials

        if model_type:
            trials = [t for t in trials if t.model_type == model_type]

        if since:
            trials = [t for t in trials if t.timestamp >= since]

        return len(trials)

    def get_cv_scores(
        self,
        model_type: str,
        since: Optional[datetime] = None
    ) -> List[float]:
        """
        Get all CV scores for a model type.

        Args:
            model_type: Model type to filter
            since: Only include trials since this datetime

        Returns:
            List of CV scores
        """
        trials = [t for t in self._trials if t.model_type == model_type]

        if since:
            trials = [t for t in trials if t.timestamp >= since]

        return [t.cv_score for t in trials]

    def get_best_trial(
        self,
        model_type: str,
        metric: str = 'cv_score'
    ) -> Optional[TrialRecord]:
        """
        Get best trial for a model type.

        Args:
            model_type: Model type to filter
            metric: Metric to maximize ('cv_score' or 'test_score')

        Returns:
            Best TrialRecord or None if no trials found
        """
        trials = [t for t in self._trials if t.model_type == model_type]

        if not trials:
            return None

        if metric == 'cv_score':
            return max(trials, key=lambda t: t.cv_score)
        elif metric == 'test_score':
            trials_with_test = [t for t in trials if t.test_score is not None]
            if not trials_with_test:
                return None
            return max(trials_with_test, key=lambda t: t.test_score if t.test_score is not None else float('-inf'))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_trials(
        self,
        model_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[TrialRecord]:
        """
        Get trials with optional filtering.

        Args:
            model_type: Filter by model type
            limit: Maximum number of trials to return (most recent first)

        Returns:
            List of TrialRecord
        """
        trials = self._trials

        if model_type:
            trials = [t for t in trials if t.model_type == model_type]

        # Sort by timestamp (most recent first)
        trials = sorted(trials, key=lambda t: t.timestamp, reverse=True)

        if limit:
            trials = trials[:limit]

        return trials

    def compute_deflated_sharpe(
        self,
        observed_sharpe: float,
        model_type: str,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0
    ) -> Dict:
        """
        Compute deflated Sharpe for observed performance.

        Uses trial count for this model type as N_trials parameter.

        Args:
            observed_sharpe: Observed Sharpe ratio
            model_type: Model type for trial filtering
            n_observations: Number of return observations
            skewness: Return skewness
            kurtosis: Return kurtosis

        Returns:
            Dict with deflated_sharpe, haircut_pct, prob_false_discovery, etc.
        """
        from neutralgrid.backtest.cpcv import DeflatedSharpeCalculator

        n_trials = self.get_trial_count(model_type=model_type)

        if n_trials == 0:
            logger.warning(
                f"No trials found for {model_type}. "
                "Deflated Sharpe will assume N=1 (no deflation)."
            )
            n_trials = 1

        calc = DeflatedSharpeCalculator(n_trials=n_trials)
        result = calc.deflate(
            raw_sharpe=observed_sharpe,
            n_observations=n_observations,
            skewness=skewness,
            kurtosis=kurtosis
        )

        return {
            'raw_sharpe': result.raw_sharpe,
            'deflated_sharpe': result.deflated_sharpe,
            'haircut_pct': result.haircut_pct,
            'n_trials': result.n_trials,
            'expected_max_sharpe': result.expected_max_sharpe,
            'prob_false_discovery': result.prob_false_discovery,
            'sharpe_ci_lower': result.sharpe_ci_lower,
            'sharpe_ci_upper': result.sharpe_ci_upper
        }

    def summary_stats(self, model_type: Optional[str] = None) -> Dict:
        """
        Get summary statistics for trials.

        Args:
            model_type: Filter by model type (None = all)

        Returns:
            Dict with summary statistics
        """
        trials = self._trials
        if model_type:
            trials = [t for t in trials if t.model_type == model_type]

        if not trials:
            return {
                'count': 0,
                'model_types': []
            }

        cv_scores = [t.cv_score for t in trials]

        return {
            'count': len(trials),
            'model_types': list(set(t.model_type for t in trials)),
            'cv_score_mean': float(sum(cv_scores) / len(cv_scores)),
            'cv_score_min': float(min(cv_scores)),
            'cv_score_max': float(max(cv_scores)),
            'first_trial': min(t.timestamp for t in trials).isoformat(),
            'last_trial': max(t.timestamp for t in trials).isoformat()
        }


# Convenience function
def get_global_tracker() -> TrialTracker:
    """
    Get global trial tracker instance.

    Uses default location: data/trial_log.json

    Returns:
        Global TrialTracker instance
    """
    return TrialTracker()
