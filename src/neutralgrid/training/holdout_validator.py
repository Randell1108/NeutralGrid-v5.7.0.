"""
Holdout Validator for AFML Station 4 Compliance.

Validates that holdout data was never used during model development.

AFML Station 4 requires proving the holdout is "true" (never examined).
This class logs all data access to detect holdout contamination.

Design principles:
- Log holdout creation with timestamp
- Log all holdout access with purpose
- Validate integrity (no premature access)
- Simple JSON persistence

Reference: AFML Station 4 (Backtesting through Cross-Validation)
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any
import json
import logging

logger = logging.getLogger(__name__)


class HoldoutValidator:
    """
    Validate that holdout data was never used during development.

    AFML Station 4 requires proving the holdout is "true" (never examined).
    This class logs all data access to detect holdout contamination.

    Usage:
        validator = HoldoutValidator()

        # When creating holdout split
        validator.mark_holdout_created(
            cv_samples=80,
            holdout_samples=20,
            cv_end_date='2025-12-31',
            holdout_start_date='2026-01-01'
        )

        # Before final evaluation (only!)
        validator.mark_holdout_accessed(
            purpose='final_evaluation',
            caller='evaluate_final_model.py'
        )

        # Validate integrity
        result = validator.validate_integrity()
        if result['contaminated']:
            raise ValueError("Holdout has been contaminated!")
    """

    def __init__(self, log_file: str = "data/holdout_access_log.json", strict: bool = True):
        """
        Initialize HoldoutValidator.

        Args:
            log_file: Path to access log file
            strict: If True, raise on non-final_evaluation access (default True)
        """
        self.log_file = Path(log_file)
        self.strict = strict
        self._access_log: List[Dict] = []
        self._load()

    def _load(self):
        """Load existing access log from disk."""
        if not self.log_file.exists():
            logger.info(f"Holdout access log not found, creating new: {self.log_file}")
            self._access_log = []
            return

        try:
            with open(self.log_file, 'r') as f:
                self._access_log = json.load(f)

            logger.info(f"Loaded {len(self._access_log)} holdout access events")

        except Exception as e:
            logger.error(f"Failed to load holdout access log: {e}")
            logger.warning("Starting with empty access log")
            self._access_log = []

    def _save(self):
        """Save access log to disk."""
        try:
            # Ensure directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file first (atomic write)
            temp_file = self.log_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self._access_log, f, indent=2)

            # Atomic rename
            temp_file.replace(self.log_file)

            logger.debug(f"Saved {len(self._access_log)} events to {self.log_file}")

        except Exception as e:
            logger.error(f"Failed to save holdout access log: {e}")
            raise

    def mark_holdout_created(
        self,
        cv_samples: int,
        holdout_samples: int,
        cv_end_date: str,
        holdout_start_date: str,
        notes: str = ""
    ):
        """
        Log when holdout was first created.

        Args:
            cv_samples: Number of samples in CV dataset
            holdout_samples: Number of samples in holdout dataset
            cv_end_date: End date of CV period (ISO format)
            holdout_start_date: Start date of holdout period (ISO format)
            notes: Optional notes
        """
        entry = {
            'event': 'holdout_created',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'cv_samples': cv_samples,
            'holdout_samples': holdout_samples,
            'cv_end_date': cv_end_date,
            'holdout_start_date': holdout_start_date,
            'notes': notes
        }

        self._access_log.append(entry)
        self._save()

        logger.info(
            f"Holdout created: {holdout_samples} samples "
            f"(CV: {cv_samples}, holdout: {holdout_start_date}+)"
        )

    def mark_holdout_accessed(
        self,
        purpose: str,
        caller: str,
        notes: str = ""
    ):
        """
        Log when holdout data was accessed.

        Args:
            purpose: Purpose of access ('final_evaluation', 'model_selection', etc.)
            caller: Script/function that accessed holdout
            notes: Optional notes
        """
        entry = {
            'event': 'holdout_accessed',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'purpose': purpose,
            'caller': caller,
            'notes': notes
        }

        self._access_log.append(entry)
        self._save()

        if purpose != 'final_evaluation':
            msg = (
                f"Holdout accessed for '{purpose}' by {caller}. "
                "This may contaminate the holdout!"
            )
            if self.strict:
                raise RuntimeError(msg)
            logger.warning(msg)
        else:
            logger.info(f"Holdout accessed for final evaluation by {caller}")

    def validate_integrity(self) -> Dict[str, Any]:
        """
        Check if holdout was accessed before final evaluation.

        Returns:
            Dict with validation results:
            - valid: bool (True if no contamination)
            - contaminated: bool (True if holdout was accessed prematurely)
            - created_at: str (timestamp of creation)
            - access_count: int (number of access events)
            - premature_access_count: int (number of non-final accesses)
            - details: List[Dict] (details of premature access)
        """
        created = [e for e in self._access_log if e['event'] == 'holdout_created']
        accessed = [e for e in self._access_log if e['event'] == 'holdout_accessed']

        if not created:
            return {
                'valid': False,
                'reason': 'No holdout creation logged',
                'contaminated': False,
                'created_at': None,
                'access_count': 0,
                'premature_access_count': 0,
                'details': []
            }

        # Check if accessed before final evaluation
        premature_access = [
            a for a in accessed
            if a['purpose'] != 'final_evaluation'
        ]

        return {
            'valid': len(premature_access) == 0,
            'contaminated': len(premature_access) > 0,
            'created_at': created[-1]['timestamp'],
            'access_count': len(accessed),
            'premature_access_count': len(premature_access),
            'details': premature_access,
            'holdout_info': created[-1]
        }

    def get_access_history(self) -> List[Dict]:
        """
        Get full access history.

        Returns:
            List of access events (sorted by timestamp)
        """
        return sorted(self._access_log, key=lambda e: e['timestamp'])

    def summary(self) -> str:
        """
        Get human-readable summary of holdout status.

        Returns:
            Summary string
        """
        result = self.validate_integrity()

        if not result['valid']:
            if result['contaminated']:
                summary = f"❌ CONTAMINATED: {result['premature_access_count']} premature access(es)\n"
                for detail in result['details']:
                    summary += f"  - {detail['timestamp']}: {detail['purpose']} by {detail['caller']}\n"
            else:
                summary = f"❌ INVALID: {result['reason']}\n"
        else:
            summary = f"✅ VALID: Holdout integrity preserved\n"
            summary += f"  Created: {result['created_at']}\n"
            summary += f"  Access count: {result['access_count']}\n"

        return summary


# Convenience function
def get_global_validator() -> HoldoutValidator:
    """
    Get global holdout validator instance.

    Uses default location: data/holdout_access_log.json

    Returns:
        Global HoldoutValidator instance
    """
    return HoldoutValidator()


# Helper function for integration
def validate_holdout_or_raise():
    """
    Validate holdout integrity or raise exception.

    Raises:
        ValueError: If holdout is contaminated
    """
    validator = get_global_validator()
    result = validator.validate_integrity()

    if not result['valid']:
        if result['contaminated']:
            raise ValueError(
                f"Holdout has been contaminated! "
                f"{result['premature_access_count']} premature access(es) detected. "
                f"See {validator.log_file} for details."
            )
        raise ValueError(
            f"Holdout integrity check failed: {result['reason']}. "
            f"See {validator.log_file} for details."
        )

    logger.info("✅ Holdout integrity validated")
