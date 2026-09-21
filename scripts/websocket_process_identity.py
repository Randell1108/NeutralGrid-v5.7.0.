"""Compatibility imports for the shared read-only process identity utility."""

from __future__ import annotations

from neutralgrid.core.process_identity import (
    ProcessObservation,
    matches_identity,
    query_process,
)

__all__ = ["ProcessObservation", "matches_identity", "query_process"]
