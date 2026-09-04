"""Core protocol types for duck-typed interfaces.

Protocols define structural sub-typing contracts so that Pylance / mypy can
verify call-sites without importing concrete implementations.  Every protocol
here is ``@runtime_checkable`` so that ``isinstance()`` guards work at runtime
as well.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class RegimePredictor(Protocol):
    """Protocol for HMM regime prediction artifacts.

    Any object whose ``predict`` method matches this signature satisfies the
    protocol.  The canonical implementation is
    ``neutralgrid.models.hmm.inference.HMMRegimePredictor``.
    """

    def predict(
        self,
        df: pd.DataFrame,
        infer_limit: Optional[int] = None,
    ) -> object:
        """Run regime inference on OHLCV data.

        Args:
            df: OHLCV DataFrame.
            infer_limit: Maximum number of bars to feed into the model.

        Returns:
            An inference result object that exposes a ``.to_dict()`` method.
        """
        ...
