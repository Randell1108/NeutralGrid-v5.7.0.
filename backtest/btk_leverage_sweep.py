"""
Leverage/margin sweep utility for the realistic backtester.

Runs the backtest at each leverage level while holding grid geometry fixed,
to find the optimal leverage that maximizes equity while staying above
liquidation threshold.

Depends on P0-A (liquidation modeling) being present in the engine.

Usage::

    from backtest.btk_leverage_sweep import sweep_leverage, find_optimal_leverage
    from backtest.backtest_realistic import GridConfig

    cfg = GridConfig(symbol="BTCUSDT", lower=40000, upper=42000, num_grids=20)
    results = sweep_leverage(cfg, klines_df, leverage_range=range(1, 26))
    optimal = find_optimal_leverage(results)
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List

import pandas as pd

from backtest.backtest_realistic import GridConfig
from backtest.btk_unified_runner import run_backtest


def sweep_leverage(
    base_config: GridConfig,
    klines_df: pd.DataFrame,
    leverage_range: range = range(1, 26),
) -> List[Dict[str, Any]]:
    """Run backtest at each leverage, return results sorted by final_equity descending.

    Parameters
    ----------
    base_config : GridConfig
        Base configuration. The ``leverage`` field will be overridden per sweep.
    klines_df : pd.DataFrame
        1-minute OHLCV data.
    leverage_range : range
        Range of leverage values to sweep (default 1..25).

    Returns
    -------
    list[dict]
        Backtest results sorted by final_equity (highest first).
        Each result includes an extra ``sweep_leverage`` key.
    """
    results: List[Dict[str, Any]] = []
    for lev in leverage_range:
        cfg = replace(base_config, leverage=lev)
        result = run_backtest(cfg, klines_df)
        result["sweep_leverage"] = lev
        results.append(result)
    return sorted(results, key=lambda r: r["final_equity"], reverse=True)


def find_optimal_leverage(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the highest-equity non-liquidated result.

    Parameters
    ----------
    results : list[dict]
        Output from ``sweep_leverage()`` (already sorted by final_equity desc).

    Returns
    -------
    dict
        The best non-liquidated result, or the lowest-leverage result if all
        were liquidated.
    """
    for r in results:
        if not r.get("liquidated", False):
            return r
    # All liquidated — return lowest leverage as safest option
    return min(results, key=lambda r: r.get("sweep_leverage", 999))
