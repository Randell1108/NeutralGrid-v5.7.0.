from __future__ import annotations

import warnings

import pandas as pd

from neutralgrid.training.data_generator import ExistingDataMapper


def test_map_dataframe_truncates_end_time_to_python_precision_without_warning() -> None:
    source = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "start_time_utc": ["2026-01-01T00:00:00Z"],
            "end_time_utc": ["2026-01-01T01:00:00.123456789Z"],
            "pnl_pct": [0.25],
            "grids_count": [10],
            "grid_spacing_pct": [0.5],
            "adx_1h": [20.0],
            "adx_15m": [21.0],
            "rsi_15m": [50.0],
            "range_size_pct": [5.0],
        }
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Discarding nonzero nanoseconds in conversion.*",
            category=UserWarning,
        )
        mapped = ExistingDataMapper().map_dataframe(source)

    assert mapped.loc[0, "t1"] == pd.Timestamp("2026-01-01T01:00:00.123456Z")
