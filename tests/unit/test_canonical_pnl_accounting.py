from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neutralgrid.training.data_generator import ExistingDataMapper


@pytest.mark.parametrize('commission', [-0.2, 0.2, 0.0, np.nan])
def test_canonical_reported_final_pnl_is_not_charged_a_second_fee(commission):
    frame = pd.DataFrame([{'symbol':'BTCUSDT', 'strategy_id':'123',
        'start_time_utc':'2026-10-08T12:00:00Z', 'end_time_utc':'2026-10-08T13:00:00Z',
        'pnl_pct':3.1, 'total_profit_usdt':0.31, 'commission_usdt':commission,
        'invested_margin_usdt':10.0, 'grid_spacing_pct':0.5}])
    result = ExistingDataMapper().map_dataframe(frame)
    assert result.iloc[0].net_pnl_pct == 3.1


def test_explicit_net_pnl_is_preserved_without_recomputing_rounded_cash_fields():
    frame = pd.DataFrame([{'symbol':'BTCUSDT','start_time_utc':'2026-10-08T12:00:00Z',
        'pnl_pct':3.1,'net_pnl_pct':2.9,'total_profit_usdt':0.31,
        'commission_usdt':-0.2,'invested_margin_usdt':10,'grid_spacing_pct':0.5}])
    row = ExistingDataMapper().map_dataframe(frame).iloc[0]
    assert row.net_pnl_pct == 2.9
    assert row.y == 0


def test_missing_reported_pnl_is_not_invented_from_ambiguous_cash_fields():
    frame = pd.DataFrame([{'symbol':'BTCUSDT','start_time_utc':'2026-10-08T12:00:00Z',
        'pnl_pct':np.nan,'total_profit_usdt':0.31,'commission_usdt':-0.2,
        'invested_margin_usdt':10,'grid_spacing_pct':0.5}])
    row = ExistingDataMapper().map_dataframe(frame).iloc[0]
    assert pd.isna(row.net_pnl_pct)
    assert pd.isna(row.y)


@pytest.mark.parametrize('pnl', [float('inf'), float('-inf')])
def test_nonfinite_final_pnl_is_not_a_training_label(pnl):
    frame = pd.DataFrame([{'symbol':'BTCUSDT','start_time_utc':'2026-10-08T12:00:00Z',
        'pnl_pct':pnl,'grid_spacing_pct':0.5}])
    row = ExistingDataMapper().map_dataframe(frame).iloc[0]
    assert pd.isna(row.net_pnl_pct)
    assert pd.isna(row.y)
