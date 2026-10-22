from __future__ import annotations

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from types import SimpleNamespace

from neutralgrid.data.funding_rate import expected_funding_carry_next_hours, FundingRateProvider


@pytest.mark.parametrize('info,horizon', [
    ({'funding_interval_hours': 2}, 7),
    ({'funding_rate_pct': .01, 'funding_interval_hours': -1}, 7),
    ({'funding_rate_pct': .01, 'funding_interval_hours': 1}, float('nan')),
    ({'funding_rate_pct': .01, 'funding_interval_hours': 1}, 0),
    ({'funding_rate_pct': float('inf'), 'funding_interval_hours': 1}, 7),
    ({'funding_rate_pct': .01, 'funding_interval_hours': float('nan')}, 7),
    ({'funding_rate_pct': True, 'funding_interval_hours': 1}, 7),
    ({'funding_rate_pct': np.bool_(True), 'funding_interval_hours': 1}, 7),
])
def test_incomplete_or_invalid_forecast_is_missing(info, horizon):
    assert expected_funding_carry_next_hours(info, horizon_hours=horizon) is None


@pytest.mark.parametrize('rate,wait,expected', [(0, 1, 0), (-.02, 7, -.02), (.03, 0, .03), (.03, 7.01, 0)])
def test_known_forecast_preserves_sign_units_and_boundary(rate, wait, expected):
    assert expected_funding_carry_next_hours({'funding_rate_pct': rate, 'funding_interval_hours': wait}) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['missing_rate', 'missing_time', 'nan', 'stale', 'timeout', 'not_object', 'boolean_rate'])
async def test_provider_invalid_response_is_unavailable_without_crash(failure):
    result = {'symbol': 'BTCUSDT', 'time': int(datetime.now(timezone.utc).timestamp()*1000), 'lastFundingRate': '0.0001', 'nextFundingTime': int(
        (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp() * 1000)}
    if failure == 'missing_rate': del result['lastFundingRate']
    if failure == 'missing_time': del result['nextFundingTime']
    if failure == 'nan': result['lastFundingRate'] = 'nan'
    if failure == 'stale': result['nextFundingTime'] = 1
    if failure == 'boolean_rate': result['lastFundingRate'] = True
    client = SimpleNamespace(get_premium_index=AsyncMock(
        return_value=None if failure == 'not_object' else result,
        side_effect=TimeoutError('offline') if failure == 'timeout' else None))
    assert await FundingRateProvider(client).get_funding_info('BTCUSDT') is None


@pytest.mark.asyncio
async def test_provider_observed_rate_units_reach_forecast():
    result = {'symbol': 'BTCUSDT', 'time': int(datetime.now(timezone.utc).timestamp()*1000), 'markPrice': '65000.25', 'indexPrice': '64999.5', 'lastFundingRate': '-0.0002', 'nextFundingTime': int(
        (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp() * 1000)}
    client = SimpleNamespace(get_premium_index=AsyncMock(return_value=result))
    info = await FundingRateProvider(client).get_funding_info('BTCUSDT')
    assert info is not None
    assert info['funding_rate_pct'] == pytest.approx(-0.02)
    assert info['mark_price'] == 65000.25
    assert info['index_price'] == 64999.5
    assert expected_funding_carry_next_hours(info) == pytest.approx(-0.02)
