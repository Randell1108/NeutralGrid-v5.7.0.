"""Complete exact archived snapshot fields; derive three costs from measured inputs."""
from __future__ import annotations
import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.core.config import get_config
from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES
from neutralgrid.scanner.enrich_grid_params import _horizon_realized_vol_pct
from neutralgrid.validation.microstructure import MicrostructureConfig, MicrostructureEstimator

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]

async def main():
    frame = pd.read_excel(OUT / 'full_57_backfilled.xlsx')
    records = json.loads((OUT / 'full_57_original_snapshot_candidates.json').read_text(encoding='utf-8'))
    cfg = get_config()
    cost_cfg = MicrostructureConfig(maker_fee=float(cfg.grid.maker_fee), taker_fee=float(cfg.grid.taker_fee), funding_extreme_threshold=float(cfg.validation.funding_extreme_threshold))
    estimator = MicrostructureEstimator(cost_cfg)
    horizon = cfg.grid.max_holding_seconds / 3600
    evidence = []
    client = BinanceClient()
    raw_dir = OUT / 'full_57_cost_market_data'
    raw_dir.mkdir(exist_ok=True)
    try:
        for index, bot in frame.iterrows():
            matches = [r for r in records if r['candidate_id'] == bot.candidate_id]
            if not matches:
                continue
            assert all(r['scan_and_recorded_file_time_before_bot_start'] for r in matches)
            source_rows = []
            for r in matches:
                path = ROOT / r['path']
                data = pd.read_csv(path, low_memory=False)
                data = data.loc[data.candidate_id.eq(bot.candidate_id)]
                assert len(data) == 1
                source_rows.append(data.iloc[0])
                r['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            record = {'strategy_id': str(bot.strategy_id), 'candidate_id': bot.candidate_id, 'sources': matches, 'fields': {}, 'timestamp_limitation': 'File times precede scan IDs. Both recorded times precede bot start; source was excluded from backtest admission. This does not independently prove archive creation time.'}
            for field in ['quote_volume_24h', 'open_interest', 'micro_round_trip_cost_pct']:
                values = [float(r[field]) for r in source_rows if pd.notna(r.get(field))]
                if values:
                    assert np.isfinite(values).all() and np.allclose(values, values[0], rtol=1e-12, atol=1e-14)
                    frame.at[index, field] = values[0]
                    record['fields'][field] = {'value': values[0], 'method': 'Recorded exact-candidate snapshot; copies agree'}
            if pd.isna(frame.at[index, 'micro_round_trip_cost_pct']):
                spreads = [float(r['spread_pct']) for r in source_rows]
                assert np.isfinite(spreads).all() and np.allclose(spreads, spreads[0], rtol=1e-12, atol=1e-14)
                scan = pd.Timestamp(matches[0]['scan'])
                raw = await client.get_klines(bot.symbol, '15m', limit=250, end_time=int(scan.timestamp()*1000)-1)
                raw = [k for k in raw if int(k[6]) < int(scan.timestamp()*1000)]
                assert len(raw) >= 193
                times = np.array([int(k[0]) for k in raw])
                assert (np.diff(times) == 900000).all()
                path = raw_dir / f'{bot.candidate_id}.json'
                path.write_text(json.dumps(raw), encoding='utf-8')
                volatility = _horizon_realized_vol_pct(raw, horizon_bars=horizon*4)
                assert volatility is not None and np.isfinite(volatility) and volatility > 0
                q = spreads[0] / 100
                spread_mid = q / (1 + q/2)
                value = round((spread_mid + 2*estimator.estimate_blended_fee() + 2*estimator.estimate_slippage(spread_mid, volatility))*100, 4)
                frame.at[index, 'micro_round_trip_cost_pct'] = value
                record['fields']['micro_round_trip_cost_pct'] = {'value': value, 'method': 'Current canonical microstructure formula, recorded bid-denominator spread converted to mid-denominator spread, historical closed-bar volatility; no fallback spread or volatility', 'recorded_spread_pct': spreads[0], 'mid_spread_decimal': spread_mid, 'volatility_pct': volatility, 'horizon_hours': horizon, 'config': dataclasses.asdict(cost_cfg), 'market_data_path': str(path), 'market_data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bars': len(raw), 'latest_close_utc': str(pd.Timestamp(int(raw[-1][6]), unit='ms', tz='UTC')), 'cutoff_utc': str(scan), 'limitation': 'Reconstructed model estimate, not an observed execution cost or a historical saved cost value'}
            evidence.append(record)
    finally:
        await client.close()
    matrix = frame[list(ACTIVE_SNAPSHOT_META_FEATURES)].apply(pd.to_numeric, errors='coerce').to_numpy()
    assert len(frame) == 57 and np.isfinite(matrix).all()
    version = json.loads((ROOT/'artifact_manifest.json').read_text())['hmm']['active_version']
    assert frame.hmm_artifact_version.eq(version).all()
    assert pd.to_datetime(frame.hmm_feature_cutoff_utc, utc=True, format='mixed').eq(pd.to_datetime(frame.start_time_utc, utc=True, format='mixed')).all()
    frame.to_excel(OUT/'full_57_completed.xlsx', index=False)
    frame.to_csv(OUT/'full_57_completed.csv', index=False)
    report = {'status': 'PASS', 'checked_at_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'rows': len(frame), 'selected_features': list(ACTIVE_SNAPSHOT_META_FEATURES), 'finite_feature_cells': int(np.isfinite(matrix).sum()), 'supplemented_snapshots': evidence, 'active_hmm': version}
    (OUT/'full_57_completion_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'supplemented_snapshots'}, indent=2))

if __name__ == '__main__':
    asyncio.run(main())
