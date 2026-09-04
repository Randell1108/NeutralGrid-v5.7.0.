import json
from pathlib import Path
import yaml

root = Path('')
manifest = json.loads((root / 'outputs' / 'audits' / 'live_telemetry_eventcomplete_current' / 'manifest.json').read_text(encoding='utf-8'))
run_dirs = manifest['symbol_run_dirs']
active_dir = Path('active bots')
run_ids = {}
active_bots = []
for yaml_path in sorted(active_dir.glob('*.yaml')):
    payload = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
    for bot in payload.get('bots', []):
        symbol = str(bot['symbol']).upper()
        strategy = str(bot.get('strategy_id', '')).strip()
        if symbol not in run_dirs:
            raise SystemExit(f'missing l2 run dir for {symbol}')
        run_dir = Path(run_dirs[symbol]).resolve()
        entry = dict(bot)
        entry['l2_stream'] = {
            'feature_path': str(run_dir / 'l2_risk_snapshots.jsonl'),
            'public_trade_path': str(run_dir / 'public_agg_trades.jsonl') if (run_dir / 'public_agg_trades.jsonl').is_file() else None,
            'manifest_path': str(run_dir / 'manifest.json'),
            'symbol': symbol,
            'strategy_id': strategy,
            'run_id': manifest['run_id'],
            'max_age_seconds': 15.0,
            'history_window_seconds': 300.0,
            'deterioration_min_duration_seconds': 60.0,
            'deterioration_min_observations': 3,
            'deterioration_fraction': 0.8,
        }
        active_bots.append(entry)
        run_ids[symbol] = bot.get('strategy_id')

out = Path('outputs/runtime/live_telemetry_controller/manual_active_bots_with_l2.yaml')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump({'bots': active_bots}, sort_keys=False), encoding='utf-8')
print(out)
print('count', len(active_bots))
print('run_ids', json.dumps(run_ids, sort_keys=True))
