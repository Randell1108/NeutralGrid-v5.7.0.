import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = sorted(Path('logs').glob('live_decisions_*.jsonl'))[-1]
rows = []
for line in log.read_text(encoding='utf-8').splitlines():
    if not line.strip():
        continue
    rows.append(json.loads(line))
adjust_rows = [r for r in rows if r.get('verdict') == 'ADJUST']
if not adjust_rows:
    raise SystemExit('no ADJUST rows in latest scanner run')
now = datetime.now(timezone.utc)
for row in adjust_rows:
    key_base = f"{row['symbol']}|{row['strategy_id']}|{row['ts']}|{row['suggested_grid_lower']}|{row['suggested_grid_upper']}"
    key = hashlib.sha256(key_base.encode('utf-8')).hexdigest()
    decision_ts = row['ts']
    iteration_id = f"live_decision_scanner_{row['ts'].replace(':','').replace('+','').replace('-','').replace('.','')}"
    intent = {
        'schema_version': 'neutralgrid_action_intent_v1',
        'action': 'ADJUST',
        'idempotency_key': key,
        'symbol': row['symbol'],
        'strategy_id': row['strategy_id'],
        'suggested_grid_lower': row['suggested_grid_lower'],
        'suggested_grid_upper': row['suggested_grid_upper'],
        'decision_ts': row['ts'],
        'iteration_id': iteration_id,
    }
    approval = {
        'schema_version': 'neutralgrid_action_approval_v1',
        'idempotency_key': key,
        'symbol': row['symbol'],
        'strategy_id': row['strategy_id'],
        'action': 'ADJUST',
        'suggested_grid_lower': row['suggested_grid_lower'],
        'suggested_grid_upper': row['suggested_grid_upper'],
        'preserve_current_position': True,
        'approved_at_utc': now.isoformat(),
        'expires_at_utc': (now + timedelta(minutes=4, seconds=45)).isoformat(),
    }
    Path('outputs/runtime/live_telemetry_controller').mkdir(parents=True, exist_ok=True)
    intent_path = Path('outputs/runtime/live_telemetry_controller') / f"intent_{row['symbol']}.json"
    approval_path = Path('outputs/runtime/live_telemetry_controller') / f"approval_{row['symbol']}.json"
    intent_path.write_text(json.dumps(intent, indent=2), encoding='utf-8')
    approval_path.write_text(json.dumps(approval, indent=2), encoding='utf-8')
    print('intent', intent_path)
    print('approval', approval_path)
    print('key', key)
