import json, urllib.request
from scripts.collect_private_grid_telemetry import CDPConnection
pages = json.loads(urllib.request.urlopen('http://127.0.0.1:9222/json/list').read().decode())
ws = [p['webSocketDebuggerUrl'] for p in pages if p.get('type')=='page' and 'trading-bots/futures/grid/' in p.get('url','')][0]
c = CDPConnection(ws, timeout_seconds=8)
expr3 = "Array.from(document.querySelectorAll('button')).map(function(e){return (e.innerText||'').trim();}).filter(function(t){return /(log in|sign in|login|sign up|connect)/i.test(t)}).slice(0,20);"
print(c.evaluate(expr3))
expr4 = "Array.from(document.querySelectorAll('*')).some(function(e){var t=(e.textContent||'').toLowerCase(); return t.includes('spot grid') || t.includes('trading bots');});"
print(c.evaluate(expr4))
expr5 = "document.body && document.body.innerText ? document.body.innerText.length : 0"
print(c.evaluate(expr5))
c.close()
