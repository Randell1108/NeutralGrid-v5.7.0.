import urllib.request, json, re, subprocess
ports=[]
for line in subprocess.check_output(["netstat","-ano"], text=True, errors="ignore").splitlines():
    if 'LISTENING' not in line:
        continue
    parts=line.split()
    if len(parts)<5:
        continue
    local=parts[1]
    m=re.match(r'^(?:\[::\]|\d+\.\d+\.\d+\.\d+):(\d+)$', local)
    if not m:
        continue
    ports.append(int(m.group(1)))
seen=[]
for p in sorted(set(ports)):
    url=f'http://127.0.0.1:{p}/json/version'
    try:
        with urllib.request.urlopen(url, timeout=0.2) as f:
            data=json.load(f)
    except Exception:
        continue
    b=data.get('Browser')
    if isinstance(b,str) and ('Chrome' in b or 'chrom' in b.lower()):
        print(f'{p}\t{b}\t{data.get("BrowserVersion","")}' )
