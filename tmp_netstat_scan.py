import subprocess,re
target='21808'
for line in subprocess.check_output(['netstat','-ano'], text=True, errors='ignore').splitlines():
    if 'LISTENING' not in line: continue
    parts=line.split()
    if len(parts)<5: continue
    addr, state, pid = parts[1], parts[3], parts[4]
    if pid!=target: continue
    print(line)
