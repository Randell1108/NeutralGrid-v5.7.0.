$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$OutLog = Join-Path $ProjectRoot "logs\live_decision_scanner_daemon.out.log"
$ErrLog = Join-Path $ProjectRoot "logs\live_decision_scanner_daemon.err.log"

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "logs") | Out-Null

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and
    $_.CommandLine -like '*live_decision_scanner.py*' -and
    $_.CommandLine -like '*--bots-dir*active bots*'
}

if ($existing) {
    $existing | Select-Object ProcessId,CommandLine
    exit 0
}

Start-Process `
    -FilePath "python" `
    -ArgumentList 'live_decision_scanner.py --interval 5m --bots-dir "active bots" --config-file config\live_discord_emit_all.yaml' `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog
