$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$AuditDir = Join-Path $ProjectRoot "outputs\audits\private_telemetry_loop_current"
$StopFile = Join-Path $AuditDir "STOP"

New-Item -ItemType Directory -Force -Path $AuditDir | Out-Null
Set-Content -LiteralPath $StopFile -Value "stop requested" -Encoding ascii

$running = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and
    $_.CommandLine -like '*scripts\collect_private_grid_telemetry.py*'
}

if ($running) {
    Write-Output "Stop requested for:"
    $running | Select-Object ProcessId,CommandLine
} else {
    Write-Output "No private telemetry collector process is running."
}
