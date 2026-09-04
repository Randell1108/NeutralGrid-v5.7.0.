param(
    [string]$AuditDir = "outputs\audits\diff_depth_current"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ResolvedAuditDir = Join-Path $ProjectRoot $AuditDir
$StopFile = Join-Path $ResolvedAuditDir "STOP"

New-Item -ItemType Directory -Force -Path $ResolvedAuditDir | Out-Null
Set-Content -LiteralPath $StopFile -Value "stop requested" -Encoding ascii

$running = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and
    $_.CommandLine -like '*scripts\collect_diff_depth.py*'
}

if ($running) {
    Write-Output "Graceful stop requested for:"
    $running | Select-Object ProcessId, CommandLine
} else {
    Write-Output "No diff-depth collector process is running."
}
