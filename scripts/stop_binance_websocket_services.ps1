param(
    [string]$AuditRoot = "outputs\audits\binance_websocket_services_current",
    [double]$TimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ResolvedAuditRoot = if ([System.IO.Path]::IsPathRooted($AuditRoot)) {
    [System.IO.Path]::GetFullPath($AuditRoot)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $AuditRoot))
}
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Checkout Python not found: $Python"
}
$StopHelper = Join-Path $PSScriptRoot "request_binance_websocket_stop.py"
& $Python $StopHelper --audit-root $ResolvedAuditRoot --timeout-seconds $TimeoutSeconds
exit $LASTEXITCODE
