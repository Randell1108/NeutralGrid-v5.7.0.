param(
    [string]$AuditRoot = "outputs\audits\binance_websocket_services_current"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ResolvedAuditRoot = if ([System.IO.Path]::IsPathRooted($AuditRoot)) {
    [System.IO.Path]::GetFullPath($AuditRoot)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $AuditRoot))
}
$StopFile = Join-Path $ResolvedAuditRoot "STOP"
$LockPath = Join-Path $ResolvedAuditRoot "supervisor.lock"

New-Item -ItemType Directory -Force -Path $ResolvedAuditRoot | Out-Null
Set-Content -LiteralPath $StopFile -Value "stop requested" -Encoding ascii

if (Test-Path -LiteralPath $LockPath -PathType Leaf) {
    $OwnerPidText = (Get-Content -LiteralPath $LockPath -Raw).Trim()
    $OwnerPid = 0
    if ([int]::TryParse($OwnerPidText, [ref]$OwnerPid)) {
        $Existing = Get-Process -Id $OwnerPid -ErrorAction SilentlyContinue
        if ($Existing) {
            Write-Output "Graceful stop requested for Binance WebSocket supervisor PID $OwnerPid."
            exit 0
        }
    }
}
Write-Output "Stop marker written; no live supervisor owner was verified."
