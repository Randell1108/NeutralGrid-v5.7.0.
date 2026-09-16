param(
    [Parameter(Mandatory = $true)]
    [string]$TargetCsv,
    [string[]]$LinkageFiles = @(),
    [double]$DurationSeconds = 0,
    [string]$AuditRoot = "outputs\audits\binance_websocket_services_current",
    [int]$FsyncEvery = 100
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectPath([string]$Value, [string]$ProjectRoot) {
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Value))
}

function Quote-ProcessArgument([string]$Value) {
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    return '"' + $Value.Replace('"', '\"') + '"'
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ResolvedTargetCsv = Resolve-ProjectPath $TargetCsv $ProjectRoot
$ResolvedAuditRoot = Resolve-ProjectPath $AuditRoot $ProjectRoot
$LogsDir = Join-Path $ProjectRoot "logs"
$SupervisorOut = Join-Path $LogsDir "binance_ws_supervisor.out.log"
$SupervisorErr = Join-Path $LogsDir "binance_ws_supervisor.err.log"
$LockPath = Join-Path $ResolvedAuditRoot "supervisor.lock"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Checkout Python not found: $Python"
}
if (-not (Test-Path -LiteralPath $ResolvedTargetCsv -PathType Leaf)) {
    throw "Exact target CSV not found: $ResolvedTargetCsv"
}
if ($DurationSeconds -lt 0) {
    throw "DurationSeconds must be non-negative."
}
if ($FsyncEvery -lt 0) {
    throw "FsyncEvery must be non-negative."
}

$ResolvedLinkages = @()
foreach ($Linkage in $LinkageFiles) {
    $Resolved = Resolve-ProjectPath $Linkage $ProjectRoot
    if (-not (Test-Path -LiteralPath $Resolved -PathType Leaf)) {
        throw "Reviewed linkage file not found: $Resolved"
    }
    $ResolvedLinkages += $Resolved
}

New-Item -ItemType Directory -Force -Path $LogsDir, $ResolvedAuditRoot | Out-Null
if (Test-Path -LiteralPath $LockPath -PathType Leaf) {
    $OwnerPidText = (Get-Content -LiteralPath $LockPath -Raw).Trim()
    $OwnerPid = 0
    if ([int]::TryParse($OwnerPidText, [ref]$OwnerPid)) {
        $Existing = Get-Process -Id $OwnerPid -ErrorAction SilentlyContinue
        if ($Existing) {
            Write-Output "Binance WebSocket supervisor is already running with PID $OwnerPid."
            Write-Output "Manifest: $(Join-Path $ResolvedAuditRoot 'manifest.json')"
            exit 0
        }
    }
}

Remove-Item -LiteralPath (Join-Path $ResolvedAuditRoot "STOP") `
    -Force `
    -ErrorAction SilentlyContinue

$SupervisorArgs = @(
    "scripts\supervise_binance_websocket_services.py",
    "--target-csv", $ResolvedTargetCsv,
    "--duration-seconds", ([string]$DurationSeconds),
    "--audit-root", $ResolvedAuditRoot,
    "--fsync-every", ([string]$FsyncEvery)
)
foreach ($Resolved in $ResolvedLinkages) {
    $SupervisorArgs += @("--linkage-file", $Resolved)
}
$QuotedArgs = $SupervisorArgs | ForEach-Object {
    Quote-ProcessArgument ([string]$_)
}

$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList $QuotedArgs `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $SupervisorOut `
    -RedirectStandardError $SupervisorErr `
    -PassThru

Write-Output "Binance WebSocket supervisor started with PID $($Process.Id)."
Write-Output "Exact target CSV: $ResolvedTargetCsv"
Write-Output "Services: /public diff-depth; /market trades, marks, klines; /private user data."
Write-Output "Manifest: $(Join-Path $ResolvedAuditRoot 'manifest.json')"
Write-Output "Verify: .\.venv\Scripts\python.exe scripts\verify_binance_websocket_services.py --target-csv `"$ResolvedTargetCsv`" --audit-root `"$ResolvedAuditRoot`""
Write-Output "Stop: powershell -ExecutionPolicy Bypass -File .\scripts\stop_binance_websocket_services.ps1 -AuditRoot `"$ResolvedAuditRoot`""
