param(
    [Parameter(Mandatory = $true)]
    [string[]]$Symbols,
    [double]$DurationSeconds = 0,
    [string]$AuditDir = "outputs\audits\diff_depth_current",
    [int]$FsyncEvery = 1
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogsDir = Join-Path $ProjectRoot "logs"
$ResolvedAuditDir = Join-Path $ProjectRoot $AuditDir
$OutLog = Join-Path $LogsDir "diff_depth_capture.out.log"
$ErrLog = Join-Path $LogsDir "diff_depth_capture.err.log"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Checkout Python not found: $Python"
}
if (-not $Symbols -or $Symbols.Count -eq 0) {
    throw "At least one symbol is required."
}

New-Item -ItemType Directory -Force -Path $LogsDir, $ResolvedAuditDir | Out-Null
Remove-Item -LiteralPath (Join-Path $ResolvedAuditDir "STOP") `
    -Force `
    -ErrorAction SilentlyContinue

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and
    $_.CommandLine -like '*scripts\collect_diff_depth.py*'
}
if ($existing) {
    $existing | Select-Object ProcessId, CommandLine
    exit 0
}

$CollectorArgs = @(
    "scripts\collect_diff_depth.py",
    "--symbols"
) + $Symbols + @(
    "--duration-seconds", $DurationSeconds,
    "--audit-dir", $AuditDir,
    "--fsync-every", $FsyncEvery
)

$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList $CollectorArgs `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru

Write-Output "Diff-depth collector started with PID $($Process.Id)."
Write-Output "Symbols: $($Symbols -join ', ')"
Write-Output "Audit manifest: $(Join-Path $ResolvedAuditDir 'manifest.json')"
Write-Output "Stop script: scripts\stop_diff_depth_capture.ps1"
