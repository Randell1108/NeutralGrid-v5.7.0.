param(
    [int]$DebugPort = 9222,
    [int]$IntervalSeconds = 300
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$Profile = Join-Path $env:LOCALAPPDATA "NeutralGrid\PrivateTelemetryChrome"
$AuditDir = Join-Path $ProjectRoot "outputs\audits\private_telemetry_loop_current"
$LogsDir = Join-Path $ProjectRoot "logs"
$OutLog = Join-Path $LogsDir "private_telemetry_loop.out.log"
$ErrLog = Join-Path $LogsDir "private_telemetry_loop.err.log"
$Endpoint = "http://127.0.0.1:$DebugPort"
$GridUrl = "https://www.binance.bh/en/trading-bots/futures/grid/SPXUSDT"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Checkout Python not found: $Python"
}
if (-not (Test-Path -LiteralPath $Chrome)) {
    throw "Chrome not found: $Chrome"
}

New-Item -ItemType Directory -Force -Path $Profile, $AuditDir, $LogsDir | Out-Null
Remove-Item -LiteralPath (Join-Path $AuditDir "STOP") -Force -ErrorAction SilentlyContinue

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and
    $_.CommandLine -like '*scripts\collect_private_grid_telemetry.py*'
}
if ($existing) {
    $existing | Select-Object ProcessId,CommandLine
    exit 0
}

$debugReady = $false
try {
    Invoke-RestMethod -Uri "$Endpoint/json/version" -TimeoutSec 2 | Out-Null
    $debugReady = $true
} catch {
    $debugReady = $false
}

if (-not $debugReady) {
    $ChromeArgs = @(
        "--remote-debugging-port=$DebugPort",
        "`"--user-data-dir=$Profile`"",
        "--no-first-run",
        "--no-default-browser-check",
        "--window-size=1920,1080",
        $GridUrl
    )
    Start-Process `
        -FilePath $Chrome `
        -ArgumentList $ChromeArgs `
        -WindowStyle Normal

    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            Invoke-RestMethod -Uri "$Endpoint/json/version" -TimeoutSec 2 | Out-Null
            $debugReady = $true
            break
        } catch {
            $debugReady = $false
        }
    }
}

if (-not $debugReady) {
    throw "Dedicated Chrome debugging endpoint did not become ready at $Endpoint"
}

$CollectorArgs = @(
    "scripts\collect_private_grid_telemetry.py",
    "--debug-endpoint", $Endpoint,
    "--interval-seconds", $IntervalSeconds,
    "--live-root", "Live",
    "--audit-dir", "outputs\audits\private_telemetry_loop_current"
)

Start-Process `
    -FilePath $Python `
    -ArgumentList $CollectorArgs `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog

Write-Output "Private telemetry collector started."
Write-Output "Dedicated Chrome profile: $Profile"
Write-Output "If Binance is not signed in in that window, sign in once; rejected cycles remain fail-closed."
Write-Output "Manifest: $AuditDir\manifest.json"
