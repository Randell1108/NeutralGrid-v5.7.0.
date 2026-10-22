param(
    [int]$DebugPort = 9222,
    [int]$IntervalSeconds = 300
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Brave = "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
$Profile = Join-Path $env:LOCALAPPDATA "NeutralGrid\PrivateTelemetryBrave"
$AuditDir = Join-Path $ProjectRoot "outputs\audits\private_telemetry_loop_current"
$LogsDir = Join-Path $ProjectRoot "logs"
$OutLog = Join-Path $LogsDir "private_telemetry_loop.out.log"
$ErrLog = Join-Path $LogsDir "private_telemetry_loop.err.log"
$Endpoint = "http://127.0.0.1:$DebugPort"
$GridUrl = "https://www.binance.bh/en/trading-bots/futures/grid/SPXUSDT"

function Test-BraveDebugEndpoint {
    param([string]$Endpoint, [int]$DebugPort, [string]$BravePath)

    $Listeners = @(Get-NetTCPConnection -State Listen -ErrorAction Stop |
        Where-Object { $_.LocalPort -eq $DebugPort })
    if (-not $Listeners.Count) { return $false }
    foreach ($Listener in $Listeners) {
        if ($Listener.LocalAddress -notin @("127.0.0.1", "::1")) {
            throw "Brave debugging endpoint must listen only on loopback: $DebugPort"
        }
        $Owner = Get-Process -Id $Listener.OwningProcess -ErrorAction Stop
        if (-not $Owner.Path -or $Owner.Path -ine $BravePath) {
            throw "Debugging port $DebugPort is not owned by Brave: $($Owner.Path)"
        }
    }
    try {
        $Version = Invoke-RestMethod -Uri "$Endpoint/json/version" -TimeoutSec 2
        return -not [string]::IsNullOrWhiteSpace($Version.webSocketDebuggerUrl)
    } catch {
        return $false
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Checkout Python not found: $Python"
}
if (-not (Test-Path -LiteralPath $Brave)) {
    throw "Brave not found: $Brave"
}

New-Item -ItemType Directory -Force -Path $Profile, $AuditDir, $LogsDir | Out-Null
Remove-Item -LiteralPath (Join-Path $AuditDir "STOP") -Force -ErrorAction SilentlyContinue

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and
    $_.CommandLine -like '*scripts\collect_private_grid_telemetry.py*'
}
$debugReady = Test-BraveDebugEndpoint -Endpoint $Endpoint -DebugPort $DebugPort -BravePath $Brave
if ($existing) {
    if (-not $debugReady) {
        throw "Existing collector has no verified Brave endpoint. Stop it with scripts\stop_private_telemetry_loop.ps1 before restarting."
    }
    $existing | Select-Object ProcessId,CommandLine
    exit 0
}

if (-not $debugReady) {
    $BraveArgs = @(
        "--remote-debugging-port=$DebugPort",
        "--remote-debugging-address=127.0.0.1",
        "`"--user-data-dir=$Profile`"",
        "--no-first-run",
        "--no-default-browser-check",
        "--window-size=1920,1080",
        $GridUrl
    )
    Start-Process `
        -FilePath $Brave `
        -ArgumentList $BraveArgs `
        -WindowStyle Hidden

    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 500
        $debugReady = Test-BraveDebugEndpoint -Endpoint $Endpoint -DebugPort $DebugPort -BravePath $Brave
        if ($debugReady) { break }
    }
}

if (-not $debugReady) {
    throw "Dedicated Brave debugging endpoint did not become ready at $Endpoint"
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
Write-Output "Dedicated Brave profile: $Profile"
Write-Output "If Binance is not signed in in that window, sign in once; rejected cycles remain fail-closed."
Write-Output "Manifest: $AuditDir\manifest.json"
