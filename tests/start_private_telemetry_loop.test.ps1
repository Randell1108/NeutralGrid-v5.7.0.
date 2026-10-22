$ErrorActionPreference = "Stop"
$Launcher = Join-Path $PSScriptRoot "..\scripts\start_private_telemetry_loop.ps1"
$Tokens = $null
$ParseErrors = $null
$Ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $Launcher, [ref]$Tokens, [ref]$ParseErrors
)
if ($ParseErrors.Count) { throw "Launcher has PowerShell syntax errors: $ParseErrors" }
$Function = $Ast.Find({
    param($Node)
    $Node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $Node.Name -eq "Test-BraveDebugEndpoint"
}, $true)
if (-not $Function) { throw "Missing Brave endpoint ownership check" }
Invoke-Expression $Function.Extent.Text

$BravePath = "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
$script:Listeners = @()
$script:OwnerPath = $BravePath
$script:HttpCalls = 0
$script:HttpUnavailable = $false
function Get-NetTCPConnection { $script:Listeners }
function Get-Process { [pscustomobject]@{ Path = $script:OwnerPath } }
function Invoke-RestMethod {
    $script:HttpCalls++
    if ($script:HttpUnavailable) { throw "Endpoint unavailable" }
    [pscustomobject]@{ webSocketDebuggerUrl = "ws://127.0.0.1:9222/devtools/browser/test" }
}
function Check-Endpoint {
    Test-BraveDebugEndpoint -Endpoint "http://127.0.0.1:9222" -DebugPort 9222 -BravePath $BravePath
}
function Expect-Rejection($Pattern) {
    $Rejected = $false
    try { Check-Endpoint | Out-Null } catch {
        if ($_.Exception.Message -notmatch $Pattern) { throw }
        $Rejected = $true
    }
    if (-not $Rejected) { throw "Expected endpoint rejection: $Pattern" }
}

if (Check-Endpoint) { throw "Absent listener accepted" }
if ($script:HttpCalls -ne 0) { throw "HTTP attempted without a verified owner" }
$script:Listeners = @([pscustomobject]@{
    LocalPort = 9222; LocalAddress = "127.0.0.1"; OwningProcess = 123
})
if (-not (Check-Endpoint)) { throw "Verified Brave endpoint rejected" }
$script:OwnerPath = "C:\OtherBrowser\browser.exe"
$Before = $script:HttpCalls
Expect-Rejection "not owned by Brave"
if ($script:HttpCalls -ne $Before) { throw "Contacted another browser's endpoint" }
$script:OwnerPath = $BravePath
$script:Listeners[0].LocalAddress = "0.0.0.0"
Expect-Rejection "loopback"
$script:Listeners[0].LocalAddress = "127.0.0.1"
$script:HttpUnavailable = $true
if (Check-Endpoint) { throw "Unavailable HTTP endpoint accepted" }
Write-Output "PASS: 5 Brave endpoint ownership and availability checks"
