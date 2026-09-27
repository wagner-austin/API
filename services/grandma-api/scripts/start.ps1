<#
.SYNOPSIS
    Start grandma-api's API and web server on this host (make up).
.DESCRIPTION
    Loads .env into the process, then starts whichever side is not already
    listening and waits until it is (GrandmaService.ps1). The project root
    is the directory make runs this from, where it was once the hub's
    checkout hard-coded, and every target is a parameter so the suite runs
    this entry against stand-ins.
.PARAMETER ProjectRoot
    services\grandma-api, relative to the current location, where make runs
    this. Not a default naming $PSScriptRoot, which is empty in an advanced
    script's param default under -File in Windows PowerShell 5.1 (API
    tools/fleet/scripts/run-agent-tick.ps1 carries the incident).
.PARAMETER WebPort
    The web server's port.
.PARAMETER Poetry
    The poetry executable.
.PARAMETER Npm
    The npm executable.
.PARAMETER ReadySeconds
    How long each side may take to listen.
#>
[CmdletBinding()]
param(
    [string]$ProjectRoot = '.',
    [int]$WebPort = 8091,
    [string]$Poetry = 'poetry',
    [string]$Npm = 'npm.cmd',
    [int]$ReadySeconds = 30
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'GrandmaService.ps1')

$root = (Resolve-Path -LiteralPath $ProjectRoot).ProviderPath
[void](Import-GrandmaEnvironment (Join-Path $root '.env'))
# PORT comes from .env or the caller's environment, as the API reads it.
$apiPort = 8090
if ($null -ne $env:PORT) {
    $apiPort = [int]$env:PORT
}
$started = @(Start-GrandmaService -ProjectRoot $root -ApiPort $apiPort -WebPort $WebPort -Poetry $Poetry -Npm $Npm -ReadySeconds $ReadySeconds)
if ($started.Count -eq 0) {
    Write-Information "Already running on ports $apiPort and $WebPort" -InformationAction Continue
} else {
    Write-Information "Started $($started -join ', '). API: https://localhost:$apiPort  Web: https://localhost:$WebPort" -InformationAction Continue
}
