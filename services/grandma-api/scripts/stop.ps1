<#
.SYNOPSIS
    Stop grandma-api's API and web server on this host (make down).
.DESCRIPTION
    Ends the process tree listening on each port, by pid
    (GrandmaService.ps1).
.PARAMETER ApiPort
    The API's port.
.PARAMETER WebPort
    The web server's port.
.PARAMETER Taskkill
    taskkill.exe.
#>
[CmdletBinding()]
param(
    [int]$ApiPort = 8090,
    [int]$WebPort = 8091,
    [string]$Taskkill = "$env:SystemRoot\System32\taskkill.exe"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'GrandmaService.ps1')

$ended = @(Stop-GrandmaService -Ports @($ApiPort, $WebPort) -Taskkill $Taskkill)
Write-Information "Stopped $($ended.Count) process tree(s) on ports $ApiPort and $WebPort." -InformationAction Continue
