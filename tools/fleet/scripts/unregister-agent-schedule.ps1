<#
.SYNOPSIS
    Remove the fleet-agent tick's Task Scheduler entry.

.DESCRIPTION
    The inverse of register-agent-schedule.ps1. With it gone the dispatch
    queue still accepts submissions -- they simply sit ``queued`` until a
    runner exists again, and ``dispatch_get`` says so honestly.

.PARAMETER TaskName
    The task's name.
#>
[CmdletBinding()]
param([string]$TaskName = 'API-FleetAgent-3min')
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'FleetSchedule.ps1')

if (Unregister-FleetTick $TaskName) {
    Write-Information "Unregistered $TaskName." -InformationAction Continue
} else {
    Write-Information "$TaskName is not registered; nothing to remove." -InformationAction Continue
}
