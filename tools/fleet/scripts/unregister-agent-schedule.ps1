<#
.SYNOPSIS
    Remove the fleet-agent tick's Task Scheduler entry.

.DESCRIPTION
    The inverse of register-agent-schedule.ps1. With it gone the dispatch
    queue still accepts submissions -- they simply sit ``queued`` until a
    runner exists again, and ``dispatch_get`` says so honestly.
#>

$ErrorActionPreference = 'Stop'

schtasks /Delete /TN 'API-FleetAgent-3min' /F
Write-Host 'Unregistered API-FleetAgent-3min.'
