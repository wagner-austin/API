<#
.SYNOPSIS
    Register the Task Scheduler entry that runs one fleet-agent tick every
    3 minutes, so the dispatch queue drains with nobody watching.

.DESCRIPTION
    Why: until 2026-09-11 the queue had submit tools, a claim engine, and NO
    standing runner -- it drained only when somebody typed ``fleet-agent`` by
    hand (verified: no scheduled task named it). Board task 3c9033ff's A4:
    an enqueued job must execute within minutes with no human in between,
    which for the ``build-bases`` lane is the entire point -- a phone
    session's rebuild request cannot wait for someone to notice it.

    Why its own task rather than a pump publisher: the hpc-wake pump's
    PUBLISHERS table is for event bridges, which are subsecond; a rebuild
    tick can legitimately block for minutes of ``make build-bases``, and a
    bake must not delay the wake bridges' posts by its own duration.

    Why 3 minutes: the pump's own cadence -- an enqueued rebuild starts
    within one bake-length of being asked for. A tick against an empty
    queue is two HTTP calls.

    The account, logon type, triggers, IgnoreNew and the forty-minute limit
    are FleetSchedule.ps1's, with the incidents behind each. mcps-manager-audit
    has run PowerShell as an S4U task every 30 minutes since 2026-09-05 and
    exits 0, which is the precedent for the action.

    Idempotent: re-running unregisters + re-registers, so this serves as
    both install and refresh.

    Unregister with: unregister-agent-schedule.ps1

.PARAMETER TaskName
    The task's name.

.PARAMETER Tick
    The script the task runs; empty means run-agent-tick.ps1 beside this one.

.PARAMETER Register
    Registers the task: FleetSchedule.ps1's Register-FleetTick. The suite
    records the call instead, where registering the real tick would start a
    real agent within three minutes.

.NOTES
    The default is resolved in the body, not the param block, because
    $PSScriptRoot is empty in an advanced script's param default under -File
    in Windows PowerShell 5.1; run-agent-tick.ps1 carries the incident.
#>
[CmdletBinding()]
param(
    [string]$TaskName = 'API-FleetAgent-3min',
    [string]$Tick = '',
    [scriptblock]$Register = {
        param([string]$TaskName, [string]$Tick, [string]$TickArguments, [string]$Description)
        Register-FleetTick -TaskName $TaskName -Tick $Tick -TickArguments $TickArguments -Description $Description
    }
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'FleetSchedule.ps1')

if ($Tick -eq '') {
    $Tick = "$PSScriptRoot\run-agent-tick.ps1"
}

$identity = & $Register $TaskName $Tick '' `
    'One fleet-agent tick: drain the dispatch queue (API tools/fleet). See register-agent-schedule.ps1.'
Write-Information "Registered $TaskName (every 3 minutes and at boot, $identity, S4U, Limited)." -InformationAction Continue
