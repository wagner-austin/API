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

    The action (poetry.exe running fleet.cli.tick's hub lane, never
    powershell.exe), the account, logon type, triggers, priority, IgnoreNew
    and the forty-minute limit are FleetSchedule.ps1's, with the incidents
    behind each.

    Idempotent: re-running unregisters + re-registers, so this serves as
    both install and refresh.

    Unregister with: unregister-agent-schedule.ps1

.PARAMETER TaskName
    The task's name.

.PARAMETER ApiRoot
    The API checkout the tick runs from; empty means the one this script is
    in.

.PARAMETER LogDirectory
    Where the tick's daily log goes.

.PARAMETER Poetry
    The poetry the action runs: a name on this console's PATH or a path,
    registered as its absolute path.

.PARAMETER Register
    Registers the task: FleetSchedule.ps1's Register-FleetTick. The suite
    records the call instead, where registering the real tick would start a
    real agent within three minutes.

.NOTES
    The default is resolved in the body, not the param block, because
    $PSScriptRoot is empty in an advanced script's param default under -File
    in Windows PowerShell 5.1: the scheduled ticks' old PowerShell entry
    lost its root that way, and every tick from 16:27Z on 2026-09-27 exited
    1 (MCPs board task d69786fa).
#>
[CmdletBinding()]
param(
    [string]$TaskName = 'API-FleetAgent-3min',
    [string]$ApiRoot = '',
    [string]$LogDirectory = "$env:LOCALAPPDATA\Temp\claude",
    [string]$Poetry = 'poetry',
    [scriptblock]$Register = {
        param([string]$TaskName, [string]$Poetry, [string]$FleetRoot, [string]$Arguments, [string]$Description)
        Register-FleetTick -TaskName $TaskName -Poetry $Poetry -FleetRoot $FleetRoot -Arguments $Arguments -Description $Description
    }
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'FleetSchedule.ps1')

if ($ApiRoot -eq '') {
    $ApiRoot = "$PSScriptRoot\..\..\.."
}
$apiRoot = [System.IO.Path]::GetFullPath($ApiRoot)
$poetry = Resolve-FleetPoetry $Poetry
$arguments = Get-FleetTickCommandLine -ApiRoot $apiRoot -LogDirectory $LogDirectory -Lane hub -Node ''

$identity = & $Register $TaskName $poetry (Join-Path $apiRoot 'tools\fleet') $arguments `
    'One fleet-agent tick: drain the dispatch queue (API tools/fleet). See register-agent-schedule.ps1.'
Write-Information "Registered $TaskName (every 3 minutes and at boot, $identity, S4U, Limited)." -InformationAction Continue
