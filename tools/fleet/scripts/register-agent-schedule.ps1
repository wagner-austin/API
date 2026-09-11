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

    RUNS AS THE INTERACTIVE USER, NOT SYSTEM, and that is load-bearing for
    the same three reasons as the MCPs fleet-audit task: the profile holds
    the ssh keys (node dispatches), the docker credentials (the API key is
    read from the live container), and the poetry environment.

    Idempotent: re-running unregisters + re-registers, so this serves as
    both install and refresh.

    Unregister with: unregister-agent-schedule.ps1
#>

$ErrorActionPreference = 'Stop'

$taskName = 'API-FleetAgent-3min'
$action = Join-Path $PSScriptRoot 'run-agent-tick.ps1'

# An absent task is the expected first-run answer, not an error; scoping the
# preference around the call is the same fix check-base-freshness.ps1 carries
# for docker-run-against-a-missing-image (a native command's stderr under
# script-level 'Stop' becomes a terminating NativeCommandError).
$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
schtasks /Delete /TN $taskName /F 2>$null | Out-Null
$ErrorActionPreference = $prevEap

schtasks /Create /TN $taskName /F `
    /SC MINUTE /MO 3 `
    /TR "powershell -NoProfile -ExecutionPolicy Bypass -File `"$action`"" `
    | Out-Null

Write-Host "Registered $taskName (every 3 minutes, interactive user):"
schtasks /Query /TN $taskName /FO LIST | Select-String 'TaskName|Status|Next Run Time'
