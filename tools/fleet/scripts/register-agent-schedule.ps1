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

    RUNS AS THE OPERATOR'S ACCOUNT, NOT SYSTEM, and that is load-bearing for
    the same three reasons as the MCPs fleet-audit task: the profile holds
    the ssh keys (node dispatches), the docker credentials (the API key is
    read from the live container), and the poetry environment.

    S4U, NOT "INTERACTIVE ONLY" (MCPs board tasks 660964d9 and d6c6bbea).
    Registered interactive-only, the task:

      * did not run from 03:28 to 10:25 local across the 2026-09-15 reboot,
        so dispatch and session-observe stopped while nobody was logged in;
      * ran with the operator's FILTERED token, and every session job needs
        more than that token can read. Measured 2026-09-17 with a one-shot
        task under the same principal: psutil.Process(<pid>).environ()
        raised AccessDenied on an sshd-spawned claude.exe and
        Win32_Process returned its CommandLine empty, so session-audit's
        pane join -- which is how restart-session and kill-session find the
        pane to type into -- could never resolve a pane from this runner.

    The same probe under an S4U task, still at RunLevel Limited, read the
    session's SSH_CONNECTION and its full command line. So the fix is the
    logon type, not elevation: RunLevel stays Limited. mcps-manager-audit
    has run PowerShell as an S4U task every 30 minutes since 2026-09-05
    and exits 0, which is the precedent for the action below.

    BOOT PLUS AN INDEFINITE REPETITION, never a logon trigger: the box is
    reached over ssh and is never logged into interactively, so a logon
    trigger fires once and never again (the WSL-Ubuntu-KeepAlive and
    DockerDesktopAutoStart outages).

    Idempotent: re-running unregisters + re-registers, so this serves as
    both install and refresh.

    Unregister with: unregister-agent-schedule.ps1
#>

$ErrorActionPreference = 'Stop'

$taskName = 'API-FleetAgent-3min'
$tick = Join-Path $PSScriptRoot 'run-agent-tick.ps1'

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$tick`""

$bootTrigger = New-ScheduledTaskTrigger -AtStartup
$timeTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
    -RepetitionInterval (New-TimeSpan -Minutes 3)

# The identity comes from WindowsIdentity, NOT "$env:USERDOMAIN\$env:USERNAME":
# austinpc is not domain-joined and "WORKGROUP\test" does not resolve
# (0x80070534). GetCurrent().Name yields "AUSTINPC\Test".
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal `
    -UserId $identity `
    -LogonType S4U `
    -RunLevel Limited

# IgnoreNew: a rebuild tick that outlasts three minutes must not be joined
# by a second tick racing it for the same queue rows.
#
# THE EXECUTION LIMIT IS THE LAST LINE, AND 72 HOURS WAS THE SCHEDULER'S
# DEFAULT, NOT A DECISION. With IgnoreNew, a tick that never exits refuses
# every tick after it until this limit ends it; measured 2026-09-17 11:15Z
# to 2026-09-20 11:15Z (board tasks 35940277 and 41ac6ed2), one ssh whose
# peer slept mid-command held the queue for exactly that long, and the
# revive queued at 10:00Z on the 20th ran at 19:21Z. Every command a tick
# runs now carries its own deadline (fleet.core._test_hooks.RunProtocol,
# mandatory): 120 s per ssh, 600 s per session verb, 1800 s for a bake.
# Forty minutes holds the longest of those with the collect and observe
# passes around it, and ends anything that escaped them thirteen ticks
# later rather than a thousand.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 40) `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger @($bootTrigger, $timeTrigger) `
    -Settings $settings `
    -Principal $principal `
    -Description 'One fleet-agent tick: drain the dispatch queue (API tools/fleet). See register-agent-schedule.ps1.' | Out-Null

Write-Host "Registered $taskName (every 3 minutes and at boot, $identity, S4U, Limited):"
Get-ScheduledTask -TaskName $taskName |
    Select-Object TaskName, State, @{ n = 'LogonType'; e = { $_.Principal.LogonType } } |
    Format-Table -AutoSize
