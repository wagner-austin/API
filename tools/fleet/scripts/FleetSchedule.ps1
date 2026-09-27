<#
.SYNOPSIS
    Register and remove the Task Scheduler entries that run fleet ticks.
.DESCRIPTION
    register-agent-schedule.ps1 (the hub lane's one task) and
    register-node-agents.ps1 (one task per enabled node) register the same
    shape; until 2026-09-27 each built it itself (MCPs board task d69786fa).

    RUNS AS THE OPERATOR'S ACCOUNT, S4U, AT RunLevel LIMITED (MCPs board
    tasks 660964d9 and d6c6bbea). The profile holds the ssh keys, the
    docker credentials and the poetry environment. Registered
    interactive-only, the task did not run from 03:28 to 10:25 local across
    the 2026-09-15 reboot, and ran with the operator's FILTERED token, under
    which psutil could not read an sshd-spawned claude.exe's environment and
    Win32_Process returned its command line empty (measured 2026-09-17), so
    session-audit's pane join could never resolve. The same probe under an
    S4U task at RunLevel Limited read both, so the fix is the logon type,
    not elevation. The identity is WindowsIdentity's name ("AUSTINPC\Test"),
    not "$env:USERDOMAIN\$env:USERNAME": the hub is not domain-joined and
    "WORKGROUP\test" does not resolve (0x80070534).

    BOOT PLUS AN INDEFINITE THREE-MINUTE REPETITION, never a logon trigger:
    the box is reached over ssh and never logged into interactively, so a
    logon trigger fires once and never again.

    IgnoreNew AND FORTY MINUTES. A tick that outlasts three minutes must not
    be joined by a second racing it for the same queue rows. With IgnoreNew
    a tick that never exits refuses every tick after it until the limit
    ends it; the scheduler's default of 72 hours held the queue from
    2026-09-17 11:15Z to 2026-09-20 11:15Z (board tasks 35940277 and
    41ac6ed2). Every command a tick runs now carries its own deadline, the
    longest a 1800 s bake, and forty minutes holds that with the passes
    around it.

    A TASK IS FOUND BY ITS EXACT NAME from the full listing, never by
    `Get-ScheduledTask -TaskName <name> -ErrorAction SilentlyContinue`: that
    form turns every failure to read the scheduler into "not registered".
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-FleetScheduledTask {
    <#
    .SYNOPSIS
        The registered tasks whose names start and end as given.
    .PARAMETER Prefix
        The name's start.
    .PARAMETER Suffix
        The name's end.
    .OUTPUTS
        CimInstance[]: the tasks, in the scheduler's order.
    #>
    [OutputType([Microsoft.Management.Infrastructure.CimInstance[]])]
    param([Parameter(Mandatory)][string]$Prefix, [Parameter(Mandatory)][AllowEmptyString()][string]$Suffix)
    return [Microsoft.Management.Infrastructure.CimInstance[]]@(Get-ScheduledTask |
        Where-Object { $_.TaskName.StartsWith($Prefix, [System.StringComparison]::Ordinal) -and $_.TaskName.EndsWith($Suffix, [System.StringComparison]::Ordinal) })
}

function Unregister-FleetTick {
    <#
    .SYNOPSIS
        Remove one task by its exact name, if it is registered.
    .PARAMETER TaskName
        The task's name.
    .OUTPUTS
        Boolean: whether a task was removed.
    #>
    [OutputType([bool])]
    param([Parameter(Mandatory)][string]$TaskName)
    $registered = @(Get-FleetScheduledTask -Prefix $TaskName -Suffix '' | Where-Object { $_.TaskName -ceq $TaskName })
    foreach ($task in $registered) {
        Unregister-ScheduledTask -TaskName $task.TaskName -TaskPath $task.TaskPath -Confirm:$false
    }
    return $registered.Count -gt 0
}

function Register-FleetTick {
    <#
    .SYNOPSIS
        Register (or replace) one tick's task: powershell.exe running a
        script every three minutes and at boot, as this account under S4U.
    .PARAMETER TaskName
        The task's name.
    .PARAMETER Tick
        The script the action runs.
    .PARAMETER TickArguments
        What follows -File <script> on the action's command line.
    .PARAMETER Description
        The task's description.
    .OUTPUTS
        String: the identity the task runs as.
    #>
    [OutputType([string])]
    param(
        [Parameter(Mandatory)][string]$TaskName,
        [Parameter(Mandatory)][string]$Tick,
        [Parameter(Mandatory)][AllowEmptyString()][string]$TickArguments,
        [Parameter(Mandatory)][string]$Description
    )
    [void](Unregister-FleetTick $TaskName)
    $argument = "-NoProfile -ExecutionPolicy Bypass -File `"$Tick`"$TickArguments"
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argument
    $bootTrigger = New-ScheduledTaskTrigger -AtStartup
    $timeTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 3)
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType S4U -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 40) -StartWhenAvailable
    [void](Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger @($bootTrigger, $timeTrigger) `
        -Settings $settings -Principal $principal -Description $Description)
    return $identity
}
