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

    THE ACTION IS poetry.exe RUNNING fleet.cli.tick, NEVER powershell.exe
    (MCPs board task 94ac1c4f). powershell.exe as an S4U task action on the
    hub can stall before its engine loads and never exit: on 2026-09-28
    diphtheria's tick hung 16 minutes and sedona's 7, and under IgnoreNew
    each hang refused every later tick of its node. A native binary under
    the same principal does not stall; poetry.exe is pip's native launcher,
    and the module it runs is fleet.cli.tick, whose docstring carries what a
    tick does. The installers here are PowerShell; they run interactively.

    PRIORITY 4, NOT THE SCHEDULER'S 7. Task Scheduler gives a task priority
    7, below-normal CPU and low I/O, unless told otherwise. Under Defender's
    real-time scan on 2026-09-28 the ticks' rolled-tree extraction, 0.8 s by
    hand, outlived its 120 s deadline at priority 7, and no node claimed a
    job for about 80 minutes (MCPs board task 94ac1c4f). 4 is the normal
    class. It is set on every registration, so re-registering can never
    bring 7 back.

    A TASK IS FOUND BY ITS EXACT NAME from the full listing, never by
    `Get-ScheduledTask -TaskName <name> -ErrorAction SilentlyContinue`: that
    form turns every failure to read the scheduler into "not registered".

    THE LISTING AND THE DELETE GO THROUGH TASK SCHEDULER'S COM SERVICE, not
    Get-ScheduledTask and Unregister-ScheduledTask, which read every task's
    definition and fail with "The system cannot find the file specified"
    when ANY task is deleted while they run. Measured on the hub 2026-09-27
    against a process registering and deleting tasks: 3 of 150
    Get-ScheduledTask listings and 172 of 178 Unregister-ScheduledTask
    -TaskName calls failed, while 0 of 1,873 COM name listings did, and
    Register-ScheduledTask by name failed 0 of 178 times (MCPs board task
    d69786fa).
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-FleetScheduledTask {
    <#
    .SYNOPSIS
        The registered tasks, in any folder, whose names start and end as
        given.
    .PARAMETER Prefix
        The name's start, compared ordinally.
    .PARAMETER Suffix
        The name's end, compared ordinally.
    .OUTPUTS
        PSCustomObject[]: each task's TaskName and FolderPath (the folder as
        the COM service names it, '\' or '\Folder'), in the scheduler's
        order.
    #>
    [OutputType([pscustomobject[]])]
    param([Parameter(Mandatory)][string]$Prefix, [Parameter(Mandatory)][AllowEmptyString()][string]$Suffix)
    $scheduler = New-Object -ComObject Schedule.Service
    $scheduler.Connect()
    $found = [System.Collections.Generic.List[pscustomobject]]::new()
    $pending = [System.Collections.Generic.Queue[object]]::new()
    $pending.Enqueue($scheduler.GetFolder('\'))
    while ($pending.Count -gt 0) {
        $folder = $pending.Dequeue()
        foreach ($task in @($folder.GetTasks(1))) {
            $name = [string]$task.Name
            if ($name.StartsWith($Prefix, [System.StringComparison]::Ordinal) -and $name.EndsWith($Suffix, [System.StringComparison]::Ordinal)) {
                $found.Add([pscustomobject]@{ TaskName = $name; FolderPath = [string]$folder.Path })
            }
        }
        foreach ($child in @($folder.GetFolders(0))) {
            $pending.Enqueue($child)
        }
    }
    return [pscustomobject[]]$found.ToArray()
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
    $scheduler = New-Object -ComObject Schedule.Service
    $scheduler.Connect()
    foreach ($task in $registered) {
        $scheduler.GetFolder($task.FolderPath).DeleteTask($task.TaskName, 0)
    }
    return $registered.Count -gt 0
}

function Resolve-FleetPoetry {
    <#
    .SYNOPSIS
        poetry's absolute path for a task action, which cannot rely on the
        PATH of the account it runs under.
    .PARAMETER Poetry
        A bare name found on this console's PATH, or a path.
    .OUTPUTS
        String: the executable's absolute path.
    .NOTES
        Raises Get-Command's CommandNotFoundException, naming what was
        asked for, when nothing answers: the installer then registers
        nothing.
    #>
    [OutputType([string])]
    param([Parameter(Mandatory)][string]$Poetry)
    return [string](Get-Command -Name $Poetry -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
}

function Get-FleetTickCommandLine {
    <#
    .SYNOPSIS
        poetry's arguments for one fleet.cli.tick lane.
    .PARAMETER ApiRoot
        The API checkout the tick runs from.
    .PARAMETER LogDirectory
        Where the tick's log goes.
    .PARAMETER Lane
        hub, node, announce, or the elevated runner's elevated and
        elevated-announce (MCPs board task a98d7083).
    .PARAMETER Node
        The node's alias for the node lanes; empty for the hub.
    .OUTPUTS
        String: everything after poetry on the command line.
    #>
    [OutputType([string])]
    param(
        [Parameter(Mandatory)][string]$ApiRoot,
        [Parameter(Mandatory)][string]$LogDirectory,
        [Parameter(Mandatory)][ValidateSet('hub', 'node', 'announce', 'elevated', 'elevated-announce')][string]$Lane,
        [Parameter(Mandatory)][AllowEmptyString()][string]$Node
    )
    $line = "run -- python -m fleet.cli.tick --api-root `"$ApiRoot`" --log-directory `"$LogDirectory`" --lane $Lane"
    if ($Node -ne '') {
        $line += " --node $Node"
    }
    return $line
}

function Register-FleetTick {
    <#
    .SYNOPSIS
        Register (or replace) one tick's task: poetry running fleet.cli.tick
        every three minutes and at boot, as this account under S4U, at
        normal priority.
    .PARAMETER TaskName
        The task's name.
    .PARAMETER Poetry
        poetry's absolute path, the action's executable.
    .PARAMETER FleetRoot
        tools\fleet, where poetry finds the package's environment and where
        the tick runs.
    .PARAMETER Arguments
        poetry's arguments: Get-FleetTickCommandLine.
    .PARAMETER Description
        The task's description.
    .OUTPUTS
        String: the identity the task runs as.
    #>
    [OutputType([string])]
    param(
        [Parameter(Mandatory)][string]$TaskName,
        [Parameter(Mandatory)][string]$Poetry,
        [Parameter(Mandatory)][string]$FleetRoot,
        [Parameter(Mandatory)][string]$Arguments,
        [Parameter(Mandatory)][string]$Description
    )
    [void](Unregister-FleetTick $TaskName)
    $action = New-ScheduledTaskAction -Execute $Poetry -Argument $Arguments -WorkingDirectory $FleetRoot
    $bootTrigger = New-ScheduledTaskTrigger -AtStartup
    $timeTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 3)
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType S4U -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 40) -StartWhenAvailable -Priority 4
    [void](Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger @($bootTrigger, $timeTrigger) `
        -Settings $settings -Principal $principal -Description $Description)
    return $identity
}
