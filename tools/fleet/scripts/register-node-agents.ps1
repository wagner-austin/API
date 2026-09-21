<#
.SYNOPSIS
    Register one Task Scheduler entry per ENABLED node of fleet.json, each
    running that node's fleet-node-agent tick every 3 minutes, and retire
    the entry of any node no longer enabled.

.DESCRIPTION
    Why (MCPs board task fd5cabfa, A1 and A5): until this script the queue
    had one runner, the hub's API-FleetAgent-3min, claiming the oldest job
    once per tick whatever it was. Nine nodes behaved as one slow node, and a
    revive waited behind a queue of checks. Now the hub's task keeps the HUB
    lane (build-bases and the session verbs) and every enabled node has a
    task of its own on the hub, ``API-FleetNode-<alias>-3min``, claiming from
    the NODE lane the jobs that name it or name no node and whose required
    tags it carries. Two checks submitted together are claimed by two
    runners within one tick.

    THE SAME REGISTRATION AS THE HUB'S, DELIBERATELY: the operator's account
    (ssh keys, docker credentials, poetry), S4U at RunLevel Limited (the
    reasons are in register-agent-schedule.ps1 and hold unchanged), boot plus
    an indefinite 3-minute repetition, IgnoreNew, and an ExecutionTimeLimit
    in minutes that ends a tick which escaped every command's own deadline.
    Forty minutes, as the hub's: a node tick's longest command is a
    ten-minute fetch, then staging and the ssh calls at 120 s each.

    THE LIST IS READ FROM fleet.json, NOT WRITTEN HERE. A node enabled in the
    registry gets a task; a task whose node is no longer enabled (or no
    longer declared) is removed; re-running is the refresh. The two-file
    roster (fleet.json here, fleet-nodes.json in MCPs) is reconciled by
    ``fleet-nodes`` on the ``enabled`` flag and the platform and gpu the
    tags derive from, so the set registered here is the set the queue can
    match.

    EACH NEW RUNNER ANNOUNCES ITSELF ONCE: the first thing a freshly
    registered task's runner needs is a session on the board's ledger (MCPs
    mig 530), so this script runs the tick with -Announce for every node it
    registers, synchronously, before the schedule's first fire. The check-in
    names the node and the tags it carries.

    Unregister one node's task with: schtasks /Delete /TN 'API-FleetNode-<alias>-3min' /F
    Unregister every node's task with: register-node-agents.ps1 -UnregisterAll

.PARAMETER Workspace
    The fleet.json to read the nodes from. Defaults to the one beside this
    package.

.PARAMETER UnregisterAll
    Remove every API-FleetNode-*-3min task and register none.
#>

param(
    [string]$Workspace = (Join-Path (Split-Path -Parent $PSScriptRoot) 'fleet.json'),
    [switch]$UnregisterAll
)

$ErrorActionPreference = 'Stop'

$taskPrefix = 'API-FleetNode-'
$taskSuffix = '-3min'
$tick = Join-Path $PSScriptRoot 'run-node-agent-tick.ps1'

function Get-NodeTasks {
    Get-ScheduledTask -ErrorAction SilentlyContinue |
        Where-Object { $_.TaskName.StartsWith($taskPrefix) -and $_.TaskName.EndsWith($taskSuffix) }
}

if ($UnregisterAll) {
    foreach ($task in @(Get-NodeTasks)) {
        Unregister-ScheduledTask -TaskName $task.TaskName -Confirm:$false
        Write-Host "Unregistered $($task.TaskName)."
    }
    exit 0
}

$document = Get-Content -LiteralPath $Workspace -Raw -Encoding UTF8 | ConvertFrom-Json
$enabled = @()
foreach ($property in $document.nodes.PSObject.Properties) {
    if ($property.Value.enabled -eq $true) {
        $enabled += $property.Name
    }
}
if ($enabled.Count -eq 0) {
    throw "fleet.json at $Workspace enables no node; nothing to register"
}

# Tasks whose node is no longer enabled (or declared) are removed first, so
# the set of tasks after this script equals the set of enabled nodes.
foreach ($task in @(Get-NodeTasks)) {
    $alias = $task.TaskName.Substring($taskPrefix.Length)
    $alias = $alias.Substring(0, $alias.Length - $taskSuffix.Length)
    if ($enabled -notcontains $alias) {
        Unregister-ScheduledTask -TaskName $task.TaskName -Confirm:$false
        Write-Host "Unregistered $($task.TaskName): $alias is not an enabled node."
    }
}

# The identity comes from WindowsIdentity, NOT "$env:USERDOMAIN\$env:USERNAME":
# austinpc is not domain-joined and "WORKGROUP\test" does not resolve
# (0x80070534). GetCurrent().Name yields "AUSTINPC\Test".
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal `
    -UserId $identity `
    -LogonType S4U `
    -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 40) `
    -StartWhenAvailable

foreach ($alias in $enabled) {
    $taskName = "$taskPrefix$alias$taskSuffix"
    $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    }
    $action = New-ScheduledTaskAction `
        -Execute 'powershell.exe' `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$tick`" -Node $alias"
    $bootTrigger = New-ScheduledTaskTrigger -AtStartup
    $timeTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
        -RepetitionInterval (New-TimeSpan -Minutes 3)
    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger @($bootTrigger, $timeTrigger) `
        -Settings $settings `
        -Principal $principal `
        -Description "One fleet-node-agent tick for ${alias}: claim the node lane's jobs ${alias} carries the tags for (API tools/fleet). See register-node-agents.ps1." | Out-Null

    # The announce runs in this console, synchronously, so a refused
    # check-in is seen here rather than in a log nobody reads yet.
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $tick -Node $alias -Announce
    if ($LASTEXITCODE -ne 0) {
        throw "the announce tick for $alias exited $LASTEXITCODE; see the fleet-node-$alias-*.log under $env:LOCALAPPDATA\Temp\claude"
    }
    Write-Host "Registered $taskName (every 3 minutes and at boot, $identity, S4U, Limited) and announced it."
}

Get-NodeTasks |
    Select-Object TaskName, State, @{ n = 'LogonType'; e = { $_.Principal.LogonType } } |
    Format-Table -AutoSize
