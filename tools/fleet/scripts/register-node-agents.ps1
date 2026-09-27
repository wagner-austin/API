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

.PARAMETER TaskPrefix
    What every node task's name starts with, before the alias.

.PARAMETER Tick
    The script each node task runs, and the announce runs once.

.PARAMETER PowerShell
    The powershell.exe the announce runs in.
#>
[CmdletBinding()]
param(
    [string]$Workspace = "$PSScriptRoot\..\fleet.json",
    [switch]$UnregisterAll,
    [string]$TaskPrefix = 'API-FleetNode-',
    [string]$Tick = "$PSScriptRoot\run-node-agent-tick.ps1",
    [string]$PowerShell = "$PSHOME\powershell.exe"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'FleetSchedule.ps1')

$taskSuffix = '-3min'

function Get-EnabledFleetNode {
    <#
    .SYNOPSIS
        The aliases fleet.json enables, in its order.
    .PARAMETER Path
        fleet.json.
    .OUTPUTS
        String[].
    #>
    [OutputType([string[]])]
    param([Parameter(Mandatory)][string]$Path)
    $document = [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
    $nodes = $document.PSObject.Properties['nodes']
    if ($null -eq $nodes) {
        throw "FLEET_NODES_MISSING: $Path has no nodes object"
    }
    $enabled = foreach ($node in $nodes.Value.PSObject.Properties) {
        $flag = $node.Value.PSObject.Properties['enabled']
        if ($null -ne $flag -and $flag.Value -eq $true) { $node.Name }
    }
    return [string[]]@($enabled)
}

$registered = @(Get-FleetScheduledTask -Prefix $TaskPrefix -Suffix $taskSuffix)
if ($UnregisterAll) {
    foreach ($task in $registered) {
        [void](Unregister-FleetTick $task.TaskName)
        Write-Information "Unregistered $($task.TaskName)." -InformationAction Continue
    }
    return
}

$enabled = @(Get-EnabledFleetNode $Workspace)
if ($enabled.Count -eq 0) {
    throw "FLEET_NODES_NONE_ENABLED: fleet.json at $Workspace enables no node; nothing to register"
}

# Tasks whose node is no longer enabled (or declared) are removed first, so
# the set of tasks after this script equals the set of enabled nodes.
foreach ($task in $registered) {
    $alias = $task.TaskName.Substring($TaskPrefix.Length, $task.TaskName.Length - $TaskPrefix.Length - $taskSuffix.Length)
    if ($enabled -notcontains $alias) {
        [void](Unregister-FleetTick $task.TaskName)
        Write-Information "Unregistered $($task.TaskName): $alias is not an enabled node." -InformationAction Continue
    }
}

foreach ($alias in $enabled) {
    $taskName = "$TaskPrefix$alias$taskSuffix"
    $identity = Register-FleetTick -TaskName $taskName -Tick $Tick -TickArguments " -Node $alias" `
        -Description "One fleet-node-agent tick for ${alias}: claim the node lane's jobs ${alias} carries the tags for (API tools/fleet). See register-node-agents.ps1."
    # The announce runs in this console, synchronously, so a refused
    # check-in is seen here rather than in a log nobody reads yet.
    & $PowerShell -NoProfile -ExecutionPolicy Bypass -File $Tick -Node $alias -Announce
    if ($LASTEXITCODE -ne 0) {
        throw "FLEET_NODE_ANNOUNCE_FAILED: the announce tick for $alias exited $LASTEXITCODE; see the fleet-node-$alias-*.log under $env:LOCALAPPDATA\Temp\claude"
    }
    Write-Information "Registered $taskName (every 3 minutes and at boot, $identity, S4U, Limited) and announced it." -InformationAction Continue
}
