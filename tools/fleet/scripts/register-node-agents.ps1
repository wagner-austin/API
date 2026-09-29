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

    THE SAME REGISTRATION AS THE HUB'S, DELIBERATELY, and FleetSchedule.ps1
    carries each part's reason: poetry.exe running fleet.cli.tick (here its
    node lane), never powershell.exe; the operator's account (ssh keys,
    docker credentials, poetry) under S4U at RunLevel Limited; boot plus an
    indefinite 3-minute repetition; priority 4; IgnoreNew; and an
    ExecutionTimeLimit that ends a tick which escaped every command's own
    deadline. Forty minutes, as the hub's: a node tick's longest command is
    a ten-minute fetch, then staging and the ssh calls at 120 s each.

    THE LIST IS READ FROM fleet.json, NOT WRITTEN HERE. A node enabled in the
    registry gets a task; a task whose node is no longer enabled (or no
    longer declared) is removed; re-running is the refresh. The two-file
    roster (fleet.json here, fleet-nodes.json in MCPs) is reconciled by
    ``fleet-nodes`` on the ``enabled`` flag and the platform and gpu the
    tags derive from, so the set registered here is the set the queue can
    match.

    A NODE DECLARING ``elevated`` GETS A SECOND TASK (MCPs board task
    a98d7083), ``API-FleetNode-<alias>-elevated-3min``, whose tick runs with
    ``-Elevated``: it claims only the jobs requiring the ``elevated`` tag and
    launches them on the node at RunLevel Highest. The hub task itself is
    registered exactly like the first, S4U at Limited, because the elevation
    happens on the node, through the ssh account's own administrator token,
    which that runner re-measures every tick.

    EACH NEW RUNNER ANNOUNCES ITSELF ONCE: the first thing a freshly
    registered task's runner needs is a session on the board's ledger (MCPs
    mig 530), so this script runs the tick's announce lane for every node it
    registers, synchronously, before the schedule's first fire. The check-in
    names the node and the tags it carries.

    Unregister one node's task with: schtasks /Delete /TN 'API-FleetNode-<alias>-3min' /F
    Unregister every node's task with: register-node-agents.ps1 -UnregisterAll

.PARAMETER Workspace
    The fleet.json to read the nodes from; empty means the one beside this
    package.

.PARAMETER UnregisterAll
    Remove every API-FleetNode-*-3min task and register none.

.PARAMETER TaskPrefix
    What every node task's name starts with, before the alias.

.PARAMETER ApiRoot
    The API checkout the ticks run from; empty means the one this script is
    in.

.PARAMETER LogDirectory
    Where each tick's daily log goes.

.PARAMETER Poetry
    The poetry the actions and the announce run: a name on this console's
    PATH or a path, registered as its absolute path.

.PARAMETER Register
    Registers one node's task: FleetSchedule.ps1's Register-FleetTick. The
    suite records the call instead, where registering the real tick would
    start a real agent within three minutes.

.NOTES
    The defaults naming this script's directory are resolved in the body,
    not the param block, because $PSScriptRoot is empty in an advanced
    script's param default under -File in Windows PowerShell 5.1;
    register-agent-schedule.ps1 carries the incident.
#>
[CmdletBinding()]
param(
    [string]$Workspace = '',
    [switch]$UnregisterAll,
    [string]$TaskPrefix = 'API-FleetNode-',
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

if ($Workspace -eq '') {
    $Workspace = "$PSScriptRoot\..\fleet.json"
}
if ($ApiRoot -eq '') {
    $ApiRoot = "$PSScriptRoot\..\..\.."
}
$apiRoot = [System.IO.Path]::GetFullPath($ApiRoot)
$fleetRoot = Join-Path $apiRoot 'tools\fleet'

$taskSuffix = '-3min'

function Get-FleetNodeRunner {
    <#
    .SYNOPSIS
        One runner per node fleet.json enables, and a second, elevated one for
        each enabled node that declares ``elevated`` (MCPs board task
        a98d7083), in the file's order.
    .PARAMETER Path
        fleet.json.
    .OUTPUTS
        PSCustomObject[] with Node (the alias), Runner (the name inside the
        task name: the alias, or ``<alias>-elevated``) and Elevated.
    #>
    [OutputType([pscustomobject[]])]
    param([Parameter(Mandatory)][string]$Path)
    $document = [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
    $nodes = $document.PSObject.Properties['nodes']
    if ($null -eq $nodes) {
        throw "FLEET_NODES_MISSING: $Path has no nodes object"
    }
    $runners = foreach ($node in $nodes.Value.PSObject.Properties) {
        $flag = $node.Value.PSObject.Properties['enabled']
        if ($null -ne $flag -and $flag.Value -eq $true) {
            [pscustomobject]@{ Node = $node.Name; Runner = $node.Name; Elevated = $false }
            $elevated = $node.Value.PSObject.Properties['elevated']
            if ($null -ne $elevated -and $elevated.Value -eq $true) {
                [pscustomobject]@{ Node = $node.Name; Runner = "$($node.Name)-elevated"; Elevated = $true }
            }
        }
    }
    return [pscustomobject[]]@($runners)
}

$registered = @(Get-FleetScheduledTask -Prefix $TaskPrefix -Suffix $taskSuffix)
if ($UnregisterAll) {
    foreach ($task in $registered) {
        [void](Unregister-FleetTick $task.TaskName)
        Write-Information "Unregistered $($task.TaskName)." -InformationAction Continue
    }
    return
}

$runners = @(Get-FleetNodeRunner $Workspace)
if ($runners.Count -eq 0) {
    throw "FLEET_NODES_NONE_ENABLED: fleet.json at $Workspace enables no node; nothing to register"
}

# Tasks naming no runner fleet.json asks for (a node no longer enabled or
# declared, or an elevated runner its node no longer declares) are removed
# first, so the set of tasks after this script equals the set of runners.
# The comparison is by whole task name: an elevated runner's name reads as
# an alias ending in -elevated, which no node carries.
$expected = @($runners | ForEach-Object { "$TaskPrefix$($_.Runner)$taskSuffix" })
foreach ($task in $registered) {
    if ($expected -notcontains $task.TaskName) {
        [void](Unregister-FleetTick $task.TaskName)
        Write-Information "Unregistered $($task.TaskName): fleet.json asks for no such runner." -InformationAction Continue
    }
}

$poetry = Resolve-FleetPoetry $Poetry
foreach ($runner in $runners) {
    $alias = $runner.Node
    $taskName = "$TaskPrefix$($runner.Runner)$taskSuffix"
    $lane = 'node'
    $announceLane = 'announce'
    $what = "claim the node lane's jobs ${alias} carries the tags for"
    if ($runner.Elevated) {
        $lane = 'elevated'
        $announceLane = 'elevated-announce'
        $what = 'claim only the jobs requiring the elevated tag, and launch them at RunLevel Highest'
    }
    $arguments = Get-FleetTickCommandLine -ApiRoot $apiRoot -LogDirectory $LogDirectory -Lane $lane -Node $alias
    $identity = & $Register $taskName $poetry $fleetRoot $arguments `
        "One fleet-node-agent tick for $($runner.Runner): $what (API tools/fleet). See register-node-agents.ps1."
    # The announce runs from this console, synchronously, so a refused
    # check-in is seen here rather than in a log nobody reads yet. The
    # process object's exit code is read directly: no redirection or pipe
    # stands between it and this check.
    $announce = Get-FleetTickCommandLine -ApiRoot $apiRoot -LogDirectory $LogDirectory -Lane $announceLane -Node $alias
    $announced = Start-Process -FilePath $poetry -ArgumentList $announce -WorkingDirectory $fleetRoot -NoNewWindow -Wait -PassThru
    if ($announced.ExitCode -ne 0) {
        throw "FLEET_NODE_ANNOUNCE_FAILED: the announce tick for $($runner.Runner) exited $($announced.ExitCode); see fleet-node-$($runner.Runner)-*.log under $LogDirectory"
    }
    Write-Information "Registered $taskName (every 3 minutes and at boot, $identity, S4U, Limited) and announced it." -InformationAction Continue
}
