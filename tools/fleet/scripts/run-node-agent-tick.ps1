<#
.SYNOPSIS
    One scheduled tick of fleet-node-agent for ONE node: resolve credentials
    fresh, run, exit with the agent's own status.

.DESCRIPTION
    The action behind each ``API-FleetNode-<alias>-3min`` scheduled task
    (register-node-agents.ps1): the node lane's runner for one enabled node
    of fleet.json (MCPs board task fd5cabfa, A1). The hub's own tick,
    run-agent-tick.ps1, keeps the hub lane; this one claims only the
    check/lint/test jobs the node's capability tags admit.

    Everything the agent needs but must not store is resolved AT TICK TIME,
    the same way and from the same places as run-agent-tick.ps1:

      * CORVIS_TENANT_ID  -- dot-sourced from the hpc-wake pump's runs/env.ps1.
      * FLEET_MCP_API_KEY -- fleet-mcp's MCP_INTERNAL_KEY, from the same
        runs/env.ps1. Not from a container: fleet-mcp runs on diphtheria
        since 2026-09-25 and the hub has no ``mcp-fleet`` to inspect (MCPs
        board tasks 91ca67f4 and 60df277e). Missing surfaces as the agent's
        named QUEUE_CREDENTIALS_MISSING refusal.
      * TASKBOARD_MCP_API_KEY -- from the same runs/env.ps1, for the
        verdict the collect pass posts to the submitting task's thread (A3)
        and for the ``--announce`` check-in. Not from a container: the
        taskboard runs on diphtheria and the hub's ``mcp-taskboard``
        forwarder is retired (MCPs board task c6fc4882). Missing surfaces
        as board-watch's named API_KEY_MISSING.

    THE RUNNER'S IDENTITY IS DERIVED FROM THE NODE, NOT PASSED HERE:
    ``fleet-node-<alias>`` with a version-5 UUID of that name
    (fleet.cli.node_agent.node_identity), so every tick of one node's runner
    is the same session on the board's ledger and ``held_by`` finds its own
    claims across ticks. ``-Announce`` posts the check-in that registers the
    session on the ledger; the registration script does it once per node.

    THE EXIT CODE IS THE AGENT'S. fleet-node-agent exits 0 whenever the
    AGENT worked, refused jobs and failed suites included; a non-zero here
    means the tick itself broke (credentials, transport, or this machine's
    records and the fleet disagreeing about a run).

.PARAMETER Node
    The node's alias in fleet.json (the key under ``nodes``).

.PARAMETER Announce
    Post the check-in that registers the runner's session on the board's
    ledger, and claim nothing. Used once, by register-node-agents.ps1.

.PARAMETER ApiRoot
    The API checkout; empty means the one this script is in.

.PARAMETER EnvironmentScript
    The hpc-wake pump's runs/env.ps1; empty means the one in ApiRoot.

.PARAMETER Poetry
    The poetry executable.

.PARAMETER LogDirectory
    Where the day's log goes: one file per node per day, beside the hub
    tick's. The credentials, the log and the run are FleetTick.ps1's.

.NOTES
    The roots are resolved in the body, not in param defaults, because
    $PSScriptRoot is empty there in an advanced script under -File in
    Windows PowerShell 5.1; run-agent-tick.ps1 carries the incident.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z][a-z0-9-]*$')]
    [string]$Node,

    [switch]$Announce,
    [string]$ApiRoot = '',
    [string]$EnvironmentScript = '',
    [string]$Poetry = 'poetry',
    [string]$LogDirectory = "$env:LOCALAPPDATA\Temp\claude"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'FleetTick.ps1')

if ($ApiRoot -eq '') {
    $ApiRoot = "$PSScriptRoot\..\..\.."
}
$apiRoot = [System.IO.Path]::GetFullPath($ApiRoot)
if ($EnvironmentScript -eq '') {
    $EnvironmentScript = "$apiRoot\tools\hpc-wake\runs\env.ps1"
}
$fleetRoot = Join-Path $apiRoot 'tools\fleet'
# THE ROLLED COMMIT RUNS, NEVER THIS CHECKOUT (board task 465689f5).
# fleet.cli.rolled extracts the commit `make fleet-roll` pointed
# refs/fleet/rolled at, once both execution suites passed there, and runs
# the node agent from it with the extracted fleet.json as its --config; with
# no roll it refuses by name, exit 2, and no agent starts. The '--' after
# 'run' is required: without it poetry parses the whole line itself when it
# holds the launcher's own '--', and refuses with 'The option "-m" does not
# exist' before any agent starts (every tick from 10:12Z on 2026-09-27).
$agentArguments = @(
    'run', '--', 'python', '-m', 'fleet.cli.rolled',
    '--repo-root', $apiRoot,
    '--agent', 'fleet-node-agent',
    '--',
    '--node', $Node
)
if ($Announce) {
    $agentArguments += '--announce'
}
exit (Invoke-FleetTick -EnvironmentScript $EnvironmentScript -Poetry $Poetry -WorkingDirectory $fleetRoot `
    -AgentArguments $agentArguments -LogDirectory $LogDirectory -LogStem "fleet-node-$Node" `
    -Header "node $Node" -RetentionDays 14)
