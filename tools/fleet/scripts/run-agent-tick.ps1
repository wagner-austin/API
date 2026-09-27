<#
.SYNOPSIS
    One scheduled tick of fleet-agent: resolve credentials fresh, run, exit
    with the agent's own status.

.DESCRIPTION
    The action behind the ``API-FleetAgent-3min`` scheduled task
    (register-agent-schedule.ps1). Everything the agent needs but must not
    store is resolved AT TICK TIME:

      * CORVIS_TENANT_ID  -- dot-sourced from the hpc-wake pump's runs/env.ps1,
        the machine's one home for that value (CI_WAKE_TASK_ID and the
        taskboard key already live there "beside the pump's other
        credentials"). A second copy would be the drift.
      * FLEET_MCP_API_KEY -- fleet-mcp's MCP_INTERNAL_KEY, from the same
        runs/env.ps1. Until 2026-09-25 this script read it from an
        ``mcp-fleet`` container on this host; fleet-mcp runs on diphtheria
        since then (MCPs board tasks 91ca67f4 and 60df277e), that read found
        nothing, and every tick failed. Missing surfaces as the agent's
        named QUEUE_CREDENTIALS_MISSING refusal, not as a silent empty tick.
        The queue's address is not set here either: the agent reads it from
        the MCPs stack's endpoint declaration.
      * TASKBOARD_MCP_API_KEY -- from the same runs/env.ps1, for the
        tick's third pass: the session-ledger observer writes to the BOARD
        (task_session_observe), not the queue, and the two services hold
        different keys. Until 2026-09-24 this script also read it from an
        ``mcp-taskboard`` container on this host; the taskboard runs on
        diphtheria and the hub's forwarder of that name is retired (MCPs
        board task c6fc4882), so that read found nothing and is gone.
        Missing surfaces as board-watch's named API_KEY_MISSING, after the
        queue passes have run.
      * --registry -- fleet-mcp/fleet-nodes.json in the MCPs checkout, the
        list of machines the observer walks. Without it the agent logs
        that observation was skipped, every tick, by design.

    THE RUNNER'S IDENTITY IS FIXED, DELIBERATELY. ``fleet-runner-austinpc``
    with one minted-once UUID is a durable service identity, the same shape
    as the wake bridges' -- a fresh UUID per tick would make ``held_by``
    (which is how a runner recovers jobs across ticks) see a stranger's
    claims.

    THE EXIT CODE IS THE AGENT'S. fleet-agent exits 0 whenever the AGENT
    worked, including refused jobs and failed suites (its module docstring
    carries the argument); a non-zero here therefore means the tick itself
    broke -- credentials, transport, or a records/fleet disagreement -- and
    Task Scheduler's "last result" shows something real changed.

    The credentials, the log and the run are FleetTick.ps1's, with the
    incidents behind each. The roots are parameters, defaulting to the
    checkout this script runs from and the MCPs checkout beside it, so the
    suite runs this entry against a stand-in poetry and a scratch log.

.PARAMETER ApiRoot
    The API checkout; empty means the one this script is in.

.PARAMETER EnvironmentScript
    The hpc-wake pump's runs/env.ps1; empty means the one in ApiRoot.

.PARAMETER Poetry
    The poetry executable.

.PARAMETER LogDirectory
    Where the day's log goes.

.NOTES
    The roots default to empty and are resolved in the body because
    $PSScriptRoot is EMPTY inside a param default of an advanced script
    ([CmdletBinding()]) when Windows PowerShell 5.1 runs it with -File, which
    is how Task Scheduler runs this one (a plain script, or the same script
    run with &, sees it set). The defaults "$PSScriptRoot\..\..\.." and
    "$PSScriptRoot\..\..\hpc-wake\runs\env.ps1" lost their root, the API
    root resolved to C:\, whose empty parent Join-Path refused before the
    tick logged anything, and every
    scheduled tick from 16:27Z on 2026-09-27 exited 1 (MCPs board task
    d69786fa).
#>
[CmdletBinding()]
param(
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
$mcpsRoot = Join-Path (Split-Path -Parent $apiRoot) 'MCPs'
# THE ROLLED COMMIT RUNS, NEVER THIS CHECKOUT (board task 465689f5).
# fleet.cli.rolled extracts the commit `make fleet-roll` pointed
# refs/fleet/rolled at, once both execution suites passed there, and runs
# fleet-agent from it with the extracted fleet.json as its --config; with no
# roll it refuses by name, exit 2, and no agent starts. The '--' after
# 'run' is required: without it poetry parses the whole line itself when it
# holds the launcher's own '--', and refuses with 'The option "-m" does not
# exist' before any agent starts (every tick from 10:12Z on 2026-09-27).
$agentArguments = @(
    'run', '--', 'python', '-m', 'fleet.cli.rolled',
    '--repo-root', $apiRoot,
    '--agent', 'fleet-agent',
    '--',
    '--agent', 'fleet-runner-austinpc',
    '--session', 'a850f688-f98d-415c-a244-e993226ca2fc',
    '--repo-root', $apiRoot,
    '--mcps-root', $mcpsRoot,
    '--registry', (Join-Path $mcpsRoot 'fleet-mcp\fleet-nodes.json')
)
exit (Invoke-FleetTick -EnvironmentScript $EnvironmentScript -Poetry $Poetry `
    -WorkingDirectory (Join-Path $apiRoot 'tools\fleet') -AgentArguments $agentArguments `
    -LogDirectory $LogDirectory -LogStem 'fleet-agent' -Header 'hub' -RetentionDays 14)
