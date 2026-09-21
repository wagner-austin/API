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
      * FLEET_MCP_API_KEY -- the live ``mcp-fleet`` container's
        MCP_INTERNAL_KEY. Missing surfaces as the agent's named
        QUEUE_CREDENTIALS_MISSING refusal.
      * TASKBOARD_MCP_API_KEY -- the live ``mcp-taskboard`` container's, for
        the verdict the collect pass posts to the submitting task's thread
        (A3) and for the ``--announce`` check-in. Missing surfaces as
        board-watch's named API_KEY_MISSING.

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
#>

param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z][a-z0-9-]*$')]
    [string]$Node,

    [switch]$Announce
)

$ErrorActionPreference = 'Stop'

$apiRoot = 'C:\Users\Test\PROJECTS\API'

. (Join-Path $apiRoot 'tools\hpc-wake\runs\env.ps1')

# Scoped preference around each native call: with docker down, `docker
# inspect` writes stderr and exits non-zero, and under script-level 'Stop'
# that raises a NativeCommandError HERE, before the agent can issue its
# named refusal (run-agent-tick.ps1 carries the same block for the same
# reason).
$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$containerEnv = docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' mcp-fleet 2>$null
$ErrorActionPreference = $prevEap
$keyLine = @($containerEnv | Where-Object { "$_".StartsWith('MCP_INTERNAL_KEY=') })
if ($keyLine.Count -eq 1) {
    $env:FLEET_MCP_API_KEY = "$($keyLine[0])".Substring('MCP_INTERNAL_KEY='.Length)
}

$ErrorActionPreference = 'Continue'
$boardEnv = docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' mcp-taskboard 2>$null
$ErrorActionPreference = $prevEap
$boardKeyLine = @($boardEnv | Where-Object { "$_".StartsWith('MCP_INTERNAL_KEY=') })
if ($boardKeyLine.Count -eq 1) {
    $env:TASKBOARD_MCP_API_KEY = "$($boardKeyLine[0])".Substring('MCP_INTERNAL_KEY='.Length)
}

$fleetRoot = Join-Path $apiRoot 'tools\fleet'
Set-Location $fleetRoot

# One log file per node per day, beside the hub tick's, with the same
# retention and the same reason: a tick that wrote nothing was found only
# by its absence (run-agent-tick.ps1 carries the incident). Start-Process
# with the streams redirected to FILES because this is an S4U task with no
# console, where a pipe has no EOF and 5.1 wraps native stderr on `*>>`.
$logDirectory = Join-Path $env:LOCALAPPDATA 'Temp\claude'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$retentionDays = 14
Get-ChildItem -Path $logDirectory -Filter "fleet-node-$Node-*.log" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$retentionDays) } |
    Remove-Item -Force
$log = Join-Path $logDirectory ("fleet-node-$Node-" + (Get-Date -Format 'yyyy-MM-dd') + '.log')
$stdoutFile = Join-Path $env:TEMP "fleet-node-$Node-tick-$PID.out"
$stderrFile = Join-Path $env:TEMP "fleet-node-$Node-tick-$PID.err"
$startedAt = Get-Date -Format o
$agentArguments = @(
    'run', 'python', '-m', 'fleet.cli.node_agent',
    '--config', 'fleet.json',
    '--node', $Node
)
if ($Announce) {
    $agentArguments += '--announce'
}
$process = Start-Process -FilePath 'poetry' -ArgumentList $agentArguments `
    -WorkingDirectory $fleetRoot `
    -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
$exitCode = $process.ExitCode
$lines = @("TICK START $startedAt node $Node pid $($process.Id) task-pid $PID")
$lines += Get-Content -Path $stdoutFile -Encoding UTF8
$lines += Get-Content -Path $stderrFile -Encoding UTF8
$lines += "TICK EXIT $exitCode $(Get-Date -Format o)"
[System.IO.File]::AppendAllLines($log, [string[]]$lines, (New-Object System.Text.UTF8Encoding($false)))
Remove-Item -Force $stdoutFile, $stderrFile
exit $exitCode
