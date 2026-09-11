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
      * FLEET_MCP_API_KEY -- read from the live ``mcp-fleet`` container's
        MCP_INTERNAL_KEY, per queue.load_credentials' own documentation.
        Nothing is written to disk; docker being down surfaces as the
        agent's named QUEUE_CREDENTIALS_MISSING refusal, not as a silent
        empty tick.

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
#>

$ErrorActionPreference = 'Stop'

$apiRoot = 'C:\Users\Test\PROJECTS\API'
$mcpsRoot = 'C:\Users\Test\PROJECTS\MCPs'

. (Join-Path $apiRoot 'tools\hpc-wake\runs\env.ps1')

# Scoped preference around the native call: with docker down, `docker
# inspect` writes stderr and exits non-zero, and under script-level 'Stop'
# that raises a NativeCommandError HERE — before fleet-agent can issue the
# named QUEUE_CREDENTIALS_MISSING refusal the comment below promises. Same
# trap, same fix as register-agent-schedule.ps1's schtasks delete (audit
# 83a7da44, standards arm).
$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$containerEnv = docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' mcp-fleet 2>$null
$ErrorActionPreference = $prevEap
$keyLine = @($containerEnv | Where-Object { "$_".StartsWith('MCP_INTERNAL_KEY=') })
if ($keyLine.Count -eq 1) {
    $env:FLEET_MCP_API_KEY = "$($keyLine[0])".Substring('MCP_INTERNAL_KEY='.Length)
}
# A missing line is NOT patched over: fleet-agent's own
# QUEUE_CREDENTIALS_MISSING names the variable and where it comes from,
# which is a better failure than anything this script could invent.

Set-Location (Join-Path $apiRoot 'tools\fleet')
poetry run fleet-agent --config fleet.json `
    --agent fleet-runner-austinpc `
    --session a850f688-f98d-415c-a244-e993226ca2fc `
    --repo-root $apiRoot `
    --mcps-root $mcpsRoot
exit $LASTEXITCODE
