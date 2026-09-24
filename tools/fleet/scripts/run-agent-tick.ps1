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

# THE TICK WRITES A LOG, BECAUSE UNTIL 2026-09-20 IT WROTE NOTHING. The
# scheduled action carried no redirection, the agent logs to its streams,
# and Task Scheduler's own history was disabled on the hub, so a tick that
# hung for three days (board tasks 35940277 and 41ac6ed2) left no record
# of itself anywhere: it was found by its absence from the fleet
# container's log. One file per day under the same directory the
# supervisor and manager-audit passes log to; both streams land in it
# with the tick's start, pid and exit; files older than the retention are
# removed at the start of each tick, so the directory bounds itself.
#
# Start-Process with the streams redirected to FILES, not `*>>` or a pipe:
# Windows PowerShell 5.1 wraps a native command's stderr lines in
# NativeCommandError records on redirection, and a pipe with no console
# (this is an S4U task) is the shape `tasklist | findstr` blocked on
# (memory: powershell-deadlocks-under-s4u-tasks). A redirected file handle
# has a real EOF. The exit code is read off the process object, which no
# redirection can clobber.
$logDirectory = Join-Path $env:LOCALAPPDATA 'Temp\claude'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$retentionDays = 14
Get-ChildItem -Path $logDirectory -Filter 'fleet-agent-*.log' |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$retentionDays) } |
    Remove-Item -Force
$log = Join-Path $logDirectory ("fleet-agent-" + (Get-Date -Format 'yyyy-MM-dd') + '.log')
$stdoutFile = Join-Path $env:TEMP "fleet-agent-tick-$PID.out"
$stderrFile = Join-Path $env:TEMP "fleet-agent-tick-$PID.err"
$startedAt = Get-Date -Format o
$agentArguments = @(
    'run', 'fleet-agent', '--config', 'fleet.json',
    '--agent', 'fleet-runner-austinpc',
    '--session', 'a850f688-f98d-415c-a244-e993226ca2fc',
    '--repo-root', $apiRoot,
    '--mcps-root', $mcpsRoot,
    '--registry', (Join-Path $mcpsRoot 'fleet-mcp\fleet-nodes.json')
)
$process = Start-Process -FilePath 'poetry' -ArgumentList $agentArguments `
    -WorkingDirectory (Join-Path $apiRoot 'tools\fleet') `
    -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
$exitCode = $process.ExitCode
$lines = @("TICK START $startedAt pid $($process.Id) task-pid $PID")
$lines += Get-Content -Path $stdoutFile -Encoding UTF8
$lines += Get-Content -Path $stderrFile -Encoding UTF8
$lines += "TICK EXIT $exitCode $(Get-Date -Format o)"
[System.IO.File]::AppendAllLines($log, [string[]]$lines, (New-Object System.Text.UTF8Encoding($false)))
Remove-Item -Force $stdoutFile, $stderrFile
exit $exitCode
