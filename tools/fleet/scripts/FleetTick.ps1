<#
.SYNOPSIS
    One scheduled fleet tick: resolve credentials, run the agent, keep its
    log, and answer with the agent's own exit code.
.DESCRIPTION
    The body run-agent-tick.ps1 (the hub lane) and run-node-agent-tick.ps1
    (one node's lane) both run; until 2026-09-27 each carried its own copy
    (MCPs board task d69786fa).

    CREDENTIALS ARE RESOLVED AT TICK TIME from the hpc-wake pump's
    runs/env.ps1, the machine's one home for CORVIS_TENANT_ID,
    FLEET_MCP_API_KEY and TASKBOARD_MCP_API_KEY. A missing key surfaces as
    the agent's own named refusal (QUEUE_CREDENTIALS_MISSING,
    API_KEY_MISSING), not as a silent empty tick. The script is dot-sourced,
    so what it sets in $env: is what the agent inherits.

    THE TICK WRITES A LOG, because until 2026-09-20 it wrote nothing: the
    scheduled action carried no redirection and Task Scheduler's history was
    off on the hub, so a tick that hung for three days (board tasks 35940277
    and 41ac6ed2) was found only by its absence from another log. One file
    per stem per day, both streams with the tick's start, pid and exit;
    files older than the retention are removed at the start of each tick,
    so the directory bounds itself.

    START-PROCESS WITH BOTH STREAMS REDIRECTED TO FILES, not `*>>` or a
    pipe: 5.1 wraps a native command's stderr lines in NativeCommandError
    records on redirection, and a pipe with no console (this is an S4U task)
    is the shape `tasklist | findstr` blocked on. A file has a real end of
    input, and the exit code is read off the process object, which no
    redirection can clobber.

    THE EXIT CODE IS THE AGENT'S. The agents exit 0 whenever they worked,
    refused jobs and failed suites included, so a non-zero means the tick
    itself broke and Task Scheduler's "last result" shows something real.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Remove-FleetTickLog {
    <#
    .SYNOPSIS
        Remove this stem's logs older than the retention.
    .PARAMETER LogDirectory
        Where the logs are.
    .PARAMETER LogStem
        The log name before its date, such as fleet-agent.
    .PARAMETER RetentionDays
        How many days a log is kept.
    .PARAMETER Now
        The time the retention is measured from.
    .OUTPUTS
        Int32: how many logs were removed.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([int])]
    param(
        [Parameter(Mandatory)][string]$LogDirectory,
        [Parameter(Mandatory)][string]$LogStem,
        [Parameter(Mandatory)][int]$RetentionDays,
        [Parameter(Mandatory)][datetime]$Now
    )
    $stale = @(Get-ChildItem -LiteralPath $LogDirectory -Filter "$LogStem-*.log" -File |
        Where-Object { $_.LastWriteTime -lt $Now.AddDays(-$RetentionDays) })
    foreach ($file in $stale) {
        if ($PSCmdlet.ShouldProcess($file.FullName, 'Remove a fleet tick log past its retention')) {
            [System.IO.File]::Delete($file.FullName)
        }
    }
    return $stale.Count
}

function Invoke-FleetTick {
    <#
    .SYNOPSIS
        Run one agent tick and append its record to the day's log.
    .PARAMETER EnvironmentScript
        runs/env.ps1, dot-sourced for the credentials.
    .PARAMETER Poetry
        The poetry executable Start-Process runs.
    .PARAMETER WorkingDirectory
        Where poetry runs: tools\fleet.
    .PARAMETER AgentArguments
        Poetry's arguments.
    .PARAMETER LogDirectory
        Where the day's log goes.
    .PARAMETER LogStem
        The log name before its date.
    .PARAMETER Header
        What the TICK START line names after its time.
    .PARAMETER RetentionDays
        How many days a log is kept.
    .OUTPUTS
        Int32: the agent's exit code.
    #>
    [OutputType([int])]
    param(
        [Parameter(Mandatory)][string]$EnvironmentScript,
        [Parameter(Mandatory)][string]$Poetry,
        [Parameter(Mandatory)][string]$WorkingDirectory,
        [Parameter(Mandatory)][string[]]$AgentArguments,
        [Parameter(Mandatory)][string]$LogDirectory,
        [Parameter(Mandatory)][string]$LogStem,
        [Parameter(Mandatory)][string]$Header,
        [Parameter(Mandatory)][int]$RetentionDays
    )
    . $EnvironmentScript
    [void][System.IO.Directory]::CreateDirectory($LogDirectory)
    $now = Get-Date
    [void](Remove-FleetTickLog -LogDirectory $LogDirectory -LogStem $LogStem -RetentionDays $RetentionDays -Now $now)
    $log = Join-Path $LogDirectory "$LogStem-$($now.ToString('yyyy-MM-dd')).log"
    $stdoutFile = Join-Path $env:TEMP "$LogStem-tick-$PID.out"
    $stderrFile = Join-Path $env:TEMP "$LogStem-tick-$PID.err"
    $startedAt = $now.ToString('o')
    $process = Start-Process -FilePath $Poetry -ArgumentList $AgentArguments -WorkingDirectory $WorkingDirectory `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
    $exitCode = $process.ExitCode
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("TICK START $startedAt $Header pid $($process.Id) task-pid $PID")
    $lines.AddRange([string[]][System.IO.File]::ReadAllLines($stdoutFile, $utf8))
    $lines.AddRange([string[]][System.IO.File]::ReadAllLines($stderrFile, $utf8))
    $lines.Add("TICK EXIT $exitCode $(Get-Date -Format o)")
    [System.IO.File]::AppendAllLines($log, $lines, $utf8)
    [System.IO.File]::Delete($stdoutFile)
    [System.IO.File]::Delete($stderrFile)
    return $exitCode
}
