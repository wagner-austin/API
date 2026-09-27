<#
.SYNOPSIS
    Relaunch the match-worker fleet, detached from any session
    (make fleet-up).
.DESCRIPTION
    The third Docker-crash recovery made this a procedure worth one command
    (log 2026-08-11): when Postgres drops, every worker dies mid-poll and
    takes its engine with it; the queue rows survive, and the first worker's
    startup reap requeues anything a dead owner held once its heartbeat
    passes the stale threshold. This launcher only starts workers that are
    not already running, so it is safe to run on a half-alive fleet.

    Same detachment and secrecy rules as door.ps1: WMI Create so no console
    handles tie a worker to this terminal, and the database password is
    read from the container at launch, never written to disk.
.PARAMETER Workers
    How many workers the fleet has.
.PARAMETER ClonePool
    The game clones the workers share.
.PARAMETER Root
    The RustedWarfareBot directory.
.PARAMETER Docker
    docker.
.PARAMETER Python
    The python that runs each worker, relative to Root.
.PARAMETER ProcessName
    The workers' image name, for finding the ones already running.
.PARAMETER NamePrefix
    What each worker's name starts with before its number.
.PARAMETER SettleSeconds
    How long to wait before reading which workers run.
#>
[CmdletBinding()]
param(
    [int]$Workers = 8,
    [string]$ClonePool = '0,1,2,3,4,5,6,7',
    [string]$Root = '.',
    [string]$Docker = 'docker',
    [string]$Python = '.venv\Scripts\python.exe',
    [string]$ProcessName = 'python.exe',
    [string]$NamePrefix = 'creepw-',
    [int]$SettleSeconds = 10
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'RwMake.ps1')

$base = (Resolve-Path -LiteralPath $Root).ProviderPath
$dsn = Get-RwDatabaseDsn $Docker
[void][System.IO.Directory]::CreateDirectory((Join-Path $base 'runs'))
$python = [System.IO.Path]::Combine($base, $Python)
$alive = Get-RwWorkerName $ProcessName $NamePrefix
$started = 0
foreach ($n in 1..$Workers) {
    $name = "$NamePrefix$n"
    if ($alive -contains $name) {
        Write-Information "$name already running; leaving it alone" -InformationAction Continue
        continue
    }
    $command = "cd /d $base & $python -u -m scripts.match_worker `"$dsn`" $name $ClonePool >> $base\runs\$name.log 2>&1"
    [void](Invoke-RwDetachedSpawn "cmd.exe /c $command" 'RW_WORKER_SPAWN_FAILED')
    $started++
}
Start-Sleep -Seconds $SettleSeconds
Write-Information "fleet up: started $started, running now: $((Get-RwWorkerName $ProcessName $NamePrefix) -join ', ')" -InformationAction Continue
