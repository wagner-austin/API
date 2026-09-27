<#
.SYNOPSIS
    Launch the match service's HTTP door, detached from any session
    (make door).
.DESCRIPTION
    The door is the queue's one submission surface (wiki:
    harness-match-service): workers poll Postgres directly and survive
    without it, but nothing can submit, reprioritize or retry until it is
    back. Running it as a terminal's child ties the control plane to that
    terminal's life; this launcher starts it detached through WMI, with its
    output in a real log, so the door outlives whoever started it.

    The database password is read from the container at launch and never
    written to disk: the DSN exists only in the door process's memory and
    command line (Get-RwDatabaseDsn).
.PARAMETER Port
    The door's port.
.PARAMETER DoorLog
    The door's log, relative to Root.
.PARAMETER Root
    The RustedWarfareBot directory.
.PARAMETER Docker
    docker.
.PARAMETER Python
    The python that runs the door.
.PARAMETER ReadySeconds
    How long the door may take to listen.
#>
[CmdletBinding()]
param(
    [int]$Port = 27501,
    [string]$DoorLog = 'runs/door.log',
    [string]$Root = '.',
    [string]$Docker = 'docker',
    [string]$Python = 'python',
    [int]$ReadySeconds = 30
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'RwMake.ps1')

$base = (Resolve-Path -LiteralPath $Root).ProviderPath
$held = @(Get-RwListener $Port)
if ($held.Count -gt 0) {
    throw "RW_DOOR_ALREADY_UP: the door is already listening on $Port (pid $($held -join ', ')); stop it first"
}
$dsn = Get-RwDatabaseDsn $Docker
[void][System.IO.Directory]::CreateDirectory((Join-Path $base 'runs'))
$log = Join-Path $base $DoorLog
$commandLine = "cmd.exe /c cd /d $base & set PYTHONPATH=$base\src & " +
    "$Python -u -m scripts.match_service `"$dsn`" >> $log 2>> $log.err"
$spawned = Invoke-RwDetachedSpawn $commandLine 'RW_DOOR_SPAWN_FAILED'
Wait-RwListener $Port $ReadySeconds 'RW_DOOR_NOT_LISTENING' "the door spawned as pid $spawned; see $DoorLog.err"
Write-Information "door up: pid $(@(Get-RwListener $Port) -join ', '), http://127.0.0.1:$Port/, log $DoorLog" -InformationAction Continue
