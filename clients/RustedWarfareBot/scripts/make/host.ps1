<#
.SYNOPSIS
    Host a LAN game the bot plays in, for a human to join (make host).
.DESCRIPTION
    The sparring launcher (wiki: multiplayer-portability-invariants).
    Differs from play.ps1 in exactly the ways a human demands: no -sandbox
    (the agent drives hostStart through the script surface from the menu),
    no lockstep and no settle (a peer cannot be world-held), and the channel
    port opens only once the human has joined and the match started, so the
    wait is lobby-length, not boot-length. The build, the match and the
    cleanup are RwMake.ps1's, shared with watch.ps1.
.PARAMETER Port
    The agent's channel port.
.PARAMETER GameDir
    The game's directory, relative to Root.
.PARAMETER PlayLog
    The game's log, relative to Root.
.PARAMETER HostMap
    The map the agent hosts.
.PARAMETER LobbyTimeoutSeconds
    How long to wait for the human to join.
.PARAMETER Module
    The planner module.
.PARAMETER Catalogue
    The unit catalogue.
.PARAMETER TypeDump
    The type dump.
.PARAMETER PlayArgs
    The planner's further arguments, space-separated.
.PARAMETER Javac
    javac.
.PARAMETER Jar
    jar.
.PARAMETER Root
    The RustedWarfareBot directory.
.PARAMETER Java
    The game's java.exe, relative to GameDir.
.PARAMETER Poetry
    poetry.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][int]$Port,
    [Parameter(Mandatory = $true)][string]$GameDir,
    [Parameter(Mandatory = $true)][string]$PlayLog,
    [Parameter(Mandatory = $true)][string]$HostMap,
    [int]$LobbyTimeoutSeconds = 900,
    [Parameter(Mandatory = $true)][string]$Module,
    [Parameter(Mandatory = $true)][string]$Catalogue,
    [Parameter(Mandatory = $true)][string]$TypeDump,
    [Parameter(Mandatory = $true)][string]$PlayArgs,
    [Parameter(Mandatory = $true)][string]$Javac,
    [Parameter(Mandatory = $true)][string]$Jar,
    [string]$Root = '.',
    [string]$Java = 'jvm64\bin\java.exe',
    [string]$Poetry = 'poetry'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'RwMake.ps1')

$base = (Resolve-Path -LiteralPath $Root).ProviderPath
[void][System.IO.Directory]::CreateDirectory((Join-Path $base 'runs'))
$stamp = [System.Guid]::NewGuid().ToString('N').Substring(0, 8)
$classesDir = "agent\build\host-$stamp"
$agentJar = "agent\build\rw-agent-host-$stamp.jar"
try {
    Invoke-RwAgentCompile $base $Javac '11' $classesDir
    Invoke-RwAgentJar $base $Jar $agentJar $classesDir
    $gameRoot = Join-Path $base $GameDir
    $gameLog = Join-Path $base $PlayLog
    $gameArguments = Get-RwGameArgument -AgentJar (Join-Path $base $agentJar) -AgentArgs "channelPort=$Port;hostMap=$HostMap" `
        -GameLog $gameLog -Display $false -Width 800 -Height 600
    $lan = @(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254*' } |
        ForEach-Object { $_.IPAddress })
    Write-Information "==> JOIN FROM YOUR GAME CLIENT: Multiplayer -> Join by IP -> $($lan -join ' or ') (port 5123). The match starts the moment you join." `
        -InformationAction Continue
    Invoke-RwMatch -Java ([System.IO.Path]::Combine($gameRoot, $Java)) -GameArguments $gameArguments -GameDir $gameRoot -PlayLog $gameLog `
        -Port $Port -WaitSeconds $LobbyTimeoutSeconds -WaitReason 'nobody joined the lobby' -Poetry $Poetry `
        -PlannerArguments (@($Module, "$Port", $Catalogue, $TypeDump) + ($PlayArgs -split ' '))
} finally {
    Remove-RwBuildPath (Join-Path $base $classesDir)
    Remove-RwBuildPath (Join-Path $base $agentJar)
}
