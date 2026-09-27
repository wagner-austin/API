<#
.SYNOPSIS
    Play one match WITH the game window, so a human can watch the bot fight
    (make watch).
.DESCRIPTION
    The spectator launcher: the same agent-driven skirmish as play.ps1,
    with rendering on and the reproducibility machinery off. Lockstep would
    hold frames on the planner's acks and a fixed logic step is pointless
    when the point is to watch. No -nodisplay, because the window is the
    point; sound stays off; no -sandbox, because the agent starts the match
    itself. The build, the match and the cleanup are RwMake.ps1's, shared
    with host.ps1.
.PARAMETER Port
    The agent's channel port.
.PARAMETER GameDir
    The game's directory, relative to Root.
.PARAMETER PlayLog
    The game's log, relative to Root.
.PARAMETER Map
    The match's map.
.PARAMETER Opponents
    How many AI opponents.
.PARAMETER Difficulty
    Their difficulty.
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
.PARAMETER ChannelSeconds
    How long the agent may take to open its channel.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][int]$Port,
    [Parameter(Mandatory = $true)][string]$GameDir,
    [Parameter(Mandatory = $true)][string]$PlayLog,
    [Parameter(Mandatory = $true)][string]$Map,
    [int]$Opponents = 1,
    [int]$Difficulty = 3,
    [Parameter(Mandatory = $true)][string]$Module,
    [Parameter(Mandatory = $true)][string]$Catalogue,
    [Parameter(Mandatory = $true)][string]$TypeDump,
    [Parameter(Mandatory = $true)][string]$PlayArgs,
    [Parameter(Mandatory = $true)][string]$Javac,
    [Parameter(Mandatory = $true)][string]$Jar,
    [string]$Root = '.',
    [string]$Java = 'jvm64\bin\java.exe',
    [string]$Poetry = 'poetry',
    [int]$ChannelSeconds = 120
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'RwMake.ps1')

$base = (Resolve-Path -LiteralPath $Root).ProviderPath
[void][System.IO.Directory]::CreateDirectory((Join-Path $base 'runs'))
$stamp = [System.Guid]::NewGuid().ToString('N').Substring(0, 8)
$classesDir = "agent\build\watch-$stamp"
$agentJar = "agent\build\rw-agent-watch-$stamp.jar"
try {
    Invoke-RwAgentCompile $base $Javac '11' $classesDir
    Invoke-RwAgentJar $base $Jar $agentJar $classesDir
    $gameRoot = Join-Path $base $GameDir
    $gameLog = Join-Path $base $PlayLog
    $agentArgs = "channelPort=$Port;matchMap=$Map;matchOpponents=$Opponents;matchDifficulty=$Difficulty"
    $gameArguments = Get-RwGameArgument -AgentJar (Join-Path $base $agentJar) -AgentArgs $agentArgs `
        -GameLog $gameLog -Display $true -Width 1280 -Height 800
    Write-Information '==> A game window will open; the bot plays, you watch.' -InformationAction Continue
    Invoke-RwMatch -Java ([System.IO.Path]::Combine($gameRoot, $Java)) -GameArguments $gameArguments -GameDir $gameRoot -PlayLog $gameLog `
        -Port $Port -WaitSeconds $ChannelSeconds -WaitReason 'the agent never opened its channel' -Poetry $Poetry `
        -PlannerArguments (@($Module, "$Port", $Catalogue, $TypeDump) + ($PlayArgs -split ' '))
} finally {
    Remove-RwBuildPath (Join-Path $base $classesDir)
    Remove-RwBuildPath (Join-Path $base $agentJar)
}
