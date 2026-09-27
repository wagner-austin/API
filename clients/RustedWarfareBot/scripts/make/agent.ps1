<#
.SYNOPSIS
    Build the agent jar, atomically (make agent).
.DESCRIPTION
    Two subtleties both live in the final move. The jar is assembled under a
    temporary name so a half-written build can never be the jar a game
    attaches, and the rename is a terminating error because a jar held open
    by a running JVM fails the move non-terminatingly otherwise: PowerShell
    printed the error, skipped the catch, and the target reported a
    successful build over a jar it never replaced. Observed live.
.PARAMETER Javac
    javac.
.PARAMETER Jar
    jar.
.PARAMETER AgentJar
    The jar to replace, relative to Root.
.PARAMETER Release
    The bytecode level. Required, never defaulted: it decides whether the
    agent can load into the Linux depot's JRE 8 at all, and a default here
    would be a second answer to a question rw_bot.harness.agent_build
    already answers. It was hardcoded to 11 while that said 8, and the jar
    this script writes is the one that ships.
.PARAMETER Root
    The RustedWarfareBot directory.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Javac,
    [Parameter(Mandatory = $true)][string]$Jar,
    [Parameter(Mandatory = $true)][string]$AgentJar,
    [Parameter(Mandatory = $true)][string]$Release,
    [string]$Root = '.'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'RwMake.ps1')

$base = (Resolve-Path -LiteralPath $Root).ProviderPath
$stamp = [System.Guid]::NewGuid().ToString('N').Substring(0, 8)
$classesDir = "agent\build\classes-$stamp"
$tmpJar = "$AgentJar.$stamp.new"
try {
    Invoke-RwAgentCompile $base $Javac $Release $classesDir
    Invoke-RwAgentJar $base $Jar $tmpJar $classesDir
    try {
        Move-Item -Force -LiteralPath (Join-Path $base $tmpJar) -Destination (Join-Path $base $AgentJar)
    } catch {
        throw "RW_AGENT_JAR_LOCKED: cannot replace ${AgentJar}: a JVM has it attached with -javaagent. Stop the running game, then retry. ($($_.Exception.Message))"
    }
} finally {
    Remove-RwBuildPath (Join-Path $base $classesDir)
    Remove-RwBuildPath (Join-Path $base $tmpJar)
}
