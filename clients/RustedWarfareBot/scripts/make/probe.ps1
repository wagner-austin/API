<#
.SYNOPSIS
    Launch the headless game with the agent attached, for the probe family
    (make sandbox-probe, discover-probe, wire-capture and type-flags).
.DESCRIPTION
    The four differ only in the agent's argument string and where output
    lands, which is exactly two parameters. {OUT} inside -AgentArgs is
    replaced with the absolute form of -Out, because the game runs from its
    own directory and the agent writes where it is told, not where the
    harness stands.
.PARAMETER GameDir
    The game's directory, relative to Root.
.PARAMETER AgentJar
    The agent jar, relative to Root.
.PARAMETER AgentArgs
    The agent's argument string, if any.
.PARAMETER Out
    What {OUT} stands for, relative to Root.
.PARAMETER Log
    The game's log, relative to Root.
.PARAMETER Root
    The RustedWarfareBot directory.
.PARAMETER Java
    The game's java.exe, relative to GameDir.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$GameDir,
    [Parameter(Mandatory = $true)][string]$AgentJar,
    [string]$AgentArgs = '',
    [string]$Out = '',
    [Parameter(Mandatory = $true)][string]$Log,
    [string]$Root = '.',
    [string]$Java = 'jvm64\bin\java.exe'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'RwMake.ps1')

$base = (Resolve-Path -LiteralPath $Root).ProviderPath
$agentArgument = $AgentArgs
if ($Out -ne '') {
    $agentArgument = $AgentArgs.Replace('{OUT}', (Join-Path $base $Out))
}
$javaAgent = "-javaagent:$(Join-Path $base $AgentJar)"
if ($agentArgument -ne '') {
    $javaAgent = "$javaAgent=$agentArgument"
}
$gameRoot = Join-Path $base $GameDir
Push-Location -LiteralPath $gameRoot
try {
    [void](Invoke-RwNative ([System.IO.Path]::Combine($gameRoot, $Java)) @('-Xmx1000M', '-Djava.library.path=.', $javaAgent,
            '-cp', 'game-lib.jar;libs/*', 'com.corrodinggames.rts.java.Main',
            '-nodisplay', '-nosound', '-sandbox', '-width', '800', '-height', '600', '-log', (Join-Path $base $Log)) 'RW_PROBE_FAILED')
} finally {
    Pop-Location
}
