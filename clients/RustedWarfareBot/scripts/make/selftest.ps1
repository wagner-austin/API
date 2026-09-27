<#
.SYNOPSIS
    Compile the agent from source and run its self-checks against the
    pinned game jar (make agent-selftest).
.DESCRIPTION
    Compiles fresh rather than depending on the built jar, deliberately: the
    gate must be runnable while a game holds rw-agent.jar open, and a gate
    that rebuilt the jar would fail for a reason that has nothing to do with
    the code under test.
.PARAMETER Javac
    javac.
.PARAMETER Java
    java.
.PARAMETER GameDir
    The game's directory, relative to Root.
.PARAMETER Root
    The RustedWarfareBot directory.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Javac,
    [Parameter(Mandatory = $true)][string]$Java,
    [Parameter(Mandatory = $true)][string]$GameDir,
    [string]$Root = '.'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'RwMake.ps1')

$base = (Resolve-Path -LiteralPath $Root).ProviderPath
$classesDir = 'agent\build\verify-' + [System.Guid]::NewGuid().ToString('N').Substring(0, 8)
try {
    Invoke-RwAgentCompile $base $Javac '11' $classesDir
    $classPath = "$(Join-Path $base $classesDir);$GameDir/game-lib.jar;$GameDir/libs/*"
    foreach ($line in (Invoke-RwNative $Java @('-cp', $classPath, 'rwbot.agent.SelfTest', "$GameDir/game-lib.jar") 'RW_SELFTEST_FAILED')) {
        Write-Information $line -InformationAction Continue
    }
} finally {
    Remove-RwBuildPath (Join-Path $base $classesDir)
}
