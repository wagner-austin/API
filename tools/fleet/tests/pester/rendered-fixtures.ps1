Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# What every suite over tools/fleet's committed renders shares: where the
# copies are, how one is run, and a stand-in for a native program (MCPs board
# task d69786fa, A2). Dot-sourced from each suite's BeforeAll.

$script:rendered = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'rendered'

function Invoke-Rendered {
    <#
    .SYNOPSIS
        Run one committed render in this process.
    .DESCRIPTION
        A render fails by throwing; a native it ran leaves its exit code in
        $LASTEXITCODE, so a non-zero one left behind is itself an error here.
    .PARAMETER Name
        The render's file stem under rendered/.
    .PARAMETER Parameters
        The render's parameters.
    #>
    param([string]$Name, [hashtable]$Parameters)
    $global:LASTEXITCODE = 0
    & (Join-Path $script:rendered "$Name.ps1") @Parameters
    if ($LASTEXITCODE -ne 0) {
        throw "TEST_RENDER_EXITED: $Name exited $LASTEXITCODE"
    }
}

function Initialize-StandIn {
    <#
    .SYNOPSIS
        A .cmd that records its arguments and exits with a chosen code.
    .PARAMETER ExitCode
        What it exits with.
    .OUTPUTS
        PSCustomObject: Path (the .cmd) and Record (where its arguments land).
    #>
    param([int]$ExitCode)
    $root = Join-Path $TestDrive ('standin-' + [guid]::NewGuid().ToString('N'))
    [void][System.IO.Directory]::CreateDirectory($root)
    $record = Join-Path $root 'arguments.txt'
    $path = Join-Path $root 'tool.cmd'
    [System.IO.File]::WriteAllText($path, "@echo off`r`necho %*>`"$record`"`r`nexit /b $ExitCode`r`n", [System.Text.Encoding]::ASCII)
    return [pscustomobject]@{ Path = $path; Record = $record }
}
