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

function Get-RenderedDefault {
    <#
    .SYNOPSIS
        A default from a render's param block, evaluated as the literal it
        is, so a suite reads the roster's values rather than restating them.
    .PARAMETER Name
        The render's file stem under rendered/.
    .PARAMETER Parameter
        The parameter's name, without the $.
    .OUTPUTS
        The default's value: a string, or an array of them.
    #>
    param([string]$Name, [string]$Parameter)
    $ast = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $script:rendered "$Name.ps1"), [ref]$null, [ref]$null)
    $found = @($ast.ParamBlock.Parameters | Where-Object { $_.Name.VariablePath.UserPath -eq $Parameter })
    return $found[0].DefaultValue.SafeGetValue()
}

function Initialize-Batch {
    <#
    .SYNOPSIS
        A .cmd that records each call's arguments, one line each, then runs
        the case's own batch lines.
    .PARAMETER Name
        The file's stem, which is the tool it stands in for.
    .PARAMETER Body
        Batch lines run after the record, ending in the exit the case wants.
    .OUTPUTS
        PSCustomObject: Path (the .cmd), Record (its calls) and Directory.
    #>
    param([string]$Name, [string[]]$Body)
    $root = Join-Path $TestDrive ('batch-' + [guid]::NewGuid().ToString('N'))
    [void][System.IO.Directory]::CreateDirectory($root)
    $record = Join-Path $root 'calls.txt'
    $path = Join-Path $root "$Name.cmd"
    # The redirection comes first: after an argument list ending in a digit,
    # as '--version 2' does, 'echo %*>>' would read '2>>' as a redirection
    # of stderr and let the line through to the caller's output.
    $lines = @('@echo off', ">>`"$record`" echo %*") + $Body
    [System.IO.File]::WriteAllText($path, ($lines -join "`r`n") + "`r`n", [System.Text.Encoding]::ASCII)
    return [pscustomobject]@{ Path = $path; Record = $record; Directory = $root }
}

function Read-CallRecord {
    <#
    .SYNOPSIS
        The calls a stand-in recorded, one argument line each; none when it
        was never run.
    .PARAMETER StandIn
        What Initialize-Batch or Initialize-StandIn answered.
    .OUTPUTS
        String[].
    #>
    param([object]$StandIn)
    if (-not (Test-Path -LiteralPath $StandIn.Record)) {
        return [string[]]@()
    }
    return [string[]]@([System.IO.File]::ReadAllLines($StandIn.Record) | ForEach-Object { $_.Trim() })
}

function Initialize-StandIn {
    <#
    .SYNOPSIS
        A .cmd that records its arguments and exits with a chosen code.
    .DESCRIPTION
        The redirection comes BEFORE the echo. Written after it, as
        'echo %*>>"file"', a call whose last argument is one digit (git's
        'config gc.auto 0', MCPs board task 939ec5c7) reads as '0>>"file"',
        a redirect of handle 0, and its line never reaches the record.
    .PARAMETER ExitCode
        What it exits with.
    .PARAMETER Append
        Record every call, one line each, instead of the last one only.
    .OUTPUTS
        PSCustomObject: Path (the .cmd) and Record (where its arguments land).
    #>
    param([int]$ExitCode, [switch]$Append)
    $root = Join-Path $TestDrive ('standin-' + [guid]::NewGuid().ToString('N'))
    [void][System.IO.Directory]::CreateDirectory($root)
    $record = Join-Path $root 'arguments.txt'
    $path = Join-Path $root 'tool.cmd'
    $redirect = @{ $true = '>>'; $false = '>' }[$Append.IsPresent]
    [System.IO.File]::WriteAllText($path, "@echo off`r`n$redirect`"$record`" echo %*`r`nexit /b $ExitCode`r`n", [System.Text.Encoding]::ASCII)
    return [pscustomobject]@{ Path = $path; Record = $record }
}
