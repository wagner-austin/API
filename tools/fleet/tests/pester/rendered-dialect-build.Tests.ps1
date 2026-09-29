Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The build a Windows node runs for one dispatch, from its committed render
# under rendered/ (fleet.core.windows_build, MCPs board task d69786fa, A2).
# Each case builds an export under $TestDrive and runs the render with
# stand-ins for the install steps and make that record where they ran and
# write to both streams, so what reaches the transcript, the order the steps
# run in and the status written last are all measured. The render runs in
# this process and sets its location and six environment variables, which
# every case restores.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')

    $script:variables = @('npm_config_cache', 'POETRY_CACHE_DIR', 'PLAYWRIGHT_BROWSERS_PATH', 'PYTEST_XDIST_AUTO_NUM_WORKERS', 'CORVIS_FLEET_ELEVATED', 'BOARD_AGENT_LABEL', 'PATH')

    # A .cmd that records the directory it ran in and its arguments, writes
    # one line to each stream, and exits with the case's code. It lands in a
    # directory of its own, so that directory can go first on PATH.
    function Initialize-Tool {
        param([string]$Name, [int]$ExitCode)
        $root = Join-Path $TestDrive ('tool-' + [guid]::NewGuid().ToString('N'))
        [void][System.IO.Directory]::CreateDirectory($root)
        $record = Join-Path $root 'record.txt'
        $path = Join-Path $root "$Name.cmd"
        $body = "@echo off`r`necho %CD% %*>>`"$record`"`r`necho $Name says out`r`n1>&2 echo $Name says err`r`nexit /b $ExitCode`r`n"
        [System.IO.File]::WriteAllText($path, $body, [System.Text.Encoding]::ASCII)
        return [pscustomobject]@{ Path = $path; Record = $record; Directory = $root }
    }

    function Initialize-Export {
        $target = Join-Path $TestDrive ('run-' + [guid]::NewGuid().ToString('N'))
        $recipe = Join-Path $target 'packages\x'
        [void][System.IO.Directory]::CreateDirectory($recipe)
        return [pscustomobject]@{ Target = $target; Recipe = $recipe; Cache = (Join-Path $TestDrive 'cache') }
    }

    function Read-Result {
        param([string]$Target)
        return [System.IO.File]::ReadAllText((Join-Path $Target 'result.txt')).Trim()
    }
}

Describe 'The build' {
    BeforeEach {
        Push-Location
        $script:saved = @{}
        foreach ($name in $script:variables) {
            $script:saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        }
    }
    AfterEach {
        Pop-Location
        foreach ($name in $script:variables) {
            [Environment]::SetEnvironmentVariable($name, $script:saved[$name], 'Process')
        }
    }

    It 'records itself, runs each install step at the root and the recipe in the project, transcribes both streams, and writes 0 last' {
        $export = Initialize-Export
        $install = Initialize-Tool 'installer' 0
        $make = Initialize-Tool 'make' 0
        Invoke-Rendered 'dialect-build' @{
            Target = $export.Target; Recipe = $export.Recipe; Workers = 3; CacheRoot = $export.Cache
            Install = @("`"$($install.Path)`" ci", "`"$($install.Path)`" second"); Make = $make.Path
        }
        [System.IO.File]::ReadAllText((Join-Path $export.Target 'build.pid')).Trim() | Should -BeExactly "$PID"
        [System.IO.File]::ReadAllLines($install.Record) | Should -Be @("$($export.Target) ci", "$($export.Target) second")
        [System.IO.File]::ReadAllLines($make.Record) | Should -Be @("$($export.Recipe) check")
        [System.IO.File]::ReadAllLines((Join-Path $export.Target 'result.txt.log')) | Should -Be @(
            "`$ `"$($install.Path)`" ci", 'installer says out', 'installer says err',
            "`$ `"$($install.Path)`" second", 'installer says out', 'installer says err',
            'make says out', 'make says err')
        Read-Result $export.Target | Should -BeExactly '0'
        $env:npm_config_cache | Should -BeExactly "$($export.Cache)/npm"
        $env:POETRY_CACHE_DIR | Should -BeExactly "$($export.Cache)/pypoetry"
        $env:PLAYWRIGHT_BROWSERS_PATH | Should -BeExactly "$($export.Cache)/ms-playwright"
        $env:PYTEST_XDIST_AUTO_NUM_WORKERS | Should -BeExactly '3'
        # The render is the ordinary lane's (MCPs board task a98d7083).
        $env:CORVIS_FLEET_ELEVATED | Should -BeExactly '0'
        # The example job's submitter (MCPs board task 6c4516af A4).
        $env:BOARD_AGENT_LABEL | Should -BeExactly 'opus-example-0929'
    }
    It 'ends at the first install step that fails, with its status, and never runs the recipe' {
        $export = Initialize-Export
        $failing = Initialize-Tool 'installer' 5
        $after = Initialize-Tool 'after' 0
        $make = Initialize-Tool 'make' 0
        Invoke-Rendered 'dialect-build' @{
            Target = $export.Target; Recipe = $export.Recipe; CacheRoot = $export.Cache
            Install = @("`"$($failing.Path)`" ci", "`"$($after.Path)`""); Make = $make.Path
        }
        Read-Result $export.Target | Should -BeExactly '5'
        Test-Path -LiteralPath $after.Record | Should -BeFalse
        Test-Path -LiteralPath $make.Record | Should -BeFalse
    }
    It 'writes the status of a recipe that fails' {
        $export = Initialize-Export
        $make = Initialize-Tool 'make' 2
        Invoke-Rendered 'dialect-build' @{ Target = $export.Target; Recipe = $export.Recipe; CacheRoot = $export.Cache; Install = @(); Make = $make.Path }
        Read-Result $export.Target | Should -BeExactly '2'
        [System.IO.File]::ReadAllLines((Join-Path $export.Target 'result.txt.log')) | Should -Be @('make says out', 'make says err')
    }
    It 'runs its rendered install step, npm ci, as whichever npm PATH names' {
        $export = Initialize-Export
        $npm = Initialize-Tool 'npm' 0
        $make = Initialize-Tool 'make' 0
        $env:PATH = "$($npm.Directory);$env:PATH"
        Invoke-Rendered 'dialect-build' @{ Target = $export.Target; Recipe = $export.Recipe; CacheRoot = $export.Cache; Make = $make.Path }
        [System.IO.File]::ReadAllLines($npm.Record) | Should -Be @("$($export.Target) ci")
        Read-Result $export.Target | Should -BeExactly '0'
    }
    It 'transcribes cmd.exe''s own refusal of a tool that does not exist, and writes its status' {
        # cmd.exe /c exits 1 for a command it cannot find; 9009 is only its
        # ERRORLEVEL inside a session.
        $export = Initialize-Export
        $make = Initialize-Tool 'make' 0
        Invoke-Rendered 'dialect-build' @{ Target = $export.Target; Recipe = $export.Recipe; CacheRoot = $export.Cache; Install = @('fleet-no-such-tool ci'); Make = $make.Path }
        Read-Result $export.Target | Should -BeExactly '1'
        [System.IO.File]::ReadAllText((Join-Path $export.Target 'result.txt.log')) | Should -BeLike "*fleet-no-such-tool*not recognized*"
        Test-Path -LiteralPath $make.Record | Should -BeFalse
    }
}
