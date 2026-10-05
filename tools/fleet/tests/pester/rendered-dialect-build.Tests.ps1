Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The build a Windows node runs for one dispatch, from its committed render
# under rendered/ (fleet.core.windows_build, MCPs board task d69786fa, A2).
# Each case builds an export under $TestDrive and runs the render with
# stand-ins for the install steps and make that record where they ran and
# write to both streams, so what reaches the transcript, the order the steps
# run in and the status written last are all measured. The render runs in
# this process and sets its location and seven environment variables, which
# every case restores. Every step and the recipe are timed as phases (MCPs
# board task 74b13c20): the transcript's fleet-phase lines are compared with
# their stamps and durations read as shapes, and one case reads the stamps
# themselves.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')

    $script:variables = @('npm_config_cache', 'POETRY_CACHE_DIR', 'PLAYWRIGHT_BROWSERS_PATH', 'PYTEST_XDIST_AUTO_NUM_WORKERS', 'CORVIS_FLEET_ELEVATED', 'BOARD_AGENT_LABEL', 'CORVIS_FLEET_CACHE', 'CORVIS_FLEET_WORKSPACE', 'PATH')

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

    # The transcript with each phase line's stamps and duration read as
    # their shapes, so a case states the order and the exit codes exactly.
    function Read-TranscriptShape {
        param([string]$Target)
        return [System.IO.File]::ReadAllLines((Join-Path $Target 'result.txt.log')) |
            ForEach-Object { $_ -replace '\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ', '<utc>' -replace 'after \d+ s', 'after <n> s' }
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

    It 'records itself, runs each install step at the root and the recipe in the project, each as a phase, and writes 0 last' {
        $export = Initialize-Export
        $install = Initialize-Tool 'installer' 0
        $make = Initialize-Tool 'make' 0
        Invoke-Rendered 'dialect-build' @{
            Target = $export.Target; Recipe = $export.Recipe; Workers = 3; CacheRoot = $export.Cache
            Install = @("`"$($install.Path)`" ci", "`"$($install.Path)`" second"); InstallPhases = @('install', 'workspace-build'); Make = $make.Path
        }
        [System.IO.File]::ReadAllText((Join-Path $export.Target 'build.pid')).Trim() | Should -BeExactly "$PID"
        [System.IO.File]::ReadAllLines($install.Record) | Should -Be @("$($export.Target) ci", "$($export.Target) second")
        [System.IO.File]::ReadAllLines($make.Record) | Should -Be @("$($export.Recipe) check")
        Read-TranscriptShape $export.Target | Should -Be @(
            "`$ `"$($install.Path)`" ci", 'fleet-phase install started <utc>', 'installer says out', 'installer says err', 'fleet-phase install ended <utc> after <n> s, exit 0',
            "`$ `"$($install.Path)`" second", 'fleet-phase workspace-build started <utc>', 'installer says out', 'installer says err', 'fleet-phase workspace-build ended <utc> after <n> s, exit 0',
            'fleet-phase check started <utc>', 'make says out', 'make says err', 'fleet-phase check ended <utc> after <n> s, exit 0')
        Read-Result $export.Target | Should -BeExactly '0'
        $env:npm_config_cache | Should -BeExactly "$($export.Cache)/npm"
        $env:POETRY_CACHE_DIR | Should -BeExactly "$($export.Cache)/pypoetry"
        $env:PLAYWRIGHT_BROWSERS_PATH | Should -BeExactly "$($export.Cache)/ms-playwright"
        $env:PYTEST_XDIST_AUTO_NUM_WORKERS | Should -BeExactly '3'
        # The node's cache, for an install step that keeps state (MCPs board task 74b13c20).
        $env:CORVIS_FLEET_CACHE | Should -BeExactly $export.Cache
        $env:CORVIS_FLEET_WORKSPACE | Should -BeExactly 'packages/maketools'
        # The render is the ordinary lane's (MCPs board task a98d7083).
        $env:CORVIS_FLEET_ELEVATED | Should -BeExactly '0'
        # The example job's submitter (MCPs board task 6c4516af A4).
        $env:BOARD_AGENT_LABEL | Should -BeExactly 'opus-example-0929'
    }
    It 'stamps each phase in UTC, ends after it starts, and counts whole seconds between' {
        $export = Initialize-Export
        $make = Initialize-Tool 'make' 0
        $before = [DateTime]::UtcNow.AddSeconds(-1)
        Invoke-Rendered 'dialect-build' @{ Target = $export.Target; Recipe = $export.Recipe; CacheRoot = $export.Cache; Install = @(); InstallPhases = @(); Make = $make.Path }
        $after = [DateTime]::UtcNow.AddSeconds(1)
        $lines = [System.IO.File]::ReadAllLines((Join-Path $export.Target 'result.txt.log'))
        $styles = [Globalization.DateTimeStyles]'AdjustToUniversal, AssumeUniversal'
        $lines[0] -match '^fleet-phase check started (\S+)$' | Should -BeTrue
        $started = [DateTime]::ParseExact($Matches[1], "yyyy-MM-dd'T'HH:mm:ss'Z'", [Globalization.CultureInfo]::InvariantCulture, $styles)
        $lines[-1] -match '^fleet-phase check ended (\S+) after (\d+) s, exit 0$' | Should -BeTrue
        $ended = [DateTime]::ParseExact($Matches[1], "yyyy-MM-dd'T'HH:mm:ss'Z'", [Globalization.CultureInfo]::InvariantCulture, $styles)
        [int]$Matches[2] | Should -BeLessOrEqual ([int]($ended - $started).TotalSeconds + 1)
        $started | Should -BeGreaterOrEqual $before.AddMilliseconds(-$before.Millisecond)
        $ended | Should -BeLessOrEqual $after
        $ended | Should -BeGreaterOrEqual $started
    }
    It 'ends at the first install step that fails, with its status in its phase line, and never runs the recipe' {
        $export = Initialize-Export
        $failing = Initialize-Tool 'installer' 5
        $after = Initialize-Tool 'after' 0
        $make = Initialize-Tool 'make' 0
        Invoke-Rendered 'dialect-build' @{
            Target = $export.Target; Recipe = $export.Recipe; CacheRoot = $export.Cache
            Install = @("`"$($failing.Path)`" ci", "`"$($after.Path)`""); InstallPhases = @('install', 'install'); Make = $make.Path
        }
        Read-Result $export.Target | Should -BeExactly '5'
        Test-Path -LiteralPath $after.Record | Should -BeFalse
        Test-Path -LiteralPath $make.Record | Should -BeFalse
        (Read-TranscriptShape $export.Target)[-1] | Should -BeExactly 'fleet-phase install ended <utc> after <n> s, exit 5'
    }
    It 'writes the status of a recipe that fails' {
        $export = Initialize-Export
        $make = Initialize-Tool 'make' 2
        Invoke-Rendered 'dialect-build' @{ Target = $export.Target; Recipe = $export.Recipe; CacheRoot = $export.Cache; Install = @(); InstallPhases = @(); Make = $make.Path }
        Read-Result $export.Target | Should -BeExactly '2'
        Read-TranscriptShape $export.Target | Should -Be @(
            'fleet-phase check started <utc>', 'make says out', 'make says err', 'fleet-phase check ended <utc> after <n> s, exit 2')
    }
    It 'runs its rendered install step, npm ci, as whichever npm PATH names, in the phase it declares' {
        $export = Initialize-Export
        $npm = Initialize-Tool 'npm' 0
        $make = Initialize-Tool 'make' 0
        $env:PATH = "$($npm.Directory);$env:PATH"
        Invoke-Rendered 'dialect-build' @{ Target = $export.Target; Recipe = $export.Recipe; CacheRoot = $export.Cache; Make = $make.Path }
        [System.IO.File]::ReadAllLines($npm.Record) | Should -Be @("$($export.Target) ci")
        (Read-TranscriptShape $export.Target)[1] | Should -BeExactly 'fleet-phase install started <utc>'
        Read-Result $export.Target | Should -BeExactly '0'
    }
    It 'runs a bash install step as the bash in GitBin, ahead of any bash already on PATH (MCPs daae17f2)' {
        $export = Initialize-Export
        $wsl = Initialize-Tool 'bash' 0
        $git = Initialize-Tool 'bash' 0
        $make = Initialize-Tool 'make' 0
        $env:PATH = "$($wsl.Directory);$env:PATH"
        Invoke-Rendered 'dialect-build' @{
            Target = $export.Target; Recipe = $export.Recipe; CacheRoot = $export.Cache
            Install = @('bash scripts/testdb-setup.sh --port 5432'); InstallPhases = @('test-database'); Make = $make.Path; GitBin = $git.Directory
        }
        [System.IO.File]::ReadAllLines($git.Record) |
            Should -Be @("$($export.Target) scripts/testdb-setup.sh --port 5432")
        Test-Path -LiteralPath $wsl.Record | Should -BeFalse
        $env:PATH | Should -BeLike "$($git.Directory);$($wsl.Directory);*"
        Read-Result $export.Target | Should -BeExactly '0'
    }
    It 'transcribes cmd.exe''s own refusal of a tool that does not exist, and writes its status' {
        # cmd.exe /c exits 1 for a command it cannot find; 9009 is only its
        # ERRORLEVEL inside a session.
        $export = Initialize-Export
        $make = Initialize-Tool 'make' 0
        Invoke-Rendered 'dialect-build' @{ Target = $export.Target; Recipe = $export.Recipe; CacheRoot = $export.Cache; Install = @('fleet-no-such-tool ci'); InstallPhases = @('install'); Make = $make.Path }
        Read-Result $export.Target | Should -BeExactly '1'
        [System.IO.File]::ReadAllText((Join-Path $export.Target 'result.txt.log')) | Should -BeLike "*fleet-no-such-tool*not recognized*"
        Test-Path -LiteralPath $make.Record | Should -BeFalse
    }
}

Describe 'Nothing the build starts outlives it (MCPs e40bca34)' {
    It 'ends an install step''s orphan, which taskkill /T cannot reach, when the stop ends the build, and frees the transcript' {
        # sedona, 2026-10-04: a hung test-database step left two bash.exe
        # whose parent had exited; the stop's taskkill /T walks parent
        # links, never reached them, and they held the transcript for 26
        # hours. Here the install step starts a process with start /b, which
        # inherits the transcript, and exits; the orphan also inherits
        # cmd.exe's output pipe, so the build cannot end on its own while it
        # lives. The build runs as a node runs it, $Target/build.ps1 in its
        # own powershell.exe, so the committed stop render can end it.
        $export = Initialize-Export
        $orphanPid = Join-Path $TestDrive ('orphan-' + [guid]::NewGuid().ToString('N') + '.pid')
        $orphaning = Initialize-Batch 'orphaning' @(
            "start `"`" /b `"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe`" -NoProfile -Command `"`$PID | Set-Content -LiteralPath '$orphanPid'; Start-Sleep 600`"",
            ':wait',
            "if not exist `"$orphanPid`" goto wait",
            'exit /b 0')
        $make = Initialize-Tool 'make' 0
        $build = "$($export.Target)/build.ps1"
        Copy-Item -LiteralPath (Join-Path $script:rendered 'dialect-build.ps1') -Destination $build
        $running = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -PassThru -WindowStyle Hidden -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$build`"", '-Target', "`"$($export.Target)`"", '-Recipe', "`"$($export.Recipe)`"",
            '-CacheRoot', "`"$($export.Cache)`"", '-Install', "`"$($orphaning.Path)`"", '-InstallPhases', 'test-database', '-Make', "`"$($make.Path)`"")
        while (-not (Test-Path -LiteralPath $orphanPid)) {
            $running.HasExited | Should -BeFalse
            Start-Sleep -Milliseconds 200
        }
        $orphan = [int]([System.IO.File]::ReadAllText($orphanPid).Trim())
        $parent = (Get-CimInstance Win32_Process -Filter "ProcessId=$orphan").ParentProcessId
        Get-CimInstance Win32_Process -Filter "ProcessId=$parent" | Should -BeNullOrEmpty
        $global:LASTEXITCODE = 0
        & (Join-Path $script:rendered 'dialect-stop.ps1') -Target $export.Target -TaskName "fleet-pester-absent-$([guid]::NewGuid().ToString('N'))" | Out-Null
        $LASTEXITCODE | Should -Be 0
        $running.WaitForExit()
        # The kernel ends a job's processes as it closes the job's last
        # handle, so the orphan is gone within moments of the build.
        $deadline = (Get-Date).AddSeconds(10)
        while (($null -ne (Get-CimInstance Win32_Process -Filter "ProcessId=$orphan")) -and ((Get-Date) -lt $deadline)) {
            Start-Sleep -Milliseconds 100
        }
        Get-CimInstance Win32_Process -Filter "ProcessId=$orphan" | Should -BeNullOrEmpty
        Move-Item -LiteralPath (Join-Path $export.Target 'result.txt.log') -Destination (Join-Path $TestDrive 'freed.log')
        Read-CallRecord $orphaning | Should -HaveCount 1
    }
}
