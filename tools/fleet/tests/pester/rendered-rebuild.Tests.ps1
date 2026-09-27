Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The rebuild's three small host scripts, executed from their committed
# copies under rendered/ (fleet.core.rendered_powershell, MCPs board task
# d69786fa, A2). shutdown.exe and wsl.exe are stand-ins that record their
# arguments and exit with the case's code, passed through the parameters the
# renders name them by, so nothing here restarts this machine or stops a
# distro. The boot instant is a read and runs for real.

BeforeAll {
    $script:rendered = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'rendered'

    # A render fails by throwing; a stand-in's exit code is left in
    # $LASTEXITCODE, so every run goes through here, where a non-zero one is
    # itself an error.
    function Invoke-Rendered {
        param([string]$Name, [hashtable]$Parameters)
        $global:LASTEXITCODE = 0
        & (Join-Path $script:rendered "$Name.ps1") @Parameters
        if ($LASTEXITCODE -ne 0) {
            throw "TEST_RENDER_EXITED: $Name exited $LASTEXITCODE"
        }
    }

    function Initialize-StandIn {
        param([int]$ExitCode)
        $root = Join-Path $TestDrive ('standin-' + [guid]::NewGuid().ToString('N'))
        [void][System.IO.Directory]::CreateDirectory($root)
        $record = Join-Path $root 'arguments.txt'
        $path = Join-Path $root 'tool.cmd'
        [System.IO.File]::WriteAllText($path, "@echo off`r`necho %*>`"$record`"`r`nexit /b $ExitCode`r`n", [System.Text.Encoding]::ASCII)
        return [pscustomobject]@{ Path = $path; Record = $record }
    }
}

Describe 'The boot instant' {
    It 'prints this machine''s last boot in UTC, round-trip formatted, before now' {
        $said = @(Invoke-Rendered 'rebuild-boot-instant' @{})
        $said.Count | Should -Be 1
        $instant = [datetime]::ParseExact($said[0], 'o', [cultureinfo]::InvariantCulture, [System.Globalization.DateTimeStyles]::RoundtripKind)
        $instant.Kind | Should -Be 'Utc'
        $instant | Should -BeLessThan ([datetime]::UtcNow)
    }
}

Describe 'The restart' {
    It 'asks shutdown.exe for a restart in ten seconds, commented as the rebuild' {
        $shutdown = Initialize-StandIn 0
        Invoke-Rendered 'rebuild-restart' @{ Shutdown = $shutdown.Path }
        [System.IO.File]::ReadAllText($shutdown.Record).Trim() | Should -BeExactly '/r /t 10 /c "fleet-runners --rebuild"'
    }
    It 'refuses by name a restart shutdown.exe refuses' {
        $shutdown = Initialize-StandIn 1190
        { Invoke-Rendered 'rebuild-restart' @{ Shutdown = $shutdown.Path } } |
            Should -Throw "FLEET_RESTART_REFUSED: $($shutdown.Path) exited 1190"
    }
    It 'runs as a host runs it, with -File and no arguments but the stand-in' {
        $shutdown = Initialize-StandIn 0
        & "$PSHOME\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File (Join-Path $script:rendered 'rebuild-restart.ps1') -Shutdown $shutdown.Path | Out-Null
        $LASTEXITCODE | Should -Be 0
        [System.IO.File]::ReadAllText($shutdown.Record).Trim() | Should -BeExactly '/r /t 10 /c "fleet-runners --rebuild"'
    }
}

Describe 'The distro terminate' {
    It 'stops the roster''s distro' {
        $wsl = Initialize-StandIn 0
        Invoke-Rendered 'rebuild-terminate-lavender' @{ Wsl = $wsl.Path }
        [System.IO.File]::ReadAllText($wsl.Record).Trim() | Should -BeExactly '--terminate Ubuntu'
    }
    It 'refuses by name a terminate wsl.exe refuses' {
        $wsl = Initialize-StandIn 1
        { Invoke-Rendered 'rebuild-terminate-lavender' @{ Wsl = $wsl.Path } } |
            Should -Throw "FLEET_TERMINATE_REFUSED: $($wsl.Path) --terminate Ubuntu exited 1"
    }
}
