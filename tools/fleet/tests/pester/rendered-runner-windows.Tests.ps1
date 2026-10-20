Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The Windows runner host's two scripts, executed from their committed
# renders, rendered/runner-windows-install.ps1 and
# rendered/runner-windows-audit.ps1 (fleet.core.runner_windows_render, MCPs
# board task 1e9ab738). sc.exe is a stand-in passed as -Sc that records each
# call and answers each subcommand as the case needs; the runner release is a
# zip this suite builds and serves as a file:// URL, holding a config.cmd
# stand-in that records its arguments, writes .runner beside itself and exits
# with the case's code. Nothing on this machine's services changes.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')
    $script:service = 'actions.runner.wagner-austin-chat.loki'

    function Initialize-Sc {
        <#
        .SYNOPSIS
            An sc.exe stand-in answering failure, query, start, qc and
            qfailure with the case's exit codes and text.
        .PARAMETER FailureExit
            What `sc failure` exits with.
        .PARAMETER QueryExit
            What `sc query` exits with.
        .PARAMETER QueryState
            The STATE word `sc query` prints.
        .PARAMETER StartExit
            What `sc start` exits with.
        .PARAMETER QcExit
            What `sc qc` exits with.
        .PARAMETER StartType
            The START_TYPE `sc qc` prints.
        .PARAMETER StartName
            The SERVICE_START_NAME `sc qc` prints; the service manager keeps
            the name config.cmd gave it, measured on loki 2026-10-20.
        .PARAMETER QfailureExit
            What `sc qfailure` exits with.
        .PARAMETER Action
            The first FAILURE_ACTIONS word `sc qfailure` prints.
        .OUTPUTS
            What Initialize-Batch answers.
        #>
        param(
            [int]$FailureExit = 0,
            [int]$QueryExit = 0,
            [string]$QueryState = '4  RUNNING',
            [int]$StartExit = 0,
            [int]$QcExit = 0,
            [string]$StartType = '2   AUTO_START',
            [string]$StartName = 'NT AUTHORITY\SYSTEM',
            [int]$QfailureExit = 0,
            [string]$Action = 'RESTART -- Delay = 60000 milliseconds.'
        )
        # Each concatenation is parenthesized: the comma binds tighter than
        # +, so unbracketed the whole list would join into the first line.
        return Initialize-Batch 'sc' @(
            ('if /i "%1"=="failure" exit /b ' + $FailureExit),
            ('if /i "%1"=="start" exit /b ' + $StartExit),
            'if /i "%1"=="query" goto query',
            'if /i "%1"=="qc" goto qc',
            'goto qfailure',
            ':query',
            'echo SERVICE_NAME: %2',
            "echo         STATE              : $QueryState",
            ('exit /b ' + $QueryExit),
            ':qc',
            "echo         START_TYPE         : $StartType",
            "echo         SERVICE_START_NAME : $StartName",
            ('exit /b ' + $QcExit),
            ':qfailure',
            "echo         FAILURE_ACTIONS              : $Action",
            ('exit /b ' + $QfailureExit)
        )
    }

    function Initialize-Release {
        <#
        .SYNOPSIS
            A runner release zip holding a config.cmd stand-in, as a file:// URL.
        .PARAMETER ConfigExit
            What its config.cmd exits with.
        .OUTPUTS
            String: the URL.
        #>
        param([int]$ConfigExit)
        $content = Join-Path $TestDrive ('release-' + [guid]::NewGuid().ToString('N'))
        [void][System.IO.Directory]::CreateDirectory($content)
        $lines = @('@echo off', '>>"%~dp0config-calls.txt" echo %*', 'type nul > "%~dp0.runner"', "exit /b $ConfigExit")
        [System.IO.File]::WriteAllText((Join-Path $content 'config.cmd'), ($lines -join "`r`n") + "`r`n", [System.Text.Encoding]::ASCII)
        $zip = "$content.zip"
        Compress-Archive -Path (Join-Path $content '*') -DestinationPath $zip
        return ([Uri]$zip).AbsoluteUri
    }

    function Initialize-Root {
        <#
        .SYNOPSIS
            An install directory under TestDrive, laid out as a case needs.
        .PARAMETER Unpacked
            Lay a config.cmd there, as an earlier run's unpack did.
        .PARAMETER Registered
            Lay .runner there, as an earlier run's config.cmd did.
        .OUTPUTS
            String: the directory.
        #>
        param([switch]$Unpacked, [switch]$Registered)
        $root = Join-Path $TestDrive ('runner-' + [guid]::NewGuid().ToString('N'))
        [void][System.IO.Directory]::CreateDirectory($root)
        if ($Unpacked) { [System.IO.File]::WriteAllText((Join-Path $root 'config.cmd'), "@exit /b 9`r`n") }
        if ($Registered) { [System.IO.File]::WriteAllText((Join-Path $root '.runner'), '{}') }
        return $root
    }

    function Invoke-Install {
        <#
        .SYNOPSIS
            Run the committed install over one install at a root.
        .PARAMETER Root
            The install's directory.
        .PARAMETER Sc
            The sc.exe stand-in.
        .PARAMETER Url
            The release URL.
        .PARAMETER Tokens
            The registration tokens.
        .OUTPUTS
            String[]: what it printed.
        #>
        param([string]$Root, [object]$Sc, [string]$Url, [hashtable]$Tokens)
        $install = @{ Repo = 'wagner-austin/chat'; Name = 'loki'; Labels = 'loki'; Root = $Root; Service = $script:service }
        $installs = ConvertTo-Json -InputObject @($install) -Compress
        return [string[]]@(Invoke-Rendered 'runner-windows-install' @{ RunnerZipUrl = $Url; Sc = $Sc.Path; Tokens = $Tokens; Installs = $installs })
    }
}

Describe 'The install' {
    It 'unpacks the release, registers the runner as SYSTEM, sets its restarts and starts it' {
        $root = Initialize-Root
        $sc = Initialize-Sc -QueryState '1  STOPPED'
        $url = Initialize-Release 0
        $printed = Invoke-Install $root $sc $url @{ 'wagner-austin/chat' = 'TOKEN1' }
        $printed | Should -Be @(
            "runner: unpacked $url into $root",
            'runner: registered wagner-austin/chat runner loki',
            "runner: wagner-austin/chat runner loki runs as $($script:service), restarted on failure"
        )
        Read-CallRecord ([pscustomobject]@{ Record = (Join-Path $root 'config-calls.txt') }) | Should -Be @(
            '--unattended --url https://github.com/wagner-austin/chat --token TOKEN1 --name loki --labels loki --replace --runasservice --windowslogonaccount "NT AUTHORITY\SYSTEM"'
        )
        Read-CallRecord $sc | Should -Be @(
            "failure $($script:service) reset= 86400 actions= restart/60000/restart/60000/restart/60000",
            "query $($script:service)",
            "start $($script:service)"
        )
        Test-Path -LiteralPath (Join-Path $root 'runner.zip') | Should -BeFalse
    }
    It 'touches neither the release nor the registration of a runner already in place, and starts nothing running' {
        $root = Initialize-Root -Unpacked -Registered
        $sc = Initialize-Sc
        $printed = Invoke-Install $root $sc 'file:///absent.zip' @{}
        $printed | Should -Be @("runner: wagner-austin/chat runner loki runs as $($script:service), restarted on failure")
        Read-CallRecord $sc | Should -Be @(
            "failure $($script:service) reset= 86400 actions= restart/60000/restart/60000/restart/60000",
            "query $($script:service)"
        )
    }
    It 'refuses to register a runner it holds no token for' {
        $root = Initialize-Root -Unpacked
        { Invoke-Install $root (Initialize-Sc) 'file:///absent.zip' @{} } | Should -Throw 'RUNNER_TOKEN_MISSING: no registration token for wagner-austin/chat; fleet-runners --rebuild yes mints one'
    }
    It 'stops when config.cmd refuses, with its status' {
        $root = Initialize-Root
        { Invoke-Install $root (Initialize-Sc) (Initialize-Release 3) @{ 'wagner-austin/chat' = 'TOKEN1' } } | Should -Throw 'RUNNER_CONFIG_FAILED: config.cmd for wagner-austin/chat exited 3'
    }
    It 'stops when the recovery actions cannot be set' {
        $root = Initialize-Root -Unpacked -Registered
        { Invoke-Install $root (Initialize-Sc -FailureExit 5) 'file:///absent.zip' @{} } | Should -Throw "RUNNER_SERVICE_FAILED: sc.exe failure $($script:service) exited 5"
    }
    It 'stops when the service does not exist' {
        $root = Initialize-Root -Unpacked -Registered
        { Invoke-Install $root (Initialize-Sc -QueryExit 1060) 'file:///absent.zip' @{} } | Should -Throw "RUNNER_SERVICE_FAILED: sc.exe query $($script:service) exited 1060"
    }
    It 'stops when a stopped service will not start' {
        $root = Initialize-Root -Unpacked -Registered
        { Invoke-Install $root (Initialize-Sc -QueryState '1  STOPPED' -StartExit 1053) 'file:///absent.zip' @{} } | Should -Throw "RUNNER_SERVICE_FAILED: sc.exe start $($script:service) exited 1053"
    }
}

Describe 'The audit' {
    It 'reads a running, automatic SYSTEM service that restarts on failure as three OK rows' {
        $sc = Initialize-Sc
        [string[]]@(Invoke-Rendered 'runner-windows-audit' @{ Sc = $sc.Path }) | Should -Be @(
            "CHECK service:$($script:service) OK",
            "CHECK account:$($script:service) OK",
            "CHECK recovery:$($script:service) OK"
        )
        Read-CallRecord $sc | Should -Be @("query $($script:service)", "qc $($script:service)", "qfailure $($script:service)")
    }
    It 'reads a stopped, manual service with no restarts as three DRIFT rows naming what sc.exe said' {
        $sc = Initialize-Sc -QueryState '1  STOPPED' -StartType '3   DEMAND_START' -Action 'NONE'
        [string[]]@(Invoke-Rendered 'runner-windows-audit' @{ Sc = $sc.Path }) | Should -Be @(
            "CHECK service:$($script:service) DRIFT sc.exe query exited 0: SERVICE_NAME: $($script:service) STATE : 1 STOPPED",
            "CHECK account:$($script:service) DRIFT sc.exe qc exited 0: START_TYPE : 3 DEMAND_START SERVICE_START_NAME : NT AUTHORITY\SYSTEM",
            "CHECK recovery:$($script:service) DRIFT sc.exe qfailure exited 0: FAILURE_ACTIONS : NONE"
        )
    }
    It 'reads an automatic service under the runner default account as account drift alone' {
        $sc = Initialize-Sc -StartName 'NT AUTHORITY\NetworkService'
        [string[]]@(Invoke-Rendered 'runner-windows-audit' @{ Sc = $sc.Path }) | Should -Be @(
            "CHECK service:$($script:service) OK",
            "CHECK account:$($script:service) DRIFT sc.exe qc exited 0: START_TYPE : 2 AUTO_START SERVICE_START_NAME : NT AUTHORITY\NetworkService",
            "CHECK recovery:$($script:service) OK"
        )
    }
    It 'reads a service sc.exe cannot open as drift on every row, and still exits 0' {
        $sc = Initialize-Sc -QueryExit 1060 -QcExit 1060 -QfailureExit 1060
        $rows = [string[]]@(Invoke-Rendered 'runner-windows-audit' @{ Sc = $sc.Path })
        $rows[0] | Should -BeLike "CHECK service:$($script:service) DRIFT sc.exe query exited 1060: *"
        $rows[1] | Should -BeLike "CHECK account:$($script:service) DRIFT sc.exe qc exited 1060: *"
        $rows[2] | Should -BeLike "CHECK recovery:$($script:service) DRIFT sc.exe qfailure exited 1060: *"
    }
}
