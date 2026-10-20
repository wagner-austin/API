Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The Windows runner host's two scripts, executed from their committed
# renders, rendered/runner-windows-install.ps1 and
# rendered/runner-windows-audit.ps1 (fleet.core.runner_windows_render and
# fleet.core.runner_windows_base_render, MCPs board task 1e9ab738). sc.exe
# and git are stand-ins passed as -Sc and -Git that record each call: sc
# keeps the service's state in a file beside it, so a stop moves it through
# STOP_PENDING to STOPPED and a start to RUNNING. The registry keys are
# scratch keys under HKCU, the machine variables are written in the Process
# scope, and the runner release is a zip this suite builds and serves as a
# file:// URL, holding a config.cmd stand-in that records its arguments,
# writes .runner beside itself and exits with the case's code. The NuGet
# CPython package (fleet.core.runner_windows_python_render) is a zip served
# the same way, whose python.cmd stand-in, passed as -PythonName, records its
# arguments and writes Scripts\pip.exe as a pip reinstall does. Nothing on
# this machine's services, registry or machine environment changes.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')
    $script:service = 'actions.runner.wagner-austin-chat.loki'
    $script:scratch = 'HKCU:\Software\corvis-runner-windows-test'

    function Initialize-Sc {
        <#
        .SYNOPSIS
            An sc.exe stand-in keeping one service's state in a file.
        .PARAMETER FailureExit
            What `sc failure` exits with.
        .PARAMETER QueryExit
            What `sc query` exits with.
        .PARAMETER QueryState
            The state the service starts in.
        .PARAMETER StartExit
            What `sc start` exits with; it leaves the service RUNNING.
        .PARAMETER StopExit
            What `sc stop` exits with.
        .PARAMETER StopTo
            The state `sc stop` leaves: STOP_PENDING reads STOPPED at the
            query after it, FAIL makes every later query exit 1060.
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
            [int]$StopExit = 0,
            [string]$StopTo = '3  STOP_PENDING',
            [int]$QcExit = 0,
            [string]$StartType = '2   AUTO_START',
            [string]$StartName = 'NT AUTHORITY\SYSTEM',
            [int]$QfailureExit = 0,
            [string]$Action = 'RESTART -- Delay = 60000 milliseconds.'
        )
        # Each concatenation is parenthesized: the comma binds tighter than
        # +, so unbracketed the whole list would join into the first line.
        $sc = Initialize-Batch 'sc' @(
            ('if /i "%1"=="failure" exit /b ' + $FailureExit),
            'if /i "%1"=="start" goto start',
            'if /i "%1"=="stop" goto stop',
            'if /i "%1"=="query" goto query',
            'if /i "%1"=="qc" goto qc',
            'goto qfailure',
            ':start',
            '>"%~dp0state.txt" echo 4  RUNNING',
            ('exit /b ' + $StartExit),
            ':stop',
            ('>"%~dp0state.txt" echo ' + $StopTo),
            ('exit /b ' + $StopExit),
            ':query',
            'set /p STATE=<"%~dp0state.txt"',
            'if "%STATE%"=="FAIL" exit /b 1060',
            'echo SERVICE_NAME: %2',
            'echo         STATE              : %STATE%',
            'if "%STATE%"=="3  STOP_PENDING" (>"%~dp0state.txt" echo 1  STOPPED)',
            ('exit /b ' + $QueryExit),
            ':qc',
            "echo         START_TYPE         : $StartType",
            "echo         SERVICE_START_NAME : $StartName",
            ('exit /b ' + $QcExit),
            ':qfailure',
            "echo         FAILURE_ACTIONS              : $Action",
            ('exit /b ' + $QfailureExit)
        )
        [System.IO.File]::WriteAllText((Join-Path $sc.Directory 'state.txt'), "$QueryState`r`n", [System.Text.Encoding]::ASCII)
        return $sc
    }

    function Initialize-Git {
        <#
        .SYNOPSIS
            A git stand-in answering `config --system --get core.longpaths`
            and `config --system core.longpaths true`.
        .PARAMETER Held
            What the get prints; empty prints nothing, as for a key not set.
        .PARAMETER GetExit
            What the get exits with: 0 when set, 1 when not, as git does.
        .PARAMETER SetExit
            What the set exits with.
        .OUTPUTS
            What Initialize-Batch answers.
        #>
        param([string]$Held = 'true', [int]$GetExit = 0, [int]$SetExit = 0)
        $get = @(':get')
        if ($Held) { $get += "echo $Held" }
        return Initialize-Batch 'git' (@('if "%3"=="--get" goto get', ('exit /b ' + $SetExit)) + $get + @('exit /b ' + $GetExit))
    }

    function Initialize-RegistryScratch {
        <#
        .SYNOPSIS
            Scratch registry keys standing in for the policy's and the file
            system's, under HKCU.
        .PARAMETER Policy
            The ExecutionPolicy value to lay, or empty for none.
        .PARAMETER LongPaths
            The LongPathsEnabled value to lay.
        .OUTPUTS
            PSCustomObject: PolicyKey and FileSystemKey.
        #>
        param([string]$Policy = 'RemoteSigned', [int]$LongPaths = 1)
        $root = Join-Path $script:scratch ([guid]::NewGuid().ToString('N'))
        $policyKey = Join-Path $root 'Policy'
        $fileSystemKey = Join-Path $root 'FileSystem'
        [void](New-Item -Path $policyKey -Force)
        [void](New-Item -Path $fileSystemKey -Force)
        if ($Policy) { Set-ItemProperty -LiteralPath $policyKey -Name ExecutionPolicy -Value $Policy }
        Set-ItemProperty -LiteralPath $fileSystemKey -Name LongPathsEnabled -Value $LongPaths -Type DWord
        return [pscustomobject]@{ PolicyKey = $policyKey; FileSystemKey = $fileSystemKey }
    }

    function Get-CaseVariableName {
        <#
        .SYNOPSIS
            A machine variable for one case, unset, in the Process scope.
        .OUTPUTS
            String: its name.
        #>
        return 'CORVIS_RUNNER_TEST_' + [guid]::NewGuid().ToString('N')
    }

    function Get-Base {
        <#
        .SYNOPSIS
            The base parameters of either script, for a host already laid
            unless a case says otherwise.
        .PARAMETER Keys
            The scratch keys.
        .PARAMETER Git
            The git stand-in.
        .PARAMETER Variable
            One machine variable's name, or empty for none declared.
        .OUTPUTS
            Hashtable: the parameters.
        #>
        param([object]$Keys, [object]$Git, [string]$Variable = '')
        $declared = '[]'
        if ($Variable) { $declared = ConvertTo-Json -InputObject @(@{ Name = $Variable; Value = 'C:\fleet\poetry' }) -Compress }
        return @{ PolicyKey = $Keys.PolicyKey; FileSystemKey = $Keys.FileSystemKey; Git = $Git.Path; VariableScope = 'Process'; MachineEnvironment = $declared }
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

    function Initialize-Package {
        <#
        .SYNOPSIS
            A NuGet CPython package zip for 3.11.9, as a file:// URL whose
            {0} the install fills with the version.
        .PARAMETER NoWheel
            Carry no bundled pip wheel.
        .PARAMETER PipExit
            What its python.cmd exits with.
        .PARAMETER NoScripts
            Make its python.cmd write no Scripts\pip.exe.
        .OUTPUTS
            String: the URL.
        #>
        param([switch]$NoWheel, [int]$PipExit = 0, [switch]$NoScripts)
        $directory = Join-Path $TestDrive ('package-' + [guid]::NewGuid().ToString('N'))
        $content = Join-Path $directory 'content'
        $bundled = Join-Path $content 'tools\Lib\ensurepip\_bundled'
        [void][System.IO.Directory]::CreateDirectory($bundled)
        if (-not $NoWheel) { [System.IO.File]::WriteAllText((Join-Path $bundled 'pip-24.0-py3-none-any.whl'), 'wheel') }
        $lines = @('@echo off', '>>"%~dp0python-calls.txt" echo %*')
        if (-not $NoScripts) { $lines += @('mkdir "%~dp0Scripts"', 'type nul > "%~dp0Scripts\pip.exe"') }
        $lines += "exit /b $PipExit"
        [System.IO.File]::WriteAllText((Join-Path $content 'tools\python.cmd'), ($lines -join "`r`n") + "`r`n", [System.Text.Encoding]::ASCII)
        Compress-Archive -Path (Join-Path $content '*') -DestinationPath (Join-Path $directory 'python-3.11.9.zip')
        return ([Uri]$directory).AbsoluteUri + '/python-{0}.zip'
    }

    function Initialize-Seeded {
        <#
        .SYNOPSIS
            Lay 3.11.9 in a runner's tool cache, as an earlier run's seed did.
        .PARAMETER Root
            The runner's directory.
        #>
        param([string]$Root)
        $python = Join-Path $Root '_work\_tool\Python'
        [void][System.IO.Directory]::CreateDirectory((Join-Path $python '3.11.9\x64\Scripts'))
        [System.IO.File]::WriteAllText((Join-Path $python '3.11.9\x64\Scripts\pip.exe'), '')
        [System.IO.File]::WriteAllText((Join-Path $python '3.11.9\x64.complete'), '')
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
        .PARAMETER Base
            The base parameters; a laid host's when omitted.
        .PARAMETER Python
            The tool cache versions the install declares.
        .PARAMETER PackageUrl
            The NuGet CPython package's URL.
        .OUTPUTS
            String[]: what it printed.
        #>
        param([string]$Root, [object]$Sc, [string]$Url, [hashtable]$Tokens, [hashtable]$Base, [string[]]$Python = @(), [string]$PackageUrl = 'file:///absent-{0}.zip')
        if (-not $Base) { $Base = Get-Base (Initialize-RegistryScratch) (Initialize-Git) }
        $install = @{ Repo = 'wagner-austin/chat'; Name = 'loki'; Labels = 'loki'; Root = $Root; Service = $script:service; Python = @($Python) }
        $installs = ConvertTo-Json -InputObject @($install) -Compress
        $parameters = @{ RunnerZipUrl = $Url; Sc = $Sc.Path; Tokens = $Tokens; Installs = $installs; StopWaitSeconds = 2; PythonPackageUrl = $PackageUrl; PythonName = 'python.cmd' } + $Base
        return [string[]]@(Invoke-Rendered 'runner-windows-install' $parameters)
    }

    function Invoke-Audit {
        <#
        .SYNOPSIS
            Run the committed audit over the example install.
        .PARAMETER Sc
            The sc.exe stand-in.
        .PARAMETER Base
            The base parameters; a laid host's when omitted.
        .PARAMETER Root
            The install's directory; one under TestDrive when omitted.
        .PARAMETER Python
            The tool cache versions the install declares.
        .OUTPUTS
            String[]: its rows.
        #>
        param([object]$Sc, [hashtable]$Base, [string]$Root, [string[]]$Python = @())
        if (-not $Base) { $Base = Get-Base (Initialize-RegistryScratch) (Initialize-Git) }
        if (-not $Root) { $Root = Initialize-Root }
        $installs = ConvertTo-Json -InputObject @(@{ Service = $script:service; Root = $Root; Python = @($Python) }) -Compress
        return [string[]]@(Invoke-Rendered 'runner-windows-audit' (@{ Sc = $Sc.Path; Installs = $installs } + $Base))
    }
}

AfterAll {
    if (Test-Path -LiteralPath $script:scratch) { Remove-Item -LiteralPath $script:scratch -Recurse -Force }
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
    It 'touches nothing on a host laid and a runner in place and running' {
        $root = Initialize-Root -Unpacked -Registered
        $sc = Initialize-Sc
        $git = Initialize-Git
        $variable = Get-CaseVariableName
        [Environment]::SetEnvironmentVariable($variable, 'C:\fleet\poetry', 'Process')
        $printed = Invoke-Install $root $sc 'file:///absent.zip' @{} (Get-Base (Initialize-RegistryScratch) $git $variable)
        $printed | Should -Be @("runner: wagner-austin/chat runner loki runs as $($script:service), restarted on failure")
        Read-CallRecord $sc | Should -Be @(
            "failure $($script:service) reset= 86400 actions= restart/60000/restart/60000/restart/60000",
            "query $($script:service)"
        )
        Read-CallRecord $git | Should -Be @('config --system --get core.longpaths')
    }
    It 'lays a stock host and restarts the running runner, which reads its variables only when it starts' {
        $root = Initialize-Root -Unpacked -Registered
        $sc = Initialize-Sc
        $git = Initialize-Git -Held '' -GetExit 1
        $keys = Initialize-RegistryScratch -Policy '' -LongPaths 0
        $variable = Get-CaseVariableName
        $printed = Invoke-Install $root $sc 'file:///absent.zip' @{} (Get-Base $keys $git $variable)
        $printed | Should -Be @(
            'base: set the LocalMachine execution policy to RemoteSigned',
            'base: enabled Win32 long paths',
            'base: set git core.longpaths for the system',
            "base: set the machine variable $variable",
            "runner: stopped $($script:service) so it starts with the machine variables",
            "runner: wagner-austin/chat runner loki runs as $($script:service), restarted on failure"
        )
        (Get-Item -LiteralPath $keys.PolicyKey).GetValue('ExecutionPolicy') | Should -Be 'RemoteSigned'
        (Get-Item -LiteralPath $keys.FileSystemKey).GetValue('LongPathsEnabled') | Should -Be 1
        [Environment]::GetEnvironmentVariable($variable, 'Process') | Should -Be 'C:\fleet\poetry'
        Read-CallRecord $git | Should -Be @('config --system --get core.longpaths', 'config --system core.longpaths true')
        Read-CallRecord $sc | Should -Be @(
            "failure $($script:service) reset= 86400 actions= restart/60000/restart/60000/restart/60000",
            "query $($script:service)",
            "stop $($script:service)",
            "query $($script:service)",
            "query $($script:service)",
            "start $($script:service)"
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
    It 'stops when git cannot read its system configuration' {
        $base = Get-Base (Initialize-RegistryScratch) (Initialize-Git -Held '' -GetExit 128)
        { Invoke-Install (Initialize-Root -Unpacked -Registered) (Initialize-Sc) 'file:///absent.zip' @{} $base } | Should -Throw 'RUNNER_BASE_FAILED: git config --system --get core.longpaths exited 128'
    }
    It 'stops when git cannot write core.longpaths' {
        $base = Get-Base (Initialize-RegistryScratch) (Initialize-Git -Held '' -GetExit 1 -SetExit 5)
        { Invoke-Install (Initialize-Root -Unpacked -Registered) (Initialize-Sc) 'file:///absent.zip' @{} $base } | Should -Throw 'RUNNER_BASE_FAILED: git config --system core.longpaths exited 5'
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
    It 'stops when a running service it must restart refuses to stop' {
        $base = Get-Base (Initialize-RegistryScratch) (Initialize-Git) (Get-CaseVariableName)
        { Invoke-Install (Initialize-Root -Unpacked -Registered) (Initialize-Sc -StopExit 1051) 'file:///absent.zip' @{} $base } | Should -Throw "RUNNER_SERVICE_FAILED: sc.exe stop $($script:service) exited 1051"
    }
    It 'stops when a service it stopped cannot be read' {
        $base = Get-Base (Initialize-RegistryScratch) (Initialize-Git) (Get-CaseVariableName)
        { Invoke-Install (Initialize-Root -Unpacked -Registered) (Initialize-Sc -StopTo 'FAIL') 'file:///absent.zip' @{} $base } | Should -Throw "RUNNER_SERVICE_FAILED: sc.exe query $($script:service) exited 1060"
    }
    It 'stops when a service it stopped never reads STOPPED' {
        $base = Get-Base (Initialize-RegistryScratch) (Initialize-Git) (Get-CaseVariableName)
        { Invoke-Install (Initialize-Root -Unpacked -Registered) (Initialize-Sc -StopTo '4  RUNNING') 'file:///absent.zip' @{} $base } | Should -Throw "RUNNER_SERVICE_FAILED: $($script:service) did not stop within 2 s"
    }
    It 'seeds each declared Python into the runner tool cache with pip entry points, and marks it complete' {
        $root = Initialize-Root -Unpacked -Registered
        $printed = Invoke-Install $root (Initialize-Sc) 'file:///absent.zip' @{} -Python '3.11.9' -PackageUrl (Initialize-Package)
        $tools = Join-Path $root '_work\_tool'
        $printed | Should -Be @(
            "python: seeded 3.11.9 into $tools for wagner-austin/chat",
            "runner: wagner-austin/chat runner loki runs as $($script:service), restarted on failure"
        )
        $target = Join-Path $tools 'Python\3.11.9\x64'
        Test-Path -LiteralPath (Join-Path $target 'Scripts\pip.exe') | Should -BeTrue
        Test-Path -LiteralPath (Join-Path $tools 'Python\3.11.9\x64.complete') | Should -BeTrue
        $wheel = Join-Path $target 'Lib\ensurepip\_bundled\pip-24.0-py3-none-any.whl'
        Read-CallRecord ([pscustomobject]@{ Record = (Join-Path $target 'python-calls.txt') }) | Should -Be @(
            "-m pip install --force-reinstall --no-deps --no-index --no-warn-script-location --disable-pip-version-check $wheel"
        )
    }
    It 'leaves a version already seeded alone' {
        $root = Initialize-Root -Unpacked -Registered
        Initialize-Seeded $root
        $printed = Invoke-Install $root (Initialize-Sc) 'file:///absent.zip' @{} -Python '3.11.9'
        $printed | Should -Be @("runner: wagner-austin/chat runner loki runs as $($script:service), restarted on failure")
    }
    It 'stops when the package holds no bundled pip wheel' {
        $root = Initialize-Root -Unpacked -Registered
        { Invoke-Install $root (Initialize-Sc) 'file:///absent.zip' @{} -Python '3.11.9' -PackageUrl (Initialize-Package -NoWheel) } | Should -Throw 'RUNNER_PYTHON_SEED_FAILED: 3.11.9 holds 0 bundled pip wheels, not one'
    }
    It 'stops when the pip reinstall fails, with its status' {
        $root = Initialize-Root -Unpacked -Registered
        { Invoke-Install $root (Initialize-Sc) 'file:///absent.zip' @{} -Python '3.11.9' -PackageUrl (Initialize-Package -PipExit 2) } | Should -Throw 'RUNNER_PYTHON_SEED_FAILED: the pip reinstall in 3.11.9 exited 2'
    }
    It 'stops when the pip reinstall writes no pip.exe' {
        $root = Initialize-Root -Unpacked -Registered
        { Invoke-Install $root (Initialize-Sc) 'file:///absent.zip' @{} -Python '3.11.9' -PackageUrl (Initialize-Package -NoScripts) } | Should -Throw 'RUNNER_PYTHON_SEED_FAILED: the pip reinstall in 3.11.9 wrote no Scripts\pip.exe'
    }
}

Describe 'The audit' {
    It 'reads a laid host and a running, automatic SYSTEM service that restarts on failure, its Python seeded, as OK rows' {
        $sc = Initialize-Sc
        $variable = Get-CaseVariableName
        [Environment]::SetEnvironmentVariable($variable, 'C:\fleet\poetry', 'Process')
        $root = Initialize-Root
        Initialize-Seeded $root
        Invoke-Audit $sc (Get-Base (Initialize-RegistryScratch) (Initialize-Git) $variable) $root '3.11.9' | Should -Be @(
            'CHECK execution-policy:LocalMachine:RemoteSigned OK',
            'CHECK long-paths OK',
            'CHECK git-longpaths OK',
            "CHECK machine-env:$variable OK",
            "CHECK service:$($script:service) OK",
            "CHECK account:$($script:service) OK",
            "CHECK recovery:$($script:service) OK",
            "CHECK python:$($script:service):3.11.9 OK"
        )
        Read-CallRecord $sc | Should -Be @("query $($script:service)", "qc $($script:service)", "qfailure $($script:service)")
    }
    It 'reads a stock host and a stopped, manual service with no restarts and no Python as DRIFT rows naming what each said' {
        $sc = Initialize-Sc -QueryState '1  STOPPED' -StartType '3   DEMAND_START' -Action 'NONE'
        $variable = Get-CaseVariableName
        $base = Get-Base (Initialize-RegistryScratch -Policy '' -LongPaths 0) (Initialize-Git -Held '' -GetExit 1) $variable
        $root = Initialize-Root
        $python = Join-Path $root '_work\_tool\Python\3.11.9'
        Invoke-Audit $sc $base $root '3.11.9' | Should -Be @(
            'CHECK execution-policy:LocalMachine:RemoteSigned DRIFT the LocalMachine ExecutionPolicy value is: ',
            'CHECK long-paths DRIFT LongPathsEnabled is: 0',
            'CHECK git-longpaths DRIFT git config --system --get core.longpaths exited 1: ',
            "CHECK machine-env:$variable DRIFT the machine variable holds: ",
            "CHECK service:$($script:service) DRIFT sc.exe query exited 0: SERVICE_NAME: $($script:service) STATE : 1 STOPPED",
            "CHECK account:$($script:service) DRIFT sc.exe qc exited 0: START_TYPE : 3 DEMAND_START SERVICE_START_NAME : NT AUTHORITY\SYSTEM",
            "CHECK recovery:$($script:service) DRIFT sc.exe qfailure exited 0: FAILURE_ACTIONS : NONE",
            "CHECK python:$($script:service):3.11.9 DRIFT the tool cache lacks $python\x64\Scripts\pip.exe or $python\x64.complete"
        )
    }
    It 'reads an automatic service under the runner default account as account drift alone' {
        $rows = Invoke-Audit (Initialize-Sc -StartName 'NT AUTHORITY\NetworkService')
        $rows[3..5] | Should -Be @(
            "CHECK service:$($script:service) OK",
            "CHECK account:$($script:service) DRIFT sc.exe qc exited 0: START_TYPE : 2 AUTO_START SERVICE_START_NAME : NT AUTHORITY\NetworkService",
            "CHECK recovery:$($script:service) OK"
        )
    }
    It 'reads a service sc.exe cannot open as drift on every row, and still exits 0' {
        $rows = Invoke-Audit (Initialize-Sc -QueryExit 1060 -QcExit 1060 -QfailureExit 1060)
        $rows[3] | Should -BeLike "CHECK service:$($script:service) DRIFT sc.exe query exited 1060: *"
        $rows[4] | Should -BeLike "CHECK account:$($script:service) DRIFT sc.exe qc exited 1060: *"
        $rows[5] | Should -BeLike "CHECK recovery:$($script:service) DRIFT sc.exe qfailure exited 1060: *"
    }
}
