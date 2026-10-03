Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The Windows side of a runner host's provision, from each roster host's
# committed provision-windows-<host> and onboard-windows-<host> renders
# (fleet.core.runner_windows_provision, MCPs board task d69786fa, A2). Each
# case passes installs of its own under TestDrive: the runner release and
# the NuGet Python package are file:// archives built here, holding a
# config.cmd and a python.cmd stand-in; sc.exe and wsl.exe are stand-in batch
# files; and the three service acts are script blocks over a case's own
# service state, so no service on this machine is touched. The provision's
# keepalive is registered for real under a minted name, read back through
# Task Scheduler's COM service, and deleted afterwards.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')
    . (Join-Path $PSScriptRoot 'task-fixtures.ps1')

    # A runner release whose config.cmd records its arguments and either
    # writes .runner or refuses with exit 1.
    function Initialize-RunnerRelease {
        param([switch]$Refuse)
        $source = Join-Path $TestDrive ('release-' + [guid]::NewGuid().ToString('N'))
        [void][System.IO.Directory]::CreateDirectory($source)
        $body = @('@echo off', '>>"%~dp0config-calls.txt" echo %*')
        if ($Refuse) {
            $body += 'exit /b 1'
        } else {
            $body += @('type nul > "%~dp0.runner"', 'exit /b 0')
        }
        [System.IO.File]::WriteAllText((Join-Path $source 'config.cmd'), ($body -join "`r`n") + "`r`n", [System.Text.Encoding]::ASCII)
        $zip = Join-Path $source 'runner.zip'
        Compress-Archive -Path (Join-Path $source 'config.cmd') -DestinationPath $zip
        return ([uri]$zip).AbsoluteUri
    }

    # A NuGet Python package for every version, as a URL template: a
    # python.cmd that records its arguments and, per variant, writes
    # Scripts\pip.exe, exits 1, or writes nothing; and one bundled pip wheel,
    # or none.
    function Initialize-PythonPackage {
        param([ValidateSet('Good', 'NoWheel', 'PipFails', 'NoPip')][string]$Variant = 'Good')
        $root = Join-Path $TestDrive ('python-' + [guid]::NewGuid().ToString('N'))
        $tools = Join-Path $root 'source\tools'
        [void][System.IO.Directory]::CreateDirectory((Join-Path $tools 'Lib\ensurepip\_bundled'))
        if ($Variant -ne 'NoWheel') {
            [System.IO.File]::WriteAllText((Join-Path $tools 'Lib\ensurepip\_bundled\pip-24.0-py3-none-any.whl'), 'wheel')
        }
        $body = @('@echo off', '>>"%~dp0python-calls.txt" echo %*')
        $body += @{
            Good = @('if not exist "%~dp0Scripts" mkdir "%~dp0Scripts"', 'type nul > "%~dp0Scripts\pip.exe"', 'exit /b 0')
            NoWheel = @('exit /b 0')
            PipFails = @('exit /b 1')
            NoPip = @('exit /b 0')
        }[$Variant]
        [System.IO.File]::WriteAllText((Join-Path $tools 'python.cmd'), ($body -join "`r`n") + "`r`n", [System.Text.Encoding]::ASCII)
        foreach ($version in '3.11.9', '3.12.10') {
            Compress-Archive -Path $tools -DestinationPath (Join-Path $root "python-$version.zip")
        }
        return ([uri]$root).AbsoluteUri + '/python-{0}.zip'
    }

    # One install under TestDrive, with a minted token variable.
    function Initialize-Install {
        param([string[]]$Python = @())
        $directory = Join-Path $TestDrive ('runner-' + [guid]::NewGuid().ToString('N'))
        $variable = 'FLEET_TEST_TOKEN_' + [guid]::NewGuid().ToString('N')
        $script:variables.Add($variable)
        return @{
            Repo = 'wagner-austin/fleet-test'; Name = 'test-runner'; Labels = 'test-runner,gpu'
            Service = 'fleet-test-service'; Directory = $directory; Workdir = (Join-Path $directory '_work')
            TokenVariable = $variable; Python = $Python
        }
    }

    # A case's service: one Win32_Service row or none, and the acts on it,
    # recorded.
    function Initialize-Service {
        param([string]$StartName = 'LocalSystem', [string]$State = 'Running', [switch]$Absent)
        $service = @{ StartName = $StartName; State = $State; Present = -not $Absent }
        $acts = [System.Collections.Generic.List[string]]::new()
        return [pscustomobject]@{
            Acts = $acts
            Get = { param([string]$Name) if ($service.Present) { [pscustomobject]@{ Name = $Name; StartName = $service.StartName; State = $service.State } } }.GetNewClosure()
            Stop = { param([string]$Name) $acts.Add("stop $Name"); $service.State = 'Stopped' }.GetNewClosure()
            Start = { param([string]$Name) $acts.Add("start $Name"); $service.State = 'Running' }.GetNewClosure()
        }
    }

    # The render's parameters for one case. The provision's also write
    # .wslconfig under TestDrive and register the keepalive under the
    # Describe's minted name, whose action is a stand-in.
    function Get-CaseParameter {
        param(
            [hashtable]$Install, [object]$Service, [object]$Sc,
            [string]$RunnerUrl = (Initialize-RunnerRelease), [string]$PythonUrl = (Initialize-PythonPackage),
            [hashtable]$Tokens = $null
        )
        if ($null -eq $Tokens) {
            $Tokens = @{ $Install.TokenVariable = 'TOKEN123' }
        }
        $parameters = @{
            Installs = @($Install); Tokens = $Tokens; RunnerUrl = $RunnerUrl; PythonPackageUrl = $PythonUrl; PythonName = 'python.cmd'
            GetService = $Service.Get; StopService = $Service.Stop; StartService = $Service.Start; Sc = $Sc.Path
        }
        if ($script:name -like 'provision-*') {
            $parameters.WslConfigPath = Join-Path $TestDrive ('wslconfig-' + [guid]::NewGuid().ToString('N'))
            $parameters.KeepaliveTask = $script:task
            $parameters.Distro = 'fleet-test-none'
            $parameters.Wsl = $script:wsl.Path
        }
        return $parameters
    }

    # What a run said, without the provision's host-level lines.
    function Select-InstallOutput {
        param([object[]]$Said)
        return [string[]]@($Said | ForEach-Object { [string]$_ } | Where-Object { $_ -notlike 'wrote .wslconfig*' -and $_ -notlike 'keepalive task *' })
    }

    # Stops and deletes a Describe's keepalive through Task Scheduler's COM
    # service. It runs in that Describe's AfterAll, before Pester removes
    # the Describe's TestDrive: the task's action is a stand-in there, and an
    # instance the scheduler started must not hold the directory.
    function Unregister-KeepaliveTask {
        param([string]$Name)
        $scheduler = New-Object -ComObject Schedule.Service
        $scheduler.Connect()
        $root = $scheduler.GetFolder('\')
        foreach ($task in @($root.GetTasks(1) | Where-Object { $_.Name -eq $Name })) {
            # Stop is a request: the instance's cmd.exe can still hold
            # TestDrive's wsl.cmd when the task is deleted, and Pester's
            # TestDrive teardown then fails with 'being used by another
            # process', as check-powershell did on the hub at f0fa90f12 (MCPs
            # board task 939ec5c7). So the case waits, bounded, for the
            # task's last instance to end before it deletes the task.
            $task.Stop(0)
            $deadline = [DateTime]::UtcNow.AddSeconds(30)
            while ($task.GetInstances(0).Count -gt 0) {
                if ([DateTime]::UtcNow -gt $deadline) {
                    throw "keepalive task $Name still has a running instance 30 s after Stop"
                }
                Start-Sleep -Milliseconds 100
            }
            $root.DeleteTask($Name, 0)
        }
    }

    $script:variables = [System.Collections.Generic.List[string]]::new()
}

AfterAll {
    foreach ($variable in $script:variables) {
        if (Test-Path -LiteralPath "env:$variable") {
            Remove-Item -LiteralPath "env:$variable"
        }
    }
}

Describe 'The Windows install loop in <_>' -ForEach @(Get-ChildItem -LiteralPath (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'rendered') -Filter '*-windows-*.ps1' | Where-Object { $_.BaseName -like 'provision-windows-*' -or $_.BaseName -like 'onboard-windows-*' } | ForEach-Object { $_.BaseName }) {
    BeforeAll {
        $script:name = $_
        $script:task = 'fleet-test-keepalive-' + [guid]::NewGuid().ToString('N')
        $script:wsl = Initialize-Batch 'wsl' @('exit /b 0')
    }
    AfterAll {
        Unregister-KeepaliveTask $script:task
    }

    It 'downloads, configures and seeds a fresh install with the token it was handed, and leaves a running SYSTEM service alone' {
        $install = Initialize-Install -Python @('3.11.9')
        $service = Initialize-Service
        $sc = Initialize-Batch 'sc' @('exit /b 0')
        $said = @(Invoke-Rendered $script:name (Get-CaseParameter -Install $install -Service $service -Sc $sc))
        Select-InstallOutput $said | Should -Be @()
        [Environment]::GetEnvironmentVariable($install.TokenVariable) | Should -BeExactly 'TOKEN123'
        Test-Path -LiteralPath (Join-Path $install.Directory 'runner.zip') | Should -BeFalse
        Read-CallRecord ([pscustomobject]@{ Record = (Join-Path $install.Directory 'config-calls.txt') }) |
            Should -Be @('--unattended --url https://github.com/wagner-austin/fleet-test --token TOKEN123 --name test-runner --labels test-runner,gpu --runasservice --windowslogonaccount "NT AUTHORITY\SYSTEM" --replace')
        Read-CallRecord $sc | Should -Be @('failure fleet-test-service reset= 86400 actions= restart/5000/restart/5000/restart/5000', 'failureflag fleet-test-service 1')
        $service.Acts | Should -Be @()
        $seed = Join-Path $install.Workdir '_tool\Python\3.11.9\x64'
        Test-Path -LiteralPath (Join-Path $seed 'Scripts\pip.exe') | Should -BeTrue
        Test-Path -LiteralPath (Join-Path $install.Workdir '_tool\Python\3.11.9\x64.complete') | Should -BeTrue
        Read-CallRecord ([pscustomobject]@{ Record = (Join-Path $seed 'python-calls.txt') }) |
            Should -Be @("-m pip install --force-reinstall --no-deps --no-index --no-warn-script-location --disable-pip-version-check $seed\Lib\ensurepip\_bundled\pip-24.0-py3-none-any.whl")
    }
    It 'leaves an install already configured and seeded alone on a second run, fetching nothing' {
        $install = Initialize-Install -Python @('3.11.9', '3.12.10')
        $service = Initialize-Service
        $sc = Initialize-Batch 'sc' @('exit /b 0')
        [void](Invoke-Rendered $script:name (Get-CaseParameter -Install $install -Service $service -Sc $sc))
        $missing = ([uri](Join-Path $TestDrive 'absent.zip')).AbsoluteUri
        [void](Invoke-Rendered $script:name (Get-CaseParameter -Install $install -Service $service -Sc $sc -RunnerUrl $missing -PythonUrl ($missing + '?{0}')))
        @(Read-CallRecord ([pscustomobject]@{ Record = (Join-Path $install.Directory 'config-calls.txt') })).Count | Should -Be 1
        foreach ($version in '3.11.9', '3.12.10') {
            @(Read-CallRecord ([pscustomobject]@{ Record = (Join-Path $install.Workdir "_tool\Python\$version\x64\python-calls.txt") })).Count | Should -Be 1
        }
    }
    It 'reads the token from the environment when it is handed none, and refuses by name when there is none' {
        $install = Initialize-Install
        $service = Initialize-Service
        $sc = Initialize-Batch 'sc' @('exit /b 0')
        { Invoke-Rendered $script:name (Get-CaseParameter -Install $install -Service $service -Sc $sc -Tokens @{}) } |
            Should -Throw "FLEET_RUNNER_TOKEN_MISSING: set $($install.TokenVariable) to a fresh registration token for wagner-austin/fleet-test"
        Set-Item -LiteralPath "env:$($install.TokenVariable)" -Value 'FROMENV'
        [void](Invoke-Rendered $script:name (Get-CaseParameter -Install $install -Service $service -Sc $sc -Tokens @{}))
        @(Read-CallRecord ([pscustomobject]@{ Record = (Join-Path $install.Directory 'config-calls.txt') }))[0] | Should -BeLike '* --token FROMENV *'
    }
    It 'provisions the roster''s own installs by default, and refuses the first by name when its token is unset' {
        $first = @(Get-RenderedDefault $script:name 'Installs')[0]
        if (Test-Path -LiteralPath "env:$($first.TokenVariable)") {
            Remove-Item -LiteralPath "env:$($first.TokenVariable)"
        }
        $parameters = Get-CaseParameter -Install (Initialize-Install) -Service (Initialize-Service) -Sc (Initialize-Batch 'sc' @('exit /b 0')) -Tokens @{}
        $parameters.Remove('Installs')
        { Invoke-Rendered $script:name $parameters } |
            Should -Throw "FLEET_RUNNER_TOKEN_MISSING: set $($first.TokenVariable) to a fresh registration token for $($first.Repo)"
    }
    It 'refuses by name a configuration config.cmd refuses' {
        $install = Initialize-Install
        $sc = Initialize-Batch 'sc' @('exit /b 0')
        { Invoke-Rendered $script:name (Get-CaseParameter -Install $install -Service (Initialize-Service) -Sc $sc -RunnerUrl (Initialize-RunnerRelease -Refuse)) } |
            Should -Throw 'FLEET_RUNNER_CONFIG_REFUSED: config.cmd for wagner-austin/fleet-test test-runner exited 1'
    }
    It 'refuses by name a service that config.cmd did not install' {
        $install = Initialize-Install
        $sc = Initialize-Batch 'sc' @('exit /b 0')
        { Invoke-Rendered $script:name (Get-CaseParameter -Install $install -Service (Initialize-Service -Absent) -Sc $sc) } |
            Should -Throw 'FLEET_RUNNER_SERVICE_MISSING: service fleet-test-service is not installed'
    }
    It 'rebinds a service under another account to SYSTEM, removing the work tree that account owned' {
        $install = Initialize-Install
        [void][System.IO.Directory]::CreateDirectory($install.Workdir)
        [System.IO.File]::WriteAllText((Join-Path $install.Workdir 'checkout.txt'), 'owned by the old account')
        $service = Initialize-Service -StartName 'NT AUTHORITY\NetworkService'
        $sc = Initialize-Batch 'sc' @('exit /b 0')
        $said = @(Invoke-Rendered $script:name (Get-CaseParameter -Install $install -Service $service -Sc $sc))
        Select-InstallOutput $said | Should -Be @('rebound fleet-test-service from NT AUTHORITY\NetworkService to LocalSystem and removed its work tree')
        $service.Acts | Should -Be @('stop fleet-test-service', 'start fleet-test-service')
        @(Read-CallRecord $sc)[0] | Should -BeExactly 'config fleet-test-service obj= LocalSystem'
        Test-Path -LiteralPath $install.Workdir | Should -BeFalse
    }
    It 'rebinds a service whose work tree was never created' {
        $install = Initialize-Install
        $service = Initialize-Service -StartName 'NT AUTHORITY\NetworkService'
        $sc = Initialize-Batch 'sc' @('exit /b 0')
        [void](Invoke-Rendered $script:name (Get-CaseParameter -Install $install -Service $service -Sc $sc))
        $service.Acts | Should -Be @('stop fleet-test-service', 'start fleet-test-service')
    }
    It 'refuses by name a rebind sc.exe refuses' {
        $install = Initialize-Install
        $sc = Initialize-Batch 'sc' @('echo %* | findstr /b /c:"config " >nul', 'if not errorlevel 1 exit /b 5', 'exit /b 0')
        { Invoke-Rendered $script:name (Get-CaseParameter -Install $install -Service (Initialize-Service -StartName 'NT AUTHORITY\NetworkService') -Sc $sc) } |
            Should -Throw 'FLEET_RUNNER_REBIND_REFUSED: sc.exe config fleet-test-service exited 5'
    }
    It 'refuses by name a restart policy sc.exe refuses to write, <Act>' -ForEach @(
        @{ Act = 'failure'; Match = 'failure ' }
        @{ Act = 'failureflag'; Match = 'failureflag ' }
    ) {
        $install = Initialize-Install
        $sc = Initialize-Batch 'sc' @("echo %* | findstr /b /c:`"$Match`" >nul", 'if not errorlevel 1 exit /b 1060', 'exit /b 0')
        { Invoke-Rendered $script:name (Get-CaseParameter -Install $install -Service (Initialize-Service) -Sc $sc) } |
            Should -Throw "FLEET_RUNNER_RECOVERY_REFUSED: sc.exe $Act fleet-test-service exited 1060"
    }
    It 'starts a SYSTEM service it finds stopped' {
        $install = Initialize-Install
        $service = Initialize-Service -State 'Stopped'
        $sc = Initialize-Batch 'sc' @('exit /b 0')
        $said = @(Invoke-Rendered $script:name (Get-CaseParameter -Install $install -Service $service -Sc $sc))
        Select-InstallOutput $said | Should -Be @('started fleet-test-service')
        $service.Acts | Should -Be @('start fleet-test-service')
    }
    It 'refuses by name a Python seed <Case>' -ForEach @(
        @{ Case = 'carrying no bundled pip wheel'; Variant = 'NoWheel'; Message = 'FLEET_PYTHON_SEED_WHEEL: expected one bundled pip wheel in seeded 3.11.9, found 0' }
        @{ Case = 'whose pip reinstall fails'; Variant = 'PipFails'; Message = 'FLEET_PYTHON_SEED_PIP: the pip reinstall in seeded 3.11.9 exited 1' }
        @{ Case = 'whose pip reinstall writes no pip.exe'; Variant = 'NoPip'; Message = 'FLEET_PYTHON_SEED_PIP_MISSING: pip.exe is missing in seeded 3.11.9' }
    ) {
        $install = Initialize-Install -Python @('3.11.9')
        $sc = Initialize-Batch 'sc' @('exit /b 0')
        { Invoke-Rendered $script:name (Get-CaseParameter -Install $install -Service (Initialize-Service) -Sc $sc -PythonUrl (Initialize-PythonPackage -Variant $Variant)) } |
            Should -Throw $Message
    }
    It 'reads a service through Get-CimInstance by default, which finds no test service' {
        $install = Initialize-Install
        $parameters = Get-CaseParameter -Install $install -Service (Initialize-Service) -Sc (Initialize-Batch 'sc' @('exit /b 0'))
        $install.Service = 'fleet-test-absent-' + [guid]::NewGuid().ToString('N')
        $parameters.Remove('GetService')
        { Invoke-Rendered $script:name $parameters } | Should -Throw "FLEET_RUNNER_SERVICE_MISSING: service $($install.Service) is not installed"
    }
    It 'reaches the service cmdlets through the default <Default>, which refuses a service that does not exist' -ForEach @(
        @{ Default = 'StopService' }
        @{ Default = 'StartService' }
    ) {
        $install = Initialize-Install
        $install.Service = 'fleet-test-absent-' + [guid]::NewGuid().ToString('N')
        $parameters = Get-CaseParameter -Install $install -Service (Initialize-Service -StartName 'NT AUTHORITY\NetworkService') -Sc (Initialize-Batch 'sc' @('exit /b 0'))
        $parameters.Remove($Default)
        { Invoke-Rendered $script:name $parameters } | Should -Throw "*$($install.Service)*"
    }
}

Describe 'The host-level provision in <_>' -ForEach @(Get-ChildItem -LiteralPath (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'rendered') -Filter 'provision-windows-*.ps1' | ForEach-Object { $_.BaseName }) {
    BeforeAll {
        $script:name = $_
        $script:task = 'fleet-test-keepalive-' + [guid]::NewGuid().ToString('N')
        $script:wsl = Initialize-Batch 'wsl' @('exit /b 0')
    }
    AfterAll {
        Unregister-KeepaliveTask $script:task
    }

    It 'writes the memory floor to .wslconfig and registers the keepalive through S4U at boot, restarting, and starts it' {
        $install = Initialize-Install
        $parameters = Get-CaseParameter -Install $install -Service (Initialize-Service) -Sc (Initialize-Batch 'sc' @('exit /b 0'))
        $said = [string[]]@(Invoke-Rendered $script:name $parameters)
        $said | Should -Be @('wrote .wslconfig; the ceiling applies when the VM next starts', "keepalive task $script:task registered through S4U and started")
        $floor = [System.IO.File]::ReadAllLines($parameters.WslConfigPath)
        $floor[0] | Should -BeExactly '[wsl2]'
        $floor[1] | Should -Match '^memory=\d+GB$'
        $floor[2] | Should -BeExactly 'swap=8GB'
        $task = (Get-TaskDefinition $script:task).Task
        $task.Principals.Principal.LogonType | Should -BeExactly 'S4U'
        $task.Principals.Principal.RunLevel | Should -BeExactly 'HighestAvailable'
        $task.Actions.Exec.Command | Should -BeExactly $script:wsl.Path
        $task.Actions.Exec.Arguments | Should -BeExactly '-d fleet-test-none --exec /usr/bin/sleep infinity'
        @($task.Triggers.ChildNodes | ForEach-Object { $_.LocalName }) | Should -Be @('BootTrigger')
        $task.Settings.RestartOnFailure.Count | Should -BeExactly '999'
        $task.Settings.ExecutionTimeLimit | Should -BeExactly 'PT0S'
    }
}
