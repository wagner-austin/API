param(
    [string]$RunnerZipUrl = 'https://github.com/actions/runner/releases/download/v2.337.0/actions-runner-win-x64-2.337.0.zip',
    [string]$Sc = "$env:SystemRoot\System32\sc.exe",
    [hashtable]$Tokens = @{},
    [string]$Policy = 'RemoteSigned',
    [string]$PolicyKey = 'HKLM:\SOFTWARE\Microsoft\PowerShell\1\ShellIds\Microsoft.PowerShell',
    [string]$FileSystemKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem',
    [string]$Git = 'git',
    [string]$VariableScope = 'Machine',
    [string]$MachineEnvironment = '[{"Name":"POETRY_CACHE_DIR","Value":"C:\\fleet\\poetry"}]',
    [string]$PythonPackageUrl = 'https://www.nuget.org/api/v2/package/python/{0}',
    [string]$PythonName = 'python.exe',
    [int]$StopWaitSeconds = 60,
    [string]$Installs = '[{"Repo":"wagner-austin/chat","Name":"loki","Labels":"loki","Root":"C:/actions-runner/chat","Service":"actions.runner.wagner-austin-chat.loki","Python":["3.11.9"]}]'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ([string](Get-Item -LiteralPath $PolicyKey).GetValue('ExecutionPolicy') -ne $Policy) {
    Set-ItemProperty -LiteralPath $PolicyKey -Name ExecutionPolicy -Value $Policy
    "base: set the LocalMachine execution policy to $Policy"
}
if ((Get-Item -LiteralPath $FileSystemKey).GetValue('LongPathsEnabled') -ne 1) {
    Set-ItemProperty -LiteralPath $FileSystemKey -Name LongPathsEnabled -Value 1 -Type DWord
    'base: enabled Win32 long paths'
}
$held = & $Git config --system --get core.longpaths
if ($LASTEXITCODE -gt 1) {
    throw "RUNNER_BASE_FAILED: git config --system --get core.longpaths exited $LASTEXITCODE"
}
if ((@($held) -join '') -ne 'true') {
    & $Git config --system core.longpaths true
    if ($LASTEXITCODE -ne 0) {
        throw "RUNNER_BASE_FAILED: git config --system core.longpaths exited $LASTEXITCODE"
    }
    'base: set git core.longpaths for the system'
}
$restart = $false
foreach ($variable in ($MachineEnvironment | ConvertFrom-Json)) {
    if ([Environment]::GetEnvironmentVariable($variable.Name, $VariableScope) -cne $variable.Value) {
        [Environment]::SetEnvironmentVariable($variable.Name, $variable.Value, $VariableScope)
        "base: set the machine variable $($variable.Name)"
        $restart = $true
    }
}
foreach ($install in ($Installs | ConvertFrom-Json)) {
    $root = $install.Root
    if (-not (Test-Path -LiteralPath "$root/config.cmd")) {
        [void][IO.Directory]::CreateDirectory($root)
        Invoke-WebRequest -Uri $RunnerZipUrl -OutFile "$root/runner.zip" -UseBasicParsing
        Expand-Archive -LiteralPath "$root/runner.zip" -DestinationPath $root -Force
        Remove-Item -LiteralPath "$root/runner.zip"
        "runner: unpacked $RunnerZipUrl into $root"
    }
    if (-not (Test-Path -LiteralPath "$root/.runner")) {
        if (-not $Tokens.ContainsKey($install.Repo)) {
            throw "RUNNER_TOKEN_MISSING: no registration token for $($install.Repo); fleet-runners --rebuild yes mints one"
        }
        & "$root/config.cmd" --unattended --url "https://github.com/$($install.Repo)" --token $Tokens[$install.Repo] --name $install.Name --labels $install.Labels --replace --runasservice --windowslogonaccount 'NT AUTHORITY\SYSTEM'
        if ($LASTEXITCODE -ne 0) {
            throw "RUNNER_CONFIG_FAILED: config.cmd for $($install.Repo) exited $LASTEXITCODE"
        }
        "runner: registered $($install.Repo) runner $($install.Name)"
    }
    $tools = Join-Path $root '_work\_tool'
    foreach ($version in @($install.Python)) {
        $target = Join-Path $tools "Python\$version\x64"
        if (-not (Test-Path -LiteralPath (Join-Path $target 'Scripts\pip.exe'))) {
            $work = Join-Path $env:TEMP ([guid]::NewGuid().ToString('N'))
            [void][IO.Directory]::CreateDirectory($work)
            $package = Join-Path $work 'python.nupkg.zip'
            Invoke-WebRequest -Uri ($PythonPackageUrl -f $version) -OutFile $package -UseBasicParsing
            Expand-Archive -LiteralPath $package -DestinationPath $work -Force
            [void][IO.Directory]::CreateDirectory($target)
            Copy-Item -Path (Join-Path $work 'tools\*') -Destination $target -Recurse -Force
            $wheel = @(Get-ChildItem -LiteralPath (Join-Path $target 'Lib\ensurepip\_bundled') -Filter 'pip-*.whl')
            if ($wheel.Count -ne 1) {
                throw "RUNNER_PYTHON_SEED_FAILED: $version holds $($wheel.Count) bundled pip wheels, not one"
            }
            & (Join-Path $target $PythonName) -m pip install --force-reinstall --no-deps --no-index --no-warn-script-location --disable-pip-version-check $wheel[0].FullName
            if ($LASTEXITCODE -ne 0) {
                throw "RUNNER_PYTHON_SEED_FAILED: the pip reinstall in $version exited $LASTEXITCODE"
            }
            if (-not (Test-Path -LiteralPath (Join-Path $target 'Scripts\pip.exe'))) {
                throw "RUNNER_PYTHON_SEED_FAILED: the pip reinstall in $version wrote no Scripts\pip.exe"
            }
            [void](New-Item -ItemType File -Force -Path (Join-Path $tools "Python\$version\x64.complete"))
            Remove-Item -LiteralPath $work -Recurse -Force
            "python: seeded $version into $tools for $($install.Repo)"
        }
    }
    $null = & $Sc failure $install.Service reset= 86400 actions= restart/60000/restart/60000/restart/60000
    if ($LASTEXITCODE -ne 0) {
        throw "RUNNER_SERVICE_FAILED: sc.exe failure $($install.Service) exited $LASTEXITCODE"
    }
    $state = (& $Sc query $install.Service) -join ' '
    if ($LASTEXITCODE -ne 0) {
        throw "RUNNER_SERVICE_FAILED: sc.exe query $($install.Service) exited $LASTEXITCODE"
    }
    $running = $state -match 'STATE\s+:\s+4\s+RUNNING'
    if ($running -and $restart) {
        $null = & $Sc stop $install.Service
        if ($LASTEXITCODE -ne 0) {
            throw "RUNNER_SERVICE_FAILED: sc.exe stop $($install.Service) exited $LASTEXITCODE"
        }
        $stopped = $false
        foreach ($second in 1..$StopWaitSeconds) {
            $state = (& $Sc query $install.Service) -join ' '
            if ($LASTEXITCODE -ne 0) {
                throw "RUNNER_SERVICE_FAILED: sc.exe query $($install.Service) exited $LASTEXITCODE"
            }
            if ($state -match 'STATE\s+:\s+1\s+STOPPED') {
                $stopped = $true
                break
            }
            Start-Sleep -Seconds 1
        }
        if (-not $stopped) {
            throw "RUNNER_SERVICE_FAILED: $($install.Service) did not stop within $StopWaitSeconds s"
        }
        "runner: stopped $($install.Service) so it starts with the machine variables"
        $running = $false
    }
    if (-not $running) {
        $null = & $Sc start $install.Service
        if ($LASTEXITCODE -ne 0) {
            throw "RUNNER_SERVICE_FAILED: sc.exe start $($install.Service) exited $LASTEXITCODE"
        }
    }
    "runner: $($install.Repo) runner $($install.Name) runs as $($install.Service), restarted on failure"
}
