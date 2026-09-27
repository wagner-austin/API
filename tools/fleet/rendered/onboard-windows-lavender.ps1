param(
    [hashtable[]]$Installs = @(
        @{ Repo = 'wagner-austin/MCPs'; Name = 'lavender'; Labels = 'lavender'; Service = 'actions.runner.wagner-austin-MCPs.lavender'; Directory = 'C:\actions-runner'; Workdir = 'C:\actions-runner\_work'; TokenVariable = 'RUNNER_TOKEN_MCPS'; Python = @('3.11.9') }
    ),
    [hashtable]$Tokens = @{},
    [string]$RunnerUrl = 'https://github.com/actions/runner/releases/download/v2.337.0/actions-runner-win-x64-2.337.0.zip',
    [string]$PythonPackageUrl = 'https://www.nuget.org/api/v2/package/python/{0}',
    [string]$PythonName = 'python.exe',
    [scriptblock]$GetService = { param([string]$Name) @(Get-CimInstance Win32_Service -Filter "Name='$Name'") },
    [scriptblock]$StopService = { param([string]$Name) Stop-Service -Name $Name },
    [scriptblock]$StartService = { param([string]$Name) Start-Service -Name $Name },
    [string]$Sc = "$env:SystemRoot\System32\sc.exe"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
foreach ($TokenName in @($Tokens.Keys)) {
    Set-Item -LiteralPath ('env:' + $TokenName) -Value $Tokens[$TokenName]
}
foreach ($Install in $Installs) {
    $Directory = [string]$Install.Directory
    $Workdir = [string]$Install.Workdir
    $ServiceName = [string]$Install.Service
    $Token = [Environment]::GetEnvironmentVariable([string]$Install.TokenVariable)
    if ([string]::IsNullOrEmpty($Token)) {
        throw ("FLEET_RUNNER_TOKEN_MISSING: set " + $Install.TokenVariable + ' to a fresh registration token for ' + $Install.Repo)
    }
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
    if (-not (Test-Path -LiteralPath (Join-Path $Directory 'config.cmd'))) {
        $Zip = Join-Path $Directory 'runner.zip'
        Invoke-WebRequest -Uri $RunnerUrl -OutFile $Zip -UseBasicParsing
        Expand-Archive -LiteralPath $Zip -DestinationPath $Directory -Force
        Remove-Item -LiteralPath $Zip
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Directory '.runner'))) {
        & (Join-Path $Directory 'config.cmd') --unattended --url ('https://github.com/' + $Install.Repo) --token $Token --name $Install.Name --labels $Install.Labels --runasservice --windowslogonaccount 'NT AUTHORITY\SYSTEM' --replace
        if ($LASTEXITCODE -ne 0) {
            throw ("FLEET_RUNNER_CONFIG_REFUSED: config.cmd for " + $Install.Repo + ' ' + $Install.Name + ' exited ' + $LASTEXITCODE)
        }
    }
    $Rows = @(& $GetService $ServiceName)
    if ($Rows.Count -eq 0) {
        throw "FLEET_RUNNER_SERVICE_MISSING: service $ServiceName is not installed"
    }
    $StartName = [string]$Rows[0].StartName
    if ($StartName -ne 'LocalSystem') {
        & $StopService $ServiceName
        & $Sc config $ServiceName obj= LocalSystem | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "FLEET_RUNNER_REBIND_REFUSED: sc.exe config $ServiceName exited $LASTEXITCODE"
        }
        if (Test-Path -LiteralPath $Workdir) {
            Remove-Item -LiteralPath $Workdir -Recurse -Force
        }
        & $StartService $ServiceName
        Write-Output ('rebound ' + $ServiceName + ' from ' + $StartName + ' to LocalSystem and removed its work tree')
    }
    & $Sc failure $ServiceName reset= 86400 actions= restart/5000/restart/5000/restart/5000 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "FLEET_RUNNER_RECOVERY_REFUSED: sc.exe failure $ServiceName exited $LASTEXITCODE"
    }
    & $Sc failureflag $ServiceName 1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "FLEET_RUNNER_RECOVERY_REFUSED: sc.exe failureflag $ServiceName exited $LASTEXITCODE"
    }
    $State = (@(& $GetService $ServiceName | ForEach-Object { [string]$_.State }) -join '')
    if ($State -ne 'Running') {
        & $StartService $ServiceName
        Write-Output ('started ' + $ServiceName)
    }
    $ToolDir = Join-Path $Workdir '_tool'
    foreach ($Version in @($Install.Python)) {
        $Target = Join-Path $ToolDir ('Python\' + $Version + '\x64')
        if (-not (Test-Path -LiteralPath (Join-Path $Target 'Scripts\pip.exe'))) {
            $Work = Join-Path $env:TEMP ([guid]::NewGuid().ToString('N'))
            New-Item -ItemType Directory -Path $Work | Out-Null
            $Package = Join-Path $Work 'python.nupkg.zip'
            Invoke-WebRequest -Uri ($PythonPackageUrl -f $Version) -OutFile $Package -UseBasicParsing
            Expand-Archive -LiteralPath $Package -DestinationPath $Work -Force
            New-Item -ItemType Directory -Force -Path $Target | Out-Null
            Copy-Item -Path (Join-Path $Work 'tools\*') -Destination $Target -Recurse -Force
            $Wheel = @(Get-ChildItem -LiteralPath (Join-Path $Target 'Lib\ensurepip\_bundled') -Filter 'pip-*.whl')
            if ($Wheel.Count -ne 1) {
                throw ("FLEET_PYTHON_SEED_WHEEL: expected one bundled pip wheel in seeded " + $Version + ', found ' + $Wheel.Count)
            }
            & (Join-Path $Target $PythonName) -m pip install --force-reinstall --no-deps --no-index --no-warn-script-location --disable-pip-version-check $Wheel[0].FullName
            if ($LASTEXITCODE -ne 0) {
                throw ("FLEET_PYTHON_SEED_PIP: the pip reinstall in seeded " + $Version + ' exited ' + $LASTEXITCODE)
            }
            if (-not (Test-Path -LiteralPath (Join-Path $Target 'Scripts\pip.exe'))) {
                throw ("FLEET_PYTHON_SEED_PIP_MISSING: pip.exe is missing in seeded " + $Version)
            }
            New-Item -ItemType File -Force -Path (Join-Path $ToolDir ('Python\' + $Version + '\x64.complete')) | Out-Null
            Remove-Item -LiteralPath $Work -Recurse -Force
        }
    }
}
