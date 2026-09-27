param(
    [string]$Scratch = 'C:/fleet/stage',
    [string[]]$Features = @('VirtualMachinePlatform', 'Microsoft-Windows-Subsystem-Linux'),
    [string]$WslVersion = '2.7.14',
    [string]$MsiUrl = 'https://github.com/microsoft/WSL/releases/download/2.7.14/wsl.2.7.14.0.x64.msi',
    [string]$MsiSha256 = 'db084e536279a59e90a26ec598d8aa8a4dff8309f41d078fd06242953ac1ebcd',
    [string]$MsiFile = 'wsl.2.7.14.0.x64.msi',
    [string]$Policy = 'RemoteSigned',
    [string]$PolicyKey = 'HKLM:\SOFTWARE\Microsoft\PowerShell\1\ShellIds\Microsoft.PowerShell',
    [string]$FileSystemKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem',
    [string[]]$PathEntries = @('C:\Program Files (x86)\GnuWin32\bin'),
    [string]$PathVariable = 'Path',
    [string]$PathScope = 'Machine',
    [string]$EnvironmentKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
    [string[]]$MachineVariables = @('POETRY_CACHE_DIR=C:\fleet\poetry'),
    [string]$WslConfigPath = "$env:USERPROFILE\.wslconfig",
    [string]$Git = 'git',
    [string]$Dism = "$env:SystemRoot\System32\dism.exe",
    [string]$Msiexec = "$env:SystemRoot\System32\msiexec.exe",
    [string]$Wsl = "$env:SystemRoot\System32\wsl.exe"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ProgressPreference = 'SilentlyContinue'
$env:WSL_UTF8 = '1'
$Reboot = $false
[void][System.IO.Directory]::CreateDirectory($Scratch)
foreach ($feature in $Features) {
    $info = & $Dism /online /english /get-featureinfo /featurename:$feature
    if ($LASTEXITCODE -ne 0) {
        throw ('dism could not read the Windows feature ' + $feature + ', exit ' + $LASTEXITCODE)
    }
    if (@($info | Where-Object { $_ -match '^State : Enabled' }).Count -eq 0) {
        $null = & $Dism /online /english /enable-feature /featurename:$feature /all /norestart
        if ($LASTEXITCODE -eq 3010) {
            $Reboot = $true
        } elseif ($LASTEXITCODE -ne 0) {
            throw ('dism could not enable the Windows feature ' + $feature + ', exit ' + $LASTEXITCODE)
        }
        Write-Output ('enabled Windows feature ' + $feature)
    }
}
$answer = & $Wsl --version
$installed = ($LASTEXITCODE -eq 0) -and ((@($answer) -join ' ') -match ('WSL[^:]*:\s*' + [regex]::Escape($WslVersion)))
if (-not $installed) {
    $msiPath = Join-Path $Scratch $MsiFile
    if (-not (Test-Path -LiteralPath $msiPath)) {
        Invoke-WebRequest -Uri $MsiUrl -OutFile $msiPath -UseBasicParsing
    }
    $Digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $msiPath).Hash.ToLower()
    if ($Digest -ne $MsiSha256) {
        Remove-Item -LiteralPath $msiPath
        throw ('WSL MSI sha256 ' + $Digest + ' does not match the pin ' + $MsiSha256)
    }
    $install = Start-Process -FilePath $Msiexec -ArgumentList @('/i', "`"$msiPath`"", '/qn', '/norestart') -Wait -PassThru -NoNewWindow
    if ($install.ExitCode -eq 3010) {
        $Reboot = $true
    } elseif ($install.ExitCode -ne 0) {
        throw ('msiexec for WSL ' + $WslVersion + ' exited ' + $install.ExitCode)
    }
    Remove-Item -LiteralPath $msiPath
    Write-Output ('installed WSL ' + $WslVersion)
}
if ([string](Get-Item -LiteralPath $PolicyKey).GetValue('ExecutionPolicy') -ne $Policy) {
    Set-ItemProperty -LiteralPath $PolicyKey -Name ExecutionPolicy -Value $Policy
    Write-Output ('set the LocalMachine execution policy to ' + $Policy)
}
if ((Get-Item -LiteralPath $FileSystemKey).GetValue('LongPathsEnabled') -ne 1) {
    Set-ItemProperty -LiteralPath $FileSystemKey -Name LongPathsEnabled -Value 1 -Type DWord
    Write-Output 'enabled Win32 long paths'
}
$held = & $Git config --system --get core.longpaths
if ($LASTEXITCODE -gt 1) {
    throw ('git config --system --get core.longpaths exited ' + $LASTEXITCODE)
}
if ((@($held) -join '') -ne 'true') {
    & $Git config --system core.longpaths true
    if ($LASTEXITCODE -ne 0) {
        throw ('git config --system core.longpaths exited ' + $LASTEXITCODE)
    }
    Write-Output 'set git core.longpaths for the system'
}
foreach ($entry in $PathEntries) {
    $current = [string][Environment]::GetEnvironmentVariable($PathVariable, $PathScope)
    $parts = @($current -split ';' | Where-Object { $_ -ne '' })
    if ($parts -notcontains $entry) {
        [Environment]::SetEnvironmentVariable($PathVariable, ((@($parts) + $entry) -join ';'), $PathScope)
        Write-Output ('added to the machine PATH: ' + $entry)
    }
}
foreach ($pair in $MachineVariables) {
    $name, $value = $pair -split '=', 2
    if ([string](Get-Item -LiteralPath $EnvironmentKey).GetValue($name) -cne $value) {
        Set-ItemProperty -LiteralPath $EnvironmentKey -Name $name -Value $value
        $Reboot = $true
        Write-Output ('set the machine variable ' + $name + ' to ' + $value)
    }
}
$WslConfig = @(
  '[wsl2]',
  'memory=26GB',
  'swap=8GB'
)
Set-Content -LiteralPath $WslConfigPath -Value $WslConfig -Encoding ascii
Write-Output 'wrote .wslconfig; the ceiling applies when the VM next starts'
if ($Reboot) {
    Write-Output 'FLEET-REBOOT-REQUIRED'
}
