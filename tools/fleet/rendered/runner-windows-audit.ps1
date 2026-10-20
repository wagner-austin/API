param(
    [string]$Sc = "$env:SystemRoot\System32\sc.exe",
    [string]$Policy = 'RemoteSigned',
    [string]$PolicyKey = 'HKLM:\SOFTWARE\Microsoft\PowerShell\1\ShellIds\Microsoft.PowerShell',
    [string]$FileSystemKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem',
    [string]$Git = 'git',
    [string]$VariableScope = 'Machine',
    [string]$MachineEnvironment = '[{"Name":"POETRY_CACHE_DIR","Value":"C:\\fleet\\poetry"}]',
    [string]$Installs = '[{"Service":"actions.runner.wagner-austin-chat.loki"}]'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
function Get-Answer {
    param([string]$Tool, [string[]]$Arguments)
    $text = ((& $Tool @Arguments) -join ' ') -replace '\s+', ' '
    return [pscustomobject]@{ Code = $LASTEXITCODE; Text = $text.Trim() }
}
$heldPolicy = [string](Get-Item -LiteralPath $PolicyKey).GetValue('ExecutionPolicy')
if ($heldPolicy -eq $Policy) {
    "CHECK execution-policy:LocalMachine:$Policy OK"
} else {
    "CHECK execution-policy:LocalMachine:$Policy DRIFT the LocalMachine ExecutionPolicy value is: $heldPolicy"
}
$heldLongPaths = (Get-Item -LiteralPath $FileSystemKey).GetValue('LongPathsEnabled')
if ($heldLongPaths -eq 1) {
    'CHECK long-paths OK'
} else {
    "CHECK long-paths DRIFT LongPathsEnabled is: $heldLongPaths"
}
$gitLongPaths = Get-Answer $Git @('config', '--system', '--get', 'core.longpaths')
if ($gitLongPaths.Code -eq 0 -and $gitLongPaths.Text -eq 'true') {
    'CHECK git-longpaths OK'
} else {
    "CHECK git-longpaths DRIFT git config --system --get core.longpaths exited $($gitLongPaths.Code): $($gitLongPaths.Text)"
}
foreach ($variable in ($MachineEnvironment | ConvertFrom-Json)) {
    $heldValue = [string][Environment]::GetEnvironmentVariable($variable.Name, $VariableScope)
    if ($heldValue -ceq $variable.Value) {
        "CHECK machine-env:$($variable.Name) OK"
    } else {
        "CHECK machine-env:$($variable.Name) DRIFT the machine variable holds: $heldValue"
    }
}
foreach ($install in ($Installs | ConvertFrom-Json)) {
    $service = $install.Service
    $state = Get-Answer $Sc @('query', $service)
    if ($state.Code -eq 0 -and $state.Text -match 'STATE : 4 RUNNING') {
        "CHECK service:$service OK"
    } else {
        "CHECK service:$service DRIFT sc.exe query exited $($state.Code): $($state.Text)"
    }
    $config = Get-Answer $Sc @('qc', $service)
    if ($config.Code -eq 0 -and $config.Text -match 'START_TYPE : 2 AUTO_START' -and $config.Text -match 'SERVICE_START_NAME : NT AUTHORITY\\SYSTEM') {
        "CHECK account:$service OK"
    } else {
        "CHECK account:$service DRIFT sc.exe qc exited $($config.Code): $($config.Text)"
    }
    $failure = Get-Answer $Sc @('qfailure', $service)
    if ($failure.Code -eq 0 -and $failure.Text -match 'FAILURE_ACTIONS : RESTART') {
        "CHECK recovery:$service OK"
    } else {
        "CHECK recovery:$service DRIFT sc.exe qfailure exited $($failure.Code): $($failure.Text)"
    }
}
exit 0
