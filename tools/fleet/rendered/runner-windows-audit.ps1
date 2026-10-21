param(
    [string]$Sc = "$env:SystemRoot\System32\sc.exe",
    [string]$Policy = 'RemoteSigned',
    [string]$PolicyKey = 'HKLM:\SOFTWARE\Microsoft\PowerShell\1\ShellIds\Microsoft.PowerShell',
    [string]$FileSystemKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem',
    [string]$Git = 'git',
    [string]$VariableScope = 'Machine',
    [string]$MachineEnvironment = '[{"Name":"POETRY_CACHE_DIR","Value":"C:\\fleet\\poetry"}]',
    [string]$JobHookPath = 'C:\fleet\runner-hooks\job-started.ps1',
    [string]$JobHookVariable = 'ACTIONS_RUNNER_HOOK_JOB_STARTED',
    [string]$JobHook = 'cGFyYW0oCiAgICBbaW50XSRTdGFydCA9ICRQSUQsCiAgICBbc3RyaW5nXSRXb3JrZXJOYW1lID0gJ1J1bm5lci5Xb3JrZXIuZXhlJwopClNldC1TdHJpY3RNb2RlIC1WZXJzaW9uIExhdGVzdAokRXJyb3JBY3Rpb25QcmVmZXJlbmNlID0gJ1N0b3AnCiRwcm9jZXNzID0gR2V0LUNpbUluc3RhbmNlIFdpbjMyX1Byb2Nlc3MgLUZpbHRlciAiUHJvY2Vzc0lkID0gJFN0YXJ0IgpkbyB7CiAgICAkY2hpbGQgPSAkcHJvY2VzcwogICAgJHByb2Nlc3MgPSBHZXQtQ2ltSW5zdGFuY2UgV2luMzJfUHJvY2VzcyAtRmlsdGVyICJQcm9jZXNzSWQgPSAkKCRjaGlsZC5QYXJlbnRQcm9jZXNzSWQpIEFORCBQcm9jZXNzSWQgPD4gMCIgfAogICAgICAgIFdoZXJlLU9iamVjdCB7ICRfLkNyZWF0aW9uRGF0ZSAtbHQgJGNoaWxkLkNyZWF0aW9uRGF0ZSB9Cn0gd2hpbGUgKCRudWxsIC1uZSAkcHJvY2VzcyAtYW5kICRwcm9jZXNzLk5hbWUgLW5lICRXb3JrZXJOYW1lKQppZiAoJG51bGwgLWVxICRwcm9jZXNzKSB7CiAgICB0aHJvdyAiUlVOTkVSX0pPQl9QUklPUklUWV9GQUlMRUQ6IG5vICRXb3JrZXJOYW1lIGlzIGFtb25nIHRoZSBhbmNlc3RvcnMgb2YgcHJvY2VzcyAkU3RhcnQsIHNvIHRoZSBqb2Igd291bGQgcnVuIGF0IHRoZSBmbGVldCdzIG93biBwcmlvcml0eSIKfQooR2V0LVByb2Nlc3MgLUlkICRwcm9jZXNzLlByb2Nlc3NJZCkuUHJpb3JpdHlDbGFzcyA9IFtTeXN0ZW0uRGlhZ25vc3RpY3MuUHJvY2Vzc1ByaW9yaXR5Q2xhc3NdOjpCZWxvd05vcm1hbAoicnVubmVyIGpvYjogJFdvcmtlck5hbWUgJCgkcHJvY2Vzcy5Qcm9jZXNzSWQpIGFuZCBldmVyeSBwcm9jZXNzIGl0IHN0YXJ0cyBydW4gYXQgQmVsb3dOb3JtYWwsIHVuZGVyIHRoZSBmbGVldCdzIGNoZWNrcyIK',
    [string]$Installs = '[{"Service":"actions.runner.wagner-austin-chat.loki","Root":"C:/actions-runner/chat","Python":["3.11.9"]}]'
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
$heldHook = ''
if (Test-Path -LiteralPath $JobHookPath) {
    $heldHook = [IO.File]::ReadAllText($JobHookPath)
}
if ($heldHook -ceq [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($JobHook))) {
    'CHECK job-hook OK'
} else {
    "CHECK job-hook DRIFT $JobHookPath does not hold the job-started hook"
}
$heldHookVariable = [string][Environment]::GetEnvironmentVariable($JobHookVariable, $VariableScope)
if ($heldHookVariable -ceq $JobHookPath) {
    'CHECK job-hook-variable OK'
} else {
    "CHECK job-hook-variable DRIFT $JobHookVariable holds: $heldHookVariable"
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
    $tools = Join-Path $install.Root '_work\_tool'
    foreach ($version in @($install.Python)) {
        $pip = Join-Path $tools "Python\$version\x64\Scripts\pip.exe"
        $complete = Join-Path $tools "Python\$version\x64.complete"
        if ((Test-Path -LiteralPath $pip) -and (Test-Path -LiteralPath $complete)) {
            "CHECK python:${service}:$version OK"
        } else {
            "CHECK python:${service}:$version DRIFT the tool cache lacks $pip or $complete"
        }
    }
}
exit 0
