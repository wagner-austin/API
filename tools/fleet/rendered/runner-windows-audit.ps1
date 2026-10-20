param(
    [string]$Sc = "$env:SystemRoot\System32\sc.exe",
    [hashtable[]]$Installs = @(
        @{ Service = 'actions.runner.wagner-austin-chat.loki' }
    )
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
function Get-Answer {
    param([string]$Tool, [string[]]$Arguments)
    $text = ((& $Tool @Arguments) -join ' ') -replace '\s+', ' '
    return [pscustomobject]@{ Code = $LASTEXITCODE; Text = $text.Trim() }
}
foreach ($install in $Installs) {
    $service = $install.Service
    $state = Get-Answer $Sc @('query', $service)
    if ($state.Code -eq 0 -and $state.Text -match 'STATE : 4 RUNNING') {
        "CHECK service:$service OK"
    } else {
        "CHECK service:$service DRIFT sc.exe query exited $($state.Code): $($state.Text)"
    }
    $config = Get-Answer $Sc @('qc', $service)
    if ($config.Code -eq 0 -and $config.Text -match 'START_TYPE : 2 AUTO_START' -and $config.Text -match 'SERVICE_START_NAME : LocalSystem') {
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
