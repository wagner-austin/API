param(
    [string]$Target = 'C:/fleet/stage/MCPs-packages-maketools-1790000000',
    [string]$TaskName = 'fleet-MCPs-packages-maketools-1790000000',
    [int]$LaunchSeconds = 30
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$build = "$Target/build.ps1"
$recorded = "$Target/build.pid"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$build`""
$settings = New-ScheduledTaskSettingsSet -Priority 4 -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value) -LogonType S4U
Register-ScheduledTask -TaskName $TaskName -Action $action -Settings $settings -Principal $principal -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
$deadline = (Get-Date).AddSeconds($LaunchSeconds)
while (-not (Test-Path -LiteralPath $recorded) -and ((Get-Date) -lt $deadline)) {
    Start-Sleep -Milliseconds 500
}
if (-not (Test-Path -LiteralPath $recorded)) {
    throw "FLEET_LAUNCH_NOT_STARTED: $TaskName registered, but its build had not recorded itself after $LaunchSeconds s"
}
Write-Output 'launched'
