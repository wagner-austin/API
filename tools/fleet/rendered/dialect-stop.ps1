param(
    [string]$Target = 'C:/fleet/stage/MCPs-packages-maketools-1790000000',
    [string]$TaskName = 'fleet-MCPs-packages-maketools-1790000000',
    [string]$Taskkill = "$env:SystemRoot\System32\taskkill.exe"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$recorded = "$Target/build.pid"
$build = "$Target/build.ps1"
if (Test-Path -LiteralPath $recorded) {
    $buildPid = [int](Get-Content -Raw -LiteralPath $recorded).Trim()
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$buildPid"
    if (($null -ne $process) -and ($process.CommandLine -like "*$build*")) {
        & $Taskkill /PID $buildPid /T /F
        if ($LASTEXITCODE -ne 0) {
            throw "FLEET_STOP_KILL_FAILED: taskkill of $buildPid exited $LASTEXITCODE"
        }
    }
}
$scheduler = New-Object -ComObject Schedule.Service
$scheduler.Connect()
$root = $scheduler.GetFolder('\')
$task = @($root.GetTasks(1) | Where-Object { $_.Name -eq $TaskName })
if ($task.Count -gt 0) {
    Stop-ScheduledTask -TaskName $TaskName
    $root.DeleteTask($TaskName, 0)
}
Write-Output "stopped $TaskName"
