param(
    [int]$Start = $PID,
    [string]$WorkerName = 'Runner.Worker.exe'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$process = Get-CimInstance Win32_Process -Filter "ProcessId = $Start"
do {
    $child = $process
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($child.ParentProcessId) AND ProcessId <> 0" |
        Where-Object { $_.CreationDate -lt $child.CreationDate }
} while ($null -ne $process -and $process.Name -ne $WorkerName)
if ($null -eq $process) {
    throw "RUNNER_JOB_PRIORITY_FAILED: no $WorkerName is among the ancestors of process $Start, so the job would run at the fleet's own priority"
}
(Get-Process -Id $process.ProcessId).PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
"runner job: $WorkerName $($process.ProcessId) and every process it starts run at BelowNormal, under the fleet's checks"
