param(
    [string]$Target = 'C:/fleet/stage/MCPs-packages-maketools-1790000000',
    [string]$TaskName = 'fleet-MCPs-packages-maketools-1790000000'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$result = "$Target/result.txt"
$log = "$Target/result.txt.log"
$scheduler = New-Object -ComObject Schedule.Service
$scheduler.Connect()
$task = @($scheduler.GetFolder('\').GetTasks(1) | Where-Object { $_.Name -eq $TaskName })
$going = ($task.Count -gt 0) -and (@(2, 4) -contains [int]$task[0].State)
if ((-not $going) -and (-not (Test-Path -LiteralPath $result))) {
    $code = 125
    $how = "task $TaskName is no longer registered"
    if ($task.Count -gt 0) {
        $states = @{ 0 = 'Unknown'; 1 = 'Disabled'; 2 = 'Queued'; 3 = 'Ready'; 4 = 'Running' }
        $state = [int]$task[0].State
        $last = [int]$task[0].LastTaskResult
        if ($last -ne 0) {
            $code = $last
        }
        $how = 'task {0} ended with Task Scheduler state {1} ({2}) and last result 0x{3:X8}' -f $TaskName, $state, $states[$state], $last
    }
    $line = "FLEET_UNIT_ENDED: $how before the build wrote its status; recorded exit $code"
    [System.IO.File]::AppendAllText($log, "$line`r`n")
    $code | Set-Content -LiteralPath $result
}
if (Test-Path -LiteralPath $result) {
    $file = Get-Item -LiteralPath $result
    $code = (Get-Content -Raw -LiteralPath $result).Trim()
    $age = $file.LastWriteTimeUtc - [datetime]'1970-01-01'
    $epoch = [int][Math]::Floor($age.TotalSeconds)
    "$code $epoch"
}
