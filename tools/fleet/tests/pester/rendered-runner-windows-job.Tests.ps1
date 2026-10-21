Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The job-started hook every Windows runner host lays, executed from its
# committed render, rendered/runner-windows-job-started.ps1
# (fleet.core.runner_windows_job_render, MCPs board task db32d739). It runs in
# this process, so the harness counts its lines. Its walk starts at a child
# this suite starts and names this process as the worker, so the one priority
# it changes is this process's own, which each case puts back.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')
    $script:selfName = (Get-CimInstance Win32_Process -Filter "ProcessId = $PID").Name

    function Start-Child {
        <#
        .SYNOPSIS
            A child of this process that waits until it is stopped.
        .OUTPUTS
            System.Diagnostics.Process: the child.
        #>
        return Start-Process -FilePath (Get-Process -Id $PID).Path -ArgumentList @('-NoProfile', '-NonInteractive', '-Command', 'Start-Sleep -Seconds 120') -WindowStyle Hidden -PassThru
    }
}

Describe 'The job-started hook' {
    BeforeEach {
        $script:priority = [System.Diagnostics.Process]::GetCurrentProcess().PriorityClass
    }
    AfterEach {
        [System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = $script:priority
    }
    It 'lowers the worker among its ancestors to BelowNormal, which every process the worker then starts inherits' {
        [System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::Normal
        $child = Start-Child
        try {
            $printed = [string[]]@(Invoke-Rendered 'runner-windows-job-started' @{ Start = $child.Id; WorkerName = $script:selfName })
            $printed | Should -Be @("runner job: $($script:selfName) $PID and every process it starts run at BelowNormal, under the fleet's checks")
            [System.Diagnostics.Process]::GetCurrentProcess().PriorityClass | Should -Be 'BelowNormal'
            $later = Start-Child
            try {
                $later.PriorityClass | Should -Be 'BelowNormal'
            } finally {
                Stop-Process -Id $later.Id -Force
            }
            $child.PriorityClass | Should -Be 'Normal'
        } finally {
            Stop-Process -Id $child.Id -Force
        }
    }
    It 'refuses a walk that reaches no worker, and lowers nothing' {
        [System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::Normal
        $child = Start-Child
        try {
            { Invoke-Rendered 'runner-windows-job-started' @{ Start = $child.Id; WorkerName = 'Runner.Worker.exe' } } | Should -Throw "RUNNER_JOB_PRIORITY_FAILED: no Runner.Worker.exe is among the ancestors of process $($child.Id), so the job would run at the fleet's own priority"
            [System.Diagnostics.Process]::GetCurrentProcess().PriorityClass | Should -Be 'Normal'
        } finally {
            Stop-Process -Id $child.Id -Force
        }
    }
}
