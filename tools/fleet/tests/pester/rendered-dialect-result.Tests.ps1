Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# What a node reports about a dispatch, executed from the committed
# rendered/dialect-result.ps1 (fleet.core.windows_result, MCPs board task
# 4e3afe4f). The render reads the build's scheduled task before the result
# file, so every case runs it against a real task registered as the launch
# registers one, an S4U principal for this account, named
# fleet-test-<guid>: still running, ended with a status of its own, or never
# registered. tests/test_windows_result.py ends a real build's process.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')
    . (Join-Path $PSScriptRoot 'task-fixtures.ps1')

    function Initialize-Target {
        param([string]$Name)
        $target = Join-Path $TestDrive $Name
        [void][System.IO.Directory]::CreateDirectory($target)
        return $target
    }

    function Initialize-CaseTask {
        <#
        .SYNOPSIS
            Register the case's task with one action and start it.
        .PARAMETER Execute
            The program the task runs.
        .PARAMETER Argument
            Its arguments.
        #>
        param([string]$Execute, [string]$Argument)
        $action = New-ScheduledTaskAction -Execute $Execute -Argument $Argument
        $principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value) -LogonType S4U
        Register-ScheduledTask -TaskName $script:taskName -Action $action -Principal $principal -Force | Out-Null
        Start-ScheduledTask -TaskName $script:taskName
    }

    function Wait-CaseTask {
        <#
        .SYNOPSIS
            Wait until the case's task reads a state and a last result.
        .PARAMETER State
            Task Scheduler's TASK_STATE number.
        .PARAMETER LastResult
            The LastTaskResult it must also read.
        #>
        param([int]$State, [int]$LastResult)
        $deadline = (Get-Date).AddSeconds(60)
        $task = Get-RootTask $script:taskName
        while ((($task.State -ne $State) -or ($task.LastTaskResult -ne $LastResult)) -and ((Get-Date) -lt $deadline)) {
            Start-Sleep -Milliseconds 200
            $task = Get-RootTask $script:taskName
        }
        $task.State | Should -Be $State
        $task.LastTaskResult | Should -Be $LastResult
    }

    function Assert-Reported {
        <#
        .SYNOPSIS
            The render's answer is the status and the UTC epoch second the
            result file was written.
        .PARAMETER Said
            What the render printed.
        .PARAMETER Target
            The dispatch directory.
        .PARAMETER Code
            The status it must report.
        #>
        param([object[]]$Said, [string]$Target, [string]$Code)
        $result = Join-Path $Target 'result.txt'
        $written = [int][Math]::Floor(((Get-Item -LiteralPath $result).LastWriteTimeUtc - [datetime]'1970-01-01').TotalSeconds)
        $Said | Should -HaveCount 1
        $Said[0] | Should -BeExactly "$Code $written"
        [System.IO.File]::ReadAllText($result).Trim() | Should -BeExactly $Code
    }
}

Describe 'The result render' {
    BeforeEach {
        $script:taskName = 'fleet-test-' + [guid]::NewGuid().ToString('N').Substring(0, 12)
    }
    AfterEach {
        Invoke-TaskCleanup $script:taskName
    }

    It 'prints nothing and writes nothing while the build''s task is still running without a result' {
        $target = Initialize-Target 'running'
        Initialize-CaseTask -Execute "$PSHOME\powershell.exe" -Argument '-NoProfile -Command Start-Sleep 300'
        Wait-CaseTask -State 4 -LastResult 0x41301
        @(Invoke-Rendered 'dialect-result' @{ Target = $target; TaskName = $script:taskName }).Count | Should -Be 0
        @([System.IO.Directory]::GetFileSystemEntries($target)).Count | Should -Be 0
    }
    It 'prints the status the build wrote and the UTC epoch second it wrote it, touching nothing' {
        $target = Initialize-Target 'finished'
        $result = Join-Path $target 'result.txt'
        [System.IO.File]::WriteAllText($result, "3`r`n")
        [System.IO.File]::SetLastWriteTimeUtc($result, [datetime]::new(2026, 9, 27, 17, 0, 0, [System.DateTimeKind]::Utc))
        Invoke-Rendered 'dialect-result' @{ Target = $target; TaskName = $script:taskName } | Should -BeExactly '3 1790528400'
        Test-Path -LiteralPath (Join-Path $target 'result.txt.log') | Should -BeFalse
    }
    It 'floors a write late in its second rather than rounding it into the next' {
        $target = Initialize-Target 'late'
        $result = Join-Path $target 'result.txt'
        [System.IO.File]::WriteAllText($result, "0`r`n")
        [System.IO.File]::SetLastWriteTimeUtc($result, [datetime]::new(2026, 9, 27, 17, 0, 0, 600, [System.DateTimeKind]::Utc))
        Invoke-Rendered 'dialect-result' @{ Target = $target; TaskName = $script:taskName } | Should -BeExactly '0 1790528400'
    }
    It 'records the exit status of a task that ended without the build''s status, with a line saying how' {
        $target = Initialize-Target 'ended'
        $log = Join-Path $target 'result.txt.log'
        [System.IO.File]::WriteAllText($log, "npm warn deprecated boolean@3.2.0`r`n")
        Initialize-CaseTask -Execute 'cmd.exe' -Argument '/c exit 3'
        Wait-CaseTask -State 3 -LastResult 3
        $said = @(Invoke-Rendered 'dialect-result' @{ Target = $target; TaskName = $script:taskName })
        Assert-Reported -Said $said -Target $target -Code '3'
        [System.IO.File]::ReadAllText($log) | Should -BeExactly ("npm warn deprecated boolean@3.2.0`r`n" +
            "FLEET_UNIT_ENDED: task $($script:taskName) ended with Task Scheduler state 3 (Ready) and last result 0x00000003 before the build wrote its status; recorded exit 3`r`n")
        $again = @(Invoke-Rendered 'dialect-result' @{ Target = $target; TaskName = $script:taskName })
        $again | Should -HaveCount 1
        $again[0] | Should -BeExactly $said[0]
        ([regex]::Matches([System.IO.File]::ReadAllText($log), 'FLEET_UNIT_ENDED')).Count | Should -Be 1
    }
    It 'never records a pass for a task that ended 0 without the build''s status' {
        $target = Initialize-Target 'zero'
        Initialize-CaseTask -Execute 'cmd.exe' -Argument '/c exit 0'
        Wait-CaseTask -State 3 -LastResult 0
        $said = @(Invoke-Rendered 'dialect-result' @{ Target = $target; TaskName = $script:taskName })
        Assert-Reported -Said $said -Target $target -Code '125'
        [System.IO.File]::ReadAllText((Join-Path $target 'result.txt.log')) | Should -BeExactly (
            "FLEET_UNIT_ENDED: task $($script:taskName) ended with Task Scheduler state 3 (Ready) and last result 0x00000000 before the build wrote its status; recorded exit 125`r`n")
    }
    It 'records the unexplained status for a task no longer registered, starting the transcript when there was none' {
        $target = Initialize-Target 'unregistered'
        $said = @(Invoke-Rendered 'dialect-result' @{ Target = $target; TaskName = $script:taskName })
        Assert-Reported -Said $said -Target $target -Code '125'
        [System.IO.File]::ReadAllText((Join-Path $target 'result.txt.log')) | Should -BeExactly (
            "FLEET_UNIT_ENDED: task $($script:taskName) is no longer registered before the build wrote its status; recorded exit 125`r`n")
    }
}
