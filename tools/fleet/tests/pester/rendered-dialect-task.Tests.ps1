Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The scheduled task a Windows build runs in, launched and stopped from the
# committed renders under rendered/ (fleet.core.windows_task, MCPs board task
# d69786fa, A2). Everything is real: each case registers a disposable task
# named fleet-test-<guid>, runs a build.ps1 it wrote under $TestDrive, and
# kills a real process tree, because a task that registers but never starts,
# or a stop that leaves a child running (sedona, 2026-09-23), is exactly what
# reading the text never caught. Only a failing taskkill is a stand-in.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')
    . (Join-Path $PSScriptRoot 'task-fixtures.ps1')

    function Initialize-Dispatch {
        param([string]$Build)
        $target = Join-Path $TestDrive ('run-' + [guid]::NewGuid().ToString('N'))
        [void][System.IO.Directory]::CreateDirectory($target)
        [System.IO.File]::WriteAllText((Join-Path $target 'build.ps1'), $Build)
        return $target
    }

    function Wait-RecordedId {
        param([string]$Path)
        $deadline = (Get-Date).AddSeconds(60)
        while (-not (Test-Path -LiteralPath $Path) -and ((Get-Date) -lt $deadline)) {
            Start-Sleep -Milliseconds 200
        }
        return [int](Get-Content -Raw -LiteralPath $Path).Trim()
    }

    function Test-Alive {
        param([int]$Id)
        return $null -ne (Get-CimInstance Win32_Process -Filter "ProcessId=$Id")
    }

    # A build blocked on a child, as sedona measured it: build.ps1 records its
    # id as a real build does, then waits on a PowerShell child that records
    # its own and sleeps. Both ids go on the case's list for AfterEach.
    function Initialize-BuildTree {
        param([string]$Target)
        $child = Join-Path $Target 'child.pid'
        $build = "`$PID | Set-Content -LiteralPath `"`$PSScriptRoot/build.pid`"`r`n" +
            "& `"`$PSHOME\powershell.exe`" -NoProfile -Command `"```$PID | Set-Content -LiteralPath '$child'; Start-Sleep 300`"`r`n" +
            "exit `$LASTEXITCODE`r`n"
        [System.IO.File]::WriteAllText((Join-Path $Target 'build.ps1'), $build)
        $process = Start-Process -FilePath "$PSHOME\powershell.exe" -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$Target/build.ps1") -PassThru -WindowStyle Hidden
        $script:started.Add($process.Id)
        $buildId = Wait-RecordedId (Join-Path $Target 'build.pid')
        $childId = Wait-RecordedId $child
        $script:started.Add($childId)
        $buildId | Should -Be $process.Id
        return [pscustomobject]@{ Build = $buildId; Child = $childId }
    }
}

Describe 'The launch' {
    BeforeEach {
        $script:taskName = 'fleet-test-' + [guid]::NewGuid().ToString('N').Substring(0, 12)
    }
    AfterEach {
        Invoke-TaskCleanup $script:taskName
    }
    It 'registers the build as an S4U task that runs on battery at priority 4, starts it, and says so once the build has recorded itself' {
        # The build sleeps before recording, so the launch is seen waiting.
        $target = Initialize-Dispatch "Start-Sleep -Seconds 2`r`n`$PID | Set-Content -LiteralPath `"`$PSScriptRoot/build.pid`"`r`n"
        Invoke-Rendered 'dialect-launch' @{ Target = $target; TaskName = $script:taskName } | Should -BeExactly 'launched'
        $task = (Get-TaskDefinition $script:taskName).Task
        $task.Actions.Exec.Command | Should -BeExactly 'powershell.exe'
        $task.Actions.Exec.Arguments | Should -BeExactly "-NoProfile -ExecutionPolicy Bypass -File `"$target/build.ps1`""
        $task.Principals.Principal.LogonType | Should -BeExactly 'S4U'
        $task.Settings.Priority | Should -BeExactly '4'
        $task.Settings.DisallowStartIfOnBatteries | Should -BeExactly 'false'
        $task.Settings.StopIfGoingOnBatteries | Should -BeExactly 'false'
        $task.Settings.MultipleInstancesPolicy | Should -BeExactly 'IgnoreNew'
        $task.Settings.ExecutionTimeLimit | Should -BeExactly 'PT0S'
    }
    It 'refuses by name a build that never records itself' {
        $target = Initialize-Dispatch "exit 0`r`n"
        { Invoke-Rendered 'dialect-launch' @{ Target = $target; TaskName = $script:taskName; LaunchSeconds = 3 } } |
            Should -Throw "FLEET_LAUNCH_NOT_STARTED: $($script:taskName) registered, but its build had not recorded itself after 3 s"
    }
    # The elevated runner's launch (MCPs board task a98d7083): registering at
    # RunLevel Highest needs an administrator's full token, which the hub's
    # sessions and CI's Windows runners hold, as an elevated fleet runner's
    # ssh session does.
    It 'registers an elevated build at RunLevel Highest and starts it like any other' {
        $target = Initialize-Dispatch "`$PID | Set-Content -LiteralPath `"`$PSScriptRoot/build.pid`"`r`n"
        Invoke-Rendered 'dialect-launch-elevated' @{ Target = $target; TaskName = $script:taskName } | Should -BeExactly 'launched'
        $task = (Get-TaskDefinition $script:taskName).Task
        $task.Principals.Principal.LogonType | Should -BeExactly 'S4U'
        $task.Principals.Principal.RunLevel | Should -BeExactly 'HighestAvailable'
        $task.Settings.Priority | Should -BeExactly '4'
    }
    It 'refuses by name an elevated build that never records itself' {
        $target = Initialize-Dispatch "exit 0`r`n"
        { Invoke-Rendered 'dialect-launch-elevated' @{ Target = $target; TaskName = $script:taskName; LaunchSeconds = 3 } } |
            Should -Throw "FLEET_LAUNCH_NOT_STARTED: $($script:taskName) registered, but its build had not recorded itself after 3 s"
    }
}

Describe 'The stop' {
    BeforeEach {
        $script:taskName = 'fleet-test-' + [guid]::NewGuid().ToString('N').Substring(0, 12)
        $script:started = [System.Collections.Generic.List[int]]::new()
    }
    AfterEach {
        Invoke-TaskCleanup $script:taskName
        foreach ($id in $script:started) {
            if (Test-Alive $id) {
                Stop-Process -Id $id -Force
            }
        }
    }

    It 'ends the build and the child it waits on, then stops and unregisters the task' {
        $target = Initialize-Dispatch ''
        $tree = Initialize-BuildTree $target
        Register-ScheduledTask -TaskName $script:taskName -Action (New-ScheduledTaskAction -Execute 'cmd.exe' -Argument '/c exit 0') | Out-Null
        Invoke-Rendered 'dialect-stop' @{ Target = $target; TaskName = $script:taskName } | Select-Object -Last 1 |
            Should -BeExactly "stopped $($script:taskName)"
        Test-Alive $tree.Build | Should -BeFalse
        Test-Alive $tree.Child | Should -BeFalse
        Get-TaskDefinition $script:taskName | Should -BeNullOrEmpty
    }
    It 'kills nothing when the recorded id names another process, since ids are reused' {
        $target = Initialize-Dispatch ''
        [System.IO.File]::WriteAllText((Join-Path $target 'build.pid'), "$PID`r`n")
        Invoke-Rendered 'dialect-stop' @{ Target = $target; TaskName = $script:taskName } | Should -BeExactly "stopped $($script:taskName)"
        Test-Alive $PID | Should -BeTrue
    }
    It 'stops by task alone a build that never recorded an id' {
        $target = Initialize-Dispatch ''
        Invoke-Rendered 'dialect-stop' @{ Target = $target; TaskName = $script:taskName } | Should -BeExactly "stopped $($script:taskName)"
    }
    It 'refuses by name a kill taskkill refuses, so a stop that ended nothing never reads as one' {
        $target = Initialize-Dispatch ''
        $tree = Initialize-BuildTree $target
        $taskkill = Initialize-StandIn 128
        { Invoke-Rendered 'dialect-stop' @{ Target = $target; TaskName = $script:taskName; Taskkill = $taskkill.Path } } |
            Should -Throw "FLEET_STOP_KILL_FAILED: taskkill of $($tree.Build) exited 128"
        [System.IO.File]::ReadAllText($taskkill.Record).Trim() | Should -BeExactly "/PID $($tree.Build) /T /F"
        Test-Alive $tree.Build | Should -BeTrue
    }
}
