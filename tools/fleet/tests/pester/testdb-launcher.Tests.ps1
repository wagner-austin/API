Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# tools/fleet/scripts/testdb-launcher.ps1, the Task Scheduler half of
# install-windows-testdb.ps1, run for real: the installer's suite hands its
# -ReadLaunchers and -RegisterLauncher seams stand-ins, and these cases are
# their real twins (MCPs' harness rule PS-EFFECT-SEAM-TWIN, board task
# 2a044a37). Each case's task has a name unique to the case and is deleted
# after it, so no case reads or changes a task it did not make.

BeforeAll {
    . (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'scripts\testdb-launcher.ps1')
    . (Join-Path $PSScriptRoot 'task-fixtures.ps1')
}

Describe 'testdb-launcher' {
    It 'registers a launcher as this account, S4U with no trigger, and reads it back by the program it runs' {
        $name = 'fleet-test-' + [guid]::NewGuid().ToString('N').Substring(0, 12)
        $desktop = Join-Path $TestDrive "$name\Docker Desktop.exe"
        try {
            @(Get-RootTaskRunning $desktop) | Should -Be @()
            Register-DockerDesktopLauncher $name $desktop
            @(Get-RootTaskRunning $desktop) | Should -Be @($name)
            $task = (Get-TaskDefinition $name).Task
            $task.Actions.Exec.Command | Should -BeExactly $desktop
            $task.Principals.Principal.UserId | Should -BeExactly ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value)
            $task.Principals.Principal.LogonType | Should -BeExactly 'S4U'
            @($task.SelectNodes("*[local-name()='Triggers']/*")).Count | Should -Be 0
        } finally {
            Invoke-TaskCleanup $name
        }
    }
    It 'refuses a task name Task Scheduler cannot hold with CimException, and registers nothing' {
        $desktop = Join-Path $TestDrive 'refused\Docker Desktop.exe'
        { Register-DockerDesktopLauncher 'fleet-test|refused' $desktop } |
            Should -Throw -ExceptionType ([Microsoft.Management.Infrastructure.CimException])
        @(Get-RootTaskRunning $desktop) | Should -Be @()
    }
    It 'refuses a folder Task Scheduler does not hold with FileNotFoundException' {
        $folder = '\fleet-test-absent-' + [guid]::NewGuid().ToString('N').Substring(0, 12)
        { Get-RootTaskRunning (Join-Path $TestDrive 'Docker Desktop.exe') $folder } |
            Should -Throw -ExceptionType ([System.IO.FileNotFoundException])
    }
}
