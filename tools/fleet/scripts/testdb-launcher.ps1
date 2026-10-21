Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The Docker Desktop launcher task install-windows-testdb.ps1 makes sure a
# Windows testdb node has (MCPs board task 801ecab0), read and registered
# through Task Scheduler. It is a library of its own so a suite can run both
# functions for real by name, which MCPs' harness requires of every seam that
# reaches a service (PS-EFFECT-SEAM-TWIN, its GuardEffectSeam.ps1; board task
# 2a044a37): the installer's -ReadLaunchers and -RegisterLauncher defaults
# call them, and it dot-sources this file from beside itself.

function Get-RootTaskRunning {
    <#
    .SYNOPSIS
        The names of a folder's tasks with an action that runs a program:
        what the installer's default -ReadLaunchers answers.
    .DESCRIPTION
        Read through Task Scheduler's COM service, hidden tasks included,
        because the scheduled-task cmdlets fail while another process
        deletes a task (MCPs board task d69786fa). An action's Path is read
        only on an exec action (type 0), the one kind that has it. A folder
        Task Scheduler does not hold raises FileNotFoundException.
    .PARAMETER Execute
        The program's full path.
    .PARAMETER Folder
        The folder whose tasks are read; the launcher lives in the root.
    .OUTPUTS
        String[]: the tasks' names, none when no task runs the program.
    #>
    [OutputType([string[]])]
    param([string]$Execute, [string]$Folder = '\')
    $scheduler = New-Object -ComObject Schedule.Service
    $scheduler.Connect()
    return [string[]]@($scheduler.GetFolder($Folder).GetTasks(1) | Where-Object {
            @($_.Definition.Actions | Where-Object { $_.Type -eq 0 -and $_.Path -eq $Execute }).Count -gt 0
        } | ForEach-Object { $_.Name })
}

function Register-DockerDesktopLauncher {
    <#
    .SYNOPSIS
        Register the root folder task that starts Docker Desktop as the
        account running this script, with no logon and no trigger: what the
        installer's default -RegisterLauncher does.
    .DESCRIPTION
        S4U at the highest run level, so it needs no logon and starts the
        program as the account that owns Docker Desktop's settings and WSL
        distro; no trigger, because MCPs' node standard is what runs it, at
        boot and whenever the daemon is down. A name Task Scheduler cannot
        hold raises CimException.
    .PARAMETER Name
        The task's name.
    .PARAMETER Execute
        Docker Desktop.exe's full path.
    #>
    param([string]$Name, [string]$Execute)
    $action = New-ScheduledTaskAction -Execute $Execute
    $principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value) -LogonType S4U -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
    [void](Register-ScheduledTask -TaskPath '\' -TaskName $Name -Action $action -Principal $principal -Settings $settings)
}
