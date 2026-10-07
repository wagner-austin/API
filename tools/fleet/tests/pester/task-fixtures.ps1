Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# A registered task's definition, read the way the fleet's scripts read the
# scheduler: through Task Scheduler's COM service. The scheduled-task cmdlets
# read every task's definition and fail while another process deletes one
# (scripts/FleetSchedule.ps1 carries the measurement), so a suite asserting a
# task's shape through them fails on tasks it never touched. MCPs board task
# d69786fa.

function Get-RootTask {
    <#
    .SYNOPSIS
        The task registered under a name in the root folder.
    .PARAMETER Name
        The task's exact name.
    .OUTPUTS
        The scheduler's registered-task object, whose State and
        LastTaskResult the result render reads; $null when no task has the
        name.
    #>
    param([Parameter(Mandatory)][string]$Name)
    $scheduler = New-Object -ComObject Schedule.Service
    $scheduler.Connect()
    $found = @($scheduler.GetFolder('\').GetTasks(1) | Where-Object { $_.Name -eq $Name })
    if ($found.Count -eq 0) {
        return $null
    }
    return $found[0]
}

function Get-TaskDefinition {
    <#
    .SYNOPSIS
        The task registered under a name in the root folder, as its XML.
    .PARAMETER Name
        The task's exact name.
    .OUTPUTS
        System.Xml.XmlDocument: the definition; $null when no task has the name.
    #>
    [OutputType([xml])]
    param([Parameter(Mandatory)][string]$Name)
    $found = Get-RootTask $Name
    if ($null -eq $found) {
        return $null
    }
    return [xml]$found.Xml
}

function Invoke-TaskCleanup {
    <#
    .SYNOPSIS
        Stop and delete a case's task, when it is registered.
    .DESCRIPTION
        Stopped before it is deleted, since deleting a running task's
        registration leaves its process running.
    .PARAMETER Name
        The task's exact name.
    #>
    param([Parameter(Mandatory)][string]$Name)
    if ($null -ne (Get-RootTask $Name)) {
        Stop-ScheduledTask -TaskName $Name
        $scheduler = New-Object -ComObject Schedule.Service
        $scheduler.Connect()
        $scheduler.GetFolder('\').DeleteTask($Name, 0)
    }
}
