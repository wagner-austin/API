Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# A registered task's definition, read the way the fleet's scripts read the
# scheduler: through Task Scheduler's COM service. The scheduled-task cmdlets
# read every task's definition and fail while another process deletes one
# (scripts/FleetSchedule.ps1 carries the measurement), so a suite asserting a
# task's shape through them fails on tasks it never touched. MCPs board task
# d69786fa.

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
    $scheduler = New-Object -ComObject Schedule.Service
    $scheduler.Connect()
    $found = @($scheduler.GetFolder('\').GetTasks(1) | Where-Object { $_.Name -eq $Name })
    if ($found.Count -eq 0) {
        return $null
    }
    return [xml]$found[0].Xml
}
