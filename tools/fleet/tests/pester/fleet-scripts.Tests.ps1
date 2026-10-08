Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# tools/fleet/scripts: the three schedule entries over FleetSchedule.ps1
# (MCPs board task d69786fa). What a tick itself does is fleet.cli.tick's,
# tested in tests/test_cli_tick.py; the tasks here only run it (MCPs board
# task 94ac1c4f).
#
# NOTHING HERE STARTS A REAL AGENT. poetry is a .cmd stand-in that exits
# with a chosen status; every task this suite registers has a disposable
# name and runs that stand-in, so a scheduler that starts it starts nothing
# that touches the queue.
#
# THE RECORD HAS ONE WRITER. Register-FleetTick's trigger repeats every
# three minutes from midnight, so Task Scheduler starts a registered task at
# the next three-minute boundary: on the hub a task registered at
# 2026-10-08T05:05:57Z was started at 05:06:01Z (MCPs board task 8e8e8769). A case
# whose run crosses a boundary would find that call in the record beside the
# announce the entry under test makes, and on a loaded hub the review of
# 8e8e8769 read a pre-registered 'stale' task's call where the announce was
# expected. So the stand-in records only the announce lanes, which only an
# entry runs, synchronously, from this console; the lanes a schedule runs
# (hub, node, elevated, and a case's 'stale') exit without writing, and each
# case compares the whole record.

BeforeAll {
    $scriptsRoot = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'scripts'
    . (Join-Path $scriptsRoot 'FleetSchedule.ps1')
    . (Join-Path $PSScriptRoot 'task-fixtures.ps1')
    $script:scriptsRoot = $scriptsRoot
    $script:registerHub = Join-Path $scriptsRoot 'register-agent-schedule.ps1'
    $script:unregisterHub = Join-Path $scriptsRoot 'unregister-agent-schedule.ps1'
    $script:registerNodes = Join-Path $scriptsRoot 'register-node-agents.ps1'
    $script:apiRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptsRoot '..\..\..'))

    # An entry fails by throwing or by `exit <code>`; the second only sets
    # $LASTEXITCODE, so every entry runs through here, where a non-zero exit
    # is itself an error.
    function Invoke-TestEntry {
        param([string]$Path, [hashtable]$Parameters)
        $global:LASTEXITCODE = 0
        & $Path @Parameters
        if ($LASTEXITCODE -ne 0) {
            throw "TEST_ENTRY_EXITED: $Path exited $LASTEXITCODE"
        }
    }

    function Initialize-FleetTickWorld {
        <#
        .SYNOPSIS
            A disposable API root with a tools\fleet directory, a log
            directory, and a stand-in poetry that exits with the given
            status, first appending the arguments and working directory of
            each announce-lane call to a record.
        .NOTES
            The lane is read with delayed expansion, so the quotes the tick's
            paths carry are text in the comparison and never end an operand.
        #>
        param([int]$ExitCode = 0)
        $root = Join-Path $TestDrive ('world-' + [guid]::NewGuid().ToString('N'))
        $api = Join-Path $root 'api'
        [void][System.IO.Directory]::CreateDirectory((Join-Path $api 'tools\fleet'))
        $record = Join-Path $root 'poetry-calls.txt'
        $poetry = Join-Path $root 'poetry.cmd'
        $body = "@echo off`r`nsetlocal EnableDelayedExpansion`r`nset `"said=%*`"`r`n" +
            "if `"!said:announce=!`"==`"!said!`" exit /b $ExitCode`r`n" +
            "echo %*>>`"$record`"`r`necho cwd=%CD%>>`"$record`"`r`nexit /b $ExitCode`r`n"
        [System.IO.File]::WriteAllText($poetry, $body, [System.Text.Encoding]::ASCII)
        return [pscustomobject]@{
            Api = $api; Fleet = Join-Path $api 'tools\fleet'; Logs = Join-Path $root 'logs'
            Poetry = $poetry; Record = $record
        }
    }

    function Read-PoetryCall {
        param([pscustomobject]$World)
        if (-not [System.IO.File]::Exists($World.Record)) {
            return [string[]]@()
        }
        return [string[]]@([System.IO.File]::ReadAllLines($World.Record) | ForEach-Object { $_.TrimEnd() })
    }

    function Get-TestTickArgument {
        param([pscustomobject]$World, [string]$Lane, [string]$Node = '')
        $line = "run -- python -m fleet.cli.tick --api-root `"$($World.Api)`" --log-directory `"$($World.Logs)`" --lane $Lane"
        if ($Node -ne '') {
            $line += " --node $Node"
        }
        return $line
    }
}

Describe 'The tick command line' {
    It 'names the <Lane> lane' -ForEach @(
        @{ Lane = 'hub'; Node = ''; Expected = 'run -- python -m fleet.cli.tick --api-root "C:\api" --log-directory "C:\logs" --lane hub' },
        @{ Lane = 'hub-announce'; Node = ''; Expected = 'run -- python -m fleet.cli.tick --api-root "C:\api" --log-directory "C:\logs" --lane hub-announce' },
        @{ Lane = 'node'; Node = 'sedona'; Expected = 'run -- python -m fleet.cli.tick --api-root "C:\api" --log-directory "C:\logs" --lane node --node sedona' },
        @{ Lane = 'announce'; Node = 'sedona'; Expected = 'run -- python -m fleet.cli.tick --api-root "C:\api" --log-directory "C:\logs" --lane announce --node sedona' },
        @{ Lane = 'elevated'; Node = 'sedona'; Expected = 'run -- python -m fleet.cli.tick --api-root "C:\api" --log-directory "C:\logs" --lane elevated --node sedona' },
        @{ Lane = 'elevated-announce'; Node = 'sedona'; Expected = 'run -- python -m fleet.cli.tick --api-root "C:\api" --log-directory "C:\logs" --lane elevated-announce --node sedona' }
    ) {
        Get-FleetTickCommandLine -ApiRoot 'C:\api' -LogDirectory 'C:\logs' -Lane $Lane -Node $Node | Should -BeExactly $Expected
    }
    It 'refuses a lane the tick does not have' {
        { Get-FleetTickCommandLine -ApiRoot 'C:\api' -LogDirectory 'C:\logs' -Lane 'agent' -Node '' } | Should -Throw '*Lane*'
    }
    It 'resolves poetry to the absolute path of what answers, and stops when nothing does' {
        $world = Initialize-FleetTickWorld
        Resolve-FleetPoetry $world.Poetry | Should -BeExactly $world.Poetry
        $missing = 'poetry-' + [guid]::NewGuid().ToString('N')
        { Resolve-FleetPoetry $missing } | Should -Throw "*'$missing'*"
    }
}

Describe 'The schedules' {
    BeforeEach {
        $script:prefix = 'API-FleetTest-' + [guid]::NewGuid().ToString('N').Substring(0, 8) + '-'
        $script:world = Initialize-FleetTickWorld
    }
    AfterEach {
        foreach ($task in @(Get-FleetScheduledTask -Prefix $script:prefix -Suffix '')) {
            [void](Unregister-FleetTick $task.TaskName)
        }
    }
    It 'registers poetry running the hub tick every three minutes and at boot, as this account under S4U at Limited, at priority 4, announces it, and replaces it on a second run' {
        $name = "${script:prefix}hub"
        $parameters = @{ TaskName = $name; ApiRoot = $script:world.Api; LogDirectory = $script:world.Logs; Poetry = $script:world.Poetry }
        Invoke-TestEntry $script:registerHub $parameters 6>$null
        # The announce ran for real: the stand-in poetry, from tools\fleet,
        # with the hub-announce lane's arguments (MCPs board task 2fecad69).
        Read-PoetryCall $script:world | Should -Be @((Get-TestTickArgument $script:world 'hub-announce'), "cwd=$($script:world.Fleet)")
        $said = @(Invoke-TestEntry $script:registerHub $parameters 6>&1 | ForEach-Object { "$_" })
        $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        $said | Should -Be @("Registered $name (every 3 minutes and at boot, $identity, S4U, Limited) and announced it.")
        @(Get-FleetScheduledTask -Prefix $name -Suffix '').Count | Should -Be 1
        # The definition as Task Scheduler stores it: Limited is the schema's
        # default run level and is stored as no RunLevel element at all,
        # where Highest would write HighestAvailable.
        $task = (Get-TaskDefinition $name).Task
        $task.Actions.Exec.Command | Should -BeExactly $script:world.Poetry
        $task.Actions.Exec.Arguments | Should -BeExactly (Get-TestTickArgument $script:world 'hub')
        $task.Actions.Exec.WorkingDirectory | Should -BeExactly $script:world.Fleet
        $task.Principals.Principal.LogonType | Should -BeExactly 'S4U'
        $task.Principals.Principal.PSObject.Properties['RunLevel'] | Should -BeNullOrEmpty
        $task.Settings.MultipleInstancesPolicy | Should -BeExactly 'IgnoreNew'
        $task.Settings.ExecutionTimeLimit | Should -BeExactly 'PT40M'
        $task.Settings.Priority | Should -BeExactly '4'
        @($task.Triggers.BootTrigger).Count | Should -Be 1
        $task.Triggers.TimeTrigger.Repetition.Interval | Should -BeExactly 'PT3M'
    }
    It 'refuses a hub registration whose announce fails' {
        $failing = Initialize-FleetTickWorld -ExitCode 7
        {
            Invoke-TestEntry $script:registerHub @{
                TaskName = "${script:prefix}hub"; ApiRoot = $failing.Api; LogDirectory = $failing.Logs; Poetry = $failing.Poetry
            } 6>$null
        } | Should -Throw "FLEET_HUB_ANNOUNCE_FAILED: the hub-announce tick exited 7; see fleet-agent-*.log under $($failing.Logs)"
    }
    It 'unregisters the hub task, and says so when there is none' {
        $name = "${script:prefix}hub"
        Invoke-TestEntry $script:registerHub @{ TaskName = $name; ApiRoot = $script:world.Api; Poetry = $script:world.Poetry } 6>$null
        @(Invoke-TestEntry $script:unregisterHub @{ TaskName = $name } 6>&1 | ForEach-Object { "$_" }) | Should -Be @("Unregistered $name.")
        @(Invoke-TestEntry $script:unregisterHub @{ TaskName = $name } 6>&1 | ForEach-Object { "$_" }) | Should -Be @("$name is not registered; nothing to remove.")
    }
    It 'registers and announces every enabled node, replaces an enabled node''s task, and retires a node no longer enabled' {
        $workspace = Join-Path $TestDrive ('fleet-' + [guid]::NewGuid().ToString('N') + '.json')
        [System.IO.File]::WriteAllText($workspace, '{"nodes": {"alpha": {"enabled": true}, "beta": {"enabled": false}, "gamma": {}}}')
        foreach ($alias in 'delta', 'alpha') {
            [void](Register-FleetTick -TaskName "${script:prefix}$alias-3min" -Poetry $script:world.Poetry -FleetRoot $script:world.Fleet `
                    -Arguments 'stale' -Description 'older')
        }
        $said = @(Invoke-TestEntry $script:registerNodes @{
                Workspace = $workspace; TaskPrefix = $script:prefix; ApiRoot = $script:world.Api
                LogDirectory = $script:world.Logs; Poetry = $script:world.Poetry
            } 6>&1 | ForEach-Object { "$_" })
        $said[0] | Should -BeExactly "Unregistered ${script:prefix}delta-3min: fleet.json asks for no such runner."
        $said[1] | Should -BeLike "Registered ${script:prefix}alpha-3min (every 3 minutes and at boot, *, S4U, Limited) and announced it."
        @(Get-FleetScheduledTask -Prefix $script:prefix -Suffix '-3min').TaskName | Should -Be @("${script:prefix}alpha-3min")
        $task = (Get-TaskDefinition "${script:prefix}alpha-3min").Task
        $task.Actions.Exec.Arguments | Should -BeExactly (Get-TestTickArgument $script:world 'node' 'alpha')
        $task.Settings.Priority | Should -BeExactly '4'
        # The announce ran for real: the stand-in poetry, from tools\fleet,
        # with the announce lane's arguments, for alpha alone.
        Read-PoetryCall $script:world | Should -Be @((Get-TestTickArgument $script:world 'announce' 'alpha'), "cwd=$($script:world.Fleet)")
    }
    # MCPs board task a98d7083: a node declaring elevated has a second runner,
    # and its task name reads as an alias no node carries, so the cleanup
    # compares whole task names; a second run must keep it, and a node that
    # stops declaring elevated loses it.
    It 'registers a second, elevated runner for a node declaring elevated, keeps it on a second run, and retires it once undeclared' {
        $workspace = Join-Path $TestDrive ('fleet-elevated-' + [guid]::NewGuid().ToString('N') + '.json')
        [System.IO.File]::WriteAllText($workspace, '{"nodes": {"alpha": {"enabled": true, "elevated": true}, "beta": {"enabled": false, "elevated": true}}}')
        $parameters = @{
            Workspace = $workspace; TaskPrefix = $script:prefix; ApiRoot = $script:world.Api
            LogDirectory = $script:world.Logs; Poetry = $script:world.Poetry
        }
        Invoke-TestEntry $script:registerNodes $parameters 6>$null
        $said = @(Invoke-TestEntry $script:registerNodes $parameters 6>&1 | ForEach-Object { "$_" })
        $said.Count | Should -Be 2
        $said[1] | Should -BeLike "Registered ${script:prefix}alpha-elevated-3min (every 3 minutes and at boot, *, S4U, Limited) and announced it."
        @(Get-FleetScheduledTask -Prefix $script:prefix -Suffix '-3min').TaskName | Sort-Object |
            Should -Be @("${script:prefix}alpha-3min", "${script:prefix}alpha-elevated-3min")
        (Get-TaskDefinition "${script:prefix}alpha-elevated-3min").Task.Actions.Exec.Arguments |
            Should -BeExactly (Get-TestTickArgument $script:world 'elevated' 'alpha')
        # Each run announces both runners from tools\fleet, the ordinary one
        # first.
        $cwd = "cwd=$($script:world.Fleet)"
        $announces = @((Get-TestTickArgument $script:world 'announce' 'alpha'), $cwd, (Get-TestTickArgument $script:world 'elevated-announce' 'alpha'), $cwd)
        Read-PoetryCall $script:world | Should -Be @($announces + $announces)
        [System.IO.File]::WriteAllText($workspace, '{"nodes": {"alpha": {"enabled": true, "elevated": false}}}')
        $retired = @(Invoke-TestEntry $script:registerNodes $parameters 6>&1 | ForEach-Object { "$_" })
        $retired[0] | Should -BeExactly "Unregistered ${script:prefix}alpha-elevated-3min: fleet.json asks for no such runner."
        @(Get-FleetScheduledTask -Prefix $script:prefix -Suffix '-3min').TaskName | Should -Be @("${script:prefix}alpha-3min")
    }
    It 'refuses a node whose announce fails, by its runner''s name' {
        $failing = Initialize-FleetTickWorld -ExitCode 7
        $workspace = Join-Path $TestDrive ('fleet-' + [guid]::NewGuid().ToString('N') + '.json')
        [System.IO.File]::WriteAllText($workspace, '{"nodes": {"alpha": {"enabled": true}}}')
        {
            Invoke-TestEntry $script:registerNodes @{
                Workspace = $workspace; TaskPrefix = $script:prefix; ApiRoot = $failing.Api; LogDirectory = $failing.Logs; Poetry = $failing.Poetry
            } 6>$null
        } | Should -Throw "FLEET_NODE_ANNOUNCE_FAILED: the announce tick for alpha exited 7; see fleet-node-alpha-*.log under $($failing.Logs)"
    }
    It 'refuses a registry with <Case>' -ForEach @(
        @{ Case = 'no nodes object'; Json = '{"hosts": {}}'; Message = 'FLEET_NODES_MISSING: *' },
        @{ Case = 'no enabled node'; Json = '{"nodes": {"alpha": {"enabled": false}}}'; Message = 'FLEET_NODES_NONE_ENABLED: *' }
    ) {
        $workspace = Join-Path $TestDrive ('fleet-' + [guid]::NewGuid().ToString('N') + '.json')
        [System.IO.File]::WriteAllText($workspace, $Json)
        { Invoke-TestEntry $script:registerNodes @{ Workspace = $workspace; TaskPrefix = $script:prefix; Poetry = $script:world.Poetry } 6>$null } |
            Should -Throw $Message
    }
    It 'registers the hub tick from this checkout into the default log directory when neither is named' {
        $record = Join-Path $TestDrive ('register-' + [guid]::NewGuid().ToString('N') + '.txt')
        $register = {
            param([string]$TaskName, [string]$Poetry, [string]$FleetRoot, [string]$Arguments, [string]$Description)
            [System.IO.File]::AppendAllText($record, "$TaskName|$Poetry|$FleetRoot|$Arguments|$Description`r`n")
            'recorded'
        }.GetNewClosure()
        $said = @(Invoke-TestEntry $script:registerHub @{ TaskName = "${script:prefix}hub"; Poetry = $script:world.Poetry; Register = $register } 6>&1 |
            ForEach-Object { "$_" })
        $said | Should -Be @("Registered ${script:prefix}hub (every 3 minutes and at boot, recorded, S4U, Limited) and announced it.")
        $arguments = "run -- python -m fleet.cli.tick --api-root `"$($script:apiRoot)`" --log-directory `"$env:LOCALAPPDATA\Temp\claude`" --lane hub"
        [System.IO.File]::ReadAllLines($record) | Should -Be @("${script:prefix}hub|$($script:world.Poetry)|$($script:apiRoot)\tools\fleet|$arguments|" +
            'One fleet-agent tick: drain the dispatch queue (API tools/fleet). See register-agent-schedule.ps1.')
    }
    It 'registers and announces each runner fleet.json asks for from this checkout, when neither is named' {
        $record = Join-Path $TestDrive ('register-' + [guid]::NewGuid().ToString('N') + '.txt')
        $register = {
            param([string]$TaskName, [string]$Poetry, [string]$FleetRoot, [string]$Arguments, [string]$Description)
            [System.IO.File]::AppendAllText($record, "$TaskName|$Poetry|$FleetRoot|$Arguments|$Description`r`n")
            'recorded'
        }.GetNewClosure()
        $document = [System.IO.File]::ReadAllText((Join-Path $script:scriptsRoot '..\fleet.json')) | ConvertFrom-Json
        $enabled = @($document.nodes.PSObject.Properties | Where-Object { $_.Value.PSObject.Properties['enabled'] -and $_.Value.enabled -eq $true })
        $enabled.Count | Should -BeGreaterThan 0
        $fleet = "$($script:apiRoot)\tools\fleet"
        $tick = "run -- python -m fleet.cli.tick --api-root `"$($script:apiRoot)`" --log-directory `"$($script:world.Logs)`""
        $expected = [System.Collections.Generic.List[string]]::new()
        $expectedCalls = [System.Collections.Generic.List[string]]::new()
        foreach ($node in $enabled) {
            $alias = $node.Name
            $expected.Add("$($script:prefix)$alias-3min|$($script:world.Poetry)|$fleet|$tick --lane node --node $alias|One fleet-node-agent tick for " +
                "${alias}: claim the node lane's jobs $alias carries the tags for (API tools/fleet). See register-node-agents.ps1.")
            $expectedCalls.Add("$tick --lane announce --node $alias")
            $expectedCalls.Add("cwd=$fleet")
            $elevated = $node.Value.PSObject.Properties['elevated']
            if ($null -ne $elevated -and $elevated.Value -eq $true) {
                $expected.Add("$($script:prefix)$alias-elevated-3min|$($script:world.Poetry)|$fleet|$tick --lane elevated --node $alias|" +
                    "One fleet-node-agent tick for $alias-elevated: claim only the jobs requiring the elevated tag, and launch them at RunLevel " +
                    'Highest (API tools/fleet). See register-node-agents.ps1.')
                $expectedCalls.Add("$tick --lane elevated-announce --node $alias")
                $expectedCalls.Add("cwd=$fleet")
            }
        }
        # serendipity declares elevated (MCPs board task a98d7083), so the
        # real registry yields one runner more than it has enabled nodes.
        $expected.Count | Should -BeGreaterThan $enabled.Count
        Invoke-TestEntry $script:registerNodes @{ TaskPrefix = $script:prefix; LogDirectory = $script:world.Logs; Poetry = $script:world.Poetry; Register = $register } 6>$null
        [string[]][System.IO.File]::ReadAllLines($record) | Should -Be @($expected)
        Read-PoetryCall $script:world | Should -Be @($expectedCalls)
    }
    It 'unregisters every node task and registers none under -UnregisterAll' {
        foreach ($alias in 'alpha', 'beta') {
            [void](Register-FleetTick -TaskName "$($script:prefix)$alias-3min" -Poetry $script:world.Poetry -FleetRoot $script:world.Fleet `
                    -Arguments (Get-TestTickArgument $script:world 'node' $alias) -Description 'test')
        }
        $said = @(Invoke-TestEntry $script:registerNodes @{ UnregisterAll = $true; TaskPrefix = $script:prefix } 6>&1 | ForEach-Object { "$_" })
        $said | Should -Be @("Unregistered $($script:prefix)alpha-3min.", "Unregistered $($script:prefix)beta-3min.")
        @(Get-FleetScheduledTask -Prefix $script:prefix -Suffix '').Count | Should -Be 0
    }
}
