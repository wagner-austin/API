Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# tools/fleet/scripts: the two tick entries over FleetTick.ps1 and the three
# schedule entries over FleetSchedule.ps1 (MCPs board task d69786fa).
#
# NOTHING HERE STARTS A REAL AGENT. poetry is a .cmd stand-in that records
# its arguments, working directory and inherited marker; every task this
# suite registers has a disposable name and runs a stand-in tick script, so
# a scheduler that starts it at once (StartWhenAvailable) starts nothing
# that touches the queue.

BeforeAll {
    $scriptsRoot = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'scripts'
    . (Join-Path $scriptsRoot 'FleetTick.ps1')
    . (Join-Path $scriptsRoot 'FleetSchedule.ps1')
    . (Join-Path $PSScriptRoot 'task-fixtures.ps1')
    $script:hubTick = Join-Path $scriptsRoot 'run-agent-tick.ps1'
    $script:nodeTick = Join-Path $scriptsRoot 'run-node-agent-tick.ps1'
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
        param([int]$ExitCode = 0)
        $root = Join-Path $TestDrive ('tick-' + [guid]::NewGuid().ToString('N'))
        $work = Join-Path $root 'work'
        [void][System.IO.Directory]::CreateDirectory($work)
        $mark = 'mark-' + [guid]::NewGuid().ToString('N')
        $environment = Join-Path $root 'env.ps1'
        [System.IO.File]::WriteAllText($environment, "`$env:FLEET_TICK_TEST_MARK = '$mark'`r`n")
        $calls = Join-Path $root 'poetry-args.txt'
        $poetry = Join-Path $root 'poetry.cmd'
        $body = "@echo off`r`necho %*>`"$calls`"`r`necho cwd=%CD%`r`necho mark=%FLEET_TICK_TEST_MARK%`r`necho to-stderr 1>&2`r`nexit /b $ExitCode`r`n"
        [System.IO.File]::WriteAllText($poetry, $body, [System.Text.Encoding]::ASCII)
        return [pscustomobject]@{
            Root = $root; Work = $work; Mark = $mark; Environment = $environment
            Calls = $calls; Poetry = $poetry; Logs = Join-Path $root 'logs'
        }
    }

    function Read-FleetTickLog {
        param([string]$Directory, [string]$Stem)
        $path = Join-Path $Directory "$Stem-$((Get-Date).ToString('yyyy-MM-dd')).log"
        return [string[]][System.IO.File]::ReadAllLines($path)
    }

    function Initialize-FleetStandInTick {
        param([string]$Name, [int]$ExitCode)
        $path = Join-Path $TestDrive "$Name.ps1"
        $record = Join-Path $TestDrive "$Name-announced.txt"
        $body = "param([string]`$Node, [switch]`$Announce)`r`n[System.IO.File]::AppendAllText('$record', `"`$Node `$Announce`r`n`")`r`nexit $ExitCode`r`n"
        [System.IO.File]::WriteAllText($path, $body)
        return [pscustomobject]@{ Path = $path; Record = $record }
    }
}

Describe 'One fleet tick' {
    It 'runs poetry with the credentials it dot-sourced, logs both streams and answers with poetry''s exit' {
        $world = Initialize-FleetTickWorld -ExitCode 3
        $code = Invoke-FleetTick -EnvironmentScript $world.Environment -Poetry $world.Poetry -WorkingDirectory $world.Work `
            -AgentArguments @('run', '--', 'x') -LogDirectory $world.Logs -LogStem 'fleet-test' -Header 'hub' -RetentionDays 14
        $code | Should -Be 3
        [System.IO.File]::ReadAllText($world.Calls).Trim() | Should -BeExactly 'run -- x'
        $log = Read-FleetTickLog $world.Logs 'fleet-test'
        $log[0] | Should -Match "^TICK START \S+ hub pid \d+ task-pid $PID$"
        $log[1..3] | Should -Be @("cwd=$($world.Work)", "mark=$($world.Mark)", 'to-stderr ')
        $log[4] | Should -Match '^TICK EXIT 3 '
        @(Get-ChildItem -LiteralPath $env:TEMP -Filter "fleet-test-tick-$PID.*").Count | Should -Be 0
    }
    It 'removes only this stem''s logs past the retention, and none under -WhatIf' {
        $directory = Join-Path $TestDrive ('retention-' + [guid]::NewGuid().ToString('N'))
        [void][System.IO.Directory]::CreateDirectory($directory)
        $now = Get-Date
        foreach ($name in 'fleet-a-old.log', 'fleet-a-new.log', 'fleet-b-old.log') {
            [System.IO.File]::WriteAllText((Join-Path $directory $name), 'x')
        }
        foreach ($name in 'fleet-a-old.log', 'fleet-b-old.log') {
            [System.IO.File]::SetLastWriteTime((Join-Path $directory $name), $now.AddDays(-20))
        }
        Remove-FleetTickLog -LogDirectory $directory -LogStem 'fleet-a' -RetentionDays 14 -Now $now -WhatIf | Should -Be 1
        [System.IO.File]::Exists((Join-Path $directory 'fleet-a-old.log')) | Should -BeTrue
        Remove-FleetTickLog -LogDirectory $directory -LogStem 'fleet-a' -RetentionDays 14 -Now $now | Should -Be 1
        @(Get-ChildItem -LiteralPath $directory -Name | Sort-Object) | Should -Be @('fleet-a-new.log', 'fleet-b-old.log')
    }
}

Describe 'The tick entries' {
    It 'runs the hub lane''s agent from the rolled commit with this checkout''s roots' {
        $world = Initialize-FleetTickWorld
        $global:LASTEXITCODE = 0
        & $script:hubTick -EnvironmentScript $world.Environment -Poetry $world.Poetry -LogDirectory $world.Logs
        $LASTEXITCODE | Should -Be 0
        $mcps = Join-Path (Split-Path -Parent $script:apiRoot) 'MCPs'
        [System.IO.File]::ReadAllText($world.Calls).Trim() | Should -BeExactly ("run -- python -m fleet.cli.rolled --repo-root $($script:apiRoot) " +
            "--agent fleet-agent -- --agent fleet-runner-austinpc --session a850f688-f98d-415c-a244-e993226ca2fc " +
            "--repo-root $($script:apiRoot) --mcps-root $mcps --registry $mcps\fleet-mcp\fleet-nodes.json")
        (Read-FleetTickLog $world.Logs 'fleet-agent')[1] | Should -BeExactly "cwd=$($script:apiRoot)\tools\fleet"
    }
    It 'exits with the agent''s own non-zero status' {
        $world = Initialize-FleetTickWorld -ExitCode 5
        $global:LASTEXITCODE = 0
        & $script:hubTick -EnvironmentScript $world.Environment -Poetry $world.Poetry -LogDirectory $world.Logs
        $LASTEXITCODE | Should -Be 5
    }
    It 'runs one node''s agent, announcing when asked, into that node''s log' {
        $world = Initialize-FleetTickWorld
        $global:LASTEXITCODE = 0
        & $script:nodeTick -Node 'alpha' -Announce -EnvironmentScript $world.Environment -Poetry $world.Poetry -LogDirectory $world.Logs
        $LASTEXITCODE | Should -Be 0
        [System.IO.File]::ReadAllText($world.Calls).Trim() | Should -BeExactly ("run -- python -m fleet.cli.rolled --repo-root $($script:apiRoot) " +
            '--agent fleet-node-agent -- --node alpha --announce')
        (Read-FleetTickLog $world.Logs 'fleet-node-alpha')[0] | Should -Match '^TICK START \S+ node alpha pid '
    }
    It 'claims without announcing by default' {
        $world = Initialize-FleetTickWorld
        $global:LASTEXITCODE = 0
        & $script:nodeTick -Node 'beta' -EnvironmentScript $world.Environment -Poetry $world.Poetry -LogDirectory $world.Logs
        $LASTEXITCODE | Should -Be 0
        [System.IO.File]::ReadAllText($world.Calls).Trim() | Should -BeLike '* -- --node beta'
    }
    It 'dot-sources runs/env.ps1 under the API root it is given, running there, in the <Entry> entry' -ForEach @(
        @{ Entry = 'hub'; Stem = 'fleet-agent'; Parameters = @{} },
        @{ Entry = 'node'; Stem = 'fleet-node-gamma'; Parameters = @{ Node = 'gamma' } }
    ) {
        $world = Initialize-FleetTickWorld
        $apiRoot = Join-Path $world.Root 'api'
        [void][System.IO.Directory]::CreateDirectory((Join-Path $apiRoot 'tools\hpc-wake\runs'))
        [void][System.IO.Directory]::CreateDirectory((Join-Path $apiRoot 'tools\fleet'))
        [System.IO.File]::Copy($world.Environment, (Join-Path $apiRoot 'tools\hpc-wake\runs\env.ps1'))
        $entry = @{ hub = $script:hubTick; node = $script:nodeTick }[$Entry]
        Invoke-TestEntry $entry ($Parameters + @{ ApiRoot = $apiRoot; Poetry = $world.Poetry; LogDirectory = $world.Logs })
        [System.IO.File]::ReadAllText($world.Calls).Trim() | Should -BeLike "run -- python -m fleet.cli.rolled --repo-root $apiRoot *"
        (Read-FleetTickLog $world.Logs $Stem)[1..2] | Should -Be @("cwd=$apiRoot\tools\fleet", "mark=$($world.Mark)")
    }
    # Task Scheduler runs these with -File, where Windows PowerShell 5.1
    # leaves $PSScriptRoot empty inside an advanced script's param default;
    # only a real -File child sees that, since & sets it (every tick from
    # 16:27Z on 2026-09-27 failed on it).
    It 'finds this checkout''s roots when powershell runs the <Entry> entry with -File' -ForEach @(
        @{ Entry = 'hub'; Arguments = @(); Expected = '--agent fleet-agent -- --agent fleet-runner-austinpc *' },
        @{ Entry = 'node'; Arguments = @('-Node', 'delta'); Expected = '--agent fleet-node-agent -- --node delta' }
    ) {
        $world = Initialize-FleetTickWorld
        $entry = @{ hub = $script:hubTick; node = $script:nodeTick }[$Entry]
        & "$PSHOME\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File $entry @Arguments `
            -EnvironmentScript $world.Environment -Poetry $world.Poetry -LogDirectory $world.Logs | Out-Null
        $LASTEXITCODE | Should -Be 0
        [System.IO.File]::ReadAllText($world.Calls).Trim() | Should -BeLike "run -- python -m fleet.cli.rolled --repo-root $($script:apiRoot) $Expected"
    }
}

Describe 'The schedules' {
    BeforeEach {
        $script:prefix = 'API-FleetTest-' + [guid]::NewGuid().ToString('N').Substring(0, 8) + '-'
        $script:standIn = Initialize-FleetStandInTick ('tick-' + [guid]::NewGuid().ToString('N').Substring(0, 8)) 0
    }
    AfterEach {
        foreach ($task in @(Get-FleetScheduledTask -Prefix $script:prefix -Suffix '')) {
            [void](Unregister-FleetTick $task.TaskName)
        }
    }
    It 'registers a tick every three minutes and at boot, as this account under S4U at Limited, and replaces it on a second run' {
        $name = "${script:prefix}hub"
        Invoke-TestEntry $script:registerHub @{ TaskName = $name; Tick = $script:standIn.Path } 6>$null
        $said = @(Invoke-TestEntry $script:registerHub @{ TaskName = $name; Tick = $script:standIn.Path } 6>&1 | ForEach-Object { "$_" })
        $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        $said | Should -Be @("Registered $name (every 3 minutes and at boot, $identity, S4U, Limited).")
        @(Get-FleetScheduledTask -Prefix $name -Suffix '').Count | Should -Be 1
        # The definition as Task Scheduler stores it: Limited is the schema's
        # default run level and is stored as no RunLevel element at all,
        # where Highest would write HighestAvailable.
        $task = (Get-TaskDefinition $name).Task
        $task.Actions.Exec.Command | Should -BeExactly 'powershell.exe'
        $task.Actions.Exec.Arguments | Should -BeExactly "-NoProfile -ExecutionPolicy Bypass -File `"$($script:standIn.Path)`""
        $task.Principals.Principal.LogonType | Should -BeExactly 'S4U'
        $task.Principals.Principal.PSObject.Properties['RunLevel'] | Should -BeNullOrEmpty
        $task.Settings.MultipleInstancesPolicy | Should -BeExactly 'IgnoreNew'
        $task.Settings.ExecutionTimeLimit | Should -BeExactly 'PT40M'
        @($task.Triggers.BootTrigger).Count | Should -Be 1
        $task.Triggers.TimeTrigger.Repetition.Interval | Should -BeExactly 'PT3M'
    }
    It 'unregisters the hub task, and says so when there is none' {
        $name = "${script:prefix}hub"
        Invoke-TestEntry $script:registerHub @{ TaskName = $name; Tick = $script:standIn.Path } 6>$null
        @(Invoke-TestEntry $script:unregisterHub @{ TaskName = $name } 6>&1 | ForEach-Object { "$_" }) | Should -Be @("Unregistered $name.")
        @(Invoke-TestEntry $script:unregisterHub @{ TaskName = $name } 6>&1 | ForEach-Object { "$_" }) | Should -Be @("$name is not registered; nothing to remove.")
    }
    It 'registers and announces every enabled node, replaces an enabled node''s task, and retires a node no longer enabled' {
        $workspace = Join-Path $TestDrive 'fleet-nodes.json'
        [System.IO.File]::WriteAllText($workspace, '{"nodes": {"alpha": {"enabled": true}, "beta": {"enabled": false}, "gamma": {}}}')
        [void](Register-FleetTick -TaskName "${script:prefix}delta-3min" -Tick $script:standIn.Path -TickArguments ' -Node delta' -Description 'stale')
        [void](Register-FleetTick -TaskName "${script:prefix}alpha-3min" -Tick $script:standIn.Path -TickArguments ' -Node old' -Description 'older')
        $said = @(Invoke-TestEntry $script:registerNodes @{ Workspace = $workspace; TaskPrefix = $script:prefix; Tick = $script:standIn.Path } 6>&1 |
            ForEach-Object { "$_" })
        $said[0] | Should -BeExactly "Unregistered ${script:prefix}delta-3min: delta is not an enabled node."
        $said[1] | Should -BeLike "Registered ${script:prefix}alpha-3min (every 3 minutes and at boot, *, S4U, Limited) and announced it."
        @(Get-FleetScheduledTask -Prefix $script:prefix -Suffix '-3min').TaskName | Should -Be @("${script:prefix}alpha-3min")
        (Get-TaskDefinition "${script:prefix}alpha-3min").Task.Actions.Exec.Arguments | Should -BeLike '* -Node alpha'
        [System.IO.File]::ReadAllText($script:standIn.Record).Trim() | Should -BeExactly 'alpha True'
    }
    It 'refuses a node whose announce fails, by name' {
        $workspace = Join-Path $TestDrive 'fleet-one.json'
        [System.IO.File]::WriteAllText($workspace, '{"nodes": {"alpha": {"enabled": true}}}')
        $failing = Initialize-FleetStandInTick ('fail-' + [guid]::NewGuid().ToString('N').Substring(0, 8)) 7
        { Invoke-TestEntry $script:registerNodes @{ Workspace = $workspace; TaskPrefix = $script:prefix; Tick = $failing.Path } 6>$null } |
            Should -Throw 'FLEET_NODE_ANNOUNCE_FAILED: the announce tick for alpha exited 7*'
    }
    It 'refuses a registry with <Case>' -ForEach @(
        @{ Case = 'no nodes object'; Json = '{"hosts": {}}'; Message = 'FLEET_NODES_MISSING: *' },
        @{ Case = 'no enabled node'; Json = '{"nodes": {"alpha": {"enabled": false}}}'; Message = 'FLEET_NODES_NONE_ENABLED: *' }
    ) {
        $workspace = Join-Path $TestDrive ('fleet-' + [guid]::NewGuid().ToString('N') + '.json')
        [System.IO.File]::WriteAllText($workspace, $Json)
        { Invoke-TestEntry $script:registerNodes @{ Workspace = $workspace; TaskPrefix = $script:prefix; Tick = $script:standIn.Path } 6>$null } |
            Should -Throw $Message
    }
    It 'registers the hub tick beside the entry when none is named' {
        $record = Join-Path $TestDrive ('register-' + [guid]::NewGuid().ToString('N') + '.txt')
        $register = {
            param([string]$TaskName, [string]$Tick, [string]$TickArguments, [string]$Description)
            [System.IO.File]::AppendAllText($record, "$TaskName|$Tick|$TickArguments|$Description`r`n")
            'recorded'
        }.GetNewClosure()
        $said = @(Invoke-TestEntry $script:registerHub @{ TaskName = "${script:prefix}hub"; Register = $register } 6>&1 | ForEach-Object { "$_" })
        $said | Should -Be @("Registered ${script:prefix}hub (every 3 minutes and at boot, recorded, S4U, Limited).")
        [System.IO.File]::ReadAllLines($record) | Should -Be @("${script:prefix}hub|$(Split-Path -Parent $script:hubTick)\run-agent-tick.ps1||" +
            'One fleet-agent tick: drain the dispatch queue (API tools/fleet). See register-agent-schedule.ps1.')
    }
    It 'registers and announces each node fleet.json enables with the tick beside the entry, when neither is named' {
        $record = Join-Path $TestDrive ('register-' + [guid]::NewGuid().ToString('N') + '.txt')
        $register = {
            param([string]$TaskName, [string]$Tick, [string]$TickArguments, [string]$Description)
            [System.IO.File]::AppendAllText($record, "$TaskName|$Tick|$TickArguments|$Description`r`n")
            'recorded'
        }.GetNewClosure()
        $announced = Join-Path $TestDrive ('announced-' + [guid]::NewGuid().ToString('N') + '.txt')
        $powerShell = Join-Path $TestDrive ('powershell-' + [guid]::NewGuid().ToString('N') + '.cmd')
        [System.IO.File]::WriteAllText($powerShell, "@echo off`r`necho %*>>`"$announced`"`r`nexit /b 0`r`n", [System.Text.Encoding]::ASCII)
        $scripts = Split-Path -Parent $script:nodeTick
        $document = [System.IO.File]::ReadAllText((Join-Path $scripts '..\fleet.json')) | ConvertFrom-Json
        $enabled = @($document.nodes.PSObject.Properties | Where-Object { $_.Value.PSObject.Properties['enabled'] -and $_.Value.enabled -eq $true } |
            ForEach-Object { $_.Name })
        $enabled.Count | Should -BeGreaterThan 0
        Invoke-TestEntry $script:registerNodes @{ TaskPrefix = $script:prefix; PowerShell = $powerShell; Register = $register } 6>$null
        $registered = [string[]][System.IO.File]::ReadAllLines($record)
        $registered.Count | Should -Be $enabled.Count
        foreach ($index in 0..($enabled.Count - 1)) {
            $alias = $enabled[$index]
            $registered[$index] | Should -BeExactly ("$($script:prefix)$alias-3min|$scripts\run-node-agent-tick.ps1| -Node $alias|One fleet-node-agent tick for " +
                "${alias}: claim the node lane's jobs $alias carries the tags for (API tools/fleet). See register-node-agents.ps1.")
        }
        [System.IO.File]::ReadAllLines($announced) | Should -Be @($enabled | ForEach-Object {
                "-NoProfile -ExecutionPolicy Bypass -File $scripts\run-node-agent-tick.ps1 -Node $_ -Announce" })
    }
    It 'unregisters every node task and registers none under -UnregisterAll' {
        foreach ($alias in 'alpha', 'beta') {
            [void](Register-FleetTick -TaskName "$($script:prefix)$alias-3min" -Tick $script:standIn.Path -TickArguments " -Node $alias" -Description 'test')
        }
        $said = @(Invoke-TestEntry $script:registerNodes @{ UnregisterAll = $true; TaskPrefix = $script:prefix } 6>&1 | ForEach-Object { "$_" })
        $said | Should -Be @("Unregistered $($script:prefix)alpha-3min.", "Unregistered $($script:prefix)beta-3min.")
        @(Get-FleetScheduledTask -Prefix $script:prefix -Suffix '').Count | Should -Be 0
    }
}
