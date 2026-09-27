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
}

Describe 'The schedules' {
    BeforeEach {
        $script:prefix = 'API-FleetTest-' + [guid]::NewGuid().ToString('N').Substring(0, 8) + '-'
        $script:standIn = Initialize-FleetStandInTick ('tick-' + [guid]::NewGuid().ToString('N').Substring(0, 8)) 0
    }
    AfterEach {
        foreach ($task in @(Get-FleetScheduledTask -Prefix $script:prefix -Suffix '')) {
            Unregister-ScheduledTask -TaskName $task.TaskName -TaskPath $task.TaskPath -Confirm:$false
        }
    }
    It 'registers a tick every three minutes and at boot, as this account under S4U at Limited, and replaces it on a second run' {
        $name = "${script:prefix}hub"
        Invoke-TestEntry $script:registerHub @{ TaskName = $name; Tick = $script:standIn.Path } 6>$null
        $said = @(Invoke-TestEntry $script:registerHub @{ TaskName = $name; Tick = $script:standIn.Path } 6>&1 | ForEach-Object { "$_" })
        $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        $said | Should -Be @("Registered $name (every 3 minutes and at boot, $identity, S4U, Limited).")
        $task = @(Get-FleetScheduledTask -Prefix $name -Suffix '')
        $task.Count | Should -Be 1
        $task[0].Actions[0].Execute | Should -BeExactly 'powershell.exe'
        $task[0].Actions[0].Arguments | Should -BeExactly "-NoProfile -ExecutionPolicy Bypass -File `"$($script:standIn.Path)`""
        $task[0].Principal.LogonType | Should -Be 'S4U'
        $task[0].Principal.RunLevel | Should -Be 'Limited'
        $task[0].Settings.MultipleInstances | Should -Be 'IgnoreNew'
        $task[0].Settings.ExecutionTimeLimit | Should -BeExactly 'PT40M'
        $task[0].Triggers.Count | Should -Be 2
        $task[0].Triggers[1].Repetition.Interval | Should -BeExactly 'PT3M'
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
        (Get-FleetScheduledTask -Prefix $script:prefix -Suffix '-3min')[0].Actions[0].Arguments | Should -BeLike '* -Node alpha'
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
    It 'unregisters every node task and registers none under -UnregisterAll' {
        foreach ($alias in 'alpha', 'beta') {
            [void](Register-FleetTick -TaskName "$($script:prefix)$alias-3min" -Tick $script:standIn.Path -TickArguments " -Node $alias" -Description 'test')
        }
        $said = @(Invoke-TestEntry $script:registerNodes @{ UnregisterAll = $true; TaskPrefix = $script:prefix } 6>&1 | ForEach-Object { "$_" })
        $said | Should -Be @("Unregistered $($script:prefix)alpha-3min.", "Unregistered $($script:prefix)beta-3min.")
        @(Get-FleetScheduledTask -Prefix $script:prefix -Suffix '').Count | Should -Be 0
    }
}
