Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# clients/RustedWarfareBot/scripts/make: the seven recipe bodies over RwMake.ps1
# (MCPs board task d69786fa). Every tool is a .cmd stand-in in a scratch root:
# javac and jar record their arguments, the game's java listens on the
# agent's channel port the way the agent does, the planner records its
# arguments, docker answers the password read, and python for the door and
# the workers are long-lived stand-ins. Ports come free from the operating
# system and worker names carry a prefix unique to this run, so no case sees
# or touches the real game, database, door or workers; every process a case
# spawns is ended by its pid.

BeforeAll {
    $makeRoot = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'scripts\make'
    . (Join-Path $makeRoot 'RwMake.ps1')
    $script:entry = @{}
    foreach ($name in 'agent', 'selftest', 'probe', 'host', 'watch', 'door', 'fleet-up') {
        $script:entry[$name] = Join-Path $makeRoot "$name.ps1"
    }
    $script:taskkill = Join-Path $env:SystemRoot 'System32\taskkill.exe'

    function Invoke-TestEntry {
        param([string]$Path, [hashtable]$Parameters)
        $global:LASTEXITCODE = 0
        & $Path @Parameters
        if ($LASTEXITCODE -ne 0) {
            throw "TEST_ENTRY_EXITED: $Path exited $LASTEXITCODE"
        }
    }

    function Get-FreePort {
        $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
        $probe.Start()
        $port = ([System.Net.IPEndPoint]$probe.LocalEndpoint).Port
        $probe.Stop()
        return $port
    }

    function Write-StandIn {
        param([string]$Path, [string[]]$Lines)
        [System.IO.File]::WriteAllText($Path, ('@echo off' + "`r`n" + ($Lines -join "`r`n") + "`r`n"), [System.Text.Encoding]::ASCII)
        return $Path
    }

    function Write-Listener {
        # A PowerShell script that holds a port until it is ended.
        param([string]$Path, [int]$Port)
        [System.IO.File]::WriteAllLines($Path, [string[]]@(
                "`$socket = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)",
                '$socket.Start()',
                'Start-Sleep -Seconds 120'))
        return $Path
    }

    function Invoke-TestListenerEnd {
        param([int]$Port)
        foreach ($owner in @(Get-RwListener $Port)) {
            $said = & $script:taskkill /F /T /PID $owner
            $LASTEXITCODE | Should -BeIn @(0, 128) -Because "$said"
        }
    }

    function Initialize-RwWorld {
        param([int]$JavacExit = 0, [int]$JarExit = 0, [int]$JavaExit = 0, [int]$PlannerExit = 0, [bool]$GameListens = $true)
        $root = Join-Path $TestDrive ('rw-' + [guid]::NewGuid().ToString('N'))
        foreach ($dir in 'agent\src\rwbot\agent', 'agent\build', 'game') {
            [void][System.IO.Directory]::CreateDirectory((Join-Path $root $dir))
        }
        [System.IO.File]::WriteAllText((Join-Path $root 'agent\src\rwbot\agent\Agent.java'), 'class Agent {}')
        [System.IO.File]::WriteAllText((Join-Path $root 'agent\manifest.mf'), 'Premain-Class: rwbot.agent.Agent')
        $calls = Join-Path $root 'calls.log'
        $port = Get-FreePort
        $listener = Write-Listener (Join-Path $root 'listen.ps1') $port
        $gameBody = @("echo java %*>>`"$calls`"", 'echo self-check passed')
        if ($GameListens) {
            $gameBody += "`"$PSHOME\powershell.exe`" -NoProfile -ExecutionPolicy Bypass -File `"$listener`""
        }
        $gameBody += "exit /b $JavaExit"
        return [pscustomobject]@{
            Root = $root; Calls = $calls; Port = $port
            Javac = Write-StandIn (Join-Path $root 'javac.cmd') @("echo javac %*>>`"$calls`"", "exit /b $JavacExit")
            # jar cfm <jar> ...: the stand-in writes the jar it was asked for.
            Jar = Write-StandIn (Join-Path $root 'jar.cmd') @("echo jar %*>>`"$calls`"", 'echo jar>"%2"', "exit /b $JarExit")
            Java = Write-StandIn (Join-Path $root 'game\java.cmd') $gameBody
            Poetry = Write-StandIn (Join-Path $root 'poetry.cmd') @("echo poetry %*>>`"$calls`"", "exit /b $PlannerExit")
        }
    }

    function Read-RwCall {
        param($World, [string]$Tool)
        if (-not [System.IO.File]::Exists($World.Calls)) { return [string[]]@() }
        return [string[]]@([System.IO.File]::ReadAllLines($World.Calls) | Where-Object { $_.StartsWith("$Tool ") } | ForEach-Object { $_.TrimEnd() })
    }
}

Describe 'Building the agent' {
    It 'compiles at the release given, packages under a temporary name and moves it into place' {
        $world = Initialize-RwWorld
        Invoke-TestEntry $script:entry['agent'] @{ Javac = $world.Javac; Jar = $world.Jar; AgentJar = 'agent\build\rw-agent.jar'; Release = '8'; Root = $world.Root }
        @(Read-RwCall $world 'javac')[0] | Should -Match '^javac --release 8 -Xlint:all -Werror -d \S+\\agent\\build\\classes-[0-9a-f]{8} \S+\\Agent\.java$'
        @(Read-RwCall $world 'jar')[0] | Should -Match '^jar cfm \S+\\agent\\build\\rw-agent\.jar\.[0-9a-f]{8}\.new \S+\\agent\\manifest\.mf -C \S+ \.$'
        [System.IO.File]::ReadAllText((Join-Path $world.Root 'agent\build\rw-agent.jar')).Trim() | Should -BeExactly 'jar'
        @(Get-ChildItem -LiteralPath (Join-Path $world.Root 'agent\build') -Name) | Should -Be @('rw-agent.jar')
    }
    It 'refuses a jar a JVM holds open, and leaves no scratch behind' {
        $world = Initialize-RwWorld
        $target = Join-Path $world.Root 'agent\build\rw-agent.jar'
        [System.IO.File]::WriteAllText($target, 'old')
        $held = [System.IO.File]::Open($target, 'Open', 'Read', 'None')
        try {
            { Invoke-TestEntry $script:entry['agent'] @{ Javac = $world.Javac; Jar = $world.Jar; AgentJar = 'agent\build\rw-agent.jar'; Release = '8'; Root = $world.Root } } |
                Should -Throw 'RW_AGENT_JAR_LOCKED: cannot replace agent\build\rw-agent.jar: a JVM has it attached*'
        } finally { $held.Dispose() }
        @(Get-ChildItem -LiteralPath (Join-Path $world.Root 'agent\build') -Name) | Should -Be @('rw-agent.jar')
    }
    It 'refuses a failed <Tool> by name' -ForEach @(
        @{ Tool = 'javac'; Code = 'RW_JAVAC_FAILED' },
        @{ Tool = 'jar'; Code = 'RW_JAR_FAILED' }
    ) {
        $world = if ($Tool -eq 'javac') { Initialize-RwWorld -JavacExit 2 } else { Initialize-RwWorld -JarExit 2 }
        { Invoke-TestEntry $script:entry['agent'] @{ Javac = $world.Javac; Jar = $world.Jar; AgentJar = 'agent\build\rw-agent.jar'; Release = '8'; Root = $world.Root } } |
            Should -Throw "${Code}: *exited 2*"
    }
    It 'runs the self-checks against the game jar and prints them' {
        $world = Initialize-RwWorld -GameListens $false
        $said = @(Invoke-TestEntry $script:entry['selftest'] @{ Javac = $world.Javac; Java = $world.Java; GameDir = 'game'; Root = $world.Root } 6>&1 | ForEach-Object { "$_" })
        $said | Should -Be @('self-check passed')
        @(Read-RwCall $world 'java')[0] | Should -Match '^java -cp \S+\\agent\\build\\verify-[0-9a-f]{8};game/game-lib\.jar;game/libs/\* rwbot\.agent\.SelfTest game/game-lib\.jar$'
        @(Read-RwCall $world 'javac')[0] | Should -Match '^javac --release 11 '
    }
    It 'refuses failed self-checks' {
        $world = Initialize-RwWorld -GameListens $false -JavaExit 1
        { Invoke-TestEntry $script:entry['selftest'] @{ Javac = $world.Javac; Java = $world.Java; GameDir = 'game'; Root = $world.Root } } |
            Should -Throw 'RW_SELFTEST_FAILED: *exited 1*'
    }
}

Describe 'The probe launcher' {
    It 'runs the headless game in its directory, placing {OUT} at the absolute output path' {
        $world = Initialize-RwWorld -GameListens $false
        Invoke-TestEntry $script:entry['probe'] @{ GameDir = 'game'; AgentJar = 'agent\build\rw-agent.jar'; AgentArgs = 'stateOutPath={OUT};exitAfterDiscovery=true'
            Out = 'runs\state.json'; Log = 'runs\probe.log'; Root = $world.Root; Java = 'java.cmd' }
        @(Read-RwCall $world 'java')[0] | Should -BeExactly ("java -Xmx1000M -Djava.library.path=. " +
            "-javaagent:$($world.Root)\agent\build\rw-agent.jar=stateOutPath=$($world.Root)\runs\state.json;exitAfterDiscovery=true " +
            "-cp game-lib.jar;libs/* com.corrodinggames.rts.java.Main -nodisplay -nosound -sandbox -width 800 -height 600 -log $($world.Root)\runs\probe.log")
    }
    It 'attaches the agent with no arguments when none are given, and refuses a failed game' {
        $world = Initialize-RwWorld -GameListens $false -JavaExit 3
        { Invoke-TestEntry $script:entry['probe'] @{ GameDir = 'game'; AgentJar = 'a.jar'; Log = 'p.log'; Root = $world.Root; Java = 'java.cmd' } } |
            Should -Throw 'RW_PROBE_FAILED: *exited 3*'
        @(Read-RwCall $world 'java')[0] | Should -BeLike "* -javaagent:$($world.Root)\a.jar -cp *"
    }
}

Describe 'Hosted and watched matches' {
    It '<Launcher> builds a scratch agent, starts the game, plays once the channel opens, and cleans up' -ForEach @(
        @{ Launcher = 'host'; Extra = @{ HostMap = 'maps/duel'; LobbyTimeoutSeconds = 60 }; Screen = '-nodisplay -nosound -width 800 -height 600'; Agent = 'hostMap=maps/duel' },
        @{ Launcher = 'watch'; Extra = @{ Map = 'maps/duel'; Opponents = 2; Difficulty = 4; ChannelSeconds = 60 }; Screen = '-nosound -width 1280 -height 800'; Agent = 'matchMap=maps/duel;matchOpponents=2;matchDifficulty=4' }
    ) {
        $world = Initialize-RwWorld
        $parameters = @{ Port = $world.Port; GameDir = 'game'; PlayLog = 'runs\play.log'; Module = 'rw_bot.play'; Catalogue = 'cat.json'; TypeDump = 'types.json'
            PlayArgs = '--samples 3 -'; Javac = $world.Javac; Jar = $world.Jar; Root = $world.Root; Java = 'java.cmd'; Poetry = $world.Poetry } + $Extra
        try {
            $said = @(Invoke-TestEntry $script:entry[$Launcher] $parameters 6>&1 | ForEach-Object { "$_" })
        } finally { Invoke-TestListenerEnd $world.Port }
        $said.Count | Should -Be 1
        @(Read-RwCall $world 'java')[0] | Should -BeLike "*-javaagent:$($world.Root)\agent\build\rw-agent-$Launcher-*.jar=channelPort=$($world.Port);$Agent*$Screen -log $($world.Root)\runs\play.log"
        @(Read-RwCall $world 'poetry') | Should -Be @("poetry run python -m rw_bot.play $($world.Port) cat.json types.json --samples 3 -")
        @(Get-ChildItem -LiteralPath (Join-Path $world.Root 'agent\build') -Name) | Should -Be @()
    }
    It 'refuses a channel that never opens, and plays nothing' {
        $world = Initialize-RwWorld -GameListens $false
        { Invoke-TestEntry $script:entry['watch'] @{ Port = $world.Port; GameDir = 'game'; PlayLog = 'runs\play.log'; Map = 'm'; Module = 'rw_bot.play'
                Catalogue = 'c'; TypeDump = 't'; PlayArgs = '-'; Javac = $world.Javac; Jar = $world.Jar; Root = $world.Root; Java = 'java.cmd'; Poetry = $world.Poetry; ChannelSeconds = 1 } } |
            Should -Throw "RW_CHANNEL_NOT_OPEN: nothing listened on port $($world.Port) within 1s: the agent never opened its channel"
        @(Read-RwCall $world 'poetry').Count | Should -Be 0
    }
    It 'refuses a planner that fails, and still stops the game' {
        $world = Initialize-RwWorld -PlannerExit 4
        try {
            { Invoke-TestEntry $script:entry['host'] @{ Port = $world.Port; GameDir = 'game'; PlayLog = 'runs\play.log'; HostMap = 'm'; Module = 'rw_bot.play'
                    Catalogue = 'c'; TypeDump = 't'; PlayArgs = '-'; Javac = $world.Javac; Jar = $world.Jar; Root = $world.Root; Java = 'java.cmd'; Poetry = $world.Poetry; LobbyTimeoutSeconds = 60 } 6>$null } |
                Should -Throw 'RW_PLANNER_FAILED: the planner exited 4'
        } finally { Invoke-TestListenerEnd $world.Port }
    }
}

Describe 'The match service launchers' {
    BeforeAll {
        function Initialize-RwServiceWorld {
            param([string[]]$DockerLines = @('echo s3cret'))
            $root = Join-Path $TestDrive ('svc-' + [guid]::NewGuid().ToString('N'))
            [void][System.IO.Directory]::CreateDirectory($root)
            $port = Get-FreePort
            $listener = Write-Listener (Join-Path $root 'listen.ps1') $port
            return [pscustomobject]@{
                Root = $root; Port = $port; Prefix = 'rwtest' + [guid]::NewGuid().ToString('N').Substring(0, 8) + '-'
                Docker = Write-StandIn (Join-Path $root 'docker.cmd') $DockerLines
                Door = Write-StandIn (Join-Path $root 'door-python.cmd') @("`"$PSHOME\powershell.exe`" -NoProfile -ExecutionPolicy Bypass -File `"$listener`"")
                Quiet = Write-StandIn (Join-Path $root 'quiet-python.cmd') @('exit /b 0')
                Worker = Write-StandIn (Join-Path $root 'worker-python.cmd') @('ping -n 60 127.0.0.1 >nul')
            }
        }
        function Invoke-TestWorkerEnd {
            # Ends each cmd.exe whose command line carries this run's unique
            # worker prefix, by pid and with its tree. A launcher and the
            # worker it started both carry the prefix, so one listing can
            # name a process that ends before its turn: taskkill then exits
            # 128 when nothing of it is left, and 255 when the root had gone
            # but its tree was still ended ("SUCCESS" for each child, seen
            # 2026-09-27 in the harness). Either way the pid must be gone
            # afterwards, which is what is asserted.
            param([string]$Prefix)
            foreach ($process in @(Get-CimInstance -ClassName Win32_Process -Filter "Name='cmd.exe'" | Where-Object { $null -ne $_.CommandLine -and $_.CommandLine.Contains($Prefix) })) {
                $said = & $script:taskkill /F /T /PID $process.ProcessId
                $LASTEXITCODE | Should -BeIn @(0, 128, 255) -Because "$said"
                @(Get-CimInstance -ClassName Win32_Process -Filter "ProcessId=$($process.ProcessId)").Count | Should -Be 0 -Because "$said"
            }
        }
    }
    It 'reads the password from the container, starts the door detached and waits for it to listen' {
        $world = Initialize-RwServiceWorld
        try {
            $said = @(Invoke-TestEntry $script:entry['door'] @{ Port = $world.Port; Root = $world.Root; Docker = $world.Docker; Python = $world.Door } 6>&1 | ForEach-Object { "$_" })
            $said | Should -Be @("door up: pid $(@(Get-RwListener $world.Port) -join ', '), http://127.0.0.1:$($world.Port)/, log runs/door.log")
        } finally { Invoke-TestListenerEnd $world.Port }
    }
    It 'refuses a door already listening, before reading the password' {
        $world = Initialize-RwServiceWorld -DockerLines @('exit /b 9')
        $held = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $world.Port)
        $held.Start()
        try {
            { Invoke-TestEntry $script:entry['door'] @{ Port = $world.Port; Root = $world.Root; Docker = $world.Docker; Python = $world.Door } } |
                Should -Throw "RW_DOOR_ALREADY_UP: the door is already listening on $($world.Port) (pid $PID); stop it first"
        } finally { $held.Stop() }
    }
    It 'refuses a door that spawns but never listens' {
        $world = Initialize-RwServiceWorld
        { Invoke-TestEntry $script:entry['door'] @{ Port = $world.Port; Root = $world.Root; Docker = $world.Docker; Python = $world.Quiet; ReadySeconds = 2 } } |
            Should -Throw "RW_DOOR_NOT_LISTENING: nothing listened on port $($world.Port) within 2s: the door spawned as pid *"
    }
    It 'refuses a password docker cannot read or that is empty' -ForEach @(
        @{ Lines = @('exit /b 1'); Message = 'RW_DATABASE_PASSWORD_UNREADABLE: *exited 1*' },
        @{ Lines = @('exit /b 0'); Message = 'RW_DATABASE_PASSWORD_UNREADABLE: platform-postgres has an empty POSTGRES_PASSWORD' }
    ) {
        $world = Initialize-RwServiceWorld -DockerLines $Lines
        { Get-RwDatabaseDsn $world.Docker } | Should -Throw $Message
    }
    It 'names the DSN with the password it read' {
        $world = Initialize-RwServiceWorld
        Get-RwDatabaseDsn $world.Docker | Should -BeExactly 'host=127.0.0.1 port=55432 user=covenant password=s3cret dbname=covenant connect_timeout=10'
    }
    It 'refuses a spawn WMI declines' {
        $absent = Join-Path $TestDrive ('absent-' + [guid]::NewGuid().ToString('N') + '\none.exe')
        { Invoke-RwDetachedSpawn $absent 'RW_TEST_SPAWN' } | Should -Throw 'RW_TEST_SPAWN: WMI Create returned *'
    }
    It 'starts only the workers not already running, and reports every one running' {
        $world = Initialize-RwServiceWorld
        try {
            $first = @(Invoke-TestEntry $script:entry['fleet-up'] @{ Workers = 1; Root = $world.Root; Docker = $world.Docker; Python = $world.Worker
                    ProcessName = 'cmd.exe'; NamePrefix = $world.Prefix; SettleSeconds = 1 } 6>&1 | ForEach-Object { "$_" })
            $first | Should -Be @("fleet up: started 1, running now: $($world.Prefix)1")
            $second = @(Invoke-TestEntry $script:entry['fleet-up'] @{ Workers = 2; Root = $world.Root; Docker = $world.Docker; Python = $world.Worker
                    ProcessName = 'cmd.exe'; NamePrefix = $world.Prefix; SettleSeconds = 1 } 6>&1 | ForEach-Object { "$_" })
            $second | Should -Be @("$($world.Prefix)1 already running; leaving it alone", "fleet up: started 1, running now: $($world.Prefix)1, $($world.Prefix)2")
        } finally { Invoke-TestWorkerEnd $world.Prefix }
        Get-RwWorkerName 'cmd.exe' $world.Prefix | Should -Be @()
    }
    It 'names no process of the image whose command line is not a worker''s' {
        # A cmd.exe of this case's own, not left to chance. On the hub some
        # other cmd.exe is always running and took this arm by accident; on a
        # clean CI runner the only ones were the workers above, so API's
        # powershell job (run 36476858256) found the arm untaken.
        $prefix = 'rwtest' + [guid]::NewGuid().ToString('N').Substring(0, 8) + '-'
        $other = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\cmd.exe') -ArgumentList '/d', '/c', 'ping -n 60 127.0.0.1 >nul' -WindowStyle Hidden -PassThru
        try {
            @(Get-CimInstance -ClassName Win32_Process -Filter "ProcessId=$($other.Id)").Count | Should -Be 1
            Get-RwWorkerName 'cmd.exe' $prefix | Should -Be @()
        } finally {
            # The exit codes Invoke-TestWorkerEnd documents: taskkill /T ends
            # the tree and can report 255 for a root it reached last. The pid
            # being gone afterwards is what is asserted.
            $said = & $script:taskkill /F /T /PID $other.Id
            $LASTEXITCODE | Should -BeIn @(0, 128, 255) -Because "$said"
            @(Get-CimInstance -ClassName Win32_Process -Filter "ProcessId=$($other.Id)").Count | Should -Be 0 -Because "$said"
        }
    }
}

Describe 'Scratch cleanup' {
    It 'removes nothing under -WhatIf, and nothing that is not there' {
        $path = Join-Path $TestDrive 'scratch.jar'
        [System.IO.File]::WriteAllText($path, 'x')
        Remove-RwBuildPath $path -WhatIf
        [System.IO.File]::Exists($path) | Should -BeTrue
        Remove-RwBuildPath (Join-Path $TestDrive 'absent.jar')
    }
}
