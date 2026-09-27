Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# services/grandma-api/scripts: start.ps1 and stop.ps1 over GrandmaService.ps1
# (MCPs board task d69786fa). poetry is a stand-in that really listens on the
# port it was told to bind, npm a stand-in that builds or fails, and every
# port is one the operating system handed out free, so the real API and web
# server are never started, and stop only ever ends processes this suite
# started. A port held by this test process itself is never given to stop.

BeforeAll {
    $scriptsRoot = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'scripts'
    . (Join-Path $scriptsRoot 'GrandmaService.ps1')
    $script:startEntry = Join-Path $scriptsRoot 'start.ps1'
    $script:stopEntry = Join-Path $scriptsRoot 'stop.ps1'
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

    function Initialize-GrandmaWorld {
        param([bool]$Listens = $true, [int]$BuildExit = 0)
        $root = Join-Path $TestDrive ('grandma-' + [guid]::NewGuid().ToString('N'))
        [void][System.IO.Directory]::CreateDirectory((Join-Path $root 'web'))
        $calls = Join-Path $root 'calls.log'
        $listener = Join-Path $root 'listen.ps1'
        # Binds the port poetry was asked for (hypercorn's --bind, or the
        # webserver's first argument) and holds it until ended.
        $listenBody = @(
            '$line = $args -join '' ''',
            "[System.IO.File]::AppendAllText('$calls', `"poetry `$line``r``n`")",
            'if ($line -match ''--bind 0\.0\.0\.0:(\d+)'') { $port = [int]$Matches[1] } elseif ($line -match ''scripts\.webserver (\d+)'') { $port = [int]$Matches[1] }',
            '$socket = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $port)',
            '$socket.Start()',
            'Start-Sleep -Seconds 120')
        if (-not $Listens) {
            $listenBody = @("[System.IO.File]::AppendAllText('$calls', `"poetry `$(`$args -join ' ')``r``n`")")
        }
        [System.IO.File]::WriteAllLines($listener, [string[]]$listenBody)
        $poetry = Join-Path $root 'poetry.cmd'
        [System.IO.File]::WriteAllText($poetry, "@echo off`r`n`"$PSHOME\powershell.exe`" -NoProfile -ExecutionPolicy Bypass -File `"$listener`" %*`r`n", [System.Text.Encoding]::ASCII)
        $npm = Join-Path $root 'npm.cmd'
        [System.IO.File]::WriteAllText($npm, "@echo off`r`necho npm %* in %CD%>>`"$calls`"`r`necho built`r`nexit /b $BuildExit`r`n", [System.Text.Encoding]::ASCII)
        return [pscustomobject]@{ Root = $root; Calls = $calls; Poetry = $poetry; Npm = $npm; ApiPort = Get-FreePort; WebPort = Get-FreePort }
    }

    function Read-GrandmaCall {
        param([string]$Path)
        return [string[]]@([System.IO.File]::ReadAllLines($Path) | ForEach-Object { $_.TrimEnd() })
    }
}

Describe 'Reading the environment and the ports' {
    It 'loads KEY=value lines, skipping comments, and loads nothing from an absent file' {
        $path = Join-Path $TestDrive 'grandma.env'
        [System.IO.File]::WriteAllLines($path, [string[]]@('# a comment', 'GRANDMA_TEST_A = "one"', "GRANDMA_TEST_B='two'", 'no equals'))
        try {
            Import-GrandmaEnvironment $path | Should -Be 2
            $env:GRANDMA_TEST_A | Should -BeExactly 'one'
            $env:GRANDMA_TEST_B | Should -BeExactly 'two'
        } finally {
            Remove-Item Env:GRANDMA_TEST_A, Env:GRANDMA_TEST_B
        }
        Import-GrandmaEnvironment (Join-Path $TestDrive 'absent.env') | Should -Be 0
    }
    It 'names the process listening on a port, and nothing on a free one' {
        $held = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
        $held.Start()
        try {
            Get-GrandmaListener ([System.Net.IPEndPoint]$held.LocalEndpoint).Port | Should -Be @([uint32]$PID)
        } finally { $held.Stop() }
        @(Get-GrandmaListener (Get-FreePort)).Count | Should -Be 0
    }
    It 'refuses by name a port that never opens' {
        { Wait-GrandmaListener (Get-FreePort) 1 'GRANDMA_TEST_NOT_READY' } | Should -Throw 'GRANDMA_TEST_NOT_READY: nothing listened on port * within 1s*'
    }
}

Describe 'Starting and stopping for real' {
    It 'starts the API and then builds and starts the web server, each once listening, and stop ends both trees' {
        $world = Initialize-GrandmaWorld
        try {
            $started = Start-GrandmaService -ProjectRoot $world.Root -ApiPort $world.ApiPort -WebPort $world.WebPort -Poetry $world.Poetry -Npm $world.Npm -ReadySeconds 60
            $started | Should -Be @("api:$($world.ApiPort)", "web:$($world.WebPort)")
            $calls = Read-GrandmaCall $world.Calls
            $calls[0] | Should -BeExactly ("poetry run hypercorn grandma_api.asgi:app --bind 0.0.0.0:$($world.ApiPort) --reload " +
                "--certfile $($world.Root)\web\cert.pem --keyfile $($world.Root)\web\key.pem")
            $calls[1] | Should -BeExactly "npm run build in $($world.Root)\web"
            $calls[2] | Should -BeExactly "poetry run python -m scripts.webserver $($world.WebPort) $($world.Root)\web"
            [System.IO.Directory]::Exists((Join-Path $world.Root 'logs')) | Should -BeTrue
            $said = @(Invoke-TestEntry $script:stopEntry @{ ApiPort = $world.ApiPort; WebPort = $world.WebPort } 6>&1 | ForEach-Object { "$_" })
            $said | Should -Be @("Stopped 2 process tree(s) on ports $($world.ApiPort) and $($world.WebPort).")
        } finally {
            [void](Stop-GrandmaService -Ports @($world.ApiPort, $world.WebPort) -Taskkill $script:taskkill)
        }
        @(Get-GrandmaListener $world.ApiPort).Count + @(Get-GrandmaListener $world.WebPort).Count | Should -Be 0
    }
    It 'refuses an API that never listens' {
        $world = Initialize-GrandmaWorld -Listens $false
        { Start-GrandmaService -ProjectRoot $world.Root -ApiPort $world.ApiPort -WebPort $world.WebPort -Poetry $world.Poetry -Npm $world.Npm -ReadySeconds 1 } |
            Should -Throw "GRANDMA_API_NOT_READY: nothing listened on port $($world.ApiPort)*"
    }
    It 'refuses a web server that never listens, with the API already up' {
        $world = Initialize-GrandmaWorld -Listens $false
        $held = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $world.ApiPort)
        $held.Start()
        try {
            { Start-GrandmaService -ProjectRoot $world.Root -ApiPort $world.ApiPort -WebPort $world.WebPort -Poetry $world.Poetry -Npm $world.Npm -ReadySeconds 1 } |
                Should -Throw "GRANDMA_WEB_NOT_READY: nothing listened on port $($world.WebPort)*"
        } finally { $held.Stop() }
    }
    It 'refuses a frontend build that fails, and starts no web server' {
        $world = Initialize-GrandmaWorld -BuildExit 3
        $held = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $world.ApiPort)
        $held.Start()
        try {
            { Start-GrandmaService -ProjectRoot $world.Root -ApiPort $world.ApiPort -WebPort $world.WebPort -Poetry $world.Poetry -Npm $world.Npm -ReadySeconds 1 } |
                Should -Throw 'GRANDMA_WEB_BUILD_FAILED: npm run build exited 3: built'
        } finally { $held.Stop() }
        Read-GrandmaCall $world.Calls | Should -Be @("npm run build in $($world.Root)\web")
    }
    It 'starts nothing under -WhatIf' {
        $world = Initialize-GrandmaWorld
        @(Start-GrandmaService -ProjectRoot $world.Root -ApiPort $world.ApiPort -WebPort $world.WebPort -Poetry $world.Poetry -Npm $world.Npm -ReadySeconds 1 -WhatIf).Count |
            Should -Be 0
        [System.IO.File]::Exists($world.Calls) | Should -BeFalse
    }
}

Describe 'Stopping against a stand-in taskkill' {
    It 'accepts exit 128 (already gone), refuses any other failure, and ends nothing under -WhatIf' {
        # The port is held by this process, so only stand-in taskkills, which
        # end nothing, are ever pointed at it.
        $held = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
        $held.Start()
        $port = ([System.Net.IPEndPoint]$held.LocalEndpoint).Port
        try {
            $gone = Join-Path $TestDrive 'taskkill-128.cmd'
            [System.IO.File]::WriteAllText($gone, "@echo off`r`necho not found`r`nexit /b 128`r`n", [System.Text.Encoding]::ASCII)
            Stop-GrandmaService -Ports @($port) -Taskkill $gone | Should -Be @("$PID on $port")
            $refused = Join-Path $TestDrive 'taskkill-5.cmd'
            [System.IO.File]::WriteAllText($refused, "@echo off`r`necho access denied`r`nexit /b 5`r`n", [System.Text.Encoding]::ASCII)
            { Stop-GrandmaService -Ports @($port) -Taskkill $refused } | Should -Throw "GRANDMA_STOP_FAILED: taskkill for pid $PID on port $port exited 5: access denied"
            @(Stop-GrandmaService -Ports @($port) -Taskkill $refused -WhatIf).Count | Should -Be 0
        } finally { $held.Stop() }
    }
}

Describe 'The start entry' {
    BeforeEach {
        $script:savedPort = [Environment]::GetEnvironmentVariable('PORT', 'Process')
        [Environment]::SetEnvironmentVariable('PORT', $null, 'Process')
    }
    AfterEach {
        [Environment]::SetEnvironmentVariable('PORT', $script:savedPort, 'Process')
    }
    It 'reads PORT from .env and says both sides are already running when they are' {
        $world = Initialize-GrandmaWorld
        [System.IO.File]::WriteAllText((Join-Path $world.Root '.env'), "PORT=$($world.ApiPort)`r`n")
        $api = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $world.ApiPort)
        $web = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $world.WebPort)
        $api.Start()
        $web.Start()
        try {
            $said = @(Invoke-TestEntry $script:startEntry @{ ProjectRoot = $world.Root; WebPort = $world.WebPort; Poetry = $world.Poetry; Npm = $world.Npm } 6>&1 |
                ForEach-Object { "$_" })
        } finally { $api.Stop(); $web.Stop() }
        $said | Should -Be @("Already running on ports $($world.ApiPort) and $($world.WebPort)")
    }
    It 'reads the API on 8090 when PORT is unset, starting nothing there' {
        # 8090 is held by this process when it is free; whatever else holds
        # it is listening too. Either way nothing is started on the real
        # port and nothing is stopped.
        $world = Initialize-GrandmaWorld
        $api = $null
        if (@(Get-GrandmaListener 8090).Count -eq 0) {
            $api = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 8090)
            $api.Start()
        }
        $web = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $world.WebPort)
        $web.Start()
        try {
            $said = @(Invoke-TestEntry $script:startEntry @{ ProjectRoot = $world.Root; WebPort = $world.WebPort; Poetry = $world.Poetry; Npm = $world.Npm } 6>&1 |
                ForEach-Object { "$_" })
        } finally {
            $web.Stop()
            if ($null -ne $api) { $api.Stop() }
        }
        $said | Should -Be @("Already running on ports 8090 and $($world.WebPort)")
        [System.IO.File]::Exists($world.Calls) | Should -BeFalse
    }
    It 'starts both sides from the entry and says where they listen' {
        $world = Initialize-GrandmaWorld
        $env:PORT = "$($world.ApiPort)"
        try {
            $said = @(Invoke-TestEntry $script:startEntry @{ ProjectRoot = $world.Root; WebPort = $world.WebPort; Poetry = $world.Poetry; Npm = $world.Npm; ReadySeconds = 60 } 6>&1 |
                ForEach-Object { "$_" })
        } finally {
            [void](Stop-GrandmaService -Ports @($world.ApiPort, $world.WebPort) -Taskkill $script:taskkill)
        }
        $said | Should -Be @("Started api:$($world.ApiPort), web:$($world.WebPort). API: https://localhost:$($world.ApiPort)  Web: https://localhost:$($world.WebPort)")
    }
}
