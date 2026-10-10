Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# tools/fleet/scripts/install-windows-testdb.ps1, the Windows fleet node's
# test database (MCPs board task daae17f2). Each case runs the real script
# against a client zip it builds under $TestDrive, fetched through a file://
# URL, and a stand-in docker.cmd whose answers are files beside it, so the
# unpack, the PATH, every docker call and the readiness wait are measured.
# PATH is the case's own process PATH and restored after it.

BeforeAll {
    $script:install = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'scripts\install-windows-testdb.ps1'
    Add-Type -AssemblyName System.IO.Compression.FileSystem

    # pg_isready: refused once and then accepting, unless a 'ready' file
    # says it already accepts or a 'never' file says it never will.
    $script:isReady = "@echo off`r`necho %*>>`"%~dp0isready.calls.txt`"`r`n" +
        "if exist `"%~dp0never`" exit /b 2`r`nif exist `"%~dp0ready`" exit /b 0`r`n" +
        "type nul > `"%~dp0ready`"`r`nexit /b 2`r`n"

    function Initialize-ClientZip {
        <#
        .SYNOPSIS
            A zip holding the given entries, each named as EDB's zip names
            its own, with its sha256.
        .PARAMETER Entries
            Entry name to its text.
        .OUTPUTS
            PSCustomObject: Url (file://) and Sha256.
        #>
        param([hashtable]$Entries)
        $path = Join-Path $TestDrive ('client-' + [guid]::NewGuid().ToString('N') + '.zip')
        $archive = [System.IO.Compression.ZipFile]::Open($path, 'Create')
        foreach ($name in $Entries.Keys) {
            $writer = New-Object System.IO.StreamWriter(($archive.CreateEntry($name)).Open())
            $writer.Write($Entries[$name])
            $writer.Dispose()
        }
        $archive.Dispose()
        return [pscustomobject]@{
            Url = ([Uri]$path).AbsoluteUri
            Sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }

    function Initialize-Docker {
        <#
        .SYNOPSIS
            A docker.cmd that records every call and answers ps, the pull,
            run, rm, port and inspect from files beside it: 'exists' (the
            container), 'pull-fails', port.txt (what an existing container
            publishes), run-port.txt (what a container run publishes) and
            mounts.txt.
        .OUTPUTS
            PSCustomObject: Path, Directory and Calls (the record).
        #>
        param(
            [string]$Port = '127.0.0.1:15432', [string]$RunPort = '127.0.0.1:15432', [string]$Mounts = '0',
            [switch]$Exists, [switch]$PullFails
        )
        $directory = Join-Path $TestDrive ('docker-' + [guid]::NewGuid().ToString('N'))
        [void][System.IO.Directory]::CreateDirectory($directory)
        $body = @(
            '@echo off', 'echo %*>>"%~dp0calls.txt"',
            'if "%1"=="ps" goto ps', 'if "%1"=="--config" goto pull', 'if "%1"=="run" goto run', 'if "%1"=="rm" goto rm',
            'if "%1"=="port" goto port', 'if "%1"=="inspect" goto inspect', 'exit /b 99',
            ':ps', 'if exist "%~dp0exists" echo 3f2a1b', 'exit /b 0',
            ':pull', 'echo PATH=%PATH%>>"%~dp0pull-path.txt"',
            'if exist "%~dp0pull-fails" (1>&2 echo error getting credentials & exit /b 1)', 'exit /b 0',
            ':run', 'type nul > "%~dp0exists"', 'copy /y "%~dp0run-port.txt" "%~dp0port.txt" >nul', 'echo 3f2a1b', 'exit /b 0',
            ':rm', 'del "%~dp0exists"', 'exit /b 0',
            ':port', 'if exist "%~dp0exists" type "%~dp0port.txt"', 'exit /b 0',
            ':inspect', 'type "%~dp0mounts.txt"', 'exit /b 0'
        ) -join "`r`n"
        [System.IO.File]::WriteAllText((Join-Path $directory 'docker.cmd'), "$body`r`n", [System.Text.Encoding]::ASCII)
        [System.IO.File]::WriteAllText((Join-Path $directory 'port.txt'), "$Port`r`n")
        [System.IO.File]::WriteAllText((Join-Path $directory 'run-port.txt'), "$RunPort`r`n")
        [System.IO.File]::WriteAllText((Join-Path $directory 'mounts.txt'), "$Mounts`r`n")
        if ($Exists) {
            [System.IO.File]::WriteAllText((Join-Path $directory 'exists'), '')
        }
        if ($PullFails) {
            [System.IO.File]::WriteAllText((Join-Path $directory 'pull-fails'), '')
        }
        return [pscustomobject]@{
            Path = Join-Path $directory 'docker.cmd'; Directory = $directory; Calls = Join-Path $directory 'calls.txt'
        }
    }

    function Initialize-Installed {
        <#
        .SYNOPSIS
            A root whose client is already unpacked, its pg_isready
            accepting at once or never.
        .OUTPUTS
            String: the root.
        #>
        param([switch]$NeverReady)
        $root = Join-Path $TestDrive ('root-' + [guid]::NewGuid().ToString('N'))
        $bin = Join-Path $root 'bin'
        [void][System.IO.Directory]::CreateDirectory($bin)
        [System.IO.File]::WriteAllText((Join-Path $bin 'pg_isready.cmd'), $script:isReady, [System.Text.Encoding]::ASCII)
        $state = @{ $true = 'never'; $false = 'ready' }[$NeverReady.IsPresent]
        [System.IO.File]::WriteAllText((Join-Path $bin $state), '')
        return $root
    }

    # The script fails by throwing; a native it ran leaves its exit code in
    # $LASTEXITCODE, so a non-zero one left behind is itself an error here.
    function Invoke-Install {
        param([hashtable]$Parameters)
        $global:LASTEXITCODE = 0
        & $script:install @Parameters
        if ($LASTEXITCODE -ne 0) {
            throw "TEST_INSTALL_EXITED: install-windows-testdb exited $LASTEXITCODE"
        }
    }

    function Get-InstallParameterSet {
        param([string]$Root, [pscustomobject]$Docker, [pscustomobject]$Client)
        $parameters = @{ Root = $Root; Docker = $Docker.Path; PathScope = 'Process'; ClientUrl = 'file:///absent.zip' }
        if ($null -ne $Client) {
            $parameters.ClientUrl = $Client.Url
            $parameters.ClientSha256 = $Client.Sha256
        }
        return $parameters
    }
}

Describe 'install-windows-testdb' {
    BeforeEach {
        $script:savedPath = $env:PATH
    }
    AfterEach {
        $env:PATH = $script:savedPath
    }

    It 'unpacks only the client''s bin, adds it to PATH, pulls anonymously, creates the container and waits for it' {
        $client = Initialize-ClientZip @{
            'pgsql/bin/pg_isready.cmd' = $script:isReady; 'pgsql/bin/psql.cmd' = '@echo off'
            'pgsql/doc/readme.txt' = 'not the client'; 'pgsql/bin/sub/nested.txt' = 'not flat'
        }
        $root = Join-Path $TestDrive 'fresh'
        [void][System.IO.Directory]::CreateDirectory((Join-Path $root 'bin.partial\stale'))
        $docker = Initialize-Docker
        $said = @(Invoke-Install (Get-InstallParameterSet $root $docker $client))
        $bin = Join-Path $root 'bin'
        $said | Should -Be @(
            "fleet-testdb: unpacked 2 client file(s) into $bin", "fleet-testdb: added $bin to the Process PATH",
            'fleet-testdb: created corvis-fleet-testdb on 127.0.0.1:15432',
            'fleet-testdb: measured corvis-fleet-testdb accepting connections on 127.0.0.1:15432, 0 volume mounts')
        @([System.IO.Directory]::GetFiles($bin) | ForEach-Object { [System.IO.Path]::GetFileName($_) } | Sort-Object) |
            Should -Be @('isready.calls.txt', 'pg_isready.cmd', 'psql.cmd', 'ready')
        Test-Path -LiteralPath (Join-Path $root 'client.zip') | Should -BeFalse
        Test-Path -LiteralPath (Join-Path $root 'bin.partial') | Should -BeFalse
        @($env:PATH -split ';')[-1] | Should -BeExactly $bin
        $anonymous = Join-Path $root 'docker-anonymous'
        [System.IO.File]::ReadAllText((Join-Path $anonymous 'config.json')) | Should -BeExactly '{}'
        [System.IO.File]::ReadAllLines($docker.Calls) | Should -Be @(
            'ps -aq --filter "name=^corvis-fleet-testdb$"',
            "--config `"$anonymous`" pull pgvector/pgvector:pg16-bookworm",
            ('run -d --name corvis-fleet-testdb --label corvis.role=fleet-testdb --restart unless-stopped ' +
                '-p 127.0.0.1:15432:5432 --tmpfs /var/lib/postgresql/data:rw,size=4g -e POSTGRES_PASSWORD=postgres ' +
                'pgvector/pgvector:pg16-bookworm'),
            'port corvis-fleet-testdb 5432/tcp',
            'inspect corvis-fleet-testdb --format "{{len .Mounts}}"')
        [System.IO.File]::ReadAllText((Join-Path $docker.Directory 'pull-path.txt')).Trim() |
            Should -BeExactly "PATH=$env:SystemRoot\System32"
        [System.IO.File]::ReadAllLines((Join-Path $bin 'isready.calls.txt')) |
            Should -Be @('-h 127.0.0.1 -p 15432 -U postgres', '-h 127.0.0.1 -p 15432 -U postgres')
    }
    It 'removes a container published on another port and creates it again on the declared one' {
        $root = Initialize-Installed
        $docker = Initialize-Docker -Exists -Port '127.0.0.1:53681'
        $env:PATH = "$env:PATH;$(Join-Path $root 'bin')"
        $said = @(Invoke-Install (Get-InstallParameterSet $root $docker $null))
        $said | Should -Be @(
            "fleet-testdb: removed corvis-fleet-testdb published on '127.0.0.1:53681', not 127.0.0.1:15432",
            'fleet-testdb: created corvis-fleet-testdb on 127.0.0.1:15432',
            'fleet-testdb: measured corvis-fleet-testdb accepting connections on 127.0.0.1:15432, 0 volume mounts')
        $calls = [System.IO.File]::ReadAllLines($docker.Calls)
        $calls[0..2] | Should -Be @(
            'ps -aq --filter "name=^corvis-fleet-testdb$"', 'port corvis-fleet-testdb 5432/tcp', 'rm -f corvis-fleet-testdb')
        $calls[4] | Should -Match '^run -d --name corvis-fleet-testdb .* -p 127\.0\.0\.1:15432:5432 '
        $calls.Count | Should -Be 7
    }
    It 'refuses a port another process already holds, naming the port and the process, and runs nothing' {
        $docker = Initialize-Docker
        $parameters = Get-InstallParameterSet (Initialize-Installed) $docker $null
        $parameters.Listener = { param([int]$LocalPort) "python (pid 4242) on $LocalPort" }
        { Invoke-Install $parameters } | Should -Throw -ExpectedMessage (
            'TESTDB_PORT_BOUND: 127.0.0.1:15432 is already bound by python (pid 4242) on 15432, so corvis-fleet-testdb cannot be published there')
        [System.IO.File]::ReadAllLines($docker.Calls) | Should -Be @('ps -aq --filter "name=^corvis-fleet-testdb$"')
        Test-Path -LiteralPath (Join-Path $docker.Directory 'exists') | Should -BeFalse
    }
    It 'reads a real loopback listener and its process through the default Listener' {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
        $listener.Start()
        try {
            $held = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
            $docker = Initialize-Docker
            $parameters = Get-InstallParameterSet (Initialize-Installed) $docker $null
            $parameters.Port = $held
            $me = (Get-Process -Id $PID).ProcessName
            { Invoke-Install $parameters } | Should -Throw -ExpectedMessage (
                "TESTDB_PORT_BOUND: 127.0.0.1:$held is already bound by $me (pid $PID), so corvis-fleet-testdb cannot be published there")
        } finally {
            $listener.Stop()
        }
    }
    It 'refuses a container that came up anywhere but the declared port' {
        $docker = Initialize-Docker -RunPort '0.0.0.0:15432'
        { Invoke-Install (Get-InstallParameterSet (Initialize-Installed) $docker $null) } |
            Should -Throw -ExpectedMessage "TESTDB_NOT_ON_DECLARED_PORT: corvis-fleet-testdb is published on '0.0.0.0:15432', not 127.0.0.1:15432"
    }
    It 'leaves an unpacked client, a PATH that has it and an existing container as they are, and verifies them' {
        $root = Initialize-Installed
        $docker = Initialize-Docker -Exists
        $env:PATH = "$env:PATH;$(Join-Path $root 'bin')"
        $before = $env:PATH
        $said = @(Invoke-Install (Get-InstallParameterSet $root $docker $null))
        $said | Should -Be @('fleet-testdb: measured corvis-fleet-testdb accepting connections on 127.0.0.1:15432, 0 volume mounts')
        $env:PATH | Should -BeExactly $before
        [System.IO.File]::ReadAllLines($docker.Calls) | Should -Be @(
            'ps -aq --filter "name=^corvis-fleet-testdb$"', 'port corvis-fleet-testdb 5432/tcp',
            'inspect corvis-fleet-testdb --format "{{len .Mounts}}"')
    }
    It 'refuses a client zip whose digest is not the pinned one, and keeps none of it' {
        $client = Initialize-ClientZip @{ 'pgsql/bin/psql.cmd' = '@echo off' }
        $root = Join-Path $TestDrive 'mismatch'
        $parameters = Get-InstallParameterSet $root (Initialize-Docker) $client
        $parameters.ClientSha256 = '0' * 64
        { Invoke-Install $parameters } | Should -Throw -ExpectedMessage "TESTDB_CLIENT_DIGEST_MISMATCH: $($client.Url) has sha256 $($client.Sha256), pinned $('0' * 64)"
        Test-Path -LiteralPath (Join-Path $root 'client.zip') | Should -BeFalse
        Test-Path -LiteralPath (Join-Path $root 'bin') | Should -BeFalse
    }
    It 'refuses a client zip with no pgsql/bin entries' {
        $client = Initialize-ClientZip @{ 'pgsql/doc/readme.txt' = 'no client here' }
        $root = Join-Path $TestDrive 'empty'
        { Invoke-Install (Get-InstallParameterSet $root (Initialize-Docker) $client) } |
            Should -Throw -ExpectedMessage "TESTDB_CLIENT_EMPTY: $($client.Url) holds no pgsql/bin entries"
        Test-Path -LiteralPath (Join-Path $root 'bin') | Should -BeFalse
    }
    It 'names a docker call that fails, with what docker said' {
        $docker = Initialize-Docker -PullFails
        $root = Initialize-Installed
        $anonymous = Join-Path $root 'docker-anonymous'
        { Invoke-Install (Get-InstallParameterSet $root $docker $null) } | Should -Throw -ExpectedMessage (
            "TESTDB_DOCKER_FAILED: docker --config `"$anonymous`" pull pgvector/pgvector:pg16-bookworm exited 1: error getting credentials*")
        Test-Path -LiteralPath (Join-Path $docker.Directory 'exists') | Should -BeFalse
    }
    It 'refuses a container whose data is on a volume' {
        $docker = Initialize-Docker -Exists -Mounts '1'
        { Invoke-Install (Get-InstallParameterSet (Initialize-Installed) $docker $null) } |
            Should -Throw -ExpectedMessage 'TESTDB_HAS_VOLUME: corvis-fleet-testdb has 1 volume mount(s); the data must be tmpfs'
    }
    It 'gives a container ReadySeconds to accept, waiting between asks, and refuses it by name after them' {
        $root = Initialize-Installed -NeverReady
        $parameters = Get-InstallParameterSet $root (Initialize-Docker -Exists) $null
        $sleeps = New-Object System.Collections.Generic.List[int]
        $parameters.ReadySeconds = 2
        $parameters.Sleep = { $sleeps.Add(1) }.GetNewClosure()
        { Invoke-Install $parameters } |
            Should -Throw -ExpectedMessage 'TESTDB_NOT_READY: corvis-fleet-testdb not accepting on 127.0.0.1:15432 after 2 s'
        $sleeps.Count | Should -Be 2
        @([System.IO.File]::ReadAllLines((Join-Path $root 'bin\isready.calls.txt'))).Count | Should -Be 3
    }
}
