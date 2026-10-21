<#
.SYNOPSIS
    Give a Windows fleet node the fleet test database: the PostgreSQL 16
    client tools MCPs' scripts/testdb-setup.sh calls, on the account's PATH,
    and the loopback corvis-fleet-testdb container it restarts before a run.

.DESCRIPTION
    Why (MCPs board task daae17f2): 27 MCPs projects require the testdb tag,
    and until this script only diphtheria and lavender-wsl carried it, with
    diphtheria's production reservation keeping it idle, so one lane ran
    every Postgres-backed check and every closure on the board waited on it.
    sedona's build account reaches Docker Desktop (measured 2026-10-03), so
    it can run the same container the Linux nodes run, and the Windows
    toolchain probe reports its image as the testdb line
    (fleet.core.windows_toolchain_probe), which is what gives the node the
    tag on its next tick.

    THE CONTAINER IS MCPs' DEFINITION, scripts/host/lib/fleet-testdb.sh:
    pgvector/pgvector:pg16-bookworm, published on 127.0.0.1:15432, its data
    on a 4 GB tmpfs so a restart empties it and no anonymous volume leaks,
    and only the throwaway roles ci-bootstrap-testdb.sh creates. An existing
    container on that port is left as it is and verified, so a rerun is the
    audit: the declared loopback port, no volume mount, accepting connections.

    THE PORT IS FIXED, NOT DOCKER'S PICK (MCPs board task c6fe2fe3). Until
    2026-10-10 the container was published on 127.0.0.1::5432, a port the
    daemon picked from the ephemeral range, and on lavender-wsl the docker
    rehearsal lane's rootless daemon picked the same port for its postgres
    (fleet job fe97a17a). 15432 is below Linux's range (32768 to 60999) and
    Windows's dynamic range (49152 to 65535), so no allocator hands it out;
    MCPs packages/maketools fleet/test_database.py declares the same number
    as TESTDB_PORT and refuses a container published anywhere else. So a
    container found on another port is removed and created again (it is
    disposable: tmpfs data, throwaway roles), and before a create the port
    is read for a listener: one already bound is refused by name with the
    port and the owning process, instead of a docker run that fails on
    'address already in use' and says nothing about who holds it.

    THE CLIENT TOOLS ARE EDB's BINARIES ZIP, PINNED BY ITS SHA256, and only
    its pgsql/bin entries are unpacked: testdb-setup.sh calls pg_isready and
    ci-bootstrap-testdb.sh calls psql, and nothing on a node runs a server
    outside the container. They are unpacked beside the final directory and
    renamed into it, so an interrupted run leaves no half-installed bin that
    a rerun would take for finished.

    THE IMAGE IS PULLED UNDER AN EMPTY DOCKER CONFIG. Docker Desktop's own
    config names its credential store, which needs an interactive logon
    session, so over ssh every pull failed with "error getting credentials
    ... A specified logon session does not exist" (sedona, 2026-10-03). The
    image is public and needs no credential, so the pull reads a config.json
    of {} under Root with System32 alone on its PATH: a config naming no
    store still makes the CLI use docker-credential-wincred when it finds it
    on the PATH, and Docker Desktop installs it beside docker.exe. The run
    then finds the image locally and asks for no credential.

    Every docker and pg_isready call runs through cmd.exe with cmd's own
    2>&1, because under Windows PowerShell 5.1 a native program's stderr
    redirected by PowerShell is an error record, and docker writes its pull
    progress there.

    DOCKER DESKTOP NEEDS A LAUNCHER, OR A REBOOT ENDS THE LANE (MCPs board
    task 801ecab0). Docker Desktop starts at its account's logon, and a node
    nobody logs on to after a reboot runs no daemon and so no container:
    loki rebooted at 2026-10-14 20:33 local and claimed no testdb job for
    about three days. MCPs' node standard (fleet-mcp
    scripts/FleetStandardDocker.ps1) runs as SYSTEM at boot and every 15
    minutes and starts a stopped Docker Desktop through the scheduled task
    whose action is Docker Desktop.exe. SYSTEM cannot start it itself, since
    it must run as the account that owns its settings and its WSL distro.
    So this script makes sure such a task exists in the root folder: one
    already there (sedona's \SlimeDockerAutoStart) is left as it is, and
    otherwise \corvis-docker-desktop-start is registered as the account
    running this script, S4U so it needs no logon, at the highest run level,
    as the one opus-mcps-1008-debe registered on loki by hand on
    2026-10-10. It has no trigger: the node standard is what runs it, at
    boot and whenever the daemon is down. The tasks are read through Task
    Scheduler's COM service, because the scheduled-task cmdlets fail while
    another process deletes a task (MCPs board task d69786fa). After a
    registration the folder is read again, and a launcher still missing is
    refused by name (TESTDB_LAUNCHER_MISSING). Both Task Scheduler functions
    are testdb-launcher.ps1's, dot-sourced from beside this script so a suite
    can run them for real (MCPs board task 2a044a37).

    A FAILED CLIENT INSTALL LEAVES NOTHING BEHIND. A download, digest or
    unpack that throws removes the zip and bin.partial before the error
    propagates, and a run removes any an interrupted one left, so Root holds
    a whole bin or none.

.PARAMETER Root
    The directory the client's bin directory is unpacked into.

.PARAMETER ClientUrl
    The EDB binaries zip of the pinned PostgreSQL 16 release.

.PARAMETER ClientSha256
    The zip's sha256, measured 2026-10-03 on the hub.

.PARAMETER Docker
    Docker Desktop's docker.exe, by its full path, since the pull runs with
    no PATH to find it on.

.PARAMETER DockerDesktop
    Docker Desktop.exe, by its full path: the program a launcher task runs.

.PARAMETER Launcher
    The name the launcher task is registered under when the root folder has
    none.

.PARAMETER ReadLaunchers
    Reads the names of the root folder's tasks whose action runs a program,
    given its path. The default asks Task Scheduler's COM service; the test
    hands in a stand-in.

.PARAMETER RegisterLauncher
    Registers a launcher task, given its name and the program it runs. The
    default registers it with Task Scheduler as the account running this
    script; the test hands in a stand-in.

.PARAMETER Container
    The container's name, the one fleet.contracts.detection.TESTDB_CONTAINER
    and the probe ask for.

.PARAMETER Image
    The container's image.

.PARAMETER Port
    The loopback port the container publishes 5432 on, the fleet's one
    declared test database port.

.PARAMETER Listener
    Reads the process listening on a loopback port, as its name and pid, or
    $null when the port is free. The default asks Windows through
    Get-NetTCPConnection; the test hands in a stand-in.

.PARAMETER ReadySeconds
    How long a container may take to accept connections.

.PARAMETER Sleep
    Waits one second between readiness asks.

.PARAMETER PathScope
    Whose PATH gains the client's bin directory: the account's own, so every
    later ssh session of the build account finds pg_isready and psql. The
    test names its own process instead.
#>
[CmdletBinding()]
param(
    [string]$Root = 'C:\fleet\pgsql-16.15',
    [string]$ClientUrl = 'https://get.enterprisedb.com/postgresql/postgresql-16.15-1-windows-x64-binaries.zip',
    [string]$ClientSha256 = '25e6fcdfb8caec38691bf461125e7564508760666f7b8e5dc6a5f0818f58f81e',
    [string]$Docker = "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe",
    [string]$DockerDesktop = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe",
    [string]$Launcher = 'corvis-docker-desktop-start',
    [scriptblock]$ReadLaunchers = { param([string]$Execute) Get-RootTaskRunning $Execute },
    [scriptblock]$RegisterLauncher = { param([string]$Name, [string]$Execute) Register-DockerDesktopLauncher $Name $Execute },
    [string]$Container = 'corvis-fleet-testdb',
    [string]$Image = 'pgvector/pgvector:pg16-bookworm',
    [int]$Port = 15432,
    [scriptblock]$Listener = { param([int]$LocalPort) Get-TestdbPortListener $LocalPort },
    [int]$ReadySeconds = 60,
    [scriptblock]$Sleep = { Start-Sleep -Seconds 1 },
    [System.EnvironmentVariableTarget]$PathScope = 'User'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
. (Join-Path $PSScriptRoot 'testdb-launcher.ps1')
. (Join-Path $PSScriptRoot '..\..\..\scripts\lib\PortListener.ps1')

$Cmd = "$env:SystemRoot\System32\cmd.exe"

function Get-TestdbPortListener {
    <#
    .SYNOPSIS
        The process listening on a loopback port, as '<name> (pid N)', or
        $null when the port is free: what the default -Listener answers.
    .DESCRIPTION
        The listener is the repository's one read, scripts/lib's
        Get-PortListener. The owner is filtered by pipeline, not by
        Get-Process -Id, which writes an error for a process that is gone,
        and the harness refuses a suppressed one; an owner that vanished
        between the two reads is named by its pid alone.
    .PARAMETER LocalPort
        The loopback port.
    #>
    param([int]$LocalPort)
    $owners = @(Get-PortListener -Port $LocalPort -Address '127.0.0.1')
    if ($owners.Count -eq 0) { return $null }
    $names = @(Get-Process | Where-Object { $_.Id -eq $owners[0] } | ForEach-Object { $_.ProcessName })
    return "$(@($names + 'an unnamed process')[0]) (pid $($owners[0]))"
}

function Assert-DockerDesktopLauncher {
    <#
    .SYNOPSIS
        Read the root folder again for a task running Docker Desktop after
        one was registered, and refuse by name when there is still none.
    .DESCRIPTION
        A registration that returned and left no task would otherwise read
        as done, and the node would run no daemon after its next reboot.
    #>
    if (@(& $ReadLaunchers $DockerDesktop).Count -eq 0) {
        throw "TESTDB_LAUNCHER_MISSING: \$Launcher was registered, but no root folder task runs $DockerDesktop"
    }
}

function Undo-ClientDownload {
    <#
    .SYNOPSIS
        Remove what a failed client install left under Root, the zip and the
        half-unpacked bin.partial, so the node is as it was before the run.
    .DESCRIPTION
        File.Delete on a file that is absent is a no-op; the directory is
        asked for first, since Directory.Delete on an absent one throws.
    .PARAMETER Zip
        The downloaded zip.
    .PARAMETER Partial
        The directory the client is unpacked into before its rename.
    #>
    param([string]$Zip, [string]$Partial)
    [System.IO.File]::Delete($Zip)
    if ([System.IO.Directory]::Exists($Partial)) {
        [System.IO.Directory]::Delete($Partial, $true)
    }
}

function Invoke-Native {
    param([string]$Command)
    $lines = & $Cmd /d /s /c "$Command 2>&1"
    $status = $LASTEXITCODE
    # A command that printed nothing leaves $lines $null, which @() would
    # count as one line, so an empty `ps -aq` would read as a container.
    return [pscustomobject]@{ Status = $status; Lines = @($lines | Where-Object { $null -ne $_ }) }
}

function Invoke-Docker {
    param([string]$Executable, [string]$Arguments, [string]$Prefix = '')
    $answer = Invoke-Native "$Prefix`"$Executable`" $Arguments"
    if ($answer.Status -ne 0) {
        throw "TESTDB_DOCKER_FAILED: docker $Arguments exited $($answer.Status): $($answer.Lines -join ' ')"
    }
    return $answer.Lines
}

$bin = Join-Path $Root 'bin'
if (-not [System.IO.Directory]::Exists($bin)) {
    [void][System.IO.Directory]::CreateDirectory($Root)
    $zip = Join-Path $Root 'client.zip'
    $partial = Join-Path $Root 'bin.partial'
    # What an earlier run left half-done is removed before this one starts.
    Undo-ClientDownload $zip $partial
    try {
        (New-Object System.Net.WebClient).DownloadFile($ClientUrl, $zip)
        $digest = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($digest -ne $ClientSha256) {
            throw "TESTDB_CLIENT_DIGEST_MISMATCH: $ClientUrl has sha256 $digest, pinned $ClientSha256"
        }
        [void][System.IO.Directory]::CreateDirectory($partial)
        $archive = [System.IO.Compression.ZipFile]::OpenRead($zip)
        try {
            $entries = @($archive.Entries | Where-Object { $_.FullName -match '^pgsql/bin/[^/]+$' })
            if ($entries.Count -eq 0) {
                throw "TESTDB_CLIENT_EMPTY: $ClientUrl holds no pgsql/bin entries"
            }
            foreach ($entry in $entries) {
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $partial $entry.Name))
            }
        } finally {
            $archive.Dispose()
        }
        [System.IO.Directory]::Move($partial, $bin)
    } catch {
        Undo-ClientDownload $zip $partial
        throw
    }
    [System.IO.File]::Delete($zip)
    Write-Output "fleet-testdb: unpacked $($entries.Count) client file(s) into $bin"
}

$parts = @([string][Environment]::GetEnvironmentVariable('Path', $PathScope) -split ';' | Where-Object { $_ -ne '' })
if ($parts -notcontains $bin) {
    [Environment]::SetEnvironmentVariable('Path', ((@($parts) + $bin) -join ';'), $PathScope)
    Write-Output "fleet-testdb: added $bin to the $PathScope PATH"
}

if (@(& $ReadLaunchers $DockerDesktop).Count -eq 0) {
    & $RegisterLauncher $Launcher $DockerDesktop
    Assert-DockerDesktopLauncher
    Write-Output "fleet-testdb: registered \$Launcher, the task the node standard starts $DockerDesktop through"
}

$declared = "127.0.0.1:$Port"
$exists = @(Invoke-Docker $Docker "ps -aq --filter `"name=^$Container$`"").Count -gt 0
$published = ''
if ($exists) {
    $published = [string](@(Invoke-Docker $Docker "port $Container 5432/tcp") | Select-Object -First 1)
    if ($published -ne $declared) {
        # A container from before the fixed port, or a stopped one publishing
        # nothing: disposable, so it is removed and created again below.
        [void](Invoke-Docker $Docker "rm -f $Container")
        Write-Output "fleet-testdb: removed $Container published on '$published', not $declared"
        $exists = $false
    }
}
if (-not $exists) {
    $holder = & $Listener $Port
    if ($null -ne $holder) {
        throw "TESTDB_PORT_BOUND: $declared is already bound by $holder, so $Container cannot be published there"
    }
    $anonymous = Join-Path $Root 'docker-anonymous'
    [void][System.IO.Directory]::CreateDirectory($anonymous)
    [System.IO.File]::WriteAllText((Join-Path $anonymous 'config.json'), '{}')
    [void](Invoke-Docker $Docker "--config `"$anonymous`" pull $Image" "set `"PATH=$env:SystemRoot\System32`" && ")
    [void](Invoke-Docker $Docker ("run -d --name $Container --label corvis.role=fleet-testdb " +
            "--restart unless-stopped -p ${declared}:5432 " +
            "--tmpfs /var/lib/postgresql/data:rw,size=4g -e POSTGRES_PASSWORD=postgres $Image"))
    Write-Output "fleet-testdb: created $Container on $declared"
    $published = [string](@(Invoke-Docker $Docker "port $Container 5432/tcp") | Select-Object -First 1)
}
if ($published -ne $declared) {
    throw "TESTDB_NOT_ON_DECLARED_PORT: $Container is published on '$published', not $declared"
}
$port = $Port
$mounts = [string](@(Invoke-Docker $Docker "inspect $Container --format `"{{len .Mounts}}`"") | Select-Object -First 1)
if ($mounts -ne '0') {
    throw "TESTDB_HAS_VOLUME: $Container has $mounts volume mount(s); the data must be tmpfs"
}
$waited = 0
while ((Invoke-Native "`"$bin\pg_isready`" -h 127.0.0.1 -p $port -U postgres").Status -ne 0) {
    if ($waited -ge $ReadySeconds) {
        throw "TESTDB_NOT_READY: $Container not accepting on 127.0.0.1:$port after $ReadySeconds s"
    }
    & $Sleep
    $waited += 1
}
Write-Output "fleet-testdb: measured $Container accepting connections on 127.0.0.1:$port, 0 volume mounts"
