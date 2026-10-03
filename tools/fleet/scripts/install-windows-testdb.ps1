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
    pgvector/pgvector:pg16-bookworm, published on a loopback port Docker
    picks, its data on a 4 GB tmpfs so a restart empties it and no anonymous
    volume leaks, and only the throwaway roles ci-bootstrap-testdb.sh
    creates. An existing container is left as it is and verified, so a rerun
    is the audit: loopback only, no volume mount, accepting connections.

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

.PARAMETER Root
    The directory the client's bin directory is unpacked into.

.PARAMETER ClientUrl
    The EDB binaries zip of the pinned PostgreSQL 16 release.

.PARAMETER ClientSha256
    The zip's sha256, measured 2026-10-03 on the hub.

.PARAMETER Docker
    Docker Desktop's docker.exe, by its full path, since the pull runs with
    no PATH to find it on.

.PARAMETER Container
    The container's name, the one fleet.contracts.detection.TESTDB_CONTAINER
    and the probe ask for.

.PARAMETER Image
    The container's image.

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
    [string]$Container = 'corvis-fleet-testdb',
    [string]$Image = 'pgvector/pgvector:pg16-bookworm',
    [int]$ReadySeconds = 60,
    [scriptblock]$Sleep = { Start-Sleep -Seconds 1 },
    [System.EnvironmentVariableTarget]$PathScope = 'User'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

$Cmd = "$env:SystemRoot\System32\cmd.exe"

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
    (New-Object System.Net.WebClient).DownloadFile($ClientUrl, $zip)
    $digest = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($digest -ne $ClientSha256) {
        [System.IO.File]::Delete($zip)
        throw "TESTDB_CLIENT_DIGEST_MISMATCH: $ClientUrl has sha256 $digest, pinned $ClientSha256"
    }
    $partial = Join-Path $Root 'bin.partial'
    if ([System.IO.Directory]::Exists($partial)) {
        [System.IO.Directory]::Delete($partial, $true)
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
    [System.IO.File]::Delete($zip)
    Write-Output "fleet-testdb: unpacked $($entries.Count) client file(s) into $bin"
}

$parts = @([string][Environment]::GetEnvironmentVariable('Path', $PathScope) -split ';' | Where-Object { $_ -ne '' })
if ($parts -notcontains $bin) {
    [Environment]::SetEnvironmentVariable('Path', ((@($parts) + $bin) -join ';'), $PathScope)
    Write-Output "fleet-testdb: added $bin to the $PathScope PATH"
}

if (@(Invoke-Docker $Docker "ps -aq --filter `"name=^$Container$`"").Count -eq 0) {
    $anonymous = Join-Path $Root 'docker-anonymous'
    [void][System.IO.Directory]::CreateDirectory($anonymous)
    [System.IO.File]::WriteAllText((Join-Path $anonymous 'config.json'), '{}')
    [void](Invoke-Docker $Docker "--config `"$anonymous`" pull $Image" "set `"PATH=$env:SystemRoot\System32`" && ")
    [void](Invoke-Docker $Docker ("run -d --name $Container --label corvis.role=fleet-testdb " +
            "--restart unless-stopped -p 127.0.0.1::5432 " +
            "--tmpfs /var/lib/postgresql/data:rw,size=4g -e POSTGRES_PASSWORD=postgres $Image"))
    Write-Output "fleet-testdb: created $Container"
}
$published = [string](@(Invoke-Docker $Docker "port $Container 5432/tcp") | Select-Object -First 1)
if (-not ($published -match '^127\.0\.0\.1:(\d+)$')) {
    throw "TESTDB_NOT_LOOPBACK: $Container is published on '$published', not loopback"
}
$port = $Matches[1]
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
