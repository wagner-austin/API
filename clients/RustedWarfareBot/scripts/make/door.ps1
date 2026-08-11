# Launch the match service's HTTP door, detached from any session.
#
# The door is the queue's one submission surface (wiki:
# harness-match-service): workers poll Postgres directly and survive
# without it, but nothing can submit, reprioritize or retry until it is
# back. Running it as a terminal's child ties the control plane to that
# terminal's life; this launcher starts it detached, with its output in
# a real log, so the door outlives whoever started it.
#
# The database password is read from the container at launch and never
# written to disk: the DSN exists only in the door process's memory and
# command line.

param(
    [int]$Port = 27501,
    [string]$DoorLog = "runs/door.log"
)

$root = (Resolve-Path -LiteralPath ".").Path

$held = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($held) {
    Write-Host "door already listening on $Port (pid $($held.OwningProcess)); stop it first" -ForegroundColor Yellow
    exit 1
}

$pw = (docker exec platform-postgres printenv POSTGRES_PASSWORD).Trim()
if (-not $pw) {
    Write-Error "could not read POSTGRES_PASSWORD from platform-postgres; is Docker up?"
    exit 1
}
$dsn = "host=127.0.0.1 port=55432 user=covenant password=$pw dbname=covenant connect_timeout=10"

New-Item -ItemType Directory -Force "runs" | Out-Null
# WMI Create rather than Start-Process: a Start-Process child inherits the
# launching console's handles, so `make door` never gets EOF and hangs
# until the door dies -- the opposite of detached. WMI spawns with fresh
# handles; the cmd shell carries the env, the cwd and the log redirects.
$cmdLine = "cmd.exe /c cd /d $root & set PYTHONPATH=$root\src & " +
    "python -u -m scripts.match_service ""$dsn"" " +
    ">> $root\$DoorLog 2>> $root\$DoorLog.err"
$spawn = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
    -Arguments @{ CommandLine = $cmdLine }
if ($spawn.ReturnValue -ne 0) {
    Write-Error "could not spawn the door (WMI Create returned $($spawn.ReturnValue))"
    exit 1
}

Start-Sleep -Seconds 3
$listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $listening) {
    Write-Error "door spawned (pid $($spawn.ProcessId)) but nothing listens on $Port; see $DoorLog.err"
    Get-Content "$root\$DoorLog.err" -Tail 10 -ErrorAction SilentlyContinue
    exit 1
}
Write-Host "door up: pid $($listening.OwningProcess), http://127.0.0.1:$Port/, log $DoorLog" -ForegroundColor Green
