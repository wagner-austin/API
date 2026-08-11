# Relaunch the match-worker fleet, detached from any session.
#
# The third Docker-crash recovery made this a procedure worth one
# command (log 2026-08-11): when Postgres drops, every worker dies
# mid-poll and takes its engine with it; the queue rows survive, and
# the first worker's startup reap requeues anything a dead owner held
# once its heartbeat passes the stale threshold. This launcher only
# starts workers that are not already running, so it is safe to run on
# a half-alive fleet.
#
# Same detachment and secrecy rules as door.ps1: WMI Create so no
# console handles tie a worker to this terminal, and the database
# password is read from the container at launch, never written to disk.

param(
    [int]$Workers = 8,
    [string]$ClonePool = "0,1,2,3,4,5,6,7"
)

$root = (Resolve-Path -LiteralPath ".").Path

$pw = (docker exec platform-postgres printenv POSTGRES_PASSWORD).Trim()
if (-not $pw) {
    Write-Error "could not read POSTGRES_PASSWORD from platform-postgres; is Docker up?"
    exit 1
}
$dsn = "host=127.0.0.1 port=55432 user=covenant password=$pw dbname=covenant connect_timeout=10"

New-Item -ItemType Directory -Force "runs" | Out-Null
$alive = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'match_worker' } |
    ForEach-Object { if ($_.CommandLine -match '(creepw-\d+)') { $Matches[1] } }) | Sort-Object -Unique
$started = 0
foreach ($n in 1..$Workers) {
    $name = "creepw-$n"
    if ($alive -contains $name) {
        Write-Host "$name already running; leaving it alone"
        continue
    }
    $cmd = "cd /d $root & $root\.venv\Scripts\python.exe -u -m scripts.match_worker " +
        """$dsn"" $name $ClonePool >> $root\runs\$name.log 2>&1"
    $spawn = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
        -Arguments @{ CommandLine = "cmd.exe /c $cmd" }
    if ($spawn.ReturnValue -ne 0) {
        Write-Error "could not spawn $name (WMI Create returned $($spawn.ReturnValue))"
        exit 1
    }
    $started++
}
Start-Sleep -Seconds 10
$now = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'match_worker' } |
    ForEach-Object { if ($_.CommandLine -match '(creepw-\d+)') { $Matches[1] } }) | Sort-Object -Unique
Write-Host "fleet up: started $started, running now: $($now -join ', ')" -ForegroundColor Green
