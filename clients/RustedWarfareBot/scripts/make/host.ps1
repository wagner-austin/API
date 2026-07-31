# Host a LAN game the bot plays in, for a human to join.
#
# The sparring launcher (wiki: multiplayer-portability-invariants). Differs
# from play.ps1 in exactly the ways a human demands: no -sandbox (the agent
# drives hostStart through the script surface from the menu), no lockstep
# and no settle (a peer cannot be world-held), and the channel port opens
# only once the human has joined and the match started -- so the wait is
# lobby-length, not boot-length.

param(
    [Parameter(Mandatory = $true)][int]$Port,
    [Parameter(Mandatory = $true)][string]$GameDir,
    [Parameter(Mandatory = $true)][string]$PlayLog,
    [Parameter(Mandatory = $true)][string]$HostMap,
    [int]$LobbyTimeoutSeconds = 900,
    [Parameter(Mandatory = $true)][string]$Module,
    [Parameter(Mandatory = $true)][string]$Catalogue,
    [Parameter(Mandatory = $true)][string]$TypeDump,
    [Parameter(Mandatory = $true)][string]$PlayArgs,
    [Parameter(Mandatory = $true)][string]$Javac,
    [Parameter(Mandatory = $true)][string]$Jar
)

New-Item -ItemType Directory -Force "runs" | Out-Null

$stamp = [System.Guid]::NewGuid().ToString("N").Substring(0, 8)
$classesDir = "agent/build/host-$stamp"
$agentJar = "agent/build/rw-agent-host-$stamp.jar"
$root = (Resolve-Path -LiteralPath ".").Path
$gameLog = $root + "\" + $PlayLog

try {
    New-Item -ItemType Directory -Force $classesDir | Out-Null
    & $Javac --release 11 -Xlint:all -Werror -d $classesDir (
        Get-ChildItem "agent/src/rwbot/agent/*.java" | ForEach-Object { $_.FullName })
    if ($LASTEXITCODE -ne 0) { Write-Host "[host] javac failed" -ForegroundColor Red; exit 1 }
    & $Jar cfm $agentJar agent/manifest.mf -C $classesDir .
    if ($LASTEXITCODE -ne 0) { Write-Host "[host] jar failed" -ForegroundColor Red; exit 1 }

    $agentArgs = "channelPort=$Port;hostMap=$HostMap"

    $gameArgs = @(
        "-Xmx1000M",
        "--add-opens", "java.base/java.lang=ALL-UNNAMED",
        "-Djava.library.path=.",
        "-javaagent:$root\$agentJar=$agentArgs",
        "-cp", "game-lib.jar;libs/*",
        "com.corrodinggames.rts.java.Main",
        "-nodisplay", "-nosound",
        "-width", "800", "-height", "600", "-log", $gameLog
    )

    $lan = (Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254*" } |
        Select-Object -First 1).IPAddress
    Write-Host ""
    Write-Host "==> JOIN FROM YOUR GAME CLIENT:" -ForegroundColor Green
    Write-Host "    Multiplayer -> Join by IP -> $lan (port 5123)" -ForegroundColor Green
    Write-Host "    The match starts automatically the moment you join." -ForegroundColor Green
    Write-Host ""

    $game = Start-Process -FilePath "$root\$GameDir\jvm64\bin\java.exe" `
        -ArgumentList $gameArgs -WorkingDirectory "$root\$GameDir" -PassThru `
        -RedirectStandardOutput "$root\$PlayLog.agent" -RedirectStandardError "$root\$PlayLog.err"
    try {
        # Lobby-length: the agent opens the channel port only after the human
        # joins and the match goes live.
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $open = $false
        while ($sw.Elapsed.TotalSeconds -lt $LobbyTimeoutSeconds) {
            try {
                $probe = New-Object Net.Sockets.TcpClient("127.0.0.1", $Port)
                $probe.Close()
                $open = $true
                break
            }
            catch { Start-Sleep -Milliseconds 1000 }
        }
        if (-not $open) {
            Write-Host "[host] nobody joined within $LobbyTimeoutSeconds seconds" -ForegroundColor Red
            exit 1
        }
        Write-Host "[host] match live; the bot is playing" -ForegroundColor Cyan
        $playArgList = $PlayArgs -split " "
        poetry run python -m $Module $Port "$Catalogue" "$TypeDump" @playArgList
    }
    finally {
        Stop-Process -Id $game.Id -Force -ErrorAction SilentlyContinue
        Write-Host "[host] game stopped" -ForegroundColor DarkGray
    }
}
finally {
    if (Test-Path $classesDir) { Remove-Item -Recurse -Force $classesDir }
    Start-Sleep -Milliseconds 500
    if (Test-Path $agentJar) { Remove-Item -Force $agentJar -ErrorAction SilentlyContinue }
}
