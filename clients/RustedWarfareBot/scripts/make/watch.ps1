# Play one match WITH the game window, so a human can watch the bot fight.
#
# The spectator launcher: the same agent-driven skirmish as play.ps1, with
# rendering on and the reproducibility machinery off -- lockstep would hold
# frames on the planner's acks and a fixed logic step is pointless when the
# point is to watch. The bot plays exactly its champion doctrine; only the
# window is new.

param(
    [Parameter(Mandatory = $true)][int]$Port,
    [Parameter(Mandatory = $true)][string]$GameDir,
    [Parameter(Mandatory = $true)][string]$PlayLog,
    [Parameter(Mandatory = $true)][string]$Map,
    [int]$Opponents = 1,
    [int]$Difficulty = 3,
    [Parameter(Mandatory = $true)][string]$Module,
    [Parameter(Mandatory = $true)][string]$Catalogue,
    [Parameter(Mandatory = $true)][string]$TypeDump,
    [Parameter(Mandatory = $true)][string]$PlayArgs,
    [Parameter(Mandatory = $true)][string]$Javac,
    [Parameter(Mandatory = $true)][string]$Jar
)

New-Item -ItemType Directory -Force "runs" | Out-Null

$stamp = [System.Guid]::NewGuid().ToString("N").Substring(0, 8)
$classesDir = "agent/build/watch-$stamp"
$agentJar = "agent/build/rw-agent-watch-$stamp.jar"
$root = (Resolve-Path -LiteralPath ".").Path
$gameLog = $root + "\" + $PlayLog

try {
    New-Item -ItemType Directory -Force $classesDir | Out-Null
    & $Javac --release 11 -Xlint:all -Werror -d $classesDir (
        Get-ChildItem "agent/src/rwbot/agent/*.java" | ForEach-Object { $_.FullName })
    if ($LASTEXITCODE -ne 0) { Write-Host "[watch] javac failed" -ForegroundColor Red; exit 1 }
    & $Jar cfm $agentJar agent/manifest.mf -C $classesDir .
    if ($LASTEXITCODE -ne 0) { Write-Host "[watch] jar failed" -ForegroundColor Red; exit 1 }

    $agentArgs = "channelPort=$Port;matchMap=$Map;matchOpponents=$Opponents;matchDifficulty=$Difficulty"

    # No -nodisplay: the window IS the point. Sound stays off; a spectator can
    # unmute in the game's own settings if wanted. No -sandbox: the agent
    # starts the requested match itself.
    $gameArgs = @(
        "-Xmx1000M",
        "--add-opens", "java.base/java.lang=ALL-UNNAMED",
        "-Djava.library.path=.",
        "-javaagent:$root\$agentJar=$agentArgs",
        "-cp", "game-lib.jar;libs/*",
        "com.corrodinggames.rts.java.Main",
        "-nosound",
        "-width", "1280", "-height", "800", "-log", $gameLog
    )

    Write-Host ""
    Write-Host "==> A game window will open; the bot plays, you watch." -ForegroundColor Green
    Write-Host ""

    $game = Start-Process -FilePath "$root\$GameDir\jvm64\bin\java.exe" `
        -ArgumentList $gameArgs -WorkingDirectory "$root\$GameDir" -PassThru `
        -RedirectStandardOutput "$root\$PlayLog.agent" -RedirectStandardError "$root\$PlayLog.err"
    try {
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $open = $false
        while ($sw.Elapsed.TotalSeconds -lt 120) {
            try {
                $probe = New-Object Net.Sockets.TcpClient("127.0.0.1", $Port)
                $probe.Close()
                $open = $true
                break
            }
            catch { Start-Sleep -Milliseconds 1000 }
        }
        if (-not $open) {
            Write-Host "[watch] the agent never opened port $Port" -ForegroundColor Red
            exit 1
        }
        $playArgList = $PlayArgs -split " "
        poetry run python -m $Module $Port "$Catalogue" "$TypeDump" @playArgList
    }
    finally {
        Stop-Process -Id $game.Id -Force -ErrorAction SilentlyContinue
        Write-Host "[watch] game stopped" -ForegroundColor DarkGray
    }
}
finally {
    if (Test-Path $classesDir) { Remove-Item -Recurse -Force $classesDir }
    Start-Sleep -Milliseconds 500
    if (Test-Path $agentJar) { Remove-Item -Force $agentJar -ErrorAction SilentlyContinue }
}
