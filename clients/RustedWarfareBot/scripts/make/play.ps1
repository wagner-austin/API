# Launch one headless match and run the planner against it.
#
# The body of the Makefile `play` recipe, moved to a file because Make runs
# each recipe line as one shell invocation and the recipe had grown into a
# four-thousand-character line nobody could read or diff. The Makefile still
# owns the variables and their defaults; this file owns the control flow.
#
# Launch plumbing, not experiment code: like the Makefile that invokes it,
# this file is read from the working tree even when the match itself imports
# a batch's frozen snapshot ($Tree). What the snapshot freezes is what the
# match RUNS -- src/rw_bot, scripts (as imported via PYTHONPATH), doctrines
# and the agent jar; how the match is BOOTED belongs to the harness
# ([[policy-loop]]).

param(
    [Parameter(Mandatory = $true)][int]$Port,
    [Parameter(Mandatory = $true)][string]$GameDir,
    [int]$Seed = 0,
    [int]$Lockstep = 0,
    [int]$Settle = 22,
    [Parameter(Mandatory = $true)][string]$PlayLog,
    [string]$Map = "",
    [int]$Opponents = 1,
    [int]$Difficulty = 0,
    [string]$Tree = "",
    [int]$PinDelta = 0,
    [int]$FastForward = 0,
    [int]$RngTap = 0,
    [string]$ExtraAgentArgs = "",
    [Parameter(Mandatory = $true)][string]$Module,
    [Parameter(Mandatory = $true)][string]$Catalogue,
    [Parameter(Mandatory = $true)][string]$TypeDump,
    [Parameter(Mandatory = $true)][string]$PlayArgs,
    [Parameter(Mandatory = $true)][string]$Javac,
    [Parameter(Mandatory = $true)][string]$Jar
)

New-Item -ItemType Directory -Force "runs" | Out-Null

$stamp = [System.Guid]::NewGuid().ToString("N").Substring(0, 8)
$classesDir = "agent/build/play-$stamp"
$agentJar = "agent/build/rw-agent-play-$stamp.jar"
$root = (Resolve-Path -LiteralPath ".").Path
$gameLog = $root + "\" + $PlayLog

try {
    if ($Tree -ne "") {
        # A frozen batch: the jar was built once when the snapshot was taken,
        # so reuse it rather than compiling per match.
        $agentJar = "$Tree/rw-agent.jar"
        Write-Host "[play] frozen tree: $Tree" -ForegroundColor DarkGray
    }
    else {
        New-Item -ItemType Directory -Force $classesDir | Out-Null
        & $Javac --release 11 -Xlint:all -Werror -d $classesDir (
            Get-ChildItem "agent/src/rwbot/agent/*.java" | ForEach-Object { $_.FullName })
        if ($LASTEXITCODE -ne 0) { Write-Host "[play] javac failed" -ForegroundColor Red; exit 1 }
        & $Jar cfm $agentJar agent/manifest.mf -C $classesDir .
        if ($LASTEXITCODE -ne 0) { Write-Host "[play] jar failed" -ForegroundColor Red; exit 1 }
    }

    $agentArgs = "channelPort=$Port"
    if ($Seed -ne 0) {
        $agentArgs += ";randomSeed=$Seed"
        Write-Host "[play] engine random pinned to seed $Seed" -ForegroundColor DarkGray
    }
    if ($Lockstep -ne 0) {
        $agentArgs += ";lockstepFrames=$Lockstep"
        Write-Host "[play] lockstep every $Lockstep frames" -ForegroundColor DarkGray
    }
    if ($Map -ne "") {
        $agentArgs += ";matchMap=$Map;matchOpponents=$Opponents;matchDifficulty=$Difficulty"
        Write-Host "[play] $Opponents opponent(s) at difficulty $Difficulty on $Map" -ForegroundColor DarkGray
    }
    if ($PinDelta -ne 0) {
        # A frame becomes a fixed quantum of simulation, which is what makes a
        # seed reproduce ([[policy-determinism]]). Off for anything a human
        # watches live; frozen trees older than the option must pass 0.
        $agentArgs += ";pinDeltaMs=$PinDelta"
        Write-Host "[play] frame delta pinned to ${PinDelta}ms" -ForegroundColor DarkGray
    }
    if ($FastForward -ne 0) {
        # The gym knob: N identical pinned steps per loop pass instead of
        # one -- same simulation, N times the wall speed, CPU permitting
        # (task #35). Zero leaves the engine at the wall clock.
        $agentArgs += ";fastForward=$FastForward"
        Write-Host "[play] fast-forward: ${FastForward}x" -ForegroundColor DarkGray
    }
    if ($RngTap -ne 0) {
        # Diagnostic: per-caller draw counts on the engine generator, one
        # line per sample window in the agent log (task #36).
        $agentArgs += ";rngTap=true"
        Write-Host "[play] rng draw tap armed" -ForegroundColor DarkGray
    }
    if ($ExtraAgentArgs -ne "") {
        # Diagnostic passthrough: any agent option the launch knobs above do
        # not name, e.g. discovery snapshots riding a live match
        # (discoverAtSeconds + stateOutPath) to capture full rosters around
        # a divergence window.
        $agentArgs += ";$ExtraAgentArgs"
        Write-Host "[play] extra agent args: $ExtraAgentArgs" -ForegroundColor DarkGray
    }

    $gameArgs = @(
        "-Xmx1000M",
        "--add-opens", "java.base/java.lang=ALL-UNNAMED",
        "--add-opens", "java.base/java.util=ALL-UNNAMED",
        "-Djava.library.path=.",
        "-javaagent:$root\$agentJar=$agentArgs",
        "-cp", "game-lib.jar;libs/*",
        "com.corrodinggames.rts.java.Main",
        "-nodisplay", "-nosound"
    )
    if ($Map -eq "") {
        # The engine's own hardcoded sandbox free-for-all.
        $gameArgs += "-sandbox"
    }
    else {
        Write-Host "[play] -sandbox withheld; the agent starts the match" -ForegroundColor DarkGray
    }
    $gameArgs += @("-width", "800", "-height", "600", "-log", $gameLog)

    # A zombie engine from a killed worker can still hold this port -- its
    # job was requeued but the process plays on, and the new agent dies at
    # the bind (vhdoom96b, 2026-08-09). Only a java engine can legitimately
    # hold a match port, so a java holder is always an orphan; anything
    # else holding the port is left alone and the bind fails loudly.
    $holder = (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue).OwningProcess
    if ($holder) {
        $holderProc = Get-Process -Id $holder -ErrorAction SilentlyContinue
        if ($holderProc -and $holderProc.ProcessName -eq "java") {
            Write-Host "[play] clearing orphaned engine (pid $holder) off port $Port" -ForegroundColor Yellow
            Stop-Process -Id $holder -Force
            Start-Sleep -Milliseconds 500
        }
    }
    $game = Start-Process -FilePath "$root\$GameDir\jvm64\bin\java.exe" `
        -ArgumentList $gameArgs -WorkingDirectory "$root\$GameDir" -PassThru `
        -RedirectStandardOutput "$root\$PlayLog.agent" -RedirectStandardError "$root\$PlayLog.err"
    try {
        # The agent opens the channel port once the game is up; the planner
        # must not connect before that or the connect is refused.
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $open = $false
        while ($sw.Elapsed.TotalSeconds -lt 90) {
            try {
                $probe = New-Object Net.Sockets.TcpClient("127.0.0.1", $Port)
                $probe.Close()
                $open = $true
                break
            }
            catch { Start-Sleep -Milliseconds 1000 }
        }
        if (-not $open) {
            Write-Host "[play] the agent never opened port $Port" -ForegroundColor Red
            exit 1
        }
        if ($Map -eq "") {
            Write-Host "[play] channel open; letting the map settle ${Settle}s" -ForegroundColor DarkGray
            Start-Sleep -Seconds $Settle
        }
        else {
            Write-Host "[play] channel open; the world is held at its first frame, no settle" -ForegroundColor DarkGray
        }

        $py = @()
        if ($Tree -ne "") {
            # -P keeps the repository root off sys.path so the snapshot wins;
            # the working directory stays the root so data paths resolve.
            $env:PYTHONPATH = "$root\$Tree;$root\$Tree\src"
            $py = @("-P")
        }
        # Split as the recipe's inline expansion used to: PLAY_ARGS is a
        # space-separated positional tail (samples, doctrine, trace).
        $playArgList = $PlayArgs -split " "
        poetry run python @py -m $Module $Port "$Catalogue" "$TypeDump" @playArgList
    }
    finally {
        Stop-Process -Id $game.Id -Force -ErrorAction SilentlyContinue
        Write-Host "[play] game stopped" -ForegroundColor DarkGray
    }
}
finally {
    if (Test-Path $classesDir) { Remove-Item -Recurse -Force $classesDir }
    Start-Sleep -Milliseconds 500
    if ($Tree -eq "" -and (Test-Path $agentJar)) {
        Remove-Item -Force $agentJar -ErrorAction SilentlyContinue
    }
}
