# Launch the headless game with the agent attached, for the probe family.
#
# One launcher for sandbox-probe, discover-probe, wire-capture and type-flags:
# the four differ only in the agent's argument string and where output lands,
# which is exactly two parameters. `{OUT}` inside -AgentArgs is replaced with
# the absolute form of -Out, because the game runs from its own directory and
# the agent writes where it is told, not where the harness stands.

param(
    [Parameter(Mandatory = $true)][string]$GameDir,
    [Parameter(Mandatory = $true)][string]$AgentJar,
    [string]$AgentArgs = "",
    [string]$Out = "",
    [Parameter(Mandatory = $true)][string]$Log
)

$root = (Resolve-Path -LiteralPath ".").Path
$agent = "$root\$AgentJar"
$logAbs = "$root\$Log"
if ($Out -ne "") {
    $AgentArgs = $AgentArgs.Replace("{OUT}", "$root\$Out")
}
$javaAgent = if ($AgentArgs -ne "") { "-javaagent:$agent=$AgentArgs" } else { "-javaagent:$agent" }

Push-Location $GameDir
try {
    & ".\jvm64\bin\java.exe" -Xmx1000M "-Djava.library.path=." $javaAgent `
        -cp "game-lib.jar;libs/*" com.corrodinggames.rts.java.Main `
        -nodisplay -nosound -sandbox -width 800 -height 600 -log $logAbs
}
finally {
    Pop-Location
}
