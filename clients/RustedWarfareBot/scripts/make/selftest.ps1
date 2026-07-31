# Compile the agent from source and run its self-checks against the pinned jar.
#
# The body of the Makefile `agent-selftest` recipe. Compiles fresh rather than
# depending on the built jar, deliberately: the gate must be runnable while a
# game holds rw-agent.jar open, and a gate that rebuilt the jar would fail for
# a reason that has nothing to do with the code under test.

param(
    [Parameter(Mandatory = $true)][string]$Javac,
    [Parameter(Mandatory = $true)][string]$Java,
    [Parameter(Mandatory = $true)][string]$GameDir
)

$stamp = [System.Guid]::NewGuid().ToString("N").Substring(0, 8)
$classesDir = "agent/build/verify-$stamp"
$failed = ""
try {
    New-Item -ItemType Directory -Force $classesDir | Out-Null
    & $Javac --release 11 -Xlint:all -Werror -d $classesDir (
        Get-ChildItem "agent/src/rwbot/agent/*.java" | ForEach-Object { $_.FullName })
    if ($LASTEXITCODE -ne 0) {
        $failed = "javac failed"
    }
    else {
        & $Java -cp "$classesDir;$GameDir/game-lib.jar;$GameDir/libs/*" `
            rwbot.agent.SelfTest "$GameDir/game-lib.jar"
        if ($LASTEXITCODE -ne 0) { $failed = "self-checks failed" }
    }
}
finally {
    if (Test-Path $classesDir) { Remove-Item -Recurse -Force $classesDir }
}
if ($failed) { Write-Host "[agent-selftest] $failed" -ForegroundColor Red; exit 1 }
