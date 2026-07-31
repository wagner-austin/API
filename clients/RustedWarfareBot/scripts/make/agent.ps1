# Build the agent jar, atomically.
#
# The body of the Makefile `agent` recipe. Two subtleties both live in the
# Move-Item: the jar is assembled under a temporary name so a half-written
# build can never be the jar a game attaches, and the final rename is
# promoted to a terminating error (-ErrorAction Stop) because a jar held
# open by a running JVM fails the move NON-terminatingly otherwise --
# PowerShell prints the error, skips the catch, and the target reports a
# successful build over a jar it never replaced. Observed live.

param(
    [Parameter(Mandatory = $true)][string]$Javac,
    [Parameter(Mandatory = $true)][string]$Jar,
    [Parameter(Mandatory = $true)][string]$AgentJar
)

$stamp = [System.Guid]::NewGuid().ToString("N").Substring(0, 8)
$classesDir = "agent/build/classes-$stamp"
$tmpJar = "$AgentJar.$stamp.new"
$failed = ""
try {
    New-Item -ItemType Directory -Force $classesDir | Out-Null
    & $Javac --release 11 -Xlint:all -Werror -d $classesDir (
        Get-ChildItem "agent/src/rwbot/agent/*.java" | ForEach-Object { $_.FullName })
    if ($LASTEXITCODE -ne 0) {
        $failed = "javac failed"
    }
    else {
        & $Jar cfm $tmpJar agent/manifest.mf -C $classesDir .
        if ($LASTEXITCODE -ne 0) {
            $failed = "jar failed"
        }
        else {
            try {
                Move-Item -Force -ErrorAction Stop -LiteralPath $tmpJar -Destination $AgentJar
            }
            catch {
                $failed = "cannot replace ${AgentJar}: a JVM has it attached with -javaagent. " +
                    "Stop the running game, then retry."
            }
        }
    }
}
finally {
    if (Test-Path $classesDir) { Remove-Item -Recurse -Force $classesDir }
    if (Test-Path $tmpJar) { Remove-Item -Force $tmpJar }
}
if ($failed) { Write-Host "[agent] $failed" -ForegroundColor Red; exit 1 }
