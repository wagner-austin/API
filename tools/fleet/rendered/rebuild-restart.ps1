param(
    [string]$Shutdown = "$env:SystemRoot\System32\shutdown.exe"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
& $Shutdown /r /t 10 /c 'fleet-runners --rebuild'
if ($LASTEXITCODE -ne 0) {
    throw "FLEET_RESTART_REFUSED: $Shutdown exited $LASTEXITCODE"
}
