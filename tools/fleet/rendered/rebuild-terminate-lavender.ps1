param(
    [string]$Wsl = "$env:SystemRoot\System32\wsl.exe"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
& $Wsl --terminate 'Ubuntu'
if ($LASTEXITCODE -ne 0) {
    throw "FLEET_TERMINATE_REFUSED: $Wsl --terminate Ubuntu exited $LASTEXITCODE"
}
