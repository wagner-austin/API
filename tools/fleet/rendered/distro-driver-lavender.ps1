param(
    [string]$Distro = 'Ubuntu',
    [string]$Payload = '/mnt/c/fleet/stage/fleet-rebuild-linux-base.sh',
    [string]$Wsl = "$env:SystemRoot\System32\wsl.exe"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$env:WSL_UTF8 = '1'
& $Wsl -d $Distro -u root -- bash $Payload
if ($LASTEXITCODE -ne 0) {
    throw "FLEET_DISTRO_PAYLOAD_FAILED: bash $Payload in $Distro exited $LASTEXITCODE"
}
