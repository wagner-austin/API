param(
    [string]$Distro = 'Ubuntu',
    [string]$Scratch = 'C:/fleet/stage',
    [string]$DistroDir = 'C:/wsl/Ubuntu',
    [string]$RootfsVersion = '24.04-20240423',
    [string]$RootfsUrl = 'https://cloud-images.ubuntu.com/wsl/releases/24.04/20240423/ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz',
    [string]$RootfsSha256 = '8251e27ffff381a4af5f41dcb94d867de3e0d9774a9241908ab34555d99315ea',
    [string]$RootfsFile = 'ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz',
    [string]$LxssKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss',
    [string]$Wsl = "$env:SystemRoot\System32\wsl.exe"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ProgressPreference = 'SilentlyContinue'
$registered = @()
if (Test-Path -LiteralPath $LxssKey) {
    $registered = @(Get-ChildItem -LiteralPath $LxssKey | ForEach-Object { [string]$_.GetValue('DistributionName') })
}
if ($registered -notcontains $Distro) {
    [void][System.IO.Directory]::CreateDirectory($Scratch)
    $image = Join-Path $Scratch $RootfsFile
    if (-not (Test-Path -LiteralPath $image)) {
        Invoke-WebRequest -Uri $RootfsUrl -OutFile $image -UseBasicParsing
    }
    $Digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $image).Hash.ToLower()
    if ($Digest -ne $RootfsSha256) {
        Remove-Item -LiteralPath $image
        throw ('rootfs sha256 ' + $Digest + ' does not match the pin ' + $RootfsSha256)
    }
    [void][System.IO.Directory]::CreateDirectory($DistroDir)
    & $Wsl --import $Distro $DistroDir $image --version 2
    if ($LASTEXITCODE -ne 0) {
        throw ('wsl --import ' + $Distro + ' exited ' + $LASTEXITCODE)
    }
    Remove-Item -LiteralPath $image
    Write-Output ('imported ' + $Distro + ' ' + $RootfsVersion)
}
