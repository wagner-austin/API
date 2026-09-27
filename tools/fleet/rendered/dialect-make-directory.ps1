param(
    [string]$Directory = 'C:/fleet/stage/MCPs-packages-maketools-1790000000'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[IO.Directory]::CreateDirectory($Directory) | Out-Null
