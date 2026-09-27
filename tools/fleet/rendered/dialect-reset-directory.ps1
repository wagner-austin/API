param(
    [string]$Directory = 'C:/fleet/stage/MCPs'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Directory) {
    Remove-Item -Recurse -Force -LiteralPath $Directory
}
[IO.Directory]::CreateDirectory($Directory) | Out-Null
