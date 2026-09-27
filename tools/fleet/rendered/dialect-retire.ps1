param(
    [string]$Target = 'C:/fleet/stage/MCPs-packages-maketools-1790000000',
    [string]$Log = 'C:/fleet/stage/MCPs-packages-maketools-1790000000/result.txt.log',
    [string]$Retained = 'C:/fleet/stage/logs/MCPs-packages-maketools-1790000000.log',
    [string]$Script0 = 'C:/fleet/stage/mkdir-MCPs-packages-maketools-1790000000.ps1',
    [string]$Script1 = 'C:/fleet/stage/stop-MCPs-packages-maketools-1790000000.ps1',
    [string]$Script2 = 'C:/fleet/stage/retire-MCPs-packages-maketools-1790000000.ps1'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Retained)) | Out-Null
if (Test-Path -LiteralPath $Log) {
    Move-Item -Force -LiteralPath $Log -Destination $Retained
}
if (Test-Path -LiteralPath $Target) {
    Remove-Item -Recurse -Force -LiteralPath $Target
}
foreach ($script in @($Script0, $Script1, $Script2)) {
    if (Test-Path -LiteralPath $script) {
        Remove-Item -Force -LiteralPath $script
    }
}
