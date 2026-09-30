param(
    [string]$Target = 'C:/fleet/stage/MCPs-packages-maketools-1790000000',
    [string]$Staging = 'C:/fleet/stage/MCPs-packages-maketools-1790000000.stage',
    [string]$Log = 'C:/fleet/stage/MCPs-packages-maketools-1790000000/result.txt.log',
    [string]$Retained = 'C:/fleet/stage/logs/MCPs-packages-maketools-1790000000.log',
    [string]$Script0 = 'C:/fleet/stage/mkdir-MCPs-packages-maketools-1790000000.ps1',
    [string]$Script1 = 'C:/fleet/stage/mkdir-MCPs-packages-maketools-1790000000.stage.ps1',
    [string]$Script2 = 'C:/fleet/stage/stop-MCPs-packages-maketools-1790000000.ps1',
    [string]$Script3 = 'C:/fleet/stage/retire-MCPs-packages-maketools-1790000000.ps1'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Retained)) | Out-Null
if (Test-Path -LiteralPath $Log) {
    Move-Item -Force -LiteralPath $Log -Destination $Retained
}
foreach ($directory in @($Target, $Staging)) {
    if (Test-Path -LiteralPath $directory) {
        Remove-Item -Recurse -Force -LiteralPath $directory
    }
}
foreach ($script in @($Script0, $Script1, $Script2, $Script3)) {
    if (Test-Path -LiteralPath $script) {
        Remove-Item -Force -LiteralPath $script
    }
}
