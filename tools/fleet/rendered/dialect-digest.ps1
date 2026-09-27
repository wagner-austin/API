param(
    [string]$Target = 'C:/fleet/stage/MCPs-packages-maketools-1790000000'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
(Get-FileHash -Algorithm SHA256 -LiteralPath "$Target/tree.tgz").Hash.ToLower()
