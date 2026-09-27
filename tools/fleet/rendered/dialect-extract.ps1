param(
    [string]$Tar = "$env:SystemRoot\System32\tar.exe"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
function Invoke-Step {
    param([string]$Tool, [string[]]$Arguments)
    & $Tool @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
Invoke-Step $Tar @('-xzmf', 'C:/fleet/stage/MCPs-packages-maketools-1790000000/tree.tgz', '-C', 'C:/fleet/stage/MCPs-packages-maketools-1790000000')
