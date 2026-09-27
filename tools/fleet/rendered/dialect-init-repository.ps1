param(
    [string]$Git = 'git'
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
Invoke-Step $Git @('-C', 'C:/fleet/stage/MCPs-packages-maketools-1790000000', 'init', '--quiet')
Invoke-Step $Git @('-C', 'C:/fleet/stage/MCPs-packages-maketools-1790000000', 'add', '--all')
Invoke-Step $Git @('-C', 'C:/fleet/stage/MCPs-packages-maketools-1790000000', '-c', 'user.name=fleet', '-c', 'user.email=fleet@corvis.invalid', 'commit', '--quiet', '--message', 'fleet export MCPs-packages-maketools-1790000000')
