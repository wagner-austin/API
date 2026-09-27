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
Invoke-Step $Git @('-C', 'C:/fleet/stage/MCPs', 'init', '--quiet')
Invoke-Step $Git @('-C', 'C:/fleet/stage/MCPs', 'add', '--all', '--force')
Invoke-Step $Git @('-C', 'C:/fleet/stage/MCPs', '-c', 'user.name=fleet', '-c', 'user.email=fleet@corvis.invalid', 'commit', '--quiet', '--message', 'fleet companion export 5f389cb3d9bdd2e9b49e8df6683a6fee71a359b2')
