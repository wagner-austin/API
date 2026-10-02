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
Invoke-Step $Git @('-C', 'C:/fleet/stage/MCPs', 'config', 'gc.auto', '0')
Invoke-Step $Git @('-C', 'C:/fleet/stage/MCPs', 'fetch', '--quiet', '--no-tags', 'C:/fleet/stage/MCPs.stage/tree.tgz', '+refs/fleet/companion:refs/remotes/origin/main')
Invoke-Step $Git @('-C', 'C:/fleet/stage/MCPs', 'checkout', '--quiet', '-B', 'main', 'refs/remotes/origin/main')
Invoke-Step $Git @('-C', 'C:/fleet/stage/MCPs', 'merge-base', '--is-ancestor', 'HEAD', '5f389cb3d9bdd2e9b49e8df6683a6fee71a359b2')
Invoke-Step $Git @('-C', 'C:/fleet/stage/MCPs', 'merge-base', '--is-ancestor', '5f389cb3d9bdd2e9b49e8df6683a6fee71a359b2', 'HEAD')
