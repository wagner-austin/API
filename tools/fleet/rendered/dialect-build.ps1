param(
    [string]$Target = 'C:/fleet/stage/MCPs-packages-maketools-1790000000',
    [string]$Recipe = 'C:/fleet/stage/MCPs-packages-maketools-1790000000/packages/maketools',
    [int]$Workers = 4,
    [string]$CacheRoot = 'C:/fleet/stage/cache',
    [string[]]$Install = @('npm ci'),
    [string]$Make = 'make',
    [string]$Cmd = "$env:SystemRoot\System32\cmd.exe"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PID | Set-Content -LiteralPath "$Target/build.pid"
$log = "$Target/result.txt.log"
$result = "$Target/result.txt"
$env:npm_config_cache = "$CacheRoot/npm"
$env:POETRY_CACHE_DIR = "$CacheRoot/pypoetry"
$env:PLAYWRIGHT_BROWSERS_PATH = "$CacheRoot/ms-playwright"
$env:PYTEST_XDIST_AUTO_NUM_WORKERS = "$Workers"
function Invoke-Logged {
    param([string]$Shell, [string]$Command)
    & $Shell /d /s /c "$Command >> `"$log`" 2>&1"
    return $LASTEXITCODE
}
Set-Location -LiteralPath $Target
$status = 0
foreach ($step in $Install) {
    if ($status -eq 0) {
        [System.IO.File]::AppendAllText($log, "`$ $step`r`n")
        $status = Invoke-Logged $Cmd $step
    }
}
if ($status -eq 0) {
    Set-Location -LiteralPath $Recipe
    $status = Invoke-Logged $Cmd "`"$Make`" check"
}
$status | Set-Content -LiteralPath $result
exit 0
