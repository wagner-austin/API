param(
    [string]$Target = 'C:/fleet/stage/MCPs-packages-maketools-1790000000',
    [string]$Recipe = 'C:/fleet/stage/MCPs-packages-maketools-1790000000/packages/maketools',
    [int]$Workers = 4,
    [string]$CacheRoot = 'C:/fleet/stage/cache',
    [string[]]$Install = @('npm ci'),
    [string[]]$InstallPhases = @('install'),
    [string]$Make = 'make',
    [string]$GitBin = "$env:ProgramFiles\Git\bin",
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
$env:CORVIS_FLEET_ELEVATED = '0'
$env:BOARD_AGENT_LABEL = 'opus-example-0929'
$env:CORVIS_FLEET_CACHE = $CacheRoot
$env:PATH = "$GitBin;$env:PATH"
function Invoke-Logged {
    param([string]$Shell, [string]$Command)
    & $Shell /d /s /c "$Command >> `"$log`" 2>&1"
    return $LASTEXITCODE
}
function Get-PhaseStamp {
    return [DateTime]::UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'")
}
function Invoke-Phase {
    param([string]$Shell, [string]$Name, [string]$Command)
    $started = [DateTime]::UtcNow
    $opening = "fleet-phase $Name started $(Get-PhaseStamp)"
    [System.IO.File]::AppendAllText($log, "$opening`r`n")
    $code = Invoke-Logged $Shell $Command
    $seconds = [int][Math]::Floor(([DateTime]::UtcNow - $started).TotalSeconds)
    $closing = "fleet-phase $Name ended $(Get-PhaseStamp) after $seconds s, exit $code"
    [System.IO.File]::AppendAllText($log, "$closing`r`n")
    return $code
}
Set-Location -LiteralPath $Target
$status = 0
for ($index = 0; $index -lt $Install.Count; $index++) {
    if ($status -eq 0) {
        [System.IO.File]::AppendAllText($log, "`$ $($Install[$index])`r`n")
        $status = Invoke-Phase -Shell $Cmd -Name $InstallPhases[$index] -Command $Install[$index]
    }
}
if ($status -eq 0) {
    Set-Location -LiteralPath $Recipe
    $status = Invoke-Phase -Shell $Cmd -Name 'check' -Command "`"$Make`" check"
}
$status | Set-Content -LiteralPath $result
exit 0
