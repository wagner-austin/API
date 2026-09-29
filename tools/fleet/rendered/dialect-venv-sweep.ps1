param(
    [string]$Venvs = 'C:/fleet/stage/cache/pypoetry/virtualenvs'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$total = 0
$removed = 0
$bytes = [long]0
if (Test-Path -LiteralPath $Venvs) {
    foreach ($venv in @(Get-ChildItem -LiteralPath $Venvs -Directory -Force)) {
        $total++
        $site = Join-Path $venv.FullName 'Lib\site-packages'
        $sources = @()
        if (Test-Path -LiteralPath $site) {
            foreach ($pth in @(Get-ChildItem -LiteralPath $site -Filter '*.pth' -File -Force)) {
                $sources += @(Get-Content -LiteralPath $pth.FullName | Where-Object { $_ -match '^[A-Za-z]:[\\/]' })
            }
        }
        if ($sources.Count -eq 0) {
            continue
        }
        if (@($sources | Where-Object { Test-Path -LiteralPath $_ }).Count -gt 0) {
            continue
        }
        $size = @(Get-ChildItem -LiteralPath $venv.FullName -Recurse -File -Force) | Measure-Object -Property Length -Sum
        Remove-Item -Recurse -Force -LiteralPath $venv.FullName
        $removed++
        $bytes += [long]$size.Sum
    }
}
'venv-sweep: removed {0} of {1} ({2} MB)' -f $removed, $total, [math]::Floor($bytes / 1MB)
