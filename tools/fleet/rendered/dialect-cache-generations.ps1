param(
    [string]$StageRoot = 'C:/fleet/stage'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$tools = "$StageRoot/cache/tools"
$currentFile = "$tools/current"
if (-not (Test-Path -LiteralPath $currentFile)) {
    "cache-generations: ${tools}: none yet"
    return
}
$current = (Get-Content -LiteralPath $currentFile -Raw).Trim()
$now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$used = @(foreach ($run in @(Get-ChildItem -LiteralPath $StageRoot -Directory -Force)) {
    $marker = Join-Path $run.FullName 'cache-generation'
    if (Test-Path -LiteralPath $marker) {
        (Get-Content -LiteralPath $marker -Raw).Trim()
    }
})
$total = 0
$removed = 0
$bytes = [long]0
foreach ($entry in @(Get-ChildItem -LiteralPath $tools -Force)) {
    if ($entry.Name -eq 'current') {
        continue
    }
    if (-not $entry.PSIsContainer -or $entry.Name -notmatch '^[0-9]+$') {
        throw "cache-generations: $($entry.FullName) is no generation"
    }
    $total++
    if ($entry.Name -eq $current -or $used -contains $entry.Name) {
        continue
    }
    $newest = [math]::Max([long]$entry.Name, [long]$current)
    if ($now - $newest -le 3600) {
        continue
    }
    $size = @(Get-ChildItem -LiteralPath $entry.FullName -Recurse -File -Force) | Measure-Object -Property Length -Sum
    Remove-Item -Recurse -Force -LiteralPath $entry.FullName
    $removed++
    $bytes += [long]$size.Sum
}
'cache-generations: {0}: generation {1} current, removed {2} of {3} ({4} MB)' -f $tools, $current, $removed, $total, [math]::Floor($bytes / 1MB)
