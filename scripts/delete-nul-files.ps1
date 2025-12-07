# Script to delete Windows reserved "NUL" files using WSL
# These files cannot be deleted through normal Windows means

param(
    [string]$Path = "C:\Users\austi\PROJECTS\API",
    [switch]$DryRun
)

Write-Host "Scanning for NUL files in: $Path" -ForegroundColor Cyan

$files = Get-ChildItem -Path $Path -Recurse -Filter 'NUL' -Force -ErrorAction SilentlyContinue

if ($files.Count -eq 0) {
    Write-Host "No NUL files found!" -ForegroundColor Green
    exit 0
}

Write-Host "Found $($files.Count) NUL files to delete" -ForegroundColor Yellow

$deleted = 0
$failed = 0

foreach ($file in $files) {
    $winPath = $file.FullName
    # Convert Windows path to WSL path
    $wslPath = $winPath -replace '^C:', '/mnt/c' -replace '\\', '/'

    if ($DryRun) {
        Write-Host "[DRY RUN] Would delete: $winPath"
    } else {
        $result = wsl rm -f "$wslPath" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $deleted++
            Write-Host "Deleted: $winPath" -ForegroundColor Green
        } else {
            $failed++
            Write-Host "Failed: $winPath - $result" -ForegroundColor Red
        }
    }
}

if (-not $DryRun) {
    Write-Host "`nSummary:" -ForegroundColor Cyan
    Write-Host "  Deleted: $deleted" -ForegroundColor Green
    Write-Host "  Failed: $failed" -ForegroundColor $(if ($failed -gt 0) { 'Red' } else { 'Green' })
}
