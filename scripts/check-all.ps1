# Run make check across all packages and display summary table
param([switch]$Verbose = $true)

$root = Get-Location
$results = @()

# Find all directories with Makefiles
$dirs = @()
foreach ($p in @("libs", "services", "clients")) {
    if (Test-Path $p) {
        Get-ChildItem -Path $p -Directory | ForEach-Object {
            if (Test-Path (Join-Path $_.FullName "Makefile")) { $dirs += $_ }
        }
    }
}

Write-Host "`nFound $($dirs.Count) packages to check`n" -ForegroundColor Cyan

# Run checks
foreach ($d in $dirs) {
    Write-Host "=== Checking $($d.Name) ===" -ForegroundColor Cyan
    Set-Location $d.FullName

    $outputLines = @()
    make check 2>&1 | ForEach-Object {
        $line = $_.ToString()
        $outputLines += $line
        if ($Verbose) { Write-Host $line }
    }
    $output = $outputLines -join "`n"

    $status = if ($LASTEXITCODE -eq 0) { "PASS" } else { "FAIL" }

    # Guards - count violations
    $guards = "0"
    if ($output -match "Guard checks passed") {
        $guards = "0"
    } elseif ($output -match "(\d+) violations? found") {
        $guards = $Matches[1]
    } elseif ($output -match "Guard rule summary:") {
        $total = 0
        [regex]::Matches($output, ":\s*(\d+)\s+violations?") | ForEach-Object { $total += [int]$_.Groups[1].Value }
        $guards = "$total"
    }

    # Ruff warnings/errors
    $ruff = "0"
    if ($output -match "Found (\d+) errors?") { $ruff = $Matches[1] }
    elseif ($output -match "(\d+) fixable") { $ruff = $Matches[1] }
    elseif ($output -match "All checks passed") { $ruff = "0" }

    # Mypy
    $mypy = "0"
    if ($output -match "Success: no issues") { $mypy = "0" }
    elseif ($output -match "Found (\d+) error") { $mypy = $Matches[1] }

    # Tests
    $tests = if ($output -match "(\d+) passed") { $Matches[1] } else { "-" }
    $failed = if ($output -match "(\d+) failed") { $Matches[1] } else { "0" }

    # Coverage
    $cov = if ($output -match "TOTAL.*?(\d+)%") { "$($Matches[1])%" } else { "-" }

    $results += [PSCustomObject]@{
        Name     = $d.Name
        Status   = $status
        Guards   = $guards
        Ruff     = $ruff
        Mypy     = $mypy
        Tests    = $tests
        Failed   = $failed
        Coverage = $cov
    }

    Set-Location $root
}

# Summary Table
Write-Host "`n"
Write-Host ("=" * 95) -ForegroundColor Blue
Write-Host "  SUMMARY" -ForegroundColor Blue
Write-Host ("=" * 95) -ForegroundColor Blue
Write-Host ""
Write-Host ("{0,-25} {1,-7} {2,-7} {3,-7} {4,-7} {5,-7} {6,-7} {7,-8}" -f "Package", "Status", "Guards", "Ruff", "Mypy", "Tests", "Failed", "Coverage") -ForegroundColor White
Write-Host ("-" * 95) -ForegroundColor DarkGray

foreach ($r in $results) {
    $statusColor = if ($r.Status -eq "PASS") { "Green" } else { "Red" }
    $guardsColor = if ($r.Guards -eq "0") { "DarkGray" } else { "Red" }
    $ruffColor = if ($r.Ruff -eq "0") { "DarkGray" } else { "Yellow" }
    $mypyColor = if ($r.Mypy -eq "0") { "DarkGray" } else { "Red" }
    $failedColor = if ($r.Failed -eq "0") { "DarkGray" } else { "Red" }
    $covColor = if ($r.Coverage -eq "100%") { "Green" } elseif ($r.Coverage -eq "-") { "DarkGray" } else { "Yellow" }

    Write-Host ("{0,-25} " -f $r.Name) -NoNewline
    Write-Host ("{0,-7} " -f $r.Status) -NoNewline -ForegroundColor $statusColor
    Write-Host ("{0,-7} " -f $r.Guards) -NoNewline -ForegroundColor $guardsColor
    Write-Host ("{0,-7} " -f $r.Ruff) -NoNewline -ForegroundColor $ruffColor
    Write-Host ("{0,-7} " -f $r.Mypy) -NoNewline -ForegroundColor $mypyColor
    Write-Host ("{0,-7} " -f $r.Tests) -NoNewline
    Write-Host ("{0,-7} " -f $r.Failed) -NoNewline -ForegroundColor $failedColor
    Write-Host ("{0,-8}" -f $r.Coverage) -ForegroundColor $covColor
}

Write-Host ("-" * 95) -ForegroundColor DarkGray

# Totals
$passed = ($results | Where-Object { $_.Status -eq "PASS" }).Count
$failedCount = ($results | Where-Object { $_.Status -eq "FAIL" }).Count

Write-Host ""
Write-Host "Total: " -NoNewline
Write-Host "$passed passed" -NoNewline -ForegroundColor Green
Write-Host ", " -NoNewline
Write-Host "$failedCount failed" -ForegroundColor $(if ($failedCount -gt 0) { "Red" } else { "Green" })
Write-Host ""

if ($failedCount -gt 0) { exit 1 }
