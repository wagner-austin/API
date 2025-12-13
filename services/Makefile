SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: lint test check

lint:
	$$root = Get-Location; $$dirs = @(); foreach ($$d in Get-ChildItem -Path . -Directory) { if (Test-Path (Join-Path $$d.FullName "Makefile")) { $$dirs += $$d } }; $$failed = @(); foreach ($$d in $$dirs) { Write-Host "`n=== Linting $$($$d.Name) ===" -ForegroundColor Cyan; Set-Location $$d.FullName; make lint; if ($$LASTEXITCODE -ne 0) { $$failed += $$d.Name }; Set-Location $$root }; if ($$failed.Count -gt 0) { Write-Host "`nFailed: $$($$failed -join ', ')" -ForegroundColor Red; exit 1 } else { Write-Host "`nAll lint passed" -ForegroundColor Green }

test:
	$$root = Get-Location; $$dirs = @(); foreach ($$d in Get-ChildItem -Path . -Directory) { if (Test-Path (Join-Path $$d.FullName "Makefile")) { $$dirs += $$d } }; $$failed = @(); foreach ($$d in $$dirs) { Write-Host "`n=== Testing $$($$d.Name) ===" -ForegroundColor Cyan; Set-Location $$d.FullName; make test; if ($$LASTEXITCODE -ne 0) { $$failed += $$d.Name }; Set-Location $$root }; if ($$failed.Count -gt 0) { Write-Host "`nFailed: $$($$failed -join ', ')" -ForegroundColor Red; exit 1 } else { Write-Host "`nAll tests passed" -ForegroundColor Green }

check:
	$$root = Get-Location; $$dirs = @(); foreach ($$d in Get-ChildItem -Path . -Directory) { if (Test-Path (Join-Path $$d.FullName "Makefile")) { $$dirs += $$d } }; $$failed = @(); foreach ($$d in $$dirs) { Write-Host "`n=== Checking $$($$d.Name) ===" -ForegroundColor Cyan; Set-Location $$d.FullName; make check; if ($$LASTEXITCODE -ne 0) { $$failed += $$d.Name }; Set-Location $$root }; if ($$failed.Count -gt 0) { Write-Host "`nFailed: $$($$failed -join ', ')" -ForegroundColor Red; exit 1 } else { Write-Host "`nAll checks passed" -ForegroundColor Green }
