SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: help install lint test check

help:
	@Write-Host ""
	@Write-Host "NavProbe -- available targets" -ForegroundColor Cyan
	@Write-Host "=============================" -ForegroundColor Cyan
	@Write-Host ""
	@Write-Host "Code health:" -ForegroundColor Yellow
	@Write-Host "  make check                       lint + test (the gate)"
	@Write-Host "  make lint                        guard + ruff + mypy over src, tests, scripts"
	@Write-Host "  make test                        pytest + coverage (100% statements and branches)"
	@Write-Host "  make install                     poetry install"
	@Write-Host ""

install:
	@Write-Host ""
	@Write-Host "==> install" -ForegroundColor Cyan
	poetry lock
	poetry install --with dev
	@Write-Host "[install] done" -ForegroundColor Green

lint:
	@Write-Host ""
	@Write-Host "==> lint" -ForegroundColor Cyan
	@$$ErrorActionPreference = 'SilentlyContinue'; poetry run mypy --version | Out-Null; if (-not $$?) { Write-Host "[lint] Stale venv detected; removing..." -ForegroundColor Yellow; if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" } }
	poetry lock
	poetry install --with dev
	@Write-Host ""
	@Write-Host "  -> guard" -ForegroundColor DarkCyan
	poetry run python -m scripts.guard; if ($$LASTEXITCODE -ne 0) { Write-Host "[lint] guard failed" -ForegroundColor Red; exit $$LASTEXITCODE }
	@Write-Host ""
	@Write-Host "  -> ruff check + format" -ForegroundColor DarkCyan
	poetry run ruff check . --fix
	poetry run ruff format .
	@Write-Host ""
	@Write-Host "  -> mypy" -ForegroundColor DarkCyan
	poetry run mypy src tests scripts
	@Write-Host ""
	@Write-Host "[lint] passed" -ForegroundColor Green

# Coverage data goes to a per-run path, not the default fixed ``.coverage``.
#
# Coverage writes one SQLite file at a fixed path, and two concurrent runs
# corrupt it. The reader then fails with "no such table: arc" and reports
# 0.00% while every test passes -- a green suite presented as a coverage
# failure, which reads as a code problem and is not one.
#
# Per-run paths cannot collide, so no lock is needed. Fragments land beside
# the combined file: xdist writes ``.coverage-<id>.<host>.<pid>.<rand>`` per
# worker and coverage combines them, so cleanup globs rather than naming one
# file. Under runs/ because that is where untracked run output already lives.
test:
	@Write-Host ""
	@Write-Host "==> test" -ForegroundColor Cyan
	poetry lock
	poetry install --with dev
	New-Item -ItemType Directory -Force "runs" | Out-Null
	$$covToken = ".coverage-" + [System.Guid]::NewGuid().ToString("N").Substring(0,8); $$env:COVERAGE_FILE = (Resolve-Path -LiteralPath ".").Path + "\runs\" + $$covToken; $$code = 0; try { poetry run python -m pytest --cov=src --cov=scripts --cov-branch --cov-report=term-missing; $$code = $$LASTEXITCODE } finally { Get-ChildItem -Path "runs" -Filter "$$covToken*" -Force -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue; Remove-Item Env:\COVERAGE_FILE -ErrorAction SilentlyContinue }; if ($$code -ne 0) { exit $$code }
	@Write-Host ""
	@Write-Host "[test] passed" -ForegroundColor Green

check: lint | test
	@Write-Host ""
	@Write-Host "=== ALL CHECKS PASSED ===" -ForegroundColor Green
	@Write-Host ""
