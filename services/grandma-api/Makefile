SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: lint test check up down logs

lint:
	@$$ErrorActionPreference = 'SilentlyContinue'; poetry run mypy --version | Out-Null; if (-not $$?) { Write-Host "[lint] Stale venv detected; removing..." -ForegroundColor Yellow; if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" } }
	poetry lock
	poetry install --with dev
	if ((Test-Path ".\\scripts\\guard.py") -or (Test-Path ".\\scripts\\guard\\__main__.py")) { poetry run python -m scripts.guard; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE } }
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy src tests scripts

test:
	poetry lock
	poetry install --with dev
	$$covArgs = @("--cov-branch","--cov-report=term-missing"); $$cands = @("src","scripts"); foreach ($$c in $$cands) { if (Test-Path (Join-Path "." $$c)) { $$covArgs += "--cov=$$c" } }; poetry run pytest -n auto -v @covArgs

check: lint | test

up:
	& .\scripts\start.ps1

down:
	& .\scripts\stop.ps1

logs:
	$$logDir = ".\logs"; if (Test-Path "$$logDir\api.log") { Write-Host "=== API Log ===" -ForegroundColor Cyan; Get-Content "$$logDir\api.log" -Tail 50 }; if (Test-Path "$$logDir\web.log") { Write-Host "`n=== Web Log ===" -ForegroundColor Cyan; Get-Content "$$logDir\web.log" -Tail 20 }
