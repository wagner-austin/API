SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: lint test check

# Lint: venv cleanup, guards, Ruff, Mypy
lint:
	# Clean stale venv if mypy not runnable; do not fail
	@$$ErrorActionPreference = 'SilentlyContinue'; poetry run mypy --version | Out-Null; if (-not $$?) { Write-Host "[lint] Stale venv detected; removing..." -ForegroundColor Yellow; if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" } }
	# Ensure dependencies (including type hints) are installed before type checking
	poetry lock
	poetry install --with dev
	# Run guard scripts if present (prefer module invocation)
	if ((Test-Path ".\\scripts\\guard.py") -or (Test-Path ".\\scripts\\guard\\__main__.py")) { poetry run python -m scripts.guard; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE } }
	# Ruff + Mypy
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy src tests scripts

# Test: install deps, then pytest with branch+statement coverage
test:
	poetry lock
	poetry install --with dev
	$$covArgs = @("--cov-branch","--cov-report=term-missing"); $$cands = @("src","scripts"); foreach ($$c in $$cands) { if (Test-Path (Join-Path "." $$c)) { $$covArgs += "--cov=$$c" } }; poetry run pytest -n auto -v @covArgs

# Check: run lint then test
check: lint | test

