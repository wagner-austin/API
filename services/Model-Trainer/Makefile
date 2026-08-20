SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: lint test check

lint:
	@$$ErrorActionPreference = 'SilentlyContinue'; poetry run mypy --version | Out-Null; if (-not $$?) { Write-Host "[lint] Stale venv detected; removing..." -ForegroundColor Yellow; if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" } }
	poetry lock
	poetry sync --with dev
	if ((Test-Path ".\\scripts\\guard.py") -or (Test-Path ".\\scripts\\guard\\__main__.py")) { poetry run python -m scripts.guard; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE } }
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy src tests scripts

# Test running lives in the shared launcher, ../../scripts/run-tests.ps1, so it is
# defined once instead of being copy-pasted into 36 Makefiles. That duplication is
# why a process-lifecycle bug had to be fixed 36 times rather than once.
#
# The launcher keeps the per-run coverage path this recipe always had (coverage
# writes one SQLite file at a fixed path, and two overlapping runs corrupt it,
# which surfaces as a spurious failure on a run whose tests all passed; several
# sessions work this repo at once, so overlap is normal). It adds a pre-run sweep
# and a kill-on-close job object so a run that dies cannot strand its xdist
# workers -- see the header of run-tests.ps1 for the 2026-08-19 incident that
# motivated it.
test:
	poetry lock
	poetry sync --with dev
	& "..\\..\\scripts\\run-tests.ps1"; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE }

check: lint | test

