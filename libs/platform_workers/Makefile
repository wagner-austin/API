SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: lint test check

lint:
	@$$ErrorActionPreference = 'SilentlyContinue'; poetry run mypy --version | Out-Null; if (-not $$?) { Write-Host "[lint] Stale venv detected; removing..." -ForegroundColor Yellow; if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" } }
	poetry lock
	poetry sync --with dev
	# Run guard scripts if present (prefer module invocation)
	if ((Test-Path ".\\scripts\\guard.py") -or (Test-Path ".\\scripts\\guard\\__main__.py")) { poetry run python -m scripts.guard; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE } }
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy src tests scripts

# Coverage data goes to a per-run path, not the default fixed ``.coverage``.
#
# Coverage writes one SQLite file at a fixed path, and two runs that overlap
# corrupt it. The reader then reports a spurious failure on a run whose tests
# all passed, which reads as a code problem and is not one. Several sessions
# work this repo at once, so overlapping runs are the normal case rather than
# the exception.
#
# Per-run paths cannot collide, so no lock is needed. Fragments land beside
# the combined file: xdist writes ``.coverage-<id>.<host>.<pid>.<rand>`` per
# worker and coverage combines them, so cleanup globs rather than naming one
# file. Under runs/ because that is gitignored (``**/runs/``) and is where
# untracked run output already belongs.
test:
	poetry lock
	poetry sync --with dev
	& "..\\..\\scripts\\run-tests.ps1"; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE }

check: lint | test
