SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: lint test check up down logs

lint:
	@$$ErrorActionPreference = 'SilentlyContinue'; poetry run mypy --version | Out-Null; if (-not $$?) { Write-Host "[lint] Stale venv detected; removing..." -ForegroundColor Yellow; if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" } }
	poetry lock
	poetry sync --with dev
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
	New-Item -ItemType Directory -Force "runs" | Out-Null
	$$covArgs = @("--cov-branch","--cov-report=term-missing"); $$cands = @("src","scripts"); foreach ($$c in $$cands) { if (Test-Path (Join-Path "." $$c)) { $$covArgs += "--cov=$$c" } }; $$env:COVERAGE_FILE = (Resolve-Path -LiteralPath ".").Path + "\runs\.coverage-" + [System.Guid]::NewGuid().ToString("N").Substring(0,8); $$code = 0; try { poetry run pytest -n auto -v @covArgs; $$code = $$LASTEXITCODE } finally { Get-ChildItem -Path "runs" -Filter ".coverage-*" -Force -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue; Remove-Item Env:\COVERAGE_FILE -ErrorAction SilentlyContinue }; if ($$code -ne 0) { exit $$code }
	Push-Location web; npm run test:unit; Pop-Location

check: lint test

up:
	& .\scripts\start.ps1

down:
	& .\scripts\stop.ps1

logs:
	$$logDir = ".\logs"; if (Test-Path "$$logDir\api.log") { Write-Host "=== API Log ===" -ForegroundColor Cyan; Get-Content "$$logDir\api.log" -Tail 50 }; if (Test-Path "$$logDir\web.log") { Write-Host "`n=== Web Log ===" -ForegroundColor Cyan; Get-Content "$$logDir\web.log" -Tail 20 }
