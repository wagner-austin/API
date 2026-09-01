SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: lint test check

lint:
	@$$ErrorActionPreference = 'SilentlyContinue'; poetry run mypy --version | Out-Null; if (-not $$?) { Write-Host "[lint] Stale venv detected; removing..." -ForegroundColor Yellow; if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" } }
	poetry lock
	poetry sync --with dev
	# First-party wheels, rebuilt fresh and installed --no-deps every lint.
	# NOT poetry path-dependencies, deliberately: platform-ml -> shap 0.50.0
	# pins llvmlite to a PyPI-deleted beta, so the path-dep chain cannot
	# solve on a fresh machine (2026-09-01). Rebuilding here keeps the wheels
	# honest against live sibling source; --no-deps keeps the dead pin out;
	# the third-party closure lives in pyproject where the solver can see it.
	foreach ($$p in @('..\\..\\libs\\platform_core','..\\..\\libs\\platform_ml','..\\..\\libs\\platform_workers','..\\..\\services\\Model-Trainer')) { Push-Location $$p; poetry build --quiet; $$code = $$LASTEXITCODE; Pop-Location; if ($$code -ne 0) { exit $$code } }
	poetry run pip install --quiet --force-reinstall --no-deps ..\\..\\libs\\platform_core\\dist\\platform_core-0.1.0-py3-none-any.whl ..\\..\\libs\\platform_ml\\dist\\platform_ml-0.1.0-py3-none-any.whl ..\\..\\libs\\platform_workers\\dist\\platform_workers-0.1.0-py3-none-any.whl ..\\..\\services\\Model-Trainer\\dist\\model_trainer_server-0.1.0-py3-none-any.whl; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE }
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
	# poetry sync REMOVES anything not in the lock -- including the
	# first-party wheels lint just installed -- so the wheels go back in
	# here too. Same reasoning as the lint block above.
	poetry run pip install --quiet --force-reinstall --no-deps ..\\..\\libs\\platform_core\\dist\\platform_core-0.1.0-py3-none-any.whl ..\\..\\libs\\platform_ml\\dist\\platform_ml-0.1.0-py3-none-any.whl ..\\..\\libs\\platform_workers\\dist\\platform_workers-0.1.0-py3-none-any.whl ..\\..\\services\\Model-Trainer\\dist\\model_trainer_server-0.1.0-py3-none-any.whl; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE }
	# NOT the shared run-tests.ps1, deliberately: it hardcodes `-n auto`,
	# and this suite is SERIAL by design -- xdist workers would each spin a
	# CUDA context on the one GPU. What is kept from the launcher: coverage
	# to a per-run path so overlapping sessions cannot corrupt one SQLite
	# file. What is safely dropped: the xdist-wreck reaper, which exists for
	# stranded execnet workers a serial run cannot produce.
	if (-not (Test-Path 'runs')) { New-Item -ItemType Directory -Path 'runs' | Out-Null }; $$env:COVERAGE_FILE = (Resolve-Path '.').Path + '\\runs\\.coverage-' + [System.Guid]::NewGuid().ToString('N').Substring(0, 8); poetry run pytest --cov=src --cov=scripts --cov-branch --cov-report=term-missing; $$code = $$LASTEXITCODE; Remove-Item -Force ($$env:COVERAGE_FILE + '*') -ErrorAction SilentlyContinue; if ($$code -ne 0) { exit $$code }

check: lint | test

