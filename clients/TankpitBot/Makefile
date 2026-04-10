SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command
OUTPUT ?= capture_session.json

.PHONY: lint test check install sniff probe teleport-probe teleport-probe-safe teleport-probe-aggressive teleport-probe-full enemy-teleport-probe enemy-teleport-probe-map enemy-teleport-probe-nearest fuel-probe bot decode discover analyze-viewport

install:
	poetry lock
	poetry install --with dev
	poetry run playwright install chromium

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

sniff: install
	$$env:TANKPIT_OUTPUT = '$(OUTPUT)'; poetry run tankpit-sniff

probe: install
	poetry run tankpit-probe

teleport-probe: teleport-probe-safe teleport-probe-aggressive

teleport-probe-safe: install
	Write-Host "[teleport-probe] safe strategy: sync_before_teleport"
	poetry run tankpit-teleport-probe --max-targets 3 --teleport-strategy sync_before_teleport

teleport-probe-aggressive: install
	Write-Host "[teleport-probe] aggressive strategy: immediate_after_map_open"
	poetry run tankpit-teleport-probe --max-targets 3 --teleport-strategy immediate_after_map_open

teleport-probe-full: install
	poetry run tankpit-teleport-probe

enemy-teleport-probe: install
	poetry run tankpit-enemy-teleport-probe --max-attempts 3 --acquisition-strategy map_open

enemy-teleport-probe-map: install
	poetry run tankpit-enemy-teleport-probe --max-attempts 3 --acquisition-strategy map_open

enemy-teleport-probe-nearest: install
	poetry run tankpit-enemy-teleport-probe --max-attempts 3 --acquisition-strategy nearest_enemy

fuel-probe: install
	poetry run tankpit-fuel-probe --target-pickups 3 --max-attempts 9

bot: install
	poetry run tankpit-bot

decode:
	poetry run python -m scripts.decode

discover:
	poetry run python discover_commands.py

analyze-viewport:
	poetry run python -m scripts.analyze_viewport
