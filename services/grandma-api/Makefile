include ../../scripts/make/shell.mk

MAKETOOLS := $(PYTHON) ../../tools/maketools/scripts/run.py

.PHONY: lint test check up down logs

lint:
	$(MAKETOOLS) venv-check
	poetry lock
	poetry sync --with dev
	$(MAKETOOLS) guard
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy src tests scripts

# Test running lives in the shared launcher, tools/maketools (the ``test``
# command); see any sibling package's Makefile for what it adds over the
# inline recipe and the incident behind it. The web frontend's unit suite
# runs through its own Makefile.
test:
	poetry lock
	poetry sync --with dev
	$(MAKETOOLS) test
	$(MAKE) -C web test

check: lint test
	@echo "=== ALL CHECKS PASSED ==="

# The host-mode start/stop scripts and the log tail are PowerShell over a
# Windows path and a Windows-only start (scripts/start.ps1 hard-codes the
# hub's checkout); the docker route (`make up-grandma` at the root) is the
# portable one.
ifeq ($(OS),Windows_NT)
up:
	& .\scripts\start.ps1

down:
	& .\scripts\stop.ps1

logs:
	$$logDir = ".\logs"; if (Test-Path "$$logDir\api.log") { Write-Host "=== API Log ===" -ForegroundColor Cyan; Get-Content "$$logDir\api.log" -Tail 50 }; if (Test-Path "$$logDir\web.log") { Write-Host "`n=== Web Log ===" -ForegroundColor Cyan; Get-Content "$$logDir\web.log" -Tail 20 }
else
up:
	@echo "up is Windows-only here (the start script under scripts hard-codes the hub's checkout): use make up-grandma at the root"
	@exit 1

down:
	@echo "down is Windows-only here (the stop script under scripts): use docker compose down in this directory"
	@exit 1

logs:
	@echo "logs is Windows-only here (it tails the host-mode logs/ directory): use docker compose logs -f"
	@exit 1
endif
