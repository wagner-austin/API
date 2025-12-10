SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: infra up-databank up-trainer up-handwriting up-qr up-transcript up-turkic up-music up-discord up-all down clean status logs lint test check

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
infra:
	docker compose up -d

# ---------------------------------------------------------------------------
# Individual Services (each starts infra first)
# ---------------------------------------------------------------------------
up-databank: infra
	Set-Location services/data-bank-api; docker compose up -d --build

up-trainer: infra
	Set-Location services/Model-Trainer; docker compose build --progress plain; docker compose up -d

up-handwriting: infra
	Set-Location services/handwriting-ai; docker compose up -d --build

up-qr: infra
	Set-Location services/qr-api; docker compose up -d --build

up-transcript: infra
	Set-Location services/transcript-api; docker compose up -d --build

up-turkic: infra
	Set-Location services/turkic-api; docker compose up -d --build

up-music: infra
	Set-Location services/music-wrapped-api; docker compose up -d --build

up-discord: infra
	Set-Location clients/DiscordBot; docker compose up -d --build

# ---------------------------------------------------------------------------
# All Services
# ---------------------------------------------------------------------------
up-all: infra up-databank up-trainer up-handwriting up-qr up-transcript up-turkic up-music up-discord
	Write-Host "All services started" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Stop/Cleanup
# ---------------------------------------------------------------------------
down:
	$$dirs = @("services/data-bank-api", "services/Model-Trainer", "services/handwriting-ai", "services/qr-api", "services/transcript-api", "services/turkic-api", "services/music-wrapped-api", "clients/DiscordBot"); foreach ($$d in $$dirs) { if (Test-Path "$$d/docker-compose.yml") { Push-Location $$d; docker compose down; Pop-Location } }; docker compose down

clean: down
	docker system prune -f
	docker volume prune -f

# ---------------------------------------------------------------------------
# Status/Logs
# ---------------------------------------------------------------------------
status:
	docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

logs:
	docker compose logs -f

# ---------------------------------------------------------------------------
# Development: lint, test, check across all libs/services/clients
# ---------------------------------------------------------------------------
lint:
	$$root = Get-Location; $$dirs = @(); foreach ($$p in @("libs","services","clients")) { foreach ($$d in Get-ChildItem -Path $$p -Directory) { if (Test-Path (Join-Path $$d.FullName "Makefile")) { $$dirs += $$d } } }; $$failed = @(); foreach ($$d in $$dirs) { Write-Host "`n=== Linting $$d.Name ===" -ForegroundColor Cyan; Set-Location $$d.FullName; make lint; if ($$LASTEXITCODE -ne 0) { $$failed += $$d.Name }; Set-Location $$root }; if ($$failed.Count -gt 0) { Write-Host "`nFailed: $$($$failed -join ', ')" -ForegroundColor Red; exit 1 } else { Write-Host "`nAll lint passed" -ForegroundColor Green }

test:
	$$root = Get-Location; $$dirs = @(); foreach ($$p in @("libs","services","clients")) { foreach ($$d in Get-ChildItem -Path $$p -Directory) { if (Test-Path (Join-Path $$d.FullName "Makefile")) { $$dirs += $$d } } }; $$failed = @(); foreach ($$d in $$dirs) { Write-Host "`n=== Testing $$d.Name ===" -ForegroundColor Cyan; Set-Location $$d.FullName; make test; if ($$LASTEXITCODE -ne 0) { $$failed += $$d.Name }; Set-Location $$root }; if ($$failed.Count -gt 0) { Write-Host "`nFailed: $$($$failed -join ', ')" -ForegroundColor Red; exit 1 } else { Write-Host "`nAll tests passed" -ForegroundColor Green }

check:
	$$root = Get-Location; $$dirs = @(); foreach ($$p in @("libs","services","clients")) { foreach ($$d in Get-ChildItem -Path $$p -Directory) { if (Test-Path (Join-Path $$d.FullName "Makefile")) { $$dirs += $$d } } }; $$failed = @(); foreach ($$d in $$dirs) { Write-Host "`n=== Checking $$d.Name ===" -ForegroundColor Cyan; Set-Location $$d.FullName; make check; if ($$LASTEXITCODE -ne 0) { $$failed += $$d.Name }; Set-Location $$root }; if ($$failed.Count -gt 0) { Write-Host "`nFailed: $$($$failed -join ', ')" -ForegroundColor Red; exit 1 } else { Write-Host "`nAll checks passed" -ForegroundColor Green }
