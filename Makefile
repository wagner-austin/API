# =============================================================================
# Root Makefile - API Monorepo
# =============================================================================
# Orchestrates shared infrastructure and services
#
# Usage:
#   make infra              # Start Redis + network
#   make up-databank        # Start Redis + data-bank-api
#   make up-trainer         # Start Redis + Model-Trainer
#   make up-all             # Start Redis + all services
#   make down               # Stop everything
#   make status             # Show running containers
#   make logs               # Tail logs from all containers
# =============================================================================

SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: infra up-databank up-trainer up-handwriting up-qr up-transcript up-turkic up-music up-discord up-all down status logs clean check-all test-all lint-all

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
	Set-Location services/Model-Trainer; docker compose up -d --build

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
	$$dirs = @("services/data-bank-api", "services/Model-Trainer", "services/handwriting-ai", "services/qr-api", "services/transcript-api", "services/turkic-api", "services/music-wrapped-api", "clients/DiscordBot"); foreach ($$d in $$dirs) { if (Test-Path "$$d/docker-compose.yml") { Set-Location $$d; docker compose down; Set-Location ../.. } }; docker compose down

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
# Development (run checks across all libs/services)
# ---------------------------------------------------------------------------
check-all:
	$$dirs = Get-ChildItem -Path libs,services,clients -Directory | Where-Object { Test-Path "$$($_.FullName)/Makefile" }; foreach ($$d in $$dirs) { Write-Host "`n=== Checking $$($_.Name) ===" -ForegroundColor Cyan; Set-Location $$d.FullName; make check; Set-Location $$PSScriptRoot }

lint-all:
	$$dirs = Get-ChildItem -Path libs,services,clients -Directory | Where-Object { Test-Path "$$($_.FullName)/Makefile" }; foreach ($$d in $$dirs) { Write-Host "`n=== Linting $$($_.Name) ===" -ForegroundColor Cyan; Set-Location $$d.FullName; make lint; Set-Location $$PSScriptRoot }

test-all:
	$$dirs = Get-ChildItem -Path libs,services,clients -Directory | Where-Object { Test-Path "$$($_.FullName)/Makefile" }; foreach ($$d in $$dirs) { Write-Host "`n=== Testing $$($_.Name) ===" -ForegroundColor Cyan; Set-Location $$d.FullName; make test; Set-Location $$PSScriptRoot }
