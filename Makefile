SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: infra up-databank up-trainer up-art-trainer up-handwriting up-qr up-transcript up-turkic up-music up-covenant up-grandma up-github-stats up-opportunity up-discord up-all down clean status logs lint test

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

# GIT_COMMIT is exported here rather than left to the operator's shell. The
# Dockerfile bakes it so every manifest and run fingerprint can name the code
# that produced a number, and it defaults to empty -- so a build that forgets
# it records null and nobody notices until the provenance audit. That is not
# hypothetical: every manifest archived by the 2026-08-18 audit has
# git_commit null for exactly this reason.
up-trainer: infra
	$$env:GIT_COMMIT = (git rev-parse HEAD); Set-Location services/Model-Trainer; docker compose build --progress plain; docker compose up -d

up-art-trainer: infra
	Set-Location services/Art-Trainer; docker compose build --progress plain; docker compose up -d

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

up-covenant: infra
	Set-Location services/covenant-radar-api; docker compose up -d --build

up-grandma: infra
	Set-Location services/grandma-api; docker compose up -d --build

up-github-stats: infra
	Set-Location services/github-stats-api; docker compose up -d --build

up-opportunity: infra
	Set-Location services/opportunity-radar-api; docker compose up -d --build

up-discord: infra
	Set-Location clients/DiscordBot; docker compose up -d --build

# ---------------------------------------------------------------------------
# All Services
# ---------------------------------------------------------------------------
up-all: infra up-databank up-trainer up-art-trainer up-handwriting up-qr up-transcript up-turkic up-music up-covenant up-grandma up-github-stats up-opportunity up-discord
	Write-Host "All services started" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Stop/Cleanup
# ---------------------------------------------------------------------------
down:
	$$dirs = @("services/data-bank-api", "services/Model-Trainer", "services/Art-Trainer", "services/handwriting-ai", "services/qr-api", "services/transcript-api", "services/turkic-api", "services/music-wrapped-api", "services/covenant-radar-api", "services/grandma-api", "services/github-stats-api", "services/opportunity-radar-api", "clients/DiscordBot"); foreach ($$d in $$dirs) { if (Test-Path "$$d/docker-compose.yml") { Push-Location $$d; docker compose down; Pop-Location } }; docker compose down

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
# Development: lint and test across all libs/services/clients
# ---------------------------------------------------------------------------
lint:
	$$root = Get-Location; $$dirs = @(); foreach ($$p in @("libs","services","clients","tools")) { foreach ($$d in Get-ChildItem -Path $$p -Directory) { if (Test-Path (Join-Path $$d.FullName "Makefile")) { $$dirs += $$d } } }; $$failed = @(); foreach ($$d in $$dirs) { Write-Host "`n=== Linting $$d.Name ===" -ForegroundColor Cyan; Set-Location $$d.FullName; make lint; if ($$LASTEXITCODE -ne 0) { $$failed += $$d.Name }; Set-Location $$root }; if ($$failed.Count -gt 0) { Write-Host "`nFailed: $$($$failed -join ', ')" -ForegroundColor Red; exit 1 } else { Write-Host "`nAll lint passed" -ForegroundColor Green }

test:
	$$root = Get-Location; $$dirs = @(); foreach ($$p in @("libs","services","clients","tools")) { foreach ($$d in Get-ChildItem -Path $$p -Directory) { if (Test-Path (Join-Path $$d.FullName "Makefile")) { $$dirs += $$d } } }; $$failed = @(); foreach ($$d in $$dirs) { Write-Host "`n=== Testing $$d.Name ===" -ForegroundColor Cyan; Set-Location $$d.FullName; make test; if ($$LASTEXITCODE -ne 0) { $$failed += $$d.Name }; Set-Location $$root }; if ($$failed.Count -gt 0) { Write-Host "`nFailed: $$($$failed -join ', ')" -ForegroundColor Red; exit 1 } else { Write-Host "`nAll tests passed" -ForegroundColor Green }

# No root `check` target. Checking every lib, service and client in one
# command rebuilt every virtualenv in the monorepo to answer a question about
# one project, so it was too slow to actually be run. Run `make check` inside
# the project you changed.

# ---------------------------------------------------------------------------
# Git hooks: point this clone at the versioned .githooks directory
# ---------------------------------------------------------------------------
# `core.hooksPath` is LOCAL config and cannot be committed, so a fresh clone
# runs no hooks until this is done once. That is the one weakness of the hook
# route and it is stated rather than hidden: `check-hooks` reports the clone's
# actual setting, so "we have a pre-commit hook" is verifiable instead of
# assumed.
install-hooks:
	git config core.hooksPath .githooks
	Write-Host "core.hooksPath = $$(git config --get core.hooksPath)" -ForegroundColor Green

check-hooks:
	$$configured = git config --get core.hooksPath; if ($$configured -eq ".githooks") { Write-Host "hooks installed: core.hooksPath = $$configured" -ForegroundColor Green } else { Write-Host "hooks NOT installed. This clone runs no pre-commit checks; the shared-index sweep is unguarded here. Run: make install-hooks" -ForegroundColor Red; exit 1 }
