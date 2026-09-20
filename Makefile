include scripts/make/shell.mk

# Every recipe below is one plain command in the grammar tools/maketools
# enforces (see its README); anything with logic is a maketools command.
MAKETOOLS := $(PYTHON) tools/maketools/scripts/run.py

.PHONY: infra up-databank up-trainer up-art-trainer up-handwriting up-qr up-transcript up-turkic up-music up-covenant up-grandma up-github-stats up-opportunity up-discord up-all down clean status logs lint test install-hooks check-hooks lint-makefiles

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
infra:
	docker compose up -d

# ---------------------------------------------------------------------------
# Individual Services (each starts infra first)
# ---------------------------------------------------------------------------
up-databank: infra
	$(MAKETOOLS) compose-up services/data-bank-api

# GIT_COMMIT is exported by compose-up rather than left to the operator's
# shell. The Dockerfile bakes it so every manifest and run fingerprint can
# name the code that produced a number, and it defaults to empty -- so a
# build that forgets it records null and nobody notices until the
# provenance audit. That is not hypothetical: every manifest archived by the
# 2026-08-18 audit has git_commit null for exactly this reason.
up-trainer: infra
	$(MAKETOOLS) compose-up services/Model-Trainer --git-commit --build-progress plain

up-art-trainer: infra
	$(MAKETOOLS) compose-up services/Art-Trainer --build-progress plain

up-handwriting: infra
	$(MAKETOOLS) compose-up services/handwriting-ai

up-qr: infra
	$(MAKETOOLS) compose-up services/qr-api

up-transcript: infra
	$(MAKETOOLS) compose-up services/transcript-api

up-turkic: infra
	$(MAKETOOLS) compose-up services/turkic-api

up-music: infra
	$(MAKETOOLS) compose-up services/music-wrapped-api

up-covenant: infra
	$(MAKETOOLS) compose-up services/covenant-radar-api

up-grandma: infra
	$(MAKETOOLS) compose-up services/grandma-api

up-github-stats: infra
	$(MAKETOOLS) compose-up services/github-stats-api

up-opportunity: infra
	$(MAKETOOLS) compose-up services/opportunity-radar-api

up-discord: infra
	$(MAKETOOLS) compose-up clients/DiscordBot

# ---------------------------------------------------------------------------
# All Services
# ---------------------------------------------------------------------------
up-all: infra up-databank up-trainer up-art-trainer up-handwriting up-qr up-transcript up-turkic up-music up-covenant up-grandma up-github-stats up-opportunity up-discord
	@echo "All services started"

# ---------------------------------------------------------------------------
# Stop/Cleanup
# ---------------------------------------------------------------------------
down:
	$(MAKETOOLS) compose-down services/data-bank-api services/Model-Trainer services/Art-Trainer services/handwriting-ai services/qr-api services/transcript-api services/turkic-api services/music-wrapped-api services/covenant-radar-api services/grandma-api services/github-stats-api services/opportunity-radar-api clients/DiscordBot
	docker compose down

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
# Development: lint and test across all libs/services/clients/tools
# ---------------------------------------------------------------------------
lint:
	$(MAKETOOLS) fan-out lint libs services clients tools

test:
	$(MAKETOOLS) fan-out test libs services clients tools

# Every tracked Makefile against the portable grammar (tools/maketools
# README, "The grammar"); tools/maketools' own lint runs it too.
lint-makefiles:
	$(MAKETOOLS) lint-makefiles

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
	$(MAKETOOLS) hooks install

check-hooks:
	$(MAKETOOLS) hooks check
