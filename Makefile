include scripts/make/shell.mk

# Every recipe below is one plain command in the grammar tools/maketools
# enforces (see its README); anything with logic is a maketools command.
MAKETOOLS := $(PYTHON) tools/maketools/scripts/run.py

.PHONY: align-commit commit-tasks executed services-executed infra fleet-roll up-databank up-trainer up-art-trainer up-handwriting up-qr up-transcript up-turkic up-music up-covenant up-grandma up-github-stats up-opportunity up-discord up-all down clean status logs lint test install-hooks check-hooks lint-makefiles

# ---------------------------------------------------------------------------
# The main checkout
# ---------------------------------------------------------------------------
# align-commit COMMIT=<sha> [DROP=<sha>,<sha>] [REPO=<path>]: move a main
# checkout, THIS one unless REPO names another, to a commit its origin's
# default branch contains, the one sanctioned mover of a checkout with linked
# session trees beside it (MCPs board task f2f89c69). The hub's hook refuses
# every hand-run git verb that moves such a checkout, and names this target
# as the route, because git inside a recipe is what the recipe does. REPO
# moves a peer that carries no mover of its own, as a session host's
# ../corvis-stick (MCPs board task a0d34aa8). The pump checkout's local
# commits are re-commits of content origin already carries, so the move may
# drop them; a local commit whose change the target does not carry is refused
# by name unless DROP names it. A dirty tracked path refuses the move; an
# untracked one the target tracks refuses it too.
REPO ?= .
align-commit:
	$(MAKETOOLS) align --repo $(REPO) --commit $(COMMIT) --drop $(DROP)

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
# A DEPLOY SHIPS A COMMIT THAT NAMES ITS TASK (MCPs board task 691b0067,
# b6eb30c8's R8). Every up-* target runs infra first, and what it builds is
# HEAD's, so HEAD's message must name a board task it serves ('board task
# <id>'), asked of the board by MCPs' maketools as MCPs' origin/main
# carries it (.githooks/published_maketools.py), with the MCPs checkout
# beside this one supplying the board's credentials and address. It reads
# no variable, and a board that cannot be asked stops the deploy.
commit-tasks:
	$(PYTHON) .githooks/published_maketools.py commit-tasks ../MCPs . HEAD^!

# A DEPLOY SHIPS A COMMIT ITS HOST CODE RAN AT (MCPs board task 465689f5,
# b6eb30c8's R3). tools/fleet's rendered sh runs on every node, and its host
# cases skip off a Linux host. MCPs' maketools publish-executed asks the
# fleet queue for a passed run of the execution project at exactly HEAD, and
# refuses a tree that differs from HEAD: EXECUTION_UNRUN, EXECUTION_TREE_DIRTY
# or, when the queue cannot be asked, EXECUTION_UNREADABLE. It reads no
# variable and has no override. Until board task e705f08c it asked for
# tools/fleet-execution too, the Windows node's half, which left with the
# fleet's last Windows node.
executed:
	$(PYTHON) .githooks/published_maketools.py publish-executed ../MCPs tools/fleet-execution-linux .

# A SERVICE DEPLOY SHIPS A COMMIT ITS SERVICES WERE DEPLOYED AT (MCPs board
# task 465689f5). tools/maketools-execution builds each covered service's
# image through the real compose_up, requires the compose file's own
# healthcheck to pass and runs the real compose_down, on a node's rootless
# daemon; the first such run found grandma-api's healthcheck calling a curl
# its image never carried. It gates infra, which every up-* target runs
# first, and not fleet-roll, since the fleet's own code is not what it runs.
services-executed:
	$(PYTHON) .githooks/published_maketools.py publish-executed ../MCPs tools/maketools-execution .

infra: commit-tasks executed services-executed
	docker compose up -d

# THE FLEET RUNS WHAT WAS ROLLED, AND A ROLL NEEDS BOTH SUITES AT HEAD (MCPs
# board task 465689f5). The hub's scheduled ticks run fleet-agent and every
# fleet-node-agent from the commit refs/fleet/rolled names, extracted by
# tools/fleet's fleet.cli.rolled, never from this checkout; this is the only
# thing that moves the ref, and only after executed passed at HEAD. The ref
# lives in the object store every worktree shares, so a roll from any of
# them is what the next tick runs. It is pushed to origin as well (MCPs
# board task c39ee056), since a machine holding a hub-runner duty other than
# the hub ticks its own lanes and has no other way to learn the roll: MCPs
# maketools release-sync fetches it there and aligns that machine's checkout
# to it. A roll may move the ref back, so the push forces it.
fleet-roll: commit-tasks executed
	git update-ref refs/fleet/rolled HEAD
	git push origin +refs/fleet/rolled:refs/fleet/rolled

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
