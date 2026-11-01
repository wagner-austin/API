# tools/maketools' deploy cases, executed on a rootless daemon (board task
# 465689f5).
#
# `make check` here runs tools/maketools' `execution` target: each service
# covered goes through the real compose_up, must pass its compose file's own
# healthcheck, and goes through the real compose_down, under a project minted
# for the run. It is the fleet project tools/maketools-execution, which
# requires the `linux` and `docker` tags, so the fleet builds it as a node's
# execdocker user against that user's rootless daemon, which cannot reach the
# stack's. tools/maketools' own `make check` skips the cases on every
# machine, so without this project a service's image, entry and healthcheck
# met a real daemon only when somebody deployed it.
include ../../scripts/make/shell.mk

# The images the deploy cases build, one target per service. The fleet runs
# each as its own install step, before the timed check (registry
# projects/tools/maketools-execution.json), so the check's own builds read a
# warm build cache, the budget measures the deploys, and the verdict's
# phases= names each service's build seconds. On loki on 2026-11-01, with
# an empty build cache, the eight cases took 460 to 654 s and the check 664
# s; warm, the whole check took 31 s (MCPs board task eb6586e4). Each line
# is the build its service's compose file declares, from the repository
# root, which tools/maketools/tests/test_execution_images.py holds to those
# files and to the registry's steps. data-bank-api, music-wrapped-api,
# qr-api and turkic-api left with their packages, archived on 2026-11-01
# (MCPs board task a276f973).
.PHONY: lint test check _check-unbudgeted image-github-stats-api image-grandma-api \
	image-opportunity-radar-api image-transcript-api

image-github-stats-api:
	docker build --quiet --file ../../services/github-stats-api/Dockerfile ../..

# grandma-api's build alone prints its steps, the slowest cold build here
# (189 s on loki): the verdict's steps= reads each step's seconds off them
# (API tools/fleet build_steps.py). Every other build stays quiet, so these
# lines stay inside the transcript tail the verdict reads.
image-grandma-api:
	docker build --progress plain --file ../../services/grandma-api/Dockerfile ../..

image-opportunity-radar-api:
	docker build --quiet --file ../../services/opportunity-radar-api/Dockerfile ../..

image-transcript-api:
	docker build --quiet --file ../../services/transcript-api/Dockerfile --target api ../..
	docker build --quiet --file ../../services/transcript-api/Dockerfile --target worker ../..

# This directory holds no code of its own, only this recipe, so its lint is
# the grammar every tracked Makefile keeps to.
lint:
	$(PYTHON) ../../tools/maketools/scripts/run.py lint-makefiles

# `make`, never $(MAKE): on serendipity $(MAKE) is a path with spaces, which
# PowerShell splits (tools/fleet-execution/Makefile carries the incident).
test:
	make -C ../maketools execution

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint test
