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

.PHONY: lint test check

# This directory holds no code of its own, only this recipe, so its lint is
# the grammar every tracked Makefile keeps to.
lint:
	$(PYTHON) ../../tools/maketools/scripts/run.py lint-makefiles

# `make`, never $(MAKE): on serendipity $(MAKE) is a path with spaces, which
# PowerShell splits (tools/fleet-execution/Makefile carries the incident).
test:
	make -C ../maketools execution

check: lint test
	@echo "=== ALL CHECKS PASSED ==="
