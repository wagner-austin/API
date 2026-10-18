# tools/fleet's hub cases, executed on the hub (MCPs board task 3d9eec0d).
#
# `make check` here runs tools/fleet's `execution-hub` target: the case that
# reads every enabled node's stage root over the hub's ssh, against the node
# runners' records in the API checkout the hub's ticks run with, and fails on
# any run directory that no live run and no owed retire holds. The fleet
# project tools/fleet-execution-hub runs it pinned to the hub. Nothing else
# can: the case reads the hub's records, so tools/fleet's own `make check`
# and the platform execution runs leave it out, and it is no roll or deploy
# gate.
include ../../scripts/make/shell.mk

.PHONY: lint test check _check-unbudgeted

# This directory holds no code of its own, only this recipe, so its lint is
# the grammar every tracked Makefile keeps to.
lint:
	$(PYTHON) ../../tools/maketools/scripts/run.py lint-makefiles

# `make`, never $(MAKE), as tools/fleet-execution's recipe says.
test:
	make -C ../fleet execution-hub

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint test
