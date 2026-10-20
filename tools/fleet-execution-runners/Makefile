# tools/fleet's runner host cases, executed on the runner host (MCPs board
# task 715b9ec5, A3).
#
# `make check` here runs tools/fleet's `execution-runners` target: the cases
# that read, on lavender itself, that every runner unit runs in the listener
# slice and every job in its scope of the CI slice, and that a slice over its
# memory high stalls its own tasks and not an unbounded sibling. The fleet
# project tools/fleet-execution-runners runs it pinned to that node, as it
# ran pinned to lavender-wsl until lavender's reimage to Ubuntu (MCPs board
# task 1e9ab738). Nothing else can: the cases read live units, so
# tools/fleet's own `make check` and the platform execution runs leave them
# out, and none of them is a roll or deploy gate.
include ../../scripts/make/shell.mk

.PHONY: lint test check _check-unbudgeted

# This directory holds no code of its own, only this recipe, so its lint is
# the grammar every tracked Makefile keeps to.
lint:
	$(PYTHON) ../../tools/maketools/scripts/run.py lint-makefiles

# `make`, never $(MAKE), as tools/fleet-execution's recipe says.
test:
	make -C ../fleet execution-runners

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint test
