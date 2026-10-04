# tools/hpc3's host case, executed where the route to the cluster is (board
# task 465689f5).
#
# `make check` here runs tools/hpc3's `execution` target: a real preflight of
# a committed run over ssh to HPC3, which uploads the rendered script and asks
# `sbatch --test-only`, so nothing is queued and no allocation is made. It is
# the fleet project tools/hpc3-execution, which requires the `hpc3` capability
# tag, carried only by a node holding the ssh route through a jump host.
# tools/hpc3's own `make check` skips the case on every machine, and API's CI
# has no route, so without this project the submission path met the real
# cluster only when somebody submitted a job.
include ../../scripts/make/shell.mk

.PHONY: lint test check _check-unbudgeted

# This directory holds no code of its own, only this recipe, so its lint is
# the grammar every tracked Makefile keeps to.
lint:
	$(PYTHON) ../../tools/maketools/scripts/run.py lint-makefiles

# `make`, never $(MAKE): on serendipity $(MAKE) is a path with spaces, which
# PowerShell splits (tools/fleet-execution/Makefile carries the incident).
test:
	make -C ../hpc3 execution

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint test
