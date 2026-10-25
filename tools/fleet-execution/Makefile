# tools/fleet's host cases, executed on a Linux host (board task 465689f5).
#
# `make check` here runs tools/fleet's `execution` target: the cases that
# run rendered sh for real, with none allowed to skip. It is the fleet
# project tools/fleet-execution-linux, and the deploys that run fleet code
# refuse a commit it has not passed at (tools/maketools publish-executed,
# read from MCPs). tools/fleet's own `make check` skips these cases off a
# Linux host. Until board task e705f08c a Windows node ran the same recipe
# as tools/fleet-execution; that project left with the fleet's last Windows
# node.
include ../../scripts/make/shell.mk

.PHONY: lint test check _check-unbudgeted

# This directory holds no code of its own, only this recipe, so its lint is
# the grammar every tracked Makefile keeps to.
lint:
	$(PYTHON) ../../tools/maketools/scripts/run.py lint-makefiles

test:
	make -C ../fleet execution

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint test
