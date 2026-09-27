# tools/fleet's host cases, executed where their platform is (board task
# 465689f5).
#
# `make check` here runs tools/fleet's `execution` target: the cases that
# run rendered PowerShell or sh for real, kept to this host's platform, with
# none allowed to skip. It is the fleet projects tools/fleet-execution (a
# Windows node) and tools/fleet-execution-linux (a Linux node), and the
# deploys that run fleet code refuse a commit either has not passed at
# (tools/maketools publish-executed, read from MCPs). tools/fleet's own
# `make check` skips these cases off their platform, and API's CI is Linux,
# so without this project the Windows ones ran on no machine but the hub.
include ../../scripts/make/shell.mk

.PHONY: lint test check

# This directory holds no code of its own, only this recipe, so its lint is
# the grammar every tracked Makefile keeps to.
lint:
	$(PYTHON) ../../tools/maketools/scripts/run.py lint-makefiles

test:
	$(MAKE) -C ../fleet execution

check: lint test
	@echo "=== ALL CHECKS PASSED ==="
