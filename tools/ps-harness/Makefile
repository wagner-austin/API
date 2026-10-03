# API's PowerShell harness as a fleet project (MCPs board task 939ec5c7).
#
# `make check` here runs MCPs' scripts/ps-harness, at the MCPs commit
# .githooks/mcps-harness.sha records, over this whole repository two
# directories up: every tracked .ps1 guarded, analysed and executed by its
# Pester suite to 100 percent of its commands and branch arms. The root's
# `make check-powershell` calls this target, so the command line exists once,
# and API's powershell workflow runs that.
#
# WHY A DIRECTORY OF ITS OWN. A fleet project names one directory whose
# `make check` a node runs (tools/fleet/fleet.json), and API's root has no
# check. Without one the harness ran only on the hub and in CI, and on
# 2026-10-03 the review of task 939ec5c7 had only a hub run to read, which
# failed in rendered-provision-windows.Tests.ps1's teardown. The project
# declares MCPs as a companion, which the hub sends as a bundle with its
# history and the node clones to ../MCPs beside the export, so the pinned
# commit is there however far main has moved.
#
# It requires windows and elevated, as MCPs/scripts/ps-harness does: the
# provision suites register real scheduled tasks under S4U, and the elevated
# runner is the one whose builds hold an administrator's token.
#
# THE PIN IS READ BY tools/maketools, NOT BY MAKE. `$(file <...)` exists from
# GNU make 4.0, and serendipity's GnuWin32 make 3.81 read it as nothing, so
# the first fleet run (job 2d949933) handed the harness an empty --ref.
include ../../scripts/make/shell.mk

.PHONY: lint test check

# This directory holds no code of its own, only this recipe, so its lint is
# the grammar every tracked Makefile keeps to.
lint:
	$(PYTHON) ../../tools/maketools/scripts/run.py lint-makefiles

test:
	$(PYTHON) ../../tools/maketools/scripts/run.py ps-harness

check: lint test
	@echo "=== ALL CHECKS PASSED ==="
