# TankPit's move from sedona to colossus, measured on sedona (board task
# c481c259).
#
# `make check` here runs clients/TankpitBot's `colossus-move-sedona` target:
# every file under sedona's C:/fleet/tankpit runs and data must still hash to
# the listing the move committed, which is the copy Codex asked to keep. It
# is the fleet project clients/TankpitBot-sedona-execution, pinned to sedona,
# because only sedona holds that directory; clients/TankpitBot-colossus-execution
# measures the other end. Without it, the move's claims about sedona rested
# on a listing this repository was handed, not on sedona.
include ../../scripts/make/shell.mk

.PHONY: lint test check _check-unbudgeted

# This directory holds no code of its own, only this recipe, so its lint is
# the grammar every tracked Makefile keeps to.
lint:
	$(PYTHON) ../../tools/maketools/scripts/run.py lint-makefiles

# `make`, never $(MAKE): on serendipity $(MAKE) is a path with spaces, which
# PowerShell splits (tools/fleet-execution/Makefile carries the incident).
test:
	make -C ../TankpitBot colossus-move-sedona

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint test
