# TankPit's move from sedona to colossus, measured on colossus (board task
# c481c259).
#
# `make check` here runs clients/TankpitBot's `colossus-move-colossus`
# target on the game host itself: every file of the copy under
# /home/user/tankpit-archive/sedona-2026-10-18 must hash to the listing the
# move committed, the move's release must be on the ladder, its image on the
# host's daemon and built after it, and no container may be named tankpit or
# started from that image. It is the fleet project
# clients/TankpitBot-colossus-execution, pinned to colossus, whose runner is
# the account that holds the copy and reaches the daemon. Once the operator
# resumes the fleet, a container from that image exists and this refuses,
# by design: it measured the move, at the commit it checked.
include ../../scripts/make/shell.mk

.PHONY: lint test check _check-unbudgeted

# This directory holds no code of its own, only this recipe, so its lint is
# the grammar every tracked Makefile keeps to.
lint:
	$(PYTHON) ../../tools/maketools/scripts/run.py lint-makefiles

test:
	make -C ../TankpitBot colossus-move-colossus

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint test
