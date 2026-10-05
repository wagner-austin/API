# clients/TankpitBot's live demo case, executed against the real service
# (MCPs board task 46934cd6).
#
# `make check` here runs clients/TankpitBot's `execution` target: the case in
# tests/live that watches austinwagner.org/tankpit play, pressing the public
# spawn button when no demo bot is up, and requires a caption from the bot's
# own state, a playlist stamped with wall-clock times and an AAC audio stream
# in a served segment. It is the fleet project clients/TankpitBot-execution,
# which requires the `windows` and `ffmpeg` tags. clients/TankpitBot's own
# `make check` skips the case on every machine, because that run is also
# API's CI, where it would start a public bot for every push; without this
# project nothing ever measured the live demo, only its code against fakes.
include ../../scripts/make/shell.mk

.PHONY: lint test check _check-unbudgeted

# This directory holds no code of its own, only this recipe, so its lint is
# the grammar every tracked Makefile keeps to.
lint:
	$(PYTHON) ../../tools/maketools/scripts/run.py lint-makefiles

# `make`, never $(MAKE): on serendipity $(MAKE) is a path with spaces, which
# PowerShell splits (tools/fleet-execution/Makefile carries the incident).
test:
	make -C ../TankpitBot execution

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint test
