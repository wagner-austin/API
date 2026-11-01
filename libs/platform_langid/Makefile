include ../../scripts/make/shell.mk

.PHONY: lint test check _check-unbudgeted

lint:
	$(PYTHON) ../../tools/maketools/scripts/run.py venv-check
	poetry lock
	poetry sync --no-cache --with dev
	$(PYTHON) ../../tools/maketools/scripts/run.py guard
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy src tests scripts

# Test running lives in the shared launcher, tools/maketools (the ``test``
# command), so it is defined once instead of being copy-pasted into every
# Makefile. That duplication is why a process-lifecycle bug had to be fixed
# 36 times rather than once.
#
# The launcher keeps the per-run coverage path this recipe always had:
# coverage writes one SQLite file at a fixed path, and two overlapping runs
# corrupt it, which surfaces as a spurious failure on a run whose tests all
# passed; several sessions work this repo at once, so overlap is normal. It
# adds a pre-run sweep and, on Windows, a kill-on-close job object so a run
# that dies cannot strand its xdist workers; its module header carries the
# 2026-08-19 incident that motivated both.
#
# --own-home runs the suite under a HOME of the run's own and fails it on any
# file left there. --hub-offline refuses the Hugging Face hub, so a test that
# resolves a hub id fails by name: two such tests pulled the 3.86 GB
# MMS-LID-126 checkpoint on pendragon, 240 s of fleet job f2af5334's 634 s,
# into the run's own HF_HOME, which --own-home alone never sees (MCPs board
# task abb0a10e).
test:
	poetry lock
	poetry sync --no-cache --with dev
	$(PYTHON) ../../tools/maketools/scripts/run.py test --own-home --hub-offline

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint | test
