include ../../scripts/make/shell.mk

.PHONY: lint test check execution

lint:
	$(PYTHON) ../../tools/maketools/scripts/run.py venv-check
	poetry lock
	poetry sync --with dev
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
test:
	poetry lock
	poetry sync --with dev
	$(PYTHON) ../../tools/maketools/scripts/run.py test

check: lint | test
	@echo "=== ALL CHECKS PASSED ==="

# The host cases, run on a host of their platform (board task 465689f5).
# ``make check`` skips them wherever the platform differs, and API's CI is
# Linux, so this target is where the Windows cases execute: the fleet runs
# it as tools/fleet-execution on a Windows node and tools/fleet-execution-
# linux on a Linux one. It keeps only this platform's host cases and fails
# with HOST_EXECUTION_INCOMPLETE when any is skipped or none ran. Coverage
# is off because the subset is not the suite; ``make check`` holds the 100.
execution:
	poetry sync --with dev
	$(PYTHON) ../../tools/maketools/scripts/run.py test --host-execution --no-cov
