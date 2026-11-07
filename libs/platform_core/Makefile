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
# --mutation (MCPs board task e52e5148): after a passing suite the launcher
# mutates the src lines this commit adds against origin/main with
# cosmic-ray, runs each mutant by the tests that ran its line, and fails
# naming every mutant they leave alive. Its time is held to its own bound,
# not the five minutes; the fleet stages origin/main because this
# package's registry entry declares base_branch main.
test:
	poetry lock
	poetry sync --no-cache --with dev
	$(PYTHON) ../../tools/maketools/scripts/run.py test --mutation

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint | test
