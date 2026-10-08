include ../../scripts/make/shell.mk

.PHONY: lint test check _check-unbudgeted

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
#
# --durations=0 with --durations-min=1.0 names every test step of a second or
# more in every check's log, as services/covenant-radar-api does, so a run
# that nears the five-minute budget says on the CI runner or fleet node itself
# which tests cost it (MCPs board task 2f90d785).
test:
	poetry lock
	poetry sync --with dev
	$(PYTHON) ../../tools/maketools/scripts/run.py test --durations=0 --durations-min=1.0

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint | test
