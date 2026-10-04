include ../../scripts/make/shell.mk

.PHONY: lint test check _check-unbudgeted execution

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

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint | test

# The case that asks the real scheduler (board task 465689f5): a preflight of
# runs/execution-preflight.json over ssh to HPC3, which uploads the rendered
# script and runs sbatch --test-only, so nothing is queued. ``make check``
# skips it everywhere, because it is also API's CI, which has no route to the
# cluster; the fleet runs this target as tools/hpc3-execution on a node that
# carries the ``hpc3`` tag. It fails with HOST_EXECUTION_INCOMPLETE when the
# case is skipped or none ran. Coverage is off because the subset is not the
# suite; ``make check`` holds the 100.
execution:
	poetry sync --with dev
	$(PYTHON) ../../tools/maketools/scripts/run.py test --host-execution --no-cov
