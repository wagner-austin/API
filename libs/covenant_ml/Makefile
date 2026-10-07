include ../../scripts/make/shell.mk

MAKETOOLS := $(PYTHON) ../../tools/maketools/scripts/run.py

.PHONY: lint test check _check-unbudgeted _deps _lint-prep _lint-guard _lint-mypy _test-suite

lint: _lint-prep _lint-guard _lint-mypy

# What lint and the suite both need first: a venv that answers, and the lock
# and the dev dependencies synced.
_deps:
	$(MAKETOOLS) venv-check
	poetry lock
	poetry sync --with dev

# The two steps that REWRITE sources (ruff's fixes and its formatter), which
# must finish before anything reads the tree.
_lint-prep: _deps
	poetry run ruff check . --fix
	poetry run ruff format .

_lint-guard:
	$(MAKETOOLS) guard

_lint-mypy:
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
test: _deps _test-suite

_test-suite:
	$(MAKETOOLS) test

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

# The guard, mypy and the suite read the tree and write nothing another of
# them reads, so once _lint-prep has synced and formatted, they run side by
# side (tools/maketools ``concurrent``) and the check is charged the longest
# of the three instead of their sum, with the lock and sync run once instead
# of twice. On serendipity at 37ba0350f the check took 363 s with the guard
# and mypy (about 30 s each there) run before a 141 s suite (board task
# 2e880b79); services/covenant-radar-api took this shape for dbbb9758.
_check-unbudgeted: _lint-prep
	$(MAKETOOLS) concurrent _lint-guard _lint-mypy _test-suite
