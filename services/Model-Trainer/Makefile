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
#
# --durations=0 with --durations-min=1.0 names every test step of a second or
# more in every check's log, as services/covenant-radar-api does, so a run
# that nears the five-minute budget says on the CI runner or fleet node itself
# which tests cost it (MCPs board task 2f90d785).
test: _deps _test-suite

_test-suite:
	$(MAKETOOLS) test --durations=0 --durations-min=1.0

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

# The guard, mypy and the suite read the tree and write nothing another of
# them reads, so once _lint-prep has synced and formatted, they run side by
# side (tools/maketools ``concurrent``) and the check is charged the longest
# of the three instead of their sum, with the lock and sync run once instead
# of twice. CI job 113153367247 at 06956410e spent 20 s in the guard and 53 s
# in mypy before the suite began, and 9 s syncing a second time (MCPs board
# task 2f90d785); libs/covenant_ml and services/covenant-radar-api took this
# shape first.
_check-unbudgeted: _lint-prep
	$(MAKETOOLS) concurrent _lint-guard _lint-mypy _test-suite
