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
# KAGGLE CREDENTIALS: PRESENT, NOT VALID. The `kaggle` package authenticates
# AT IMPORT: it reads these two variables, checks both are set, stores them
# and returns, with no network call. With neither set it prints "Could not
# find kaggle.json" and calls `exit(1)`, which fails
# tests/test_production.py's import of it with `SystemExit: 1` and leaves two
# lines of _production.py uncovered. Dummy values make the module importable,
# so the gate needs no secret. They lived in GitHub's packages workflow until
# MCPs board task b1089284 registered this package on the fleet, whose nodes
# run this Makefile and nothing else: fleet job 39f0b4b2 on colossus failed
# exactly so on 2026-10-11. `?=` keeps a caller's real pair, which the one
# live test (opt-in behind KAGGLE_LIVE=1) needs.
test: export KAGGLE_USERNAME ?= ci
test: export KAGGLE_KEY ?= not-a-real-key
test:
	poetry lock
	poetry sync --with dev
	$(PYTHON) ../../tools/maketools/scripts/run.py test

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint | test
