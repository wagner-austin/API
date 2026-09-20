include ../../scripts/make/shell.mk

MAKETOOLS := $(PYTHON) ../../tools/maketools/scripts/run.py

.PHONY: help install lint test check

help:
	@echo ""
	@echo "NavProbe -- available targets"
	@echo "============================="
	@echo ""
	@echo "Code health:"
	@echo "  make check                       lint + test (the gate)"
	@echo "  make lint                        guard + ruff + mypy over src, tests, scripts"
	@echo "  make test                        pytest + coverage (100% statements and branches)"
	@echo "  make install                     poetry install"
	@echo ""

install:
	@echo "=== install"
	poetry lock
	poetry install --with dev
	@echo "[install] done"

lint:
	@echo "=== lint"
	$(MAKETOOLS) venv-check
	poetry lock
	poetry install --with dev
	@echo "  -- guard"
	poetry run python -m scripts.guard
	@echo "  -- ruff check + format"
	poetry run ruff check . --fix
	poetry run ruff format .
	@echo "  -- mypy"
	poetry run mypy src tests scripts
	@echo "[lint] passed"

# Test running lives in the shared launcher, tools/maketools (the ``test``
# command): a per-run COVERAGE_FILE under runs/ so two concurrent runs cannot
# corrupt one SQLite file (the reader then fails with "no such table: arc"
# and reports 0.00% while every test passes), the pre-run sweep, and the
# kill-on-close job object on Windows.
test:
	@echo "=== test"
	poetry lock
	poetry install --with dev
	$(MAKETOOLS) test
	@echo "[test] passed"

check: lint | test
	@echo ""
	@echo "=== ALL CHECKS PASSED ==="
	@echo ""
