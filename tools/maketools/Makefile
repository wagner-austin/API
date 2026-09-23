include ../../scripts/make/shell.mk

.PHONY: lint test check lint-makefiles

lint:
	$(PYTHON) scripts/run.py venv-check
	poetry lock
	poetry sync --with dev
	$(PYTHON) scripts/run.py guard
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy src tests scripts
	$(PYTHON) scripts/run.py lint-makefiles

# Coverage data goes to a per-run path under runs/, not the default fixed
# .coverage: two overlapping runs would corrupt one file, and several
# sessions work this repo at once. The launcher owns the token and the
# scoped cleanup; see src/maketools/test_run.py.
test:
	poetry lock
	poetry sync --with dev
	$(PYTHON) scripts/run.py test

check: lint | test
	@echo "=== ALL CHECKS PASSED ==="

lint-makefiles:
	$(PYTHON) scripts/run.py lint-makefiles
