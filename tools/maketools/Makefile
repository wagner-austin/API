include ../../scripts/make/shell.mk

.PHONY: lint test check lint-makefiles execution

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

# The cases that deploy a service for real (board task 465689f5): compose_up
# builds each service's image from its committed Dockerfile, the container
# must pass the compose file's own healthcheck, and compose_down must leave
# nothing of the run's project. ``make check`` skips them everywhere, because
# it is also API's CI and the hub; the fleet runs this target as
# tools/maketools-execution, as a Linux node's execdocker user against its
# rootless daemon, and each case refuses any other daemon. It fails with
# HOST_EXECUTION_INCOMPLETE when a case is skipped or none ran. Coverage is
# off because the subset is not the suite; ``make check`` holds the 100.
execution:
	poetry sync --with dev
	$(PYTHON) scripts/run.py test --host-execution --no-cov
