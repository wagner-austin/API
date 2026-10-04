include ../../scripts/make/shell.mk

.PHONY: lint test check _check-unbudgeted lint-makefiles lift-check lift-refresh execution

# lift-check runs LAST, after ruff has fixed and formatted the tree: a
# lifted file the formatter rewrote would then fail it by name rather than
# drift unseen (src/maketools/lift.py, MCPs board task 1b152218).
lint:
	$(PYTHON) scripts/run.py venv-check
	poetry lock
	poetry sync --with dev
	$(PYTHON) scripts/run.py guard
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy src tests scripts
	$(PYTHON) scripts/run.py lint-makefiles
	$(PYTHON) scripts/run.py lift-check

# Coverage data goes to a per-run path under runs/, not the default fixed
# .coverage: two overlapping runs would corrupt one file, and several
# sessions work this repo at once. The launcher owns the token and the
# scoped cleanup; see src/maketools/test_run.py.
test:
	poetry lock
	poetry sync --with dev
	$(PYTHON) scripts/run.py test

# Every package's check runs under the five-minute budget (MCPs board task
# 1b152218): check-budget runs _check-unbudgeted, the old prerequisites,
# and fails a passing check past 300 s with exit 3, naming its time.
check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint | test

lint-makefiles:
	$(PYTHON) scripts/run.py lint-makefiles

lift-check:
	$(PYTHON) scripts/run.py lift-check

# Re-lift every file lift-lock.json names from a commit of the MCPs checkout
# beside this repository: make lift-refresh REVISION=<sha>.
lift-refresh:
	$(PYTHON) scripts/run.py lift-refresh ../../../MCPs $(REVISION)

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
