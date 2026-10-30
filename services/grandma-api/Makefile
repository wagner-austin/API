include ../../scripts/make/shell.mk

MAKETOOLS := $(PYTHON) ../../tools/maketools/scripts/run.py

.PHONY: lint test check _check-unbudgeted up down logs

lint:
	$(MAKETOOLS) venv-check
	poetry lock
	poetry sync --no-cache --with dev
	$(MAKETOOLS) guard
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy src tests scripts

# Test running lives in the shared launcher, tools/maketools (the ``test``
# command); see any sibling package's Makefile for what it adds over the
# inline recipe and the incident behind it. The web frontend's unit suite
# runs through its own Makefile.
test:
	poetry lock
	poetry sync --no-cache --with dev
	$(MAKETOOLS) test
	make -C web test

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint test

# up, down and logs drive this directory's compose file, the route the
# root's up-grandma takes; the container joins platform-network, which the
# root's `make infra` creates. They were a Windows PowerShell host mode over
# the Windows socket table until MCPs board task 703fec5a; the web
# frontend's local HTTPS server is `make -C web serve`.
up:
	$(MAKETOOLS) compose-up .

down:
	$(MAKETOOLS) compose-down .

logs:
	docker compose logs -f
