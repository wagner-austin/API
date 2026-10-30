include ../../scripts/make/shell.mk

.PHONY: lint test check _check-unbudgeted execution execution-runners execution-hub render-documents

lint:
	$(PYTHON) ../../tools/maketools/scripts/run.py venv-check
	poetry lock
	poetry sync --no-cache --with dev
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
	poetry sync --no-cache --with dev
	$(PYTHON) ../../tools/maketools/scripts/run.py test

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint | test

# The host cases, run on a Linux host (board task 465689f5). ``make check``
# skips them off one; the fleet runs this target as
# tools/fleet-execution-linux on a Linux node. It keeps only the host cases
# and fails with HOST_EXECUTION_INCOMPLETE when any is skipped or none ran.
# Coverage is off because the subset is not the suite; ``make check`` holds
# the 100.
execution:
	poetry sync --no-cache --with dev
	$(PYTHON) ../../tools/maketools/scripts/run.py test --host-execution --no-cov

# The runner host's cases (MCPs board task 715b9ec5): where lavender's
# runner units and their jobs run, read on lavender, and a full slice
# measured beside an unbounded sibling there. The fleet runs it as
# tools/fleet-execution-runners, pinned to that node; --runner-host names the
# runners.json host the cases read, and no other run has their capability.
execution-runners:
	poetry sync --no-cache --with dev
	$(PYTHON) ../../tools/maketools/scripts/run.py test --host-execution --runner-host=lavender --no-cov

# The hub's cases (MCPs board task 3d9eec0d): every enabled node's stage root,
# read over the hub's ssh against the node runners' records in the API
# checkout the hub's ticks run with. The fleet runs it as
# tools/fleet-execution-hub, pinned to the hub; --fleet-hub names that
# checkout, and no other run has its records.
execution-hub:
	poetry sync --no-cache --with dev
	$(PYTHON) ../../tools/maketools/scripts/run.py test --host-execution --fleet-hub=$(HOME)/PROJECTS/API --no-cov

# The committed JSON copies of the pins readers outside this package need:
# the hooks check pins (fleet.core.hooks_check_document, MCPs board task
# 88ca2402), read by MCPs' session-host provision.sh and the fleet audit, and
# the toolchain's Poetry (fleet.core.toolchain_document, MCPs board task
# ce698bfb), read by MCPs' host provisioning. Run this after a pin in
# fleet.contracts.tagged_tools or fleet.contracts.toolchain moves;
# tests/test_hooks_check_document.py and tests/test_toolchain_document.py
# fail until the copies match.
render-documents:
	poetry run python -m fleet.cli.render_documents --out rendered
