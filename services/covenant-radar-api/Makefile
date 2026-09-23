include ../../scripts/make/shell.mk

MAKETOOLS := $(PYTHON) ../../tools/maketools/scripts/run.py

.PHONY: lint test check native-deps

# cleargbm_rs is a maturin (Rust) package and the only path dependency declared
# without develop = true, with its version frozen at 0.1.0. `poetry install`
# therefore considers it satisfied and never reinstalls it, so rebuilding the
# crate leaves this venv running whatever build it first saw. That is not a
# theoretical drift: it presented as `module 'cleargbm_rs' has no attribute
# 'py_gbm_model_to_json_rs'` at conftest import, which took the whole test
# suite down. This target reinstalls from the crate's built wheel whenever that
# wheel is newer than the installed package (tools/maketools, ``native-wheel``).
native-deps:
	$(MAKETOOLS) native-wheel --crate ../../libs/cleargbm_rs --package cleargbm_rs

lint: native-deps
	$(MAKETOOLS) require-tool poetry "Install it with pipx: pip install pipx, then pipx install poetry"
	$(MAKETOOLS) venv-check
	poetry lock
	poetry sync --with dev
	$(MAKETOOLS) guard
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy src tests scripts

# Test running lives in the shared launcher, tools/maketools (the ``test``
# command); see any sibling package's Makefile for what it adds over the
# inline recipe and the incident behind it.
test: native-deps
	$(MAKETOOLS) require-tool poetry "Install it with pipx: pip install pipx, then pipx install poetry"
	poetry lock
	poetry sync --with dev
	$(MAKETOOLS) test

check: lint | test
	@echo "=== ALL CHECKS PASSED ==="
