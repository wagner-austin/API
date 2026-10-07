include ../../scripts/make/shell.mk

MAKETOOLS := $(PYTHON) ../../tools/maketools/scripts/run.py

.PHONY: lint test check _check-unbudgeted native-deps _deps _lint-prep _lint-guard _lint-mypy _test-suite

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

lint: _lint-prep _lint-guard _lint-mypy

# What lint and the suite both need first: the native wheel, a venv that
# answers, and the lock and the dev dependencies synced.
_deps: native-deps
	$(MAKETOOLS) require-tool poetry "Install it with pipx: pip install pipx, then pipx install poetry"
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
# command); see any sibling package's Makefile for what it adds over the
# inline recipe and the incident behind it. --durations names the suite's 25
# slowest steps in every check's log, so the next run that nears the
# five-minute budget says on the fleet node itself which tests cost it.
test: _deps _test-suite

_test-suite:
	$(MAKETOOLS) test --durations=25

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

# The guard, mypy and the suite read the tree and write nothing another of
# them reads, so once _lint-prep has synced and formatted, they run side by
# side (tools/maketools ``concurrent``) and the check is charged the longest
# of the three instead of their sum, with the lock and sync run once instead
# of twice (board task dbbb9758: on serendipity at 3f1bbe1c7 the check took
# 497 s, 244 s of it the suite run after the whole of lint).
_check-unbudgeted: _lint-prep
	$(MAKETOOLS) concurrent _lint-guard _lint-mypy _test-suite
