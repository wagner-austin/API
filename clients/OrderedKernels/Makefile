include ../../scripts/make/shell.mk

MAKETOOLS := $(PYTHON) ../../tools/maketools/scripts/run.py

.PHONY: lint test check _check-unbudgeted _deps _lint-ruff _lint-guard _lint-mypy _test-suite

# First-party wheels, rebuilt fresh and installed --no-deps during setup.
# NOT poetry path-dependencies, deliberately: platform-ml -> shap 0.50.0
# pins llvmlite to a PyPI-deleted beta, so the path-dep chain cannot
# solve on a fresh machine (2026-09-01). Rebuilding here keeps the wheels
# honest against live sibling source; --no-deps keeps the dead pin out;
# the third-party closure lives in pyproject where the solver can see it.
FIRST_PARTY_WHEELS := ../../libs/platform_core/dist/platform_core-0.1.0-py3-none-any.whl ../../libs/platform_ml/dist/platform_ml-0.1.0-py3-none-any.whl ../../libs/platform_workers/dist/platform_workers-0.1.0-py3-none-any.whl ../../services/Model-Trainer/dist/model_trainer_server-0.1.0-py3-none-any.whl

lint: _deps
	$(MAKETOOLS) concurrent _lint-ruff _lint-guard _lint-mypy

_deps:
	$(MAKETOOLS) venv-check
	poetry sync --no-cache --with dev
	$(MAKETOOLS) poetry-build ../../libs/platform_core ../../libs/platform_ml ../../libs/platform_workers ../../services/Model-Trainer
	poetry run pip install --quiet --force-reinstall --no-deps $(FIRST_PARTY_WHEELS)

_lint-ruff:
	poetry run ruff check .
	poetry run ruff format --check .

_lint-guard:
	$(MAKETOOLS) guard

_lint-mypy:
	poetry run mypy src tests scripts

# SERIAL, deliberately: the shared launcher's default is `-n auto`, and this
# suite would have every xdist worker spin a CUDA context on the one GPU.
# What is kept from the launcher: coverage to a per-run path so overlapping
# sessions cannot corrupt one SQLite file, and the scoped cleanup. What the
# serial run does not need: xdist's crash-loop cap, which only applies to
# workers.
test: _deps
	$(MAKETOOLS) concurrent _test-suite

_test-suite:
	$(MAKETOOLS) test --serial

check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

# Sync the committed lock and install sibling wheels once. The remaining
# checks only read source, so they share the existing concurrent launcher.
_check-unbudgeted: _deps
	$(MAKETOOLS) concurrent _lint-ruff _lint-guard _lint-mypy _test-suite
