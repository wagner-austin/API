include ../../scripts/make/shell.mk

MAKETOOLS := $(PYTHON) ../../tools/maketools/scripts/run.py

.PHONY: lint test check

# First-party wheels, rebuilt fresh and installed --no-deps every lint.
# NOT poetry path-dependencies, deliberately: platform-ml -> shap 0.50.0
# pins llvmlite to a PyPI-deleted beta, so the path-dep chain cannot
# solve on a fresh machine (2026-09-01). Rebuilding here keeps the wheels
# honest against live sibling source; --no-deps keeps the dead pin out;
# the third-party closure lives in pyproject where the solver can see it.
FIRST_PARTY_WHEELS := ../../libs/platform_core/dist/platform_core-0.1.0-py3-none-any.whl ../../libs/platform_ml/dist/platform_ml-0.1.0-py3-none-any.whl ../../libs/platform_workers/dist/platform_workers-0.1.0-py3-none-any.whl ../../services/Model-Trainer/dist/model_trainer_server-0.1.0-py3-none-any.whl

lint:
	$(MAKETOOLS) venv-check
	poetry lock
	poetry sync --with dev
	$(MAKETOOLS) poetry-build ../../libs/platform_core ../../libs/platform_ml ../../libs/platform_workers ../../services/Model-Trainer
	poetry run pip install --quiet --force-reinstall --no-deps $(FIRST_PARTY_WHEELS)
	$(MAKETOOLS) guard
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy src tests scripts

# poetry sync REMOVES anything not in the lock -- including the first-party
# wheels lint just installed -- so the wheels go back in here too. Same
# reasoning as the lint block above.
#
# SERIAL, deliberately: the shared launcher's default is `-n auto`, and this
# suite would have every xdist worker spin a CUDA context on the one GPU.
# What is kept from the launcher: coverage to a per-run path so overlapping
# sessions cannot corrupt one SQLite file, and the scoped cleanup. What the
# serial run does not need: xdist's crash-loop cap, which only applies to
# workers.
test:
	poetry lock
	poetry sync --with dev
	poetry run pip install --quiet --force-reinstall --no-deps $(FIRST_PARTY_WHEELS)
	$(MAKETOOLS) test --serial

check: lint | test
