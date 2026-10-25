include ../../scripts/make/shell.mk

MAKETOOLS := $(PYTHON) ../../tools/maketools/scripts/run.py

.PHONY: lint test check _check-unbudgeted rust-lint rust-test rust-cov python-lint python-test setup

# Setup: create the uv venv and install the dev extras (a no-op when the
# venv already answers; tools/maketools, ``uv-venv-check``).
setup:
	$(MAKETOOLS) uv-venv-check

# Every cargo step that compiles or runs the crate goes through
# ``venv-embed``: the Rust tests embed Python through pyo3 and import numpy
# there, and an embedded interpreter is whatever installation pyo3 linked,
# never the venv, so a node whose system Python lacked numpy failed 94 of
# 1680 tests (fleet job 7722e5fa, MCPs board task 6dd73ada). ``venv-embed``
# links pyo3 against the venv's interpreter and puts its site-packages on
# PYTHONPATH. Clippy goes through it too, so all three steps share one pyo3
# configuration and none of them rebuilds the crate for another.
#
# All three name ``--features python``, never ``--all-features``: that would
# turn on ``extension-module``, which leaves libpython unlinked for maturin's
# wheel, and on Linux the test binary then fails to link pyo3's symbols
# (fleet job e8a8b1e9, API board task 38cabb1b; Windows links the python DLL
# either way, which is why only Windows nodes passed before).

# Rust linting: fmt + clippy
rust-lint:
	$(MAKETOOLS) uv-venv-check
	cargo fmt
	$(MAKETOOLS) venv-embed cargo clippy --all-targets --features python -- -D warnings

# Rust tests
rust-test:
	$(MAKETOOLS) uv-venv-check
	$(MAKETOOLS) venv-embed cargo test --features python

# Rust coverage (requires cargo-llvm-cov) - 100% segment coverage
# Uses segment-based check to avoid false negatives from generic instantiations
rust-cov:
	$(MAKETOOLS) uv-venv-check
	$(MAKETOOLS) venv-embed cargo llvm-cov --features python --json --output-path coverage.json
	$(MAKETOOLS) venv-embed cargo run --bin check_segment_coverage -- --threshold 100

# Python linting: ruff + mypy + guard, through the uv venv's own executables
# (``venv-exec`` runs them from .venv/bin).
python-lint:
	$(MAKETOOLS) uv-venv-check
	$(MAKETOOLS) venv-exec python -m scripts.guard
	$(MAKETOOLS) venv-exec ruff check . --fix
	$(MAKETOOLS) venv-exec ruff format .
	$(MAKETOOLS) venv-exec mypy tests scripts

# Python tests with coverage, through the shared launcher on the uv venv's
# interpreter; the per-run COVERAGE_FILE under runs/ and its scoped cleanup
# come with it.
python-test:
	$(MAKETOOLS) uv-venv-check
	$(MAKETOOLS) test --runner venv

# Lint: Rust + Python (guard runs during python-lint)
lint: rust-lint python-lint

# Test: Rust + Python
test: rust-test python-test

# Full check: lint, test, then Rust coverage
check:
	$(PYTHON) ../../tools/maketools/scripts/run.py check-budget
	@echo "=== ALL CHECKS PASSED ==="

_check-unbudgeted: lint test rust-cov
