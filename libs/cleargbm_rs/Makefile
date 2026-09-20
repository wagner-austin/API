include ../../scripts/make/shell.mk

MAKETOOLS := $(PYTHON) ../../tools/maketools/scripts/run.py

.PHONY: lint test check rust-lint rust-test rust-cov python-lint python-test setup

# Setup: create the uv venv and install the dev extras (a no-op when the
# venv already answers; tools/maketools, ``uv-venv-check``).
setup:
	$(MAKETOOLS) uv-venv-check

# Rust linting: fmt + clippy
rust-lint:
	cargo fmt
	cargo clippy --all-targets --all-features -- -D warnings

# Rust tests
rust-test:
	cargo test --all-features

# Rust coverage (requires cargo-llvm-cov) - 100% segment coverage
# Uses segment-based check to avoid false negatives from generic instantiations
rust-cov:
	cargo llvm-cov --all-features --json --output-path coverage.json
	cargo run --bin check_segment_coverage -- --threshold 100

# Python linting: ruff + mypy + guard, through the uv venv's own executables
# (``venv-exec`` resolves .venv/Scripts on Windows and .venv/bin elsewhere).
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
check: lint test rust-cov
	@echo "=== ALL CHECKS PASSED ==="
