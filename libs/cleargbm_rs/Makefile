SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: lint test check rust-lint rust-test rust-cov python-lint python-test setup

# Setup: create venv and install dev dependencies with uv
setup:
	if (-not (Test-Path ".venv")) { uv venv }
	uv pip install -e ".[dev]"

# Rust linting: fmt + clippy
rust-lint:
	cargo fmt
	cargo clippy --all-targets --all-features -- -D warnings

# Rust tests
rust-test:
	cargo test --all-features

# Rust coverage (requires cargo-llvm-cov)
rust-cov:
	cargo llvm-cov --all-features --fail-under-lines 100

# Python linting: ruff + mypy + guard
python-lint:
	@$$ErrorActionPreference = 'SilentlyContinue'; .venv\Scripts\mypy --version | Out-Null; if (-not $$?) { Write-Host "[lint] Stale venv detected; removing..." -ForegroundColor Yellow; if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" }; uv venv; uv pip install -e ".[dev]" }
	if ((Test-Path ".\\scripts\\guard.py") -or (Test-Path ".\\scripts\\guard\\__main__.py")) { .venv\Scripts\python -m scripts.guard; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE } }
	.venv\Scripts\ruff check . --fix
	.venv\Scripts\ruff format .
	.venv\Scripts\mypy python tests scripts

# Python tests with coverage
python-test:
	@if (-not (Test-Path ".venv")) { uv venv; uv pip install -e ".[dev]" }
	$$covArgs = @("--cov-branch","--cov-report=term-missing"); $$cands = @("python","scripts"); foreach ($$c in $$cands) { if (Test-Path (Join-Path "." $$c)) { $$covArgs += "--cov=$$c" } }; .venv\Scripts\pytest -n auto -v @covArgs

# Lint (Rust only for this crate)
lint: rust-lint

# Test (Rust only for this crate)
test: rust-test

# Full check: lint, test, then coverage
check: lint test rust-cov
