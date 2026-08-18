SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

.PHONY: lint test check native-deps

# cleargbm_rs is a maturin (Rust) package and the only path dependency declared
# without develop = true, with its version frozen at 0.1.0. `poetry install`
# therefore considers it satisfied and never reinstalls it, so rebuilding the
# crate leaves this venv running whatever build it first saw. That is not a
# theoretical drift: it presented as `module 'cleargbm_rs' has no attribute
# 'py_gbm_model_to_json_rs'` at conftest import, which took the whole test
# suite down. This target reinstalls from the crate's built wheel whenever that
# wheel is newer than the installed package.
native-deps:
	@$$wheel = Get-ChildItem "../../libs/cleargbm_rs/target/wheels/*.whl" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1; if ($$null -eq $$wheel) { Write-Host "[native-deps] no cleargbm_rs wheel built; skipping" -ForegroundColor DarkGray } else { $$info = Get-ChildItem ".venv/Lib/site-packages/cleargbm_rs-*.dist-info" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1; if ($$null -ne $$info -and $$info.LastWriteTime -ge $$wheel.LastWriteTime) { Write-Host "[native-deps] cleargbm_rs up to date" -ForegroundColor DarkGray } else { Write-Host "[native-deps] reinstalling cleargbm_rs from $$($$wheel.Name)" -ForegroundColor Yellow; poetry run pip install --force-reinstall --no-deps $$wheel.FullName } }

lint: native-deps
	@if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) { Write-Host "ERROR: poetry not found. Install it with: pip install pipx && pipx install poetry" -ForegroundColor Red; exit 1 }
	@$$ErrorActionPreference = 'SilentlyContinue'; poetry run mypy --version | Out-Null; if (-not $$?) { Write-Host "[lint] Stale venv detected; removing..." -ForegroundColor Yellow; if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" } }
	poetry lock
	poetry sync --with dev
	if ((Test-Path ".\\scripts\\guard.py") -or (Test-Path ".\\scripts\\guard\\__main__.py")) { poetry run python -m scripts.guard; if ($$LASTEXITCODE -ne 0) { exit $$LASTEXITCODE } }
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy src tests scripts

test: native-deps
	@if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) { Write-Host "ERROR: poetry not found. Install it with: pip install pipx && pipx install poetry" -ForegroundColor Red; exit 1 }
	poetry lock
	poetry sync --with dev
	New-Item -ItemType Directory -Force "runs" | Out-Null
	$$covArgs = @("--cov-branch","--cov-report=term-missing"); $$cands = @("src","scripts"); foreach ($$c in $$cands) { if (Test-Path (Join-Path "." $$c)) { $$covArgs += "--cov=$$c" } }; $$env:COVERAGE_FILE = (Resolve-Path -LiteralPath ".").Path + "\runs\.coverage-" + [System.Guid]::NewGuid().ToString("N").Substring(0,8); $$code = 0; try { poetry run pytest -v @covArgs; $$code = $$LASTEXITCODE } finally { Get-ChildItem -Path "runs" -Filter ".coverage-*" -Force -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue; Remove-Item Env:\COVERAGE_FILE -ErrorAction SilentlyContinue }; if ($$code -ne 0) { exit $$code }

check: lint | test
