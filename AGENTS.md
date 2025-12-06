# Repository Guidelines

## Configuration
- Codex CLI configuration is located at `C:\Users\austi\PROJECTS\API\config.toml`.
- Full permissions granted: `approval_policy = "never"`, `sandbox_mode = "danger-full-access"`.
- All code quality standards and type safety rules are enforced via configuration flags.

## Project Structure & Module Organization
- Monorepo with `clients/` (user-facing apps like `DiscordBot`), `libs/` (shared packages: `platform_core`, `platform_discord`, `platform_workers`, `monorepo_guards`), and `services/` (APIs such as `data-bank-api`, `qr-api`, `transcript-api`, `turkic-api`, `handwriting-ai`, `Model-Trainer`).
- Each project is self-contained: code in `src/`, tests in `tests/`, helpers in `scripts/`, schemas or assets alongside their module (for example, `platform_core/models`). Add new code inside the target project and mirror module paths in tests.

## Build, Test, and Development Commands
- Run from the project root you are changing (e.g., `cd services/qr-api`).
- `make lint`: removes stale venvs, runs guard checks (`python -m scripts.guard` when present), installs dev deps via Poetry, formats with Ruff, and type-checks with MyPy strict.
- `make test`: installs dev deps then runs `pytest -n auto -v` with branch coverage across `src` and `scripts`.
- `make check`: convenience target executing `lint` then `test`. Always run before opening a PR.

## Coding Style & Typing Rules
- Python 3.11, 4-space indents, Ruff line length 100; Ruff handles import ordering and formatting.
- Strict typing everywhere: no `Any`, `cast`, `type: ignore`, stubs, or `.pyi`. Dataclasses are banned in `src/`; use `TypedDict`/`Protocol` and concrete type aliases instead.
- Parse/validate at boundaries: JSON via `json.loads` into a typed alias then validate with internal `_decode*`/`_load_json*` helpers; TOML must be converted to `TypedDict` (no `typing.Any` or `type_checking` blocks); ASGI/request objects go through a minimal `Protocol` rather than `dict[str, Any]`.
- For dynamic imports, call `__import__`, fetch attributes with `getattr`, and annotate the variable with the target `Protocol` at assignment to avoid `Any`.
- Redis access goes through module-level helpers or a `Protocol`-typed client; never reference `Redis[Any]`.

## Testing Guidelines
- Tests live in `tests/` and use `test_*.py`. Prefer fixtures over inline data; avoid `print` (guarded).
- Coverage uses branch reporting; many packages set `fail_under = 100`. Match that bar for new code even if a package does not yet enforce it.
- Add unit and integration tests for every code path and error branch; no best-effort fallbacks or silent try/except.

## Commit & Pull Request Guidelines
- Commit subjects are imperative and may be prefixed with the package (e.g., `platform_core: add envelope validator`). Keep PRs small and focused.
- PR checklist: brief description, linked issue/ticket, commands run (`make check`), and evidence for user-visible changes (logs, screenshots).
- Document any new env vars in the project `README` or `.env.example`; keep dependencies pinned via `poetry.lock`.
