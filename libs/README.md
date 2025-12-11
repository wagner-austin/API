# libs/

Shared Python libraries for the monorepo. Each library is a standalone Poetry package with strict typing, 100% test coverage, and no external state.

## Libraries

| Library | Description | Dependencies |
|---------|-------------|--------------|
| [covenant-domain](./covenant_domain) | Pure business logic for covenant monitoring | None |
| [covenant-ml](./covenant_ml) | XGBoost wrapper for breach risk prediction | covenant-domain, xgboost, sklearn |
| [covenant-persistence](./covenant_persistence) | PostgreSQL repository layer for covenants | covenant-domain, psycopg |
| [instrument-io](./instrument_io) | Readers/writers for analytical chemistry formats | rainbow-api, pyteomics, openpyxl |
| [monorepo-guards](./monorepo_guards) | Code quality enforcement rules | None |
| [platform-core](./platform_core) | Shared utilities: errors, validation, logging | FastAPI, httpx |
| [platform-discord](./platform_discord) | Discord bot helpers and embed utilities | discord.py, platform-workers |
| [platform-ml](./platform_ml) | ML artifact storage and manifest schemas | platform-core |
| [platform-music](./platform_music) | Music analytics for Music Wrapped | None |
| [platform-workers](./platform_workers) | Redis/RQ background job processing | platform-core, redis, rq |

## Design Principles

All libraries follow these conventions:

- **Strict typing**: No `Any`, `cast`, `type: ignore`, or `.pyi` stubs
- **TypedDict and Protocol only**: No dataclasses or classes with mutable state
- **100% test coverage**: Statement and branch coverage enforced
- **Parse at edges**: Validate/decode at boundaries, pure logic inside
- **No global state**: All dependencies passed explicitly

## Development

Each library has a Makefile with standard targets:

```bash
make lint   # Run guard checks, ruff, mypy
make test   # Run pytest with coverage
make check  # Run both lint and test
```

## Adding a New Library

1. Create directory: `libs/my_library/`
2. Initialize Poetry: `poetry init`
3. Add standard structure:
   ```
   libs/my_library/
   ├── src/my_library/
   │   └── __init__.py
   ├── tests/
   │   └── conftest.py
   ├── scripts/
   │   └── guard.py
   ├── pyproject.toml
   ├── Makefile
   └── README.md
   ```
4. Configure `pyproject.toml` with strict mypy and ruff settings
5. Add to monorepo root `pyproject.toml` if needed

## Requirements

- Python 3.11+ (some libraries require 3.12+)
- Poetry for dependency management
