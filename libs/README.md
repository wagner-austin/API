# libs/

Shared Python libraries for the monorepo. Each library is a standalone Poetry package with strict typing, 100% test coverage, and no external state.

## Libraries

| Library | Description | Dependencies |
|---------|-------------|--------------|
| [covenant-domain](./covenant_domain) | Pure business logic for covenant monitoring | None |
| [covenant-ml](./covenant_ml) | Pluggable ML backends, Optuna optimizers, DART boosting, feature explainers | covenant-domain, platform-ml, xgboost, lightgbm, torch, optuna |
| [covenant-persistence](./covenant_persistence) | PostgreSQL repository layer for covenants | covenant-domain, psycopg |
| [monorepo-guards](./monorepo_guards) | Code quality enforcement rules | None |
| [platform-core](./platform_core) | Shared utilities: errors, validation, logging, service config | FastAPI, httpx |
| [platform-ml](./platform_ml) | ML artifacts, manifests, device selection, explainers (SHAP, permutation, gradient) | platform-core, shap, torch |
| [platform-workers](./platform_workers) | Redis/RQ background job processing | platform-core, redis, rq |
| [cleargbm](./cleargbm) | From-scratch interpretable gradient boosting (pure stdlib) | None |
| [platform-calendar](./platform_calendar) | Google Calendar API for competition deadlines | platform-core |
| [platform-codebase](./platform_codebase) | Codebase capability detection and profiling | platform-core |
| [platform-devpost](./platform_devpost) | Devpost hackathon discovery + capability matching | platform-core, platform-codebase, httpx |
| [platform-kaggle](./platform_kaggle) | Kaggle competition discovery + capability matching | platform-core, platform-codebase, kaggle |
| [platform-stt](./platform_stt) | Speech-to-text with Whisper, chunking, language detection | platform-core, numpy, openai, fasttext-wheel |

instrument_io, platform_discord, platform_email, platform_music and procart were archived on 2026-11-01 and live on at the tag `archive/api-packages-2026-11-01`.

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
