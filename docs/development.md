# Development

## Prerequisites

- Python 3.11+
- Poetry (dependency management)
- Docker + Docker Compose
- Make (via PowerShell on Windows)
- NVIDIA GPU + CUDA 12.4 (for Model-Trainer only)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/wagner-austin/API.git
cd API

# Start shared infrastructure (Redis + network)
make infra

# Start a specific service
make up-trainer      # Model-Trainer (GPU)
make up-databank     # data-bank-api
make up-handwriting  # handwriting-ai

# View running containers
make status
```

## Local Development

### Setting Up a Service

```bash
cd services/Model-Trainer

# Install dependencies (creates .venv)
poetry install --with dev

# Run checks
make check   # lint + test
make lint    # just linting
make test    # just tests
```

### Running Checks Across Everything

From the repository root:

```bash
make check   # Run lint + test on all libs/services/clients
make lint    # Lint only
make test    # Test only
```

## Code Standards

### Type Safety

All code uses strict mypy with no `Any` types:

```toml
[tool.mypy]
strict = true
disallow_any_generics = true
disallow_subclassing_any = true
```

### Test Coverage

100% test coverage is required:

```toml
[tool.coverage.report]
fail_under = 100
```

See [Testing Patterns](testing-patterns.md) for techniques to achieve 100% coverage without mocks.

### Linting

Ruff handles linting and formatting:

```bash
poetry run ruff check .
poetry run ruff format .
```

## Project Structure

```
API/
├── libs/                    # Shared libraries
│   ├── covenant_domain/     # Covenant domain models & rules
│   ├── covenant_ml/         # XGBoost training for covenants
│   ├── covenant_persistence/# PostgreSQL repositories
│   ├── instrument_io/       # Scientific formats
│   ├── monorepo_guards/     # Static analysis
│   ├── platform_core/       # Config, logging, HTTP
│   ├── platform_discord/    # Discord utilities
│   ├── platform_ml/         # ML artifacts
│   ├── platform_music/      # Music service adapters
│   └── platform_workers/    # RQ job infrastructure
├── services/                # Microservices
│   ├── covenant-radar-api/
│   ├── data-bank-api/
│   ├── handwriting-ai/
│   ├── Model-Trainer/
│   ├── music-wrapped-api/
│   ├── qr-api/
│   ├── transcript-api/
│   └── turkic-api/
├── clients/                 # Client applications
│   └── DiscordBot/
├── docs/                    # Documentation
├── docker-compose.yml       # Shared infrastructure
├── Makefile                 # Root orchestration
└── README.md
```

## Docker Workflow

### Starting Services

```bash
# Infrastructure first (required)
make infra

# Individual services
make up-databank
make up-trainer
make up-handwriting

# All services
make up-all

# Stop everything
make down
```

### Viewing Logs

```bash
# All infrastructure logs
make logs

# Specific service logs
cd services/Model-Trainer
docker compose logs -f
```

### Rebuilding

```bash
# Rebuild a specific service
cd services/Model-Trainer
docker compose up -d --build

# Clean rebuild (remove volumes)
make down
make clean
make up-all
```

## Adding a New Service

1. Create directory under `services/`:
   ```
   services/new-service/
   ├── src/new_service/
   │   ├── __init__.py
   │   └── main.py
   ├── tests/
   ├── Dockerfile
   ├── docker-compose.yml
   ├── pyproject.toml
   └── Makefile
   ```

2. Add path dependencies in `pyproject.toml`:
   ```toml
   [tool.poetry.dependencies]
   platform-core = { path = "../../libs/platform_core", develop = true }
   ```

3. Configure `docker-compose.yml` with external network:
   ```yaml
   networks:
     platform-network:
       external: true
   ```

4. Add to root `Makefile`:
   ```makefile
   up-newservice: infra
   	Set-Location services/new-service; docker compose up -d --build
   ```

5. Update `up-all` target to include new service.

## Adding a New Library

1. Create directory under `libs/`:
   ```
   libs/new-lib/
   ├── src/new_lib/
   │   └── __init__.py
   ├── tests/
   ├── pyproject.toml
   └── Makefile
   ```

2. Reference from services:
   ```toml
   [tool.poetry.dependencies]
   new-lib = { path = "../../libs/new-lib", develop = true }
   ```

## Environment Variables

Services use `.env` files (git-ignored). Copy from examples:

```bash
cp .env.example .env
```

Common variables:
- `REDIS_URL` - Redis connection string
- `DATA_BANK_URL` - data-bank-api endpoint
- `LOG_LEVEL` - Logging verbosity (DEBUG, INFO, WARNING, ERROR)

## Troubleshooting

### Poetry can't find dependencies

```bash
# Ensure you're in the right directory
cd services/Model-Trainer

# Reinstall
poetry install --with dev
```

### Docker network not found

```bash
# Start infrastructure first
make infra
```

### GPU not detected (Model-Trainer)

1. Verify NVIDIA Container Toolkit is installed
2. Check Docker GPU access: `docker run --rm --gpus all nvidia/cuda:12.4-base nvidia-smi`
3. Restart Docker daemon

### Tests failing with import errors

```bash
# Install in development mode
poetry install --with dev

# Run tests through poetry
poetry run pytest
```
