# API Services Monorepo

Production-grade microservices platform with strict type safety, 100% test coverage, and zero technical debt.

## Overview

This monorepo contains a suite of interconnected API services for language processing, machine learning, file storage, and utility operations. All services share common patterns, quality standards, and platform libraries.

## Architecture

```
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │                     Clients                                  │
                                    │  Discord Bot  •  Web UI  •  External Services               │
                                    └─────────────────────────────┬───────────────────────────────┘
                                                                  │
                                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           API Services Layer                                                 │
│                                                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │  handwriting-ai  │  │   turkic-api     │  │  Model-Trainer   │  │  transcript-api  │  │    qr-api      │ │
│  │                  │  │                  │  │                  │  │                  │  │                │ │
│  │  MNIST digit     │  │  Turkic corpus   │  │  LLM training    │  │  Video caption   │  │  QR code       │ │
│  │  recognition     │  │  processing &    │  │  & tokenizer     │  │  & speech-to-    │  │  generation    │ │
│  │  (inference)     │  │  transliteration │  │  service         │  │  text            │  │                │ │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  └────────────────┘ │
│           │                     │                     │                     │                               │
└───────────┼─────────────────────┼─────────────────────┼─────────────────────┼───────────────────────────────┘
            │                     │                     │                     │
            │                     ▼                     ▼                     │
            │           ┌─────────────────────────────────────────┐           │
            │           │           data-bank-api                 │           │
            │           │                                         │           │
            │           │  Central file storage for artifacts,    │           │
            │           │  corpus files, and model weights        │           │
            │           └─────────────────────────────────────────┘           │
            │                              │                                  │
            ▼                              ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         Infrastructure Layer                                                 │
│                                                                                                              │
│  ┌──────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────┐                   │
│  │         Redis            │  │    Persistent Volumes    │  │       Railway            │                   │
│  │  Job queues (RQ)         │  │  Artifacts, corpus,      │  │  Deployment platform     │                   │
│  │  Event pub/sub           │  │  model weights           │  │  Private networking      │                   │
│  │  Status tracking         │  │                          │  │                          │                   │
│  └──────────────────────────┘  └──────────────────────────┘  └──────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Services

| Service | Purpose | Port | Docs |
|---------|---------|------|------|
| [data-bank-api](./data-bank-api/) | Central file storage for service-to-service data exchange | 8000 | [README](./data-bank-api/README.md) • [DESIGN](./data-bank-api/DESIGN.md) |
| [handwriting-ai](./handwriting-ai/) | MNIST digit recognition with ResNet-18 inference | 8081 | [README](./handwriting-ai/README.md) |
| [Model-Trainer](./Model-Trainer/) | LLM training and tokenizer service (GPT-2, BPE) | 8000 | [README](./Model-Trainer/README.md) • [DESIGN](./Model-Trainer/DESIGN.md) |
| [qr-api](./qr-api/) | QR code generation via Segno | 8080 | [README](./qr-api/README.md) |
| [transcript-api](./transcript-api/) | Video captions (YouTube) and speech-to-text (OpenAI) | 8000 | [README](./transcript-api/README.md) |
| [turkic-api](./turkic-api/) | Turkic language corpus processing and IPA transliteration | 8000 | [README](./turkic-api/README.md) • [DESIGN](./turkic-api/DESIGN.md) |

### Service Details

#### data-bank-api
Central file storage service for internal service-to-service exchange. Provides streaming uploads/downloads with HTTP Range support, atomic writes, SHA256 checksums, and disk-space guards.

**Key Features:**
- Streaming uploads/downloads with Range support
- Atomic writes with temporary files
- Multi-key authentication (upload/read/delete permissions)
- SHA256 checksums for integrity verification

**Consumers:** turkic-api, Model-Trainer

---

#### handwriting-ai
Standalone HTTP service for MNIST digit inference. Uses a ResNet-18 backbone with temperature scaling and optional test-time augmentation.

**Key Features:**
- Multi-stage preprocessing (grayscale → Otsu → deskew → center → normalize)
- Temperature-scaled calibration
- Test-time augmentation (TTA) support
- Confidence-based uncertainty flagging

**Consumers:** Discord Bot (`/read` command)

---

#### Model-Trainer
Modular system for training and evaluating small language models and tokenizers on CPU. Features pluggable backends, durable job execution, and deterministic artifacts.

**Key Features:**
- Tokenizer backends: BPE (HF Tokenizers), SentencePiece
- Model backends: GPT-2 (Transformers)
- Redis + RQ job queue with heartbeats and cancellation
- Artifact management with manifests

**Integrations:** data-bank-api (artifact storage), Discord Bot (training notifications)

---

#### qr-api
Minimal, strictly-typed QR code generation service using Segno.

**Key Features:**
- URL validation and sanitization
- Configurable ECC levels (L/M/Q/H)
- Custom colors and sizing
- PNG output

---

#### transcript-api
Transcript service providing YouTube captions and OpenAI Whisper speech-to-text.

**Key Features:**
- YouTube caption extraction with language preferences
- OpenAI Whisper STT integration
- Audio chunking for long videos
- Parallel chunk processing

**Consumers:** Discord Bot

---

#### turkic-api
Production-grade REST API for Turkic language corpus processing with language detection and IPA transliteration.

**Key Features:**
- Corpus streaming from OSCAR and Wikipedia
- FastText language detection
- Rules-based IPA transliteration
- Support for 7 languages (Kazakh, Kyrgyz, Uzbek, Turkish, Uyghur, Finnish, Azerbaijani)

**Integrations:** data-bank-api (result storage)

---

## Shared Libraries

Located in `../libs/`:

| Library | Purpose | Used By |
|---------|---------|---------|
| [platform_core](../libs/platform_core/) | Centralized logging, error handling, config, health checks, JSON utilities | All services |
| [platform_workers](../libs/platform_workers/) | Redis/RQ job queue abstractions and worker harness | turkic-api, Model-Trainer, handwriting-ai |
| [platform_discord](../libs/platform_discord/) | Discord integration helpers and embed builders | handwriting-ai |
| [monorepo_guards](../libs/monorepo_guards/) | Code quality enforcement scripts | All services |

### platform_core Exports

```python
# Logging (structured JSON)
from platform_core.logging import setup_logging, get_logger

# Error handling
from platform_core.errors import AppError, ErrorCode

# FastAPI helpers
from platform_core.fastapi import (
    install_exception_handlers_fastapi,
    install_request_id_middleware,
    create_api_key_dependency,
)

# Health checks
from platform_core.health import healthz, HealthResponse, ReadyResponse

# Data bank client
from platform_core.data_bank_client import DataBankClient

# JSON utilities
from platform_core.json_utils import load_json_bytes, InvalidJsonError
```

### platform_workers Exports

```python
# Redis protocols
from platform_workers.redis import RedisStrProto

# RQ worker harness
from platform_workers.rq_harness import WorkerConfig, run_rq_worker
```

---

## Quality Standards

All services enforce **zero technical debt** policies:

### Type Safety
- `mypy --strict` with zero warnings
- **Zero `Any` types** (`disallow_any_expr = true`)
- **Zero type casts** (banned via Ruff)
- **Zero `type: ignore` comments**
- TypedDict models (no Pydantic in most services)
- Protocol-based dependency injection

### Test Coverage
- **100% statement coverage** required
- **100% branch coverage** required
- All tests must pass before merge

### Code Quality
- Guard scripts enforce patterns (no stdlib logging, no os.getenv, etc.)
- Ruff linting and formatting
- Structured JSON logging with correlation IDs
- Standardized error responses: `{code, message, request_id}`

### Guard Rules (enforced by `scripts/guard.py`)

| Rule | Description |
|------|-------------|
| No stdlib logging | Use `platform_core.logging` only |
| No Pydantic | Use TypedDict for data models |
| No dataclasses | Use TypedDict for data models |
| No os.getenv | Use `platform_core.config` |
| No subprocess | Security restriction |
| No type: ignore | Resolve at source |
| No relative imports | Use absolute imports in tests |

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **Poetry 1.8+** (package management)
- **Redis 7.0+** (for services with job queues)
- **Docker** (optional, for containerized development)

### Quick Start (Any Service)

```bash
# Navigate to a service
cd services/<service-name>

# Install dependencies
poetry install --with dev

# Run quality checks
make check

# Start the service
make serve  # or see service-specific README
```

### Development Commands

All services use a consistent Makefile interface:

```bash
make install      # Install dependencies
make install-dev  # Install with dev dependencies
make lint         # Run guards + ruff + mypy
make test         # Run pytest with coverage
make check        # Run lint + test (full quality gate)
make serve        # Start development server
make start        # Start with Docker
make stop         # Stop Docker containers
make clean        # Clean up containers/volumes
```

---

## Development Workflow

### Before Committing

```bash
# Run full quality checks
make check

# This runs:
# 1. Guard script validation
# 2. Ruff linting and formatting
# 3. Mypy strict type checking
# 4. Pytest with 100% coverage enforcement
```

### Adding a New Service

1. Create service directory under `services/`
2. Set up Poetry with platform library dependencies:
   ```toml
   [tool.poetry.dependencies]
   platform-core = { path = "../../libs/platform_core", develop = true }
   platform-workers = { path = "../../libs/platform_workers", develop = true }
   ```
3. Create `scripts/guard.py` using monorepo_guards
4. Configure mypy strict mode in `pyproject.toml`
5. Add Makefile with standard targets
6. Write README.md and DESIGN.md

### Platform Library Integration

```python
# Logging - always use platform_core
from platform_core.logging import setup_logging, get_logger

setup_logging(
    level="INFO",
    format_mode="json",
    service_name="my-service",
)
logger = get_logger(__name__)
logger.info("Starting service", extra={"port": 8000})

# Error handling
from platform_core.errors import AppError, ErrorCode

raise AppError(ErrorCode.NOT_FOUND, "Resource not found")

# Health checks
from platform_core.health import healthz

@app.get("/healthz")
def health() -> dict[str, str]:
    return healthz()
```

---

## Configuration

### Environment Variable Naming

Services use nested environment variable naming with double underscores:

```bash
# Pattern: SECTION__KEY=value
APP__PORT=8000
APP__DATA_ROOT=/data
REDIS_URL=redis://localhost:6379/0
SECURITY__API_KEY=secret
LOGGING__LEVEL=INFO
```

### Common Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP__PORT` or `PORT` | Service port | Service-specific |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `LOG_LEVEL` or `LOGGING__LEVEL` | Log level | `INFO` |
| `SECURITY__API_KEY` | API key (optional) | Empty (disabled) |

### Service-Specific Configuration

See each service's README for detailed configuration options.

---

## Deployment

### Railway (Production)

All services are configured for Railway deployment:

1. **Multi-stage Dockerfile** - Separate build and runtime stages
2. **Health checks** - `/healthz` for liveness, `/readyz` for readiness
3. **Private networking** - Services communicate via `*.railway.internal`
4. **Persistent volumes** - For artifacts and data storage

Example Railway setup:
```bash
# API service
railway up --service api

# Worker service (if applicable)
railway up --service worker
```

### Docker (Local)

```bash
# Build and run a service
cd services/<service-name>
docker build -t <service-name>:latest .
docker run -p 8000:8000 --env-file .env <service-name>:latest

# Or use docker-compose
docker-compose up -d
```

### Inter-Service Communication

Services communicate via private Railway network:

```
turkic-api → data-bank-api:
  URL: http://data-bank-api.railway.internal:8000

Model-Trainer → data-bank-api:
  URL: http://data-bank-api.railway.internal:8000
```

---

## API Standards

### Health Endpoints

All services implement:

```
GET /healthz → {"status": "ok"}
GET /readyz  → {"status": "ready", ...service-specific fields}
```

### Error Response Format

All services return consistent error responses:

```json
{
  "code": "NOT_FOUND",
  "message": "Resource not found",
  "request_id": "req-abc123"
}
```

### Request Correlation

All services support request correlation via `X-Request-ID` header:
- If provided, echoed in response and logs
- If absent, generated automatically (UUIDv4)

### Authentication

Services optionally require `X-API-Key` header when `SECURITY__API_KEY` is set.

---

## Testing

### Running Tests

```bash
# Run all tests with coverage
make test

# Run specific test file
poetry run pytest tests/test_api.py -v

# Run with coverage report
poetry run pytest --cov-report=html
```

### Test Patterns

- **Unit tests**: Fast, isolated, no external dependencies
- **Integration tests**: Service-to-service communication
- **Fixtures**: Use pytest fixtures, avoid mocking where possible
- **Parallel execution**: Tests run in parallel via pytest-xdist

### Coverage Requirements

```toml
[tool.coverage.report]
fail_under = 100
branch = true
```

---

## Repository Structure

```
API/
├── services/                    # Microservices (this directory)
│   ├── data-bank-api/
│   ├── handwriting-ai/
│   ├── Model-Trainer/
│   ├── qr-api/
│   ├── transcript-api/
│   ├── turkic-api/
│   └── README.md               # This file
├── libs/                        # Shared libraries
│   ├── platform_core/
│   ├── platform_workers/
│   ├── platform_discord/
│   └── monorepo_guards/
├── clients/                     # Client applications
│   └── DiscordBot/
└── docs/                        # Additional documentation
```

---

## Contributing

1. Follow the quality standards (100% coverage, strict typing)
2. Use platform libraries for logging, config, and errors
3. Run `make check` before committing
4. Write tests for all new functionality
5. Update documentation as needed

See [turkic-api/docs/CONTRIBUTING.md](./turkic-api/docs/CONTRIBUTING.md) for detailed guidelines.

---

## License

Apache-2.0
