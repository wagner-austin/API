# Architecture

## System Overview

The API platform is a Python monorepo using FastAPI for HTTP services, RQ (Redis Queue) for background job processing, and Redis for both job queuing and pub/sub messaging.

```
                         +------------------+
                         |   Discord Bot    |
                         +--------+---------+
                                  |
                    +-------------+-------------+
                    |                           |
              +-----v-----+               +-----v-----+
              |  Redis    |               |  Services |
              | (pub/sub) |               | (FastAPI) |
              +-----+-----+               +-----+-----+
                    |                           |
              +-----v-----+               +-----v-----+
              | RQ Workers|               | data-bank |
              +-----------+               +-----------+
```

## Shared Infrastructure

All services connect to shared infrastructure started via `make infra`:

| Component | Container Name | Port | Purpose |
|-----------|---------------|------|---------|
| Redis | platform-redis | 6379 | Job queue, pub/sub, caching |
| Network | platform-network | - | Docker bridge for inter-service communication |

## Service Architecture

Each service follows a consistent pattern:

```
service-name/
├── src/service_name/
│   ├── __init__.py
│   ├── main.py          # FastAPI app entry point
│   ├── routes/          # API endpoints
│   ├── services/        # Business logic
│   ├── models/          # Pydantic models
│   └── workers/         # RQ job handlers (if applicable)
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── Makefile
```

### Standard Endpoints

All services expose:
- `GET /healthz` - Liveness probe (is the process running?)
- `GET /readyz` - Readiness probe (is the service ready to handle requests?)

### Configuration

Services use environment variables for configuration, managed via:
- `.env` files (local development, git-ignored)
- `.env.example` files (documented examples, committed)
- Docker Compose environment sections

## Shared Libraries

Libraries provide reusable functionality across services:

| Library | Purpose |
|---------|---------|
| platform_core | Config loading, logging, HTTP clients, FastAPI utilities |
| platform_workers | RQ job infrastructure, Redis connection utilities |
| platform_ml | ML artifact storage, model manifests, tarball packaging |
| platform_discord | Discord embed builders, Redis pub/sub event subscribers |
| platform_music | Music streaming service adapters (Spotify, Apple, Last.fm) |
| instrument_io | Scientific instrument data format readers |
| monorepo_guards | Static analysis rules for code quality |

Libraries are installed as path dependencies in each service's `pyproject.toml`:

```toml
[tool.poetry.dependencies]
platform-core = { path = "../../libs/platform_core", develop = true }
```

## Job Processing

Long-running tasks use RQ workers:

1. **Client** submits job via HTTP POST
2. **FastAPI** enqueues job to Redis
3. **RQ Worker** picks up and processes job
4. **Worker** publishes progress/completion to Redis pub/sub
5. **Subscribers** (Discord Bot, etc.) receive events

### Job Event Flow

```
Client → POST /jobs/train → Redis Queue → RQ Worker
                                              ↓
                                         Processing
                                              ↓
Discord Bot ← Redis Pub/Sub ← Job Events (progress, complete, error)
```

## Docker Networking

Services communicate over `platform-network`:

- **Internal**: Services reference each other by container name (e.g., `http://data-bank-api:8001`)
- **External**: Host machine accesses via localhost ports

### Port Assignments

| Service | Port |
|---------|------|
| turkic-api | 8000 |
| data-bank-api | 8001 |
| qr-api | 8002 |
| transcript-api | 8003 |
| handwriting-ai | 8004 |
| Model-Trainer | 8005 |
| music-wrapped-api | 8006 |

## Data Flow

### Content-Addressed Storage (data-bank-api)

All services store files through data-bank-api using content-addressed storage:

```
Service → POST /files (multipart) → data-bank-api → SHA256 hash → Filesystem
        ← { "hash": "abc123...", "size": 1234 }
```

Files are retrieved by hash, ensuring deduplication and integrity.

### ML Artifacts (platform_ml)

Trained models are packaged as tarballs with manifests:

```
model-artifact.tar.xz
├── manifest.json      # Metadata, hyperparameters, metrics
├── model.pt          # PyTorch weights
├── config.json       # Model configuration
└── vocab.json        # Tokenizer vocabulary (if applicable)
```

## GPU Support

Model-Trainer uses NVIDIA GPU acceleration:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Requires:
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- Docker configured for GPU access
