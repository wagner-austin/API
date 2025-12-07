# API Platform

Typed Python monorepo for ML training, NLP, and media services. Strict mypy (no `Any`), 100% test coverage, FastAPI + RQ + Redis architecture.

## Services

| Service | Port | Description |
|---------|------|-------------|
| [data-bank-api](services/data-bank-api) | 8001 | Content-addressed file storage with atomic writes |
| [Model-Trainer](services/Model-Trainer) | 8005 | GPT-2 and Char-LSTM training with CUDA support |
| [handwriting-ai](services/handwriting-ai) | 8004 | MNIST digit recognition with calibrated confidence |
| [turkic-api](services/turkic-api) | 8000 | Turkic language detection and IPA transliteration |
| [transcript-api](services/transcript-api) | 8003 | YouTube video transcription |
| [qr-api](services/qr-api) | 8002 | QR code generation |
| [music-wrapped-api](services/music-wrapped-api) | 8006 | Music listening analytics (Spotify, Apple, Last.fm) |

## Clients

| Client | Description |
|--------|-------------|
| [DiscordBot](clients/DiscordBot) | Discord bot integrating all platform services |

## Shared Libraries

| Library | Description |
|---------|-------------|
| [platform_core](libs/platform_core) | Config, logging, HTTP clients, FastAPI utilities |
| [platform_workers](libs/platform_workers) | RQ job infrastructure, Redis utilities |
| [platform_ml](libs/platform_ml) | ML artifact storage, manifests, tarballs |
| [platform_discord](libs/platform_discord) | Discord embed builders, event subscribers |
| [platform_music](libs/platform_music) | Music analytics, streaming service adapters |
| [instrument_io](libs/instrument_io) | Scientific instrument data format readers |
| [monorepo_guards](libs/monorepo_guards) | Code quality rules (20+ static analysis checks) |

## Quick Start

```bash
# Start shared infrastructure (Redis + network)
make infra

# Start a specific service
make up-trainer      # Model-Trainer (GPU)
make up-databank     # data-bank-api
make up-handwriting  # handwriting-ai

# Start all services
make up-all

# Stop everything
make down

# Run checks across all libs/services
make check
make lint
make test

# View running containers
make status
```

## Architecture

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

All services share:
- `platform-network` (Docker bridge)
- `platform-redis` (job queue + pub/sub)
- Structured JSON logging
- Health endpoints (`/healthz`, `/readyz`)

## Development

```bash
# Install dependencies for a service
cd services/Model-Trainer
poetry install --with dev

# Run checks
make check

# Run just lint or test
make lint
make test
```

See [docs/](docs/) for detailed documentation.

## Requirements

- Python 3.11+
- Docker + Docker Compose
- Poetry
- Make (via PowerShell on Windows)
- NVIDIA GPU + CUDA 12.4 (for Model-Trainer)

## License

MIT
