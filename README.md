# API Platform

Typed Python monorepo for ML training, NLP, media services, and quant-ML risk modeling. Strict mypy (no `Any`, no `cast`, no `type: ignore`), 100% statement + branch test coverage, FastAPI + RQ + Redis + Kafka architecture.

## For recruiters — start here

If evaluating this repo, read these three (15 min total):

1. **[`libs/cleargbm/`](libs/cleargbm)** + **[`libs/cleargbm_rs/`](libs/cleargbm_rs)** — From-scratch interpretable gradient boosting: numpy Python orchestration with a Rust core (histogram building, tree construction, prediction pipeline) exposed via PyO3 bindings. Rare skill demo.
2. **[`services/covenant-radar-api/`](services/covenant-radar-api)** + **[`libs/covenant_ml/`](libs/covenant_ml)** + **[`libs/covenant_nn/`](libs/covenant_nn)** — Multi-model risk prediction: pluggable ML backends (XGBoost, LightGBM, ClearGBM, LogReg, RF) and PyTorch NN backends (MLP, LSTM), Optuna optimization, Kafka streaming. Quant-ML shop quality.
3. **[`libs/monorepo_guards/`](libs/monorepo_guards)** — Custom static-analysis framework: 20+ architecture-enforcement rules (Python + Rust) that lint invariants CI-time. Enforcement-as-code pattern.

Skip: hobby services (`grandma-api`, `github-stats-api`, `procart-api`) unless curious.

## Services

| Service | Port | Description |
|---------|------|-------------|
| [data-bank-api](services/data-bank-api) | 8001 | Content-addressed file storage with atomic writes |
| [Model-Trainer](services/Model-Trainer) | 8005 | Language model training with LoRA/QLoRA/Unsloth fine-tuning, GPT-2, Char-LSTM, CUDA support |
| [Art-Trainer](services/Art-Trainer) | 8011 | Image generation model training (LoRAs for SD 1.5, SDXL, FLUX) with Kohya-ss backend |
| [handwriting-ai](services/handwriting-ai) | 8004 | MNIST digit recognition with calibrated confidence |
| [turkic-api](services/turkic-api) | 8000 | Turkic language detection and IPA transliteration (Kazakh, Kyrgyz, Uzbek, Turkish, Russian, +4 more) |
| [transcript-api](services/transcript-api) | 8003 | YouTube video transcription |
| [qr-api](services/qr-api) | 8002 | QR code generation |
| [music-wrapped-api](services/music-wrapped-api) | 8006 | Music listening analytics (Spotify, Apple, YouTube Music, Last.fm) |
| [covenant-radar-api](services/covenant-radar-api) | 8007 | Multi-domain risk prediction with pluggable domains (Covenant, Weather/McKinnon), Kafka streaming (XGBoost, LightGBM, ClearGBM, LogReg, RF, MLP, LSTM) |
| [grandma-api](services/grandma-api) | 8008 | Multi-language audio-to-English translation (Whisper STT + language detection, GPT-4o-mini translation) |
| [github-stats-api](services/github-stats-api) | 8009 | GitHub stats SVG card generation |
| [opportunity-radar-api](services/opportunity-radar-api) | 8010 | Hackathon and competition discovery |
| [doc-extract-api](services/doc-extract-api) | 8012 | PDF text extraction with pdfplumber + docTR OCR fallback |
| [procart-api](services/procart-api) | - | Procedural art rendering orchestration |

## Clients

| Client | Description |
|--------|-------------|
| [DiscordBot](clients/DiscordBot) | Discord bot integrating all platform services |
| [TankpitBot](clients/TankpitBot) | Tankpit.com WebSocket protocol reverse-engineering and game bot |

## Shared Libraries

| Library | Description |
|---------|-------------|
| [platform_core](libs/platform_core) | Config, logging, HTTP clients, FastAPI utilities, OAuth 2.0 with PKCE, job event schemas |
| [platform_workers](libs/platform_workers) | RQ job infrastructure, Redis utilities |
| [platform_ml](libs/platform_ml) | ML artifact storage, manifests, device/precision auto-detection |
| [platform_discord](libs/platform_discord) | Discord embed builders, event subscribers |
| [platform_music](libs/platform_music) | Music analytics, streaming service adapters |
| [platform_email](libs/platform_email) | Email integration (Outlook Graph API, Gmail) with OAuth 2.0 |
| [platform_calendar](libs/platform_calendar) | Google Calendar API for competition deadline tracking with auto-reminders |
| [platform_codebase](libs/platform_codebase) | Codebase capability detection and profiling |
| [platform_devpost](libs/platform_devpost) | Devpost hackathon discovery + capability matching |
| [platform_kaggle](libs/platform_kaggle) | Kaggle competition discovery + capability matching |
| [platform_stt](libs/platform_stt) | Speech-to-text with OpenAI Whisper API, chunking, language detection |
| [platform_langid](libs/platform_langid) | Spoken language identification from audio using Meta MMS-LID |
| [platform_translate](libs/platform_translate) | Text translation with pluggable backends (Anthropic, OpenAI) |
| [instrument_io](libs/instrument_io) | Scientific instrument data format readers and writers (mass spec, mzML, Excel, PDF) |
| [monorepo_guards](libs/monorepo_guards) | Code quality rules (20+ static analysis checks, Python + Rust) |
| [covenant_domain](libs/covenant_domain) | Loan covenant domain models and rule engine |
| [covenant_ml](libs/covenant_ml) | Classification and regression backends (XGBoost, LightGBM, ClearGBM, LogReg, RF) with Optuna, temporal/NetCDF features |
| [covenant_nn](libs/covenant_nn) | PyTorch neural network backends (MLP, LSTM) for classification and regression |
| [covenant_persistence](libs/covenant_persistence) | PostgreSQL repositories for covenant data |
| [cleargbm](libs/cleargbm) | From-scratch interpretable gradient boosting (numpy only, Rust-accelerated via hooks) |
| [cleargbm_rs](libs/cleargbm_rs) | Rust core for ClearGBM: histogram building, tree construction, prediction pipeline, PyO3 bindings |
| [procart](libs/procart) | Procedural art core (neon visuals, HDR pipeline) |

## Quick Start

```bash
# Start shared infrastructure (Redis + PostgreSQL + Traefik)
make infra

# Start a specific service
make up-databank      # data-bank-api
make up-trainer       # Model-Trainer (GPU)
make up-art-trainer   # Art-Trainer (GPU)
make up-handwriting   # handwriting-ai
make up-qr            # qr-api
make up-transcript    # transcript-api
make up-turkic        # turkic-api
make up-music         # music-wrapped-api
make up-covenant      # covenant-radar-api
make up-grandma       # grandma-api
make up-github-stats  # github-stats-api
make up-opportunity   # opportunity-radar-api
make up-discord       # DiscordBot

# Start all services
make up-all

# Stop everything
make down

# Stop all services and prune Docker volumes
make clean

# Run checks across all libs/services
make check
make lint
make test

# View running containers
make status

# Stream logs
make logs
```

## Architecture

```
                              +--------------+
                              | Traefik (80) |
                              +------+-------+
                                     |
                         +-----------+-----------+
                         |                       |
                   +-----v-----+           +-----v-----+
                   | Discord   |           |  Services |
                   |   Bot     |           | (FastAPI) |
                   +-----+-----+           +-----+-----+
                         |                       |
           +-------------+-------------+---------+
           |             |             |
     +-----v-----+ +-----v-----+ +-----v-----+
     |   Redis   | | PostgreSQL| | data-bank |
     | (pub/sub) | | (covenant)| | (storage) |
     +-----+-----+ +-----------+ +-----------+
           |
     +-----v-----+
     | RQ Workers|
     +-----------+
```

All services share:
- `platform-network` (Docker bridge)
- `platform-redis` (job queue + pub/sub for async services)
- `platform-postgres` (covenant-radar persistence)
- Structured JSON logging
- Health endpoints (`/healthz`, `/readyz`)

## Infrastructure

| Component | Port | Purpose |
|-----------|------|---------|
| Redis 7 | 6379 | Job queue, pub/sub, status tracking |
| PostgreSQL 16 | 5432 | Covenant-radar persistence |
| Traefik 3 | 80, 8080 | API gateway + dashboard |

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

## Code Standards

- **Type Safety**: mypy strict mode, zero `Any` types, zero `cast`, zero `type: ignore`
- **Test Coverage**: 100% statement + branch coverage enforced
- **Linting**: Ruff formatting and linting
- **Architecture**: Protocol-based dependency injection, `_test_hooks.py` pattern

## Requirements

- Python 3.11+
- Docker + Docker Compose
- Poetry
- Make (via PowerShell on Windows)
- NVIDIA GPU + CUDA 12.4 (for Model-Trainer, Art-Trainer)

## License

MIT
