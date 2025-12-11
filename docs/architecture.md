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

---

## Directory Structure

```
C:\Users\austi\PROJECTS\API/
├── libs/                          # Shared libraries (10 packages)
│   ├── covenant_domain/           # Loan covenant domain models & rule engine
│   ├── covenant_ml/               # XGBoost training and prediction for covenants
│   ├── covenant_persistence/      # PostgreSQL repositories for covenant data
│   ├── instrument_io/             # IO for analytical chemistry data formats
│   ├── monorepo_guards/           # Guard rules for monorepo integrity
│   ├── platform_core/             # Typed event schemas & platform utilities
│   ├── platform_discord/          # Discord integration helpers
│   ├── platform_ml/               # ML artifact handling (manifests, storage)
│   ├── platform_music/            # Music analytics library
│   └── platform_workers/          # Workers tooling (Redis helpers, RQ harness)
├── services/                      # API microservices (8 services)
│   ├── covenant-radar-api/        # Loan covenant monitoring & ML prediction
│   ├── data-bank-api/             # Central file storage for artifacts
│   ├── handwriting-ai/            # MNIST digit recognition (ResNet-18)
│   ├── Model-Trainer/             # LLM training & tokenizer service
│   ├── music-wrapped-api/         # Music analytics API
│   ├── qr-api/                    # QR code generation
│   ├── transcript-api/            # Video captions & speech-to-text
│   └── turkic-api/                # Turkic language processing
├── clients/                       # Client applications
│   └── DiscordBot/                # Discord bot for user interaction
└── docs/                          # Monorepo documentation
```

---

## Shared Infrastructure

All services connect to shared infrastructure started via `make infra`:

| Component | Container Name | Port | Purpose |
|-----------|---------------|------|---------|
| Redis | platform-redis | 6379 | Job queue, pub/sub, caching |
| Network | platform-network | - | Docker bridge for inter-service communication |

---

## Shared Libraries (libs/)

Libraries provide reusable functionality across services. Each is installed as a path dependency in service `pyproject.toml` files:

```toml
[tool.poetry.dependencies]
platform-core = { path = "../../libs/platform_core", develop = true }
```

### platform_core

**Purpose:** Core platform utilities shared by all services.

**Key Components:**
| Module | Description |
|--------|-------------|
| `events/` | Typed event schemas (job events, training metrics, data-bank events) |
| `errors/` | Typed error codes and HTTP status mappers |
| `clients/data_bank.py` | `DataBankClient` for inter-service file exchange |
| `logging/` | JSON formatter, structured logging fields |
| `config/` | Environment-based configuration management |
| `http/` | HTTP client utilities and retry logic |
| `validators/` | Common validation functions |

**Used by:** All services, DiscordBot

---

### platform_ml

**Purpose:** ML artifact handling, model manifests, and experiment tracking.

**Key Components:**
| Module | Description |
|--------|-------------|
| `artifact_store.py` | Upload/download model tarballs via data-bank-api |
| `manifest.py` | `ModelManifestV2` schema with versioning |
| `wandb_publisher.py` | Weights & Biases integration for experiment tracking |
| `wandb_types.py` | W&B-specific type definitions |

**Artifact Structure:**
```
model-artifact.tar.xz
├── manifest.json      # Metadata, hyperparameters, metrics
├── model.pt          # PyTorch weights
├── config.json       # Model configuration
└── vocab.json        # Tokenizer vocabulary (if applicable)
```

**Used by:** Model-Trainer, handwriting-ai

---

### platform_workers

**Purpose:** Redis and RQ job infrastructure.

**Key Components:**
|     Module       | Description |
|------------------|-------------|
| `redis.py`       | Typed Redis client protocols |
| `job_context.py` | Job execution context with cancellation support |
| `job_store.py`   | Generic job status storage in Redis |
| `rq_harness.py`  | RQ worker wrapper with heartbeats |
| `testing.py`     | Test utilities (fakeredis fixtures) |

**Job Lifecycle:**
1. **Enqueue** - Service creates job payload, pushes to Redis queue
2. **Execute** - RQ worker picks up job, runs handler with context
3. **Heartbeat** - Worker periodically updates job status in Redis
4. **Complete** - Worker marks job done, publishes completion event

**Used by:** Model-Trainer, DiscordBot (for job status polling)

---

### platform_discord

**Purpose:** Discord integration helpers for the bot client.

**Key Components:**
| Module | Description |
|--------|-------------|
| `embeds/` | Rich embed builders for Discord messages |
| `subscribers/` | Redis pub/sub event handlers |
| `commands/` | Slash command registration helpers |

**Used by:** DiscordBot

---

### platform_music

**Purpose:** Music streaming service adapters and analytics.

**Key Components:**
| Module | Description |
|--------|-------------|
| `adapters/` | Spotify, Apple Music, Last.fm API clients |
| `models.py` | TypedDict-based track/album/artist types |
| `aggregation.py` | Listening statistics computation |
| `redis_cache.py` | Caching layer for API responses |

**Design Philosophy:** TypedDict and Protocol-only (no classes with mutable state).

**Used by:** music-wrapped-api

---

### instrument_io

**Purpose:** Scientific instrument data format readers and document writers.

**Key Components:**
| Module | Description |
|--------|-------------|
| `readers/` | Agilent, Thermo, Waters, mzML, Excel, PDF, DOCX, PPTX readers |
| `writers/` | ExcelWriter, WordWriter, PDFWriter for document generation |
| `types/` | TypedDicts for all data structures (spectra, chromatograms, documents) |
| `_protocols/` | Protocol definitions for openpyxl, python-docx, reportlab |

**Supported Formats:**
|    Format   | Read | Write | Library |
|-------------|:----:|:-----:|---------|
| Agilent .D  |   ✓  |       | rainbow-api |
| Thermo .raw |   ✓  |       | ThermoRawFileParser |
| Waters .raw |   ✓  |       | rainbow-api |
| mzML/mzXML  |   ✓  |       | pyteomics |
| Excel       |   ✓  |   ✓   | openpyxl, polars |
| Word        |   ✓  |   ✓   | python-docx |
| PDF         |   ✓  |   ✓   | pdfplumber, reportlab |
| PowerPoint  |   ✓  |       | python-pptx |

**Used by:** Standalone library (not currently used by services)

---

### monorepo_guards

**Purpose:** Static analysis rules for code quality enforcement.

**Key Components:**
| Module | Description |
|--------|-------------|
| `rules/` | Guard rules (no `Any`, no `cast`, etc.) |
| `checker.py` | AST-based rule enforcement |

**Used by:** CI/CD pipelines

---

### covenant_domain

**Purpose:** Loan covenant domain models, formula parser, and rule engine.

**Key Components:**
| Module | Description |
|--------|-------------|
| `models.py` | TypedDict models for deals, covenants, measurements |
| `parser.py` | Formula parser for covenant expressions |
| `engine.py` | Rule evaluation engine |

**Used by:** covenant-radar-api

---

### covenant_ml

**Purpose:** XGBoost training and prediction for covenant breach probability.

**Key Components:**
| Module | Description |
|--------|-------------|
| `trainer.py` | XGBoost model training |
| `predictor.py` | Breach probability prediction |
| `features.py` | Feature engineering for tabular data |

**Used by:** covenant-radar-api

---

### covenant_persistence

**Purpose:** PostgreSQL repositories for covenant data.

**Key Components:**
| Module | Description |
|--------|-------------|
| `repositories/` | Deal, covenant, measurement repositories |
| `connection.py` | Database connection management |

**Used by:** covenant-radar-api

---

## Dependency Graph

```
                    ┌─────────────────┐
                    │  platform_core  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         v                   v                   v
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│platform_workers│  │  platform_ml   │  │platform_discord│
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                   │                   │
        │         ┌─────────┴─────────┐         │
        │         │                   │         │
        v         v                   v         v
┌──────────────────────┐    ┌──────────────┐  ┌──────────┐
│    Model-Trainer     │    │ handwriting  │  │ Discord  │
│  (workers + ml)      │    │    -ai       │  │   Bot    │
└──────────────────────┘    └──────────────┘  └──────────┘
```

**Service → Library Dependencies:**

| Service | platform_core | platform_workers | platform_ml | platform_discord | platform_music |
|---------|:-------------:|:----------------:|:-----------:|:----------------:|:--------------:|
| Model-Trainer | ✓ | ✓ | ✓ | | |
| handwriting-ai | ✓ | | ✓ | | |
| covenant-radar-api | ✓ | ✓ | ✓ | | |
| data-bank-api | ✓ | ✓ | | | |
| turkic-api | ✓ | ✓ | | | |
| transcript-api | ✓ | ✓ | | | |
| qr-api | ✓ | ✓ | | | |
| music-wrapped-api | ✓ | | | | ✓ |
| DiscordBot | ✓ | ✓ | | ✓ | |

---

## Service Architecture Pattern

Each service follows a consistent structure:

```
service-name/
├── src/service_name/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── api/
│   │   ├── routes/          # HTTP endpoint handlers
│   │   ├── schemas/         # Request/response Pydantic models
│   │   └── validators/      # Request validation
│   ├── core/
│   │   ├── config/          # Settings and environment
│   │   ├── contracts/       # Protocol interfaces
│   │   └── services/        # Business logic implementations
│   └── worker/              # RQ job handlers (if applicable)
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

Services use environment variables for configuration:
- `.env` files (local development, git-ignored)
- `.env.example` files (documented examples, committed)
- Docker Compose environment sections

---

## Services In-Depth

### Model-Trainer

**Purpose:** Production-grade LLM training and tokenizer service.

**Architecture:**
```
src/model_trainer/
├── api/                          # FastAPI routes & schemas
│   ├── routes/
│   │   ├── health.py             # /healthz, /readyz
│   │   ├── runs.py               # Training/eval endpoints
│   │   └── tokenizers.py         # Tokenizer endpoints
│   └── schemas/                  # Pydantic request/response models
├── core/
│   ├── contracts/                # Protocol-based interfaces
│   │   ├── model.py              # ModelBackend, PreparedLMModel
│   │   ├── tokenizer.py          # TokenizerBackend, TokenizerHandle
│   │   └── queue.py              # Job payloads
│   └── services/
│       ├── model/
│       │   ├── backend_factory.py
│       │   └── backends/
│       │       ├── gpt2/         # GPT-2 implementation
│       │       └── char_lstm/    # Char-LSTM implementation
│       ├── tokenizer/
│       │   ├── bpe_backend.py
│       │   └── spm_backend.py
│       └── training/
│           └── base_trainer.py   # Unified training loop
├── orchestrators/                # High-level coordination
│   ├── training_orchestrator.py
│   └── inference_orchestrator.py
└── worker/                       # RQ job handlers
    ├── train_job.py
    ├── eval_job.py
    └── generate_job.py
```

**Model Backends:**
| Backend | Sizes | Capabilities |
|---------|-------|--------------|
| GPT-2 | tiny, small, medium, large | train, evaluate, score, generate |
| Char-LSTM | small | train, evaluate, score, generate |

**Tokenizer Backends:**
| Backend | Description |
|---------|-------------|
| BPE | Hugging Face Tokenizers library |
| SentencePiece | Optional, requires binaries |

**Key Design Patterns:**
- **Protocol-based contracts** - All backends implement `ModelBackend` protocol
- **Backend factory pattern** - `create_backend()` takes function dictionaries
- **Unified training loop** - `BaseTrainer` handles gradient descent, checkpointing, early stopping
- **Service container** - Explicit dependency injection via `ServiceContainer`

**Dependencies:**
- `torch` ^2.5 (CUDA 12.4)
- `transformers` ^4.45
- `tokenizers` ^0.20
- `platform-core`, `platform-workers`, `platform-ml`

---

### handwriting-ai

**Purpose:** MNIST digit recognition with calibrated confidence scores.

**Architecture:**
- ResNet-18 based classifier
- Temperature-scaled calibration for reliable confidence
- Batch prediction support

**ML Pipeline:**
```
Image → Preprocessing → ResNet-18 → Temperature Scaling → Calibrated Probabilities
```

**Dependencies:**
- `torch`
- `torchvision`
- `platform-core`, `platform-ml`

---

### data-bank-api

**Purpose:** Content-addressed file storage with SHA256 hashing.

**Storage Model:**
```
POST /files (multipart) → SHA256 hash → Filesystem
                        ← { "hash": "abc123...", "size": 1234 }
```

**Features:**
- Automatic deduplication (same content = same hash)
- Atomic writes (no partial files)
- File retrieval by content hash

**Used by:** All services store artifacts through data-bank-api

---

### turkic-api

**Purpose:** Turkic language NLP (detection and IPA transliteration).

**Supported Languages:**
- Turkish, Kazakh, Uzbek, Kyrgyz, Azerbaijani
- Multiple scripts: Latin, Cyrillic, Arabic

**Architecture:**
- Rule-based IPA transliteration engine
- Language detection via character frequency analysis

---

### transcript-api

**Purpose:** YouTube video transcription extraction.

**Features:**
- Auto-generated caption extraction
- Multiple language support
- Timestamp preservation

---

### qr-api

**Purpose:** QR code generation service.

**Output Formats:** PNG, SVG

---

### music-wrapped-api

**Purpose:** Music listening analytics and yearly wrapped reports.

**Streaming Services:**
- Spotify
- Apple Music
- Last.fm

**Features:**
- Listening history aggregation
- Top artists/tracks/albums computation
- Yearly "wrapped" report generation

---

### covenant-radar-api

**Purpose:** Loan covenant monitoring and breach prediction using XGBoost.

**Architecture:**
```
src/covenant_radar_api/
├── api/                          # FastAPI routes
│   ├── routes/
│   │   ├── health.py             # /healthz, /readyz
│   │   ├── deals.py              # CRUD for loan deals
│   │   ├── covenants.py          # CRUD for covenant rules
│   │   ├── measurements.py       # Financial metric ingestion
│   │   ├── evaluate.py           # Deterministic evaluation
│   │   └── ml.py                 # Prediction & training
│   └── decode.py                 # JSON request decoders
├── core/
│   ├── config.py                 # Settings re-export
│   ├── container.py              # ServiceContainer for DI
│   └── _test_hooks.py            # Container hooks for testing
├── worker/
│   ├── train_job.py              # XGBoost training job
│   └── evaluate_job.py           # Batch evaluation job
├── worker_entry.py               # RQ worker entry point
└── _test_hooks.py                # Worker runner hooks
```

**Domain Libraries:**
| Library | Purpose |
|---------|---------|
| `covenant_domain` | TypedDict models, formula parser, rule engine |
| `covenant_ml` | XGBoost training and prediction |
| `covenant_persistence` | PostgreSQL repositories |

**Features:**
- CRUD operations for deals and covenants
- Deterministic covenant rule evaluation
- XGBoost-based breach probability prediction
- Background training jobs via RQ

**Dependencies:**
- `xgboost` ^3.1
- `numpy` ^2.3
- `psycopg` ^3.3
- `platform-core`, `platform-workers`, `platform-ml`

---

## ML Capabilities Summary

| Service | ML Type | Framework | Model Architecture |
|---------|---------|-----------|-------------------|
| Model-Trainer | Sequence modeling | PyTorch | GPT-2, Char-LSTM |
| handwriting-ai | Image classification | PyTorch | ResNet-18 |
| covenant-radar-api | Tabular classification | XGBoost | Gradient boosted trees |

**Model-Trainer is NOT suitable for:**
- Tree-based models (XGBoost, LightGBM, Random Forest)
- Tabular data classification/regression
- Non-sequence ML tasks

**Reasoning:** Model-Trainer's contracts assume `torch.nn.Module` models with tokenization pipelines. The training loop is designed for gradient descent, not boosting iterations.

**covenant-radar-api** handles tabular ML:
- XGBoost gradient boosted trees for breach probability prediction
- Standalone service with dedicated `covenant_ml` library
- Uses `platform_workers` for background training jobs

---

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

### Job Types (Model-Trainer)

| Job Type | Description | Output |
|----------|-------------|--------|
| train | Train model on corpus | Model artifact tarball |
| evaluate | Compute loss/perplexity | Metrics JSON |
| generate | Generate text | Generated text |
| score | Token-level scoring | Surprisal values |
| tokenizer_train | Train BPE/SPM tokenizer | Tokenizer files |

---

## Docker Networking

Services communicate over `platform-network`:

- **Internal**: Services reference each other by container name (e.g., `http://data-bank-api:8001`)
- **External**: Host machine accesses via localhost ports

### Port Assignments

| Service | Port | GPU Required |
|---------|------|:------------:|
| turkic-api | 8000 | |
| data-bank-api | 8001 | |
| qr-api | 8002 | |
| transcript-api | 8003 | |
| handwriting-ai | 8004 | |
| Model-Trainer | 8005 | ✓ |
| music-wrapped-api | 8006 | |
| covenant-radar-api | 8007 | |

---

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

**Requirements:**
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- Docker configured for GPU access

---

## Type Safety Standards

The monorepo enforces extremely strict typing across all services - stricter than mypy's default `strict` mode.

**MyPy Configuration:**
```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_ignores = true
warn_redundant_casts = true
warn_unused_configs = true
disallow_subclassing_any = true
disallow_any_generics = true
no_implicit_optional = true
check_untyped_defs = true
no_implicit_reexport = true
show_error_codes = true
explicit_package_bases = true

# Maximum Any restrictions (beyond strict mode)
disallow_any_unimported = true
disallow_any_expr = true
disallow_any_decorated = true
disallow_any_explicit = true
```

**Ruff Banned APIs:**
```toml
[tool.ruff.lint.flake8-tidy-imports.banned-api]
"typing.Any" = { msg = "Do not use typing.Any; prefer precise types or Protocols/TypedDicts." }
"typing.cast" = { msg = "Do not use typing.cast; prefer adapters or precise types." }

[tool.ruff.lint.flake8-annotations]
allow-star-arg-any = false
```

**What This Means:**
- `disallow_any_expr` - Cannot use expressions that evaluate to `Any`
- `disallow_any_explicit` - Cannot write `Any` in annotations
- `disallow_any_decorated` - Decorators cannot produce `Any`
- `disallow_any_unimported` - Cannot use types from untyped modules
- `no_implicit_reexport` - Must explicitly re-export in `__init__.py`
- Ruff bans importing `Any` or `cast` at the lint level

**Enforced by:** MyPy strict + Ruff banned APIs + 100% coverage requirement

---

## Testing Standards

All services require:
- 100% statement coverage
- 100% branch coverage
- Pytest with xdist parallel execution

```bash
make check  # Runs mypy + ruff + pytest --cov --cov-branch
```

---

## Adding a New Service

1. Create service directory under `services/`
2. Initialize with standard structure (see Service Architecture Pattern)
3. Add `pyproject.toml` with platform library dependencies
4. Create `Makefile` with standard targets (`check`, `test`, `lint`)
5. Create `Dockerfile` and `docker-compose.yml`
6. Add port assignment to this document
7. Update `docs/services.md` with service documentation

**For ML Services:**
- If sequence/language modeling → extend Model-Trainer backends
- If tabular/tree-based ML → create standalone service using `platform_workers` + `platform_ml`
- If image classification → follow handwriting-ai pattern
