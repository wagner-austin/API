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
API/
├── libs/                          # Shared libraries (22 packages)
│   ├── cleargbm/                  # Pure-Python gradient boosting (ClearGBM)
│   ├── cleargbm_rs/               # Rust core for ClearGBM (PyO3 bindings)
│   ├── covenant_domain/           # Loan covenant domain models & rule engine
│   ├── covenant_ml/               # Pluggable ML framework for covenants & climate
│   ├── covenant_nn/               # PyTorch neural network backends (MLP, LSTM)
│   ├── covenant_persistence/      # PostgreSQL repositories for covenant data
│   ├── instrument_io/             # IO for analytical chemistry data formats
│   ├── monorepo_guards/           # Guard rules for monorepo integrity
│   ├── platform_calendar/         # Google Calendar API for deadline tracking
│   ├── platform_codebase/         # Codebase capability detection & profiling
│   ├── platform_core/             # Typed event schemas & platform utilities
│   ├── platform_devpost/          # Devpost hackathon discovery & matching
│   ├── platform_discord/          # Discord integration helpers
│   ├── platform_email/            # Email integration (Outlook Graph API, Gmail)
│   ├── platform_kaggle/           # Kaggle competition discovery & matching
│   ├── platform_langid/           # Spoken language ID via Meta MMS-LID
│   ├── platform_ml/               # ML artifact handling (manifests, storage)
│   ├── platform_music/            # Music analytics library
│   ├── platform_stt/              # Speech-to-text (Whisper, chunking, merging)
│   ├── platform_translate/        # Text translation (Anthropic, OpenAI backends)
│   ├── platform_workers/          # Workers tooling (Redis helpers, RQ harness)
│   └── procart/                   # Procedural art core (neon visuals, HDR)
├── services/                      # API microservices (13 services)
│   ├── Art-Trainer/               # Image generation model training (LoRA)
│   ├── covenant-radar-api/        # Loan covenant monitoring & ML prediction
│   ├── data-bank-api/             # Central file storage for artifacts
│   ├── github-stats-api/          # GitHub stats SVG card generation
│   ├── grandma-api/               # Multi-language audio-to-English translation
│   ├── handwriting-ai/            # MNIST digit recognition (ResNet-18)
│   ├── Model-Trainer/             # LLM training & tokenizer service
│   ├── music-wrapped-api/         # Music analytics API
│   ├── opportunity-radar-api/     # Hackathon & competition discovery
│   ├── procart-api/               # Procedural art rendering orchestration
│   ├── qr-api/                    # QR code generation
│   ├── transcript-api/            # Video captions & speech-to-text
│   └── turkic-api/                # Turkic language processing
├── clients/                       # Client applications
│   ├── DiscordBot/                # Discord bot integrating all services
│   ├── TankpitBot/                # Tankpit.com WebSocket bot
│   └── RustedWarfareBot/          # Headless Rusted Warfare client (JVM agent + planner)
└── docs/                          # Monorepo documentation
```

---

## Shared Infrastructure

All services connect to shared infrastructure started via `make infra`:

| Component | Container Name | Port | Purpose |
|-----------|---------------|------|---------|
| Redis 7 | platform-redis | 6379 | Job queue, pub/sub, status tracking |
| PostgreSQL 16 | platform-postgres | 5432 | Covenant-radar persistence |
| Traefik 3 | traefik | 80, 8080 | API gateway + dashboard |
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
| `adapters/` | Spotify, Apple Music, YouTube Music, Last.fm API clients |
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

**Purpose:** Pluggable ML framework for covenant breach prediction and climate trend analysis.

**Key Components:**
| Module | Description |
|--------|-------------|
| `backends/` | Pluggable model backends (XGBoost, LightGBM, Random Forest, LogReg, ClearGBM) |
| `backends/regressor_registry.py` | Regression backend registry (LightGBM, XGBoost regressors) |
| `optimizer/` | Optuna-based hyperparameter optimization with TPE, grid, and random search |
| `ensemble/` | Weighted ensemble and regression ensemble optimization |
| `validation/` | Cross-validation runners, splitters, strategies (stratified k-fold, time series, etc.) |
| `calibration/` | Probability calibration with Platt scaling |
| `preprocessing/` | Feature preprocessing pipelines |
| `features.py` | Feature engineering for tabular data |
| `metrics.py` | Classification and regression metrics |
| `trainer.py` | Unified training orchestrator |
| `predictor.py` | Prediction interface |
| `datasets/` | Dataset loaders (CSV, Parquet, ARFF, time series, NetCDF temporal) |
| `datasets/loaders/_netcdf_temporal.py` | McKinnon PNAS 2024 temporal feature extraction (steps 1-3) |
| `datasets/loaders/_netcdf_trend_testing.py` | McKinnon rank-trend hypothesis testing (steps 4-7) |
| `datasets/types.py` | Temporal types, rank-trend types, heat metric constants |
| `explainers/` | Model explainability (permutation importance, ClearGBM SHAP) |

**Stats:** 1924 tests, 100% statement + branch coverage

**Used by:** covenant-radar-api

---

### covenant_nn

**Purpose:** PyTorch neural network backends for covenant breach prediction and regression, extracted from covenant_ml to isolate CUDA/torch dependencies.

**Key Components:**
| Module | Description |
|--------|-------------|
| `backends/mlp/backend.py` | MLP classifier (binary classification, early stopping on AUC) |
| `backends/mlp/regressor.py` | MLP regressor (continuous prediction, early stopping on RMSE) |
| `backends/lstm/backend.py` | LSTM classifier (temporal sequence classification) |
| `backends/lstm/regressor.py` | LSTM regressor (temporal sequence regression) |
| `backends/lstm/sequences.py` | Sequence building (sliding windows, entity grouping) |
| `objectives/` | Optuna objectives for all 4 backends |

**Features:**
- CUDA support with mixed precision (fp16/bf16)
- Early stopping on validation metrics (AUC for classifiers, RMSE for regressors)
- Class weighting for imbalanced classification
- Model serialization (.pt weights + .json metadata)
- Bidirectional LSTM support

**Dependencies:** `covenant_ml` (protocols, types, metrics), `platform_ml` (device selection), `torch` ^2.5

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

### cleargbm

**Purpose:** Pure-Python gradient boosting implementation with strict typing.

**Key Components:**
| Module | Description |
|--------|-------------|
| `histogram.py` | Histogram-based split finding |
| `tree.py` | Decision tree construction |
| `losses.py` | Loss functions (binary logistic, sigmoid) |
| `_test_hooks.py` | Protocol-based dependency injection for testing |

**Stats:** 486 tests, 100% coverage

**Used by:** covenant_ml (as a backend)

---

### cleargbm_rs

**Purpose:** High-performance Rust core for ClearGBM via PyO3 bindings.

**Key Components:**
| Module | Description |
|--------|-------------|
| `src/histogram/` | O(n) histogram building with NaN handling |
| `src/split/` | Split finding with monotonicity constraints |
| `src/tree/` | Tree construction with histogram subtraction trick |
| `src/predict/` | Ensemble prediction with sigmoid |
| `src/pyo3_module/` | PyO3 bindings (PyCFunction::new_closure pattern) |

**Stats:** 1122 tests (Rust), 100% segment coverage. All clippy lints at `forbid`.

**Used by:** cleargbm (Python integration pending)

---

### platform_email

**Purpose:** Email integration with OAuth 2.0 authentication for Outlook Graph API and Gmail.

**Key Components:**
| Module | Description |
|--------|-------------|
| `outlook/` | Microsoft Graph API client for sending and reading emails |
| `gmail/` | Gmail API client with OAuth 2.0 |
| `cli.py` | CLI for sending emails and managing OAuth tokens |
| `testing.py` | Production hook validation and test fakes |

**Used by:** Services requiring email notifications

---

### platform_stt

**Purpose:** Reusable speech-to-text library (Whisper API, audio chunking, parallel transcription).

**Key Components:**
| Module | Description |
|--------|-------------|
| `whisper_client.py` | OpenAI Whisper API client |
| `chunker.py` | Silence-based audio splitting via ffmpeg |
| `parallel.py` | Bounded-concurrency parallel transcription |
| `merger.py` | Segment merging with time offset correction |
| `langid.py` | FastText language detection |

**Used by:** transcript-api, grandma-api

---

### platform_langid

**Purpose:** Spoken language identification from audio using Meta's MMS-LID model (`facebook/mms-lid-4017`).

**Key Components:**
| Module | Description |
|--------|-------------|
| `detector.py` | `detect_spoken_language()` one-shot detection |
| `config.py` | `DetectorConfig` (model_id, device, confidence_threshold) |
| `types.py` | `SpokenLanguageResult` (ISO language code, confidence, model_id) |

**Features:**
- Returns ISO 639 language code with confidence score (0-1)
- Returns `"und"` (undetermined) when confidence is below threshold
- Audio resampled to 16000 Hz target sample rate
- CUDA device support

**Dependencies:** PyTorch, torchaudio, Transformers, soundfile, langcodes

**Used by:** grandma-api

---

### platform_translate

**Purpose:** Text translation with pluggable backends.

**Key Components:**
| Module | Description |
|--------|-------------|
| `translator.py` | `translate_text()` one-shot, `create_translator()` persistent |
| `config.py` | `TranslatorConfig` (backend, api_key, model) |
| `backends/` | Anthropic Claude (default), OpenAI |

**Features:**
- Default backend: Anthropic Claude (`claude-3-haiku-20240307`)
- Pluggable backend architecture for future DeepL / NLLB-200 support

**Dependencies:** `anthropic`, `openai`

**Used by:** grandma-api

---

### platform_calendar

**Purpose:** Google Calendar API integration for tracking competition deadlines with automatic reminders.

**Key Components:**
| Module | Description |
|--------|-------------|
| `client.py` | `CalendarClientProtocol` — list, get, create, update, delete events |
| `auth.py` | OAuth 2.0 PKCE authentication with auto token refresh |
| `tracking.py` | `TrackedCompetition` linked to calendar events |
| `cli.py` | Rich CLI (list, tomorrow, week, calendars, create, delete) |

**Features:**
- Multi-account support via environment variable prefixes
- Automatic reminders (1 day + 1 hour before deadline)
- 24+ injectable hooks covering all external I/O

**Dependencies:** `platform-core` (OAuth types, PKCE utilities)

**Used by:** opportunity-radar-api

---

### platform_codebase

**Purpose:** Codebase capability detection and profiling for monorepos.

**Key Components:**
| Module | Description |
|--------|-------------|
| `scanner.py` | `scan_libs()`, `scan_services()` — local filesystem scanning |
| `github.py` | `GitHubClient` — remote scanning for containerized deployments |
| `types.py` | `LibInfo`, `ServiceInfo`, `CodebaseProfile`, `CodebaseCapability` |
| `matching.py` | Strength levels (strong/moderate/basic), recommendation scoring |

**Used by:** github-stats-api, opportunity-radar-api

---

### platform_devpost

**Purpose:** Devpost hackathon discovery with codebase capability matching.

**Key Components:**
| Module | Description |
|--------|-------------|
| `client.py` | `DevpostClient` — list and get hackathons from Devpost API |
| `matching.py` | Score hackathons against `CodebaseProfile` |
| `filters.py` | Interest filtering by themes, states, featured-only |

**Dependencies:** `platform-core`, `platform-codebase`

**Used by:** opportunity-radar-api

---

### platform_kaggle

**Purpose:** Kaggle competition discovery with codebase capability matching.

**Key Components:**
| Module | Description |
|--------|-------------|
| `client.py` | `KaggleClient` — list and get competitions via official `kaggle` library |
| `matching.py` | Score competitions against `CodebaseProfile` |
| `profile.py` | Auto-detect capabilities from pyproject.toml dependencies |
| `pages.py` | `KagglePageFetcher` — fetch competition description, evaluation, timeline |

**Dependencies:** `platform-core`, `platform-codebase`, `kaggle`

**Used by:** github-stats-api, opportunity-radar-api

---

### procart

**Purpose:** Procedural art core library for generating looping neon visual scenes.

**Key Components:**
| Module | Description |
|--------|-------------|
| `scene.py` | `Scene` / `Layer` composition with camera and parallax |
| `modules/` | Pluggable `VisualModule` registry (background, neon_orbs, recursive_rects, fractal_mandelbrot, spiral_flow) |
| `hdr.py` | HDR linear RGBA render pipeline with exposure/gamma tone mapping |
| `camera.py` | Camera paths registry with typed `CameraConfig` builder |
| `schedule.py` | `ScheduleConfig` (constant | linear) for parameter evolution |
| `ffmpeg.py` | `FfmpegRunner` Protocol for video encoding |

**Features:**
- Supersampling and deterministic seeds to prevent frame drift
- `MathBackend` Protocol (NumPy default) — swappable math implementation
- PNG output via Pillow

**Dependencies:** NumPy, Pillow

**Used by:** procart-api

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

| Service | core | workers | ml | discord | music | stt | langid | translate | codebase | calendar | devpost | kaggle | procart |
|---------|:----:|:-------:|:--:|:-------:|:-----:|:---:|:------:|:---------:|:--------:|:--------:|:-------:|:------:|:-------:|
| Model-Trainer | ✓ | ✓ | ✓ | | | | | | | | | | |
| Art-Trainer | ✓ | ✓ | ✓ | | | | | | | | | | |
| handwriting-ai | ✓ | | ✓ | | | | | | | | | | |
| covenant-radar-api | ✓ | ✓ | ✓ | | | | | | | | | | |
| data-bank-api | ✓ | ✓ | | | | | | | | | | | |
| turkic-api | ✓ | ✓ | | | | | | | | | | | |
| transcript-api | ✓ | ✓ | | | | | | | | | | | |
| qr-api | ✓ | ✓ | | | | | | | | | | | |
| music-wrapped-api | ✓ | | | | ✓ | | | | | | | | |
| grandma-api | ✓ | | | | | ✓ | ✓ | ✓ | | | | | |
| github-stats-api | ✓ | | | | | | | | ✓ | | | ✓ | |
| opportunity-radar-api | ✓ | | | | | | | | ✓ | ✓ | ✓ | ✓ | |
| procart-api | ✓ | | | | | | | | | | | | ✓ |
| DiscordBot | ✓ | ✓ | | ✓ | | | | | | | | | |
| TankpitBot | ✓ | | | | | | | | | | | | |
| RustedWarfareBot | | | | | | | | | | | | | |

Every service and client additionally depends on `monorepo_guards` (not a
column above) for lint-rule enforcement. RustedWarfareBot is deliberately
standalone — `monorepo_guards` is its only in-repo dependency.

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
- Turkish, Kazakh, Uzbek, Kyrgyz, Russian, Azerbaijani, +3 more
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
- YouTube Music
- Last.fm

**Features:**
- Listening history aggregation
- Top artists/tracks/albums computation
- Yearly "wrapped" report generation

---

### covenant-radar-api

**Purpose:** Loan covenant monitoring and breach prediction with pluggable ML backends.

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
│   │   └── ml.py                 # Prediction, training, optimization, explanation
│   ├── decode.py                 # JSON request decoders
│   └── error_handlers.py         # Decode failures -> 400 responses
├── core/
│   ├── config.py                 # Settings re-export
│   ├── container.py              # ServiceContainer for DI
│   ├── model_paths.py            # Model path resolution under the models root
│   └── _test_hooks.py            # Container hooks for testing
├── worker/
│   ├── train_job.py              # Internal-data training job
│   ├── train_external_job.py     # External-dataset training job
│   ├── evaluate_job.py           # Deterministic evaluation job
│   ├── explain_job.py            # Explanation job
│   ├── optimize_job.py           # Hyperparameter optimization, all backends
│   ├── optimize_regression_job.py # Regression hyperparameter optimization
│   ├── _model_loaders.py         # Model deserialization per backend
│   ├── _explain_loaders.py       # Explainer construction per backend
│   ├── _optimize_common.py       # Shared optimization utilities
│   └── _test_hooks.py            # Worker DI hooks (registry, loaders, datasets)
├── worker_entry.py               # RQ worker entry point
├── streaming_worker_entry.py     # Kafka streaming worker entry point
└── _test_hooks.py                # Worker runner hooks
```

**Domain Libraries:**
| Library | Purpose |
|---------|---------|
| `covenant_domain` | TypedDict models, formula parser, rule engine |
| `covenant_ml` | Tree-based ML (XGBoost, LightGBM, Random Forest, LogReg, ClearGBM) |
| `covenant_nn` | Neural network ML (MLP, LSTM classifiers and regressors) |
| `covenant_persistence` | PostgreSQL repositories |

**Features:**
- CRUD operations for deals and covenants
- Deterministic covenant rule evaluation
- Pluggable ML backends: XGBoost, LightGBM, ClearGBM, LogReg, Random Forest (tree-based via covenant_ml); MLP, LSTM (neural via covenant_nn)
- Optuna hyperparameter optimization with Bayesian TPE
- Model explainability (permutation importance, SHAP, gradient, integrated gradients)
- External dataset training (Taiwan, US, Polish bankruptcy datasets)
- Temporal feature extraction (McKinnon PNAS 2024) and rank-trend hypothesis testing
- Background training and optimization jobs via RQ

**Dependencies:**
- `xgboost` ^3.1, `lightgbm`, `scikit-learn`
- `numpy` ^2.3
- `psycopg` ^3.3
- `platform-core`, `platform-workers`, `platform-ml`
- `covenant_ml`, `covenant_nn` (neural backends, optional torch dependency)

---

### Art-Trainer

**Purpose:** Image generation model training service (LoRAs for SD 1.5, SDXL, FLUX) with Kohya-ss backend.

**Features:**
- Multi-model support: SD 1.5, SDXL, FLUX LoRA training
- Style, character, and concept training presets
- Dataset upload + multi-backend auto-captioning (BLIP / Gemini / GPT-4o)
- Durable job queue with progress tracking, cancellation, retry
- ComfyUI automatic deployment of trained LoRAs
- Data-bank API integration for artifact upload/download

**Key Endpoints:**
```
POST /lora/train                 Enqueue LoRA training job
GET  /lora/{job_id}              Get job status
GET  /lora/{job_id}/progress     Get training progress
POST /lora/{job_id}/cancel       Cancel job
POST /dataset/upload             Upload images for training
GET  /dataset/{dataset_id}       Get dataset info
POST /dataset/{dataset_id}/caption  Caption images
```

**Dependencies:**
- `torch` (CUDA), `transformers`, `diffusers`, `peft`, `safetensors`
- `platform-core`, `platform-workers`, `platform-ml`

---

### grandma-api

**Purpose:** Multi-language audio-to-English translation API. Supports 57 input languages with automatic language detection.

**Architecture:**
- Python FastAPI backend + vanilla TypeScript browser frontend
- Audio recording in browser, real-time translation to English

**Features:**
- Translates speech from 57 languages to English text
- Automatic language detection (no language parameter needed)
- Supports WebM, MP3, WAV, M4A, OGG audio formats
- Token-based authentication (`API_TOKEN`)

**Key Endpoints:**
```
POST /translate    Translate audio to English text
```

**Dependencies:**
- `platform-core`, `platform-stt`, `platform-langid`, `platform-translate`

---

### github-stats-api

**Purpose:** GitHub statistics SVG card generation API (Python reimplementation of github-readme-stats).

**Features:**
- User stats cards and top languages cards as SVG
- Capabilities card, hero card, skills/tech stack card
- 10 themes: 5 basic + 5 premium animated (cyberpunk, synthwave, neon, aurora, radical)
- Premium themes: gradient backgrounds, pulsing glow, twinkling sparkles (CSS animations)
- Language card layouts: default, compact, donut, pie
- Configurable caching (default 30 min TTL)

**Key Endpoints:**
```
GET /api            Generate user stats SVG card
GET /api/top-langs  Generate top languages SVG card
```

**Dependencies:**
- `platform-core`, `platform-codebase`, `platform-kaggle`

---

### opportunity-radar-api

**Purpose:** Discover Kaggle competitions and Devpost hackathons matching the monorepo's codebase capabilities.

**Features:**
- Codebase scanning: detects ML backends, frameworks, technologies from pyproject.toml
- Dual scanning modes: local filesystem or GitHub API (for containers)
- Kaggle competition matching with scored recommendations
- Devpost hackathon matching with theme/state/featured filtering
- Recommendation levels: `strong_fit`, `good_fit`, `stretch`, `new_territory`

**Key Endpoints:**
```
GET /codebase/profile          Get capability profile
GET /codebase/libs             List monorepo libraries
GET /codebase/services         List monorepo services
GET /kaggle/competitions       Find matching competitions
GET /kaggle/competitions/{ref} Get competition by ref
GET /devpost/hackathons        Find matching hackathons
GET /devpost/hackathons/{id}   Get hackathon by ID
```

**Dependencies:**
- `platform-core`, `platform-codebase`, `platform-kaggle`, `platform-devpost`, `platform-calendar`

---

### procart-api

**Purpose:** Procedural art rendering orchestration via the `procart` library.

**Features:**
- Registry endpoints for discovering available visual modules, camera paths, and tone mappers
- Frame preview rendering (single frame)
- Full frame sequence rendering to disk
- Video encoding via ffmpeg (injected runner)

**Key Endpoints:**
```
GET  /registries/modules       List available visual modules
GET  /registries/camera-paths  List camera path types
GET  /registries/tone-mappers  List tone mapping types
POST /render/preview           Render a single frame preview
POST /render/frames            Render all frames to disk
POST /render/video             Encode frames to video via ffmpeg
```

**Dependencies:**
- `procart`, `platform-core`

---

## ML Capabilities Summary

| Service | ML Type | Framework | Model Architecture |
|---------|---------|-----------|-------------------|
| Model-Trainer | Sequence modeling | PyTorch | GPT-2, Char-LSTM |
| Art-Trainer | Image generation | PyTorch, Diffusers, Kohya-ss | SD 1.5, SDXL, FLUX LoRAs |
| handwriting-ai | Image classification | PyTorch | ResNet-18 |
| covenant-radar-api | Tabular classification | XGBoost, LightGBM, ClearGBM, LogReg, Random Forest | Tree-based + linear (covenant_ml) |
| covenant-radar-api | Neural classification/regression | PyTorch | MLP, LSTM (covenant_nn) |
| covenant-radar-api | Tabular regression | XGBoost, LightGBM | Tree-based regressors (covenant_ml) |
| grandma-api | Speech-to-text | OpenAI Whisper, Meta MMS-LID | Language detection + translation |

**Model-Trainer is NOT suitable for:**
- Tree-based models (XGBoost, LightGBM, Random Forest)
- Tabular data classification/regression
- Non-sequence ML tasks

**Reasoning:** Model-Trainer's contracts assume `torch.nn.Module` models with tokenization pipelines. The training loop is designed for gradient descent, not boosting iterations.

**covenant-radar-api** handles tabular ML:
- 7 classifier backends + 4 regressor backends via pluggable registry
- Standalone service with dedicated `covenant_ml` + `covenant_nn` libraries
- Uses `platform_workers` for background training and optimization jobs

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
| grandma-api | 8008 | |
| github-stats-api | 8009 | |
| opportunity-radar-api | 8010 | |
| Art-Trainer | 8011 | ✓ |
| procart-api | - | |

---

## GPU Support

Model-Trainer and Art-Trainer use NVIDIA GPU acceleration:

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
- If image generation → follow Art-Trainer pattern (Kohya-ss + diffusers)
- If tabular/tree-based ML → create standalone service using `platform_workers` + `platform_ml`
- If image classification → follow handwriting-ai pattern
- If speech/audio → use `platform_stt` + `platform_langid` (follow grandma-api pattern)
