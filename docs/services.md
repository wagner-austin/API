# Services

## Overview

| Service | Port | Description | GPU |
|---------|------|-------------|-----|
| data-bank-api | 8001 | Content-addressed file storage | No |
| Model-Trainer | 8005 | Language model training with LoRA/QLoRA/Unsloth, GPT-2, Char-LSTM | Yes |
| Art-Trainer | 8011 | Image generation model training (LoRA) | Yes |
| handwriting-ai | 8004 | MNIST digit recognition | No |
| turkic-api | 8000 | Turkic language NLP | No |
| transcript-api | 8003 | YouTube transcription | No |
| qr-api | 8002 | QR code generation | No |
| music-wrapped-api | 8006 | Music analytics | No |
| covenant-radar-api | 8007 | Loan covenant monitoring | No |
| grandma-api | 8008 | Multi-language audio-to-English translation | No |
| github-stats-api | 8009 | GitHub stats SVG card generation | No |
| opportunity-radar-api | 8010 | Hackathon & competition discovery | No |
| procart-api | - | Procedural art rendering | No |

---

## data-bank-api

Content-addressed file storage service with atomic writes and SHA256 hashing.

**Features:**
- Upload files via multipart form data
- Retrieve files by content hash
- Automatic deduplication
- Atomic writes (no partial files)

**Key Endpoints:**
```
POST /files          Upload file, returns hash
GET  /files/{hash}   Download file by hash
HEAD /files/{hash}   Check if file exists
```

**Start:**
```bash
make up-databank
```

**Docs:** [README](../services/data-bank-api/README.md) | [API](../services/data-bank-api/docs/api.md)

---

## Model-Trainer

GPT-2 and Char-LSTM model training with CUDA GPU support.

**Features:**
- Fine-tune GPT-2 on custom text corpora
- Train character-level LSTM models
- Background job processing via RQ
- Progress streaming via Redis pub/sub
- Model artifact packaging with manifests

**Key Endpoints:**
```
POST /jobs/train     Start training job
GET  /jobs/{id}      Get job status
GET  /jobs/{id}/logs Stream training logs
POST /generate       Generate text from trained model
```

**Requirements:**
- NVIDIA GPU with CUDA 12.4+
- NVIDIA Container Toolkit

**Start:**
```bash
make up-trainer
```

**Docs:** [README](../services/Model-Trainer/README.md) | [API](../services/Model-Trainer/docs/api.md) | [Design](../services/Model-Trainer/DESIGN.md)

---

## handwriting-ai

MNIST digit recognition with calibrated confidence scores.

**Features:**
- Recognize handwritten digits (0-9)
- Calibrated probability estimates
- Support for multiple image formats
- Batch prediction

**Key Endpoints:**
```
POST /predict        Predict digit from image
POST /predict/batch  Predict multiple images
```

**Start:**
```bash
make up-handwriting
```

**Docs:** [README](../services/handwriting-ai/README.md) | [API](../services/handwriting-ai/docs/api.md)

---

## turkic-api

Turkic language detection and IPA transliteration.

**Features:**
- Detect Turkic languages (Turkish, Kazakh, Uzbek, etc.)
- Transliterate to IPA (International Phonetic Alphabet)
- Support for multiple scripts (Latin, Cyrillic, Arabic)

**Key Endpoints:**
```
POST /detect         Detect language from text
POST /transliterate  Convert to IPA
```

**Start:**
```bash
make up-turkic
```

**Docs:** [README](../services/turkic-api/README.md) | [API](../services/turkic-api/docs/api.md) | [Design](../services/turkic-api/DESIGN.md)

---

## transcript-api

YouTube video transcription service.

**Features:**
- Extract transcripts from YouTube videos
- Support for auto-generated captions
- Multiple language support
- Timestamp extraction

**Key Endpoints:**
```
POST /transcripts    Get transcript for YouTube URL
```

**Start:**
```bash
make up-transcript
```

**Docs:** [README](../services/transcript-api/README.md) | [API](../services/transcript-api/docs/api.md)

---

## qr-api

QR code generation service.

**Features:**
- Generate QR codes from text/URLs
- Customizable size and error correction
- Multiple output formats (PNG, SVG)

**Key Endpoints:**
```
POST /generate       Generate QR code
```

**Start:**
```bash
make up-qr
```

**Docs:** [README](../services/qr-api/README.md) | [API](../services/qr-api/docs/api.md)

---

## music-wrapped-api

Music listening analytics aggregating data from streaming services.

**Features:**
- Spotify listening history analysis
- Apple Music integration
- YouTube Music integration
- Last.fm scrobble aggregation
- Yearly wrapped-style reports

**Key Endpoints:**
```
POST /connect/{service}  Connect streaming service
GET  /stats              Get listening statistics
GET  /wrapped/{year}     Generate yearly wrapped
```

**Start:**
```bash
make up-music
```

**Docs:** [README](../services/music-wrapped-api/README.md)

---

## covenant-radar-api

Loan covenant monitoring and breach prediction service with pluggable ML backends.

**Features:**
- CRUD operations for loan deals and covenant definitions
- Financial measurement ingestion
- Deterministic covenant rule evaluation
- Pluggable ML backends: XGBoost, LightGBM, Random Forest, Logistic Regression, ClearGBM (tree-based via covenant_ml); MLP, LSTM classifiers and regressors (neural via covenant_nn)
- Feature importance ranking for tree-based models (XGBoost, LightGBM, ClearGBM)
- Model explainability (permutation importance, ClearGBM SHAP)
- Temporal feature extraction (McKinnon PNAS 2024) and rank-trend hypothesis testing
- Optuna hyperparameter optimization with Bayesian TPE
- External dataset training (Taiwan, US, Polish bankruptcy datasets)
- Background training and optimization jobs via RQ

**Key Endpoints:**
```
POST /deals                  Create loan deal
GET  /deals                  List all deals
POST /covenants              Create covenant rule
POST /measurements           Add financial measurements
POST /evaluate               Evaluate covenants for period
POST /ml/predict             Predict breach probability
POST /ml/train               Train model on internal data (background job)
POST /ml/train-external      Train model on external datasets (background job)
POST /ml/optimize            Optimize hyperparameters with Optuna (background job)
POST /ml/explain             Run feature importance explanation (background job)
GET  /ml/jobs/{job_id}       Get training/optimization job status
GET  /ml/models/active       Get active model info
```

**Start:**
```bash
make up-covenant
```

**Docs:** [README](../services/covenant-radar-api/README.md) | [API](../services/covenant-radar-api/docs/api.md)

---

## Art-Trainer

Image generation model training (LoRAs for SD 1.5, SDXL, FLUX) with Kohya-ss backend.

**Features:**
- Multi-model support: SD 1.5, SDXL, FLUX LoRA training
- Style, character, and concept training presets
- Dataset upload + multi-backend auto-captioning (BLIP / Gemini / GPT-4o)
- Durable job queue with progress tracking, cancellation, retry
- ComfyUI automatic deployment of trained LoRAs

**Key Endpoints:**
```
POST /lora/train                    Enqueue LoRA training job
GET  /lora/{job_id}                 Get job status
GET  /lora/{job_id}/progress        Get training progress
POST /lora/{job_id}/cancel          Cancel job
POST /dataset/upload                Upload images for training
GET  /dataset/{dataset_id}          Get dataset info
POST /dataset/{dataset_id}/caption  Caption images
```

**Requirements:**
- NVIDIA GPU with CUDA 12.4+

**Start:**
```bash
make up-art-trainer
```

**Docs:** [README](../services/Art-Trainer/README.md)

---

## grandma-api

Multi-language audio-to-English translation API. Supports 57 input languages with automatic language detection.

**Features:**
- Translates speech from 57 languages to English text
- Automatic language detection (no language parameter needed)
- Supports WebM, MP3, WAV, M4A, OGG audio formats
- Companion browser frontend for audio recording + real-time translation

**Key Endpoints:**
```
POST /translate    Translate audio to English text
```

**Start:**
```bash
make up-grandma
```

**Docs:** [README](../services/grandma-api/README.md)

---

## github-stats-api

GitHub statistics SVG card generation API (Python reimplementation of github-readme-stats).

**Features:**
- User stats cards and top languages cards as SVG
- 10 themes: 5 basic + 5 premium animated (cyberpunk, synthwave, neon, aurora, radical)
- Language card layouts: default, compact, donut, pie
- Configurable caching (default 30 min TTL)

**Key Endpoints:**
```
GET /api            Generate user stats SVG card
GET /api/top-langs  Generate top languages SVG card
```

**Start:**
```bash
make up-github-stats
```

**Docs:** [README](../services/github-stats-api/README.md)

---

## opportunity-radar-api

Discover Kaggle competitions and Devpost hackathons matching the monorepo's codebase capabilities.

**Features:**
- Codebase scanning: detects ML backends, frameworks, technologies
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

**Start:**
```bash
make up-opportunity
```

**Docs:** [README](../services/opportunity-radar-api/README.md)

---

## procart-api

Procedural art rendering orchestration via the `procart` library.

**Features:**
- Registry endpoints for discovering available visual modules, camera paths, and tone mappers
- Frame preview rendering (single frame)
- Full frame sequence rendering to disk
- Video encoding via ffmpeg

**Key Endpoints:**
```
GET  /registries/modules       List available visual modules
GET  /registries/camera-paths  List camera path types
GET  /registries/tone-mappers  List tone mapping types
POST /render/preview           Render a single frame preview
POST /render/frames            Render all frames to disk
POST /render/video             Encode frames to video via ffmpeg
```

**Start:**
```bash
cd services/procart-api
poetry install --with dev
poetry run hypercorn src/procart_api/main:app --bind 0.0.0.0:8000
```

**Docs:** [README](../services/procart-api/README.md)

---

## Clients

### DiscordBot

Discord bot integrating all platform services.

**Features:**
- Slash commands for all services
- Real-time job progress updates
- Rich embeds for results
- Redis pub/sub event subscription

**Start:**
```bash
make up-discord
```

**Docs:** [README](../clients/DiscordBot/README.md)

---

### TankpitBot

Automated bot client for Tankpit.com browser game. Uses Playwright and Chrome DevTools Protocol (CDP) to capture and reverse-engineer the game's WebSocket protocol, with a durable HFSM AI system for autonomous tank control.

**Features:**
- Autonomous AI bot with durable HFSM — two mode owners (`HUNT`, `COLLECT`) plus an SPA-pinnable `UNSET` idle, with rank-derived readiness thresholds
- WebSocket traffic capture via Chrome DevTools Protocol
- XOR codec for message encoding/decoding (static + session keys)
- Complete wire coverage: every V-table message type has exactly one decoder, and encoders round-trip byte-identically against the capture archive
- A server twin (`sim/`) the production bot plays full sessions against — no browser, no live server
- Machine-checked physics: each `physics/` symbol is bound to a wiki claim and re-derived from the runs archive on every `make check`
- Long-running aiohttp + SSE bot service (`tankpit-bot-service`, port 27100) driven by a phone SPA, with MJPEG live view
- 30+ CLI entry points; the main ones are `tankpit-bot`, `tankpit-bot-service`, `tankpit-sniff`, `tankpit-sim-run`, `tankpit-audit`, `tankpit-shadow`, `tankpit-roundtrip`

**Docs:** [README](../clients/TankpitBot/README.md) — the client's own [wiki](../clients/TankpitBot/wiki/index.md) (6 hubs, 67 pages) is the source of truth for protocol and game mechanics

---

### RustedWarfareBot

Headless Rusted Warfare client. A Java agent inside the game's JVM dispatches orders and serialises simulation state; a Python package plans and evaluates. The game boots fully headless via `-nodisplay`, so there is no virtual framebuffer, no screen scraping, and no input synthesis.

**Features:**
- In-JVM Java agent (`agent/`) for order dispatch and state streaming
- Python planning and evaluation layer with doctrine files (`doctrines/`)
- Sweep harness for batch match evaluation
- Standing goal: measured 100% win rate against the built-in AI at Impossible and below, with champion matches watchable live

**Docs:** [README](../clients/RustedWarfareBot/README.md) — the client's own [wiki](../clients/RustedWarfareBot/wiki/index.md) is the source of truth for engine internals; claims are pinned to a game build because the jar is obfuscated
