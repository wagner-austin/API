# Art-Trainer

Service for training image generation models (LoRAs, IP-Adapter, etc.) for SD 1.5, SDXL, and FLUX. Features dataset management with multi-backend auto-captioning, Kohya-ss training integration, durable job execution via Redis + RQ, and automatic deployment to ComfyUI.

## Features

- **Multi-Model Support**: Train LoRAs for Stable Diffusion 1.5, SDXL, and FLUX architectures
- **Training Types**: Style, character, and concept training with optimized presets
- **Dataset Management**: Upload images, auto-generate captions, manage training datasets
- **Multi-Backend Captioning**: BLIP (local), Google Gemini, and OpenAI GPT-4o for caption generation
- **Kohya-ss Integration**: Production-grade LoRA training via Kohya-ss/sd-scripts
- **Durable Jobs**: Redis + RQ with progress tracking, cancellation, retry logic
- **ComfyUI Deployment**: Automatic deployment of trained LoRAs to ComfyUI models directory
- **Data-Bank Integration**: Upload/download datasets and artifacts via data-bank API
- **Type Safety**: mypy strict mode, zero `Any` types, Protocol-based DI via `_test_hooks.py` pattern
- **100% Test Coverage**: Statements and branches

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+
- Docker Desktop (for Redis and containerized deployment)
- NVIDIA GPU with CUDA support (for training)

### Installation

```bash
cd services/Art-Trainer
poetry install --with dev
```

### Start with Docker

From the repo root (`C:\Users\Test\PROJECTS\API`):

```bash
# Start infra (Redis) + Art-Trainer (API + Worker)
make up-art-trainer

# Verify
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz

# Stop all services
make down
```

### Local Development (without Docker)

```bash
# Start Redis via infra from repo root
make infra

# Run API (from services/Art-Trainer)
poetry run hypercorn 'art_trainer.api.main:create_app()' --bind 0.0.0.0:8000

# Run Worker (separate terminal)
poetry run arttrainer-rq-worker
```

## API Reference

For complete API documentation, see [docs/api.md](./docs/api.md).

### Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/readyz` | GET | Readiness probe (checks Redis + workers) |
| `/lora/train` | POST | Enqueue LoRA training job |
| `/lora/{job_id}` | GET | Get job status |
| `/lora/{job_id}/progress` | GET | Get training progress |
| `/lora/{job_id}/cancel` | POST | Request job cancellation |
| `/datasets/upload` | POST | Upload images for training |
| `/datasets/{dataset_id}` | GET | Get dataset info |
| `/datasets/{dataset_id}/caption` | POST | Caption images with BLIP/Gemini/OpenAI |

---

## Configuration

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APP_ENV` | string | `dev` | Environment (`dev` or `prod`) |
| `LOGGING__LEVEL` | string | `INFO` | Log level |
| `REDIS__ENABLED` | bool | `true` | Enable Redis connection |
| `REDIS__URL` | string | `redis://redis:6379/0` | Redis connection URL |
| `RQ__QUEUE_NAME` | string | `art-trainer` | RQ queue name |
| `RQ__JOB_TIMEOUT_SEC` | int | `86400` | Job timeout (24h) |
| `RQ__RESULT_TTL_SEC` | int | `86400` | Result retention (24h) |
| `RQ__FAILURE_TTL_SEC` | int | `604800` | Failure retention (7d) |
| `RQ__RETRY_MAX` | int | `1` | Max retry attempts |
| `RQ__RETRY_INTERVALS_SEC` | string | `300` | Retry interval in seconds |
| `APP__DATA_ROOT` | string | `/data` | Base data directory |
| `APP__OUTPUT_ROOT` | string | `/data/output` | Training outputs directory |
| `APP__LOGS_ROOT` | string | `/data/logs` | Logs directory |
| `APP__KOHYA_SS_PATH` | string | `/opt/kohya_ss` | Path to Kohya-ss installation |
| `APP__COMFYUI_LORA_PATH` | string | `/opt/ComfyUI/models/loras` | ComfyUI LoRA models directory |
| `API_GATEWAY_URL` | string | - | Gateway base URL. When set, the data-bank URL becomes `$API_GATEWAY_URL/data-bank` and `APP__DATA_BANK_API_URL` is ignored |
| `APP__DATA_BANK_API_URL` | string | - | Data-bank API URL, used only when `API_GATEWAY_URL` is unset |
| `APP__DATA_BANK_API_KEY` | string | - | Data-bank API key |
| `APP__BLIP_MODEL_NAME` | string | `Salesforce/blip-image-captioning-base` | BLIP model for captioning |
| `APP__CAPTION_TRIGGER_WORD` | string | `sks person` | Default trigger word |
| `GEMINI_API_KEY` | string | - | Google Gemini API key for captioning |
| `OPENAI_API_KEY` | string | - | OpenAI API key for captioning |
| `SECURITY__API_KEY` | string | - | Optional API key for authentication |
| `REDIS_URL` | string | - | Required by `arttrainer-rq-worker`. The API reads `REDIS__URL`; the worker entry point reads this one |

### Example .env

```bash
APP_ENV=dev
LOGGING__LEVEL=DEBUG
REDIS__URL=redis://redis:6379/0
RQ__QUEUE_NAME=art-trainer
APP__DATA_ROOT=./data
APP__OUTPUT_ROOT=./data/output
APP__KOHYA_SS_PATH=/opt/kohya_ss
APP__COMFYUI_LORA_PATH=/opt/ComfyUI/models/loras
APP__DATA_BANK_API_URL=http://localhost:8001
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key
```

---

## Job Execution

### Queue Architecture

```
+------------------+
|    FastAPI       |
|    API Server    |
+--------+---------+
         | enqueue
         v
+------------------+     +------------------+
|     Redis        |<----|   RQ Worker      |
|   Job Queue      |     |                  |
|                  |     |  - Training      |
|  lora:hb:{id}    |     |  - Captioning    |
|  lora:{id}:stat  |     |  - Deployment    |
+------------------+     +------------------+
```

### Progress Tracking

Workers emit progress to Redis during training:
- Current phase (preparing, training, saving, uploading)
- Training step and total steps
- Current loss value
- Learning rate
- Timestamp of last update

### Cancellation

Set `lora:<job_id>:cancelled=1` in Redis to request cancellation. Worker checks this flag between training steps and performs graceful shutdown.

---

## Captioning Backends

Art-Trainer supports multiple backends for auto-generating image captions:

### BLIP (Local)

Local inference using Salesforce BLIP models. No API key required.

```json
{
  "backend": "blip",
  "model_name": "Salesforce/blip-image-captioning-large",
  "trigger_word": "sks person"
}
```

### Google Gemini

Cloud-based captioning using Google's Gemini models. Requires `GEMINI_API_KEY`.

```json
{
  "backend": "gemini",
  "model_name": "gemini-2.0-flash",
  "trigger_word": "sks person"
}
```

### OpenAI GPT-4o

Cloud-based captioning using OpenAI's vision models. Requires `OPENAI_API_KEY`.

```json
{
  "backend": "openai",
  "model_name": "gpt-4o",
  "trigger_word": "sks person"
}
```

---

## Development

### Commands

```bash
make check       # Run lint + typecheck + tests
make lint        # Run ruff + mypy
make test        # Run pytest with coverage
```

### Quality Gates

All code must pass:

1. **Ruff**: Linting and formatting
2. **Mypy**: Strict type checking
3. **Pytest**: 100% statement and branch coverage

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
poetry run pytest tests/core/services/captioning/test_gemini_backend.py -v

# Run with coverage report
poetry run pytest --cov-report=html
```

---

## Project Structure

```
Art-Trainer/
+-- src/art_trainer/
|   +-- api/
|   |   +-- main.py             # App factory
|   |   +-- routes/
|   |   |   +-- health.py       # Health endpoints
|   |   |   +-- lora.py         # LoRA training endpoints
|   |   |   +-- dataset.py      # Dataset endpoints
|   |   +-- schemas/            # Request/response types
|   |   +-- validators/         # Request validation
|   +-- core/
|   |   +-- contracts/          # Protocols and TypedDicts
|   |   +-- services/
|   |   |   +-- captioning/     # BLIP, Gemini, OpenAI backends
|   |   |   +-- dataset/        # Dataset upload/download
|   |   |   +-- deployment/     # ComfyUI deployment
|   |   |   +-- training/
|   |   |   |   +-- backends/
|   |   |   |       +-- kohya/  # Kohya-ss integration
|   |   |   +-- container.py    # DI container
|   |   +-- config/             # Settings
|   |   +-- infra/              # Paths, Redis keys
|   +-- orchestrators/          # Job orchestration
|   +-- worker/                 # RQ worker jobs
+-- tests/
+-- docs/
|   +-- api.md                  # API documentation
+-- scripts/
+-- Dockerfile
+-- docker-compose.yml
+-- pyproject.toml
+-- Makefile
```

---

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `hypercorn` | ASGI server |
| `redis` | Job queue backend |
| `rq` | Redis Queue |
| `transformers` | BLIP captioning |
| `torch` | ML framework |
| `pillow` | Image processing |
| `openai` | OpenAI API client |
| `google-genai` | Google Gemini API client |
| `accelerate` | Training acceleration |
| `diffusers` | Stable Diffusion utilities |
| `safetensors` | Model serialization |
| `peft` | LoRA implementation |
| `lycoris-lora` | LoCon / LoHa / LoKr training algorithms |
| `xformers` | Memory-efficient attention |
| `bitsandbytes` | 8-bit optimizers and quantization |
| `timm` | Vision backbones used by the Kohya stack |
| `torchvision` | Image transforms and dataset utilities |
| `sentencepiece` | Tokenizer for the caption models |
| `einops` | Tensor rearrangement |
| `omegaconf` | Kohya-ss config handling |
| `huggingface-hub` | Model and dataset downloads |
| `wandb` | Optional training run logging |
| `prodigyopt`, `prodigy-plus-schedule-free`, `dadaptation`, `lion-pytorch`, `pytorch-optimizer`, `schedulefree` | Optimizer implementations selectable per training run |
| `httpx` | Outbound HTTP (data-bank client) |
| `python-multipart` | Multipart upload parsing |
| `python-dotenv` | `.env` loading for local runs |
| `python-json-logger` | JSON log formatting |
| `rich` | Console output |
| `typing-extensions` | Backported typing constructs |
| `platform-core` | Logging, errors, config |
| `platform-workers` | RQ worker harness |
| `platform-ml` | Device selection, artifact storage |

### Development

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage reporting |
| `pytest-xdist` | Parallel tests |
| `mypy` | Type checking |
| `ruff` | Linting/formatting |

---

## License

Apache-2.0
