# Model Trainer

Strictly typed, modular system for training and evaluating small language models and tokenizers. Features pluggable backends, GPU acceleration with mixed-precision training, durable job execution via Redis + RQ, and deterministic artifacts with manifests.

## Features

- **Pluggable Backends**: Protocol-based tokenizers (BPE, SentencePiece) and models (GPT-2, Char-LSTM)
- **GPU Acceleration**: CUDA support with automatic device selection
- **Mixed-Precision Training**: FP16/BF16 with automatic loss scaling for faster training
- **Durable Jobs**: Redis + RQ with heartbeats, cancellation, retry logic
- **Artifact Management**: Deterministic manifests, model weights, evaluation metrics
- **Weights & Biases Integration**: Optional experiment tracking with full metrics logging
- **Discord Integration**: Training notifications via Redis pub/sub
- **Type Safety**: mypy strict mode, zero `Any` types, Protocol-based DI
- **100% Test Coverage**: Statements and branches

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+
- Docker Desktop (for Redis and containerized deployment)

### Installation

```bash
cd services/Model-Trainer
poetry install --with dev
```

### Start with Docker

```bash
# Copy environment file
cp .env.example .env

# Start stack (API + Worker + Redis)
make start

# Verify
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

### GPU (CUDA) Build and Run

Prereqs (Windows + 3090 Ti):
- Docker Desktop with WSL 2 backend enabled
- NVIDIA Windows driver (recent), and WSL GPU support enabled
- NVIDIA Container Toolkit (provided by Docker Desktop with WSL 2)

Build and run with GPU support:

```bash
# From services/Model-Trainer
# Build images with CUDA wheels (compose passes TORCH_INDEX=cu124)
docker compose build --no-cache

# Start API + Worker with GPU access
docker compose up -d

# Verify containers are healthy
docker ps
```

Usage notes:
- Use `"device": "auto"` in training requests to prefer CUDA when available, or `"device": "cuda"` to require it.
- Use `"precision": "auto"` to automatically select FP16 on CUDA or FP32 on CPU.
- Explicit precision options: `"fp32"` (full precision), `"fp16"` (mixed precision with loss scaling), `"bf16"` (mixed precision, CUDA only).
- To override the PyTorch CUDA wheel index, set `TORCH_INDEX` in `.env` before building.

### Device Selection (platform_ml)

Device and precision resolution uses the centralized `platform_ml` library to ensure consistent behavior across all ML services:

```python
from platform_ml import (
    RequestedDevice,      # "cpu", "cuda", "auto"
    ResolvedDevice,       # "cpu", "cuda"
    RequestedPrecision,   # "fp32", "fp16", "bf16", "auto"
    ResolvedPrecision,    # "fp32", "fp16", "bf16"
    resolve_device,
    resolve_precision,
    recommended_batch_size,
)

# Device resolution
device = resolve_device("auto")  # Returns "cuda" if available, else "cpu"

# Precision resolution (raises on fp16/bf16 + CPU)
precision = resolve_precision("auto", device)  # "fp16" on CUDA, "fp32" on CPU

# Batch size recommendation (bumps small batches on CUDA)
batch = recommended_batch_size(4, device)  # 8 on CUDA, 4 on CPU
```

Model-Trainer extends this with model-family-specific batch sizing via `recommended_batch_size_for(model_family, batch, device)`.

### Local Development

```bash
# Start Redis
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Run API
poetry run hypercorn 'model_trainer.api.main:create_app()' --bind 0.0.0.0:8000

# Run Worker (separate terminal)
poetry run modeltrainer-rq-worker
```

## API Reference

For complete API documentation, see [docs/api.md](./docs/api.md).

### Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/readyz` | GET | Readiness probe (checks Redis + workers) |
| `/runs/train` | POST | Enqueue model training job |
| `/runs/{run_id}` | GET | Get training run status |
| `/runs/{run_id}/evaluate` | POST | Enqueue evaluation job |
| `/runs/{run_id}/eval` | GET | Get evaluation results |
| `/runs/{run_id}/artifact` | GET | Get artifact pointer |
| `/runs/{run_id}/logs` | GET | Get training logs (tail) |
| `/runs/{run_id}/logs/stream` | GET | Stream logs via SSE |
| `/runs/{run_id}/cancel` | POST | Request job cancellation |
| `/runs/{run_id}/generate` | POST | Enqueue text generation |
| `/runs/{run_id}/generate/{request_id}` | GET | Get generation results |
| `/runs/{run_id}/score` | POST | Enqueue text scoring |
| `/runs/{run_id}/score/{request_id}` | GET | Get scoring results |
| `/runs/{run_id}/chat` | POST | Start/continue chat conversation |
| `/runs/{run_id}/chat/{session_id}/{request_id}` | GET | Poll for chat response |
| `/runs/{run_id}/chat/{session_id}` | GET | Get conversation history |
| `/runs/{run_id}/chat/{session_id}` | DELETE | Delete chat session |
| `/tokenizers/train` | POST | Enqueue tokenizer training |
| `/tokenizers/{tokenizer_id}` | GET | Get tokenizer status |

---

## Configuration

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDIS__URL` | string | `redis://redis:6379/0` | Redis connection URL |
| `REDIS__ENABLED` | bool | `true` | Enable Redis connection |
| `RQ__QUEUE_NAME` | string | `trainer` | RQ queue name |
| `RQ__JOB_TIMEOUT_SEC` | int | `86400` | Job timeout (24h) |
| `RQ__RESULT_TTL_SEC` | int | `86400` | Result retention (24h) |
| `RQ__FAILURE_TTL_SEC` | int | `604800` | Failure retention (7d) |
| `RQ__RETRY_MAX` | int | `1` | Max retry attempts |
| `RQ__RETRY_INTERVALS_SEC` | string | `300` | Retry interval in seconds |
| `APP__DATA_ROOT` | string | `/data` | Base data directory |
| `APP__ARTIFACTS_ROOT` | string | `/data/artifacts` | Artifact storage path |
| `APP__RUNS_ROOT` | string | `/data/runs` | Training runs directory |
| `APP__LOGS_ROOT` | string | `/data/logs` | Logs directory |
| `APP__THREADS` | int | `0` | Thread count (0=auto) |
| `APP__TOKENIZER_SAMPLE_MAX_LINES` | int | `10000` | Max lines for tokenizer sampling |
| `APP__DATA_BANK_API_URL` | string | - | Data bank API URL (or use `API_GATEWAY_URL`) |
| `APP__DATA_BANK_API_KEY` | string | - | Data bank API key |
| `API_GATEWAY_URL` | string | - | Gateway URL (auto-appends `/data-bank` for data bank) |
| `APP_ENV` | string | `dev` | Environment (`dev` or `prod`) |
| `LOGGING__LEVEL` | string | `INFO` | Log level |
| `HF_HOME` | string | `/hf-cache` | Hugging Face cache |
| `SECURITY__API_KEY` | string | - | Optional API key |
| `WANDB__ENABLED` | bool | `false` | Enable Weights & Biases logging |
| `WANDB__PROJECT` | string | `model-trainer` | Wandb project name |

**Cleanup Settings:**

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APP__CLEANUP__ENABLED` | bool | `true` | Enable post-upload cleanup |
| `APP__CLEANUP__VERIFY_UPLOAD` | bool | `true` | Verify upload before cleanup |
| `APP__CLEANUP__GRACE_PERIOD_SECONDS` | int | `0` | Wait time before cleanup |
| `APP__CLEANUP__DRY_RUN` | bool | `false` | Log deletions without deleting |
| `APP__CORPUS_CACHE_CLEANUP__ENABLED` | bool | `false` | Enable corpus cache cleanup |
| `APP__CORPUS_CACHE_CLEANUP__MAX_BYTES` | int | `10737418240` | Max cache size (10GB) |
| `APP__CORPUS_CACHE_CLEANUP__MIN_FREE_BYTES` | int | `2147483648` | Min free space (2GB) |
| `APP__CORPUS_CACHE_CLEANUP__EVICTION_POLICY` | string | `lru` | `lru` or `oldest` |
| `APP__TOKENIZER_CLEANUP__ENABLED` | bool | `false` | Enable tokenizer cleanup |
| `APP__TOKENIZER_CLEANUP__MIN_UNUSED_DAYS` | int | `30` | Days before cleanup |

### Example .env

```bash
REDIS__URL=redis://redis:6379/0
RQ__QUEUE_NAME=trainer
RQ__JOB_TIMEOUT_SEC=86400
APP__ARTIFACTS_ROOT=/data/artifacts
LOGGING__LEVEL=INFO
HF_HOME=/hf-cache
```

---

## Job Execution

### Queue Architecture

```
┌─────────────────┐
│    FastAPI      │
│    API Server   │
└────────┬────────┘
         │ enqueue
         ▼
┌─────────────────┐     ┌─────────────────┐
│     Redis       │◄────│   RQ Worker     │
│   Job Queue     │     │                 │
│                 │     │  - Training     │
│  runs:hb:{id}   │     │  - Evaluation   │
│  runs:{id}:stat │     │  - Tokenizer    │
└─────────────────┘     └─────────────────┘
```

### Heartbeats

Workers emit heartbeats to `runs:hb:<run_id>` during training:
- Epoch progress
- Current loss
- Samples processed
- Memory usage

### Cancellation

Set `runs:<run_id>:cancelled=1` in Redis to request cancellation. Worker checks this flag between batches and performs graceful shutdown.

### Event Publishing

Training publishes to `trainer:events`:

- Job lifecycle (via `platform_core.job_events`): `trainer.job.started.v1`, `.progress.v1`, `.completed.v1`, `.failed.v1`
- Metrics (via `platform_core.trainer_metrics_events`): `trainer.metrics.config.v1`, `.progress.v1`, `.completed.v1`

Lifecycle events carry `job_id` (run_id), `user_id`, `queue`, progress, and optional message/payload. Metrics events carry training-specific fields (epochs, steps, loss, artifact path) alongside the same `job_id`/`user_id`.

---

## Artifacts

### Directory Layout

```
/data/artifacts/
├── tokenizers/
│   └── my-bpe-tokenizer/
│       ├── tokenizer.json      # HF tokenizer file
│       └── manifest.json       # Training metadata
└── models/
    └── gpt2-run-001/
        ├── pytorch_model.bin   # Model weights
        ├── manifest.json       # Training manifest
        └── eval/
            └── metrics.json    # Evaluation results
```

### Manifest Schema

```json
{
  "run_id": "gpt2-run-001",
  "model_family": "gpt2",
  "tokenizer_id": "my-bpe-tokenizer",
  "epochs": 3,
  "batch_size": 8,
  "learning_rate": 0.0001,
  "max_seq_len": 128,
  "seed": 42,
  "device": "cuda",
  "precision": "fp16",
  "final_loss": 3.21,
  "final_perplexity": 24.8,
  "training_time_seconds": 3600,
  "created_at": "2024-11-27T10:00:00Z",
  "completed_at": "2024-11-27T11:00:00Z"
}
```

---

## Inference

Use trained models to generate text or score input sequences.

### Text Generation

Generate text from a trained model with configurable sampling parameters.

```bash
# Enqueue generation
curl -X POST http://localhost:8000/runs/{run_id}/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_text": "Once upon a time",
    "max_new_tokens": 100,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95,
    "stop_on_eos": true,
    "stop_sequences": [],
    "seed": null,
    "num_return_sequences": 1
  }'

# Poll for results
curl http://localhost:8000/runs/{run_id}/generate/{request_id}
```

**GenerateRequest Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `prompt_text` | string \| null | Inline prompt text |
| `prompt_path` | string \| null | Path to prompt file (mutually exclusive with prompt_text) |
| `max_new_tokens` | int | Maximum tokens to generate (1-512) |
| `temperature` | float | Sampling temperature (0=greedy, >0=stochastic) |
| `top_k` | int | Top-K filtering (0=disabled) |
| `top_p` | float | Nucleus sampling probability |
| `stop_on_eos` | bool | Stop at end-of-sequence token |
| `stop_sequences` | list[str] | Custom stop strings |
| `seed` | int \| null | Random seed for reproducibility |
| `num_return_sequences` | int | Batch generation (1-10) |

**GenerateResponse Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | string | Unique request identifier |
| `status` | string | `queued`, `running`, `completed`, or `failed` |
| `outputs` | list[str] \| null | Generated text sequences |
| `steps` | int \| null | Number of tokens generated |
| `eos_terminated` | list[bool] \| null | Whether each sequence hit EOS |

### Text Scoring

Score input text to get loss, perplexity, and per-token surprisal.

```bash
# Enqueue scoring
curl -X POST http://localhost:8000/runs/{run_id}/score \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog.",
    "path": null,
    "detail_level": "summary",
    "top_k": null,
    "seed": null
  }'

# Poll for results
curl http://localhost:8000/runs/{run_id}/score/{request_id}
```

**ScoreRequest Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | string \| null | Text to score |
| `path` | string \| null | Path to text file (mutually exclusive with text) |
| `detail_level` | string | `summary` or `per_char` |
| `top_k` | int \| null | Return top-K predictions per token |
| `seed` | int \| null | Random seed |

**ScoreResponse Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | string | Unique request identifier |
| `status` | string | `queued`, `running`, `completed`, or `failed` |
| `loss` | float \| null | Cross-entropy loss |
| `perplexity` | float \| null | exp(loss) |
| `surprisal` | list[float] \| null | Per-token log-probabilities |
| `topk` | list \| null | Top-K tokens per position |
| `tokens` | list[str] \| null | Tokenized input |

---

## Architecture

### Component Overview

```
model_trainer/
├── api/                    # FastAPI routes
│   ├── main.py            # App factory
│   └── routes/            # Endpoint handlers
├── core/
│   ├── contracts/         # Protocol interfaces
│   ├── services/
│   │   ├── container.py   # DI container
│   │   ├── tokenizer/     # BPE, SentencePiece backends
│   │   ├── model/         # GPT-2 backend
│   │   └── dataset/       # Dataset builders
│   └── config/            # Settings
├── orchestrators/         # Job orchestration
├── worker/                # RQ worker entry
└── infra/                 # Storage, persistence
```

### Protocol-Based Design

```python
class TokenizerBackend(Protocol):
    def train(self, config: TokenizerConfig) -> TokenizerArtifact: ...
    def load(self, path: str) -> TokenizerHandle: ...
    def encode(self, handle: TokenizerHandle, text: str) -> list[int]: ...
    def decode(self, handle: TokenizerHandle, ids: list[int]) -> str: ...

class ModelBackend(Protocol):
    def prepare(self, config: ModelConfig) -> PreparedModel: ...
    def train(self, model: PreparedModel, data: DataConfig) -> TrainingResult: ...
    def evaluate(self, model: PreparedModel, split: str) -> EvalResult: ...
    def save(self, model: PreparedModel, path: str) -> ModelArtifact: ...
```

See [DESIGN.md](./DESIGN.md) for detailed architecture documentation.

---

## Development

### Commands

```bash
make install      # Install dependencies
make install-dev  # Install with dev dependencies
make lint         # Run guards + ruff + mypy
make test         # Run pytest with coverage
make check        # Run lint + test
make start        # Docker compose up (API + Worker + Redis)
make stop         # Docker compose stop
make restart      # Stop then start
make clean        # Remove containers/volumes
```

### Quality Gates

All code must pass:

1. **Guard Scripts**: No `Any`, no `cast`, no `type: ignore`
2. **Ruff**: Linting and formatting
3. **Mypy**: Strict type checking
4. **Pytest**: 100% statement and branch coverage

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
poetry run pytest tests/test_tokenizer.py -v

# Run with coverage report
poetry run pytest --cov-report=html
```

---

## Project Structure

```
Model-Trainer/
├── src/model_trainer/
│   ├── api/
│   │   ├── main.py             # App factory
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   ├── runs.py
│   │   │   └── tokenizers.py
│   │   └── schemas/            # Request/response models
│   ├── core/
│   │   ├── contracts/          # Protocols
│   │   ├── services/
│   │   │   ├── container.py    # DI container
│   │   │   ├── tokenizer/      # Tokenizer backends
│   │   │   ├── model/          # Model backends
│   │   │   └── dataset/        # Dataset builders
│   │   └── config/             # Settings
│   ├── orchestrators/          # Job orchestration
│   ├── worker/                 # RQ workers
│   └── infra/                  # Storage helpers
├── tests/
├── scripts/
│   ├── cleanup_tokenizers.py
│   └── cleanup_corpus_cache.py
├── config/
│   └── app.toml
├── corpus/                     # Training data (mounted)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── Makefile
```

---

## Deployment

### Docker Compose

```yaml
version: "3.8"
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS__URL=redis://redis:6379/0
      - APP__ARTIFACTS_ROOT=/data/artifacts
    volumes:
      - ./corpus:/data/corpus:ro
      - ./artifacts:/data/artifacts
    depends_on:
      - redis

  worker:
    build: .
    command: modeltrainer-rq-worker
    environment:
      - REDIS__URL=redis://redis:6379/0
      - APP__ARTIFACTS_ROOT=/data/artifacts
    volumes:
      - ./corpus:/data/corpus:ro
      - ./artifacts:/data/artifacts
    depends_on:
      - redis
```

### Railway Deployment

1. **Create services**: API and Worker from same Dockerfile
2. **Add Redis addon**
3. **Configure volumes** for artifacts
4. **Set environment variables**:
   ```
   REDIS__URL=${{Redis.REDIS_URL}}
   APP__ARTIFACTS_ROOT=/data/artifacts
   RQ__QUEUE_NAME=trainer
   ```

### Health Checks

- **API**: `/healthz` and `/readyz`
- **Worker**: Monitored via RQ heartbeats

---

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `hypercorn` | ASGI server |
| `redis` | Job queue backend |
| `rq` | Redis Queue |
| `transformers` | GPT-2 model |
| `tokenizers` | BPE tokenizer |
| `datasets` | Data loading |
| `torch` | Training framework |
| `platform-core` | Logging, errors, config |
| `platform-ml` | Device selection, artifact storage |
| `platform-workers` | RQ worker harness |

### Development

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage reporting |
| `pytest-xdist` | Parallel tests |
| `mypy` | Type checking |
| `ruff` | Linting/formatting |

---

## Weights & Biases Integration

Optional experiment tracking via [Weights & Biases](https://wandb.ai). When enabled, training runs automatically log:

### Logged Metrics

**Per-Step (every batch):**
- `train_loss`, `train_ppl` - Training loss and perplexity
- `grad_norm` - Gradient norm before clipping
- `samples_per_sec` - Training throughput
- `global_step`, `epoch` - Progress tracking

**Per-Epoch (after validation):**
- `val_loss`, `val_ppl` - Validation metrics
- `best_val_loss`, `epochs_no_improve` - Early stopping state

**Final (end of training):**
- `test_loss`, `test_ppl` - Test set evaluation
- `early_stopped` - Whether training stopped early
- `epoch_summary` - Table with all epoch metrics

### Configuration

```bash
# Enable wandb logging
WANDB__ENABLED=true
WANDB__PROJECT=my-project

# Authenticate (set in environment or via wandb login)
WANDB_API_KEY=your-api-key
```

### Requirements

- Install wandb: `pip install wandb` (optional dependency)
- If `WANDB__ENABLED=true` but wandb is not installed, training continues without logging (warning emitted)

---

## Discord Bot Integration

Training progress is published to Redis for Discord bot notifications:

**Channel:** `trainer:events`

**Event Types (Job Lifecycle):**
- `trainer.job.started.v1`
- `trainer.job.progress.v1`
- `trainer.job.completed.v1`
- `trainer.job.failed.v1`

**Event Types (Training Metrics):**
- `trainer.metrics.config.v1`
- `trainer.metrics.progress.v1`
- `trainer.metrics.completed.v1`

**Event Schema:**
```json
{
  "type": "trainer.job.progress.v1",
  "job_id": "gpt2-run-001",
  "user_id": 12345,
  "queue": "trainer",
  "progress": 66,
  "message": "Training epoch 2/3"
}
```

**Metrics Event Schema:**
```json
{
  "type": "trainer.metrics.progress.v1",
  "job_id": "gpt2-run-001",
  "user_id": 12345,
  "epoch": 2,
  "total_epochs": 3,
  "loss": 3.45,
  "perplexity": 31.5,
  "samples_per_sec": 125.4
}
```

---

## Quality Standards

- **Type Safety**: mypy strict mode, no `Any`, no `cast`
- **Coverage**: 100% statements and branches
- **Guard Rules**: Enforced via `scripts/guard.py`
- **Logging**: Structured JSON via platform_core
- **Errors**: Consistent `{code, message, request_id}` format

---

## Notes

- **GPU Support**: CUDA acceleration with mixed-precision (FP16/BF16) training for faster iteration
- **Discord Bot**: Status notifications via `clients/DiscordBot`; subscribes to `trainer:events`
- **Web UI**: Planned; API-first design enables future UI development

---

## License

Apache-2.0
