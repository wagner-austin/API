# Handwriting AI

Production-ready MNIST digit recognition service with ResNet-18 inference, advanced preprocessing pipeline, and optional training capabilities. Built for Discord bot integration with strict type safety and comprehensive error handling.

## Features

- **MNIST Inference**: ResNet-18 (CIFAR-style) with temperature-scaled calibration
- **Advanced Preprocessing**: Grayscale → Otsu → LCC extraction → deskew → center → normalize
- **Test-Time Augmentation (TTA)**: Optional multi-shift/rotation averaging for improved accuracy
- **Model Management**: Hot-reload, manifest validation, admin upload endpoint
- **Training Pipeline**: RQ-based background training with calibration and augmentation
- **Type Safety**: mypy strict mode, zero `Any` types, TypedDict models
- **100% Test Coverage**: Statements and branches

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+
- Redis (for training worker)

### Installation

```bash
cd services/handwriting-ai
poetry install --with dev
```

### Run the Service

```bash
# Development
poetry run hypercorn 'handwriting_ai.api.main:create_app()' --bind 0.0.0.0:8081 --reload

# Production
poetry run hypercorn 'handwriting_ai.api.main:create_app()' --bind [::]:${PORT:-8081}
```

### Verify

```bash
curl http://localhost:8081/healthz
# {"status": "ok"}

curl http://localhost:8081/readyz
# {"status": "ready", "model_loaded": true, "model_id": "mnist_resnet18_v1", ...}
```

## API Reference

For complete API documentation, see [docs/api.md](./docs/api.md).

### Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/readyz` | GET | Readiness probe (checks model loaded) |
| `/v1/models/active` | GET | Get active model metadata |
| `/v1/read` | POST | Classify handwritten digit image |
| `/v1/predict` | POST | Alias for `/v1/read` |
| `/v1/admin/models/upload` | POST | Upload new model (admin) |
| `/api/v1/training/jobs` | POST | Enqueue training job |

---

## Configuration

### Configuration File

Create `config/handai.toml`:

```toml
[app]
data_root = "/data"
artifacts_root = "/data/artifacts"
logs_root = "/data/logs"
threads = 0                    # 0 = auto-detect
port = 8081

[digits]
model_dir = "/data/digits/models"
active_model = "mnist_resnet18_v1"
tta = false                    # Test-time augmentation
uncertain_threshold = 0.85     # Flag as uncertain below this
max_image_mb = 2               # Max upload size
max_image_side_px = 1024       # Max dimension
predict_timeout_seconds = 5    # Inference timeout
visualize_max_kb = 16          # Max visualization size

[security]
api_key_enabled = false
# api_key set via SECURITY__API_KEY env var
```

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APP__PORT` | int | `8081` | Server port |
| `APP__THREADS` | int | `0` | Worker threads (0 = auto) |
| `DIGITS__MODEL_DIR` | string | `/data/digits/models` | Model directory |
| `DIGITS__ACTIVE_MODEL` | string | - | Active model ID |
| `DIGITS__TTA` | bool | `false` | Enable test-time augmentation |
| `DIGITS__UNCERTAIN_THRESHOLD` | float | `0.85` | Confidence threshold |
| `DIGITS__MAX_IMAGE_MB` | int | `2` | Max upload size (MB) |
| `DIGITS__MAX_IMAGE_SIDE_PX` | int | `1024` | Max image dimension |
| `DIGITS__PREDICT_TIMEOUT_SECONDS` | int | `5` | Inference timeout |
| `SECURITY__API_KEY_ENABLED` | bool | `false` | Require API key |
| `SECURITY__API_KEY` | string | - | API key value |
| `REDIS_URL` | string | - | Redis URL (for worker) |
| `HANDWRITING_VERSION` | string | - | Version override |

---

## Preprocessing Pipeline

The service applies a deterministic preprocessing pipeline to normalize input images:

```
Input Image
    │
    ▼
┌─────────────────────┐
│  Load & Grayscale   │  EXIF transpose, alpha composite, mode convert
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Auto-Invert        │  Detect dark background, invert if needed
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Otsu Binarization  │  Adaptive threshold for clean binary image
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Largest Component  │  BFS flood-fill to isolate digit
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Deskew             │  PCA-based rotation (±10° max, confidence gated)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Center             │  Center of mass on square canvas with margin
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Resize to 28×28    │  Aspect-preserving resize
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  MNIST Normalize    │  mean=0.1307, std=0.3081
└──────────┬──────────┘
           │
           ▼
    Tensor (1, 1, 28, 28)
```

**Preprocessing Signature:** `v1/grayscale+otsu+lcc+deskew{angle_conf}+center+resize28+mnistnorm`

---

## Inference Engine

### Architecture

- **Model**: ResNet-18 (CIFAR-style stem, 1-channel input)
- **Output**: 10-class softmax probabilities
- **Calibration**: Temperature scaling for confidence calibration

### Test-Time Augmentation (TTA)

When enabled (`DIGITS__TTA=true`), predictions average over:
- Identity transform
- 4 pixel shifts (±1 px horizontal/vertical)
- 4 small rotations (±3°, ±6°)

TTA improves accuracy at the cost of ~3x latency.

### Threading

```python
# Default configuration
torch.set_num_threads(1)  # Avoid oversubscription
ThreadPoolExecutor(max_workers=min(8, cpu_count))
```

Recommendation: Run with `--workers 1` for hypercorn to avoid thread explosion.

---

## Model Management

### Model Directory Structure

```
/data/digits/models/
└── mnist_resnet18_v1/
    ├── model.pt           # PyTorch state dict
    └── manifest.json      # Model metadata
```

### Manifest Schema (v1.1)

```json
{
  "schema_version": "v1.1",
  "model_id": "mnist_resnet18_v1",
  "arch": "resnet18",
  "n_classes": 10,
  "version": "1.0.0",
  "created_at": "2024-01-15T10:30:00Z",
  "preprocess_hash": "v1/grayscale+otsu+lcc+deskew{angle_conf}+center+resize28+mnistnorm",
  "val_acc": 0.9912,
  "temperature": 1.0
}
```

**Validation Rules:**
- `schema_version`: Must be "v1" or "v1.1"
- `arch`: Must be "resnet18"
- `n_classes`: Must be ≥ 2
- `preprocess_hash`: Must match current pipeline signature
- `temperature`: Must be > 0

### Hot Reload

The engine monitors model directory mtime and reloads when changes are detected (16 stable reads to avoid transient FS issues).

---

## Training (Worker)

### Start Worker

```bash
export REDIS_URL=redis://localhost:6379/0
poetry run handwriting-rq-worker
```

### Training Job Schema

```json
{
  "type": "digits.train.v1",
  "request_id": "uuid",
  "user_id": 12345,
  "model_id": "mnist_resnet18_v2",
  "epochs": 10,
  "batch_size": 64,
  "lr": 0.001,
  "seed": 42,
  "augment": true,
  "notes": "Training run notes"
}
```

### Training Features

- **Data Augmentation**: Rotation, translation, noise, dots, blur, morphological ops
- **Calibration**: Auto-tune dataloader settings (workers, threads)
- **Memory Guards**: Prevent runaway memory during training
- **Progress Events**: Publish to Redis for Discord bot integration
- **Temperature Scaling**: Post-hoc confidence calibration

### Augmentation Options

| Option | Description |
|--------|-------------|
| `aug_rotate` | Max rotation degrees |
| `aug_translate` | Max translation fraction |
| `noise_prob` | Salt & pepper noise probability |
| `dots_prob` | Random dots probability |
| `blur_sigma` | Gaussian blur sigma |
| `morph` | Morphological operations |

---

## Discord Bot Integration

The service integrates with Discord bot via:

1. **HTTP API**: Bot calls `/v1/read` with uploaded images
2. **Redis Events**: Training progress published to `digits:events` channel

### Bot Usage

```
/read image:<attachment>
```

Response format:
```
Digit: 7 (98.7% confidence)
Top-3: 7=0.987, 1=0.011, 9=0.002
Model: mnist_resnet18_v1
```

If `uncertain=true`:
```
Low confidence; try larger digits or darker ink.
```

---

## Development

### Commands

```bash
make install      # Install dependencies
make install-dev  # Install with dev dependencies
make lint         # Run guards + ruff + mypy
make test         # Run pytest with coverage
make check        # Run lint + test
make serve        # Start development server
make guards       # Check for Any, cast, type: ignore
make start        # Docker compose up
make stop         # Docker compose stop
make clean        # Clean containers/volumes
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
poetry run pytest tests/test_preprocess.py -v

# Run with coverage report
poetry run pytest --cov-report=html
```

---

## Project Structure

```
handwriting-ai/
├── src/handwriting_ai/
│   ├── __init__.py
│   ├── api/
│   │   ├── main.py             # App factory
│   │   ├── schemas.py          # Response schemas
│   │   ├── types.py            # API types
│   │   ├── dependencies.py     # Dependency injection
│   │   └── routes/
│   │       ├── health.py       # Health endpoints
│   │       ├── read.py         # Read/predict endpoints
│   │       ├── models.py       # Model metadata endpoint
│   │       ├── admin.py        # Admin upload endpoint
│   │       └── training.py     # Training job endpoint
│   ├── inference/
│   │   ├── engine.py           # Inference engine
│   │   ├── manifest.py         # Model manifest handling
│   │   └── types.py            # Inference types
│   ├── training/
│   │   ├── __init__.py
│   │   ├── mnist_train.py      # Training loop
│   │   ├── augment.py          # Data augmentation
│   │   ├── calibrate.py        # Dataloader calibration
│   │   ├── loops.py            # Training loops
│   │   ├── runtime.py          # Training runtime
│   │   ├── resources.py        # Resource management
│   │   ├── safety.py           # Memory safety guards
│   │   ├── train_utils.py      # Training utilities
│   │   ├── train_config.py     # Training configuration
│   │   ├── dataset.py          # Dataset handling
│   │   ├── metrics.py          # Training metrics
│   │   ├── artifacts.py        # Artifact management
│   │   ├── progress.py         # Progress tracking
│   │   ├── optim.py            # Optimizer utilities
│   │   └── calibration/        # Calibration subsystem
│   │       ├── calibrator.py
│   │       ├── orchestrator.py
│   │       ├── runner.py
│   │       ├── measure.py
│   │       ├── cache.py
│   │       ├── candidates.py
│   │       └── checkpoint.py
│   ├── jobs/
│   │   └── digits.py           # RQ job handler
│   ├── preprocess.py           # Preprocessing pipeline
│   ├── config.py               # Configuration loading
│   ├── monitoring.py           # Memory monitoring
│   ├── middleware.py           # Custom middleware
│   ├── version.py              # Version info
│   ├── worker_entry.py         # RQ worker entry point
│   └── _test_hooks.py          # Test hooks for DI
├── seed/                       # Seed model artifacts
├── config/
│   └── handai.toml             # Default configuration
├── tests/
│   ├── test_api*.py
│   ├── test_engine*.py
│   ├── test_preprocess*.py
│   └── ...
├── scripts/
│   └── guard.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── Makefile
```

---

## Deployment

### Docker

```bash
# Build API
docker build --target api -t handwriting-ai:latest .

# Build Worker
docker build --target worker -t handwriting-ai-worker:latest .

# Run API
docker run -p 8081:8081 \
  -v ./models:/data/digits/models \
  -e DIGITS__ACTIVE_MODEL=mnist_resnet18_v1 \
  handwriting-ai:latest

# Run Worker
docker run \
  -e REDIS_URL=redis://redis:6379/0 \
  handwriting-ai-worker:latest
```

### Docker Compose

```yaml
version: "3.8"
services:
  handai:
    build: .
    ports:
      - "8081:8081"
    volumes:
      - ./config:/app/config:ro
      - ./models:/data/digits/models
    environment:
      - APP__PORT=8081
      - DIGITS__ACTIVE_MODEL=mnist_resnet18_v1
```

### Railway

```bash
# Set environment variables in Railway dashboard
railway up
```

**Health Check Path:** `/healthz`

---

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `hypercorn` | ASGI server |
| `torch` | Neural network inference |
| `torchvision` | ResNet-18 architecture |
| `pillow` | Image processing |
| `numpy` | Numerical operations |
| `redis` | Job queue backend |
| `rq` | Background job processing |
| `platform-core` | Logging, errors, config |
| `platform-workers` | RQ worker harness |
| `platform-discord` | Discord integration |

### Development

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage reporting |
| `pytest-xdist` | Parallel tests |
| `mypy` | Type checking |
| `ruff` | Linting/formatting |

---

## Performance Targets

**On laptop i9 CPU (TTA off):**
- P50 latency: < 20 ms
- P95 latency: < 60 ms

**With TTA enabled:**
- P50 latency: < 60 ms
- P95 latency: < 180 ms

---

## Quality Standards

- **Type Safety**: mypy strict mode, no `Any`, no `cast`
- **Coverage**: 100% statements and branches
- **Guard Rules**: Enforced via `scripts/guard.py`
- **Logging**: Structured JSON via platform_core
- **Errors**: Consistent `{code, message, request_id}` format

---

## License

Apache-2.0
