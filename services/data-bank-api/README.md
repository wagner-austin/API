# Data Bank API

Production-grade file storage API for internal service-to-service data exchange. Provides streaming uploads/downloads with HTTP Range support, atomic writes, SHA256 checksums, disk-space guards, and multi-key authentication.

## Features

- **Atomic Writes**: Temp file → fsync → rename for crash safety
- **Content-Addressed Storage**: SHA256 as file ID (deduplication built-in)
- **HTTP Range Support**: Resume downloads, partial content (206)
- **Disk Guards**: Pre-upload free space checks, size limits
- **Multi-Key Auth**: Separate permissions for upload/read/delete
- **Type Safety**: mypy strict mode, zero `Any` types
- **100% Test Coverage**: Statements and branches

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+

### Installation

```bash
cd services/data-bank-api
poetry install --with dev
```

### Run the Service

```bash
# Development
poetry run hypercorn 'data_bank_api.api.main:create_app()' --bind 0.0.0.0:8000 --reload

# Production
poetry run hypercorn 'data_bank_api.api.main:create_app()' --bind [::]:${PORT:-8000}
```

### Verify

```bash
curl http://localhost:8000/healthz
# {"status": "ok"}

curl http://localhost:8000/readyz
# {"status": "ready"}
```

## API Reference

For complete API documentation, see [docs/api.md](./docs/api.md).

### Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/readyz` | GET | Readiness probe (checks disk space) |
| `/files` | POST | Upload file (returns SHA256 file ID) |
| `/files/{file_id}` | GET | Download file (supports Range) |
| `/files/{file_id}` | HEAD | Probe metadata |
| `/files/{file_id}/info` | GET | Get metadata as JSON |
| `/files/{file_id}` | DELETE | Delete file |

---

## Configuration

### Environment Variables

| Variable | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `REDIS_URL` | string | **Yes** | - | Redis connection URL |
| `API_UPLOAD_KEYS` | string | **Yes** | - | Comma-separated upload keys |
| `API_READ_KEYS` | string | No | (inherits from upload) | Comma-separated read keys |
| `API_DELETE_KEYS` | string | No | (inherits from upload) | Comma-separated delete keys |
| `PORT` | int | No | `8000` | Server port |

### Fixed Configuration

The following values are fixed in the codebase and cannot be changed via environment variables:

| Setting | Value | Description |
|---------|-------|-------------|
| `DATA_ROOT` | `/data/files` | Storage root directory |
| `MIN_FREE_GB` | `1` | Minimum free disk space (GB) |
| `MAX_FILE_BYTES` | `0` | Max upload size (0 = unlimited) |
| `DELETE_STRICT_404` | `false` | Return 404 on missing delete |

### Example Configuration

```bash
REDIS_URL=redis://localhost:6379/0
API_UPLOAD_KEYS=turkic-api-key,model-trainer-key
API_READ_KEYS=model-trainer-key
API_DELETE_KEYS=admin-key
```

---

## Storage Architecture

### File Layout

```
/data/files/
├── ab/                    # First 2 chars of file_id
│   └── cd/                # Next 2 chars
│       ├── abcd1234...bin # Blob content
│       └── abcd1234...meta # Metadata sidecar
└── tmp/                   # Temp files during upload
```

### Atomic Write Process

1. Stream upload to temporary file
2. Compute SHA256 hash while streaming
3. Verify size limits
4. `fsync()` file descriptor
5. Atomic `os.replace()` to final path
6. Write metadata sidecar with fsync

### Content-Addressed Storage

- **File ID** = SHA256 hex digest of content
- Automatic deduplication (same content = same ID)
- Immutable after upload

### Metadata Sidecar

```
sha256=a1b2c3d4e5f6...
content_type=text/plain; charset=utf-8
created_at=2024-11-27T10:30:00+00:00
```

---

## Python Client

The service includes a typed client for consuming services:

```python
from data_bank_api.client import DataBankClient
from pathlib import Path
import os

# Create client
client = DataBankClient(
    base_url=os.environ["DATA_BANK_URL"],
    api_key=os.environ["DATA_BANK_KEY"]
)

# Upload
with open("corpus.txt", "rb") as f:
    response = client.upload("corpus.txt", f, "text/plain")
    file_id = response["file_id"]

# Download with resume support
client.download_to_path(
    file_id,
    Path("local.txt"),
    resume=True,        # Resume partial downloads
    verify_etag=True    # Verify SHA256 after download
)

# Probe metadata
head = client.head(file_id)  # Returns size, ETag, content-type
info = client.info(file_id)  # Returns full metadata

# Delete
client.delete(file_id)
```

### Client Error Handling

```python
from data_bank_api.client import (
    DataBankClientError,
    AuthorizationError,
    ForbiddenError,
    NotFoundError,
    InsufficientStorageClientError,
)

try:
    client.upload(...)
except InsufficientStorageClientError:
    # Handle 507 - disk full
except AuthorizationError:
    # Handle 401 - missing key
except ForbiddenError:
    # Handle 403 - invalid key
except NotFoundError:
    # Handle 404 - file not found
except DataBankClientError as e:
    # Generic error
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
poetry run pytest tests/test_storage.py -v

# Run with coverage report
poetry run pytest --cov-report=html
```

---

## Project Structure

```
data-bank-api/
├── src/data_bank_api/
│   ├── __init__.py         # Public API exports
│   ├── storage.py          # Storage logic (atomic writes, ranges)
│   ├── config.py           # Configuration loading
│   ├── client.py           # Re-export typed client
│   ├── api/
│   │   ├── main.py         # App factory
│   │   ├── config.py       # TypedDict settings
│   │   ├── jobs.py         # Job processing integration
│   │   └── routes/         # Route handlers
│   └── core/
│       └── corpus_download.py  # Corpus helper
├── tests/
│   ├── test_files_endpoints.py
│   ├── test_auth_and_ranges.py
│   ├── test_storage_branches.py
│   ├── test_health_ready.py
│   └── ...
├── scripts/
│   └── guard.py
├── Dockerfile
├── pyproject.toml
└── Makefile
```

---

## Deployment

### Docker

```bash
# Build
docker build -t data-bank-api:latest .

# Run
docker run -p 8000:8000 \
  -e DATA_ROOT=/data/files \
  -e API_UPLOAD_KEYS=test-key \
  -v $(pwd)/data:/data/files \
  data-bank-api:latest
```

### Railway Deployment

1. **Create service** from Dockerfile
2. **Add persistent volume** mounted at `/data/files`
3. **Configure environment:**
   ```
   DATA_ROOT=/data/files
   MIN_FREE_GB=2
   API_UPLOAD_KEYS=turkic-key-xyz
   API_READ_KEYS=trainer-key-abc
   API_DELETE_KEYS=admin-key-123
   ```
4. **Keep on private network** - Access via `http://data-bank-api.railway.internal:8000`
5. **Configure clients:**
   - turkic-api: `DATA_BANK_URL=http://data-bank-api.railway.internal:8000`
   - model-trainer: `DATA_BANK_URL=http://data-bank-api.railway.internal:8000`

### Health Checks

- **Liveness:** `/healthz`
- **Readiness:** `/readyz` (checks disk space and writability)

---

## Integration Examples

### turkic-api → data-bank-api (Upload)

```python
from data_bank_api.client import DataBankClient

client = DataBankClient(
    base_url=settings["data_bank_api_url"],
    api_key=settings["data_bank_api_key"]
)

# After job completion, upload result
with out_path.open("rb") as f:
    response = client.upload(f"{job_id}.txt", f, "text/plain")
    file_id = response["file_id"]

# Store file_id in Redis for consumers
redis.hset(f"job:{job_id}", mapping={"file_id": file_id})
```
### Job Events

When run as an RQ job, corpus uploads emit lifecycle events via `platform_workers.job_context` to the `databank:events` channel:
- `databank.job.started.v1`
- `databank.job.progress.v1`
- `databank.job.completed.v1`
- `databank.job.failed.v1`

Events include `job_id`, `user_id` (default 0), and `queue`, and errors are propagated without fallback handling.


### model-trainer → data-bank-api (Download)

```python
from data_bank_api.client import DataBankClient
from pathlib import Path

client = DataBankClient(
    base_url=settings.data_bank_api_url,
    api_key=settings.data_bank_api_key
)

# Download corpus for training
corpus_path = Path(settings.data_root) / "corpus_cache" / f"{file_id}.txt"
corpus_path.parent.mkdir(parents=True, exist_ok=True)

client.download_to_path(
    file_id,
    corpus_path,
    resume=True,
    verify_etag=True
)

# Use corpus for training
train(corpus_path=corpus_path, ...)
```

---

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `hypercorn` | ASGI server |
| `python-multipart` | Multipart form parsing |
| `httpx` | HTTP client |
| `redis` | Job queue integration |
| `platform-core` | Logging, errors, config, client |
| `platform-workers` | RQ integration |

### Development

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage reporting |
| `pytest-xdist` | Parallel tests |
| `mypy` | Type checking |
| `ruff` | Linting/formatting |

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
