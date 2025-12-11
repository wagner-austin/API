# platform-core

Shared platform utilities: error handling, validation, logging, health checks, typed event schemas, and service clients.

## Installation

```bash
poetry add platform-core
```

## Quick Start

```python
from platform_core import (
    AppError, ErrorCode,
    validate_str, validate_int_range,
    get_logger, setup_logging,
    healthz,
)

# Raise structured errors
raise AppError(code=ErrorCode.NOT_FOUND, message="User not found", http_status=404)

# Validate input
name = validate_str(data.get("name"), "name")
count = validate_int_range(data.get("count"), "count", ge=1, le=100)

# Setup logging
setup_logging()
logger = get_logger("my-service")
```

## Error Handling

```python
from platform_core import AppError, ErrorCode, install_exception_handlers

# Raise structured errors
raise AppError(
    code=ErrorCode.NOT_FOUND,
    message="User not found",
    http_status=404
)

# Install FastAPI exception handlers
from fastapi import FastAPI
app = FastAPI()
install_exception_handlers(app, logger_name="my-api")
```

### Error Codes

| Code | HTTP | Description |
|------|------|-------------|
| `INVALID_INPUT` | 400 | Validation failed |
| `INVALID_JSON` | 400 | JSON parse error |
| `UNAUTHORIZED` | 401 | Missing/invalid auth |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Resource conflict |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Unexpected server error |
| `EXTERNAL_SERVICE_ERROR` | 502 | Upstream service failed |
| `SERVICE_UNAVAILABLE` | 503 | Service not ready |
| `TIMEOUT` | 504 | Operation timed out |

### Domain-Specific Error Codes

```python
from platform_core import HandwritingErrorCode, TranscriptErrorCode

# Handwriting service errors
raise AppError(code=HandwritingErrorCode.MODEL_NOT_READY, ...)

# Transcript service errors
raise AppError(code=TranscriptErrorCode.AUDIO_TOO_LONG, ...)
```

## Validation

```python
from platform_core import (
    validate_str,
    validate_int_range,
    validate_float_range,
    validate_bool,
    validate_optional_literal,
    validate_required_literal,
    load_json_dict,
)

# Validates and raises AppError on failure
name = validate_str(data.get("name"), "name")
count = validate_int_range(data.get("count"), "count", ge=1, le=100)
ratio = validate_float_range(data.get("ratio"), "ratio", ge=0.0, le=1.0)
enabled = validate_bool(data.get("enabled"), "enabled", default=False)
mode = validate_optional_literal(data.get("mode"), "mode", frozenset(["fast", "slow"]))
status = validate_required_literal(data.get("status"), "status", frozenset(["ok", "error"]))

# Load JSON from string
config = load_json_dict(json_string, "config")
```

## Health Checks

```python
from platform_core import healthz, HealthResponse, ReadyResponse

# Liveness probe (always returns ok)
@app.get("/healthz")
def health() -> HealthResponse:
    return healthz()

# For readiness probes with Redis, use platform_workers.health
```

## Logging

```python
from platform_core import get_logger, setup_logging, stdlib_logging

setup_logging()
logger = get_logger("my-service")
logger.info("started", extra={"port": 8000})

# Get stdlib logger for libraries
std_logger = stdlib_logging("my-lib")
```

## Request Context

```python
from platform_core import RequestIdMiddleware, request_id_var

app.add_middleware(RequestIdMiddleware)

# Access request ID anywhere
rid = request_id_var.get()
```

## Security

```python
from platform_core import create_api_key_dependency

api_key_dep = create_api_key_dependency(expected_key="secret")

@app.get("/protected")
def protected(key: str = Depends(api_key_dep)):
    return {"status": "ok"}
```

## Data Bank Client

HTTP client for data-bank-api file storage:

```python
from platform_core import DataBankClient

client = DataBankClient(
    base_url="http://data-bank-api:8000",
    api_key="secret",
    timeout_seconds=30.0,
)

# Upload file
response = client.upload(
    file_bytes=b"...",
    filename="model.bin",
    content_type="application/octet-stream",
    request_id="req-123",
)
print(response["file_id"], response["sha256"])

# Download file
data = client.download(file_id="model.bin", request_id="req-456")

# Check file exists
info = client.head(file_id="model.bin", request_id="req-789")
```

### Client Errors

```python
from platform_core import (
    DataBankClientError,
    NotFoundError,
    AuthorizationError,
    BadRequestError,
    ConflictError,
    ForbiddenError,
    RangeNotSatisfiableError,
    InsufficientStorageClientError,
)
```

## Event Schemas

Typed event schemas for inter-service communication:

### Job Events (Generic)

```python
from platform_core.job_events import (
    JobEventV1,
    decode_job_event,
    encode_job_event,
    default_events_channel,
)

# Domain-parametric events (turkic, handwriting, etc.)
channel = default_events_channel("turkic")  # "events:turkic"
```

### Digits Metrics Events

```python
from platform_core.digits_metrics_events import (
    DigitsMetricsEventV1,
    decode_digits_metrics_event,
    encode_digits_metrics_event,
)
```

### Trainer Metrics Events

```python
from platform_core.trainer_metrics_events import (
    TrainerMetricsEventV1,
    decode_trainer_metrics_event,
    encode_trainer_metrics_event,
)
```

### Data Bank Events

```python
from platform_core.data_bank_events import (
    DataBankEventV1,
    decode_data_bank_event,
    encode_data_bank_event,
)
```

## Torch Types

Protocol types for PyTorch compatibility without importing torch:

```python
from platform_core import (
    TensorProtocol,
    DeviceProtocol,
    DTypeProtocol,
    TrainableModel,
    ImageClassificationDataset,
    TensorIterable,
    TensorIterator,
    ThreadConfig,
    PILImage,
    configure_torch_threads,
    get_num_threads,
    set_manual_seed,
)

# Configure threading
config: ThreadConfig = configure_torch_threads(max_threads=4)
print(f"Using {config['threads']} threads")
```

## API Reference

### Error Types

| Type | Description |
|------|-------------|
| `AppError` | Structured application error |
| `ErrorCode` | Standard error code enum |
| `ErrorCodeBase` | Base for custom error codes |
| `HandwritingErrorCode` | Handwriting service errors |
| `TranscriptErrorCode` | Transcript service errors |

### Validation Functions

| Function | Description |
|----------|-------------|
| `validate_str` | Validate string field |
| `validate_int_range` | Validate integer in range |
| `validate_float_range` | Validate float in range |
| `validate_bool` | Validate boolean field |
| `validate_optional_literal` | Validate optional literal |
| `validate_required_literal` | Validate required literal |
| `load_json_dict` | Parse JSON to dict |

### Health Types

| Type | Description |
|------|-------------|
| `HealthResponse` | Liveness response |
| `ReadyResponse` | Readiness response |
| `healthz` | Create liveness response |

### Logging Functions

| Function | Description |
|----------|-------------|
| `setup_logging` | Configure structured logging |
| `get_logger` | Get named logger |
| `stdlib_logging` | Get stdlib logger |

### Client Types

| Type | Description |
|------|-------------|
| `DataBankClient` | Data bank API client |
| `FileInfoDict` | File info response |
| `HeadInfo` | HEAD response info |

### Torch Protocol Types

| Type | Description |
|------|-------------|
| `TensorProtocol` | PyTorch tensor interface |
| `DeviceProtocol` | Device specification |
| `DTypeProtocol` | Data type specification |
| `TrainableModel` | Model with forward/parameters |
| `ImageClassificationDataset` | Dataset protocol |
| `ThreadConfig` | Thread configuration result |

### Utility Types

| Type | Description |
|------|-------------|
| `JSONValue` | JSON-compatible value type |
| `FastAPIAppAdapter` | FastAPI app protocol |

## Development

```bash
make lint   # guard checks, ruff, mypy
make test   # pytest with coverage
make check  # lint + test
```

## Requirements

- Python 3.11+
- FastAPI, httpx
- 100% test coverage enforced
