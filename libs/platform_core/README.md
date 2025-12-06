# platform-core

Shared platform utilities: error handling, validation, logging, health checks, and typed event schemas.

## Installation

```bash
poetry add platform-core
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

## Validation

```python
from platform_core import (
    validate_str,
    validate_int_range,
    validate_bool,
    validate_optional_literal,
    load_json_dict,
)

# Validates and raises AppError on failure
name = validate_str(data.get("name"), "name")
count = validate_int_range(data.get("count"), "count", ge=1, le=100)
enabled = validate_bool(data.get("enabled"), "enabled", default=False)
mode = validate_optional_literal(data.get("mode"), "mode", frozenset(["fast", "slow"]))
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
from platform_core import get_logger, setup_logging

setup_logging()
logger = get_logger("my-service")
logger.info("started", extra={"port": 8000})
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

## Event Schemas

Typed event schemas for inter-service communication:

- `platform_core.job_events` - Generic job lifecycle events (domain-parametric, shared)
- `platform_core.digits_metrics_events` - Digits training metrics events
- `platform_core.trainer_metrics_events` - Model trainer metrics events
- `platform_core.data_bank_events` - Data bank job events

## Requirements

- Python 3.11+
- FastAPI, httpx
- 100% test coverage enforced
