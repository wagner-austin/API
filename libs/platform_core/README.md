# platform-core

Shared platform utilities: error handling, validation, logging, health checks, typed event schemas, and service clients.

## Installation

```bash
poetry add platform-core
```

## Quick Start

```python
from platform_core import (
    AppError,
    ErrorCode,
    validate_str,
    validate_int_range,
    get_logger,
    setup_logging,
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
raise AppError(code=ErrorCode.NOT_FOUND, message="User not found", http_status=404)

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

## OAuth 2.0

Reusable OAuth 2.0 utilities for authorization flows with PKCE support.

### Types

```python
from platform_core import (
    OAuthCredentials,
    OAuthTokens,
    OAuthTokenResponse,
    TokenType,
    encode_oauth_credentials,
    decode_oauth_credentials,
    encode_oauth_tokens,
    decode_oauth_tokens,
    encode_oauth_token_response,
    decode_oauth_token_response,
)

# Client credentials
credentials = OAuthCredentials(
    client_id="your-client-id",
    client_secret="your-client-secret",
    redirect_uri="http://localhost:8080/callback",
)

# Access and refresh tokens
tokens = OAuthTokens(
    access_token="access-token",
    refresh_token="refresh-token",
    expires_at=1735200000,
    token_type="Bearer",
)
```

### PKCE Functions

```python
from platform_core import (
    generate_code_verifier,
    generate_code_challenge,
    generate_state,
)

# Generate PKCE values for authorization
verifier = generate_code_verifier()
challenge = generate_code_challenge(verifier)
state = generate_state()
```

### Token Utilities

```python
from platform_core import is_token_expired, build_authorization_url

# Check if token needs refresh
current_time = int(time.time())
if is_token_expired(tokens, current_time, buffer_seconds=60):
    # Refresh the token
    ...

# Build authorization URL
auth_url = build_authorization_url(
    auth_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
    client_id=credentials["client_id"],
    redirect_uri=credentials["redirect_uri"],
    code_challenge=challenge,
    state=state,
    scopes=("openid", "email"),
)
```

### Token Exchange

```python
from platform_core import exchange_authorization_code, refresh_access_token

# Exchange authorization code for tokens
tokens = exchange_authorization_code(
    token_endpoint="https://oauth2.googleapis.com/token",
    credentials=credentials,
    code="auth-code-from-callback",
    code_verifier=verifier,
    http_post=your_http_post_function,
    current_time=int(time.time()),
)

# Refresh expired access token
new_tokens = refresh_access_token(
    token_endpoint="https://oauth2.googleapis.com/token",
    credentials=credentials,
    refresh_token=tokens["refresh_token"],
    http_post=your_http_post_function,
    current_time=int(time.time()),
)
```

### OAuth Testing Utilities

```python
from platform_core.oauth_testing import (
    make_fake_http_post,
    make_raising_http_post,
    make_sequenced_http_post,
    make_fake_current_time,
    make_advancing_current_time,
    make_token_response_json,
    make_error_response_json,
    make_test_credentials,
    make_test_tokens,
    make_test_token_response,
)

# Create fake HTTP POST that returns token response
http_post = make_fake_http_post(make_token_response_json())

# Create test credentials and tokens
creds = make_test_credentials()
tokens = make_test_tokens(expired=True, current_time=1735200000)
```

### OAuth Error Codes

```python
from platform_core import OAuthErrorCode, AppError

# OAuth-specific errors
raise AppError(OAuthErrorCode.TOKEN_EXPIRED, "Token has expired")
```

| Code | Description |
|------|-------------|
| `AUTH_FAILED` | Authorization failed |
| `INVALID_GRANT` | Invalid authorization grant |
| `INVALID_STATE` | State mismatch (CSRF) |
| `TOKEN_EXPIRED` | Access token expired |
| `TOKEN_EXCHANGE_FAILED` | Code exchange failed |
| `TOKEN_REFRESH_FAILED` | Token refresh failed |
| `MISSING_REFRESH_TOKEN` | No refresh token in response |
| `TOKEN_ENDPOINT_ERROR` | Token endpoint returned error |

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

## Service Configuration

Typed configuration loaders for service-specific settings.

### Covenant Radar Settings

```python
from platform_core import (
    CovenantRadarSettings,
    CovenantRadarDatadogConfig,
    load_covenant_radar_settings,
)

# Load settings from environment
settings = load_covenant_radar_settings()

# Access typed configuration
print(settings["redis"]["url"])
print(settings["database_url"])
print(settings["datadog"]["enabled"])
print(settings["app"]["ml_backend"])
```

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `""` | PostgreSQL connection URL |
| `REDIS_URL` or `REDIS__URL` | `redis://redis:6379/0` | Redis connection URL |
| `APP_ENV` | `dev` | Environment (`dev` or `prod`) |
| `LOGGING__LEVEL` | `INFO` | Log level |
| `APP__ML_BACKEND` | `xgboost` | ML backend (`xgboost`, `mlp`, `lstm`, `lightgbm`) |
| `APP__DATA_ROOT` | `/data` | Data root directory |
| `APP__MODELS_ROOT` | `/data/models` | Models directory |
| `APP__LOGS_ROOT` | `/data/logs` | Logs directory |

**Datadog Configuration:**

| Variable | Default | Description |
|----------|---------|-------------|
| `DATADOG__ENABLED` | `false` | Enable Datadog integration |
| `DATADOG__SERVICE` | `covenant-radar-api` | Service name for traces |
| `DATADOG__ENV` | `dev` | Environment (`dev`, `staging`, `production`) |
| `DATADOG__VERSION` | `0.0.0` | Service version |
| `DATADOG__AGENT_HOST` | `localhost` | Datadog agent host |
| `DATADOG__DOGSTATSD_PORT` | `8125` | DogStatsD UDP port |
| `DATADOG__TRACE_ENABLED` | `true` | Enable APM tracing |

### Configuration Types

| Type | Description |
|------|-------------|
| `CovenantRadarSettings` | Root settings TypedDict |
| `CovenantRadarRedisConfig` | Redis connection config |
| `CovenantRadarAppConfig` | Application paths and ML backend |
| `CovenantRadarLoggingConfig` | Logging level config |
| `CovenantRadarRQConfig` | RQ queue settings |
| `CovenantRadarDatadogConfig` | Datadog APM and metrics config |

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
| `OAuthErrorCode` | OAuth 2.0 errors |

### OAuth Types

| Type | Description |
|------|-------------|
| `OAuthCredentials` | OAuth client credentials (client_id, client_secret, redirect_uri) |
| `OAuthTokens` | Access and refresh tokens with expiry |
| `OAuthTokenResponse` | Raw token endpoint response |
| `TokenType` | Token type literal ("Bearer") |

### OAuth Functions

| Function | Description |
|----------|-------------|
| `generate_code_verifier` | Generate PKCE code verifier |
| `generate_code_challenge` | Generate PKCE code challenge (S256) |
| `generate_state` | Generate CSRF state parameter |
| `is_token_expired` | Check if token needs refresh |
| `build_authorization_url` | Build OAuth authorization URL |
| `exchange_authorization_code` | Exchange code for tokens |
| `refresh_access_token` | Refresh expired access token |
| `encode_oauth_credentials` | Serialize credentials to JSON |
| `decode_oauth_credentials` | Deserialize credentials from JSON |
| `encode_oauth_tokens` | Serialize tokens to JSON |
| `decode_oauth_tokens` | Deserialize tokens from JSON |
| `encode_oauth_token_response` | Serialize token response to JSON |
| `decode_oauth_token_response` | Deserialize token response from JSON |

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
