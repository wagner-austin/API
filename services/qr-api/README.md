# QR API

Strict, typed QR code generation service built with FastAPI. Produces PNG QR codes via the Segno library with full input validation, customizable styling, and comprehensive error handling.

## Features

- **URL Validation**: Scheme enforcement (http/https), host validation (domains, IPv4, IPv6, localhost)
- **Customizable Output**: Error correction levels, colors, sizing
- **Type Safety**: mypy strict mode, zero `Any` types, TypedDict models
- **100% Test Coverage**: Statements and branches
- **Monorepo Integration**: Guard rules, platform_core logging and error handling

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+

### Installation

```bash
cd services/qr-api
poetry install --with dev
```

### Run the Service

```bash
# Development
poetry run hypercorn qr_api.asgi:app --bind 0.0.0.0:8080 --reload

# Production
poetry run hypercorn qr_api.asgi:app --bind [::]:${PORT:-8080}
```

### Verify

```bash
curl http://localhost:8080/healthz
# {"status": "ok"}
```

## API Reference

For complete API documentation, see [docs/api.md](./docs/api.md).

### Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/v1/qr` | POST | Generate QR code PNG from URL |

---

## Configuration

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PORT` | int | `8080` | Server port |
| `QR_DEFAULT_ERROR_CORRECTION` | string | `"M"` | Default ECC level (L/M/Q/H) |
| `QR_DEFAULT_BOX_SIZE` | int | `10` | Default pixels per module (5-20) |
| `QR_DEFAULT_BORDER` | int | `1` | Default quiet zone width (1-10) |
| `QR_DEFAULT_FILL_COLOR` | string | `"#000000"` | Default dark color |
| `QR_DEFAULT_BACK_COLOR` | string | `"#FFFFFF"` | Default light color |

**Example `.env`:**

```bash
PORT=8080
QR_DEFAULT_ERROR_CORRECTION=M
QR_DEFAULT_BOX_SIZE=10
QR_DEFAULT_BORDER=1
QR_DEFAULT_FILL_COLOR=#000000
QR_DEFAULT_BACK_COLOR=#FFFFFF
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
poetry run pytest tests/test_validators.py -v

# Run with coverage report
poetry run pytest --cov-report=html
```

### Type Checking

```bash
poetry run mypy src tests scripts
```

---

## Project Structure

```
qr-api/
├── src/qr_api/
│   ├── __init__.py      # Package exports
│   ├── app.py           # FastAPI factory and routes
│   ├── asgi.py          # ASGI entry point
│   ├── types.py         # TypedDict models (QRPayload, QROptions)
│   ├── validators.py    # URL, color, dimension validation
│   ├── generator.py     # Segno QR generation
│   └── settings.py      # Environment config loading
├── tests/
│   ├── test_health.py
│   ├── test_qr_handler.py
│   ├── test_validators.py
│   ├── test_generator.py
│   ├── test_settings.py
│   └── ...
├── scripts/
│   └── guard.py         # Monorepo guard orchestrator
├── Dockerfile           # Multi-stage build
├── pyproject.toml       # Poetry + tool config
├── Makefile             # Development commands
└── railway.toml         # Railway deployment config
```

---

## Deployment

### Docker

```bash
# Build
docker build -t qr-api:latest .

# Run
docker run -p 8080:8080 qr-api:latest

# With custom defaults
docker run -p 8080:8080 \
  -e QR_DEFAULT_ERROR_CORRECTION=H \
  -e QR_DEFAULT_BOX_SIZE=12 \
  qr-api:latest
```

### Railway

The service is configured for Railway deployment:

```toml
# railway.toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "/healthz"
healthcheckTimeout = 30
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 3
```

Deploy via:
```bash
railway up
```

---

## Architecture

### Request Flow

```
Client Request
      │
      ▼
┌─────────────────┐
│  ASGI Entry     │  (asgi.py)
│  Request ID MW  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI App    │  (app.py)
│  Route Handler  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validators     │  (validators.py)
│  URL, colors,   │
│  dimensions     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generator      │  (generator.py)
│  Segno QR       │
└────────┬────────┘
         │
         ▼
   PNG Response
```

### Type Models

```python
# Raw request payload (partial fields)
class QRPayload(TypedDict, total=False):
    url: str
    ecc: ECCLevel
    box_size: int
    border: int
    fill_color: str
    back_color: str

# Validated options (all fields required)
class QROptions(TypedDict, total=True):
    url: str
    ecc: ECCLevel        # Literal["L", "M", "Q", "H"]
    box_size: int        # 5-20
    border: int          # 1-10
    fill_color: str      # #RGB or #RRGGBB
    back_color: str      # #RGB or #RRGGBB
```

---

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `hypercorn` | ASGI server |
| `segno` | QR code generation |
| `platform-core` | Logging, errors, config |

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
