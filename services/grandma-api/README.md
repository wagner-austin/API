# Grandma API

Strictly typed, modular multi-language audio to English translation API using OpenAI Whisper. Supports 57 input languages with automatic language detection. Features Protocol-based dependency injection, structured error handling, and 100% test coverage.

## Features

- **Multi-Language Translation**: Translates speech from 57 languages to English text using OpenAI Whisper
- **Automatic Language Detection**: Whisper auto-detects the input language
- **Protocol-Based DI**: ServiceContainer with typed dependency injection
- **Strict Typing**: mypy strict mode, zero `Any` types, no casts, no type ignores
- **Structured Errors**: Consistent `{code, message, request_id}` error responses
- **Request Tracing**: Automatic request ID generation and propagation
- **100% Test Coverage**: Statements and branches

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+
- OpenAI API key with Whisper access

### Installation

```bash
cd services/grandma-api
poetry install --with dev
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your values
# Required: OPENAI_API_KEY, API_TOKEN
```

### Run Locally

```bash
# Start the API server
poetry run hypercorn 'grandma_api.asgi:app' --bind 0.0.0.0:8080

# Verify
curl http://localhost:8080/healthz
```

### Run with Docker

```bash
# Build and run
docker compose up --build

# Verify
curl http://localhost:8080/healthz
```

## API Reference

For complete API documentation, see [docs/api.md](./docs/api.md).

### Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/translate` | POST | Translate audio (any supported language) to English |

### Translation Example

```bash
curl -X POST http://localhost:8080/translate \
  -F "audio=@recording.webm" \
  -F "token=your-api-token"
```

**Response:**
```json
{
  "text": "Hello, how are you?"
}
```

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key for Whisper |
| `API_TOKEN` | Yes | - | Authentication token for `/translate` |
| `PORT` | No | `8080` | Server port |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `LOG_FORMAT` | No | `json` | Log format (json or text) |

### Example .env

```bash
OPENAI_API_KEY=sk-your-openai-api-key
API_TOKEN=your-secure-api-token
PORT=8080
LOG_LEVEL=INFO
LOG_FORMAT=json
```

---

## Architecture

### Component Overview

```
grandma_api/
├── api/                    # FastAPI routes and middleware
│   ├── main.py            # App factory with ServiceContainer
│   ├── middleware.py      # API key authentication
│   ├── routes/            # Endpoint handlers
│   │   ├── health.py      # Liveness probe
│   │   └── translate.py   # Translation endpoint
│   ├── schemas/           # Request/response TypedDicts
│   │   └── translate.py   # TranslationResponse
│   └── validators/        # Request validation
│       └── translate.py   # Token and audio validation
├── core/                   # Core infrastructure
│   ├── container.py       # ServiceContainer for DI
│   └── _test_hooks.py     # Test dependency injection
├── config.py              # Settings with encode/decode
├── health.py              # Health check implementation
├── types.py               # Re-exports from api/schemas
└── asgi.py                # ASGI application entry point
```

### Protocol-Based Design

```python
class STTClientFactoryProtocol(Protocol):
    """Protocol for STT client factory function."""

    def __call__(self, api_key: str) -> STTClientProtocol:
        """Create STT client with given API key."""
        ...


class ServiceContainer:
    """Dependency injection container for grandma-api services."""

    settings: GrandmaApiSettings
    stt_client_factory: STTClientFactoryProtocol

    @classmethod
    def from_settings(cls, settings: GrandmaApiSettings) -> ServiceContainer:
        """Create a ServiceContainer with production defaults."""
        ...

    def get_stt_client(self) -> STTClientProtocol:
        """Get an STT client configured with the API key."""
        ...
```

### Request Flow

```
┌─────────────────┐
│   Client        │
│   (Browser/App) │
└────────┬────────┘
         │ POST /translate
         ▼
┌─────────────────┐
│   FastAPI       │
│   (CORS, ReqID) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Middleware    │
│   (Validation)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   Translate     │────▶│   STT Client    │
│   Route         │     │   (OpenAI)      │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Response      │
│   (JSON)        │
└─────────────────┘
```

---

## Development

### Commands

```bash
make lint         # Run guards + ruff + mypy
make test         # Run pytest with coverage
make check        # Run lint + test
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
poetry run pytest tests/test_api_routes_translate.py -v

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
grandma-api/
├── src/grandma_api/
│   ├── api/
│   │   ├── main.py             # App factory
│   │   ├── middleware.py       # API key auth
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   └── translate.py
│   │   ├── schemas/
│   │   │   └── translate.py
│   │   ├── validators/
│   │   │   └── translate.py
│   │   └── _test_hooks.py
│   ├── core/
│   │   ├── container.py        # ServiceContainer
│   │   └── _test_hooks.py
│   ├── asgi.py                 # ASGI entry point
│   ├── config.py               # Settings
│   ├── health.py               # Health check
│   └── types.py                # Type re-exports
├── tests/
├── scripts/
├── docs/
│   └── api.md                  # API reference
├── web/                        # Browser frontend
│   ├── src/                    # TypeScript source
│   ├── tests/                  # Unit tests
│   ├── assets/                 # CSS and built JS
│   ├── index.html
│   └── package.json
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── Makefile
├── .env.example
├── DEPLOYING_RAILWAY.md
└── README.md
```

---

## Deployment

### Docker

```bash
# Build image
docker build -t grandma-api -f Dockerfile ../..

# Run container
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=sk-your-key \
  -e API_TOKEN=your-token \
  grandma-api
```

### Docker Compose

```bash
# Start service
docker compose up -d

# Check logs
docker compose logs -f

# Stop service
docker compose down
```

### Railway Deployment

See [DEPLOYING_RAILWAY.md](./DEPLOYING_RAILWAY.md) for Railway deployment instructions.

### Health Checks

- **Endpoint**: `/healthz`
- **Response**: `{"status": "ok"}`

---

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `hypercorn` | ASGI server |
| `python-multipart` | Form data parsing |
| `platform-core` | Logging, errors, config |
| `platform-stt` | OpenAI Whisper client |

### Development

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |
| `pytest-asyncio` | Async test support |
| `pytest-cov` | Coverage reporting |
| `pytest-xdist` | Parallel tests |
| `mypy` | Type checking |
| `ruff` | Linting/formatting |
| `httpx` | HTTP test client |

---

## Error Handling

### Error Response Format

```json
{
  "code": "ERROR_CODE",
  "message": "Human-readable description",
  "request_id": "uuid-for-tracing"
}
```

### Standard Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_INPUT` | 400 | Invalid request (empty audio, etc.) |
| `UNAUTHORIZED` | 401 | Invalid authentication token |
| `INTERNAL_ERROR` | 500 | Internal server error |

---

## Quality Standards

- **Type Safety**: mypy strict mode, no `Any`, no `cast`, no `type: ignore`
- **Coverage**: 100% statements and branches
- **TypedDicts**: All have encode/decode functions with require_* validation
- **Test Hooks**: Each module has `_test_hooks.py` for dependency injection
- **Logging**: Structured JSON via platform_core
- **Errors**: Consistent `{code, message, request_id}` format

---

## Language Support

### Input Languages (57)

Whisper auto-detects the source language. The following languages are supported:

| Language | Code | | Language | Code | | Language | Code |
|----------|------|-|----------|------|-|----------|------|
| Afrikaans | `af` | | Hindi | `hi` | | Portuguese | `pt` |
| Arabic | `ar` | | Hungarian | `hu` | | Romanian | `ro` |
| Armenian | `hy` | | Icelandic | `is` | | Russian | `ru` |
| Azerbaijani | `az` | | Indonesian | `id` | | Serbian | `sr` |
| Belarusian | `be` | | Italian | `it` | | Slovak | `sk` |
| Bosnian | `bs` | | Japanese | `ja` | | Slovenian | `sl` |
| Bulgarian | `bg` | | Kannada | `kn` | | Spanish | `es` |
| Catalan | `ca` | | Kazakh | `kk` | | Swahili | `sw` |
| Chinese | `zh` | | Korean | `ko` | | Swedish | `sv` |
| Croatian | `hr` | | Latvian | `lv` | | Tagalog | `tl` |
| Czech | `cs` | | Lithuanian | `lt` | | Tamil | `ta` |
| Danish | `da` | | Macedonian | `mk` | | Thai | `th` |
| Dutch | `nl` | | Malay | `ms` | | Turkish | `tr` |
| English | `en` | | Marathi | `mr` | | Ukrainian | `uk` |
| Estonian | `et` | | Maori | `mi` | | Urdu | `ur` |
| Finnish | `fi` | | Nepali | `ne` | | Vietnamese | `vi` |
| French | `fr` | | Norwegian | `no` | | Welsh | `cy` |
| Galician | `gl` | | Persian | `fa` | | | |
| German | `de` | | Polish | `pl` | | | |
| Greek | `el` | | | | | | |
| Hebrew | `he` | | | | | | |

### Output Language

- **English only** - Whisper API limitation; the translate endpoint only supports English output

### Language Detection

- **Automatic** - Whisper performs internal language detection; no `language` parameter required
- **Accuracy** - Best for major languages (English, Spanish, French, German, etc.)
- **Low-resource languages** - May have reduced accuracy (Icelandic, Welsh, Swahili, etc.)

---

## Web Frontend

The `web/` directory contains a vanilla TypeScript browser application for audio recording and translation.

### Features

- **Token-based authentication** - Simple password login
- **Audio recording** - MediaRecorder API with WebM format
- **Real-time translation** - Sends recordings to the API and displays results
- **Offline-capable** - Static files, no server-side rendering

### Tech Stack

| Tool | Purpose |
|------|---------|
| TypeScript | Strict typing, ES2022 target |
| Vitest | Testing with jsdom |
| ESLint | Linting (flat config) |
| V8 Coverage | 100% coverage enforcement |

### Development

```bash
cd web

# Install dependencies
npm install

# Run checks (lint + typecheck + tests)
npm run check

# Build for production
npm run build

# Watch tests
npm run test
```

### Architecture

```
web/
├── src/
│   ├── main.ts           # Entry point (auto-init)
│   ├── app.ts            # App class with Promise-returning handlers
│   ├── api.ts            # translateAudio() API client
│   ├── config.ts         # loadConfig() from config.json
│   ├── recorder.ts       # startRecording/stopRecording
│   ├── storage.ts        # Token persistence (localStorage)
│   ├── types.ts          # TypedDicts with encode/decode/require
│   ├── dom.ts            # DOM utilities
│   ├── _test_hooks.ts    # Dependency injection for testing
│   └── testing.ts        # Public test utilities (fakes)
├── tests/unit/           # Unit tests (100% coverage)
├── assets/
│   ├── css/main.css      # Styles
│   └── js/               # Build output (git-ignored)
├── index.html            # Single-page app
├── config.json           # API_BASE_URL configuration
├── tsconfig.json         # Type checking config
├── tsconfig.build.json   # Production build config
├── vitest.config.ts      # Test configuration
├── eslint.config.js      # ESLint flat config
└── package.json
```

### Quality Standards

- **Type Safety**: Strict mode, no `any`, no casts, no type ignores
- **Coverage**: 100% statements, branches, functions, lines
- **Test Hooks**: Dependency injection via `_test_hooks.ts`
- **Fakes**: No mocks - all test doubles are fakes via hooks

---

## Audio Format Support

| Format | Extension | MIME Type |
|--------|-----------|-----------|
| WebM | `.webm` | `audio/webm` |
| MP3 | `.mp3` | `audio/mpeg` |
| WAV | `.wav` | `audio/wav` |
| M4A | `.m4a` | `audio/m4a` |
| OGG | `.ogg` | `audio/ogg` |

**Recommended**: WebM for browser recordings (native MediaRecorder format)

---

## License

Apache-2.0
