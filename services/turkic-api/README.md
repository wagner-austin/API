# Turkic API

Production-grade REST API for Turkic language corpus processing with **100% test coverage**, **strict type safety**, and **zero technical debt**.

## Features

- **Corpus Download**: Stream from OSCAR, Wikipedia, and CulturaX datasets
- **Language Detection**: FastText-based language identification
- **IPA Transliteration**: Convert text to International Phonetic Alphabet
- **Script Support**: Cyrillic, Latin, and Arabic scripts
- **Supported Languages**: Kazakh (kk), Kyrgyz (ky), Uzbek (uz), Turkish (tr), Uyghur (ug), Finnish (fi), Azerbaijani (az), English (en)

## Technology Stack

- **Framework**: FastAPI (async)
- **Job Queue**: Redis + RQ
- **Deployment**: Railway
- **Package Management**: Poetry
- **Type Checking**: mypy (strict mode)
- **Testing**: pytest (100% coverage required)
- **Quality**: Guard scripts enforcing strict patterns

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+

### Installation

1. **Clone the repository**
```bash
cd services/turkic-api
```

2. **Install dependencies**
```bash
poetry install --with dev
```

Or use the setup script:
```bash
python scripts/setup_dev.py
```

3. **Run tests**
```bash
make check
```

This runs:
- Guard scripts (code quality enforcement)
- Ruff (linting)
- Mypy (strict type checking)
- Pytest (100% coverage required)

### Run the Service

```bash
# Development
poetry run hypercorn 'turkic_api.api.main:create_app()' --bind 0.0.0.0:8000 --reload

# Production
poetry run hypercorn 'turkic_api.api.main:create_app()' --bind [::]:${PORT:-8000}
```

### Verify

```bash
curl http://localhost:8000/healthz
# {"status": "ok"}
```

### Development Commands

```bash
make check      # Run all checks (guards, linting, types, tests)
make test       # Run tests with coverage
make lint       # Run linters only
make format     # Auto-format code
```

## Project Structure

```
turkic-api/
├── src/turkic_api/          # Source code
│   ├── api/                 # FastAPI application
│   │   ├── main.py         # App factory & endpoints
│   │   ├── models.py       # TypedDict request/response models
│   │   ├── services.py     # Business logic
│   │   ├── jobs.py         # RQ job definitions
│   │   ├── dependencies.py # Dependency injection
│   │   └── health.py       # Health check logic
│   └── core/               # Core business logic
│       ├── translit.py     # IPA transliteration engine
│       ├── langid.py       # Language detection
│       ├── corpus.py       # Corpus streaming
│       └── rules/          # Transliteration rules
├── scripts/                # Utility scripts
│   ├── guard.py           # Code quality enforcement
│   └── setup_dev.py       # Development setup
├── tests/                 # Test suite (100% coverage)
├── docs/                  # Documentation
├── Makefile              # Development commands
├── pyproject.toml        # Poetry configuration
└── Dockerfile            # Container image
```

## API Reference

For complete API documentation, see [docs/api.md](./docs/api.md).

### Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/readyz` | GET | Readiness probe (checks Redis + data volume) |
| `/api/v1/jobs` | POST | Create corpus processing job |
| `/api/v1/jobs/{job_id}` | GET | Get job status |
| `/api/v1/jobs/{job_id}/result` | GET | Download processed result file |

## Quality Standards

### Type Safety
- ✅ `mypy --strict` with zero warnings
- ✅ **Zero `Any` types** (disallow_any_expr)
- ✅ **Zero type casts**
- ✅ **Zero `type: ignore` comments**
- ✅ TypedDict models (no Pydantic)

### Testing
- ✅ **100% statement coverage**
- ✅ **100% branch coverage**
- ✅ All tests must pass before commit

### Code Quality
- ✅ Guard scripts enforce patterns
- ✅ No `TYPE_CHECKING` blocks
- ✅ No try/except blocks (explicit error handling)
- ✅ Ruff + Black formatting

## Environment Variables

```bash
# Redis connection
TURKIC_REDIS_URL=redis://localhost:6379/0

# Data storage
TURKIC_DATA_DIR=/data

# Data bank integration (either direct URL or gateway)
TURKIC_DATA_BANK_API_URL=https://api.databank.example.com
TURKIC_DATA_BANK_API_KEY=your-api-key

# Alternative: Use API gateway (auto-appends /data-bank)
# API_GATEWAY_URL=https://gateway.example.com
```

## Deployment

### Railway

```bash
# Deploy automatically via git push
git push railway main
```

Configuration in `railway.toml`:
- Dockerfile-based build
- Health check: `/healthz`
- Auto-restart on failure

### Docker

```bash
# Build image
docker build -t turkic-api .

# Run container
docker run -p 8000:8000 \
  -e TURKIC_REDIS_URL=redis://redis:6379/0 \
  -e TURKIC_DATA_DIR=/data \
  -e TURKIC_DATA_BANK_API_KEY=your-api-key \
  turkic-api
```

## Development Workflow

### Before Committing

```bash
make check  # Must pass 100%
```

### Building Balanced Corpora

Download and build IPA-balanced corpora from OSCAR:

```bash
poetry run python scripts/build_balanced_corpora.py \
  --source oscar --threshold 0.95 --out-dir data/balanced/oscar
```

The script:
- Streams from HuggingFace OSCAR dataset (set `HF_TOKEN` env var)
- Filters with FastText language ID at the confidence threshold
- Transliterates to IPA
- Balances all corpora to the bottleneck language's character count
- Writes manifest.json with run metadata

### Adding New Language Support

1. Add transliteration rules to `src/turkic_api/core/rules/`:
   - `{lang}_ipa.rules` for IPA transliteration
   - `{lang}_lat.rules` for Latin transliteration

2. Add tests in `tests/`:
   - Letter-level tests: `test_{lang}_ipa_letters.py`
   - Word-level tests: `test_{lang}_words.py`

3. Run `make check` to verify 100% coverage

### Guard Scripts

The project uses guard scripts to enforce code quality:
- No `Any`, `cast`, or `type: ignore`
- No `TYPE_CHECKING` imports
- No Pydantic imports
- No try/except blocks
- Proper Protocol usage

Run manually: `poetry run python -m scripts.guard`

## Architecture

### Pure Python Transliteration
- No PyICU dependency
- Rules-based transliteration engine
- Unicode normalization (NFC)
- Deterministic output

### Dependency Injection
- FastAPI `Depends()` for all dependencies
- No global state
- Protocol-based interfaces
- Explicit logger injection

### Job Processing
- Redis for job queue
- RQ workers for background processing
- Atomic status updates
- Data bank integration for result storage

## Testing

```bash
# Run all tests
make test

# Run specific test file
poetry run pytest tests/test_translit.py -v

# Run with coverage report
poetry run pytest --cov-report=html
```

### Coverage Requirements
- 100% statement coverage (no exceptions)
- 100% branch coverage (no exceptions)
- All tests must pass

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

Apache-2.0

## Documentation

- [API Reference](docs/api.md) - Complete endpoint documentation
- [Setup Guide](docs/setup_guide.md) - Development environment setup
- [Architecture & Design](DESIGN.md) - System design and patterns
- [Contributing Guidelines](docs/CONTRIBUTING.md) - Code standards and workflow
