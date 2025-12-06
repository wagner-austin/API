# Turkic API Documentation

Welcome to the documentation for the Turkic API. This is a production-grade REST API service for Turkic language corpus processing with 100% test coverage and strict type safety.

## Quick Links

- [API Reference](api.md) - Complete endpoint documentation
- [Setup Guide](setup_guide.md) - Development environment setup
- [Architecture & Design](../DESIGN.md) - System design and patterns
- [Contributing Guidelines](CONTRIBUTING.md) - Code standards and workflow
- [Main README](../README.md) - Project overview

## Project Overview

Turkic API is a FastAPI-based service that provides:
- Corpus download from OSCAR, Wikipedia, and CulturaX datasets
- Language detection using FastText
- IPA transliteration for multiple Turkic languages
- Script conversion (Cyrillic ↔ Latin ↔ Arabic)
- Background job processing with Redis + RQ

### Supported Languages

- Kazakh (kk)
- Kyrgyz (ky)
- Uzbek (uz)
- Turkish (tr)
- Uyghur (ug)
- Finnish (fi)
- Azerbaijani (az)

## Project Structure

```
turkic-api/
├── src/
│   └── turkic_api/              # Main package
│       ├── api/                 # FastAPI application (9 modules)
│       │   ├── main.py         # App factory & routes
│       │   ├── services.py     # Endpoint handlers
│       │   ├── jobs.py         # Job processing implementation
│       │   ├── worker_entry.py # RQ worker entrypoint
│       │   ├── dependencies.py # Dependency injection
│       │   ├── config.py       # Settings TypedDict
│       │   ├── models.py       # Request/response parsers
│       │   ├── types.py        # Protocols & type aliases
│       │   ├── errors.py       # Error handlers
│       │   └── job_store.py    # Redis job state store
│       └── core/               # Core business logic (12 modules)
│           ├── translit.py     # IPA transliteration engine
│           ├── langid.py       # Language identification
│           ├── corpus.py       # Corpus service
│           ├── corpus_download.py  # Corpus filtering
│           ├── transliteval.py # Rule-based transliteration
│           ├── models.py       # Core TypedDict models
│           └── rules/          # Transliteration rule files
├── scripts/
│   ├── guard.py               # Monorepo rule enforcement
│   ├── setup_dev.py          # Development setup automation
│   └── README.md
├── tests/                     # Test suite (52 modules, 503 tests, 100% coverage)
├── docs/                      # Documentation (you are here)
├── Makefile                  # PowerShell-based development commands
├── Dockerfile                # Multi-stage build (api + worker targets)
├── pyproject.toml            # Poetry dependencies, strict mypy config
└── railway.toml              # Railway deployment config
```

## Getting Started

### Installation

```bash
# Install with Poetry
poetry install --with dev

# Verify installation
poetry run python -c "import turkic_api; print('Installation successful')"
```

### Run Tests

```bash
make check  # Runs guards, linters, type checks, and tests
```

### Start Development Server

```bash
# API service
poetry run hypercorn turkic_api.api.main:create_app --bind localhost:8000 --reload

# Worker service (separate terminal)
poetry run python -m turkic_api.api.worker_entry
```

## Architecture Highlights

### Type Safety First
- 100% type coverage with mypy --strict
- Zero `Any` types allowed
- TypedDict models (no Pydantic)
- Protocol-based dependency injection

### Quality Standards
- 100% test coverage (statements + branches)
- Guard scripts enforce coding patterns
- No try/except blocks (explicit error handling)
- Automated quality checks in CI/CD

### Pure Python
- No PyICU dependency
- Rules-based transliteration
- Cross-platform compatibility
- Deterministic output

## Development

### Code Quality Checks

```bash
make check     # All checks
make lint      # Linting only
make test      # Tests only
make format    # Auto-format
```

### Running the API

```bash
# API service (development with auto-reload)
poetry run hypercorn turkic_api.api.main:create_app --bind localhost:8000 --reload

# Worker service (separate terminal)
poetry run python -m turkic_api.api.worker_entry

# Production (Railway)
# Automatic deployment via git push to main branch
```

## API Reference

See [api.md](api.md) for complete endpoint documentation including:

- Health endpoints (`/healthz`, `/readyz`)
- Job management (`POST /api/v1/jobs`, `GET /api/v1/jobs/{job_id}`)
- Result download (`GET /api/v1/jobs/{job_id}/result`)
- Error handling and status codes
- Request/response examples

### Quick Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/jobs` | Create a corpus processing job |
| `GET` | `/api/v1/jobs/{job_id}` | Get job status |
| `GET` | `/api/v1/jobs/{job_id}/result` | Download processed result |
| `GET` | `/healthz` | Liveness probe |
| `GET` | `/readyz` | Readiness probe |

## Support

For issues or questions:
- Check the [setup guide](setup_guide.md) first
- Review [DESIGN.md](../DESIGN.md) for architectural context
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
