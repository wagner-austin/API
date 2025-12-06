# Setup Guide for Turkic API

This document provides detailed setup instructions for the Turkic API service, a production-grade FastAPI microservice for Turkic language corpus processing.

## Prerequisites

Before setting up the project, ensure you have:

- **Python 3.11 or later** (tested with Python 3.11+)
- **Poetry 1.8+** (package and dependency management)
- **Redis 7.0+** (message queue and caching)
- **Git** (version control)
- **PowerShell** (Windows) or Bash (Linux/macOS)

## System Setup

### Python Installation

**Windows:**
```powershell
# Download and install Python 3.11+ from python.org
# Or use winget:
winget install Python.Python.3.11
```

**Linux/macOS:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# macOS (Homebrew)
brew install python@3.11
```

### Poetry Installation

Poetry is required for dependency management:

```bash
# Install Poetry (cross-platform)
curl -sSL https://install.python-poetry.org | python3 -

# Or via pip
pip install poetry

# Verify installation
poetry --version
```

Add Poetry to your PATH if needed:
- **Windows:** `C:\Users\<username>\AppData\Roaming\Python\Scripts`
- **Linux/macOS:** `~/.local/bin`

### Redis Installation

#### Option 1: Docker (Recommended)

```bash
# Pull and run Redis container
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Verify Redis is running
docker ps | grep redis
```

#### Option 2: Local Installation

**Windows:**
```powershell
# Install via Chocolatey
choco install redis-64

# Or download from: https://github.com/microsoftarchive/redis/releases
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

**macOS:**
```bash
brew install redis
brew services start redis
```

Verify Redis is running:
```bash
redis-cli ping
# Expected output: PONG
```

## Project Setup

### 1. Clone the Monorepo

The turkic-api service is part of a larger monorepo:

```bash
git clone <monorepo-url>
cd services/turkic-api
```

### 2. Install Dependencies

Install all dependencies including development tools:

```bash
# Install main and dev dependencies
poetry install --with dev

# Verify installation
poetry run python -c "import turkic_api; print('Installation successful')"
```

This installs:
- **Runtime dependencies:** FastAPI, Hypercorn, Redis, RQ, httpx
- **Platform libraries:** platform-core, platform-workers (from monorepo)
- **Dev dependencies:** pytest, pytest-cov, pytest-xdist, mypy, ruff
- **Test dependencies:** fakeredis

### 3. Platform Library Setup

The turkic-api service depends on two platform libraries located in the monorepo:

- `libs/platform_core` - Centralized logging, config, events
- `libs/platform_workers` - RQ worker harness, Redis protocols

These are installed automatically by Poetry using path dependencies:

```toml
[tool.poetry.dependencies]
platform-core = { path = "../../libs/platform_core", develop = true }
platform-workers = { path = "../../libs/platform_workers", develop = true }
```

If you encounter import errors, ensure the monorepo structure is correct:
```
monorepo/
├── libs/
│   ├── platform_core/
│   └── platform_workers/
└── services/
    └── turkic-api/
```

### 4. Environment Configuration

Create a `.env` file in the project root for local development:

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Data Directory (corpus files, language models)
DATA_DIR=./data

# Environment
ENVIRONMENT=local

# Data Bank API (file upload service)
DATA_BANK_API_URL=http://localhost:8001
DATA_BANK_API_KEY=dev-key-12345

# Optional: Railway deployment
RAILWAY_ENVIRONMENT=development
```

**Environment variables explained:**

- **REDIS_URL:** Redis connection string for job queue and caching
- **DATA_DIR:** Directory for corpus files and language ID models
- **ENVIRONMENT:** Deployment environment (local, test, staging, production)
- **DATA_BANK_API_URL:** URL for data-bank-api file upload service
- **DATA_BANK_API_KEY:** API key for data-bank-api authentication

### 5. Create Data Directory Structure

```bash
# Create required directories
mkdir -p data/corpus
mkdir -p data/models

# Download language ID model (optional for local dev)
# curl -L <model-url> -o data/models/langid.bin
```

The directory structure should be:
```
data/
├── corpus/           # Corpus text files (oscar_kk.txt, wikipedia_uz.txt, etc.)
└── models/           # Language identification models (langid.bin)
```

### 6. Verify Installation

Run the guard script to ensure monorepo rules are satisfied:

```bash
poetry run python scripts/guard.py
```

This checks for:
- No stdlib logging imports (must use platform_core.logging)
- No Pydantic or dataclass usage (must use TypedDict)
- No direct environment variable access (must use platform_core.config)
- No subprocess/shell execution
- No type: ignore comments
- No relative imports in tests

Run full checks:

```bash
make check
```

This runs:
1. Guard script validation
2. Ruff linting and formatting
3. Mypy strict type checking
4. Pytest with 100% coverage enforcement

Expected output:
```
Guard scripts: ✅ All 11 rules passed
Ruff: ✅ No issues found
Mypy: ✅ Success: no issues found in 21 source files
Pytest: ✅ 503 passed in ~20 seconds
Coverage: ✅ 1417 statements, 100% coverage
```

## Running the Services

### API Service

Start the FastAPI application:

```bash
# Development mode with auto-reload
poetry run hypercorn turkic_api.api.main:create_app --bind 127.0.0.1:8000 --reload

# Or use the Railway start command
poetry run hypercorn 'turkic_api.api.main:create_app' --bind [::]:8000
```

Verify the API is running:
```bash
curl http://localhost:8000/api/v1/health
# Expected: {"status":"healthy","redis":true,"volume":true,"timestamp":"..."}
```

### Worker Service

Start the RQ worker in a separate terminal:

```bash
poetry run python -m turkic_api.api.worker_entry
```

Expected output:
```json
{
  "level": "INFO",
  "message": "Starting RQ worker",
  "service": "turkic-worker",
  "queue": "turkic-queue",
  "redis_url": "redis://localhost:6379/0"
}
```

The worker will:
- Connect to Redis queue
- Listen for corpus processing jobs
- Process jobs asynchronously
- Publish events to Redis pub/sub

### Testing the Full Pipeline

Create a corpus processing job:

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "source": "oscar",
    "language": "kk",
    "max_sentences": 10,
    "transliterate": true
  }'
```

Expected response:
```json
{
  "job_id": "abc123...",
  "status": "queued",
  "created_at": "2025-11-22T12:00:00Z"
}
```

Check job status:
```bash
curl http://localhost:8000/api/v1/jobs/abc123...
```

## Development Workflow

### Code Quality Checks

Before committing, always run:

```bash
# Full check suite
make check

# Individual checks
make lint        # Guard + Ruff + Mypy
make test        # Pytest with coverage
```

### Running Tests

```bash
# Run all tests in parallel
poetry run pytest -n auto -v

# Run with coverage report
poetry run pytest -n auto -v --cov-branch --cov=src --cov=scripts

# Run specific test file
poetry run pytest tests/test_jobs_api.py -v

# Run specific test case
poetry run pytest tests/test_jobs_api.py::test_process_corpus_impl -v
```

### Type Checking

```bash
# Check all source files
poetry run mypy src tests scripts

# Check specific file
poetry run mypy src/turkic_api/api/jobs.py
```

### Linting and Formatting

```bash
# Auto-fix issues
poetry run ruff check . --fix

# Format code
poetry run ruff format .

# Check without fixing
poetry run ruff check .
```

## Deployment

### Railway Deployment

The project is configured for Railway deployment with two services:

1. **API Service** - FastAPI application
2. **Worker Service** - RQ background worker

#### Railway Configuration (railway.toml)

```toml
[deploy]
startCommand = "sh -c \"exec hypercorn 'turkic_api.api.main:create_app' --bind [::]:${PORT:-8000}\""
healthcheckPath = "/api/v1/health"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
```

#### Environment Variables (Railway)

Set these in the Railway dashboard:

- `REDIS_URL` - Railway Redis addon connection string
- `DATA_DIR` - Persistent volume mount path (e.g., `/data`)
- `ENVIRONMENT` - `production`
- `DATA_BANK_API_URL` - Production data-bank-api URL
- `DATA_BANK_API_KEY` - Production API key

#### Multi-Stage Docker Build

The Dockerfile defines two build targets:

```dockerfile
# API service
FROM runtime-base AS api
CMD ["hypercorn", "turkic_api.api.main:create_app", "--bind", "[::]:8000"]

# Worker service
FROM runtime-base AS worker
CMD ["python", "-m", "turkic_api.api.worker_entry"]
```

Build and run locally:

```bash
# Build API service
docker build --target api -t turkic-api:latest .

# Build worker service
docker build --target worker -t turkic-worker:latest .

# Run API
docker run -p 8000:8000 --env-file .env turkic-api:latest

# Run worker
docker run --env-file .env turkic-worker:latest
```

### Health Checks

The API provides a health endpoint:

```bash
curl http://localhost:8000/api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "redis": true,
  "volume": true,
  "timestamp": "2025-11-22T12:00:00.000000Z"
}
```

Health checks verify:
- Redis connectivity
- Volume mount accessibility
- System timestamp

## Troubleshooting

### Poetry Installation Issues

**Error:** `poetry: command not found`

**Solution:** Add Poetry to PATH:
```bash
# Linux/macOS
export PATH="$HOME/.local/bin:$PATH"

# Windows (PowerShell)
$env:Path += ";C:\Users\$env:USERNAME\AppData\Roaming\Python\Scripts"
```

### Redis Connection Errors

**Error:** `redis.exceptions.ConnectionError: Error 10061 connecting to localhost:6379`

**Solution:** Start Redis:
```bash
# Docker
docker start redis

# Linux
sudo systemctl start redis-server

# macOS
brew services start redis

# Windows
redis-server
```

### Platform Library Import Errors

**Error:** `ModuleNotFoundError: No module named 'platform_core'`

**Solution:** Ensure monorepo structure is correct and reinstall:
```bash
# Check directory structure
ls ../../libs/platform_core
ls ../../libs/platform_workers

# Reinstall dependencies
poetry install --with dev
```

### Guard Script Failures

**Error:** `Guard check failed: Found stdlib logging import`

**Solution:** Use platform_core.logging instead:
```python
# WRONG
import logging
logger = logging.getLogger(__name__)

# CORRECT
from platform_core.logging import get_logger
logger = get_logger(__name__)
```

### Coverage Below 100%

**Error:** `FAILED: coverage: total coverage is 99.5%, expected 100%`

**Solution:** Add tests for uncovered code:
```bash
# Find uncovered lines
poetry run pytest -v --cov-branch --cov=src --cov-report=term-missing

# Output shows: src/turkic_api/api/jobs.py:234
# Add test case for line 234
```

### Mypy Type Errors

**Error:** `error: Incompatible types in assignment`

**Solution:** Use type narrowing:
```python
# WRONG
value: str | int = get_value()
result = value.upper()  # Error: int has no upper()

# CORRECT
value: str | int = get_value()
if isinstance(value, str):
    result = value.upper()  # OK: mypy knows value is str
else:
    result = str(value)
```

### Windows PowerShell Execution Policy

**Error:** `... cannot be loaded because running scripts is disabled`

**Solution:** Allow script execution:
```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Additional Resources

- **Architecture Documentation:** See `DESIGN.md`
- **Contributing Guide:** See `docs/CONTRIBUTING.md`
- **API Documentation:** See `docs/index.md`
- **Transliteration Rules:** See `src/turkic_api/core/rules/README.md`

## Quick Reference

### Essential Commands

```bash
# Setup
poetry install --with dev
make check

# Development
poetry run hypercorn turkic_api.api.main:create_app --bind 127.0.0.1:8000 --reload
poetry run python -m turkic_api.api.worker_entry

# Testing
poetry run pytest -n auto -v
make test

# Type checking
poetry run mypy src tests scripts
make lint

# Deployment
docker build --target api -t turkic-api .
docker build --target worker -t turkic-worker .
```

### API Endpoints

- `GET /api/v1/health` - Health check
- `POST /api/v1/jobs` - Create corpus processing job
- `GET /api/v1/jobs/{job_id}` - Get job status
- `GET /api/v1/jobs/{job_id}/result` - Download processed corpus

### Supported Languages

- Kazakh (kk)
- Kyrgyz (ky)
- Uzbek (uz)
- Turkish (tr)
- Uyghur (ug)
- Finnish (fi)
- Azerbaijani (az)

### Supported Corpus Sources

- oscar - OSCAR corpus (via Hugging Face datasets)
- wikipedia - Wikipedia XML dumps
- culturax - CulturaX corpus (mC4 + OSCAR combined)
