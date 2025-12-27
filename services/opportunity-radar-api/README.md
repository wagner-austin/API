# opportunity-radar-api

FastAPI service for discovering Kaggle competitions and Devpost hackathons that match your codebase capabilities. Features codebase scanning, capability-based matching, and scored recommendations.

## Features

- **Codebase Scanning**: Detects ML backends, frameworks, and technologies from monorepo
- **Kaggle Integration**: Find competitions matching your capabilities via `platform-kaggle`
- **Devpost Integration**: Find hackathons matching your capabilities via `platform-devpost`
- **Capability Matching**: Score opportunities against codebase profile
- **Recommendations**: Get `strong_fit`, `good_fit`, `stretch`, or `new_territory` ratings
- **Type Safety**: mypy strict mode, zero `Any` types, Protocol-based DI via `_test_hooks.py` pattern
- **100% Test Coverage**: Statements and branches

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+
- Kaggle API credentials (`~/.kaggle/kaggle.json`)

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `KAGGLE_API_TOKEN` | No | Kaggle API token (or use `~/.kaggle/kaggle.json`) |
| `PORT` | No | Server port (default: 8010) |
| `LOG_LEVEL` | No | Logging level (default: INFO) |
| `LOG_FORMAT` | No | Log format: `json` or `text` (default: json) |
| `GITHUB_TOKEN` | No | GitHub personal access token for API scanning |
| `GITHUB_REPO` | No | Repository to scan (format: `owner/repo`) |

### GitHub-based Codebase Scanning

When running in Docker or environments without local filesystem access, configure GitHub scanning:

```bash
# Set GitHub credentials
export GITHUB_TOKEN="ghp_your_personal_access_token"
export GITHUB_REPO="wagner-austin/API"
```

With these set, the `/codebase/*` endpoints scan the repository via GitHub API instead of the local filesystem. This enables:

- **`/codebase/libs`**: Lists libraries by fetching `libs/*/pyproject.toml` from GitHub
- **`/codebase/services`**: Lists services by fetching `services/*/pyproject.toml` from GitHub
- **`/codebase/profile`**: Detects capabilities from GitHub-scanned dependencies

The GitHub token needs read access to repository contents (no special scopes required for public repos).

### Installation

```bash
cd services/opportunity-radar-api
poetry install --with dev
```

### Run the Server

```bash
poetry run hypercorn 'opportunity_radar_api.api.main:create_app()' --bind 0.0.0.0:8000
```

### Example Requests

```bash
# Health check
curl http://localhost:8000/healthz

# Get codebase capability profile
curl http://localhost:8000/codebase/profile

# List all libraries in monorepo
curl http://localhost:8000/codebase/libs

# List all services in monorepo
curl http://localhost:8000/codebase/services

# Find Kaggle competitions matching "tabular" tag with 30%+ match score
curl "http://localhost:8000/kaggle/competitions?tags=tabular&min_score=0.3"

# Find open Devpost hackathons matching AI themes
curl "http://localhost:8000/devpost/hackathons?themes=AI&themes=Machine%20Learning&states=open"
```

---

## API Reference

### Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |

### Codebase

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/codebase/profile` | GET | Get capability profile |
| `/codebase/libs` | GET | List monorepo libraries |
| `/codebase/services` | GET | List monorepo services |

### Kaggle

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/kaggle/competitions` | GET | Find matching competitions |
| `/kaggle/competitions/{ref}` | GET | Get competition by reference |

**Query Parameters for `/kaggle/competitions`:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tags` | list[str] | `[]` | Tags to include (repeatable) |
| `exclude` | list[str] | `[]` | Tags to exclude (repeatable) |
| `min_score` | float | `0.0` | Minimum match score (0.0-1.0) |
| `match_codebase` | bool | `true` | Score against capabilities |

### Devpost

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/devpost/hackathons` | GET | Find matching hackathons |
| `/devpost/hackathons/{id}` | GET | Get hackathon by ID |

**Query Parameters for `/devpost/hackathons`:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `themes` | list[str] | `[]` | Theme names to include (repeatable) |
| `exclude` | list[str] | `[]` | Theme names to exclude (repeatable) |
| `states` | list[str] | `["open"]` | Allowed states (open, upcoming, ended, submissions) |
| `min_score` | float | `0.0` | Minimum match score (0.0-1.0) |
| `match_codebase` | bool | `true` | Score against capabilities |
| `featured_only` | bool | `false` | Only featured hackathons |

---

## Response Formats

### Competition Match

```json
{
  "competition": {
    "ref": "amex-default-prediction",
    "title": "American Express Default Prediction",
    "category": "Featured",
    "reward": "$100,000",
    "deadline": "2024-08-01",
    "team_count": 5000,
    "tags": ["tabular", "classification", "finance"],
    "description": "...",
    "url": "https://www.kaggle.com/competitions/amex-default-prediction"
  },
  "match_score": 0.85,
  "matched_capabilities": ["xgboost", "classification"],
  "missing_capabilities": [],
  "recommendation": "strong_fit"
}
```

### Hackathon Match

```json
{
  "hackathon": {
    "id": 12345,
    "title": "AI for Good Hackathon",
    "url": "https://devpost.com/...",
    "open_state": "open",
    "time_left_to_submission": "5 days",
    "themes": [{"id": 1, "name": "Machine Learning"}],
    "prize_amount": "$10,000"
  },
  "match_score": 0.75,
  "matched_capabilities": ["python", "fastapi"],
  "missing_capabilities": ["mobile"],
  "recommendation": "good_fit"
}
```

### Codebase Profile

```json
{
  "capabilities": [
    {
      "name": "xgboost_tabular",
      "strength": "strong",
      "tags": ["tabular", "classification", "regression", "xgboost"],
      "description": "XGBoost gradient boosting for tabular data"
    },
    {
      "name": "huggingface_transformers",
      "strength": "strong",
      "tags": ["nlp", "transformers", "huggingface", "text-classification", "text-generation", "llm"],
      "description": "Hugging Face Transformers for NLP and LLMs"
    },
    {
      "name": "torchvision_cv",
      "strength": "strong",
      "tags": ["computer-vision", "image", "pytorch", "image-classification"],
      "description": "TorchVision for computer vision tasks"
    }
  ],
  "technologies": ["python"],
  "frameworks": ["fastapi", "flask"],
  "ml_backends": ["xgboost", "lightgbm", "pytorch", "transformers", "torchvision"],
  "data_formats": ["csv", "parquet", "excel"],
  "task_types": ["binary_classification", "regression", "image_classification", "text_generation"]
}
```

---

## Architecture

### Component Overview

```
opportunity_radar_api/
├── __init__.py           # Package exports
├── _test_hooks.py        # DI hooks for testing
└── api/
    ├── main.py           # App factory
    ├── container.py      # DI container
    └── routes/
        ├── health.py     # Health checks
        ├── codebase.py   # Codebase endpoints
        ├── kaggle.py     # Kaggle endpoints
        └── devpost.py    # Devpost endpoints
```

### Container-Based DI

The service uses a container for dependency injection:

```python
from opportunity_radar_api.api.container import ServiceContainer
from opportunity_radar_api.api.main import create_app

# Production (auto-configured)
app = create_app()

# Testing (inject fakes)
container = ServiceContainer(
    monorepo_root=Path("/path/to/monorepo"),
    kaggle_client_factory=lambda: fake_kaggle_client,
    devpost_client_factory=lambda: fake_devpost_client,
    codebase_profile_factory=lambda root: fake_profile,
    libs_scanner=lambda root: (),
    services_scanner=lambda root: (),
)
app = create_app(container=container)
```

### Test Hooks Pattern (`_test_hooks.py`)

Production code uses hooks for testability without mocks:

```python
# In _test_hooks.py
class ContainerFindMonorepoRootProto(Protocol):
    def __call__(self) -> Path: ...

container_find_monorepo_root: ContainerFindMonorepoRootProto | None = None
guard_is_dir: IsDirProto | None = None
guard_find_monorepo_root: FindMonorepoRootProto | None = None
guard_load_orchestrator: LoadOrchestratorProto | None = None

# In production code
def _find_monorepo_root() -> Path:
    if _test_hooks.container_find_monorepo_root is not None:
        return _test_hooks.container_find_monorepo_root()
    return _find_monorepo_root_impl()
```

Tests install fake implementations; production uses default implementations.

### Data Flow

```
                        FastAPI Application
 ┌─────────────────────────────────────────────────────────────┐
 │  ┌──────────┐    ┌──────────┐    ┌──────────────────┐      │
 │  │/codebase │    │ /kaggle  │    │    /devpost      │      │
 │  │  routes  │    │  routes  │    │     routes       │      │
 │  └────┬─────┘    └────┬─────┘    └────────┬─────────┘      │
 │       └───────────────┼───────────────────┘                │
 │                       │                                    │
 │               ┌───────▼───────┐                            │
 │               │  Container    │                            │
 │               └───────┬───────┘                            │
 └───────────────────────┼────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
 ┌───────▼───────┐ ┌─────▼─────┐ ┌───────▼───────┐
 │ platform_     │ │ platform_ │ │ platform_     │
 │ codebase      │ │ kaggle    │ │ devpost       │
 │               │ │           │ │               │
 │ - scan_libs   │ │ - client  │ │ - client      │
 │ - scan_svcs   │ │ - match   │ │ - match       │
 │ - GitHub scan │ │ - filter  │ │ - filter      │
 │ - profile     │ │ - profile │ │               │
 └───────┬───────┘ └─────┬─────┘ └───────┬───────┘
         │               │               │
 ┌───────▼───────┐ ┌─────▼─────┐   ┌─────▼─────┐
 │   GitHub      │ │  Kaggle   │   │  Devpost  │
 │   API         │ │   API     │   │   API     │
 │ (if enabled)  │ │           │   │           │
 └───────────────┘ └───────────┘   └───────────┘
```

### Codebase Scanning Modes

The container automatically selects the scanning mode based on configuration:

| Mode | Condition | How it works |
|------|-----------|--------------|
| **Local** | `GITHUB_TOKEN` not set | Scans local `libs/` and `services/` directories |
| **GitHub** | `GITHUB_TOKEN` + `GITHUB_REPO` set | Fetches pyproject.toml via GitHub API |

In GitHub mode, `platform_codebase` handles the API calls and `platform_kaggle.build_profile()` detects capabilities from the scanned data.

### Design Documentation

See [docs/architecture-plan.md](./docs/architecture-plan.md) for the full design document.

---

## Development

### Commands

```bash
make lint    # Run guards + ruff + mypy
make test    # Run pytest with coverage
make check   # Run lint + test
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
poetry run pytest tests/test_kaggle.py -v

# Run with coverage report
poetry run pytest --cov-report=html
```

---

## Project Structure

```
opportunity-radar-api/
├── src/opportunity_radar_api/
│   ├── __init__.py             # Package exports
│   ├── _test_hooks.py          # DI hooks for testing
│   ├── py.typed                # PEP 561 marker
│   └── api/
│       ├── main.py             # App factory
│       ├── container.py        # DI container
│       └── routes/
│           ├── health.py       # Health checks
│           ├── codebase.py     # Codebase endpoints
│           ├── kaggle.py       # Kaggle endpoints
│           └── devpost.py      # Devpost endpoints
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_container.py
│   ├── test_main.py
│   ├── test_health.py
│   ├── test_codebase.py
│   ├── test_kaggle.py
│   ├── test_devpost.py
│   └── test_guard.py
├── scripts/
│   └── guard.py                # Code quality enforcement
├── docs/
│   └── architecture-plan.md    # Design documentation
├── pyproject.toml
├── Makefile
└── README.md
```

---

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `hypercorn` | ASGI server |
| `httpx` | HTTP client |
| `platform-core` | JSON utilities, error handling |
| `platform-codebase` | Codebase scanning |
| `platform-kaggle` | Kaggle API wrapper |
| `platform-devpost` | Devpost API wrapper |
| `platform-calendar` | Calendar integration |

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
- **Errors**: Consistent format via `platform_core.fastapi`

---

## Future Enhancements

- **Calendar Integration**: Sync deadlines to Google Calendar via `platform-calendar`
- **Notifications**: Send alerts for new matching opportunities
- **Caching**: Cache competition/hackathon data to reduce API calls
- **Webhooks**: Push notifications when new opportunities match

---

## License

Apache-2.0
