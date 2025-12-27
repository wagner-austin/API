# Architecture: opportunity-radar-api

## Overview

The `opportunity-radar-api` is a FastAPI service that integrates multiple platform libraries to discover Kaggle competitions and Devpost hackathons that match the codebase capabilities. It provides a unified API for opportunity discovery with capability-based scoring.

## Dependencies

- `platform-core` - JSON utilities, exception handlers, error handling
- `platform-codebase` - Codebase scanning and capability detection
- `platform-kaggle` - Kaggle competition discovery and matching
- `platform-devpost` - Devpost hackathon discovery and matching
- `platform-calendar` - Calendar integration for deadline tracking (future)

## Directory Structure

```
services/opportunity-radar-api/
├── pyproject.toml
├── README.md
├── Makefile
├── Dockerfile
├── docs/
│   ├── architecture-plan.md
│   └── api.md
├── scripts/
│   ├── __init__.py
│   └── guard.py
├── src/opportunity_radar_api/
│   ├── __init__.py           # Package exports
│   ├── _test_hooks.py        # DI hooks for testing
│   ├── py.typed              # PEP 561 marker
│   └── api/
│       ├── __init__.py
│       ├── main.py           # FastAPI app factory
│       ├── container.py      # DI container
│       └── routes/
│           ├── __init__.py
│           ├── health.py     # Health checks
│           ├── codebase.py   # Codebase profile endpoints
│           ├── kaggle.py     # Kaggle competition endpoints
│           └── devpost.py    # Devpost hackathon endpoints
└── tests/
    ├── __init__.py
    ├── conftest.py           # Shared fixtures
    ├── test_container.py
    ├── test_main.py
    ├── test_health.py
    ├── test_codebase.py
    ├── test_kaggle.py
    ├── test_devpost.py
    └── test_guard.py
```

## Core Components

### ServiceContainer

Dependency injection container for API routes:

```python
class ServiceContainer:
    """Container for service dependencies."""
    __slots__ = (
        "monorepo_root",
        "_kaggle_client_factory",
        "_devpost_client_factory",
        "_codebase_profile_factory",
        "_libs_scanner",
        "_services_scanner",
    )

    def __init__(
        self,
        *,
        monorepo_root: Path,
        kaggle_client_factory: Callable[[], KaggleClientProtocol],
        devpost_client_factory: Callable[[], DevpostClientProtocol],
        codebase_profile_factory: Callable[[Path], CodebaseProfile],
        libs_scanner: Callable[[Path], tuple[LibInfo, ...]],
        services_scanner: Callable[[Path], tuple[ServiceInfo, ...]],
    ) -> None: ...

    def get_kaggle_client(self) -> KaggleClientProtocol: ...
    def get_devpost_client(self) -> DevpostClientProtocol: ...
    def get_codebase_profile(self) -> CodebaseProfile: ...
    def scan_libs(self) -> tuple[LibInfo, ...]: ...
    def scan_services(self) -> tuple[ServiceInfo, ...]: ...
```

### App Factory

```python
def create_app(
    container: ServiceContainer | None = None,
    monorepo_root: Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
```

### Test Hooks Pattern (`_test_hooks.py`)

Following the model-trainer pattern, production code uses hooks for testability:

```python
# Protocol definitions
class FindMonorepoRootProto(Protocol):
    def __call__(self, start: Path) -> Path: ...

class RunForProjectProto(Protocol):
    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int: ...

class LoadOrchestratorProto(Protocol):
    def __call__(self, monorepo_root: Path) -> RunForProjectProto: ...

class IsDirProto(Protocol):
    def __call__(self, path: Path) -> bool: ...

class ContainerFindMonorepoRootProto(Protocol):
    def __call__(self) -> Path: ...

# Hooks - None means use default behavior (production implementation)
guard_find_monorepo_root: FindMonorepoRootProto | None = None
guard_load_orchestrator: LoadOrchestratorProto | None = None
guard_is_dir: IsDirProto | None = None
container_find_monorepo_root: ContainerFindMonorepoRootProto | None = None
```

Production code checks hooks:

```python
def _find_monorepo_root() -> Path:
    """Find monorepo root, using hook if set."""
    from opportunity_radar_api import _test_hooks

    if _test_hooks.container_find_monorepo_root is not None:
        return _test_hooks.container_find_monorepo_root()
    return _find_monorepo_root_impl()
```

Tests install fake implementations; production uses default implementations.

## API Endpoints

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
- `tags` - Tags to include (repeatable)
- `exclude` - Tags to exclude (repeatable)
- `min_score` - Minimum match score (0.0-1.0)
- `match_codebase` - Score against capabilities (boolean)

### Devpost

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/devpost/hackathons` | GET | Find matching hackathons |
| `/devpost/hackathons/{id}` | GET | Get hackathon by ID |

**Query Parameters for `/devpost/hackathons`:**
- `themes` - Theme names to include (repeatable)
- `exclude` - Theme names to exclude (repeatable)
- `states` - Allowed states (open, upcoming, ended, submissions)
- `min_score` - Minimum match score (0.0-1.0)
- `match_codebase` - Score against capabilities (boolean)
- `featured_only` - Only featured hackathons (boolean)

## Data Flow

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
 │ - profile     │ │ - filter  │ │ - filter      │
 └───────────────┘ └─────┬─────┘ └───────┬───────┘
                         │               │
                   ┌─────▼─────┐   ┌─────▼─────┐
                   │  Kaggle   │   │  Devpost  │
                   │   API     │   │   API     │
                   └───────────┘   └───────────┘
```

## Route Implementation Pattern

Routes use `add_api_route()` instead of decorators to avoid `Any` types from FastAPI:

```python
def build_router(container: ServiceContainer) -> APIRouter:
    """Build router with injected container."""
    router = APIRouter(prefix="/kaggle", tags=["kaggle"])

    def _list_competitions(
        tags: Annotated[list[str], Query()] = _TAGS_QUERY,
        exclude: Annotated[list[str], Query()] = _EXCLUDE_QUERY,
        min_score: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
        match_codebase: bool = True,
    ) -> list[JSONObject]:
        """Find Kaggle competitions matching criteria."""
        client = container.get_kaggle_client()
        competitions = client.list_competitions()
        # ... filtering and matching logic
        return [encode_match(m) for m in matches]

    router.add_api_route(
        "/competitions", _list_competitions, methods=["GET"], response_model=None
    )
    return router
```

## Testing Strategy

### Container-Based DI

Tests inject fake implementations via the container:

```python
def _make_fake_container(
    fake_kaggle_client: FakeKaggleClient,
    fake_devpost_client: FakeDevpostClient,
    fake_profile: CodebaseProfile,
    fake_lib_info: LibInfo,
    fake_service_info: ServiceInfo,
    tmp_path: Path,
) -> ServiceContainer:
    return ServiceContainer(
        monorepo_root=tmp_path,
        kaggle_client_factory=lambda: fake_kaggle_client,
        devpost_client_factory=lambda: fake_devpost_client,
        codebase_profile_factory=lambda root: fake_profile,
        libs_scanner=lambda root: (fake_lib_info,),
        services_scanner=lambda root: (fake_service_info,),
    )

fake_container = pytest.fixture(_make_fake_container)
```

### Using Platform Lib Fakes

Tests use fakes from platform libraries:

```python
from platform_kaggle.testing import FakeKaggleClient, make_fake_competition
from platform_devpost.testing import FakeDevpostClient, make_fake_hackathon
from platform_codebase.testing import make_fake_profile, make_fake_lib_info
```

### Fixture Pattern

Following model-trainer, fixtures use function call pattern instead of decorators:

```python
def _make_fake_competition() -> Competition:
    """Create a fake Kaggle competition."""
    return make_fake_competition(
        ref="test-comp",
        title="Test Competition",
        category="Playground",
        tags=("tabular", "classification"),
    )

fake_competition = pytest.fixture(_make_fake_competition)
```

### Hook Testing

Tests set hooks for monorepo root detection and guard functions:

```python
def test_find_monorepo_root_uses_impl_when_no_hook() -> None:
    """Test _find_monorepo_root uses impl when no hook set."""
    original = _test_hooks.container_find_monorepo_root
    _test_hooks.container_find_monorepo_root = None

    try:
        result = _find_monorepo_root()
        assert result == expected_root
    finally:
        _test_hooks.container_find_monorepo_root = original
```

### Guard Script Coverage

The `if __name__ == "__main__"` block is covered using `runpy.run_path()`:

```python
def test_guard_main_via_runpy() -> None:
    """Test guard.py if __name__ == '__main__' block via runpy."""
    # Set up hooks
    _test_hooks.guard_is_dir = fake_is_dir
    _test_hooks.guard_find_monorepo_root = fake_find_root
    _test_hooks.guard_load_orchestrator = fake_load

    guard_path = Path(__file__).parent.parent / "scripts" / "guard.py"

    try:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(guard_path), run_name="__main__")
        assert exc_info.value.code == 0
    finally:
        # Reset hooks
        ...
```

### Test Coverage

- 100% statement and branch coverage required
- Tests for all endpoints
- Tests for container methods
- Tests for guard.py including `__main__` block
- Tests for hook usage (both with and without hooks set)

## Quality Standards

- **Type Safety**: mypy strict mode, no `Any`, no `cast`, no `type: ignore`
- **Coverage**: 100% statements and branches
- **Guard Rules**: Enforced via `scripts/guard.py`
- **Docstrings**: Google-style with Args/Returns/Raises
- **No Mocks**: Use fakes and hooks pattern instead

## Response Formats

### Competition Match Response

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

### Hackathon Match Response

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

## Deployment

### Docker

The service uses a multi-stage Dockerfile:

1. **Builder stage**: Builds the wheel from source
2. **Runtime stage**: Installs the wheel and platform libs

```bash
# Build from monorepo root
docker build -f services/opportunity-radar-api/Dockerfile -t opportunity-radar-api .

# Run
docker run -p 8000:8000 \
  -e KAGGLE_API_TOKEN="$KAGGLE_API_TOKEN" \
  opportunity-radar-api
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `PORT` | Server port (default: 8000) |
| `KAGGLE_API_TOKEN` | Kaggle API token (required) |

### Kaggle Credentials

The Kaggle API requires the `KAGGLE_API_TOKEN` environment variable to be set.

## Future Enhancements

1. **Calendar Integration**: Sync deadlines to Google Calendar via `platform-calendar`
2. **Notifications**: Send alerts for new matching opportunities
3. **Caching**: Cache competition/hackathon data to reduce API calls
4. **Webhooks**: Push notifications when new opportunities match
