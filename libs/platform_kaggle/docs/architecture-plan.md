# Architecture: platform_kaggle Library

## Overview

The `platform_kaggle` library provides Kaggle competition discovery with codebase capability matching. It wraps the Kaggle API, filters competitions by user interests, and matches them to codebase capabilities detected via `platform_codebase`.

## Dependencies

- `platform-core` - JSON utilities (`require_*` helpers), type validation
- `platform-codebase` - Shared codebase scanning (`scan_libs`, `scan_services`) and types (`CodebaseCapability`, `CodebaseProfile`)
- `kaggle` - Official Kaggle Python API client

## Directory Structure

```
libs/platform_kaggle/
├── pyproject.toml
├── README.md
├── Makefile
├── docs/
│   └── architecture-plan.md
├── scripts/
│   ├── __init__.py
│   └── guard.py
├── src/platform_kaggle/
│   ├── __init__.py           # Public exports
│   ├── py.typed              # PEP 561 marker
│   ├── types.py              # Immutable classes + Protocols
│   ├── client.py             # KaggleClient wrapper
│   ├── capabilities.py       # Codebase capability detection (uses platform_codebase)
│   ├── matcher.py            # Competition-to-capability matching
│   ├── filters.py            # Interest-based filtering
│   ├── _production.py        # Production implementations
│   └── testing.py            # Fakes + hooks container
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_types.py
    ├── test_client.py
    ├── test_capabilities.py
    ├── test_matcher.py
    ├── test_filters.py
    ├── test_testing.py
    ├── test_production.py
    ├── test_init.py
    └── test_guard.py
```

## Kaggle API

The library wraps the official Kaggle Python API (`kaggle.api`):

```python
# Raw Kaggle API returns objects like:
# competition.ref = "titanic"
# competition.title = "Titanic - Machine Learning from Disaster"
# competition.category = "Getting Started"
# competition.reward = "Knowledge"
# competition.deadline = "2030-01-01"
# competition.teamCount = 15000
# competition.tags = [Tag(ref="tabular"), Tag(ref="classification")]
```

## Core Types (types.py)

### Literal Types

```python
CompetitionCategory = Literal[
    "Featured", "Research", "Playground",
    "Getting Started", "Masters", "Kudos"
]
CapabilityStrength = Literal["strong", "moderate", "basic"]
MatchRecommendation = Literal["strong_fit", "good_fit", "stretch", "new_territory"]
```

### Immutable Classes

All domain types use `__slots__` for immutability and memory efficiency:

```python
class Competition:
    """Kaggle competition metadata."""
    __slots__ = (
        "ref", "title", "category", "reward", "deadline",
        "team_count", "tags", "description", "url",
    )

class CompetitionMatch:
    """A competition scored against codebase capabilities."""
    __slots__ = (
        "competition", "match_score", "matched_capabilities",
        "missing_capabilities", "recommendation",
    )

class InterestFilter:
    """User interest filter for competitions."""
    __slots__ = ("include_tags", "exclude_tags", "min_reward", "categories")
```

### Protocols

```python
class KaggleApiProtocol(Protocol):
    """Protocol for low-level Kaggle API."""
    def competitions_list(
        self, *, search: str = "", category: str = ""
    ) -> tuple[KaggleCompetition, ...]: ...

class KaggleClientProtocol(Protocol):
    """Protocol for high-level Kaggle client."""
    def list_competitions(
        self, *, search: str | None = None, category: CompetitionCategory | None = None
    ) -> tuple[Competition, ...]: ...

    def get_competition(self, ref: str) -> Competition | None: ...

class KaggleModuleProtocol(Protocol):
    """Protocol for Kaggle module wrapper."""
    @property
    def api(self) -> KaggleApiProtocol: ...
```

## Key Modules

### 1. client.py - Kaggle API Wrapper

High-level client that uses hooks for testability:

```python
class KaggleClient:
    """Production Kaggle API client."""

    def __init__(self) -> None:
        self._api = hooks.kaggle_api_factory()

    def list_competitions(
        self, *, search: str | None = None, category: CompetitionCategory | None = None
    ) -> tuple[Competition, ...]:
        raw = self._api.competitions_list(
            search=search or "",
            category=category or "",
        )
        return tuple(_convert_competition(c) for c in raw)
```

### 2. _production.py - Production Implementations

```python
def create_kaggle_api() -> KaggleApiProtocol:
    """Create production Kaggle API client."""
    module = hooks.kaggle_module()
    module.api.authenticate()
    return module.api

def default_kaggle_api_factory() -> KaggleApiProtocol:
    """Default factory using real Kaggle module."""
    return create_kaggle_api()

def make_kaggle_client() -> KaggleClientProtocol:
    """Production factory for KaggleClient."""
    return KaggleClient()
```

### 3. capabilities.py - Codebase Capability Detection

Uses `platform_codebase` for scanning:

```python
from platform_codebase import scan_libs, scan_services
from platform_codebase.types import LibInfo, ServiceInfo

def scan_codebase(root: Path) -> CodebaseProfile:
    """Scan codebase and return capability profile."""
    libs = scan_libs(root)
    services = scan_services(root)
    return _build_profile(libs, services)

def _build_profile(
    libs: tuple[LibInfo, ...], services: tuple[ServiceInfo, ...]
) -> CodebaseProfile:
    """Build capability profile from scanned libs/services."""
    # Detects capabilities based on dependencies:
    # - xgboost/lightgbm -> tabular ML
    # - torch/tensorflow -> deep learning
    # - sklearn -> classical ML
    # - optuna -> hyperparameter optimization
```

### 4. matcher.py - Competition Matching

Scores competitions against codebase capabilities:

```python
def match_competition(
    competition: Competition, profile: CodebaseProfile
) -> CompetitionMatch:
    """Score a competition against codebase capabilities."""
    # Matches competition tags to profile ml_backends/task_types
    # Returns match_score (0.0-1.0) and recommendation

def match_competitions(
    competitions: tuple[Competition, ...], profile: CodebaseProfile
) -> tuple[CompetitionMatch, ...]:
    """Score multiple competitions and sort by match score (descending)."""
```

### 5. filters.py - Interest Filtering

```python
def filter_competitions(
    competitions: tuple[Competition, ...], interests: InterestFilter
) -> tuple[Competition, ...]:
    """Filter competitions by user interests."""
    # Filters by:
    # - include_tags: must have at least one matching tag
    # - exclude_tags: must not have any matching tag
    # - min_reward: minimum prize amount (parses "$1,000" format)
    # - categories: must be in allowed categories
```

## Testing Strategy (testing.py)

### Hooks Pattern

```python
class HooksContainer:
    """Container for dependency injection hooks."""
    kaggle_api_factory: KaggleApiFactoryProtocol
    kaggle_client: KaggleClientHook
    kaggle_module: KaggleModuleHook
    profile_scanner: ProfileScannerHook

hooks = HooksContainer()

def _init_production_hooks() -> None:
    """Initialize hooks with production implementations."""
    hooks.kaggle_api_factory = default_kaggle_api_factory
    hooks.kaggle_client = make_kaggle_client
    hooks.kaggle_module = _default_kaggle_module
    hooks.profile_scanner = scan_codebase

def reset_hooks() -> None:
    """Reset hooks to production implementations."""
    _init_production_hooks()
```

### Fake Implementations

```python
class FakeKaggleApi:
    """Fake Kaggle API for testing."""
    def __init__(self, competitions: tuple[KaggleCompetition, ...] = ()) -> None:
        self._competitions = competitions
        self._authenticated = False
        self._list_calls: list[dict[str, str]] = []

    def authenticate(self) -> None:
        self._authenticated = True

    def competitions_list(
        self, *, search: str = "", category: str = ""
    ) -> tuple[KaggleCompetition, ...]: ...

class FakeKaggleClient:
    """Fake Kaggle client for testing."""
    def __init__(self, competitions: tuple[Competition, ...] = ()) -> None:
        self._competitions = competitions
        self._list_calls: list[dict[str, str | CompetitionCategory | None]] = []
        self._get_calls: list[str] = []

class FakeKaggleModule:
    """Fake Kaggle module for testing."""
    def __init__(self, api: KaggleApiProtocol | None = None) -> None:
        self._api = api or FakeKaggleApi()
```

### Factory Functions

```python
def make_fake_competition(...) -> Competition
def make_fake_kaggle_competition(...) -> KaggleCompetition  # Raw API object
def make_fake_capability(...) -> CodebaseCapability
def make_fake_profile(...) -> CodebaseProfile
def make_interest_filter(...) -> InterestFilter
```

## Public API (__init__.py)

```python
# Main functions
def find_competitions(
    *, interests: InterestFilter | None = None,
    match_codebase: bool = True,
    min_match_score: float = 0.0,
    root: Path | None = None,
) -> tuple[CompetitionMatch, ...]: ...

def get_codebase_profile(root: Path | None = None) -> CodebaseProfile: ...

# Re-exported types
from platform_kaggle.types import (
    CapabilityStrength, CodebaseCapability, CodebaseProfile,
    Competition, CompetitionCategory, CompetitionMatch,
    InterestFilter, KaggleApiProtocol, KaggleClientProtocol,
    MatchRecommendation,
)

# Testing utilities
from platform_kaggle.testing import (
    FakeKaggleApi, FakeKaggleClient, FakeKaggleModule, hooks,
    make_fake_capability, make_fake_competition, make_fake_kaggle_competition,
    make_fake_profile, make_interest_filter, reset_hooks,
)

# Encode/decode functions
from platform_kaggle.types import (
    decode_capability, decode_competition, decode_filter, decode_match,
    decode_profile, encode_capability, encode_competition, encode_filter,
    encode_match, encode_profile,
)
```

## Test Coverage

- 100% statement and branch coverage required
- Tests for each encode/decode pair with round-trip validation
- Tests for each filter combination
- Tests for match scoring edge cases
- Tests for capability detection from pyproject.toml via platform_codebase
- Tests for Kaggle API wrapper with fake module
- Tests for guard.py using runpy
- Integration tests with fake client and fake profile
