# Architecture: platform_devpost Library

## Overview

The `platform_devpost` library provides Devpost hackathon discovery with codebase capability matching. It fetches hackathon listings from Devpost's JSON API, filters by user interests, and matches hackathons to codebase capabilities detected via `platform_codebase`.

## Dependencies

- `platform-core` - JSON utilities (`load_json_str`, `narrow_json_to_dict`, `require_*` helpers), HTTP client protocols
- `platform-codebase` - Shared codebase scanning (`scan_libs`, `scan_services`) and types (`CodebaseCapability`, `CodebaseProfile`)
- `httpx` - HTTP client for API requests

## Directory Structure

```
libs/platform_devpost/
├── pyproject.toml
├── README.md
├── Makefile
├── docs/
│   └── architecture-plan.md
├── scripts/
│   ├── __init__.py
│   └── guard.py
├── src/platform_devpost/
│   ├── __init__.py           # Public exports
│   ├── py.typed              # PEP 561 marker
│   ├── types.py              # Immutable classes + Protocols
│   ├── client.py             # DevpostClient wrapper
│   ├── capabilities.py       # Codebase capability detection (uses platform_codebase)
│   ├── matcher.py            # Hackathon-to-capability matching
│   ├── filters.py            # Interest-based filtering
│   ├── _production.py        # Production implementations with HTTP client hook
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

## Devpost API

### Endpoints

- `GET https://devpost.com/api/hackathons` - List hackathons with pagination

### API Response Structure

```json
{
  "hackathons": [
    {
      "id": 12345,
      "title": "Hackathon Title",
      "displayed_location": {"icon": "globe", "location": "Online"},
      "open_state": "open",
      "thumbnail_url": "https://...",
      "url": "https://hackathon.devpost.com/",
      "time_left_to_submission": "5 days left",
      "submission_period_dates": "Jan 01 - Feb 01, 2025",
      "themes": [{"id": 1, "name": "Machine Learning"}],
      "prize_amount": "$10,000",
      "registrations_count": 500,
      "featured": true,
      "organization_name": "Tech Corp",
      "winners_announced": false,
      "invite_only": false
    }
  ],
  "meta": {
    "total_count": 150,
    "per_page": 10
  }
}
```

## Core Types (types.py)

### Literal Types

```python
HackathonState = Literal["open", "upcoming", "ended", "submissions"]
CapabilityStrength = Literal["strong", "moderate", "basic"]
MatchRecommendation = Literal["strong_fit", "good_fit", "stretch", "new_territory"]
```

### Immutable Classes

All domain types use `__slots__` for immutability and memory efficiency:

```python
class Theme:
    """A hackathon theme/category."""

    __slots__ = ("id", "name")


class DisplayedLocation:
    """Location information for a hackathon."""

    __slots__ = ("icon", "location")


class Hackathon:
    """Devpost hackathon metadata."""

    __slots__ = (
        "id",
        "title",
        "url",
        "thumbnail_url",
        "organization_name",
        "displayed_location",
        "open_state",
        "time_left_to_submission",
        "submission_period_dates",
        "themes",
        "prize_amount",
        "registrations_count",
        "featured",
        "winners_announced",
        "invite_only",
    )


class CodebaseCapability:
    """A capability the codebase has."""

    __slots__ = ("name", "strength", "tags", "description")


class CodebaseProfile:
    """Full profile of codebase capabilities."""

    __slots__ = ("capabilities", "technologies", "frameworks")


class HackathonMatch:
    """A hackathon scored against codebase capabilities."""

    __slots__ = (
        "hackathon",
        "match_score",
        "matched_capabilities",
        "missing_capabilities",
        "recommendation",
    )


class InterestFilter:
    """User interest filter for hackathons."""

    __slots__ = ("include_themes", "exclude_themes", "states", "featured_only")


class HackathonListMeta:
    """Metadata for hackathon list response."""

    __slots__ = ("total_count", "per_page")


class HackathonListResponse:
    """Response from hackathon list API."""

    __slots__ = ("hackathons", "meta")
```

### Protocols

```python
class DevpostApiProtocol(Protocol):
    """Protocol for low-level Devpost API client."""

    def fetch_hackathons(
        self, *, page: int = 1, search: str | None = None
    ) -> HackathonListResponse: ...


class DevpostClientProtocol(Protocol):
    """Protocol for high-level Devpost client."""

    def list_hackathons(
        self, *, search: str | None = None, state: HackathonState | None = None
    ) -> tuple[Hackathon, ...]: ...

    def get_hackathon(self, hackathon_id: int) -> Hackathon | None: ...
```

## Key Modules

### 1. client.py - Devpost API Wrapper

High-level client that uses hooks for testability:

```python
class DevpostClient:
    """Production Devpost API client."""

    def __init__(self) -> None:
        self._api = hooks.devpost_api_factory()

    def list_hackathons(
        self, *, search: str | None = None, state: HackathonState | None = None
    ) -> tuple[Hackathon, ...]:
        response = self._api.fetch_hackathons(search=search)
        result = response.hackathons
        if state is not None:
            result = tuple(h for h in result if h.open_state == state)
        return result
```

### 2. _production.py - Production Implementations

Contains HTTP-based API implementation with testable hook:

```python
from collections.abc import Callable
from platform_core.http_client import HttpxClient, SyncTransport, build_client

# HTTP Client Builder Hook for testability
HttpClientBuilder = Callable[[float, SyncTransport | None], HttpxClient]
_http_client_builder: HttpClientBuilder = build_client


def _set_http_client_builder(builder: HttpClientBuilder) -> None:
    """Set HTTP client builder for testing."""
    global _http_client_builder
    _http_client_builder = builder


def _reset_http_client_builder() -> None:
    """Reset HTTP client builder to production implementation."""
    global _http_client_builder
    _http_client_builder = build_client


class _HttpDevpostApi:
    """Production HTTP-based Devpost API."""

    def __init__(self) -> None:
        self._client = _http_client_builder(DEFAULT_TIMEOUT_SECONDS, None)

    def fetch_hackathons(
        self, *, page: int = 1, search: str | None = None
    ) -> HackathonListResponse:
        params: dict[str, str | int] = {"page": page}
        if search is not None:
            params["search"] = search
        response = self._client.get(DEVPOST_API_URL, params=params)
        response.raise_for_status()
        data = narrow_json_to_dict(load_json_str(response.text))
        return decode_list_response(data)
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


def _build_profile(libs: tuple[LibInfo, ...], services: tuple[ServiceInfo, ...]) -> CodebaseProfile:
    """Build capability profile from scanned libs/services."""
    # Detects capabilities based on dependencies:
    # - xgboost/lightgbm -> tabular ML
    # - torch/tensorflow -> deep learning
    # - polars/pandas -> data processing
    # - openai -> AI integration
    # etc.
```

### 4. matcher.py - Hackathon Matching

Scores hackathons against codebase capabilities:

```python
def match_hackathon(hackathon: Hackathon, profile: CodebaseProfile) -> HackathonMatch:
    """Score a hackathon against codebase capabilities."""
    # Matches hackathon themes to profile technologies/frameworks
    # Returns match_score (0.0-1.0) and recommendation


def match_hackathons(
    hackathons: tuple[Hackathon, ...], profile: CodebaseProfile
) -> tuple[HackathonMatch, ...]:
    """Score multiple hackathons and sort by match score (descending)."""
```

### 5. filters.py - Interest Filtering

```python
def filter_hackathons(
    hackathons: tuple[Hackathon, ...], interests: InterestFilter
) -> tuple[Hackathon, ...]:
    """Filter hackathons by user interests."""
    # Filters by:
    # - include_themes: must have at least one matching theme
    # - exclude_themes: must not have any matching theme
    # - states: must be in allowed states
    # - featured_only: only featured hackathons
```

## Testing Strategy (testing.py)

### Hooks Pattern

```python
class HooksContainer:
    """Container for dependency injection hooks."""

    devpost_api_factory: DevpostApiFactoryProtocol
    devpost_client: DevpostClientHook
    profile_scanner: ProfileScannerHook


hooks = HooksContainer()


def _init_production_hooks() -> None:
    """Initialize hooks with production implementations."""
    hooks.devpost_api_factory = create_devpost_api
    hooks.devpost_client = make_devpost_client
    hooks.profile_scanner = scan_codebase


def reset_hooks() -> None:
    """Reset hooks to production implementations."""
    _init_production_hooks()
```

### Fake Implementations

```python
class FakeDevpostApi:
    """Fake Devpost API for testing."""

    def __init__(self, hackathons: tuple[Hackathon, ...] = ()) -> None:
        self._hackathons = hackathons
        self._fetch_calls: list[dict[str, int | str | None]] = []


class FakeDevpostClient:
    """Fake Devpost client for testing."""

    def __init__(self, hackathons: tuple[Hackathon, ...] = ()) -> None:
        self._hackathons = hackathons
        self._list_calls: list[dict[str, str | HackathonState | None]] = []
        self._get_calls: list[int] = []
```

### Factory Functions

```python
def make_fake_hackathon(*, id: int = 1, title: str = "Test Hackathon", ...) -> Hackathon
def make_fake_theme(*, id: int = 1, name: str = "Test Theme") -> Theme
def make_fake_displayed_location(*, icon: str = "globe", location: str = "Online") -> DisplayedLocation
def make_fake_capability(*, name: str = "test_capability", ...) -> CodebaseCapability
def make_fake_profile(*, capabilities: tuple[...] = (), ...) -> CodebaseProfile
def make_interest_filter(*, include_themes: tuple[str, ...] = (), ...) -> InterestFilter
```

## Testing HTTP Layer

Tests use `httpx.BaseTransport` to mock HTTP responses:

```python
class FakeHttpTransport(httpx.BaseTransport):
    """Fake HTTP transport for testing."""

    def __init__(self, response_text: str, status_code: int = 200) -> None:
        self._response_text = response_text
        self._status_code = status_code

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=self._status_code,
            content=self._response_text.encode(),
            request=request,
        )


# In tests:
def test_fetch_hackathons() -> None:
    fake_transport = FakeHttpTransport(response_json)

    def test_builder(timeout: float, transport: SyncTransport | None) -> HttpxClient:
        return build_client(timeout, transport=fake_transport)

    _set_http_client_builder(test_builder)
    try:
        api = _HttpDevpostApi()
        result = api.fetch_hackathons()
        # assertions...
    finally:
        _reset_http_client_builder()
```

## Public API (__init__.py)

```python
# Main functions
def find_hackathons(
    *,
    interests: InterestFilter | None = None,
    match_codebase: bool = True,
    min_match_score: float = 0.0,
    root: Path | None = None,
) -> tuple[HackathonMatch, ...]: ...


def get_codebase_profile(root: Path | None = None) -> CodebaseProfile: ...


# Re-exported types
from platform_devpost.types import (
    CapabilityStrength,
    CodebaseCapability,
    CodebaseProfile,
    DevpostApiProtocol,
    DevpostClientProtocol,
    DisplayedLocation,
    Hackathon,
    HackathonMatch,
    HackathonState,
    InterestFilter,
    MatchRecommendation,
    Theme,
)

# Testing utilities
from platform_devpost.testing import (
    FakeDevpostApi,
    FakeDevpostClient,
    hooks,
    make_fake_capability,
    make_fake_hackathon,
    make_fake_profile,
    make_fake_theme,
    make_interest_filter,
    reset_hooks,
)

# Encode/decode functions
from platform_devpost.types import (
    decode_capability,
    decode_filter,
    decode_hackathon,
    decode_match,
    decode_profile,
    decode_theme,
    encode_capability,
    encode_filter,
    encode_hackathon,
    encode_match,
    encode_profile,
    encode_theme,
)
```

## Test Coverage

- 100% statement and branch coverage required
- Tests for each encode/decode pair with round-trip validation
- Tests for each filter combination
- Tests for match scoring edge cases
- Tests for capability detection from pyproject.toml
- Tests for HTTP client with fake transport
- Tests for guard.py using runpy
- Integration tests with fake client and fake profile
