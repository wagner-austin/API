# platform-devpost

Devpost hackathon discovery with codebase capability matching.

This library provides tools for discovering Devpost hackathons that match
your codebase capabilities and personal interests.

## Installation

```bash
poetry add platform-devpost
```

## Dependencies

- `platform-core` - JSON utilities, HTTP client protocols
- `platform-codebase` - Shared codebase scanning and capability detection

## Quick Start

```python
from platform_devpost import find_hackathons, make_interest_filter

# Find hackathons that fit the codebase
matches = find_hackathons(
    interests=make_interest_filter(
        include_themes=("Machine Learning", "API", "Web"),
        exclude_themes=("Gaming",),
        states=("open", "upcoming"),
        featured_only=False,
    ),
    match_codebase=True,
    min_match_score=0.3,
)

for match in matches:
    title = match.hackathon.title
    score = f"{match.match_score:.0%}"
    recommendation = match.recommendation
    time_left = match.hackathon.time_left_to_submission
    # Use title, score, recommendation, time_left as needed
```

## Core Concepts

### Hackathons

The library fetches hackathon metadata from the Devpost API:

```python
from platform_devpost import DevpostClient

client = DevpostClient()
hackathons = client.list_hackathons(
    search="machine learning",
    state="open",
)
```

### Interest Filtering

Filter hackathons by themes, states, and featured status:

```python
from platform_devpost import make_interest_filter, filter_hackathons

interests = make_interest_filter(
    include_themes=("AI", "Machine Learning"),  # Must have at least one
    exclude_themes=("Gaming",),                  # Must not have any
    states=("open", "upcoming"),                 # None = all states
    featured_only=True,                          # Only featured hackathons
)

filtered = filter_hackathons(hackathons, interests)
```

### Codebase Capability Matching

The library scans your codebase to detect capabilities using `platform_codebase`:

```python
from platform_devpost import get_codebase_profile

profile = get_codebase_profile()
# Returns capabilities like:
# - technologies: ("python", "javascript", "rust")
# - frameworks: ("flask", "react", "pytorch")
# - capabilities: detected from pyproject.toml dependencies
```

Hackathons are matched against these capabilities:

```python
from platform_devpost import match_hackathon, match_hackathons

# Match single hackathon
match = match_hackathon(hackathon, profile)
# match.match_score: 0.0 - 1.0
# match.recommendation: "strong_fit" | "good_fit" | "stretch" | "new_territory"
# match.matched_capabilities: ("python", "machine_learning")
# match.missing_capabilities: ("mobile_development",)

# Match multiple hackathons at once
matches = match_hackathons(hackathons, profile)
```

### Scanning Codebase Directly

For advanced use cases:

```python
from pathlib import Path
from platform_devpost import scan_codebase

# Scan local filesystem for capabilities
profile = scan_codebase(Path("/path/to/monorepo"))
```

## Testing

The library provides fakes and hooks for testing:

```python
from platform_devpost import (
    FakeDevpostClient,
    hooks,
    make_fake_hackathon,
    reset_hooks,
)

# Create fake client with test data
fake_client = FakeDevpostClient(
    hackathons=(
        make_fake_hackathon(id=1, title="Test Hackathon"),
    )
)

# Install via hooks
hooks.devpost_client = lambda: fake_client

try:
    # Your test code here
    ...
finally:
    reset_hooks()
```

### Available Fakes

- `FakeDevpostApi` - Low-level API fake with `fetch_hackathons`
- `FakeDevpostClient` - High-level client fake with `list_hackathons` and `get_hackathon`

### Factory Functions

- `make_fake_hackathon()` - Create test Hackathon instances
- `make_fake_theme()` - Create test Theme instances
- `make_fake_displayed_location()` - Create test DisplayedLocation instances
- `make_fake_capability()` - Create test CodebaseCapability instances
- `make_fake_profile()` - Create test CodebaseProfile instances
- `make_interest_filter()` - Create InterestFilter instances

## Types

All types are exported from the package:

- `Hackathon` - Hackathon metadata (immutable `__slots__` class)
- `Theme` - Hackathon theme
- `DisplayedLocation` - Location information
- `HackathonMatch` - Match result with score
- `CodebaseProfile` - Detected capabilities
- `CodebaseCapability` - Individual capability
- `InterestFilter` - Filter configuration
- `HackathonState` - Literal type: `"open" | "upcoming" | "ended" | "submissions"`
- `CapabilityStrength` - Literal type: `"strong" | "moderate" | "basic"`
- `MatchRecommendation` - Literal type: `"strong_fit" | "good_fit" | "stretch" | "new_territory"`

## JSON Serialization

Encode/decode functions for all types:

```python
from platform_devpost import encode_hackathon, decode_hackathon

# Serialize to JSON-compatible dict
data = encode_hackathon(hackathon)

# Deserialize with validation
hackathon = decode_hackathon(data)
```

## Hooks System

The library uses a hooks pattern for dependency injection:

| Hook | Type | Purpose |
|------|------|---------|
| `hooks.devpost_api_factory` | `() -> DevpostApiProtocol` | Creates low-level API |
| `hooks.devpost_client` | `() -> DevpostClientProtocol` | Creates high-level client |
| `hooks.profile_scanner` | `(Path) -> CodebaseProfile` | Scans codebase for capabilities |

## Service Integration

### Adding to Your Project

1. Add the dependency to your `pyproject.toml`:

```toml
[tool.poetry.dependencies]
platform-devpost = { path = "../platform_devpost", develop = true }
```

2. Import and use:

```python
from platform_devpost import find_hackathons, make_interest_filter
```

### Basic Usage in a Script

```python
from platform_devpost import (
    find_hackathons,
    make_interest_filter,
    encode_match,
)

def main() -> None:
    # Define what you're interested in
    interests = make_interest_filter(
        include_themes=("Machine Learning", "AI", "API"),
        exclude_themes=("Gaming",),
        states=("open", "upcoming"),
        featured_only=False,
    )

    # Find matching hackathons
    matches = find_hackathons(
        interests=interests,
        match_codebase=True,  # Score against your codebase capabilities
        min_match_score=0.3,  # Only show 30%+ matches
    )

    # Process results
    for match in matches:
        print(f"{match.hackathon.title}")
        print(f"  Score: {match.match_score:.0%}")
        print(f"  Fit: {match.recommendation}")
        print(f"  Time left: {match.hackathon.time_left_to_submission}")
        print(f"  Prize: {match.hackathon.prize_amount}")
        print()

if __name__ == "__main__":
    main()
```

### Using in a FastAPI Service

```python
from fastapi import FastAPI, Query, HTTPException
from platform_devpost import (
    find_hackathons,
    make_interest_filter,
    encode_match,
    DevpostClient,
    encode_hackathon,
    HackathonState,
)

app = FastAPI()

@app.get("/hackathons")
def list_hackathons(
    themes: list[str] = Query(default=[]),
    exclude: list[str] = Query(default=[]),
    states: list[HackathonState] = Query(default=["open"]),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0),
    featured_only: bool = False,
):
    """Find Devpost hackathons matching criteria."""
    interests = make_interest_filter(
        include_themes=tuple(themes),
        exclude_themes=tuple(exclude),
        states=tuple(states) if states else None,
        featured_only=featured_only,
    )

    matches = find_hackathons(
        interests=interests,
        match_codebase=True,
        min_match_score=min_score,
    )

    return [encode_match(m) for m in matches]


@app.get("/hackathons/{hackathon_id}")
def get_hackathon(hackathon_id: int):
    """Get a specific hackathon by ID."""
    client = DevpostClient()
    hackathon = client.get_hackathon(hackathon_id)
    if hackathon is None:
        raise HTTPException(404, f"Hackathon {hackathon_id} not found")
    return encode_hackathon(hackathon)


@app.get("/codebase/profile")
def get_profile():
    """Get the codebase capability profile."""
    from platform_devpost import get_codebase_profile, encode_profile

    profile = get_codebase_profile()
    return encode_profile(profile)
```

### Testing Your Integration

```python
import pytest
from platform_devpost import (
    hooks,
    reset_hooks,
    FakeDevpostClient,
    make_fake_hackathon,
    make_fake_profile,
)

@pytest.fixture(autouse=True)
def reset_devpost_hooks():
    """Reset hooks after each test."""
    yield
    reset_hooks()

def test_my_hackathon_endpoint():
    # Set up fake data
    fake_hackathon = make_fake_hackathon(
        id=123,
        title="AI Challenge",
        open_state="open",
    )
    fake_client = FakeDevpostClient(hackathons=(fake_hackathon,))
    hooks.devpost_client = lambda: fake_client

    # Optionally mock the profile scanner
    hooks.profile_scanner = lambda root: make_fake_profile(
        technologies=("python",),
        frameworks=("fastapi",),
    )

    # Test your code
    from myapp import list_hackathons
    result = list_hackathons(states=["open"])

    assert len(result) == 1
    assert result[0]["hackathon"]["id"] == 123
```

## Architecture

See [docs/architecture-plan.md](docs/architecture-plan.md) for the full design document.

## Development

```bash
# Run all checks (guard, lint, format, type check, tests)
make check

# Run tests only
make test

# Run linting only
make lint
```
