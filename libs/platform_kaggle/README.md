# platform-kaggle

Kaggle competition discovery with codebase capability matching.

This library provides tools for discovering Kaggle competitions that match
your codebase capabilities and personal interests.

## Installation

```bash
poetry add platform-kaggle
```

Requires valid Kaggle API credentials in `~/.kaggle/kaggle.json` or via
environment variables (`KAGGLE_USERNAME`, `KAGGLE_KEY`).

## Quick Start

```python
from platform_kaggle import find_competitions, make_interest_filter

# Find ML/linguistics competitions that fit the codebase
matches = find_competitions(
    interests=make_interest_filter(
        include_tags=("tabular", "nlp", "classification"),
        exclude_tags=("computer-vision", "image"),
    ),
    match_codebase=True,
    min_match_score=0.3,
)

for match in matches:
    title = match.competition.title
    score = f"{match.match_score:.0%}"
    recommendation = match.recommendation
    deadline = match.competition.deadline
    # Use title, score, recommendation, deadline as needed
```

## Core Concepts

### Competitions

The library fetches competition metadata from the Kaggle API:

```python
from platform_kaggle import KaggleClient

client = KaggleClient()
competitions = client.list_competitions(
    search="tabular",
    category="Featured",
)
```

### Interest Filtering

Filter competitions by tags, categories, and reward amounts:

```python
from platform_kaggle import make_interest_filter, filter_competitions

interests = make_interest_filter(
    include_tags=("tabular", "nlp"),       # Must have at least one
    exclude_tags=("computer-vision",),     # Must not have any
    min_reward=1000,                       # Minimum prize (None = include Knowledge)
    categories=("Featured", "Research"),   # None = all categories
)

filtered = filter_competitions(competitions, interests)
```

### Codebase Capability Matching

The library scans your codebase to detect ML capabilities:

```python
from platform_kaggle import get_codebase_profile

profile = get_codebase_profile()
# Returns capabilities like:
# - ml_backends: ("xgboost", "lightgbm", "sklearn")
# - data_formats: ("csv", "parquet")
# - task_types: ("binary_classification", "regression")
```

Competitions are matched against these capabilities:

```python
from platform_kaggle import match_competition

match = match_competition(competition, profile)
# match.match_score: 0.0 - 1.0
# match.recommendation: "strong_fit" | "good_fit" | "stretch" | "new_territory"
# match.matched_capabilities: ("xgboost", "classification")
# match.missing_capabilities: ("deep_learning",)
```

## Testing

The library provides fakes and hooks for testing:

```python
from platform_kaggle import (
    FakeKaggleClient,
    hooks,
    make_fake_competition,
)

# Create fake client with test data
fake_client = FakeKaggleClient(
    competitions=(
        make_fake_competition(ref="test-comp", title="Test"),
    )
)

# Install via hooks
original = hooks.kaggle_client
hooks.kaggle_client = lambda: fake_client

try:
    # Your test code here
    ...
finally:
    hooks.kaggle_client = original
```

## Types

All types are exported from the package:

- `Competition` - Competition metadata
- `CompetitionMatch` - Match result with score
- `CodebaseProfile` - Detected capabilities
- `CodebaseCapability` - Individual capability
- `InterestFilter` - Filter configuration
- `CompetitionCategory` - Literal type for categories
- `CapabilityStrength` - Literal type for strength levels
- `MatchRecommendation` - Literal type for recommendations

## JSON Serialization

Encode/decode functions for all types:

```python
from platform_kaggle import encode_competition, decode_competition

# Serialize to JSON-compatible dict
data = encode_competition(competition)

# Deserialize with validation
competition = decode_competition(data)
```

## Architecture

See [docs/architecture-plan.md](docs/architecture-plan.md) for the full design document.
