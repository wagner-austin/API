# platform-kaggle

Kaggle competition discovery with codebase capability matching.

This library provides tools for discovering Kaggle competitions that match
your codebase capabilities and personal interests.

## Installation

```bash
poetry add platform-kaggle
```

Requires valid Kaggle API credentials via the `KAGGLE_API_TOKEN` environment variable.

## Dependencies

- `platform-core` - JSON utilities, HTTP client protocols
- `platform-codebase` - Shared codebase scanning and capability detection
- `kaggle` - Official Kaggle API client

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

The library scans your codebase to detect ML capabilities using `platform_codebase`:

```python
from platform_kaggle import get_codebase_profile

profile = get_codebase_profile()
# Returns capabilities like:
# - ml_backends: ("xgboost", "lightgbm", "pytorch", "transformers", ...)
# - data_formats: ("csv", "parquet", "excel")
# - task_types: ("binary_classification", "image_classification", "text_generation", ...)
# - capabilities: detected from pyproject.toml dependencies
```

#### Detected Capabilities

The library automatically detects capabilities based on dependencies in `pyproject.toml`:

| Dependency | Capability Detected | Tags |
|------------|---------------------|------|
| `xgboost` | `xgboost_tabular` | tabular, classification, regression |
| `lightgbm` | `lightgbm_tabular` | tabular, classification, regression |
| `catboost` | (ml_backend only) | - |
| `torch` | `pytorch_deep_learning` | deep-learning, neural-network |
| `scikit-learn` | `sklearn_ml` | tabular, classification, regression |
| `optuna` | `hyperparameter_optimization` | optimization, hyperparameter-tuning |
| `torchvision` | `torchvision_cv` | computer-vision, image, image-classification |
| `pillow` | `image_processing` | image, image-processing |
| `opencv-python` | `opencv_cv` | computer-vision, image, video |
| `transformers` | `huggingface_transformers` | nlp, text-classification, text-generation, llm |
| `datasets` | `huggingface_datasets` | data, huggingface |
| `tokenizers` | `tokenization` | nlp, tokenization |
| `sentencepiece` | `sentencepiece_tokenization` | nlp, tokenization |
| `fasttext` / `fasttext-wheel` | `language_identification` | nlp, language-detection, multilingual |
| `openai` | `speech_to_text` | nlp, speech, transcription, whisper |
| `*.rules` files | `transliteration` | nlp, transliteration, script-conversion |

#### Detected Task Types

| Dependency | Task Types Added |
|------------|-----------------|
| `xgboost`, `lightgbm` | binary_classification, multiclass_classification, regression |
| `torch` | time_series, sequence_modeling |
| `scikit-learn` | clustering |
| `torchvision` | image_classification, object_detection |
| `transformers` | text_classification, text_generation, token_classification, question_answering, summarization, translation |
| `openai` | speech_recognition, translation |

#### Building Profiles from Pre-scanned Data

For containerized environments that scan via GitHub API, use `build_profile` with pre-scanned libs and services:

```python
from platform_codebase import scan_libs_from_github, scan_services_from_github, GitHubClient
from platform_kaggle import build_profile

# Scan via GitHub
client = GitHubClient(token="ghp_your_token")
libs = scan_libs_from_github(client, "owner", "repo")
services = scan_services_from_github(client, "owner", "repo")

# Build capability profile from scanned data
profile = build_profile(libs, services)

# Same profile structure as get_codebase_profile()
print(profile.ml_backends)    # ("xgboost", "lightgbm", "pytorch")
print(profile.capabilities)   # Detected capabilities
```

This separation allows capability detection to work with data from any source (local filesystem, GitHub API, etc.).

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
    reset_hooks,
)

# Create fake client with test data
fake_client = FakeKaggleClient(
    competitions=(
        make_fake_competition(ref="test-comp", title="Test"),
    )
)

# Install via hooks
hooks.kaggle_client = lambda: fake_client

try:
    # Your test code here
    ...
finally:
    reset_hooks()
```

### Available Fakes

- `FakeKaggleApi` - Low-level API fake implementing `KaggleApiProtocol`
- `FakeKaggleClient` - High-level client fake with `list_competitions` and `get_competition`
- `FakeKagglePageFetcher` - Page fetcher fake with `fetch_pages` and `get_competition_id`

### Factory Functions

- `make_fake_competition()` - Create test Competition instances
- `make_fake_kaggle_competition()` - Create raw Kaggle API competition objects
- `make_fake_capability()` - Create test CodebaseCapability instances
- `make_fake_profile()` - Create test CodebaseProfile instances
- `make_fake_competition_page()` - Create test CompetitionPage instances
- `make_fake_competition_pages()` - Create test CompetitionPages instances
- `make_interest_filter()` - Create InterestFilter instances

## Types

All types are exported from the package:

- `Competition` - Competition metadata (immutable `__slots__` class)
- `CompetitionMatch` - Match result with score
- `CompetitionPage` - Single page with id, name, content
- `CompetitionPages` - All pages for a competition (description, evaluation, timeline, rules)
- `CodebaseProfile` - Detected capabilities (from `platform_codebase`)
- `CodebaseCapability` - Individual capability (from `platform_codebase`)
- `InterestFilter` - Filter configuration
- `CompetitionCategory` - Literal type: `"Featured" | "Research" | "Playground" | ...`
- `CapabilityStrength` - Literal type: `"strong" | "moderate" | "basic"`
- `MatchRecommendation` - Literal type: `"strong_fit" | "good_fit" | "stretch" | "new_territory"`

## JSON Serialization

Encode/decode functions for all types:

```python
from platform_kaggle import encode_competition, decode_competition

# Serialize to JSON-compatible dict
data = encode_competition(competition)

# Deserialize with validation
competition = decode_competition(data)
```

## Hooks System

The library uses a hooks pattern for dependency injection:

| Hook | Type | Purpose |
|------|------|---------|
| `hooks.kaggle_api_factory` | `() -> KaggleApiProtocol` | Returns pre-authenticated Kaggle API |
| `hooks.kaggle_client` | `() -> KaggleClientProtocol` | Creates high-level client |
| `hooks.profile_scanner` | `(Path) -> CodebaseProfile` | Scans codebase for capabilities |
| `hooks.page_fetcher` | `() -> KagglePageFetcherProtocol` | Creates page fetcher for competition content |

## Service Integration

### Adding to Your Project

1. Add the dependency to your `pyproject.toml`:

```toml
[tool.poetry.dependencies]
platform-kaggle = { path = "../platform_kaggle", develop = true }
```

2. Set up Kaggle credentials:
   - Set `KAGGLE_API_TOKEN` environment variable with your API token

3. Import and use:

```python
from platform_kaggle import find_competitions, make_interest_filter
```

### Basic Usage in a Script

```python
from platform_kaggle import (
    find_competitions,
    make_interest_filter,
    encode_match,
)

def main() -> None:
    # Define what you're interested in
    interests = make_interest_filter(
        include_tags=("tabular", "nlp", "classification"),
        exclude_tags=("computer-vision",),
        min_reward=None,  # Include "Knowledge" competitions
        categories=("Featured", "Research", "Playground"),
    )

    # Find matching competitions
    matches = find_competitions(
        interests=interests,
        match_codebase=True,  # Score against your codebase capabilities
        min_match_score=0.3,  # Only show 30%+ matches
    )

    # Process results
    for match in matches:
        print(f"{match.competition.title}")
        print(f"  Score: {match.match_score:.0%}")
        print(f"  Fit: {match.recommendation}")
        print(f"  Deadline: {match.competition.deadline}")
        print()

if __name__ == "__main__":
    main()
```

## Fetching Competition Page Content

The library provides access to full competition page content (Description, Evaluation, Timeline, Rules) via Kaggle's internal API:

```python
from platform_kaggle import create_page_fetcher

# Create an initialized page fetcher
fetcher = create_page_fetcher()

# Get competition ID from slug
comp_id = fetcher.get_competition_id("titanic")  # Returns 3136

# Fetch all pages for a competition
pages = fetcher.fetch_pages(comp_id)

# Access structured content
print(pages.description)  # Full description markdown
print(pages.evaluation)   # Evaluation criteria
print(pages.timeline)     # Competition timeline
print(pages.rules)        # Competition rules

# Access individual pages
for page in pages.pages:
    print(f"{page.name}: {len(page.content)} chars")
```

### Session Management

For multiple requests, reuse the session:

```python
from platform_kaggle import KaggleSession, KagglePageFetcher

# Create and initialize session once
session = KaggleSession()
session.initialize()

# Create fetcher with session
fetcher = KagglePageFetcher(session)

# Make multiple requests
pages1 = fetcher.fetch_pages(3136)  # Titanic
pages2 = fetcher.fetch_pages(5407)  # House Prices
```

### Testing Page Fetcher

```python
from platform_kaggle import (
    FakeKagglePageFetcher,
    make_fake_competition_pages,
    hooks,
    reset_hooks,
)

# Create fake with test data
fake_pages = make_fake_competition_pages(
    competition_id=12345,
    description="Test description",
    evaluation="Test evaluation",
)
fake_fetcher = FakeKagglePageFetcher(
    pages={12345: fake_pages},
    competition_ids={"test-comp": 12345},
)

# Install via hooks
hooks.page_fetcher = lambda: fake_fetcher

try:
    # Your test code
    pages = fake_fetcher.fetch_pages(12345)
    assert pages.description == "Test description"
finally:
    reset_hooks()
```

### Using in a FastAPI Service

```python
from fastapi import FastAPI, Query
from platform_kaggle import (
    find_competitions,
    make_interest_filter,
    encode_match,
    CompetitionCategory,
)

app = FastAPI()

@app.get("/competitions")
def list_competitions(
    tags: list[str] = Query(default=["tabular"]),
    exclude: list[str] = Query(default=[]),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0),
    category: CompetitionCategory | None = None,
):
    """Find Kaggle competitions matching criteria."""
    interests = make_interest_filter(
        include_tags=tuple(tags),
        exclude_tags=tuple(exclude),
        categories=(category,) if category else None,
    )

    matches = find_competitions(
        interests=interests,
        match_codebase=True,
        min_match_score=min_score,
    )

    return [encode_match(m) for m in matches]


@app.get("/competitions/{ref}")
def get_competition(ref: str):
    """Get a specific competition by reference."""
    from platform_kaggle import KaggleClient, encode_competition

    client = KaggleClient()
    comp = client.get_competition(ref)
    if comp is None:
        raise HTTPException(404, f"Competition {ref} not found")
    return encode_competition(comp)
```

### Testing Your Integration

```python
import pytest
from platform_kaggle import (
    hooks,
    reset_hooks,
    FakeKaggleClient,
    make_fake_competition,
)

@pytest.fixture(autouse=True)
def reset_kaggle_hooks():
    """Reset hooks after each test."""
    yield
    reset_hooks()

def test_my_competition_endpoint():
    # Set up fake data
    fake_comp = make_fake_competition(
        ref="test-comp",
        title="Test Competition",
        tags=("tabular", "classification"),
    )
    fake_client = FakeKaggleClient(competitions=(fake_comp,))
    hooks.kaggle_client = lambda: fake_client

    # Test your code
    from myapp import list_competitions
    result = list_competitions(tags=["tabular"])

    assert len(result) == 1
    assert result[0]["competition"]["ref"] == "test-comp"
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
