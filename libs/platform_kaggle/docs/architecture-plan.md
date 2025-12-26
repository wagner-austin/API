# Plan: platform_kaggle Library

## Goal
Create a new `libs/platform_kaggle` library that:
1. Wraps Kaggle API for listing competitions
2. Filters by user interests (ML, linguistics, etc.)
3. Matches competitions to codebase capabilities
4. Follows existing platform_* patterns (Protocols, TypedDict, hooks)

---

## Architecture

### Directory Structure
```
libs/platform_kaggle/
├── pyproject.toml
├── README.md
├── src/platform_kaggle/
│   ├── __init__.py           # Public exports
│   ├── py.typed              # PEP 561 marker
│   ├── types.py              # TypedDicts + Protocols
│   ├── client.py             # KaggleClient wrapper
│   ├── capabilities.py       # Codebase capability detection
│   ├── matcher.py            # Competition-to-capability matching
│   ├── filters.py            # Interest-based filtering
│   └── testing.py            # Fakes + hooks container
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_client.py
    ├── test_capabilities.py
    ├── test_matcher.py
    └── test_filters.py
```

---

## Core Types (types.py)

### Literal Types
```python
CompetitionCategory = Literal["Featured", "Research", "Playground", "Getting Started", "Masters", "Kudos"]
CapabilityStrength = Literal["strong", "moderate", "basic"]
MatchRecommendation = Literal["strong_fit", "good_fit", "stretch", "new_territory"]
```

### Internal Validation Helpers
```python
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_str,
    require_int,
    require_float,
    require_list,
    require_dict,
    optional_int,
)


def _require_list_str(obj: JSONObject, key: str) -> list[str]:
    """Extract required list of strings from JSON object."""
    items = require_list(obj, key)
    result: list[str] = []
    for i, item in enumerate(items):
        if not isinstance(item, str):
            raise JSONTypeError(f"Field '{key}[{i}]' must be a string, got {type(item).__name__}")
        result.append(item)
    return result


def _require_dict_value(value: JSONValue, context: str) -> JSONObject:
    """Require value to be a dict."""
    if not isinstance(value, dict):
        raise JSONTypeError(f"{context} must be an object, got {type(value).__name__}")
    return value


def _require_category(obj: JSONObject, key: str) -> CompetitionCategory:
    """Extract and validate CompetitionCategory from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Field key.

    Returns:
        Validated CompetitionCategory.

    Raises:
        JSONTypeError: If field is missing or not a valid category.
    """
    value = require_str(obj, key)
    if value == "Featured":
        return "Featured"
    if value == "Research":
        return "Research"
    if value == "Playground":
        return "Playground"
    if value == "Getting Started":
        return "Getting Started"
    if value == "Masters":
        return "Masters"
    if value == "Kudos":
        return "Kudos"
    raise JSONTypeError(f"Field '{key}' must be a valid category, got '{value}'")


def _require_strength(obj: JSONObject, key: str) -> CapabilityStrength:
    """Extract and validate CapabilityStrength from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Field key.

    Returns:
        Validated CapabilityStrength.

    Raises:
        JSONTypeError: If field is missing or not a valid strength.
    """
    value = require_str(obj, key)
    if value == "strong":
        return "strong"
    if value == "moderate":
        return "moderate"
    if value == "basic":
        return "basic"
    raise JSONTypeError(f"Field '{key}' must be strong/moderate/basic, got '{value}'")


def _require_recommendation(obj: JSONObject, key: str) -> MatchRecommendation:
    """Extract and validate MatchRecommendation from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Field key.

    Returns:
        Validated MatchRecommendation.

    Raises:
        JSONTypeError: If field is missing or not a valid recommendation.
    """
    value = require_str(obj, key)
    if value == "strong_fit":
        return "strong_fit"
    if value == "good_fit":
        return "good_fit"
    if value == "stretch":
        return "stretch"
    if value == "new_territory":
        return "new_territory"
    raise JSONTypeError(f"Field '{key}' must be a valid recommendation, got '{value}'")
```

### Competition Model
```python
class Competition(TypedDict):
    """Kaggle competition metadata."""
    ref: str                    # e.g., "amex-default-prediction"
    title: str
    category: CompetitionCategory
    reward: str                 # "$100,000" or "Knowledge"
    deadline: str               # ISO 8601 date
    team_count: int
    tags: tuple[str, ...]       # ("tabular", "classification", "finance")
    description: str
    url: str


def encode_competition(comp: Competition) -> dict[str, object]:
    """Encode Competition to JSON-serializable dict."""
    return {
        "ref": comp["ref"],
        "title": comp["title"],
        "category": comp["category"],
        "reward": comp["reward"],
        "deadline": comp["deadline"],
        "team_count": comp["team_count"],
        "tags": list(comp["tags"]),
        "description": comp["description"],
        "url": comp["url"],
    }


def decode_competition(data: JSONObject) -> Competition:
    """Decode Competition from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated Competition.

    Raises:
        JSONTypeError: If validation fails.
    """
    return Competition(
        ref=require_str(data, "ref"),
        title=require_str(data, "title"),
        category=_require_category(data, "category"),
        reward=require_str(data, "reward"),
        deadline=require_str(data, "deadline"),
        team_count=require_int(data, "team_count"),
        tags=tuple(_require_list_str(data, "tags")),
        description=require_str(data, "description"),
        url=require_str(data, "url"),
    )
```

### Capability Model
```python
class CodebaseCapability(TypedDict):
    """A capability the codebase has."""
    name: str
    strength: CapabilityStrength
    tags: tuple[str, ...]
    description: str


def encode_capability(cap: CodebaseCapability) -> dict[str, object]:
    """Encode CodebaseCapability to JSON-serializable dict."""
    return {
        "name": cap["name"],
        "strength": cap["strength"],
        "tags": list(cap["tags"]),
        "description": cap["description"],
    }


def decode_capability(data: JSONObject) -> CodebaseCapability:
    """Decode CodebaseCapability from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CodebaseCapability.

    Raises:
        JSONTypeError: If validation fails.
    """
    return CodebaseCapability(
        name=require_str(data, "name"),
        strength=_require_strength(data, "strength"),
        tags=tuple(_require_list_str(data, "tags")),
        description=require_str(data, "description"),
    )


class CodebaseProfile(TypedDict):
    """Full profile of codebase capabilities."""
    capabilities: tuple[CodebaseCapability, ...]
    ml_backends: tuple[str, ...]
    data_formats: tuple[str, ...]
    task_types: tuple[str, ...]


def encode_profile(profile: CodebaseProfile) -> dict[str, object]:
    """Encode CodebaseProfile to JSON-serializable dict."""
    return {
        "capabilities": [encode_capability(c) for c in profile["capabilities"]],
        "ml_backends": list(profile["ml_backends"]),
        "data_formats": list(profile["data_formats"]),
        "task_types": list(profile["task_types"]),
    }


def decode_profile(data: JSONObject) -> CodebaseProfile:
    """Decode CodebaseProfile from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CodebaseProfile.

    Raises:
        JSONTypeError: If validation fails.
    """
    caps_raw = require_list(data, "capabilities")
    return CodebaseProfile(
        capabilities=tuple(
            decode_capability(_require_dict_value(c, f"capabilities[{i}]"))
            for i, c in enumerate(caps_raw)
        ),
        ml_backends=tuple(_require_list_str(data, "ml_backends")),
        data_formats=tuple(_require_list_str(data, "data_formats")),
        task_types=tuple(_require_list_str(data, "task_types")),
    )
```

### Match Result
```python
class CompetitionMatch(TypedDict):
    """A competition scored against codebase capabilities."""
    competition: Competition
    match_score: float          # 0.0 - 1.0
    matched_capabilities: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    recommendation: MatchRecommendation


def encode_match(match: CompetitionMatch) -> dict[str, object]:
    """Encode CompetitionMatch to JSON-serializable dict."""
    return {
        "competition": encode_competition(match["competition"]),
        "match_score": match["match_score"],
        "matched_capabilities": list(match["matched_capabilities"]),
        "missing_capabilities": list(match["missing_capabilities"]),
        "recommendation": match["recommendation"],
    }


def decode_match(data: JSONObject) -> CompetitionMatch:
    """Decode CompetitionMatch from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated CompetitionMatch.

    Raises:
        JSONTypeError: If validation fails.
    """
    comp_raw = data.get("competition")
    return CompetitionMatch(
        competition=decode_competition(_require_dict_value(comp_raw, "competition")),
        match_score=require_float(data, "match_score"),
        matched_capabilities=tuple(_require_list_str(data, "matched_capabilities")),
        missing_capabilities=tuple(_require_list_str(data, "missing_capabilities")),
        recommendation=_require_recommendation(data, "recommendation"),
    )
```

### Filter Types
```python
class InterestFilter(TypedDict):
    """User interest filter for competitions."""
    include_tags: tuple[str, ...]   # Must have at least one
    exclude_tags: tuple[str, ...]   # Must not have any
    min_reward: int | None          # Minimum prize (None = include Knowledge)
    categories: tuple[CompetitionCategory, ...] | None  # None = all


def encode_filter(f: InterestFilter) -> dict[str, object]:
    """Encode InterestFilter to JSON-serializable dict."""
    return {
        "include_tags": list(f["include_tags"]),
        "exclude_tags": list(f["exclude_tags"]),
        "min_reward": f["min_reward"],
        "categories": list(f["categories"]) if f["categories"] else None,
    }


def decode_filter(data: JSONObject) -> InterestFilter:
    """Decode InterestFilter from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated InterestFilter.

    Raises:
        JSONTypeError: If validation fails.
    """
    cats_raw = data.get("categories")
    categories: tuple[CompetitionCategory, ...] | None = None
    if cats_raw is not None:
        if not isinstance(cats_raw, list):
            raise JSONTypeError(f"Field 'categories' must be an array, got {type(cats_raw).__name__}")
        categories = tuple(_require_category_value(c, f"categories[{i}]") for i, c in enumerate(cats_raw))

    return InterestFilter(
        include_tags=tuple(_require_list_str(data, "include_tags")),
        exclude_tags=tuple(_require_list_str(data, "exclude_tags")),
        min_reward=optional_int(data, "min_reward"),
        categories=categories,
    )


def _require_category_value(value: JSONValue, context: str) -> CompetitionCategory:
    """Require value to be a valid CompetitionCategory."""
    if not isinstance(value, str):
        raise JSONTypeError(f"{context} must be a string, got {type(value).__name__}")
    if value == "Featured":
        return "Featured"
    if value == "Research":
        return "Research"
    if value == "Playground":
        return "Playground"
    if value == "Getting Started":
        return "Getting Started"
    if value == "Masters":
        return "Masters"
    if value == "Kudos":
        return "Kudos"
    raise JSONTypeError(f"{context} must be a valid category, got '{value}'")
```

---

## Protocols

### KaggleClientProtocol
```python
@runtime_checkable
class KaggleClientProtocol(Protocol):
    """Protocol for Kaggle API client."""

    def list_competitions(
        self,
        *,
        search: str | None = None,
        category: CompetitionCategory | None = None,
    ) -> tuple[Competition, ...]:
        """List active competitions with optional filters."""
        ...

    def get_competition(self, ref: str) -> Competition | None:
        """Get a specific competition by ref."""
        ...
```

---

## Key Modules

### 1. client.py - Kaggle API Wrapper
- Wraps `kaggle.api.kaggle_api_extended.KaggleApi`
- Converts API response to typed `Competition` TypedDict
- Handles authentication via kaggle.json or env vars
- Uses hooks pattern for testability

### 2. capabilities.py - Dynamic Codebase Capability Detection

#### Internal Types for Scanner
```python
class LibInfo(TypedDict):
    """Information about a scanned library."""
    name: str
    path: Path
    dependencies: tuple[str, ...]


class ServiceInfo(TypedDict):
    """Information about a scanned service."""
    name: str
    path: Path
    dependencies: tuple[str, ...]
    has_rules_files: bool  # For turkic-api transliteration detection
```

Scans libs/ and services/ pyproject.toml files to detect installed capabilities:

```python
class CapabilityScanner:
    """Scans codebase to detect ML/NLP capabilities."""

    def __init__(self, root_path: Path) -> None:
        self._root = root_path

    def scan(self) -> CodebaseProfile:
        """Scan codebase and return capability profile."""
        libs = self._scan_libs()
        services = self._scan_services()
        return self._build_profile(libs, services)

    def _scan_libs(self) -> tuple[LibInfo, ...]:
        """Scan libs/ directory for pyproject.toml files."""
        # Parse each pyproject.toml, extract name and dependencies
        ...

    def _scan_services(self) -> tuple[ServiceInfo, ...]:
        """Scan services/ directory for pyproject.toml files."""
        ...

    def _build_profile(
        self,
        libs: tuple[LibInfo, ...],
        services: tuple[ServiceInfo, ...],
    ) -> CodebaseProfile:
        """Build capability profile from scanned libs/services."""
        capabilities: list[CodebaseCapability] = []

        # Detect ML backends from covenant_ml dependencies
        if self._has_lib("covenant_ml"):
            if self._has_dep("xgboost"):
                capabilities.append(CodebaseCapability(...))
            if self._has_dep("lightgbm"):
                capabilities.append(CodebaseCapability(...))
            # etc.

        # Detect NLP from turkic-api, grandma-api
        if self._has_service("turkic-api"):
            capabilities.append(CodebaseCapability(
                name="language_identification",
                strength="moderate",
                tags=("nlp", "language-detection", "multilingual"),
                description="FastText LID-218e for 218+ languages",
            ))

        return CodebaseProfile(
            capabilities=tuple(capabilities),
            ml_backends=self._detect_backends(),
            data_formats=self._detect_formats(),
            task_types=self._detect_task_types(),
        )


def scan_codebase(root: Path | None = None) -> CodebaseProfile:
    """Scan codebase and return capability profile.

    Args:
        root: Path to monorepo root. Defaults to PROJECTS/API.

    Returns:
        CodebaseProfile with detected capabilities.
    """
    if root is None:
        root = _detect_monorepo_root()
    scanner = CapabilityScanner(root)
    return scanner.scan()
```

#### Capability Detection Rules

| Lib/Service | Dependency/File | Capability Detected |
|-------------|-----------------|---------------------|
| covenant_ml | xgboost | tabular_classification (strong) |
| covenant_ml | lightgbm | large_tabular (strong) |
| covenant_ml | torch + lstm | time_series (strong) |
| covenant_ml | optuna | hyperparameter_optimization (strong) |
| turkic-api | fasttext | language_identification (moderate) |
| turkic-api | *.rules files | transliteration (moderate) |
| grandma-api | openai | speech_translation (moderate) |
| platform_stt | openai + whisper | speech_to_text (moderate) |

### 3. matcher.py - Competition Matching
Scores competitions against codebase capabilities:

```python
def match_competition(
    competition: Competition,
    profile: CodebaseProfile,
) -> CompetitionMatch:
    """Score a competition against codebase capabilities."""
    # Match competition tags to capability tags
    # Calculate match_score based on overlap
    # Identify matched and missing capabilities
    # Determine recommendation level
```

### 4. filters.py - Interest Filtering
Filter competitions by user interests:

```python
class InterestFilter(TypedDict):
    """User interest filter."""
    include_tags: tuple[str, ...]  # Must have at least one
    exclude_tags: tuple[str, ...]  # Must not have any
    min_reward: int | None         # Minimum prize amount
    categories: tuple[CompetitionCategory, ...] | None

def filter_competitions(
    competitions: tuple[Competition, ...],
    interests: InterestFilter,
) -> tuple[Competition, ...]:
    """Filter competitions by user interests."""
```

---

## Public API (from __init__.py)

```python
# Main functions
def find_competitions(
    *,
    interests: InterestFilter | None = None,
    match_codebase: bool = True,
    min_match_score: float = 0.0,
) -> tuple[CompetitionMatch, ...]:
    """Find competitions matching interests and codebase capabilities."""

def get_codebase_profile() -> CodebaseProfile:
    """Get the capability profile of this codebase."""

# Types (re-exported)
Competition, CompetitionMatch, CodebaseProfile, InterestFilter, ...
```

---

## Testing Strategy (testing.py)

### Hooks Pattern
Following libs pattern: `testing.py` is public, exports fakes and hooks for consumers.

```python
"""Public test utilities for platform_kaggle consumers."""
from typing import Callable
from platform_kaggle.types import Competition, CodebaseProfile


# Hook types
KaggleClientHook = Callable[[], "KaggleClientProtocol"]
ProfileScannerHook = Callable[[], CodebaseProfile]


class HooksContainer:
    """Container for dependency injection hooks."""

    kaggle_client: KaggleClientHook
    profile_scanner: ProfileScannerHook


hooks = HooksContainer()


def _init_production_hooks() -> None:
    """Initialize hooks with production implementations."""
    from platform_kaggle.client import KaggleClient
    from platform_kaggle.capabilities import scan_codebase

    hooks.kaggle_client = KaggleClient
    hooks.profile_scanner = scan_codebase


def reset_hooks() -> None:
    """Reset hooks to production implementations (for test teardown)."""
    _init_production_hooks()


# Initialize on module load
_init_production_hooks()


# --- Fake Implementations ---

class FakeKaggleClient:
    """Fake Kaggle client for testing."""

    def __init__(self, competitions: tuple[Competition, ...] = ()) -> None:
        self._competitions = competitions
        self._list_calls: list[dict[str, object]] = []

    def list_competitions(
        self,
        *,
        search: str | None = None,
        category: str | None = None,
    ) -> tuple[Competition, ...]:
        """Return configured competitions, optionally filtered."""
        self._list_calls.append({"search": search, "category": category})
        result = self._competitions
        if search:
            result = tuple(c for c in result if search.lower() in c["title"].lower())
        if category:
            result = tuple(c for c in result if c["category"] == category)
        return result

    def get_competition(self, ref: str) -> Competition | None:
        """Get competition by ref."""
        for c in self._competitions:
            if c["ref"] == ref:
                return c
        return None


def make_fake_competition(
    *,
    ref: str = "test-competition",
    title: str = "Test Competition",
    category: str = "Playground",
    reward: str = "Knowledge",
    deadline: str = "2025-12-31",
    team_count: int = 100,
    tags: tuple[str, ...] = ("tabular",),
    description: str = "Test description",
) -> Competition:
    """Factory for creating test Competition instances."""
    return Competition(
        ref=ref,
        title=title,
        category=category,
        reward=reward,
        deadline=deadline,
        team_count=team_count,
        tags=tags,
        description=description,
        url=f"https://www.kaggle.com/competitions/{ref}",
    )


def make_fake_profile(
    *,
    capabilities: tuple[CodebaseCapability, ...] = (),
    ml_backends: tuple[str, ...] = ("xgboost",),
    data_formats: tuple[str, ...] = ("csv",),
    task_types: tuple[str, ...] = ("binary_classification",),
) -> CodebaseProfile:
    """Factory for creating test CodebaseProfile instances."""
    return CodebaseProfile(
        capabilities=capabilities,
        ml_backends=ml_backends,
        data_formats=data_formats,
        task_types=task_types,
    )
```

### Test Structure
```
tests/
├── __init__.py
├── conftest.py           # Fixtures: reset_hooks, fake_client, fake_profile
├── test_types.py         # encode/decode round-trip tests
├── test_client.py        # KaggleClient tests with fake API
├── test_capabilities.py  # CapabilityScanner tests
├── test_matcher.py       # Match scoring tests
├── test_filters.py       # Interest filter tests
└── test_integration.py   # End-to-end with fakes
```

### Test Coverage Requirements
- 100% statement and branch coverage
- Tests for each encode/decode pair with round-trip validation
- Tests for each filter combination
- Tests for match scoring edge cases (0%, 50%, 100% matches)
- Tests for capability detection from pyproject.toml
- Integration test with fake client and fake profile

---

## Full pyproject.toml

```toml
[build-system]
requires = ["poetry-core>=1.3.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "platform-kaggle"
version = "0.1.0"
description = "Kaggle competition discovery with codebase capability matching"
authors = ["Austin Wagner <austinwagner@msn.com>"]
packages = [{ include = "platform_kaggle", from = "src" }]
readme = "README.md"
include = ["src/platform_kaggle/py.typed"]

[tool.poetry.dependencies]
python = "^3.11"
kaggle = "^1.8.0"
platform-core = { path = "../platform_core", develop = true }

[tool.poetry.group.dev.dependencies]
pytest = "^9.0.0"
pytest-xdist = "^3.6.1"
pytest-cov = "^7.0.0"
mypy = "^1.13.0"
ruff = "^0.14.4"

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
no_implicit_reexport = true
strict_equality = true
disallow_subclassing_any = true
disallow_any_expr = true
disallow_any_decorated = true
disallow_any_explicit = true
disallow_any_generics = true
extra_checks = true

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "F",      # pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "BLE",    # flake8-blind-except
    "UP",     # pyupgrade
    "N",      # pep8-naming
    "C4",     # flake8-comprehensions
    "SIM",    # flake8-simplify
    "RET",    # flake8-return
    "C90",    # mccabe complexity
    "RUF",    # ruff-specific
    "ANN",    # flake8-annotations
]
ignore = [
    "ANN101",  # Missing type annotation for self
    "ANN102",  # Missing type annotation for cls
]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["ANN201"]

[tool.coverage.run]
branch = true
source = ["src/platform_kaggle"]
omit = ["tests/*"]

[tool.coverage.report]
fail_under = 100
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
]
```

---

## Example Usage

```python
from platform_kaggle import find_competitions, InterestFilter

# Find ML/linguistics competitions that fit the codebase
matches = find_competitions(
    interests=InterestFilter(
        include_tags=("tabular", "nlp", "linguistics", "classification"),
        exclude_tags=("computer-vision", "image"),
        min_reward=None,  # Include Knowledge competitions
        categories=("Featured", "Research", "Playground"),
    ),
    match_codebase=True,
    min_match_score=0.3,
)

for match in matches:
    print(f"{match['competition']['title']}")
    print(f"  Score: {match['match_score']:.0%}")
    print(f"  Fit: {match['recommendation']}")
    print(f"  Deadline: {match['competition']['deadline']}")
```

---

## Implementation Order

1. **types.py** - All TypedDicts and Protocols
2. **testing.py** - Hooks container and FakeKaggleClient
3. **client.py** - Real Kaggle API wrapper
4. **capabilities.py** - Codebase profile definition
5. **matcher.py** - Competition matching logic
6. **filters.py** - Interest filtering
7. **__init__.py** - Public API exports
8. **Tests** - Full coverage for each module

---

## Files to Create

| File | Purpose |
|------|---------|
| `libs/platform_kaggle/pyproject.toml` | Package config, deps, tool settings |
| `libs/platform_kaggle/README.md` | Usage documentation |
| `libs/platform_kaggle/src/platform_kaggle/__init__.py` | Public exports |
| `libs/platform_kaggle/src/platform_kaggle/py.typed` | PEP 561 marker |
| `libs/platform_kaggle/src/platform_kaggle/types.py` | TypedDicts + Protocols |
| `libs/platform_kaggle/src/platform_kaggle/client.py` | Kaggle API wrapper |
| `libs/platform_kaggle/src/platform_kaggle/capabilities.py` | Codebase profile |
| `libs/platform_kaggle/src/platform_kaggle/matcher.py` | Match scoring |
| `libs/platform_kaggle/src/platform_kaggle/filters.py` | Interest filtering |
| `libs/platform_kaggle/src/platform_kaggle/testing.py` | Fakes + hooks |
| `libs/platform_kaggle/tests/__init__.py` | Test package |
| `libs/platform_kaggle/tests/conftest.py` | Fixtures |
| `libs/platform_kaggle/tests/test_*.py` | Test modules |
