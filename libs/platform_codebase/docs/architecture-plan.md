# Architecture: platform_codebase Library

## Overview

The `platform_codebase` library provides shared types and utilities for codebase capability detection and profiling. It serves as the foundation for:

- `platform_kaggle` - matching Kaggle competitions to codebase capabilities
- `platform_devpost` - matching Devpost hackathons to codebase capabilities

## Dependencies

- `platform-core` - JSON utilities (`require_*` helpers), type validation

## Directory Structure

```
libs/platform_codebase/
├── pyproject.toml
├── Makefile
├── README.md
├── docs/
│   └── plan.md
├── scripts/
│   ├── __init__.py
│   └── guard.py
├── src/platform_codebase/
│   ├── __init__.py      # Public exports
│   ├── py.typed         # PEP 561 marker
│   ├── types.py         # Core types and encode/decode
│   ├── toml.py          # Regex-based TOML parsing
│   ├── scanner.py       # Directory scanning utilities
│   └── testing.py       # Test factories and utilities
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_types.py
    ├── test_toml.py
    ├── test_scanner.py
    ├── test_testing.py
    └── test_guard.py
```

## Core Types (types.py)

### Literal Types

```python
CapabilityStrength = Literal["strong", "moderate", "basic"]
MatchRecommendation = Literal["strong_fit", "good_fit", "stretch", "new_territory"]
```

### Immutable Classes

All types use `__slots__` for immutability and memory efficiency:

```python
class CodebaseCapability:
    """A capability the codebase has."""
    __slots__ = ("description", "name", "strength", "tags")

    name: str                    # e.g., "tabular_classification"
    strength: CapabilityStrength
    tags: tuple[str, ...]        # e.g., ("tabular", "classification")
    description: str

class CodebaseProfile:
    """Full profile of codebase capabilities."""
    __slots__ = (
        "capabilities", "data_formats", "frameworks",
        "ml_backends", "task_types", "technologies",
    )

    capabilities: tuple[CodebaseCapability, ...]
    technologies: tuple[str, ...]   # e.g., ("python", "javascript")
    frameworks: tuple[str, ...]     # e.g., ("fastapi", "django")
    ml_backends: tuple[str, ...]    # e.g., ("xgboost", "lightgbm")
    data_formats: tuple[str, ...]   # e.g., ("csv", "parquet")
    task_types: tuple[str, ...]     # e.g., ("binary_classification",)

class LibInfo:
    """Information about a scanned library."""
    __slots__ = ("dependencies", "name", "path")

    name: str
    path: Path
    dependencies: tuple[str, ...]

class ServiceInfo:
    """Information about a scanned service."""
    __slots__ = ("dependencies", "has_rules_files", "name", "path")

    name: str
    path: Path
    dependencies: tuple[str, ...]
    has_rules_files: bool  # For transliteration detection
```

## TOML Parsing (toml.py)

Regex-based parsing to avoid tomllib (which is banned in this codebase):

```python
def extract_poetry_name(content: str) -> str:
    """Extract name from [tool.poetry] section using regex.

    Returns empty string if not found.
    """

def extract_poetry_dependencies(content: str) -> list[str]:
    """Extract dependencies from [tool.poetry.dependencies] section.

    Returns list of dependency names (strips extras like [dev]).
    """

def parse_pyproject(path: Path) -> tuple[str, tuple[str, ...]]:
    """Parse pyproject.toml and extract name and dependencies.

    Args:
        path: Path to pyproject.toml file.

    Returns:
        Tuple of (name, dependencies).
    """
```

### Regex Patterns

```python
# Extract name from [tool.poetry] section
NAME_PATTERN = r'\[tool\.poetry\][^\[]*name\s*=\s*"([^"]+)"'

# Extract [tool.poetry.dependencies] section
DEPS_SECTION_PATTERN = r'\[tool\.poetry\.dependencies\](.*?)(?:\[|$)'

# Extract individual dependency names
DEP_NAME_PATTERN = r'^([a-zA-Z][a-zA-Z0-9_-]*)'
```

## Scanner (scanner.py)

```python
def scan_libs(root: Path) -> tuple[LibInfo, ...]:
    """Scan libs/ directory for pyproject.toml files.

    Skips directories without pyproject.toml.
    Returns LibInfo for each valid library.
    """

def scan_services(root: Path) -> tuple[ServiceInfo, ...]:
    """Scan services/ directory for pyproject.toml files.

    Also detects *.rules files for transliteration services.
    Returns ServiceInfo for each valid service.
    """
```

### Scanning Logic

1. List subdirectories in `libs/` or `services/`
2. Check each for `pyproject.toml`
3. Parse pyproject.toml for name and dependencies
4. For services, also check for `*.rules` files
5. Return tuple of LibInfo/ServiceInfo

## Testing Utilities (testing.py)

Factory functions for creating test data:

```python
def make_fake_capability(
    *,
    name: str = "test_capability",
    strength: CapabilityStrength = "moderate",
    tags: tuple[str, ...] = ("test",),
    description: str = "Test capability",
) -> CodebaseCapability: ...

def make_fake_profile(
    *,
    capabilities: tuple[CodebaseCapability, ...] = (),
    technologies: tuple[str, ...] = ("python",),
    frameworks: tuple[str, ...] = ("fastapi",),
    ml_backends: tuple[str, ...] = ("xgboost",),
    data_formats: tuple[str, ...] = ("csv",),
    task_types: tuple[str, ...] = ("binary_classification",),
) -> CodebaseProfile: ...

def make_fake_lib_info(
    *,
    name: str = "test-lib",
    path: Path | None = None,
    dependencies: tuple[str, ...] = (),
) -> LibInfo: ...

def make_fake_service_info(
    *,
    name: str = "test-service",
    path: Path | None = None,
    dependencies: tuple[str, ...] = (),
    has_rules_files: bool = False,
) -> ServiceInfo: ...
```

## Consumer Usage

### platform_kaggle

```python
from platform_codebase import scan_libs, scan_services, CodebaseProfile
from platform_codebase.types import LibInfo, ServiceInfo

def scan_codebase(root: Path) -> CodebaseProfile:
    libs = scan_libs(root)
    services = scan_services(root)
    return _build_profile(libs, services)

# Uses: capabilities, ml_backends, data_formats, task_types
```

### platform_devpost

```python
from platform_codebase import scan_libs, scan_services, CodebaseProfile

def scan_codebase(root: Path) -> CodebaseProfile:
    libs = scan_libs(root)
    services = scan_services(root)
    return _build_profile(libs, services)

# Uses: capabilities, technologies, frameworks
```

## Public API (__init__.py)

```python
# Types
from platform_codebase.types import (
    CapabilityStrength,
    CodebaseCapability,
    CodebaseProfile,
    LibInfo,
    MatchRecommendation,
    ServiceInfo,
)

# Encode/decode
from platform_codebase.types import (
    decode_capability,
    decode_lib_info,
    decode_profile,
    decode_service_info,
    encode_capability,
    encode_lib_info,
    encode_profile,
    encode_service_info,
)

# TOML parsing
from platform_codebase.toml import (
    extract_poetry_dependencies,
    extract_poetry_name,
    parse_pyproject,
)

# Scanning
from platform_codebase.scanner import (
    scan_libs,
    scan_services,
)
```

## Test Coverage

- 100% statement and branch coverage required
- Tests for each encode/decode pair with round-trip validation
- Tests for TOML parsing edge cases:
  - Empty sections
  - No dependencies
  - Missing name field
  - Dependencies with extras (`pkg[extra]`)
- Tests for scanner with temp directories:
  - Empty libs/services directories
  - Non-directory entries
  - Missing pyproject.toml
  - Services with *.rules files
- Tests for guard.py using runpy
