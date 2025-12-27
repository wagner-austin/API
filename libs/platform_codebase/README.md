# platform-codebase

Shared codebase capability detection and profiling for monorepos.

## Overview

This library provides types and utilities for scanning monorepo codebases to detect capabilities, technologies, and frameworks. It serves as the foundation for capability matching in:

- `platform_kaggle` - matching Kaggle competitions to codebase capabilities
- `platform_devpost` - matching Devpost hackathons to codebase capabilities

## Installation

```bash
poetry add platform-codebase
```

## Dependencies

- `platform-core` - JSON utilities (`require_*` helpers), type validation

## Usage

### Scanning a Local Codebase

```python
from pathlib import Path
from platform_codebase import scan_libs, scan_services

# Scan libs/ and services/ directories
root = Path("/path/to/monorepo")
libs = scan_libs(root)
services = scan_services(root)

# Each lib/service has name, path, and dependencies
for lib in libs:
    print(f"{lib.name}: {lib.dependencies}")
```

### Scanning via GitHub API

For containerized environments without local file access, scan via GitHub:

```python
from platform_codebase import (
    GitHubClient,
    scan_libs_from_github,
    scan_services_from_github,
    parse_github_repo,
)

# Create authenticated GitHub client
client = GitHubClient(token="ghp_your_token")

# Parse repo string (handles "owner/repo" format)
owner, repo = parse_github_repo("wagner-austin/API")

# Scan libs and services via GitHub API
libs = scan_libs_from_github(client, owner, repo)
services = scan_services_from_github(client, owner, repo)

# Same LibInfo/ServiceInfo objects as local scanning
for lib in libs:
    print(f"{lib.name}: {lib.dependencies}")
```

The GitHub scanner fetches `pyproject.toml` files via the GitHub Contents API and parses them to extract dependency information.

### Parsing pyproject.toml

The library provides regex-based TOML parsing (tomllib is banned in this codebase):

```python
from pathlib import Path
from platform_codebase import parse_pyproject

path = Path("libs/my_lib/pyproject.toml")
name, dependencies = parse_pyproject(path)
print(f"Package: {name}")
print(f"Dependencies: {dependencies}")
```

### Building Profiles

Consumer libraries (platform_kaggle, platform_devpost) use the scanned data to build capability profiles:

```python
from platform_codebase import CodebaseCapability, CodebaseProfile

# Create a capability
cap = CodebaseCapability(
    name="tabular_classification",
    strength="strong",
    tags=("tabular", "classification", "xgboost"),
    description="XGBoost gradient boosting for tabular data",
)

# Create a profile
profile = CodebaseProfile(
    capabilities=(cap,),
    technologies=("python",),
    frameworks=("fastapi",),
    ml_backends=("xgboost", "lightgbm"),
    data_formats=("csv", "parquet"),
    task_types=("binary_classification",),
)
```

## Types

### CodebaseCapability

Represents a capability the codebase has:

```python
cap = CodebaseCapability(
    name="tabular_classification",
    strength="strong",           # "strong" | "moderate" | "basic"
    tags=("tabular", "xgboost"),
    description="XGBoost for tabular data",
)
```

### CodebaseProfile

Full profile of codebase capabilities:

```python
profile = CodebaseProfile(
    capabilities=(cap1, cap2),
    technologies=("python", "javascript"),
    frameworks=("fastapi", "react"),
    ml_backends=("xgboost", "lightgbm"),
    data_formats=("csv", "parquet"),
    task_types=("binary_classification", "regression"),
)
```

### LibInfo / ServiceInfo

Information about scanned libraries and services:

```python
lib = LibInfo(
    name="my-lib",
    path=Path("libs/my_lib"),
    dependencies=("fastapi", "httpx"),
)

service = ServiceInfo(
    name="my-service",
    path=Path("services/my_service"),
    dependencies=("flask",),
    has_rules_files=False,  # True if *.rules files exist
)
```

### Literal Types

```python
CapabilityStrength = Literal["strong", "moderate", "basic"]
MatchRecommendation = Literal["strong_fit", "good_fit", "stretch", "new_territory"]
```

## JSON Serialization

Encode/decode functions for all types:

```python
from platform_codebase import encode_capability, decode_capability

# Serialize to JSON-compatible dict
data = encode_capability(cap)

# Deserialize with validation
cap = decode_capability(data)
```

## Testing

The library provides test utilities via the `testing` module:

```python
from platform_codebase.testing import (
    make_fake_capability,
    make_fake_profile,
    make_fake_lib_info,
    make_fake_service_info,
)

# Create test data
cap = make_fake_capability(name="test_cap", tags=("test",))
profile = make_fake_profile(capabilities=(cap,))
lib = make_fake_lib_info(name="test-lib", dependencies=("pytest",))
service = make_fake_service_info(name="test-svc", has_rules_files=True)
```

### Testing GitHub Scanning

Use `FakeGitHubClient` to test code that uses GitHub scanning:

```python
from platform_codebase import FakeGitHubClient, scan_libs_from_github

# Create fake client with test data
fake_client = FakeGitHubClient(
    directories={"libs": ["my-lib"]},
    files={
        "libs/my-lib/pyproject.toml": """
[tool.poetry]
name = "my-lib"

[tool.poetry.dependencies]
python = "^3.11"
xgboost = "^2.0.0"
""",
    },
    path_patterns={("services/my-api", ".rules"): True},  # For has_rules_files detection
)

# Use in tests
libs = scan_libs_from_github(fake_client, "owner", "repo")
assert libs[0].name == "my-lib"
assert "xgboost" in libs[0].dependencies
```

## Service Integration

### Adding to Your Project

1. Add the dependency to your `pyproject.toml`:

```toml
[tool.poetry.dependencies]
platform-codebase = { path = "../platform_codebase", develop = true }
```

2. Import and use:

```python
from platform_codebase import scan_libs, scan_services, CodebaseProfile
```

### Building a Custom Capability Scanner

```python
from pathlib import Path
from platform_codebase import (
    scan_libs,
    scan_services,
    CodebaseCapability,
    CodebaseProfile,
)

def build_my_profile(root: Path) -> CodebaseProfile:
    """Build a capability profile for your specific needs."""
    libs = scan_libs(root)
    services = scan_services(root)

    # Collect all dependencies
    all_deps: set[str] = set()
    for lib in libs:
        all_deps.update(lib.dependencies)
    for svc in services:
        all_deps.update(svc.dependencies)

    # Detect capabilities based on dependencies
    capabilities: list[CodebaseCapability] = []

    if "xgboost" in all_deps:
        capabilities.append(CodebaseCapability(
            name="gradient_boosting",
            strength="strong",
            tags=("ml", "tabular", "xgboost"),
            description="XGBoost gradient boosting",
        ))

    if "fastapi" in all_deps:
        capabilities.append(CodebaseCapability(
            name="rest_api",
            strength="strong",
            tags=("web", "api", "fastapi"),
            description="FastAPI REST services",
        ))

    return CodebaseProfile(
        capabilities=tuple(capabilities),
        technologies=("python",),
        frameworks=tuple(d for d in all_deps if d in ("fastapi", "flask", "django")),
        ml_backends=tuple(d for d in all_deps if d in ("xgboost", "lightgbm", "torch")),
        data_formats=("csv", "json"),
        task_types=(),
    )
```

### Using in a FastAPI Service

```python
from fastapi import FastAPI
from platform_codebase import scan_libs, encode_lib_info

app = FastAPI()

@app.get("/codebase/libs")
def list_libs():
    """List all libraries in the monorepo."""
    libs = scan_libs(Path("/path/to/monorepo"))
    return [encode_lib_info(lib) for lib in libs]
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
