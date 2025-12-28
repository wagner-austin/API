# AI Instructions for platform_codebase

## What This Library Does

Scans the monorepo to detect:
- Libraries in `libs/` directory
- Services in `services/` directory
- Dependencies from pyproject.toml files
- Capabilities (ML backends, frameworks, data formats, task types)

## Do NOT Use This Directly

**Use the opportunity-radar-api instead:**

```bash
# Get codebase profile
curl "https://opportunity-radar-api-production.up.railway.app/codebase/profile"

# List all libs
curl "https://opportunity-radar-api-production.up.railway.app/codebase/libs"

# List all services
curl "https://opportunity-radar-api-production.up.railway.app/codebase/services"
```

## When To Use This Library Directly

Only use this library directly when:
1. Writing tests for opportunity-radar-api
2. Extending capability detection in platform_kaggle
3. The API is unavailable and you need local scanning

## Key Functions (for lib development only)

```python
from platform_codebase import (
    scan_libs,           # Scan libs/ directory -> tuple[LibInfo, ...]
    scan_services,       # Scan services/ directory -> tuple[ServiceInfo, ...]
    collect_all_dependencies,  # Get all deps from libs+services
    has_dependency,      # Check if a dependency exists
)
```

## Types

- `LibInfo`: name, path, dependencies
- `ServiceInfo`: name, path, dependencies, has_rules_files
- `CodebaseProfile`: capabilities, technologies, frameworks, ml_backends, data_formats, task_types
- `CodebaseCapability`: name, strength, tags, description

## GitHub Scanner

For Docker deployments without local filesystem access:

```python
from platform_codebase import (
    GitHubClient,
    scan_libs_from_github,
    scan_services_from_github,
)

client = GitHubClient(token="...", repo="owner/repo")
libs = scan_libs_from_github(client)
services = scan_services_from_github(client)
```

## Testing Fakes

```python
from platform_codebase import (
    make_fake_lib_info,
    make_fake_service_info,
    make_fake_profile,
    make_fake_capability,
    FakeGitHubClient,
)
```
