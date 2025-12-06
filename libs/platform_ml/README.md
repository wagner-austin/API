# platform-ml

Shared ML artifact handling for the monorepo. Provides tarball utilities, versioned manifest schemas, and remote artifact storage via data-bank-api.

## Installation

```toml
[tool.poetry.dependencies]
platform-ml = { path = "../libs/platform_ml", develop = true }
```

## ArtifactStore

Remote artifact storage wrapping `DataBankClient`. Upload directories as tarballs and download them back with integrity checks.

```python
from pathlib import Path
from platform_ml import ArtifactStore

store = ArtifactStore(
    base_url="http://data-bank-api:8000",
    api_key="secret",
    timeout_seconds=600.0,
)

# Upload a directory as a tarball
resp = store.upload_artifact(
    Path("./model-output"),
    artifact_name="model-v1",
    request_id="req-123",
)
print(resp["file_id"], resp["sha256"])

# Download and extract
root = store.download_artifact(
    file_id="model-v1.tar.gz",
    dest_dir=Path("./downloaded"),
    request_id="req-456",
    expected_root="model-v1",
)
print(root)  # Path to extracted directory
```

## Manifest Schema

Typed model manifest schema (v2.0) for tracking ML artifacts with training metadata.

```python
from pathlib import Path
from platform_ml import (
    ModelManifestV2,
    TrainingRunMetadata,
    from_json_manifest_v2,
    from_path_manifest_v2,
)

# Parse from JSON string
manifest = from_json_manifest_v2('{"schema_version": "v2.0", ...}')

# Parse from file
manifest = from_path_manifest_v2(Path("manifest.json"))

# Access typed fields
print(manifest["model_type"])      # Literal["resnet18", "gpt2"]
print(manifest["file_id"])         # Remote artifact file ID
print(manifest["file_sha256"])     # SHA256 hash for integrity
print(manifest["training"]["epochs"])  # Training run config
```

### ModelManifestV2 Fields

| Field | Type | Required |
|-------|------|----------|
| `schema_version` | `Literal["v2.0"]` | Yes |
| `model_type` | `Literal["resnet18", "gpt2"]` | Yes |
| `model_id` | `str` | Yes |
| `created_at` | `str` (ISO 8601) | Yes |
| `arch` | `str` | Yes |
| `file_id` | `str` | Yes |
| `file_size` | `int` | Yes |
| `file_sha256` | `str` | Yes |
| `training` | `TrainingRunMetadata` | Yes |
| `n_classes` | `int` | No |
| `vocab_size` | `int` | No |
| `val_acc` | `float` | No |
| `val_loss` | `float` | No |
| `preprocess_hash` | `str` | No |

## Tarball Utilities

Create and extract gzip-compressed tarballs with security validation.

```python
from pathlib import Path
from platform_ml import create_tarball, extract_tarball, TarballError

# Create a tarball from a directory
tar_path = create_tarball(
    src_dir=Path("./model-files"),
    dest_file=Path("./artifacts/model.tar.gz"),
    root_name="model-v1",
)

# Extract with root validation (prevents path traversal)
try:
    root = extract_tarball(
        tar_path=Path("./artifacts/model.tar.gz"),
        dest_dir=Path("./extracted"),
        expected_root="model-v1",
    )
except TarballError as e:
    print(f"Extraction failed: {e}")
```

## Development

```bash
make lint    # Run ruff linter
make test    # Run pytest with coverage
make check   # Run both lint and test
```

## Requirements

- Python 3.12+
- platform-core (for DataBankClient, JSON utilities)

