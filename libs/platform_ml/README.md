# platform-ml

Shared ML artifact handling: tarball utilities, versioned manifest schemas, remote artifact storage, and Weights & Biases integration.

## Installation

```toml
[tool.poetry.dependencies]
platform-ml = { path = "../libs/platform_ml", develop = true }
```

## Quick Start

```python
from pathlib import Path
from platform_ml import ArtifactStore, create_tarball, extract_tarball

# Upload artifacts
store = ArtifactStore(base_url="http://data-bank-api:8000", api_key="secret")
resp = store.upload_artifact(Path("./model-output"), artifact_name="model-v1", request_id="req-123")

# Download and extract
root = store.download_artifact(file_id="model-v1.tar.gz", dest_dir=Path("./downloaded"), request_id="req-456")
```

## ArtifactStore

Remote artifact storage wrapping `DataBankClient`. Upload directories as tarballs and download them back with integrity checks.

```python
from pathlib import Path
from platform_ml import ArtifactStore, ArtifactStoreError

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
try:
    root = store.download_artifact(
        file_id="model-v1.tar.gz",
        dest_dir=Path("./downloaded"),
        request_id="req-456",
        expected_root="model-v1",
    )
    print(root)  # Path to extracted directory
except ArtifactStoreError as e:
    print(f"Failed: {e}")
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
    MANIFEST_SCHEMA_VERSION,
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

## WandbPublisher

Protocol-based Weights & Biases integration for experiment tracking across ML services.

```python
from platform_ml import WandbPublisher, WandbUnavailableError

# Create publisher (requires wandb package installed)
try:
    publisher = WandbPublisher(
        project="my-ml-project",
        run_name="gpt2-run-001",
        enabled=True,
    )
except WandbUnavailableError:
    # wandb not installed, continue without tracking
    publisher = None

# Log training config at start
if publisher:
    publisher.log_config({
        "model_family": "gpt2",
        "batch_size": 8,
        "learning_rate": 0.001,
    })

# Log per-step metrics during training
if publisher:
    publisher.log_step({
        "global_step": step,
        "train_loss": loss,
        "train_ppl": ppl,
        "grad_norm": grad_norm,
    })

# Log epoch-end validation metrics
if publisher:
    publisher.log_epoch({
        "epoch": epoch,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "best_val_loss": best_val_loss,
    })

# Log final test metrics
if publisher:
    publisher.log_final({
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "early_stopped": early_stopped,
    })

# Log summary table
if publisher:
    publisher.log_table(
        "epoch_summary",
        columns=["epoch", "train_loss", "val_loss"],
        data=[[0, 2.5, 2.3], [1, 1.8, 1.7]],
    )

# Finish run
if publisher:
    publisher.finish()
```

### WandbPublisher Methods

| Method | Description |
|--------|-------------|
| `log_config(config)` | Log training configuration dict |
| `log_step(metrics)` | Log per-step training metrics |
| `log_epoch(metrics)` | Log epoch-end validation metrics |
| `log_final(metrics)` | Log final test metrics |
| `log_table(name, columns, data)` | Log summary table |
| `finish()` | Close the wandb run |
| `get_init_result()` | Get status and run_id |
| `is_enabled` | Property: whether wandb is active |

### Disabled Mode

When `enabled=False`, all methods are no-ops (safe to call without checks):

```python
publisher = WandbPublisher(project="x", run_name="y", enabled=False)
publisher.log_step({"loss": 1.0})  # No-op, no error
```

### Wandb Types

```python
from platform_ml import (
    WandbRunConfig,
    WandbPublisherConfig,
    WandbStepMetrics,
    WandbEpochMetrics,
    WandbFinalMetrics,
    WandbTableRow,
    WandbInitResult,
)
```

## Device Selection

Centralized device detection and configuration for ML training across services. Prevents drift by providing a single source of truth for device resolution, precision selection, and batch size recommendations.

```python
from platform_ml import (
    RequestedDevice,
    ResolvedDevice,
    RequestedPrecision,
    ResolvedPrecision,
    resolve_device,
    resolve_precision,
    recommended_batch_size,
)

# Resolve device: "auto" detects CUDA availability
device: ResolvedDevice = resolve_device("auto")  # "cuda" or "cpu"
device: ResolvedDevice = resolve_device("cuda")  # passthrough
device: ResolvedDevice = resolve_device("cpu")   # passthrough

# Resolve precision based on device
precision: ResolvedPrecision = resolve_precision("auto", device)
# "auto" on CUDA -> "fp16", "auto" on CPU -> "fp32"
precision: ResolvedPrecision = resolve_precision("fp16", "cuda")  # OK
# resolve_precision("fp16", "cpu")  # RuntimeError: fp16 not supported on CPU

# Recommended batch size (bumps small batches on CUDA)
batch_size = recommended_batch_size(4, "cuda")  # 8
batch_size = recommended_batch_size(4, "cpu")   # 4
batch_size = recommended_batch_size(16, "cuda") # 16 (preserved)
```

### Device Types

| Type | Values | Description |
|------|--------|-------------|
| `RequestedDevice` | `"cpu"`, `"cuda"`, `"auto"` | User-requested device |
| `ResolvedDevice` | `"cpu"`, `"cuda"` | Concrete device after resolution |
| `RequestedPrecision` | `"fp32"`, `"fp16"`, `"bf16"`, `"auto"` | User-requested precision |
| `ResolvedPrecision` | `"fp32"`, `"fp16"`, `"bf16"` | Concrete precision after resolution |

### Device Functions

| Function | Description |
|----------|-------------|
| `resolve_device(requested)` | Resolve "auto" to concrete device via CUDA check |
| `resolve_precision(requested, device)` | Resolve precision based on device capabilities |
| `recommended_batch_size(current, device)` | Recommend batch size (bump small batches on CUDA) |

### Testing Device Selection

Use `FakeTorchModule` from `platform_ml.testing` to test device paths without GPU hardware:

```python
from platform_ml import torch_types, resolve_device
from platform_ml.testing import FakeTorchModule
from platform_ml.torch_types import _TorchModuleProtocol

# Test CUDA available path
fake_torch = FakeTorchModule(cuda_available=True)
def _fake_import() -> _TorchModuleProtocol:
    return fake_torch
torch_types._import_torch = _fake_import
assert resolve_device("auto") == "cuda"

# Test CPU fallback path
fake_torch = FakeTorchModule(cuda_available=False)
torch_types._import_torch = _fake_import
assert resolve_device("auto") == "cpu"
```

## API Reference

### Artifact Store

| Type | Description |
|------|-------------|
| `ArtifactStore` | Remote artifact storage client |
| `ArtifactStoreError` | Store operation error |

### Manifest

| Type | Description |
|------|-------------|
| `ModelManifestV2` | Model manifest schema |
| `TrainingRunMetadata` | Training run metadata |
| `from_json_manifest_v2` | Parse manifest from JSON |
| `from_path_manifest_v2` | Parse manifest from file |
| `MANIFEST_SCHEMA_VERSION` | Current schema version |

### Tarball

| Function | Description |
|----------|-------------|
| `create_tarball` | Create gzip tarball from directory |
| `extract_tarball` | Extract tarball with security checks |
| `TarballError` | Tarball operation error |

### Wandb

| Type | Description |
|------|-------------|
| `WandbPublisher` | W&B experiment publisher |
| `WandbUnavailableError` | W&B not installed error |
| `WandbRunConfig` | Run configuration |
| `WandbPublisherConfig` | Publisher configuration |
| `WandbStepMetrics` | Step metrics TypedDict |
| `WandbEpochMetrics` | Epoch metrics TypedDict |
| `WandbFinalMetrics` | Final metrics TypedDict |
| `WandbTableRow` | Table row type |
| `WandbInitResult` | Init result TypedDict |

### Device Selection

| Type | Description |
|------|-------------|
| `RequestedDevice` | Device requested by user (`"cpu"`, `"cuda"`, `"auto"`) |
| `ResolvedDevice` | Concrete device after resolution (`"cpu"`, `"cuda"`) |
| `RequestedPrecision` | Precision requested by user |
| `ResolvedPrecision` | Concrete precision after resolution |
| `resolve_device` | Resolve device with CUDA auto-detection |
| `resolve_precision` | Resolve precision based on device |
| `recommended_batch_size` | Device-aware batch size recommendation |

## Development

```bash
make lint    # Run ruff linter
make test    # Run pytest with coverage
make check   # Run both lint and test
```

## Requirements

- Python 3.12+
- platform-core (for DataBankClient, JSON utilities)
- wandb (optional, for experiment tracking)
- 100% test coverage enforced

## Torch Protocols and Test Fakes

Strict, framework-agnostic interfaces for PyTorch-like functionality, with public fakes for tests. This enables deterministic, fully-typed ML backends without importing `torch` when unavailable.

### Strict Protocols (no Any)

Provided in `platform_ml.torch_types`:
- `_TorchModuleProtocol`, `_CudaModuleProtocol` — minimal surface for device checks, seeding, tensor creation, save/load
- `TensorProtocol`, `DeviceProtocol`, `DTypeProtocol` — tensor/device/dtype primitives used by backends

Dynamic import pattern with immediate Protocol annotation (no Any leaks):

```python
from platform_ml.torch_types import _TorchModuleProtocol, _import_torch

torch: _TorchModuleProtocol = _import_torch()  # returns real torch or compat module
torch.set_num_threads(1)
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.zeros(2, 3, device=device)
```

All attributes used from `torch` must exist on the Protocol. When you need new capabilities, extend the Protocols (and the fakes) rather than using untyped access.

### Fake Torch for Tests

Provided in `platform_ml.testing`:
- `FakeTorchModule`, `FakeCudaModule`, `FakeDevice`, `FakeDType`, `FakeNoGradContext`
- `FakeTensor` supports methods used by typical training loops (`tolist`, `detach`, `cpu`, `clone`, `to`, `backward`, `numpy`, `argmax`)

Usage example in tests:

```python
from platform_ml.testing import FakeTorchModule

torch = FakeTorchModule()  # satisfies _TorchModuleProtocol
torch.set_num_threads(1)
torch.manual_seed(123)

with torch.no_grad():
    t = torch.zeros(2, 3)
    assert t.numpy().shape == (2, 3)
```

### Determinism Guidance

- Seed at component prep with a single source of truth (e.g., `config["random_state"]`).
- Use `set_num_threads(1)` for reproducible CPU behavior where appropriate.
- Backend-specific deterministic toggles (e.g., CUDA) should live in the training library; the Protocol remains minimal and portable.
