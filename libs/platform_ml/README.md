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
root = store.download_artifact(
    file_id="model-v1.tar.gz", dest_dir=Path("./downloaded"), request_id="req-456"
)
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
print(manifest["model_type"])  # Literal["resnet18", "gpt2"]
print(manifest["file_id"])  # Remote artifact file ID
print(manifest["file_sha256"])  # SHA256 hash for integrity
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
    publisher.log_config(
        {
            "model_family": "gpt2",
            "batch_size": 8,
            "learning_rate": 0.001,
        }
    )

# Log per-step metrics during training
if publisher:
    publisher.log_step(
        {
            "global_step": step,
            "train_loss": loss,
            "train_ppl": ppl,
            "grad_norm": grad_norm,
        }
    )

# Log epoch-end validation metrics
if publisher:
    publisher.log_epoch(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "best_val_loss": best_val_loss,
        }
    )

# Log final test metrics
if publisher:
    publisher.log_final(
        {
            "test_loss": test_loss,
            "test_ppl": test_ppl,
            "early_stopped": early_stopped,
        }
    )

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
device: ResolvedDevice = resolve_device("cpu")  # passthrough

# Resolve precision based on device
precision: ResolvedPrecision = resolve_precision("auto", device)
# "auto" on CUDA -> "fp16", "auto" on CPU -> "fp32"
precision: ResolvedPrecision = resolve_precision("fp16", "cuda")  # OK
# resolve_precision("fp16", "cpu")  # RuntimeError: fp16 not supported on CPU

# Recommended batch size (bumps small batches on CUDA)
batch_size = recommended_batch_size(4, "cuda")  # 8
batch_size = recommended_batch_size(4, "cpu")  # 4
batch_size = recommended_batch_size(16, "cuda")  # 16 (preserved)
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

## Feature Importance Explainers

Pluggable feature importance explainers for ML models. Provides three methods with different trade-offs between accuracy, speed, and model requirements.

### SHAP Integration

The `shap` library has poor typing (`Any` everywhere), which violates our strict MyPy standards. We provide two approaches:

1. **Custom Explainers** (below): Fully typed alternatives that don't depend on SHAP
2. **ShapTreeWrapper**: Anti-corruption layer for SHAP TreeExplainer with strict typing

For tree-based models (XGBoost, LightGBM, sklearn GradientBoosting), use `ShapTreeWrapper` to get SHAP's fast TreeExplainer with full type safety. See the ShapTreeWrapper section below.

### Available Explainers

| Explainer | Model Requirements | Speed | Accuracy |
|-----------|-------------------|-------|----------|
| `PermutationExplainer` | Any predictor | Medium | Good |
| `GradientExplainer` | Differentiable (neural nets) | Fast | Moderate |
| `IntegratedGradientsExplainer` | Differentiable (neural nets) | Slow | High |

### Quick Start

```python
from platform_ml.explainers import (
    PermutationExplainer,
    PermutationConfig,
    create_permutation_explainer,
    FeatureImportanceScore,
)

# Configure explainer
config: PermutationConfig = {"n_repeats": 10, "random_state": 42}
explainer = create_permutation_explainer(config)

# Compute feature importance
importance: list[FeatureImportanceScore] = explainer.compute_importance(
    model=model,  # Any model with predict_proba()
    x_data=x_test,  # NDArray[np.float64]
    feature_names=["age", "income", "score"],
    target_class=1,  # Class index for importance
)

# Results are ranked by importance
for score in importance:
    print(f"{score['rank']}. {score['name']}: {score['importance']:.4f}")
```

### PermutationExplainer

Model-agnostic method that measures prediction change when features are shuffled. Works with any model implementing `PredictorProtocol`.

```python
from platform_ml.explainers import (
    PermutationExplainer,
    PermutationConfig,
    PERMUTATION_CAPABILITIES,
)

config: PermutationConfig = {
    "n_repeats": 10,  # Number of shuffle repeats
    "random_state": 42,  # For reproducibility
}
explainer = PermutationExplainer(config)

# Check capabilities
caps = explainer.capabilities()
assert caps["requires_gradients"] is False  # Works with any model
assert caps["computational_cost"] == "medium"
```

### GradientExplainer

Fast gradient-based attribution for neural networks. Requires models implementing `GradientModelProtocol`.

```python
from platform_ml.explainers import (
    GradientExplainer,
    GradientConfig,
    GRADIENT_CAPABILITIES,
)

config: GradientConfig = {
    "multiply_by_input": True,  # Gradient * input attribution
    "absolute_value": True,  # Use absolute gradients
}
explainer = GradientExplainer(config)

# Check capabilities
caps = explainer.capabilities()
assert caps["requires_gradients"] is True  # Neural networks only
assert caps["computational_cost"] == "low"
```

### IntegratedGradientsExplainer

Accurate path-integrated gradients (Sundararajan et al., 2017). More accurate than simple gradients but computationally expensive.

```python
from platform_ml.explainers import (
    IntegratedGradientsExplainer,
    IntegratedGradientsConfig,
    INTEGRATED_GRADIENTS_CAPABILITIES,
)

config: IntegratedGradientsConfig = {
    "n_steps": 50,  # Integration steps (more = accurate)
    "baseline_mode": "zeros",  # "zeros" or "mean"
}
explainer = IntegratedGradientsExplainer(config)

# Check capabilities
caps = explainer.capabilities()
assert caps["requires_gradients"] is True
assert caps["requires_background_data"] is True
assert caps["computational_cost"] == "high"
```

### Model Protocols

Models must implement the appropriate protocol:

```python
from platform_ml.explainers import PredictorProtocol, GradientModelProtocol


# For PermutationExplainer - any model with predict_proba
class MyClassifier:
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # Return shape (n_samples, n_classes)
        ...


# For GradientExplainer/IntegratedGradientsExplainer - neural networks
class MyNeuralNet:
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def compute_gradients(self, x: NDArray[np.float64], target_class: int) -> NDArray[np.float64]:
        # Return gradients with shape (n_samples, n_features)
        ...
```

### Explainer Types

| Type | Description |
|------|-------------|
| `ExplainerName` | `Literal["permutation", "gradient", "integrated_gradients"]` |
| `ComputationalCost` | `Literal["low", "medium", "high"]` |
| `ExplainerCapabilities` | TypedDict with requirements and cost |
| `FeatureImportanceScore` | TypedDict with name, importance, rank |
| `PermutationConfig` | Config for PermutationExplainer |
| `GradientConfig` | Config for GradientExplainer |
| `IntegratedGradientsConfig` | Config for IntegratedGradientsExplainer |
| `PredictorProtocol` | Protocol for models with predict_proba |
| `GradientModelProtocol` | Protocol for models with gradients |
| `FeatureExplainer` | Protocol for all explainers |

### Factory Functions

```python
from platform_ml.explainers import (
    create_permutation_explainer,
    create_gradient_explainer,
    create_integrated_gradients_explainer,
)

# Each returns the corresponding explainer type
explainer = create_permutation_explainer(config)
explainer = create_gradient_explainer(config)
explainer = create_integrated_gradients_explainer(config)
```

### ShapTreeWrapper

Type-safe wrapper for SHAP TreeExplainer. Provides local (per-sample) Shapley value explanations for tree-based models while maintaining strict typing.

```python
from platform_ml import ShapTreeWrapper, LocalExplanation, TreeModelProtocol
import numpy as np
from numpy.typing import NDArray

# Any tree-based model with predict_proba works
# (XGBoost, LightGBM, sklearn GradientBoosting, RandomForest)
model = train_gradient_boosting_classifier(x_train, y_train)

# Create wrapper (dynamically imports shap internally)
wrapper = ShapTreeWrapper(model)

# Compute local explanations
x_test: NDArray[np.float64] = get_test_data()
feature_names = ["age", "income", "credit_score"]

explanations: list[LocalExplanation] = wrapper.explain_local(x_test, feature_names)

# Each explanation contains:
for expl in explanations:
    print(f"Base value: {expl['base_value']}")  # Model's expected output
    print(f"Features: {expl['feature_names']}")  # Feature names
    print(f"SHAP values: {expl['values']}")  # Per-feature contributions
```

#### TreeModelProtocol

Models must implement `predict_proba` returning class probabilities:

```python
from platform_ml import TreeModelProtocol


class MyTreeModel:
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Probabilities with shape (n_samples, n_classes).
        """
        ...
```

#### LocalExplanation TypedDict

```python
from platform_ml import LocalExplanation

# TypedDict structure
explanation: LocalExplanation = {
    "base_value": 0.5,  # float: model's expected output
    "feature_names": ["a", "b", "c"],  # list[str]: feature names
    "values": [0.1, -0.2, 0.15],  # list[float]: SHAP values per feature
}
```

#### Anti-Corruption Pattern

ShapTreeWrapper uses dynamic import to isolate the untyped `shap` library:

- No top-level `import shap` (avoids untyped module pollution)
- Protocol-typed assignment: `TreeExplainerConstructor` Protocol overrides `Any` from `getattr`
- TypeGuard for safe type narrowing of `expected_value` (can be float or ndarray)
- All outputs converted to strict Python types (`float`, `list[float]`)

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

### Explainers

| Type | Description |
|------|-------------|
| `PermutationExplainer` | Model-agnostic permutation importance |
| `GradientExplainer` | Gradient-based feature attribution |
| `IntegratedGradientsExplainer` | Path-integrated gradients |
| `ShapTreeWrapper` | Type-safe wrapper for SHAP TreeExplainer |
| `PermutationConfig` | Permutation explainer config |
| `GradientConfig` | Gradient explainer config |
| `IntegratedGradientsConfig` | Integrated gradients config |
| `FeatureImportanceScore` | Importance result TypedDict |
| `LocalExplanation` | SHAP local explanation TypedDict |
| `ExplainerCapabilities` | Explainer requirements TypedDict |
| `PredictorProtocol` | Protocol for any predictor |
| `GradientModelProtocol` | Protocol for gradient-capable models |
| `TreeModelProtocol` | Protocol for tree-based models |
| `FeatureExplainer` | Protocol for all explainers |
| `create_permutation_explainer` | Factory for PermutationExplainer |
| `create_gradient_explainer` | Factory for GradientExplainer |
| `create_integrated_gradients_explainer` | Factory for IntegratedGradientsExplainer |
| `PERMUTATION_CAPABILITIES` | Permutation explainer capabilities |
| `GRADIENT_CAPABILITIES` | Gradient explainer capabilities |
| `INTEGRATED_GRADIENTS_CAPABILITIES` | Integrated gradients capabilities |

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
