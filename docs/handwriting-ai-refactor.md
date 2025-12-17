# RFC: Generalizable Vision Training Harness and Handwriting-AI Refactor

## Status
- Target: merge after alignment with owners of `handwriting-ai`, `Model-Trainer`, `covenant-radar-api`, and `libs/platform_ml`
- Scope: services/handwriting-ai, libs/platform_ml (additions only), new libs/platform_vision (new package)
- Non-goals: back-compat layers, best-effort behavior, partial/temporary shims

## Goals
- Extract a reusable, strictly-typed image training harness leveraged by handwriting-ai and future CV tasks.
- Reduce drift by centralizing device/precision, runtime/threading, training loops, manifests, artifacts, and inference patterns.
- Enforce strict typing and boundary validation everywhere (no Any, no casts, no stubs, no pyi, no type: ignore, no noqa; exceptions propagate).
- Achieve and sustain 100% statement and branch coverage for all new modules and refactored surfaces.
- Keep services thin: domain-specific wiring and APIs; move mechanics into libs with stable Protocols and TypedDict configs.

## Principles (Hard Requirements)
- Strict typing only:
  - No `Any`, no `cast`, no `type: ignore`, no `.pyi`, no `noqa`.
  - Use `TypedDict`, `Protocol`, type aliases. No dataclasses in `src/`.
- Parse/validate at edges:
  - JSON: `json.loads` → assign to recursive TypeAlias → validate via internal `_decode*`/`_load_json*`.
  - TOML: convert to concrete `TypedDict` before use; no `typing.Any` or `type_checking` gates.
  - ASGI boundaries: define minimal `Protocol` (e.g., `async def body() -> bytes`), avoid `dict[str, Any]` scopes.
- Dynamic import pattern:
  - `mod = __import__("package.module")`; `sym: TargetProtocol = getattr(mod, "Symbol")` (annotate to avoid Any).
- Dependency injection via hooks:
  - Services: `_test_hooks.py` with functions set at startup to real implementations.
  - Libs: `testing.py` expose public test utilities; tests override hooks directly without conditionals.
- Fail loud and early:
  - No best-effort fallbacks; we name functions and document failure modes; exceptions propagate.
  - No internal try/except for “softening” errors (only at process edge for translation/logging).
- DRY and centralized:
  - Shared logic belongs in `libs/` (platform_ml, platform_vision). Services import; no duplication.

## High-Level Architecture
- libs/platform_ml (existing)
  - Keep: `device_selector.py`, `torch_types.py`, `manifest.py` (v2), `artifact_store.py`, `testing.py`, `wandb_*`.
- libs/platform_vision (new)
  - `config.py`: `VisionTrainConfig` (TypedDict), validation helpers, JSON/TOML loaders; immutable-like usage.
  - `datasets/`: `ImageClassificationDataset` Protocols, adapters (MNIST adapter, CIFAR adapter, local folder adapter).
  - `preprocess/`: PIL→tensor, normalization, size/channel policies; deterministic hashing of preprocess spec.
  - `augment/`: affine/noise/blur/morph knobs with deterministic RNG (seeded) and pure functions.
  - `models/`: model registry via Protocol-typed builders (e.g., `resnet18`, `mobilenet_v3_small`), no Any.
  - `train/`: supervised CE loop, early stopping, schedulers, mixed-precision, resource calibration, thread config, memory guard; progress callbacks.
  - `metrics/`: accuracy/loss surfaces with typed results.
  - `inference/`: CPU-first engine, optional TTA, manifest-aware preprocess.
- services/handwriting-ai (refactored)
  - Thin task plugin for digits: dataset builder, model spec (`arch=resnet18`, `n_classes=10`, `in_channels=1`), inference wiring and API routes/jobs.
  - Training/inference implementations delegate to platform_vision harness.

## Surface Designs (Typed)

### Config: VisionTrainConfig
Minimal, immutable-by-convention configuration for a single training run. All fields are required by the time training starts. JSON/TOML loaders must validate and return concrete `VisionTrainConfig`.

```python
from typing import Literal, TypedDict
from pathlib import Path
from platform_ml.device_selector import RequestedDevice, RequestedPrecision

class VisionTrainConfig(TypedDict):
    data_root: Path
    out_dir: Path
    model_id: str
    arch: Literal["resnet18", "mobilenet_v3_small"]
    in_channels: int
    n_classes: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    device: RequestedDevice
    precision: RequestedPrecision
    optim: Literal["adamw", "sgd"]
    scheduler: Literal["none", "step"]
    step_size: int
    gamma: float
    min_lr: float
    patience: int
    min_delta: float
    threads: int
    # Augment
    augment: bool
    aug_rotate: float
    aug_translate: float
    noise_prob: float
    noise_salt_vs_pepper: float
    dots_prob: float
    dots_count: int
    dots_size_px: int
    blur_sigma: float
    morph: Literal["none", "erode", "dilate"]
    morph_kernel_px: int
    # Progress/Calibration
    progress_every_epochs: int
    progress_every_batches: int
    calibrate: bool
    calibration_samples: int
    force_calibration: bool
    memory_guard: bool
```

Rules:
- Loaders perform complete type validation before returning; use `_decode_*` helpers.
- No defaults at use sites. Defaults only exist in a `default_config()` constructor returning `VisionTrainConfig`.

### Dataset Protocols

```python
class ImageClassificationDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[PILImage, int]: ...
```

Adapters:
- `MNISTRawDataset` (migrated from handwriting-ai), `CIFAR10Dataset`, `FolderDataset(root/cls/*.png)`.

### Model Registry
Dynamic import with Protocol-typed assignment; no Any.

```python
class ResNet18Builder(Protocol):
    def __call__(self, *, weights: None, num_classes: int) -> torch.nn.Module: ...

class ModelsModule(Protocol):
    @property
    def resnet18(self) -> ResNet18Builder: ...

def build_model(arch: str, n_classes: int, in_channels: int) -> TrainableModel:
    tv_raw = __import__("torchvision.models")
    tv: ModelsModule = tv_raw
    if arch == "resnet18":
        fn: ResNet18Builder = tv.resnet18
        m = fn(weights=None, num_classes=n_classes)
        # Replace stem for in_channels (1-channel MNIST)
        # Assign via _modules to keep state_dict registration strict
        # No try/except: invalid state raises
        ...
    else:
        raise RuntimeError("unsupported arch")
    return _wrap_torch_model(m)
```

### Training Runtime and Threads
- Reuse `platform_ml.torch_types.configure_torch_threads`.
- Add `train.runtime.build_effective_config(cfg)` in platform_vision to compute loader threads, prefetch, batch size caps based on resource detection hooks (migrated from handwriting-ai runtime.py) with no best-effort.

### Training Loop
- Migrate `train_epoch`, `evaluate`, mixed precision behavior, and memory guard from handwriting-ai to `platform_vision.train.loops`.
- Keep precision logic: `fp16` uses scaler on CUDA; `bf16` / `fp32` standard.
- Thread application via `torch.set_num_threads` happens once per effective config.
- Progress callback typed `Protocol` to emit batch/epoch snapshots.
- No internal try/except; invalid device/precision/config yields immediate errors.

### Inference Engine
- Generic CPU-first engine with optional TTA, moved to `platform_vision.inference`.
- Manifest v2 from `platform_ml.manifest`. Strict typed decoding; no partial loads.
- Service-specific wrapper (handwriting-ai) remains to bind settings and routes.

### Artifacts and Manifests
- Standardize on `ModelManifestV2` in `platform_ml.manifest`.
- Training returns typed `TrainingResult` with `state_dict` and `metadata` to create the manifest (service or orchestrator writes artifacts via `artifact_store`).

## Migration Plan (Phased)
1) Freeze types and configs
   - Define `VisionTrainConfig`, dataset/model Protocols, training result/metadata types.
   - Align owners and accept no-backcompat stance.

2) Extract generic components
   - Move runtime/thread calibration, loops, safety, memory diagnostics into `libs/platform_vision/train`.
   - Move augment + preprocess into `libs/platform_vision/augment` and `preprocess`.

3) Wire artifacts and manifests
   - Ensure training returns `TrainingResult`; writer composes `ModelManifestV2` with required fields.

4) Refactor handwriting-ai
   - Replace internal imports with `platform_vision.*`; keep digits dataset adapter, API routes, jobs.
   - Inference engine becomes a thin wrapper configuring TTA and model dir.

5) Tests (must reach 100% statements/branches)
   - Unit: config decoders, augment functions (all branches), model builder (error branches), runtime calibration, training loop precision paths, memory guard, manifest decode.
   - Integration: end-to-end training on tiny synthetic dataset (CPU), artifact write + manifest read, inference execution with TTA.
   - No mocks: tests call real code paths. Use hooks to control torch/device availability and RNG seeding.

6) Remove legacy code
   - Delete duplicated logic from handwriting-ai once green.

## Testing Strategy
- Hooks: Production sets real impls at startup; tests set deterministic fakes (no conditionals; just call hook).
- Determinism: fixed seeds for RNG; assert exact numeric invariants where practical; otherwise assert tight tolerances.
- Coverage: enforce `fail_under = 100` for new libs; add branch-specific tests (e.g., CPU vs CUDA, fp16 vs fp32).
- Performance: use minimal synthetic datasets to keep CI fast; ensure thread calibration branches run under constrained limits.

## Risks and Mitigations
- Risk: Hidden MNIST assumptions (1-channel tensors) leak into generic layers.
  - Mitigation: Explicit `in_channels` in model build; channel checks in preprocess; tests for 1/3-channel inputs.
- Risk: Divergence between services and libs over time.
  - Mitigation: Single source of truth in libs; services prohibited from re-implementing loops/device logic.
- Risk: CI flakiness due to device/precision detection.
  - Mitigation: All device/precision checks go through hooks; tests pin to CPU and override availability deterministically.

## Rollout
- Week 1: Land platform_vision scaffolding and types; unit tests for config/augment/manifest.
- Week 2: Migrate loops/runtime/memory guard; integration tests pass on CPU.
- Week 3: Refactor handwriting-ai to use platform_vision; delete duplicates; all tests 100%.
- Week 4: Optional: Add `services/image-trainer` orchestrator if multiple tasks need a shared API.

## Open Questions
- Package name: `libs/platform_vision` vs `libs/platform_ml/vision` as a subpackage?
- Model registry scope: start with `resnet18` and add on demand, or include `mobilenet` now?
- Do we want a generic folder dataset with class-per-dir semantics immediately?

