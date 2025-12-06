# Model-Trainer Architecture Refactor

## Overview

This document specifies the complete architectural refactor for `services/model-trainer` to establish a production-grade ML research platform. The refactor eliminates technical debt, removes boilerplate duplication, and creates extensible abstractions for supporting many model architectures.

**Guiding Principles:**
- Strict typing: No `Any`, `cast`, `type: ignore`, `.pyi`, `noqa`, or stubs
- Immutable data: TypedDict for all configs and results (no dataclasses)
- DRY: Eliminate ~800 lines of duplicated backend boilerplate
- Explicit failures: No try/except recovery, no best-effort, no fallbacks
- 100% test coverage: Statements and branches
- Protocol-based: Dynamic imports use Protocol type annotations at assignment

---

## Phase 1: Foundation - Config Classes to TypedDict ✅ COMPLETED

### 1.1 Replace Manual Config Classes

**Problem (SOLVED):**
`core/contracts/model.py` had 6 config classes with ~200 lines of manual `__init__` boilerplate.

**Files to modify:**
- `core/contracts/model.py`

**Before:**
```python
class ScoreConfig:
    text: str | None
    path: str | None
    detail_level: Literal["summary", "per_char"]
    top_k: int | None
    seed: int | None

    def __init__(
        self: ScoreConfig,
        *,
        text: str | None,
        path: str | None,
        detail_level: Literal["summary", "per_char"],
        top_k: int | None,
        seed: int | None,
    ) -> None:
        self.text = text
        self.path = path
        self.detail_level = detail_level
        self.top_k = top_k
        self.seed = seed
```

**After:**
```python
class ScoreConfig(TypedDict):
    text: str | None
    path: str | None
    detail_level: Literal["summary", "per_char"]
    top_k: int | None
    seed: int | None
```

**Classes to convert (6 total):**
1. `ScoreConfig` - 33 lines → 6 lines
2. `GenerateConfig` - 38 lines → 12 lines
3. `ModelTrainConfig` - 30 lines → 9 lines
4. `TrainOutcome` - 21 lines → 7 lines
5. `EvalOutcome` - 8 lines → 4 lines
6. `ScoreOutcome` - 24 lines → 7 lines
7. `GenerateOutcome` - 17 lines → 5 lines
8. `ModelArtifact` - 5 lines → 3 lines

**Reduction achieved:** ~175 lines

### 1.2 Add Strict Validators

**File created:** `core/contracts/validators.py`

```python
from __future__ import annotations

from typing import Literal, Sequence

from model_trainer.core.contracts.model import (
    GenerateConfig,
    ModelTrainConfig,
    ScoreConfig,
)


def validate_score_config(cfg: ScoreConfig) -> None:
    """Validate ScoreConfig. Raises ValueError on invalid input."""
    if cfg["text"] is None and cfg["path"] is None:
        raise ValueError("ScoreConfig requires either text or path")
    if cfg["text"] is not None and cfg["path"] is not None:
        raise ValueError("ScoreConfig cannot have both text and path")
    if cfg["top_k"] is not None and cfg["top_k"] < 1:
        raise ValueError("top_k must be >= 1")


def validate_generate_config(cfg: GenerateConfig) -> None:
    """Validate GenerateConfig. Raises ValueError on invalid input."""
    if cfg["prompt_text"] is None and cfg["prompt_path"] is None:
        raise ValueError("GenerateConfig requires either prompt_text or prompt_path")
    if cfg["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if cfg["temperature"] <= 0.0:
        raise ValueError("temperature must be > 0")
    if cfg["top_k"] < 1:
        raise ValueError("top_k must be >= 1")
    if not (0.0 < cfg["top_p"] <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    if cfg["num_return_sequences"] < 1:
        raise ValueError("num_return_sequences must be >= 1")


def validate_model_train_config(cfg: ModelTrainConfig) -> None:
    """Validate ModelTrainConfig. Raises ValueError on invalid input."""
    if cfg["max_seq_len"] < 1:
        raise ValueError("max_seq_len must be >= 1")
    if cfg["num_epochs"] < 1:
        raise ValueError("num_epochs must be >= 1")
    if cfg["batch_size"] < 1:
        raise ValueError("batch_size must be >= 1")
    if cfg["learning_rate"] <= 0.0:
        raise ValueError("learning_rate must be > 0")
```

---

## Phase 2: Unified PreparedModel ✅ COMPLETED

### 2.1 Replace Opaque PreparedModel Protocol

**Problem (SOLVED):**
`GPT2Prepared` and `CharLSTMPrepared` were identical 6-field classes defined separately.

**Files modified:**
- `core/contracts/model.py` - Added `PreparedLMModel` class
- `core/services/model/backends/gpt2/types.py` - DELETED
- `core/services/model/backends/char_lstm/types.py` - DELETED

**Unified type in `core/contracts/model.py`:**
```python
class PreparedLMModel(TypedDict):
    """Prepared language model ready for training or inference."""
    model: LMModelProto
    tokenizer_id: str
    eos_id: int
    pad_id: int
    max_seq_len: int
    tok_for_dataset: Encoder


class LMModelProto(Protocol):
    """Protocol for language model forward pass."""
    def __call__(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor: ...
    def parameters(self) -> Iterator[Parameter]: ...
    def train(self, mode: bool = True) -> Self: ...
    def eval(self) -> Self: ...
    def to(self, device: str | device) -> Self: ...
    def state_dict(self) -> dict[str, Tensor]: ...
    def load_state_dict(self, state_dict: dict[str, Tensor]) -> None: ...


class Encoder(Protocol):
    """Protocol for tokenizer encoding."""
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
```

**Reduction achieved:** ~50 lines (2 duplicate files removed)

---

## Phase 3: Extract Common Training Loop ✅ COMPLETED

### 3.1 Create Base Trainer

**Problem (SOLVED):**
`gpt2/train.py` and `char_lstm/train.py` had ~300 lines of nearly identical code:
- `_gather_lib_versions()` - identical
- `_write_manifest()` - identical structure
- `_build_train_loader()` - identical pattern
- `_train_one_epoch()` - identical loop structure
- `_run_training_loop()` - identical

**File created:** `core/services/training/base_trainer.py`

Implementation includes:
- `TrainResult` class for training outcomes
- `BaseTrainer` class with unified training loop
- `_gather_lib_versions()` helper
- `_maybe_git_commit()` for reproducibility
- `_clip_grad_norm()` with Protocol typing

**Actual implementation:**

```python
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, TypedDict

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from model_trainer.core.contracts.model import PreparedLMModel, TrainOutcome


class EpochMetrics(TypedDict):
    epoch: int
    loss: float
    perplexity: float
    steps: int


class TrainingCallbacks(TypedDict):
    heartbeat: Callable[[float], None]
    cancelled: Callable[[], bool]
    progress: Callable[[int, int, float], None] | None


class BaseTrainer:
    """Base trainer with common training loop logic."""

    def train_epochs(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        train_loader: DataLoader[tuple[Tensor, Tensor]],
        num_epochs: int,
        device: str,
        callbacks: TrainingCallbacks,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Iterator[EpochMetrics]:
        """Yield metrics after each epoch. Caller handles early stopping."""
        model.train()
        model.to(device)

        for epoch in range(num_epochs):
            if callbacks["cancelled"]():
                return

            epoch_loss = 0.0
            steps = 0

            for batch_idx, (input_ids, targets) in enumerate(train_loader):
                if callbacks["cancelled"]():
                    return

                input_ids = input_ids.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(input_ids)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                steps += 1

                callbacks["heartbeat"](time.time())
                if callbacks["progress"] is not None:
                    callbacks["progress"](steps, epoch, loss.item())

            avg_loss = epoch_loss / max(steps, 1)
            perplexity = math.exp(min(avg_loss, 100.0))

            yield EpochMetrics(
                epoch=epoch,
                loss=avg_loss,
                perplexity=perplexity,
                steps=steps,
            )

    def build_causal_lm_loader(
        self,
        *,
        texts: list[str],
        encoder: Encoder,
        max_seq_len: int,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader[tuple[Tensor, Tensor]]:
        """Build DataLoader for causal language modeling."""
        dataset = CausalLMDataset(texts, encoder, max_seq_len)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
```

### 3.2 Refactor Backend Train Functions

**Files refactored:**
- `core/services/model/backends/gpt2/train.py` - Now thin wrapper (~50 lines)
- `core/services/model/backends/char_lstm/train.py` - Now thin wrapper (~50 lines)

**Actual implementation (both backends follow this pattern):**
```python
from __future__ import annotations

from model_trainer.core.training.base_trainer import BaseTrainer, TrainingCallbacks


def train_gpt2(
    prepared: PreparedLMModel,
    cfg: GPT2TrainConfig,
    callbacks: TrainingCallbacks,
) -> TrainOutcome:
    """Train GPT-2 model."""
    trainer = BaseTrainer()

    train_loader = trainer.build_causal_lm_loader(
        texts=load_corpus(cfg["corpus_path"]),
        encoder=prepared["tok_for_dataset"],
        max_seq_len=cfg["max_seq_len"],
        batch_size=cfg["batch_size"],
    )

    optimizer = AdamW(prepared["model"].parameters(), lr=cfg["learning_rate"])

    final_metrics: EpochMetrics | None = None
    for metrics in trainer.train_epochs(
        model=prepared["model"],
        optimizer=optimizer,
        train_loader=train_loader,
        num_epochs=cfg["num_epochs"],
        device=cfg["device"],
        callbacks=callbacks,
        loss_fn=cross_entropy_loss,
    ):
        final_metrics = metrics

    if final_metrics is None:
        raise TrainingCancelledError("Training cancelled before first epoch")

    return TrainOutcome(
        loss=final_metrics["loss"],
        perplexity=final_metrics["perplexity"],
        steps=final_metrics["steps"],
        out_dir=cfg["out_dir"],
        cancelled=False,
    )
```

**Reduction achieved:** ~400 lines across both backends

---

## Phase 4: Backend Adapter Elimination ✅ COMPLETED

### 4.1 Problem Solved

**Previous Problem:**
`GPT2BackendImpl` and `CharLSTMBackendImpl` were ~147 lines each with identical structure:
- Type guards (`isinstance` checks)
- Config conversion boilerplate
- Outcome wrapping

### 4.2 Generic Backend Adapter

**New file:** `core/services/model/backend_adapter.py`

```python
from __future__ import annotations

from typing import Generic, TypeVar

from model_trainer.core.contracts.model import (
    EvalOutcome,
    GenerateConfig,
    GenerateOutcome,
    ModelArtifact,
    ModelBackend,
    ModelTrainConfig,
    PreparedLMModel,
    ScoreConfig,
    ScoreOutcome,
    TrainOutcome,
)
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.tokenizer import TokenizerHandle

TBackendConfig = TypeVar("TBackendConfig")


class BackendFunctions(TypedDict, Generic[TBackendConfig]):
    """Functions that define a backend implementation."""
    name: str
    convert_config: Callable[[ModelTrainConfig], TBackendConfig]
    prepare: Callable[[TBackendConfig, Settings, TokenizerHandle], PreparedLMModel]
    train: Callable[[PreparedLMModel, TBackendConfig, TrainingCallbacks], TrainOutcome]
    evaluate: Callable[[str, TBackendConfig, Settings], EvalOutcome]
    score: Callable[[PreparedLMModel, ScoreConfig, Settings], ScoreOutcome]
    generate: Callable[[PreparedLMModel, GenerateConfig, Settings], GenerateOutcome]
    save: Callable[[PreparedLMModel, str], ModelArtifact]
    load: Callable[[str, Settings, TokenizerHandle], PreparedLMModel]


def create_backend(funcs: BackendFunctions[TBackendConfig]) -> ModelBackend:
    """Create a ModelBackend from a set of implementation functions."""

    class _Backend:
        def name(self) -> str:
            return funcs["name"]

        def prepare(
            self,
            cfg: ModelTrainConfig,
            settings: Settings,
            *,
            tokenizer: TokenizerHandle,
        ) -> PreparedLMModel:
            backend_cfg = funcs["convert_config"](cfg)
            return funcs["prepare"](backend_cfg, settings, tokenizer)

        def train(
            self,
            cfg: ModelTrainConfig,
            settings: Settings,
            *,
            run_id: str,
            heartbeat: Callable[[float], None],
            cancelled: Callable[[], bool],
            prepared: PreparedLMModel,
            progress: Callable[[int, int, float], None] | None = None,
        ) -> TrainOutcome:
            backend_cfg = funcs["convert_config"](cfg)
            callbacks = TrainingCallbacks(
                heartbeat=heartbeat,
                cancelled=cancelled,
                progress=progress,
            )
            return funcs["train"](prepared, backend_cfg, callbacks)

        def evaluate(
            self,
            *,
            run_id: str,
            cfg: ModelTrainConfig,
            settings: Settings,
        ) -> EvalOutcome:
            backend_cfg = funcs["convert_config"](cfg)
            return funcs["evaluate"](run_id, backend_cfg, settings)

        def score(
            self,
            *,
            prepared: PreparedLMModel,
            cfg: ScoreConfig,
            settings: Settings,
        ) -> ScoreOutcome:
            return funcs["score"](prepared, cfg, settings)

        def generate(
            self,
            *,
            prepared: PreparedLMModel,
            cfg: GenerateConfig,
            settings: Settings,
        ) -> GenerateOutcome:
            return funcs["generate"](prepared, cfg, settings)

        def save(self, prepared: PreparedLMModel, out_dir: str) -> ModelArtifact:
            return funcs["save"](prepared, out_dir)

        def load(
            self,
            artifact_path: str,
            settings: Settings,
            *,
            tokenizer: TokenizerHandle,
        ) -> PreparedLMModel:
            return funcs["load"](artifact_path, settings, tokenizer)

    return _Backend()
```

### 4.3 Backend Registration

**Refactored GPT-2 backend definition (~30 lines total):**

**File:** `core/services/model/backends/gpt2/__init__.py`
```python
from __future__ import annotations

from model_trainer.core.services.model.backend_adapter import BackendFunctions, create_backend
from model_trainer.core.services.model.backends.gpt2.config import GPT2TrainConfig
from model_trainer.core.services.model.backends.gpt2.evaluate import evaluate_gpt2
from model_trainer.core.services.model.backends.gpt2.generate import generate_gpt2
from model_trainer.core.services.model.backends.gpt2.io import load_gpt2, save_gpt2
from model_trainer.core.services.model.backends.gpt2.prepare import prepare_gpt2
from model_trainer.core.services.model.backends.gpt2.score import score_gpt2
from model_trainer.core.services.model.backends.gpt2.train import train_gpt2


def _convert_config(cfg: ModelTrainConfig) -> GPT2TrainConfig:
    return GPT2TrainConfig(
        model_size=cfg["model_size"],
        max_seq_len=cfg["max_seq_len"],
        num_epochs=cfg["num_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        corpus_path=cfg["corpus_path"],
    )


GPT2_BACKEND_FUNCS: BackendFunctions[GPT2TrainConfig] = {
    "name": "gpt2",
    "convert_config": _convert_config,
    "prepare": prepare_gpt2,
    "train": train_gpt2,
    "evaluate": evaluate_gpt2,
    "score": score_gpt2,
    "generate": generate_gpt2,
    "save": save_gpt2,
    "load": load_gpt2,
}

gpt2_backend = create_backend(GPT2_BACKEND_FUNCS)
```

**Files to delete:**
- `core/services/model/gpt2_backend_impl.py` (~156 lines)
- `core/services/model/char_lstm_backend_impl.py` (~156 lines)

**Estimated reduction:** ~280 lines

---

## Phase 5: Flatten Directory Structure ❌ CANCELLED

### Decision

**Cancelled** - Elected to avoid monolithic files.

The current modular structure (`backends/gpt2/*.py` with 9 focused files) is preferred:
- Each file has single responsibility
- Avoids 500+ line monolithic files
- Easier to test and maintain
- Better for code review diffs

**Current structure retained:**
```
core/services/model/backends/gpt2/
├── __init__.py     (backend registration)
├── _dl.py          (DataLoader)
├── evaluate.py     (evaluation)
├── generate.py     (text generation)
├── hf_gpt2.py      (HuggingFace model wrapper)
├── io.py           (save/load)
├── prepare.py      (model initialization)
├── score.py        (per-token loss)
└── train.py        (training entry point)
```

---

## Phase 6: Registry Improvements ✅ COMPLETED

### 6.1 Capability Discovery - DONE

**Implemented in:** `core/contracts/model.py`

```python
class BackendCapabilities(TypedDict):
    """What a backend can do."""
    supports_train: bool
    supports_evaluate: bool
    supports_score: bool
    supports_generate: bool
    supports_distributed: bool
    supported_sizes: tuple[str, ...]
```

### 6.2 Lazy Backend Loading - DONE

**Implemented in:** `core/services/registries.py`

- `BackendRegistration` stores factory + capabilities
- `ModelRegistry` has lazy loading via `_cache`
- `get_capabilities()` returns capabilities without instantiation

---

## Phase 7: Worker Refactor ✅ COMPLETED

### 7.1 Problem Solved

Workers are now split into focused modules.

### 7.2 Current Structure

```
worker/
├── __init__.py
├── worker_entry.py       (RQ worker entry point)
├── train_job.py          (process_train_job)
├── eval_job.py           (process_eval_job)
├── score_job.py          (process_score_job)
├── generate_job.py       (process_generate_job)
├── tokenizer_worker.py   (process_tokenizer_job)
├── job_utils.py          (shared utilities)
├── manifest.py           (manifest handling)
└── trainer_job_store.py  (job state persistence)
```

### 7.3 Job Context

`JobContext` comes from `platform_workers.job_context`

---

## Phase 8: ML Event System ✅ COMPLETED

### 8.1 Architecture

Two complementary event systems:

| System | File | Purpose |
|--------|------|---------|
| `job_events` | `platform_core/job_events.py` | Generic job lifecycle (started/progress%/completed/failed) |
| `trainer_metrics_events` | `platform_core/trainer_metrics_events.py` | ML-specific metrics (config, loss, perplexity, epochs) |

**Job lifecycle events** (from `job_events.py` with `domain="trainer"`):
- `trainer.job.started.v1` - job_id, user_id, queue
- `trainer.job.progress.v1` - job_id, user_id, progress (%), message, payload
- `trainer.job.completed.v1` - job_id, user_id, result_id, result_bytes
- `trainer.job.failed.v1` - job_id, user_id, error_kind (user/system), message

**ML metrics events** (from `trainer_metrics_events.py`):
- `trainer.metrics.config.v1` - model config + resource allocation at start
- `trainer.metrics.progress.v1` - epoch, step, loss during training
- `trainer.metrics.completed.v1` - final loss, perplexity, artifact path

### 8.2 Trainer Metrics Events

**Implemented in:** `libs/platform_core/src/platform_core/trainer_metrics_events.py`

Versioned event types for training metrics with full encode/decode support:

```python
from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

TrainerMetricsEventType = Literal[
    "trainer.metrics.config.v1",
    "trainer.metrics.progress.v1",
    "trainer.metrics.completed.v1",
]


class TrainerConfigV1(TypedDict):
    """Training configuration event published at job start."""
    type: Literal["trainer.metrics.config.v1"]
    job_id: str
    user_id: int
    model_family: str
    model_size: str
    total_epochs: int
    queue: str
    cpu_cores: NotRequired[int]
    memory_mb: NotRequired[int]
    optimal_threads: NotRequired[int]
    optimal_workers: NotRequired[int]
    batch_size: NotRequired[int]
    learning_rate: NotRequired[float]


class TrainerProgressMetricsV1(TypedDict):
    """Training progress metrics event published during training."""
    type: Literal["trainer.metrics.progress.v1"]
    job_id: str
    user_id: int
    epoch: int
    total_epochs: int
    step: int
    loss: float


class TrainerCompletedMetricsV1(TypedDict):
    """Training completion metrics event published at job completion."""
    type: Literal["trainer.metrics.completed.v1"]
    job_id: str
    user_id: int
    loss: float
    perplexity: float
    artifact_path: str


TrainerMetricsEventV1 = TrainerConfigV1 | TrainerProgressMetricsV1 | TrainerCompletedMetricsV1
```

### 8.3 Factory Functions

```python
def make_config_event(*, job_id: str, user_id: int, ...) -> TrainerConfigV1: ...
def make_progress_metrics_event(*, job_id: str, user_id: int, ...) -> TrainerProgressMetricsV1: ...
def make_completed_metrics_event(*, job_id: str, user_id: int, ...) -> TrainerCompletedMetricsV1: ...
```

### 8.4 Encode/Decode + TypeGuards

```python
def encode_trainer_metrics_event(event: TrainerMetricsEventV1) -> str: ...
def decode_trainer_metrics_event(payload: str) -> TrainerMetricsEventV1: ...

def is_config(ev: TrainerMetricsEventV1) -> TypeGuard[TrainerConfigV1]: ...
def is_progress_metrics(ev: TrainerMetricsEventV1) -> TypeGuard[TrainerProgressMetricsV1]: ...
def is_completed_metrics(ev: TrainerMetricsEventV1) -> TypeGuard[TrainerCompletedMetricsV1]: ...
```

---

## Phase 9: Error Codes

### 9.1 Add ML-Specific Error Codes

**File:** `core/errors.py`

```python
from __future__ import annotations

from typing import Literal

from platform_core.errors import AppError


ModelTrainerErrorCode = Literal[
    # Training errors
    "TRAINING_CANCELLED",
    "TRAINING_OOM",
    "TRAINING_NAN_LOSS",
    "TRAINING_DIVERGED",
    # Model errors
    "MODEL_NOT_FOUND",
    "MODEL_LOAD_FAILED",
    "MODEL_INCOMPATIBLE",
    "INVALID_MODEL_SIZE",
    # Tokenizer errors
    "TOKENIZER_NOT_FOUND",
    "TOKENIZER_LOAD_FAILED",
    "TOKENIZER_TRAIN_FAILED",
    # Dataset errors
    "CORPUS_NOT_FOUND",
    "CORPUS_EMPTY",
    "CORPUS_TOO_LARGE",
    # Infrastructure errors
    "CUDA_NOT_AVAILABLE",
    "CUDA_OOM",
    "ARTIFACT_UPLOAD_FAILED",
    "ARTIFACT_DOWNLOAD_FAILED",
]


# HTTP status mapping
ERROR_STATUS: dict[ModelTrainerErrorCode, int] = {
    "TRAINING_CANCELLED": 499,
    "TRAINING_OOM": 507,
    "TRAINING_NAN_LOSS": 500,
    "TRAINING_DIVERGED": 500,
    "MODEL_NOT_FOUND": 404,
    "MODEL_LOAD_FAILED": 500,
    "MODEL_INCOMPATIBLE": 400,
    "INVALID_MODEL_SIZE": 400,
    "TOKENIZER_NOT_FOUND": 404,
    "TOKENIZER_LOAD_FAILED": 500,
    "TOKENIZER_TRAIN_FAILED": 500,
    "CORPUS_NOT_FOUND": 404,
    "CORPUS_EMPTY": 400,
    "CORPUS_TOO_LARGE": 413,
    "CUDA_NOT_AVAILABLE": 503,
    "CUDA_OOM": 507,
    "ARTIFACT_UPLOAD_FAILED": 502,
    "ARTIFACT_DOWNLOAD_FAILED": 502,
}


class ModelTrainerError(AppError[ModelTrainerErrorCode]):
    """Model trainer specific error."""
    pass
```

---

## Phase 10: Testing Strategy ⚠️ IN PROGRESS

### 10.1 Test Structure

```
tests/
├── unit/
│   ├── core/
│   │   ├── test_contracts.py      (TypedDict validation)
│   │   ├── test_validators.py     (config validators)
│   │   └── test_training.py       (BaseTrainer)
│   ├── backends/
│   │   ├── test_gpt2.py
│   │   └── test_char_lstm.py
│   ├── worker/
│   │   ├── test_training_worker.py
│   │   └── test_inference_worker.py
│   └── api/
│       ├── test_routes.py
│       └── test_schemas.py
├── integration/
│   ├── test_training_e2e.py
│   ├── test_inference_e2e.py
│   └── test_tokenizer_e2e.py
└── conftest.py
```

### 10.2 Coverage Requirements

**pyproject.toml:**
```toml
[tool.pytest.ini_options]
addopts = "--cov=model_trainer --cov-report=term-missing --cov-fail-under=100 --cov-branch"
```

### 10.3 Test Patterns

**No mocking of internal types:**
```python
# BAD - mocking internal types
@patch("model_trainer.core.backends.gpt2.GPT2Model")
def test_prepare(mock_model):
    ...

# GOOD - use real types with minimal fixtures
def test_prepare():
    cfg = GPT2TrainConfig(
        model_size="tiny",
        max_seq_len=32,
        num_epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        corpus_path="/tmp/test_corpus.txt",
        device="cpu",
    )
    tokenizer = create_test_tokenizer()
    prepared = prepare_gpt2(cfg, test_settings(), tokenizer)

    assert prepared["model"] is not None
    assert prepared["max_seq_len"] == 32
```

---

## Phase 11: Migration Plan ✅ N/A

**Not needed** - No backwards compatibility required per project requirements.

- No fallback code
- No legacy support
- No shims or stubs
- Clean break from old patterns

---

## Summary of Changes

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Config Classes → TypedDict | ✅ COMPLETED |
| 2 | Unified PreparedModel | ✅ COMPLETED |
| 3 | BaseTrainer extraction | ✅ COMPLETED |
| 4 | Backend Adapter Factory | ✅ COMPLETED |
| 5 | Flatten Directory Structure | ❌ CANCELLED (prefer modular) |
| 6 | Registry + Capabilities | ✅ COMPLETED |
| 7 | Worker Refactor | ✅ COMPLETED |
| 8 | ML Event System | ✅ COMPLETED |
| 9 | Error Codes | Pending |
| 10 | Testing Strategy | ⚠️ IN PROGRESS |
| 11 | Migration Plan | ✅ N/A (no back compat) |

**Total lines saved: ~900+ lines of boilerplate**

---

## Verification Checklist

Before merge, verify:

- [x] `make check` passes (mypy strict, ruff, pytest) - Phases 1-7 verified
- [x] No `Any` in codebase (except dynamic import patterns)
- [x] No `cast` usage
- [x] No `type: ignore` comments
- [x] No `.pyi` stub files
- [x] All TypedDict are immutable (no mutation after creation)
- [ ] All errors use ModelTrainerError with proper codes (Phase 9)
- [x] All workers emit proper ML events (Phase 8)
- [x] 100% statement coverage - ACHIEVED (457 tests)
- [x] 100% branch coverage - ACHIEVED (457 tests)
- [x] All public functions have docstrings
- [x] No TODO/FIXME/HACK markers

**Resolved:** All fake backends in test stubs now have `capabilities()` method. 457 tests pass with 100% coverage.
