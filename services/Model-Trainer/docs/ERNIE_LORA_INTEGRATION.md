# ERNIE + LoRA/Unsloth Integration — Refactor Document

## Status Summary

| Phase | Status | Notes |
|-------|--------|-------|
| **Phase 1: Core Infrastructure** | ✅ DONE | TypedDicts, validation, contracts |
| **Phase 2: Backend Implementation** | ✅ DONE | Implemented as `hf_lm` backend (generic HF LM) |
| **Phase 3: Finetuning Strategies** | ✅ DONE | full, lora, qlora, unsloth in `services/finetuning/` |
| **Phase 4: Testing** | ✅ DONE | 987 tests, 100% coverage |
| **Phase 5: API Integration** | ✅ DONE | API schema, validators, worker wiring |
| **Phase 6: Documentation** | ✅ DONE | README, API docs complete |

### Completed (Phase 6)

1. **API Documentation** (`docs/api.md`) - ✅ DONE:
   - Documented all new request fields (`hub_model_id`, `finetuning_strategy`, `lora`, `quantization`, `unsloth`)
   - Added LoRA, Quantization, and Unsloth configuration tables
   - Updated `model_family` to clarify `hf_lm` for HuggingFace models (LLaMA, Qwen, etc.)
   - Updated `tokenizer_id` as conditional (required for non-hf_lm, optional for hf_lm)

### Completed (Phase 5)

1. **API Schema** (`api/schemas/runs.py`) - ✅ DONE:
   - Added `hub_model_id: str | None` field
   - Added `finetuning_strategy: Literal["full", "lora", "qlora", "unsloth"]` field
   - Added `lora: LoraConfigRequest | None` field
   - Added `quantization: QuantizationConfigRequest | None` field
   - Added `unsloth: UnslothConfigRequest | None` field
   - Updated `model_family` to include `"hf_lm"`

2. **API Validators** (`api/validators/runs.py`) - ✅ DONE:
   - Validate `hub_model_id` required when `model_family == "hf_lm"`
   - Validate `lora` config when `finetuning_strategy` in `["lora", "qlora", "unsloth"]`
   - Validate `quantization` config when `finetuning_strategy == "qlora"`
   - Validate `unsloth` config when `finetuning_strategy == "unsloth"`

3. **Worker Wiring** - ✅ DONE:
   - `hf_lm` backend registered in `container.py`
   - `job_utils.py` passes through all new fields in `build_cfg`
   - Training orchestrator passes new fields from TrainRequest to TrainRequestPayload

---

## 1. Overview

This document specifies the integration of ERNIE model family support with **pluggable fine-tuning strategies** (LoRA, Unsloth, QLoRA, Full) into the Model-Trainer architecture. The implementation follows the covenant_ml pattern of pluggable backends via Protocol + Registry.

### Goals

- Add ERNIE 4.5 model family as a new backend following existing GPT-2/Char-LSTM patterns.
- **Create pluggable FineTuningStrategy system** with registry (like covenant_ml backends).
- Implement strategies: `full`, `lora`, `qlora`, `unsloth`.
- Maintain 100% type safety: no `Any`, `cast`, `type: ignore`, `.pyi`, or stubs.
- Achieve 100% test coverage for statements and branches.
- Zero fallback/best-effort behavior; explicit failure propagation.

### Non-Goals

- Distributed multi-GPU training (deferred).
- Full PaddlePaddle framework integration (use HuggingFace ERNIE weights).
- LLaMA-Factory integration (separate future work).

### Constraints

- All new code follows `_test_hooks.py` pattern for dependency injection.
- TypedDicts with `_encode_*`/`_decode_*` functions and `require_*` validation.
- Protocol types for all dynamic imports (Unsloth, PEFT, ERNIE models).
- No mocks in tests; test actual code paths with fakes via hooks.


## 1.1 Pluggable Architecture (covenant_ml Pattern)

The architecture separates two concerns:

1. **Model Backend** (existing): Defines base model architecture (GPT-2, Char-LSTM, ERNIE)
2. **Fine-Tuning Strategy** (new): Defines how to adapt the model for training

```
┌─────────────────────────────────────────────────────────────┐
│                    ModelTrainConfig                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ model_family │  │ finetuning   │  │ strategy_config  │  │
│  │ = "ernie"    │  │ = "lora"     │  │ = LoraConfig     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │                   │                    │
         ▼                   ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ ModelBackend    │  │ FineTuningReg   │  │ Strategy gets   │
│ Registry        │  │ Registry        │  │ its config      │
│ → ERNIEBackend  │  │ → LoRAStrategy  │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                   │
         ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│              PreparedLMModel (with LoRA adapters)           │
└─────────────────────────────────────────────────────────────┘
```


## 2. Architecture Changes

### 2.1 Model Family Expansion

Update `ModelTrainConfig` in `core/contracts/model.py`:

```python
# Before
model_family: Literal["gpt2", "llama", "qwen", "char_lstm"]

# After
model_family: Literal["gpt2", "llama", "qwen", "char_lstm", "ernie"]
```

### 2.2 New Configuration Types

Add to `core/contracts/model.py`:

```python
class LoraConfig(TypedDict):
    """Configuration for LoRA fine-tuning."""

    enabled: bool
    r: int                    # LoRA rank (8, 16, 32, 64)
    lora_alpha: int           # Scaling factor
    lora_dropout: float       # Dropout probability
    target_modules: tuple[str, ...]  # Modules to apply LoRA
    bias: Literal["none", "all", "lora_only"]


class QuantizationConfig(TypedDict):
    """Configuration for model quantization."""

    load_in_4bit: bool
    load_in_8bit: bool
    bnb_4bit_compute_dtype: Literal["float16", "bfloat16", "float32"]
    bnb_4bit_quant_type: Literal["nf4", "fp4"]


class UnslothConfig(TypedDict):
    """Configuration for Unsloth optimization."""

    enabled: bool
    max_seq_length: int
    dtype: Literal["float16", "bfloat16"] | None  # None = auto
```

### 2.3 Extended ModelTrainConfig

```python
class ModelTrainConfig(TypedDict):
    """Configuration for model training - extended for LoRA/Unsloth."""

    # Existing fields...
    model_family: Literal["gpt2", "llama", "qwen", "char_lstm", "ernie"]
    model_size: str
    # ... other existing fields ...

    # New fields for LoRA/Unsloth
    lora: LoraConfig | None
    quantization: QuantizationConfig | None
    unsloth: UnslothConfig | None
    hub_model_id: str | None  # HuggingFace model ID for pretrained loading
```


## 3. Protocol Definitions

### 3.1 Unsloth Protocol (`core/contracts/unsloth.py`)

```python
"""Protocols for Unsloth integration with strict typing."""

from __future__ import annotations

from typing import Protocol

from model_trainer.core.types import LMModelProto


class FastLanguageModelProto(Protocol):
    """Protocol for Unsloth's FastLanguageModel class."""

    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: int,
        dtype: str | None,
        load_in_4bit: bool,
    ) -> tuple[LMModelProto, TokenizerProto]: ...

    @staticmethod
    def get_peft_model(
        model: LMModelProto,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: tuple[str, ...],
        bias: str,
    ) -> LMModelProto: ...


class TokenizerProto(Protocol):
    """Protocol for tokenizer returned by Unsloth."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    @property
    def eos_token_id(self) -> int: ...
    @property
    def pad_token_id(self) -> int | None: ...
    def get_vocab_size(self) -> int: ...
```

### 3.2 PEFT Protocol (`core/contracts/peft.py`)

```python
"""Protocols for PEFT library integration."""

from __future__ import annotations

from typing import Protocol

from model_trainer.core.types import LMModelProto


class PeftModelProto(Protocol):
    """Protocol for PEFT-wrapped models."""

    def save_pretrained(self, save_directory: str) -> None: ...
    def merge_and_unload(self) -> LMModelProto: ...

    # Inherit LMModelProto methods
    def forward(
        self,
        input_ids: object,
        labels: object | None = None,
    ) -> object: ...

    def parameters(self) -> object: ...
    def train(self) -> None: ...
    def eval(self) -> None: ...
    def to(self, device: str) -> PeftModelProto: ...


class LoraConfigProto(Protocol):
    """Protocol for PEFT LoraConfig."""

    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    bias: str
    task_type: str
```


## 4. Backend Implementation

### 4.1 Directory Structure

```
services/Model-Trainer/src/model_trainer/core/services/model/backends/ernie/
├── __init__.py
├── _test_hooks.py        # Hooks for Unsloth/PEFT/model loading
├── hf_ernie.py           # HuggingFace ERNIE model loading
├── unsloth_loader.py     # Unsloth-optimized loading
├── peft_wrapper.py       # PEFT/LoRA wrapper utilities
├── prepare.py            # prepare_ernie_with_handle
├── train.py              # train_prepared_ernie (delegates to BaseTrainer)
├── io.py                 # save/load with LoRA adapter support
├── evaluate.py           # evaluation
├── generate.py           # text generation
└── score.py              # scoring/perplexity
```

### 4.2 Test Hooks (`_test_hooks.py`)

```python
"""Internal hooks for ERNIE backend dependency injection.

Production code sets hooks to real implementations at startup.
Tests set hooks to fakes. No conditionals - call hooks directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model_trainer.core.contracts.peft import PeftModelProto
    from model_trainer.core.contracts.unsloth import (
        FastLanguageModelProto,
        TokenizerProto,
    )
    from model_trainer.core.types import LMModelProto


# === Unsloth Hooks ===

def _real_unsloth_from_pretrained(
    model_name: str,
    max_seq_length: int,
    dtype: str | None,
    load_in_4bit: bool,
) -> tuple[LMModelProto, TokenizerProto]:
    """Real implementation using Unsloth."""
    unsloth_mod = __import__("unsloth", fromlist=["FastLanguageModel"])
    FastLanguageModel: FastLanguageModelProto = unsloth_mod.FastLanguageModel
    return FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )


def _real_unsloth_get_peft_model(
    model: LMModelProto,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: tuple[str, ...],
    bias: str,
) -> LMModelProto:
    """Real implementation for LoRA wrapping via Unsloth."""
    unsloth_mod = __import__("unsloth", fromlist=["FastLanguageModel"])
    FastLanguageModel: FastLanguageModelProto = unsloth_mod.FastLanguageModel
    return FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
    )


# Hook functions - tests replace these
unsloth_from_pretrained = _real_unsloth_from_pretrained
unsloth_get_peft_model = _real_unsloth_get_peft_model


# === PEFT Hooks (fallback when Unsloth unavailable) ===

def _real_peft_get_peft_model(
    model: LMModelProto,
    peft_config: object,
) -> PeftModelProto:
    """Real implementation using PEFT directly."""
    peft_mod = __import__("peft", fromlist=["get_peft_model"])
    get_peft_model_fn = peft_mod.get_peft_model
    result: PeftModelProto = get_peft_model_fn(model, peft_config)
    return result


def _real_peft_lora_config(
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    bias: str,
    task_type: str,
) -> object:
    """Create PEFT LoraConfig."""
    peft_mod = __import__("peft", fromlist=["LoraConfig", "TaskType"])
    LoraConfig = peft_mod.LoraConfig
    TaskType = peft_mod.TaskType
    task_type_enum = getattr(TaskType, task_type)
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type_enum,
    )


peft_get_peft_model = _real_peft_get_peft_model
peft_lora_config = _real_peft_lora_config


# === HuggingFace Model Hooks ===

def _real_load_ernie_model(model_id: str) -> LMModelProto:
    """Load ERNIE model from HuggingFace."""
    transformers_mod = __import__("transformers", fromlist=["AutoModelForCausalLM"])
    AutoModelForCausalLM = transformers_mod.AutoModelForCausalLM
    model: LMModelProto = AutoModelForCausalLM.from_pretrained(model_id)
    return model


def _real_load_ernie_tokenizer(model_id: str) -> TokenizerProto:
    """Load ERNIE tokenizer from HuggingFace."""
    transformers_mod = __import__("transformers", fromlist=["AutoTokenizer"])
    AutoTokenizer = transformers_mod.AutoTokenizer
    tokenizer: TokenizerProto = AutoTokenizer.from_pretrained(model_id)
    return tokenizer


load_ernie_model = _real_load_ernie_model
load_ernie_tokenizer = _real_load_ernie_tokenizer


# === Availability Check Hooks ===

def _real_unsloth_available() -> bool:
    """Check if Unsloth is installed and importable."""
    try:
        __import__("unsloth")
        return True
    except ImportError:
        return False


def _real_peft_available() -> bool:
    """Check if PEFT is installed and importable."""
    try:
        __import__("peft")
        return True
    except ImportError:
        return False


unsloth_available = _real_unsloth_available
peft_available = _real_peft_available
```

### 4.3 Model Preparation (`prepare.py`)

```python
"""ERNIE model preparation with optional Unsloth/LoRA optimization."""

from __future__ import annotations

from model_trainer.core.contracts.model import (
    LoraConfig,
    ModelTrainConfig,
    PreparedLMModel,
    QuantizationConfig,
    UnslothConfig,
)
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.services.model.backends.ernie import _test_hooks
from model_trainer.core.services.model.backends.ernie.io import (
    encoder_from_ernie_tokenizer,
    token_ids_from_ernie,
)


def prepare_ernie_with_handle(
    tokenizer: TokenizerHandle,
    cfg: ModelTrainConfig,
) -> PreparedLMModel:
    """Prepare an ERNIE model for training.

    Supports three modes:
    1. Unsloth + LoRA (fastest, requires Unsloth)
    2. PEFT LoRA (fallback, requires PEFT)
    3. Full fine-tuning (no LoRA)

    Args:
        tokenizer: TokenizerHandle for encoding text.
        cfg: Training configuration including LoRA/Unsloth settings.

    Returns:
        PreparedLMModel ready for training.

    Raises:
        RuntimeError: If required libraries are not available.
    """
    hub_model_id = cfg.get("hub_model_id")
    if hub_model_id is None:
        raise ValueError("hub_model_id required for ERNIE models")

    lora_cfg = cfg.get("lora")
    unsloth_cfg = cfg.get("unsloth")
    quant_cfg = cfg.get("quantization")

    # Determine loading strategy
    use_unsloth = (
        unsloth_cfg is not None
        and unsloth_cfg["enabled"]
        and _test_hooks.unsloth_available()
    )
    use_lora = lora_cfg is not None and lora_cfg["enabled"]

    if use_unsloth:
        model, ernie_tokenizer = _prepare_with_unsloth(
            hub_model_id, unsloth_cfg, lora_cfg
        )
    elif use_lora:
        model = _prepare_with_peft(hub_model_id, lora_cfg, quant_cfg)
        ernie_tokenizer = _test_hooks.load_ernie_tokenizer(hub_model_id)
    else:
        model = _test_hooks.load_ernie_model(hub_model_id)
        ernie_tokenizer = _test_hooks.load_ernie_tokenizer(hub_model_id)

    eos_id, pad_id, vocab_size = token_ids_from_ernie(ernie_tokenizer)

    return PreparedLMModel(
        model=model,
        tokenizer_id=cfg["tokenizer_id"],
        eos_id=eos_id,
        pad_id=pad_id,
        max_seq_len=cfg["max_seq_len"],
        tok_for_dataset=encoder_from_ernie_tokenizer(ernie_tokenizer),
    )


def _prepare_with_unsloth(
    model_id: str,
    unsloth_cfg: UnslothConfig,
    lora_cfg: LoraConfig | None,
) -> tuple[object, object]:
    """Load model with Unsloth optimization.

    Args:
        model_id: HuggingFace model ID.
        unsloth_cfg: Unsloth configuration.
        lora_cfg: Optional LoRA configuration.

    Returns:
        Tuple of (model, tokenizer).
    """
    model, tokenizer = _test_hooks.unsloth_from_pretrained(
        model_name=model_id,
        max_seq_length=unsloth_cfg["max_seq_length"],
        dtype=unsloth_cfg["dtype"],
        load_in_4bit=True,  # Unsloth default for efficiency
    )

    if lora_cfg is not None and lora_cfg["enabled"]:
        model = _test_hooks.unsloth_get_peft_model(
            model,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
        )

    return model, tokenizer


def _prepare_with_peft(
    model_id: str,
    lora_cfg: LoraConfig,
    quant_cfg: QuantizationConfig | None,
) -> object:
    """Load model with PEFT LoRA (without Unsloth).

    Args:
        model_id: HuggingFace model ID.
        lora_cfg: LoRA configuration.
        quant_cfg: Optional quantization configuration.

    Returns:
        PEFT-wrapped model.

    Raises:
        RuntimeError: If PEFT is not available.
    """
    if not _test_hooks.peft_available():
        raise RuntimeError("PEFT library required for LoRA without Unsloth")

    model = _test_hooks.load_ernie_model(model_id)

    peft_config = _test_hooks.peft_lora_config(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=list(lora_cfg["target_modules"]),
        bias=lora_cfg["bias"],
        task_type="CAUSAL_LM",
    )

    return _test_hooks.peft_get_peft_model(model, peft_config)
```


## 5. TypedDict Encode/Decode Functions

### 5.1 LoRA Config Validation (`core/contracts/model_validation.py`)

```python
"""Validation functions for model configuration TypedDicts."""

from __future__ import annotations

from typing import Literal

from model_trainer.core.contracts.model import (
    LoraConfig,
    QuantizationConfig,
    UnslothConfig,
)


def require_lora_config(data: dict[str, object]) -> LoraConfig:
    """Validate and convert dict to LoraConfig.

    Args:
        data: Raw dictionary from JSON/TOML.

    Returns:
        Validated LoraConfig TypedDict.

    Raises:
        ValueError: If required fields are missing or invalid.
        TypeError: If field types are incorrect.
    """
    if not isinstance(data.get("enabled"), bool):
        raise TypeError("lora.enabled must be bool")
    if not isinstance(data.get("r"), int):
        raise TypeError("lora.r must be int")
    if not isinstance(data.get("lora_alpha"), int):
        raise TypeError("lora.lora_alpha must be int")
    if not isinstance(data.get("lora_dropout"), float):
        raise TypeError("lora.lora_dropout must be float")

    target_modules = data.get("target_modules")
    if not isinstance(target_modules, (list, tuple)):
        raise TypeError("lora.target_modules must be list or tuple")
    target_modules_tuple = tuple(str(m) for m in target_modules)

    bias = data.get("bias")
    if bias not in ("none", "all", "lora_only"):
        raise ValueError("lora.bias must be 'none', 'all', or 'lora_only'")
    bias_literal: Literal["none", "all", "lora_only"] = bias  # type narrowing

    return LoraConfig(
        enabled=data["enabled"],
        r=data["r"],
        lora_alpha=data["lora_alpha"],
        lora_dropout=data["lora_dropout"],
        target_modules=target_modules_tuple,
        bias=bias_literal,
    )


def encode_lora_config(cfg: LoraConfig) -> dict[str, object]:
    """Encode LoraConfig to JSON-serializable dict.

    Args:
        cfg: LoraConfig TypedDict.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    return {
        "enabled": cfg["enabled"],
        "r": cfg["r"],
        "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg["lora_dropout"],
        "target_modules": list(cfg["target_modules"]),
        "bias": cfg["bias"],
    }


def require_quantization_config(data: dict[str, object]) -> QuantizationConfig:
    """Validate and convert dict to QuantizationConfig.

    Args:
        data: Raw dictionary from JSON/TOML.

    Returns:
        Validated QuantizationConfig TypedDict.

    Raises:
        ValueError: If required fields are missing or invalid.
        TypeError: If field types are incorrect.
    """
    if not isinstance(data.get("load_in_4bit"), bool):
        raise TypeError("quantization.load_in_4bit must be bool")
    if not isinstance(data.get("load_in_8bit"), bool):
        raise TypeError("quantization.load_in_8bit must be bool")

    compute_dtype = data.get("bnb_4bit_compute_dtype")
    if compute_dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError("quantization.bnb_4bit_compute_dtype invalid")

    quant_type = data.get("bnb_4bit_quant_type")
    if quant_type not in ("nf4", "fp4"):
        raise ValueError("quantization.bnb_4bit_quant_type must be 'nf4' or 'fp4'")

    return QuantizationConfig(
        load_in_4bit=data["load_in_4bit"],
        load_in_8bit=data["load_in_8bit"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_type,
    )


def require_unsloth_config(data: dict[str, object]) -> UnslothConfig:
    """Validate and convert dict to UnslothConfig.

    Args:
        data: Raw dictionary from JSON/TOML.

    Returns:
        Validated UnslothConfig TypedDict.

    Raises:
        TypeError: If field types are incorrect.
    """
    if not isinstance(data.get("enabled"), bool):
        raise TypeError("unsloth.enabled must be bool")
    if not isinstance(data.get("max_seq_length"), int):
        raise TypeError("unsloth.max_seq_length must be int")

    dtype = data.get("dtype")
    if dtype is not None and dtype not in ("float16", "bfloat16"):
        raise ValueError("unsloth.dtype must be 'float16', 'bfloat16', or null")

    dtype_value: Literal["float16", "bfloat16"] | None = dtype

    return UnslothConfig(
        enabled=data["enabled"],
        max_seq_length=data["max_seq_length"],
        dtype=dtype_value,
    )
```


## 6. Save/Load with LoRA Adapters

### 6.1 IO Functions (`backends/ernie/io.py`)

```python
"""ERNIE model save/load with LoRA adapter support."""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import dump_json_str, load_json_str

from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.encoding import Encoder, HandleEncoder
from model_trainer.core.services.model.backends.ernie import _test_hooks


class ErnieTokenizerEncoder(Encoder):
    """Encoder adapter for ERNIE tokenizer."""

    def __init__(self, tokenizer: object) -> None:
        self._tok = tokenizer

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        result: list[int] = self._tok.encode(text)
        return result


def encoder_from_ernie_tokenizer(tokenizer: object) -> Encoder:
    """Create Encoder from ERNIE tokenizer."""
    return ErnieTokenizerEncoder(tokenizer)


def token_ids_from_ernie(tokenizer: object) -> tuple[int, int, int]:
    """Extract special token IDs from ERNIE tokenizer.

    Args:
        tokenizer: ERNIE tokenizer instance.

    Returns:
        Tuple of (eos_id, pad_id, vocab_size).
    """
    eos_id: int = tokenizer.eos_token_id
    pad_id_opt = tokenizer.pad_token_id
    pad_id: int = pad_id_opt if pad_id_opt is not None else eos_id
    vocab_size: int = len(tokenizer)
    return eos_id, pad_id, vocab_size


def save_prepared_ernie(
    prepared: PreparedLMModel,
    out_dir: str,
    *,
    is_lora: bool = False,
) -> None:
    """Save ERNIE model to disk.

    For LoRA models, saves only the adapter weights.
    For full models, saves complete weights.

    Args:
        prepared: Prepared model to save.
        out_dir: Output directory path.
        is_lora: Whether this is a LoRA adapter model.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if is_lora:
        # LoRA models have save_pretrained that saves only adapters
        prepared.model.save_pretrained(out_dir)
        # Write metadata
        metadata = {"is_lora": True, "tokenizer_id": prepared.tokenizer_id}
        Path(out_dir, "ernie_metadata.json").write_text(
            dump_json_str(metadata), encoding="utf-8"
        )
    else:
        prepared.model.save_pretrained(out_dir)
        metadata = {"is_lora": False, "tokenizer_id": prepared.tokenizer_id}
        Path(out_dir, "ernie_metadata.json").write_text(
            dump_json_str(metadata), encoding="utf-8"
        )


def load_prepared_ernie_from_handle(
    artifact_path: str,
    tokenizer: TokenizerHandle,
) -> PreparedLMModel:
    """Load ERNIE model from saved artifact.

    Automatically detects if artifact is LoRA adapter or full model.

    Args:
        artifact_path: Path to saved model directory.
        tokenizer: TokenizerHandle for encoding.

    Returns:
        PreparedLMModel ready for inference or continued training.
    """
    metadata_path = Path(artifact_path, "ernie_metadata.json")
    metadata = load_json_str(metadata_path.read_text(encoding="utf-8"))

    is_lora = metadata.get("is_lora", False)

    if is_lora:
        # Load base model + LoRA adapter
        # TODO: Need to know base model ID - store in metadata
        raise NotImplementedError("LoRA model loading requires base model ID")
    else:
        model = _test_hooks.load_ernie_model(artifact_path)

    # Use provided tokenizer handle for encoding
    eos_id, pad_id, _ = token_ids_from_handle(tokenizer)
    max_seq_len = _get_model_max_seq_len(model)

    return PreparedLMModel(
        model=model,
        tokenizer_id=metadata.get("tokenizer_id", "unknown"),
        eos_id=eos_id,
        pad_id=pad_id,
        max_seq_len=max_seq_len,
        tok_for_dataset=HandleEncoder(tokenizer),
    )


def token_ids_from_handle(tokenizer: TokenizerHandle) -> tuple[int, int, int]:
    """Extract token IDs from TokenizerHandle."""
    eos_id_opt = tokenizer.token_to_id("[EOS]")
    eos_id = int(eos_id_opt) if eos_id_opt is not None else 0
    pad_id_opt = tokenizer.token_to_id("[PAD]")
    pad_id = int(pad_id_opt) if pad_id_opt is not None else 0
    vocab_size = int(tokenizer.get_vocab_size())
    return eos_id, pad_id, vocab_size


def _get_model_max_seq_len(model: object) -> int:
    """Extract max sequence length from model config."""
    config = model.config
    # Try common attribute names
    for attr in ("max_position_embeddings", "n_positions", "max_seq_length"):
        val = getattr(config, attr, None)
        if isinstance(val, int):
            return val
    return 2048  # ERNIE default
```


## 7. Backend Factory Integration

### 7.1 Update `backend_factory.py`

```python
# Add to backend_factory.py

ERNIE_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_evaluate": True,
    "supports_score": True,
    "supports_generate": True,
    "supports_distributed": False,
    "supported_sizes": ("0.3B", "7B", "21B"),
}


def ernie_backend_funcs() -> BackendFuncs:
    """Get function references for ERNIE backend."""
    from .backends.ernie.evaluate import evaluate_ernie
    from .backends.ernie.generate import generate_ernie
    from .backends.ernie.io import load_prepared_ernie_from_handle, save_prepared_ernie
    from .backends.ernie.prepare import prepare_ernie_with_handle
    from .backends.ernie.score import score_ernie
    from .backends.ernie.train import train_prepared_ernie

    return BackendFuncs(
        name="ernie",
        prepare=prepare_ernie_with_handle,
        save=lambda p, d: save_prepared_ernie(p, d, is_lora=False),
        load=load_prepared_ernie_from_handle,
        train=train_prepared_ernie,
        evaluate=evaluate_ernie,
        score=score_ernie,
        generate=generate_ernie,
    )


def create_ernie_backend(dataset_builder: DatasetBuilder) -> ModelBackend:
    """Create an ERNIE ModelBackend."""
    return create_backend(ernie_backend_funcs(), dataset_builder, ERNIE_CAPABILITIES)
```


## 8. API Schema Updates

### 8.1 Run Creation Schema (`api/schemas/runs.py`)

Add optional LoRA/Unsloth fields to `RunCreateRequest`:

```python
class LoraConfigSchema(BaseModel):
    """API schema for LoRA configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    r: int = Field(default=16, ge=4, le=128)
    lora_alpha: int = Field(default=16, ge=1, le=256)
    lora_dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: Literal["none", "all", "lora_only"] = "none"


class UnslothConfigSchema(BaseModel):
    """API schema for Unsloth configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_seq_length: int = Field(default=2048, ge=128, le=8192)
    dtype: Literal["float16", "bfloat16"] | None = None


class RunCreateRequest(BaseModel):
    """Request to create a new training run."""

    # ... existing fields ...

    # New optional fields for ERNIE + LoRA
    hub_model_id: str | None = None
    lora: LoraConfigSchema | None = None
    unsloth: UnslothConfigSchema | None = None
```


## 9. Testing Strategy

### 9.1 Test Structure

```
tests/core/services/model/backends/ernie/
├── test_prepare.py           # Test all preparation paths
├── test_io.py                # Test save/load with LoRA
├── test_train.py             # Integration with BaseTrainer
├── test_hooks.py             # Test hook replacement works
├── test_validation.py        # Test require_* functions
└── conftest.py               # Shared fixtures (fake models/tokenizers)
```

### 9.2 Fake Implementations for Tests (`conftest.py`)

```python
"""Test fixtures for ERNIE backend tests."""

from __future__ import annotations

import pytest

from model_trainer.core.services.model.backends.ernie import _test_hooks


class FakeERNIEModel:
    """Fake ERNIE model for testing."""

    def __init__(self) -> None:
        self._is_training = True
        self._device = "cpu"
        self._params: list[object] = []

    def forward(
        self,
        input_ids: object,
        labels: object | None = None,
    ) -> object:
        """Fake forward pass."""
        class FakeOutput:
            loss = 0.5
        return FakeOutput()

    def parameters(self) -> list[object]:
        return self._params

    def train(self) -> None:
        self._is_training = True

    def eval(self) -> None:
        self._is_training = False

    def to(self, device: str) -> FakeERNIEModel:
        self._device = device
        return self

    def save_pretrained(self, path: str) -> None:
        """Fake save."""
        pass

    @property
    def config(self) -> object:
        class FakeConfig:
            max_position_embeddings = 2048
        return FakeConfig()


class FakeERNIETokenizer:
    """Fake ERNIE tokenizer for testing."""

    @property
    def eos_token_id(self) -> int:
        return 2

    @property
    def pad_token_id(self) -> int:
        return 0

    def __len__(self) -> int:
        return 32000

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 1000 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "decoded"


@pytest.fixture
def fake_ernie_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install fake hooks for ERNIE testing."""

    def fake_load_model(model_id: str) -> FakeERNIEModel:
        return FakeERNIEModel()

    def fake_load_tokenizer(model_id: str) -> FakeERNIETokenizer:
        return FakeERNIETokenizer()

    def fake_unsloth_from_pretrained(
        model_name: str,
        max_seq_length: int,
        dtype: str | None,
        load_in_4bit: bool,
    ) -> tuple[FakeERNIEModel, FakeERNIETokenizer]:
        return FakeERNIEModel(), FakeERNIETokenizer()

    def fake_unsloth_get_peft_model(
        model: object,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: tuple[str, ...],
        bias: str,
    ) -> FakeERNIEModel:
        return FakeERNIEModel()

    monkeypatch.setattr(_test_hooks, "load_ernie_model", fake_load_model)
    monkeypatch.setattr(_test_hooks, "load_ernie_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr(_test_hooks, "unsloth_from_pretrained", fake_unsloth_from_pretrained)
    monkeypatch.setattr(_test_hooks, "unsloth_get_peft_model", fake_unsloth_get_peft_model)
    monkeypatch.setattr(_test_hooks, "unsloth_available", lambda: True)
    monkeypatch.setattr(_test_hooks, "peft_available", lambda: True)
```

### 9.3 Example Test (`test_prepare.py`)

```python
"""Tests for ERNIE model preparation."""

from __future__ import annotations

import pytest

from model_trainer.core.contracts.model import LoraConfig, ModelTrainConfig, UnslothConfig
from model_trainer.core.services.model.backends.ernie.prepare import prepare_ernie_with_handle


class TestPrepareERNIE:
    """Test ERNIE model preparation paths."""

    def test_prepare_with_unsloth_lora(
        self,
        fake_ernie_hooks: None,
        fake_tokenizer_handle: object,
    ) -> None:
        """Test preparation with Unsloth + LoRA."""
        cfg: ModelTrainConfig = {
            "model_family": "ernie",
            "model_size": "7B",
            "max_seq_len": 2048,
            "hub_model_id": "baidu/ERNIE-4.5-7B-PT",
            "tokenizer_id": "test-tok",
            "lora": LoraConfig(
                enabled=True,
                r=16,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=("q_proj", "k_proj"),
                bias="none",
            ),
            "unsloth": UnslothConfig(
                enabled=True,
                max_seq_length=2048,
                dtype=None,
            ),
            # ... other required fields ...
        }

        prepared = prepare_ernie_with_handle(fake_tokenizer_handle, cfg)

        assert prepared.max_seq_len == 2048
        assert prepared.eos_id == 2
        assert prepared.pad_id == 0

    def test_prepare_without_lora(
        self,
        fake_ernie_hooks: None,
        fake_tokenizer_handle: object,
    ) -> None:
        """Test preparation without LoRA (full fine-tuning)."""
        cfg: ModelTrainConfig = {
            "model_family": "ernie",
            "model_size": "7B",
            "max_seq_len": 2048,
            "hub_model_id": "baidu/ERNIE-4.5-7B-PT",
            "tokenizer_id": "test-tok",
            "lora": None,
            "unsloth": None,
            # ... other required fields ...
        }

        prepared = prepare_ernie_with_handle(fake_tokenizer_handle, cfg)

        assert prepared is not None

    def test_prepare_missing_hub_model_id_raises(
        self,
        fake_ernie_hooks: None,
        fake_tokenizer_handle: object,
    ) -> None:
        """Test that missing hub_model_id raises ValueError."""
        cfg: ModelTrainConfig = {
            "model_family": "ernie",
            "model_size": "7B",
            "max_seq_len": 2048,
            "hub_model_id": None,  # Missing!
            # ...
        }

        with pytest.raises(ValueError, match="hub_model_id required"):
            prepare_ernie_with_handle(fake_tokenizer_handle, cfg)
```


## 10. Dependencies

### 10.1 New Dependencies (`pyproject.toml`)

```toml
[tool.poetry.dependencies]
# Existing...

# New optional dependencies for ERNIE + LoRA
peft = { version = ">=0.10.0", optional = true }
bitsandbytes = { version = ">=0.42.0", optional = true }
# unsloth installed separately (requires specific CUDA version)

[tool.poetry.extras]
lora = ["peft", "bitsandbytes"]
```

### 10.2 Unsloth Installation Notes

Unsloth requires CUDA-specific installation:

```bash
# For CUDA 12.x
pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"

# For CUDA 11.8
pip install "unsloth[cu118-ampere] @ git+https://github.com/unslothai/unsloth.git"
```


## 11. Implementation Phases

### Phase 1: Core Infrastructure ✅ DONE
1. ✅ Add TypedDicts to `core/contracts/model.py` (LoraConfig, UnslothConfig, QuantizationConfig)
2. ✅ Add validation functions to `core/contracts/model_validation.py`
3. ✅ Add `finetuning.py` contracts
4. ✅ Add `hf_lm` backend skeleton with `_test_hooks.py`

### Phase 2: Backend Implementation ✅ DONE
5. ✅ Implement `hf_lm/prepare.py` with all loading paths
6. ✅ Implement `hf_lm/io.py` for save/load
7. ✅ Implement `hf_lm/train.py` (delegates to BaseTrainer)
8. ✅ Implement `hf_lm/evaluate.py`, `generate.py`, `score.py`

### Phase 3: Finetuning Strategies ✅ DONE
9. ✅ Create `services/finetuning/registry.py`
10. ✅ Implement `strategies/full.py`, `lora.py`, `qlora.py`, `unsloth.py`
11. ✅ Update `backend_factory.py` with `hf_lm` backend

### Phase 4: Testing ✅ DONE
12. ✅ Create test fixtures with fakes in `tests/core/services/model/backends/hf_lm/testing.py`
13. ✅ Write unit tests for each module
14. ✅ Write integration tests
15. ✅ Achieve 100% coverage (872 tests passing)

### Phase 5: API Integration ✅ DONE
16. ✅ Update `api/schemas/runs.py` with new fields:
    - `model_family: Literal["gpt2", "llama", "qwen", "char_lstm", "hf_lm"]`
    - `hub_model_id: str | None`
    - `finetuning_strategy: Literal["full", "lora", "qlora", "unsloth"]`
    - `lora: LoraConfigRequest | None`
    - `quantization: QuantizationConfigRequest | None`
    - `unsloth: UnslothConfigRequest | None`
17. ✅ Update `api/validators/runs.py`:
    - Validate `hub_model_id` required when `model_family == "hf_lm"`
    - Validate strategy-specific configs (lora, quantization, unsloth)
18. ✅ Update worker/container/orchestrator:
    - `hf_lm` backend registered in `container.py`
    - `job_utils.py` passes new fields in `build_cfg`
    - `training_orchestrator.py` passes new fields to TrainRequestPayload

### Phase 6: Documentation ⚠️ PARTIAL
19. ✅ Update README.md with hf_lm backend info
20. ❌ Update `docs/api.md` with new request/response fields
21. ❌ Add usage examples for LoRA fine-tuning


## 12. Success Criteria

- [x] All new code passes `make check` (mypy strict, ruff, pytest)
- [x] 100% test coverage for statements and branches
- [x] No `Any`, `cast`, `type: ignore`, `.pyi`, or stubs
- [x] TypedDicts have encode/decode with require_* validation
- [x] All hooks follow _test_hooks.py pattern
- [x] HF LM model can be loaded with Unsloth + LoRA **via API**
- [x] Training produces valid checkpoints **via API**
- [x] Checkpoints can be saved/loaded correctly **via API**
- [x] LoRA adapters can be merged and exported **via API**

**Note:** All core functionality is complete and exposed via API. Phase 6 (API documentation) is the only remaining work.
