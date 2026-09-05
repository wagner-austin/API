"""Internal hooks for HuggingFace LM backend dependency injection.

Production code sets hooks to real implementations at startup.
Tests set hooks to fakes. No conditionals - call hooks directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import torch
from platform_core.determinism_record import DeterminismRecord
from platform_ml.wandb_publisher import WandbPublisher

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import (
    ModelTrainConfig,
    PreparedLMModel,
    ProgressCallback,
    QuantizationConfig,
)
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.encoding import Encoder
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import (
    BitsAndBytesConfigProto,
    CausalLMDatasetProto,
    CreateCausalDatasetFn,
    CreateDataLoaderFn,
    CreateTrainerFn,
    DataLoaderProto,
    EvalDirFn,
    HFModelLoader,
    HFTokenizerLoader,
    HFTokenizerProto,
    ModelDirFn,
    PreparedModelLoader,
    ReadTextFileFn,
    TokenizerLoader,
    TrainerProto,
    _BitsAndBytesConfigClassProto,
    _DataLoaderClassProto,
    _HFModelClassProto,
    _HFTokenizerClassProto,
)
from model_trainer.core.types import LMModelProto

#: Compute dtypes a quantization config may name, resolved to torch dtypes
#: here rather than by ``getattr(torch, name)`` so the accepted set is the
#: Literal on :class:`QuantizationConfig` and not the whole torch namespace.
_COMPUTE_DTYPES: Final[dict[str, torch.dtype]] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _compute_dtype(quantization: QuantizationConfig) -> torch.dtype:
    """Resolve a config's compute dtype name to the torch dtype.

    Args:
        quantization: The quantization configuration.

    Returns:
        The torch dtype the 4-bit storage is dequantized to.
    """
    return _COMPUTE_DTYPES[quantization["bnb_4bit_compute_dtype"]]


def _bits_and_bytes_config(quantization: QuantizationConfig) -> BitsAndBytesConfigProto:
    """Build the transformers quantization config this run asked for.

    Every field is passed. ``bnb_4bit_quant_type`` and
    ``bnb_4bit_compute_dtype`` carry upstream defaults that are not the
    QLoRA paper's arm, so leaving either unset would quietly change what the
    run measures.

    Args:
        quantization: The quantization configuration to translate.

    Returns:
        The constructed BitsAndBytesConfig.
    """
    transformers = __import__("transformers", fromlist=["BitsAndBytesConfig"])
    config_cls: _BitsAndBytesConfigClassProto = transformers.BitsAndBytesConfig
    return config_cls(
        load_in_4bit=quantization["load_in_4bit"],
        load_in_8bit=quantization["load_in_8bit"],
        bnb_4bit_compute_dtype=_compute_dtype(quantization),
        bnb_4bit_quant_type=quantization["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quantization["bnb_4bit_use_double_quant"],
    )


def load_arguments(
    quantization: QuantizationConfig | None,
) -> tuple[BitsAndBytesConfigProto | None, torch.dtype]:
    """Decide the two loading arguments a quantization choice implies.

    Separated from the load itself so the decision is testable without a
    CUDA device: building a 4-bit model requires one, and asserting what
    would be REQUESTED does not.

    Args:
        quantization: The quantization to apply, or None for unquantized.

    Returns:
        A tuple of (quantization config or None, dtype for the layers
        quantization does not replace).
    """
    if quantization is None:
        return None, torch.float32
    return _bits_and_bytes_config(quantization), _compute_dtype(quantization)


def _default_load_hf_model(
    model_id_or_path: str, quantization: QuantizationConfig | None
) -> LMModelProto:
    """Production implementation for loading HuggingFace models.

    When a quantization config is given, the model's linear layers are
    replaced during loading, which is the only point at which that can
    happen: the QLoRA strategy adapts an ALREADY-quantized model and does
    not quantize one itself.

    ``torch_dtype`` is passed in both cases rather than left to default. The
    4-bit quantizer overrides a None with ``torch.float16`` and reports it
    only through a log line, so accepting the default means the dtype of
    every layer quantization did not replace is decided somewhere the run
    never records.

    Args:
        model_id_or_path: HuggingFace model ID or local path.
        quantization: Quantization to apply while loading, or None to load
            the weights at their stored precision.

    Returns:
        Loaded language model instance.
    """
    config, dtype = load_arguments(quantization)
    transformers = __import__("transformers", fromlist=["AutoModelForCausalLM"])
    model_cls: _HFModelClassProto = transformers.AutoModelForCausalLM
    return model_cls.from_pretrained(
        model_id_or_path, quantization_config=config, torch_dtype=dtype
    )


def _default_load_hf_tokenizer(model_id_or_path: str) -> HFTokenizerProto:
    """Production implementation for loading HuggingFace tokenizers.

    Args:
        model_id_or_path: HuggingFace model ID or local path.

    Returns:
        Loaded tokenizer instance.
    """
    transformers = __import__("transformers", fromlist=["AutoTokenizer"])
    tokenizer_cls: _HFTokenizerClassProto = transformers.AutoTokenizer
    return tokenizer_cls.from_pretrained(model_id_or_path)


def _default_create_trainer(
    prepared: PreparedLMModel,
    cfg: ModelTrainConfig,
    settings: Settings,
    *,
    run_id: str,
    redis_hb: Callable[[float], None],
    cancelled: Callable[[], bool],
    resume: bool,
    progress: ProgressCallback | None,
    service_name: str,
    wandb_publisher: WandbPublisher | None,
    determinism: DeterminismRecord,
) -> TrainerProto:
    """Production implementation for creating BaseTrainer.

    Args:
        prepared: Prepared model to train.
        cfg: Training configuration.
        settings: Application settings.
        run_id: Unique identifier for this training run.
        redis_hb: Heartbeat callback.
        cancelled: Callback to check if training was cancelled.
        progress: Optional progress callback.
        service_name: Name of the service.
        wandb_publisher: Optional W&B publisher.
        determinism: What the worker pinned before any CUDA work, so the
            manifest can record it rather than leaving it to a log line.

    Returns:
        Trainer instance.
    """
    from model_trainer.core.services.training.base_trainer import BaseTrainer

    trainer: TrainerProto = BaseTrainer(
        prepared,
        cfg,
        settings,
        run_id=run_id,
        redis_hb=redis_hb,
        cancelled=cancelled,
        resume=resume,
        progress=progress,
        service_name=service_name,
        wandb_publisher=wandb_publisher,
        determinism=determinism,
    )
    return trainer


def _default_load_tokenizer(path: str) -> TokenizerHandle:
    """Production implementation for loading tokenizers with automatic detection.

    Args:
        path: Path to tokenizer artifact directory.

    Returns:
        Loaded tokenizer handle.
    """
    from model_trainer.core.services.tokenizer.loader import load_tokenizer_from_dir

    return load_tokenizer_from_dir(path)


def _default_load_prepared_model(
    model_path: str, tokenizer_handle: TokenizerHandle | None
) -> PreparedLMModel:
    """Production implementation for loading prepared models.

    For HF LM models, the tokenizer_handle is optional (can be None) because
    the HF tokenizer is loaded from hub_model_id stored in metadata.

    Args:
        model_path: Path to model artifacts.
        tokenizer_handle: Optional tokenizer handle (unused for HF LM).

    Returns:
        Prepared model instance.
    """
    from model_trainer.core.services.model.backends.hf_lm.io import (
        load_prepared_hf_lm_from_handle,
    )

    return load_prepared_hf_lm_from_handle(model_path, tokenizer_handle)


def _default_create_causal_dataset(
    *,
    lines: Sequence[str],
    tokenizer: Encoder,
    max_len: int,
    eos_id: int,
    pad_id: int,
) -> CausalLMDatasetProto:
    """Production implementation for creating CausalLMDataset.

    Args:
        lines: Corpus lines to tokenize, already partitioned.
        tokenizer: Encoder for tokenization.
        max_len: Maximum sequence length.
        eos_id: End of sequence token ID.
        pad_id: Padding token ID.

    Returns:
        Dataset instance.
    """
    from model_trainer.core.services.training.dataset_builder import CausalLMDataset

    dataset: CausalLMDatasetProto = CausalLMDataset(
        lines=lines,
        tokenizer=tokenizer,
        max_len=max_len,
        eos_id=eos_id,
        pad_id=pad_id,
    )
    return dataset


def _default_create_dataloader(
    dataset: CausalLMDatasetProto,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoaderProto:
    """Production implementation for creating DataLoader.

    Args:
        dataset: Dataset to load from.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of workers.
        pin_memory: Whether to pin memory.

    Returns:
        DataLoader instance.
    """
    torch_data = __import__("torch.utils.data", fromlist=["DataLoader"])
    loader_cls: _DataLoaderClassProto = torch_data.DataLoader
    loader: DataLoaderProto = loader_cls(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


def _default_get_model_dir(settings: Settings, run_id: str) -> Path:
    """Production implementation for getting model directory.

    Args:
        settings: Application settings.
        run_id: Run identifier.

    Returns:
        Path to model directory.
    """
    from model_trainer.core.infra.paths import model_dir

    return model_dir(settings, run_id)


def _default_get_eval_dir(settings: Settings, run_id: str) -> Path:
    """Production implementation for getting evaluation directory.

    Args:
        settings: Application settings.
        run_id: Run identifier.

    Returns:
        Path to evaluation directory.
    """
    from model_trainer.core.infra.paths import model_eval_dir

    return model_eval_dir(settings, run_id)


def _default_read_text_file(path: Path) -> str:
    """Production implementation for reading text files.

    Args:
        path: Path to file.

    Returns:
        File contents as string.
    """
    return path.read_text(encoding="utf-8")


class Hooks:
    """Container for HF LM backend hooks, each bound to its real implementation.

    Production calls these without wiring anything first. Tests assign a fake
    and call reset() afterwards, which puts the real implementation back.
    """

    # io.py / prepare.py hooks
    load_hf_model: HFModelLoader = _default_load_hf_model
    load_hf_tokenizer: HFTokenizerLoader = _default_load_hf_tokenizer

    # train.py hooks
    create_trainer: CreateTrainerFn = _default_create_trainer

    # evaluate.py hooks
    load_tokenizer: TokenizerLoader = _default_load_tokenizer
    load_prepared_model: PreparedModelLoader = _default_load_prepared_model
    create_causal_dataset: CreateCausalDatasetFn = _default_create_causal_dataset
    create_dataloader: CreateDataLoaderFn = _default_create_dataloader
    get_model_dir: ModelDirFn = _default_get_model_dir
    get_eval_dir: EvalDirFn = _default_get_eval_dir

    # generate.py / score.py hooks
    read_text_file: ReadTextFileFn = _default_read_text_file

    @classmethod
    def reset(cls) -> None:
        """Restore every hook to its real implementation.

        The restoration `reset_hooks()` performs, exposed as a classmethod so
        an autouse fixture can name the container it protects.
        """
        reset_hooks()


def reset_hooks() -> None:
    """Restore every hook to the production implementation it is bound to."""
    Hooks.load_hf_model = _default_load_hf_model
    Hooks.load_hf_tokenizer = _default_load_hf_tokenizer
    Hooks.create_trainer = _default_create_trainer
    Hooks.load_tokenizer = _default_load_tokenizer
    Hooks.load_prepared_model = _default_load_prepared_model
    Hooks.create_causal_dataset = _default_create_causal_dataset
    Hooks.create_dataloader = _default_create_dataloader
    Hooks.get_model_dir = _default_get_model_dir
    Hooks.get_eval_dir = _default_get_eval_dir
    Hooks.read_text_file = _default_read_text_file


__all__ = [
    "CausalLMDatasetProto",
    "CreateCausalDatasetFn",
    "CreateDataLoaderFn",
    "CreateTrainerFn",
    "DataLoaderProto",
    "EvalDirFn",
    "HFModelLoader",
    "HFTokenizerLoader",
    "HFTokenizerProto",
    "Hooks",
    "ModelDirFn",
    "PreparedModelLoader",
    "ProgressCallback",
    "ReadTextFileFn",
    "TokenizerLoader",
    "TrainerProto",
    "reset_hooks",
]
