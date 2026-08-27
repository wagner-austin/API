"""HuggingFace GPT-2 model integration with strict typing.

This module provides typed access to transformers.GPT2LMHeadModel and GPT2Config
using dynamic imports with Protocol-based type annotations. All public functions
return strictly typed values with no Any, cast, or type: ignore.
"""

from __future__ import annotations

from typing import Protocol

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for

from model_trainer.core.services.model.model_sizes import GPT2_MODEL_SIZES as MODEL_SIZES
from model_trainer.core.types import LMModelProto, TracedLMModelProto


class GPT2ConfigProto(Protocol):
    """Protocol for GPT2Config from transformers."""

    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int
    bos_token_id: int
    eos_token_id: int


class _GPT2ConfigCtorProto(Protocol):
    """Protocol for GPT2Config constructor."""

    def __call__(
        self,
        *,
        vocab_size: int,
        n_positions: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        bos_token_id: int,
        eos_token_id: int,
    ) -> GPT2ConfigProto: ...


class _GPT2LMHeadModelCtorProto(Protocol):
    """Protocol for GPT2LMHeadModel constructor.

    Returns the TRACED protocol rather than the plain one. A GPT2LMHeadModel
    is a torch module and has always had a module graph; declaring it here is
    what lets the forward trace reach it without a cast. Callers wanting only
    :class:`LMModelProto` are unaffected -- this is a subtype of it.
    """

    def __call__(self, config: GPT2ConfigProto) -> TracedLMModelProto: ...


class _GPT2LMHeadModelLoaderProto(Protocol):
    """Protocol for GPT2LMHeadModel.from_pretrained class method."""

    @staticmethod
    def from_pretrained(path: str) -> LMModelProto: ...


def _get_gpt2_config_class() -> _GPT2ConfigCtorProto:
    """Get transformers.GPT2Config with typed interface via dynamic import."""
    transformers_mod = __import__("transformers", fromlist=["GPT2Config"])
    cls: _GPT2ConfigCtorProto = transformers_mod.GPT2Config
    return cls


def _get_gpt2_lm_head_model_ctor() -> _GPT2LMHeadModelCtorProto:
    """Get transformers.GPT2LMHeadModel constructor with typed interface."""
    transformers_mod = __import__("transformers", fromlist=["GPT2LMHeadModel"])
    cls: _GPT2LMHeadModelCtorProto = transformers_mod.GPT2LMHeadModel
    return cls


def _get_gpt2_lm_head_model_loader() -> _GPT2LMHeadModelLoaderProto:
    """Get transformers.GPT2LMHeadModel class for from_pretrained."""
    transformers_mod = __import__("transformers", fromlist=["GPT2LMHeadModel"])
    cls: _GPT2LMHeadModelLoaderProto = transformers_mod.GPT2LMHeadModel
    return cls


def create_gpt2_config(
    *,
    vocab_size: int,
    max_seq_len: int,
    model_size: str,
) -> GPT2ConfigProto:
    """Create a GPT2Config with the specified parameters.

    Args:
        vocab_size: Size of the vocabulary (from tokenizer).
        max_seq_len: Maximum sequence length (n_positions).
        model_size: One of "tiny", "small", "medium", "large", "xl".

    Returns:
        A GPT2Config instance with the specified architecture.

    Raises:
        AppError: If model_size is not a valid size key.
    """
    # Was a bare ``MODEL_SIZES[model_size]``. An unknown size therefore escaped as
    # KeyError -- an untyped 500 for what is a caller mistake -- while char_lstm's
    # equivalent lookup had always raised AppError(INVALID_MODEL_SIZE). Same error
    # code here so the two backends reject an unknown size identically.
    size_config = MODEL_SIZES.get(model_size)
    if size_config is None:
        raise AppError(
            ModelTrainerErrorCode.INVALID_MODEL_SIZE,
            "invalid model_size for gpt2",
            model_trainer_status_for(ModelTrainerErrorCode.INVALID_MODEL_SIZE),
        )
    config_cls = _get_gpt2_config_class()
    return config_cls(
        vocab_size=vocab_size,
        n_positions=max_seq_len,
        n_embd=size_config["hidden_size"],
        n_layer=size_config["n_layer"],
        n_head=size_config["n_head"],
        bos_token_id=0,
        eos_token_id=1,
    )


def create_gpt2_model(
    *,
    vocab_size: int,
    max_seq_len: int,
    model_size: str,
) -> TracedLMModelProto:
    """Create a new GPT2LMHeadModel with the specified configuration.

    Args:
        vocab_size: Size of the vocabulary (from tokenizer).
        max_seq_len: Maximum sequence length (n_positions).
        model_size: One of "tiny", "small", "medium", "large", "xl".

    Returns:
        A newly initialized GPT2LMHeadModel. Typed as TracedLMModelProto,
        which is LMModelProto plus the module graph a forward trace hooks;
        every existing caller reads it as the former.

    Raises:
        AppError: If model_size is not a valid size key. Propagated from
            create_gpt2_config, which is where the size is resolved.
    """
    config = create_gpt2_config(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        model_size=model_size,
    )
    model_ctor = _get_gpt2_lm_head_model_ctor()
    return model_ctor(config)


def load_gpt2_model(path: str) -> LMModelProto:
    """Load a GPT2LMHeadModel from a pretrained checkpoint.

    Args:
        path: Path to the model directory containing config.json and model weights.

    Returns:
        The loaded GPT2LMHeadModel conforming to LMModelProto.
    """
    loader = _get_gpt2_lm_head_model_loader()
    return loader.from_pretrained(path)
