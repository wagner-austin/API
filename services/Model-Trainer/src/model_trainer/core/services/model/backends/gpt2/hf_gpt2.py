"""HuggingFace GPT-2 model integration with strict typing.

This module provides typed access to transformers.GPT2LMHeadModel and GPT2Config
using dynamic imports with Protocol-based type annotations. All public functions
return strictly typed values with no Any, cast, or type: ignore.
"""

from __future__ import annotations

from typing import Final, Protocol

from typing_extensions import TypedDict

from model_trainer.core.types import LMModelProto


class GPT2ModelSizeConfig(TypedDict, total=True):
    """Configuration for GPT-2 model architecture by size."""

    hidden_size: int
    n_layer: int
    n_head: int


# GPT-2 model size configurations
# Reference: https://huggingface.co/docs/transformers/model_doc/gpt2
MODEL_SIZES: Final[dict[str, GPT2ModelSizeConfig]] = {
    "small": {"hidden_size": 768, "n_layer": 12, "n_head": 12},  # ~124M params
    "medium": {"hidden_size": 1024, "n_layer": 24, "n_head": 16},  # ~355M params
    "large": {"hidden_size": 1280, "n_layer": 36, "n_head": 20},  # ~774M params
    "xl": {"hidden_size": 1600, "n_layer": 48, "n_head": 25},  # ~1.5B params
}


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
    """Protocol for GPT2LMHeadModel constructor returning LMModelProto."""

    def __call__(self, config: GPT2ConfigProto) -> LMModelProto: ...


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
        model_size: One of "small", "medium", "large", "xl".

    Returns:
        A GPT2Config instance with the specified architecture.

    Raises:
        KeyError: If model_size is not a valid size key.
    """
    size_config = MODEL_SIZES[model_size]
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
) -> LMModelProto:
    """Create a new GPT2LMHeadModel with the specified configuration.

    Args:
        vocab_size: Size of the vocabulary (from tokenizer).
        max_seq_len: Maximum sequence length (n_positions).
        model_size: One of "small", "medium", "large", "xl".

    Returns:
        A newly initialized GPT2LMHeadModel conforming to LMModelProto.

    Raises:
        KeyError: If model_size is not a valid size key.
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
