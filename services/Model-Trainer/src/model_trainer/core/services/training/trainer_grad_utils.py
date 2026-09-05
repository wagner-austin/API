"""Gradient utilities shared by the training loop."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Literal, Protocol

import torch
from platform_core.logging import get_logger

from model_trainer.core.contracts.strategy_names import StrategyName
from model_trainer.core.types import (
    LMModelProto,
    OptimizerProto,
    ParameterLike,
)

_logger = get_logger(__name__)


class _GradScalerProto(Protocol):
    """Protocol for torch.cuda.amp.GradScaler."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor: ...
    def unscale_(self, optimizer: OptimizerProto) -> None: ...
    def step(self, optimizer: OptimizerProto) -> None: ...
    def update(self) -> None: ...


class _ClipGradNormProto(Protocol):
    """Protocol for torch.nn.utils.clip_grad_norm_ function."""

    def __call__(
        self,
        parameters: Sequence[ParameterLike],
        max_norm: float,
    ) -> torch.Tensor: ...


def _get_clip_grad_norm() -> _ClipGradNormProto:
    """Get torch.nn.utils.clip_grad_norm_ with typed interface."""
    torch_nn_utils = __import__("torch.nn.utils", fromlist=["clip_grad_norm_"])
    fn: _ClipGradNormProto = torch_nn_utils.clip_grad_norm_
    return fn


def _clip_grad_norm(parameters: Sequence[ParameterLike], *, max_norm: float) -> None:
    """Clip gradients of model parameters.

    Args:
        parameters: Model parameters from model.parameters().
        max_norm: Maximum gradient norm.
    """
    clip_fn = _get_clip_grad_norm()
    _ = clip_fn(parameters, max_norm)


def _clip_grad_norm_with_return(parameters: Sequence[ParameterLike], *, max_norm: float) -> float:
    """Clip gradients of model parameters and return the total norm before clipping.

    Args:
        parameters: Model parameters from model.parameters().
        max_norm: Maximum gradient norm.

    Returns:
        Total gradient norm before clipping (as float).
    """
    clip_fn = _get_clip_grad_norm()
    total_norm = clip_fn(parameters, max_norm)
    return float(total_norm.item())


def _freeze_embeddings(model: LMModelProto) -> None:
    """Freeze embedding layer parameters for fine-tuning.

    Attempts to find and freeze embedding layers using common naming conventions.
    Works with transformers models (wte, embed_tokens) and custom models (embedding).

    Args:
        model: The language model with an embedding layer.
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        # Match common embedding layer names across different architectures
        if any(
            embed_name in name.lower()
            for embed_name in ("wte", "embed_tokens", "embedding", "word_embedding")
        ):
            param.requires_grad = False
            frozen_count += 1
    _logger.info(
        "Froze %d embedding parameters",
        frozen_count,
        extra={"category": "model", "event": "freeze_embeddings"},
    )


def _get_autocast_context(
    precision: Literal["fp32", "fp16", "bf16"], device: torch.device
) -> AbstractContextManager[None]:
    """Get autocast context manager based on precision and device.

    Args:
        precision: The precision to use.
        device: The device (cpu or cuda).

    Returns:
        A context manager for autocast, or nullcontext for fp32.
    """
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        return nullcontext()
    # Get autocast from torch.amp (PyTorch 2.0+ API)
    torch_amp = __import__("torch.amp", fromlist=["autocast"])
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    ctx: AbstractContextManager[None] = torch_amp.autocast(device_type="cuda", dtype=dtype)
    return ctx


def _create_grad_scaler() -> _GradScalerProto:
    """Create a GradScaler for fp16 mixed precision training.

    Returns:
        A GradScaler instance for scaling gradients.
    """
    torch_amp = __import__("torch.amp", fromlist=["GradScaler"])
    scaler: _GradScalerProto = torch_amp.GradScaler()
    return scaler


def _enable_gradient_checkpointing_if_supported(
    model: LMModelProto,
    strategy: StrategyName,
) -> bool:
    """Put a model about to be trained into its activation-checkpointing posture.

    THE INVARIANT THIS OWNS. Gradient checkpointing is a property of TRAINING,
    not of how a model was constructed, so it belongs at the point where every
    model about to be trained converges -- ``BaseTrainer.train`` -- rather than
    in one of the two paths that reach it.

    Before this existed, only one path enabled it. A fresh run reaches the
    trainer through a strategy's ``apply``, which called the hook; a
    CONTINUATION -- a run with ``pretrained_run_id`` set -- reaches it through
    ``load_adapted``, which did not. The asymmetry cost a 124M model 22.84 GiB
    on a 24 GB card, roughly 180x its own weights, on a configuration that had
    trained all afternoon on the same card as a fresh run
    (2026-09-04, gpt2 at batch 8, seq 512).

    ``load_adapted`` is deliberately NOT the place to fix that. The same
    loader serves ``modeltrainer-score-run``, which reads a trained artifact
    for inference; enabling checkpointing there would force ``use_cache=False``
    on the scorer and put a training concern inside a reload path.

    WHY IT ASKS THE STRATEGY. ``supports_gradient_checkpointing`` was declared
    by all four strategies and read by nothing outside their own tests -- each
    one instead hard-coded the hook call in its ``apply``, and ``cartridge``
    encoded its "no" by omitting the call. The capability was documentation.
    Reading it here is what makes the declaration load-bearing, and it is why
    cartridge is still not checkpointed: a checkpointed model discards the
    key-value cache it is handed, so the prefix never reaches attention and
    the memory saving would be bought by not training the thing the run
    exists to train.

    Args:
        model: The model about to be trained.
        strategy: Fine-tuning strategy this run uses, whose declared
            capabilities decide whether checkpointing is applicable.

    Returns:
        True when checkpointing was enabled, False when the strategy declares
        it unsupported. The bool is the observable: a caller cannot otherwise
        tell the two apart, and neither could a test.
    """
    from model_trainer.core.services.finetuning.registry import default_registry

    if not default_registry().get_capabilities(strategy)["supports_gradient_checkpointing"]:
        _logger.info(
            "Gradient checkpointing not supported by strategy",
            extra={"category": "training", "event": "grad_ckpt_unsupported", "strategy": strategy},
        )
        return False
    model.gradient_checkpointing_enable()
    _logger.info(
        "Gradient checkpointing enabled",
        extra={"category": "training", "event": "grad_ckpt_enabled", "strategy": strategy},
    )
    return True
