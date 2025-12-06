from __future__ import annotations

from typing import Protocol

from PIL import Image
from platform_core.torch_types import (
    ThreadConfig,
    configure_torch_threads,
    set_manual_seed,
)
from torch.nn import Module

from handwriting_ai.config import MNIST_N_CLASSES

from .augment import apply_affine as _apply_affine_impl

UnknownJson = dict[str, "UnknownJson"] | list["UnknownJson"] | str | int | float | bool | None


def _apply_affine(img: Image.Image, deg_max: float, tx_frac: float) -> Image.Image:
    return _apply_affine_impl(img, deg_max, tx_frac)


def _ensure_image(obj: Image.Image | int | str | float) -> Image.Image:
    if not isinstance(obj, Image.Image):
        raise RuntimeError("MNIST returned a non-image sample")
    return obj


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    set_manual_seed(seed)


class _TorchvisionPkg(Protocol):
    @property
    def models(self) -> _ModelsModule: ...


class _ModelsModule(Protocol):
    @property
    def resnet18(self) -> _ResNet18Builder: ...


class _ResNet18Builder(Protocol):
    def __call__(self, *, weights: None, num_classes: int) -> Module: ...


def _build_model() -> Module:
    """Build a trainable model for MNIST classification.

    Returns a torch.nn.Module that can be used for training.
    Uses dynamic import to construct model without direct torch dependency.
    """
    import torch.nn as nn

    # Dynamic import with Protocol typing to avoid Any
    tv_pkg_raw = __import__("torchvision.models")
    tv_pkg: _TorchvisionPkg = tv_pkg_raw
    models_mod: _ModelsModule = tv_pkg.models
    resnet_fn: _ResNet18Builder = models_mod.resnet18
    model: Module = resnet_fn(weights=None, num_classes=int(MNIST_N_CLASSES))

    # CIFAR-style stem and 1-channel adaptation
    # Use _modules dict to properly register submodules for state_dict() serialization
    conv1_new = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    maxpool_new = nn.Identity()
    model._modules["conv1"] = conv1_new
    model._modules["maxpool"] = maxpool_new

    return model


def _configure_threads(cfg: ThreadConfig) -> None:
    configure_torch_threads(cfg)


def bytes_of_model_and_grads(model: Module) -> tuple[int, int]:
    """Compute raw bytes for model parameters and their gradients.

    Returns a tuple: (parameter_bytes, gradient_bytes).
    """
    param_bytes = 0
    grad_bytes = 0
    for p in model.parameters():
        pn = int(p.numel()) * int(p.element_size())
        param_bytes += pn
        g = p.grad
        if g is not None:
            gn = int(g.numel()) * int(g.element_size())
            grad_bytes += gn
    return param_bytes, grad_bytes


class _CudaModule(Protocol):
    """Protocol for torch.cuda module interface."""

    def is_available(self) -> bool: ...
    def current_device(self) -> int: ...
    def memory_allocated(self, device: int | None = ...) -> int: ...
    def memory_reserved(self, device: int | None = ...) -> int: ...
    def max_memory_allocated(self, device: int | None = ...) -> int: ...


class _TorchModule(Protocol):
    """Protocol for torch module interface."""

    cuda: _CudaModule


def torch_allocator_stats() -> tuple[bool, int, int, int]:
    """Return CUDA allocator stats if available: (available, allocated, reserved, max_allocated).

    Values are bytes; when CUDA is unavailable or on CPU-only builds, returns
    (False, 0, 0, 0). Safe across environments.
    """
    torch_mod: _TorchModule = __import__("torch")
    cuda: _CudaModule = torch_mod.cuda
    if not cuda.is_available():
        return False, 0, 0, 0
    dev: int = cuda.current_device()
    allocated = int(cuda.memory_allocated(dev))
    reserved = int(cuda.memory_reserved(dev))
    max_alloc = int(cuda.max_memory_allocated(dev))
    return True, allocated, reserved, max_alloc
