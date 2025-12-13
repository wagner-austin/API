"""Device selection and configuration for ML training.

This module provides strict device resolution logic for training workflows,
supporting CPU, CUDA, and automatic device detection. All device detection
goes through test hooks to enable reliable testing without GPU hardware.

Centralized in platform_ml to prevent drift across services (handwriting-ai,
Model-Trainer, covenant-radar-api).
"""

from __future__ import annotations

from typing import Final, Literal

from . import torch_types

RequestedDevice = Literal["cpu", "cuda", "auto"]
"""Device requested by user: explicit cpu/cuda or auto-detection."""

ResolvedDevice = Literal["cpu", "cuda"]
"""Concrete device after resolution (no 'auto')."""

RequestedPrecision = Literal["fp32", "fp16", "bf16", "auto"]
"""Precision requested by user: explicit or auto-detection based on device."""

ResolvedPrecision = Literal["fp32", "fp16", "bf16"]
"""Concrete precision after resolution (no 'auto')."""

_CUDA: Final[ResolvedDevice] = "cuda"
_CPU: Final[ResolvedDevice] = "cpu"
_FP32: Final[ResolvedPrecision] = "fp32"
_FP16: Final[ResolvedPrecision] = "fp16"
_BF16: Final[ResolvedPrecision] = "bf16"


def resolve_device(requested: RequestedDevice) -> ResolvedDevice:
    """Resolve requested device to a concrete device.

    This function centralizes device detection logic so other modules do not import
    torch directly. It performs a single check using torch.cuda.is_available() when
    'auto' is requested; otherwise returns the requested concrete device.

    Args:
        requested: The device requested by the user ("cpu", "cuda", or "auto").

    Returns:
        Concrete device to use for training ("cpu" or "cuda").

    Examples:
        >>> resolve_device("cpu")
        'cpu'
        >>> resolve_device("cuda")
        'cuda'
        >>> # When CUDA is available
        >>> resolve_device("auto")  # doctest: +SKIP
        'cuda'
    """
    if requested == _CUDA:
        return _CUDA
    if requested == _CPU:
        return _CPU

    # Use hook for CUDA availability check - allows testing without torch import
    torch = torch_types._import_torch()
    return _CUDA if torch.cuda.is_available() else _CPU


def resolve_precision(requested: RequestedPrecision, device: ResolvedDevice) -> ResolvedPrecision:
    """Resolve requested precision to a concrete precision.

    Resolution rules:
    - "auto" on CUDA resolves to "fp16" (safe default for modern GPUs)
    - "auto" on CPU resolves to "fp32" (mixed precision not useful on CPU)
    - Explicit "fp32" is always valid on any device
    - Explicit "fp16" or "bf16" on CPU raises RuntimeError

    Args:
        requested: The precision requested by the user.
        device: The resolved device (must be concrete, not "auto").

    Returns:
        Concrete precision to use for training.

    Raises:
        RuntimeError: If fp16/bf16 is requested on CPU.

    Examples:
        >>> resolve_precision("fp32", "cpu")
        'fp32'
        >>> resolve_precision("auto", "cuda")
        'fp16'
        >>> resolve_precision("auto", "cpu")
        'fp32'
    """
    if requested == _FP32:
        return _FP32
    if requested == _FP16:
        if device == _CPU:
            raise RuntimeError("fp16 precision is not supported on CPU")
        return _FP16
    if requested == _BF16:
        if device == _CPU:
            raise RuntimeError("bf16 precision is not supported on CPU")
        return _BF16
    # requested == "auto"
    if device == _CUDA:
        return _FP16
    return _FP32


def recommended_batch_size(current: int, device: ResolvedDevice) -> int:
    """Return a recommended batch size given the resolved device.

    We avoid implicit overrides: only bump modestly when device is CUDA and the
    current batch size is at or below the conservative default of 4.

    Args:
        current: The batch size requested by the user.
        device: The resolved device.

    Returns:
        Recommended batch size (bumped to 8 on CUDA if current <= 4).

    Examples:
        >>> recommended_batch_size(4, "cuda")
        8
        >>> recommended_batch_size(8, "cuda")
        8
        >>> recommended_batch_size(4, "cpu")
        4
    """
    if device == _CUDA and current <= 4:
        return 8
    return current


__all__ = [
    "RequestedDevice",
    "RequestedPrecision",
    "ResolvedDevice",
    "ResolvedPrecision",
    "recommended_batch_size",
    "resolve_device",
    "resolve_precision",
]
