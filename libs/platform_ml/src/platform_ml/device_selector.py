"""Device selection and configuration for ML training.

This module provides strict device resolution logic for training workflows,
supporting CPU, CUDA, and automatic device detection. All device detection
goes through test hooks to enable reliable testing without GPU hardware.

Centralized in platform_ml to prevent drift across services (handwriting-ai,
Model-Trainer, covenant-radar-api).
"""

from __future__ import annotations

from enum import StrEnum

from . import torch_types


class RequestedDevice(StrEnum):
    """Device requested by user: explicit cpu/cuda or auto-detection."""

    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class ResolvedDevice(StrEnum):
    """Concrete device after resolution (no 'auto')."""

    CPU = "cpu"
    CUDA = "cuda"


class RequestedPrecision(StrEnum):
    """Precision requested by user: explicit or auto-detection based on device."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    AUTO = "auto"


class ResolvedPrecision(StrEnum):
    """Concrete precision after resolution (no 'auto')."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


def resolve_device(requested: RequestedDevice) -> ResolvedDevice:
    """Resolve requested device to a concrete device.

    This function centralizes device detection logic so other modules do not import
    torch directly. It performs a single check using torch.cuda.is_available() when
    AUTO is requested; otherwise returns the requested concrete device.

    Args:
        requested: The device requested by the user (CPU, CUDA, or AUTO).

    Returns:
        Concrete device to use for training (CPU or CUDA).

    Examples:
        >>> resolve_device(RequestedDevice.CPU)
        <ResolvedDevice.CPU: 'cpu'>
        >>> resolve_device(RequestedDevice.CUDA)
        <ResolvedDevice.CUDA: 'cuda'>
        >>> # When CUDA is available
        >>> resolve_device(RequestedDevice.AUTO)  # doctest: +SKIP
        <ResolvedDevice.CUDA: 'cuda'>
    """
    if requested is RequestedDevice.CUDA:
        return ResolvedDevice.CUDA
    if requested is RequestedDevice.CPU:
        return ResolvedDevice.CPU

    # Use hook for CUDA availability check - allows testing without torch import
    torch = torch_types._import_torch()
    return ResolvedDevice.CUDA if torch.cuda.is_available() else ResolvedDevice.CPU


def resolve_precision(requested: RequestedPrecision, device: ResolvedDevice) -> ResolvedPrecision:
    """Resolve requested precision to a concrete precision.

    Resolution rules:
    - AUTO on CUDA resolves to FP16 (safe default for modern GPUs)
    - AUTO on CPU resolves to FP32 (mixed precision not useful on CPU)
    - Explicit FP32 is always valid on any device
    - Explicit FP16 or BF16 on CPU raises RuntimeError

    Args:
        requested: The precision requested by the user.
        device: The resolved device (concrete, never AUTO).

    Returns:
        Concrete precision to use for training.

    Raises:
        RuntimeError: If fp16/bf16 is requested on CPU.

    Examples:
        >>> resolve_precision(RequestedPrecision.FP32, ResolvedDevice.CPU)
        <ResolvedPrecision.FP32: 'fp32'>
        >>> resolve_precision(RequestedPrecision.AUTO, ResolvedDevice.CUDA)
        <ResolvedPrecision.FP16: 'fp16'>
        >>> resolve_precision(RequestedPrecision.AUTO, ResolvedDevice.CPU)
        <ResolvedPrecision.FP32: 'fp32'>
    """
    if requested is RequestedPrecision.FP32:
        return ResolvedPrecision.FP32
    if requested is RequestedPrecision.FP16:
        if device is ResolvedDevice.CPU:
            raise RuntimeError("fp16 precision is not supported on CPU")
        return ResolvedPrecision.FP16
    if requested is RequestedPrecision.BF16:
        if device is ResolvedDevice.CPU:
            raise RuntimeError("bf16 precision is not supported on CPU")
        return ResolvedPrecision.BF16
    # requested is RequestedPrecision.AUTO
    if device is ResolvedDevice.CUDA:
        return ResolvedPrecision.FP16
    return ResolvedPrecision.FP32


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
        >>> recommended_batch_size(4, ResolvedDevice.CUDA)
        8
        >>> recommended_batch_size(8, ResolvedDevice.CUDA)
        8
        >>> recommended_batch_size(4, ResolvedDevice.CPU)
        4
    """
    if device is ResolvedDevice.CUDA and current <= 4:
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
