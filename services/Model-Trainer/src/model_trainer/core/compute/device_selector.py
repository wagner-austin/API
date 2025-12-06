from __future__ import annotations

from typing import Final, Literal

RequestedDevice = Literal["cpu", "cuda", "auto"]
ResolvedDevice = Literal["cpu", "cuda"]
RequestedPrecision = Literal["fp32", "fp16", "bf16", "auto"]
ResolvedPrecision = Literal["fp32", "fp16", "bf16"]
ModelFamily = Literal["gpt2", "llama", "qwen", "char_lstm"]

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
    """
    if requested == _CUDA:
        return _CUDA
    if requested == _CPU:
        return _CPU

    import torch as _torch  # local import to avoid import-time side effects

    return _CUDA if _torch.cuda.is_available() else _CPU


def recommended_batch_size(current: int, device: ResolvedDevice) -> int:
    """Return a recommended batch size given the resolved device.

    We avoid implicit overrides: only bump modestly when device is CUDA and the
    current batch size is at or below the conservative default of 4.
    """
    if device == _CUDA and current <= 4:
        return 8
    return current


def recommended_batch_size_for(
    model_family: ModelFamily, current: int, device: ResolvedDevice
) -> int:
    """Recommend a batch size based on model family and device.

    - On CUDA, increase conservative defaults to backend-appropriate values when
      users provided very small batches (<=4).
    - On CPU, leave the user-provided batch unchanged.
    """
    if device == _CPU:
        return current
    if current > 4:
        return current
    if model_family == "char_lstm":
        return 64
    if model_family == "gpt2":
        return 32
    # Other families default to a modest bump
    return 16


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
