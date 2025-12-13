"""Device selection utilities for Model-Trainer.

Re-exports types and functions from platform_ml for centralized device detection,
precision resolution, and batch size recommendations. Adds model-family-specific
batch size logic that is unique to Model-Trainer.
"""

from __future__ import annotations

from typing import Final, Literal

from platform_ml import RequestedDevice as RequestedDevice
from platform_ml import RequestedPrecision as RequestedPrecision
from platform_ml import ResolvedDevice as ResolvedDevice
from platform_ml import ResolvedPrecision as ResolvedPrecision
from platform_ml import recommended_batch_size as recommended_batch_size
from platform_ml import resolve_device as resolve_device
from platform_ml import resolve_precision as resolve_precision

ModelFamily = Literal["gpt2", "llama", "qwen", "char_lstm"]

_CUDA: Final[ResolvedDevice] = "cuda"
_CPU: Final[ResolvedDevice] = "cpu"


def recommended_batch_size_for(
    model_family: ModelFamily, current: int, device: ResolvedDevice
) -> int:
    """Recommend a batch size based on model family and device.

    This is Model-Trainer-specific logic that extends platform_ml's generic
    recommended_batch_size with model-family-aware defaults.

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
