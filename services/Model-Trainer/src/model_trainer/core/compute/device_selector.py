"""Device selection utilities for Model-Trainer.

Re-exports types and functions from platform_ml for centralized device
detection and precision resolution.

IT USED TO ADD ONE THING OF ITS OWN, and that thing was removed on
2026-09-04. ``recommended_batch_size_for`` rewrote any declared batch size of
4 or less on CUDA to a family default, so a payload declaring 4 trained at 16
and reported 4. Batch size decides the optimization trajectory, so that made
one document describe two different experiments depending on which entry
point ran it. What a payload declares is not a suggestion.
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

ModelFamily = Literal["gpt2", "llama", "qwen", "char_lstm", "hf_lm"]

_CUDA: Final[ResolvedDevice] = "cuda"
_CPU: Final[ResolvedDevice] = "cpu"
