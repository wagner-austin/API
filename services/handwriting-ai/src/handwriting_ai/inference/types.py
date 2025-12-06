from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

from torch import Tensor


class PredictOutput(TypedDict):
    digit: int
    confidence: float
    probs: tuple[float, ...]  # length 10
    model_id: str


class PreprocessOutput(TypedDict):
    tensor: Tensor
    visual_png: bytes | None


Probs = Sequence[float]
