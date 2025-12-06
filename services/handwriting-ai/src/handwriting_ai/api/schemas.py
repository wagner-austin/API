from __future__ import annotations

from typing_extensions import TypedDict


class PredictResponse(TypedDict):
    digit: int
    confidence: float
    probs: list[float]
    model_id: str
    visual_png_b64: str | None
    uncertain: bool
    latency_ms: int
