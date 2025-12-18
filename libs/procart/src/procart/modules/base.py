from __future__ import annotations

from typing import Protocol

from ..math_backend import FloatArray
from ..types import Resolution


class VisualModule(Protocol):
    """Abstract base interface for a procedural visual module.

    Args:
        t_normalized: Animation time in [0, 1).
        resolution: Target frame resolution.
        seed: Deterministic seed for this module.
        camera_x: Camera x offset for parallax.
        camera_y: Camera y offset for parallax.

    Returns:
        FloatArray: RGBA float32 array of shape (H, W, 4) in linear HDR domain.

    Raises:
        ValueError: If arguments are invalid.
    """

    def render_frame(
        self,
        t_normalized: float,
        resolution: Resolution,
        seed: int,
        camera_x: float,
        camera_y: float,
    ) -> FloatArray: ...


__all__ = ["VisualModule"]
