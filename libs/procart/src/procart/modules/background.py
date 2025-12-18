from __future__ import annotations

from ..math_backend import BACKEND, FloatArray
from ..types import Resolution


class BlackBackground:
    """Solid black background module with alpha=1.0.

    Ignores time, camera, and seed; only uses resolution.

    Args:
        t_normalized: Unused.
        resolution: Target resolution; width and height must be positive.
        seed: Unused.
        camera_x: Unused.
        camera_y: Unused.

    Returns:
        FloatArray: RGBA array (H, W, 4) with RGB=0 and A=1.

    Raises:
        ValueError: If resolution is invalid.
    """

    def render_frame(
        self,
        t_normalized: float,
        resolution: Resolution,
        seed: int,
        camera_x: float,
        camera_y: float,
    ) -> FloatArray:
        w = int(resolution["width"])  # Width
        h = int(resolution["height"])  # Height
        if w <= 0 or h <= 0:
            raise ValueError("resolution width and height must be positive")
        zeros = BACKEND.zeros(h, w)
        ones = BACKEND.ones(h, w)
        return BACKEND.stack_rgba(zeros, zeros, zeros, ones)


__all__ = ["BlackBackground"]
