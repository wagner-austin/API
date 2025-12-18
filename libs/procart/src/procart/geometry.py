from __future__ import annotations

from .math_backend import BACKEND, FloatArray
from .types import Resolution


def create_normalized_grid(resolution: Resolution) -> tuple[FloatArray, FloatArray]:
    """Create normalized coordinate grids in [0,1] for y (rows) and x (cols).

    Args:
        resolution: Target resolution.

    Returns:
        Tuple of (yy, xx) arrays with shape (H, W).

    Raises:
        ValueError: If width or height is not positive.
    """
    w = int(resolution["width"])  # Width
    h = int(resolution["height"])  # Height
    if w <= 0 or h <= 0:
        raise ValueError("resolution width and height must be positive")
    yy, xx = BACKEND.normalized_grid(h, w)
    return yy, xx


def normalized_distance(
    xx: FloatArray,
    yy: FloatArray,
    center_x: float,
    center_y: float,
) -> FloatArray:
    """Compute Euclidean distance in normalized [0,1] coordinates.

    Args:
        xx: X grid in [0,1] with shape (H, W).
        yy: Y grid in [0,1] with shape (H, W).
        center_x: Center X in [0,1].
        center_y: Center Y in [0,1].

    Returns:
        Distances of shape (H, W).

    Raises:
        ValueError: If grids differ in shape.
    """
    if xx.shape != yy.shape:
        raise ValueError("xx and yy must have the same shape")
    dx = xx - float(center_x)
    dy = yy - float(center_y)
    return BACKEND.hypot(dx, dy)


__all__ = ["create_normalized_grid", "normalized_distance"]
