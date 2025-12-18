from __future__ import annotations

from ..geometry import create_normalized_grid
from ..math_backend import BACKEND, FloatArray
from ..types import FractalMandelbrotLayerConfig, Resolution


class FractalMandelbrot:
    """Simple Mandelbrot renderer with array math and bounded iterations.

    Uses a fixed number of iterations (configurable) without early-exit masks to
    keep implementation simple and strictly typed. Produces grayscale HDR RGBA.
    """

    def __init__(self, config: FractalMandelbrotLayerConfig) -> None:
        self._cfg = config

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

        max_iter = int(self._cfg["max_iter"])  # Iterations
        bailout = float(self._cfg["bailout"])  # Escape radius
        zoom = float(self._cfg["zoom"])  # Zoom factor
        pan_x_cfg = float(self._cfg["pan_x"])  # Pan X
        pan_y_cfg = float(self._cfg["pan_y"])  # Pan Y

        yy, xx = create_normalized_grid(resolution)
        # Map to complex plane centered at (pan_x, pan_y)
        cx = (xx - 0.5 + pan_x_cfg + camera_x) * (3.0 / max(zoom, 1e-6))
        cy = (yy - 0.5 + pan_y_cfg + camera_y) * (3.0 / max(zoom, 1e-6))

        zr = BACKEND.zeros(h, w)
        zi = BACKEND.zeros(h, w)
        for _ in range(max_iter):
            zr2 = zr * zr - zi * zi + cx
            zi2 = (zr * zi) * 2.0 + cy
            zr = zr2
            zi = zi2

        mag = BACKEND.hypot(zr, zi)
        # Normalize brightness from magnitude using bailout as a scale
        b = mag / (bailout + mag)
        b = BACKEND.clip(b, 0.0, 1.0)
        a = BACKEND.ones(h, w)
        return BACKEND.stack_rgba(b, b, b, a)


__all__ = ["FractalMandelbrot"]
