from __future__ import annotations

from typing import Final

from ..color import hsv_to_rgb
from ..math_backend import BACKEND, FloatArray
from ..types import RecursiveRectsLayerConfig, Resolution


class RecursiveRects:
    """Recursive subdivision into colored neon rectangles.

    Uses a simple deterministic split pattern. Returns HDR-linear RGBA.
    """

    def __init__(self, config: RecursiveRectsLayerConfig) -> None:
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
        acc_r = BACKEND.zeros(h, w)
        acc_g = BACKEND.zeros(h, w)
        acc_b = BACKEND.zeros(h, w)
        acc_a = BACKEND.zeros(h, w)

        max_depth = int(self._cfg["max_depth"])  # Depth
        min_size = int(self._cfg["min_size"])  # Min size

        stack: list[tuple[int, int, int, int, int]] = [(0, 0, w, h, 0)]
        border: Final[int] = max(1, min(h, w) // 64)
        while stack:
            x0, y0, x1, y1, depth = stack.pop()
            width = x1 - x0
            height = y1 - y0
            if depth >= max_depth or width < min_size or height < min_size:
                cx = (x0 + x1) * 0.5 / float(w) + camera_x * 0.1
                cy = (y0 + y1) * 0.5 / float(h) + camera_y * 0.1
                hue = (cx + cy + float(depth) * 0.13 + t_normalized * 0.2) % 1.0
                rgb = hsv_to_rgb(hue, 0.9, 1.2)
                rmask = BACKEND.rect_mask(h, w, x0 + border, y0 + border, x1 - border, y1 - border)
                acc_r = acc_r + rmask * rgb[0]
                acc_g = acc_g + rmask * rgb[1]
                acc_b = acc_b + rmask * rgb[2]
                acc_a = BACKEND.clip(acc_a + rmask * 0.6, 0.0, 1.0)
            else:
                if width >= height:
                    mid = x0 + width // 2
                    stack.append((x0, y0, mid, y1, depth + 1))
                    stack.append((mid, y0, x1, y1, depth + 1))
                else:
                    mid = y0 + height // 2
                    stack.append((x0, y0, x1, mid, depth + 1))
                    stack.append((x0, mid, x1, y1, depth + 1))

        return BACKEND.stack_rgba(acc_r, acc_g, acc_b, acc_a)


__all__ = ["RecursiveRects"]
